#!/usr/bin/env python3
"""
Improved Diffusion-StyleGAN2 Own-Code DDP Training Script for CIFAR-10.

This file is self-contained and does not import or clone the official Diffusion-GAN repository.
It follows the paper-level Diffusion-GAN procedure:
1) Generate fake images with a StyleGAN2-style generator.
2) Diffuse both real and fake images with the same adaptive forward diffusion process.
3) Train a timestep-conditioned discriminator D(y, t).
4) Update the adaptive maximum timestep T from the discriminator's real-image behavior.
5) Use PyTorch DistributedDataParallel to split one global minibatch across multiple GPUs.
6) Improvement: use sinusoidal timestep embedding and block-wise FiLM conditioning in D.

Run with:
    torchrun --standalone --nproc_per_node=2 train_diffusion_gan_ddp.py
"""

import os
import math
import random
import copy
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Speed-oriented backend settings. These do not change the model architecture or objective.
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class TrainConfig:
    # Dataset settings
    dataset: str = "CIFAR10"
    data_root: str = "./data"
    resolution: int = 32
    img_channels: int = 3
    num_workers: int = 4

    # StyleGAN2 CIFAR-like model settings
    z_dim: int = 512
    w_dim: int = 512
    mapping_layers: int = 2
    use_film_t_conditioning: bool = True
    t_fourier_dim: int = 128
    channel_base: int = 16384
    channel_max: int = 512

    # Training settings. batch_size is the global batch size across all GPUs.
    batch_size: int = 64
    total_kimg: int = 25000
    g_lr: float = 0.0025
    d_lr: float = 0.0025
    betas: Tuple[float, float] = (0.0, 0.99)

    # StyleGAN2 lazy regularization settings
    r1_gamma: float = 0.01
    d_reg_interval: int = 16
    pl_weight: float = 2.0
    g_reg_interval: int = 4
    pl_batch_shrink: int = 2
    pl_decay: float = 0.01

    # Exponential moving average for the generator
    ema_kimg: float = 500.0

    # Diffusion-GAN pixel-level diffusion settings
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    t_min: int = 5
    t_max: int = 1000
    noise_sd: float = 0.05
    dtarget: float = 0.6
    ts_dist: str = "priority"
    update_T_every: int = 4
    T_step: int = 1
    tepl_size: int = 64
    tepl_zero_count: int = 32

    # Logging and output settings
    outdir: str = "./runs/owncode_diffusion_stylegan2_cifar10_ddp"
    sample_every_kimg: int = 100
    save_every_kimg: int = 500
    print_every: int = 50
    seed: int = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Own-code Diffusion-StyleGAN2 DDP trainer for CIFAR-10")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Global batch size across all GPUs")
    parser.add_argument("--total-kimg", type=int, default=None, help="Thousands of real images to process")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--sample-every-kimg", type=int, default=None)
    parser.add_argument("--save-every-kimg", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ts-dist", type=str, choices=["priority", "uniform"], default=None)
    parser.add_argument("--t-max", type=int, default=None)
    parser.add_argument("--pl-weight", type=float, default=None)
    parser.add_argument("--r1-gamma", type=float, default=None)
    parser.add_argument("--disable-film-t-conditioning", action="store_true", help="Disable the improved block-wise timestep FiLM conditioning")
    parser.add_argument("--t-fourier-dim", type=int, default=None, help="Sinusoidal timestep embedding dimension")
    return parser.parse_args()


def make_config_from_args(args):
    cfg = TrainConfig()
    for field_name, arg_name in [
        ("data_root", "data_root"),
        ("outdir", "outdir"),
        ("batch_size", "batch_size"),
        ("total_kimg", "total_kimg"),
        ("num_workers", "num_workers"),
        ("sample_every_kimg", "sample_every_kimg"),
        ("save_every_kimg", "save_every_kimg"),
        ("print_every", "print_every"),
        ("seed", "seed"),
        ("ts_dist", "ts_dist"),
        ("t_max", "t_max"),
        ("pl_weight", "pl_weight"),
        ("r1_gamma", "r1_gamma"),
        ("t_fourier_dim", "t_fourier_dim"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, field_name, value)
    if getattr(args, 'disable_film_t_conditioning', False):
        cfg.use_film_t_conditioning = False
    return cfg


def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "DDP environment variables are missing. Run this script with torchrun, for example: "
            "torchrun --standalone --nproc_per_node=2 train_diffusion_gan_ddp.py"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # PyTorch versions that support device_id can avoid NCCL's
    # "devices used by this process are currently unknown" barrier warning.
    try:
        dist.init_process_group(backend="nccl", device_id=device)
    except TypeError:
        dist.init_process_group(backend="nccl")
        dist.barrier(device_ids=[local_rank])
    is_main = rank == 0
    return local_rank, rank, world_size, device, is_main


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def build_dataloader(cfg, rank, world_size):
    transform = transforms.Compose([
        transforms.Resize(cfg.resolution),
        transforms.CenterCrop(cfg.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * cfg.img_channels, [0.5] * cfg.img_channels),
    ])

    # Download only on rank 0 to avoid multiple processes writing the same files simultaneously.
    if rank == 0:
        datasets.CIFAR10(root=cfg.data_root, train=True, transform=transform, download=True)
    dist.barrier()

    dataset = datasets.CIFAR10(root=cfg.data_root, train=True, transform=transform, download=False)

    assert cfg.batch_size % world_size == 0, (
        f"Global batch_size={cfg.batch_size} must be divisible by world_size={world_size}."
    )
    per_gpu_batch_size = cfg.batch_size // world_size

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    return loader, sampler, per_gpu_batch_size


def infinite_loader(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * torch.rsqrt(x.square().mean(dim=dim, keepdim=True) + eps)


class FullyConnectedLayer(nn.Module):
    """
    Fully connected layer with equalized learning rate.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation="linear",
        lr_multiplier=1.0,
        bias_init=0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / math.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias * self.bias_gain if self.bias is not None else None
        x = F.linear(x, w, b)

        if self.activation == "lrelu":
            x = F.leaky_relu(x, 0.2) * math.sqrt(2)
        elif self.activation == "linear":
            pass
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return x


class Conv2dLayer(nn.Module):
    """
    Convolution layer with equalized learning rate and optional up/downsampling.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation="lrelu",
        up=False,
        down=False,
    ):
        super().__init__()
        self.up = up
        self.down = down
        self.activation = activation
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.weight_gain = 1 / math.sqrt(in_channels * kernel_size * kernel_size)

    def forward(self, x):
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        w = self.weight * self.weight_gain
        x = F.conv2d(x, w, bias=self.bias, padding=self.padding)

        if self.down:
            x = F.avg_pool2d(x, kernel_size=2)

        if self.activation == "lrelu":
            x = F.leaky_relu(x, 0.2) * math.sqrt(2)
        elif self.activation == "linear":
            pass
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return x


class MappingNetwork(nn.Module):
    """
    StyleGAN2 mapping network from latent vector z to intermediate latent vector w.
    """
    def __init__(self, z_dim, w_dim, num_layers=2, lr_multiplier=0.01):
        super().__init__()
        layers = []
        in_dim = z_dim

        for _ in range(num_layers):
            layers.append(
                FullyConnectedLayer(
                    in_dim,
                    w_dim,
                    activation="lrelu",
                    lr_multiplier=lr_multiplier,
                )
            )
            in_dim = w_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        z = normalize_2nd_moment(z)
        x = z

        for layer in self.layers:
            x = layer(x)

        return x


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Transformer/DDPM-style sinusoidal embedding for the diffusion timestep.

    This is used only in the improved discriminator conditioning path.
    """
    def __init__(self, t_max=1000, embedding_dim=128):
        super().__init__()
        self.t_max = float(t_max)
        self.embedding_dim = int(embedding_dim)

        half_dim = max(1, self.embedding_dim // 2)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        t = t.float().view(-1, 1)
        args = t * self.freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if emb.shape[1] < self.embedding_dim:
            emb = F.pad(emb, (0, self.embedding_dim - emb.shape[1]))
        elif emb.shape[1] > self.embedding_dim:
            emb = emb[:, : self.embedding_dim]

        # Keep the normalized raw timestep as an explicit low-frequency coordinate.
        emb = torch.cat([emb, t / self.t_max], dim=1)

        return emb


class TimestepMappingNetwork(nn.Module):
    """
    Mapping network used to inject the diffusion timestep into the discriminator.

    Baseline Diffusion-GAN injects the discrete timestep through the discriminator's
    mapping network. This improved version gives that mapping network a sinusoidal
    timestep embedding, which makes nearby timesteps easier to relate to each other.
    """
    def __init__(self, t_max=1000, w_dim=512, num_layers=2, lr_multiplier=0.01, fourier_dim=128):
        super().__init__()
        self.t_max = float(t_max)
        self.embedding = SinusoidalTimestepEmbedding(t_max=t_max, embedding_dim=fourier_dim)
        self.layers = nn.ModuleList()

        in_dim = int(fourier_dim) + 1
        for _ in range(num_layers):
            self.layers.append(
                FullyConnectedLayer(
                    in_dim,
                    w_dim,
                    activation="lrelu",
                    lr_multiplier=lr_multiplier,
                )
            )
            in_dim = w_dim

    def forward(self, t):
        x = self.embedding(t)

        for layer in self.layers:
            x = layer(x)

        return x


class ModulatedConv2d(nn.Module):
    """
    StyleGAN2 modulated convolution with optional demodulation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        w_dim,
        demodulate=True,
        up=False,
        eps=1e-8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.up = up
        self.eps = eps
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        # StyleGAN2-style equalized weight gain for modulated convolution.
        # Without this factor, activations become too large and ToRGB can saturate.
        self.weight_gain = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1.0)

    def forward(self, x, w):
        batch, in_ch, height, width = x.shape

        styles = self.affine(w).view(batch, 1, in_ch, 1, 1)
        weight = (self.weight * self.weight_gain) * styles

        if self.demodulate:
            demod = torch.rsqrt(weight.square().sum(dim=(2, 3, 4)) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels,
            in_ch,
            self.kernel_size,
            self.kernel_size,
        )

        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            height, width = height * 2, width * 2

        x = x.reshape(1, batch * in_ch, height, width)
        x = F.conv2d(x, weight, padding=self.padding, groups=batch)
        x = x.reshape(batch, self.out_channels, height, width)

        return x


class SynthesisLayer(nn.Module):
    """
    One StyleGAN2 synthesis layer: modulated convolution, noise injection, bias, and leaky ReLU.
    """
    def __init__(self, in_channels, out_channels, w_dim, resolution, kernel_size=3, up=False):
        super().__init__()
        self.resolution = resolution
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, w_dim, up=up)
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w, noise_mode="random"):
        x = self.conv(x, w)

        if noise_mode == "random":
            noise = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
            x = x + noise * self.noise_strength
        elif noise_mode == "const":
            # Constant noise is omitted in this compact own-code version.
            # During training, random noise is normally used.
            pass
        elif noise_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported noise_mode: {noise_mode}")

        x = x + self.bias.view(1, -1, 1, 1)
        x = F.leaky_relu(x, 0.2) * math.sqrt(2)

        return x


class ToRGBLayer(nn.Module):
    """
    StyleGAN2 ToRGB layer.
    """
    def __init__(self, in_channels, out_channels, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            w_dim=w_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w):
        x = self.conv(x, w)
        x = x + self.bias.view(1, -1, 1, 1)
        return x


def num_channels(resolution, channel_base=16384, channel_max=512):
    return min(channel_base // resolution, channel_max)


class SynthesisBlock(nn.Module):
    """
    StyleGAN2 synthesis block for one spatial resolution.
    """
    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels, is_first):
        super().__init__()
        self.is_first = is_first
        self.resolution = resolution

        if is_first:
            self.const = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim, resolution=4, up=False)
        else:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim, resolution=resolution, up=True)
            self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim, resolution=resolution, up=False)

        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim)

    @property
    def num_conv(self):
        return 1 if self.is_first else 2

    @property
    def num_torgb(self):
        return 1

    def forward(self, x, img, ws, noise_mode="random"):
        # ws has shape [batch, num_conv + num_torgb, w_dim].
        w_iter = iter(ws.unbind(dim=1))

        if self.is_first:
            batch = ws.shape[0]
            x = self.const.repeat(batch, 1, 1, 1)
            x = self.conv1(x, next(w_iter), noise_mode=noise_mode)
        else:
            x = self.conv0(x, next(w_iter), noise_mode=noise_mode)
            x = self.conv1(x, next(w_iter), noise_mode=noise_mode)

        y = self.torgb(x, next(w_iter))

        if img is not None:
            img = F.interpolate(img, scale_factor=2, mode="nearest")
            img = img + y
        else:
            img = y

        return x, img


class Generator(nn.Module):
    """
    StyleGAN2-style generator for CIFAR-10 resolution.
    """
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        img_resolution=32,
        img_channels=3,
        mapping_layers=2,
        channel_base=16384,
        channel_max=512,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.mapping = MappingNetwork(z_dim, w_dim, num_layers=mapping_layers)

        resolutions = [2 ** i for i in range(2, int(math.log2(img_resolution)) + 1)]
        channels = {res: num_channels(res, channel_base, channel_max) for res in resolutions}

        blocks = []
        in_ch = 0
        self.num_ws = 0

        for res in resolutions:
            out_ch = channels[res]
            is_first = res == 4

            block = SynthesisBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_first=is_first,
            )

            blocks.append((str(res), block))
            self.num_ws += block.num_conv + block.num_torgb
            in_ch = out_ch

        self.blocks = nn.ModuleDict(blocks)

    def forward(self, z, noise_mode="random", return_ws=False):
        w = self.mapping(z)
        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)

        x = None
        img = None
        w_idx = 0

        for _, block in self.blocks.items():
            block_ws = ws[:, w_idx : w_idx + block.num_conv + block.num_torgb]
            w_idx += block.num_conv + block.num_torgb
            x, img = block(x, img, block_ws, noise_mode=noise_mode)

        # StyleGAN2 uses a linear ToRGB output. Do not apply tanh here.
        # Clamping/normalization is done only when saving image grids.
        if return_ws:
            return img, ws

        return img

class MinibatchStdLayer(nn.Module):
    """
    Minibatch standard deviation layer used in the StyleGAN2 discriminator.
    """
    def __init__(self, group_size=4, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N)

        if N % G != 0:
            G = N

        y = x.view(G, -1, self.num_channels, C // self.num_channels, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.square().mean(dim=0) + 1e-8)
        y = y.mean(dim=(2, 3, 4), keepdim=True)
        y = y.mean(dim=2)
        y = y.repeat(G, 1, H, W)

        return torch.cat([x, y], dim=1)


class TimeFiLM(nn.Module):
    """
    Feature-wise affine timestep conditioning for discriminator feature maps.

    The parameters are zero-initialized, so the improved discriminator starts as the
    baseline discriminator and learns to use block-wise timestep conditioning only
    when it helps.
    """
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.zeros(channels * 2, cond_dim))
        self.bias = nn.Parameter(torch.zeros(channels * 2))

    def forward(self, x, cond):
        gamma_beta = F.linear(cond, self.weight, self.bias)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)

        return x * (1.0 + gamma) + beta


class DiscriminatorBlock(nn.Module):
    """
    Residual discriminator block.

    Improved version: optionally applies timestep-conditioned FiLM after each
    convolution so the discriminator can use different feature statistics at
    different diffusion noise levels.
    """
    def __init__(
        self,
        in_channels,
        tmp_channels,
        out_channels,
        resolution,
        img_channels,
        first_layer=False,
        cmap_dim=512,
        use_film_t_conditioning=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.first_layer = first_layer
        self.use_film_t_conditioning = use_film_t_conditioning

        if first_layer:
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation="lrelu")

        self.conv0 = Conv2dLayer(in_channels, tmp_channels, kernel_size=3, activation="lrelu")
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation="lrelu", down=True)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, activation="linear", down=True, bias=False)

        if self.use_film_t_conditioning:
            self.film0 = TimeFiLM(cmap_dim, tmp_channels)
            self.film1 = TimeFiLM(cmap_dim, out_channels)

    def forward(self, x, img=None, cmap=None):
        if self.first_layer:
            x = self.fromrgb(img)

        skip = self.skip(x)

        x = self.conv0(x)
        if self.use_film_t_conditioning:
            x = self.film0(x, cmap)

        x = self.conv1(x)
        if self.use_film_t_conditioning:
            x = self.film1(x, cmap)

        x = (x + skip) / math.sqrt(2)

        return x


class DiscriminatorEpilogue(nn.Module):
    """
    Final discriminator block with projection conditioning.
    """
    def __init__(self, in_channels, cmap_dim=512, resolution=4):
        super().__init__()
        self.mbstd = MinibatchStdLayer()
        self.conv = Conv2dLayer(in_channels + 1, in_channels, kernel_size=3, activation="lrelu")
        self.fc = FullyConnectedLayer(in_channels * resolution * resolution, in_channels, activation="lrelu")
        self.out = FullyConnectedLayer(in_channels, cmap_dim, activation="linear")
        self.cmap_dim = cmap_dim

    def forward(self, x, cmap):
        x = self.mbstd(x)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.out(x)

        # Projection discriminator: the final logit depends on the timestep conditioning vector.
        logits = (x * cmap).sum(dim=1) / math.sqrt(self.cmap_dim)

        return logits


class Discriminator(nn.Module):
    """
    Diffusion-StyleGAN2 discriminator D(y, t).

    The discriminator follows the StyleGAN2 conditional-discriminator idea,
    but the conditioning variable is the diffusion timestep t instead of a class label.
    """
    def __init__(
        self,
        img_resolution=32,
        img_channels=3,
        channel_base=16384,
        channel_max=512,
        cmap_dim=512,
        t_max=1000,
        mapping_layers=2,
        use_film_t_conditioning=True,
        t_fourier_dim=128,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.cmap_dim = cmap_dim
        self.use_film_t_conditioning = use_film_t_conditioning

        self.t_mapping = TimestepMappingNetwork(
            t_max=t_max,
            w_dim=cmap_dim,
            num_layers=mapping_layers,
            fourier_dim=t_fourier_dim,
        )

        resolutions = [2 ** i for i in range(int(math.log2(img_resolution)), 2, -1)]
        channels = {res: num_channels(res, channel_base, channel_max) for res in [4] + resolutions}

        blocks = []
        in_ch = channels[img_resolution]

        for idx, res in enumerate(resolutions):
            out_ch = channels[res // 2]

            block = DiscriminatorBlock(
                in_channels=in_ch,
                tmp_channels=in_ch,
                out_channels=out_ch,
                resolution=res,
                img_channels=img_channels,
                first_layer=(idx == 0),
                cmap_dim=cmap_dim,
                use_film_t_conditioning=use_film_t_conditioning,
            )

            blocks.append((str(res), block))
            in_ch = out_ch

        self.blocks = nn.ModuleDict(blocks)
        self.epilogue = DiscriminatorEpilogue(in_channels=in_ch, cmap_dim=cmap_dim, resolution=4)

    def forward(self, img, t):
        cmap = self.t_mapping(t)

        x = None
        for _, block in self.blocks.items():
            x = block(x, img if x is None else None, cmap=cmap)

        logits = self.epilogue(x, cmap)

        return logits

class AdaptiveDiffusion(nn.Module):
    """
    Forward diffusion sampler for Diffusion-GAN.

    It implements:

        q(y | x, t) = N(y; sqrt(alpha_bar_t) x, (1 - alpha_bar_t) sigma^2 I)

    Using reparameterization:

        y = sqrt(alpha_bar_t) x + sqrt(1 - alpha_bar_t) sigma epsilon

    It also maintains the adaptive maximum timestep T and the exploration list t_epl.
    """
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=2e-2,
        t_min=5,
        t_max=1000,
        sigma=0.05,
        dtarget=0.6,
        ts_dist="priority",
        update_step=1,
        tepl_size=64,
        tepl_zero_count=32,
        device="cuda",
    ):
        super().__init__()
        assert ts_dist in ["priority", "uniform"]

        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.T = int(t_min)
        self.sigma = float(sigma)
        self.dtarget = float(dtarget)
        self.ts_dist = ts_dist
        self.update_step = int(update_step)
        self.tepl_size = int(tepl_size)
        self.tepl_zero_count = int(tepl_zero_count)

        beta = torch.zeros(t_max + 1, device=device)
        beta[1:] = torch.linspace(beta_start, beta_end, t_max, device=device)

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar[0] = 1.0

        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)

        self.tepl = self.make_tepl().to(device)

    @torch.no_grad()
    def sample_from_p_pi(self, n):
        """
        Sample timesteps from p_pi.

        Uniform sampling:
            P(t) = 1 / T

        Priority sampling:
            P(t) is proportional to t.
        """
        if self.T <= 0:
            return torch.zeros(n, dtype=torch.long)

        if self.ts_dist == "uniform":
            return torch.randint(1, self.T + 1, (n,), dtype=torch.long)

        weights = torch.arange(1, self.T + 1, dtype=torch.float)
        probs = weights / weights.sum()

        return torch.multinomial(probs, n, replacement=True).long() + 1

    @torch.no_grad()
    def make_tepl(self):
        """
        Build the exploration list:

            t_epl = [0, ..., 0, t_1, ..., t_32]

        The default length is 64, with 32 zeros and 32 sampled timesteps.
        """
        n_random = self.tepl_size - self.tepl_zero_count
        zeros = torch.zeros(self.tepl_zero_count, dtype=torch.long)
        sampled = self.sample_from_p_pi(n_random)

        return torch.cat([zeros, sampled], dim=0)

    @torch.no_grad()
    def sample_t(self, batch_size, device):
        """
        Sample timesteps uniformly from the current exploration list t_epl.
        """
        idx = torch.randint(0, self.tepl.numel(), (batch_size,), device=device)
        return self.tepl.to(device)[idx]

    def diffuse(self, x, t):
        """
        Apply the forward diffusion process to a batch of images.

        x must be scaled to [-1, 1].
        """
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x)
        y = torch.sqrt(a_bar) * x + torch.sqrt(1.0 - a_bar) * self.sigma * eps

        return y

    @torch.no_grad()
    def update_T(self, real_logits):
        """
        Adaptive timestep update.

        The paper defines:

            r_d = E[sign(D(y, t) - 0.5)]
            T = T + sign(r_d - d_target) * C

        This implementation uses logits. Since sigmoid(logit) > 0.5 is equivalent
        to logit > 0, sign(logit) is used.
        """
        rd = torch.sign(real_logits.detach()).float().mean().item()

        direction = 1 if rd > self.dtarget else -1
        self.T += direction * self.update_step
        self.T = int(max(self.t_min, min(self.t_max, self.T)))

        self.tepl = self.make_tepl().to(real_logits.device)

        return rd, self.T

def d_logistic_loss(real_logits, fake_logits):
    """
    Non-saturating logistic discriminator loss.
    """
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def g_nonsaturating_loss(fake_logits):
    """
    Non-saturating generator loss.
    """
    return F.softplus(-fake_logits).mean()


def d_r1_loss(real_logits, real_img):
    """
    R1 gradient penalty on real images.
    """
    grad_real = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_img,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return grad_real.square().reshape(real_img.shape[0], -1).sum(1).mean()


def g_path_length_regularize(fake_img, ws, pl_mean, decay=0.01):
    """
    StyleGAN2 path length regularization.
    """
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])

    grad = torch.autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=ws,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    path_lengths = torch.sqrt(grad.square().sum(2).mean(1) + 1e-8)
    pl_mean_new = pl_mean + decay * (path_lengths.mean() - pl_mean)
    penalty = (path_lengths - pl_mean_new).square().mean()

    return penalty, pl_mean_new.detach(), path_lengths.detach()


@torch.no_grad()
def update_ema(G_ema, G, batch_size, ema_kimg):
    """Update the exponential moving average generator from the DDP-wrapped generator."""
    source_G = unwrap_model(G)
    ema_beta = 0.5 ** (batch_size / max(ema_kimg * 1000.0, 1e-8))

    for p_ema, p in zip(G_ema.parameters(), source_G.parameters()):
        p_ema.copy_(p.lerp(p_ema, ema_beta))

    for b_ema, b in zip(G_ema.buffers(), source_G.buffers()):
        b_ema.copy_(b)


@torch.no_grad()
def save_samples(G_ema, cfg, device, cur_kimg, n=64, seed=0):
    """Generate and save sample images from the EMA generator."""
    G_ema.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(n, cfg.z_dim, generator=gen, device=device)

    img = G_ema(z, noise_mode="const").detach().cpu()
    # Clamp only for visualization. The generator itself remains linear.
    img_vis = img.clamp(-1, 1)
    grid = utils.make_grid(img_vis, nrow=8, normalize=True, value_range=(-1, 1))

    out_path = Path(cfg.outdir) / f"samples_kimg_{int(cur_kimg):06d}.png"
    utils.save_image(grid, out_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"kimg={cur_kimg:.1f}")
    plt.tight_layout()
    plt.savefig(Path(cfg.outdir) / f"samples_kimg_{int(cur_kimg):06d}_preview.png", dpi=150)
    plt.close()

    return out_path


def save_checkpoint(cfg, G, D, G_ema, opt_G, opt_D, diffusion, pl_mean, cur_nimg, history):
    """Save a DDP-safe checkpoint using unwrapped model state_dicts."""
    ckpt = {
        "G": unwrap_model(G).state_dict(),
        "D": unwrap_model(D).state_dict(),
        "G_ema": G_ema.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "diffusion_T": diffusion.T,
        "diffusion_tepl": diffusion.tepl.detach().cpu(),
        "pl_mean": pl_mean.detach().cpu(),
        "cur_nimg": cur_nimg,
        "cfg": asdict(cfg),
        "history": history,
    }

    path = Path(cfg.outdir) / f"ckpt_kimg_{cur_nimg // 1000:06d}.pt"
    torch.save(ckpt, path)
    return path


@torch.no_grad()
def update_diffusion_T_ddp(diffusion, real_logits, rank, world_size):
    """
    Update adaptive maximum timestep T using all ranks.

    The original update is based on the discriminator's behavior on diffused real images.
    In DDP, every rank only sees a shard of the global minibatch, so the local statistics
    are all-reduced before changing T. Then rank 0 creates the new t_epl and broadcasts it.
    """
    local_rd = torch.sign(real_logits.detach()).float().mean()
    dist.all_reduce(local_rd, op=dist.ReduceOp.SUM)
    rd = (local_rd / world_size).item()

    direction = 1 if rd > diffusion.dtarget else -1
    diffusion.T += direction * diffusion.update_step
    diffusion.T = int(max(diffusion.t_min, min(diffusion.t_max, diffusion.T)))

    if rank == 0:
        diffusion.tepl = diffusion.make_tepl().to(real_logits.device)
    dist.broadcast(diffusion.tepl, src=0)

    return rd, diffusion.T


def main():
    args = parse_args()
    cfg = make_config_from_args(args)

    local_rank, rank, world_size, device, is_main = setup_ddp()
    seed_everything(cfg.seed, rank)
    Path(cfg.outdir).mkdir(parents=True, exist_ok=True)

    if is_main:
        print("DDP initialized")
        print("world_size:", world_size)
        print("CUDA devices visible:", torch.cuda.device_count())
        print("config:", asdict(cfg))

    loader, sampler, per_gpu_batch_size = build_dataloader(cfg, rank, world_size)
    data_iter = infinite_loader(loader, sampler)

    if is_main:
        print("global batch size:", cfg.batch_size)
        print("per-GPU batch size:", per_gpu_batch_size)

    G = Generator(
        z_dim=cfg.z_dim,
        w_dim=cfg.w_dim,
        img_resolution=cfg.resolution,
        img_channels=cfg.img_channels,
        mapping_layers=cfg.mapping_layers,
        channel_base=cfg.channel_base,
        channel_max=cfg.channel_max,
    ).to(device)

    D = Discriminator(
        img_resolution=cfg.resolution,
        img_channels=cfg.img_channels,
        channel_base=cfg.channel_base,
        channel_max=cfg.channel_max,
        cmap_dim=cfg.w_dim,
        t_max=cfg.t_max,
        mapping_layers=cfg.mapping_layers,
        use_film_t_conditioning=cfg.use_film_t_conditioning,
        t_fourier_dim=cfg.t_fourier_dim,
    ).to(device)

    G_ema = copy.deepcopy(G).eval().requires_grad_(False)

    G = DDP(G, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    D = DDP(D, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    diffusion = AdaptiveDiffusion(
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        sigma=cfg.noise_sd,
        dtarget=cfg.dtarget,
        ts_dist=cfg.ts_dist,
        update_step=cfg.T_step,
        tepl_size=cfg.tepl_size,
        tepl_zero_count=cfg.tepl_zero_count,
        device=device,
    ).to(device)

    # Make the initial exploration list identical across ranks.
    dist.broadcast(diffusion.tepl, src=0)

    g_mb_ratio = cfg.g_reg_interval / (cfg.g_reg_interval + 1)
    d_mb_ratio = cfg.d_reg_interval / (cfg.d_reg_interval + 1)

    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=cfg.g_lr * g_mb_ratio,
        betas=(cfg.betas[0] ** g_mb_ratio, cfg.betas[1] ** g_mb_ratio),
    )

    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=cfg.d_lr * d_mb_ratio,
        betas=(cfg.betas[0] ** d_mb_ratio, cfg.betas[1] ** d_mb_ratio),
    )

    pl_mean = torch.zeros([], device=device)

    if is_main:
        G_unwrapped = unwrap_model(G)
        D_unwrapped = unwrap_model(D)
        print("G parameters:", sum(p.numel() for p in G_unwrapped.parameters()))
        print("D parameters:", sum(p.numel() for p in D_unwrapped.parameters()))
        print("G.num_ws:", G_unwrapped.num_ws)
        print("Initial T:", diffusion.T)
        print("Improvement: block-wise timestep FiLM conditioning =", cfg.use_film_t_conditioning)
        print("Timestep Fourier embedding dim:", cfg.t_fourier_dim)
        print("Initial t_epl:", diffusion.tepl[:20].detach().cpu().tolist())

    total_nimg = cfg.total_kimg * 1000
    cur_nimg = 0
    step = 0
    last_sample_kimg = -1
    last_save_kimg = -1

    history = {
        "kimg": [],
        "d_loss": [],
        "g_loss": [],
        "r1": [],
        "pl": [],
        "rd": [],
        "T": [],
        "real_score": [],
        "fake_score": [],
    }

    pbar = tqdm(total=cfg.total_kimg, desc="training kimg") if is_main else None

    while cur_nimg < total_nimg:
        step += 1

        # ----------------------------
        # Step I: Update discriminator.
        # ----------------------------
        for p in D.parameters():
            p.requires_grad_(True)
        for p in G.parameters():
            p.requires_grad_(False)

        real_img, _ = next(data_iter)
        real_img = real_img.to(device, non_blocking=True)

        z = torch.randn(per_gpu_batch_size, cfg.z_dim, device=device)
        with torch.no_grad():
            # G is frozen here. Calling the unwrapped local module avoids unnecessary
            # DDP reducer bookkeeping during the discriminator update.
            fake_img = unwrap_model(G)(z, noise_mode="random")

        t = diffusion.sample_t(per_gpu_batch_size, device)
        y_real = diffusion.diffuse(real_img, t)
        y_fake = diffusion.diffuse(fake_img, t)

        real_logits = D(y_real, t)
        fake_logits = D(y_fake, t)

        d_loss = d_logistic_loss(real_logits, fake_logits)

        opt_D.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_D.step()

        # Apply R1 regularization lazily.
        r1_val = torch.zeros([], device=device)
        if step % cfg.d_reg_interval == 0:
            real_img, _ = next(data_iter)
            real_img = real_img.to(device, non_blocking=True).requires_grad_(True)

            t_r1 = diffusion.sample_t(per_gpu_batch_size, device)
            y_real_r1 = diffusion.diffuse(real_img, t_r1)
            real_logits_r1 = D(y_real_r1, t_r1)

            r1_penalty = d_r1_loss(real_logits_r1, real_img)
            r1_loss = r1_penalty * (cfg.r1_gamma * cfg.d_reg_interval / 2)

            # DDP note:
            # R1 is a gradient penalty with respect to the input image. Some discriminator
            # parameters, especially final additive biases, can be mathematically unused by
            # the R1 term alone. Adding this zero-valued logit term does not change the loss,
            # but it keeps every parameter connected to the graph so DDP reduction finishes.
            r1_loss = r1_loss + 0.0 * real_logits_r1.sum()

            opt_D.zero_grad(set_to_none=True)
            r1_loss.backward()
            opt_D.step()

            r1_val = r1_penalty.detach()

        # ----------------------------
        # Step II: Update generator.
        # ----------------------------
        for p in D.parameters():
            p.requires_grad_(False)
        for p in G.parameters():
            p.requires_grad_(True)

        z = torch.randn(per_gpu_batch_size, cfg.z_dim, device=device)
        fake_img, ws = G(z, noise_mode="random", return_ws=True)

        t_g = diffusion.sample_t(per_gpu_batch_size, device)
        y_fake_g = diffusion.diffuse(fake_img, t_g)

        # D is frozen in the generator update. We only need gradients through D with
        # respect to y_fake_g, not D parameter gradients. Using the unwrapped local module
        # avoids unnecessary DDP reducer bookkeeping.
        fake_logits_g = unwrap_model(D)(y_fake_g, t_g)

        g_loss = g_nonsaturating_loss(fake_logits_g)

        opt_G.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_G.step()

        # Apply path length regularization lazily.
        pl_val = torch.zeros([], device=device)
        if cfg.pl_weight > 0 and step % cfg.g_reg_interval == 0:
            pl_batch = max(1, per_gpu_batch_size // cfg.pl_batch_shrink)
            z_pl = torch.randn(pl_batch, cfg.z_dim, device=device)

            fake_pl, ws_pl = G(z_pl, noise_mode="random", return_ws=True)

            pl_penalty, pl_mean, _ = g_path_length_regularize(
                fake_pl,
                ws_pl,
                pl_mean,
                decay=cfg.pl_decay,
            )

            pl_loss = pl_penalty * cfg.pl_weight * cfg.g_reg_interval

            # DDP note:
            # Path length regularization is defined through gradients with respect to ws.
            # A zero-valued fake image term keeps all generator parameters connected to the
            # graph during this lazy regularization step without changing the objective.
            pl_loss = pl_loss + 0.0 * fake_pl.sum()

            opt_G.zero_grad(set_to_none=True)
            pl_loss.backward()
            opt_G.step()

            pl_val = pl_penalty.detach()

        update_ema(G_ema, G, cfg.batch_size, cfg.ema_kimg)

        # ----------------------------
        # Step III: Update adaptive diffusion.
        # ----------------------------
        if step % cfg.update_T_every == 0:
            rd, T_now = update_diffusion_T_ddp(diffusion, real_logits, rank, world_size)
        else:
            rd = None
            T_now = diffusion.T

        cur_nimg += cfg.batch_size
        cur_kimg = cur_nimg / 1000.0

        if is_main:
            pbar.n = cur_kimg
            pbar.refresh()

            if step % cfg.print_every == 0:
                with torch.no_grad():
                    real_score = real_logits.mean().item()
                    fake_score = fake_logits.mean().item()

                history["kimg"].append(cur_kimg)
                history["d_loss"].append(d_loss.item())
                history["g_loss"].append(g_loss.item())
                history["r1"].append(r1_val.item())
                history["pl"].append(pl_val.item())
                history["rd"].append(float(rd) if rd is not None else None)
                history["T"].append(T_now)
                history["real_score"].append(real_score)
                history["fake_score"].append(fake_score)

                pbar.set_description(
                    f"kimg {cur_kimg:.1f} | D {d_loss.item():.3f} | G {g_loss.item():.3f} | "
                    f"real {real_score:.3f} fake {fake_score:.3f} | T {T_now}"
                )

            cur_kimg_int = int(cur_kimg)

            if (
                cur_kimg_int > 0
                and cur_kimg_int % cfg.sample_every_kimg == 0
                and cur_kimg_int != last_sample_kimg
            ):
                last_sample_kimg = cur_kimg_int
                sample_path = save_samples(G_ema, cfg, device, cur_kimg, n=64, seed=42)
                print("sample saved:", sample_path)

            if (
                cur_kimg_int > 0
                and cur_kimg_int % cfg.save_every_kimg == 0
                and cur_kimg_int != last_save_kimg
            ):
                last_save_kimg = cur_kimg_int
                ckpt_path = save_checkpoint(cfg, G, D, G_ema, opt_G, opt_D, diffusion, pl_mean, cur_nimg, history)
                print("checkpoint saved:", ckpt_path)

    if is_main:
        pbar.close()
        ckpt_path = save_checkpoint(cfg, G, D, G_ema, opt_G, opt_D, diffusion, pl_mean, cur_nimg, history)
        sample_path = save_samples(G_ema, cfg, device, cur_nimg / 1000.0, n=64, seed=123)
        print("final checkpoint saved:", ckpt_path)
        print("final sample saved:", sample_path)
        print("Training finished.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
