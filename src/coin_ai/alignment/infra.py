from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np
import kornia
import kornia.geometry as KG
from torch import nn, Tensor
from einops import rearrange, repeat
from torchvision.transforms import v2 as transforms
from kornia.utils import image_to_tensor


class DenseDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dino.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def preprocess(self, x: np.ndarray | Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            x = image_to_tensor(x, keepdim=False)
        elif x.ndim == 3:
            x = rearrange(x, "h w c -> 1 h w c")

        b, c, h, w = x.shape
        assert c == 3, c
        assert h % 14 == 0, h
        assert w % 14 == 0, w

        if x.dtype == torch.uint8:
            x = transforms.ToDtype(torch.float32, scale=True)(x)
        x = transforms.Grayscale(num_output_channels=3)(x)
        return x

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x = self.preprocess(x)
        b, c, h, w = x.shape
        x = x.to(self.device)
        x = self.dino.forward_features(x)["x_norm_patchtokens"]
        return rearrange(x, "b (h w) c -> b h w c", h=h // 14, w=w // 14)


class EmbeddingHead(nn.Module):
    def __init__(self, dim: int = 32, normalized: bool = True, pre_norm: bool = False):
        super().__init__()
        self.pre_norm = (
            nn.LayerNorm(384, elementwise_affine=False, bias=False)
            if pre_norm
            else nn.Identity()
        )
        self.linear = nn.Linear(384, dim)
        self.normalized = normalized

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.normalized:
            x = nn.functional.normalize(x, dim=-1)
        return x


class Embedder(nn.Module):
    def __init__(self, dim: int = 32, normalized: bool = True):
        super().__init__()
        self.dense_dino = DenseDino()
        self.head = EmbeddingHead(dim, normalized=normalized)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x = self.dense_dino(x)
        return self.head(x)
