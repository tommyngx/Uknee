"""ViTPose-S full-frame heatmap baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, channels: int, ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * ratio
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTPoseS(nn.Module):
    """Small ViTPose encoder (384/12/6) with a deconvolution heatmap head."""

    def __init__(
        self,
        input_channels: int = 3,
        num_landmarks: int = 129,
        image_size: int = 640,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        attention_heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, patch_size, patch_size)
        grid = max(image_size // patch_size, 1)
        self.position = nn.Parameter(torch.zeros(1, embed_dim, grid, grid))
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, attention_heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_landmarks, 1),
        )
        nn.init.trunc_normal_(self.position, std=0.02)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(image)
        batch, channels, height, width = features.shape
        position = F.interpolate(self.position, (height, width), mode="bicubic", align_corners=False)
        tokens = (features + position).flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
        features = self.norm(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        return self.heatmap_head(features)


ViTPoseLandmarkBaseline = ViTPoseS

__all__ = ["ViTPoseS", "ViTPoseLandmarkBaseline"]
