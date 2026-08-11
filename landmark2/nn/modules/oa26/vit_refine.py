# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight ViT-style refinement block for OA26 pose experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTRefine(nn.Module):
    """Refine one feature map with lightweight global token mixing and preserve its shape."""

    def __init__(
        self,
        c: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        max_tokens: int = 4096,
    ):
        """Initialize a ViTPose-inspired refinement block."""
        super().__init__()
        c = int(c)
        embed_dim = int(embed_dim or c)
        num_heads = int(num_heads)
        depth = int(depth)
        max_tokens = int(max_tokens)
        if c <= 0 or embed_dim <= 0:
            raise ValueError("c and embed_dim must be positive.")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self.max_tokens = max_tokens
        self.use_checkpoint = bool(use_checkpoint)
        self.in_proj = nn.Conv2d(c, embed_dim, 1, 1)
        self.pos = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=max(int(embed_dim * float(mlp_ratio)), embed_dim),
                dropout=float(dropout),
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, c, 1, 1)

    def _pool_size(self, h: int, w: int) -> tuple[int, int]:
        """Return a pooled spatial size with token count at or below `max_tokens`."""
        if h * w <= self.max_tokens:
            return h, w
        scale = math.sqrt(self.max_tokens / float(h * w))
        ph, pw = max(1, int(math.floor(h * scale))), max(1, int(math.floor(w * scale)))
        while ph * pw > self.max_tokens:
            if ph >= pw and ph > 1:
                ph -= 1
            elif pw > 1:
                pw -= 1
            else:
                break
        return ph, pw

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer encoder blocks to token tensor `[B, HW, C]`."""
        if self.use_checkpoint and self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint

            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a refined feature map with the same `[B, C, H, W]` shape as input."""
        b, _, h, w = x.shape
        y = self.in_proj(x)
        pooled_h, pooled_w = self._pool_size(h, w)
        if (pooled_h, pooled_w) != (h, w):
            y = F.adaptive_avg_pool2d(y, (pooled_h, pooled_w))

        # [B, C, H, W] -> [B, H*W, C]
        y = y + self.pos(y)
        y = y.flatten(2).transpose(1, 2)
        y = self._encode(y)

        # [B, H*W, C] -> [B, C, H, W]
        y = y.transpose(1, 2).reshape(b, -1, pooled_h, pooled_w)
        if (pooled_h, pooled_w) != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return x + self.out_proj(y)
