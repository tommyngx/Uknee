from __future__ import annotations

import math

import torch
from torch import nn

from .heatmap_baseline import decode_global_heatmaps


def _sincos_position_2d(
    height: int,
    width: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic 2D absolute positions in raster token order."""
    frequencies_per_axis = max(math.ceil(channels / 4), 1)
    frequencies = torch.arange(
        frequencies_per_axis, dtype=torch.float32, device=device
    )
    frequencies = torch.exp(
        -math.log(10_000.0)
        * frequencies
        / max(frequencies_per_axis - 1, 1)
    )
    ys = torch.linspace(0, 1, height, dtype=torch.float32, device=device)
    xs = torch.linspace(0, 1, width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x_angles = xx[..., None] * frequencies
    y_angles = yy[..., None] * frequencies
    encoding = torch.cat(
        (x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()),
        dim=-1,
    )[..., :channels]
    return encoding.reshape(1, height * width, channels).to(dtype=dtype)


class ViTPoseLandmarkBaseline(nn.Module):
    """Compact ViTPose-style baseline with a transformer and deconvolution head."""

    def __init__(
        self,
        input_channels: int = 1,
        num_landmarks: int = 129,
        embed_dim: int = 192,
        patch_size: int = 16,
        depth: int = 6,
        attention_heads: int = 6,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        layer = nn.TransformerEncoderLayer(
            embed_dim,
            attention_heads,
            embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_landmarks, 1),
        )

    def forward(self, image, **_):
        features = self.patch_embed(image)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + _sincos_position_2d(
            height,
            width,
            channels,
            tokens.dtype,
            tokens.device,
        )
        tokens = self.norm(self.transformer(tokens))
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return decode_global_heatmaps(self.heatmap_head(features))
