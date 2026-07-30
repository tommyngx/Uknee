from __future__ import annotations

import torch
from torch import nn

from landmark.utils.coordinates import soft_argmax_2d


def _groups(channels: int) -> int:
    value = min(8, channels)
    while channels % value:
        value -= 1
    return value


class QueryConditionedLocalHeatmapHead(nn.Module):
    def __init__(self, query_dim: int, temperature: float = 0.1):
        super().__init__()
        hidden = max(query_dim // 2, 8)
        self.temperature = temperature
        self.query_to_film = nn.Linear(query_dim, query_dim * 2)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, 3, padding=1),
            nn.GroupNorm(_groups(query_dim), query_dim),
            nn.GELU(),
            nn.Conv2d(query_dim, hidden, 3, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(
        self,
        local_patches_high: torch.Tensor,
        queries: torch.Tensor,
        reference_coordinates: torch.Tensor,
        patch_radius_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, landmarks, channels, height, width = local_patches_high.shape
        gamma, beta = self.query_to_film(queries).chunk(2, dim=-1)
        conditioned = (
            local_patches_high * (1 + gamma[..., None, None])
            + beta[..., None, None]
        )
        heatmaps = self.heatmap_head(
            conditioned.reshape(batch * landmarks, channels, height, width)
        ).view(batch, landmarks, height, width)
        offsets, confidence = soft_argmax_2d(heatmaps, self.temperature)
        final = (reference_coordinates + offsets * patch_radius_xy).clamp(0, 1)
        return {
            "local_heatmaps": heatmaps,
            "local_offsets": offsets,
            "final_coordinates": final,
            "confidence": confidence,
        }
