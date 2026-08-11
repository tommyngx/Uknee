# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Landmark query encoding for OA26 per-region refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class OA26RegionQueryEncoder(nn.Module):
    """Encode ROI coordinates, confidence, local landmark identity, and anatomical region identity."""

    def __init__(
        self,
        max_landmarks: int,
        num_regions: int,
        d_model: int = 128,
        coord_fourier_bands: int = 6,
    ):
        """Initialize metadata projections and learnable identity embeddings."""
        super().__init__()
        self.coord_fourier_bands = int(coord_fourier_bands)
        self.coord_projection = nn.Sequential(
            nn.Linear(4 * self.coord_fourier_bands, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.conf_projection = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.landmark_embedding = nn.Embedding(max_landmarks, d_model)
        self.region_embedding = nn.Embedding(num_regions, d_model)

    def _fourier(self, xy: torch.Tensor) -> torch.Tensor:
        """Return sine/cosine Fourier features for ROI-normalized x/y coordinates."""
        frequencies = xy.new_tensor([2.0**index * math.pi for index in range(self.coord_fourier_bands)])
        angles = xy.unsqueeze(-1) * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def forward(
        self,
        coarse_xy_roi: torch.Tensor,
        confidence: torch.Tensor,
        region_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return M x Kmax x d_model queries with padded landmarks zeroed."""
        m, k = coarse_xy_roi.shape[:2]
        local_ids = torch.arange(k, device=coarse_xy_roi.device).view(1, k)
        queries = self.coord_projection(self._fourier(coarse_xy_roi))
        queries = queries + self.conf_projection(confidence.unsqueeze(-1))
        queries = queries + self.landmark_embedding(local_ids)
        queries = queries + self.region_embedding(region_ids).view(m, 1, -1)
        return queries.masked_fill(~valid_mask.unsqueeze(-1), 0)
