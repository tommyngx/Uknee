# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""ROI-wide heatmap localization for OA26 per-region refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class OA26RegionLocalizationHead(nn.Module):
    """Localize every landmark over the complete region ROI using query/image similarity maps."""

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 0.1,
        use_coarse_prior: bool = False,
        coarse_prior_sigma: float = 0.25,
        coarse_prior_gain: float = 0.5,
    ):
        """Initialize query/image projections and optional weak coarse-coordinate prior."""
        super().__init__()
        self.temperature = float(temperature)
        self.use_coarse_prior = bool(use_coarse_prior)
        self.coarse_prior_sigma = float(coarse_prior_sigma)
        self.coarse_prior_gain = float(coarse_prior_gain)
        self.query_projection = nn.Linear(d_model, d_model)
        self.image_projection = nn.Linear(d_model, d_model)

    @staticmethod
    def coordinate_grid(
        height: int, width: int, reference: torch.Tensor
    ) -> torch.Tensor:
        """Return flattened ROI bin-center xy coordinates normalized to [0, 1]."""
        ys = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) / max(height, 1)
        xs = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) / max(width, 1)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)

    def forward(
        self,
        landmark_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        roi_hw: tuple[int, int],
        coarse_xy_roi: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return logits, probability heatmaps, and ROI-normalized soft-argmax coordinates."""
        height, width = roi_hw
        queries = self.query_projection(landmark_tokens)
        image = self.image_projection(image_tokens)
        logits = torch.matmul(queries, image.transpose(-1, -2)) / math.sqrt(max(queries.shape[-1], 1))
        if self.use_coarse_prior:
            grid = self.coordinate_grid(height, width, logits)
            distance = (grid.view(1, 1, -1, 2) - coarse_xy_roi.unsqueeze(2)).pow(2).sum(dim=-1)
            sigma = max(self.coarse_prior_sigma, 1e-6)
            logits = logits - self.coarse_prior_gain * distance / (2 * sigma**2)
        probability = (logits / max(self.temperature, 1e-6)).softmax(dim=-1)
        probability = probability * valid_mask.unsqueeze(-1).to(probability.dtype)
        grid = self.coordinate_grid(height, width, probability)
        refined_xy_roi = torch.matmul(probability, grid)
        shape = (*probability.shape[:2], height, width)
        return logits.view(shape), probability.view(shape), refined_xy_roi
