# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SimCC target and decode helpers for OA26 pose experiments."""

from __future__ import annotations

import torch


def gaussian_simcc_targets(
    gt_xy: torch.Tensor,
    valid: torch.Tensor,
    image_size: torch.Tensor,
    x_bins: int,
    y_bins: int,
    sigma: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert image-space keypoints to soft x/y classification targets."""
    device, dtype = gt_xy.device, gt_xy.dtype
    image_h, image_w = image_size.to(device=device, dtype=dtype).clamp(min=1)
    x_centers = (gt_xy[..., 0] / image_w * (x_bins - 1)).clamp(0, x_bins - 1)
    y_centers = (gt_xy[..., 1] / image_h * (y_bins - 1)).clamp(0, y_bins - 1)

    x_grid = torch.arange(x_bins, device=device, dtype=dtype).view(1, 1, x_bins)
    y_grid = torch.arange(y_bins, device=device, dtype=dtype).view(1, 1, y_bins)
    denom = 2 * max(float(sigma), 1e-6) ** 2
    target_x = torch.exp(-(x_grid - x_centers.unsqueeze(-1)).pow(2) / denom)
    target_y = torch.exp(-(y_grid - y_centers.unsqueeze(-1)).pow(2) / denom)
    target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return target_x, target_y, valid


def decode_simcc_logits(x_logits: torch.Tensor, y_logits: torch.Tensor, split_ratio: float = 2.0) -> torch.Tensor:
    """Decode SimCC logits to `[B, K, 3]` keypoints in image coordinates."""
    px = x_logits.softmax(dim=-1)
    py = y_logits.softmax(dim=-1)
    x_conf, x_idx = px.max(dim=-1)
    y_conf, y_idx = py.max(dim=-1)
    xy = torch.stack((x_idx.to(x_logits.dtype), y_idx.to(y_logits.dtype)), dim=-1) / max(float(split_ratio), 1e-6)
    conf = (x_conf * y_conf).sqrt().unsqueeze(-1)
    return torch.cat((xy, conf), dim=-1)
