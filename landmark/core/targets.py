# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Heatmap target helpers for OA26 pose experiments."""

from __future__ import annotations

import torch


# Uknee modification, 2026-08-10: MESKO stores four class-local object rows,
# while the OA26 auxiliary branch predicts one canonical image-level heatmap.
REGION_COUNTS = (45, 51, 24, 9)
REGION_OFFSETS = (0, 45, 96, 120)


def extract_image_keypoints(
    batch: dict[str, torch.Tensor],
    batch_size: int,
    num_keypoints: int,
    image_size: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one normalized-label landmark set per image as image-space xy plus visibility mask."""
    gt_xy = torch.zeros(batch_size, num_keypoints, 2, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, num_keypoints, device=device, dtype=torch.bool)
    keypoints = batch.get("keypoints")
    batch_idx = batch.get("batch_idx")
    if keypoints is None or batch_idx is None or keypoints.numel() == 0:
        return gt_xy, valid

    keypoints = keypoints.to(device=device, dtype=dtype).clone()
    batch_idx = batch_idx.to(device=device).long().flatten()
    h, w = image_size.to(device=device, dtype=dtype)
    keypoints[..., 0] *= w
    keypoints[..., 1] *= h
    n = min(num_keypoints, keypoints.shape[1])

    for image_i in range(batch_size):
        object_ids = (batch_idx == image_i).nonzero(as_tuple=False).flatten()
        if object_ids.numel() == 0:
            continue
        sample = keypoints[object_ids[0], :n]
        gt_xy[image_i, :n] = sample[:, :2]
        valid[image_i, :n] = sample[:, 2] > 0 if sample.shape[-1] == 3 else True
    return gt_xy, valid


def extract_canonical_image_keypoints(
    batch: dict[str, torch.Tensor],
    batch_size: int,
    image_size: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map four MESKO class rows to one ordered 129-point target per image."""
    gt_xy = torch.zeros(batch_size, sum(REGION_COUNTS), 2, device=device, dtype=dtype)
    valid = torch.zeros(batch_size, sum(REGION_COUNTS), device=device, dtype=torch.bool)
    keypoints, batch_idx, classes = (
        batch.get("keypoints"),
        batch.get("batch_idx"),
        batch.get("cls"),
    )
    if keypoints is None or batch_idx is None or classes is None or keypoints.numel() == 0:
        return gt_xy, valid

    keypoints = keypoints.to(device=device, dtype=dtype)
    batch_idx = batch_idx.to(device=device).long().flatten()
    classes = classes.to(device=device).long().flatten()
    h, w = image_size.to(device=device, dtype=dtype)
    for image_i in range(batch_size):
        image_rows = torch.nonzero(batch_idx == image_i, as_tuple=False).flatten()
        for class_id, (offset, count) in enumerate(zip(REGION_OFFSETS, REGION_COUNTS)):
            rows = image_rows[classes[image_rows] == class_id]
            if rows.numel() > 1:
                raise ValueError(
                    "OA26 canonical auxiliary targets support one instance per class; "
                    "disable mosaic/mixup/cutmix for V1/V9 training."
                )
            if rows.numel() == 0:
                continue
            sample = keypoints[rows[0], :count]
            gt_xy[image_i, offset : offset + count, 0] = sample[:, 0] * w
            gt_xy[image_i, offset : offset + count, 1] = sample[:, 1] * h
            valid[image_i, offset : offset + count] = sample[:, 2] > 0
    return gt_xy, valid


def gaussian_heatmap_targets(
    gt_xy: torch.Tensor,
    valid: torch.Tensor,
    heatmap_hw: tuple[int, int],
    image_size: torch.Tensor,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Generate Gaussian heatmaps from image-space keypoints."""
    h, w = heatmap_hw
    b, k = valid.shape
    device, dtype = gt_xy.device, gt_xy.dtype
    image_h, image_w = image_size.to(device=device, dtype=dtype).clamp(min=1)
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")

    x0 = (gt_xy[..., 0] * w / image_w).clamp(0, w - 1).view(b, k, 1, 1)
    y0 = (gt_xy[..., 1] * h / image_h).clamp(0, h - 1).view(b, k, 1, 1)
    dist = (x_grid.view(1, 1, h, w) - x0).pow(2) + (y_grid.view(1, 1, h, w) - y0).pow(2)
    target = torch.exp(-dist / (2 * max(float(sigma), 1e-6) ** 2))
    return target * valid.view(b, k, 1, 1).to(dtype)
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SimCC target and decode helpers for OA26 pose experiments."""


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
