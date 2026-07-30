from __future__ import annotations

import torch

from landmark.data.yolo_pose import BONE_NAMES, POINT_BONE_IDS

from .coordinates import denormalize_coordinates


def radial_errors(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    predicted_px = denormalize_coordinates(predicted_xy, image_height, image_width)
    target_px = denormalize_coordinates(target_xy, image_height, image_width)
    return torch.linalg.vector_norm(predicted_px - target_px, dim=-1)


def mean_radial_error(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    image_height: int,
    image_width: int,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    errors = radial_errors(predicted_xy, target_xy, image_height, image_width)
    if visibility is None:
        return errors.mean()
    weights = visibility.to(errors.dtype)
    return (errors * weights).sum() / weights.sum().clamp_min(1)


def per_landmark_radial_error(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    image_height: int,
    image_width: int,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    errors = radial_errors(predicted_xy, target_xy, image_height, image_width)
    if visibility is None:
        return errors.mean(dim=0)
    weights = visibility.to(errors.dtype)
    return (errors * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1)


def pck(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    threshold_pixels: float,
    image_height: int,
    image_width: int,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    correct = (
        radial_errors(predicted_xy, target_xy, image_height, image_width)
        <= threshold_pixels
    ).float()
    if visibility is None:
        return correct.mean()
    weights = visibility.to(correct.dtype)
    return (correct * weights).sum() / weights.sum().clamp_min(1)


def landmark_metrics(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    visibility: torch.Tensor,
    image_height: int,
    image_width: int,
) -> dict[str, float]:
    errors = radial_errors(predicted_xy, target_xy, image_height, image_width)
    selected = errors[visibility.bool()]
    if selected.numel() == 0:
        keys = ("mre", "median", "p95", "pck2", "pck4", "pck8", "failure_gt_8")
        return {key: float("nan") for key in keys}
    metrics = {
        "mre": selected.mean().item(),
        "median": selected.median().item(),
        "p95": torch.quantile(selected.float(), 0.95).item(),
        "pck2": (selected <= 2).float().mean().item(),
        "pck4": (selected <= 4).float().mean().item(),
        "pck8": (selected <= 8).float().mean().item(),
        "failure_gt_8": (selected > 8).float().mean().item(),
    }
    bone_ids = POINT_BONE_IDS.to(errors.device)
    for bone_id, bone_name in enumerate(BONE_NAMES):
        bone_mask = visibility.bool() & (bone_ids[None] == bone_id)
        bone_errors = errors[bone_mask]
        metrics[f"mre_{bone_name}"] = (
            bone_errors.mean().item() if bone_errors.numel() else float("nan")
        )
    return metrics
