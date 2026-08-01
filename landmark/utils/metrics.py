from __future__ import annotations

import torch
from torch.nn import functional as F

from landmark.data.yolo_pose import BONE_NAMES, POINT_BONE_IDS
from landmark.models.metadata import LANDMARK_PATH_RANGES, TOPOLOGY_EDGES

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


def _project_to_polyline_arclength(
    points: torch.Tensor,
    polyline: torch.Tensor,
) -> torch.Tensor:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = vectors.square().sum(dim=-1).clamp_min(1.0e-8)
    relative = points[:, None] - starts[None]
    fractions = (
        (relative * vectors[None]).sum(dim=-1) / squared_lengths[None]
    ).clamp(0, 1)
    projections = starts[None] + fractions[..., None] * vectors[None]
    distances = (points[:, None] - projections).square().sum(dim=-1)
    closest = distances.argmin(dim=-1)
    segment_lengths = squared_lengths.sqrt()
    cumulative = torch.cat(
        [segment_lengths.new_zeros(1), segment_lengths.cumsum(dim=0)[:-1]]
    )
    return (
        cumulative[closest]
        + fractions.gather(1, closest[:, None]).squeeze(1)
        * segment_lengths[closest]
    )


def topology_metrics(
    predicted_xy: torch.Tensor,
    target_xy: torch.Tensor,
    visibility: torch.Tensor,
    image_height: int,
    image_width: int,
) -> dict[str, float]:
    """Measure sequence order, duplicates and local contour geometry."""
    scale = predicted_xy.new_tensor([image_width - 1, image_height - 1])
    predicted_px = predicted_xy * scale
    target_px = target_xy * scale

    inversion_count = 0
    ordered_pair_count = 0
    for batch_index in range(predicted_xy.shape[0]):
        for start, stop in LANDMARK_PATH_RANGES:
            valid = visibility[batch_index, start:stop].bool()
            if valid.sum() < 2:
                continue
            predicted_path = predicted_px[batch_index, start:stop][valid]
            target_path = target_px[batch_index, start:stop][valid]
            positions = _project_to_polyline_arclength(
                predicted_path, target_path
            )
            # Count every pair, not only adjacent points. This is the
            # normalized Kendall inversion rate and catches long-range swaps
            # that an adjacent-only metric can miss.
            reversed_or_tied = positions[:, None] >= positions[None, :]
            pair_mask = torch.triu(
                torch.ones_like(reversed_or_tied, dtype=torch.bool), diagonal=1
            )
            inversion_count += int((reversed_or_tied & pair_mask).sum())
            point_count = positions.numel()
            ordered_pair_count += point_count * (point_count - 1) // 2

    edge_start = torch.tensor(
        [start for start, _ in TOPOLOGY_EDGES], device=predicted_xy.device
    )
    edge_end = torch.tensor(
        [end for _, end in TOPOLOGY_EDGES], device=predicted_xy.device
    )
    predicted_delta = predicted_px[:, edge_end] - predicted_px[:, edge_start]
    target_delta = target_px[:, edge_end] - target_px[:, edge_start]
    predicted_length = torch.linalg.vector_norm(predicted_delta, dim=-1)
    target_length = torch.linalg.vector_norm(target_delta, dim=-1)
    valid_edges = (
        visibility[:, edge_start].bool() & visibility[:, edge_end].bool()
    )
    if valid_edges.any():
        relative_length_error = (
            (predicted_length - target_length).abs()
            / target_length.clamp_min(1.0)
        )[valid_edges].mean()
        cosine = F.cosine_similarity(
            predicted_delta[valid_edges], target_delta[valid_edges], dim=-1
        ).clamp(-1, 1)
        direction_error = torch.rad2deg(torch.acos(cosine)).mean()
        duplicate_rate = (predicted_length[valid_edges] <= 0.5).float().mean()
    else:
        relative_length_error = predicted_xy.new_tensor(float("nan"))
        direction_error = predicted_xy.new_tensor(float("nan"))
        duplicate_rate = predicted_xy.new_tensor(float("nan"))

    return {
        "order_inversion_rate": (
            inversion_count / ordered_pair_count
            if ordered_pair_count
            else float("nan")
        ),
        "adjacent_duplicate_rate": float(duplicate_rate),
        "edge_length_relative_error": float(relative_length_error),
        "direction_error_degrees": float(direction_error),
    }


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
        keys = (
            "mre",
            "median",
            "p95",
            "pck2",
            "pck4",
            "pck8",
            "failure_gt_8",
            "order_inversion_rate",
            "adjacent_duplicate_rate",
            "edge_length_relative_error",
            "direction_error_degrees",
        )
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
    metrics.update(
        topology_metrics(
            predicted_xy,
            target_xy,
            visibility,
            image_height,
            image_width,
        )
    )
    return metrics
