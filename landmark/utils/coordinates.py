from __future__ import annotations

import torch


def normalize_pixel_coordinates(
    coordinates_xy: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert pixel (x, y) coordinates to unit coordinates in [0, 1]."""
    scale = coordinates_xy.new_tensor(
        [max(image_width - 1, 1), max(image_height - 1, 1)]
    )
    return coordinates_xy / scale


def denormalize_coordinates(
    coordinates_xy: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert unit (x, y) coordinates to zero-indexed pixel coordinates."""
    scale = coordinates_xy.new_tensor(
        [max(image_width - 1, 1), max(image_height - 1, 1)]
    )
    return coordinates_xy * scale


def unit_to_grid_sample(coordinates_xy: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] (x, y) coordinates to grid_sample's [-1, 1] space."""
    return coordinates_xy * 2.0 - 1.0


def create_local_sampling_grid(
    centers_xy: torch.Tensor,
    patch_size: int,
    feature_height: int,
    feature_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a grid around each centre and its half-width in unit coordinates.

    The grid has shape ``[B, N, P, P, 2]`` and keeps coordinate order ``(x, y)``.
    """
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    dtype, device = centers_xy.dtype, centers_xy.device
    offset = torch.arange(patch_size, dtype=dtype, device=device)
    offset = offset - (patch_size - 1) / 2.0
    offset_x = offset / max(feature_width - 1, 1)
    offset_y = offset / max(feature_height - 1, 1)
    yy, xx = torch.meshgrid(offset_y, offset_x, indexing="ij")
    offsets = torch.stack([xx, yy], dim=-1)
    unit_grid = centers_xy[:, :, None, None, :] + offsets[None, None]
    radius = centers_xy.new_tensor(
        [
            float(offset_x.abs().max()) if patch_size > 1 else 0.0,
            float(offset_y.abs().max()) if patch_size > 1 else 0.0,
        ]
    )
    return unit_to_grid_sample(unit_grid), radius


def soft_argmax_2d(
    heatmaps: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiably decode heatmaps into local coordinates in [-1, 1].

    Args:
        heatmaps: ``[..., H, W]`` unnormalised logits.
        temperature: positive softmax temperature.

    Returns:
        ``coordinates_xy`` with shape ``[..., 2]`` and maximum spatial
        probability as confidence with shape ``[...]``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    height, width = heatmaps.shape[-2:]
    probabilities = torch.softmax(heatmaps.flatten(-2) / temperature, dim=-1)
    xs = torch.linspace(-1, 1, width, dtype=heatmaps.dtype, device=heatmaps.device)
    ys = torch.linspace(-1, 1, height, dtype=heatmaps.dtype, device=heatmaps.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x = (probabilities * xx.flatten()).sum(dim=-1)
    y = (probabilities * yy.flatten()).sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    return torch.stack([x, y], dim=-1), confidence
