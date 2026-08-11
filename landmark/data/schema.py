"""MESKO4GF2 pose schema shared by models, adapters, losses and metrics."""

from __future__ import annotations

import torch


REGION_NAMES = ("femur", "tibia", "fibula", "patella")
REGION_KEYPOINT_COUNTS = (45, 51, 24, 9)
REGION_OFFSETS = (0, 45, 96, 120)
NUM_REGIONS = len(REGION_NAMES)
MAX_REGION_KEYPOINTS = 51
NUM_LANDMARKS = sum(REGION_KEYPOINT_COUNTS)
LANDMARK_PATH_RANGES = (
    (0, 45),
    (45, 86),
    (86, 91),
    (91, 96),
    (96, 120),
    (120, 129),
)
POINT_REGION_IDS = torch.tensor(
    [region for region, count in enumerate(REGION_KEYPOINT_COUNTS) for _ in range(count)],
    dtype=torch.long,
)


def validate_region_schema(nc: int, kpt_shape: tuple[int, int] | list[int]) -> None:
    """Fail fast when a dataset/model cannot represent the MESKO annotations."""
    if int(nc) != NUM_REGIONS:
        raise ValueError(f"MESKO4GF2 requires nc={NUM_REGIONS}, got {nc}")
    if tuple(map(int, kpt_shape)) != (MAX_REGION_KEYPOINTS, 3):
        raise ValueError(
            f"MESKO4GF2 requires kpt_shape=[{MAX_REGION_KEYPOINTS}, 3], got {kpt_shape}"
        )


def class_keypoint_mask(
    class_ids: torch.Tensor, max_keypoints: int = MAX_REGION_KEYPOINTS
) -> torch.Tensor:
    """Return valid local keypoint slots for each anatomical class."""
    class_ids = class_ids.long()
    if ((class_ids < 0) | (class_ids >= NUM_REGIONS)).any():
        raise ValueError("Region class IDs must be in [0, 3]")
    counts = class_ids.new_tensor(REGION_KEYPOINT_COUNTS)
    if max_keypoints < 1:
        raise ValueError("max_keypoints must be positive")
    slots = torch.arange(max_keypoints, device=class_ids.device)
    return slots.view(*((1,) * class_ids.ndim), -1) < counts[class_ids].unsqueeze(-1)


def class_path_masks(class_ids: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Return starts that remain inside each anatomical path for pairs/triples."""
    if order not in {2, 3}:
        raise ValueError("order must be 2 or 3")
    class_ids = class_ids.long().flatten()
    if ((class_ids < 0) | (class_ids >= NUM_REGIONS)).any():
        raise ValueError("Region class IDs must be in [0, 3]")
    masks = torch.zeros(
        class_ids.numel(),
        MAX_REGION_KEYPOINTS - order + 1,
        dtype=torch.bool,
        device=class_ids.device,
    )
    local_paths = {
        0: ((0, 45),),
        1: ((0, 41), (41, 46), (46, 51)),
        2: ((0, 24),),
        3: ((0, 9),),
    }
    for row, class_id in enumerate(class_ids.tolist()):
        for start, stop in local_paths[class_id]:
            masks[row, start : max(start, stop - order + 1)] = True
    return masks


def objects_to_canonical(
    keypoints: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map arbitrary YOLO object rows to one best 129-point anatomical target."""
    if keypoints.ndim != 3 or keypoints.shape[1] != MAX_REGION_KEYPOINTS:
        raise ValueError(f"Expected keypoints [N, 51, D], got {tuple(keypoints.shape)}")
    if keypoints.shape[-1] < 2:
        raise ValueError("Keypoints require at least x and y coordinates")
    if class_ids.numel() != keypoints.shape[0]:
        raise ValueError("class_ids and keypoints must contain the same number of objects")

    coordinates = keypoints.new_zeros((NUM_LANDMARKS, 2))
    confidence = keypoints.new_zeros(NUM_LANDMARKS)
    class_ids = class_ids.long().flatten()
    scores = (
        keypoints.new_ones(keypoints.shape[0])
        if scores is None
        else scores.to(device=keypoints.device, dtype=keypoints.dtype).flatten()
    )
    for class_id, (offset, count) in enumerate(zip(REGION_OFFSETS, REGION_KEYPOINT_COUNTS)):
        candidates = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        selected = candidates[scores[candidates].argmax()]
        coordinates[offset : offset + count] = keypoints[selected, :count, :2]
        if keypoints.shape[-1] >= 3:
            confidence[offset : offset + count] = keypoints[selected, :count, 2].clamp(0, 1)
        else:
            confidence[offset : offset + count] = scores[selected].clamp(0, 1)
    return coordinates, confidence
