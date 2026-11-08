"""Anatomical landmark schema shared by models, adapters, losses and metrics.

The default is the MESKO4GF2 knee schema (femur/tibia/fibula/patella, 129
landmarks). Another four-region schema is activated by pointing
``UKNEE_LANDMARK_SCHEMA`` at a dataset YAML that declares ``names``,
``region_keypoint_counts``, ``landmark_path_ranges`` and ``kpt_shape``;
``landmark.train`` sets it from ``--dataset``. The schema is resolved once at
import so every module, and every DDP worker (which inherits the environment),
sees the same constants.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


SCHEMA_ENV = "UKNEE_LANDMARK_SCHEMA"
# The trainers, plots and refinement heads are written for four anatomical
# regions; a different count must fail here rather than misbehave downstream.
SUPPORTED_NUM_REGIONS = 4


@dataclass(frozen=True)
class LandmarkSchema:
    """Ordered regions, their keypoint counts and the independent landmark paths."""

    names: tuple[str, ...]
    counts: tuple[int, ...]
    path_ranges: tuple[tuple[int, int], ...]

    @property
    def offsets(self) -> tuple[int, ...]:
        starts, total = [], 0
        for count in self.counts:
            starts.append(total)
            total += count
        return tuple(starts)

    @property
    def max_keypoints(self) -> int:
        return max(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def local_paths(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Per-region paths in region-local slot coordinates."""
        result = []
        for offset, count in zip(self.offsets, self.counts):
            result.append(
                tuple(
                    (start - offset, stop - offset)
                    for start, stop in self.path_ranges
                    if offset <= start and stop <= offset + count
                )
            )
        return tuple(result)


KNEE_SCHEMA = LandmarkSchema(
    names=("femur", "tibia", "fibula", "patella"),
    counts=(45, 51, 24, 9),
    path_ranges=((0, 45), (45, 86), (86, 91), (91, 96), (96, 120), (120, 129)),
)


def build_schema(
    names: Any,
    counts: Any,
    path_ranges: Any = None,
    kpt_shape: Any = None,
) -> LandmarkSchema:
    """Validate and freeze a region schema; ``path_ranges`` defaults to one path per region."""
    names = tuple(str(name) for name in names)
    counts = tuple(int(count) for count in counts)
    if len(names) != SUPPORTED_NUM_REGIONS or len(counts) != SUPPORTED_NUM_REGIONS:
        raise ValueError(
            f"Landmark schema needs exactly {SUPPORTED_NUM_REGIONS} regions; "
            f"got names={names!r}, region_keypoint_counts={counts!r}"
        )
    if min(counts) < 1:
        raise ValueError(f"region_keypoint_counts must be positive, got {counts}")
    offsets, total = [], 0
    for count in counts:
        offsets.append(total)
        total += count
    if kpt_shape is not None and tuple(int(v) for v in kpt_shape) != (max(counts), 3):
        raise ValueError(
            f"kpt_shape must be [max(region_keypoint_counts)={max(counts)}, 3], got {list(kpt_shape)}"
        )
    if path_ranges is None:
        path_ranges = [(offset, offset + count) for offset, count in zip(offsets, counts)]
    paths = tuple((int(start), int(stop)) for start, stop in path_ranges)
    cursor = 0
    for start, stop in paths:
        if start != cursor or stop <= start:
            raise ValueError(
                f"landmark_path_ranges must be contiguous and tile [0, {total}); got {list(paths)}"
            )
        cursor = stop
        inside = any(o <= start and stop <= o + c for o, c in zip(offsets, counts))
        if not inside:
            raise ValueError(f"landmark path ({start}, {stop}) crosses a region boundary")
    if cursor != total:
        raise ValueError(f"landmark_path_ranges end at {cursor} but the regions hold {total} landmarks")
    return LandmarkSchema(names=names, counts=counts, path_ranges=paths)


def schema_from_metadata(metadata: Mapping[str, Any]) -> LandmarkSchema | None:
    """Return the schema declared by a dataset YAML, or ``None`` if it declares none.

    A YAML describing the knee regions (same names and counts) always maps to
    ``KNEE_SCHEMA`` so existing knee datasets keep their six structure paths.
    """
    if "region_keypoint_counts" not in metadata:
        return None
    counts = tuple(int(value) for value in metadata["region_keypoint_counts"])
    names = metadata.get("names", {})
    if isinstance(names, Mapping):
        names = [names[index] for index in range(len(counts))]
    names = tuple(str(name) for name in names)
    declared_paths = metadata.get("landmark_path_ranges")
    if names == KNEE_SCHEMA.names and counts == KNEE_SCHEMA.counts:
        if declared_paths is not None:
            paths = tuple((int(start), int(stop)) for start, stop in declared_paths)
            if paths != KNEE_SCHEMA.path_ranges:
                warnings.warn(
                    f"Knee dataset declares landmark_path_ranges={list(paths)}; the fixed knee "
                    f"paths {list(KNEE_SCHEMA.path_ranges)} are used instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return KNEE_SCHEMA
    return build_schema(names, counts, declared_paths, metadata.get("kpt_shape"))


def load_schema(path: str | Path) -> LandmarkSchema | None:
    metadata = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return schema_from_metadata(metadata)


def _resolve_active_schema() -> LandmarkSchema:
    source = os.environ.get(SCHEMA_ENV, "").strip()
    if not source:
        return KNEE_SCHEMA
    schema = load_schema(source)
    return KNEE_SCHEMA if schema is None else schema


ACTIVE_SCHEMA = _resolve_active_schema()
REGION_NAMES = ACTIVE_SCHEMA.names
REGION_KEYPOINT_COUNTS = ACTIVE_SCHEMA.counts
REGION_OFFSETS = ACTIVE_SCHEMA.offsets
NUM_REGIONS = len(REGION_NAMES)
MAX_REGION_KEYPOINTS = ACTIVE_SCHEMA.max_keypoints
NUM_LANDMARKS = ACTIVE_SCHEMA.total
LANDMARK_PATH_RANGES = ACTIVE_SCHEMA.path_ranges
REGION_LOCAL_PATHS = ACTIVE_SCHEMA.local_paths
POINT_REGION_IDS = torch.tensor(
    [region for region, count in enumerate(REGION_KEYPOINT_COUNTS) for _ in range(count)],
    dtype=torch.long,
)


def validate_region_schema(nc: int, kpt_shape: tuple[int, int] | list[int]) -> None:
    """Fail fast when a dataset/model cannot represent the active region annotations."""
    if int(nc) != NUM_REGIONS:
        raise ValueError(f"Landmark schema requires nc={NUM_REGIONS}, got {nc}")
    if tuple(map(int, kpt_shape)) != (MAX_REGION_KEYPOINTS, 3):
        raise ValueError(
            f"Landmark schema {REGION_NAMES} requires kpt_shape=[{MAX_REGION_KEYPOINTS}, 3], got {kpt_shape}"
        )


def class_keypoint_mask(
    class_ids: torch.Tensor, max_keypoints: int = MAX_REGION_KEYPOINTS
) -> torch.Tensor:
    """Return valid local keypoint slots for each anatomical class."""
    class_ids = class_ids.long()
    if ((class_ids < 0) | (class_ids >= NUM_REGIONS)).any():
        raise ValueError(f"Region class IDs must be in [0, {NUM_REGIONS - 1}]")
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
        raise ValueError(f"Region class IDs must be in [0, {NUM_REGIONS - 1}]")
    masks = torch.zeros(
        class_ids.numel(),
        MAX_REGION_KEYPOINTS - order + 1,
        dtype=torch.bool,
        device=class_ids.device,
    )
    for row, class_id in enumerate(class_ids.tolist()):
        for start, stop in REGION_LOCAL_PATHS[class_id]:
            masks[row, start : max(start, stop - order + 1)] = True
    return masks


def objects_to_canonical(
    keypoints: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map arbitrary YOLO object rows to one best canonical anatomical target."""
    if keypoints.ndim != 3 or keypoints.shape[1] != MAX_REGION_KEYPOINTS:
        raise ValueError(
            f"Expected keypoints [N, {MAX_REGION_KEYPOINTS}, D], got {tuple(keypoints.shape)}"
        )
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
