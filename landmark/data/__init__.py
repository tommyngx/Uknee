"""Pose-only data pipeline and canonical MESKO4GF2 schema."""

from .build import build_dataloader, build_yolo_dataset, load_inference_source
from .dataset import YOLODataset
from .prepare import PreparedDataset, prepare_dataset
from .schema import (
    LANDMARK_PATH_RANGES,
    MAX_REGION_KEYPOINTS,
    NUM_LANDMARKS,
    NUM_REGIONS,
    POINT_REGION_IDS,
    REGION_KEYPOINT_COUNTS,
    REGION_NAMES,
    REGION_OFFSETS,
    class_keypoint_mask,
    class_path_masks,
    objects_to_canonical,
    validate_region_schema,
)

__all__ = [
    "LANDMARK_PATH_RANGES",
    "MAX_REGION_KEYPOINTS",
    "NUM_LANDMARKS",
    "NUM_REGIONS",
    "POINT_REGION_IDS",
    "PreparedDataset",
    "REGION_KEYPOINT_COUNTS",
    "REGION_NAMES",
    "REGION_OFFSETS",
    "YOLODataset",
    "build_dataloader",
    "build_yolo_dataset",
    "class_keypoint_mask",
    "class_path_masks",
    "load_inference_source",
    "objects_to_canonical",
    "prepare_dataset",
    "validate_region_schema",
]
