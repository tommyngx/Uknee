"""Pose-only data pipeline with lazy exports to avoid import cycles."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "build_dataloader": ".build",
    "build_yolo_dataset": ".build",
    "load_inference_source": ".build",
    "YOLODataset": ".dataset",
    "PreparedDataset": ".prepare",
    "prepare_dataset": ".prepare",
    "LANDMARK_PATH_RANGES": ".schema",
    "MAX_REGION_KEYPOINTS": ".schema",
    "NUM_LANDMARKS": ".schema",
    "NUM_REGIONS": ".schema",
    "POINT_REGION_IDS": ".schema",
    "REGION_KEYPOINT_COUNTS": ".schema",
    "REGION_NAMES": ".schema",
    "REGION_OFFSETS": ".schema",
    "class_keypoint_mask": ".schema",
    "class_path_masks": ".schema",
    "objects_to_canonical": ".schema",
    "validate_region_schema": ".schema",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, package=__name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
