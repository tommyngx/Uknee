# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Utilities used only by the OA26 per-region refinement experiment."""

from .region_schema import (
    REGION_KEYPOINT_COUNTS,
    SOURCE_KEYPOINT_RANGES,
    MAX_REGION_KEYPOINTS,
    NUM_REGIONS,
    REGION_NAMES,
    class_keypoint_mask,
    validate_region_schema,
)
from .loss import OA26RegionRefinePoseLoss
from .training_plot_v9 import (
    plot_v9_performance_on_epoch_end,
    render_pose_detection_performance,
    render_v9_training_dashboard,
)

__all__ = (
    "MAX_REGION_KEYPOINTS",
    "NUM_REGIONS",
    "OA26RegionRefinePoseLoss",
    "REGION_KEYPOINT_COUNTS",
    "REGION_NAMES",
    "SOURCE_KEYPOINT_RANGES",
    "class_keypoint_mask",
    "plot_v9_performance_on_epoch_end",
    "render_pose_detection_performance",
    "render_v9_training_dashboard",
    "validate_region_schema",
)
