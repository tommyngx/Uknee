"""YOLO-compatible results and standardized pose-report exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from landmark.data.schema import NUM_LANDMARKS, objects_to_canonical
from landmark.core.plotting import plot_dashboard_pose, plot_pose_metrics, plot_validation_samples


@dataclass
class KneePoseResult:
    """Delegate standard result fields while exposing canonical landmarks."""

    raw: Any
    landmarks_xy: torch.Tensor
    landmark_confidence: torch.Tensor

    def __getattr__(self, name: str):
        return getattr(self.raw, name)

    @property
    def boxes_xyxy(self) -> torch.Tensor:
        return self.raw.boxes.xyxy

    @property
    def scores(self) -> torch.Tensor:
        return self.raw.boxes.conf

    @property
    def class_ids(self) -> torch.Tensor:
        return self.raw.boxes.cls.long()


def adapt_yolo_result(result: Any) -> KneePoseResult:
    """Add a normalized 129-landmark view to one Ultralytics Result."""
    device = result.boxes.xyxy.device if result.boxes is not None else torch.device("cpu")
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        xy = torch.zeros(NUM_LANDMARKS, 2, device=device)
        confidence = torch.zeros(NUM_LANDMARKS, device=device)
    else:
        xy, confidence = objects_to_canonical(
            result.keypoints.data, result.boxes.cls, scores=result.boxes.conf
        )
        height, width = result.orig_shape
        xy = (xy / xy.new_tensor([max(width, 1), max(height, 1)])).clamp(0, 1)
    return KneePoseResult(result, xy, confidence)


def plot_landmark_curves(csv_file: str, output_png: str | None = None, pixel_spacing: float = 0.1, **_: Any):
    """Backward-compatible alias for the standardized dashboard."""
    return plot_dashboard_pose(csv_file, output_png, pixel_spacing=pixel_spacing)


__all__ = [
    "KneePoseResult",
    "adapt_yolo_result",
    "plot_dashboard_pose",
    "plot_landmark_curves",
    "plot_pose_metrics",
    "plot_validation_samples",
]
