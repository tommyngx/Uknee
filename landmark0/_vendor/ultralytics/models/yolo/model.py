# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Local-only YOLO frontend narrowed to the detect and pose tasks used by Uknee."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models.yolo import detect, pose
from ultralytics.nn.tasks import DetectionModel, PoseModel


class YOLO(Model):
    """Ultralytics YOLO frontend exposing only detection and pose estimation."""

    def __init__(self, model: str | Path = "yolo26n-pose.pt", task: str | None = None, verbose: bool = False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map the two retained tasks to their model, trainer, validator and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": detect.DetectionTrainer,
                "validator": detect.DetectionValidator,
                "predictor": detect.DetectionPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": pose.PoseTrainer,
                "validator": pose.PoseValidator,
                "predictor": pose.PosePredictor,
            },
        }


__all__ = ("YOLO",)
