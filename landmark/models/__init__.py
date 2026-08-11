"""Standalone canonical landmark models exposed by Uknee."""

from .hrnet import HRNetLandmarkBaseline, HRNetW32, HRNetW48
from .rtmo import RTMOKneePose
from .vitpose import ViTPoseB, ViTPoseLandmarkBaseline, ViTPoseS


def __getattr__(name: str):
    if name in {"HeatmapPose", "HeatmapPoseModel", "HeatmapPoseTrainer", "HeatmapPoseValidator"}:
        from . import heatmap_adapter

        return getattr(heatmap_adapter, name)
    raise AttributeError(name)


__all__ = [
    "HRNetW32",
    "HRNetW48",
    "ViTPoseS",
    "ViTPoseB",
    "RTMOKneePose",
    "HRNetLandmarkBaseline",
    "ViTPoseLandmarkBaseline",
    "HeatmapPose",
    "HeatmapPoseModel",
    "HeatmapPoseTrainer",
    "HeatmapPoseValidator",
]
