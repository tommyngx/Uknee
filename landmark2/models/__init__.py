"""The two standalone full-frame heatmap baselines exposed by Uknee."""

from .hrnet import HRNetLandmarkBaseline, HRNetW32
from .vitpose import ViTPoseLandmarkBaseline, ViTPoseS


def __getattr__(name: str):
    if name in {"HeatmapPose", "HeatmapPoseModel", "HeatmapPoseTrainer", "HeatmapPoseValidator"}:
        from . import heatmap_adapter

        return getattr(heatmap_adapter, name)
    raise AttributeError(name)


__all__ = [
    "HRNetW32",
    "ViTPoseS",
    "HRNetLandmarkBaseline",
    "ViTPoseLandmarkBaseline",
    "HeatmapPose",
    "HeatmapPoseModel",
    "HeatmapPoseTrainer",
    "HeatmapPoseValidator",
]
