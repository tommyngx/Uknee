"""The two standalone full-frame heatmap baselines exposed by Uknee."""

from .hrnet import HRNetLandmarkBaseline, HRNetW32
from .vitpose import ViTPoseLandmarkBaseline, ViTPoseS

__all__ = ["HRNetW32", "ViTPoseS", "HRNetLandmarkBaseline", "ViTPoseLandmarkBaseline"]
