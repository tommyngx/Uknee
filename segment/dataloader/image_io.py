"""Shared image decoding contract for segmentation datasets."""

from __future__ import annotations

from os import PathLike

import cv2
import numpy as np


def read_rgb_image(path: str | PathLike[str]) -> np.ndarray | None:
    """Read a color image and expose the model-facing RGB channel order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


__all__ = ["read_rgb_image"]
