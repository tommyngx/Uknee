"""Deterministic inference preprocessing shared by export and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    target_height: int
    target_width: int
    pad_top: int
    pad_left: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def resolve_target_hw(img_size) -> tuple[int, int]:
    if isinstance(img_size, (list, tuple)):
        if len(img_size) != 2:
            raise ValueError(f"img_size must contain [height, width], got {img_size}")
        return int(img_size[0]), int(img_size[1])
    size = int(img_size)
    return size, size


def letterbox_array(
    image: np.ndarray,
    target_hw: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
    pad_value: int | float = 0,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize with unchanged aspect ratio and center-pad to the network canvas."""
    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D or HWC image, got shape={image.shape}")
    original_height, original_width = image.shape[:2]
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    if min(original_height, original_width, target_height, target_width) <= 0:
        raise ValueError(
            f"Image and target dimensions must be positive, got image={image.shape}, target={target_hw}"
        )

    scale = min(target_height / original_height, target_width / original_width)
    resized_height = max(1, min(target_height, int(round(original_height * scale))))
    resized_width = max(1, min(target_width, int(round(original_width * scale))))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    border_value = pad_value if image.ndim == 2 else tuple([pad_value] * image.shape[2])
    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
    transform = LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        target_height=target_height,
        target_width=target_width,
        pad_top=pad_top,
        pad_left=pad_left,
    )
    return canvas, transform


def restore_letterbox_mask(mask: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Remove network padding and restore a label map to the source image size."""
    mask = np.asarray(mask)
    top = transform.pad_top
    left = transform.pad_left
    cropped = mask[
        top : top + transform.resized_height,
        left : left + transform.resized_width,
    ]
    return cv2.resize(
        cropped,
        (transform.original_width, transform.original_height),
        interpolation=cv2.INTER_NEAREST,
    )


__all__ = [
    "LetterboxTransform",
    "letterbox_array",
    "resolve_target_hw",
    "restore_letterbox_mask",
]
