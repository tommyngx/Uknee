"""Shared-weight global/local RWKV-UNet V6b for osteophyte segmentation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .RWKV_UNetV6a import RWKV_UNetV6a
from .RWKV_UNetV6 import (
    DEFAULT_DEPTHS, DEFAULT_EMBED_DIMS, DEFAULT_EXP_RATIOS,
    DEFAULT_MATRIX_STATE_STAGES, DEFAULT_NUM_HEADS,
)

SOURCE_SIZE = (1024, 640)
GLOBAL_SIZE = (720, 448)
LOCAL_SIZE = 640


def compose_global_local_logits(global_logits, local_logits, output_size, crop_y0, crop_x0):
    """Map global logits to source coordinates and replace the local window."""
    output_h, output_w = (int(value) for value in output_size)
    local_h, local_w = local_logits.shape[-2:]
    if crop_y0 < 0 or crop_x0 < 0 or crop_y0 + local_h > output_h or crop_x0 + local_w > output_w:
        raise ValueError("Local logits do not fit inside the requested output canvas")
    composed = F.interpolate(
        global_logits, size=(output_h, output_w), mode="bilinear", align_corners=False
    ).clone()
    composed[..., crop_y0:crop_y0 + local_h, crop_x0:crop_x0 + local_w] = local_logits
    return composed


class RWKV_UNetV6b(nn.Module):
    """Two spatial views evaluated by one and only one V6a segmentation core."""

    def __init__(
        self, input_channels: int = 3, num_classes: int = 5, stem_dim: int = 24,
        depths: Tuple[int, ...] = DEFAULT_DEPTHS,
        embed_dims: Tuple[int, ...] = DEFAULT_EMBED_DIMS,
        exp_ratios: Tuple[float, ...] = DEFAULT_EXP_RATIOS,
        num_heads: Tuple[int, ...] = DEFAULT_NUM_HEADS,
        matrix_state_stages: Tuple[int, ...] = DEFAULT_MATRIX_STATE_STAGES,
        drop_path_rate: float = 0.1, matrix_state_backend: Optional[str] = None,
        source_size: Tuple[int, int] = SOURCE_SIZE,
        global_size: Tuple[int, int] = GLOBAL_SIZE, local_size: int = LOCAL_SIZE,
    ) -> None:
        super().__init__()
        self.source_size = tuple(int(value) for value in source_size)
        self.global_size = tuple(int(value) for value in global_size)
        self.local_size = int(local_size)
        if self.local_size > min(self.source_size):
            raise ValueError(f"local_size={self.local_size} must fit source_size={self.source_size}")
        core_kwargs = {
            "input_channels": input_channels, "num_classes": num_classes, "stem_dim": stem_dim,
            "depths": tuple(depths), "embed_dims": tuple(embed_dims),
            "exp_ratios": tuple(exp_ratios), "num_heads": tuple(num_heads),
            "matrix_state_stages": tuple(matrix_state_stages), "drop_path_rate": drop_path_rate,
            "matrix_state_backend": matrix_state_backend,
        }
        # Deliberately one core: both branches call these exact same parameters.
        self.core = RWKV_UNetV6a(**core_kwargs)
        self._core_kwargs = core_kwargs

    def crop_origin(self, images: torch.Tensor, jitter_y: int = 0):
        height, width = images.shape[-2:]
        if (height, width) != self.source_size:
            raise ValueError(
                f"RWKV_UNetV6b expects source images in [H, W]={self.source_size}, received {(height, width)}"
            )
        crop_y0 = (height - self.local_size) // 2 + int(jitter_y)
        crop_x0 = (width - self.local_size) // 2
        if crop_y0 < 0 or crop_y0 + self.local_size > height:
            raise ValueError(f"jitter_y={jitter_y} moves the local crop outside height={height}")
        return crop_y0, crop_x0

    def forward(self, images: torch.Tensor, *, return_branches: bool = False, jitter_y: int = 0):
        crop_y0, crop_x0 = self.crop_origin(images, jitter_y=jitter_y)
        global_images = F.interpolate(
            images, size=self.global_size, mode="bilinear", align_corners=False
        )
        local_images = images[
            ..., crop_y0:crop_y0 + self.local_size, crop_x0:crop_x0 + self.local_size
        ]
        global_logits = self.core(global_images)
        local_logits = self.core(local_images)
        final_logits = compose_global_local_logits(
            global_logits, local_logits, self.source_size, crop_y0, crop_x0
        )
        if not return_branches:
            return final_logits
        return {"out": final_logits, "global_logits": global_logits, "local_logits": local_logits}

    def to_v6_reference(self):
        """Return the same wrapper with a CUDA-extension-free V6 core."""
        kwargs = dict(self._core_kwargs)
        kwargs["matrix_state_backend"] = "reference"
        reference = type(self)(
            **kwargs, source_size=self.source_size,
            global_size=self.global_size, local_size=self.local_size,
        )
        reference.core = self.core.to_v6_reference()
        reference.train(self.training)
        return reference


def rwkv_unet_v6b(input_channel: int = 3, num_classes: int = 5, **kwargs):
    return RWKV_UNetV6b(input_channels=input_channel, num_classes=num_classes, **kwargs)


__all__ = ["GLOBAL_SIZE", "LOCAL_SIZE", "SOURCE_SIZE", "RWKV_UNetV6b",
           "compose_global_local_logits", "rwkv_unet_v6b"]
