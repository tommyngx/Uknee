# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""P4 ROI feature extraction for OA26 per-region refinement."""

from __future__ import annotations

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.extension import _has_ops
from torchvision.ops import roi_align

from landmark.nn.modules.conv import Conv
from landmark.core.debug import debug_event, mark_backward


_roi_align_module = import_module("torchvision.ops.roi_align")
_roi_align_eager = getattr(_roi_align_module._roi_align, "__wrapped__", _roi_align_module._roi_align)


class OA26RegionROIExtractor(nn.Module):
    """Extract a fixed-resolution projected feature map for every anatomical region ROI."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        output_size: tuple[int, int] = (20, 20),
        sampling_ratio: int = 2,
        aligned: bool = True,
    ):
        """Initialize the feature projection and ROIAlign settings."""
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = bool(aligned)
        self.projection = nn.Sequential(Conv(in_channels, d_model, 1), Conv(d_model, d_model, 3))

    def forward(
        self,
        feature: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        spatial_scale: float,
    ) -> torch.Tensor:
        """Return M x d_model x Hroi x Wroi features, including a safe empty-ROI path."""
        projected = self.projection(feature)
        mark_backward(projected, "roi-align-backward-complete")
        if boxes.numel() == 0:
            return projected.new_empty((0, projected.shape[1], *self.output_size))
        # THOP's stride-based FLOPs probe evaluates the full model with a synthetic 32x32 image, leaving P4 at only
        # 2x2. torchvision 0.19's CPU ROIAlign native kernel can segfault (not raise Python) when expanding that tiny
        # map to the 20x20 ROI grid. Real v9 inputs are much larger; keep profiling safe with a differentiable
        # per-image fallback only for these degenerate feature maps.
        if min(projected.shape[-2:]) <= 2:
            debug_event("roi-align-tiny-feature-fallback", feature_shape=tuple(projected.shape), rois=boxes.shape[0])
            pooled = F.adaptive_avg_pool2d(projected, self.output_size)
            return pooled.index_select(0, batch_indices.long())
        # ROI coordinates are metadata, not differentiable model values. Torchvision 0.19/CUDA builds are also more
        # robust when ROIAlign forward/backward stays in FP32 instead of entering its AMP half-precision native path.
        rois = torch.cat((batch_indices.to(boxes.dtype).unsqueeze(1), boxes.detach()), dim=1)
        output_dtype = projected.dtype
        debug_event("roi-align-forward-enter", feature_shape=tuple(projected.shape), rois=rois.shape[0])
        with torch.autocast(device_type=projected.device.type, enabled=False):
            roi_input, roi_boxes = projected.float(), rois.float()
            # Torchvision 0.19 otherwise invokes a hidden torch.compile path here. Its undecorated implementation is
            # fully differentiable and avoids the incompatible user-site Triton installed on the school machine.
            use_eager = not _has_ops() or (
                torch.are_deterministic_algorithms_enabled() and projected.device.type in {"cuda", "mps", "xpu"}
            )
            if use_eager:
                debug_event(
                    "roi-align-eager-fallback",
                    device=projected.device.type,
                    native_ops=_has_ops(),
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                )
                aligned = _roi_align_eager(
                    roi_input,
                    roi_boxes,
                    float(spatial_scale),
                    self.output_size[0],
                    self.output_size[1],
                    self.sampling_ratio,
                    self.aligned,
                )
            else:
                aligned = roi_align(
                    roi_input,
                    roi_boxes,
                    output_size=self.output_size,
                    spatial_scale=float(spatial_scale),
                    sampling_ratio=self.sampling_ratio,
                    aligned=self.aligned,
                )
        debug_event("roi-align-forward-complete", output_shape=tuple(aligned.shape))
        mark_backward(aligned, "roi-align-backward-enter")
        return aligned.to(output_dtype)
