# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Per-instance anatomical-region refinement for MESKO4GF2 pose labels."""

from __future__ import annotations

import torch
import torch.nn as nn

from landmark2.data.schema import (
    MAX_REGION_KEYPOINTS,
    NUM_REGIONS,
    class_keypoint_mask,
    validate_region_schema,
)
from landmark2.core.debug import debug_event, mark_backward

from .landmark_query_encoder import OA26RegionQueryEncoder
from .localization_head import OA26RegionLocalizationHead
from .region_transformer import OA26RegionTransformer
from .roi_feature_extractor import OA26RegionROIExtractor


class OA26PerRegionRefinementHead(nn.Module):
    """Refine each detected bone independently using its P4 ROI and class-local landmarks."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_REGIONS,
        kpt_shape: tuple[int, int] = (MAX_REGION_KEYPOINTS, 3),
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        roi_output_size: tuple[int, int] = (20, 20),
        roi_sampling_ratio: int = 2,
        roi_padding: float = 0.25,
        min_roi_size_px: float = 48.0,
        heatmap_temperature: float = 0.1,
        use_coarse_prior: bool = False,
        coarse_prior_sigma: float = 0.25,
        coarse_prior_gain: float = 0.5,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        """Initialize ROI extraction, landmark queries, attention, and spatial localization."""
        super().__init__()
        validate_region_schema(num_classes, kpt_shape)
        self.num_keypoints = int(kpt_shape[0])
        self.roi_padding = float(roi_padding)
        self.min_roi_size_px = float(min_roi_size_px)
        self.roi_extractor = OA26RegionROIExtractor(
            in_channels, d_model, roi_output_size, roi_sampling_ratio, aligned=True
        )
        self.query_encoder = OA26RegionQueryEncoder(
            MAX_REGION_KEYPOINTS, NUM_REGIONS, d_model, coord_fourier_bands=6
        )
        self.transformer = OA26RegionTransformer(
            d_model, num_heads, num_layers, 4.0, dropout, gradient_checkpointing
        )
        self.localization_head = OA26RegionLocalizationHead(
            d_model,
            heatmap_temperature,
            use_coarse_prior,
            coarse_prior_sigma,
            coarse_prior_gain,
        )

    def _expand_boxes(
        self, boxes: torch.Tensor, image_size: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Pad detected boxes into stable search ROIs while remaining inside the image."""
        image_h, image_w = image_size
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=2.0) * (1.0 + 2.0 * self.roi_padding)
        size = size.clamp(min=self.min_roi_size_px)
        xy1, xy2 = center - size * 0.5, center + size * 0.5
        xy1 = torch.stack((xy1[:, 0].clamp(0, image_w), xy1[:, 1].clamp(0, image_h)), dim=-1)
        xy2 = torch.stack((xy2[:, 0].clamp(0, image_w), xy2[:, 1].clamp(0, image_h)), dim=-1)
        # Small cold-start boxes near an image edge can collapse after clipping.
        xy2 = torch.maximum(xy2, xy1 + 2.0)
        xy2 = torch.stack((xy2[:, 0].clamp(max=image_w), xy2[:, 1].clamp(max=image_h)), dim=-1)
        xy1 = torch.minimum(xy1, xy2 - 2.0)
        return torch.cat((xy1, xy2), dim=-1)

    @staticmethod
    def _image_to_roi(xy: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Convert image pixels to normalized coordinates in each instance ROI."""
        origin = boxes[:, None, :2]
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1).unsqueeze(1)
        return ((xy - origin) / size).clamp(0.0, 1.0)

    @staticmethod
    def _roi_to_image(xy: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Convert normalized ROI coordinates back to input-image pixels."""
        origin = boxes[:, None, :2]
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1).unsqueeze(1)
        return origin + xy * size

    def forward(
        self,
        p4_feature: torch.Tensor,
        instance_boxes: torch.Tensor,
        coarse_keypoints: torch.Tensor,
        class_ids: torch.Tensor,
        batch_indices: torch.Tensor,
        selected_anchor_indices: torch.Tensor,
        image_size: tuple[torch.Tensor, torch.Tensor],
        p4_stride: float,
    ) -> dict[str, torch.Tensor]:
        """Return one independent refined 51-slot pose for every selected class instance."""
        debug_event("refiner-forward-enter", p4_shape=tuple(p4_feature.shape), instances=instance_boxes.shape[0])
        valid_mask = class_keypoint_mask(class_ids, self.num_keypoints)
        # torchvision ROIAlign differentiates feature values, not ROI coordinates. Keep predicted boxes outside the
        # refiner autograd graph to avoid unsupported/native ROI-coordinate backward paths on older CUDA/torchvision
        # builds and to prevent refinement losses from perturbing the detector through crop geometry.
        region_boxes = self._expand_boxes(instance_boxes.detach(), image_size)
        coarse_xy = coarse_keypoints[..., :2].masked_fill(~valid_mask.unsqueeze(-1), 0)
        coarse_conf = coarse_keypoints[..., 2].masked_fill(~valid_mask, 0)
        coarse_xy_roi = self._image_to_roi(coarse_xy, region_boxes)

        roi_features = self.roi_extractor(
            p4_feature, region_boxes, batch_indices, 1.0 / max(float(p4_stride), 1.0)
        )
        debug_event("refiner-after-roi", roi_shape=tuple(roi_features.shape))
        height, width = roi_features.shape[-2:]
        image_tokens = roi_features.flatten(2).transpose(1, 2)
        queries = self.query_encoder(coarse_xy_roi, coarse_conf, class_ids, valid_mask)
        landmark_tokens = self.transformer(queries, image_tokens, valid_mask)
        debug_event("refiner-after-transformer", token_shape=tuple(landmark_tokens.shape))
        logits, probability, refined_xy_roi = self.localization_head(
            landmark_tokens, image_tokens, (height, width), coarse_xy_roi, valid_mask
        )
        mark_backward(probability, "refiner-heatmap-backward")
        debug_event("refiner-forward-complete", heatmap_shape=tuple(probability.shape))
        refined_xy = self._roi_to_image(refined_xy_roi, region_boxes)
        refined_xy = torch.where(valid_mask.unsqueeze(-1), refined_xy, coarse_keypoints[..., :2])
        refined_kpts = torch.cat((refined_xy, coarse_keypoints[..., 2:3]), dim=-1)
        return {
            "coarse_region_kpts": coarse_keypoints,
            "refined_region_kpts": refined_kpts,
            "region_boxes": region_boxes,
            "region_batch_indices": batch_indices,
            "region_ids": class_ids,
            "region_valid_mask": valid_mask,
            "region_selected_anchor_indices": selected_anchor_indices,
            "region_coarse_xy_roi": coarse_xy_roi,
            "region_refined_xy_roi": refined_xy_roi,
            "region_heatmap_logits": logits,
            "region_heatmaps": probability,
        }
