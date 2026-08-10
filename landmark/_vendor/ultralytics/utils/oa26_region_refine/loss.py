# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Losses used only by the OA26 per-region refinement experiment."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.cfg import get_cfg
from ultralytics.utils.oa26_region_refine.debug import debug_event, mark_backward
from ultralytics.utils.oa26.loss import OA26HeatmapPoseLoss
from ultralytics.utils.oa26_region_refine.region_schema import (
    NUM_REGIONS,
    class_keypoint_mask,
    class_path_masks,
)


class OA26RegionRefinePoseLoss(OA26HeatmapPoseLoss):
    """Preserve every v1 loss and append ROI heatmap/coordinate/intra-region structure losses."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Read v9-only loss settings without changing the old OA26 loss implementation."""
        # PoseModel constructed directly (outside the YOLO/Trainer wrapper) has no `.args`, while every base YOLO loss
        # expects its hyperparameters there. Supply normal pose defaults so the public v9 criterion also works in
        # notebooks and standalone diagnostics instead of failing during criterion construction.
        if not hasattr(model, "args"):
            model.args = get_cfg(overrides={"task": "pose", "mode": "train"})
        super().__init__(model, tal_topk, tal_topk2)
        self.debug_branch = f"topk-{tal_topk}-topk2-{tal_topk2}"
        cfg = getattr(model, "yaml", {}).get("oa26_region_refine", {})
        self.refined_heatmap_gain = float(cfg.get("refined_heatmap_gain", 1.0))
        self.refined_heatmap_sigma = float(cfg.get("refined_heatmap_sigma", 1.5))
        self.refined_coord_gain = float(cfg.get("refined_coord_gain", 0.5))
        self.refined_neighbour_gain = float(cfg.get("refined_neighbour_gain", 0.1))
        self.refined_curve_gain = float(cfg.get("refined_curve_gain", 0.1))

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the original ten v1/RLE losses followed by four region-refinement losses."""
        debug_event("v9-loss-enter", branch=self.debug_branch, batch=preds["boxes"].shape[0])
        base_loss, base_detach = super().loss(preds, batch)
        debug_event("v9-loss-base-complete", branch=self.debug_branch)
        region_loss = self.region_refinement_loss(preds, batch)
        batch_size = preds["boxes"].shape[0]
        total = torch.cat((base_loss, region_loss * batch_size))
        mark_backward(total, "v9-loss-backward-enter", branch=self.debug_branch)
        debug_event("v9-loss-complete", branch=self.debug_branch, items=total.shape[0])
        # Auxiliary terms remain in `total` and still train; only five standard YOLO Pose items are reported/logged.
        return total, base_detach[:5]

    def region_refinement_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute heatmap, coordinate, neighbour, and curvature loss independently per region row."""
        probability = preds.get("region_heatmaps")
        if probability is None:
            return preds["boxes"].new_zeros(4)
        batch_size = preds["boxes"].shape[0]
        num_keypoints = self.kpt_shape[0]
        image_size = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=probability.dtype)
        image_size *= self.stride[0]
        region_ids = preds["region_ids"].long()
        region_batch = preds["region_batch_indices"].long()
        rows = region_batch * NUM_REGIONS + region_ids

        # MESKO stores four object rows per image. Each row has 51 local slots and is matched by class ID.
        region_gt = probability.new_zeros((batch_size * NUM_REGIONS, num_keypoints, 3))
        region_present = torch.zeros(batch_size * NUM_REGIONS, device=self.device, dtype=torch.bool)
        gt_batch = batch["batch_idx"].long().view(-1)
        gt_class = batch["cls"].long().view(-1)
        valid_object = (gt_batch >= 0) & (gt_batch < batch_size) & (gt_class >= 0) & (gt_class < NUM_REGIONS)
        if valid_object.any():
            gt_slots = gt_batch[valid_object] * NUM_REGIONS + gt_class[valid_object]
            region_gt[gt_slots] = batch["keypoints"][valid_object].to(self.device, probability.dtype)
            region_present[gt_slots] = True
        region_gt = region_gt[rows]
        region_gt_xy = region_gt[..., :2] * image_size.flip(0)
        schema_valid = class_keypoint_mask(region_ids, num_keypoints)
        valid = schema_valid & region_present[rows, None] & (region_gt[..., 2] > 0)
        loss = probability.new_zeros(4)
        if not valid.any():
            return loss

        boxes = preds["region_boxes"]
        origin = boxes[:, None, :2]
        size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1).unsqueeze(1)
        gt_xy_roi = ((region_gt_xy - origin) / size).clamp(0.0, 1.0)
        pred_xy_roi = preds["region_refined_xy_roi"]

        if self.refined_heatmap_gain:
            target = self._gaussian_targets(gt_xy_roi, valid, probability.shape[-2:])
            pixels = probability.shape[-2] * probability.shape[-1]
            loss[0] = F.mse_loss(probability[valid], target[valid]) * pixels * self.refined_heatmap_gain

        if self.refined_coord_gain:
            loss[1] = F.smooth_l1_loss(pred_xy_roi[valid], gt_xy_roi[valid], beta=0.01)
            loss[1] *= self.refined_coord_gain

        pair_mask = valid[:, 1:] & valid[:, :-1] & class_path_masks(region_ids, order=2)
        if self.refined_neighbour_gain and pair_mask.any():
            pred_vector = pred_xy_roi[:, 1:] - pred_xy_roi[:, :-1]
            gt_vector = gt_xy_roi[:, 1:] - gt_xy_roi[:, :-1]
            loss[2] = F.smooth_l1_loss(pred_vector[pair_mask], gt_vector[pair_mask], beta=0.01)
            loss[2] *= self.refined_neighbour_gain

        curve_mask = (
            valid[:, 2:]
            & valid[:, 1:-1]
            & valid[:, :-2]
            & class_path_masks(region_ids, order=3)
        )
        if self.refined_curve_gain and curve_mask.any():
            pred_curve = pred_xy_roi[:, :-2] - 2 * pred_xy_roi[:, 1:-1] + pred_xy_roi[:, 2:]
            gt_curve = gt_xy_roi[:, :-2] - 2 * gt_xy_roi[:, 1:-1] + gt_xy_roi[:, 2:]
            loss[3] = F.smooth_l1_loss(pred_curve[curve_mask], gt_curve[curve_mask], beta=0.01)
            loss[3] *= self.refined_curve_gain
        return loss

    def _gaussian_targets(
        self, gt_xy_roi: torch.Tensor, valid: torch.Tensor, roi_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Create normalized Gaussian probability targets on the ROI grid."""
        height, width = roi_hw
        ys = (torch.arange(height, device=gt_xy_roi.device, dtype=gt_xy_roi.dtype) + 0.5) / max(height, 1)
        xs = (torch.arange(width, device=gt_xy_roi.device, dtype=gt_xy_roi.dtype) + 0.5) / max(width, 1)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((x_grid, y_grid), dim=-1)
        distance = (grid.view(1, 1, height, width, 2) - gt_xy_roi[:, :, None, None]).pow(2).sum(dim=-1)
        sigma_x = self.refined_heatmap_sigma / max(width, 1)
        sigma_y = self.refined_heatmap_sigma / max(height, 1)
        sigma = max((sigma_x + sigma_y) * 0.5, 1e-6)
        target = torch.exp(-distance / (2 * sigma**2)) * valid[:, :, None, None].to(gt_xy_roi.dtype)
        return target / target.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-9)
