# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""OA26 auxiliary pose losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import PoseLoss26
from ultralytics.utils.oa26.heatmap import (
    extract_canonical_image_keypoints,
    extract_image_keypoints,
    gaussian_heatmap_targets,
)
from ultralytics.utils.oa26.simcc import gaussian_simcc_targets


class OA26HeatmapPoseLoss(PoseLoss26):
    """Pose26 loss plus optional heatmap, soft-coordinate and structure-aware losses."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize loss gains from `oa26_loss` in the model YAML."""
        super().__init__(model, tal_topk, tal_topk2)
        cfg = getattr(model, "yaml", {}).get("oa26_loss", {})
        self.heatmap_gain = float(cfg.get("heatmap", cfg.get("hm_loss_gain", 1.0)))
        self.coord_gain = float(cfg.get("coord", 0.0))
        self.neighbour_gain = float(cfg.get("neighbour", 0.0))
        self.curve_gain = float(cfg.get("curve", 0.0))
        self.sigma = float(cfg.get("sigma", cfg.get("hm_sigma", 1.5)))

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate base pose loss and append OA26 auxiliary components."""
        base_loss, base_detach = super().loss(preds, batch)
        if "heatmaps" not in preds or "hm_kpts" not in preds:
            return base_loss, base_detach

        batch_size = preds["heatmaps"].shape[0]
        aux = self.auxiliary_loss(preds, batch)
        return torch.cat((base_loss, aux * batch_size)), torch.cat((base_detach, aux.detach()))

    def auxiliary_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute heatmap, coordinate, neighbour and curvature losses."""
        heatmaps = preds["heatmaps"]
        batch_size, num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
        image_size = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=heatmaps.dtype) * self.stride[0]
        if num_keypoints == 129:
            gt_xy, valid = extract_canonical_image_keypoints(
                batch, batch_size, image_size, self.device, heatmaps.dtype
            )
        else:
            gt_xy, valid = extract_image_keypoints(
                batch, batch_size, num_keypoints, image_size, self.device, heatmaps.dtype
            )
        loss = torch.zeros(4, device=self.device, dtype=heatmaps.dtype)
        if not valid.any():
            return loss

        if self.heatmap_gain:
            target = gaussian_heatmap_targets(gt_xy, valid, (heatmap_h, heatmap_w), image_size, self.sigma)
            loss[0] = F.mse_loss(heatmaps.sigmoid(), target) * self.heatmap_gain

        pred_xy = preds["hm_kpts"].clone()
        pred_xy[..., 0] *= image_size[1] / max(float(heatmap_w), 1.0)
        pred_xy[..., 1] *= image_size[0] / max(float(heatmap_h), 1.0)

        if self.coord_gain:
            loss[1] = F.smooth_l1_loss(pred_xy[valid], gt_xy[valid], beta=2.0) * self.coord_gain

        edge_start, edge_end, curve_start, curve_middle, curve_end = self._path_indices(
            num_keypoints, heatmaps.device
        )
        neighbour_mask = valid[:, edge_start] & valid[:, edge_end]
        if self.neighbour_gain and neighbour_mask.any():
            pred_vec = pred_xy[:, edge_end] - pred_xy[:, edge_start]
            gt_vec = gt_xy[:, edge_end] - gt_xy[:, edge_start]
            loss[2] = F.smooth_l1_loss(pred_vec[neighbour_mask], gt_vec[neighbour_mask], beta=2.0)
            loss[2] *= self.neighbour_gain

        curve_mask = valid[:, curve_start] & valid[:, curve_middle] & valid[:, curve_end]
        if self.curve_gain and curve_mask.any():
            pred_curve = pred_xy[:, curve_start] - 2 * pred_xy[:, curve_middle] + pred_xy[:, curve_end]
            gt_curve = gt_xy[:, curve_start] - 2 * gt_xy[:, curve_middle] + gt_xy[:, curve_end]
            loss[3] = F.smooth_l1_loss(pred_curve[curve_mask], gt_curve[curve_mask], beta=2.0)
            loss[3] *= self.curve_gain
        return loss

    @staticmethod
    def _path_indices(num_keypoints: int, device: torch.device):
        """Return adjacency indices without joining independent tibial paths."""
        ranges = ((0, 45), (45, 86), (86, 91), (91, 96), (96, 120), (120, 129))
        if num_keypoints != 129:
            ranges = ((0, num_keypoints),)
        edges = [(index, index + 1) for start, stop in ranges for index in range(start, stop - 1)]
        curves = [
            (index, index + 1, index + 2)
            for start, stop in ranges
            for index in range(start, stop - 2)
        ]
        return (
            torch.tensor([item[0] for item in edges], device=device),
            torch.tensor([item[1] for item in edges], device=device),
            torch.tensor([item[0] for item in curves], device=device),
            torch.tensor([item[1] for item in curves], device=device),
            torch.tensor([item[2] for item in curves], device=device),
        )


class OA26SimCCPoseLoss(PoseLoss26):
    """Pose26 loss plus auxiliary SimCC x/y classification loss."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize SimCC loss config from `oa26_simcc` in the model YAML."""
        super().__init__(model, tal_topk, tal_topk2)
        cfg = getattr(model, "yaml", {}).get("oa26_simcc", {})
        self.simcc_loss_gain = float(cfg.get("simcc_loss_gain", 1.0))
        self.simcc_sigma = float(cfg.get("simcc_sigma", cfg.get("label_smooth_sigma", 6.0)))

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate base pose loss and append SimCC auxiliary loss."""
        base_loss, base_detach = super().loss(preds, batch)
        if "simcc_x" not in preds or "simcc_y" not in preds:
            return base_loss, base_detach

        batch_size = preds["simcc_x"].shape[0]
        aux = self.auxiliary_loss(preds, batch).view(1)
        return torch.cat((base_loss, aux * batch_size)), torch.cat((base_detach, aux.detach()))

    def auxiliary_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute masked KL-style SimCC loss."""
        x_logits, y_logits = preds["simcc_x"], preds["simcc_y"]
        if not self.simcc_loss_gain:
            return x_logits.new_zeros(())

        batch_size, num_keypoints, x_bins = x_logits.shape
        y_bins = y_logits.shape[-1]
        image_size = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=x_logits.dtype) * self.stride[0]
        gt_xy, valid = extract_image_keypoints(
            batch, batch_size, num_keypoints, image_size, self.device, x_logits.dtype
        )
        if not valid.any():
            return x_logits.new_zeros(())

        target_x, target_y, valid = gaussian_simcc_targets(gt_xy, valid, image_size, x_bins, y_bins, self.simcc_sigma)
        log_px = x_logits.log_softmax(dim=-1)
        log_py = y_logits.log_softmax(dim=-1)
        loss_x = F.kl_div(log_px, target_x, reduction="none").sum(dim=-1)
        loss_y = F.kl_div(log_py, target_y, reduction="none").sum(dim=-1)
        return ((loss_x + loss_y)[valid].mean()) * self.simcc_loss_gain
