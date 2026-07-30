from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from landmark.config.loader import LossConfig
from landmark.models.metadata import POINT_BONE_IDS
from landmark.utils.coordinates import unit_to_grid_sample


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none").sum(dim=-1)
    weights = visibility.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def _gaussian_heatmaps(
    centers_xy: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    ys = torch.arange(height, dtype=centers_xy.dtype, device=centers_xy.device)
    xs = torch.arange(width, dtype=centers_xy.dtype, device=centers_xy.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    distance = (
        (xx - centers_xy[..., 0, None, None]).square()
        + (yy - centers_xy[..., 1, None, None]).square()
    )
    return torch.exp(-distance / (2 * sigma**2))


def _spatial_kl(
    logits: torch.Tensor,
    target_heatmaps: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    log_prediction = F.log_softmax(logits.flatten(-2), dim=-1)
    target = target_heatmaps.flatten(-2)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    per_point = (
        target * (target.clamp_min(1e-8).log() - log_prediction)
    ).sum(dim=-1)
    weights = valid.to(per_point.dtype)
    return (per_point * weights).sum() / weights.sum().clamp_min(1)


class LandmarkLoss(nn.Module):
    def __init__(
        self,
        config: LossConfig,
        point_bone_ids: torch.Tensor = POINT_BONE_IDS,
    ):
        super().__init__()
        self.config = config
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())

    def _local_heatmap_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        heatmaps = outputs["local_heatmaps"]
        reference = outputs["refinement_reference"]
        radius = outputs["local_patch_radius_xy"].clamp_min(1e-8)
        local = (target - reference) / radius
        valid = visibility.bool() & (local.abs() <= 1).all(dim=-1)
        height, width = heatmaps.shape[-2:]
        centers = (local + 1) / 2
        centers = centers * target.new_tensor([width - 1, height - 1])
        target_heatmaps = _gaussian_heatmaps(
            centers, height, width, self.config.heatmap_sigma
        )
        return _spatial_kl(heatmaps, target_heatmaps, valid)

    def _global_heatmap_loss(
        self,
        heatmaps: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        height, width = heatmaps.shape[-2:]
        centers = target * target.new_tensor([width - 1, height - 1])
        target_heatmaps = _gaussian_heatmaps(
            centers, height, width, self.config.heatmap_sigma
        )
        return _spatial_kl(heatmaps, target_heatmaps, visibility.bool())

    def _bone_constraint_loss(
        self,
        outputs: dict[str, torch.Tensor],
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        if "bone_probabilities" not in outputs:
            return outputs["final_landmarks"].sum() * 0
        probabilities = outputs["bone_probabilities"]
        coordinates = outputs["final_landmarks"]
        grid = unit_to_grid_sample(coordinates)[:, :, None, :]
        sampled = F.grid_sample(
            probabilities,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        correct = sampled.gather(
            -1,
            self.point_bone_ids[None, :, None].expand(coordinates.shape[0], -1, 1),
        ).squeeze(-1)
        loss = -correct.clamp_min(1e-6).log()
        weights = visibility.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        phase: str = "full",
    ) -> dict[str, torch.Tensor]:
        target = batch["landmarks"]
        visibility = batch["landmark_visibility"]
        coarse = _masked_smooth_l1(
            outputs["coarse_landmarks"], target, visibility
        )
        coordinate = _masked_smooth_l1(
            outputs["final_landmarks"], target, visibility
        )
        zero = coordinate * 0
        if "local_heatmaps" in outputs:
            heatmap = self._local_heatmap_loss(outputs, target, visibility)
        elif "global_heatmaps" in outputs:
            heatmap = self._global_heatmap_loss(
                outputs["global_heatmaps"], target, visibility
            )
        else:
            heatmap = zero
        bone = self._bone_constraint_loss(outputs, visibility)

        if phase == "coarse":
            total = self.config.coarse_weight * coarse
        else:
            total = (
                self.config.coarse_weight * coarse
                + self.config.coordinate_weight * coordinate
                + self.config.heatmap_weight * heatmap
                + self.config.bone_constraint_weight * bone
            )
        return {
            "loss": total,
            "coarse_loss": coarse,
            "coordinate_loss": coordinate,
            "heatmap_loss": heatmap,
            "bone_constraint_loss": bone,
        }
