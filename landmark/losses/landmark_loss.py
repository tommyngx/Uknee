from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from landmark.config.loader import LossConfig
from landmark.models.metadata import (
    LANDMARK_PATH_RANGES,
    POINT_BONE_IDS,
    TOPOLOGY_EDGES,
    TOPOLOGY_TRIPLETS,
)
from landmark.utils.coordinates import unit_to_grid_sample


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none").sum(dim=-1)
    weights = visibility.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    loss = F.l1_loss(prediction, target, reduction="none").sum(dim=-1)
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
    temperature: float = 1.0,
) -> torch.Tensor:
    log_prediction = F.log_softmax(logits.flatten(-2) / temperature, dim=-1)
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
        if config.heatmap_temperature <= 0:
            raise ValueError("heatmap_temperature must be positive")
        self.config = config
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        self.register_buffer(
            "topology_edge_start",
            torch.tensor([start for start, _ in TOPOLOGY_EDGES]),
        )
        self.register_buffer(
            "topology_edge_end",
            torch.tensor([end for _, end in TOPOLOGY_EDGES]),
        )
        self.register_buffer(
            "topology_curve_start",
            torch.tensor([start for start, _, _ in TOPOLOGY_TRIPLETS]),
        )
        self.register_buffer(
            "topology_curve_middle",
            torch.tensor([middle for _, middle, _ in TOPOLOGY_TRIPLETS]),
        )
        self.register_buffer(
            "topology_curve_end",
            torch.tensor([end for _, _, end in TOPOLOGY_TRIPLETS]),
        )
        self.topology_scale = 1.0

    def set_topology_scale(self, scale: float) -> None:
        """Set the curriculum multiplier used by topology-aware models."""
        self.topology_scale = min(max(float(scale), 0.0), 1.0)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1)

    def _topology_edge_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        predicted_delta = (
            prediction[:, self.topology_edge_end]
            - prediction[:, self.topology_edge_start]
        )
        target_delta = (
            target[:, self.topology_edge_end]
            - target[:, self.topology_edge_start]
        )
        # Relative normalization makes the loss comparable for short patellar
        # edges and longer femoral/tibial edges.
        target_length = torch.linalg.vector_norm(
            target_delta, dim=-1, keepdim=True
        ).clamp_min(1.0e-3)
        per_edge = F.smooth_l1_loss(
            predicted_delta / target_length,
            target_delta / target_length,
            reduction="none",
        ).sum(dim=-1)
        valid = (
            visibility[:, self.topology_edge_start]
            * visibility[:, self.topology_edge_end]
        )
        return self._weighted_mean(per_edge, valid)

    def _topology_curvature_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        def turning_vectors(points: torch.Tensor) -> torch.Tensor:
            before = points[:, self.topology_curve_middle] - points[
                :, self.topology_curve_start
            ]
            after = points[:, self.topology_curve_end] - points[
                :, self.topology_curve_middle
            ]
            before = F.normalize(before, dim=-1, eps=1.0e-6)
            after = F.normalize(after, dim=-1, eps=1.0e-6)
            return after - before

        per_triplet = F.smooth_l1_loss(
            turning_vectors(prediction),
            turning_vectors(target),
            reduction="none",
        ).sum(dim=-1)
        valid = (
            visibility[:, self.topology_curve_start]
            * visibility[:, self.topology_curve_middle]
            * visibility[:, self.topology_curve_end]
        )
        return self._weighted_mean(per_triplet, valid)

    def _topology_duplicate_loss(
        self,
        probabilities: torch.Tensor | None,
        visibility: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if probabilities is None:
            return zero
        losses = []
        weights = []
        uniform_overlap = 1.0 / probabilities.shape[-1]
        for start, stop in LANDMARK_PATH_RANGES:
            path_probabilities = probabilities[:, start:stop]
            overlap = torch.matmul(
                path_probabilities, path_probabilities.transpose(1, 2)
            )
            path_visibility = visibility[:, start:stop]
            valid_pairs = (
                path_visibility[:, :, None] * path_visibility[:, None, :]
            ).bool()
            diagonal = torch.eye(
                stop - start,
                dtype=torch.bool,
                device=probabilities.device,
            )[None]
            valid_pairs = valid_pairs & ~diagonal
            collision = (overlap - uniform_overlap).clamp_min(0)
            collision = collision.masked_fill(~valid_pairs, 0).max(dim=-1).values
            losses.append(collision)
            weights.append(path_visibility * valid_pairs.any(dim=-1))
        return self._weighted_mean(
            torch.cat(losses, dim=1), torch.cat(weights, dim=1)
        )

    def _local_heatmap_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        heatmaps = outputs["local_heatmaps"]
        # Heatmap targets are labels, not a path for gradients into the model's
        # current reference prediction. Coordinate losses supervise reference
        # quality separately.
        reference = outputs["refinement_reference"].detach()
        radius = outputs["local_patch_radius_xy"].detach().clamp_min(1e-8)
        local = (target - reference) / radius
        valid = visibility.bool() & (local.abs() <= 1).all(dim=-1)
        height, width = heatmaps.shape[-2:]
        centers = (local + 1) / 2
        centers = centers * target.new_tensor([width - 1, height - 1])
        target_heatmaps = _gaussian_heatmaps(
            centers, height, width, self.config.heatmap_sigma
        )
        return _spatial_kl(
            heatmaps,
            target_heatmaps,
            valid,
            self.config.heatmap_temperature,
        )

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
        return _spatial_kl(
            heatmaps,
            target_heatmaps,
            visibility.bool(),
            self.config.heatmap_temperature,
        )

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
        # A bounded penalty avoids the 1/p gradient explosion of -log(p) when
        # early landmark predictions fall outside their bone.
        loss = 1 - correct.clamp(0, 1)
        weights = visibility.to(loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1)

    def _contour_assignment_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = outputs["contour_candidate_coordinates"]
        logits = outputs["contour_assignment_logits"]
        squared_distance = (candidates - target[:, :, None]).square().sum(dim=-1)
        nearest = squared_distance.argmin(dim=-1)
        per_landmark = F.cross_entropy(
            logits.flatten(0, 1),
            nearest.flatten(),
            reduction="none",
        ).view_as(visibility)
        weights = visibility.to(per_landmark.dtype)
        assignment = (per_landmark * weights).sum() / weights.sum().clamp_min(1)
        oracle = (
            squared_distance.gather(-1, nearest[..., None])
            .squeeze(-1)
            .clamp_min(0)
            .sqrt()
        )
        oracle = (oracle * weights).sum() / weights.sum().clamp_min(1)
        return assignment, oracle

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        phase: str = "full",
    ) -> dict[str, torch.Tensor]:
        target = batch["landmarks"]
        visibility = batch["landmark_visibility"]
        contour_constrained = "contour_assignment_logits" in outputs
        coordinate_loss = _masked_l1 if contour_constrained else _masked_smooth_l1
        coarse = coordinate_loss(outputs["coarse_landmarks"], target, visibility)
        coordinate = coordinate_loss(outputs["final_landmarks"], target, visibility)
        zero = coordinate * 0
        coarse_heatmap = (
            self._global_heatmap_loss(
                outputs["coarse_heatmaps"], target, visibility
            )
            if "coarse_heatmaps" in outputs
            else zero
        )
        contour_oracle = zero
        if contour_constrained:
            heatmap, contour_oracle = self._contour_assignment_loss(
                outputs, target, visibility
            )
        elif "local_heatmaps" in outputs:
            heatmap = self._local_heatmap_loss(outputs, target, visibility)
        elif "global_heatmaps" in outputs:
            heatmap = self._global_heatmap_loss(
                outputs["global_heatmaps"], target, visibility
            )
        else:
            heatmap = zero
        bone = (
            self._bone_constraint_loss(outputs, visibility)
            if self.config.bone_constraint_weight
            else zero
        )
        topology_aware = "topology_soft_landmarks" in outputs
        if topology_aware:
            topology_coordinates = outputs["topology_soft_landmarks"]
            topology_edge = self._topology_edge_loss(
                topology_coordinates, target, visibility
            )
            topology_curvature = self._topology_curvature_loss(
                topology_coordinates, target, visibility
            )
            topology_duplicate = self._topology_duplicate_loss(
                outputs.get("contour_assignment_probabilities"),
                visibility,
                zero,
            )
        else:
            topology_edge = topology_curvature = topology_duplicate = zero
        topology = self.topology_scale * (
            self.config.topology_edge_weight * topology_edge
            + self.config.topology_curvature_weight * topology_curvature
            + self.config.topology_duplicate_weight * topology_duplicate
        )

        if phase == "coarse":
            total = (
                self.config.coarse_weight * coarse
                + self.config.coarse_heatmap_weight * coarse_heatmap
            )
        else:
            total = (
                self.config.coarse_weight * coarse
                + self.config.coarse_heatmap_weight * coarse_heatmap
                + self.config.coordinate_weight * coordinate
                + self.config.heatmap_weight * heatmap
                + self.config.bone_constraint_weight * bone
                + topology
            )
        return {
            "loss": total,
            "coarse_loss": coarse,
            "coarse_heatmap_loss": coarse_heatmap,
            "coordinate_loss": coordinate,
            "heatmap_loss": heatmap,
            "bone_constraint_loss": bone,
            "contour_oracle_loss": contour_oracle,
            "topology_loss": topology,
            "topology_edge_loss": topology_edge,
            "topology_curvature_loss": topology_curvature,
            "topology_duplicate_loss": topology_duplicate,
        }
