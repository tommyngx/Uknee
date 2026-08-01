from __future__ import annotations

import torch
from torch import nn

from landmark.config.loader import ModelConfig

from .backbone_adapter import RWKVUNetBackboneAdapter
from .coarse_reference_head import CoarseReferenceHead
from .landmark_transformer import LandmarkQueryTransformer
from .local_feature_sampler import MultiScaleLocalFeatureSampler
from .local_heatmap_head import QueryConditionedLocalHeatmapHead
from .metadata import POINT_BONE_IDS
from .query_initializer import LandmarkQueryInitializer


class RWKVUNetLandmarkModel(nn.Module):
    """RWKV U-Net guided global-to-local landmark query network.

    All model landmark coordinates are ordered ``(x, y)`` and normalised to
    ``[0, 1]``. Ground-truth references are optional training-only inputs;
    inference needs only the image.
    """

    def __init__(
        self,
        segmentation_backbone: nn.Module,
        config: ModelConfig,
        point_bone_ids: torch.Tensor = POINT_BONE_IDS,
    ):
        super().__init__()
        if config.num_landmarks != point_bone_ids.numel():
            raise ValueError(
                f"Configured {config.num_landmarks} landmarks but metadata contains "
                f"{point_bone_ids.numel()}"
            )
        if len(config.bone_class_indices) != config.num_bones:
            raise ValueError("bone_class_indices length must equal num_bones")
        if max(config.bone_class_indices) >= config.num_mask_classes:
            raise ValueError("bone_class_indices references a missing segmentation class")
        if config.bone_class_groups:
            if len(config.bone_class_groups) != config.num_bones:
                raise ValueError("bone_class_groups length must equal num_bones")
            invalid = [
                class_id
                for group in config.bone_class_groups
                for class_id in group
                if class_id < 0 or class_id >= config.num_mask_classes
            ]
            if invalid:
                raise ValueError(
                    f"bone_class_groups references missing segmentation classes: {invalid}"
                )
        self.config = config
        self.bone_class_groups = tuple(
            tuple(group) for group in config.bone_class_groups
        )
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        self.register_buffer(
            "bone_class_indices",
            torch.tensor(config.bone_class_indices, dtype=torch.long),
        )
        self.backbone_adapter = RWKVUNetBackboneAdapter(
            segmentation_backbone,
            query_dim=config.query_dim,
            freeze_backbone=config.freeze_backbone,
        )
        self.coarse_reference_head = CoarseReferenceHead(
            config.num_landmarks,
            config.num_bones,
            config.query_dim,
            point_bone_ids,
            config.transformer_dropout,
        )
        self.local_feature_sampler = MultiScaleLocalFeatureSampler(
            config.query_dim,
            config.num_landmarks,
            point_bone_ids,
            config.local_patch_size,
            config.token_patch_size,
        )
        self.query_initializer = LandmarkQueryInitializer(
            config.num_landmarks,
            config.num_bones,
            config.query_dim,
            point_bone_ids,
        )
        self.landmark_transformer = LandmarkQueryTransformer(
            config.query_dim,
            config.attention_heads,
            config.transformer_ffn_dim,
            config.transformer_layers,
            config.transformer_dropout,
        )
        self.local_heatmap_head = QueryConditionedLocalHeatmapHead(config.query_dim)

    def get_bone_probabilities(self, segmentation_logits: torch.Tensor) -> torch.Tensor:
        if segmentation_logits.shape[1] == 1:
            return torch.sigmoid(segmentation_logits)
        probabilities = torch.softmax(segmentation_logits, dim=1)
        if self.bone_class_groups:
            bone_maps = [
                probabilities[:, group].sum(dim=1)
                for group in self.bone_class_groups
            ]
            return torch.stack(bone_maps, dim=1).clamp(0, 1)
        return probabilities.index_select(1, self.bone_class_indices)

    def set_training_phase(self, phase: str) -> None:
        """Select coarse, refinement, or full landmark-decoder optimisation."""
        if phase not in {"coarse", "refinement", "full"}:
            raise ValueError(f"Unknown training phase: {phase}")
        landmark_modules = (
            self.coarse_reference_head,
            self.local_feature_sampler,
            self.query_initializer,
            self.landmark_transformer,
            self.local_heatmap_head,
        )
        for module in landmark_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(phase == "full")
        if phase == "coarse":
            for parameter in self.coarse_reference_head.parameters():
                parameter.requires_grad_(True)
            for projection in (
                self.backbone_adapter.high_projection,
                self.backbone_adapter.mid_projection,
                self.backbone_adapter.low_projection,
            ):
                for parameter in projection.parameters():
                    parameter.requires_grad_(True)
        elif phase == "refinement":
            for module in landmark_modules[1:]:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for projection in (
                self.backbone_adapter.high_projection,
                self.backbone_adapter.mid_projection,
                self.backbone_adapter.low_projection,
            ):
                for parameter in projection.parameters():
                    parameter.requires_grad_(True)

    def forward(
        self,
        image: torch.Tensor,
        reference_landmarks: torch.Tensor | None = None,
        return_heatmaps: bool = True,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        backbone = self.backbone_adapter(image)
        bone_probabilities = self.get_bone_probabilities(
            backbone["segmentation_logits"]
        )
        coarse_outputs = self.coarse_reference_head(
            backbone["feature_mid"],
            backbone["feature_low"],
            bone_probabilities,
        )
        coarse = coarse_outputs["coordinates"]
        reference = coarse if reference_landmarks is None else reference_landmarks.clamp(0, 1)
        # Stop local appearance/heatmap gradients from moving the coarse patch
        # centres. The final coordinate still contains `reference`, so its
        # coordinate loss supervises the coarse head directly.
        refinement_context = reference.detach()
        sampled = self.local_feature_sampler(
            backbone["feature_high"],
            backbone["feature_mid"],
            backbone["feature_low"],
            bone_probabilities,
            refinement_context,
        )
        queries = self.query_initializer(
            refinement_context, sampled["local_tokens"]
        )
        queries = self.landmark_transformer(queries)
        refined = self.local_heatmap_head(
            sampled["local_patches_high"],
            queries,
            reference,
            sampled["patch_radius_xy"],
        )
        outputs = {
            "segmentation_logits": backbone["segmentation_logits"],
            "bone_probabilities": bone_probabilities,
            "coarse_landmarks": coarse,
            "coarse_landmark_confidence": coarse_outputs["confidence"],
            "refinement_reference": reference,
            "local_patch_radius_xy": sampled["patch_radius_xy"],
            "final_landmarks": refined["final_coordinates"],
            "landmark_confidence": refined["confidence"],
        }
        if return_heatmaps:
            outputs["coarse_heatmaps"] = coarse_outputs["heatmaps"]
            outputs["local_heatmaps"] = refined["local_heatmaps"]
        if return_features:
            outputs.update(
                {
                    "feature_high": backbone["feature_high"],
                    "feature_mid": backbone["feature_mid"],
                    "feature_low": backbone["feature_low"],
                }
            )
        return outputs
