from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from landmark.config.loader import ModelConfig

from .backbone_adapter import RWKVUNetBackboneAdapter
from .metadata import POINT_BONE_IDS


class KneePV1ContourDETR(nn.Module):
    """Ordered landmark queries constrained to a frozen segmentation contour.

    The segmentation backbone proposes a fixed set of contour tokens for each
    bone. Every anatomical query can attend only to tokens from its own bone,
    and the final coordinate is snapped to one of those tokens. Landmark
    identity is therefore learned as an ordered contour-token assignment
    problem rather than unconstrained coordinate regression.
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
        if config.contour_tokens_per_bone < 1:
            raise ValueError("contour_tokens_per_bone must be positive")
        if config.contour_temperature <= 0:
            raise ValueError("contour_temperature must be positive")
        if config.contour_kernel_size < 3 or config.contour_kernel_size % 2 == 0:
            raise ValueError("contour_kernel_size must be an odd value >= 3")
        if len(config.bone_class_groups) != config.num_bones:
            raise ValueError("bone_class_groups length must equal num_bones")

        invalid_classes = [
            class_id
            for group in config.bone_class_groups
            for class_id in group
            if class_id < 0 or class_id >= config.num_mask_classes
        ]
        if invalid_classes:
            raise ValueError(
                f"bone_class_groups references missing classes: {invalid_classes}"
            )

        self.config = config
        self.num_landmarks = config.num_landmarks
        self.num_bones = config.num_bones
        self.tokens_per_bone = config.contour_tokens_per_bone
        self.temperature = config.contour_temperature
        self.contour_kernel_size = config.contour_kernel_size
        self.bone_class_groups = tuple(
            tuple(group) for group in config.bone_class_groups
        )
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())

        if not torch.equal(
            self.point_bone_ids, self.point_bone_ids.sort().values
        ):
            raise ValueError("KneePV1 expects landmarks grouped by bone")

        self.backbone_adapter = RWKVUNetBackboneAdapter(
            segmentation_backbone,
            query_dim=config.query_dim,
            freeze_backbone=config.freeze_backbone,
        )
        self.landmark_queries = nn.Embedding(config.num_landmarks, config.query_dim)
        self.bone_embedding = nn.Embedding(config.num_bones, config.query_dim)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(2, config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.strength_embedding = nn.Linear(1, config.query_dim)
        self.token_norm = nn.LayerNorm(config.query_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.query_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.transformer_layers,
        )
        self.query_projection = nn.Linear(config.query_dim, config.query_dim)
        self.token_projection = nn.Linear(config.query_dim, config.query_dim)

        memory_bone_ids = torch.arange(config.num_bones).repeat_interleave(
            self.tokens_per_bone
        )
        memory_mask = self.point_bone_ids[:, None] != memory_bone_ids[None]
        self.register_buffer("memory_mask", memory_mask, persistent=False)

    def get_bone_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        maps = [
            probabilities[:, group].sum(dim=1)
            for group in self.bone_class_groups
        ]
        return torch.stack(maps, dim=1).clamp(0, 1)

    def _contour_maps(self, bone_probabilities: torch.Tensor) -> torch.Tensor:
        padding = self.contour_kernel_size // 2
        dilated = F.max_pool2d(
            bone_probabilities,
            self.contour_kernel_size,
            stride=1,
            padding=padding,
        )
        eroded = -F.max_pool2d(
            -bone_probabilities,
            self.contour_kernel_size,
            stride=1,
            padding=padding,
        )
        return (dilated - eroded).clamp_min(0)

    def _build_contour_tokens(
        self,
        feature_high: torch.Tensor,
        bone_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feature_high.shape
        if self.tokens_per_bone > height * width:
            raise ValueError(
                f"Requested {self.tokens_per_bone} contour tokens from a "
                f"{height}x{width} feature map"
            )

        resized_bones = F.interpolate(
            bone_probabilities,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        contour_maps = self._contour_maps(resized_bones)
        strengths, indices = contour_maps.flatten(2).topk(
            self.tokens_per_bone,
            dim=-1,
            sorted=True,
        )

        xs = (indices % width).to(feature_high.dtype) / max(width - 1, 1)
        ys = torch.div(indices, width, rounding_mode="floor").to(
            feature_high.dtype
        ) / max(height - 1, 1)
        coordinates = torch.stack([xs, ys], dim=-1)

        flattened_features = feature_high.flatten(2).transpose(1, 2)
        expanded_features = flattened_features[:, None].expand(
            -1, self.num_bones, -1, -1
        )
        sampled_features = torch.gather(
            expanded_features,
            dim=2,
            index=indices[..., None].expand(-1, -1, -1, channels),
        )
        bone_ids = torch.arange(
            self.num_bones, device=feature_high.device
        )[None, :, None]
        tokens = (
            sampled_features
            + self.coordinate_embedding(coordinates)
            + self.strength_embedding(strengths[..., None])
            + self.bone_embedding(bone_ids)
        )
        return self.token_norm(tokens), coordinates, strengths, contour_maps

    def _assignment_logits(
        self,
        decoded_queries: torch.Tensor,
        contour_tokens: torch.Tensor,
    ) -> torch.Tensor:
        queries = F.normalize(self.query_projection(decoded_queries), dim=-1)
        tokens = F.normalize(self.token_projection(contour_tokens), dim=-1)
        chunks = []
        for bone_id in range(self.num_bones):
            selected_queries = queries[:, self.point_bone_ids == bone_id]
            chunks.append(
                torch.einsum(
                    "bnd,bkd->bnk",
                    selected_queries,
                    tokens[:, bone_id],
                )
            )
        return torch.cat(chunks, dim=1) / self.temperature

    def forward(self, image: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        backbone = self.backbone_adapter(image)
        bone_probabilities = self.get_bone_probabilities(
            backbone["segmentation_logits"]
        )
        contour_tokens, contour_coordinates, strengths, contour_maps = (
            self._build_contour_tokens(
                backbone["feature_high"],
                bone_probabilities,
            )
        )

        batch = image.shape[0]
        queries = self.landmark_queries.weight[None].expand(batch, -1, -1)
        queries = queries + self.bone_embedding(self.point_bone_ids)[None]
        memory = contour_tokens.flatten(1, 2)
        decoded = self.decoder(
            queries,
            memory,
            memory_mask=self.memory_mask,
        )
        assignment_logits = self._assignment_logits(decoded, contour_tokens)
        probabilities = torch.softmax(assignment_logits, dim=-1)

        candidates = contour_coordinates[:, self.point_bone_ids]
        soft_coordinates = (
            probabilities[..., None] * candidates
        ).sum(dim=-2)
        hard_indices = assignment_logits.argmax(dim=-1)
        hard_coordinates = torch.gather(
            candidates,
            dim=2,
            index=hard_indices[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        # Forward values are exactly on the contour; gradients follow the soft
        # assignment so coordinate supervision remains differentiable.
        final_coordinates = soft_coordinates + (
            hard_coordinates - soft_coordinates
        ).detach()

        return {
            "segmentation_logits": backbone["segmentation_logits"],
            "bone_probabilities": bone_probabilities,
            "contour_maps": contour_maps,
            "contour_token_strengths": strengths,
            "contour_candidate_coordinates": candidates,
            "contour_assignment_logits": assignment_logits,
            "coarse_landmarks": soft_coordinates,
            "final_landmarks": final_coordinates,
            "landmark_confidence": probabilities.max(dim=-1).values,
        }
