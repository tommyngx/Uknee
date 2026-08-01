from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from landmark.config.loader import ModelConfig

from .kneepv1 import KneePV1ContourDETR
from .metadata import LANDMARK_PATH_RANGES, POINT_BONE_IDS


class _PathMixerBlock(nn.Module):
    """Mix immediate contour neighbours without crossing path boundaries."""

    def __init__(self, query_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(query_dim)
        self.depthwise = nn.Conv1d(
            query_dim,
            query_dim,
            kernel_size=3,
            padding=1,
            groups=query_dim,
        )
        self.pointwise = nn.Conv1d(query_dim, query_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        residual = queries
        mixed = self.norm(queries).transpose(1, 2)
        mixed = self.pointwise(F.gelu(self.depthwise(mixed)))
        return residual + self.dropout(mixed.transpose(1, 2))


class KneePV2TopologyDETR(KneePV1ContourDETR):
    """Contour DETR with explicit path order and topology-aware decoding.

    KneePV1 identifies points independently inside a bone.  V2 additionally
    identifies the anatomical path and normalized position within that path,
    then mixes neighbouring decoded queries before contour assignment.  At
    evaluation time, greedy confidence-priority matching prevents two points
    in the same path from snapping to the same contour token.
    """

    def __init__(
        self,
        segmentation_backbone: nn.Module,
        config: ModelConfig,
        point_bone_ids: torch.Tensor = POINT_BONE_IDS,
    ):
        super().__init__(segmentation_backbone, config, point_bone_ids)
        if config.topology_mixer_layers < 1:
            raise ValueError("topology_mixer_layers must be positive")
        if config.topology_unique_inference and config.contour_tokens_per_bone < max(
            stop - start for start, stop in LANDMARK_PATH_RANGES
        ):
            raise ValueError(
                "contour_tokens_per_bone must cover the longest landmark path "
                "when topology_unique_inference is enabled"
            )

        path_ids = torch.empty(config.num_landmarks, dtype=torch.long)
        order_features = torch.empty(config.num_landmarks, 5)
        for path_id, (start, stop) in enumerate(LANDMARK_PATH_RANGES):
            position = torch.linspace(0, 1, stop - start)
            path_ids[start:stop] = path_id
            order_features[start:stop] = torch.stack(
                (
                    position,
                    torch.sin(math.pi * position),
                    torch.cos(math.pi * position),
                    torch.sin(2 * math.pi * position),
                    torch.cos(2 * math.pi * position),
                ),
                dim=-1,
            )
        self.register_buffer("landmark_path_ids", path_ids)
        self.register_buffer("landmark_order_features", order_features)

        self.path_embedding = nn.Embedding(
            len(LANDMARK_PATH_RANGES), config.query_dim
        )
        self.order_embedding = nn.Sequential(
            nn.Linear(order_features.shape[-1], config.query_dim),
            nn.GELU(),
            nn.Linear(config.query_dim, config.query_dim),
        )
        self.path_mixers = nn.ModuleList(
            _PathMixerBlock(config.query_dim, config.transformer_dropout)
            for _ in range(config.topology_mixer_layers)
        )
        self.unique_inference = config.topology_unique_inference

    def _build_landmark_queries(self, batch_size: int) -> torch.Tensor:
        queries = super()._build_landmark_queries(batch_size)
        topology = self.path_embedding(self.landmark_path_ids)
        topology = topology + self.order_embedding(self.landmark_order_features)
        return queries + topology[None]

    def _refine_decoded_queries(self, decoded: torch.Tensor) -> torch.Tensor:
        for mixer in self.path_mixers:
            decoded = torch.cat(
                [mixer(decoded[:, start:stop]) for start, stop in LANDMARK_PATH_RANGES],
                dim=1,
            )
        return decoded

    def _hard_assignment_indices(
        self, assignment_logits: torch.Tensor
    ) -> torch.Tensor:
        independent = super()._hard_assignment_indices(assignment_logits)
        if self.training or not self.unique_inference:
            return independent

        # Confidence-priority greedy matching is deterministic and guarantees
        # uniqueness within each anatomical path.  Separate tibial paths may
        # still share nearby candidates because their annotations overlap.
        selected_indices = independent.clone()
        token_count = assignment_logits.shape[-1]
        for batch_index in range(assignment_logits.shape[0]):
            for start, stop in LANDMARK_PATH_RANGES:
                scores = assignment_logits[batch_index, start:stop]
                confidence_order = scores.max(dim=-1).values.argsort(descending=True)
                used_tokens = torch.zeros(
                    token_count, dtype=torch.bool, device=scores.device
                )
                selected = torch.empty(
                    stop - start, dtype=torch.long, device=scores.device
                )
                for rank in range(stop - start):
                    point_index = confidence_order[rank]
                    available_scores = scores[point_index].masked_fill(
                        used_tokens, -torch.inf
                    )
                    token_index = available_scores.argmax()
                    selected[point_index] = token_index
                    used_tokens[token_index] = True
                selected_indices[batch_index, start:stop] = selected
        return selected_indices

    def forward(self, image: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        outputs = super().forward(image, **kwargs)
        # This explicit key activates V2-only topology losses without changing
        # the behavior of existing KneePV1 configurations.
        outputs["topology_soft_landmarks"] = outputs["coarse_landmarks"]
        return outputs
