from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CoarseReferenceHead(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        num_bones: int,
        query_dim: int,
        point_bone_ids: torch.Tensor,
        dropout: float = 0.1,
    ):
        super().__init__()
        if point_bone_ids.numel() != num_landmarks:
            raise ValueError("point_bone_ids length must equal num_landmarks")
        self.num_landmarks = num_landmarks
        self.num_bones = num_bones
        self.landmark_embedding = nn.Embedding(num_landmarks, query_dim)
        self.bone_embedding = nn.Embedding(num_bones, query_dim)
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(query_dim * 4, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 2),
        )

    def forward(
        self,
        feature_mid: torch.Tensor,
        feature_low: torch.Tensor,
        bone_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels = feature_low.shape[:2]
        global_token = F.adaptive_avg_pool2d(feature_low, 1).flatten(1)
        global_tokens = global_token[:, None].expand(-1, self.num_landmarks, -1)
        bone_maps = F.interpolate(
            bone_probabilities,
            feature_mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        weighted = (
            feature_mid[:, None] * bone_maps[:, :, None]
        ).sum(dim=(-2, -1))
        normalizer = bone_maps.sum(dim=(-2, -1)).unsqueeze(-1).clamp_min(1e-6)
        bone_tokens = weighted / normalizer
        selected_bone_tokens = bone_tokens[:, self.point_bone_ids]
        landmark_ids = torch.arange(self.num_landmarks, device=feature_mid.device)
        landmark_tokens = self.landmark_embedding(landmark_ids)[None].expand(batch, -1, -1)
        bone_identity = self.bone_embedding(self.point_bone_ids)[None].expand(batch, -1, -1)
        inputs = torch.cat(
            [landmark_tokens, bone_identity, global_tokens, selected_bone_tokens],
            dim=-1,
        )
        return torch.sigmoid(self.coordinate_mlp(inputs))
