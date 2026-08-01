from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from landmark.utils.coordinates import soft_argmax_2d


def _groups(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


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
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        hidden = max(query_dim // 2, 16)
        self.spatial_head = nn.Sequential(
            nn.Conv2d(query_dim, query_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(query_dim), query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(query_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_landmarks, kernel_size=1),
        )
        self.temperature = 1.0
        self.bone_prior_strength = 0.1

    def forward(
        self,
        feature_mid: torch.Tensor,
        feature_low: torch.Tensor,
        bone_probabilities: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del feature_low  # Kept in the interface for adapter compatibility.
        bone_maps = F.interpolate(
            bone_probabilities,
            feature_mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        landmark_bone_prior = bone_maps[:, self.point_bone_ids]
        heatmaps = self.spatial_head(feature_mid)
        heatmaps = heatmaps + self.bone_prior_strength * torch.log(
            landmark_bone_prior.clamp_min(1.0e-4)
        )
        local_coordinates, confidence = soft_argmax_2d(
            heatmaps, self.temperature
        )
        return {
            "coordinates": (local_coordinates + 1) / 2,
            "heatmaps": heatmaps,
            "confidence": confidence,
        }
