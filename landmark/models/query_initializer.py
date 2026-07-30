from __future__ import annotations

import math

import torch
from torch import nn


class CoordinateFourierEmbedding(nn.Module):
    def __init__(self, output_dim: int, num_frequencies: int = 16):
        super().__init__()
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(num_frequencies * 4, output_dim)

    def forward(self, coordinates_xy: torch.Tensor) -> torch.Tensor:
        angles = coordinates_xy[..., None] * self.frequencies * (2 * math.pi)
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.projection(encoding)


class LandmarkQueryInitializer(nn.Module):
    def __init__(
        self,
        num_landmarks: int,
        num_bones: int,
        query_dim: int,
        point_bone_ids: torch.Tensor,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        self.landmark_embedding = nn.Embedding(num_landmarks, query_dim)
        self.bone_embedding = nn.Embedding(num_bones, query_dim)
        self.coordinate_embedding = CoordinateFourierEmbedding(query_dim)
        self.local_projection = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        coarse_coordinates: torch.Tensor,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ids = torch.arange(self.num_landmarks, device=coarse_coordinates.device)
        identity = self.landmark_embedding(ids)[None]
        bone = self.bone_embedding(self.point_bone_ids)[None]
        query = (
            identity
            + bone
            + self.local_projection(local_tokens)
            + self.coordinate_embedding(coarse_coordinates)
        )
        return self.norm(query)
