from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from landmark.utils.coordinates import create_local_sampling_grid, unit_to_grid_sample


def _sample_patches(
    features: torch.Tensor,
    coordinates: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = features.shape
    landmark_count = coordinates.shape[1]
    grid, radius = create_local_sampling_grid(
        coordinates, patch_size, height, width
    )
    merged_grid = grid.reshape(batch, landmark_count * patch_size, patch_size, 2)
    sampled = F.grid_sample(
        features,
        merged_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    patches = sampled.view(
        batch, channels, landmark_count, patch_size, patch_size
    ).permute(0, 2, 1, 3, 4).contiguous()
    return patches, radius


class MultiScaleLocalFeatureSampler(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_landmarks: int,
        point_bone_ids: torch.Tensor,
        local_patch_size: int = 24,
        token_patch_size: int = 3,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.local_patch_size = local_patch_size
        self.token_patch_size = token_patch_size
        self.register_buffer("point_bone_ids", point_bone_ids.long().clone())
        self.local_projection = nn.Sequential(
            nn.Linear(query_dim * 3 + 1, query_dim * 2),
            nn.GELU(),
            nn.Linear(query_dim * 2, query_dim),
        )

    def forward(
        self,
        feature_high: torch.Tensor,
        feature_mid: torch.Tensor,
        feature_low: torch.Tensor,
        bone_probabilities: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        high_patches, radius = _sample_patches(
            feature_high, coordinates, self.local_patch_size
        )
        high_tokens = high_patches.mean(dim=(-2, -1))
        mid_patches, _ = _sample_patches(feature_mid, coordinates, self.token_patch_size)
        low_patches, _ = _sample_patches(feature_low, coordinates, self.token_patch_size)
        mid_tokens = mid_patches.mean(dim=(-2, -1))
        low_tokens = low_patches.mean(dim=(-2, -1))

        point_grid = unit_to_grid_sample(coordinates)[:, :, None, :]
        sampled_bones = F.grid_sample(
            bone_probabilities,
            point_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(-1).transpose(1, 2)
        selected = sampled_bones.gather(
            dim=-1,
            index=self.point_bone_ids[None, :, None].expand(coordinates.shape[0], -1, 1),
        )
        tokens = self.local_projection(
            torch.cat([high_tokens, mid_tokens, low_tokens, selected], dim=-1)
        )
        return {
            "local_tokens": tokens,
            "local_patches_high": high_patches,
            "patch_radius_xy": radius,
        }
