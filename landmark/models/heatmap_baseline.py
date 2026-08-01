from __future__ import annotations

import torch

from landmark.utils.coordinates import soft_argmax_2d


def decode_global_heatmaps(heatmaps: torch.Tensor) -> dict[str, torch.Tensor]:
    local, confidence = soft_argmax_2d(heatmaps, temperature=1.0)
    coordinates = (local + 1.0) / 2.0
    return {
        "coarse_landmarks": coordinates,
        "final_landmarks": coordinates,
        "landmark_confidence": confidence,
        "global_heatmaps": heatmaps,
    }
