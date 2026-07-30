from __future__ import annotations

import torch

from landmark.data.yolo_pose import BONE_NAMES, NUM_LANDMARKS, POINT_BONE_IDS, POINT_COUNTS

LANDMARK_NAMES = tuple(
    f"{BONE_NAMES[bone_id]}_{point_index:02d}"
    for bone_id, point_count in enumerate(POINT_COUNTS)
    for point_index in range(point_count)
)

# Contour adjacency is explicit per bone and never crosses class boundaries.
TOPOLOGY_EDGES = tuple(
    (offset + index, offset + index + 1)
    for offset, count in zip((0, 45, 96, 120), POINT_COUNTS)
    for index in range(count - 1)
)


def validate_metadata() -> None:
    if NUM_LANDMARKS != 129 or len(POINT_BONE_IDS) != NUM_LANDMARKS:
        raise RuntimeError("Landmark metadata must describe exactly 129 points")
    expected = torch.tensor([0] * 45 + [1] * 51 + [2] * 24 + [3] * 9)
    if not torch.equal(POINT_BONE_IDS.cpu(), expected):
        raise RuntimeError("Unexpected class-wise landmark mapping")


validate_metadata()
