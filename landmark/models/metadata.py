from __future__ import annotations

import torch

from landmark.data.yolo_pose import BONE_NAMES, NUM_LANDMARKS, POINT_BONE_IDS, POINT_COUNTS

LANDMARK_NAMES = tuple(
    f"{BONE_NAMES[bone_id]}_{point_index:02d}"
    for bone_id, point_count in enumerate(POINT_COUNTS)
    for point_index in range(point_count)
)

# The tibial annotation contains one main contour followed by two shorter
# plateau contours.  Those three paths must not be joined merely because they
# share the same bone class.  Keeping the paths explicit also gives models a
# stable definition of landmark order that is independent of image direction.
LANDMARK_PATH_RANGES = (
    (0, 45),    # femur
    (45, 86),   # tibia main contour
    (86, 91),   # tibia plateau contour A
    (91, 96),   # tibia plateau contour B
    (96, 120),  # fibula
    (120, 129), # patella
)
LANDMARK_PATHS = tuple(
    tuple(range(start, stop)) for start, stop in LANDMARK_PATH_RANGES
)

# Contour adjacency is explicit per anatomical path.  In particular, it does
# not create the false tibial edges 85->86 or 90->91.
TOPOLOGY_EDGES = tuple(
    (index, index + 1)
    for start, stop in LANDMARK_PATH_RANGES
    for index in range(start, stop - 1)
)
TOPOLOGY_TRIPLETS = tuple(
    (index, index + 1, index + 2)
    for start, stop in LANDMARK_PATH_RANGES
    for index in range(start, stop - 2)
)


def validate_metadata() -> None:
    if NUM_LANDMARKS != 129 or len(POINT_BONE_IDS) != NUM_LANDMARKS:
        raise RuntimeError("Landmark metadata must describe exactly 129 points")
    expected = torch.tensor([0] * 45 + [1] * 51 + [2] * 24 + [3] * 9)
    if not torch.equal(POINT_BONE_IDS.cpu(), expected):
        raise RuntimeError("Unexpected class-wise landmark mapping")
    flattened_paths = tuple(index for path in LANDMARK_PATHS for index in path)
    if flattened_paths != tuple(range(NUM_LANDMARKS)):
        raise RuntimeError("Landmark paths must cover every landmark exactly once")


validate_metadata()
