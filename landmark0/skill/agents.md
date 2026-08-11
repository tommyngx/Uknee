# Landmark maintenance invariants

1. Import `landmark0` before any absolute `ultralytics` import and keep runtime
   pinned to `landmark0/_vendor/ultralytics` version 8.4.87.
2. Preserve class order femur/tibia/fibula/patella, padded shape `[51,3]` and
   actual counts `45/51/24/9`.
3. Never connect neighbour/curve losses across the six path ranges in
   `landmark0.data.schema.LANDMARK_PATH_RANGES`.
4. Keep multi-image augmentation disabled for V1, V9, HRNet and ViTPose unless
   their target representation is redesigned for multiple instances/class.
5. Do not replace ROIAlign during V9 training or ordinary inference.
6. Keep `last.pt` resumable and select the flat-layout `best.pt` by minimum validation MRE.
7. Preserve standard YOLO `Results`, checkpoint and CLI semantics.
8. Run `python -m unittest discover -s landmark0/tests -v` after changes and a
   real 8–16 image overfit plus two-process DDP smoke before paper experiments.
