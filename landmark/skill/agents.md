# Agent Guide

When updating this package:

1. Preserve `(x, y)` coordinate order and `[0, 1]` model coordinates.
2. Preserve the class-derived landmark mapping 45/51/24/9. Do not infer a
   different mapping from point indices.
3. Do not modify the default output of the existing RWKV U-Net.
4. Do not use forward hooks in the deployable adapter.
5. Do not update frozen backbone BatchNorm statistics.
6. Mask every coordinate and heatmap loss by `landmark_visibility`.
7. Do not enable horizontal flips without a verified identity remap.
8. Add a YAML file and registry entry for each new model baseline.
9. Run `python -m unittest discover -s landmark/tests -v` after changes.
10. Run a small 8–16 image overfit experiment before full training.

`bone_class_groups` refers to the seven-class segmentation checkpoint, not the
four YOLO-Pose class IDs. The default uses `[[1,6], [2,4], [3,4], [5]]` from
the bundled metadata. Confirm the historical checkpoint used that class
definition before publishing measurements.
