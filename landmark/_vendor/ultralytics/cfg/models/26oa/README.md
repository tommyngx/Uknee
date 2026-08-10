# YOLO26 OA Pose Experiments

This folder contains experimental YOLO26 pose model variants for dense medical knee X-ray landmarks. V1-v8 default to
`nc: 1`, `kpt_shape: [129, 3]`; v9 follows MESKO4GF2 with `nc: 4`, `kpt_shape: [51, 3]`.

All models keep standard YOLO pose prediction compatibility unless noted. Prediction, validation, export, and downstream
postprocessing should still receive normal YOLO pose keypoints.

## Model Summary

| Model | YAML | Backbone | Head | Extra loss | Output | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | `ultralytics/cfg/models/26oa/yolo26-posev1.yaml` | YOLO26-style P2-P5 | `OA26HeatmapPose` | Heatmap + coord + neighbour + curve | Standard pose | Test full auxiliary landmark supervision |
| v2 | `ultralytics/cfg/models/26oa/yolo26-posev2.yaml` | YOLO26-style P2-P5 | `OA26HeatmapPose` | Heatmap only | Standard pose | Isolate heatmap supervision effect |
| v3 | `ultralytics/cfg/models/26oa/yolo26-posev3.yaml` | YOLO26-style P2-P5 | `OA26SimCCPose` | SimCC x/y | Standard pose | Test coordinate-distribution supervision |
| v4 | `ultralytics/cfg/models/26oa/yolo26-posev4.yaml` | `HRNetLite` | `Pose26` | None | Standard pose | Lightweight HRNet-style baseline |
| v5 | `ultralytics/cfg/models/26oa/yolo26-posev5.yaml` | `ConvNeXtV2N` pretrained | `Pose26` | None | Standard pose | Isolate pretrained ConvNeXtV2-Nano backbone effect |
| v6 | `ultralytics/cfg/models/26oa/yolo26-posev6.yaml` | YOLO26-style P2-P5 + `ViTRefine` on P4 | `Pose26` | None | Standard pose | Test ViTPose-style global token refinement |
| v7 | `ultralytics/cfg/models/26oa/yolo26-posev7.yaml` | canonical HRNet-W32 | `Pose26` | None | Standard pose | Balanced canonical HRNet baseline |
| v8 | `ultralytics/cfg/models/26oa/yolo26-posev8.yaml` | canonical HRNet-W48 | `Pose26` | None | Standard pose | Accuracy-first canonical HRNet variant |
| v9 | `ultralytics/cfg/models/26oa/yolo26-posev9.yaml` | YOLO26-style P2-P5 | `OA26RegionRefinePose` | v1 + region heatmap/coord/structure | Standard 51-point pose | Independent P4 ROI query refinement per detected bone |

## Module Locations

Shared v1-v8 modules remain in `ultralytics/nn/modules/oa26/`:

- `pose_heads.py`: `OA26HeatmapPose`, `OA26SimCCPose`
- `hrnet.py`: canonical `HRNet` plus lightweight `HRNetLite` used by v4
- `convnextv2_n.py`: `ConvNeXtV2N` for Nano
- `convnextv2_t.py`: `ConvNeXtV2T` for Tiny, kept for quick future swaps
- `vit_refine.py`: `ViTRefine` for lightweight transformer refinement

V9 is isolated under `ultralytics/nn/modules/oa26_region_refine/`; its schema and loss live under
`ultralytics/utils/oa26_region_refine/`. This keeps v1-v8 implementation files unchanged.

## Quick Training Commands

Auxiliary-loss variants:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev1.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev2.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev3.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Backbone-only variants:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev4.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev5.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev7.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev8.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Transformer-refinement variant:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev6.yaml data=your_knee_pose.yaml imgsz=896 epochs=100
```

Landmark-refinement variant:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev9.yaml data=/path/to/yolo_mesko4GF2/data.yaml imgsz=896 epochs=100
```

Safer smoke runs:

```bash
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev1.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev5.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
yolo pose train model=ultralytics/cfg/models/26oa/yolo26-posev6.yaml data=your_knee_pose.yaml imgsz=768 epochs=5 batch=2
```

Swap the model path in the smoke command to compare all six variants under the same data, image size, batch size, and
seed.

## Recommended Experiment Order

1. Build each model from YAML.
2. Run a small dummy forward smoke test.
3. Train for 5 epochs at `imgsz=768`, small batch.
4. Train the best candidates at `imgsz=896`.
5. Compare against original `ultralytics/cfg/models/26/yolo26-pose.yaml`.

## Memory Notes

- v1-v3 include a P2 stride-4 path and auxiliary branches, so they can use more VRAM than original YOLO26 pose.
- v4 is the lightweight HRNet-style baseline. v7 and v8 implement canonical HRNet-Pose W32 and W48 respectively; all three keep the standard `Pose26` head and a P2 stride-4 feature path.
- v8 has substantially higher VRAM and latency cost than v7. Benchmark both under an identical training protocol before selecting a production model.
- v6 applies transformer attention on P4/16 only; do not move full attention to P2/4 at `imgsz=896`.
- v9 extracts one `20 x 20` P4 ROI per detected class instance. Queries cross-attend to the complete ROI and
  self-attend only within their own bone; it has no cross-class attention and no patch-per-landmark sampling.
- MESKO4GF2 uses femur/tibia/fibula/patella as classes 0/1/2/3 with 45/51/24/9 valid points padded to 51 slots.
- V9 writes `pose_detection_performance.png` in its run folder after every epoch (2 detection + 2 pose panels, top 3).
- Start with `imgsz=768` and small batch size before moving to `imgsz=896`.
- v5 requires `timm` for `convnextv2_nano` and downloads pretrained weights on first use if they are not cached.

## Medical Landmark Metrics

Do not rely only on COCO-style pose mAP for this task. Track:

- mean radial error in pixels
- normalized mean error using knee crop width or tibial width
- percentage of keypoints within 2, 4, and 8 pixels
- per-region landmark error: femur, tibia, joint margin, osteophyte-related points
- downstream B-score error
- downstream JSW measurement error
- per-image failure rate

## Export Notes

The public output is standard YOLO pose keypoints. Still test ONNX/export separately for custom variants; v9 adds
ROIAlign, transformer attention, class-specific pose selection, and soft-argmax operations.
