# Landmark architecture

## Data and public contract

MESKO4GF2 is represented as four YOLO-Pose objects in fixed class order:
femur, tibia, fibula and patella. Every row allocates 51 `(x,y,visibility)`
slots; the real counts are `45/51/24/9`. The canonical adapter concatenates
those regions into 129 points and preserves six independent paths:

```text
femur 0:45
tibia main 45:86 | tibia path 2 86:91 | tibia path 3 91:96
fibula 96:120
patella 120:129
```

No neighbour or curvature loss crosses these boundaries.

## Runtime

```text
landmark import
  -> bootstrap pinned landmark/_vendor/ultralytics
  -> KneePose API / CLI
  -> dataset preflight + leakage-safe manifests
  -> vendored trainer, validator, predictor and exporter
       |-- YOLO26 Pose / V1 / V9
       `-- HeatmapPoseTrainer -> HRNet-W32 / ViTPose-S
  -> Ultralytics Results + canonical 129-point adapter
```

The YOLO branch retains the vendored parser, backbone/PAN, Pose26 end-to-end
head, assigner, detection/pose/RLE losses and data pipeline. V1 adds a global
129-channel auxiliary heatmap with visibility-masked heatmap, coordinate,
neighbour and curvature terms. V9 preserves V1 and injects refined class-local
keypoints from P4 ROIAlign, class queries, a region transformer and localization
head.

HRNet-W32 follows the multi-resolution stage layout (2, 3 and 4 branches with
repeated fusion) and predicts stride-4 heatmaps. ViTPose-S uses a 16-pixel patch
embedding, 384-dimensional 12-layer/6-head transformer encoder and two-stage
deconvolution head. Both predict 129 full-frame heatmaps and derive four boxes
from each region's landmark envelope.

## Validation and export

The pose validator preserves box/pose precision, recall and mAP. Its extension
adds pixel MRE, median/p95, PCK2/4/8, image failure rate, per-region MRE and
path order/topology measurements. Upstream fitness still selects `best.pt`;
MRE independently selects `best_mre.pt`.

The deployment wrapper has three fixed outputs: four detection rows,
`num_detections`, and canonical `[B,129,3]`. V1 training-only auxiliary tensors
are not exported. V9 export uses its refined detections and leaves ROIAlign in
place. During ONNX export only, the PyTorch fused MHA evaluation fast-path is
disabled so attention decomposes into equivalent standard ONNX operators.
