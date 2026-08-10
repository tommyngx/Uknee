# Uknee Landmark Pose

`landmark` is a standalone knee-landmark package. It vendors Ultralytics
8.4.87 under `_vendor/ultralytics`, so **do not install the `ultralytics` pip
package**. Importing `landmark` bootstraps and verifies the local snapshot;
accidentally importing a pip or `Ref` copy first raises an error.

## Public models

| YAML | Role | Output |
|---|---|---|
| `yolo26-pose.yaml` | Stable default, P3–P5 Pose26 | 4 objects × 51 padded points |
| `yolo26-pose-v1.yaml` | P2–P5 plus 129-point global auxiliary loss | same |
| `yolo26-pose-v9.yaml` | V1 plus P4 ROIAlign region refinement | same, refined |
| `hrnet-w32-pose.yaml` | Full HRNet-W32 heatmap baseline | canonical 129, adapted to 4×51 |
| `vitpose-s-pose.yaml` | ViTPose-S 384/12/6 heatmap baseline | canonical 129, adapted to 4×51 |

The three YOLO YAMLs use normal Ultralytics compound scaling. Their checked-in
default is `scale: n`; change it to `s`, `m`, `l` or `x` for the corresponding
depth/width profile.

## Install and train

Install only the runtime dependencies:

```bash
pip install -r landmark/requirements.txt
```

Single process:

```bash
python -m landmark.train \
  --model landmark/cfg/models/yolo26-pose-v9.yaml \
  --data landmark/cfg/datasets/mesko4gf2.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0 \
  --project landmark/runs/pose --name pose-v9
```

Two-process DDP:

```bash
torchrun --standalone --nproc-per-node=2 -m landmark.train \
  --model landmark/cfg/models/yolo26-pose.yaml \
  --data landmark/cfg/datasets/mesko4gf2.yaml \
  --batch 16 --device 0,1
```

Any additional vendored trainer setting can be appended in native
`KEY=VALUE` form, for example `optimizer=AdamW lr0=0.001 cos_lr=true`.

The data preflight validates all labels, enforces four classes and
`kpt_shape=[51,3]`, verifies the actual `45/51/24/9` visible slots, and creates
a deterministic case-grouped split (seed 2006) when train and val point to the
same images. The resolved YAML is copied into the run directory.

V1, V9, HRNet and ViTPose force `mosaic=mixup=cutmix=0`; their global target
supports one instance per anatomical class. RandomPerspective, HSV and other
single-image transforms remain the vendored implementations. Base YOLO26 keeps
the upstream multi-image augmentation defaults.

## Python API

```python
from landmark import KneePose

model = KneePose("landmark/cfg/models/yolo26-pose.yaml")
model.train(data="landmark/cfg/datasets/mesko4gf2.yaml", epochs=100, device=0)
metrics = model.val(data="landmark/cfg/datasets/mesko4gf2.yaml")
results = model.predict("image.png")
artifact = model.export(format="onnx", imgsz=640)
```

The package root intentionally exposes only the training entry point. The
`KneePose` API and result, validation, prediction and export helpers live under
`landmark/utils`; `KneePose` remains importable directly from `landmark`.

Each prediction delegates the normal Ultralytics `Results` fields and adds:

- `boxes_xyxy`, `scores`, `class_ids`;
- `keypoints.data[N,51,3]` in original-image pixels;
- `landmarks_xy[129,2]` normalized to `[0,1]`;
- `landmark_confidence[129]`.

The adapter takes the highest-score object of each class. Missing classes and
padded slots have zero confidence. ONNX and TorchScript export return fixed
`detections[B,4,159]`, `num_detections[B]`, and `canonical[B,129,3]` outputs.

Training keeps upstream AMP, EMA, nominal-batch accumulation, optimizer auto,
warmup/scheduler, scaled weight decay, early stopping, resume and DDP sampler.
It writes `last.pt`, `best.pt`, and a separate `best_mre.pt`. Use
`export_state_dict()` for a weights-only paper archive.

See [architecture](skill/architecture.md),
[paper experiments](PAPER_EXPERIMENTS.md), and
[_vendor/ultralytics/VENDORED.md](_vendor/ultralytics/VENDORED.md).

## License

The combined project is AGPL-3.0. Vendored files retain their original
Ultralytics headers. Historical Apache-2.0 text is retained at the repository
root for attribution; see `THIRD_PARTY_NOTICES.md`.
