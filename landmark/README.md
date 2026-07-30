# Uknee Landmark Models

This package is independent from the existing segmentation training code. It
loads class-wise YOLO-Pose labels, builds one ordered 129-point target, and
supports three model names:

- `adaptive_rwkv` / `adaptive_detr_rwkv`: frozen RWKV_UNetV3 segmentation
  backbone plus coarse-to-local anatomical landmark queries.
- `vitpose`: compact ViTPose-style heatmap baseline.
- `hrnet`: compact two-resolution HRNet-style heatmap baseline.

The landmark order is fixed by the reference data:

| Bone | Class | Points | Global indices |
|---|---:|---:|---:|
| femur | 0 | 45 | 0–44 |
| tibia | 1 | 51 | 45–95 |
| fibula | 2 | 24 | 96–119 |
| patella | 3 | 9 | 120–128 |

All coordinates inside the package use normalised `(x, y)` order. Missing
points are represented only by `landmark_visibility=0`; they are excluded from
losses and metrics.

## Train

From the repository root:

```bash
python -m landmark.train --config landmark/config/adaptive_rwkv.yaml
python -m landmark.train --config landmark/config/vitpose.yaml
python -m landmark.train --config landmark/config/hrnet.yaml
```

Each run creates:

```text
landmark/runs/<experiment>/<timestamp>/
├── checkpoints/
│   ├── best.pt
│   └── epoch_XXXX.pt
├── logs/train.log
├── plots/
│   ├── training_curves.png
│   └── epochs/epoch_XXXX.png
├── config_resolved.yaml
├── history.csv
└── result.json
```

The adaptive model uses this curriculum:

1. coarse head and feature projections;
2. local refinement around noisy ground-truth references;
3. complete landmark decoder using its predicted references.

The segmentation backbone remains frozen and stays in evaluation mode.

## Evaluate

```bash
python -m landmark.evaluate \
  --config landmark/config/adaptive_rwkv.yaml \
  --checkpoint landmark/runs/<experiment>/<timestamp>/checkpoints/best.pt
```

## Important configuration

The supplied segmentation checkpoint was trained as `RWKV_UNetV3`, 640×640,
3-channel input and 7 mask classes. The bundled segmentation metadata defines
background 0, femur 1, tibia 2, fibula 3, tibia–fibula overlap 4, patella 5,
and a femoral child region 6. The default four soft anatomical maps therefore
merge `[[1,6], [2,4], [3,4], [5]]`. This is editable as
`model.bone_class_groups`; verify the historical checkpoint used the same
class definition before publishing a full experiment.

The default loader keeps pixels in `[0, 1]` (`mean=0`, `std=1`) to match the
preprocessing used to train that checkpoint. The adapter repeats the single
grayscale channel to three channels only at the backbone boundary.

The `path` inside `Ref/yolo_mesko4GF2/data.yaml` points to a different machine.
The loader detects this and falls back to the directory containing that YAML.
Because its train/val entries both name the training directory, the loader
creates a deterministic validation split and prevents leakage.
