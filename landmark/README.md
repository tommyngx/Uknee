# Uknee Landmark Models

This package is independent from the existing segmentation training code. It
loads class-wise YOLO-Pose labels, builds one ordered 129-point target, and
supports three model names:

- `adaptive_rwkv` / `adaptive_detr_rwkv`: frozen RWKV_UNetV3 segmentation
  backbone plus coarse-to-local anatomical landmark queries.
- `kneepv1`: frozen 11-class RWKV segmentation, differentiable contour
  extraction and ordered DETR queries whose final points are snapped to the
  predicted contour of the correct bone.
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
python -m landmark.train \
  --checkpoint /data/tommyngu/BMammo/Knee/Run_OApheno/Unet_mesko_640_RWKV3_9class2/checkpoint_last.pth \
  --yaml-path /projects/BMammo/Knee/data/yolo_mesko4GF2/data.yaml \
  --batch-size 8 \
  --epochs 100 \
  --device cuda \
  --output-dir /projects/BMammo/Knee/Run_OApheno \
  --exp-name Landmark_RWKV3_11class \
  --seed 2006 \
  --img-size 640 \
  --aug-strategy xray
```

`--num-mask-classes` now defaults to `11`, so it is not needed in the command
above. The CLI also accepts underscore aliases matching `main.py`, for example
`--output_dir`, `--img_size`, `--aug_strategy`, `--base_lr`, and
`--pretrained_path`.

Baseline examples:

```bash
python -m landmark.train --config landmark/config/kneepv1.yaml
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

`kneepv1` does not use that curriculum. It trains its contour-token assignment
decoder end-to-end from epoch 1 while keeping only the segmentation backbone
frozen. Its `contour_oracle_px` history value is the lower-bound error supplied
by the current segmentation contours.

## Evaluate

```bash
python -m landmark.evaluate \
  --config landmark/config/adaptive_rwkv.yaml \
  --checkpoint landmark/runs/<experiment>/<timestamp>/checkpoints/best.pt
```

## Important configuration

The landmark configuration now matches the 11-class `RWKV_UNetV3` run:
640×640, three backbone input channels and eleven segmentation outputs. The
bundled metadata defines background 0, core bone classes 1–5, femoral child
regions 6–7 and tibial child regions 8–10. The four soft anatomical maps merge
`[[1,6,7], [2,4,8,9,10], [3,4], [5]]` for femur, tibia, fibula and patella.
This remains editable as `model.bone_class_groups`.

The default loader keeps pixels in `[0, 1]` (`mean=0`, `std=1`) to match the
preprocessing used to train that checkpoint. The adapter repeats the single
grayscale channel to three channels only at the backbone boundary.

The `path` inside `Ref/yolo_mesko4GF2/data.yaml` points to a different machine.
The loader detects this and falls back to the directory containing that YAML.
Because its train/val entries both name the training directory, the loader
creates a deterministic validation split and prevents leakage.
