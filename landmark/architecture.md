# Landmark architecture

## Contract

`landmark` is the active self-contained knee detection and pose runtime. It
does not depend on `landmark0` or the PyPI `ultralytics` package.

- Detection: two knee-side classes with YOLO `xywh` labels.
- Pose: four regions in fixed order `femur/tibia/fibula/patella`, padded to
  51 keypoints per object and adapted to 129 canonical landmarks. This knee
  layout is the default schema; another four-region schema (e.g. hip
  `59/7/9/10` → 85 landmarks) is declared by a dataset YAML, see Invariants.
- Spatial CLI values and exported metadata use `[height, width]`.
- Public deployment formats are TorchScript and ONNX.

## Entry points and flow

```text
landmark.train / KneePose API       landmark.train_det
          |                                |
          +---------- cfg + data ----------+
                         |
                core trainer/validator
                         |
              nn model built from YAML
                         |
        checkpoint + metrics + reports + ONNX
```

- `train.py`: pose CLI and GPU visibility bootstrap.
- `train_det.py`: detection CLI, label audit, leakage-safe split, reports.
- `utils/api.py`: public `KneePose` lifecycle API.
- `core/model.py`, `trainer.py`, `validator.py`: runtime lifecycle.
- `core/pose.py`, `detect.py`: task-specific trainer/validator/predictor.
- `nn/tasks.py`: YAML-to-model construction and checkpoint loading.

## Ownership map

| Area | Primary files |
|---|---|
| Dataset schema/splits | `data/schema.py`, `data/prepare.py`, `train_det.py` |
| Loading/augmentation | `data/dataset.py`, `data/build.py`, `data/augment.py` |
| Losses/targets | `core/loss.py`, `core/targets.py`, `core/tal.py` |
| Model heads/backbones | `nn/modules/`, `models/` |
| Metrics/reports | `utils/validation.py`, `core/metrics.py`, `core/plotting*.py` |
| Export contract | `utils/exporting.py`, `core/exporter.py` |
| User configuration | `cfg/default.yaml`, `cfg/detect/`, `cfg/models/`, `cfg/datasets/` |

## Invariants

- `data/schema.py` is the only source of region names, counts, offsets, slot
  count, canonical landmark count and path boundaries. Never copy those numbers
  elsewhere; import them. Default (knee) values must stay `45/51/24/9`, 129
  landmarks and the six paths; structure losses must not connect separate paths.
- A non-default schema is read once at import from the dataset YAML named by
  `UKNEE_LANDMARK_SCHEMA` (`names`, `region_keypoint_counts`,
  `landmark_path_ranges`, `kpt_shape`). `landmark.train` sets it from
  `--dataset` before importing models; DDP workers inherit it. Exactly four
  regions are supported. `prepare_dataset` rejects a dataset whose schema
  differs from the active one.
- Model YAMLs belong to one anatomy branch and are checked against the active
  schema at construction (`num_landmarks`, `kpt_shape`, `nc`,
  `oa26_region_refine.auxiliary_heatmap_channels`): knee uses `rtmo-pose`,
  `yolo26-pose-v9`, `hrnet-*`, `vitpose-*`; hip uses `rtmo-pose-hip` and
  `yolo26-pose-hip`. Never let a model silently adopt another schema; add a
  `-hip` YAML instead.
- Result images (`samples/landmark_sample_e*.png`, dashboard, metrics) show
  progress as `Epoch <completed>/<total>`.
- Keep horizontal flips disabled when detection classes encode left/right.
- Keep non-square inputs letterboxed; mosaic/multi-scale require square input.
- Keep `last.pt` resumable and select pose `best.pt` by minimum MRE; detection
  selects best by mAP fitness.
- Keep export outputs fixed and validate ONNX Runtime parity after export edits.
- Do not consult `landmark0` as an implementation reference unless legacy
  parity or migration is explicitly in scope.

## Focused verification

```bash
python -m unittest landmark.tests.test_train_det -v
python -m unittest landmark.tests.test_data_and_adapter -v
python -m unittest landmark.tests.test_schema -v
python -m unittest landmark.tests.test_exporter -v
python -m unittest discover -s landmark/tests -v
```

