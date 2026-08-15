# Segment architecture

## Contract

`segment` is the active standalone medical-image segmentation runtime. It
supports binary and multiclass datasets through a lazy model registry, shared
training/reporting, and PyTorch/ONNX deployment.

- CLI and preprocessing dimensions use `[height, width]`.
- Images and masks are letterboxed; masks use nearest-neighbour interpolation.
- `best.pt` is inference-only; `last.pt` retains optimizer state for resume.
- Model imports stay lazy so unused optional model dependencies are not loaded.

## Entry points and flow

```text
segment.cli -> segment.main
      |            |
 cfg/default   model registry (lazy import)
                   |
          dataloader + augmentation
                   |
       train/validate + reporting
                   |
      checkpoints + reports + ONNX
```

- `cli.py`: YAML/CLI normalization and legacy argument aliases.
- `main.py`: train, validate, resume, checkpoint, report, and export orchestration.
- `models/__init__.py`: `MODEL_REGISTRY`, lazy imports, model construction.
- `dataloader/dataloader.py`: dataset selection and loader construction.
- `dataloader/dataset_mesko.py`: MESKO multiclass image/mask pairing.
- `deploy/app_function.py`: reusable inference and preprocessing API.

## Ownership map

| Area | Primary files |
|---|---|
| Configuration/CLI | `cfg/default.yaml`, `cli.py` |
| Training/checkpoints | `main.py` |
| Model selection | `models/__init__.py`, `models/model_id.json` |
| A specific architecture | Only its named folder under `models/` |
| Dataset routing | `dataloader/dataloader.py`, `dataloader/dataset*.py` |
| Augmentation/preprocess | `dataloader/augment.py`, `utils/preprocessing.py` |
| Metrics/reports | `utils/binary_metrics.py`, `utils/segment_reporting.py`, `utils/training_logs.py` |
| ONNX/deployment | `utils/onnx_export.py`, `deploy/app_function.py`, `tools/export_deploy_model.py` |

## Invariants

- Query `MODEL_REGISTRY` by exact model name; never scan or import the complete
  `models/` tree for a single-model task.
- Preserve lazy imports and give missing optional dependencies model-specific
  errors.
- Keep mask class IDs discrete through augmentation and resizing.
- Validate configured/inferred class counts before training multiclass MESKO.
- Keep checkpoint architecture checks strict; do not partially load a mismatched
  model silently.
- Validate PyTorch/ONNX numerical parity and embedded preprocessing metadata
  after export changes.

## Focused verification

```bash
python -m unittest tests.code.test_cli_contract -v
python -m unittest tests.code.test_segment_reporting -v
python -m unittest tests.code.test_segment_onnx_and_preprocessing -v
python -m unittest discover -s tests/code -p 'test_segment*.py' -v
```

