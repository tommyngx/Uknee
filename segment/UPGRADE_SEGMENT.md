# Uknee Segment — Component Architecture, API & Upgrade Specification

> **Goal**: Provide a clean, standalone, modular specification and upgrade roadmap for the **Uknee Medical Image Segment** package.

---

## 1. Public API & Command Line Interface (CLI)

### 1.1 Python API

```python
from segment.models import build_model
from segment.dataloader.dataloader import getDataloader
from segment.utils.segment_reporting import SegmentationEvaluator, plot_segmentation_metrics
from segment.utils.training_logs import EpochLogWriter, plot_training_dashboard

# Build segmentation model from registry (118+ architectures supported)
model = build_model(config=args, input_channel=3, num_classes=4)

# Create train & validation loaders
train_loader, val_loader = getDataloader(args)

# Evaluate predictions & generate visual dashboard
evaluator = SegmentationEvaluator(num_classes=4, class_names=["Background", "Femur", "Tibia", "Patella"])
evaluator.update(logits, targets)
snapshot = evaluator.snapshot()
plot_segmentation_metrics(snapshot, "runs/segmentation_metrics.png")
```

### 1.2 CLI Commands

```bash
# 2D Segmentation Training (e.g. CMUNeXt on MESKO dataset)
python -m segment.main \
  --model CMUNeXt \
  --base_dir ./data/mesko \
  --dataset_name mesko \
  --num_classes 4 \
  --batch_size 16 \
  --max_epochs 150 \
  --gpu 0

# Interactive Sample Visualization Tool
python -m segment.visualize \
  --model U_Net \
  --model_path ./output/U_Net/mesko/exp1/checkpoint_best.pth \
  --base_dir ./data/mesko \
  --dataset_name mesko \
  --save_path ./output/visualizations

# ONNX / TorchScript Model Exporter
python -m segment.tools.export_deploy_model \
  --checkpoint ./runs/segment/U_Net_mesko/best.pt \
  --data_dir ./data/mesko \
  --output_dir ./deploy/U_Net_mesko \
  --export_format auto

# Download U-Bench data or pretrained weights
python -m segment.utils.downloads dataset
python -m segment.utils.downloads weights
```

---

## 2. Directory Structure (Completed AS-IS State)

The segmentation framework has been isolated into its own dedicated package [segment/](file:///Users/francistommy/Desktop/BugHunter/Project/Uknee/segment) at the workspace root, standing alongside `landmark/`, `landmark2/`, `Ref/`, `info/`, and `tests/`:

```text
Uknee/                                    # Workspace Root
├── segment/                              # 📦 STANDALONE SEGMENTATION FRAMEWORK
│   ├── __init__.py                       # Package initialization & public exports
│   │
│   ├── models/                           # 🧠 118+ MODEL ARCHITECTURES (5 FAMILIES)
│   │   ├── __init__.py                   # Dynamic MODEL_REGISTRY & build_model()
│   │   ├── model_id.json                 # Model metadata & deep supervision IDs
│   │   ├── CNN/                          # 59 CNN models (SimpleUNet, AttU_Net, CMUNeXt, U_KAN,...)
│   │   ├── Hybrid/                       # 37 Hybrid models (TransUnet, SwinUNETR, TransFuse, EMCAD,...)
│   │   ├── Mamba/                        # 12 Mamba models (VMUNet, Swin_umamba, MambaUnet,...)
│   │   ├── RWKV/                         # 5 RWKV models (RWKV_UNet V1-V4, U_RWKV, Zig_RiR)
│   │   └── Transformer/                  # 5 Transformer models (SwinUnet, MedT, BATFormer,...)
│   │
│   ├── dataloader/                       # 📊 DATASETS & AUGMENTATION PIPELINE
│   │   ├── augment.py                    # Albumentations & torchvision transforms
│   │   ├── dataloader.py                 # getDataloader(), getZeroShotDataloader()
│   │   ├── dataset.py                    # General binary/multiclass dataset loader
│   │   ├── dataset_ACDC.py               # ACDC cardiac 3D dataset loader
│   │   ├── dataset_XRay.py               # Knee X-Ray dataset loader
│   │   ├── dataset_mesko.py              # MESKO 5-class segmentation loader
│   │   └── dataset_synapse.py            # Synapse multi-organ dataset loader
│   │
│   ├── utils/                            # 📐 MEDICAL METRICS, LOSSES & REPORTING
│   │   ├── binary_metrics.py             # Dice, IoU, Accuracy, Sensitivity, Specificity
│   │   ├── losses.py                     # DiceLoss, BCE, FocalLoss, Combined losses
│   │   ├── metrics.py                    # Standard metrics helper
│   │   ├── metrics_medpy.py              # MedPy 3D metrics (HD95, ASD, Dice)
│   │   ├── segment_reporting.py          # SegmentationEvaluator & evaluation grid plotting
│   │   ├── training_logs.py              # EpochLogWriter, CSV logging, Dashboard plotting
│   │   ├── downloads.py                  # Unified Hugging Face download CLI
│   │   ├── medsegbench.py                # MedSegBench datasets and download helpers
│   │   └── util.py                       # AverageMeter and volume evaluation helpers
│   │
│   ├── tools/                            # 🛠️ VERIFICATION & EXPORT TOOLS
│   │   ├── check_dataset.py              # Dataset integrity validator
│   │   ├── check_masks.py                # Segmentation mask boundary validator
│   │   └── export_deploy_model.py        # ONNX & TorchScript exporter tool
│   │
│   ├── deploy/                           # 🚀 WEB DEMO & INFERENCE API HANDLERS
│   │   ├── app.py                        # FastAPI / Gradio application backend
│   │   ├── app_function.py               # Inference preprocessing & postprocessing
│   │   └── test_function.py              # Deployment API integration tests
│   │
│   ├── main.py                           # 2D Segmentation Training Entrypoint
│   ├── visualize.py                      # Interactive Prediction Visualizer
│   └── UPGRADE_SEGMENT.md                # This specification document
│
├── landmark/                             # 📍 Legacy Pose Detection
├── landmark2/                            # 📍 Successor Pose Detection
├── Ref/                                  # 💾 Reference Datasets & Weights
├── info/                                 # 📄 Project License & Documentation
└── tests/                                # 🧪 Shared Root Test Suite
```

---

## 3. Model Registry Inventory (5 Architecture Families)

The `segment/models/` package organizes 118+ deep learning models across 5 distinct architectural paradigms:

| Family | Subdirectory | Models Count | Key Examples |
|---|---|---|---|
| **CNN** | `models/CNN` | 59 | `CMUNeXt`, `AttU_Net`, `SimpleUNet`, `UNetplus`, `ResU_KAN`, `CaraNet`, `Egeunet` |
| **Hybrid** | `models/Hybrid` | 37 | `TransUnet`, `SwinUNETR`, `TransFuse`, `EMCAD`, `HiFormer`, `MERIT`, `UCTransNet` |
| **Mamba** | `models/Mamba` | 12 | `VMUNet`, `VMUNetV2`, `Swin_umamba`, `MambaUnet`, `H_vmunet`, `UltraLight_VM_UNet` |
| **RWKV** | `models/RWKV` | 5 | `RWKV_UNet` (V1-V4), `U_RWKV`, `Zig_RiR` |
| **Transformer** | `models/Transformer` | 5 | `SwinUnet`, `MedT`, `BATFormer`, `Polyp_PVT`, `SCUNet_plus_plus` |

> [!NOTE]
> All models are instantiated through `build_model(config, input_channel, num_classes)` via string identifiers mapped in `segment/models/model_id.json`.

---

## 4. Completed Refactoring (AS-IS Status Summary)

| Aspect | Legacy State (Root Mix) | Current State (`segment/`) | Status |
|---|---|---|---|
| **Module Isolation** | Mixed directly in root | Isolated in `segment/` package | ✅ Complete |
| **Root Cleanliness** | 15+ root scripts & folders | Clean 6 top-level directories | ✅ Complete |
| **Import Integrity** | `from models import ...` | Package-qualified `segment.*` imports | ✅ Complete |
| **Download Utilities** | Three scattered implementations | Unified under `segment.utils` | ✅ Complete |
| **Root Tests** | Scattered reporting tests | Consolidated in `tests/` with package resolution | ✅ Complete |
| **Deployment App** | Depended on root paths | Self-resolving `REPO_ROOT` & `SEGMENTATION_ROOT` | ✅ Complete |

---

## 5. Upgrade Roadmap & Future Enhancements (TO-BE)

### 5.1 YAML Configuration System (`segment/cfg/default.yaml`)
Currently, `main.py` uses command-line flags. Future upgrade will support centralized YAML configurations:

```yaml
# segment/cfg/default.yaml
experiment:
  project: "runs/segment"
  name: "cmunext_mesko"
  seed: 2006
  device: "cuda:0"

model:
  name: "CMUNeXt"
  num_classes: 4
  input_channels: 3
  img_size: 256
  deep_supervision: false

dataset:
  name: "mesko"
  root: "./data/mesko"
  batch_size: 16
  num_workers: 4

training:
  epochs: 150
  lr: 0.01
  optimizer: "sgd"
  loss: "dice_ce"
```

### 5.2 Pythonic Naming Standardisation
Rename subdirectories in `segment/models/` to PEP 8 lower-case (`cnn/`, `hybrid/`, `mamba/`, `rwkv/`, `transformer/`) and update `MODEL_REGISTRY` accordingly.

### 5.3 Unified `core/` Package Organization
Group `losses.py`, `binary_metrics.py`, `metrics_medpy.py`, `segment_reporting.py`, and `training_logs.py` into a unified `segment/core/` subpackage to align 1:1 with `landmark2/core/`.

### 5.4 Unified Evaluation CLI (`segment/eval.py`)
Add a dedicated evaluation script for batch testing, zero-shot validation, and automated per-class Dice/HD95 table generation.

### 5.5 Joint Landmark + Segmentation Pipeline
Enable joint multi-task deployment where `landmark2` detects knee ROI landmarks to crop bounding boxes, and `segmentation` segments cartilage/bone masks on the cropped region.

---

## 6. Verification Suite

Run verification commands to confirm system integrity:

```bash
# Run root test suite
python tests/test_segment_reporting.py
python tests/test_landmark2_sample_plotting.py

# Run landmark2 test suite
python -m unittest discover -s landmark2/tests -v

# Run landmark test suite
python -m unittest discover -s landmark/tests -v

# Test visualization CLI help
python -m segment.visualize --help
python -m segment.utils.downloads --help
```
