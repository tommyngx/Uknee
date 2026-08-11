# Landmark Upgrade — Tích hợp Ultralytics vào sourcecode, xoá `_vendor/`

> **Mục tiêu**: Trích xuất code Ultralytics 8.4.87 cần thiết cho **Pose Detection** duy nhất, merge code trùng lặp, tối ưu cấu trúc gọn nhất có thể, xoá hoàn toàn `_vendor/`. Kết quả: **5 thư mục chính**, không bootstrap hack, không code thừa.

---

## 1. Tổng quan hiện trạng

### Vendor stats

| | Files | LoC | Size |
|---|---|---|---|
| `_vendor/ultralytics/` | 184 | ~55,800 | 5.6 MB |
| Landmark code riêng | 24 | ~2,650 | — |

### Landmark chỉ cần cho 1 task: **Pose Detection**

- YOLO26-Pose (v1, v9): train / val / predict / export
- HRNet-W32, ViTPose-S: heatmap baselines
- OA26 auxiliary: heatmap targets, structure-aware loss, region refinement
- Medical metrics: MRE, PCK, HD95, topology

---

## 2. Phân tích code trùng lặp & cần merge

### 2.1 Schema bị duplicate

`landmark/data/schema.py` và `_vendor/ultralytics/utils/oa26_region_refine/region_schema.py` **định nghĩa cùng** MESKO4GF2 schema:

| Constant | `data/schema.py` | `oa26_region_refine/region_schema.py` |
|---|---|---|
| `REGION_NAMES` | ✅ `("femur","tibia","fibula","patella")` | ✅ Giống hệt |
| `REGION_KEYPOINT_COUNTS` | ✅ `(45, 51, 24, 9)` | ✅ Giống hệt |
| `MAX_REGION_KEYPOINTS` | ✅ `51` | ✅ Giống hệt |
| `NUM_REGIONS` | ✅ `4` | ✅ Giống hệt |
| `class_keypoint_mask()` | ✅ Có | ✅ Có (gần giống) |
| `validate_region_schema()` | ✅ Có | ✅ Có (gần giống) |
| `class_path_masks()` | ❌ Không có | ✅ Có |
| `objects_to_canonical()` | ✅ Có | ❌ Không có |
| `LANDMARK_PATH_RANGES` | ✅ Có | ❌ Không có |
| `POINT_REGION_IDS` | ✅ Có | ❌ Không có |

**→ Merge**: Giữ `data/schema.py` làm **canonical source** duy nhất, bổ sung `class_path_masks()` vào. Xoá region_schema.py, tất cả import từ `data.schema`.

### 2.2 Config bị tách thừa

Hiện tại có 3 lớp config:
1. `_vendor/ultralytics/cfg/default.yaml` — 136 dòng, chứa mọi setting (classify, segment, OBB, tracker...)
2. `landmark/cfg/default.yaml` — 45 dòng, override cho pose only
3. `_vendor/ultralytics/cfg/__init__.py` — config parser (`get_cfg`, `get_save_dir`...)

Model YAMLs cũng tách 3 nơi:
1. `landmark/cfg/models/` — 5 YAML (yolo26-pose, hrnet, vitpose)
2. `_vendor/ultralytics/cfg/models/26/` — 10 YAML (phần lớn ko dùng: cls, obb, seg...)
3. `_vendor/ultralytics/cfg/models/26oa/` — 11 YAML (9 pose variants)

**→ Merge**:
- Tạo **1 file `default.yaml` duy nhất**: lọc bỏ setting classify/segment/obb/tracker, giữ detect+pose only
- Gom tất cả model YAML cần vào **1 thư mục flat** `cfg/models/`
- Config parser gộp vào `core/__init__.py`

### 2.3 Plotting bị tách 2 chỗ

| File | Nội dung |
|---|---|
| `landmark/utils/plotting.py` | `plot_dashboard_pose`, `plot_pose_metrics`, `plot_validation_samples` — 350 LoC |
| `_vendor/ultralytics/utils/plotting.py` | `Annotator`, `colors`, `save_one_box`, `plot_results` — ~1,100 LoC |

**→ Merge**: Gom hết vào `core/plotting.py`. Landmark plotting import trực tiếp từ cùng file.

### 2.4 Loss bị tách 3 chỗ

| File | Class |
|---|---|
| `_vendor/ultralytics/utils/loss.py` | `DetectionLoss`, `PoseLoss26`, `PoseRLELoss` |
| `_vendor/ultralytics/utils/oa26/loss.py` | `OA26HeatmapPoseLoss(PoseLoss26)`, `OA26SimCCPoseLoss(PoseLoss26)` |
| `_vendor/ultralytics/utils/oa26_region_refine/loss.py` | `OA26RegionRefinePoseLoss(OA26HeatmapPoseLoss)` |

Inheritance chain: `DetectionLoss → PoseLoss26 → OA26HeatmapPoseLoss → OA26RegionRefinePoseLoss`

**→ Merge**: Gom tất cả vào `core/loss.py` (1 file, 1 module). Xoá bỏ `SegmentationLoss`, `OBBLoss`, `ClassificationLoss`.

### 2.5 Heatmap targets bị tách 2 chỗ

| File | Hàm |
|---|---|
| `_vendor/ultralytics/utils/oa26/heatmap.py` | `extract_canonical_image_keypoints`, `gaussian_heatmap_targets` |
| `_vendor/ultralytics/utils/oa26/simcc.py` | `gaussian_simcc_targets`, `decode_simcc_logits` |

Đều là helper functions nhỏ (~110 + ~42 LoC). Dùng bởi: `heatmap/engine.py`, OA26 loss classes, tests.

**→ Merge**: Gom vào `core/targets.py` (1 file chứa tất cả target generation: heatmap + simcc).

### 2.6 Validation bị tách 2 chỗ

| File | Class |
|---|---|
| `_vendor/.../models/yolo/pose/val.py` | `PoseValidator` — base YOLO pose validation |
| `landmark/utils/validation.py` | `KneePoseValidator(PoseValidator)` — thêm MRE/PCK/HD95 |

**→ Giữ nguyên hierarchy**: PoseValidator ở `core/`, KneePoseValidator ở `utils/`. Nhưng `UkneePoseMetrics(PoseMetrics)` và `FlatPoseTrainerMixin` đều nằm trong cùng 1 file.

### 2.7 Heatmap engine riêng 1 thư mục — thừa

`landmark/heatmap/` chỉ có 2 files:
- `__init__.py` (6 LoC)
- `engine.py` (279 LoC) — adapter gắn HRNet/ViTPose vào ultralytics training loop

Đây thực chất là **model adapter**, không phải engine riêng.

**→ Merge**: Dời `engine.py` → `landmark/models/heatmap_adapter.py`. Xoá thư mục `heatmap/`.

---

## 3. Cấu trúc mới: 5 thư mục chính

```text
landmark/
│
├── cfg/                            # ① CONFIG — flat, đơn giản
│   ├── default.yaml                #    1 file defaults duy nhất (merged + cleaned)
│   └── models/                     #    Tất cả model YAML flat
│       ├── yolo26-pose.yaml        #    YOLO26 base pose
│       ├── yolo26-pose-v1.yaml     #    YOLO26 OA26 v1
│       ├── yolo26-pose-v9.yaml     #    YOLO26 OA26 v9 (region refine)
│       ├── yolo26-posev2.yaml      #    Từ vendor 26oa/ (nếu dùng)
│       ├── ...                     #    Các variant khác từ 26oa/
│       ├── yolo26-pose-base.yaml   #    Từ vendor 26/yolo26-pose.yaml (base architecture)
│       ├── hrnet-w32-pose.yaml     #    HRNet heatmap
│       └── vitpose-s-pose.yaml     #    ViTPose heatmap
│
├── core/                           # ② CORE — Mọi infrastructure gom 1 chỗ
│   ├── __init__.py                 #    LOGGER, RANK, SETTINGS, DEFAULT_CFG, get_cfg()
│   │                               #    ← merge utils/__init__.py + cfg/__init__.py
│   │
│   │  ── Engine ──
│   ├── model.py                    #    YOLO model lifecycle API
│   ├── trainer.py                  #    BaseTrainer
│   ├── validator.py                #    BaseValidator
│   ├── predictor.py                #    BasePredictor
│   ├── exporter.py                 #    Export (ONNX + TorchScript only)
│   ├── results.py                  #    Results container
│   │
│   │  ── Detect/Pose frontends (merged, không cần thư mục riêng) ──
│   ├── detect.py                   #    DetectionTrainer + Validator + Predictor
│   │                               #    ← merge models/yolo/detect/*.py vào 1 file
│   ├── pose.py                     #    PoseTrainer + PoseValidator + PosePredictor
│   │                               #    ← merge models/yolo/pose/*.py vào 1 file
│   │
│   │  ── Utilities ──
│   ├── ops.py                      #    NMS, scale_coords, xywh2xyxy
│   ├── loss.py                     #    ALL losses: Detection → Pose → OA26Heatmap → RegionRefine
│   │                               #    ← merge 3 files loss vào 1
│   ├── metrics.py                  #    DetMetrics, PoseMetrics, AP, ConfusionMatrix
│   │                               #    Bỏ: SegmentMetrics, OBBMetrics, ClassifyMetrics
│   ├── targets.py                  #    Heatmap + SimCC target generation
│   │                               #    ← merge oa26/heatmap.py + oa26/simcc.py
│   ├── tal.py                      #    Task-Aligned Label Assignment
│   ├── torch_utils.py              #    EMA, model_info, select_device, smart_optimizer
│   ├── plotting.py                 #    Annotator + dashboard_pose + pose_metrics plot
│   │                               #    ← merge 2 plotting files
│   ├── checks.py                   #    check_version, check_amp, check_imgsz
│   ├── downloads.py                #    Download utilities
│   ├── patches.py                  #    torch_load/save wrappers
│   ├── instance.py                 #    Instances, Bboxes classes
│   ├── files.py                    #    File utilities
│   ├── dist.py                     #    DDP utilities
│   ├── autobatch.py                #    Auto batch size
│   ├── autodevice.py               #    Auto device
│   ├── errors.py                   #    Error classes
│   ├── callbacks.py                #    Callback registry (1 file, không cần thư mục)
│   │                               #    ← chỉ giữ base callbacks
│   └── export_utils.py             #    ONNX + TorchScript helpers (1 file)
│                                   #    ← merge utils/export/onnx.py + torchscript.py + engine.py
│
├── data/                           # ③ DATA — Dataset pipeline
│   ├── __init__.py                 #    Re-export schema constants
│   ├── schema.py                   #    MESKO schema (CANONICAL source duy nhất)
│   │                               #    ← merge data/schema + oa26_region_refine/region_schema
│   │                               #    Thêm: class_path_masks()
│   ├── prepare.py                  #    Leakage-safe split + validation
│   ├── base.py                     #    ← BaseDataset (từ vendor)
│   ├── build.py                    #    ← build_yolo_dataset, InfiniteDataLoader (từ vendor)
│   ├── dataset.py                  #    ← YOLODataset (từ vendor)
│   ├── augment.py                  #    ← Mosaic, MixUp, LetterBox (từ vendor)
│   │                               #    Bỏ: ClassifyAugment
│   ├── loaders.py                  #    ← Image loaders (từ vendor)
│   └── data_utils.py              #    ← check_det_dataset, verify_image_label (từ vendor)
│
├── nn/                             # ④ NEURAL NETWORK — Model building blocks
│   ├── __init__.py                 #    Module registry
│   ├── tasks.py                    #    Model builder (parse YAML → model)
│   ├── autobackend.py              #    PyTorch + ONNX runtime only
│   │                               #    Bỏ: TensorRT, CoreML, OpenVINO, Paddle...
│   └── modules/
│       ├── __init__.py             #    ALL module exports
│       ├── activation.py
│       ├── block.py
│       ├── conv.py
│       ├── head.py                 #    Detect, Pose, OA26Heatmap, OA26RegionRefine heads
│       ├── transformer.py
│       ├── utils.py
│       ├── oa26/                   #    OA26 backbone modules
│       │   ├── __init__.py
│       │   ├── convnextv2_n.py
│       │   ├── convnextv2_t.py
│       │   ├── hrnet.py            #    OA26 HRNet backbone (khác với landmark/models/hrnet.py)
│       │   ├── pose_heads.py
│       │   └── vit_refine.py
│       └── oa26_region_refine/     #    Region refinement modules
│           ├── __init__.py
│           ├── landmark_query_encoder.py
│           ├── localization_head.py
│           ├── pose_head.py
│           ├── refinement_head.py
│           ├── region_transformer.py
│           └── roi_feature_extractor.py
│
├── models/                         # ⑤ MODEL ARCHITECTURES
│   ├── __init__.py                 #    Export: HRNetW32, ViTPoseS, HeatmapPose
│   ├── hrnet.py                    #    HRNet-W32 (pure PyTorch, không import ultralytics)
│   ├── vitpose.py                  #    ViTPose-S (pure PyTorch, không import ultralytics)
│   └── heatmap_adapter.py          #    HeatmapPoseModel + HeatmapPose + Trainer/Validator/Predictor
│                                   #    ← từ heatmap/engine.py, gom adapter vào models
│
├── utils/                          # Landmark-specific utilities
│   ├── __init__.py
│   ├── api.py                      #    KneePose public API (train/val/predict/export)
│   ├── validation.py               #    KneePoseValidator, UkneePoseMetrics, FlatPoseTrainerMixin
│   ├── results.py                  #    KneePoseResult, adapt_yolo_result
│   └── exporting.py                #    KneePoseExportWrapper
│
├── tests/                          #    Tests (import paths updated)
├── __init__.py                     #    Clean: just `from .utils.api import KneePose`
├── train.py                        #    CLI entry point
├── requirements.txt
├── README.md
├── PAPER_EXPERIMENTS.md
└── UPGRADE.md
```

**So sánh phiên bản cũ vs mới**:

| | Plan cũ (v1) | Plan mới (v2) |
|---|---|---|
| Thư mục chính | 10 (`core`, `engine`, `nn`, `yolo`, `optim`, `cfg`, `data`, `heatmap`, `models`, `utils`) | **5** (`cfg`, `core`, `data`, `nn`, `models`) + `utils` (landmark-specific) |
| Config files | 2 YAML + parser riêng | **1 YAML merged** + parser trong core |
| Loss files | 3 files tách biệt | **1 file merged** |
| Plotting files | 2 files tách biệt | **1 file merged** |
| Schema files | 2 files duplicate | **1 file canonical** |
| Heatmap targets | 2 files (heatmap + simcc) | **1 file merged** (`targets.py`) |
| YOLO detect/pose | 2 thư mục con (6 files) | **2 files** (`detect.py`, `pose.py`) |
| Heatmap engine | Thư mục riêng | Gộp vào `models/` |
| Callbacks | Thư mục con | **1 file** |
| Export utils | Thư mục con (3 files) | **1 file** |
| Optim | Thư mục riêng | Gộp vào `core/torch_utils.py` |

---

## 4. Chi tiết merge từng phần

### 4.1 `cfg/default.yaml` — Merge 2 configs thành 1

Lấy base từ `_vendor/ultralytics/cfg/default.yaml`, **loại bỏ** settings không dùng:

```yaml
# XOÁ các dòng sau:
overlap_mask: True    # segment only
mask_ratio: 4         # segment only
dropout: 0.0          # classify only
angle: 1.0            # obb only
copy_paste: 0.0       # segment only
copy_paste_mode: flip # segment only
auto_augment: ...     # classify only
erasing: 0.4          # classify only
tracker: ...          # tracking only
retina_masks: False   # segment only
```

Sau đó **override** bằng giá trị từ `landmark/cfg/default.yaml`:

```yaml
# OVERRIDE cho Uknee Pose
task: pose
seed: 2006            # thay vì 0
plots: false           # landmark tự render
save_period: -1
```

Kết quả: **1 file ~80 dòng** thay vì 136 + 45.

### 4.2 `cfg/models/` — Gom flat

```text
cfg/models/
├── yolo26-pose.yaml          # ← giữ từ landmark/cfg/models/
├── yolo26-pose-v1.yaml       # ← giữ
├── yolo26-pose-v9.yaml       # ← giữ
├── yolo26-pose-base.yaml     # ← từ vendor cfg/models/26/yolo26-pose.yaml (base architecture, chứa layer defs)
├── yolo26-posev2.yaml        # ← từ vendor cfg/models/26oa/ (nếu dùng)
├── yolo26-posev3.yaml        # ...
├── yolo26-posev4.yaml
├── yolo26-posev5.yaml
├── yolo26-posev6.yaml
├── yolo26-posev7.yaml
├── yolo26-posev8.yaml
├── hrnet-w32-pose.yaml       # ← giữ
└── vitpose-s-pose.yaml       # ← giữ
```

**Xoá**: `yolo26-cls.yaml`, `yolo26-obb.yaml`, `yolo26-seg.yaml`, `yolo26-sem.yaml`, `yoloe-26.yaml`, `yoloe-26-seg.yaml`, `yolo26-p2.yaml`, `yolo26-p6.yaml`, toàn bộ `cfg/models/v3,v5,v6,v8,v9,v10,v11,12,rt-detr/`.

### 4.3 `data/schema.py` — Merge canonical schema

Bổ sung vào file hiện tại:

```python
# Thêm từ oa26_region_refine/region_schema.py:
SOURCE_KEYPOINT_RANGES = ((0, 44), (45, 95), (96, 119), (120, 128))

def class_path_masks(class_ids: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Return valid adjacent-pair or curvature-triplet starts per region row."""
    # ... code từ region_schema.py
```

### 4.4 `core/loss.py` — Merge 3 files loss

```python
# File: core/loss.py
# Gom theo thứ tự inheritance:

class DetectionLoss: ...          # ← từ utils/loss.py (bỏ SegmentLoss, OBBLoss)
class PoseLoss26(DetectionLoss): ... # ← từ utils/loss.py
class OA26HeatmapPoseLoss(PoseLoss26): ... # ← từ utils/oa26/loss.py
class OA26SimCCPoseLoss(PoseLoss26): ...   # ← từ utils/oa26/loss.py
class OA26RegionRefinePoseLoss(OA26HeatmapPoseLoss): ... # ← từ utils/oa26_region_refine/loss.py
```

### 4.5 `core/targets.py` — Merge heatmap + simcc

```python
# File: core/targets.py (~150 LoC)

def extract_image_keypoints(...): ...           # ← oa26/heatmap.py
def extract_canonical_image_keypoints(...): ...  # ← oa26/heatmap.py
def gaussian_heatmap_targets(...): ...           # ← oa26/heatmap.py
def gaussian_simcc_targets(...): ...             # ← oa26/simcc.py
def decode_simcc_logits(...): ...                # ← oa26/simcc.py
```

### 4.6 `core/detect.py` + `core/pose.py` — Flatten detect/pose

Thay vì tạo thư mục `yolo/detect/` (3 files) và `yolo/pose/` (3 files):

```python
# core/detect.py — 1 file chứa cả 3 class
class DetectionTrainer(BaseTrainer): ...
class DetectionValidator(BaseValidator): ...
class DetectionPredictor(BasePredictor): ...
```

```python
# core/pose.py — 1 file chứa cả 3 class
class PoseTrainer(DetectionTrainer): ...
class PoseValidator(DetectionValidator): ...
class PosePredictor(DetectionPredictor): ...
```

### 4.7 `core/plotting.py` — Merge 2 plotting

```python
# File: core/plotting.py

# === Từ vendor ultralytics/utils/plotting.py (cần cho engine) ===
class Annotator: ...
def colors(...): ...
def save_one_box(...): ...
def plot_results(...): ...

# === Từ landmark/utils/plotting.py (medical dashboard) ===
def plot_dashboard_pose(...): ...
def plot_pose_metrics(...): ...
def plot_validation_samples(...): ...
```

**Lưu ý**: `landmark/utils/results.py` hiện import từ `landmark.utils.plotting` — sau merge, đổi thành import từ `landmark.core.plotting`.

### 4.8 `core/callbacks.py` — 1 file thay thư mục

Chỉ cần **callback registry** (default hooks cho on_train_start, on_batch_end, etc.):

```python
# core/callbacks.py (~200 LoC)
# ← từ utils/callbacks/base.py
# Bỏ: ClearML, Comet, DVC, MLflow, Neptune, WandB, Ray, Platform, TensorBoard
default_callbacks = { ... }
def add_integration_callbacks(instance): ...
```

### 4.9 `core/export_utils.py` — 1 file thay thư mục

```python
# core/export_utils.py
# ← merge utils/export/onnx.py + torchscript.py + engine.py
# Chỉ giữ ONNX + TorchScript, bỏ 14 format khác
```

### 4.10 `core/torch_utils.py` — Merge optim vào

```python
# core/torch_utils.py
# ← từ utils/torch_utils.py: EMA, model_info, select_device, smart_optimizer...
# ← THÊM MuSGD từ optim/muon.py (chỉ ~300 LoC, không cần thư mục riêng)
class MuSGD(Optimizer): ...
```

### 4.11 `models/heatmap_adapter.py` — Heatmap vào models

```python
# models/heatmap_adapter.py
# ← nguyên heatmap/engine.py (279 LoC)
# Import đổi: ultralytics.* → landmark.core.*
# Class: HeatmapPoseModel, HeatmapPose, HeatmapPoseTrainer, HeatmapPoseValidator, HeatmapPosePredictor
```

Xoá thư mục `landmark/heatmap/` sau khi dời.

---

## 5. Import mapping tổng hợp

| Import cũ (`ultralytics.*`) | Import mới (`landmark.*`) |
|---|---|
| `from ultralytics import YOLO` | `from landmark.core.model import YOLO` |
| `from ultralytics.engine.model import Model` | `from landmark.core.model import Model` |
| `from ultralytics.engine.trainer import BaseTrainer` | `from landmark.core.trainer import BaseTrainer` |
| `from ultralytics.engine.validator import BaseValidator` | `from landmark.core.validator import BaseValidator` |
| `from ultralytics.engine.predictor import BasePredictor` | `from landmark.core.predictor import BasePredictor` |
| `from ultralytics.engine.results import Results` | `from landmark.core.results import Results` |
| `from ultralytics.engine.exporter import Exporter` | `from landmark.core.exporter import Exporter` |
| `from ultralytics.models.yolo.detect import DetectionTrainer` | `from landmark.core.detect import DetectionTrainer` |
| `from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator` | `from landmark.core.pose import PoseTrainer, PoseValidator` |
| `from ultralytics.nn.tasks import ...` | `from landmark.nn.tasks import ...` |
| `from ultralytics.cfg import get_cfg` | `from landmark.core import get_cfg` |
| `from ultralytics.utils import LOGGER, RANK, ops, DEFAULT_CFG` | `from landmark.core import LOGGER, RANK, DEFAULT_CFG` + `from landmark.core.ops import ...` |
| `from ultralytics.utils.loss import PoseLoss26` | `from landmark.core.loss import PoseLoss26` |
| `from ultralytics.utils.metrics import PoseMetrics` | `from landmark.core.metrics import PoseMetrics` |
| `from ultralytics.utils.torch_utils import model_info` | `from landmark.core.torch_utils import model_info` |
| `from ultralytics.utils.oa26.heatmap import ...` | `from landmark.core.targets import ...` |
| `from ultralytics.utils.oa26.loss import ...` | `from landmark.core.loss import ...` |
| `from ultralytics.utils.oa26.simcc import ...` | `from landmark.core.targets import ...` |
| `from ultralytics.utils.oa26_region_refine.region_schema import ...` | `from landmark.data.schema import ...` |
| `from ultralytics.utils.oa26_region_refine.loss import ...` | `from landmark.core.loss import ...` |
| `from ultralytics.data import build_yolo_dataset` | `from landmark.data.build import build_yolo_dataset` |
| `from ultralytics.data.augment import LetterBox` | `from landmark.data.augment import LetterBox` |
| `from ultralytics.data.utils import check_det_dataset` | `from landmark.data.data_utils import check_det_dataset` |
| `from ultralytics.optim import MuSGD` | `from landmark.core.torch_utils import MuSGD` |
| `from ultralytics.utils.plotting import Annotator` | `from landmark.core.plotting import Annotator` |

---

## 6. Files XOÁ hoàn toàn

### Xoá toàn bộ thư mục

| Thư mục | Lý do |
|---|---|
| `models/fastsam/` | FastSAM — không liên quan |
| `models/nas/` | NAS — không dùng |
| `models/rtdetr/` | RT-DETR — không dùng |
| `models/yolo/classify/` | Classification |
| `models/yolo/segment/` | Instance segmentation |
| `models/yolo/obb/` | Oriented bounding box |
| `models/yolo/semantic/` | Semantic segmentation |
| `models/yolo/world/` | YOLO-World open-vocab |
| `models/yolo/yoloe/` | YOLOE efficient |
| `cfg/datasets/` (42 files) | Không dùng dataset config nào |
| `cfg/trackers/` (6 files) | Tracking — không dùng |
| `cfg/models/v3,v5,v6,v8,v9,v10,v11,v12,rt-detr/` | Cũ, không dùng |
| `nn/backends/` (16/18 files) | Chỉ giữ pytorch.py, onnx.py |
| `data/scripts/` | Download scripts |
| `utils/callbacks/` (9/11 files) | Chỉ giữ base.py |
| `utils/export/` (14/17 files) | Chỉ giữ onnx + torchscript + engine |

### Xoá từng file

| File | Lý do |
|---|---|
| `nn/text_model.py` | CLIP/text — không dùng |
| `nn/distill_model.py` | Distillation — cân nhắc |
| `engine/tuner.py` | Hyperparameter tuner |
| `data/converter.py` | Format converter |
| `data/split.py`, `data/split_dota.py` | Splitter |
| `utils/benchmarks.py` | Benchmarking |
| `utils/events.py` | Telemetry |
| `utils/uploads.py` | HUB upload |
| `utils/git.py` | Git integration |
| `utils/triton.py` | Triton server |
| `utils/tuner.py` | Tuner utils |
| `utils/logger.py` | Remote logger |
| `utils/tqdm.py` | Custom tqdm |
| `utils/cpu.py` | CPU optimization (nhỏ, optional) |
| `_vendor/bootstrap.py` | Xoá cuối cùng |

### Xoá thư mục landmark cũ

| Thư mục/File | Lý do |
|---|---|
| `landmark/heatmap/` | Đã dời vào `models/heatmap_adapter.py` |
| `landmark/utils/plotting.py` | Đã merge vào `core/plotting.py` |
| `landmark/_vendor/` | Mục tiêu chính — xoá hoàn toàn |

### Ước tính giảm

| | Trước | Sau | Giảm |
|---|---|---|---|
| Python files | 208 (184 vendor + 24 landmark) | ~55 | **~74%** |
| LoC | ~58,400 | ~22,000 | **~62%** |
| Thư mục chính | 10+ | **5** | **50%** |

---

## 7. Thứ tự thực hiện

### Phase 1: Foundation — `core/` + `data/schema.py` merge

| Step | Việc | Chi tiết |
|---|---|---|
| 1.1 | Merge `data/schema.py` | Thêm `class_path_masks()`, `SOURCE_KEYPOINT_RANGES` từ region_schema |
| 1.2 | Tạo `core/__init__.py` | Extract LOGGER, RANK, SETTINGS, DEFAULT_CFG + get_cfg() |
| 1.3 | Tạo `core/ops.py` | Copy từ vendor, bỏ functions không dùng |
| 1.4 | Tạo `core/torch_utils.py` | Copy + merge MuSGD optimizer vào |
| 1.5 | Tạo `core/patches.py` | Copy torch_load/save |
| 1.6 | Tạo `core/errors.py` | Copy error classes |
| 1.7 | Tạo `core/callbacks.py` | Chỉ base callback registry |
| 1.8 | Tạo `core/checks.py` | check_version, check_amp, check_imgsz |
| 1.9 | Tạo `core/files.py`, `dist.py`, `autobatch.py`, `autodevice.py`, `downloads.py`, `instance.py` | Copy trực tiếp |
| 1.10 | Test imports | `python -c "from landmark.core import LOGGER, RANK"` |

### Phase 2: Training pipeline — `core/` engine + detect + pose

| Step | Việc |
|---|---|
| 2.1 | Tạo `core/results.py` ← vendor engine/results.py |
| 2.2 | Tạo `core/model.py` ← vendor engine/model.py |
| 2.3 | Tạo `core/trainer.py` ← vendor engine/trainer.py |
| 2.4 | Tạo `core/validator.py` ← vendor engine/validator.py |
| 2.5 | Tạo `core/predictor.py` ← vendor engine/predictor.py |
| 2.6 | Tạo `core/detect.py` ← merge yolo/detect/*.py vào 1 file |
| 2.7 | Tạo `core/pose.py` ← merge yolo/pose/*.py vào 1 file |
| 2.8 | Tạo `core/exporter.py` + `core/export_utils.py` |

### Phase 3: Loss, metrics, targets, plotting merge

| Step | Việc |
|---|---|
| 3.1 | Tạo `core/tal.py` ← copy |
| 3.2 | Tạo `core/metrics.py` ← copy, bỏ Segment/OBB/Classify metrics |
| 3.3 | Tạo `core/loss.py` ← merge 3 files (DetLoss → PoseLoss → OA26 → RegionRefine) |
| 3.4 | Tạo `core/targets.py` ← merge heatmap + simcc |
| 3.5 | Tạo `core/plotting.py` ← merge vendor plotting + landmark plotting |

### Phase 4: Data + NN

| Step | Việc |
|---|---|
| 4.1 | Bổ sung `data/base.py`, `build.py`, `dataset.py`, `augment.py`, `loaders.py`, `data_utils.py` |
| 4.2 | Copy `nn/tasks.py`, `nn/autobackend.py`, `nn/modules/` |
| 4.3 | Merge `cfg/default.yaml` thành 1 file |
| 4.4 | Gom model YAMLs vào `cfg/models/` |

### Phase 5: Models adapter + landmark code update + cleanup

| Step | Việc |
|---|---|
| 5.1 | Dời `heatmap/engine.py` → `models/heatmap_adapter.py` |
| 5.2 | Update imports trong `utils/api.py`, `utils/validation.py`, `utils/results.py`, `utils/exporting.py` |
| 5.3 | Xoá `landmark/utils/plotting.py` (đã merge) |
| 5.4 | Update `landmark/__init__.py` — xoá bootstrap |
| 5.5 | Update tests |
| 5.6 | **Xoá `_vendor/`** |
| 5.7 | Xoá `landmark/heatmap/` |
| 5.8 | Test toàn bộ |

---

## 8. Rủi ro & lưu ý

### Checkpoint compatibility

Checkpoint `.pt` chứa module paths `ultralytics.*`. Cần:
- `torch.load` với custom `map_location` + `pickle_module` rename
- Hoặc viết migration script đổi key paths

### `nn/tasks.py` dynamic imports

File này dùng `getattr` để resolve layer classes từ `nn.modules.*`. Module registry trong `nn/modules/__init__.py` phải map đầy đủ tất cả class names.

### `utils/__init__.py` initialization order

File `_vendor/ultralytics/utils/__init__.py` (~58KB) chứa global state phức tạp. Khi extract vào `core/__init__.py`, cần giữ đúng thứ tự khởi tạo: SETTINGS → LOGGER → RANK → DEFAULT_CFG.

### Test strategy

```bash
# Sau mỗi phase:
python -m pytest landmark/tests/ -v

# Smoke test:
python landmark/train.py --model cfg/models/yolo26-pose-v1.yaml --data <dataset>.yaml --epochs 2 --batch 4 --device cpu
python landmark/train.py --model cfg/models/hrnet-w32-pose.yaml --data <dataset>.yaml --epochs 2 --batch 4 --device cpu

# Export test:
python -c "from landmark import KneePose; m = KneePose('best.pt'); m.export(format='onnx')"
```

---

## Tóm tắt

| Metric | Trước | Sau |
|---|---|---|
| Vendor files | 184 | **0** |
| Thư mục chính | ~10+ | **5** |
| Duplicate schema | 2 files | **1 canonical** |
| Config files | 136 + 45 dòng | **~80 dòng** |
| Loss files | 3 | **1 merged** |
| Plotting files | 2 | **1 merged** |
| Separate heatmap dir | ✅ | ❌ (vào `models/`) |
| Separate yolo dir | ✅ (plan v1) | ❌ (vào `core/`) |
| Separate optim dir | ✅ (plan v1) | ❌ (vào `core/torch_utils`) |
| Bootstrap hack | ✅ | ❌ |
| External ultralytics dep | sys.path hack | **Không có** |
