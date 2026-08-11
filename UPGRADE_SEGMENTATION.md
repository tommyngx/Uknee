# Uknee Segmentation — Upgrade & Refactoring Plan

> **Mục tiêu**: Đánh giá hiện trạng phần **Segmentation** ngoài `landmark/`, xây dựng kế hoạch sắp xếp lại thư mục gọn gàng, tách biệt 100% với bài toán **Landmark (Pose Detection)**, đồng thời bổ sung các định hướng nâng cấp chuyên nghiệp cho pipeline Segmentation.

---

## 1. Đánh giá hiện trạng cập nhật (AS-IS Check)

### 1.1 Phân tích tổng quan cấu trúc ngoài `landmark/`

Phần Segmentation của Uknee hiện tại bao gồm các thành phần:

```text
Uknee/ (Root)
├── models/                           # 100+ Models phân loại thành 5 họ kiến trúc
│   ├── CNN/                          # 59 models (SimpleUNet, AttU_Net, EGEUNet, U_KAN,...)
│   ├── Hybrid/                       # 37 models (TransUnet, SwinUNETR, TransFuse, EMCAD,...)
│   ├── Mamba/                        # 12 models (VMUNet, Swin_umamba, MambaUnet,...)
│   ├── RWKV/                         # RWKV_UNet (V1, V2, V3, V4), U_RWKV, Zig_RiR
│   ├── Transformer/                  # 5 models (SwinUnet, MedT, BATFormer, Polyp_PVT,...)
│   ├── __init__.py                   # Dynamic MODEL_REGISTRY + build_model()
│   └── model_id.json                 # Metadata ID & Deep Supervision config
│
├── dataloader/                       # Dataset & Augmentation Pipeline
│   ├── augment.py                    # Albumentations / torchvision transforms
│   ├── dataloader.py                 # getDataloader(), getZeroShotDataloader()
│   ├── dataset.py                    # General Segmentation dataset
│   ├── dataset_ACDC.py               # ACDC cardiac dataset
│   ├── dataset_XRay.py               # Knee X-Ray segmentation dataset
│   ├── dataset_mesko.py              # MESKO segmentation dataset
│   ├── dataset_synapse.py           # Synapse multi-organ dataset
│   └── download.py                   # Auto-download helpers
│
├── utils/                            # Metrics, Losses & Reporting
│   ├── losses.py                     # DiceLoss, BCE, Focal, Combo losses
│   ├── metrics.py / binary_metrics.py# Dice, IoU, Accuracy, Sensitivity, Specificity
│   ├── metrics_medpy.py              # MedPy metrics (HD95, ASD, Dice)
│   ├── segmentation_reporting.py     # SegmentationEvaluator & evaluation plotting
│   ├── training_logs.py              # EpochLogWriter, CSV logging, Dashboard plotting
│   └── util.py                       # AverageMeter, seed, checkpoint helpers
│
├── tools/                            # Verification & Export tools
│   ├── check_dataset.py              # Kiểm tra tính hợp lệ của dataset
│   ├── check_masks.py                # Kiểm tra nhãn mask
│   └── export_deploy_model.py        # ONNX / TorchScript exporter
│
├── deploy/                           # Gradio Web Demo Application
│   ├── app.py                        # Gradio UI entry point
│   ├── app_function.py               # Inference backend logic
│   └── test_function.py              # Deploy test suite
│
├── tests/                            # Unit tests cho reporting
│   └── test_segmentation_reporting.py
│
├── imgs/                             # Hình ảnh báo cáo / README visualizations
├── main.py                           # 2D Segmentation Training Entrypoint
├── main_multi3d.py                   # 3D / Multi-slice Segmentation Entrypoint
├── visualize.py                      # Interactive visualization tool
├── download_datasets_from_huggingface.py
└── download_weights_from_huggingface.py
```

### 1.2 Nhận xét & Đánh giá (Check ổn không?)

✅ **Điểm RẤT ỔN & Chuyên nghiệp hiện tại**:
1. **Model Registry ấn tượng (`models/`)**: Phân chia 5 họ mô hình (CNN, Hybrid, Mamba, RWKV, Transformer) cực kỳ khoa học. `build_model()` kết hợp `model_id.json` cho phép gọi hơn 100+ mô hình linh hoạt thông qua string ID.
2. **Dataloader đa dạng (`dataloader/`)**: Hỗ trợ nhiều bộ dữ liệu y tế thực tế (MESKO, XRay, ACDC, Synapse) và hỗ trợ chế độ Zero-Shot evaluation.
3. **Medical Metrics chuẩn mực (`utils/`)**: Đã có MedPy (HD95, ASD), Dice/IoU per-class, kết hợp với `segmentation_reporting.py` và `training_logs.py` tự động xuất dashboard kết quả đẹp mắt.
4. **Đã có sẵn Tooling & Deployment (`tools/`, `deploy/`)**: Có sẵn web app Gradio (`deploy/app.py`) và công cụ export model (`export_deploy_model.py`).

⚠️ **Điểm CẦN CẢI THIỆN (Vấn đề duy nhất)**:
- Phần Segmentation hiện vẫn đang nằm **trực tiếp ở thư mục Root (`Uknee/`)**.
- Điều này khiến Root bị trộn lẫn giữa:
  1. Code Segmentation (`models/`, `dataloader/`, `main.py`, `main_multi3d.py`)
  2. Code Landmark Pose (`landmark/`)
  3. Các file cấu hình/doc dùng chung (`README.md`, `requirements.txt`)
- Sự khác biệt bản chất: **Landmark** là bài toán Pose Keypoint Detection (output là các điểm toạ độ 129 landmarks / bounding box), còn **Segmentation** là Pixel-level Mask Prediction (output là 2D/3D segmentation mask). Do đó 2 bài toán hoàn toàn độc lập và nên được gom thư mục tách biệt.

---

## 2. Kế hoạch Tái cấu trúc & Sắp xếp gọn gàng (TO-BE)

### 2.1 Cấu trúc thư mục Segmentation đề xuất (`segmentation/`)

Gom toàn bộ thành phần của Segmentation vào thư mục `segmentation/` đứng **ngang hàng** với `landmark/`:

```text
Uknee/                                # Root dự án
├── segmentation/                     # 📦 TOÀN BỘ CODE SEGMENTATION GOM VÀO ĐÂY
│   ├── cfg/                          # File cấu hình YAML cho train/eval (MỚI)
│   │   ├── default.yaml              # Hyperparameters mặc định cho Segmentation
│   │   └── models/                   # Configs riêng cho các họ model
│   │
│   ├── dataloader/                   # Dataset & Data Augmentation
│   │   ├── __init__.py
│   │   ├── dataset.py                # Dataset base & MESKO / XRay loaders
│   │   ├── dataset_acdc.py
│   │   ├── dataset_synapse.py
│   │   ├── augment.py
│   │   └── loader.py                 # get_dataloader(), get_zeroshot_dataloader()
│   │
│   ├── models/                       # 100+ Models Registry
│   │   ├── __init__.py               # build_model(), MODEL_REGISTRY
│   │   ├── model_id.json
│   │   ├── cnn/                      # (Đổi tên folder viết hoa -> viết thường cho chuẩn python)
│   │   ├── hybrid/
│   │   ├── mamba/
│   │   ├── rwkv/
│   │   └── transformer/
│   │
│   ├── core/                         # Losses, Metrics & Logging Engine
│   │   ├── __init__.py
│   │   ├── losses.py                 # DiceLoss, BCE, Focal, Combined Losses
│   │   ├── metrics.py                # Dice, IoU, MedPy (HD95, ASD)
│   │   ├── reporting.py              # Evaluation reporting & per-class tables
│   │   ├── logger.py                 # EpochLogWriter & dashboard plotting
│   │   └── utils.py                  # AverageMeter, seed, checkpoint helpers
│   │
│   ├── tools/                        # Tools kiểm tra & export
│   │   ├── check_dataset.py
│   │   ├── check_masks.py
│   │   └── export_model.py
│   │
│   ├── deploy/                       # Web Application / Demo
│   │   ├── app.py
│   │   └── app_function.py
│   │
│   ├── train.py                      # (Đổi tên từ main.py -> train.py cho rõ nghĩa)
│   ├── train_multi3d.py              # (Đổi tên từ main_multi3d.py)
│   ├── eval.py                       # Evaluation script chuyên biệt
│   └── visualize.py                  # Interactive visualizer
│
├── landmark/                         # 📍 TOÀN BỘ CODE LANDMARK POSE DETECTION
│   ├── cfg/
│   ├── core/
│   ├── data/
│   ├── engine/
│   ├── models/
│   ├── utils/
│   └── train.py
│
├── shared/                           # 🤝 COMPONENT DÙNG CHUNG (Tùy chọn)
│   ├── download_datasets.py          # Download data HuggingFace
│   └── download_weights.py           # Download pretrained weights
│
├── imgs/                             # Visualizations cho Root README
├── README.md                         # Project overview
├── requirements.txt                  # Python dependencies
└── UPGRADE_SEGMENTATION.md           # File này
```

---

## 3. Các bước thực hiện Refactoring (Step-by-step)

### Bước 1: Tạo thư mục `segmentation/` & Di chuyển thành phần
- Tạo thư mục `segmentation/`.
- Move các thư mục `models/`, `dataloader/`, `utils/`, `tools/`, `deploy/` ở root vào trong `segmentation/`.
- Move các entry script: `main.py` -> `segmentation/train.py`, `main_multi3d.py` -> `segmentation/train_multi3d.py`, `visualize.py` -> `segmentation/visualize.py`.

### Bước 2: Cập nhật đường dẫn Import (Import Paths)
- Trong `segmentation/train.py` và các module con:
  - Thay `from models import build_model` -> `from segmentation.models import build_model` (hoặc dùng relative import `from .models import build_model`).
  - Thay `import utils.losses as losses` -> `from segmentation.core.losses import ...`.
  - Thay `from dataloader.dataloader import getDataloader` -> `from segmentation.dataloader import get_dataloader`.

### Bước 3: Đổi tên thư mục con theo chuẩn Pythonic (PEP 8)
- Đổi tên các subfolder trong `models/`:
  - `models/CNN` -> `models/cnn`
  - `models/Hybrid` -> `models/hybrid`
  - `models/Mamba` -> `models/mamba`
  - `models/RWKV` -> `models/rwkv`
  - `models/Transformer` -> `models/transformer`
- Cập nhật tương ứng chuỗi module path trong `MODEL_REGISTRY` ở `segmentation/models/__init__.py`.

### Bước 4: Tách biệt `core/` trong Segmentation
- Gom các file `losses.py`, `metrics.py`, `metrics_medpy.py`, `segmentation_reporting.py`, `training_logs.py`, `util.py` từ `utils/` vào thư mục `segmentation/core/` để nhất quán với cấu trúc `landmark/core/`.

---

## 4. Bổ sung Update & Nâng cấp Tính năng (Feature Upgrades)

Ngoài việc sắp xếp lại thư mục, phần Segmentation nên được bổ sung 4 cập nhật quan trọng sau:

### 4.1 Thêm Config System bằng YAML (`segmentation/cfg/default.yaml`)
Hiện tại `main.py` nhận tham số qua `argparse` với hàng chục cờ lệnh dài dòng.
**Nâng cấp**:
Tạo file `segmentation/cfg/default.yaml` quản lý thông số huấn luyện tập trung:

```yaml
# segmentation/cfg/default.yaml
experiment:
  project: "runs/segmentation"
  name: "rwkv_unet_mesko"
  seed: 2006
  device: "cuda:0"

model:
  name: "RWKV_UNetV2"
  num_classes: 4
  input_channels: 1
  img_size: 256
  deep_supervision: true

dataset:
  name: "mesko"
  root: "data/mesko"
  batch_size: 16
  num_workers: 4
  val_fraction: 0.15

training:
  epochs: 150
  optimizer: "adamw"
  lr: 0.0003
  weight_decay: 0.01
  scheduler: "cosine"
  loss: "dice_ce" # dice, ce, focal, combo

reporting:
  pixel_spacing: 0.10
  save_dashboard: true
```

### 4.2 Chuẩn hóa Output & Logging Format
- Đảm bảo khi chạy `train.py`, kết quả lưu về thư mục `runs/segmentation/<experiment_name>/` bao gồm:
  - `best_model.pth` / `last_model.pth`
  - `metrics.csv` (lưu Dice, IoU, HD95, Loss theo từng epoch)
  - `dashboard_segmentation.png` (đồ thị loss & metrics tự động cập nhật)
  - `eval_report.json` (kết quả đánh giá chi tiết per-class)

### 4.3 Tích hợp Unified Evaluation Script (`segmentation/eval.py`)
Tạo một script đánh giá chuyên biệt `eval.py` cho phép load checkpoint đã train và chạy test trên tập Validation/Test hoặc Zero-shot dataset:

```bash
python segmentation/eval.py --weights runs/segmentation/rwkv_unet/best_model.pth --dataset mesko --save-vis
```

### 4.4 Kết hợp Pipeline (Joint Pipeline - Segmentation + Landmark)
Trong tương lai, nếu hệ thống Uknee cần chạy kết hợp 2 bài toán (ví dụ: dùng **Landmark** để định vị khớp gối → crop ROI → dùng **Segmentation** để phân vùng sụn/xương):
- Thư mục `deploy/` hoặc một module pipeline ở root sẽ import cả 2 module:
  - `from landmark.utils.api import KneePose`
  - `from segmentation.models import build_model`
- Việc gom nhóm `segmentation/` và `landmark/` riêng biệt giúp pipeline kết hợp này cực kỳ sạch và dễ bảo trì.

---

## 5. Tóm tắt So sánh Cấu trúc

| Tiêu chí | Cấu trúc Cũ (Root Mix) | Cấu trúc Mới (Modular Split) |
|---|---|---|
| Phân tách Task | ❌ Trộn lẫn ở Root | ✅ Tách biệt 100%: `segmentation/` & `landmark/` |
| Thư mục Root | ❌ Bị rác bởi 15+ files & folders | ✅ Sạch sẽ, chỉ chứa 2 package chính + docs |
| Tên thư mục Model | ⚠️ Viết hoa (`CNN`, `Hybrid`...) | ✅ Chuẩn Pythonic (`cnn`, `hybrid`...) |
| Quản lý Config | ⚠️ `argparse` thủ công dài dòng | ✅ File YAML tập trung (`cfg/default.yaml`) |
| Quản lý Loss & Metrics | 📁 Tản mạn ở `utils/` | 📂 Gom gọn vào `segmentation/core/` |
| Đặt tên Entrypoint | ⚠️ `main.py`, `main_multi3d.py` | ✅ `train.py`, `train_multi3d.py`, `eval.py` |

---

> **Kết luận**: Cập nhật nội bộ các thư mục Segmentation của bạn (`models/`, `dataloader/`, `utils/`) hiện tại **đã rất tốt về mặt chức năng và độ phong phú của mô hình**. Việc gom toàn bộ các thư mục này vào `segmentation/` sẽ giúp dự án Uknee trở nên hoàn chỉnh, chuyên nghiệp và tách biệt tuyệt đối giữa 2 bài toán **Segmentation** và **Landmark Pose**.
