# SUGGEST — Scope đã chốt cho `segment` và `landmark`

> Cập nhật: 2026-08-12
>
> Tài liệu này thay thế bản audit ban đầu. Nội dung bên dưới phản ánh các quyết định đã chốt cho dữ liệu hiện tại: ưu tiên RWKV V3/V6, YOLO landmark, ONNX tự động, RGB thống nhất và augmentation X-ray không flip.

## 1. Phạm vi được áp dụng

| Hạng mục | Quyết định | Trạng thái |
|---|---|---|
| Segment model | Chỉ tập trung `RWKV_UNetV3` và `RWKV_UNetV6` | Đã áp dụng |
| Landmark model | Tự export ONNX cho YOLO trong `landmark` | Đã áp dụng |
| Input color | Model-facing tensor dùng RGB, float32 | Đã áp dụng |
| Kích thước ảnh nguồn | Chấp nhận mọi tỉ lệ/kích thước, letterbox giữ aspect ratio | Đã áp dụng |
| Segment augmentation | Tắt horizontal/vertical flip; X-ray không Cutout/CoarseDropout | Đã áp dụng |
| Landmark augmentation | Khóa `fliplr=0`, `flipud=0`, `erasing=0`, `cutmix=0`, `copy_paste=0` | Đã áp dụng |
| Best model | Tự tạo ONNX từ `weights/best.pt` khi train hoàn tất | Đã áp dụng |
| ONNX metadata | Lưu input shape, RGB, normalization, letterbox và output contract | Đã áp dụng |
| `summary.yaml` | Bổ sung preprocessing và deployment/ONNX schema | Đã áp dụng |
| MESKO split | Giữ cơ chế hiện tại do dữ liệu còn ít | Không thay đổi |
| Pixel spacing | Xem là đã xử lý trước, không mở rộng trong đợt này | Bỏ khỏi scope |
| Crop | Giữ augmentation crop hiện tại | Không thay đổi |
| Các model segment khác | Chưa đầu tư export/compatibility | Bỏ khỏi scope hiện tại |

## 2. Output sau huấn luyện

### Segment

Ví dụ train `RWKV_UNetV3`:

```text
runs/segment/<experiment>/
├── args.yaml
├── summary.yaml
├── results.csv
├── segment_dashboard.png
├── segment_metrics.png
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── rwkv_unetv3.onnx
└── samples/
    └── segment_sample_e{epoch}.png
```

Ví dụ train `RWKV_UNetV6`, artifact tương ứng là `rwkv_unetv6.onnx`.

### Landmark2 / YOLO

Tên ONNX lấy từ model YAML đang train:

```text
runs/landmark/<experiment>/
├── summary.yaml
├── results.csv
├── landmark_dashboard.png
├── landmark_metrics.png
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── yolo26_pose.onnx
└── samples/
    └── landmark_sample_e{epoch}.png
```

Với `yolo26-pose-v9.yaml`, tên artifact được chuẩn hóa thành `yolo26_pose_v9.onnx`.

## 3. Contract ảnh đầu vào

### 3.1 Màu và normalization

Contract chung cho segment và landmark:

```yaml
layout: NCHW
dtype: float32
color_space: RGB
value_range: [0.0, 1.0]
normalization:
  mode: scale_0_1
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
```

- Dataset segment đọc bằng OpenCV nhưng chuyển BGR → RGB ngay sau decode.
- Landmark2 tiếp tục đọc OpenCV BGR ở tầng I/O, sau đó `Format`/predictor chuyển sang RGB trước model; `bgr=0.0` được khóa trong train config.
- PIL inference dùng `.convert("RGB")` nên khớp train contract.
- Checkpoint segment lưu thêm `preprocess` để không phải suy luận normalization theo tên dataset.

### 3.2 Kích thước tự do nhưng không bóp méo

Ảnh nguồn có thể là ngang, dọc hoặc bất kỳ độ phân giải nào. Pipeline dùng:

```text
source image → resize giữ aspect ratio → center padding → network canvas
prediction → bỏ padding → nearest resize về source shape
```

Điểm cần phân biệt:

- **Source spatial shape là dynamic**: API nhận ảnh kích thước bất kỳ.
- **Network canvas là fixed**: ví dụ `256×256` cho segment hoặc `640×640` cho YOLO.
- Ảnh không bị kéo giãn thành hình vuông; phần thiếu được padding.
- ONNX RWKV V3/V6 chỉ dynamic theo batch. Không quảng bá dynamic height/width trong graph vì các nhánh RWKV được trace theo network canvas.
- Metadata ONNX ghi cả `source_spatial_shape=dynamic` và `network_input_shape` để runtime xử lý đúng.

Segment dùng padding 0. YOLO dùng letterbox padding 114 theo pipeline của landmark.

## 4. Augmentation X-ray đã chốt

### Segment

Preset `xray` giữ:

- crop có bảo toàn foreground;
- affine nhẹ;
- CLAHE, brightness/contrast, gamma, tone curve;
- noise/blur/sharpen nhẹ;
- downscale nhẹ;
- grid distortion xác suất thấp.

Preset không dùng:

- horizontal flip;
- vertical flip;
- Cutout;
- CoarseDropout.

Flip cũng đã được bỏ khỏi spatial policy chung của segment. Crop hiện tại được giữ vì dữ liệu ít và cần tăng đa dạng; chưa áp dụng class-aware crop.

### Landmark2

Các giá trị được khóa khi gọi `KneePose.train()` nên override bên ngoài không vô tình bật lại:

```yaml
flipud: 0.0
fliplr: 0.0
cutmix: 0.0
copy_paste: 0.0
erasing: 0.0
bgr: 0.0
```

Mosaic vẫn được giữ cho model phù hợp. Các model single-image/head đặc biệt vẫn tắt mosaic theo logic sẵn có.

## 5. ONNX tự động

### 5.1 Segment RWKV V3/V6

Sau epoch cuối:

1. Load lại chính xác weight từ `best.pt`.
2. Wrap output về một tensor `logits` ổn định.
3. Export opset 17, input `images`, output `logits`.
4. Gắn metadata vào ONNX.
5. Chạy `onnx.checker`.
6. Chạy cùng dummy input bằng PyTorch và ONNX Runtime CPU.
7. Kiểm tra numerical parity bằng `assert_allclose`.
8. Ghi SHA-256, file size và parity result vào `summary.yaml`.

Nếu export hoặc parity fail, file ONNX chưa hoàn chỉnh bị xóa và quá trình kết thúc bằng lỗi rõ ràng. `best.pt` vẫn được giữ.

Auto-export hiện chỉ chạy với:

```text
RWKV_UNetV3
RWKV_UNetV6
```

Model khác vẫn train/save checkpoint bình thường nhưng `deployment.onnx.status` là `not_supported`.

Có thể tắt khi debug:

```bash
python -m segment.main ... --no-auto-export-onnx
```

### 5.2 Landmark2 YOLO

Sau final evaluation:

1. Load `best.pt`.
2. Export bằng fixed Uknee pose output contract.
3. Đặt tên theo model YAML nguồn.
4. Gắn metadata và chạy `onnx.checker`.
5. Chạy ONNX Runtime và so numerical parity với PyTorch.
6. Ghi SHA-256/file size/metadata/parity vào `summary.yaml`.

Có thể tắt bằng:

```bash
python -m landmark.train ... --no-auto-export-onnx
```

## 6. Metadata nằm trong ONNX

Các key dùng chung:

```text
uknee.schema_version
uknee.task
uknee.model_name
uknee.source_checkpoint
uknee.opset
uknee.preprocess
uknee.output
```

Segment có thêm:

```text
uknee.class_names
```

`uknee.preprocess` là JSON chứa:

- source shape dynamic;
- network input `[N,C,H,W]`;
- RGB/GRAYSCALE;
- float32 và range `[0,1]`;
- mean/std;
- letterbox target height/width;
- keep-aspect flag;
- padding value/placement;
- interpolation image/mask.

`uknee.output` mô tả:

- segment: logits NCHW, số class, sigmoid/argmax và raw class-ID label map;
- landmark: `detections`, `num_detections`, `canonical`, keypoint format và coordinate space.

Runtime sau này phải đọc metadata thay vì hard-code preprocessing.

## 7. `summary.yaml` schema version 2

Cả hai pipeline bổ sung:

```yaml
schema_version: 2
preprocessing:
  # cùng contract với metadata ONNX
deployment:
  auto_export_onnx: true
  onnx:
    status: ready
    path: weights/model_name.onnx
    format: onnx
    opset: 17
    sha256: "..."
    file_size_bytes: 0
    metadata: {}
    parity: {}       # segment V3/V6
artifacts:
  best_checkpoint: weights/best.pt
  last_checkpoint: weights/last.pt
  onnx_model: weights/model_name.onnx
```

Các trường paper hiện có như parameter, GFLOPs, train duration, best/final metrics và sample artifacts được giữ nguyên.

## 8. Các đề xuất cũ đã bỏ hoặc hoãn

Không thực hiện trong đợt này:

- bỏ MESKO fallback hoặc bắt buộc train/val/test riêng;
- patient-level split validator;
- thay đổi pixel spacing/HD95/ASSD;
- class-aware crop;
- thay loss hiện tại;
- AMP/DDP/refactor toàn bộ trainer segment;
- DICOM/DICOM SEG;
- support/export toàn bộ model zoo;
- sửa API concurrency/PHI ở mức production.

Các mục trên có thể xem lại khi dữ liệu lớn hơn hoặc bước vào clinical deployment. Chúng không còn là blocker cho scope nghiên cứu hiện tại.

## 9. Tiêu chí kiểm tra hiện tại

- [x] RGB decode segment đúng thứ tự channel.
- [x] Landmark train/predict đưa RGB vào model.
- [x] X-ray augment không có horizontal/vertical flip hoặc CoarseDropout/Cutout.
- [x] Ảnh không vuông được letterbox, không kéo giãn.
- [x] Segment prediction được đưa về kích thước ảnh nguồn.
- [x] Multiclass mask giữ raw class IDs.
- [x] `RWKV_UNetV6` có trong registry và model metadata.
- [x] RWKV V3 export ONNX và chạy ONNX Runtime.
- [x] RWKV V6 export ONNX và chạy ONNX Runtime.
- [x] YOLO pose export ONNX, metadata đọc được và output shapes đúng.
- [x] `summary.yaml` có preprocessing/deployment schema.

## 10. Việc nên làm tiếp theo

Sau khi chạy một training thật cho từng model, chỉ cần kiểm tra:

1. `best.pt` và ONNX có cùng model name/checksum record trong `summary.yaml`.
2. Dùng 5–10 ảnh thật với nhiều aspect ratio để so PyTorch và ONNX output.
3. Kiểm tra overlay sau inverse letterbox nằm đúng vị trí trên ảnh gốc.
4. Benchmark latency/VRAM cho đúng canvas sẽ deploy.
5. Archive chung thư mục `weights/`, `summary.yaml`, `args.yaml`, class metadata và code commit dùng cho paper.
