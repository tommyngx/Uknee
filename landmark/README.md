# landmark

`landmark` is the compact, pose-only successor to the archived `landmark0`. It embeds
the required YOLO26/OA26 implementation directly under `core`, `data`, and
`nn`; it does not bootstrap `_vendor` and does not require the `ultralytics`
PyPI package.

## Public API

```python
from landmark import KneePose

model = KneePose("landmark/cfg/models/yolo26-pose-v9.yaml")
model.train(data="landmark/cfg/datasets/mesko4gf2.yaml", epochs=100, device=0)
results = model.predict("image.png")
artifact = model.export(format="onnx", imgsz=640)
```

## CLI thống nhất

Mọi giá trị truyền từ CLI sẽ ghi đè cấu hình mặc định trong
`cfg/default.yaml`. Kết quả luôn nằm tại `<project>/runs/<name>`; `weights/`,
`samples/`, `args.yaml`, `summary.yaml` và ONNX nằm bên trong run đó.
Pose đồng bộ ONNX mỗi khi `weights/best.pt` được cập nhật. Detection export
best đầu tiên, sau đó mặc định mỗi 10 epoch khi có best mới, và hoàn tất từ
best checkpoint cuối. Ảnh sample có chiều rộng 800 px và giữ nguyên tỷ lệ.

Nếu chạy từ notebook/thư mục ngoài repository, cài source một lần bằng đúng
Python của kernel:

```python
import sys
UKNEE_SOURCE = "/path/to/Uknee"
!{sys.executable} -m pip install --no-deps --no-build-isolation -e "{UKNEE_SOURCE}"
```

`--project` là thư mục data/output, không phải source code. Nếu chưa cài
package thì cần `%cd /path/to/Uknee` trước khi dùng `-m landmark.train`.

`--gpu '[0,1]'` được áp dụng trước khi import PyTorch và đặt
`CUDA_VISIBLE_DEVICES=0,1`. Sau dòng khởi động, log phải báo
`torch.cuda.device_count(): 2`/hai CUDA device. Nếu scheduler của trường chỉ
cấp một GPU, cần yêu cầu hai GPU từ scheduler trước; `nvidia-smi` có thể vẫn
hiện card mà job không được phép sử dụng.

```bash
python -m landmark.train \
  --model yolo26-pose-v9 \
  --project /projects/BMammo/Knee \
  --dataset /projects/BMammo/Knee/data/mesko_landmark \
  --imgsz 540x640 \
  --batch 4 \
  --epochs 200 \
  --base_lr 0.001 \
  --gpu '[0,1]' \
  --seed 2026 \
  --aug_strategy xray \
  --name yolo26_pose_v9_540x640
```

`--model` nhận đường dẫn YAML/checkpoint hoặc tên YAML trong `cfg/models`
(có thể bỏ `.yaml`). `--dataset`/`--data` nhận trực tiếp YAML hoặc folder. Nếu
nhận folder, chương trình tìm `data.yaml`, `dataset.yaml` hoặc YAML đầu tiên,
sau đó tạo bản runtime tại `<project>/.uknee/datasets` với `path` tuyệt đối.
Nếu folder chưa có YAML, schema landmark 4 vùng/51 keypoint được tạo tự động.

`--dataset_link` (hoặc `--dataset-link`) trỏ tới thư mục dữ liệu thật (chứa
`images/` và `labels/`) và thay `path:` của dataset YAML, nhờ đó dùng chung một
YAML trong repo cho nhiều máy mà không phải sửa file. YAML gốc không bị ghi đè:
bản runtime với `path` mới nằm ở `<project>/.uknee/datasets`. Link rỗng hoặc
không tồn tại sẽ báo lỗi, không tự quay về `path:` cũ.

```bash
python -m landmark.train \
  --model rtmo-pose \
  --project /scratch/phenooa/run \
  --dataset landmark/cfg/datasets/hip_point.yaml \
  --dataset_link /scratch/phenooa/data/HipX02 \
  --name rtmo_hipx02
```

### Hai nhánh cấu hình: knee và hip

Dataset YAML quyết định schema (tên vùng, số keypoint mỗi vùng, các path), và
model YAML phải thuộc cùng nhánh. Ghép sai nhánh sẽ báo lỗi ngay khi dựng model.

| Nhánh | Dataset YAML | Schema | Model YAML |
|---|---|---|---|
| Knee (mặc định) | `cfg/datasets/mesko4gf2.yaml` | femur/tibia/fibula/patella, `45/51/24/9` → 129 điểm, 51 slot | `rtmo-pose`, `yolo26-pose-v9`, `yolo26-pose-v1`, `hrnet-*`, `vitpose-*` |
| Hip | `cfg/datasets/hip_point.yaml` | femur/lesser_trochanter/greater_trochanter/acetabulum, `59/7/9/10` → 85 điểm, 59 slot | `rtmo-pose-hip`, `yolo26-pose-hip` |

`landmark.train` đọc `--dataset` và kích hoạt schema qua biến môi trường
`UKNEE_LANDMARK_SCHEMA` trước khi import model (DDP worker kế thừa biến này).
Khi gọi API trực tiếp (không qua `landmark.train`), đặt biến này thành đường
dẫn dataset YAML trước khi `import landmark`. Cần thêm model hip khác (HRNet,
ViTPose) thì copy YAML knee tương ứng và đổi `num_landmarks: 85`,
`kpt_shape: [59, 3]`.

Label hip: mỗi dòng có `5 + 59 * 3 = 182` giá trị, mỗi ảnh đủ 4 dòng class
0..3, và `count` keypoint đầu của vùng phải có visibility > 0 (slot còn lại
bằng 0). Ảnh kết quả (`samples/landmark_sample_e*.png`, dashboard, metrics)
ghi tiến độ dạng `Epoch 5/20`.

```bash
python -m landmark.train \
  --model rtmo-pose-hip \
  --project /scratch/phenooa/run/hiplandmark \
  --dataset landmark/cfg/datasets/hip_point.yaml \
  --dataset_link /scratch/phenooa/data/HipX02 \
  --imgsz 736x448 --batch 4 --epochs 200 --base_lr 0.001 \
  --gpu '[0]' --seed 2026 --aug_strategy xray \
  --name rtmo_hipx02 --no-auto-export-onnx
```

### Train tiếp từ checkpoint (fine-tune)

Đưa `best.pt` của run cũ vào `--model` (kiến trúc lấy từ YAML lưu trong
checkpoint) hoặc giữ `--model <yaml>` và thêm `--pretrained <best.pt>`; hai
cách nạp đúng toàn bộ trọng số như nhau. Vẫn phải truyền `--dataset` cùng nhánh
để kích hoạt schema, và đặt `--name` mới vì `--exist_ok` mặc định bật sẽ ghi
đè run cũ. Optimizer, scheduler, warmup và bộ đếm epoch bắt đầu lại từ đầu nên
nên hạ `--base_lr` so với lần train gốc.

```bash
python -m landmark.train \
  --model /scratch/phenooa/runs/rtmo_hipx02/weights/best.pt \
  --project /scratch/phenooa \
  --dataset landmark/cfg/datasets/hip_point.yaml \
  --dataset_link /scratch/phenooa/data/HipX02 \
  --imgsz 736x448 --batch 32 --epochs 100 --base_lr 0.0003 \
  --gpu '[0]' --seed 2026 --aug_strategy xray \
  --name rtmo_hipx02_ft warmup_epochs=1
```

`--resume <last.pt>` chỉ để chạy nốt một run bị ngắt (giữ nguyên tổng epoch,
nạp lại optimizer/EMA); `best.pt` và `last.pt` của run đã kết thúc đã bị strip
optimizer nên không resume được, và `--resume` không tăng thêm epoch.

`yolo26-pose-v9` dùng compound scale được khai báo trực tiếp trong YAML
(hiện tại là `l`); CLI vẫn ghi `--model yolo26-pose-v9`.

Các YOLO pose model dùng curriculum detection-first: epoch 1–50 chỉ tối ưu
box/class/DFL; epoch 51–75 mở tuyến tính pose, keypoint-object, heatmap và
structure/refinement loss. Weight pose cuối ramp bằng 50% cấu hình gốc để
auxiliary landmark loss không lấn át việc học bounding box. Cấu hình đầy đủ
được lưu trong `summary.yaml` tại `training.loss_curriculum`.

Đường dẫn dataset rút gọn `--dataset /mesko_landmark` hoặc
`--dataset mesko_landmark` được hiểu là
`<project>/data/mesko_landmark` nếu đường dẫn được truyền không tồn tại.

Kích thước dùng thứ tự **HxW (height x width)**:

- `--imgsz 640` → `[640, 640]`.
- `--imgsz 540x640` → `[540, 640]`.
- `--imgz` và `--img_size` là alias tương thích.

Các argument chung: `--project`, `--dataset`, `--model`, `--imgsz`, `--batch`,
`--epochs`, `--base_lr`, `--gpu`, `--seed`, `--aug_strategy`, `--name`,
`--exist_ok/--no-exist_ok`. `--gpu` nhận `[0]`, `[0,1]`, `0,1`; `[-1]` chọn
CPU. Mặc định `exist_ok=True`, nên các folder cần thiết được tạo bằng
`parents=True, exist_ok=True`. Argument backend ít dùng hơn vẫn có thể đặt sau
các option trên dưới dạng `KEY=VALUE`.

Với canvas chữ nhật, landmark tự tắt mosaic/multi-scale (hai pipeline này giả
định canvas vuông) và dùng letterbox/affine để không kéo giãn giải phẫu. H/W
được lưu thống nhất trong `args.yaml`, `summary.yaml` và metadata ONNX. Trainer
sẽ làm tròn từng cạnh lên bội số stride của model (ví dụ H=540 có thể thành
544); metadata luôn ghi kích thước mạng thực sự sau bước này.

The public configs are base YOLO26 Pose, OA26 V1, OA26 V9, HRNet-W32,
HRNet-W48, ViTPose-S, ViTPose-B, and RTMO. RTMO is adapted to the repository's
four anatomical regions and canonical 129-landmark contract while retaining
its multi-level grouped pose head and DCC/GAU coordinate classifier. Export is
intentionally limited to TorchScript and ONNX. Unsupported tasks, remote
tracking integrations, and deployment backends fail explicitly instead of
importing optional Ultralytics modules.

Canonical model configs:

```text
cfg/models/hrnet-w32-pose.yaml
cfg/models/hrnet-w48-pose.yaml
cfg/models/vitpose-s-pose.yaml
cfg/models/vitpose-b-pose.yaml
cfg/models/rtmo-pose.yaml
```

Run the regression suite with:

```bash
python -m unittest discover -s landmark/tests -v
```

## Knee detection CLI

Bounding-box detection is intentionally separate from the landmark/pose entry
point. `train_det` validates five-column YOLO labels and creates a deterministic
train/validation split when a dataset currently contains only a train folder.

```bash
python -m landmark.train_det \
  --model yolo26-detect \
  --config landmark/cfg/detect/kneelocation.yaml \
  --data landmark/cfg/datasets/kneelocation.yaml \
  --project /projects/BMammo/Knee \
  --gpu '[0,1]' \
  --name yolo26_detect_kneelocation
```

Outputs use the same `<project>/runs/<name>` convention, including `weights/`,
`samples/`, `args.yaml`, `results.csv`, the detection dashboard/metric report,
validation plots, ONNX and `summary.yaml`. Detection ONNX is exported for the
first best checkpoint, then every `--onnx-export-interval` epochs only when the
best changed, and finalized after training. The detection preset is used by
default and can be overridden from the CLI (for example `--epochs 300`). Medical
landmark metrics are not emitted for detection. Use `--no-auto-export-onnx` when
only PyTorch checkpoints are required.

See the repository-level `report.yaml` for measured structural, parity, loss,
dataset, and export differences against `landmark0`.
