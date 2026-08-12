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

See the repository-level `report.yaml` for measured structural, parity, loss,
dataset, and export differences against `landmark0`.
