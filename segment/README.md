# segment

## CLI thống nhất

Default được đặt trong `cfg/default.yaml`; option CLI ghi đè trực tiếp. Model
là tên trong `segment.models.MODEL_REGISTRY`. Mỗi lần train lưu vào
`<project>/runs/<name>` với `weights/`, `samples/`, `args.yaml`, `summary.yaml`
và ONNX (RWKV_UNetV3/V5/V6).
`best.pt` vẫn được cập nhật ngay khi Dice tốt hơn. Để không làm chậm training,
ONNX chỉ đồng bộ ở epoch 1, mỗi 10 epoch và epoch cuối, với điều kiện có best
mới kể từ lần export trước. Có thể đổi chu kỳ bằng `--onnx-export-interval`.
Ảnh sample có chiều rộng 800 px và giữ nguyên tỷ lệ.

`weights/best.pt` là checkpoint inference gọn, không chứa AdamW state.
`weights/last.pt` giữ đầy đủ optimizer để `--resume`, nên thường lớn gần gấp ba
weight FP32. ONNX là FP32 deployment graph nên kích thước gần `best.pt` là bình thường.

Để chạy từ notebook hoặc thư mục ngoài repository, cài source một lần bằng
đúng Python của kernel:

```python
import sys
UKNEE_SOURCE = "/path/to/Uknee"
!{sys.executable} -m pip install --no-deps --no-build-isolation -e "{UKNEE_SOURCE}"
```

```bash
python -m segment.main \
  --model RWKV_UNetV3 \
  --project /projects/BMammo/Knee \
  --dataset /projects/BMammo/Knee/data/unet_mesko5seg \
  --imgsz 540x640 \
  --batch 4 \
  --epochs 200 \
  --base_lr 0.001 \
  --gpu '[0,1]' \
  --seed 2026 \
  --aug_strategy xray \
  --name Unet_mesko_540x640_RWKV3
```

`--dataset /unet_mesko5seg` hoặc `--dataset unet_mesko5seg` là dạng rút gọn
cho `<project>/data/unet_mesko5seg`. Đường dẫn tuyệt đối tồn tại luôn được giữ
nguyên. Alias cũ `--base_dir`, `--batch_size`, `--max_epochs`, `--img_size`,
`--exp_name` tiếp tục hoạt động.

Kích thước luôn theo **HxW (height x width)**:

- `--imgsz 640` → `[640, 640]`.
- `--imgsz 540x640` → `[540, 640]`.
- `--imgz` là alias để tương thích lệnh cũ/gõ nhầm.

Ảnh và mask được letterbox tới đúng HxW, giữ tỷ lệ giải phẫu và dùng nearest
neighbor cho mask. Cặp H/W được ghi giống nhau trong `args.yaml`,
`summary.yaml` và metadata ONNX (`target_height`, `target_width`).

Các argument chung với landmark: `--project`, `--dataset`, `--model`,
`--imgsz`, `--batch`, `--epochs`, `--base_lr`, `--gpu`, `--seed`,
`--aug_strategy`, `--name`, `--exist_ok/--no-exist_ok`. `--gpu` nhận `[0]`,
`[0,1]`, `0,1`; `[-1]` chọn CPU. Mặc định `exist_ok=True`, và mọi folder cần
thiết được tạo bằng `parents=True, exist_ok=True`.

Tên phiên bản hiện tại:

- `RWKV_UNetV5`: bản MedAxial gọn trước đây có tên V6.
- `RWKV_UNetV6`: bản matrix-state V6 đã cân lại để có thể huấn luyện ở độ
  phân giải y tế. Cấu hình giữ width nông `48/72`, dùng
  `depths=(2,2,3,2)`, `dims=(48,72,136,216)` và chỉ chạy matrix-state hai
  chiều tại bottleneck stage 4. Model còn khoảng `16.05M` tham số; phép đo
  autograd `128x128` giảm khoảng `58.8%` activation được lưu so với việc bật
  matrix-state ở cả stage 3 và 4.
- ONNX tương ứng là `rwkv_unetv5.onnx` và `rwkv_unetv6.onnx`.

Checkpoint của run V6 gọn cũ phải resume bằng `--model RWKV_UNetV5`; loader
sẽ từ chối kiến trúc không khớp thay vì âm thầm load một phần weight.

Muốn dùng YAML khác:

```bash
python -m segment.main --config segment/cfg/default.yaml \
  --model RWKV_UNetV5 --project /projects/BMammo/Knee \
  --dataset /unet_mesko5seg --name rwkv5_compact_baseline
```
