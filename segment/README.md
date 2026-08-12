# segment

## CLI thống nhất

Default được đặt trong `cfg/default.yaml`; option CLI ghi đè trực tiếp. Model
là tên trong `segment.models.MODEL_REGISTRY`. Mỗi lần train lưu vào
`<project>/runs/<name>` với `weights/`, `samples/`, `args.yaml`, `summary.yaml`
và ONNX (RWKV_UNetV3/V6).
Mỗi khi `weights/best.pt` được cập nhật, ONNX tương ứng cũng được export và thay thế ngay; ảnh sample có chiều rộng 800 px và giữ nguyên tỷ lệ.

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

Muốn dùng YAML khác:

```bash
python -m segment.main --config segment/cfg/default.yaml \
  --model RWKV_UNetV6 --project /projects/BMammo/Knee \
  --dataset /unet_mesko5seg --name rwkv6_baseline
```
