# Landmark model guide

Tài liệu này là chuẩn chung để train, đánh giá và so sánh các model landmark
trong `landmark/models`. Mọi model dự đoán 129 điểm theo tọa độ chuẩn hóa
`(x, y)` trong `[0, 1]`.

## Dataset và topology chuẩn

Dataset hiện có 442 ảnh nhưng chỉ có 188 nhóm ca độc lập do original/Flip và
ảnh hai bên cùng ca được gom chung. Split mặc định theo nhóm gồm 378 ảnh train
và 64 ảnh validation, không có leakage. Tất cả label hiện có đủ 129 điểm.

129 điểm phải được xem như sáu path độc lập:

| Path | Global indices | Số điểm |
|---|---:|---:|
| Femur | 0-44 | 45 |
| Tibia main | 45-85 | 41 |
| Tibia plateau A | 86-90 | 5 |
| Tibia plateau B | 91-95 | 5 |
| Fibula | 96-119 | 24 |
| Patella | 120-128 | 9 |

Không nối cạnh `85→86` hoặc `90→91`. Không sắp xếp landmark theo `x`/`y`, vì
contour cong và ảnh Flip đảo hướng không gian nhưng không đổi landmark identity.

## Model inventory

| Model | Tổng params | Trainable mặc định | Đầu ra chính | Vai trò khuyến nghị |
|---|---:|---:|---|---|
| `hrnet` | 275,329 | 275,329 | Global heatmaps | Baseline chính, nhẹ và giữ spatial layout |
| `vitpose` | 3,617,025 | 3,617,025 | Global heatmaps | Baseline transformer; cần theo dõi overfit |
| `adaptive_rwkv` | 17,983,715 | 983,298 | Spatial coarse + local heatmaps | Coarse-to-local ablation |
| `kneepv1` | 17,524,961 | 524,544 | Independent contour assignment | Contour-constrained baseline |
| `kneepv2` | 17,577,569 | 577,152 | Topology-aware contour assignment | Model ưu tiên cho thứ tự landmark |

Số trainable của các model RWKV không tính backbone 17,000,417 tham số đang
được đóng băng.

## Kiến trúc và giới hạn

### HRNet

- Giữ feature map không gian và dự đoán 129 heatmap độc lập.
- Phù hợp nhất làm control model trên dataset nhỏ.
- Dùng BatchNorm; nếu batch thực tế nhỏ hơn 4 nên cân nhắc GroupNorm.
- Không có topology loss nên vẫn phải báo cáo order metrics.

### ViTPose

- Patch size 16, sáu transformer layer và decoder heatmap ×4.
- Có deterministic 2D sine/cosine positional encoding.
- Chưa dùng pretrained encoder; với 188 nhóm ca độc lập, nguy cơ overfit cao.
- Chỉ so sánh công bằng khi cùng split, augmentation và số epoch với HRNet.

### Adaptive RWKV

- Backbone segmentation 11-class được đóng băng.
- Coarse head là spatial heatmap có bone prior; không dùng global pooling để
  tránh làm mất vị trí ở cặp original/Flip.
- Local refinement dùng patch khoảng 46 px ở input 640×640.
- Local sampling context được stop-gradient khỏi coarse head. Coarse coordinate
  và coarse heatmap loss chịu trách nhiệm học vị trí patch.
- Curriculum mặc định: coarse epoch 1-30, teacher refinement epoch 31-45, full
  inference từ epoch 46. Không dùng checkpoint dừng trước full phase.

`Ref/kneept640v0.pt` là artifact legacy ở epoch 23 của kiến trúc coarse pooling
cũ, MRE khoảng 97 px và chưa vào full phase. File này không tương thích với
spatial coarse head mới và không được dùng làm kết quả chuẩn.

### KneePV1

- Lấy 512 contour candidate cho mỗi bone từ segmentation backbone.
- Landmark query chỉ được attend candidate thuộc bone tương ứng.
- Hard output luôn nằm trên candidate contour.
- Assignment và argmax độc lập, vì vậy có thể trùng hoặc đảo thứ tự.
- `contour_oracle_px` là lower bound bắt buộc phải theo dõi. Spot-check sáu ảnh
  validation với checkpoint hiện tại: mean 2.50 px, median 1.71 px, p95 8.40 px.

### KneePV2

- Kế thừa KneePV1, thêm path identity, normalized order embedding và local path
  mixer.
- Loss gồm edge vector, curvature và assignment-overlap penalty.
- Topology loss bắt đầu epoch 5 và ramp trong 15 epoch.
- Evaluation ngăn trùng token trong từng path bằng confidence-priority matching.
- Cơ chế hiện tại bảo đảm uniqueness nhưng chưa bảo đảm monotonic arc-length;
  vì vậy phải chọn và báo cáo thêm `best_order.pt`.

## Loss chuẩn

Heatmap model dùng cùng temperature `1.0` cho spatial KL và soft-argmax. Target
heatmap cục bộ được detach khỏi predicted reference. Bone constraint dùng
penalty bị chặn `1-p` thay cho `-log(p)` để tránh gradient explosion.

| Model | Coarse coord | Coarse heatmap | Final coord | Heatmap/assignment | Topology |
|---|---:|---:|---:|---:|---:|
| HRNet | 0 | 0 | 1.0 | 1.0 | 0 |
| ViTPose | 0 | 0 | 1.0 | 1.0 | 0 |
| Adaptive RWKV | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| KneePV1 | 0 | 0 | 1.0 | 1.0 | 0 |
| KneePV2 | 0 | 0 | 1.0 | 1.0 | edge 0.25, curvature 0.10, duplicate 0.05 |

## Metric bắt buộc

Không kết luận model chỉ bằng MRE. Mỗi run phải báo cáo:

- `mre`, `median`, `p95` theo pixel ở resolution train.
- `pck2`, `pck4`, `pck8`, `failure_gt_8`.
- MRE riêng femur, tibia, fibula và patella.
- `order_inversion_rate` (Kendall inversion trên mọi cặp trong từng path).
- `adjacent_duplicate_rate`.
- `edge_length_relative_error`.
- `direction_error_degrees`.
- `contour_oracle_px` cho KneePV1/V2.

`best.pt` tối ưu MRE. `best_order.pt` tối ưu theo thứ tự từ ưu tiên cao xuống:
inversion rate, duplicate rate, rồi MRE. Khi báo cáo KneePV2 phải đánh giá cả hai.

## Checkpoint policy

- Model RWKV với `freeze_backbone: true` bắt buộc có segmentation checkpoint.
- Checkpoint chuẩn hiện tại là `Ref/checkpoint_last.pth`: RWKV_UNetV3, input 3
  channel, output 11 class, image 640×640.
- Train CLI fail-fast nếu backbone bị đóng băng nhưng checkpoint trống hoặc file
  không tồn tại.
- Full landmark checkpoint phải được evaluate bằng đúng model config đã lưu.
- Không dùng segmentation checkpoint 7-class `Ref/RWKV_200_0.966.pth` cho config
  11-class.

## Lệnh train chuẩn

```bash
python -m landmark.train --config landmark/config/hrnet.yaml
python -m landmark.train --config landmark/config/vitpose.yaml
python -m landmark.train --config landmark/config/adaptive_rwkv.yaml
python -m landmark.train --config landmark/config/kneepv1.yaml
python -m landmark.train --config landmark/config/kneepv2.yaml
```

CLI chỉ override giá trị YAML khi argument thực sự được truyền. Config resolved,
history, plot theo epoch và checkpoint được ghi dưới:

```text
landmark/runs/<experiment>/<timestamp>/
```

`landmark.evaluate` xuất thêm epoch checkpoint, tổng/trainable parameters và
`contour_oracle_px` khi model có contour candidates, giúp điền bảng kết quả mà
không phải tính thủ công.

## Protocol so sánh

1. Dùng cùng seed 2006 và cùng generated group split.
2. Không thay đổi augmentation giữa các model trong cùng bảng so sánh.
3. Giữ budget mặc định 120 epoch cho tất cả model; nếu đổi phải ghi rõ trong bảng.
4. Báo cáo cả `best.pt` và `best_order.pt` trên đúng validation split.
5. Ghi trainable params, epoch tốt nhất và contour oracle nếu có.
6. Kiểm tra overlay đã nối sáu path, không chỉ nhìn scatter point.
7. Không công bố kết quả từ checkpoint đang ở coarse/teacher phase.

### Result table template

| Model/checkpoint | Epoch | MRE ↓ | P95 ↓ | PCK4 ↑ | Inversion ↓ | Duplicate ↓ | Oracle ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| HRNet | | | | | | | n/a |
| ViTPose | | | | | | | n/a |
| Adaptive RWKV | | | | | | | n/a |
| KneePV1 | | | | | | | |
| KneePV2 best MRE | | | | | | | |
| KneePV2 best order | | | | | | | |
