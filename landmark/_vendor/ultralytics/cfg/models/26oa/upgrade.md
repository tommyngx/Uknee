# Tài liệu phân tích & Giải pháp cải thiện tốc độ huấn luyện mô hình Pose

Tài liệu này ghi lại phân tích chi tiết về sự khác biệt hiệu năng giữa hai phiên bản cấu hình **v4 (HRNetLite)** và **v7 (HRNet)**, đồng thời gợi ý các hướng tối ưu hóa để cân bằng giữa tốc độ huấn luyện và độ chính xác.

---

## 1. Phân tích nguyên nhân v7 (HRNet) học chậm hơn v4 (HRNetLite)

Sự khác biệt chính nằm ở cấu trúc backbone được sử dụng:
* **v4**: Sử dụng [HRNetLite](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L101) (bản rút gọn siêu nhẹ).
* **v7**: Sử dụng [HRNet](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L164) (bản canonical/standard).

### Chi tiết so sánh độ phức tạp tính toán:

| Đặc tính so sánh | [HRNetLite](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L101) (v4) | [HRNet](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L164) (v7) | Ảnh hưởng hiệu năng |
| :--- | :--- | :--- | :--- |
| **Số lượng BasicBlock** | **12 blocks** | **104 blocks** | Lượng tính toán của v7 gấp gần 10 lần v4. |
| **Khối Bottleneck ở Stage 1** | Không sử dụng | 4 khối Bottleneck nặng | Stage 1 của v7 tốn nhiều tài nguyên hơn trước khi tách nhánh. |
| **Tính toán ở nhánh P2 (stride 4)** | Chỉ chạy 3 blocks | Chạy tới 32 blocks và 4 Bottleneck | Nhánh P2 có độ phân giải lớn nhất ($160 \times 160$ với ảnh đầu vào 640). Chạy nhiều block ở độ phân giải này là nguyên nhân chính gây thắt cổ chai (bottleneck) tốc độ trên GPU. |
| **Số lần kết hợp nhánh (Fusion)** | 3 lần | 8 lần | Phép toán nội suy (Upsampling) và cộng chéo nhánh diễn ra liên tục ở v7, tốn băng thông bộ nhớ GPU. |

---

## 2. Các đề xuất cải thiện (Nhanh hơn v7, Chính xác hơn v4)

### Cách 1: Tối ưu hóa cấu hình `HRNetLite` trực tiếp qua YAML (Không cần sửa code)
Lớp `HRNetLite` hỗ trợ cấu hình số lượng block (`num_blocks`) và số lượng stage (`num_stages`) trực tiếp từ file YAML cấu hình. Bạn có thể thay đổi để tăng dung lượng mô hình:

* **Tăng gấp đôi số block (Khuyên dùng thử nghiệm đầu tiên)**:
  Giúp nâng cao khả năng biểu diễn đặc trưng nhưng vẫn nhanh hơn rất nhiều so với v7.
  ```yaml
  # Sửa trong yolo26-posev4.yaml
  backbone:
    - [-1, 1, HRNetLite, [w32, False, [128, 256, 512, 512], True, 2, 3]]
  ```

* **Tăng cả số block và số stage (`num_blocks=2`, `num_stages=4`)**:
  ```yaml
  backbone:
    - [-1, 1, HRNetLite, [w32, False, [128, 256, 512, 512], True, 2, 4]]
  ```

* **Sử dụng phiên bản rộng hơn (`w48`) nhưng giữ số lượng block nhỏ**:
  ```yaml
  backbone:
    - [-1, 1, HRNetLite, [w48, False, [128, 256, 512, 512], True, 1, 3]]
  ```

---

### Cách 2: Làm nhẹ backbone `HRNet` tiêu chuẩn (Cần sửa code nhẹ)
Hiện tại lớp [HRNet](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L164) đang bị fix cứng `num_blocks=4` và danh sách module `[1, 4, 3]`. Ta có thể cấu hình lại lớp này để chấp nhận các tham số truyền vào từ YAML.

**Các bước thực hiện:**
1. Cập nhật phương thức **[_make_stage](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L212)** trong file [hrnet.py](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py) để nhận `num_blocks`:
   ```python
   @staticmethod
   def _make_stage(channels: tuple[int, ...], num_modules: int, num_blocks: int = 4) -> nn.Sequential:
       return nn.Sequential(*(HRFusionStage(channels, num_blocks=num_blocks) for _ in range(num_modules)))
   ```
2. Cập nhật phương thức khởi tạo **[__init__](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/modules/oa26/hrnet.py#L177)** của `HRNet` để nhận và truyền các tham số này:
   ```python
   def __init__(
       self,
       variant: str = "w32",
       pretrained: bool = False,
       out_channels: tuple[int, int, int, int] | list[int] = (128, 256, 512, 512),
       return_p2: bool = True,
       num_blocks: int = 2,                  # Giảm số block mặc định xuống 2
       modules_list: list[int] = [1, 2, 2],  # Giảm số lượng module ở stage 3 và 4
   ):
       ...
       self.stage2 = self._make_stage(widths[:2], num_modules=modules_list[0], num_blocks=num_blocks)
       self.stage3 = self._make_stage(widths[:3], num_modules=modules_list[1], num_blocks=num_blocks)
       self.stage4 = self._make_stage(widths, num_modules=modules_list[2], num_blocks=num_blocks)
   ```

---

### Cách 3: Sử dụng các Backbone tối ưu hóa GPU tốt hơn
HRNet có cấu trúc nhiều nhánh phức tạp dẫn đến việc đọc/ghi bộ nhớ GPU liên tục không tối ưu (memory access overhead). Bạn có thể thử nghiệm các backbone dạng chuỗi hiện đại đã có sẵn trong mã nguồn dự án tại [tasks.py](file:///Users/francistommy/Desktop/BugHunter/Project/OApheno/ultralytics/nn/tasks.py#L113-L121):
* **`ConvNeXtV2T` (Tiny)** hoặc **`ConvNeXtV2N` (Nano)**:
  Sử dụng cấu trúc tích chập khối sâu (Depthwise Separable Convolution) tối ưu phần cứng cực tốt, giúp học nhanh và có độ chính xác rất cao nhờ dung lượng biểu diễn tốt.
