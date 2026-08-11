# RWKV-UNet Model Zoo (Versions 1-6)

Thư mục này chứa các biến thể của kiến trúc **RWKV-UNet** được tích hợp và tối ưu hóa cho bài toán phân đoạn ảnh y tế trong bộ benchmark **U-Bench** (Uknee).

---

## Bảng so sánh các phiên bản

| Đặc tính | RWKV_UNet (V1) | RWKV_UNetV2 | RWKV_UNetV3 | RWKV_UNetV4 | RWKV_UNetV6 (Mới nhất) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bản gốc tương ứng** | [juntaoJianggavin/RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet) | Nâng cấp từ V1 | Bản tối giản hóa của V1 | Nâng cấp lên RWKV-6 | Thiết kế nâng cấp MedAxial-RWKV6 |
| **Kiến trúc RWKV lõi** | RWKV-4 (Vision-RWKV v1) | RWKV-4 (Vision-RWKV v1) | RWKV-4 (Sửa đổi) | RWKV-6 (Finch) | **RWKV-6 (Finch) Matrix-State** |
| **Nhân CUDA (WKV)** | Bắt buộc (`wkv_cuda.cu`) | Bắt buộc (`wkv_cuda.cu`) | Không cần (PyTorch thuần) | Không cần (PyTorch thuần) | **Không cần** (PyTorch thuần có Recurrent Scan) |
| **Cơ chế Token Shift** | Static LERP (Nội suy tĩnh) | Static LERP (Nội suy tĩnh) | Static LERP (Nội suy tĩnh) | Dynamic LERP (ddlerp) | **Partial Dynamic LERP (Nội suy động w và v)** |
| **Cơ chế quét không gian** | Trải phẳng chuỗi 1D | Trộn theo hàng/cột (2 hướng) | Trải phẳng chuỗi 1D | Trải phẳng chuỗi 1D | **Trộn 4 hướng trục (Quad-Axial Scan)** |
| **Cổng Gate phụ** | Không có | Không có | Không có | Có (`self.gate` + SiLU) | **Có (`self.gate` + SiLU) & SkipGate** |
| **Kích thước ảnh tối đa** | $\le 256 \times 256$ | $\le 1024 \times 1024$ | Không giới hạn | Không giới hạn | **Không giới hạn** |
| **Ưu điểm** | Giống bản gốc nhất. | Chạy được ảnh to hơn trên CUDA. | Dễ cài đặt, không lỗi biên dịch CUDA. | Học động tốt hơn, hội tụ nhanh hơn. | Cơ chế Matrix-state thực thụ, quét đa hướng, tiết kiệm tham số và cực kỳ ổn định trên ảnh y tế độ phân giải cao. |

---

## Chi tiết kỹ thuật từng phiên bản

### 1. [RWKV_UNet (V1)](./RWKV_UNet.py)
* **Kiến trúc:** Triển khai chính xác theo paper gốc của tác giả Juntao Jiang. Bộ mã hóa sử dụng khối **IR-RWKV** (Inverted Residual RWKV) kết hợp phép tích chập cục bộ với trộn không gian toàn cục thông qua toán tử WKV của RWKV-4.
* **Hạn chế:** Nhân CUDA đi kèm được biên dịch cứng với `T_MAX = 1024`. Vì ảnh được làm dẹt hoàn toàn ($H \times W$ tokens), ảnh đầu vào có kích thước $> 256 \times 256$ (ví dụ: $512 \times 512$ sẽ tạo ra $64 \times 64 = 4096$ tokens ở stage đầu) sẽ gây tràn bộ nhớ đệm CUDA và crash chương trình.

### 2. [RWKV_UNetV2](./RWKV_UNetV2.py)
* **Kiến trúc:** Khắc phục giới hạn kích thước chuỗi của V1 bằng cơ chế **Strip-wise RWKV mixing**. Thay vì trải phẳng toàn bộ ảnh, mô hình thực hiện trộn thông tin theo hàng dọc và hàng ngang một cách độc lập (chiều dài chuỗi tối đa lúc này chỉ là $\max(H, W)$).
* **Cải tiến khác:** Tích hợp khối lọc biên tự động **Boundary-Aware Refinement** để tăng độ sắc nét ở rìa vùng phân đoạn.

### 3. [RWKV_UNetV3](./RWKV_UNetV3.py)
* **Kiến trúc:** Thiết kế cho các hệ thống không có GPU CUDA mạnh hoặc muốn chạy huấn luyện ảnh lớn mà không gặp rắc rối khi cài đặt/biên dịch nhân C++/CUDA.
* **Cơ chế:** Lớp `SafeVRWKVSpatialMix` loại bỏ hoàn toàn nhân CUDA WKV và thay bằng một phép tính xấp xỉ song song trong PyTorch thuần túy:
  $$x_{wkv} = \text{LayerNorm}(k + v)$$
  Đồng thời mô hình loại bỏ các tham số suy giảm lũy thừa tuần tự (decay) để tối ưu tốc độ huấn luyện trên ảnh lớn.

### 4. [RWKV_UNetV4](./RWKV_UNetV4.py)
* **Kiến trúc:** Nâng cấp lớp trộn không gian lên lõi **RWKV-6 (Finch)** nhưng vẫn giữ tính chất "Safe" (PyTorch thuần, không cần biên dịch nhân CUDA).
* **Cải tiến nổi bật:**
  1. **Dynamic LERP (Nội suy động):** Thay thế hệ số trộn tĩnh (`spatial_mix`) bằng các ma trận chiếu Low-Rank (LoRA-style). Tỉ lệ đóng góp của pixel dịch chuyển (`xx`) và pixel hiện tại (`x`) được tính toán động dựa vào ngữ cảnh của chính ảnh đầu vào thông qua lớp `spatial_maa_x`.
  2. **Gating Mechanism:** Bổ sung thêm cổng chiếu `gate` kết hợp hàm kích hoạt `SiLU` để lọc và làm nổi bật các đặc trưng quan trọng trước khi đưa qua lớp chiếu đầu ra `self.output`.

### 5. [RWKV_UNetV6 (MedAxial-RWKV6 U-Net)](./RWKV_UNetV6.py)
* **Kiến trúc:** Đây là biến thể cao cấp và hiện đại nhất, được thiết kế đặc biệt cho các bài toán phân đoạn ảnh y tế với các đặc điểm nổi bật:
  1. **Quét trục 4 hướng (Quad-Axial Matrix RWKV):** Quét qua các trục của ảnh theo 4 hướng: Trái sang Phải, Phải sang Trái, Trên xuống Dưới, và Dưới lên Trên. Các hướng quét ngược chia sẻ trọng số với hướng quét xuôi giúp tối ưu dung lượng tham số và tăng tính đối xứng.
  2. **Quét trạng thái ma trận bằng PyTorch thuần (`MatrixStateScan`):** Thực hiện tính toán cập nhật trạng thái ma trận (Matrix-state recurrence) nguyên bản của RWKV-6 hoàn toàn bằng PyTorch thuần túy. Trạng thái ma trận được tính ở kiểu dữ liệu `float32` để đảm bảo độ ổn định số học mà không cần nhân CUDA.
  3. **Trộn động một phần (Partial Dynamic LERP):** Để giảm độ phức tạp tính toán khi quét 4 hướng, mô hình chỉ tính toán động cho nhánh decay ($w$) và value ($v$), trong khi giữ các nhánh $r$, $k$, $g$ là các hệ số sigmoid tĩnh học được.
  4. **Cổng bỏ qua skip connection (`SkipGate`):** Sử dụng cơ chế cổng học được tại decoder để kiểm soát lượng thông tin chi tiết truyền từ encoder sang decoder, giúp hạn chế nhiễu tần số cao trên các tập dữ liệu y tế nhỏ.
