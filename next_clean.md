# NEXT CLEAN — Runbook refactor `landmark` và `segment`

> Loại tài liệu: hướng dẫn triển khai cho agent ở các phiên sau
>
> Ngày audit: 2026-08-13
>
> Trạng thái: chưa thực hiện refactor source

## 1. Agent phải đọc phần này trước

Mục tiêu của đợt clean là tách các file active quá lớn thành module nhỏ để giảm
context/token và làm test dễ cô lập. `landmark/` và `segment/` phải tiếp tục là hai
package hoàn toàn riêng.

Các luật không được vi phạm:

1. Không merge `landmark` với `segment`.
2. Không tạo `common/`, `shared/` hoặc trainer chung giữa hai package.
3. Chỉ tách code; không thay thuật toán trong cùng commit.
4. Không đổi public API, CLI, YAML symbol, checkpoint path hoặc ONNX contract.
5. Giữ file import cũ làm compatibility facade khi cần.
6. Không quét hoặc copy code từ `landmark0/_vendor` để refactor `landmark`.
7. Không refactor toàn model zoo `segment` nếu model chưa được xác nhận active.
8. Không xóa class được YAML parser hoặc pickle resolve động chỉ dựa trên kết quả
   tìm import tĩnh.
9. Mỗi work package bên dưới là một task/commit độc lập.
10. Khi gặp parity fail, dừng work package và sửa ngay; không tiếp tục tách file
    khác để “làm xong một lượt”.

Tài liệu chiến lược dài hơn nằm tại:

```text
landmark0/_vendor/ultralytics/cfg/models/26oa/upgrade.md
```

Nếu `landmark0` đã được chuyển sang legacy, đường dẫn tương ứng là:

```text
legacy/landmark0/_vendor/ultralytics/cfg/models/26oa/upgrade.md
```

`next_clean.md` là runbook thi công. Agent không cần đọc toàn bộ upgrade plan nếu
chỉ thực hiện một work package đã được giao rõ.

## 2. Cách dùng runbook

Agent nhận task phải thực hiện đúng trình tự:

1. Xác định work package ID, ví dụ `L1` hoặc `S2`.
2. Chỉ đọc phần “Input”, “Output”, “Symbol map” và “Validation” của work package đó.
3. Kiểm tra `git status --short`; giữ nguyên thay đổi ngoài scope của user/agent cũ.
4. Chạy baseline nhỏ trước khi sửa.
5. Move một nhóm symbol, thêm facade, chuyển consumer nội bộ.
6. Chạy validation nhỏ sau mỗi nhóm.
7. Chạy validation đầy đủ của work package.
8. Ghi kết quả vào mục “Nhật ký thực hiện” cuối file hoặc báo cáo riêng trong PR.
9. Không tự động bắt đầu work package kế tiếp nếu user chỉ giao một package.

### 2.1 Prompt mẫu giao cho agent

```text
Thực hiện work package <ID> trong next_clean.md.
Chỉ sửa các file thuộc mục Allowed files.
Giữ đường import cũ bằng compatibility facade.
Không đổi thuật toán, checkpoint, YAML hoặc output contract.
Chạy đầy đủ validation của work package và báo cáo file đã đổi, test đã chạy,
parity result và phần việc còn lại.
```

### 2.2 Điều kiện dừng và hỏi lại

Agent phải dừng nếu gặp một trong các trường hợp:

- cần đổi schema checkpoint hoặc state-dict key;
- cần thay tên class được lưu trong full-model checkpoint;
- không xác định được model/dataset còn được dùng hay không;
- phải sửa cả `landmark` và `segment` để hoàn thành một work package;
- baseline đã fail trước khi sửa;
- dependency GPU/ONNX không có và không thể xác minh parity cần thiết;
- phát hiện thay đổi user chưa commit chồng lên file định sửa;
- phải xóa file/model thay vì chỉ tách;
- import cycle chỉ giải được bằng global `sys.path` hoặc monkey patch.

## 3. Baseline và lệnh khám phá chuẩn

### 3.1 Không quét toàn repo

Task landmark:

```bash
rg -n "PATTERN" landmark tests/code -g '*.py' -g '*.yaml'
```

Task segment:

```bash
rg -n "PATTERN" segment tests/code -g '*.py' -g '*.yaml'
```

Không tìm mặc định trong:

```text
legacy/
landmark0/
Ref/
runs/
**/__pycache__/
**/.cache/
```

### 3.2 Baseline chung

```bash
git status --short
python -m compileall -q landmark segment
python -m landmark.train --help
python -m landmark.train_det --help
python -m segment.main --help
```

### 3.3 Baseline landmark

```bash
python -m unittest discover -s landmark/tests -v
python -m unittest discover -s tests/code -p 'test_landmark*.py' -v
python -m unittest discover -s tests/code -p 'test_detection*.py' -v
```

### 3.4 Baseline segment

```bash
python -m unittest discover -s tests/code -p 'test_segment*.py' -v
python -m unittest discover -s tests/code -p 'test_cli_contract.py' -v
```

Test GPU, DDP và ONNX chỉ chạy khi environment có dependency phù hợp. Không ghi
`pass` nếu test bị skip vì thiếu dependency; báo `not run` kèm lý do.

## 4. Kiến trúc compatibility dùng cho mọi file lớn

Không biến ngay file cũ thành thư mục cùng tên. Ví dụ không thể đồng thời có:

```text
landmark/core/metrics.py
landmark/core/metrics/
```

Vì vậy dùng các module sibling có tiền tố và giữ file cũ làm facade:

```text
landmark/core/metrics.py              # path cũ, chỉ re-export
landmark/core/metric_geometry.py      # implementation mới
landmark/core/metric_ap.py
landmark/core/metric_detection.py
landmark/core/metric_pose.py
```

Mẫu facade:

```python
"""Compatibility exports; implementations live in focused modules."""

from .metric_geometry import bbox_ioa, bbox_iou, box_iou, kpt_iou
from .metric_detection import ConfusionMatrix, DetMetrics, Metric
from .metric_pose import PoseMetrics

__all__ = [
    "bbox_ioa",
    "bbox_iou",
    "box_iou",
    "kpt_iou",
    "ConfusionMatrix",
    "DetMetrics",
    "Metric",
    "PoseMetrics",
]
```

Quy tắc facade:

- không chứa implementation mới;
- không dùng wildcard import;
- có `__all__` rõ;
- import module mới không được tạo side effect;
- test cả path cũ và path mới;
- consumer nội bộ mới import trực tiếp module focused;
- path cũ chỉ phục vụ compatibility và external consumer.

Với class có thể nằm trong pickle, xác minh `__module__` và full checkpoint trước
khi move. Nếu cần giữ module path tuyệt đối, để định nghĩa class ở file cũ và chỉ
tách helper trước; không ép move class.

## 5. Dependency order

```text
B0 baseline + contract tests
 |
 +-- L1 core init split
 |    +-- L2 augment split
 |    +-- L3 tasks/checkpoint/parser split
 |         +-- L4 block split
 |         +-- L5 head split
 |    +-- L6 metrics split
 |         +-- L7 loss split
 |    +-- L8 plotting split
 |         +-- L9 results split
 |    +-- L10 secondary landmark files
 |
 +-- S1 segment main split
 +-- S2 RWKV V5 split
 +-- S3 RWKV V6 split
 +-- S4 dataset split
 |    +-- S5 MedSegBench split
 +-- S6 model-zoo classification
 |
 +-- C1 agent scope + SKILL.md
 +-- C2 landmark0 legacy move
```

`landmark` và `segment` branches ở trên độc lập. Có thể làm song song ở các PR khác
nhau nếu không cùng sửa root tests/tài liệu, nhưng không gộp chúng vào một commit.

## 6. B0 — Khóa contract bằng characterization tests

### Mục tiêu

Thêm test cho import path, symbol, checkpoint và output trước khi move code.

### Allowed files

```text
landmark/tests/
tests/code/
info/                         # chỉ khi lưu baseline machine-readable
```

### Test cần bổ sung

Landmark:

- import toàn bộ symbol hiện được `landmark.nn.modules.__all__` expose;
- construct tám YAML public trong `landmark/cfg/models/`;
- class cuối của YOLO YAML đúng `Pose26`, `OA26HeatmapPose`,
  `OA26RegionRefinePose`;
- fixed export shapes: `(B,4,159)`, `(B,)`, `(B,129,3)`;
- `load_checkpoint` qua path cũ;
- `PoseMetrics`, `PoseLoss26`, `Results`, `Keypoints`, `LetterBox` import từ path
  cũ;
- state-dict key snapshot cho model nhỏ đại diện.

Segment:

- registry tuple của V3/V5/V6 không đổi;
- model ID V5=124 và V6=125 không đổi;
- factory V3/V5/V6 construct được;
- checkpoint best không optimizer, last có optimizer;
- ONNX filename/metadata/output contract;
- các helper hiện được test import từ `segment.main` có snapshot trước khi move.

### Validation

Chạy toàn bộ baseline ở mục 3. Không refactor nếu baseline mới chưa pass.

## 7. L1 — Tách `landmark/core/__init__.py`

### Hiện trạng

`landmark/core/__init__.py`: khoảng 1.509 dòng. Nó chứa constants, logging,
serialization, environment detection, settings và concurrency; đồng thời import
Torch/OpenCV và khởi tạo settings khi import.

### Output files

```text
landmark/core/__init__.py          # facade/lazy exports, mục tiêu <150 dòng
landmark/core/constants.py         # path, version, scalar flags
landmark/core/logging.py           # logger và text formatting
landmark/core/serialization.py     # YAML/JSON/export mixins
landmark/core/environment.py       # platform/runtime probes
landmark/core/settings.py          # SettingsManager và path settings
landmark/core/concurrency.py       # decorators/locks/retry
```

### Symbol map

| Module mới | Symbol chuyển |
|---|---|
| `constants.py` | `RANK`, `LOCAL_RANK`, `ROOT`, `ASSETS`, `DEFAULT_CFG_PATH`, `NUM_THREADS`, version/platform constants, `FLOAT_OR_INT`, `STR_OR_PATH` |
| `logging.py` | `plt_settings`, `set_logging`, `LOGGER`, `emojis`, `colorstr`, `remove_colorstr`, `PREFIX` |
| `serialization.py` | `DataExportMixin`, `SimpleClass`, `IterableSimpleNamespace`, `YAML`, `JSONDict` |
| `environment.py` | `read_device_model`, toàn bộ `is_*`, `get_ubuntu_version`, `get_user_config_dir`, `DEVICE_MODEL`, `ONLINE`, `IS_*`, `ENVIRONMENT`, `TESTS_RUNNING` |
| `settings.py` | `SettingsManager`, `SETTINGS_FILE`, `SETTINGS`, `PERSISTENT_CACHE`, `DATASETS_DIR`, `WEIGHTS_DIR`, `RUNS_DIR`, default config load |
| `concurrency.py` | `ThreadingLocked`, `TryExcept`, `Retry`, `threaded` |

Các helper `deprecation_warn`, `clean_url`, `url2file`, `vscode_msg`, `set_sentry`
phải audit consumer. Không chuyển vào module active mới nếu không có consumer.

### Thứ tự thực hiện

1. Tạo `constants.py` không import từ `landmark.core`.
2. Tạo `serialization.py`; tránh phụ thuộc `settings.py`.
3. Tạo `logging.py`; tránh vòng `logging -> core -> logging`.
4. Tạo `environment.py`; không chạy network probe ở import nếu có thể trì hoãn.
5. Tạo `settings.py`; import trực tiếp constants/serialization.
6. Tạo `concurrency.py`.
7. Đổi consumer nội bộ sang import module cụ thể.
8. Làm mỏng `core/__init__.py` thành lazy facade.
9. Chạy import-time test để xác nhận `import landmark` vẫn lazy.

### Không được làm

- không đổi key/default trong `cfg/default.yaml`;
- không đổi vị trí `RUNS_DIR` hoặc cache;
- không gọi network ở import;
- không thay format LOGGER;
- không xóa settings key trong commit này.

### Validation

```bash
python -c "import landmark; assert 'torch' not in __import__('sys').modules"
python -c "from landmark.core import LOGGER, YAML, SETTINGS, RANK"
python -m unittest discover -s landmark/tests -v
python -m unittest discover -s tests/code -p 'test_cli_contract.py' -v
```

Nếu yêu cầu `torch` chưa thể không import do contract hiện tại, ghi baseline và chỉ
yêu cầu refactor không làm import nặng hơn; không sửa bằng hack.

## 8. L2 — Tách `landmark/data/augment.py`

### Hiện trạng

Khoảng 3.133 dòng. Đây là file lớn nhất trong `landmark`.

### Output files

```text
landmark/data/augment.py            # facade cũ
landmark/data/transform_base.py
landmark/data/transform_mix.py
landmark/data/transform_geometry.py
landmark/data/transform_photometric.py
landmark/data/transform_format.py
landmark/data/transform_factory.py
landmark/data/transform_classify.py  # chỉ giữ nếu reachability chứng minh cần
```

### Symbol map

| Module mới | Symbol |
|---|---|
| `transform_base.py` | `BaseTransform`, `Compose` |
| `transform_mix.py` | `BaseMixTransform`, `Mosaic`, `MixUp`, `CutMix`, `CopyPaste` |
| `transform_geometry.py` | `RandomPerspective`, `RandomFlip`, `LetterBox` |
| `transform_photometric.py` | `RandomHSV`, `Albumentations` |
| `transform_format.py` | `Format`, `SemanticFormat` nếu còn dùng |
| `transform_factory.py` | `v8_transforms` và factory pose/detect active |
| `transform_classify.py` | `classify_transforms`, `classify_augmentations`, `ClassifyLetterBox`, `CenterCrop`, `ToTensor` nếu còn consumer |

`LoadVisualPrompt` và `RandomLoadText` là ứng viên ngoài scope. Audit parser/dataset
trước khi quyết định giữ hoặc prune ở commit khác.

### Import constraints

- `transform_base.py` không import dataset.
- `transform_mix.py` có thể phụ thuộc base/Instances nhưng không import factory.
- `transform_factory.py` là lớp composition cuối, được phép import các transform.
- `LetterBox` phải tiếp tục import được từ `landmark.data.augment` vì
  `landmark.core.results` đang dùng path này.

### Validation

- fixed seed cho `LetterBox`, `RandomPerspective`, `Mosaic` nếu active;
- rectangular HxW không bị stretch;
- X-ray policy giữ flip/cutmix/copy-paste/erasing bằng 0 theo contract;
- dataset/model smoke;
- full landmark tests.

## 9. L3 — Tách `landmark/nn/tasks.py`

### Hiện trạng

Khoảng 2.194 dòng, gồm model classes, dynamic YAML parser, checkpoint safe load và
task guessing.

### Output files

```text
landmark/nn/tasks.py                 # compatibility facade
landmark/nn/model_base.py
landmark/nn/model_tasks.py
landmark/nn/model_parser.py
landmark/nn/checkpoint.py
```

### Symbol map

| Module mới | Symbol |
|---|---|
| `model_base.py` | `BaseModel`, common forward/predict/fuse/load |
| `model_tasks.py` | `_initialize_yolo_model`, `DetectionModel`, `PoseModel`, model active khác đã xác nhận |
| `model_parser.py` | `parse_model`, `yaml_model_load`, `guess_model_scale`, `guess_model_task` |
| `checkpoint.py` | `temporary_modules`, `_SafeLoad`, `torch_safe_load`, `load_checkpoint` |

Các class `OBBModel`, `SegmentationModel`, `SemanticSegmentationModel`,
`ClassificationModel`, `RTDETRDetectionModel`, `WorldModel`, `YOLOEModel`,
`YOLOESegModel` là ứng viên prune. Không move chúng vào `model_tasks.py` trước khi
audit model YAML active và compatibility checkpoint.

### Critical compatibility

- `landmark.nn.tasks.load_checkpoint` vẫn tồn tại;
- test đang patch đúng path này, nên facade phải gọi symbol re-export được;
- parser vẫn resolve toàn bộ class trong tám YAML public;
- temporary alias chỉ active trong context load checkpoint;
- không cài alias `ultralytics` global lúc import thường.

### Validation

- construct tám YAML;
- compare state-dict keys và initialized values ở fixed seed;
- load checkpoint cũ;
- `AutoBackend` patch path cũ pass;
- forward/backward V1/V9/HRNet/ViTPose/RTMO theo test hiện có.

## 10. L4 — Tách `landmark/nn/modules/block.py`

### Hiện trạng

Khoảng 2.073 dòng. Parser import block qua `landmark.nn.modules`, do đó sai export
sẽ làm YAML build fail ngay.

### Output files

```text
landmark/nn/modules/block.py          # compatibility facade
landmark/nn/modules/block_common.py
landmark/nn/modules/block_csp.py
landmark/nn/modules/block_attention.py
landmark/nn/modules/block_yolo26.py
landmark/nn/modules/block_external.py  # chỉ nếu class active cần
```

### Symbol map đề xuất

| Module | Symbol chính |
|---|---|
| `block_common.py` | `DFL`, `Proto`, `SPP`, `SPPF`, `Bottleneck`, `GhostBottleneck`, `ResNetBlock`, `ResNetLayer` |
| `block_csp.py` | `C1`, `C2`, `C2f`, `C3`, `C3x`, `C3TR`, `C3Ghost`, `BottleneckCSP`, `RepC3`, `C3f`, `C3k`, `C3k2` |
| `block_attention.py` | `Attention`, `PSABlock`, `PSA`, `C2PSA`, `C2fPSA`, `MaxSigmoidAttnBlock`, `C2fAttn` nếu active |
| `block_yolo26.py` | `RepVGGDW`, `CIB`, `C2fCIB`, `SCDown`, `A2C2f`, `Proto26`, các block YOLO26 YAML dùng |
| `block_external.py` | `TorchVision`, contrastive/world blocks nếu còn consumer |

Trước khi move từng symbol, tìm tên trong:

```bash
rg -n "SYMBOL" landmark/cfg landmark/nn/tasks.py landmark/tests -g '*.yaml' -g '*.py'
```

### Validation

- `landmark.nn.modules.__all__` không mất symbol cần thiết;
- parser build tất cả YAML;
- state-dict parity;
- checkpoint full-model load nếu có fixture;
- không circular import giữa `block_*` và `head.py`/`transformer.py`.

## 11. L5 — Tách `landmark/nn/modules/head.py`

### Hiện trạng

Khoảng 1.877 dòng. Chỉ detect/pose là task public, nhưng file còn head của nhiều
task upstream.

### Output files

```text
landmark/nn/modules/head.py           # compatibility facade
landmark/nn/modules/head_detect.py
landmark/nn/modules/head_pose.py
landmark/nn/modules/head_end2end.py
landmark/nn/modules/head_external.py  # chỉ nếu compatibility cần
```

### Symbol map

| Module | Symbol |
|---|---|
| `head_detect.py` | `Detect` và detect primitives |
| `head_pose.py` | `Pose`, `Pose26` |
| `head_end2end.py` | end-to-end detect helper/head nếu tách được không tạo cycle |
| `head_external.py` | `Segment`, `Segment26`, `OBB`, `OBB26`, `Classify`, `WorldDetect`, `YOLOE*`, `RTDETRDecoder`, `SemanticSegment`, `v10Detect` chỉ khi còn compatibility consumer |

OA26 head trong `nn/modules/oa26/` và region refine trong
`nn/modules/oa26_region_refine/` phải tiếp tục riêng; không merge vào `head_pose.py`.

### Validation

- class cuối của các YAML đúng tên;
- V1/V9 auxiliary gradients;
- detection CLI build và train smoke;
- export contract.

## 12. L6 — Tách `landmark/core/metrics.py`

### Output files

```text
landmark/core/metrics.py              # facade
landmark/core/metric_geometry.py
landmark/core/metric_ap.py
landmark/core/metric_detection.py
landmark/core/metric_pose.py
landmark/core/metric_external.py       # chỉ khi compatibility cần
```

### Symbol map

| Module | Symbol |
|---|---|
| `metric_geometry.py` | `bbox_ioa`, `box_iou`, `bbox_iou`, `mask_iou`, `kpt_iou`, covariance helpers, `probiou`, `batch_probiou`, `smooth_bce` |
| `metric_ap.py` | `smooth`, `plot_pr_curve`, `plot_mc_curve`, `compute_ap`, `ap_per_class` |
| `metric_detection.py` | `ConfusionMatrix`, `Metric`, `DetMetrics` |
| `metric_pose.py` | `PoseMetrics` |
| `metric_external.py` | `SegmentMetrics`, `ClassifyMetrics`, `OBBMetrics`, `SemanticMetrics` nếu checkpoint/import compatibility cần |

`CITYSCAPES_WEIGHT`, `OKS_SIGMA`, `RLE_WEIGHT` đang được loss import; đặt chúng ở
module geometry/constants rõ ràng và re-export từ path cũ.

### Validation

- numerical test IoU/OKS/AP trước và sau;
- detection/pose metrics keys không đổi;
- reporting test;
- loss imports không tạo cycle.

## 13. L7 — Tách `landmark/core/loss.py`

### Output files

```text
landmark/core/loss.py                 # facade
landmark/core/loss_common.py
landmark/core/loss_detection.py
landmark/core/loss_pose.py
landmark/core/loss_oa26.py
landmark/core/loss_region_refine.py
landmark/core/loss_external.py         # chỉ nếu compatibility cần
```

### Symbol map

| Module | Symbol |
|---|---|
| `loss_common.py` | `LandmarkLossCurriculum`, `VarifocalLoss`, `FocalLoss`, `DFLoss`, `BboxLoss`, `RLELoss`, `KeypointLoss` |
| `loss_detection.py` | `v8DetectionLoss`, `E2EDetectLoss`, `E2ELoss`, detect losses active |
| `loss_pose.py` | `v8PoseLoss`, `PoseLoss26` |
| `loss_oa26.py` | `OA26HeatmapPoseLoss`, `OA26SimCCPoseLoss` |
| `loss_region_refine.py` | `OA26RegionRefinePoseLoss` |
| `loss_external.py` | segmentation/classification/OBB/semantic losses nếu còn compatibility consumer |

Inheritance order bắt buộc:

```text
v8DetectionLoss
  -> v8PoseLoss
      -> PoseLoss26
          -> OA26HeatmapPoseLoss
              -> OA26RegionRefinePoseLoss
```

Không tạo import ngược `loss_common -> loss_pose` hoặc
`loss_pose -> loss_region_refine`.

### Validation

- fixed batch loss vector shape và value parity;
- detection-first curriculum zeros đúng indices;
- V1/V9 gradient reachability;
- backward finite;
- checkpoint resume.

## 14. L8 — Tách `landmark/core/plotting.py`

### Output files

```text
landmark/core/plotting.py             # facade
landmark/core/plot_colors.py
landmark/core/plot_annotator.py
landmark/core/plot_detection.py
landmark/core/plot_features.py
landmark/core/plot_pose.py
```

### Symbol map

| Module | Symbol |
|---|---|
| `plot_colors.py` | `Colors`, `colors` |
| `plot_annotator.py` | `Annotator`, `save_one_box` |
| `plot_detection.py` | `plot_labels`, `plot_images`, `plot_results`, `plot_multitrain_results`, tuning/scatter helpers nếu active |
| `plot_features.py` | `feature_visualization` |
| `plot_pose.py` | `_style`, `_read_csv`, `_training_report_title`, resize helper, `plot_dashboard_pose`, `plot_pose_metrics`, `plot_validation_samples` |

`landmark/core/plotting_det.py` tiếp tục riêng. Nó chỉ import primitive từ các
module mới; không merge hai file reporting.

### Validation

- ảnh sample rộng 800 px;
- title/time format không đổi;
- detection plotting test;
- robust y-limit test;
- matplotlib backend không mở GUI.

## 15. L9 — Tách `landmark/core/results.py`

### Output files

```text
landmark/core/results.py              # facade
landmark/core/result_base.py
landmark/core/result_container.py
landmark/core/result_detection.py
landmark/core/result_pose.py
landmark/core/result_external.py       # chỉ nếu compatibility cần
```

### Symbol map

| Module | Symbol |
|---|---|
| `result_base.py` | `BaseTensor` |
| `result_container.py` | `Results` |
| `result_detection.py` | `Boxes` |
| `result_pose.py` | `Keypoints` |
| `result_external.py` | `SemanticMask`, `Masks`, `Probs`, `OBB` nếu compatibility cần |

`landmark/utils/results.py` là MESKO canonical adapter và tiếp tục riêng; không
merge vào core result container.

### Validation

- predict returns `Results` như trước;
- boxes/keypoints fields và device conversions;
- canonical adapter `[129,3]`;
- export wrapper.

## 16. L10 — Landmark secondary backlog

Chỉ bắt đầu sau L1–L9:

| File | Dòng hiện tại | Điểm tách đề xuất |
|---|---:|---|
| `core/trainer.py` | ~1.300 | `trainer_optimizer.py`, `trainer_checkpoint.py`, `trainer_ddp.py`; giữ lifecycle ở file cũ |
| `core/model.py` | ~1.178 | public lifecycle giữ ở file cũ, task dispatch sang `model_dispatch.py` |
| `data/dataset.py` | ~1.143 | base dataset và pose/detect implementation |
| `core/checks.py` | ~1.129 | dependency, file/YAML, image/font, environment checks |
| `core/torch_utils.py` | ~1.088 | device/AMP, EMA, optimizer, model inspection |
| `train_det.py` | ~865 | dataset preflight/reporting helper nếu tiếp tục phình; giữ CLI ở file cũ |

Không tách secondary backlog theo phỏng đoán. Agent phải tạo symbol/consumer map
giống L1–L9 trước khi sửa.

## 17. S1 — Tách `segment/main.py`

### Hiện trạng quan trọng

`segment/cli.py` đã tồn tại và xử lý YAML/argument. **Không tạo CLI parser mới**.
`segment/main.py` đang chứa runtime setup, checkpoint, model load, zero-shot,
validation, train loop và export orchestration.

### Output files

```text
segment/main.py                 # bootstrap GPU sớm + dispatch, mục tiêu <150 dòng
segment/cli.py                  # giữ nguyên vai trò hiện có
segment/core/__init__.py
segment/core/runtime.py
segment/core/checkpoint.py
segment/core/optim.py
segment/core/evaluator.py
segment/core/trainer.py
segment/core/export.py
segment/core/run_dir.py
```

### Exact function map

| Module mới | Hàm hiện có từ `segment/main.py` |
|---|---|
| `runtime.py` | `convert_to_numpy`, `seed_torch`, `_requested_gpu_count`, `_unwrap_model`, `_wrap_data_parallel_if_needed`, `_validate_runtime_config`, `_tensor_stats`, `_raise_non_finite_error` |
| `checkpoint.py` | `_load_model_state_dict`, `_save_checkpoint`, `_compact_existing_best_checkpoint`, `load_model` |
| `optim.py` | `deep_supervision_loss`, `_build_criterion`, `_as_float`, `_build_optimizer` |
| `evaluator.py` | `zero_shot`, `_validate_epoch`, `validate`, `_dataset_class_names` |
| `export.py` | `_should_export_pending_best`, `_export_best_segment_onnx` |
| `run_dir.py` | `_safe_run_component`, `init_dir`, `_read_history` |
| `trainer.py` | `train` |
| `main.py` | `main` và bootstrap cần chạy trước khi import Torch |

`parse_arguments` nếu chỉ wrap `parse_segment_args` thì xóa sau compatibility
audit; không tạo parser thứ hai.

### Critical import order

`CUDA_VISIBLE_DEVICES` phải được đặt trước khi import Torch. Không move bootstrap
vào module bị import sau Torch. Phương án an toàn:

1. `main.py` parse GPU tối thiểu và set environment;
2. sau đó import `segment.core.runtime`/trainer;
3. `segment.cli.parse_segment_args` xử lý config đầy đủ.

### Compatibility tests phải cập nhật

`tests/code/test_segment_reporting.py` hiện import các helper từ `segment.main`:

```text
_compact_existing_best_checkpoint
_export_best_segment_onnx
_save_checkpoint
_should_export_pending_best
```

Trong giai đoạn chuyển đổi:

- test path mới trực tiếp;
- thêm một test facade path cũ;
- `segment.main` re-export bốn helper trên;
- patch target ONNX đổi sang nơi symbol được lookup thực tế, không patch mù.

### Validation

- CLI help trước/sau giống nhau;
- GPU bootstrap test;
- checkpoint reporting tests;
- 1 epoch/small batch smoke;
- best/last semantics;
- ONNX refresh schedule.

## 18. S2 — Tách `RWKV_UNetV5.py`

### Hiện trạng

Khoảng 1.347 dòng. Registry contract:

```python
"RWKV_UNetV5": (
    ".RWKV.RWKV_UNet.RWKV_UNetV5",
    "med_axial_rwkv5_unet",
)
```

Model ID phải giữ `124`.

### Output files

```text
segment/models/RWKV/RWKV_UNet/RWKV_UNetV5.py   # facade
segment/models/RWKV/RWKV_UNet/rwkv_v5/
├── __init__.py
├── shifts.py
├── mixers.py
├── blocks.py
├── encoder.py
├── decoder.py
├── model.py
└── diagnostics.py
```

### Exact symbol map

| Module | Symbol |
|---|---|
| `shifts.py` | `_valid_group_count`, `q_shift_2d`, `LightweightDynamicShift`, `PartialDynamicLerpV6` |
| `mixers.py` | `MatrixStateScan`, `UniRWKV6SequenceMix`, `QuadAxialMatrixRWKV`, `LightweightChannelMix` |
| `blocks.py` | `DropPath` nếu local bắt buộc, `ConvGNAct`, `LayerScale`, `LocalIRBlock`, `MatrixRWKVBottleneckBlock`, `SkipGate` |
| `encoder.py` | `MedRWKV6Encoder` |
| `decoder.py` | `DecoderBlock` |
| `model.py` | `MedAxialRWKV6UNet`, `med_axial_rwkv5_unet` |
| `diagnostics.py` | `count_trainable_parameters`, `check_finite_gradients`, `report_missing_gradients`, `self_test` |

Không đổi các tên nội bộ có chữ RWKV6 trong commit này. Đổi tên sẽ thay pickle
path và làm diff không còn thuần structural.

### Validation

- registry tuple không đổi;
- ID 124;
- factory signature không đổi;
- parameter count, state-dict keys, fixed-seed output;
- backward finite và missing-gradient test;
- ONNX parity nếu V5 nằm trong auto-export support.

## 19. S3 — Tách `RWKV_UNetV6.py`

### Registry contract

```python
"RWKV_UNetV6": (
    ".RWKV.RWKV_UNet.RWKV_UNetV6",
    "rwkv_unet_v6",
)
```

Model ID phải giữ `125`.

### Output files

```text
segment/models/RWKV/RWKV_UNet/RWKV_UNetV6.py   # facade
segment/models/RWKV/RWKV_UNet/rwkv_v6/
├── __init__.py
├── mixers.py
├── blocks.py
├── encoder.py
├── decoder.py
├── model.py
└── diagnostics.py
```

### Exact symbol map

| Module | Symbol |
|---|---|
| `mixers.py` | `DynamicLerpV6`, `RWKV6MatrixStateScan`, `RWKV6SequenceMix`, `AxialRWKV6SpatialMix`, `RWKV6ChannelMix` |
| `blocks.py` | `IRRWKV6Block` |
| `encoder.py` | `RWKV6UNetEncoder` |
| `decoder.py` | `ConvDecoderBlock` |
| `model.py` | `RWKV_UNetV6`, `rwkv_unet_v6` |
| `diagnostics.py` | `count_parameters`, `self_test` |

Không tạo shared implementation V5/V6 trong work package này. Code giống nhau chỉ
được share ở proposal riêng sau khi có numerical parity test cho cả hai model.

### Validation

- registry tuple/ID 125;
- test direct import path cũ;
- parameter/state-dict/output parity;
- backward;
- ONNX metadata và runtime parity.

## 20. S4 — Tách `segment/dataloader/dataset.py`

### Output files

```text
segment/dataloader/dataset.py         # facade
segment/dataloader/datasets/
├── __init__.py
├── generic.py
├── glas.py
├── busbra.py
├── kvasir.py
├── skin.py
├── retinal.py
├── covid_ct.py
├── monuseg.py
└── data_science_bowl.py
```

### Exact symbol map

| Module | Symbol |
|---|---|
| `generic.py` | `_normalize_sample_name`, `_resolve_generic_mask_dir`, `_index_generic_files`, `_resolve_generic_pair`, `MedicalDataSets`, `MedicalDataSetsVal`, `MedicalDataSetsVal_withscale` |
| `glas.py` | `GlasDataSets` |
| `busbra.py` | `BUSBRADatasets` |
| `kvasir.py` | `KvasirSEGDataset`, `KvasirSEGDatagen`, `KvasirSEGDatagenVAL`, `KvasirSEGDatasetVAL` |
| `skin.py` | `PH2Dataset` |
| `retinal.py` | `CHASEDB1Dataset`, `DRIVEdataset` |
| `covid_ct.py` | `Covid19CTScanDataset` |
| `monuseg.py` | `MonuSeg2018Dataset` |
| `data_science_bowl.py` | `DataScienceBowl2018Dataset` |

`dataset_mesko.py`, `dataset_XRay.py`, `dataset_ACDC.py`, `dataset_synapse.py` đã
riêng và phải tiếp tục riêng.

### Consumer cần cập nhật

`segment/dataloader/dataloader.py` đang import một danh sách dài từ
`segment.dataloader.dataset`. Chuyển nó sang import module focused, nhưng giữ
facade để external code không gãy.

### Validation

- import tất cả dataset cũ;
- generic image/mask pairing;
- MESKO path không bị ảnh hưởng;
- worker seeding;
- train/val DataLoader smoke cho dataset available.

## 21. S5 — Tách `segment/utils/medsegbench.py`

### Hiện trạng

Khoảng 1.052 dòng. `INFO` lớn chiếm phần đầu; class `MedSegBench` bắt đầu gần dòng
846, `FlexibleMedSegBench` gần dòng 992.

### Output files

```text
segment/utils/medsegbench.py          # facade cũ
segment/medsegbench/
├── __init__.py
├── registry.py
├── paths.py
├── dataset.py
└── factory.py
```

### Symbol map

| Module | Symbol |
|---|---|
| `paths.py` | `get_default_root` |
| `registry.py` | `HOMEPAGE`, `INFO` |
| `dataset.py` | `MedSegBench`, `FlexibleMedSegBench` |
| `factory.py` | `get_MedSegBench_dataset` |

Không import/download network khi chỉ cần `INFO` để route dataset.

### Consumers

```text
segment/dataloader/dataloader.py
  -> INFO
  -> get_MedSegBench_dataset
```

### Validation

- registry length/keys snapshot;
- factory behavior cho một flag hợp lệ và invalid flag;
- import `INFO` không khởi tạo dataset/download;
- DataLoader routing.

## 22. S6 — Quyết định cho `U_RWKV.py` và model zoo

### `U_RWKV.py`

File khoảng 6.905 dòng, model ID 113. Có hai định nghĩa `q_shift` và nhiều factory
gần/trùng tên. Không được tự động tách trước khi owner xác nhận model còn support.

Nếu inactive:

- giữ nguyên code;
- bỏ khỏi registry active ở proposal riêng;
- chuyển nguyên thư mục sang experimental/legacy sau compatibility review;
- ghi checkpoint ID và cách phục hồi;
- không đọc file trong task V3/V5/V6.

Nếu active, tạo proposal riêng với các module:

```text
cuda_wkv.py
scans.py
mixers.py
wavelet.py
blocks.py
encoders.py
decoders.py
variants.py
model.py
```

Phải tạo test xác định định nghĩa trùng nào đang có hiệu lực trước khi move.

### Model zoo khác

Tạo manifest trạng thái, không refactor hàng loạt:

```yaml
- name: G_CASCADE
  status: experimental       # supported | experimental | broken | archived
  registry_path: segment.models.Hybrid.G_CASCADE.G_CASCADE
  checkpoint_in_use: false
  tested: false
  owner: null
```

Chỉ model `supported` mới được nhận work package tách file. Snapshot third-party
không active nên archive nguyên trạng thay vì sửa và merge thư viện vendored.

## 23. C1 — `AGENTS.md` và `SKILL.md`

Tạo riêng:

```text
landmark/AGENTS.md
landmark/SKILL.md
segment/AGENTS.md
segment/SKILL.md
```

### Landmark instructions

- scope mặc định `landmark/` + landmark tests;
- không đọc segment/legacy;
- schema 129 điểm, path ranges, YAML/checkpoint/export invariants;
- module map sau refactor;
- test ladder.

### Segment instructions

- scope mặc định `segment/` + segment tests;
- chỉ đọc model đúng tên user yêu cầu;
- allowlist V3/V5/V6 và compatibility model đã chốt;
- RGB/letterbox/checkpoint/ONNX invariants;
- test ladder.

`SKILL.md` tối đa khoảng 200 dòng. Chi tiết file map nằm trong `next_clean.md`,
không copy toàn bộ vào skill.

## 24. C2 — Di chuyển `landmark0` sang legacy

Đây là commit thuần move:

```text
landmark0/ -> legacy/landmark0/
```

Không sửa code baseline trong commit này. Cập nhật:

- root README;
- `info/report.yaml`;
- link đến upgrade plan;
- test discovery;
- agent scope.

Xác minh:

```bash
rg -n "landmark0|legacy\.landmark0" landmark segment tests -g '*.py'
```

Active runtime không được import legacy. Compatibility checkpoint alias chỉ tồn
tại trong load context của `landmark`.

## 25. Quy tắc test sau mỗi move symbol

Sau khi move một nhóm, không chờ đến cuối file mới test:

1. compile module mới;
2. import path mới;
3. import path facade cũ;
4. test consumer gần nhất;
5. compare fixed output nếu là model/loss/metric;
6. kiểm tra circular import bằng clean Python process;
7. xem `git diff --check`;
8. tiếp tục nhóm sau.

Ví dụ:

```bash
python -m compileall -q landmark/core
python -c "from landmark.core.metrics import PoseMetrics"
python -c "from landmark.core.metric_pose import PoseMetrics"
python -m unittest landmark.tests.test_reporting -v
git diff --check
```

## 26. Quy tắc checkpoint và pickle

State dict thường chỉ lưu tensor keys, nhưng full checkpoint có thể pickle class
theo `module.ClassName`. Trước khi move class model:

```bash
rg -n "torch.save|torch.load|weights_only|temporary_modules|load_checkpoint" \
  landmark segment tests -g '*.py'
```

Agent phải xác minh:

- best/last checkpoint format;
- `__module__` trước/sau;
- alias path cũ;
- load bằng process sạch không dựa vào module đã import sẵn;
- resume optimizer/scheduler/epoch;
- no global alias leak sau load.

Không thay state-dict key bằng cách đổi tên attribute model trong work package
tách file.

## 27. Quy tắc YAML/dynamic registry

Landmark YAML parser resolve class theo tên, segment registry resolve module/attr
theo string. Vì vậy:

- không đổi tên symbol trong commit move;
- `landmark.nn.modules.__all__` phải export tên YAML cần;
- segment registry tuple V3/V5/V6 giữ nguyên khi facade còn tồn tại;
- model ID không renumber;
- test construct model từ YAML/registry, không chỉ import class.

## 28. Quy tắc giảm token thực tế

Tách file chỉ có tác dụng nếu agent không mở facade và tất cả implementation cùng
lúc. Sau refactor:

- facade có docstring + re-export, không chứa hướng dẫn dài;
- mỗi module mới có responsibility rõ trong tên;
- `AGENTS.md` chỉ route agent đến module liên quan;
- model experimental không nằm trong tìm kiếm mặc định;
- không copy architecture docs vào từng module;
- test log giới hạn theo package trước khi chạy full suite;
- dùng `rg` theo path/symbol, không in toàn file lớn;
- `next_clean.md` chỉ đọc work package được giao.

## 29. Definition of Done cho từng work package

Một work package chỉ được đánh dấu xong khi:

- [ ] Implementation đã rời file lớn theo symbol map.
- [ ] File cũ là facade/dispatcher nhỏ hoặc có lý do giữ logic còn lại.
- [ ] Không wildcard import.
- [ ] Consumer nội bộ import module focused.
- [ ] Public path cũ vẫn hoạt động nếu contract yêu cầu.
- [ ] Không circular import.
- [ ] Compile pass.
- [ ] Unit test vùng sửa pass.
- [ ] Package regression liên quan pass.
- [ ] Model/checkpoint/output parity pass nếu áp dụng.
- [ ] `git diff --check` sạch.
- [ ] Không có file unrelated bị sửa.
- [ ] Tài liệu/module map được cập nhật nếu path đích thay đổi.

## 30. Definition of Done toàn đợt clean

- [ ] `landmark` và `segment` vẫn hoàn toàn riêng.
- [ ] `landmark0` nằm trong legacy và không được scan mặc định.
- [ ] Không file infrastructure active nào trên 1.500 dòng nếu không có waiver.
- [ ] `landmark/core/__init__.py` dưới 150 dòng hoặc lazy tương đương.
- [ ] `landmark/data/augment.py` là facade nhỏ.
- [ ] `landmark/nn/tasks.py` là facade nhỏ.
- [ ] `landmark/core/{metrics,loss,plotting,results}.py` là facade rõ.
- [ ] `segment/main.py` chỉ bootstrap/dispatch.
- [ ] V5/V6 implementation được tách và giữ parity.
- [ ] Dataset/MedSegBench được tách mà routing không đổi.
- [ ] U_RWKV/model zoo có status rõ, không làm phình context mặc định.
- [ ] Mỗi package có hướng dẫn agent riêng.
- [ ] CLI/API/YAML/checkpoint/resume/reporting/ONNX test pass.

## 31. Nhật ký thực hiện

Agent hoàn thành work package thêm một dòng, không ghi log dài:

| Ngày | Work package | Commit/PR | Test chính | Parity | Ghi chú |
|---|---|---|---|---|---|
| — | — | — | — | — | Chưa bắt đầu |

## 32. Thứ tự khuyến nghị bắt đầu

Thứ tự ít rủi ro và tạo hiệu quả context sớm:

1. `B0` — contract tests.
2. `L1` — làm mỏng `landmark/core/__init__.py`.
3. `S1` — làm mỏng `segment/main.py` nhưng giữ GPU bootstrap.
4. `L2` — tách augmentation.
5. `L3` — parser/checkpoint/model classes.
6. `S2` và `S3` — V5/V6 độc lập.
7. `L6` rồi `L7` — metrics trước loss.
8. `L8` rồi `L9` — plotting trước results consumer update.
9. `L4` và `L5` — blocks/heads sau khi characterization đủ mạnh.
10. `S4` rồi `S5` — dataset và MedSegBench.
11. `S6` — model-zoo decision, không tự động refactor.
12. `C1` — cập nhật skill/module map theo cấu trúc thực tế cuối.
13. `C2` — legacy move có thể làm sớm hoặc cuối, nhưng luôn ở commit riêng.

Không giao “làm toàn bộ next_clean.md” cho một agent trong một turn. Giao đúng một
work package để context nhỏ, diff review được và rollback an toàn.
