# Kế hoạch refactor tách module cho `landmark` và `segment`

> Trạng thái: kế hoạch, chưa thực hiện refactor source
>
> Phạm vi audit: repository Uknee ngày 2026-08-13

## 1. Mục tiêu và nguyên tắc bắt buộc

`landmark/` và `segment/` tiếp tục là **hai package độc lập**. Kế hoạch này không
merge hai package, không tạo framework dùng chung lớn và không gom nhiều file vào
một file mới.

Mục tiêu:

1. Tách các file active quá lớn thành module nhỏ theo đúng trách nhiệm.
2. Giúp agent chỉ cần đọc một phần nhỏ source cho mỗi task, giảm token/context.
3. Đưa `landmark0/` vào vùng legacy, không để agent quét mặc định.
4. Giữ nguyên public API, CLI, YAML, checkpoint, training, inference và ONNX.
5. Tạo hướng dẫn `SKILL.md` riêng cho `landmark` và `segment`.

Nguyên tắc thực hiện:

- **Chỉ tách, không merge**: không nhập `landmark` vào `segment`, không nhập
  `segment` vào `landmark`, không tạo `common/` chứa logic của cả hai package.
- Một commit chỉ tách một file hoặc một nhóm phụ thuộc chặt.
- Commit di chuyển code không được đồng thời thay thuật toán.
- Giữ file cũ làm facade/re-export tạm thời nếu import path hoặc pickle path đã
  được phát hành.
- Không tách file chỉ để đạt số dòng đẹp; điểm tách phải theo trách nhiệm và import
  graph.
- Không refactor sâu model research không được dùng. Đánh dấu và loại nó khỏi
  context mặc định hiệu quả hơn việc sửa hàng nghìn dòng không có consumer.
- Không xóa class được resolve động từ YAML hoặc checkpoint chỉ vì `rg` không thấy
  import tĩnh.

## 2. Ngưỡng quyết định tách file

| Kích thước file active | Quyết định |
|---:|---|
| Trên 1.500 dòng | Bắt buộc audit và tách, trừ file generated có checksum |
| 800–1.500 dòng | Tách nếu chứa từ hai trách nhiệm độc lập trở lên |
| 400–800 dòng | Chỉ tách khi import nặng, thay đổi thường xuyên hoặc test khó cô lập |
| Dưới 400 dòng | Giữ nguyên, trừ khi có circular import hoặc contract riêng rõ ràng |

Mục tiêu sau refactor:

- module hạ tầng: khoảng 150–500 dòng;
- module model toán học: tối đa khoảng 800 dòng;
- facade `__init__.py` hoặc file compatibility: dưới 100 dòng;
- agent khám phá ban đầu không cần mở quá 8 file;
- không file active nào trên 1.500 dòng nếu không có lý do ghi trong manifest.

Số dòng không phải Definition of Done duy nhất. Refactor chỉ đạt khi import graph,
checkpoint và test vẫn ổn định.

## 3. Hiện trạng đã đo

### 3.1 `landmark`

| File | Dòng | Vấn đề chính |
|---|---:|---|
| `landmark/data/augment.py` | 3.133 | base, mix, geometry, color, format và classify transform cùng file |
| `landmark/nn/tasks.py` | 2.194 | model class, parser YAML, checkpoint loader và compatibility cùng file |
| `landmark/nn/modules/block.py` | 2.073 | nhiều họ block không liên quan nằm chung |
| `landmark/core/metrics.py` | 1.895 | geometry, AP, detection, pose, segment, classify, OBB |
| `landmark/nn/modules/head.py` | 1.877 | nhiều task head cùng file |
| `landmark/core/loss.py` | 1.761 | common, detection, pose, OA26 và task ngoài scope |
| `landmark/core/plotting.py` | 1.600 | annotator, detect plot và medical pose report |
| `landmark/core/results.py` | 1.591 | container, tensor wrapper và kết quả mọi task |
| `landmark/core/__init__.py` | 1.509 | logging, environment, YAML, settings, helper và side effect import |
| `landmark/core/trainer.py` | 1.300 | lifecycle, optimizer, checkpoint và DDP orchestration |
| `landmark/core/model.py` | 1.178 | public lifecycle và task dispatch |
| `landmark/data/dataset.py` | 1.143 | dataset base và task-specific behavior |
| `landmark/core/checks.py` | 1.129 | environment, file, dependency, font và model checks |
| `landmark/core/torch_utils.py` | 1.088 | device, AMP, EMA, optimizer và model inspection |

`landmark` đã thay thế phần lớn chức năng active của `landmark0`, vì vậy đây là
refactor tách module trên implementation hiện tại, không copy code ngược từ
`landmark0/_vendor`.

### 3.2 `segment`

Các file active hoặc gần active cần ưu tiên:

| File | Dòng | Quyết định đề xuất |
|---|---:|---|
| `segment/main.py` | 964 | tách train, validate, checkpoint, runtime; giữ main làm dispatcher |
| `segment/models/RWKV/RWKV_UNet/RWKV_UNetV5.py` | 1.347 | tách mixer, block, encoder/decoder, model |
| `segment/models/RWKV/RWKV_UNet/RWKV_UNetV6.py` | 985 | tách block/mixer và model nếu V6 tiếp tục active |
| `segment/utils/medsegbench.py` | 1.052 | tách registry, download và dataset adapters |
| `segment/dataloader/dataset.py` | 854 | tách mỗi nhóm dataset thành file riêng |
| `segment/tools/export_deploy_model.py` | 465 | chưa bắt buộc; chỉ tách nếu tiếp tục phình |
| `segment/utils/segment_reporting.py` | 450 | giữ nếu API cohesive; có thể tách rendering khỏi evaluator |

File rất lớn nhưng cần decision gate trước khi sửa:

| File | Dòng | Lý do chưa tách ngay |
|---|---:|---|
| `segment/models/RWKV/U_RWKV/U_RWKV.py` | 6.905 | research model lớn, có hàm trùng tên và nhiều variant; không thuộc scope RWKV V3/V5/V6 hiện tại |
| `segment/models/Hybrid/G_CASCADE/models_timm/efficientnet.py` | 2.403 | vendored model-zoo code, không nên refactor nếu model không active |
| `segment/models/Transformer/MedT/model_codes.py` | 2.323 | research implementation độc lập |
| `segment/models/Mamba/MUCM_Net/MUCM_Net.py` | 2.023 | research implementation độc lập |
| `segment/models/Hybrid/{MERIT,G_CASCADE}/...` | 1.500–1.900 | nhiều snapshot `timm` trùng nhau, không thuộc runtime trọng tâm |

Với các file research trên, chọn một trong hai:

- nếu model là supported: tách theo kế hoạch và bổ sung test;
- nếu model không supported: giữ nguyên trong `segment/experimental/` hoặc archive,
  bỏ khỏi registry/context mặc định; không tốn công refactor sâu.

## 4. Cấu trúc đích — hai package vẫn riêng

```text
Uknee/
├── landmark/
│   ├── AGENTS.md
│   ├── SKILL.md
│   ├── cfg/
│   ├── core/
│   ├── data/
│   ├── models/
│   ├── nn/
│   ├── utils/
│   └── tests/
├── segment/
│   ├── AGENTS.md
│   ├── SKILL.md
│   ├── cfg/
│   ├── core/
│   ├── dataloader/
│   ├── models/
│   ├── deploy/
│   ├── tools/
│   ├── utils/
│   └── tests/
└── legacy/
    └── landmark0/
```

Không đổi tên `segment/dataloader/` trong đợt tách file đầu tiên. Đổi tên thư mục
cùng lúc với tách logic làm diff lớn và khó xác minh.

`SKILL.md` phải viết hoa đúng chuẩn. Mỗi package có file riêng; agent sửa landmark
không cần nạp skill của segment và ngược lại.

## 5. Phase 0 — Khóa contract trước khi tách

### 5.1 Contract của `landmark`

Phải giữ nguyên:

- `from landmark import KneePose`;
- `python -m landmark.train` và `python -m landmark.train_det`;
- model YAML public và symbol được parser resolve;
- bốn vùng `femur/tibia/fibula/patella`;
- padded `[51,3]`, số điểm thật `45/51/24/9`, canonical 129 điểm;
- checkpoint cũ, resume, state-dict key và alias unpickle;
- output TorchScript/ONNX hiện tại.

### 5.2 Contract của `segment`

Phải giữ nguyên:

- `python -m segment.main` và script `uknee-segment`;
- registry/build_model cho model supported;
- RWKV V3/V5/V6 theo scope hiện tại;
- compatibility của checkpoint V5/V6 và model ID hiện có;
- RGB, float32, letterbox, output logits/mask;
- `args.yaml`, `summary.yaml`, `results.csv`, dashboard và ONNX metadata.

### 5.3 Baseline

Trước mỗi phase, lưu:

- danh sách import path public;
- model parameter count và state-dict keys;
- output shape với fixed seed;
- unit test và thời gian chạy;
- checkpoint load/resume smoke;
- ONNX output name/shape/parity nếu module liên quan export.

Các lệnh tối thiểu:

```bash
python -m compileall -q landmark segment
python -m unittest discover -s landmark/tests -v
python -m unittest discover -s tests/code -v
python -m landmark.train --help
python -m landmark.train_det --help
python -m segment.main --help
```

## 6. Phase 1 — Đưa `landmark0` vào legacy

Dùng một commit chỉ di chuyển:

```text
landmark0/ -> legacy/landmark0/
```

Yêu cầu:

- dùng `git mv` để giữ history;
- không sửa implementation bên trong `landmark0` trong commit di chuyển;
- cập nhật link tài liệu và `info/report.yaml`;
- `pyproject.toml` tiếp tục chỉ package `landmark` và `segment`;
- test mặc định không discover `legacy/`;
- active source không import `legacy.landmark0`;
- chỉ đọc legacy khi làm parity hoặc migrate checkpoint.

File kế hoạch này sẽ đi theo đường dẫn mới tương ứng trong legacy. Runtime active
không phụ thuộc vào vị trí của tài liệu.

Lưu ý: di chuyển thư mục không tự giảm token nếu agent vẫn quét toàn repository.
Vì vậy cần `AGENTS.md`/`SKILL.md` giới hạn path tìm kiếm như Phase 2.

## 7. Phase 2 — Scope và skill để giảm token

### 7.1 `landmark/AGENTS.md`

Chỉ chứa các rule ngắn:

- mặc định tìm trong `landmark/` và test landmark;
- không mở `segment/` hoặc `legacy/` nếu task không yêu cầu;
- xem entrypoint/import trước khi mở file model lớn;
- giữ schema/checkpoint/export invariants;
- chạy test liên quan nhỏ trước full suite.

### 7.2 `segment/AGENTS.md`

Chỉ chứa:

- mặc định tìm trong `segment/` và test segment;
- model active là allowlist đã chốt, không đọc toàn model zoo;
- không mở `landmark/` hoặc `legacy/` nếu task không yêu cầu;
- giữ preprocessing/checkpoint/ONNX contract;
- model experimental chỉ đọc khi user gọi đúng tên.

### 7.3 `landmark/SKILL.md`

Nội dung tối đa khoảng 150–200 dòng:

- metadata trigger cho task dưới `landmark/`;
- module map: CLI, data, model parser, loss, metrics, plotting, export;
- invariants 129 landmark và checkpoint;
- lệnh `rg` có scope;
- test ladder;
- cấm đọc legacy theo mặc định.

### 7.4 `segment/SKILL.md`

Nội dung tối đa khoảng 150–200 dòng:

- metadata trigger cho task dưới `segment/`;
- danh sách model supported từ một manifest ngắn;
- module map: CLI, dataloader, train, checkpoint, reporting, export;
- preprocessing/output contract;
- test ladder CPU/GPU/ONNX;
- cấm quét model zoo theo mặc định.

Không copy toàn bộ README/kiến trúc vào skill. Chi tiết dài đặt trong
`references/` và chỉ đọc khi task cần.

## 8. Phase 3 — Tách file lớn trong `landmark`

### 8.1 `core/__init__.py` — ưu tiên số 1

Tách thành:

```text
landmark/core/
├── __init__.py          # facade và lazy re-export, dưới 100 dòng
├── logging.py           # LOGGER, logging setup, color output
├── environment.py       # is_colab, is_docker, is_online, platform checks
├── serialization.py     # YAML, JSONDict, DataExportMixin
├── settings.py          # paths, SettingsManager, defaults
└── concurrency.py       # Retry, ThreadingLocked, threaded
```

Không khởi tạo Torch/network/filesystem nặng chỉ vì `import landmark.core`.

Giữ re-export tên cũ trong `core/__init__.py` để các import hiện tại không gãy.
Sau khi consumer chuyển hết sang module cụ thể mới xóa re-export không cần thiết.

### 8.2 `data/augment.py`

Tách theo pipeline:

```text
landmark/data/transforms/
├── __init__.py
├── base.py          # BaseTransform, Compose
├── mix.py           # BaseMixTransform, Mosaic, MixUp, CutMix, CopyPaste
├── geometry.py      # RandomPerspective, RandomFlip, LetterBox
├── photometric.py   # RandomHSV, Albumentations
├── format.py        # Format và tensor conversion
└── factory.py       # pose/detect transform builders
```

Các transform classify/visual-prompt không thuộc public scope phải qua reachability
audit. Nếu không dùng bởi YAML, CLI, checkpoint hoặc test thì xóa; nếu còn dùng
thì chuyển sang file riêng, không nhét vào pose factory.

Giữ `landmark/data/augment.py` làm facade re-export trong một release để tránh phá
import từ dataset và checkpoint.

### 8.3 `nn/tasks.py`

Tách thành:

```text
landmark/nn/
├── tasks.py             # compatibility facade
├── model_base.py        # BaseModel và common forward/fuse/load
├── model_tasks.py       # DetectionModel, PoseModel và model active
├── parser.py            # parse_model, yaml_model_load, task/scale inference
└── checkpoint.py        # safe load, temporary aliases, load_checkpoint
```

Các model OBB, segmentation, classification, world, YOLOE và RT-DETR chỉ giữ nếu
reachability test chứng minh runtime landmark cần. Nếu không cần, prune ở commit
riêng sau khi tách, không xóa cùng commit move.

`temporary_modules`, `_SafeLoad` và alias pickle phải nằm trong `checkpoint.py`;
không đặt alias global khi import package.

### 8.4 `nn/modules/block.py`

Tách theo họ block:

```text
landmark/nn/modules/blocks/
├── common.py        # DFL, SPP, SPPF, Bottleneck
├── csp.py           # C1/C2/C2f/C3/C3k2 và CSP variants
├── attention.py     # PSA/attention blocks thực sự dùng
├── yolo26.py        # block riêng của YAML YOLO26 active
└── compatibility.py # alias cần cho checkpoint cũ
```

Không chuyển class một cách tùy ý: trước mỗi nhóm phải liệt kê YAML nào resolve
class đó. `block.py` tiếp tục re-export class để full-model checkpoint có đường
phục hồi.

### 8.5 `nn/modules/head.py`

Tách thành:

```text
landmark/nn/modules/heads/
├── detect.py
├── pose.py
├── end2end.py
└── compatibility.py
```

OA26 head đã có thư mục riêng thì giữ riêng; không merge ngược vào `head.py`.
Head của task ngoài detect/pose chỉ prune sau reachability audit.

### 8.6 `core/metrics.py`

Tách thành:

```text
landmark/core/metrics/
├── geometry.py      # IoU, OKS, covariance helpers
├── ap.py            # PR curve, AP calculation
├── detection.py     # ConfusionMatrix, Metric, DetMetrics
├── pose.py          # PoseMetrics
└── compatibility.py
```

Không đưa medical reporting vào `metrics.py`; phần MESKO-specific tiếp tục ở
`landmark/utils/validation.py` hoặc module medical riêng.

Segment/Classify/OBB/Semantic metrics không dùng phải được đánh dấu ứng viên prune,
không chuyển sang module active mới nếu không có consumer.

### 8.7 `core/loss.py`

Tách theo inheritance chain:

```text
landmark/core/losses/
├── common.py        # Focal, DFL, bbox, keypoint primitives
├── detection.py     # detection/end-to-end losses
├── pose.py          # v8PoseLoss, PoseLoss26
├── oa26.py          # OA26 heatmap và SimCC
└── region_refine.py # OA26RegionRefinePoseLoss
```

Không merge OA26 và region-refine thành một file. Giữ inheritance và tên class.
Loss segmentation/classification/OBB chỉ chuyển nếu active; nếu không, prune ở
commit riêng.

### 8.8 `core/plotting.py`

Tách thành:

```text
landmark/core/plotting/
├── colors.py
├── annotator.py
├── detection.py
├── features.py
└── pose_reporting.py
```

`pose_reporting.py` chứa dashboard, metric plot và validation samples. Không merge
với `plotting_det.py`; sau tách, hai module chỉ dùng chung primitive màu/annotator.

### 8.9 `core/results.py`

Tách thành:

```text
landmark/core/results/
├── base.py          # BaseTensor
├── containers.py    # Results
├── detection.py     # Boxes
├── pose.py          # Keypoints
└── compatibility.py
```

Masks/Probs/OBB/SemanticMask chỉ giữ nếu public runtime còn consumer. Kết quả
adapter 129 điểm ở `landmark/utils/results.py` vẫn riêng, không merge vào core.

### 8.10 Các file 1.000–1.500 dòng còn lại

Chỉ làm sau nhóm trên:

- `core/trainer.py`: tách optimizer, checkpoint, DDP orchestration;
- `core/model.py`: tách task dispatch khỏi lifecycle public;
- `data/dataset.py`: tách base dataset và pose/detect dataset;
- `core/checks.py`: tách dependency/file/environment checks;
- `core/torch_utils.py`: tách device/AMP, EMA, optimizer, inspection.

Mỗi file phải có import-cycle test trước khi chọn điểm tách.

## 9. Phase 4 — Tách file lớn trong `segment`

### 9.1 `segment/main.py`

Giữ `main.py` dưới 120 dòng, chỉ parse và dispatch:

```text
segment/
├── main.py
├── cli.py
└── core/
    ├── runtime.py       # seed, GPU count, validation config, model wrapping
    ├── checkpoint.py    # save/load/resume/compact best
    ├── optim.py         # criterion, optimizer, deep-supervision loss
    ├── evaluator.py     # validate và zero-shot
    ├── trainer.py       # epoch/batch train loop
    └── export.py        # orchestration gọi ONNX utility hiện có
```

Không merge `segment` core với `landmark/core`. Hai bên có checkpoint/trainer riêng
vì contract và model output khác nhau.

Thứ tự tách an toàn:

1. runtime helpers thuần;
2. checkpoint helpers;
3. optimizer/loss builder;
4. evaluator;
5. train loop;
6. export orchestration;
7. làm mỏng `main.py`.

### 9.2 `RWKV_UNetV5.py`

Tách thành package riêng nhưng giữ file cũ làm facade:

```text
segment/models/RWKV/RWKV_UNet/rwkv_v5/
├── shifts.py       # q_shift_2d, dynamic shift/lerp
├── mixers.py       # sequence/spatial/channel mix
├── blocks.py       # local IR và bottleneck block
├── encoder.py
├── decoder.py
├── model.py        # MedAxialRWKV6UNet + factory
└── diagnostics.py  # parameter/gradient checks, self_test
```

`RWKV_UNetV5.py` re-export `med_axial_rwkv5_unet` và tên class cũ. Registry path
giữ nguyên trong giai đoạn compatibility.

Không đổi tên V5/V6 trong cùng commit tách file, dù tên class nội bộ V5 còn chữ
RWKV6. Việc chuẩn hóa tên là migration riêng có checkpoint mapping.

### 9.3 `RWKV_UNetV6.py`

Tách tương tự nhưng **không dùng chung module với V5 chỉ để giảm dòng**:

```text
segment/models/RWKV/RWKV_UNet/rwkv_v6/
├── mixers.py
├── blocks.py
├── encoder.py
├── decoder.py
├── model.py
└── diagnostics.py
```

Nếu V5 và V6 có đoạn giống nhau, chỉ tạo shared primitive sau khi có parity test
chứng minh semantics thực sự giống. Mặc định giữ riêng để tránh coupling hai model.

### 9.4 `U_RWKV.py` — decision gate

Không tách 6.905 dòng trước khi xác nhận model `U_RWKV` còn support.

Nếu không support:

- bỏ khỏi registry active;
- chuyển nguyên thư mục sang `segment/experimental/RWKV/U_RWKV/` hoặc archive;
- thêm manifest nói rõ dependency/checkpoint/status;
- agent không đọc theo mặc định.

Nếu support:

```text
U_RWKV/
├── cuda_wkv.py
├── scans.py
├── mixers.py
├── wavelet.py
├── blocks.py
├── encoders.py
├── decoders.py
├── variants.py
└── model.py
```

Phải xử lý hai định nghĩa `q_shift` và các factory trùng tên bằng test trước khi
tách; không âm thầm chọn một bản.

### 9.5 `dataloader/dataset.py`

Tách dataset độc lập:

```text
segment/dataloader/datasets/
├── common.py
├── medical.py
├── glas.py
├── busbra.py
├── kvasir.py
├── retinal.py
├── covid_ct.py
├── monuseg.py
└── data_science_bowl.py
```

`segment/dataloader/dataset.py` re-export tên cũ trong giai đoạn compatibility.
Không merge MESKO/XRay vào file này; các dataset đang có file riêng tiếp tục riêng.

Dataset không support cần decision gate giống model, tránh duy trì facade vô hạn.

### 9.6 `utils/medsegbench.py`

Tách thành:

```text
segment/medsegbench/
├── registry.py
├── download.py
├── datasets.py
└── __init__.py
```

Không import downloader/network khi chỉ cần đọc registry. Giữ shim
`segment.utils.medsegbench` đến khi consumer được chuyển hết.

### 9.7 Model zoo khác

Không tách hàng loạt `G_CASCADE`, `MERIT`, `MedT`, `Mamba` trong phase active.
Trước tiên tạo manifest:

```yaml
name: G_CASCADE
status: experimental  # supported | experimental | broken | archived
entrypoint: segment.models.Hybrid.G_CASCADE.G_CASCADE
checkpoint_in_use: false
tests: []
```

Chỉ model `supported` mới được đưa vào backlog tách file. Model experimental nằm
trong package nhưng bị loại khỏi context mặc định; model archived chuyển nguyên
trạng, không merge hoặc chỉnh nội bộ.

## 10. Kiểm soát import và compatibility

Mẫu migration cho mọi file lớn:

1. Viết characterization test cho symbol public.
2. Tạo module mới và move một nhóm class/hàm.
3. Re-export từ path cũ.
4. Chuyển consumer nội bộ sang import module mới trực tiếp.
5. Kiểm tra không có circular import.
6. Load checkpoint cũ và so state-dict/output.
7. Chỉ xóa facade sau một release hoặc giữ vĩnh viễn nếu pickle yêu cầu.

Các check bắt buộc:

```bash
rg -n "from landmark.core import" landmark
rg -n "from landmark.data.augment import" landmark
rg -n "from landmark.nn.tasks import" landmark
rg -n "from segment.main import" segment tests
rg -n "RWKV_UNetV[56]" segment tests
```

Không dùng wildcard import trong module mới. Không đặt logic mới vào `__init__.py`.

## 11. Test ladder

### Level 1 — mỗi commit

```bash
python -m compileall -q landmark segment
python -m landmark.train --help
python -m landmark.train_det --help
python -m segment.main --help
```

### Level 2 — unit theo module

- import path cũ và mới cùng hoạt động;
- class/factory identity hoặc behavior tương đương;
- fixed-seed forward output không đổi;
- không có import active sang legacy.

### Level 3 — package regression

```bash
python -m unittest discover -s landmark/tests -v
python -m unittest discover -s tests/code -v
```

### Level 4 — checkpoint/model

- build tất cả YAML landmark public;
- forward/backward smoke cho landmark model bị ảnh hưởng;
- build và backward RWKV V3/V5/V6;
- load/resume checkpoint cũ;
- so state-dict key, parameter count và output shape.

### Level 5 — export/training

- TorchScript/ONNX checker và runtime parity;
- overfit 8–16 ảnh;
- DDP smoke nếu thay trainer/checkpoint/device;
- xác nhận artifact tree và `summary.yaml` không đổi.

## 12. Thứ tự commit đề xuất

1. `docs: record module split plan and baselines`
2. `chore: move landmark0 to legacy without code changes`
3. `docs: add separate landmark and segment agent scopes`
4. `refactor(landmark): split core initialization`
5. `refactor(landmark): split data transforms`
6. `refactor(landmark): split model parser and checkpoint loading`
7. `refactor(landmark): split blocks and heads`
8. `refactor(landmark): split metrics losses plotting and results`
9. `refactor(segment): extract runtime and checkpoint from main`
10. `refactor(segment): extract trainer evaluator and export orchestration`
11. `refactor(segment): split RWKV V5 implementation`
12. `refactor(segment): split RWKV V6 implementation`
13. `refactor(segment): split datasets and medsegbench`
14. `chore: classify experimental model zoo`
15. `docs: add focused SKILL.md files`

Không gộp nhiều mục trên vào một commit lớn.

## 13. Definition of Done

Refactor hoàn thành khi:

- `landmark` và `segment` vẫn độc lập, không import chéo logic nội bộ;
- `landmark0` nằm trong legacy và không được quét/test mặc định;
- file active trên 1.500 dòng đã được tách hoặc có waiver rõ;
- `landmark/core/__init__.py`, `landmark/data/augment.py`,
  `landmark/nn/tasks.py`, `segment/main.py` chỉ còn facade/dispatcher nhỏ;
- RWKV V5/V6 giữ nguyên registry, checkpoint và numerical behavior;
- model research lớn không làm phình context mặc định;
- mỗi package có `AGENTS.md` và `SKILL.md` riêng;
- task landmark không cần đọc segment và task segment không cần đọc landmark;
- public API, CLI, YAML, checkpoint, resume, reporting và ONNX đều pass test;
- rollback được bằng revert từng commit, không cần import hack.

## 14. Ngoài phạm vi

- Merge `landmark` với `segment`.
- Thiết kế model/loss mới.
- Đổi dataset split hoặc class/landmark schema.
- Đổi accuracy target hoặc training recipe.
- Đổi tên hàng loạt module trong cùng commit tách code.
- Refactor toàn bộ third-party model zoo khi chưa có consumer và test.
