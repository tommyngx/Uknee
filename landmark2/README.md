# landmark2

`landmark2` is the side-by-side, pose-only successor to `landmark`. It embeds
the required YOLO26/OA26 implementation directly under `core`, `data`, and
`nn`; it does not bootstrap `_vendor` and does not require the `ultralytics`
PyPI package.

## Public API

```python
from landmark2 import KneePose

model = KneePose("landmark2/cfg/models/yolo26-pose-v9.yaml")
model.train(data="landmark2/cfg/datasets/mesko4gf2.yaml", epochs=100, device=0)
results = model.predict("image.png")
artifact = model.export(format="onnx", imgsz=640)
```

CLI:

```bash
python -m landmark2.train \
  --model landmark2/cfg/models/yolo26-pose-v9.yaml \
  --data landmark2/cfg/datasets/mesko4gf2.yaml \
  --epochs 100 --imgsz 640 --batch 16 --device 0
```

The five retained configs are base YOLO26 Pose, OA26 V1, OA26 V9,
HRNet-W32, and ViTPose-S. Export is intentionally limited to TorchScript and
ONNX. Unsupported tasks, remote tracking integrations, and deployment
backends fail explicitly instead of importing optional Ultralytics modules.

Run the regression suite with:

```bash
python -m unittest discover -s landmark2/tests -v
```

See the repository-level `report.yaml` for measured structural, parity, loss,
dataset, and export differences against `landmark`.
