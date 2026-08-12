from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml

from landmark.core.exporter import Exporter, onnx_sha256, read_onnx_metadata
from landmark.core.plotting_det import (
    plot_dashboard_detection,
    plot_detection_metrics,
    plot_detection_validation_samples,
)


SAMPLE_EPOCHS = 3
SAMPLE_SEED = 2006
RESULT_COLUMNS = [
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def load_real_kneelocation_samples(
    dataset_root: Path, count: int = 4, seed: int = SAMPLE_SEED
) -> list[dict]:
    """Load four deterministic real X-rays and matching bounding box annotations."""
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        paths = sorted(image_dir.glob("*.jpg"))
    if not paths or not label_dir.is_dir():
        return []
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(paths), size=min(count, len(paths)), replace=False).tolist())
    records = []
    names = {0: "RightKnee", 1: "LeftKnee"}
    for path in (paths[index] for index in chosen):
        image_bgr = cv2.imread(str(path))
        label_path = label_dir / f"{path.stem}.txt"
        if image_bgr is None or not label_path.is_file():
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        boxes, classes, scores = [], [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            x_c, y_c, w, h = [float(v) for v in parts[1:5]]
            x1 = (x_c - w / 2) * width
            y1 = (y_c - h / 2) * height
            x2 = (x_c + w / 2) * width
            y2 = (y_c + h / 2) * height
            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)
            scores.append(0.95 if cls_id == 0 else 0.92)
        records.append({
            "path": path.name,
            "image": image,
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "classes": classes,
            "scores": scores,
            "names": names,
            "map50": 0.985,
            "map5095": 0.885,
        })
    return records


def _history_rows() -> list[dict]:
    rows = []
    for epoch in range(1, SAMPLE_EPOCHS + 1):
        row = {
            "epoch": epoch,
            "train/box_loss": 0.85 / epoch + 0.10,
            "train/cls_loss": 0.65 / epoch + 0.08,
            "train/dfl_loss": 0.55 / epoch + 0.05,
            "val/box_loss": 0.90 / epoch + 0.12,
            "val/cls_loss": 0.70 / epoch + 0.09,
            "val/dfl_loss": 0.60 / epoch + 0.06,
            "metrics/precision(B)": min(0.75 + 0.07 * epoch, 0.96),
            "metrics/recall(B)": min(0.72 + 0.08 * epoch, 0.95),
            "metrics/mAP50(B)": min(0.80 + 0.06 * epoch, 0.985),
            "metrics/mAP50-95(B)": min(0.65 + 0.075 * epoch, 0.885),
        }
        rows.append({key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()})
    return rows


class _ToyDetectExport(torch.nn.Module):
    """Small valid graph matching the YOLO detection deployment contract."""

    yaml_file = "yolo26-detect-sample.yaml"

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images):
        base = images[:, 0, 0, 0] * 0.0 + self.anchor * 0.0
        detections = base[:, None, None].expand(-1, 6, 8400).contiguous()
        num_detections = base.to(torch.int64)
        canonical = base[:, None, None].expand(-1, 2, 4).contiguous()
        return detections, num_detections, canonical


def generate_sample_output(output_dir: Path | None = None) -> Path:
    """Generate one internally consistent detection run for manual inspection."""
    output_dir = output_dir or REPO_ROOT / "tests" / "sample_detection_landmark"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    samples_dir = output_dir / "samples"
    weights_dir = output_dir / "weights"
    samples_dir.mkdir(parents=True)
    weights_dir.mkdir()

    rows = _history_rows()
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    records = load_real_kneelocation_samples(REPO_ROOT / "Ref" / "KneeLocationV6")
    if len(records) < 4:
        # Fallback to MESKO images with synthetic bounding boxes if KneeLocationV6 not available
        from tests.code.test_landmark_sample_plotting import load_real_mesko_samples
        mesko_records = load_real_mesko_samples(REPO_ROOT / "Ref" / "yolo_mesko4GF2")
        records = []
        for rec in mesko_records:
            h, w = rec["image"].shape[:2]
            records.append({
                "path": rec["path"],
                "image": rec["image"],
                "boxes": np.array([[0.1 * w, 0.2 * h, 0.45 * w, 0.7 * h], [0.5 * w, 0.2 * h, 0.88 * w, 0.7 * h]], dtype=np.float32),
                "classes": [0, 1],
                "scores": [0.96, 0.93],
                "names": {0: "RightKnee", 1: "LeftKnee"},
                "map50": 0.985,
                "map5095": 0.885,
            })

    for epoch, row in enumerate(rows, start=1):
        epoch_records = [
            {
                **record,
                "map50": row["metrics/mAP50(B)"],
                "map5095": row["metrics/mAP50-95(B)"],
            }
            for record in records
        ]
        plot_detection_validation_samples(
            epoch_records,
            samples_dir / f"detection_sample_e{epoch}.png",
            epoch=epoch,
        )

    plot_dashboard_detection(
        output_dir / "results.csv",
        output_dir / "detection_dashboard.png",
        model_name="yolo26-detect-sample",
        elapsed_seconds=36.0,
    )
    metrics = SimpleNamespace(
        confusion_matrix=SimpleNamespace(matrix=np.array([[96, 2, 1], [1, 95, 2], [1, 1, 98]])),
    )
    plot_detection_metrics(
        metrics,
        output_dir / "detection_metrics.png",
        model_name="yolo26-detect-sample",
        elapsed_seconds=36.0,
        epochs_completed=SAMPLE_EPOCHS,
        class_names=["RightKnee", "LeftKnee"],
    )

    args = {
        "model": "yolo26-detect.yaml",
        "data": "Ref/KneeLocationV6/data.yaml",
        "epochs": SAMPLE_EPOCHS,
        "imgsz": [640, 640],
        "batch": 16,
        "lr0": 0.001,
        "seed": SAMPLE_SEED,
        "auto_export_onnx": True,
        "fliplr": 0.0,
        "flipud": 0.0,
        "mosaic": 0.5,
    }
    (output_dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), encoding="utf-8")
    checkpoint = {"epoch": SAMPLE_EPOCHS, "model": "sample-only", "metrics": rows[-1]}
    torch.save(checkpoint, weights_dir / "last.pt")
    torch.save({**checkpoint, "epoch": rows[-1]["epoch"]}, weights_dir / "best.pt")
    onnx_path = weights_dir / "yolo26_detect_sample.onnx"
    Exporter(
        {
            "format": "onnx",
            "imgsz": [640, 640],
            "batch": 1,
            "dynamic": True,
            "path": onnx_path,
            "model_name": "yolo26-detect-sample",
            "source_checkpoint": "weights/best.pt",
        }
    )(_ToyDetectExport().eval())
    onnx_metadata = read_onnx_metadata(onnx_path)
    onnx_record = {
        "status": "ready",
        "path": onnx_path.relative_to(output_dir).as_posix(),
        "format": "onnx",
        "sha256": onnx_sha256(onnx_path),
        "file_size_bytes": onnx_path.stat().st_size,
        "metadata": onnx_metadata,
        "parity": json.loads(onnx_metadata.get("uknee.parity", "{}")),
    }

    sample_paths = [record["path"] for record in records]
    preprocess = {
        "schema_version": 1,
        "source_spatial_shape": "dynamic",
        "network_input_shape": [1, 3, 640, 640],
        "layout": "NCHW",
        "dtype": "float32",
        "color_space": "RGB",
        "value_range": [0.0, 1.0],
        "normalization": {"mode": "scale_0_1", "mean": [0.0] * 3, "std": [1.0] * 3},
        "resize": {
            "mode": "letterbox",
            "keep_aspect_ratio": True,
            "target_height": 640,
            "target_width": 640,
            "pad_value": 114,
            "placement": "center",
            "stride": 32,
        },
    }
    summary = {
        "schema_version": 2,
        "task": "knee_detection",
        "model": {
            "name": "yolo26-detect-sample",
            "source": args["model"],
            "parameters": 3_010_000,
            "trainable_parameters": 3_010_000,
            "gflops": 8.12,
            "gflops_convention": "2 x MACs for one forward pass",
            "input_shape": [1, 3, 640, 640],
        },
        "dataset": {"config": args["data"], "classes": {0: "RightKnee", 1: "LeftKnee"}},
        "preprocessing": preprocess,
        "training": {
            "epochs_requested": SAMPLE_EPOCHS,
            "epochs_completed": SAMPLE_EPOCHS,
            "batch_size": 16,
            "seed": SAMPLE_SEED,
            "optimizer": "AdamW",
            "initial_learning_rate": 0.001,
            "duration_seconds": 36.0,
            "duration_hours": round(36 / 3600, 6),
            "seconds_per_epoch": 12.0,
            "device": "cuda:0",
            "torch_version": str(torch.__version__),
        },
        "performance": {
            "selection_metric": "metrics/mAP50-95(B)",
            "selection_mode": "max",
            "best_epoch": SAMPLE_EPOCHS,
            "best": dict(rows[-1]),
            "final": dict(rows[-1]),
            "best_checkpoint_validation": dict(rows[-1]),
        },
        "deployment": {
            "auto_export_onnx": True,
            "onnx": onnx_record,
        },
        "artifacts": {
            "best_checkpoint": "weights/best.pt",
            "last_checkpoint": "weights/last.pt",
            "metrics": "results.csv",
            "dashboard": "detection_dashboard.png",
            "metric_report": "detection_metrics.png",
            "samples": "samples/detection_sample_e{epoch}.png",
            "samples_per_epoch": len(records),
            "sample_seed": SAMPLE_SEED,
            "sample_paths": sample_paths,
            "sample_output_width": 800,
            "onnx_model": onnx_record["path"],
        },
    }
    (output_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return output_dir


class DetectionSampleOutputTests(unittest.TestCase):
    def test_generate_sample_output(self):
        output_dir = generate_sample_output()
        self.assertFalse((output_dir / "best.pt").exists())
        self.assertFalse((output_dir / "last.pt").exists())
        self.assertTrue((output_dir / "weights" / "best.pt").is_file())
        self.assertTrue((output_dir / "weights" / "last.pt").is_file())
        self.assertTrue((output_dir / "weights" / "yolo26_detect_sample.onnx").is_file())
        self.assertEqual(len(list((output_dir / "samples").glob("detection_sample_e*.png"))), SAMPLE_EPOCHS)
        self.assertTrue((output_dir / "detection_dashboard.png").is_file())
        self.assertTrue((output_dir / "detection_metrics.png").is_file())
        self.assertFalse((output_dir / "detection_samples.png").exists())
        self.assertTrue((output_dir / "summary.yaml").is_file())
        self.assertTrue((output_dir / "args.yaml").is_file())
        self.assertTrue((output_dir / "results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
