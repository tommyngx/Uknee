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
from landmark.core.plotting import plot_dashboard_pose, plot_pose_metrics, plot_validation_samples


SAMPLE_EPOCHS = 3
SAMPLE_SEED = 2006
RESULT_COLUMNS = [
    "epoch", "train/loss", "val/loss", "metrics/MRE", "metrics/PCK2",
    "metrics/PCK4", "metrics/PCK8", "metrics/HD95", "metrics/mAP50-95(B)",
    "metrics/MRE_femur", "metrics/MRE_tibia", "metrics/MRE_fibula", "metrics/MRE_patella",
]


def load_real_mesko_samples(dataset_root: Path, count: int = 4, seed: int = SAMPLE_SEED) -> list[dict]:
    """Load four deterministic real X-rays and their 129-keypoint annotations."""
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    paths = sorted(image_dir.glob("*.png"))
    if not paths or not label_dir.is_dir():
        return []
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(paths), size=min(count, len(paths)), replace=False).tolist())
    records = []
    for path in (paths[index] for index in chosen):
        image_bgr = cv2.imread(str(path))
        label_path = label_dir / f"{path.stem}.txt"
        if image_bgr is None or not label_path.is_file():
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        prediction = np.zeros((129, 2), dtype=np.float32)
        valid = np.zeros(129, dtype=bool)
        offsets = {0: 0, 1: 45, 2: 96, 3: 120}
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            offset = offsets.get(int(parts[0]), 0)
            values = [float(value) for value in parts[5:]]
            for point in range(len(values) // 3):
                index = offset + point
                if index < 129 and values[point * 3 + 2] > 0:
                    prediction[index] = [values[point * 3] * width, values[point * 3 + 1] * height]
                    valid[index] = True
        records.append({"path": path.name, "image": image, "pred": prediction, "valid": valid})
    return records


def _history_rows() -> list[dict]:
    rows = []
    for epoch in range(1, SAMPLE_EPOCHS + 1):
        mre = 4.2 / epoch
        row = {
            "epoch": epoch,
            "train/loss": 0.42 / epoch + 0.03,
            "val/loss": 0.48 / epoch + 0.04,
            "metrics/MRE": mre,
            "metrics/PCK2": min(0.72 + 0.09 * epoch, 0.99),
            "metrics/PCK4": min(0.82 + 0.055 * epoch, 0.995),
            "metrics/PCK8": min(0.91 + 0.025 * epoch, 0.999),
            "metrics/HD95": 7.5 / epoch,
            "metrics/mAP50-95(B)": min(0.62 + 0.1 * epoch, 0.95),
            "metrics/MRE_femur": mre * 0.90,
            "metrics/MRE_tibia": mre * 1.05,
            "metrics/MRE_fibula": mre * 1.20,
            "metrics/MRE_patella": mre * 0.85,
        }
        rows.append({key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()})
    return rows


class _Metric:
    def __init__(self):
        self.px = np.linspace(0, 1, 100)
        self.prec_values = np.tile(np.linspace(1, 0.2, 100), (4, 1))
        self.f1_curve = np.tile(np.linspace(0, 0.95, 100), (4, 1))


class _ToyPoseExport(torch.nn.Module):
    """Small valid graph matching the three-output landmark deployment contract."""

    yaml_file = "yolo26-pose-sample.yaml"

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.stride = torch.tensor([32.0])

    def forward(self, images):
        base = images[:, 0, 0, 0] * 0.0 + self.anchor * 0.0
        detections = base[:, None, None].expand(-1, 4, 159).contiguous()
        num_detections = base.to(torch.int64)
        canonical = base[:, None, None].expand(-1, 129, 3).contiguous()
        return detections, num_detections, canonical


def generate_sample_output(output_dir: Path | None = None) -> Path:
    """Generate one internally consistent landmark run for manual inspection."""
    output_dir = output_dir or REPO_ROOT / "tests" / "sample_output_landmark"
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

    records = load_real_mesko_samples(REPO_ROOT / "Ref" / "yolo_mesko4GF2")
    if len(records) != 4:
        raise RuntimeError("The landmark sample generator requires four valid MESKO images and labels")
    for epoch, row in enumerate(rows, start=1):
        epoch_records = [
            {
                **record,
                "mre_px": row["metrics/MRE"],
                "pck2": row["metrics/PCK2"],
                "hd95_px": row["metrics/HD95"],
                "box_iou": row["metrics/mAP50-95(B)"],
            }
            for record in records
        ]
        plot_validation_samples(
            epoch_records,
            samples_dir / f"landmark_sample_e{epoch}.png",
            epoch=epoch,
        )

    plot_dashboard_pose(
        output_dir / "results.csv",
        output_dir / "landmark_dashboard.png",
        model_name="yolo26-pose-sample",
        elapsed_seconds=48.0,
    )
    metrics = SimpleNamespace(
        confusion_matrix=SimpleNamespace(matrix=np.array([
            [95, 2, 1, 0], [1, 92, 3, 0], [1, 2, 88, 1], [0, 1, 1, 96],
        ])),
        pose=_Metric(),
        box=_Metric(),
    )
    plot_pose_metrics(
        metrics,
        output_dir / "landmark_metrics.png",
        model_name="yolo26-pose-sample",
        elapsed_seconds=48.0,
        epochs_completed=SAMPLE_EPOCHS,
    )
    plot_validation_samples(epoch_records, output_dir / "landmark_samples.png", epoch=SAMPLE_EPOCHS)

    args = {
        "model": "yolo26-pose-sample.yaml", "data": "Ref/yolo_mesko4GF2/data.yaml",
        "epochs": SAMPLE_EPOCHS, "imgsz": [512, 320], "batch": 16,
        "lr0": 0.001, "seed": SAMPLE_SEED, "auto_export_onnx": True,
        "fliplr": 0.0, "flipud": 0.0, "erasing": 0.0, "bgr": 0.0,
    }
    (output_dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), encoding="utf-8")
    checkpoint = {"epoch": SAMPLE_EPOCHS, "model": "sample-only", "metrics": rows[-1]}
    torch.save(checkpoint, weights_dir / "last.pt")
    torch.save({**checkpoint, "epoch": rows[-1]["epoch"]}, weights_dir / "best.pt")
    onnx_path = weights_dir / "yolo26_pose_sample.onnx"
    Exporter(
        {
            "format": "onnx", "imgsz": [512, 320], "batch": 1, "dynamic": True,
            "path": onnx_path, "model_name": "yolo26-pose-sample",
            "source_checkpoint": "weights/best.pt",
        }
    )(_ToyPoseExport().eval())
    onnx_metadata = read_onnx_metadata(onnx_path)
    onnx_record = {
        "status": "ready", "path": onnx_path.relative_to(output_dir).as_posix(),
        "format": "onnx", "sha256": onnx_sha256(onnx_path),
        "file_size_bytes": onnx_path.stat().st_size, "metadata": onnx_metadata,
        "parity": json.loads(onnx_metadata["uknee.parity"]),
    }

    sample_paths = [record["path"] for record in records]
    preprocess = {
        "schema_version": 1, "source_spatial_shape": "dynamic",
        "network_input_shape": [1, 3, 512, 320], "layout": "NCHW",
        "dtype": "float32", "color_space": "RGB", "value_range": [0.0, 1.0],
        "normalization": {"mode": "scale_0_1", "mean": [0.0] * 3, "std": [1.0] * 3},
        "resize": {"mode": "letterbox", "keep_aspect_ratio": True, "target_height": 512,
                   "target_width": 320, "pad_value": 114, "placement": "center", "stride": 32},
    }
    summary = {
        "schema_version": 2,
        "task": "landmark_detection",
        "model": {
            "name": "yolo26-pose-sample", "source": args["model"],
            "parameters": 3_120_000, "trainable_parameters": 3_120_000,
            "gflops": 8.65, "gflops_convention": "2 x MACs for one forward pass",
            "input_shape": [1, 3, 512, 320],
        },
        "dataset": {"config": args["data"], "pixel_spacing_mm": 0.1},
        "preprocessing": preprocess,
        "training": {
            "epochs_requested": SAMPLE_EPOCHS, "epochs_completed": SAMPLE_EPOCHS,
            "batch_size": 16, "seed": SAMPLE_SEED, "optimizer": "AdamW",
            "initial_learning_rate": 0.001, "duration_seconds": 48.0,
            "duration_hours": round(48 / 3600, 6), "seconds_per_epoch": 16.0,
            "device": "cuda:0", "torch_version": str(torch.__version__),
        },
        "performance": {
            "selection_metric": "metrics/MRE", "selection_mode": "min",
            "best_epoch": SAMPLE_EPOCHS, "best": dict(rows[-1]), "final": dict(rows[-1]),
            "best_checkpoint_validation": dict(rows[-1]), "distance_unit_in_metrics": "pixel",
            "pixel_spacing_mm": 0.1, "pck_thresholds_pixels": [2, 4, 8],
            "pck_thresholds_mm": [0.2, 0.4, 0.8], "inference_ms_per_image": 2.5,
        },
        "deployment": {
            "auto_export_onnx": True,
            "onnx": onnx_record,
        },
        "artifacts": {
            "best_checkpoint": "weights/best.pt", "last_checkpoint": "weights/last.pt",
            "metrics": "results.csv", "dashboard": "landmark_dashboard.png",
            "metric_report": "landmark_metrics.png",
            "samples": "samples/landmark_sample_e{epoch}.png",
            "samples_per_epoch": 4, "sample_seed": SAMPLE_SEED, "sample_paths": sample_paths,
            "sample_output_width": 800,
            "onnx_model": onnx_record["path"],
        },
    }
    (output_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return output_dir


class Landmark2SampleOutputTests(unittest.TestCase):
    def test_generate_sample_output(self):
        output_dir = generate_sample_output()
        self.assertFalse((output_dir / "best.pt").exists())
        self.assertFalse((output_dir / "last.pt").exists())
        self.assertTrue((output_dir / "weights" / "best.pt").is_file())
        self.assertTrue((output_dir / "weights" / "last.pt").is_file())
        self.assertTrue((output_dir / "weights" / "yolo26_pose_sample.onnx").is_file())
        self.assertEqual(len(list((output_dir / "samples").glob("landmark_sample_e*.png"))), SAMPLE_EPOCHS)


if __name__ == "__main__":
    unittest.main()
