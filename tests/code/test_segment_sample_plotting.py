from __future__ import annotations

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

from segment.utils.segment_reporting import SegmentationEvaluator, plot_segmentation_metrics
from segment.utils.onnx_export import export_segment_onnx, onnx_filename
from segment.utils.training_logs import EpochLogWriter, plot_training_dashboard, save_summary_yaml, save_training_args


SAMPLE_EPOCHS = 3
SAMPLE_SEED = 2006
RESULT_COLUMNS = [
    "epoch", "train/loss", "val/loss", "val/dice", "val/iou",
    "val/hd95", "val/assd", "val/sens", "val/prec",
]
CLASS_NAMES = [
    "Femur", "Tibia", "Fibula", "Overlap", "Patella",
    "Lat Fem Osteophyte", "Med Fem Osteophyte", "Lat Tib Osteophyte",
    "Med Tib Osteophyte", "Tibial Plateau",
]


class _ToySegmentExport(torch.nn.Module):
    """Small valid graph used only to demonstrate the generated ONNX artifact."""

    def __init__(self, classes: int = 11):
        super().__init__()
        self.head = torch.nn.Conv2d(3, classes, kernel_size=1)

    def forward(self, images):
        return self.head(images)


def load_real_mesko_images(
    dataset_root: Path, count: int = 4, seed: int = SAMPLE_SEED
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Load four deterministic real X-rays and matching 11-class masks."""
    image_dir = dataset_root / "images" / "train"
    mask_dir = dataset_root / "masks" / "train"
    paths = sorted(image_dir.glob("*.png"))
    if not paths or not mask_dir.is_dir():
        raise RuntimeError("The segment sample generator requires the MESKO images and masks")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(paths), size=min(count, len(paths)), replace=False).tolist())
    images, targets, valid_indices = [], [], []
    for dataset_index in indices:
        path = paths[dataset_index]
        image_bgr = cv2.imread(str(path))
        mask = cv2.imread(str(mask_dir / path.name), cv2.IMREAD_UNCHANGED)
        if image_bgr is None or mask is None:
            continue
        source_height, source_width = image_bgr.shape[:2]
        fixed_height = 512
        relative_width = max(1, round(source_width * fixed_height / source_height))
        size = (relative_width, fixed_height)
        image = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        images.append(torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0)
        targets.append(torch.from_numpy(mask.astype(np.int64)))
        valid_indices.append(dataset_index)
    if len(images) != count:
        raise RuntimeError(f"Expected {count} valid MESKO samples, found {len(images)}")
    return images, targets, valid_indices


def _history_rows() -> list[dict]:
    rows = []
    for epoch in range(1, SAMPLE_EPOCHS + 1):
        row = {
            "epoch": epoch,
            "train/loss": 0.72 / epoch + 0.04,
            "val/loss": 0.78 / epoch + 0.05,
            "val/dice": min(0.70 + 0.085 * epoch, 0.98),
            "val/iou": min(0.58 + 0.105 * epoch, 0.95),
            "val/hd95": 6.6 / epoch,
            "val/assd": 3.0 / epoch,
            "val/sens": min(0.72 + 0.08 * epoch, 0.98),
            "val/prec": min(0.70 + 0.085 * epoch, 0.98),
        }
        rows.append({key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()})
    return rows


def generate_sample_output(output_dir: Path | None = None) -> Path:
    """Generate one internally consistent segmentation run for manual inspection."""
    output_dir = output_dir or REPO_ROOT / "tests" / "sample_output_segment"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    samples_dir = output_dir / "samples"
    weights_dir = output_dir / "weights"
    samples_dir.mkdir(parents=True)
    weights_dir.mkdir()

    rows = _history_rows()
    writer = EpochLogWriter(output_dir, "results", fieldnames=RESULT_COLUMNS, write_auxiliary=False)
    for row in rows:
        writer.append(row)

    images, targets, dataset_indices = load_real_mesko_images(REPO_ROOT / "Ref" / "unet_meskoseg_9class")
    evaluator = SegmentationEvaluator(
        11,
        pixel_spacing_mm=0.1,
        class_names=CLASS_NAMES,
        sample_indices=range(4),
        seed=SAMPLE_SEED,
    )
    for sample_index, (image, target) in enumerate(zip(images, targets)):
        height, width = target.shape
        logits = torch.full((1, 11, height, width), -5.0)
        logits.scatter_(1, target[None, None], 8.0)
        evaluator.update(logits, target[None], images=image[None], start_index=sample_index)
    snapshot = evaluator.snapshot()
    for epoch in range(1, SAMPLE_EPOCHS + 1):
        evaluator.save_samples(samples_dir / f"segment_sample_e{epoch}.png", epoch=epoch)

    plot_segmentation_metrics(snapshot, output_dir / "segment_metrics.png")
    evaluator.save_samples(output_dir / "segment_samples.png", epoch=SAMPLE_EPOCHS)
    plot_training_dashboard(
        output_dir,
        rows,
        loss_keys=[("train/loss", "Training Loss"), ("val/loss", "Validation Loss")],
        metric_keys=[
            ("val/dice", "Val Dice"), ("val/iou", "Val IoU"),
            ("val/hd95", "HD95"), ("val/assd", "ASSD"),
            ("val/sens", "Sensitivity"), ("val/prec", "Precision"),
        ],
        ranking_key="val/dice",
        maximize=True,
        filename="segment_dashboard.png",
        model_name="RWKV_UNetV5-sample",
        title="RWKV_UNetV5-sample | unet_meskoseg_9class",
        elapsed_seconds=36.0,
    )

    args = {
        "model": "RWKV_UNetV5", "dataset_name": "unet_meskoseg_9class",
        "max_epochs": SAMPLE_EPOCHS, "img_size": 256, "input_channel": 3,
        "num_classes": 11, "batch_size": 8, "base_lr": 0.001,
        "pixel_spacing_mm": 0.1, "seed": SAMPLE_SEED, "auto_export_onnx": True,
    }
    save_training_args(output_dir, args, filename="args.yaml")
    preprocess = {
        "schema_version": 1, "source_spatial_shape": "dynamic",
        "network_input_shape": [1, 3, 256, 256], "layout": "NCHW",
        "dtype": "float32", "color_space": "RGB", "value_range": [0.0, 1.0],
        "normalization": {"mode": "scale_0_1", "mean": [0.0] * 3, "std": [1.0] * 3},
        "resize": {"mode": "letterbox", "keep_aspect_ratio": True, "target_height": 256,
                   "target_width": 256, "pad_value": 0, "placement": "center"},
    }
    checkpoint = {
        "epoch": SAMPLE_EPOCHS, "state_dict": {}, "metrics": rows[-1],
        "config": args, "preprocess": preprocess,
    }
    torch.save(checkpoint, weights_dir / "last.pt")
    torch.save(checkpoint, weights_dir / "best.pt")
    onnx_path = weights_dir / onnx_filename(args["model"])
    onnx_record = export_segment_onnx(
        _ToySegmentExport().eval(),
        SimpleNamespace(**args),
        onnx_path,
        class_names=["Background", *CLASS_NAMES],
    )
    onnx_record["path"] = onnx_path.relative_to(output_dir).as_posix()

    summary = {
        "schema_version": 2,
        "task": "segmentation",
        "model": {
            "name": args["model"], "parameters": 2_450_000,
            "trainable_parameters": 2_450_000, "gflops": 12.84,
            "gflops_convention": "2 x MACs for one forward pass",
            "input_shape": [1, 3, 256, 256],
        },
        "dataset": {
            "name": args["dataset_name"], "train_manifest": "train.txt",
            "validation_manifest": "val.txt", "classes": 11, "pixel_spacing_mm": 0.1,
        },
        "preprocessing": preprocess,
        "training": {
            "epochs_requested": SAMPLE_EPOCHS, "epochs_completed": SAMPLE_EPOCHS,
            "batch_size": 8, "seed": SAMPLE_SEED, "optimizer": "AdamW",
            "criterion": "DiceCELoss", "initial_learning_rate": 0.001,
            "duration_seconds": 36.0, "duration_hours": 0.01,
            "seconds_per_epoch": 12.0, "device": "cuda:0",
            "torch_version": str(torch.__version__),
        },
        "performance": {
            "selection_metric": "val/dice", "selection_mode": "max",
            "best_epoch": SAMPLE_EPOCHS, "best": rows[-1], "final": rows[-1],
            "distance_unit": "mm",
        },
        "deployment": {
            "auto_export_onnx": True,
            "onnx": onnx_record,
        },
        "artifacts": {
            "best_checkpoint": "weights/best.pt", "last_checkpoint": "weights/last.pt",
            "best_checkpoint_type": "inference_best_without_optimizer",
            "last_checkpoint_type": "resumable_with_optimizer",
            "metrics": "results.csv", "dashboard": "segment_dashboard.png",
            "metric_report": "segment_metrics.png",
            "samples": "samples/segment_sample_e{epoch}.png",
            "samples_per_epoch": 4, "sample_seed": SAMPLE_SEED,
            "sample_indices": dataset_indices,
            "sample_display_height": 512,
            "sample_display_width": "preserve_aspect_ratio",
            "sample_output_width": 800,
            "onnx_model": onnx_record["path"],
        },
    }
    save_summary_yaml(output_dir, summary)
    return output_dir


class SegmentationSampleOutputTests(unittest.TestCase):
    def test_generate_sample_output(self):
        output_dir = generate_sample_output()
        self.assertFalse((output_dir / "best.pt").exists())
        self.assertFalse((output_dir / "last.pt").exists())
        self.assertTrue((output_dir / "weights" / "best.pt").is_file())
        self.assertTrue((output_dir / "weights" / "last.pt").is_file())
        self.assertTrue((output_dir / "weights" / "rwkv_unetv5.onnx").is_file())
        self.assertEqual(len(list((output_dir / "samples").glob("segment_sample_e*.png"))), SAMPLE_EPOCHS)


if __name__ == "__main__":
    unittest.main()
