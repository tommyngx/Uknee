from __future__ import annotations

import csv
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from landmark.core.plotting import _training_report_title, plot_pose_metrics, plot_validation_samples
from landmark.utils.validation import FlatPoseTrainerMixin, RESULT_COLUMNS


def plot_v9_performance_on_epoch_end(_):
    pass


class _BaseTrainer:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.wdir = save_dir / "weights"
        self.wdir.mkdir(parents=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"
        self.args = SimpleNamespace(save_period=10, resume=False)
        self.save_period = 10
        self.callbacks = defaultdict(list, {"on_fit_epoch_end": [plot_v9_performance_on_epoch_end]})
        self.best_fitness = None
        self.metrics_for_validation = {"metrics/MRE": 10.0}
        self.csv = save_dir / "results.csv"
        self.epoch = 0

    def validate(self):
        self.best_fitness = 999.0  # emulate upstream mAP fitness mutation
        return dict(self.metrics_for_validation), 999.0

    def save_model(self):
        self.last.write_bytes(b"last")
        if self.best_fitness == self.fitness:
            self.best.write_bytes(b"best")
        return True


class _Trainer(FlatPoseTrainerMixin, _BaseTrainer):
    pass


class _Metric:
    def __init__(self):
        self.px = np.linspace(0, 1, 8)
        self.prec_values = np.tile(np.linspace(1, 0, 8), (4, 1))
        self.f1_curve = np.tile(np.linspace(0, 1, 8), (4, 1))


class ReportingTests(unittest.TestCase):
    def test_onnx_filename_uses_portable_underscores(self):
        self.assertEqual(_Trainer._onnx_name("yolo26-pose-v9.yaml"), "yolo26_pose_v9.onnx")

    def test_weight_checkpoints_mre_fitness_and_compact_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer(Path(directory))
            self.assertEqual(trainer.wdir, Path(directory) / "weights")
            self.assertEqual(trainer.best, Path(directory) / "weights" / "best.pt")
            self.assertEqual(trainer.last, Path(directory) / "weights" / "last.pt")
            self.assertEqual(trainer.callbacks["on_fit_epoch_end"], [])
            _, fitness = trainer.validate()
            self.assertEqual(fitness, -10.0)
            self.assertEqual(trainer.best_fitness, -10.0)
            trainer.metrics_for_validation["metrics/MRE"] = 5.0
            _, fitness = trainer.validate()
            self.assertEqual(fitness, -5.0)
            trainer.save_metrics(
                {
                    "train/pose_loss": 2.0,
                    "val/pose_loss": 3.0,
                    "metrics/MRE": 5.0,
                    "metrics/PCK2": 0.5,
                    "metrics/PCK4": 0.7,
                    "metrics/PCK8": 0.9,
                    "metrics/HD95": 8.0,
                    "metrics/mAP50-95(B)": 0.4,
                }
            )
            with trainer.csv.open(newline="", encoding="utf-8") as stream:
                self.assertEqual(tuple(next(csv.reader(stream))), RESULT_COLUMNS)
            self.assertTrue((Path(directory) / "landmark_dashboard.png").is_file())

            trainer.model = torch.nn.Conv2d(3, 2, kernel_size=1)
            trainer.metrics = {"metrics/MRE": 5.0, "metrics/PCK2": 0.5}
            trainer.validator = SimpleNamespace(_sample_paths=["a", "b", "c", "d"], speed={"inference": 2.5})
            trainer.device = torch.device("cpu")
            trainer._write_summary()
            summary = yaml.safe_load((Path(directory) / "summary.yaml").read_text(encoding="utf-8"))
            self.assertEqual(summary["task"], "landmark_detection")
            self.assertEqual(summary["model"]["parameters"], 8)
            self.assertEqual(summary["artifacts"]["samples_per_epoch"], 4)

    def test_consolidated_metrics_and_fixed_sample_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = SimpleNamespace(
                confusion_matrix=SimpleNamespace(matrix=np.eye(5)),
                pose=_Metric(),
                box=_Metric(),
            )
            self.assertTrue(plot_pose_metrics(metrics, root / "landmark_metrics.png").is_file())
            records = [
                {
                    "path": f"image_{index}.png",
                    "image": np.zeros((32, 32, 3), dtype=float),
                    "pred": np.zeros((129, 2), dtype=float),
                    "valid": np.ones(129, dtype=bool),
                    "mre_px": 1.0,
                    "pck2": 1.0,
                    "hd95_px": 2.0,
                    "box_iou": 0.8,
                }
                for index in range(4)
            ]
            output = plot_validation_samples(records, root / "samples" / "landmark_sample_e1.png")
            self.assertIsNotNone(output)
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.width, 800)

    def test_report_title_keeps_total_time_and_time_per_epoch(self):
        title = _training_report_title("Landmark Dashboard", "pose-v9", 125.0, 5)
        self.assertEqual(
            title,
            "Landmark Dashboard: pose-v9 | Train Time: 2m 5s | Time/Epoch: 25.0s",
        )

    def test_onnx_is_refreshed_immediately_when_best_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer(Path(directory))
            trainer.fitness = trainer.best_fitness = -1.25
            record = {"status": "ready", "path": "weights/pose.onnx"}
            with patch.object(trainer, "_export_best_onnx", return_value=record) as export:
                self.assertTrue(trainer.save_model())
            self.assertTrue(trainer.best.is_file())
            export.assert_called_once_with()
            self.assertEqual(trainer._onnx_export_record, record)


if __name__ == "__main__":
    unittest.main()
