from __future__ import annotations

import csv
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from landmark0.utils.plotting import plot_pose_metrics, plot_validation_samples
from landmark0.utils.validation import FlatPoseTrainerMixin, RESULT_COLUMNS


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


class _Trainer(FlatPoseTrainerMixin, _BaseTrainer):
    pass


class _Metric:
    def __init__(self):
        self.px = np.linspace(0, 1, 8)
        self.prec_values = np.tile(np.linspace(1, 0, 8), (4, 1))
        self.f1_curve = np.tile(np.linspace(0, 1, 8), (4, 1))


class ReportingTests(unittest.TestCase):
    def test_flat_checkpoints_mre_fitness_and_compact_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = _Trainer(Path(directory))
            self.assertEqual(trainer.wdir, Path(directory))
            self.assertEqual(trainer.best, Path(directory) / "best.pt")
            self.assertEqual(trainer.last, Path(directory) / "last.pt")
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
            self.assertTrue((Path(directory) / "dashboard_pose.png").is_file())

    def test_consolidated_metrics_and_fixed_sample_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = SimpleNamespace(
                confusion_matrix=SimpleNamespace(matrix=np.eye(5)),
                pose=_Metric(),
                box=_Metric(),
            )
            self.assertTrue(plot_pose_metrics(metrics, root / "pose_metrics.png").is_file())
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
            output = plot_validation_samples(records, root / "samples" / "val_samples_e1.png")
            self.assertIsNotNone(output)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
