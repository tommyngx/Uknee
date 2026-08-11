import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the repository root and segment package are importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (str(REPO_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from segment.utils.segment_reporting import SegmentationEvaluator, plot_segmentation_metrics
from segment.utils.training_logs import (
    EpochLogWriter,
    model_paper_profile,
    plot_training_dashboard,
    save_summary_yaml,
    save_training_args,
)


RESULT_COLUMNS = [
    "epoch", "train/loss", "val/loss", "val/dice", "val/iou",
    "val/hd95", "val/assd", "val/sens", "val/prec",
]


class SegmentReportingTests(unittest.TestCase):
    def test_binary_evaluator_and_reports_use_real_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target = torch.zeros(2, 1, 16, 16)
            target[:, :, 4:12, 4:12] = 1
            logits = torch.full_like(target, -8.0)
            logits[:, :, 4:12, 4:12] = 8.0
            images = torch.rand(2, 3, 16, 16)

            evaluator = SegmentationEvaluator(1, pixel_spacing_mm=0.1, sample_indices=[0, 1])
            evaluator.update(logits, target, images=images)
            snapshot = evaluator.snapshot()

            self.assertEqual(snapshot.dice, 1.0)
            self.assertEqual(snapshot.iou, 1.0)
            self.assertEqual(snapshot.hd95, 0.0)
            self.assertEqual(snapshot.assd, 0.0)
            plot_segmentation_metrics(snapshot, tmp_path / "segment_metrics.png")
            evaluator.save_samples(tmp_path / "samples" / "segment_sample_e1.png", epoch=1)
            self.assertTrue((tmp_path / "segment_metrics.png").is_file())
            self.assertTrue((tmp_path / "samples" / "segment_sample_e1.png").is_file())

    def test_multiclass_region_metrics_and_fixed_samples_are_deterministic(self):
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, 1:4, 1:4] = 1
        target[:, 4:7, 4:7] = 2
        logits = torch.full((1, 3, 8, 8), -9.0)
        logits.scatter_(1, target.unsqueeze(1), 9.0)
        evaluator = SegmentationEvaluator(3, class_names=["Background", "Femur", "Tibia"])
        evaluator.update(logits, target)
        snapshot = evaluator.snapshot()

        np.testing.assert_allclose(snapshot.class_dice, [1.0, 1.0])
        self.assertEqual(snapshot.class_names, ["Femur", "Tibia"])
        self.assertEqual(SegmentationEvaluator.fixed_sample_indices(30, seed=2006), SegmentationEvaluator.fixed_sample_indices(30, seed=2006))

    def test_flat_csv_yaml_and_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            writer = EpochLogWriter(tmp_path, "results", fieldnames=RESULT_COLUMNS, write_auxiliary=False)
            row = dict(zip(RESULT_COLUMNS, [1, 0.8, 0.7, 0.6, 0.5, 1.2, 0.4, 0.7, 0.8]))
            writer.append(row)
            save_training_args(tmp_path, {"seed": 2006}, filename="args.yaml")
            plot_training_dashboard(
                tmp_path, [row],
                loss_keys=[("train/loss", "Training Loss"), ("val/loss", "Validation Loss")],
                metric_keys=[
                    ("val/dice", "Val Dice"), ("val/iou", "Val IoU"),
                    ("val/hd95", "HD95"), ("val/assd", "ASSD"),
                    ("val/sens", "Sensitivity"), ("val/prec", "Precision"),
                ],
                ranking_key="val/dice", filename="segment_dashboard.png",
            )

            with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(next(csv.reader(file)), RESULT_COLUMNS)
            self.assertTrue((tmp_path / "args.yaml").is_file())
            self.assertTrue((tmp_path / "segment_dashboard.png").is_file())
            self.assertFalse((tmp_path / "results.jsonl").exists())

    def test_paper_summary_contains_model_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model = torch.nn.Conv2d(3, 2, kernel_size=1)
            profile = model_paper_profile(model, (1, 3, 16, 16))
            save_summary_yaml(tmp_path, {"task": "segmentation", "model": profile})

            import yaml

            summary = yaml.safe_load((tmp_path / "summary.yaml").read_text(encoding="utf-8"))
            self.assertEqual(summary["task"], "segmentation")
            self.assertEqual(summary["model"]["parameters"], 8)
            self.assertEqual(summary["model"]["input_shape"], [1, 3, 16, 16])


if __name__ == "__main__":
    unittest.main()
