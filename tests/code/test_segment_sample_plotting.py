from __future__ import annotations

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

import cv2
import numpy as np
import torch

from segment.utils.segment_reporting import SegmentationEvaluator, plot_segmentation_metrics
from segment.utils.training_logs import EpochLogWriter, plot_training_dashboard, save_training_args


def load_real_mesko_images(dataset_root: Path, count: int = 4, seed: int = 2006) -> tuple[torch.Tensor, torch.Tensor]:
    """Load up to `count` real knee X-ray images and matching 11-class masks from Ref/unet_meskoseg_9class."""
    img_dir = dataset_root / "images" / "train"
    mask_dir = dataset_root / "masks" / "train"
    if not img_dir.exists() or not mask_dir.exists():
        images = torch.rand(count, 3, 256, 256)
        targets = torch.zeros(count, 256, 256, dtype=torch.long)
        return images, targets

    all_img_paths = sorted(img_dir.glob("*.png"))
    rng = np.random.RandomState(seed)
    if len(all_img_paths) > count:
        indices = sorted(rng.choice(len(all_img_paths), size=count, replace=False))
        img_paths = [all_img_paths[i] for i in indices]
    else:
        img_paths = all_img_paths[:count]
    images_list = []
    targets_list = []

    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        img_bgr = cv2.imread(str(img_path))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None or mask_raw is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        new_h = 512
        new_w = int(w * (512.0 / h))
        img_resized = cv2.resize(img_rgb, (320, 512), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_raw, (320, 512), interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized.astype(np.int64))

        images_list.append(img_tensor)
        targets_list.append(mask_tensor)

    if not images_list:
        images_list = [torch.rand(3, 512, 320) for _ in range(count)]
        targets_list = [torch.zeros(512, 320, dtype=torch.long) for _ in range(count)]

    return torch.stack(images_list), torch.stack(targets_list)


class SegmentationSamplePlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).resolve().parent.parent
        cls.ref_dataset = cls.tests_dir.parent / "Ref" / "unet_meskoseg_9class"

    def test_01_plot_segmentation_val_samples_and_metrics(self):
        images, targets = load_real_mesko_images(self.ref_dataset, count=4)
        num_classes = 11

        # Generate realistic logits with slight random noise for validation predictions
        logits = torch.full((len(images), num_classes, 512, 320), -5.0)
        for b in range(len(images)):
            t_mask = targets[b]
            logits[b].scatter_(0, t_mask.unsqueeze(0), 8.0)
            noise = torch.randn_like(logits[b]) * 0.5
            logits[b] += noise

        evaluator = SegmentationEvaluator(
            num_classes,
            pixel_spacing_mm=0.10,
            class_names=[
                "Femur", "Tibia", "Fibula", "Overlap", "Patella",
                "Lat Fem Osteophyte", "Med Fem Osteophyte", "Lat Tib Osteophyte",
                "Med Tib Osteophyte", "Tibial Plateau"
            ],
            sample_indices=list(range(len(images))),
            seed=2006
        )

        evaluator.update(logits, targets, images=images, start_index=0)
        snapshot = evaluator.snapshot()

        # 1. Save validation samples grid
        val_samples_png = self.tests_dir / "segment_samples.png"
        evaluator.save_samples(val_samples_png, epoch=150)
        self.assertTrue(val_samples_png.is_file())
        self.assertGreater(val_samples_png.stat().st_size, 10000)

        # 2. Save evaluation metrics dashboard
        metrics_png = self.tests_dir / "segment_metrics.png"
        plot_segmentation_metrics(snapshot, metrics_png)
        self.assertTrue(metrics_png.is_file())
        self.assertGreater(metrics_png.stat().st_size, 10000)

    def test_02_plot_segmentation_training_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            history_rows = []
            for ep in range(1, 31):
                tr_loss = 0.8 * np.exp(-ep / 8) + 0.05
                va_loss = 0.85 * np.exp(-ep / 8) + 0.07 + np.random.uniform(-0.01, 0.01)
                val_dice = 0.25 + 0.72 * (1 - np.exp(-ep / 6))
                val_iou = 0.15 + 0.70 * (1 - np.exp(-ep / 6))
                val_hd95 = 25.0 * np.exp(-ep / 7) + 1.8
                val_assd = 12.0 * np.exp(-ep / 7) + 0.8
                val_sens = 0.3 + 0.67 * (1 - np.exp(-ep / 5))
                val_prec = 0.35 + 0.63 * (1 - np.exp(-ep / 5))

                row = {
                    "epoch": ep,
                    "train/loss": tr_loss,
                    "val/loss": va_loss,
                    "val/dice": val_dice,
                    "val/iou": val_iou,
                    "val/hd95": val_hd95,
                    "val/assd": val_assd,
                    "val/sens": val_sens,
                    "val/prec": val_prec,
                }
                history_rows.append(row)

            dashboard_png = self.tests_dir / "segment_dashboard.png"
            plot_path, top_epochs = plot_training_dashboard(
                log_dir=tmp_path,
                history_rows=history_rows,
                loss_keys=[("train/loss", "Training Loss"), ("val/loss", "Validation Loss")],
                metric_keys=[
                    ("val/dice", "Val Dice"), ("val/iou", "Val IoU"),
                    ("val/hd95", "Val HD95 (mm)"), ("val/assd", "Val ASSD (mm)"),
                    ("val/sens", "Sensitivity"), ("val/prec", "Precision"),
                ],
                ranking_key="val/dice",
                maximize=True,
                filename=str(dashboard_png),
                model_name="CMUNeXt-MESKO5Seg",
                elapsed_seconds=320.0,
            )

            self.assertTrue(dashboard_png.is_file())
            self.assertGreater(dashboard_png.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
