from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from landmark2.core.plotting import plot_dashboard_pose, plot_pose_metrics, plot_validation_samples


def load_real_mesko_samples(dataset_root: Path, count: int = 4) -> list[dict]:
    """Load up to `count` real knee X-ray images and matching 129-keypoint annotations from Ref/yolo_mesko4GF2."""
    img_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    if not img_dir.exists() or not label_dir.exists():
        return []

    img_paths = sorted(img_dir.glob("*.png"))[:count]
    records = []

    for img_path in img_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        pred = np.zeros((129, 2), dtype=float)
        valid = np.zeros(129, dtype=bool)

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").strip().splitlines()
            # Class offsets: Femur (45), Tibia (51), Fibula (24), Patella (9)
            class_kpt_counts = {0: 45, 1: 51, 2: 24, 3: 9}
            class_offsets = {0: 0, 1: 45, 2: 96, 3: 120}

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                kpt_data = parts[5:]
                num_kpts = class_kpt_counts.get(cls_id, 0)
                start_idx = class_offsets.get(cls_id, 0)

                for i in range(num_kpts):
                    if i * 3 + 2 < len(kpt_data):
                        kx = float(kpt_data[i * 3]) * w
                        ky = float(kpt_data[i * 3 + 1]) * h
                        kv = float(kpt_data[i * 3 + 2])
                        idx = start_idx + i
                        if idx < 129:
                            pred[idx] = [kx, ky]
                            valid[idx] = (kv > 0)

        # Convert BGR image to RGB for matplotlib plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        records.append({
            "path": img_path.name,
            "image": img_rgb,
            "pred": pred,
            "valid": valid,
            "mre_px": 1.25,
            "pck2": 0.985,
            "hd95_px": 2.10,
            "box_iou": 0.942,
        })

    return records


class _Metric:
    def __init__(self):
        self.px = np.linspace(0, 1, 100)
        self.prec_values = np.tile(np.linspace(1, 0.2, 100), (4, 1))
        self.f1_curve = np.tile(np.linspace(0, 0.95, 100), (4, 1))


class Landmark2SamplePlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tests_dir = Path(__file__).parent
        cls.ref_dataset = cls.tests_dir.parent / "Ref" / "yolo_mesko4GF2"

    def test_01_plot_validation_samples(self):
        records = load_real_mesko_samples(self.ref_dataset, count=4)
        if not records:
            records = [
                {
                    "path": f"sample_knee_{i}.png",
                    "image": np.full((512, 512, 3), 128, dtype=np.uint8),
                    "pred": np.random.uniform(100, 400, (129, 2)),
                    "valid": np.ones(129, dtype=bool),
                    "mre_px": 1.15 + i * 0.1,
                    "pck2": 0.98,
                    "hd95_px": 2.0,
                    "box_iou": 0.95,
                }
                for i in range(4)
            ]

        output_png = self.tests_dir / "landmark2_val_samples.png"
        res = plot_validation_samples(records, output_png)
        self.assertIsNotNone(res)
        self.assertTrue(output_png.is_file())
        self.assertGreater(output_png.stat().st_size, 10000)

    def test_02_plot_dashboard_pose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            columns = [
                "epoch", "train/loss", "val/loss", "metrics/MRE", "metrics/PCK2",
                "metrics/PCK4", "metrics/PCK8", "metrics/HD95", "metrics/mAP50-95(B)",
                "metrics/MRE_femur", "metrics/MRE_tibia", "metrics/MRE_fibula", "metrics/MRE_patella"
            ]

            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for ep in range(1, 21):
                    loss_tr = 0.5 * np.exp(-ep / 5) + 0.05
                    loss_va = 0.55 * np.exp(-ep / 5) + 0.07 + np.random.uniform(-0.01, 0.01)
                    mre = 15.0 * np.exp(-ep / 6) + 1.2
                    pck2 = 0.3 + 0.65 * (1 - np.exp(-ep / 4))
                    pck4 = 0.5 + 0.48 * (1 - np.exp(-ep / 4))
                    pck8 = 0.7 + 0.29 * (1 - np.exp(-ep / 4))
                    hd95 = 20.0 * np.exp(-ep / 5) + 2.5
                    map_box = 0.2 + 0.75 * (1 - np.exp(-ep / 5))
                    femur = mre * 0.9
                    tibia = mre * 1.05
                    fibula = mre * 1.2
                    patella = mre * 0.85

                    writer.writerow([
                        ep, loss_tr, loss_va, mre, pck2, pck4, pck8, hd95, map_box,
                        femur, tibia, fibula, patella
                    ])

            output_png = self.tests_dir / "landmark2_dashboard_pose.png"
            res = plot_dashboard_pose(csv_path, output_png, model_name="landmark2-v9-pose-sample", elapsed_seconds=320.5)
            self.assertIsNotNone(res)
            self.assertTrue(output_png.is_file())
            self.assertGreater(output_png.stat().st_size, 10000)

    def test_03_plot_pose_metrics(self):
        metrics = SimpleNamespace(
            confusion_matrix=SimpleNamespace(matrix=np.array([
                [95, 2, 1, 0],
                [1, 92, 3, 0],
                [1, 2, 88, 1],
                [0, 1, 1, 96]
            ])),
            pose=_Metric(),
            box=_Metric(),
        )

        output_png = self.tests_dir / "landmark2_pose_metrics.png"
        res = plot_pose_metrics(metrics, output_png)
        self.assertIsNotNone(res)
        self.assertTrue(output_png.is_file())
        self.assertGreater(output_png.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
