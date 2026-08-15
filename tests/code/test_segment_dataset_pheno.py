from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from segment.dataloader.augment import build_val_transform
from segment.dataloader.dataset_pheno import (
    PhenoSegDataset,
    infer_pheno_num_classes,
    is_pheno_dataset,
)


class PhenoSegDatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path):
        image_dir = root / "images" / "train"
        mask_dir = root / "masks" / "train"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dataset_name": "PhenoX01",
                    "class_pixels": {"0": 100, "1": 20, "10": 3},
                }
            ),
            encoding="utf-8",
        )
        # This is landmark metadata and must not reduce segmentation to 4 classes.
        (root / "data_points.yaml").write_text(
            "names:\n  0: femur\n  1: tibia\n  2: fibula\n  3: patella\n",
            encoding="utf-8",
        )
        image = np.zeros((12, 8, 3), dtype=np.uint8)
        image[:, 2:6] = 200
        mask = np.zeros((12, 8), dtype=np.uint8)
        mask[3:9, 2:6] = 10
        self.assertTrue(cv2.imwrite(str(image_dir / "case.png"), image))
        self.assertTrue(cv2.imwrite(str(mask_dir / "case.png"), mask))

    def test_detects_pheno_and_uses_segmentation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            self.assertTrue(is_pheno_dataset(root))
            self.assertEqual(infer_pheno_num_classes(root), 11)

    def test_loads_discrete_mask_and_falls_back_for_empty_val_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            dataset = PhenoSegDataset(
                root,
                mode="val",
                transform=build_val_transform([24, 16]),
                num_classes=1,
            )
            sample = dataset[0]
            self.assertEqual(dataset.num_classes, 11)
            self.assertEqual(sample["image"].shape, (3, 24, 16))
            self.assertEqual(sample["label"].shape, (24, 16))
            self.assertEqual(set(np.unique(sample["label"])), {0, 10})


if __name__ == "__main__":
    unittest.main()
