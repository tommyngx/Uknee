from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from segment.dataloader.augment import build_val_transform
from segment.dataloader.dataloader import getDataloader
from segment.dataloader.dataset_handoa import (
    HandOASegDataset,
    handoa_class_info,
    infer_handoa_num_classes,
    is_handoa_dataset,
)


class HandOASegDatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path):
        image_dir = root / "images" / "train"
        mask_dir = root / "masks" / "train"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dataset_name": "HandOA_segX1",
                    "num_classes": 11,
                    "class_pixels": {"0": 100, "1": 20, "10": 3},
                }
            ),
            encoding="utf-8",
        )
        (root / "classes.json").write_text(
            json.dumps(
                [
                    {"id": 0, "name": "background"},
                    {"id": 1, "name": "distal_phalanges"},
                    {"id": 2, "name": "middle_phalanges"},
                    {"id": 3, "name": "proximal_phalanges"},
                    {"id": 4, "name": "metacarpal_1"},
                    {"id": 5, "name": "metacarpals_other"},
                    {"id": 6, "name": "trapezium"},
                    {"id": 7, "name": "trapezoid"},
                    {"id": 8, "name": "scaphoid"},
                    {"id": 9, "name": "interphalangeal_plateau"},
                    {"id": 10, "name": "metacarpal_overlap"},
                ]
            ),
            encoding="utf-8",
        )
        image = np.zeros((16, 12, 3), dtype=np.uint8)
        image[:, 2:10] = 200
        mask = np.zeros((16, 12), dtype=np.uint8)
        mask[3:12, 2:10] = 10
        self.assertTrue(cv2.imwrite(str(image_dir / "case_01.png"), image))
        self.assertTrue(cv2.imwrite(str(mask_dir / "case_01.png"), mask))

    def test_detects_handoa_and_uses_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            self.assertTrue(is_handoa_dataset(root))
            self.assertEqual(infer_handoa_num_classes(root), 11)
            info = handoa_class_info(root)
            self.assertEqual(len(info), 11)
            self.assertEqual(info[0]["name"], "background")
            self.assertEqual(info[4]["name"], "metacarpal_1")
            self.assertEqual(info[6]["name"], "trapezium")

    def test_loads_sample_and_falls_back_for_empty_val_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            dataset = HandOASegDataset(
                root,
                mode="val",
                transform=build_val_transform(img_size=[16, 12]),
                num_classes=11,
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["case"], "case_01")
            self.assertEqual(item["image"].shape, (3, 16, 12))
            self.assertEqual(item["label"].shape, (16, 12))
            self.assertEqual(int(item["label"].max()), 10)

    def test_dataloader_routing_for_handoa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            args = SimpleNamespace(
                base_dir=str(root),
                dataset_name="HandOA_segX1",
                img_size=[16, 12],
                batch_size=1,
                workers=0,
                aug_strategy="none",
                num_classes=11,
                seed=2006,
            )
            train_loader, val_loader = getDataloader(args)
            self.assertIsInstance(train_loader.dataset, HandOASegDataset)
            self.assertIsInstance(val_loader.dataset, HandOASegDataset)
            batch = next(iter(train_loader))
            self.assertEqual(batch["image"].shape, (1, 3, 16, 12))
            self.assertEqual(batch["label"].shape, (1, 16, 12))


if __name__ == "__main__":
    unittest.main()
