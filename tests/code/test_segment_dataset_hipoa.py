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
from segment.dataloader.dataset_hipoa import (
    HIPOA_DEFAULT_CLASSES,
    HipOASegDataset,
    hipoa_class_ids,
    hipoa_class_info,
    infer_hipoa_num_classes,
    is_hipoa_dataset,
)


class HipOASegDatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path):
        image_dir = root / "images" / "train"
        mask_dir = root / "masks" / "train"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "dataset_name": "HipOA_segX1",
                    "num_classes": 7,
                    "class_pixels": {"0": 100, "1": 20, "6": 5},
                }
            ),
            encoding="utf-8",
        )
        (root / "classes.json").write_text(
            json.dumps(HIPOA_DEFAULT_CLASSES),
            encoding="utf-8",
        )
        image = np.zeros((16, 12, 3), dtype=np.uint8)
        image[:, 2:10] = 200
        mask = np.zeros((16, 12), dtype=np.uint8)
        mask[3:12, 2:10] = 6
        self.assertTrue(cv2.imwrite(str(image_dir / "case_01.png"), image))
        self.assertTrue(cv2.imwrite(str(mask_dir / "case_01.png"), mask))

    def test_detects_hipoa_and_uses_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            self.assertTrue(is_hipoa_dataset(root))
            self.assertEqual(infer_hipoa_num_classes(root), 7)
            info = hipoa_class_info(root)
            self.assertEqual(len(info), 7)
            self.assertEqual(info[0]["name"], "background")
            self.assertEqual(info[1]["name"], "right_hip_joint_space")
            self.assertEqual(info[2]["name"], "left_hip_joint_space")
            self.assertEqual(info[3]["name"], "right_pelvis")
            self.assertEqual(info[4]["name"], "left_pelvis")
            self.assertEqual(info[5]["name"], "right_femur")
            self.assertEqual(info[6]["name"], "left_femur")
            self.assertEqual(info[1]["color"], [139, 92, 246])
            self.assertEqual(info[3]["color"], [255, 141, 23])
            self.assertEqual(info[4]["color"], [20, 83, 45])
            self.assertEqual(info[5]["color"], [0, 180, 255])
            self.assertEqual(info[6]["color"], [255, 120, 220])

    def test_loads_sample_and_falls_back_for_empty_val_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            dataset = HipOASegDataset(
                root,
                mode="val",
                transform=build_val_transform(img_size=[16, 12]),
                num_classes=7,
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["case"], "case_01")
            self.assertEqual(item["image"].shape, (3, 16, 12))
            self.assertEqual(item["label"].shape, (16, 12))
            self.assertEqual(int(item["label"].max()), 6)

    def test_dataloader_routing_for_hipoa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            args = SimpleNamespace(
                base_dir=str(root),
                dataset_name="HipOA_segX1",
                img_size=[16, 12],
                batch_size=1,
                workers=0,
                aug_strategy="none",
                num_classes=7,
                seed=2006,
            )
            train_loader, val_loader = getDataloader(args)
            self.assertIsInstance(train_loader.dataset, HipOASegDataset)
            self.assertIsInstance(val_loader.dataset, HipOASegDataset)
            batch = next(iter(train_loader))
            self.assertEqual(batch["image"].shape, (1, 3, 16, 12))
            self.assertEqual(batch["label"].shape, (1, 16, 12))

    def test_real_ref_dataset_detection_and_classes(self):
        ref_path = Path("/Users/francistommy/Desktop/BugHunter/Project/Uknee/Ref/HipOA_segX1")
        if not ref_path.is_dir():
            self.skipTest("Ref/HipOA_segX1 directory not found.")
        self.assertTrue(is_hipoa_dataset(ref_path))
        self.assertEqual(infer_hipoa_num_classes(ref_path), 7)
        ids = hipoa_class_ids(ref_path)
        self.assertEqual(ids, [0, 1, 2, 3, 4, 5, 6])
        info = hipoa_class_info(ref_path)
        self.assertEqual(len(info), 7)
        self.assertEqual(info[0]["name"], "background")
        self.assertEqual(info[1]["name"], "right_hip_joint_space")
        self.assertEqual(info[2]["name"], "left_hip_joint_space")
        self.assertEqual(info[3]["name"], "right_pelvis")
        self.assertEqual(info[4]["name"], "left_pelvis")
        self.assertEqual(info[5]["name"], "right_femur")
        self.assertEqual(info[6]["name"], "left_femur")

    def test_parse_arguments_auto_updates_num_classes_for_hipoa(self):
        from segment.main import parse_arguments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_dataset(root)
            args = parse_arguments(["--base_dir", str(root), "--num_classes", "1"])
            self.assertEqual(args.num_classes, 7)


if __name__ == "__main__":
    unittest.main()
