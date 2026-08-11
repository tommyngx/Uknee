from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from landmark.data.augment import Format
import landmark  # noqa: F401 - bootstrap the pinned backend before absolute imports
from landmark.core import ops

from landmark.data import (
    NUM_LANDMARKS,
    REGION_KEYPOINT_COUNTS,
    objects_to_canonical,
    prepare_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def group_key(path: Path) -> str:
    stem = path.stem.removesuffix("Flip")
    return stem.removesuffix("_L").removesuffix("_R")


class DatasetAndAdapterTests(unittest.TestCase):
    def test_training_formatter_uses_rgb_contract(self):
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
        tensor = Format(bgr=0.0)._format_img(bgr)
        self.assertEqual(tensor[:, 0, 0].tolist(), [30, 20, 10])

    @classmethod
    def setUpClass(cls):
        cls.prepared = prepare_dataset(ROOT / "cfg" / "datasets" / "mesko4gf2.yaml")

    def test_all_442_labels_are_validated_and_split_without_case_leakage(self):
        prepared = self.prepared
        self.assertEqual(len(prepared.train_images) + len(prepared.val_images), 442)
        self.assertFalse(
            {group_key(path) for path in prepared.train_images}
            & {group_key(path) for path in prepared.val_images}
        )
        self.assertTrue(prepared.yaml_path.is_file())

    def test_adapter_selects_best_duplicate_and_zero_fills_missing_classes(self):
        keypoints = torch.zeros(4, 51, 3)
        keypoints[0, :, :2] = 1
        keypoints[1, :, :2] = 2
        keypoints[2, :, :2] = 3
        keypoints[3, :, :2] = 9
        keypoints[..., 2] = 0.75
        classes = torch.tensor([0, 1, 1, 3])
        scores = torch.tensor([0.5, 0.2, 0.9, 0.8])
        coordinates, confidence = objects_to_canonical(keypoints, classes, scores=scores)
        self.assertEqual(tuple(coordinates.shape), (NUM_LANDMARKS, 2))
        self.assertTrue(torch.all(coordinates[45:96] == 3))
        self.assertTrue(torch.all(coordinates[96:120] == 0))
        self.assertTrue(torch.all(confidence[96:120] == 0))

    def test_adapter_discards_padded_slots(self):
        keypoints = torch.ones(4, 51, 3)
        coordinates, confidence = objects_to_canonical(keypoints, torch.arange(4))
        self.assertEqual(coordinates.shape[0], sum(REGION_KEYPOINT_COUNTS))
        self.assertEqual(confidence.shape[0], sum(REGION_KEYPOINT_COUNTS))

    def test_inverse_letterbox_coordinates(self):
        original = torch.tensor([[100.0, 200.0, 1.0], [900.0, 700.0, 1.0]])
        letterboxed = original.clone()
        letterboxed[:, 0] = original[:, 0] * 0.64
        letterboxed[:, 1] = original[:, 1] * 0.64 + 64.0
        restored = ops.scale_coords(
            (640, 640), letterboxed, (800, 1000), ratio_pad=((0.64, 0.64), (0.0, 64.0))
        )
        torch.testing.assert_close(restored[:, :2], original[:, :2], rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
