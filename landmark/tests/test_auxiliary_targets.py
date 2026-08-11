from __future__ import annotations

import unittest

import torch
import landmark  # noqa: F401 - bootstrap the pinned backend before absolute imports

from landmark.core.targets import extract_canonical_image_keypoints
from landmark.core.loss import OA26HeatmapPoseLoss
from landmark.data.schema import class_path_masks


class AuxiliaryTargetTests(unittest.TestCase):
    def setUp(self):
        counts = (45, 51, 24, 9)
        self.keypoints = torch.zeros(4, 51, 3)
        for class_id, count in enumerate(counts):
            self.keypoints[class_id, :count, 0] = (class_id + 1) / 10
            self.keypoints[class_id, :count, 1] = (class_id + 2) / 10
            self.keypoints[class_id, :count, 2] = 2
        self.batch = {
            "keypoints": self.keypoints,
            "batch_idx": torch.zeros(4),
            "cls": torch.arange(4).view(-1, 1).float(),
        }

    def test_all_four_classes_populate_canonical_target(self):
        coordinates, valid = extract_canonical_image_keypoints(
            self.batch, 1, torch.tensor((100.0, 200.0)), torch.device("cpu"), torch.float32
        )
        self.assertEqual(int(valid.sum()), 129)
        self.assertTrue(torch.allclose(coordinates[0, 45], torch.tensor((40.0, 30.0))))
        self.assertTrue(torch.allclose(coordinates[0, 96], torch.tensor((60.0, 40.0))))
        self.assertTrue(torch.allclose(coordinates[0, 120], torch.tensor((80.0, 50.0))))

    def test_duplicate_class_fails_fast(self):
        duplicate = {key: value.clone() for key, value in self.batch.items()}
        duplicate["keypoints"] = torch.cat((duplicate["keypoints"], duplicate["keypoints"][:1]))
        duplicate["batch_idx"] = torch.cat((duplicate["batch_idx"], torch.zeros(1)))
        duplicate["cls"] = torch.cat((duplicate["cls"], torch.zeros(1, 1)))
        with self.assertRaisesRegex(ValueError, "disable mosaic"):
            extract_canonical_image_keypoints(
                duplicate, 1, torch.tensor((100.0, 200.0)), torch.device("cpu"), torch.float32
            )

    def test_six_paths_do_not_have_cross_boundary_edges(self):
        starts, ends, c0, c1, c2 = OA26HeatmapPoseLoss._path_indices(129, torch.device("cpu"))
        edges = set(zip(starts.tolist(), ends.tolist()))
        curves = set(zip(c0.tolist(), c1.tolist(), c2.tolist()))
        for boundary in (45, 86, 91, 96, 120):
            self.assertNotIn((boundary - 1, boundary), edges)
            self.assertFalse(any(a < boundary <= c for a, _, c in curves))

    def test_v9_local_tibia_masks_preserve_three_paths(self):
        tibia = torch.tensor([1])
        edge_mask = class_path_masks(tibia, order=2)[0]
        curve_mask = class_path_masks(tibia, order=3)[0]
        self.assertFalse(bool(edge_mask[40]))
        self.assertFalse(bool(edge_mask[45]))
        self.assertFalse(bool(curve_mask[39]))
        self.assertFalse(bool(curve_mask[40]))


if __name__ == "__main__":
    unittest.main()
