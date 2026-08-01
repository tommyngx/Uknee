import unittest
from pathlib import Path

from landmark.config.loader import DataConfig
from landmark.data.yolo_pose import (
    POINT_BONE_IDS,
    _split_group_key,
    build_dataloaders,
    parse_yolo_pose_label,
)


class DatasetTests(unittest.TestCase):
    def test_missing_label_fails_instead_of_becoming_all_invisible(self):
        with self.assertRaises(FileNotFoundError):
            parse_yolo_pose_label(Path("/definitely/missing/label.txt"))

    def test_generated_split_keeps_flip_and_bilateral_case_together(self):
        config = DataConfig(
            yaml_path="Ref/yolo_mesko4GF2/data.yaml",
            num_workers=0,
        )
        train_loader, val_loader = build_dataloaders(config, batch_size=4)
        train_groups = {
            _split_group_key(path)
            for path in train_loader.dataset.image_paths
        }
        val_groups = {
            _split_group_key(path)
            for path in val_loader.dataset.image_paths
        }
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_reference_yolo_label_becomes_129_points(self):
        path = Path("Ref/yolo_mesko4GF2/labels/train/01-002-B_R.txt")
        landmarks, visibility = parse_yolo_pose_label(path)
        self.assertEqual(landmarks.shape, (129, 2))
        self.assertEqual(visibility.shape, (129,))
        self.assertEqual(int(visibility.sum()), 129)
        self.assertEqual(POINT_BONE_IDS.bincount().tolist(), [45, 51, 24, 9])


if __name__ == "__main__":
    unittest.main()
