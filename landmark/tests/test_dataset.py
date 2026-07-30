import unittest
from pathlib import Path

from landmark.data.yolo_pose import POINT_BONE_IDS, parse_yolo_pose_label


class DatasetTests(unittest.TestCase):
    def test_reference_yolo_label_becomes_129_points(self):
        path = Path("Ref/yolo_mesko4GF2/labels/train/01-002-B_R.txt")
        landmarks, visibility = parse_yolo_pose_label(path)
        self.assertEqual(landmarks.shape, (129, 2))
        self.assertEqual(visibility.shape, (129,))
        self.assertEqual(int(visibility.sum()), 129)
        self.assertEqual(POINT_BONE_IDS.bincount().tolist(), [45, 51, 24, 9])


if __name__ == "__main__":
    unittest.main()
