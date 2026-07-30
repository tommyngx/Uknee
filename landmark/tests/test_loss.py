import unittest

import torch

from landmark.config.loader import LossConfig
from landmark.losses import LandmarkLoss


class LossTests(unittest.TestCase):
    def test_invisible_landmarks_do_not_change_coordinate_loss(self):
        target = torch.zeros(1, 129, 2)
        prediction = target.clone()
        prediction[:, 0] = 1
        outputs = {
            "coarse_landmarks": prediction,
            "final_landmarks": prediction,
            "landmark_confidence": torch.ones(1, 129),
        }
        batch = {
            "landmarks": target,
            "landmark_visibility": torch.ones(1, 129),
        }
        batch["landmark_visibility"][:, 0] = 0
        losses = LandmarkLoss(LossConfig())(outputs, batch)
        self.assertEqual(losses["coarse_loss"].item(), 0)
        self.assertEqual(losses["coordinate_loss"].item(), 0)


if __name__ == "__main__":
    unittest.main()
