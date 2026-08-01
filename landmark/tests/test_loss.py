import unittest

import torch

from landmark.config.loader import LossConfig
from landmark.losses import LandmarkLoss


class LossTests(unittest.TestCase):
    @staticmethod
    def _ordered_landmarks() -> torch.Tensor:
        target = torch.zeros(1, 129, 2)
        ranges = ((0, 45), (45, 86), (86, 91), (91, 96), (96, 120), (120, 129))
        for path_id, (start, stop) in enumerate(ranges):
            target[0, start:stop, 0] = torch.linspace(0.1, 0.9, stop - start)
            target[0, start:stop, 1] = 0.1 + path_id * 0.1
        return target

    def test_topology_edge_loss_penalizes_reversed_path(self):
        config = LossConfig(
            coarse_weight=0,
            coordinate_weight=0,
            heatmap_weight=0,
            bone_constraint_weight=0,
            topology_edge_weight=1,
        )
        criterion = LandmarkLoss(config)
        target = self._ordered_landmarks()
        reversed_prediction = target.clone()
        reversed_prediction[:, :45] = reversed_prediction[:, :45].flip(1)
        batch = {
            "landmarks": target,
            "landmark_visibility": torch.ones(1, 129),
        }

        def outputs(prediction):
            return {
                "coarse_landmarks": prediction,
                "final_landmarks": prediction,
                "topology_soft_landmarks": prediction,
            }

        correct = criterion(outputs(target), batch)
        reversed_loss = criterion(outputs(reversed_prediction), batch)
        self.assertEqual(correct["topology_edge_loss"].item(), 0)
        self.assertGreater(
            reversed_loss["topology_edge_loss"].item(),
            correct["topology_edge_loss"].item(),
        )

    def test_topology_duplicate_loss_penalizes_shared_token(self):
        config = LossConfig(
            coarse_weight=0,
            coordinate_weight=0,
            heatmap_weight=0,
            bone_constraint_weight=0,
            topology_duplicate_weight=1,
        )
        criterion = LandmarkLoss(config)
        target = self._ordered_landmarks()
        visibility = torch.ones(1, 129)
        distinct = torch.zeros(1, 129, 64)
        shared = torch.zeros_like(distinct)
        for start, stop in ((0, 45), (45, 86), (86, 91), (91, 96), (96, 120), (120, 129)):
            distinct[0, start:stop, : stop - start] = torch.eye(stop - start)
            shared[0, start:stop, 0] = 1

        def outputs(probabilities):
            return {
                "coarse_landmarks": target,
                "final_landmarks": target,
                "topology_soft_landmarks": target,
                "contour_assignment_probabilities": probabilities,
            }

        batch = {"landmarks": target, "landmark_visibility": visibility}
        distinct_loss = criterion(outputs(distinct), batch)
        shared_loss = criterion(outputs(shared), batch)
        self.assertEqual(distinct_loss["topology_duplicate_loss"].item(), 0)
        self.assertGreater(
            shared_loss["topology_duplicate_loss"].item(),
            distinct_loss["topology_duplicate_loss"].item(),
        )

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

    def test_local_heatmap_target_does_not_backpropagate_to_reference(self):
        criterion = LandmarkLoss(LossConfig())
        reference = torch.full((1, 129, 2), 0.5, requires_grad=True)
        heatmaps = torch.randn(1, 129, 8, 8, requires_grad=True)
        outputs = {
            "local_heatmaps": heatmaps,
            "refinement_reference": reference,
            "local_patch_radius_xy": torch.full((1, 129, 2), 0.25),
        }
        target = torch.full((1, 129, 2), 0.55)
        visibility = torch.ones(1, 129)
        criterion._local_heatmap_loss(outputs, target, visibility).backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(heatmaps.grad)

    def test_bone_constraint_is_finite_and_bounded(self):
        criterion = LandmarkLoss(LossConfig(bone_constraint_weight=1.0))
        outputs = {
            "final_landmarks": torch.full((1, 129, 2), 0.5),
            "bone_probabilities": torch.zeros(1, 4, 8, 8),
        }
        loss = criterion._bone_constraint_loss(outputs, torch.ones(1, 129))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0)
        self.assertLessEqual(loss.item(), 1)

    def test_coarse_heatmap_loss_is_used_when_enabled(self):
        config = LossConfig(
            coarse_weight=0,
            coarse_heatmap_weight=1,
            coordinate_weight=0,
            heatmap_weight=0,
            bone_constraint_weight=0,
        )
        prediction = torch.zeros(1, 129, 2)
        outputs = {
            "coarse_landmarks": prediction,
            "final_landmarks": prediction,
            "coarse_heatmaps": torch.zeros(1, 129, 8, 8),
        }
        batch = {
            "landmarks": torch.rand(1, 129, 2),
            "landmark_visibility": torch.ones(1, 129),
        }
        losses = LandmarkLoss(config)(outputs, batch, phase="coarse")
        self.assertGreater(losses["coarse_heatmap_loss"].item(), 0)
        self.assertEqual(losses["loss"].item(), losses["coarse_heatmap_loss"].item())


if __name__ == "__main__":
    unittest.main()
