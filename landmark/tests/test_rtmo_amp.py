from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from landmark import KneePose
from landmark.models.heatmap_adapter import _rtmo_spatial_classification_loss


ROOT = Path(__file__).resolve().parents[1]


class RTMOAutomaticMixedPrecisionTests(unittest.TestCase):
    @staticmethod
    def _pose_batch() -> dict[str, torch.Tensor]:
        keypoints = torch.zeros(4, 51, 3)
        for class_id, count in enumerate((45, 51, 24, 9)):
            keypoints[class_id, :count, 0] = torch.linspace(0.2, 0.8, count)
            keypoints[class_id, :count, 1] = 0.3 + class_id * 0.1
            keypoints[class_id, :count, 2] = 2
        return {
            "img": torch.rand(1, 3, 64, 64),
            "keypoints": keypoints,
            "batch_idx": torch.zeros(4),
            "cls": torch.arange(4).view(-1, 1).float(),
            "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.5]]).repeat(4, 1),
        }

    def test_rtmo_loss_does_not_call_amp_unsafe_probability_bce(self):
        model = KneePose(ROOT / "cfg" / "models" / "rtmo-pose.yaml").model.train()
        with patch(
            "landmark.models.heatmap_adapter.F.binary_cross_entropy",
            side_effect=AssertionError("probability BCE is unsafe under autocast"),
        ):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss, items = model(self._pose_batch())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(items).all())
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_rtmo_auxiliary_logits_match_canonical_layout(self):
        model = KneePose(ROOT / "cfg" / "models" / "rtmo-pose.yaml").model.network.eval()
        with torch.no_grad():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                predictions = model(torch.zeros(2, 3, 64, 96), return_aux=True)
        self.assertEqual(tuple(predictions["visibility_logits"].shape), (2, 129))
        self.assertEqual(tuple(predictions["region_logits"].shape), (2, 4))
        self.assertEqual(predictions["candidate_logits"].shape[0], 2)
        self.assertEqual(predictions["candidate_logits"].shape[2], 4)
        self.assertEqual(
            predictions["candidate_logits"].shape[1], predictions["candidate_grids"].shape[0]
        )
        self.assertEqual(predictions["dcc"]["x_log_probability"].dtype, torch.float32)
        self.assertEqual(predictions["dcc"]["y_log_probability"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(predictions["dcc"]["x_log_probability"]).all())
        self.assertTrue(torch.isfinite(predictions["dcc"]["y_log_probability"]).all())
        self.assertTrue(
            torch.allclose(
                predictions["visibility_logits"].sigmoid().float(), predictions["canonical"][..., 2].float()
            )
        )

    def test_float16_probability_product_reproduces_the_old_mle_underflow(self):
        probability = torch.tensor([1e-5], dtype=torch.float16)
        old_joint_probability = (probability * probability).clamp_min(1e-9)
        stable_log_probability = probability.float().log() * 2

        self.assertEqual(old_joint_probability.item(), 0.0)
        self.assertFalse(torch.isfinite(old_joint_probability.log()).item())
        self.assertTrue(torch.isfinite(stable_log_probability).item())

    def test_rtmo_spatial_loss_rewards_the_candidate_nearest_the_region(self):
        grids = torch.tensor([[0.2, 0.2], [0.8, 0.8]])
        boxes = torch.tensor([[[0.2, 0.2, 0.2, 0.2]]])
        present = torch.ones(1, 1, dtype=torch.bool)
        correct = torch.tensor([[[6.0], [-6.0]]])
        reversed_logits = correct.flip(1)
        uniform = torch.zeros(1, 2, 1, requires_grad=True)

        correct_loss = _rtmo_spatial_classification_loss(correct, grids, boxes, present)
        reversed_loss = _rtmo_spatial_classification_loss(reversed_logits, grids, boxes, present)
        _rtmo_spatial_classification_loss(uniform, grids, boxes, present).backward()

        self.assertLess(correct_loss, reversed_loss)
        self.assertLess(uniform.grad[0, 0, 0], 0)
        self.assertGreater(uniform.grad[0, 1, 0], 0)


if __name__ == "__main__":
    unittest.main()
