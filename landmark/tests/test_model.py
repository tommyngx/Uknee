import unittest

import torch

from landmark.config.loader import ExperimentConfig
from landmark.losses import LandmarkLoss
from landmark.models.registry import build_model


def _small_config():
    config = ExperimentConfig()
    config.data.image_height = config.data.image_width = 64
    config.model.checkpoint = ""
    config.model.query_dim = 32
    config.model.attention_heads = 4
    config.model.transformer_ffn_dim = 64
    config.model.local_patch_size = 8
    return config


class ModelTests(unittest.TestCase):
    def test_kneepv1_outputs_are_snapped_to_same_bone_contour_tokens(self):
        config = _small_config()
        config.model.name = "kneepv1"
        config.model.transformer_layers = 1
        config.model.contour_tokens_per_bone = 16
        model = build_model(config)
        image = torch.randn(1, 1, 64, 64)
        outputs = model(image)

        self.assertEqual(outputs["final_landmarks"].shape, (1, 129, 2))
        self.assertEqual(
            outputs["contour_assignment_logits"].shape,
            (1, 129, 16),
        )
        candidates = outputs["contour_candidate_coordinates"]
        distance = torch.linalg.vector_norm(
            outputs["final_landmarks"][:, :, None] - candidates,
            dim=-1,
        )
        self.assertTrue(bool((distance.min(dim=-1).values <= 1.0e-6).all()))

        batch = {
            "landmarks": torch.rand(1, 129, 2),
            "landmark_visibility": torch.ones(1, 129),
        }
        losses = LandmarkLoss(config.loss)(outputs, batch)
        losses["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.backbone_adapter.backbone.parameters()
            )
        )
        self.assertTrue(model.landmark_queries.weight.grad is not None)

    def test_adaptive_model_shapes_ranges_and_frozen_backbone(self):
        config = _small_config()
        model = build_model(config)
        image = torch.randn(1, 1, 64, 64)
        outputs = model(image)
        self.assertEqual(outputs["segmentation_logits"].shape, (1, 11, 64, 64))
        self.assertEqual(outputs["coarse_landmarks"].shape, (1, 129, 2))
        self.assertEqual(outputs["final_landmarks"].shape, (1, 129, 2))
        self.assertEqual(outputs["landmark_confidence"].shape, (1, 129))
        self.assertEqual(outputs["local_heatmaps"].shape, (1, 129, 8, 8))
        self.assertTrue(
            bool(((outputs["coarse_landmarks"] >= 0) & (outputs["coarse_landmarks"] <= 1)).all())
        )
        self.assertTrue(
            bool(((outputs["final_landmarks"] >= 0) & (outputs["final_landmarks"] <= 1)).all())
        )

        batch = {
            "landmarks": torch.rand(1, 129, 2),
            "landmark_visibility": torch.ones(1, 129),
        }
        LandmarkLoss(config.loss)(outputs, batch)["loss"].backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.backbone_adapter.backbone.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.coarse_reference_head.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.backbone_adapter.mid_projection.parameters()
            )
        )

    def test_adapter_segmentation_matches_original_forward(self):
        config = _small_config()
        model = build_model(config).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = model.backbone_adapter.backbone(image)
            actual = model.backbone_adapter(image)["segmentation_logits"]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
