import unittest

import torch

from landmark.config.loader import ExperimentConfig
from landmark.losses import LandmarkLoss
from landmark.models.metadata import LANDMARK_PATH_RANGES, TOPOLOGY_EDGES
from landmark.models.registry import build_model
from landmark.models.vitpose import _sincos_position_2d


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
    def test_heatmap_baselines_produce_normalized_landmarks(self):
        for name in ("hrnet", "vitpose"):
            with self.subTest(model=name):
                config = _small_config()
                config.model.name = name
                config.model.input_channels = 1
                config.model.query_dim = 32
                config.model.attention_heads = 4
                config.model.vit_depth = 1
                config.model.vit_patch_size = 16
                config.model.hrnet_width = 8
                outputs = build_model(config).eval()(torch.randn(1, 1, 64, 64))
                self.assertEqual(outputs["final_landmarks"].shape, (1, 129, 2))
                self.assertEqual(outputs["global_heatmaps"].shape[:2], (1, 129))
                self.assertTrue(
                    bool(
                        (
                            (outputs["final_landmarks"] >= 0)
                            & (outputs["final_landmarks"] <= 1)
                        ).all()
                    )
                )

    def test_topology_paths_do_not_join_independent_tibial_contours(self):
        self.assertNotIn((85, 86), TOPOLOGY_EDGES)
        self.assertNotIn((90, 91), TOPOLOGY_EDGES)
        self.assertEqual(len(LANDMARK_PATH_RANGES), 6)
        self.assertEqual(len(TOPOLOGY_EDGES), 123)

    def test_kneepv2_adds_topology_and_unique_path_assignments(self):
        config = _small_config()
        config.model.name = "kneepv2"
        config.model.transformer_layers = 1
        config.model.topology_mixer_layers = 1
        config.model.contour_tokens_per_bone = 64
        model = build_model(config).eval()
        image = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            outputs = model(image)

        self.assertEqual(outputs["topology_soft_landmarks"].shape, (1, 129, 2))
        indices = outputs["contour_assignment_indices"][0]
        for start, stop in LANDMARK_PATH_RANGES:
            self.assertEqual(indices[start:stop].unique().numel(), stop - start)

    def test_kneepv2_topology_modules_receive_gradients(self):
        config = _small_config()
        config.model.name = "kneepv2"
        config.model.transformer_layers = 1
        config.model.topology_mixer_layers = 1
        config.model.contour_tokens_per_bone = 64
        config.loss.topology_edge_weight = 0.25
        config.loss.topology_curvature_weight = 0.1
        config.loss.topology_duplicate_weight = 0.05
        model = build_model(config)
        outputs = model(torch.randn(1, 1, 64, 64))
        batch = {
            "landmarks": torch.rand(1, 129, 2),
            "landmark_visibility": torch.ones(1, 129),
        }
        losses = LandmarkLoss(config.loss)(outputs, batch)
        losses["loss"].backward()

        self.assertTrue(model.path_embedding.weight.grad is not None)
        self.assertTrue(model.path_mixers[0].depthwise.weight.grad is not None)

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
        self.assertEqual(outputs["coarse_landmark_confidence"].shape, (1, 129))
        self.assertEqual(outputs["coarse_heatmaps"].shape, (1, 129, 8, 8))
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

    def test_vitpose_position_encoding_distinguishes_raster_locations(self):
        encoding = _sincos_position_2d(
            4,
            5,
            32,
            torch.float32,
            torch.device("cpu"),
        )
        self.assertEqual(encoding.shape, (1, 20, 32))
        self.assertFalse(torch.equal(encoding[:, 0], encoding[:, 1]))
        self.assertFalse(torch.equal(encoding[:, 0], encoding[:, 5]))

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
