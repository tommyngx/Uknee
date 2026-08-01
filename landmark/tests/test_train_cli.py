import sys
import unittest
from unittest.mock import patch

from landmark.config.loader import ExperimentConfig, load_config
from landmark.train import (
    _topology_scale,
    _validate_training_config,
    apply_cli_overrides,
    parse_args,
)


class TrainCliTests(unittest.TestCase):
    def test_kneepv2_config_and_topology_ramp(self):
        config = load_config("landmark/config/kneepv2.yaml")
        self.assertEqual(config.model.name, "kneepv2")
        self.assertEqual(_topology_scale(4, 5, 15), 0)
        self.assertAlmostEqual(_topology_scale(5, 5, 15), 1 / 15)
        self.assertEqual(_topology_scale(19, 5, 15), 1)

    def test_project_defaults_are_11_class_640_xray(self):
        with patch.object(sys, "argv", ["landmark.train"]):
            args = parse_args()
        config = apply_cli_overrides(ExperimentConfig(), args)
        self.assertEqual(config.model.num_mask_classes, 11)
        self.assertEqual(config.data.image_height, 640)
        self.assertEqual(config.data.image_width, 640)
        self.assertEqual(config.training.seed, 2006)
        self.assertEqual(config.data.seed, 2006)
        self.assertEqual(config.data.aug_strategy, "xray")

    def test_omitted_cli_values_preserve_loaded_config(self):
        config = ExperimentConfig()
        config.model.num_mask_classes = 7
        config.training.seed = 99
        config.data.seed = 99
        config.data.image_height = config.data.image_width = 512
        config.data.aug_strategy = "basic"
        with patch.object(sys, "argv", ["landmark.train"]):
            args = parse_args()
        resolved = apply_cli_overrides(config, args)
        self.assertEqual(resolved.model.num_mask_classes, 7)
        self.assertEqual(resolved.training.seed, 99)
        self.assertEqual(resolved.data.image_height, 512)
        self.assertEqual(resolved.data.aug_strategy, "basic")

    def test_frozen_rwkv_requires_existing_segmentation_checkpoint(self):
        config = ExperimentConfig()
        with self.assertRaisesRegex(ValueError, "randomly initialized frozen"):
            _validate_training_config(config)

        config.model.checkpoint = "/definitely/missing/segmentation.pth"
        with self.assertRaises(FileNotFoundError):
            _validate_training_config(config)

        bundled = load_config("landmark/config/adaptive_rwkv.yaml")
        _validate_training_config(bundled)

    def test_missing_resume_checkpoint_fails_early(self):
        config = load_config("landmark/config/hrnet.yaml")
        config.training.resume = "/definitely/missing/landmark.pt"
        with self.assertRaisesRegex(FileNotFoundError, "Resume checkpoint"):
            _validate_training_config(config)

    def test_heatmap_baseline_configs_disable_irrelevant_coarse_loss(self):
        for name in ("hrnet", "vitpose"):
            config = load_config(f"landmark/config/{name}.yaml")
            self.assertEqual(config.loss.coarse_weight, 0)
            self.assertEqual(config.loss.heatmap_temperature, 1)

    def test_main_py_style_aliases_override_config(self):
        argv = [
            "landmark.train",
            "--output_dir",
            "/tmp/landmark-output",
            "--img_size",
            "512",
            "--aug_strategy",
            "none",
            "--base_lr",
            "0.0002",
            "--exp_name",
            "alias-test",
            "--num_mask_classes",
            "11",
            "--pretrained_path",
            "/tmp/segmentation.pth",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        config = apply_cli_overrides(ExperimentConfig(), args)
        self.assertEqual(config.training.output_dir, "/tmp/landmark-output")
        self.assertEqual(config.data.image_height, 512)
        self.assertFalse(config.data.augment)
        self.assertEqual(config.training.learning_rate, 0.0002)
        self.assertEqual(config.training.experiment_name, "alias-test")
        self.assertEqual(config.model.checkpoint, "/tmp/segmentation.pth")


if __name__ == "__main__":
    unittest.main()
