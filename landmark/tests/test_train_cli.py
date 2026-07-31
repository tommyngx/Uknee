import sys
import unittest
from unittest.mock import patch

from landmark.config.loader import ExperimentConfig
from landmark.train import apply_cli_overrides, parse_args


class TrainCliTests(unittest.TestCase):
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
