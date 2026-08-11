from __future__ import annotations

import unittest

from landmark.train import _parse_overrides, build_parser, resolve_model_source


class CliTests(unittest.TestCase):
    def test_documented_training_command_parses(self):
        args = build_parser().parse_args(
            [
                "--model", "landmark/cfg/models/yolo26-pose-v9.yaml",
                "--data", "landmark/cfg/datasets/mesko4gf2.yaml",
                "--epochs", "100", "--imgsz", "540x640", "--batch", "16",
                "--gpu", "[0,1]", "--name", "pose_rect",
            ]
        )
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.imgsz, [540, 640])
        self.assertEqual(args.gpu, [0, 1])
        self.assertEqual(args.name, "pose_rect")

    def test_short_model_name_resolves_from_registry_folder(self):
        self.assertEqual(resolve_model_source("yolo26-pose-v9").name, "yolo26-pose-v9.yaml")

    def test_native_key_value_overrides(self):
        self.assertEqual(
            _parse_overrides(["optimizer=AdamW", "lr0=0.001", "cos_lr=true"]),
            {"optimizer": "AdamW", "lr0": 0.001, "cos_lr": True},
        )


if __name__ == "__main__":
    unittest.main()
