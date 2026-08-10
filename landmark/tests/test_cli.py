from __future__ import annotations

import unittest

from landmark.train import _parse_overrides, build_parser


class CliTests(unittest.TestCase):
    def test_documented_training_command_parses(self):
        args = build_parser().parse_args(
            [
                "--model", "landmark/cfg/models/yolo26-pose-v9.yaml",
                "--data", "landmark/cfg/datasets/mesko4gf2.yaml",
                "--epochs", "100", "--imgsz", "640", "--batch", "16",
                "--device", "0", "--project", "landmark/runs/pose", "--name", "pose-v9",
            ]
        )
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.name, "pose-v9")

    def test_native_key_value_overrides(self):
        self.assertEqual(
            _parse_overrides(["optimizer=AdamW", "lr0=0.001", "cos_lr=true"]),
            {"optimizer": "AdamW", "lr0": 0.001, "cos_lr": True},
        )


if __name__ == "__main__":
    unittest.main()
