from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from landmark.train import _parse_overrides, build_parser, main, resolve_model_source
from landmark.core.config import get_cfg
from landmark.utils.api import KneePose


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

    def test_api_never_overwrites_resolved_model_with_default_null(self):
        model_path = resolve_model_source("yolo26-pose-v9")
        pose = KneePose(model_path)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.yaml"
            source.write_text(yaml.safe_dump({"dataset_name": "test"}), encoding="utf-8")
            prepared = SimpleNamespace(
                yaml_path=source,
                source_yaml=source,
                root=root,
            )
            with (
                patch.object(pose, "_prepare_data", return_value=prepared),
                patch.object(pose.backend, "train", return_value="ok") as backend_train,
            ):
                self.assertEqual(pose.train(data=source, epochs=1), "ok")
            self.assertEqual(backend_train.call_args.kwargs["model"], str(model_path))

    def test_validator_accepts_runtime_save_dir(self):
        args = get_cfg(overrides={"save_dir": "/tmp/uknee-run"})
        self.assertEqual(args.save_dir, "/tmp/uknee-run")

    def test_explicit_base_lr_is_not_discarded_by_auto_optimizer(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            dataset = project / "data" / "pose"
            dataset.mkdir(parents=True)
            (dataset / "data.yaml").write_text(
                yaml.safe_dump({"dataset_name": "pose"}), encoding="utf-8"
            )
            with patch("landmark.train.KneePose") as pose_class:
                pose_class.return_value.train.return_value = "ok"
                result = main(
                    [
                        "--model", "yolo26-pose-v9",
                        "--project", str(project),
                        "--dataset", "/pose",
                        "--base_lr", "0.001",
                    ]
                )
            self.assertEqual(result, "ok")
            kwargs = pose_class.return_value.train.call_args.kwargs
            self.assertEqual(kwargs["lr0"], 0.001)
            self.assertEqual(kwargs["optimizer"], "AdamW")

    def test_native_key_value_overrides(self):
        self.assertEqual(
            _parse_overrides(["optimizer=AdamW", "lr0=0.001", "cos_lr=true"]),
            {"optimizer": "AdamW", "lr0": 0.001, "cos_lr": True},
        )


if __name__ == "__main__":
    unittest.main()
