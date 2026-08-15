from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import yaml
from PIL import Image

from landmark.train_det import (
    DEFAULT_CFG,
    DetectionReportTrainer,
    _load_yaml,
    _xray_detection_defaults,
    prepare_detection_dataset,
    write_detection_summary,
)
from landmark.core.detect import DetectionTrainer
from landmark.nn.modules.head import Detect
from landmark.utils.exporting import KneeDetectionExportWrapper


class _FakeDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.names = {0: "right_knee", 1: "left_knee"}
        self.model = torch.nn.ModuleList()
        self.stride = torch.tensor([32.0])

    def forward(self, images):
        rows = images.new_tensor(
            [
                [1.0, 2.0, 10.0, 12.0, 0.90, 0.0],
                [4.0, 5.0, 14.0, 16.0, 0.80, 1.0],
                [3.0, 3.0, 8.0, 8.0, 0.10, 0.0],
            ]
        )
        return rows.unsqueeze(0).expand(images.shape[0], -1, -1)


class DetectionTrainingTests(unittest.TestCase):
    def test_xray_defaults_do_not_corrupt_left_right_classes(self):
        defaults = _xray_detection_defaults()
        self.assertEqual(defaults["fliplr"], 0.0)
        self.assertEqual(defaults["flipud"], 0.0)
        self.assertTrue(defaults["plots"])

    def test_kneelocation_preset_reuses_detection_settings_safely(self):
        config = _load_yaml(DEFAULT_CFG)
        self.assertEqual(config["pretrained"], "yolo26m.pt")
        self.assertEqual(config["optimizer"], "auto")
        self.assertEqual(config["epochs"], 1000)
        self.assertEqual(config["patience"], 200)
        self.assertEqual(config["fliplr"], 0.0)
        self.assertNotIn("pose", config)

    def test_detection_export_contract_selects_one_box_per_class(self):
        wrapper = KneeDetectionExportWrapper(_FakeDetectionModel(), confidence=0.25).eval()
        images = torch.zeros(2, 3, 32, 32)
        detections, count, canonical = wrapper(images)
        self.assertEqual(tuple(detections.shape), (2, 3, 6))
        self.assertEqual(count.tolist(), [2, 2])
        self.assertEqual(tuple(canonical.shape), (2, 2, 4))
        self.assertEqual(canonical[0, 0].tolist(), [1.0, 2.0, 10.0, 12.0])
        self.assertEqual(canonical[0, 1].tolist(), [4.0, 5.0, 14.0, 16.0])

        graph = str(torch.jit.trace(wrapper, images).inlined_graph)
        self.assertNotIn("aten::index", graph)
        self.assertIn("aten::gather", graph)

    def test_end2end_topk_is_stable_for_saturated_pretrained_scores(self):
        head = Detect(nc=2, end2end=True, ch=(8,))
        head.export = True
        scores = torch.full((1, 4, 2), 1.0)

        selected_scores, classes, anchors = head.get_topk_index(scores, max_det=3)

        self.assertEqual(selected_scores.flatten().tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(classes.flatten().tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(anchors.flatten().tolist(), [0, 0, 1])

        class TopKGraph(torch.nn.Module):
            def __init__(self, detect_head):
                super().__init__()
                self.detect_head = detect_head

            def forward(self, values):
                return self.detect_head.get_topk_index(values, max_det=3)

        graph = str(torch.jit.trace(TopKGraph(head), scores).inlined_graph)
        self.assertNotIn("aten::index", graph)
        self.assertIn("aten::gather", graph)

    def test_pretrained_classifier_is_not_transferred_across_class_taxonomies(self):
        head = Detect(nc=2, end2end=True, ch=(8,))
        fake_model = SimpleNamespace(
            model=torch.nn.ModuleList([torch.nn.Identity(), head]),
            load=MagicMock(),
        )
        source = SimpleNamespace(names={0: "cat", 1: "dog"})
        trainer = object.__new__(DetectionTrainer)
        trainer.data = {"nc": 2, "channels": 3, "names": {0: "RightKnee", 1: "LeftKnee"}}

        with patch("landmark.core.detect.DetectionModel", return_value=fake_model):
            result = trainer.get_model(cfg="model.yaml", weights=source, verbose=False)

        self.assertIs(result, fake_model)
        excluded = fake_model.load.call_args.kwargs["exclude"]
        self.assertTrue(any(".cv3." in prefix for prefix in excluded))
        self.assertTrue(any(".one2one_cv3." in prefix for prefix in excluded))

    def test_pretrained_classifier_can_transfer_for_the_same_taxonomy(self):
        head = Detect(nc=2, end2end=True, ch=(8,))
        fake_model = SimpleNamespace(
            model=torch.nn.ModuleList([torch.nn.Identity(), head]),
            load=MagicMock(),
        )
        source = SimpleNamespace(names={0: "right_knee", 1: "left-knee"})
        trainer = object.__new__(DetectionTrainer)
        trainer.data = {"nc": 2, "channels": 3, "names": {0: "RightKnee", 1: "LeftKnee"}}

        with patch("landmark.core.detect.DetectionModel", return_value=fake_model):
            trainer.get_model(cfg="model.yaml", weights=source, verbose=False)

        self.assertEqual(fake_model.load.call_args.kwargs["exclude"], ())

    def test_detection_report_trainer_has_stable_ddp_import(self):
        from landmark.core.dist import generate_ddp_file

        trainer = object.__new__(DetectionReportTrainer)
        trainer.args = SimpleNamespace(model="model.yaml", augmentations=None)
        trainer.hub_session = SimpleNamespace(model_url="model.yaml")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("landmark.core.dist.USER_CONFIG_DIR", Path(directory)),
                patch.object(DetectionReportTrainer, "__module__", "__main__"),
            ):
                generated = Path(generate_ddp_file(trainer))
                content = generated.read_text(encoding="utf-8")
        self.assertIn("from landmark.train_det import DetectionReportTrainer", content)
        self.assertNotIn("from __main__ import DetectionReportTrainer", content)

    def test_detection_checkpoint_save_exports_first_best_then_waits_for_interval(self):
        trainer = object.__new__(DetectionReportTrainer)
        trainer.args = SimpleNamespace(auto_export_onnx=True, onnx_export_interval=10)
        trainer.best_fitness = trainer.fitness = 0.5
        trainer.epoch = 0
        trainer.epochs = 100
        with (
            patch("landmark.core.trainer.BaseTrainer.save_model", return_value=True) as save_checkpoint,
            patch.object(trainer, "_export_detection_onnx") as export_onnx,
        ):
            self.assertTrue(trainer.save_model())
            trainer.best_fitness = trainer.fitness = 0.6
            trainer.epoch = 1
            self.assertTrue(trainer.save_model())
        self.assertEqual(save_checkpoint.call_count, 2)
        export_onnx.assert_called_once_with()

    def test_detection_checkpoint_save_refreshes_pending_best_at_interval(self):
        trainer = object.__new__(DetectionReportTrainer)
        trainer.args = SimpleNamespace(auto_export_onnx=True, onnx_export_interval=10)
        trainer.best_fitness = trainer.fitness = 0.5
        trainer.epoch = 0
        trainer.epochs = 100
        with (
            patch("landmark.core.trainer.BaseTrainer.save_model", return_value=True),
            patch.object(trainer, "_export_detection_onnx") as export_onnx,
        ):
            trainer.save_model()
            trainer.best_fitness = trainer.fitness = 0.6
            trainer.epoch = 4
            trainer.save_model()
            trainer.fitness = 0.55
            trainer.epoch = 9
            trainer.save_model()
        self.assertEqual(export_onnx.call_count, 2)

    def test_ddp_validation_uses_rank_zero_plot_decision_on_every_rank(self):
        from landmark.core.detect import DetectionValidator

        rank_zero_validator = object.__new__(DetectionValidator)
        rank_zero_validator.args = SimpleNamespace(plots=True)
        rank_zero_validator.device = torch.device("cpu")
        rank_one_validator = object.__new__(DetectionValidator)
        rank_one_validator.args = SimpleNamespace(plots=True)
        rank_one_validator.device = torch.device("cpu")
        rank_zero_trainer = SimpleNamespace(
            world_size=2,
            stopper=SimpleNamespace(possible_stop=True),
            epoch=337,
            epochs=1000,
        )
        rank_one_trainer = SimpleNamespace(
            world_size=2,
            stopper=SimpleNamespace(possible_stop=False),
            epoch=337,
            epochs=1000,
        )
        wire = {}

        def send_rank_zero_decision(tensor, src):
            self.assertEqual(src, 0)
            wire["plots"] = int(tensor.item())

        def receive_rank_zero_decision(tensor, src):
            self.assertEqual(src, 0)
            tensor.fill_(wire["plots"])

        with (
            patch("landmark.core.validator.RANK", 0),
            patch("landmark.core.validator.dist.broadcast", side_effect=send_rank_zero_decision),
        ):
            rank_zero_validator._sync_training_plots(rank_zero_trainer)
        with (
            patch("landmark.core.validator.RANK", 1),
            patch("landmark.core.validator.dist.broadcast", side_effect=receive_rank_zero_decision),
        ):
            rank_one_validator._sync_training_plots(rank_one_trainer)

        self.assertTrue(rank_zero_validator.args.plots)
        self.assertTrue(rank_one_validator.args.plots)

    def test_ddp_parent_summary_uses_reloaded_best_model(self):
        with tempfile.TemporaryDirectory() as directory:
            save_dir = Path(directory)
            (save_dir / "results.csv").write_text(
                "epoch,metrics/mAP50-95(B)\n1,0.75\n", encoding="utf-8"
            )
            trainer = SimpleNamespace(
                model="landmark/cfg/models/yolo26-detect.yaml",
                validator=None,
                metrics=None,
                optimizer=None,
                args=SimpleNamespace(
                    imgsz=[32, 32], epochs=1, batch=4, seed=2026,
                    optimizer="AdamW", lr0=0.001, auto_export_onnx=False,
                ),
                device=torch.device("cuda:0"),
            )
            trained_model = torch.nn.Conv2d(3, 2, kernel_size=1)
            summary_path = write_detection_summary(
                save_dir,
                Path("yolo26-detect.yaml"),
                {"class_instances": {"right_knee": 1, "left_knee": 1}},
                trainer=trainer,
                trained_model=trained_model,
                validation_metrics={"metrics/mAP50-95(B)": 0.75},
            )
            summary = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["model"]["parameters"], 8)
            self.assertEqual(summary["training"]["optimizer"], "AdamW")
            self.assertEqual(summary["performance"]["best_checkpoint_validation"]["metrics/mAP50-95(B)"], 0.75)

    def test_detection_dataset_is_split_without_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            images = root / "images" / "train"
            labels = root / "labels" / "train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(12):
                image = Image.new("RGB", (32, 32), (index, index, index))
                image.save(images / f"case_{index}.png")
                (labels / f"case_{index}.txt").write_text(
                    f"{index % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                )
            # This exact duplicate must follow its source into the same split.
            (images / "duplicate.png").write_bytes((images / "case_0.png").read_bytes())
            (labels / "duplicate.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "images/train",
                        "val": "images/train",
                        "names": {0: "right_knee", 1: "left_knee"},
                        "val_fraction": 0.25,
                        "split_seed": 2026,
                    }
                ),
                encoding="utf-8",
            )

            resolved, audit = prepare_detection_dataset(source, Path(directory) / "output")
            metadata = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            train = set(Path(metadata["train"]).read_text(encoding="utf-8").splitlines())
            val = set(Path(metadata["val"]).read_text(encoding="utf-8").splitlines())

            self.assertFalse(train & val)
            self.assertEqual(len(train | val), 13)
            self.assertEqual(str(images / "case_0.png") in train, str(images / "duplicate.png") in train)
            self.assertEqual(audit["instances"], 13)


if __name__ == "__main__":
    unittest.main()
