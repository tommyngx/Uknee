from __future__ import annotations

import tempfile
import unittest
import json
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from landmark import KneePose
from landmark.core.exporter import portable_model_stem, read_onnx_metadata
from landmark.nn.tasks import LEGACY_LANDMARK2_MODULES, temporary_modules


ROOT = Path(__file__).resolve().parents[1]


class ExporterTests(unittest.TestCase):
    def test_landmark2_checkpoint_aliases_are_scoped(self):
        self.assertNotIn("landmark2.models.rtmo", sys.modules)
        with temporary_modules(LEGACY_LANDMARK2_MODULES):
            legacy = importlib.import_module("landmark2.models.rtmo")
            current = importlib.import_module("landmark.models.rtmo")
            self.assertIs(legacy, current)
        self.assertNotIn("landmark2.models.rtmo", sys.modules)

    def test_default_export_name_uses_portable_underscores(self):
        self.assertEqual(portable_model_stem("yolo26-pose-v9.yaml"), "yolo26_pose_v9")

    def test_torchscript_round_trip_keeps_fixed_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pose.torchscript"
            exported = KneePose(ROOT / "cfg/models/yolo26-pose.yaml").export(
                format="torchscript", imgsz=64, path=destination
            )
            model = torch.jit.load(str(exported), map_location="cpu")
            with torch.no_grad():
                outputs = model(torch.zeros(1, 3, 64, 64))
            self.assertEqual([tuple(value.shape) for value in outputs], [(1, 4, 159), (1,), (1, 129, 3)])

    def test_onnx_round_trip_embeds_rgb_letterbox_contract(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "yolo26_pose.onnx"
            exported = KneePose(ROOT / "cfg/models/yolo26-pose.yaml").export(
                format="onnx", imgsz=64, path=destination, model_name="yolo26-pose"
            )
            metadata = read_onnx_metadata(exported)
            preprocess = json.loads(metadata["uknee.preprocess"])
            outputs = ort.InferenceSession(
                str(exported), providers=["CPUExecutionProvider"]
            ).run(None, {"images": np.zeros((1, 3, 64, 64), dtype=np.float32)})
            self.assertEqual([tuple(value.shape) for value in outputs], [(1, 4, 159), (1,), (1, 129, 3)])
            self.assertEqual(metadata["uknee.model_name"], "yolo26-pose")
            self.assertEqual(preprocess["color_space"], "RGB")
            self.assertEqual(preprocess["resize"]["mode"], "letterbox")
            self.assertTrue(json.loads(metadata["uknee.parity"])["validated"])


if __name__ == "__main__":
    unittest.main()
