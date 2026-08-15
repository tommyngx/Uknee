from __future__ import annotations

import tempfile
import unittest
import json
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from landmark import KneePose
from landmark.core.exporter import portable_model_stem, read_onnx_metadata
from landmark.nn.tasks import LEGACY_LANDMARK2_MODULES, temporary_modules
from landmark.utils.exporting import KneePoseExportWrapper


ROOT = Path(__file__).resolve().parents[1]


class _DummyYoloPoseCore(nn.Module):
    def __init__(self):
        super().__init__()
        predictions = torch.zeros(1, 5, 159)
        predictions[0, :, 4] = torch.tensor([0.2, 0.9, 0.8, 0.7, 0.95])
        predictions[0, :, 5] = torch.tensor([0.0, 0.0, 1.0, 2.0, 9.0])
        predictions[0, :, 6] = torch.arange(5, dtype=torch.float32)
        self.register_buffer("predictions", predictions)
        self.model = nn.ModuleList([nn.Identity()])
        self.names = {0: "femur", 1: "tibia", 2: "fibula", 3: "patella"}
        self.stride = torch.tensor([32.0])

    def forward(self, images):
        return self.predictions.expand(images.shape[0], -1, -1)


class ExporterTests(unittest.TestCase):
    def test_pose_class_selection_avoids_onnx_advanced_indexing(self):
        wrapper = KneePoseExportWrapper(_DummyYoloPoseCore(), family="yolo").eval()
        image = torch.zeros(1, 3, 8, 8)
        traced = torch.jit.trace(wrapper, image, check_trace=False)
        node_kinds = {node.kind() for node in traced.inlined_graph.nodes()}
        self.assertNotIn("aten::index", node_kinds)

        detections, count, canonical = wrapper(image)
        self.assertEqual(tuple(detections.shape), (1, 4, 159))
        self.assertEqual(tuple(canonical.shape), (1, 129, 3))
        self.assertEqual(count.tolist(), [3])
        self.assertEqual(detections[0, 0, 6].item(), 1.0)
        self.assertTrue(torch.equal(detections[0, 3], torch.zeros(159)))

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
