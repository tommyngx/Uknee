from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from landmark2 import KneePose


ROOT = Path(__file__).resolve().parents[1]


class ExporterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
