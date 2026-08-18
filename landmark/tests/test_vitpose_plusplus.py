from __future__ import annotations

import unittest
from pathlib import Path

import torch

from landmark import KneePose
from landmark.models.vitpose_plusplus import MixtureOfExpertsMLP
from landmark.utils.exporting import KneePoseExportWrapper


ROOT = Path(__file__).resolve().parents[1]


class ViTPosePlusPlusTests(unittest.TestCase):
    def test_router_and_explicit_expert_preserve_shape(self):
        module = MixtureOfExpertsMLP(channels=32, num_experts=3, part_features=8, ratio=2)
        inputs = torch.randn(2, 5, 32)
        self.assertEqual(tuple(module(inputs).shape), tuple(inputs.shape))
        self.assertEqual(tuple(module(inputs, torch.tensor([0, 2])).shape), tuple(inputs.shape))

    def test_small_yaml_builds_and_all_experts_receive_gradients(self):
        wrapper = KneePose(ROOT / "cfg" / "models" / "vitpose-s-plusplus.yaml")
        model = wrapper.model.train()
        self.assertEqual(type(model.network).__name__, "ViTPosePlusPlusS")
        heatmaps = model.network(torch.rand(1, 3, 32, 32))
        heatmaps.mean().backward()
        self.assertEqual(tuple(heatmaps.shape), (1, 129, 8, 8))
        for block in model.network.blocks:
            self.assertTrue(all(expert.weight.grad is not None for expert in block.mlp.experts))

    def test_base_yaml_builds_with_canonical_output(self):
        wrapper = KneePose(ROOT / "cfg" / "models" / "vitpose-b-plusplus.yaml")
        model = wrapper.model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 32, 48))
        self.assertEqual(type(model.network).__name__, "ViTPosePlusPlusB")
        self.assertEqual(tuple(output.shape), (1, 129, 3))

    def test_small_model_has_fixed_export_contract(self):
        wrapper = KneePose(ROOT / "cfg" / "models" / "vitpose-s-plusplus.yaml")
        export_model = KneePoseExportWrapper(wrapper.model.eval(), wrapper.family, confidence=0.0).eval()
        with torch.no_grad():
            detections, count, canonical = export_model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(tuple(detections.shape), (1, 4, 159))
        self.assertEqual(tuple(count.shape), (1,))
        self.assertEqual(tuple(canonical.shape), (1, 129, 3))


if __name__ == "__main__":
    unittest.main()
