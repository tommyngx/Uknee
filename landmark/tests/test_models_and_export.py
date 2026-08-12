from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from landmark import KneePose
from landmark.nn.autobackend import AutoBackend
from landmark.utils.exporting import KneePoseExportWrapper
from landmark.core.config import get_cfg


ROOT = Path(__file__).resolve().parents[1]


class ModelAndExportTests(unittest.TestCase):
    @staticmethod
    def _pose_batch():
        keypoints = torch.zeros(4, 51, 3)
        for class_id, count in enumerate((45, 51, 24, 9)):
            keypoints[class_id, :count, 0] = torch.linspace(0.2, 0.8, count)
            keypoints[class_id, :count, 1] = 0.3 + class_id * 0.1
            keypoints[class_id, :count, 2] = 2
        return {
            "img": torch.rand(1, 3, 64, 64),
            "keypoints": keypoints,
            "batch_idx": torch.zeros(4),
            "cls": torch.arange(4).view(-1, 1).float(),
            "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.5]]).repeat(4, 1),
        }

    def test_three_yolo_heads_build_with_configured_default_scale(self):
        expected = {
            "yolo26-pose.yaml": ("Pose26", "n"),
            "yolo26-pose-v1.yaml": ("OA26HeatmapPose", "n"),
            "yolo26-pose-v9.yaml": ("OA26RegionRefinePose", "m"),
        }
        for filename, (head_name, scale) in expected.items():
            wrapper = KneePose(ROOT / "cfg" / "models" / filename)
            self.assertEqual(type(wrapper.model.model[-1]).__name__, head_name)
            self.assertEqual(wrapper.model.yaml["scale"], scale)

    def test_pt_backend_casts_loaded_checkpoint_to_requested_precision(self):
        model = torch.nn.Conv2d(3, 4, 1)
        model.names = {0: "landmark"}
        model.stride = torch.tensor([32])

        with patch("landmark.nn.tasks.load_checkpoint", return_value=(model, {})):
            backend = AutoBackend("best.pt", device=torch.device("cpu"), fp16=True, fuse=False)

        self.assertTrue(backend.fp16)
        self.assertEqual(backend.model.weight.dtype, torch.float16)
        self.assertEqual(backend.model.bias.dtype, torch.float16)
        output = backend.model(torch.zeros(1, 3, 8, 8, dtype=torch.float16))
        self.assertEqual(output.dtype, torch.float16)

    def test_heatmap_models_have_full_architectures_and_gradients(self):
        for filename, architecture in (
            ("hrnet-w32-pose.yaml", "HRNetW32"),
            ("vitpose-s-pose.yaml", "ViTPoseS"),
        ):
            wrapper = KneePose(ROOT / "cfg" / "models" / filename)
            model = wrapper.model.train()
            self.assertEqual(type(model.network).__name__, architecture)
            keypoints = torch.zeros(4, 51, 3)
            for class_id, count in enumerate((45, 51, 24, 9)):
                keypoints[class_id, :count, :2] = torch.rand(count, 2) * 0.8 + 0.1
                keypoints[class_id, :count, 2] = 2
            loss, items = model(
                {
                    "img": torch.rand(1, 3, 64, 64),
                    "keypoints": keypoints,
                    "batch_idx": torch.zeros(4),
                    "cls": torch.arange(4).view(-1, 1).float(),
                }
            )
            loss.backward()
            self.assertEqual(tuple(items.shape), (2,))
            self.assertTrue(all(torch.isfinite(items)))
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_hrnet_w48_and_vitpose_b_build_with_canonical_output(self):
        for filename, architecture, minimum_parameters in (
            ("hrnet-w48-pose.yaml", "HRNetW48", 60_000_000),
            ("vitpose-b-pose.yaml", "ViTPoseB", 85_000_000),
        ):
            wrapper = KneePose(ROOT / "cfg" / "models" / filename)
            model = wrapper.model.eval()
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 64, 96))
            self.assertEqual(type(model.network).__name__, architecture)
            self.assertEqual(tuple(output.shape), (1, 129, 3))
            self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), minimum_parameters)

    def test_rtmo_dcc_and_grouped_pose_branches_receive_gradients(self):
        wrapper = KneePose(ROOT / "cfg" / "models" / "rtmo-pose.yaml")
        self.assertEqual(wrapper.family, "rtmo")
        with torch.no_grad():
            rectangular_output = wrapper.model.eval()(torch.zeros(1, 3, 64, 96))
        self.assertEqual(tuple(rectangular_output.shape), (1, 129, 3))
        model = wrapper.model.train()
        loss, items = model(self._pose_batch())
        loss.backward()
        self.assertEqual(tuple(items.shape), (5,))
        self.assertTrue(torch.isfinite(loss))
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
        for prefix in (
            "network.backbone",
            "network.neck",
            "network.head.classification",
            "network.head.pose",
            "network.head.out_offset",
            "network.dcc.gau",
            "network.dcc.x_projection",
            "network.dcc.y_projection",
        ):
            self.assertTrue(
                any(value is not None for name, value in gradients.items() if name.startswith(prefix)),
                f"No RTMO gradient reached {prefix}",
            )

    def test_all_models_have_fixed_export_contract(self):
        filenames = (
            "yolo26-pose.yaml",
            "yolo26-pose-v1.yaml",
            "yolo26-pose-v9.yaml",
            "hrnet-w32-pose.yaml",
            "hrnet-w48-pose.yaml",
            "vitpose-s-pose.yaml",
            "vitpose-b-pose.yaml",
            "rtmo-pose.yaml",
        )
        for filename in filenames:
            wrapper = KneePose(ROOT / "cfg" / "models" / filename)
            export_model = KneePoseExportWrapper(wrapper.model.eval(), wrapper.family, confidence=0.0).eval()
            with torch.no_grad():
                detections, count, canonical = export_model(torch.zeros(1, 3, 64, 64))
            self.assertEqual(tuple(detections.shape), (1, 4, 159))
            self.assertEqual(tuple(count.shape), (1,))
            self.assertEqual(tuple(canonical.shape), (1, 129, 3))

    def test_v1_v9_auxiliary_and_refinement_branches_receive_gradients(self):
        required = {
            "yolo26-pose-v1.yaml": ("cv2", "cv3", "cv4_kpts", "hm_head"),
            "yolo26-pose-v9.yaml": (
                "cv2",
                "cv3",
                "cv4_kpts",
                "hm_head",
                "region_refine_head.roi_extractor",
                "region_refine_head.query_encoder",
                "region_refine_head.transformer",
                "region_refine_head.localization_head",
            ),
        }
        for filename, prefixes in required.items():
            model = KneePose(ROOT / "cfg" / "models" / filename).model.train()
            model.args = get_cfg(overrides={"task": "pose", "mode": "train", "imgsz": 64})
            model.criterion = model.init_criterion()
            for _ in range(50):
                model.criterion.update()
            loss, _ = model(self._pose_batch())
            loss.sum().backward()
            head = model.model[-1]
            parameter_grads = {name: value.grad for name, value in head.named_parameters()}
            for prefix in prefixes:
                self.assertTrue(
                    any(gradient is not None for name, gradient in parameter_grads.items() if name.startswith(prefix)),
                    f"{filename}: no gradient reached {prefix}",
                )

    def test_detection_only_phase_zeros_every_landmark_loss(self):
        model = KneePose(ROOT / "cfg" / "models" / "yolo26-pose-v1.yaml").model.train()
        model.args = get_cfg(overrides={"task": "pose", "mode": "train", "imgsz": 64})
        loss, reported = model(self._pose_batch())

        # V1 vector: box, pose, kobj, cls, dfl, heatmap, coord,
        # neighbour, curve. E2E returns the same layout after branch fusion.
        landmark_indices = [1, 2, *range(5, len(loss))]
        self.assertTrue(torch.equal(loss[landmark_indices], torch.zeros_like(loss[landmark_indices])))
        self.assertEqual(float(reported[1]), 0.0)
        self.assertEqual(float(reported[2]), 0.0)
        self.assertGreater(float(loss[[0, 3, 4]].sum().detach()), 0.0)

    def test_legacy_name_is_rejected_clearly(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            KneePose("adaptive_rwkv.pt")


if __name__ == "__main__":
    unittest.main()
