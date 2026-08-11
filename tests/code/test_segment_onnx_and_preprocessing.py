from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segment.dataloader.augment import build_train_transform, build_val_transform
from segment.dataloader.image_io import read_rgb_image
from segment.deploy.app_function import predict_mask
from segment.utils.onnx_export import export_segment_onnx, onnx_filename, read_onnx_metadata
from segment.utils.preprocessing import letterbox_array, restore_letterbox_mask


class _ToySegment(torch.nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.head = torch.nn.Conv2d(3, classes, kernel_size=1)

    def forward(self, images):
        return self.head(images)


class _AlwaysClassTwo(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.shape[0], 3, images.shape[2], images.shape[3], device=images.device)
        logits[:, 2] = 5.0
        return logits


class SegmentONNXAndPreprocessingTests(unittest.TestCase):
    def test_onnx_filename_uses_portable_underscores(self):
        self.assertEqual(onnx_filename("RWKV-UNet V6"), "rwkv_unet_v6.onnx")

    def test_letterbox_preserves_aspect_and_restores_mask(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        canvas, transform = letterbox_array(image, (256, 256))
        self.assertEqual(canvas.shape, (256, 256, 3))
        self.assertEqual((transform.resized_height, transform.resized_width), (128, 256))
        self.assertTrue(np.all(canvas[:64] == 0))
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[20:80, 40:160] = 7
        letterboxed_mask, mask_transform = letterbox_array(
            mask, (256, 256), interpolation=cv2.INTER_NEAREST
        )
        restored = restore_letterbox_mask(letterboxed_mask, mask_transform)
        self.assertEqual(restored.shape, mask.shape)
        self.assertEqual(set(np.unique(restored)), {0, 7})

    def test_albumentations_xray_policy_has_no_flip_or_cutout(self):
        policy = build_train_transform(256, strategy="xray")
        policy_text = repr(policy).lower()
        for forbidden in ("horizontalflip", "verticalflip", "coarsedropout", "cutout"):
            self.assertNotIn(forbidden, policy_text)
        sample = build_val_transform(256)(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            mask=np.zeros((100, 200), dtype=np.uint8),
        )
        self.assertEqual(sample["image"].shape, (256, 256, 3))

        source = np.random.default_rng(2006).integers(0, 256, (101, 237, 3), dtype=np.uint8)
        train_contract = build_val_transform(256)(image=source)["image"]
        deploy_contract, _ = letterbox_array(source, (256, 256))
        np.testing.assert_array_equal(train_contract, deploy_contract)

    def test_segment_decoder_exposes_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixel.png"
            bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), bgr))
            rgb = read_rgb_image(path)
            self.assertEqual(rgb[0, 0].tolist(), [30, 20, 10])

    def test_multiclass_deploy_keeps_raw_class_ids_and_source_shape(self):
        runtime = {
            "model": _AlwaysClassTwo().eval(),
            "config": SimpleNamespace(img_size=64, input_channel=3),
            "device": torch.device("cpu"),
            "threshold": 0.5,
            "preprocess": {"mean": None, "std": None, "source": "scale_0_1"},
        }
        source = Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8), mode="RGB")
        mask = predict_mask(runtime, source, return_pil=False, resize_back=True)
        self.assertEqual(mask.shape, (20, 40))
        self.assertEqual(set(np.unique(mask)), {2})

    def test_onnx_contains_preprocess_metadata_and_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.onnx"
            args = SimpleNamespace(model="RWKV_UNetV6", img_size=32, input_channel=3, num_classes=3)
            record = export_segment_onnx(
                _ToySegment().eval(),
                args,
                path,
                class_names=["background", "femur", "tibia"],
            )
            metadata = read_onnx_metadata(path)
            preprocess = json.loads(metadata["uknee.preprocess"])
            self.assertEqual(record["status"], "ready")
            self.assertTrue(record["parity"]["validated"])
            self.assertEqual(preprocess["color_space"], "RGB")
            self.assertEqual(preprocess["resize"]["mode"], "letterbox")
            self.assertEqual(json.loads(metadata["uknee.class_names"]), ["background", "femur", "tibia"])


if __name__ == "__main__":
    unittest.main()
