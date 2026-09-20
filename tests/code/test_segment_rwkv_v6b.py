import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from segment.dataloader.osteophyte import remap_osteophyte_mask
from segment.cli import parse_segment_args
from segment.main import _build_criterion, _load_model_state_dict
from segment.models import MODEL_REGISTRY, build_model, load_model_id
from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6b import RWKV_UNetV6b
from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6a import RWKV_UNetV6a
from segment.utils.onnx_export import export_segment_onnx, read_onnx_metadata

TINY_KWARGS = dict(
    input_channels=1, num_classes=5, stem_dim=4, depths=(1, 1, 1, 1),
    embed_dims=(4, 8, 12, 16), exp_ratios=(1.0, 1.0, 1.0, 1.0),
    num_heads=(1, 1, 1, 1), matrix_state_stages=(), drop_path_rate=0.0,
    matrix_state_backend="reference", source_size=(64, 40),
    global_size=(48, 32), local_size=32,
)


class RWKVUNetV6bTests(unittest.TestCase):
    def test_registry_metadata_and_single_shared_core(self):
        self.assertEqual(MODEL_REGISTRY["RWKV_UNetV6b"],
                         (".RWKV.RWKV_UNet.RWKV_UNetV6b", "rwkv_unet_v6b"))
        self.assertEqual(load_model_id("RWKV_UNetV6b"), (127, 0))
        config = SimpleNamespace(model="RWKV_UNetV6b", do_deeps=False)
        build_kwargs = dict(TINY_KWARGS)
        build_kwargs["input_channel"] = build_kwargs.pop("input_channels")
        model = build_model(config, **build_kwargs)
        self.assertIsInstance(model, RWKV_UNetV6b)
        self.assertEqual([name for name, _ in model.named_children()], ["core"])
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in model.core.parameters()))

    def test_two_core_calls_and_composed_output(self):
        model = RWKV_UNetV6b(**TINY_KWARGS).eval()
        calls = []
        handle = model.core.register_forward_hook(
            lambda _module, inputs, _output: calls.append(tuple(inputs[0].shape[-2:])))
        with torch.no_grad():
            branches = model(torch.randn(1, 1, 64, 40), return_branches=True)
        handle.remove()
        self.assertEqual(calls, [(48, 32), (32, 32)])
        self.assertEqual(branches["out"].shape, (1, 5, 64, 40))
        torch.testing.assert_close(branches["out"][..., 16:48, 4:36], branches["local_logits"])

    def test_production_geometry_uses_720x448_and_center_640_crop(self):
        kwargs = {key: value for key, value in TINY_KWARGS.items()
                  if key not in {"source_size", "global_size", "local_size"}}
        model = RWKV_UNetV6b(**kwargs).eval()
        model.core = torch.nn.Conv2d(1, 5, kernel_size=1)
        calls = []
        handle = model.core.register_forward_hook(
            lambda _module, inputs, _output: calls.append(tuple(inputs[0].shape[-2:])))
        with torch.no_grad():
            branches = model(torch.randn(1, 1, 1024, 640), return_branches=True)
        handle.remove()
        self.assertEqual(calls, [(720, 448), (640, 640)])
        torch.testing.assert_close(branches["out"][..., 192:832, :], branches["local_logits"])

    def test_checkpoint_keys_are_one_core_prefix_from_v6a(self):
        model = RWKV_UNetV6b(**TINY_KWARGS)
        v6a_kwargs = {key: value for key, value in TINY_KWARGS.items()
                      if key not in {"source_size", "global_size", "local_size"}}
        source = RWKV_UNetV6a(**v6a_kwargs)
        self.assertEqual(list(model.state_dict()), [f"core.{key}" for key in source.state_dict()])
        _load_model_state_dict(model, source.state_dict(), strict=True)
        torch.testing.assert_close(
            model.core.encoder.stem[0].weight, source.encoder.stem[0].weight
        )

    def test_both_branch_shapes_backward(self):
        model = RWKV_UNetV6b(**TINY_KWARGS).train()
        image = torch.randn(1, 1, 64, 40, requires_grad=True)
        branches = model(image, return_branches=True, jitter_y=3)
        (branches["global_logits"].mean() + branches["local_logits"].mean()).backward()
        self.assertIsNotNone(image.grad)
        self.assertIsNotNone(model.core.segmentation_head.weight.grad)

    def test_mask_remap_keeps_only_four_osteophyte_ids(self):
        source = np.asarray([[0, 1, 6, 7, 8, 9, 10]], dtype=np.int64)
        expected = np.asarray([[0, 0, 1, 2, 3, 4, 0]], dtype=np.int64)
        np.testing.assert_array_equal(remap_osteophyte_mask(source), expected)

    def test_rejects_transposed_source_dimensions(self):
        model = RWKV_UNetV6b(**TINY_KWARGS)
        with self.assertRaisesRegex(ValueError, "expects source images"):
            model(torch.randn(1, 1, 40, 64))

    def test_onnx_exports_composed_logits_through_reference_core(self):
        model = RWKV_UNetV6b(**TINY_KWARGS).eval()
        args = SimpleNamespace(
            model="RWKV_UNetV6b", img_size=[64, 40], source_size=[64, 40],
            global_size=[48, 32], local_size=32, input_channel=1, num_classes=5
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rwkv_unetv6b.onnx"
            record = export_segment_onnx(model, args, path, validate=True)
            self.assertTrue(record["parity"]["validated"])
            self.assertEqual(record["preprocess"]["global_local"]["shared_weights"], True)
            self.assertEqual(read_onnx_metadata(path)["uknee.model_name"], "RWKV_UNetV6b")

    def test_new_config_preserves_11_classes_and_controls_view_sizes(self):
        config = Path("segment/cfg/rwkv_unetv6b_global_local_11class.yaml").resolve()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "mesko"
            dataset.mkdir()
            args = parse_segment_args([
                "--config", str(config), "--dataset", str(dataset), "--workers", "0"
            ])
        self.assertEqual(args.model, "RWKV_UNetV6b")
        self.assertEqual(args.num_classes, 11)
        self.assertFalse(args.osteophyte_only)
        self.assertEqual(args.source_size, [1024, 640])
        self.assertEqual(args.global_size, [720, 448])
        self.assertEqual(args.local_size, 640)
        self.assertEqual(args.osteophyte_class_ids, [6, 7, 8, 9])
        self.assertEqual(args.loss, "osteophyte_focal_tversky_ce")
        model = build_model(
            args, input_channel=1, num_classes=11, stem_dim=4,
            depths=(1, 1, 1, 1), embed_dims=(4, 8, 12, 16),
            exp_ratios=(1.0, 1.0, 1.0, 1.0), num_heads=(1, 1, 1, 1),
            matrix_state_stages=(), drop_path_rate=0.0, matrix_state_backend="reference",
        )
        self.assertEqual(model.source_size, (1024, 640))
        self.assertEqual(model.global_size, (720, 448))
        self.assertEqual(model.local_size, 640)
        self.assertEqual(model.core.segmentation_head.out_channels, 11)

    def test_focal_tversky_is_finite_and_penalises_absent_class_false_positives(self):
        args = SimpleNamespace(
            loss="osteophyte_focal_tversky_ce", num_classes=11,
            osteophyte_class_ids=[6, 7, 8, 9], focal_tversky_fp_weight=0.30,
            focal_tversky_fn_weight=0.70, focal_tversky_gamma=1.30,
            focal_tversky_smooth=1e-6, lambda_ft=1.0,
        )
        criterion, name = _build_criterion(args)
        self.assertEqual(name, "OsteophyteFocalTverskyCELoss")
        self.assertEqual(criterion.fp_weight, 0.30)
        self.assertEqual(criterion.fn_weight, 0.70)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        low_fp = torch.full((1, 11, 4, 4), -6.0)
        low_fp[:, 0] = 6.0
        high_fp = low_fp.clone()
        high_fp[:, 6] = 8.0
        self.assertLess(
            criterion.focal_tversky(low_fp, target),
            criterion.focal_tversky(high_fp, target),
        )
        high_fp.requires_grad_(True)
        loss = criterion(high_fp, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(high_fp.grad).all())
        self.assertGreater(high_fp.grad[:, 6].abs().sum().item(), 0.0)

        positive_target = target.clone()
        positive_target[:, 1:3, 1:3] = 6
        positive_loss = criterion(low_fp, positive_target)
        self.assertTrue(torch.isfinite(positive_loss))


if __name__ == "__main__":
    unittest.main()
