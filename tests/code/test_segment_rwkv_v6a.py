import os
import unittest
from types import SimpleNamespace

import torch

from segment.models import MODEL_REGISTRY, build_model, load_model_id
from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6 import (
    AxialRWKV6SpatialMix,
    RWKV6MatrixStateScan,
    RWKV6SequenceMix,
    RWKV_UNetV6,
)
from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6a import (
    AxialRWKV6aSpatialMix,
    RWKV6aMatrixStateScan,
    RWKV6aSequenceMix,
    RWKV_UNetV6a,
)


def _clones(*tensors):
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


class RWKVUNetV6aReferenceTests(unittest.TestCase):
    def test_registry_and_metadata_are_independent(self):
        self.assertEqual(
            MODEL_REGISTRY["RWKV_UNetV6a"],
            (".RWKV.RWKV_UNet.RWKV_UNetV6a", "rwkv_unet_v6a"),
        )
        self.assertEqual(load_model_id("RWKV_UNetV6a"), (126, 0))
        config = SimpleNamespace(model="RWKV_UNetV6a", do_deeps=False)
        model = build_model(
            config,
            input_channel=1,
            num_classes=2,
            matrix_state_backend="reference",
        )
        self.assertIsInstance(model, RWKV_UNetV6a)
        self.assertEqual(config.model_id, 126)

    def test_default_parameter_and_checkpoint_contract(self):
        torch.manual_seed(2006)
        v6 = RWKV_UNetV6(input_channels=1, num_classes=2)
        v6a = RWKV_UNetV6a(
            input_channels=1,
            num_classes=2,
            matrix_state_backend="reference",
        )
        count_v6 = sum(parameter.numel() for parameter in v6.parameters())
        count_v6a = sum(parameter.numel() for parameter in v6a.parameters())
        self.assertEqual(count_v6, count_v6a)
        self.assertEqual(list(v6.state_dict()), list(v6a.state_dict()))
        self.assertEqual(
            [tensor.shape for tensor in v6.state_dict().values()],
            [tensor.shape for tensor in v6a.state_dict().values()],
        )
        v6a.load_state_dict(v6.state_dict(), strict=True)
        v6.load_state_dict(v6a.state_dict(), strict=True)
        self.assertEqual(
            sum(isinstance(module, RWKV6aMatrixStateScan) for module in v6a.modules()),
            4,
        )

    def test_reference_scan_forward_and_backward_equal_frozen_v6(self):
        torch.manual_seed(11)
        shape = (2, 4, 6)
        source = (
            torch.randn(shape),
            torch.sigmoid(torch.randn(shape)),
            torch.randn(shape),
            torch.randn(shape),
        )
        grad_output = torch.randn(shape)
        for reverse in (False, True):
            v6_inputs = _clones(*source)
            v6a_inputs = _clones(*source)
            v6 = RWKV6MatrixStateScan(dim=6, num_heads=2)
            v6a = RWKV6aMatrixStateScan(dim=6, num_heads=2, backend="reference")
            output_v6 = v6(*v6_inputs, reverse=reverse)
            output_v6a = v6a(*v6a_inputs, reverse=reverse)
            torch.testing.assert_close(output_v6a, output_v6, rtol=0, atol=0)
            grads_v6 = torch.autograd.grad(output_v6, v6_inputs, grad_output)
            grads_v6a = torch.autograd.grad(output_v6a, v6a_inputs, grad_output)
            for actual, expected in zip(grads_v6a, grads_v6):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sequence_and_axial_modules_match_v6(self):
        torch.manual_seed(19)
        sequence_v6 = RWKV6SequenceMix(dim=12, num_heads=3, low_rank_dim=4)
        sequence_v6a = RWKV6aSequenceMix(
            dim=12,
            num_heads=3,
            low_rank_dim=4,
            matrix_state_backend="reference",
        )
        sequence_v6a.load_state_dict(sequence_v6.state_dict(), strict=True)
        sequence_input = torch.randn(2, 5, 12)
        torch.testing.assert_close(
            sequence_v6a(sequence_input), sequence_v6(sequence_input), rtol=0, atol=0
        )

        axial_v6 = AxialRWKV6SpatialMix(dim=12, num_heads=3, low_rank_dim=4)
        axial_v6a = AxialRWKV6aSpatialMix(
            dim=12,
            num_heads=3,
            low_rank_dim=4,
            matrix_state_backend="reference",
        )
        axial_v6a.load_state_dict(axial_v6.state_dict(), strict=True)
        axial_input = torch.randn(2, 12, 12)
        torch.testing.assert_close(
            axial_v6a(axial_input, (3, 4)),
            axial_v6(axial_input, (3, 4)),
            rtol=0,
            atol=0,
        )

    def test_complete_model_forward_and_representative_gradients_match(self):
        torch.manual_seed(23)
        model_kwargs = dict(
            input_channels=1,
            num_classes=2,
            stem_dim=8,
            depths=(1, 1, 1, 2),
            embed_dims=(8, 12, 16, 24),
            exp_ratios=(2.0, 2.0, 2.0, 2.0),
            num_heads=(1, 1, 2, 3),
            drop_path_rate=0.0,
        )
        v6 = RWKV_UNetV6(**model_kwargs).eval()
        v6a = RWKV_UNetV6a(
            **model_kwargs,
            matrix_state_backend="reference",
        ).eval()
        v6a.load_state_dict(v6.state_dict(), strict=True)
        input_v6 = torch.randn(1, 1, 32, 32, requires_grad=True)
        input_v6a = input_v6.detach().clone().requires_grad_(True)
        output_v6 = v6(input_v6)
        output_v6a = v6a(input_v6a)
        torch.testing.assert_close(output_v6a, output_v6, rtol=0, atol=0)
        output_v6.square().mean().backward()
        output_v6a.square().mean().backward()
        torch.testing.assert_close(input_v6a.grad, input_v6.grad, rtol=0, atol=0)
        for name in (
            "encoder.stem.0.weight",
            "encoder.stage4.1.spatial_mix.horizontal_mix.output.weight",
            "segmentation_head.weight",
        ):
            grad_v6 = dict(v6.named_parameters())[name].grad
            grad_v6a = dict(v6a.named_parameters())[name].grad
            torch.testing.assert_close(grad_v6a, grad_v6, rtol=0, atol=0)

    def test_explicit_cuda_never_silently_falls_back(self):
        if torch.cuda.is_available():
            self.skipTest("This failure-path assertion is CPU-only")
        with self.assertRaisesRegex(RuntimeError, "CUDA backend was requested"):
            RWKV_UNetV6a(matrix_state_backend="cuda")


@unittest.skipUnless(torch.cuda.is_available(), "NVIDIA CUDA is unavailable")
class RWKVUNetV6aCudaTests(unittest.TestCase):
    def _compare_cuda_scan(self, shape, dtype=torch.float32):
        from segment.models.RWKV.RWKV_UNet.cuda_v6a import matrix_scan

        torch.manual_seed(29)
        source = (
            torch.randn(shape, device="cuda", dtype=dtype),
            torch.sigmoid(torch.randn(shape, device="cuda", dtype=dtype)),
            torch.randn(shape, device="cuda", dtype=dtype),
            torch.randn(shape, device="cuda", dtype=dtype),
        )
        reference_inputs = _clones(*source)
        cuda_inputs = _clones(*source)
        from segment.models.RWKV.RWKV_UNet.RWKV_UNetV6a import reference_matrix_scan

        reference = reference_matrix_scan(*reference_inputs)
        actual = matrix_scan(*(tensor.contiguous() for tensor in cuda_inputs))
        grad_output = torch.randn_like(reference)
        reference_grads = torch.autograd.grad(reference, reference_inputs, grad_output)
        actual_grads = torch.autograd.grad(actual, cuda_inputs, grad_output)
        if dtype == torch.float32:
            tolerances = dict(rtol=1e-4, atol=1e-5)
        else:
            tolerances = dict(rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual, reference, **tolerances)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, **tolerances)

    def test_torch_library_opcheck(self):
        from segment.models.RWKV.RWKV_UNet.cuda_v6a import custom_op

        args = tuple(
            tensor.contiguous()
            for tensor in (
                torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
                torch.sigmoid(torch.randn(1, 3, 2, 3, device="cuda")).requires_grad_(),
                torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
                torch.randn(1, 3, 2, 3, device="cuda", requires_grad=True),
            )
        )
        torch.library.opcheck(custom_op(), args)

    def test_fp32_small_and_real_head_dimension(self):
        self._compare_cuda_scan((2, 4, 2, 3))
        lengths = [16]
        if os.environ.get("RWKV_V6A_EXTENDED_CUDA_TESTS") == "1":
            lengths.extend([28, 32, 45, 64])
        for length in lengths:
            with self.subTest(length=length):
                self._compare_cuda_scan((1, length, 6, 90))

    def test_bfloat16_with_fp32_state(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("GPU does not support bfloat16")
        self._compare_cuda_scan((1, 4, 2, 3), dtype=torch.bfloat16)

    def test_axial_and_complete_model_cuda_parity(self):
        torch.manual_seed(37)
        axial_reference = AxialRWKV6aSpatialMix(
            dim=12, num_heads=3, low_rank_dim=4, matrix_state_backend="reference"
        ).cuda()
        with torch.no_grad():
            for name, parameter in axial_reference.named_parameters():
                if name.endswith("output.weight"):
                    parameter.normal_(std=0.02)
        axial_cuda = AxialRWKV6aSpatialMix(
            dim=12, num_heads=3, low_rank_dim=4, matrix_state_backend="cuda"
        ).cuda()
        axial_cuda.load_state_dict(axial_reference.state_dict(), strict=True)
        axial_input_reference = torch.randn(1, 12, 12, device="cuda", requires_grad=True)
        axial_input_cuda = axial_input_reference.detach().clone().requires_grad_(True)
        axial_expected = axial_reference(axial_input_reference, (3, 4))
        axial_actual = axial_cuda(axial_input_cuda, (3, 4))
        torch.testing.assert_close(axial_actual, axial_expected, rtol=1e-4, atol=1e-5)
        axial_expected.square().mean().backward()
        axial_actual.square().mean().backward()
        torch.testing.assert_close(
            axial_input_cuda.grad, axial_input_reference.grad, rtol=1e-4, atol=1e-5
        )

        model_kwargs = dict(
            input_channels=1,
            num_classes=2,
            stem_dim=8,
            depths=(1, 1, 1, 2),
            embed_dims=(8, 12, 16, 24),
            exp_ratios=(2.0, 2.0, 2.0, 2.0),
            num_heads=(1, 1, 2, 3),
            drop_path_rate=0.0,
        )
        model_reference = RWKV_UNetV6a(
            **model_kwargs, matrix_state_backend="reference"
        ).cuda().eval()
        with torch.no_grad():
            for name, parameter in model_reference.named_parameters():
                if "spatial_mix" in name and name.endswith("output.weight"):
                    parameter.normal_(std=0.02)
        model_cuda = RWKV_UNetV6a(
            **model_kwargs, matrix_state_backend="cuda"
        ).cuda().eval()
        model_cuda.load_state_dict(model_reference.state_dict(), strict=True)
        image_reference = torch.randn(1, 1, 32, 32, device="cuda", requires_grad=True)
        image_cuda = image_reference.detach().clone().requires_grad_(True)
        expected = model_reference(image_reference)
        actual = model_cuda(image_cuda)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(image_cuda.grad, image_reference.grad, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
