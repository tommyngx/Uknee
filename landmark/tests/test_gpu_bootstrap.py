from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class GpuBootstrapTests(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_torch(self):
        code = "import landmark,sys; print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_explicit_gpu_list_is_applied_before_torch_import(self):
        code = (
            "import sys,os; "
            "sys.argv=['landmark.train','--gpu','[0,1]']; "
            "import landmark.train; "
            "print(os.environ.get('CUDA_VISIBLE_DEVICES'))"
        )
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "0,1")

    def test_ddp_temp_file_injects_repository_before_landmark_import(self):
        from landmark.core.dist import generate_ddp_file

        class FakeTrainer:
            def __init__(self):
                self.args = SimpleNamespace(model="model.yaml", augmentations=None)
                self.hub_session = SimpleNamespace(model_url="model.yaml")

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with patch("landmark.core.dist.USER_CONFIG_DIR", Path(directory)):
                generated = Path(generate_ddp_file(FakeTrainer()))
                content = generated.read_text(encoding="utf-8")
        path_insert = content.index("sys.path.insert")
        trainer_import = content.index("from landmark.core")
        self.assertLess(path_insert, trainer_import)
        self.assertIn(str(REPOSITORY_ROOT), content)

    def test_amp_check_uses_nested_training_output_without_reference_download(self):
        import torch

        from landmark.core.checks import _amp_comparison_tensor, _first_output_tensor, check_amp

        nested = {"primary": [None, torch.ones(1, 2)]}
        self.assertIs(_first_output_tensor(nested), nested["primary"][1])
        postprocessed = torch.zeros(1, 300, 6)
        raw_scores = torch.ones(1, 2, 8400)
        end2end = (postprocessed, {"one2one": {"scores": raw_scores}})
        self.assertIs(_amp_comparison_tensor(end2end), raw_scores)
        model = torch.nn.Conv2d(3, 4, 1)
        model.stride = torch.tensor([32])
        self.assertFalse(check_amp(model, imgsz=[64, 96]))  # CPU intentionally disables CUDA AMP


if __name__ == "__main__":
    unittest.main()
