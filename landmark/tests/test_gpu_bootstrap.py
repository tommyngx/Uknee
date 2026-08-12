from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()
