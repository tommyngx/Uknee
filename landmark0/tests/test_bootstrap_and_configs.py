from __future__ import annotations

import unittest
from pathlib import Path

import landmark0
import ultralytics


ROOT = Path(__file__).resolve().parents[1]


class BootstrapAndConfigTests(unittest.TestCase):
    def test_vendored_backend_is_pinned(self):
        backend = landmark0.load_vendored_ultralytics()
        expected = (ROOT / "_vendor" / "ultralytics").resolve()
        self.assertEqual(backend.__version__, "8.4.87")
        self.assertIn(expected, Path(backend.__file__).resolve().parents)
        self.assertIs(backend, ultralytics)

    def test_only_five_public_model_yamls_exist(self):
        names = {path.name for path in (ROOT / "cfg" / "models").glob("*.yaml")}
        self.assertEqual(
            names,
            {
                "yolo26-pose.yaml",
                "yolo26-pose-v1.yaml",
                "yolo26-pose-v9.yaml",
                "hrnet-w32-pose.yaml",
                "vitpose-s-pose.yaml",
            },
        )

    def test_core_vendor_files_match_reference(self):
        repository = ROOT.parent
        pairs = (
            "data/augment.py",
            "data/dataset.py",
            "engine/trainer.py",
            "utils/loss.py",
            "utils/tal.py",
        )
        for relative in pairs:
            source = repository / "Ref" / "ultralytics" / relative
            vendored = ROOT / "_vendor" / "ultralytics" / relative
            self.assertEqual(source.read_bytes(), vendored.read_bytes(), relative)


if __name__ == "__main__":
    unittest.main()
