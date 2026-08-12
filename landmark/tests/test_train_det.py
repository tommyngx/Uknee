from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from landmark.train_det import _xray_detection_defaults, prepare_detection_dataset


class DetectionTrainingTests(unittest.TestCase):
    def test_xray_defaults_do_not_corrupt_left_right_classes(self):
        defaults = _xray_detection_defaults()
        self.assertEqual(defaults["fliplr"], 0.0)
        self.assertEqual(defaults["flipud"], 0.0)
        self.assertTrue(defaults["plots"])

    def test_detection_dataset_is_split_without_duplicate_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            images = root / "images" / "train"
            labels = root / "labels" / "train"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(12):
                image = Image.new("RGB", (32, 32), (index, index, index))
                image.save(images / f"case_{index}.png")
                (labels / f"case_{index}.txt").write_text(
                    f"{index % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8"
                )
            # This exact duplicate must follow its source into the same split.
            (images / "duplicate.png").write_bytes((images / "case_0.png").read_bytes())
            (labels / "duplicate.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "images/train",
                        "val": "images/train",
                        "names": {0: "right_knee", 1: "left_knee"},
                        "val_fraction": 0.25,
                        "split_seed": 2026,
                    }
                ),
                encoding="utf-8",
            )

            resolved, audit = prepare_detection_dataset(source, Path(directory) / "output")
            metadata = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            train = set(Path(metadata["train"]).read_text(encoding="utf-8").splitlines())
            val = set(Path(metadata["val"]).read_text(encoding="utf-8").splitlines())

            self.assertFalse(train & val)
            self.assertEqual(len(train | val), 13)
            self.assertEqual(str(images / "case_0.png") in train, str(images / "duplicate.png") in train)
            self.assertEqual(audit["instances"], 13)


if __name__ == "__main__":
    unittest.main()
