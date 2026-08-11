from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from segment.cli import parse_segment_args
from segment.dataloader.augment import build_val_transform
from uknee_cli import parse_gpu_ids, parse_image_size, resolve_dataset_path


class SharedCliContractTests(unittest.TestCase):
    def test_image_size_and_gpu_formats(self):
        self.assertEqual(parse_image_size("640"), [640, 640])
        self.assertEqual(parse_image_size("540x640"), [540, 640])
        self.assertEqual(parse_image_size("[540, 640]"), [540, 640])
        self.assertEqual(parse_gpu_ids("[0,1]"), [0, 1])

    def test_dataset_short_name_uses_project_data_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            dataset = project / "data" / "mesko"
            dataset.mkdir(parents=True)
            self.assertEqual(resolve_dataset_path("/mesko", project), dataset.resolve())

    def test_segment_aliases_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            dataset = project / "data" / "mesko"
            dataset.mkdir(parents=True)
            args = parse_segment_args(
                [
                    "--project", str(project), "--dataset", "/mesko",
                    "--imgz", "540x640", "--batch", "4", "--epochs", "200",
                    "--gpu", "[0,1]", "--name", "rect_run",
                ]
            )
            self.assertEqual(args.base_dir, str(dataset.resolve()))
            self.assertEqual(args.dataset_name, "mesko")
            self.assertEqual(args.img_size, [540, 640])
            self.assertEqual(args.gpu_ids, [0, 1])
            self.assertEqual((args.batch_size, args.max_epochs), (4, 200))

    def test_rectangular_letterbox_does_not_stretch_mask(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=np.uint8)
        image[:, 50:150] = 255
        mask[:, 50:150] = 1
        output = build_val_transform([300, 500])(image=image, mask=mask)
        self.assertEqual(output["image"].shape[:2], (300, 500))
        self.assertEqual(output["mask"].shape, (300, 500))
        self.assertEqual(set(np.unique(output["mask"])), {0, 1})


if __name__ == "__main__":
    unittest.main()
