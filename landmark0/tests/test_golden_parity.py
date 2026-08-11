from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from landmark0.data import prepare_dataset


REPOSITORY = Path(__file__).resolve().parents[2]
MODEL_YAML = REPOSITORY / "landmark0" / "cfg" / "models" / "yolo26-pose.yaml"


PROBE = textwrap.dedent(
    """
    import random
    import sys
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    model_yaml, data_yaml, destination = sys.argv[1:]
    torch.manual_seed(2006)
    model = YOLO(model_yaml, task="pose").model
    model.kpt_shape = list(model.model[-1].kpt_shape)
    model.args = get_cfg(overrides={"task": "pose", "mode": "train", "imgsz": 64})
    image = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction

    keypoints = torch.zeros(4, 51, 3)
    for class_id, count in enumerate((45, 51, 24, 9)):
        keypoints[class_id, :count, 0] = torch.linspace(0.2, 0.8, count)
        keypoints[class_id, :count, 1] = 0.3 + class_id * 0.1
        keypoints[class_id, :count, 2] = 2
    batch = {
        "img": image,
        "batch_idx": torch.zeros(4),
        "cls": torch.arange(4).view(-1, 1).float(),
        "bboxes": torch.tensor([[0.5, 0.5, 0.6, 0.5]]).repeat(4, 1),
        "keypoints": keypoints,
    }
    model.train()
    loss, items = model(batch)
    random.seed(2006)
    np.random.seed(2006)
    torch.manual_seed(2006)
    data = check_det_dataset(data_yaml)
    data_args = get_cfg(
        overrides={
            "task": "pose", "mode": "train", "model": model_yaml, "data": data_yaml,
            "imgsz": 64, "batch": 1, "workers": 0, "mosaic": 1.0,
            "mixup": 0.0, "cutmix": 0.0, "cache": False,
        }
    )
    dataset = build_yolo_dataset(data_args, data["train"], 1, data, mode="train", rect=False, stride=32)
    sample = dataset[0]
    torch.save(
        {
            "prediction": prediction.detach().cpu(),
            "loss": loss.detach().cpu(),
            "items": items.detach().cpu(),
            "parameters": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "sample": {
                key: value.detach().cpu()
                for key, value in sample.items()
                if key in {"img", "cls", "bboxes", "keypoints"}
            },
        },
        destination,
    )
    """
)


class GoldenParityTests(unittest.TestCase):
    def test_reference_and_vendor_forward_and_fp32_loss_match(self):
        prepared = prepare_dataset(REPOSITORY / "landmark0" / "cfg" / "datasets" / "mesko4gf2.yaml")
        with tempfile.TemporaryDirectory(prefix="uknee-parity-") as directory:
            directory = Path(directory)
            outputs = []
            for name, python_path in (
                ("reference", REPOSITORY / "Ref"),
                ("vendored", REPOSITORY / "landmark0" / "_vendor"),
            ):
                destination = directory / f"{name}.pt"
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(python_path)
                environment["MPLCONFIGDIR"] = str(directory / "matplotlib")
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        PROBE,
                        str(MODEL_YAML),
                        str(prepared.yaml_path),
                        str(destination),
                    ],
                    cwd=REPOSITORY,
                    env=environment,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                outputs.append(torch.load(destination, map_location="cpu", weights_only=True))

        reference, vendored = outputs
        self.assertEqual(reference["parameters"].keys(), vendored["parameters"].keys())
        for key in reference["parameters"]:
            torch.testing.assert_close(reference["parameters"][key], vendored["parameters"][key], rtol=0, atol=0)
        torch.testing.assert_close(reference["prediction"], vendored["prediction"], rtol=0, atol=1e-6)
        torch.testing.assert_close(reference["loss"], vendored["loss"], rtol=0, atol=1e-6)
        torch.testing.assert_close(reference["items"], vendored["items"], rtol=0, atol=1e-6)
        self.assertEqual(reference["sample"].keys(), vendored["sample"].keys())
        for key in reference["sample"]:
            torch.testing.assert_close(reference["sample"][key], vendored["sample"][key], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
