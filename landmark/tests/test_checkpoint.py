import tempfile
import unittest
from pathlib import Path

import torch

from landmark.utils.checkpoint import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_order_tracking_survives_resume_checkpoint(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        tracking = {
            "best_order_key": (0.1, 0.02, 3.5),
            "best_order_metrics": {"epoch": 7, "order_inversion_rate": 0.1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, None, 7, 3.5, {}, tracking)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["tracking"], tracking)


if __name__ == "__main__":
    unittest.main()
