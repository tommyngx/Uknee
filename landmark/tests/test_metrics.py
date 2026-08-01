import unittest

import torch

from landmark.evaluate import _contour_oracle_statistics
from landmark.models.metadata import LANDMARK_PATH_RANGES
from landmark.utils.metrics import landmark_metrics


class TopologyMetricTests(unittest.TestCase):
    @staticmethod
    def _ordered_landmarks() -> torch.Tensor:
        target = torch.zeros(1, 129, 2)
        for path_id, (start, stop) in enumerate(LANDMARK_PATH_RANGES):
            target[0, start:stop, 0] = torch.linspace(0.1, 0.9, stop - start)
            target[0, start:stop, 1] = 0.1 + path_id * 0.1
        return target

    def test_order_metrics_distinguish_ordered_and_reversed_paths(self):
        target = self._ordered_landmarks()
        visibility = torch.ones(1, 129)
        ordered = landmark_metrics(target, target, visibility, 640, 640)
        reversed_prediction = target.clone()
        reversed_prediction[:, :45] = reversed_prediction[:, :45].flip(1)
        reversed_metrics = landmark_metrics(
            reversed_prediction, target, visibility, 640, 640
        )

        self.assertEqual(ordered["order_inversion_rate"], 0)
        self.assertEqual(ordered["adjacent_duplicate_rate"], 0)
        self.assertGreater(reversed_metrics["order_inversion_rate"], 0)
        self.assertGreater(reversed_metrics["direction_error_degrees"], 0)

    def test_order_metric_catches_non_adjacent_swap(self):
        target = self._ordered_landmarks()
        prediction = target.clone()
        prediction[:, [0, 10]] = prediction[:, [10, 0]]
        metrics = landmark_metrics(
            prediction,
            target,
            torch.ones(1, 129),
            640,
            640,
        )
        self.assertGreater(metrics["order_inversion_rate"], 0)

    def test_contour_oracle_uses_pixel_axis_scales(self):
        target = torch.zeros(1, 129, 2)
        candidates = torch.zeros(1, 129, 2, 2)
        candidates[:, :, 0, 0] = 0.1
        candidates[:, :, 1, 1] = 0.2
        result = _contour_oracle_statistics(
            {
                "final_landmarks": target,
                "contour_candidate_coordinates": candidates,
            },
            {
                "landmarks": target,
                "landmark_visibility": torch.ones(1, 129),
            },
            image_height=51,
            image_width=101,
        )
        self.assertIsNotNone(result)
        error_sum, count = result
        self.assertEqual(count, 129)
        self.assertAlmostEqual(error_sum / count, 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
