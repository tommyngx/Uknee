import unittest

import torch

from landmark.utils.coordinates import (
    create_local_sampling_grid,
    denormalize_coordinates,
    normalize_pixel_coordinates,
    soft_argmax_2d,
)


class CoordinateTests(unittest.TestCase):
    def test_pixel_coordinate_round_trip_and_xy_order(self):
        xy = torch.tensor([[17.0, 9.0]])
        unit = normalize_pixel_coordinates(xy, image_height=21, image_width=41)
        self.assertTrue(torch.allclose(unit, torch.tensor([[17 / 40, 9 / 20]])))
        self.assertTrue(torch.allclose(denormalize_coordinates(unit, 21, 41), xy))

    def test_soft_argmax_center(self):
        heatmap = torch.full((1, 1, 7, 9), -20.0)
        heatmap[0, 0, 3, 4] = 20.0
        coordinate, confidence = soft_argmax_2d(heatmap, temperature=0.1)
        self.assertTrue(torch.allclose(coordinate, torch.zeros_like(coordinate), atol=1e-4))
        self.assertGreater(confidence.item(), 0.99)

    def test_sampling_grid_keeps_x_before_y(self):
        center = torch.tensor([[[0.75, 0.25]]])
        grid, _ = create_local_sampling_grid(center, 1, 11, 11)
        self.assertTrue(torch.allclose(grid[0, 0, 0, 0], torch.tensor([0.5, -0.5])))


if __name__ == "__main__":
    unittest.main()
