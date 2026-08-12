from __future__ import annotations

import unittest
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "uknee-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from uknee_plotting import apply_robust_y_limit


class RobustPlotLimitTests(unittest.TestCase):
    def test_early_loss_spikes_do_not_flatten_converged_history(self):
        epochs = np.arange(1, 101)
        loss = np.concatenate((np.linspace(5000.0, 100.0, 30), np.linspace(2.0, 0.2, 70)))
        figure, axis = plt.subplots()
        axis.plot(epochs, loss)

        result = apply_robust_y_limit(axis, [loss], epochs=epochs)

        self.assertTrue(result["applied"])
        self.assertGreater(result["clipped_points"], 0)
        self.assertLess(axis.get_ylim()[1], 10.0)
        self.assertIn("max=5,000", axis.texts[0].get_text())
        plt.close(figure)

    def test_normal_history_keeps_matplotlib_auto_upper_limit(self):
        epochs = np.arange(1, 51)
        loss = np.linspace(1.0, 0.2, 50)
        figure, axis = plt.subplots()
        axis.plot(epochs, loss)

        result = apply_robust_y_limit(axis, [loss], epochs=epochs)

        self.assertFalse(result["applied"])
        self.assertGreaterEqual(axis.get_ylim()[1], 1.0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
