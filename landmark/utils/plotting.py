from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "uknee-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class TrainingPlotter:
    """Persist CSV history, metric plots and qualitative overlays per epoch."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.plot_dir = self.run_dir / "plots"
        self.epoch_dir = self.plot_dir / "epochs"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, float]] = []

    def update(self, values: dict[str, float]) -> None:
        self.history.append(values)
        fieldnames = sorted({key for row in self.history for key in row})
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        epoch = [row["epoch"] for row in self.history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ("train_loss", "val_loss", "coarse_loss", "coordinate_loss", "heatmap_loss"):
            if any(key in row for row in self.history):
                axes[0].plot(epoch, [row.get(key, np.nan) for row in self.history], label=key)
        axes[0].set_title("Loss by epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(fontsize=8)
        for key in ("val_mre", "val_pck4", "val_pck8", "learning_rate"):
            if any(key in row for row in self.history):
                axes[1].plot(epoch, [row.get(key, np.nan) for row in self.history], label=key)
        axes[1].set_title("Validation / learning rate")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(self.plot_dir / "training_curves.png", dpi=160)
        plt.close(figure)

    def overlay(
        self,
        images: torch.Tensor,
        predicted: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor,
        epoch: int,
        limit: int = 4,
    ) -> None:
        count = min(limit, images.shape[0])
        figure, axes = plt.subplots(1, count, figsize=(5 * count, 5), squeeze=False)
        for index in range(count):
            image = images[index, 0].detach().float().cpu().numpy()
            image = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
            height, width = image.shape
            mask = visibility[index].detach().bool().cpu().numpy()
            pred = predicted[index].detach().cpu().numpy()
            truth = target[index].detach().cpu().numpy()
            axis = axes[0, index]
            axis.imshow(image, cmap="gray")
            axis.scatter(truth[mask, 0] * (width - 1), truth[mask, 1] * (height - 1), s=7, c="lime", label="GT")
            axis.scatter(pred[mask, 0] * (width - 1), pred[mask, 1] * (height - 1), s=7, c="red", label="Pred")
            axis.axis("off")
            if index == 0:
                axis.legend(loc="lower right", fontsize=7)
        figure.tight_layout()
        figure.savefig(self.epoch_dir / f"epoch_{epoch:04d}.png", dpi=150)
        plt.close(figure)
