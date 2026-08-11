"""Metrics and compact visual reports for 2D segmentation training."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "uknee-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.binary_metrics import assd, hd95


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _style_axis(axis):
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    axis.tick_params(axis="both", which="both", direction="out", length=5, colors="black")


def _distance_metrics(prediction, target, pixel_spacing_mm):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        # A finite, reproducible penalty keeps CSV/plots usable for failed masks.
        penalty = float(np.hypot(*prediction.shape[-2:]) * pixel_spacing_mm)
        return penalty, penalty
    spacing = (float(pixel_spacing_mm),) * prediction.ndim
    return hd95(prediction, target, voxelspacing=spacing), assd(
        prediction, target, voxelspacing=spacing
    )


def _to_display_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    finite = image[np.isfinite(image)]
    if finite.size:
        low, high = np.percentile(finite, (1, 99))
        image = np.clip((image - low) / max(high - low, 1e-8), 0, 1)
    return image


@dataclass
class ValidationSnapshot:
    dice: float
    iou: float
    hd95: float
    assd: float
    sensitivity: float
    precision: float
    confusion: np.ndarray
    class_names: list[str]
    class_dice: np.ndarray
    class_iou: np.ndarray
    recall_curve: np.ndarray
    precision_curve: np.ndarray
    hd95_values: np.ndarray
    assd_values: np.ndarray


class SegmentationEvaluator:
    """Accumulate real validation predictions for metrics and report artifacts."""

    def __init__(
        self,
        num_classes,
        pixel_spacing_mm=0.10,
        class_names=None,
        sample_indices=(),
        seed=2006,
        max_pr_points=200_000,
    ):
        self.num_classes = int(num_classes)
        self.pixel_spacing_mm = float(pixel_spacing_mm)
        self.class_names = self._resolve_class_names(class_names)
        self.sample_indices = set(int(index) for index in sample_indices)
        self.rng = np.random.default_rng(seed)
        self.max_pr_points = int(max_pr_points)
        self.confusion = np.zeros((2, 2), dtype=np.int64)
        foreground_classes = 1 if self.num_classes <= 1 else self.num_classes - 1
        self.class_counts = np.zeros((foreground_classes, 3), dtype=np.int64)
        self.hd95_values = []
        self.assd_values = []
        self.pr_probabilities = []
        self.pr_targets = []
        self.samples = []

    def _resolve_class_names(self, names):
        count = 1 if self.num_classes <= 1 else self.num_classes - 1
        if not names:
            return ["Mask"] if count == 1 else [f"Region {index}" for index in range(1, count + 1)]
        names = list(names)
        if self.num_classes > 1 and len(names) == self.num_classes:
            names = names[1:]
        names = [str(name) for name in names[:count]]
        return names + [f"Region {index}" for index in range(len(names) + 1, count + 1)]

    @staticmethod
    def fixed_sample_indices(dataset_size, count=4, seed=2006):
        dataset_size = int(dataset_size)
        if dataset_size <= 0:
            return []
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(dataset_size, size=min(count, dataset_size), replace=False).tolist())

    def update(self, logits, target, images=None, start_index=0):
        logits = logits.detach()
        target = target.detach()
        if self.num_classes > 1:
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            if target.ndim == 4 and target.shape[1] == self.num_classes:
                target_ids = target.argmax(dim=1)
            elif target.ndim == 4 and target.shape[1] == 1:
                target_ids = target[:, 0].long()
            else:
                target_ids = target.long()
            foreground_probability = 1.0 - probabilities[:, 0]
        else:
            foreground_probability = torch.sigmoid(logits[:, 0])
            predictions = (foreground_probability >= 0.5).long()
            target_ids = target[:, 0].long() if target.ndim == 4 else target.long()

        prediction_np = predictions.cpu().numpy()
        target_np = target_ids.cpu().numpy()
        probability_np = foreground_probability.float().cpu().numpy()
        images_np = images.detach().cpu().numpy() if images is not None else None

        for batch_index, (prediction, truth, probability) in enumerate(
            zip(prediction_np, target_np, probability_np)
        ):
            foreground_prediction = prediction > 0
            foreground_truth = truth > 0
            tp = np.logical_and(foreground_prediction, foreground_truth).sum()
            fp = np.logical_and(foreground_prediction, ~foreground_truth).sum()
            fn = np.logical_and(~foreground_prediction, foreground_truth).sum()
            tn = np.logical_and(~foreground_prediction, ~foreground_truth).sum()
            self.confusion += np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)

            class_ids = [1] if self.num_classes <= 1 else range(1, self.num_classes)
            for class_offset, class_id in enumerate(class_ids):
                pred_class = foreground_prediction if self.num_classes <= 1 else prediction == class_id
                true_class = foreground_truth if self.num_classes <= 1 else truth == class_id
                self.class_counts[class_offset] += (
                    np.logical_and(pred_class, true_class).sum(),
                    np.logical_and(pred_class, ~true_class).sum(),
                    np.logical_and(~pred_class, true_class).sum(),
                )

            sample_hd95, sample_assd = _distance_metrics(
                foreground_prediction, foreground_truth, self.pixel_spacing_mm
            )
            self.hd95_values.append(sample_hd95)
            self.assd_values.append(sample_assd)

            flat_probability = probability.reshape(-1)
            flat_truth = foreground_truth.reshape(-1)
            remaining = max(self.max_pr_points - sum(map(len, self.pr_probabilities)), 0)
            if remaining:
                take = min(remaining, flat_probability.size)
                if take < flat_probability.size:
                    indices = self.rng.choice(flat_probability.size, size=take, replace=False)
                    flat_probability = flat_probability[indices]
                    flat_truth = flat_truth[indices]
                self.pr_probabilities.append(flat_probability.astype(np.float32, copy=False))
                self.pr_targets.append(flat_truth.astype(bool, copy=False))

            dataset_index = int(start_index) + batch_index
            if dataset_index in self.sample_indices and images_np is not None:
                self.samples.append(
                    {
                        "index": dataset_index,
                        "image": images_np[batch_index],
                        "target": truth,
                        "prediction": prediction,
                        "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
                        "iou": _safe_ratio(tp, tp + fp + fn),
                        "hd95": sample_hd95,
                        "assd": sample_assd,
                    }
                )

    def snapshot(self):
        tn, fp = self.confusion[0]
        fn, tp = self.confusion[1]
        class_dice = []
        class_iou = []
        for class_tp, class_fp, class_fn in self.class_counts:
            class_dice.append(_safe_ratio(2 * class_tp, 2 * class_tp + class_fp + class_fn))
            class_iou.append(_safe_ratio(class_tp, class_tp + class_fp + class_fn))

        probabilities = np.concatenate(self.pr_probabilities) if self.pr_probabilities else np.array([])
        targets = np.concatenate(self.pr_targets) if self.pr_targets else np.array([], dtype=bool)
        precision_curve, recall_curve = [], []
        for threshold in np.linspace(0.0, 1.0, 101):
            predicted = probabilities >= threshold
            curve_tp = np.logical_and(predicted, targets).sum()
            curve_fp = np.logical_and(predicted, ~targets).sum()
            curve_fn = np.logical_and(~predicted, targets).sum()
            precision_curve.append(_safe_ratio(curve_tp, curve_tp + curve_fp))
            recall_curve.append(_safe_ratio(curve_tp, curve_tp + curve_fn))

        return ValidationSnapshot(
            dice=_safe_ratio(2 * tp, 2 * tp + fp + fn),
            iou=_safe_ratio(tp, tp + fp + fn),
            hd95=float(np.mean(self.hd95_values)) if self.hd95_values else 0.0,
            assd=float(np.mean(self.assd_values)) if self.assd_values else 0.0,
            sensitivity=_safe_ratio(tp, tp + fn),
            precision=_safe_ratio(tp, tp + fp),
            confusion=self.confusion.copy(),
            class_names=self.class_names,
            class_dice=np.asarray(class_dice, dtype=float),
            class_iou=np.asarray(class_iou, dtype=float),
            recall_curve=np.asarray(recall_curve, dtype=float),
            precision_curve=np.asarray(precision_curve, dtype=float),
            hd95_values=np.asarray(self.hd95_values, dtype=float),
            assd_values=np.asarray(self.assd_values, dtype=float),
        )

    def save_samples(self, output_path, epoch):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=150)
        for axis in axes.flat:
            _style_axis(axis)
            axis.set_xticks([])
            axis.set_yticks([])

        for axis, sample in zip(axes.flat, sorted(self.samples, key=lambda item: item["index"])):
            image = _to_display_image(sample["image"])
            target = sample["target"] > 0
            prediction = sample["prediction"] > 0
            axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
            gt_overlay = np.zeros((*target.shape, 4), dtype=float)
            gt_overlay[target] = (0.1, 0.85, 0.25, 0.28)
            pred_overlay = np.zeros((*prediction.shape, 4), dtype=float)
            pred_overlay[prediction] = (0.95, 0.1, 0.25, 0.24)
            axis.imshow(gt_overlay)
            axis.imshow(pred_overlay)
            if target.any() and not target.all():
                axis.contour(target, levels=[0.5], colors=["#00d45a"], linewidths=1.2)
            if prediction.any() and not prediction.all():
                axis.contour(prediction, levels=[0.5], colors=["#ff1744"], linewidths=1.2)
            banner = (
                f"Dice {100 * sample['dice']:.2f}% | IoU {100 * sample['iou']:.2f}%\n"
                f"HD95 {sample['hd95']:.3f} mm | ASSD {sample['assd']:.3f} mm"
            )
            axis.text(
                0.01, 0.99, banner, transform=axis.transAxes, va="top", ha="left",
                fontsize=9, color="white", bbox={"facecolor": "black", "alpha": 0.72, "pad": 4},
            )
            axis.set_title(f"Validation sample {sample['index']}", fontsize=10)

        for axis in axes.flat[len(self.samples):]:
            axis.set_visible(False)
        fig.suptitle(f"Validation predictions — epoch {int(epoch)} | GT green, prediction red", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return output_path


def plot_segmentation_metrics(snapshot, output_path):
    """Write the required four-panel segmentation evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)
    for axis in axes.flat:
        _style_axis(axis)

    confusion = snapshot.confusion.astype(float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)
    axis = axes[0, 0]
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            value = normalized[row, column]
            axis.text(column, row, f"{100 * value:.1f}%", ha="center", va="center",
                      color="white" if value > 0.55 else "black")
    axis.set_xticks([0, 1], ["Background", "Mask"])
    axis.set_yticks([0, 1], ["Background", "Mask"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Normalized Confusion Matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis = axes[0, 1]
    positions = np.arange(len(snapshot.class_names))
    width = 0.38
    axis.bar(positions - width / 2, 100 * snapshot.class_dice, width, label="Dice", color="#2ca02c")
    axis.bar(positions + width / 2, 100 * snapshot.class_iou, width, label="IoU", color="#1f77b4")
    axis.set_xticks(positions, snapshot.class_names, rotation=20, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Score (%)")
    axis.set_title("Per-Region Dice and IoU")
    axis.legend()

    axis = axes[1, 0]
    axis.plot(snapshot.recall_curve, snapshot.precision_curve, color="#9467bd", linewidth=2.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Mask Precision–Recall Curve")

    axis = axes[1, 1]
    values = [snapshot.hd95_values, snapshot.assd_values]
    if any(len(item) for item in values):
        axis.boxplot(values, tick_labels=["HD95", "ASSD"], patch_artist=True,
                     boxprops={"facecolor": "#dbeafe", "edgecolor": "black"},
                     medianprops={"color": "#dc2626", "linewidth": 1.5})
    axis.set_ylabel("Distance (mm)")
    axis.set_title("Boundary Distance Distribution")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
