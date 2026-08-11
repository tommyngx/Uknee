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

from segment.utils.binary_metrics import assd, hd95


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


def _resize_sample_for_display(image, target, prediction, fixed_height=512):
    """Resize a sample to a fixed height while preserving its original aspect ratio."""
    import cv2

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, target, prediction
    display_width = max(1, round(width * int(fixed_height) / height))
    size = (display_width, int(fixed_height))
    image_interpolation = cv2.INTER_AREA if fixed_height < height else cv2.INTER_CUBIC
    image = cv2.resize(image, size, interpolation=image_interpolation)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
    target = cv2.resize(np.asarray(target), size, interpolation=cv2.INTER_NEAREST)
    prediction = cv2.resize(np.asarray(prediction), size, interpolation=cv2.INTER_NEAREST)
    return image, target, prediction


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
        """Render four validation segmentation images in a 2x2 grid with MESKO multi-class colors, black outlines, boundary keypoints, and centered title."""
        import cv2
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=150)
        axes_list = list(axes.flat)

        MESKO_COLOR_PALETTE = [
            "#00B4FF",  # 1: Femur (Cyan Blue)
            "#EA580C",  # 2: Tibia (Warm Orange)
            "#FF78DC",  # 3: Fibula (Bright Pink)
            "#00BCD4",  # 4: Overlap (Teal)
            "#14532D",  # 5: Patella (Dark Green)
            "#22C55E",  # 6: Lat Fem Osteophyte (Vibrant Green)
            "#EF4444",  # 7: Med Fem Osteophyte (Vibrant Red)
            "#3B82F6",  # 8: Lat Tib Osteophyte (Royal Blue)
            "#EAB308",  # 9: Med Tib Osteophyte (Gold Yellow)
            "#8B5CF6",  # 10: Tibial Plateau (Purple)
        ]
        OSTEOPHYTE_LEGEND_ITEMS = [
            ("Lat Fem", "#22C55E"),
            ("Med Fem", "#EF4444"),
            ("Lat Tib", "#3B82F6"),
            ("Med Tib", "#EAB308"),
        ]
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=color, edgecolor="black", alpha=0.75, label=label)
            for label, color in OSTEOPHYTE_LEGEND_ITEMS
        ]

        sorted_samples = sorted(self.samples, key=lambda item: item["index"])[:4]
        for ax, sample in zip(axes_list, sorted_samples):
            image = _to_display_image(sample["image"])
            target = sample["target"]
            prediction = sample["prediction"]
            image, target, prediction = _resize_sample_for_display(image, target, prediction)

            ax.imshow(image, cmap="gray" if image.ndim == 2 else None, aspect="equal")

            # Multi-class semi-transparent mask overlay
            overlay = np.zeros((*prediction.shape, 4), dtype=float)
            max_cls = max(int(target.max()), int(prediction.max()))
            for c_id in range(1, max_cls + 1):
                color_hex = MESKO_COLOR_PALETTE[(c_id - 1) % len(MESKO_COLOR_PALETTE)]
                rgb = [int(color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
                mask_c = (prediction == c_id)
                if mask_c.any():
                    overlay[mask_c] = (*rgb, 0.50)

            ax.imshow(overlay, aspect="equal")

            banner = (
                f"Dice {100 * sample['dice']:.2f}% | IoU {100 * sample['iou']:.2f}%\n"
                f"HD95 {sample['hd95']:.3f} mm | ASSD {sample['assd']:.3f} mm"
            )
            ax.text(
                0.01, 0.99, banner, transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5, color="white",
                bbox=dict(facecolor="black", alpha=0.75, pad=3.5, edgecolor="#334155", linewidth=0.8),
            )
            ax.set_title(f"Val Sample {sample['index']}", fontsize=9.5, fontweight="bold", color="#1e293b", pad=6)
            ax.axis("off")

        for ax in axes_list[len(sorted_samples):]:
            ax.axis("off")

        fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9.5, frameon=True, facecolor="white", edgecolor="#94a3b8")
        fig.suptitle(f"Validation Segmentation Predictions — Epoch {int(epoch)}", fontsize=13.5, fontweight="bold", color="#1e293b", ha="center", y=0.955)
        fig.tight_layout(rect=(0, 0.055, 1, 0.945))
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

    # 1. Confusion Matrix Panel (Matching landmark 1:1 with Frosted Glass Badges & grid(False))
    confusion = snapshot.confusion.astype(float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums != 0)
    axis = axes[0, 0]
    axis.grid(False)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

    n_rows, n_cols = normalized.shape
    for row in range(n_rows):
        for column in range(n_cols):
            val = normalized[row, column]
            raw_cnt = int(confusion[row, column])
            if val > 0.45:
                text_color = "#ffffff"
                bg_box = "#000000"
                bg_alpha = 0.25
            else:
                text_color = "#0f172a"
                bg_box = "#ffffff"
                bg_alpha = 0.55

            cell_text = f"{val * 100:.1f}%\n(n={raw_cnt})"
            axis.text(
                column, row, cell_text,
                ha="center", va="center", color=text_color, fontsize=8.5, fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.4",
                    facecolor=bg_box,
                    alpha=bg_alpha,
                    edgecolor="none",
                ),
            )

    gt_counts = confusion.sum(axis=1).astype(int)
    labels = ["Background", "Mask"] if n_rows == 2 else (snapshot.class_names or [f"Class {i}" for i in range(n_rows)])
    x_labels = [f"{name}\n(N={gt_counts[i]})" if i < len(gt_counts) else name for i, name in enumerate(labels)]
    axis.set_xticks(range(n_cols), x_labels, rotation=20 if n_cols > 2 else 0, ha="right" if n_cols > 2 else "center")
    axis.set_yticks(range(n_rows))
    axis.set_yticklabels(labels, rotation=90, va="center", ha="right")
    axis.set_xlabel("Predicted Label", fontsize=9.5, fontweight="bold")
    axis.set_ylabel("True Groundtruth Label", fontsize=9.5, fontweight="bold")
    axis.set_title("Normalized Confusion Matrix (Counts & %)", fontsize=11, fontweight="bold", pad=8)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    # 2. Per-Region Dice and IoU Panel (Sufficient Headroom & Clear Non-Overlapping Bar Numbers)
    axis = axes[0, 1]
    axis.grid(True, linestyle="--", alpha=0.3)
    positions = np.arange(len(snapshot.class_names))
    width = 0.36
    bars_dice = axis.bar(positions - width / 2, 100 * snapshot.class_dice, width, color="#22c55e", edgecolor="#15803d", linewidth=1.0)
    bars_iou = axis.bar(positions + width / 2, 100 * snapshot.class_iou, width, color="#3b82f6", edgecolor="#1d4ed8", linewidth=1.0)

    for bar in bars_dice:
        h = bar.get_height()
        if h > 0:
            axis.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{round(h)}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#15803d")
    for bar in bars_iou:
        h = bar.get_height()
        if h > 0:
            axis.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{round(h)}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1d4ed8")

    axis.set_xticks(positions, snapshot.class_names, rotation=20, ha="right")
    axis.set_ylim(0, 130)
    axis.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold")
    axis.set_title("Per-Region Dice and IoU Scores", fontsize=11, fontweight="bold", pad=8)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#22c55e", edgecolor="#15803d", linewidth=1.0, label="Dice Score (%)"),
        Patch(facecolor="#3b82f6", edgecolor="#1d4ed8", linewidth=1.0, label="IoU Score (%)"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#cbd5e1")

    # 3. Precision-Recall Curve Panel
    axis = axes[1, 0]
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.plot(snapshot.recall_curve, snapshot.precision_curve, color="#9467bd", linewidth=2.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall", fontsize=9.5, fontweight="bold")
    axis.set_ylabel("Precision", fontsize=9.5, fontweight="bold")
    axis.set_title("Mask Precision–Recall Curve", fontsize=11, fontweight="bold", pad=8)

    # 4. Boundary Distance Distribution Panel
    axis = axes[1, 1]
    axis.grid(True, linestyle="--", alpha=0.3)
    values = [snapshot.hd95_values, snapshot.assd_values]
    if any(len(item) for item in values):
        axis.boxplot(values, tick_labels=["HD95", "ASSD"], patch_artist=True,
                     boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8"},
                     medianprops={"color": "#dc2626", "linewidth": 1.8})
    axis.set_ylabel("Distance (mm)", fontsize=9.5, fontweight="bold")
    axis.set_title("Boundary Distance Distribution (mm)", fontsize=11, fontweight="bold", pad=8)

    fig.suptitle("Segmentation Evaluation & Metric Performance Report", fontsize=14, fontweight="bold", color="#1e293b", ha="center", y=0.965)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
