"""Only the three standardized visual artifacts emitted by landmark training."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_CACHE = Path(tempfile.gettempdir()) / "uknee-matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from landmark.data.schema import LANDMARK_PATH_RANGES, REGION_NAMES


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")


def _style(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=5, width=1.0, colors="black")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = rows[0].keys() if rows else ()
    return {
        key: np.asarray([float(row[key]) if row.get(key, "") else np.nan for row in rows], dtype=float)
        for key in keys
    }


def plot_dashboard_pose(
    csv_file: str | Path,
    output_png: str | Path | None = None,
    *,
    pixel_spacing: float = 0.10,
    model_name: str | None = None,
    elapsed_seconds: float | None = None,
) -> Path | None:
    """Write the fixed 2x2 epoch dashboard with balanced marker sizes, vibrant colors, and seaborn darkgrid style."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        return None
    values = _read_csv(csv_path)
    if not values:
        return None
    epochs = values.get("epoch", np.arange(1, len(next(iter(values.values()))) + 1))
    destination = Path(output_png) if output_png else csv_path.parent / "dashboard_pose.png"

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    # Core vibrant color palette
    c_blue = "#2563eb"
    c_red = "#dc2626"
    c_green = "#16a34a"
    c_purple = "#9333ea"

    # Balanced uniform marker sizes
    marker_styles = [
        ("#facc15", "*", 12, "#b45309"),  # Top 1: Gold Star
        ("#94a3b8", "D", 9, "#475569"),   # Top 2: Slate Diamond
        ("#b45309", "o", 8, "#78350f"),   # Top 3: Bronze Circle
    ]

    # Title formatting
    m_name = model_name or csv_path.parent.name or "yolo26-pose-v1"
    time_info = ""
    if elapsed_seconds and elapsed_seconds > 0:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        avg_ep = elapsed_seconds / max(1, len(epochs))
        time_info = f" | Train Time: {mins}m {secs}s ({avg_ep:.1f}s/ep)"

    fig.suptitle(f"Landmark Pose: {m_name}{time_info}", color="#1e293b", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Train & Val Loss + Top 1, 2, 3 Val Loss Markers
    ax1 = axes[0, 0]
    _style(ax1)
    if "train/loss" in values:
        ax1.plot(epochs, values["train/loss"], label="Train Loss", color=c_blue, lw=2.2)
    if "val/loss" in values:
        val_losses = values["val/loss"]
        ax1.plot(epochs, val_losses, label="Val Loss", color=c_red, lw=2.2, linestyle="--")
        finite = np.flatnonzero(np.isfinite(val_losses))
        if finite.size:
            top_indices = finite[np.argsort(val_losses[finite])[:3]]
            for rank, idx in enumerate(top_indices):
                c, m, ms, ec = marker_styles[rank % len(marker_styles)]
                ep = epochs[idx]
                val = val_losses[idx]
                ax1.plot(ep, val, marker=m, markersize=ms, color=c, markeredgecolor=ec,
                         markeredgewidth=1.2, linestyle="None", label=f"Top{rank+1} Val Loss: {val:.4f} (E{ep})", zorder=5)

    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold", color="#1e293b")
    ax1.set_xlabel("Epochs", fontsize=10, color="black")
    ax1.set_ylabel("Loss", fontsize=10, color="black", fontweight="semibold")
    ax1.set_ylim(bottom=-0.005)
    leg1 = ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg1:
        leg1.get_frame().set_alpha(0.96)
        leg1.set_zorder(100)

    # Subplot 2: Overall MRE & Box mAP50-95
    ax2 = axes[0, 1]
    _style(ax2)
    best_mre_idx = 0
    if "metrics/MRE" in values:
        mre_mm = values["metrics/MRE"] * pixel_spacing
        best_mre_idx = int(np.nanargmin(mre_mm))
        best_mre_ep = epochs[best_mre_idx]
        best_mre_val = mre_mm[best_mre_idx]

        ax2.plot(epochs, mre_mm, label="Overall MRE (mm)", color=c_red, lw=2.5)
        ax2.plot(best_mre_ep, best_mre_val, marker="*", markersize=12, color="#facc15",
                 markeredgecolor="#991b1b", markeredgewidth=1.2,
                 label=f"Best MRE: {best_mre_val:.4f} mm (E{best_mre_ep})", zorder=5)

    ax2.plot([], [], " ", label=f"Pixel Spacing: {pixel_spacing:.2f} mm/px")

    has_bbox_map = "metrics/mAP50-95(B)" in values and np.isfinite(values["metrics/mAP50-95(B)"]).any()
    if has_bbox_map:
        bbox_vals = values["metrics/mAP50-95(B)"]
        ax2_right = ax2.twinx()
        _style(ax2_right)
        ax2_right.grid(False)
        ax2_right.plot(epochs, bbox_vals, label="Box mAP50-95", color=c_blue, lw=2.0, linestyle="--")
        best_map_idx = int(np.nanargmax(bbox_vals))
        best_map_ep = epochs[best_map_idx]
        best_map_val = bbox_vals[best_map_idx]
        ax2_right.plot(best_map_ep, best_map_val, marker="*", markersize=12,
                       color="#60a5fa", markeredgecolor="#1e3a8a", markeredgewidth=1.2,
                       label=f"Best Box mAP: {best_map_val:.4f} (E{best_map_ep})", zorder=5)
        ax2_right.set_ylabel("Box mAP50-95", fontsize=10, color=c_blue, fontweight="semibold")
        ax2_right.set_ylim(bottom=-0.04, top=1.05)

    ax2.set_title("Mean Radial Error (MRE)" + (" & Box mAP50-95" if has_bbox_map else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Epochs", fontsize=10, color="black")
    ax2.set_ylabel("MRE Error (mm)", fontsize=10, color=c_red, fontweight="semibold")
    ax2.set_ylim(bottom=-0.05)

    handles_l2, labels_l2 = ax2.get_legend_handles_labels()
    if has_bbox_map:
        handles_r2, labels_r2 = ax2_right.get_legend_handles_labels()
        leg2 = ax2_right.legend(handles_l2 + handles_r2, labels_l2 + labels_r2,
                                loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    else:
        leg2 = ax2.legend(handles_l2, labels_l2, loc="upper left", frameon=True,
                          facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg2:
        leg2.get_frame().set_alpha(0.96)
        leg2.set_zorder(100)

    # Subplot 3: Per-Region MRE
    ax3 = axes[1, 0]
    _style(ax3)
    region_cols = [("metrics/MRE_femur", "Femur", c_blue),
                   ("metrics/MRE_tibia", "Tibia", c_red),
                   ("metrics/MRE_fibula", "Fibula", c_green),
                   ("metrics/MRE_patella", "Patella", c_purple)]
    for col, name, color in region_cols:
        if col in values:
            r_mm = values[col] * pixel_spacing
            best_val = r_mm[best_mre_idx] if len(r_mm) > best_mre_idx else r_mm[-1]
            ax3.plot(epochs, r_mm, label=f"{name}: {best_val:.4f} mm", color=color, lw=2.0)

    if "metrics/MRE" in values and len(mre_mm) > best_mre_idx:
        ax3.plot(epochs[best_mre_idx], mre_mm[best_mre_idx], marker="*", markersize=12, color="#facc15",
                 markeredgecolor="#b45309", markeredgewidth=1.2, label=f"Best MRE Epoch (E{best_mre_ep})", zorder=5)

    ax3.set_title("Per-Region MRE (Femur, Tibia, Fibula, Patella in mm)", fontsize=12, fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Epochs", fontsize=10, color="black")
    ax3.set_ylabel("Error (mm)", fontsize=10, color="black", fontweight="semibold")
    ax3.set_ylim(bottom=-0.05)
    leg3 = ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg3:
        leg3.get_frame().set_alpha(0.96)
        leg3.set_zorder(100)

    # Subplot 4: PCK Accuracy & HD95
    ax4 = axes[1, 1]
    _style(ax4)
    has_hd95 = "metrics/HD95" in values and np.isfinite(values["metrics/HD95"]).any()
    pck_cols = [("metrics/PCK2", "PCK@0.2mm", c_blue),
                ("metrics/PCK4", "PCK@0.4mm", c_green),
                ("metrics/PCK8", "PCK@0.8mm", c_purple)]
    for col, name, color in pck_cols:
        if col in values:
            p_vals = values[col] * 100.0
            best_pck = p_vals[best_mre_idx] if len(p_vals) > best_mre_idx else p_vals[-1]
            ax4.plot(epochs, p_vals, label=f"{name}: {best_pck:.2f}%", color=color, lw=2.0)

    if "metrics/PCK2" in values:
        p2_vals = values["metrics/PCK2"] * 100.0
        best_p2_idx = int(np.nanargmax(p2_vals))
        best_p2_ep = epochs[best_p2_idx]
        best_p2_val = p2_vals[best_p2_idx]
        ax4.plot(best_p2_ep, best_p2_val, marker="*", markersize=12,
                 color="#facc15", markeredgecolor="#1e3a8a", markeredgewidth=1.2,
                 label=f"Best PCK@0.2mm: {best_p2_val:.2f}% (E{best_p2_ep})", zorder=5)

    if has_hd95:
        hd95_vals = values["metrics/HD95"] * pixel_spacing
        ax4_right = ax4.twinx()
        _style(ax4_right)
        ax4_right.grid(False)
        ax4_right.plot(epochs, hd95_vals, label="Val HD95 (mm)", color=c_red, lw=2.0, linestyle="-.")
        best_hd95_idx = int(np.nanargmin(hd95_vals))
        best_hd95_ep = epochs[best_hd95_idx]
        best_hd95_val = hd95_vals[best_hd95_idx]
        ax4_right.plot(best_hd95_ep, best_hd95_val, marker="D", markersize=9,
                       color="#f87171", markeredgecolor="#991b1b", markeredgewidth=1.2,
                       label=f"Best HD95: {best_hd95_val:.4f} mm (E{best_hd95_ep})", zorder=5)
        ax4_right.set_ylabel("HD95 (mm)", fontsize=10, color=c_red, fontweight="semibold")
        ax4_right.set_ylim(bottom=-0.05)

    ax4.set_title("PCK Accuracy" + (" & Hausdorff Distance (HD95)" if has_hd95 else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax4.set_xlabel("Epochs", fontsize=10, color="black")
    ax4.set_ylabel("Accuracy (%)", fontsize=10, color="black", fontweight="semibold")
    ax4.set_ylim(bottom=-5, top=108)

    handles_l4, labels_l4 = ax4.get_legend_handles_labels()
    if has_hd95:
        handles_r4, labels_r4 = ax4_right.get_legend_handles_labels()
        leg4 = ax4_right.legend(handles_l4 + handles_r4, labels_l4 + labels_r4,
                                loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    else:
        leg4 = ax4.legend(handles_l4, labels_l4, loc="upper left", frameon=True,
                          facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg4:
        leg4.get_frame().set_alpha(0.96)
        leg4.set_zorder(100)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def _mean_curve(curve: Any) -> np.ndarray:
    array = np.asarray(curve, dtype=float)
    return np.nanmean(array, axis=0) if array.ndim > 1 else array


def plot_pose_metrics(metrics: Any, output_png: str | Path) -> Path:
    """Write normalized confusion, pose/box PR, and F1-confidence in one figure."""
    destination = Path(output_png)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    raw_matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=float)[:4, :4]
    norm_matrix = raw_matrix / np.maximum(raw_matrix.sum(axis=0, keepdims=True), 1e-9)
    gt_counts = raw_matrix.sum(axis=0).astype(int)

    ax = axes[0, 0]
    image = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    _style(ax)
    ax.grid(False)  # Remove grid lines slicing through heatmap text

    for row in range(4):
        for column in range(4):
            val = norm_matrix[row, column]
            raw_cnt = int(raw_matrix[row, column])
            # Subtle adaptive frosted glass pill: harmonious, elegant, high legibility
            if val > 0.45:
                text_color = "#ffffff"
                bg_box = "#000000"
                bg_alpha = 0.25
            else:
                text_color = "#0f172a"
                bg_box = "#ffffff"
                bg_alpha = 0.55

            cell_text = f"{val * 100:.1f}%\n(n={raw_cnt})"
            ax.text(
                column,
                row,
                cell_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25,rounding_size=0.4",
                    facecolor=bg_box,
                    alpha=bg_alpha,
                    edgecolor="none",
                ),
            )

    x_labels = [f"{name.title()}\n(N={gt_counts[i]})" for i, name in enumerate(REGION_NAMES)]
    ax.set_xticks(range(4), x_labels, rotation=30, ha="right")
    ax.set_yticks(range(4), [name.title() for name in REGION_NAMES])
    ax.set(title="Normalized Confusion Matrix (Counts & %)", xlabel="True", ylabel="Predicted")
    fig.colorbar(image, ax=ax, fraction=0.046)

    for ax, metric, title in (
        (axes[0, 1], metrics.pose, "Pose PR Curve"),
        (axes[1, 0], metrics.box, "Box PR Curve"),
    ):
        x = np.asarray(getattr(metric, "px", np.linspace(0, 1, 1000)))
        precision = _mean_curve(getattr(metric, "prec_values", np.zeros_like(x)))
        ax.plot(x, precision, color=COLORS[0], linewidth=2)
        ax.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
        _style(ax)

    ax = axes[1, 1]
    for metric, label, color in ((metrics.pose, "Pose F1", COLORS[1]), (metrics.box, "Box F1", COLORS[0])):
        x = np.asarray(getattr(metric, "px", np.linspace(0, 1, 1000)))
        f1 = _mean_curve(getattr(metric, "f1_curve", np.zeros_like(x)))
        ax.plot(x, f1, color=color, linewidth=2, label=label)
    ax.set(title="F1-Score / Confidence", xlabel="Confidence", ylabel="F1", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    _style(ax)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_validation_samples(records: list[dict[str, Any]], output_png: str | Path) -> Path | None:
    """Render four validation images side-by-side in one row (1x4 grid)."""
    if not records:
        return None
    destination = Path(output_png)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=150)
    axes_list = list(axes) if isinstance(axes, np.ndarray) else [axes]
    for ax, record in zip(axes_list, records[:4]):
        image = record["image"]
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        pred, valid = record["pred"], record["valid"]
        offset = 0
        for color, name, count in zip(COLORS, REGION_NAMES, (45, 51, 24, 9)):
            local = pred[offset : offset + count]
            local_valid = valid[offset : offset + count] & np.isfinite(local).all(axis=1)
            ax.scatter(local[local_valid, 0], local[local_valid, 1], s=7, color=color, label=name.title())
            offset += count
        for start, stop in LANDMARK_PATH_RANGES:
            path = pred[start:stop]
            mask = valid[start:stop] & np.isfinite(path).all(axis=1)
            if mask.sum() >= 2:
                ax.plot(path[mask, 0], path[mask, 1], color="#fde047", linewidth=0.8)
        banner = (
            f"MRE {record['mre_px'] * 0.10:.3f} mm | PCK {record['pck2'] * 100:.1f}%\n"
            f"HD95 {record['hd95_px'] * 0.10:.3f} mm | IoU {record['box_iou'] * 100:.1f}%"
        )
        ax.text(0.01, 0.99, banner, transform=ax.transAxes, va="top", ha="left", color="white", fontsize=7.5,
                bbox={"facecolor": "black", "alpha": 0.72, "pad": 3})
        ax.set_title(Path(record["path"]).name, fontsize=9)
        ax.axis("off")
    for ax in axes_list[len(records[:4]) :]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)
    return destination


__all__ = ["plot_dashboard_pose", "plot_pose_metrics", "plot_validation_samples"]
