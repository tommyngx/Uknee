# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V9 epoch dashboard plot for loss, detection, refinement loss, and pose validation performance."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

from ultralytics.utils import RANK


def find_matching_key(headers: Sequence[str], candidate_keys: Sequence[str]) -> str | None:
    """Find the first matching key from candidate keys in the available headers."""
    for candidate in candidate_keys:
        if candidate in headers:
            return candidate
    for candidate in candidate_keys:
        clean_cand = candidate.split("/")[-1].lower()
        for header in headers:
            if clean_cand in header.lower():
                return header
    return None


def get_top_indices(values: list[float], epochs: list[int], top_k: int = 3, mode: str = "max") -> list[int]:
    """Return top_k indices based on values (either max or min) excluding non-finite values."""
    candidates = [i for i, v in enumerate(values) if math.isfinite(v)]
    if not candidates:
        return []
    if mode == "max":
        sorted_indices = sorted(candidates, key=lambda i: (-values[i], epochs[i]))
    else:
        sorted_indices = sorted(candidates, key=lambda i: (values[i], epochs[i]))
    return sorted_indices[:top_k]


def render_v9_training_dashboard(csv_path: str | Path, output_path: str | Path) -> Path | None:
    """Render a 2x2 dashboard containing:
    1. Bounding Box Loss & Pose Loss (with top losses in legend).
    2. Detection Performance mAP50 & mAP50-95 (Orange for mAP50, Red for mAP50-95).
    3. Refinement Loss (2 key refinement losses with top losses in legend).
    4. Pose Performance mAP50 & mAP50-95 (Sky Blue for mAP50, Red for mAP50-95).
    Missing metrics/losses for any model are skipped cleanly.
    """
    csv_path, output_path = Path(csv_path), Path(output_path)
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None

    headers = list(rows[0].keys())
    epochs = [int(float(row["epoch"])) for row in rows if "epoch" in row]
    if not epochs:
        return None

    import matplotlib.pyplot as plt

    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.patch.set_facecolor("#f7f7f7")
    fig.suptitle("YOLO26 Pose v9 — Loss & Performance Dashboard", fontsize=20, fontweight="bold", color="#161A1F")

    # Define candidate keys for each plot
    box_loss_candidates = ["val/box_loss", "train/box_loss", "box_loss", "train/box", "val/box"]
    pose_loss_candidates = ["val/pose_loss", "train/pose_loss", "pose_loss", "train/pose", "val/pose"]

    det_map50_candidates = ["metrics/mAP50(B)", "val/mAP50(B)", "mAP50(B)", "metrics/mAP50"]
    det_map50_95_candidates = ["metrics/mAP50-95(B)", "val/mAP50-95(B)", "mAP50-95(B)", "metrics/mAP50-95"]

    refine_loss1_candidates = [
        "train/hm_loss", "val/hm_loss", "hm_loss", "train/refine_hm_loss", "val/refine_hm_loss",
        "train/kobj_loss", "val/kobj_loss", "train/cls_loss", "val/cls_loss"
    ]
    refine_loss2_candidates = [
        "train/hm_coord_loss", "val/hm_coord_loss", "hm_coord_loss", "train/refine_coord_loss",
        "val/refine_coord_loss", "train/dfl_loss", "val/dfl_loss", "train/rle_loss", "val/rle_loss"
    ]

    pose_map50_candidates = ["metrics/mAP50(P)", "val/mAP50(P)", "mAP50(P)"]
    pose_map50_95_candidates = ["metrics/mAP50-95(P)", "val/mAP50-95(P)", "mAP50-95(P)"]

    def plot_line_and_get_values(ax, candidate_keys, fallback_name, line_color, line_style="-"):
        key = find_matching_key(headers, candidate_keys)
        if key and key in rows[0]:
            vals = [float(r[key]) for r in rows]
            clean_label = key.replace("metrics/", "").replace("train/", "").replace("val/", "")
            line, = ax.plot(epochs, vals, color=line_color, linestyle=line_style, linewidth=2.5, marker="o", markersize=4, label=clean_label)
            return vals, clean_label, line
        return None, fallback_name, None

    def finalize_legend(ax):
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(facecolor="white", edgecolor="#222831", fontsize=9.5, loc="best")

    # -------------------------------------------------------------------------
    # Plot 1: Bounding Box Loss & Pose Loss
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#f7f7f7")
    vals_box, label_box, _ = plot_line_and_get_values(ax1, box_loss_candidates, "Box Loss", "#E63946")
    vals_pose_loss, label_pose_loss, _ = plot_line_and_get_values(ax1, pose_loss_candidates, "Pose Loss", "#2A9D8F")

    if vals_box:
        best_box_idx = get_top_indices(vals_box, epochs, top_k=1, mode="min")
        if best_box_idx:
            idx = best_box_idx[0]
            val = vals_box[idx]
            ep = epochs[idx]
            ax1.scatter(ep, val, s=130, color="#D4AF37", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st {label_box}: {val:.4f} (Ep {ep})")

    if vals_pose_loss:
        best_pose_loss_idx = get_top_indices(vals_pose_loss, epochs, top_k=1, mode="min")
        if best_pose_loss_idx:
            idx = best_pose_loss_idx[0]
            val = vals_pose_loss[idx]
            ep = epochs[idx]
            ax1.scatter(ep, val, s=130, color="#CD7F32", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st {label_pose_loss}: {val:.4f} (Ep {ep})")

    ax1.set_title("1. Box & Pose Loss", color="#222831", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.45, color="navy")
    finalize_legend(ax1)

    # -------------------------------------------------------------------------
    # Plot 2: Detection Performance (mAP50 Orange #F4A261, mAP50-95 Red #E63946)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#f7f7f7")
    vals_det50, label_det50, _ = plot_line_and_get_values(ax2, det_map50_candidates, "Detection mAP50", "#F4A261")
    vals_det5095, label_det5095, _ = plot_line_and_get_values(ax2, det_map50_95_candidates, "Detection mAP50-95", "#E63946")

    if vals_det50:
        best_det50_idx = get_top_indices(vals_det50, epochs, top_k=1, mode="max")
        if best_det50_idx:
            idx = best_det50_idx[0]
            val = vals_det50[idx]
            ep = epochs[idx]
            ax2.scatter(ep, val, s=140, color="#D4AF37", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st mAP50(B): {val:.4f} (Ep {ep})")

    if vals_det5095:
        top2_det5095 = get_top_indices(vals_det5095, epochs, top_k=2, mode="max")
        top_colors = ["#A7A7AD", "#CD7F32"]
        rank_labels = ["1st", "2nd"]
        markers = ["*", "^"]
        for r_idx, idx in enumerate(top2_det5095):
            val = vals_det5095[idx]
            ep = epochs[idx]
            ax2.scatter(ep, val, s=130 if r_idx == 0 else 110, color=top_colors[r_idx], edgecolor="#222831", marker=markers[r_idx], zorder=5,
                        label=f"{rank_labels[r_idx]} mAP50-95(B): {val:.4f} (Ep {ep})")

    ax2.set_title("2. Detection Performance", color="#222831", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.grid(True, linestyle="--", alpha=0.45, color="navy")
    finalize_legend(ax2)

    # -------------------------------------------------------------------------
    # Plot 3: Refinement Loss
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#f7f7f7")
    vals_ref1, label_ref1, _ = plot_line_and_get_values(ax3, refine_loss1_candidates, "Refine Loss 1", "#8E44AD")
    vals_ref2, label_ref2, _ = plot_line_and_get_values(ax3, refine_loss2_candidates, "Refine Loss 2", "#2980B9")

    if vals_ref1:
        best_ref1_idx = get_top_indices(vals_ref1, epochs, top_k=1, mode="min")
        if best_ref1_idx:
            idx = best_ref1_idx[0]
            val = vals_ref1[idx]
            ep = epochs[idx]
            ax3.scatter(ep, val, s=130, color="#D4AF37", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st {label_ref1}: {val:.4f} (Ep {ep})")

    if vals_ref2:
        best_ref2_idx = get_top_indices(vals_ref2, epochs, top_k=1, mode="min")
        if best_ref2_idx:
            idx = best_ref2_idx[0]
            val = vals_ref2[idx]
            ep = epochs[idx]
            ax3.scatter(ep, val, s=130, color="#CD7F32", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st {label_ref2}: {val:.4f} (Ep {ep})")

    ax3.set_title("3. Refinement Loss", color="#222831", fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, linestyle="--", alpha=0.45, color="navy")
    finalize_legend(ax3)

    # -------------------------------------------------------------------------
    # Plot 4: Pose Performance (mAP50 Sky Blue #1E88E5, mAP50-95 Red #E63946)
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#f7f7f7")
    vals_pose50, label_pose50, _ = plot_line_and_get_values(ax4, pose_map50_candidates, "Pose mAP50", "#1E88E5")
    vals_pose5095, label_pose5095, _ = plot_line_and_get_values(ax4, pose_map50_95_candidates, "Pose mAP50-95", "#E63946")

    if vals_pose50:
        best_pose50_idx = get_top_indices(vals_pose50, epochs, top_k=1, mode="max")
        if best_pose50_idx:
            idx = best_pose50_idx[0]
            val = vals_pose50[idx]
            ep = epochs[idx]
            ax4.scatter(ep, val, s=140, color="#D4AF37", edgecolor="#222831", marker="*", zorder=5,
                        label=f"1st mAP50(P): {val:.4f} (Ep {ep})")

    if vals_pose5095:
        top2_pose5095 = get_top_indices(vals_pose5095, epochs, top_k=2, mode="max")
        top_colors = ["#A7A7AD", "#CD7F32"]
        rank_labels = ["1st", "2nd"]
        markers = ["*", "^"]
        for r_idx, idx in enumerate(top2_pose5095):
            val = vals_pose5095[idx]
            ep = epochs[idx]
            ax4.scatter(ep, val, s=130 if r_idx == 0 else 110, color=top_colors[r_idx], edgecolor="#222831", marker=markers[r_idx], zorder=5,
                        label=f"{rank_labels[r_idx]} mAP50-95(P): {val:.4f} (Ep {ep})")

    ax4.set_title("4. Pose Performance", color="#222831", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("mAP")
    ax4.grid(True, linestyle="--", alpha=0.45, color="navy")
    finalize_legend(ax4)

    # Format axis spines for all subplots
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_color("#161A1F")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_pose_detection_performance(csv_path: str | Path, output_path: str | Path) -> Path | None:
    """Backward compatibility wrapper rendering the v9 training dashboard."""
    return render_v9_training_dashboard(csv_path, output_path)


def plot_v9_performance_on_epoch_end(trainer) -> None:
    """Update the v9 dashboard after the current epoch metrics have been written for any model."""
    if RANK not in {-1, 0} or not getattr(trainer, "csv", None):
        return
    output = render_v9_training_dashboard(
        trainer.csv, Path(trainer.save_dir) / "pose_detection_performance.png"
    )
    if output is not None:
        trainer.on_plot(output)
