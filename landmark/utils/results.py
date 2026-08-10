"""YOLO-compatible results enriched with the canonical 129-point view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from landmark.data.schema import NUM_LANDMARKS, objects_to_canonical


@dataclass
class KneePoseResult:
    """Delegate standard result fields while exposing canonical landmarks."""

    raw: Any
    landmarks_xy: torch.Tensor
    landmark_confidence: torch.Tensor

    def __getattr__(self, name: str):
        return getattr(self.raw, name)

    @property
    def boxes_xyxy(self) -> torch.Tensor:
        return self.raw.boxes.xyxy

    @property
    def scores(self) -> torch.Tensor:
        return self.raw.boxes.conf

    @property
    def class_ids(self) -> torch.Tensor:
        return self.raw.boxes.cls.long()


def adapt_yolo_result(result: Any) -> KneePoseResult:
    """Add a normalized 129-landmark view to one Ultralytics Result."""
    device = result.boxes.xyxy.device if result.boxes is not None else torch.device("cpu")
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        xy = torch.zeros(NUM_LANDMARKS, 2, device=device)
        confidence = torch.zeros(NUM_LANDMARKS, device=device)
    else:
        data = result.keypoints.data
        xy, confidence = objects_to_canonical(data, result.boxes.cls, scores=result.boxes.conf)
        height, width = result.orig_shape
        scale = xy.new_tensor([max(width, 1), max(height, 1)])
        xy = (xy / scale).clamp(0, 1)
    return KneePoseResult(result, xy, confidence)


def plot_landmark_curves(
    csv_file: str,
    output_png: str | None = None,
    pixel_spacing: float = 0.1,
    model_name: str | None = None,
    elapsed_seconds: float | None = None,
) -> str | None:
    """Plot landmark training curves with safe, optional bbox mAP handling."""
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    csv_path = Path(csv_file)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "epoch" in df.columns:
        epochs = df["epoch"].values
    else:
        epochs = np.arange(1, len(df) + 1)

    save_path = output_png or str(csv_path.parent / "training_curves.png")

    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    # Title formatting
    m_name = model_name or csv_path.parent.name or "Landmark Model"
    time_info = ""
    if elapsed_seconds and elapsed_seconds > 0:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        avg_ep = elapsed_seconds / max(1, len(epochs))
        time_info = f" | Train Time: {mins}m {secs}s ({avg_ep:.1f}s/ep)"

    fig.suptitle(f"{m_name}{time_info} | Pixel Spacing: {pixel_spacing} mm/px",
                 color="#1e293b", fontsize=14, fontweight="bold", y=0.98)

    # Subplot 1: Clean Train & Val Loss
    ax1 = axes[0, 0]
    if "train/loss" in df.columns or "train/pose_loss" in df.columns:
        t_loss_col = "train/loss" if "train/loss" in df.columns else "train/pose_loss"
        ax1.plot(epochs, df[t_loss_col], label="Train Loss", color="#1f77b4", lw=2.2)
    if "val/loss" in df.columns or "val/pose_loss" in df.columns:
        v_loss_col = "val/loss" if "val/loss" in df.columns else "val/pose_loss"
        val_losses = df[v_loss_col].values
        ax1.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", lw=2.2, linestyle="--")
        best_loss_idx = np.nanargmin(val_losses)
        ax1.plot(epochs[best_loss_idx], val_losses[best_loss_idx], marker="*", markersize=18,
                 color="#FFD700", markeredgecolor="#B8860B", markeredgewidth=1.5,
                 label=f"Val Loss: {val_losses[best_loss_idx]:.4f} (Epoch {epochs[best_loss_idx]})", zorder=5)
        ax1.axvline(x=epochs[best_loss_idx], color="#B8860B", linestyle=":", alpha=0.7, lw=1.5)

    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold", color="#1e293b")
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_ylim(bottom=-0.08)
    leg1 = ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.5)
    if leg1:
        leg1.get_frame().set_alpha(0.96)

    # Subplot 2: Overall MRE & optional Box mAP50-95 twin Y axis
    ax2 = axes[0, 1]
    best_mre_idx = 0
    if "metrics/MRE" in df.columns:
        mre_px = df["metrics/MRE"].values
        mre_mm = mre_px * pixel_spacing
        best_mre_idx = np.nanargmin(mre_mm)
        best_mre_epoch = epochs[best_mre_idx]
        best_mre_val = mre_mm[best_mre_idx]

        ax2.plot(epochs, mre_mm, label="Overall MRE", color="#d62728", lw=2.5)
        ax2.axhline(y=best_mre_val, color="#64748b", linestyle=":", alpha=0.7)
        ax2.axvline(x=best_mre_epoch, color="#B8860B", linestyle=":", alpha=0.7, lw=1.5)
        ax2.plot(best_mre_epoch, best_mre_val, marker="*", markersize=20, color="#FFD700",
                 markeredgecolor="#8B0000", markeredgewidth=1.5,
                 label=f"MRE: {best_mre_val:.4f} mm (Epoch {best_mre_epoch})", zorder=5)
        ax2.annotate(f"{best_mre_val:.4f} mm\n(Epoch {best_mre_epoch})",
                     xy=(best_mre_epoch, best_mre_val),
                     xytext=(max(1, best_mre_epoch - 14), best_mre_val + 0.15),
                     arrowprops=dict(facecolor="#FFD700", edgecolor="#1e293b", shrink=0.08, width=1.5, headwidth=8),
                     fontsize=9.5, fontweight="bold", color="#1e293b",
                     bbox=dict(boxstyle="round,pad=0.4", fc="#fffbea", ec="#B8860B", lw=1.2))

    # Safely check and plot Box mAP50-95 only if available and non-empty
    has_bbox_map = "metrics/mAP50-95(B)" in df.columns and not df["metrics/mAP50-95(B)"].dropna().empty
    if has_bbox_map:
        bbox_vals = df["metrics/mAP50-95(B)"].values
        valid_mask = np.isfinite(bbox_vals)
        if valid_mask.any():
            ax2_right = ax2.twinx()
            ax2_right.grid(False)
            ax2_right.plot(epochs, bbox_vals, label="Box mAP50-95", color="#2563eb", lw=2.0, linestyle="--")
            best_map_idx = np.nanargmax(bbox_vals)
            best_map_epoch = epochs[best_map_idx]
            best_map_val = bbox_vals[best_map_idx]
            ax2_right.plot(best_map_epoch, best_map_val, marker="*", markersize=18,
                           color="#60a5fa", markeredgecolor="#1e3a8a", markeredgewidth=1.5,
                           label=f"Box mAP50-95: {best_map_val:.4f} (Epoch {best_map_epoch})", zorder=5)
            ax2_right.set_ylabel("Box mAP50-95", fontsize=10, color="#2563eb", fontweight="semibold")
            ax2_right.set_ylim(bottom=-0.04, top=1.05)

    ax2.set_title("Mean Radial Error (MRE)" + (" & Box mAP50-95" if has_bbox_map else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("MRE Error (mm)", fontsize=10, color="#d62728", fontweight="semibold")
    ax2.set_ylim(bottom=-0.05)

    # Combine legends safely at upper left with nearly opaque frame
    handles_left, labels_left = ax2.get_legend_handles_labels()
    if has_bbox_map:
        handles_right, labels_right = ax2_right.get_legend_handles_labels()
        leg2 = ax2.legend(handles_left + handles_right, labels_left + labels_right,
                          loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    else:
        leg2 = ax2.legend(handles_left, labels_left, loc="upper left", frameon=True,
                          facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg2:
        leg2.get_frame().set_alpha(0.96)

    # Subplot 3: Per-Region MRE
    ax3 = axes[1, 0]
    region_cols = [("metrics/MRE_femur", "Femur", "#2ca02c"),
                   ("metrics/MRE_tibia", "Tibia", "#9467bd"),
                   ("metrics/MRE_fibula", "Fibula", "#8c564b"),
                   ("metrics/MRE_patella", "Patella", "#e377c2")]
    for col, name, color in region_cols:
        if col in df.columns:
            r_mm = df[col].values * pixel_spacing
            best_val = r_mm[best_mre_idx] if len(r_mm) > best_mre_idx else r_mm[-1]
            ax3.plot(epochs, r_mm, label=f"{name}: {best_val:.4f} mm", color=color, lw=2.0)

    if "metrics/MRE" in df.columns:
        ax3.axvline(x=epochs[best_mre_idx], color="#B8860B", linestyle=":", alpha=0.7, lw=1.5,
                    label=f"Best Epoch ({epochs[best_mre_idx]})")

    ax3.set_title("Per-Region MRE (Femur, Tibia, Fibula, Patella in mm)", fontsize=12, fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Epochs", fontsize=10)
    ax3.set_ylabel("Error (mm)", fontsize=10)
    ax3.set_ylim(bottom=-0.05)
    leg3 = ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9.0)
    if leg3:
        leg3.get_frame().set_alpha(0.96)

    # Subplot 4: PCK Accuracy Metrics + optional HD95 twin Y axis
    ax4 = axes[1, 1]
    has_hd95 = "metrics/HD95" in df.columns or "val/hd95" in df.columns
    pck_cols = [("metrics/PCK2", "PCK@0.2mm", "#17becf"),
                ("metrics/PCK4", "PCK@0.4mm", "#bcbd22"),
                ("metrics/PCK8", "PCK@0.8mm", "#64748b")]
    for col, name, color in pck_cols:
        if col in df.columns:
            p_vals = df[col].values * 100.0  # convert 0-1 ratio to %
            best_pck = p_vals[best_mre_idx] if len(p_vals) > best_mre_idx else p_vals[-1]
            ax4.plot(epochs, p_vals, label=f"{name}: {best_pck:.2f}%", color=color, lw=2.0)

    if "metrics/MRE" in df.columns:
        ax4.axvline(x=epochs[best_mre_idx], color="#B8860B", linestyle=":", alpha=0.7, lw=1.5,
                    label=f"Best Epoch ({epochs[best_mre_idx]})")

    if has_hd95:
        hd95_col = "metrics/HD95" if "metrics/HD95" in df.columns else "val/hd95"
        hd95_vals = df[hd95_col].values * pixel_spacing
        ax4_right = ax4.twinx()
        ax4_right.grid(False)
        ax4_right.plot(epochs, hd95_vals, label="Val HD95 (mm)", color="#10b981", lw=2.0, linestyle="-.")
        best_hd95_idx = np.nanargmin(hd95_vals)
        best_hd95_epoch = epochs[best_hd95_idx]
        best_hd95_val = hd95_vals[best_hd95_idx]
        ax4_right.plot(best_hd95_epoch, best_hd95_val, marker="D", markersize=11,
                       color="#34d399", markeredgecolor="#047857", markeredgewidth=1.5,
                       label=f"HD95: {best_hd95_val:.4f} mm (Epoch {best_hd95_epoch})", zorder=5)
        ax4_right.set_ylabel("HD95 (mm)", fontsize=10, color="#047857", fontweight="semibold")
        ax4_right.set_ylim(bottom=-0.05)

    ax4.set_title("PCK Accuracy" + (" & Hausdorff Distance (HD95)" if has_hd95 else ""),
                  fontsize=12, fontweight="bold", color="#1e293b")
    ax4.set_xlabel("Epochs", fontsize=10)
    ax4.set_ylabel("Accuracy (%)", fontsize=10)
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

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path



