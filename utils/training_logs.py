import csv
import json
import logging
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "uknee-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def to_python_number(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_serializable(value):
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return to_python_number(value)


def to_serializable_dict(row):
    return {key: to_serializable(value) for key, value in row.items()}


def setup_logger(log_file, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_training_args(log_dir, args_dict, filename="training_args.json"):
    path = Path(log_dir) / filename
    serializable = to_serializable(args_dict)
    with path.open("w", encoding="utf-8") as file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml

                yaml.safe_dump(serializable, file, sort_keys=False, allow_unicode=True)
            except ImportError:
                # JSON is valid YAML 1.2 and keeps this utility dependency-light.
                json.dump(serializable, file, indent=2)
        else:
            json.dump(serializable, file, indent=4)
    return path


class EpochLogWriter:
    def __init__(self, log_dir, file_stem="epoch_metrics", fieldnames=None, write_auxiliary=True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{file_stem}.csv"
        self.jsonl_path = self.log_dir / f"{file_stem}.jsonl"
        self.summary_path = self.log_dir / f"{file_stem}_summary.json"
        self._fieldnames = list(fieldnames) if fieldnames else None
        self.write_auxiliary = bool(write_auxiliary)

    def append(self, row):
        clean_row = to_serializable_dict(row)
        if self._fieldnames is None:
            self._fieldnames = list(clean_row.keys())

        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(clean_row)

        if self.write_auxiliary:
            with self.jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(clean_row) + "\n")

    def write_summary(self, summary):
        if self.write_auxiliary:
            with self.summary_path.open("w", encoding="utf-8") as file:
                json.dump(to_serializable(summary), file, indent=4)


def _extract_series(history_rows, key):
    values = []
    for row in history_rows:
        if key not in row:
            values.append(np.nan)
            continue
        value = to_python_number(row[key])
        values.append(float(value) if value is not None else np.nan)
    return np.asarray(values, dtype=float)


def _top_ranked_epochs(epochs, scores, maximize=True, top_k=2):
    if len(epochs) == 0:
        return []
    pairs = [
        (int(epoch), float(score))
        for epoch, score in zip(epochs, scores)
        if score is not None and np.isfinite(score)
    ]
    if not pairs:
        return []
    pairs.sort(key=lambda item: item[1], reverse=maximize)
    return pairs[:top_k]


def plot_training_dashboard(
    log_dir,
    history_rows,
    loss_keys,
    metric_keys,
    ranking_key,
    maximize=True,
    top_k=3,
    filename="training_dashboard.png",
    title=None,
    model_name="RWKV_UNet",
    elapsed_seconds=None,
):
    if not history_rows:
        return None, []

    log_dir = Path(log_dir)
    epochs = np.asarray([int(row["epoch"]) for row in history_rows], dtype=int)
    loss_series = {
        label: _extract_series(history_rows, key) for key, label in loss_keys
    }
    metric_series = {
        label: _extract_series(history_rows, key) for key, label in metric_keys
    }
    ranking_scores = _extract_series(history_rows, ranking_key)
    top_epochs = _top_ranked_epochs(epochs, ranking_scores, maximize=maximize, top_k=top_k)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    def style_axes(ax):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
        ax.tick_params(axis="both", which="both", direction="out", length=5, width=1.0, colors="black")

    time_info = ""
    if elapsed_seconds and elapsed_seconds > 0:
        mins = int(elapsed_seconds // 60)
        secs = int(elapsed_seconds % 60)
        avg_ep = elapsed_seconds / max(1, len(epochs))
        time_info = f" | Train Time: {mins}m {secs}s ({avg_ep:.1f}s/ep)"

    main_title = f"{title or f'Segmentation: {model_name}'}{time_info}"
    fig.suptitle(main_title, color="#1e293b", fontsize=14, fontweight="bold", y=0.98)

    marker_styles = [("#FFD700", "*"), ("#ff7f0e", "D"), ("#1f77b4", "o")]

    # Subplot 1: Loss by Epoch + Top 1,2,3 Val Loss markers in Legend
    ax1 = axes[0, 0]
    style_axes(ax1)
    line_styles = ["-", "--", "-.", ":"]
    for idx, (label, values) in enumerate(loss_series.items()):
        ax1.plot(epochs, values, lw=2.2, linestyle=line_styles[idx % len(line_styles)], label=label)

    val_loss_vals = next(
        (values for label, values in loss_series.items() if "val" in label.lower()),
        None,
    )
    top_loss_indices = (
        np.argsort(np.where(np.isfinite(val_loss_vals), val_loss_vals, np.inf))[:top_k]
        if val_loss_vals is not None and np.isfinite(val_loss_vals).any()
        else []
    )
    for rank, i in enumerate(top_loss_indices):
        c, m = marker_styles[rank % len(marker_styles)]
        ep = epochs[i]
        val = val_loss_vals[i]
        ax1.plot(ep, val, marker=m, markersize=14 if m == "*" else 8, color=c, markeredgecolor="black",
                 markeredgewidth=1.2, linestyle="None", label=f"Top{rank+1} Val Loss: {val:.4f} (E{ep})", zorder=5)

    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold", color="#1e293b")
    ax1.set_xlabel("Epochs", fontsize=10, color="black")
    ax1.set_ylabel("Loss", fontsize=10, color="black", fontweight="semibold")
    ax1.set_ylim(bottom=-0.05)
    leg1 = ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg1:
        leg1.get_frame().set_alpha(0.96)
        leg1.set_zorder(100)

    # Subplot 2: Overlap Metrics (Dice & IoU) + Top 1,2,3 Dice markers in Legend
    ax2 = axes[0, 1]
    style_axes(ax2)
    overlap_series = {
        label: values for label, values in metric_series.items()
        if "dice" in label.lower() or "iou" in label.lower() or "jaccard" in label.lower()
    }
    for idx, (label, values) in enumerate(overlap_series.items()):
        display = values * 100.0 if np.nanmax(np.abs(values)) <= 1.01 else values
        ax2.plot(epochs, display, lw=2.2, linestyle=line_styles[idx % len(line_styles)], label=label)

    for rank_idx, (ep, score) in enumerate(top_epochs):
        c, m = marker_styles[rank_idx % len(marker_styles)]
        ep_pos = np.where(epochs == ep)[0]
        if len(ep_pos) > 0:
            i = ep_pos[0]
            display_score = score * 100.0 if abs(score) <= 1.01 else score
            ax2.plot(ep, display_score, marker=m, markersize=14 if m == "*" else 8, color=c,
                     markeredgecolor="black", markeredgewidth=1.2, linestyle="None",
                     label=f"Top{rank_idx+1} Dice: {display_score:.2f}% (E{ep})", zorder=5)

    ax2.set_title("Segmentation Overlap Metrics (Dice & IoU)", fontsize=12, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Epochs", fontsize=10, color="black")
    ax2.set_ylabel("Score (%)", fontsize=10, color="black", fontweight="semibold")
    ax2.set_ylim(bottom=-5, top=105)
    leg2 = ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg2:
        leg2.get_frame().set_alpha(0.96)
        leg2.set_zorder(100)

    # Subplot 3: Boundary Distance Metrics (HD95 & ASSD in mm)
    ax3 = axes[1, 0]
    style_axes(ax3)
    ax3_right = ax3.twinx()
    style_axes(ax3_right)
    ax3_right.grid(False)

    hd95_key = next((key for key, _ in metric_keys if "hd95" in key.lower()), "val/hd95")
    assd_key = next((key for key, _ in metric_keys if "assd" in key.lower() or "asd" in key.lower()), "val/assd")
    hd95_vals = _extract_series(history_rows, hd95_key)
    assd_vals = _extract_series(history_rows, assd_key)
    if np.isfinite(hd95_vals).any():
        best_hd95_i = int(np.nanargmin(hd95_vals))
        ax3.plot(epochs, hd95_vals, label="HD95 (mm)", color="#d62728", lw=2.2)
        ax3.plot(epochs[best_hd95_i], hd95_vals[best_hd95_i], marker="*", markersize=16,
                 color="#FFD700", markeredgecolor="black", label=f"Best HD95: {hd95_vals[best_hd95_i]:.2f} mm")
    if np.isfinite(assd_vals).any():
        best_assd_i = int(np.nanargmin(assd_vals))
        ax3_right.plot(epochs, assd_vals, label="ASSD (mm)", color="#9467bd", lw=2.0, linestyle="--")
        ax3_right.plot(epochs[best_assd_i], assd_vals[best_assd_i], marker="D", markersize=8,
                       color="#c084fc", markeredgecolor="black", label=f"Best ASSD: {assd_vals[best_assd_i]:.2f} mm")

    ax3.set_title("Boundary Distance Metrics (HD95 & ASSD in mm)", fontsize=12, fontweight="bold", color="#1e293b")
    ax3.set_xlabel("Epochs", fontsize=10, color="black")
    ax3.set_ylabel("HD95 (mm)", fontsize=10, color="#d62728", fontweight="semibold")
    ax3_right.set_ylabel("ASSD (mm)", fontsize=10, color="#9467bd", fontweight="semibold")
    ax3.set_ylim(bottom=-0.1)
    ax3_right.set_ylim(bottom=-0.05)

    h_l3, l_l3 = ax3.get_legend_handles_labels()
    h_r3, l_r3 = ax3_right.get_legend_handles_labels()
    leg3 = ax3_right.legend(h_l3 + h_r3, l_l3 + l_r3, loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg3:
        leg3.get_frame().set_alpha(0.96)
        leg3.set_zorder(100)

    # Subplot 4: Classification Metrics (Sensitivity & Precision)
    ax4 = axes[1, 1]
    style_axes(ax4)
    sens_key = next((key for key, _ in metric_keys if "sens" in key.lower() or "recall" in key.lower()), "val/sens")
    prec_key = next((key for key, _ in metric_keys if "prec" in key.lower()), "val/prec")
    sens_vals = _extract_series(history_rows, sens_key)
    prec_vals = _extract_series(history_rows, prec_key)
    if np.isfinite(sens_vals).any():
        sens_display = sens_vals * 100.0 if np.nanmax(np.abs(sens_vals)) <= 1.01 else sens_vals
        ax4.plot(epochs, sens_display, label="Sensitivity / Recall (%)", color="#e377c2", lw=2.2)
    if np.isfinite(prec_vals).any():
        prec_display = prec_vals * 100.0 if np.nanmax(np.abs(prec_vals)) <= 1.01 else prec_vals
        ax4.plot(epochs, prec_display, label="Precision (%)", color="#bcbd22", lw=2.2, linestyle="--")

    ax4.set_title("Classification Metrics (Sensitivity & Precision)", fontsize=12, fontweight="bold", color="#1e293b")
    ax4.set_xlabel("Epochs", fontsize=10)
    ax4.set_ylabel("Score (%)", fontsize=10, color="#1e293b", fontweight="semibold")
    ax4.set_ylim(bottom=-5, top=105)
    leg4 = ax4.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.8)
    if leg4:
        leg4.get_frame().set_alpha(0.96)
        leg4.set_zorder(100)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = log_dir / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_summary = [
        {"rank": rank + 1, "epoch": epoch, "value": score}
        for rank, (epoch, score) in enumerate(top_epochs)
    ]
    return output_path, top_summary
