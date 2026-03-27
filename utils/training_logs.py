import csv
import json
import logging
from pathlib import Path

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

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_training_args(log_dir, args_dict):
    path = Path(log_dir) / "training_args.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(args_dict), file, indent=4)
    return path


class EpochLogWriter:
    def __init__(self, log_dir, file_stem="epoch_metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{file_stem}.csv"
        self.jsonl_path = self.log_dir / f"{file_stem}.jsonl"
        self.summary_path = self.log_dir / f"{file_stem}_summary.json"
        self._fieldnames = None

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

        with self.jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(clean_row) + "\n")

    def write_summary(self, summary):
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_loss, ax_metric = axes

    line_styles = ["-", "--", "-.", ":"]
    for index, (label, values) in enumerate(loss_series.items()):
        ax_loss.plot(
            epochs,
            values,
            linewidth=2.2,
            linestyle=line_styles[index % len(line_styles)],
            label=label,
        )

    ax_loss.set_title("Loss by Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3, linestyle="--")

    metric_handles = []
    for index, (label, values) in enumerate(metric_series.items()):
        handle = ax_metric.plot(
            epochs,
            values,
            linewidth=2.0,
            linestyle=line_styles[index % len(line_styles)],
            label=label,
        )[0]
        metric_handles.append(handle)

    ranking_metric_values = _extract_series(history_rows, ranking_key)
    val_loss_values = loss_series.get("Validation Loss")
    marker_styles = [
        ("gold", "*", 220),
        ("orangered", "D", 90),
        ("dodgerblue", "o", 80),
    ]
    top_handles = []
    for rank_index, (epoch, score) in enumerate(top_epochs):
        epoch_position = np.where(epochs == epoch)[0]
        if len(epoch_position) == 0:
            continue
        idx = int(epoch_position[0])
        color, marker, size = marker_styles[min(rank_index, len(marker_styles) - 1)]
        if val_loss_values is not None and idx < len(val_loss_values):
            ax_loss.scatter(
                epoch,
                val_loss_values[idx],
                color=color,
                marker=marker,
                s=size,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        top_handle = ax_metric.scatter(
            epoch,
            ranking_metric_values[idx],
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"Top{rank_index + 1} {ranking_key}: epoch {epoch}, {score:.4f}",
        )
        ax_metric.annotate(
            f"E{epoch}",
            (epoch, ranking_metric_values[idx]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
            color=color,
            weight="bold",
        )
        top_handles.append(top_handle)

    ax_metric.set_title("Metrics by Epoch")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel("Score")
    ax_metric.grid(alpha=0.3, linestyle="--")

    ax_loss.legend(loc="best")
    ax_metric.legend(handles=[*metric_handles, *top_handles], loc="best", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    output_path = log_dir / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_summary = [
        {"rank": rank + 1, "epoch": epoch, "value": score}
        for rank, (epoch, score) in enumerate(top_epochs)
    ]
    return output_path, top_summary
