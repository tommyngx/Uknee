"""Plotting helpers shared by landmark and segmentation reports."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _compact_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.4g}"


def apply_robust_y_limit(
    axis: Any,
    series: Iterable[Any],
    *,
    epochs: Any | None = None,
    lower_bound: float = 0.0,
    warmup_epochs: int = 30,
    annotation_position: tuple[float, float] = (0.02, 0.98),
) -> dict[str, Any]:
    """Focus an unbounded metric plot on its converged region.

    The raw curves are left unchanged. Only the visible y-range is reduced
    when early spikes are materially larger than the post-warm-up values. A
    note records how many points are outside the view and their true maximum.
    Percentage/probability axes should not call this helper.
    """
    arrays = [np.asarray(values, dtype=float).reshape(-1) for values in series]
    arrays = [values for values in arrays if values.size]
    if not arrays:
        axis.set_ylim(bottom=lower_bound)
        return {"applied": False, "clipped_points": 0}

    max_length = max(len(values) for values in arrays)
    # At least two thirds of the available history remains in the focus set.
    # Once a run is long enough this settles at the requested 30-epoch warm-up.
    warmup_count = min(max(0, int(warmup_epochs)), max_length // 3)
    focus_values = np.concatenate(
        [values[min(warmup_count, len(values)) :] for values in arrays]
    )
    focus_values = focus_values[np.isfinite(focus_values)]
    all_values = np.concatenate(arrays)
    all_values = all_values[np.isfinite(all_values)]

    if focus_values.size < 4 or all_values.size < 4:
        axis.set_ylim(bottom=lower_bound)
        return {"applied": False, "clipped_points": 0}

    low_quantile, high_quantile = np.percentile(focus_values, [2.0, 98.0])
    scale = max(
        float(high_quantile - low_quantile),
        abs(float(high_quantile)) * 0.05,
        1e-9,
    )
    visible_top = max(
        float(high_quantile + 0.15 * scale),
        float(lower_bound + scale),
    )
    global_max = float(np.max(all_values))
    # Ordinary monotonic convergence should keep Matplotlib's natural range.
    # Zoom only when the global peak is clearly on a different scale from the
    # post-warm-up distribution.
    material_margin = max(3.0 * scale, 0.75 * max(abs(visible_top), 1e-9))

    if global_max <= visible_top + material_margin:
        axis.set_ylim(bottom=lower_bound)
        return {
            "applied": False,
            "clipped_points": 0,
            "global_max": global_max,
        }

    clipped_points = int(np.count_nonzero(all_values > visible_top))
    axis.set_ylim(bottom=lower_bound, top=visible_top)

    if epochs is not None:
        epoch_values = np.asarray(epochs).reshape(-1)
        focus_epoch = epoch_values[min(warmup_count, len(epoch_values) - 1)] if epoch_values.size else warmup_count + 1
    else:
        focus_epoch = warmup_count + 1

    axis.text(
        *annotation_position,
        f"Focused after E{focus_epoch}\n"
        f"{clipped_points} high point{'s' if clipped_points != 1 else ''} hidden; "
        f"max={_compact_number(global_max)}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#7f1d1d",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#fff7ed",
            "edgecolor": "#fdba74",
            "alpha": 0.92,
        },
        zorder=90,
    )
    return {
        "applied": True,
        "clipped_points": clipped_points,
        "global_max": global_max,
        "visible_top": visible_top,
        "focus_epoch": int(focus_epoch),
    }



def format_duration(elapsed_seconds: float | None) -> str:
    """Format duration in seconds into 'Xh Ym Zs' or 'Ym Zs'."""
    if elapsed_seconds is None or elapsed_seconds < 0:
        return "0m 0s"
    elapsed = float(elapsed_seconds)
    hours, remainder = divmod(int(round(elapsed)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"


__all__ = ["apply_robust_y_limit", "format_duration"]
