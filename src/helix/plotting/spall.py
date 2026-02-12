"""Spall analysis diagnostic plots."""
from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("helix")


def plot_spall_analysis(
    plot_path: str,
    time_window: np.ndarray,
    vel_window: np.ndarray,
    uncert_window: Optional[np.ndarray],
    peak_idx: Optional[int],
    valley_idx: Optional[int],
    spall_strength,
    spall_unc: float,
    base_name: str,
    lines_info=None,
    intersections=None,
) -> None:
    """Generate a spall analysis diagnostic plot and save to *plot_path*."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_window, vel_window, "b-", linewidth=1.5, alpha=0.7, label="Velocity")
    if uncert_window is not None and len(uncert_window) == len(vel_window):
        ax.fill_between(
            time_window, vel_window - uncert_window, vel_window + uncert_window,
            alpha=0.2, color="blue", label="Uncertainty",
        )

    if peak_idx is not None and peak_idx < len(time_window):
        ax.plot(time_window[peak_idx], vel_window[peak_idx], "ro", markersize=10,
                label=f"Peak: {vel_window[peak_idx]:.1f} m/s")
    if valley_idx is not None and valley_idx < len(time_window):
        ax.plot(time_window[valley_idx], vel_window[valley_idx], "go", markersize=10,
                label=f"Valley: {vel_window[valley_idx]:.1f} m/s")

    if (peak_idx is not None and valley_idx is not None
            and peak_idx < len(time_window) and valley_idx < len(time_window)):
        ax.plot(
            [time_window[peak_idx], time_window[valley_idx]],
            [vel_window[peak_idx], vel_window[valley_idx]],
            "r--", linewidth=2, alpha=0.5, label="Pullback",
        )

    # Overlay 5-segment hybrid if available
    if lines_info and intersections:
        _overlay_segments(ax, time_window, lines_info, intersections)

    # Title
    try:
        sv = float(spall_strength) if not isinstance(spall_strength, str) else None
    except (TypeError, ValueError):
        sv = None

    if sv is not None and np.isfinite(sv):
        if spall_unc is not None and not np.isnan(spall_unc) and spall_unc > 0:
            title = f"Spall Analysis: {base_name}\nSpall Strength: {sv:.3f} \u00b1 {spall_unc:.3f} GPa"
        else:
            title = f"Spall Analysis: {base_name}\nSpall Strength: {sv:.3f} GPa"
    else:
        title = f"Spall Analysis: {base_name}\nSpall Strength: {spall_strength}"

    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("Velocity (m/s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _overlay_segments(ax, time_window, lines_info, intersections):
    """Overlay 5-segment hybrid model lines."""
    import pandas as pd

    t_min = float(np.min(time_window)) if len(time_window) > 0 else 0.0
    t_max = float(np.max(time_window)) if len(time_window) > 0 else 0.0

    boundaries = [t_min]
    for pt in (intersections or []):
        if pt and len(pt) >= 2 and not any(pd.isna(c) for c in pt):
            boundaries.append(float(pt[0]))
    boundaries.append(t_max)

    while len(boundaries) <= len(lines_info):
        boundaries.append(boundaries[-1] + 1.0)
    if len(boundaries) > len(lines_info) + 1:
        boundaries = boundaries[: len(lines_info) + 1]

    colors = ["#FF8C00", "#B22222", "#228B22", "#1E90FF", "#8A2BE2"]
    for idx, ((m, c), color) in enumerate(zip(lines_info, (colors * 3)[: len(lines_info)])):
        ts, te = boundaries[idx], boundaries[idx + 1]
        if te <= ts:
            continue
        t = np.linspace(ts, te, 100)
        ax.plot(t, m * t + c, "--", linewidth=1.2, color=color, label=f"Segment {idx + 1}")

    for idx, pt in enumerate(intersections or [], start=1):
        if not pt or any(pd.isna(c) for c in pt):
            continue
        ax.plot(pt[0], pt[1], marker="X", color="black", markersize=7)
        ax.text(pt[0], pt[1], f"P{idx}", fontsize=9, ha="left", va="bottom")
