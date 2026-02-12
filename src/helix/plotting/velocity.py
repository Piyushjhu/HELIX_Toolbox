"""Combined and all-traces velocity plotting."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("helix")


def plot_combined_velocity(
    vel_files: List[str],
    output_dir: str,
    spade_params: Dict[str, Any],
    param_data: Optional[Dict[str, Dict]] = None,
    material_resolver=None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Generate a combined velocity traces plot (main + zoomed panels).

    Returns the saved figure path, or None on failure.
    """
    def _emit(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    if not vel_files:
        _emit("No velocity files for combined plot.")
        return None

    threshold_vel = spade_params.get("align_velocity_threshold_ms", 10.0)
    x_min = spade_params.get("x_min_main", -40)
    x_max = spade_params.get("x_max_main", 800)
    y_min = spade_params.get("y_min_main", 0)
    y_max = spade_params.get("y_max_main", 1000)
    include_unc = spade_params.get("include_uncert_bands", True)
    unc_alpha = spade_params.get("uncert_alpha", 0.2)

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(20, 8))

    for vel_file in vel_files:
        try:
            df = pd.read_csv(vel_file, header=None)
            time_ns = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
            velocity = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
            uncertainty = (
                pd.to_numeric(df.iloc[:, 2], errors="coerce").values
                if len(df.columns) >= 3
                else None
            )
        except Exception:
            continue

        valid = ~np.isnan(velocity)
        if np.sum(valid) < 10:
            continue

        # Align to threshold
        above = np.where(velocity[valid] >= threshold_vel)[0]
        if len(above) == 0:
            t_offset = 0
        else:
            t_offset = time_ns[valid][above[0]]

        t_aligned = time_ns - t_offset

        label = os.path.splitext(os.path.basename(vel_file))[0]
        for suffix in ("--vel-smooth-with-uncert", "--velocity--smooth"):
            label = label.replace(suffix, "")

        line, = ax_main.plot(t_aligned, velocity, linewidth=1, alpha=0.7, label=label)
        ax_zoom.plot(t_aligned, velocity, linewidth=1, alpha=0.7, label=label)

        if include_unc and uncertainty is not None:
            ax_main.fill_between(t_aligned, velocity - uncertainty, velocity + uncertainty,
                                  alpha=unc_alpha, color=line.get_color())

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel("Time (ns)")
    ax_main.set_ylabel("Velocity (m/s)")
    ax_main.set_title("Combined Velocity Traces (Full)")
    ax_main.grid(True, alpha=0.3)

    zoom_x_min = spade_params.get("x_min_zoom", -10)
    zoom_x_max = spade_params.get("x_max_zoom", 400)
    zoom_y_min = spade_params.get("y_min_zoom", 0)
    zoom_y_max = spade_params.get("y_max_zoom", 1000)
    ax_zoom.set_xlim(zoom_x_min, zoom_x_max)
    ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
    ax_zoom.set_xlabel("Time (ns)")
    ax_zoom.set_ylabel("Velocity (m/s)")
    ax_zoom.set_title("Combined Velocity Traces (Zoomed)")
    ax_zoom.grid(True, alpha=0.3)

    plt.tight_layout()
    dest = os.path.join(output_dir, "combined_velocity_traces.png")
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _emit(f"Combined velocity plot saved: {dest}")
    return dest


def plot_all_velocity_traces(
    vel_files: List[str],
    output_dir: str,
    spade_params: Dict[str, Any],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Generate an overlay of all velocity traces in a single panel."""
    def _emit(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    if not vel_files:
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    threshold_vel = spade_params.get("align_velocity_threshold_ms", 10.0)

    for vel_file in vel_files:
        try:
            df = pd.read_csv(vel_file, header=None)
            time_ns = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
            velocity = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        except Exception:
            continue

        valid = ~np.isnan(velocity)
        if np.sum(valid) < 10:
            continue

        above = np.where(velocity[valid] >= threshold_vel)[0]
        t_offset = time_ns[valid][above[0]] if len(above) > 0 else 0
        ax.plot(time_ns - t_offset, velocity, linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("All Velocity Traces")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    dest = os.path.join(output_dir, "all_velocity_traces.png")
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _emit(f"All traces plot saved: {dest}")
    return dest
