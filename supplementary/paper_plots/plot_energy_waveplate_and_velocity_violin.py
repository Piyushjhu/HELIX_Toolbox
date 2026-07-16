#!/usr/bin/env python3
"""
Two-panel figure from enhanced_spall_summary.csv (SPADE analysis output):

  Top panel  – Scatter: Laser Target Energy on x (continuous variation is obvious),
               Waveplate Angle on y (discrete nominal settings), coloured by Cu / Zn / Brass,
               light jitter only to reduce overlap.

  Bottom panel – Violin plot: Binned Laser Target Energy (x) vs Peak Free-Surface
                 Velocity (First_Maxima_m_s, y) coloured by material.

Usage:
    python plot_energy_waveplate_and_velocity_violin.py
    python plot_energy_waveplate_and_velocity_violin.py --csv /path/to/enhanced_spall_summary.csv
    python plot_energy_waveplate_and_velocity_violin.py --out my_figure.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_CSV = (
    "/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/"
    "Output_new/SPADE_analysis/enhanced_spall_summary.csv"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "energy_waveplate_velocity_violin.png")

# ---------------------------------------------------------------------------
# Material styling
# ---------------------------------------------------------------------------
MATERIALS = ["Cu", "Zn", "Brass"]

MATERIAL_COLORS = {
    "Cu":    "#B87333",   # copper
    "Zn":    "#4A90D9",   # steel-blue
    "Brass": "#D4A017",   # golden-brass
}

MATERIAL_MARKERS = {
    "Cu":    "o",
    "Zn":    "s",
    "Brass": "^",
}

MATERIAL_LABELS = {
    "Cu":    "Cu",
    "Zn":    "Zn",
    "Brass": "Brass",
}

# ---------------------------------------------------------------------------
# Energy-bin definitions
# The raw energies cluster naturally around these set-points (mJ):
#   ~890, ~1100, ~1350, ~1575, ~1800
# ---------------------------------------------------------------------------
ENERGY_BIN_EDGES   = [840, 1000, 1250, 1470, 1720, 1950]   # 6 edges → 5 bins
ENERGY_BIN_LABELS  = ["~890", "~1100", "~1350", "~1575", "~1800"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_filter(csv_path: str) -> pd.DataFrame:
    """Load CSV, keep Cu/Zn/Brass rows only, ensure required columns exist."""
    df = pd.read_csv(csv_path)

    # Normalise material column (strip whitespace, title-case Brass)
    df["Material"] = df["Material"].astype(str).str.strip()

    df = df[df["Material"].isin(MATERIALS)].copy()
    if df.empty:
        sys.exit(
            f"ERROR: No rows with Material in {MATERIALS} found in {csv_path}"
        )

    required = ["Waveplate_Angle (Degrees)", "Laser_Target_Energy (mJ)", "First_Maxima_m_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns: {missing}")

    return df


def bin_energies(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'Energy_Bin' categorical column using the defined edges/labels."""
    df = df.copy()
    df["Energy_Bin"] = pd.cut(
        df["Laser_Target_Energy (mJ)"],
        bins=ENERGY_BIN_EDGES,
        labels=ENERGY_BIN_LABELS,
        right=False,
    )
    return df


def add_jitter(values: np.ndarray, width: float = 0.12, seed: int = 42) -> np.ndarray:
    """Return array with small uniform jitter for scatter visibility."""
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-width, width, size=len(values))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure(df: pd.DataFrame, out_path: str) -> None:
    df_binned = bin_energies(df)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 9),
        gridspec_kw={"hspace": 0.45},
    )

    ax_top, ax_bot = axes

    # -----------------------------------------------------------------------
    # TOP: Scatter – Energy on x (large horizontal spread = what varies shot-to-shot),
    #       waveplate on y (discrete nominal angles, not a continuous “sweep” along x).
    # -----------------------------------------------------------------------
    waveplate_angles = sorted(df["Waveplate_Angle (Degrees)"].dropna().unique())
    if not len(waveplate_angles):
        sys.exit("ERROR: No waveplate angles in data.")

    step = min(np.diff(sorted(waveplate_angles))) if len(waveplate_angles) > 1 else 2.5
    y_pad = 0.35 * float(step)
    energy_jitter_mj = 9.0
    angle_jitter_deg = 0.18

    for i_mat, mat in enumerate(MATERIALS):
        sub = df[df["Material"] == mat]
        if sub.empty:
            continue
        x_e = sub["Laser_Target_Energy (mJ)"].astype(float).values
        y_w = sub["Waveplate_Angle (Degrees)"].astype(float).values
        x_j = add_jitter(x_e, width=energy_jitter_mj, seed=5000 + i_mat)
        y_j = add_jitter(y_w, width=angle_jitter_deg, seed=6000 + i_mat)
        ax_top.scatter(
            x_j,
            y_j,
            color=MATERIAL_COLORS[mat],
            marker=MATERIAL_MARKERS[mat],
            s=26,
            alpha=0.72,
            linewidths=0.45,
            edgecolors="k",
            zorder=3,
        )

    ax_top.set_xlabel("Laser target energy (mJ)", fontsize=12, labelpad=6)
    ax_top.set_ylabel("Waveplate angle (°)\n(discrete settings)", fontsize=12, labelpad=6)
    ax_top.set_title(
        "Laser energy varies along shots; angle is a fixed condition per group",
        fontsize=12,
        fontweight="bold",
    )
    ax_top.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax_top.tick_params(axis="both", labelsize=10)
    ax_top.xaxis.set_major_locator(ticker.MultipleLocator(200))

    ax_top.set_yticks(waveplate_angles)
    ax_top.set_yticklabels([f"{a:g}" for a in waveplate_angles], fontsize=11)
    ax_top.set_ylim(waveplate_angles[0] - y_pad, waveplate_angles[-1] + y_pad)

    legend_handles = [
        mpatches.Patch(facecolor=MATERIAL_COLORS[m], edgecolor="k", linewidth=0.8, label=m)
        for m in MATERIALS
    ]
    ax_top.legend(
        handles=legend_handles,
        title="Material",
        title_fontsize=10,
        fontsize=10,
        framealpha=0.85,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )

    # -----------------------------------------------------------------------
    # BOTTOM: Violin – Binned Laser Energy (x) vs Peak Free-Surface Velocity (y)
    # -----------------------------------------------------------------------
    # Build per-material, per-bin data matrices for matplotlib violin
    x_positions = np.arange(len(ENERGY_BIN_LABELS))
    n_mat = len(MATERIALS)
    group_width = 0.72          # total width used by all materials at one x position
    mat_width = group_width / n_mat

    violin_plotted = False

    for i_mat, mat in enumerate(MATERIALS):
        sub = df_binned[
            (df_binned["Material"] == mat) &
            df_binned["First_Maxima_m_s"].notna() &
            df_binned["Energy_Bin"].notna()
        ]

        mat_offset = (i_mat - (n_mat - 1) / 2.0) * mat_width
        color = MATERIAL_COLORS[mat]

        for i_bin, bin_label in enumerate(ENERGY_BIN_LABELS):
            bin_data = sub[sub["Energy_Bin"] == bin_label]["First_Maxima_m_s"].values

            xpos = x_positions[i_bin] + mat_offset

            if len(bin_data) < 4:
                # Too few points for a violin – draw a boxplot stub instead
                if len(bin_data) > 0:
                    ax_bot.scatter(
                        [xpos] * len(bin_data),
                        bin_data,
                        color=color,
                        marker=MATERIAL_MARKERS[mat],
                        s=30,
                        alpha=0.8,
                        zorder=4,
                        edgecolors="k",
                        linewidths=0.4,
                    )
                continue

            parts = ax_bot.violinplot(
                bin_data,
                positions=[xpos],
                widths=mat_width * 0.88,
                showmedians=True,
                showextrema=True,
            )
            violin_plotted = True

            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("k")
                pc.set_linewidth(0.6)
                pc.set_alpha(0.72)

            for partname in ("cmedians", "cmins", "cmaxes", "cbars"):
                if partname in parts:
                    parts[partname].set_edgecolor("k")
                    parts[partname].set_linewidth(1.0)
                    if partname == "cmedians":
                        parts[partname].set_linewidth(1.8)

    ax_bot.set_xticks(x_positions)
    ax_bot.set_xticklabels([f"{lbl} mJ" for lbl in ENERGY_BIN_LABELS], fontsize=10)
    ax_bot.set_xlabel("Laser Target Energy (mJ)", fontsize=12, labelpad=6)
    ax_bot.set_ylabel("Peak Free-Surface Velocity (m/s)", fontsize=12, labelpad=6)
    ax_bot.set_title("Peak Velocity Distribution vs Laser Energy by Material", fontsize=13, fontweight="bold")
    ax_bot.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax_bot.tick_params(axis="both", labelsize=10)

    # Legend (violin patches)
    violin_legend = [
        mpatches.Patch(
            facecolor=MATERIAL_COLORS[m],
            edgecolor="k",
            linewidth=0.8,
            alpha=0.72,
            label=m,
        )
        for m in MATERIALS
    ]
    ax_bot.legend(
        handles=violin_legend,
        title="Material",
        title_fontsize=10,
        fontsize=10,
        framealpha=0.85,
        loc="upper left",
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to enhanced_spall_summary.csv")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output figure path (.png/.pdf/.svg)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: CSV not found: {args.csv}")

    print(f"Loading data from: {args.csv}")
    df = load_and_filter(args.csv)
    print(f"  Rows after filtering: {len(df)}  |  Materials: {df['Material'].value_counts().to_dict()}")

    plot_figure(df, args.out)


if __name__ == "__main__":
    main()
