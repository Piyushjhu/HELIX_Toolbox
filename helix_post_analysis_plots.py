"""
helix_post_analysis_plots.py
============================
Standalone post-processing generator for HELIX Toolbox v2. It reads the single
consolidated master summary (<IGSN>-Data_Summary.csv) produced by a HELIX run and
generates a suite of data-analysis plots (PNG + PDF) into the same SPADE_analysis
directory. All per-shot quantities the plots need -- peak shock stress, rise times,
plastic/compressive strain rates, HEL, spall -- are computed upstream in the main
toolbox and read directly from the master file (no raw-trace reprocessing here).

The master file is discovered by pattern (*Data_Summary.csv) inside the SPADE_analysis
directory resolved from the config, so the IGSN-prefixed filename need not be known.

Run:
    python helix_post_analysis_plots.py                              # uses helix_master_config.yml
    python helix_post_analysis_plots.py --config helix_master_config.yml
    python helix_post_analysis_plots.py --spade-dir /path/to/Output/SPADE_analysis
    python helix_post_analysis_plots.py --csv /path/to/JHXXXX-Data_Summary.csv

Plots produced
--------------
Original (PA_01 … PA_13):
1.  Shock Stress vs Compressive Strain Rate      (scatter, per material)
2.  Shock Stress vs Rise Time to Peak Shock      (scatter, per material)
2b. Peak Free-Surface Velocity vs Rise Time to Peak Shock      (scatter, per material)
2c. Grady law log–log fit (log10 ε̇ vs log10 shock stress)   (all materials, linear regression)
2d. Grady law log–log (axes swapped: strain rate on x, shock stress on y)
2e. Grady log–log (like 2d): plastic strain rate (80%-20% backward walk) vs shock stress
2f. Grady log–log plastic (same as 2e; axes flipped: shock stress on x, plastic strain rate on y)
2g. Rise time t_r vs shock stress on log–log axes (per-material power law t_r = A·σ^n)
2h. Shock stress vs rise time t_r on log–log axes (2g axes flipped; refit σ = A·t_r^n)
3.  HEL Strength distribution per material            (violin + box hybrid)
4.  Spall Strength distribution per material          (violin + swarm)
5.  Ridgeline — HEL GPa per material                  (stacked KDE)
6.  Detection rate lollipop                           (HEL & Spall counts per material)
7.  Pairwise scatter matrix (correlogram)             (key physics variables, incl. HEL)
7b. Pairwise scatter matrix (correlogram)             (same as 7, Laser Energy swapped for Peak Shock Stress)
7c. HEL distribution + HEL vs 1/Elastic Rise Time     (first-row pair from 7b only; in-panel legends)
8.  HEL GPa vs Shock Stress                      (scatter, per material)
9.  HEL vs 1/Elastic Rise Time                        (scatter, per material)
10. HEL vs 1/Elastic Rise Time                        (scatter + per-material regressions)
11. HEL vs 1/Elastic Rise Time                        (per-material mean ± 1σ vs strain rate)
12. HEL vs 1/Elastic Rise Time                        (rolling mean ± 1σ, stride=10)
13. Shock Stress vs Flyer Row × Column           (per material, stratified by laser energy ±50 mJ)
14. PDV Return Power vs HEL Uncertainty               (scatter, per material)
15. HEL vs 1/Elastic Rise Time                        (per-material faceted, log x-axis, regression + ±1σ band)

Tensile-only, no HEL (PA_tensile_*):
9.  Shock Stress vs Tensile Strain Rate          (scatter; tensile = line 3 pullback slope)
10. Shock Stress vs Tensile Strain Rate          (bubble, Laser Energy)
11. Correlogram — tensile/spall only                  (Peak Stress, Tensile Strain Rate, Spall, Pullback, Laser Energy)
12. Spall Strength vs Shock Stress               (scatter)
13. Spall Strength vs Tensile Strain Rate             (scatter; color = material, marker = Shock Stress bin in 0.5 GPa steps)
14. Spall Strength vs Shock Stress               (scatter; fill color = Tensile Strain Rate bin in 0.5e6 s⁻¹ steps; edge = material)

Spatial 3-D maps (binary / spatial analysis; generated in `Binary_metal_analysis_config_version.py`):
- Spall Strength spatial 3-D contour surfaces (X/Y = flyer position, Z = laser energy or waveplate angle).
- HEL spatial 3-D contour surfaces (X/Y = flyer position, Z = laser energy or waveplate angle).
- Spatial 2-D contour maps from Z-collapsed (mean) surfaces (X/Y = flyer position, colour = mean property).
"""

import argparse
import glob
import io
import json
import os
import sys
import warnings
from functools import partial

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, linregress

# ── Spatial helper utilities (inlined from HELIX v1 Binary_metal_analysis so this
#    standalone generator has no dependency on the v1 module) ──────────────────
def _find_waveplate_col(df):
    """Return the first column in *df* that looks like a waveplate-angle column, or None."""
    candidates = [
        'Waveplate_Angle (Degrees)', 'Waveplate_Angle', 'Waveplate Angle',
        'WP_Angle', 'WP_angle', 'waveplate_angle', 'Wave_Plate_Angle',
        'Wave Plate Angle', 'WaveplateAngle',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _find_laser_energy_col(df):
    """Return the laser-energy column name if present in *df*, or None.

    Searches for the primary name 'Laser_Target_Energy (mJ)' first, then
    a list of common aliases, then falls back to a case-insensitive scan.
    """
    candidates = [
        'Laser_Target_Energy (mJ)',
        'Laser_Target_Energy(mJ)',
        'Laser Target Energy (mJ)',
        'Laser_Energy (mJ)',
        'Laser_Energy(mJ)',
        'Laser Energy (mJ)',
        'LaserEnergy_mJ',
        'Laser_Energy_mJ',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _build_xy_positions(output_df, flyer_row, flyer_column, comp_pct,
                        START_X=8.0, START_Y=8.0, SPACING=6.0,
                        Y_MIN=8.0, Y_MAX=32.0):
    """Compute (x_pos, y_pos) Series from flyer grid columns.

    Physical coordinate convention (matches the target diagram):
    ┌─────────────────────────────────────────────────────────────────┐
    │  X axis  = Column direction                                     │
    │            C1 → X=32 mm,  C2 → 26,  C3 → 20,  C4 → 14,       │
    │            C5 → X=8 mm   (column number DECREASES as X grows)  │
    │  Y axis  = Row direction                                        │
    │            R1 → Y=8 mm,  R2 → 14,  R3 → 20,  R4 → 26,        │
    │            R5 → Y=32 mm  (row number INCREASES with Y)         │
    └─────────────────────────────────────────────────────────────────┘
    Falls back to sequential positions when grid data is absent.
    """
    x_pos = pd.Series(dtype=float, index=output_df.index)
    y_pos = pd.Series(dtype=float, index=output_df.index)

    # ── X position: from Flyer_Column ────────────────────────────────────────
    # Desired convention:
    # - Corner is (0, 0)
    # - C5R1 is at (START_X, START_Y) = (8, 8) mm
    # - Each successive row/column step is exactly SPACING = 6 mm
    # - Column number decreases as X increases (C5=8, C4=14, ..., C1=32)
    col_numeric = pd.to_numeric(flyer_column, errors='coerce')
    if col_numeric.isna().all():
        try:
            col_numeric = flyer_column.astype(str).str.extract(r'(\d+)', expand=False).astype(float)
        except Exception:
            col_numeric = pd.Series(dtype=float, index=output_df.index)

    if col_numeric.notna().any():
        col_min, col_max = col_numeric.min(), col_numeric.max()
        if col_max > col_min:
            # Higher column number → smaller X (C1=largest X, C5=smallest X)
            x_pos = START_X + (col_max - col_numeric) * SPACING
        else:
            x_pos = pd.Series(START_X, index=output_df.index)
    else:
        # Fallback: sequential X from shot_index or row counter
        row_num = pd.to_numeric(flyer_row.astype(str).str.extract(r'(\d+)', expand=False),
                                errors='coerce')
        if row_num.notna().any():
            x_pos = START_X + (row_num - row_num.min()) * SPACING
        elif 'shot_index' in output_df.columns:
            si = pd.to_numeric(output_df['shot_index'], errors='coerce')
            x_pos = START_X + (si - si.min()) * SPACING if si.notna().any() else \
                    pd.Series([START_X + i * SPACING for i in range(len(output_df))],
                               index=output_df.index)
        else:
            x_pos = pd.Series([START_X + i * SPACING for i in range(len(output_df))],
                               index=output_df.index)
    x_pos.index = output_df.index

    # ── Y position: from Flyer_Row ───────────────────────────────────────────
    # Desired convention: R1=START_Y=8, R2=14, ..., increasing by SPACING=6 mm
    row_numeric = pd.to_numeric(flyer_row, errors='coerce')
    if row_numeric.isna().all():
        try:
            row_numeric = flyer_row.astype(str).str.extract(r'(\d+)', expand=False).astype(float)
        except Exception:
            row_numeric = pd.Series(dtype=float, index=output_df.index)

    if row_numeric.notna().any():
        row_min = row_numeric.min()
        y_pos = START_Y + (row_numeric - row_min) * SPACING
    elif comp_pct.notna().any():
        # Binary mode fallback: map composition to Y
        unique_comps = sorted(comp_pct[comp_pct.notna()].unique())
        if len(unique_comps) > 1:
            spacing_y = (Y_MAX - START_Y) / (len(unique_comps) - 1)
            comp_to_y = {c: START_Y + i * spacing_y for i, c in enumerate(unique_comps)}
            y_pos = comp_pct.map(lambda c: comp_to_y.get(c, np.nan) if pd.notna(c) else np.nan)
        else:
            y_pos = pd.Series(START_Y, index=output_df.index)
    else:
        y_pos = pd.Series(START_Y, index=output_df.index)
    y_pos.index = output_df.index

    return x_pos, y_pos


def _plot_3d_contour_surface(output_df, x_pos, y_pos, values, unique_angles,
                              wp_angles, value_label, cmap_name, title_str,
                              out_path_base, z_label='Laser Energy (mJ)'):
    """Render layered 3D contour surfaces — one per Z-axis level — and save PNG + PDF.

    Parameters
    ----------
    z_label : str
        Label for the Z axis (default: 'Laser Energy (mJ)').
        Pass 'Waveplate Angle (°)' when using waveplate angles instead.
    """
    from scipy.interpolate import griddata
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

    valid_vals = np.array([float(v) if pd.notna(v) else np.nan for v in values])
    global_vmin = np.nanmin(valid_vals)
    global_vmax = np.nanmax(valid_vals)

    # Wider figure gives room for the Z-axis label on the left side
    # (azim=225 places Z on the far-left edge of the canvas).
    fig = plt.figure(figsize=(16, 13))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap(cmap_name)
    # Shared normalisation so plane colours and scatter points match the legend
    norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
    sorted_angles = sorted(unique_angles)

    for plane_idx, angle in enumerate(sorted_angles):
        amask = (wp_angles == angle).values
        x_sub = x_pos[amask].values
        y_sub = y_pos[amask].values
        z_sub = values[amask].values.astype(float)

        # Drop NaN values before griddata
        finite = np.isfinite(z_sub)
        x_sub, y_sub, z_sub = x_sub[finite], y_sub[finite], z_sub[finite]
        if len(x_sub) < 3:
            # Too few points for interpolation — plot scatter dots instead
            norm_vals = (z_sub - global_vmin) / max(global_vmax - global_vmin, 1e-12)
            colors = cmap(norm_vals)
            ax.scatter(x_sub, y_sub,
                       [angle] * len(x_sub),
                       c=colors, s=80, zorder=5,
                       depthshade=False)
            continue

        x_range = np.linspace(x_pos.min(), x_pos.max(), 60)
        y_range = np.linspace(y_pos.min(), y_pos.max(), 60)
        xi_mesh, yi_mesh = np.meshgrid(x_range, y_range)

        method = 'cubic' if len(x_sub) >= 4 else 'linear'
        try:
            zi = griddata((x_sub, y_sub), z_sub, (xi_mesh, yi_mesh), method=method)
        except Exception:
            try:
                zi = griddata((x_sub, y_sub), z_sub, (xi_mesh, yi_mesh), method='linear')
            except Exception:
                continue

        zi = np.ma.masked_invalid(zi)

        # For HEL maps, colour each laser-energy plane by the *mean* HEL value
        # on that plane so the solid plane colour matches the HEL legend.
        if 'HEL' in str(value_label):
            plane_mean = float(np.nanmean(z_sub)) if z_sub.size else np.nan
            if np.isfinite(plane_mean):
                plane_color = cmap(norm(plane_mean))
                z_plane = np.full_like(xi_mesh, float(angle), dtype=float)
                ax.plot_surface(
                    xi_mesh, yi_mesh, z_plane,
                    color=plane_color,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=False,
                    shade=False, alpha=0.55,
                )
        else:
            # Original behaviour for non-HEL fields: full contour colormap per plane
            ax.contourf(
                xi_mesh, yi_mesh, zi,
                zdir='z', offset=float(angle),
                levels=15, cmap=cmap_name,
                norm=norm,
                alpha=0.45,
            )

    # Scatter actual data points on their surfaces
    finite_global = np.isfinite(valid_vals)
    wp_clean = wp_angles.values[finite_global]
    ax.scatter(x_pos.values[finite_global],
               y_pos.values[finite_global],
               wp_clean,
               c=valid_vals[finite_global],
               cmap=cmap_name,
               vmin=global_vmin, vmax=global_vmax,
               s=60, edgecolors='black', linewidth=0.8,
               zorder=10, depthshade=False)

    # Increase inter-plane visual spacing via z-axis padding and box aspect
    z_range = max(unique_angles) - min(unique_angles)
    z_pad = max(z_range * 0.18, 2.0)
    ax.set_zlim(min(unique_angles) - z_pad, max(unique_angles) + z_pad)

    ax.set_xlabel('X Position (mm)', fontsize=32, labelpad=16)
    ax.set_ylabel('Y Position (mm)', fontsize=32, labelpad=16)
    # Z-axis label: keep labelpad moderate; the extra left margin and pad_inches
    # on savefig are what actually prevent clipping for large tick values like "2000".
    ax.set_zlabel(z_label, fontsize=32, labelpad=22)
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=14)

    # Isometric-ish view — taller z box spreads planes further apart visually
    ax.view_init(elev=28, azim=225)
    ax.set_box_aspect([1, 1, 1.1])

    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    ax.zaxis.set_tick_params(labelsize=24)

    # Single colourbar for all surfaces
    sm = plt.cm.ScalarMappable(
        cmap=cmap_name,
        norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.04)
    cbar.set_label(value_label, fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)

    # tight_layout does not work well with 3D axes; give the 3D box generous
    # margins — especially on the left where the Z label lives (azim=225).
    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.06, top=0.92)
    png_path = out_path_base + '.png'
    pdf_path = out_path_base + '.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.6)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.6)
    plt.close(fig)
    return png_path if os.path.exists(png_path) else None


warnings.filterwarnings('ignore')

# ── Aesthetic constants ────────────────────────────────────────────────────────
PALETTE    = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']   # material colours
MARKERS    = ['o', 's', '^', 'D', 'v', 'P']

# Fixed colour/marker per material so identity is stable across plots and datasets
# regardless of which other materials happen to be combined into the same df
# (positional PALETTE/MARKERS assignment shifts alphabetically, e.g. adding 'Al'
# used to bump Cu/Zn onto the wrong marker).
FIXED_MATERIAL_STYLE = {
    'Brass':    ('#2196F3', 'o'),   # blue circle
    'Cu':       ('#FF5722', 's'),   # dark/deep orange square
    # Cu shot at two different sample thicknesses across input files gets split
    # into thickness-suffixed labels (see split_material_by_thickness()) so the
    # two batches aren't silently pooled together; keep them in the same
    # orange family but visually distinguishable.
    'Cu 1mm':   ('#FF5722', 's'),   # same as base Cu (1 mm is the default batch)
    'Cu 0.5mm': ('#FFAB40', 'P'),   # lighter amber, plus marker
    'Zn':       ('#4CAF50', '^'),   # green triangle
    'Al':       ('#9E9E9E', 'D'),   # grey diamond
}

# Cu markers get extra transparency in scatter plots where its cluster commonly
# overlaps Brass's (both binary-alloy compositions sit in a similar stress range).
MATERIAL_ALPHA_OVERRIDE = {'Cu': 0.55, 'Cu 1mm': 0.55, 'Cu 0.5mm': 0.55}


def mat_alpha(mat, default):
    return MATERIAL_ALPHA_OVERRIDE.get(mat, default)
FIG_DPI    = 300
FONT_TITLE = 14
FONT_AXIS  = 20
FONT_TICK  = 18
FONT_LEG   = 13

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top':    True,
    'axes.spines.right':  True,
    'axes.edgecolor':     'black',
    'axes.linewidth':     1.2,
    'xtick.major.width':  1.2,
    'ytick.major.width':  1.2,
    'xtick.major.size':   5,
    'ytick.major.size':   5,
})

# ── Config loading ─────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """Load the HELIX master config. Supports YAML (helix_master_config.yml) and JSON."""
    with open(config_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if config_path.lower().endswith(('.yml', '.yaml')):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def find_summary_csv(spade_dir: str) -> str:
    """Return the consolidated master summary inside a SPADE_analysis directory.

    HELIX Toolbox v2 writes a single consolidated master per run named
    <IGSN>-Data_Summary.csv (the IGSN prefix is derived from the run folder, so the
    exact name varies) -- discover it by pattern rather than hardcoding a name. Falls
    back to the legacy spall_summary.csv / velocity_shots_summary.csv only if no master
    is present.
    """
    masters = sorted(glob.glob(os.path.join(spade_dir, '*Data_Summary.csv')))
    if masters:
        return masters[0]
    for name in ('spall_summary.csv', 'velocity_shots_summary.csv'):
        cand = os.path.join(spade_dir, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"No *Data_Summary.csv (or legacy spall_summary.csv/velocity_shots_summary.csv) "
        f"found in {spade_dir}")


# ── Data helpers ───────────────────────────────────────────────────────────────
def to_num(series):
    return pd.to_numeric(series, errors='coerce')


def split_material_by_thickness(df):
    """Split any material into thickness-suffixed labels (e.g. 'Cu 1mm', 'Cu
    0.5mm') whenever more than one distinct 'Sample Thickness (mm)' value is
    present for that material. Without this, batches of the same composition
    shot at different thicknesses across different input files (e.g. the HEL
    run's 1 mm Cu vs the Spall run's 0.5 mm Cu) get silently pooled together
    under one 'Cu' label. Materials with a single thickness (or no thickness
    column at all) keep their plain name, so single-CSV runs are unaffected.
    """
    thick_col = next((c for c in df.columns
                      if c.strip().lower() == 'sample thickness (mm)'), None)
    if not thick_col:
        return df
    thick = pd.to_numeric(df[thick_col], errors='coerce')
    for mat in df['_material'].dropna().unique():
        mask = df['_material'] == mat
        distinct = thick[mask].dropna().unique()
        if len(distinct) > 1:
            suffix = thick[mask].apply(lambda t: f' {t:g}mm' if pd.notna(t) else '')
            df.loc[mask, '_material'] = df.loc[mask, '_material'].astype(str) + suffix
    return df


def material_groups(df):
    """Return sorted list of unique materials and a colour/marker mapping.

    Brass/Cu/Zn always get their fixed style from FIXED_MATERIAL_STYLE; any other
    material falls back to the next PALETTE/MARKERS entry not already claimed by
    the fixed set, so it can't collide with Brass/Cu/Zn's colours or markers.
    """
    mats = sorted(df['_material'].dropna().unique())
    used_colours = {c for c, _m in FIXED_MATERIAL_STYLE.values()}
    used_markers = {m for _c, m in FIXED_MATERIAL_STYLE.values()}
    fallback_colours = [c for c in PALETTE if c not in used_colours]
    fallback_markers = [m for m in MARKERS if m not in used_markers]

    colours, markers = {}, {}
    fi = 0
    for m in mats:
        if m in FIXED_MATERIAL_STYLE:
            colours[m], markers[m] = FIXED_MATERIAL_STYLE[m]
        else:
            colours[m] = fallback_colours[fi % len(fallback_colours)]
            markers[m] = fallback_markers[fi % len(fallback_markers)]
            fi += 1
    return mats, colours, markers


def _scaled_mad_1d(a):
    """MAD × 1.4826 on `a`: comparable to Gaussian σ for outlier gates."""
    a = np.asarray(a, dtype=float)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad <= 0 or not np.isfinite(mad):
        mad = 1e-30
    return 1.4826 * mad


def _grady_loglog_outlier_keep(x, y, k_sigma, use_mad):
    """
    Rectangular gate in log–log space: keep points within k_sigma times spread on each axis.
    If use_mad: center = median, spread = scaled MAD per axis (robust; used for Zn).
    Else: center = mean, spread = sample std (other materials).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if use_mad:
        cx, cy = np.median(x), np.median(y)
        sx, sy = _scaled_mad_1d(x), _scaled_mad_1d(y)
    else:
        cx, cy = np.mean(x), np.mean(y)
        sx, sy = np.std(x), np.std(y)
        if sx <= 0:
            sx = 1e-30
        if sy <= 0:
            sy = 1e-30
    return (np.abs(x - cx) <= k_sigma * sx) & (np.abs(y - cy) <= k_sigma * sy)


def save(fig, out_dir, name):
    png = os.path.join(out_dir, name + '.png')
    pdf = os.path.join(out_dir, name + '.pdf')
    # Delete existing files before writing so PIL always creates a new file.
    # OneDrive holds a sync lock on existing files; open("w+b") on a locked file
    # blocks indefinitely. Unlinking first forces a clean create with no lock.
    for _p in (png, pdf):
        try:
            os.remove(_p)
        except FileNotFoundError:
            pass
    # pad_inches avoids tight bbox clipping axis spines on log / multi-panel figures
    fig.savefig(png, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(pdf,              bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"  ✓  {png}")


def apply_full_axis_frame(ax, linewidth=1.5):
    """Draw all four spines in black (log axes and shared axes sometimes omit them)."""
    ax.set_axis_on()
    ax.set_axisbelow(True)
    for side in ('bottom', 'top', 'left', 'right'):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_linewidth(linewidth)
        sp.set_edgecolor('black')
    ax.tick_params(which='major', direction='out', top=True, right=True,
                   labeltop=False, labelright=False, length=5, width=1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 1 — Shock Stress vs Compressive Strain Rate (scatter, from meta_data.csv)
# ══════════════════════════════════════════════════════════════════════════════
def plot_stress_vs_strainrate_scatter(df, out_dir):
    """Scatter: Shock Stress (GPa) vs Compressive Strain Rate (s⁻¹), per material.
    Uses compressive strain rate directly from meta_data.csv. Cu is semi-transparent
    so overlapping Brass / Zn points are visible."""
    stress_col = 'Peak_Shock_Stress_GPa'
    if stress_col not in df.columns:
        print("  [skip] PA_01: missing Peak_Shock_Stress_GPa column.")
        return

    # Prefer Hugoniot-based strain rate if present, else fall back to average column
    if 'Compressive_StrainRate_Ufs_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Ufs_s^-1'
    elif 'Compressive_StrainRate_Avg_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Avg_s^-1'
    else:
        print("  [skip] PA_01: no compressive strain-rate column found in meta_data.csv.")
        return

    df2 = df.copy()
    df2[stress_col] = pd.to_numeric(df2[stress_col], errors='coerce')
    df2[sr_col] = pd.to_numeric(df2[sr_col], errors='coerce')

    valid = df2.dropna(subset=[stress_col, sr_col])
    valid = valid[valid[sr_col] > 0]
    valid = valid[valid[stress_col] > 0]

    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] PA_01: no valid rows with positive stress and strain rate.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # draw Cu last (on top) but semi-transparent
    cu_mats  = [m for m in mats if m.lower().startswith('cu')]
    non_cu   = [m for m in mats if not m.lower().startswith('cu')]
    for mat in (non_cu + cu_mats):
        sub   = valid[valid['_material'] == mat]
        alpha = 0.35 if mat.lower().startswith('cu') else 0.85
        ax.scatter(sub[sr_col],
                   sub[stress_col],
                   c=colours[mat], marker=mkrs[mat],
                   s=60, edgecolors='black', linewidths=0.5,
                   alpha=alpha, label=f'{mat}  (n={len(sub)})', zorder=3)

    ax.set_xlabel('Compressive Strain Rate  (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('Shock Stress  (GPa)',             fontsize=FONT_AXIS)
    ax.set_title('Shock Stress vs Compressive Strain Rate', fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlim(0, 1e6)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, 'PA_01_stress_vs_strainrate_scatter')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2 — Shock Stress vs Rise Time to Peak Shock (scatter, per material)
# ══════════════════════════════════════════════════════════════════════════════
def _mad_filter_per_material(xv, yv, mv, mats, k_sigma=1.5):
    """Keep only points within k_sigma x scaled-MAD of the per-material median on both axes.
    Materials with fewer than 2 points are kept as-is (nothing to filter against)."""
    keep_mask = np.zeros(len(xv), dtype=bool)
    for mat in mats:
        idx = mv == mat
        if idx.sum() < 2:
            keep_mask |= idx
            continue
        keep_local = _grady_loglog_outlier_keep(xv[idx], yv[idx], k_sigma, use_mad=True)
        keep_mask[np.where(idx)[0][keep_local]] = True
    return xv[keep_mask], yv[keep_mask], mv[keep_mask]


def _add_material_regression_lines(ax, xv, yv, mv, mats, colours):
    """Overlay a per-material linear least-squares fit (dashed line) on a scatter plot."""
    for mat in mats:
        idx = mv == mat
        x = xv[idx]
        y = yv[idx]
        if len(x) < 2 or np.allclose(x, x[0]):
            continue
        try:
            m, b = np.polyfit(x, y, 1)
        except Exception:
            continue
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, color=colours[mat], linewidth=2.0,
                linestyle='--', alpha=0.9, zorder=4)


def plot_stress_vs_strainrate_bubble(df, out_dir):
    """Scatter: Shock Stress (GPa) vs arrival-to-peak rise time (ns), per material.

    Uses RiseTime_ArrivalToPeak_ns (AIC-detected first motion to peak) — the full rise span,
    including any elastic precursor."""
    df2 = df.copy()
    for col in ['Peak_Shock_Stress_GPa', 'RiseTime_ArrivalToPeak_ns']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    valid = df2.dropna(subset=['Peak_Shock_Stress_GPa',
                                'RiseTime_ArrivalToPeak_ns'])
    valid = valid[valid['Peak_Shock_Stress_GPa'] > 0]

    mats, colours, mkrs = material_groups(valid)
    valid = valid.reset_index(drop=True)

    xv = valid['RiseTime_ArrivalToPeak_ns'].values.astype(float)
    yv = valid['Peak_Shock_Stress_GPa'].values.astype(float)
    mv = valid['_material'].values

    # MAD outlier filter: keep points within 1.5 x scaled-MAD of the per-material
    # median on both axes (robust to skew, unlike a mean +/- std gate).
    xv, yv, mv = _mad_filter_per_material(xv, yv, mv, mats, k_sigma=1.5)

    fig, ax = plt.subplots(figsize=(10, 7))
    for mat in mats:
        idx = mv == mat
        ax.scatter(xv[idx], yv[idx],
                   s=60, c=colours[mat], marker=mkrs[mat],
                   edgecolors='black', linewidths=0.4,
                   alpha=mat_alpha(mat, 0.8), label=mat, zorder=3)
    _add_material_regression_lines(ax, xv, yv, mv, mats, colours)

    ax.set_xlabel('Rise Time to Peak Shock  (ns)', fontsize=FONT_AXIS)
    ax.set_ylabel('Shock Stress  (GPa)',       fontsize=FONT_AXIS)
    ax.set_title('Shock Stress vs Rise Time to Peak Shock\n(MAD filter: median ±1.5×scaled MAD per material)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlim(0, xv.max() + 10)
    ax.set_ylim(0, yv.max() * 1.1)
    ax.tick_params(labelsize=FONT_TICK)
    mat_handles = [
        Line2D([0], [0], marker=mkrs[m], color='w',
               markerfacecolor=colours[m], markeredgecolor='black',
               markersize=8, label=f'{m} (n={int((mv == m).sum())})')
        for m in mats
    ]
    ax.legend(handles=mat_handles, fontsize=FONT_LEG, framealpha=0.9,
              title='Material', loc='upper right')
    fig.tight_layout()
    save(fig, out_dir, 'PA_02_stress_vs_risetime_scatter')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2b — Peak Free-Surface Velocity vs Rise Time to Peak Shock
# ══════════════════════════════════════════════════════════════════════════════
def plot_peak_fsv_vs_risetime_scatter(df, out_dir):
    """Scatter: Peak free-surface velocity (m/s) vs arrival-to-peak rise time (ns), per material.

    Uses RiseTime_ArrivalToPeak_ns (AIC-detected first motion to peak) — the full rise span,
    including any elastic precursor."""
    required_cols = ['RiseTime_ArrivalToPeak_ns', 'First_Maxima_m_s']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("  [skip] PA_02b: Missing required columns:", missing)
        return

    df2 = df.copy()
    for col in required_cols:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')

    valid = df2.dropna(subset=required_cols)
    valid = valid[valid['First_Maxima_m_s'] > 0]
    if valid.empty:
        print("  [skip] PA_02b: No valid rows after filtering.")
        return

    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] PA_02b: No materials found for plot.")
        return

    valid = valid.reset_index(drop=True)
    xv = valid['RiseTime_ArrivalToPeak_ns'].values.astype(float)
    yv = valid['First_Maxima_m_s'].values.astype(float)
    mv = valid['_material'].values

    fig, ax = plt.subplots(figsize=(10, 7))
    for mat in mats:
        idx = mv == mat
        ax.scatter(xv[idx], yv[idx],
                   s=60, c=colours[mat], marker=mkrs[mat],
                   edgecolors='black', linewidths=0.4,
                   alpha=0.8, label=mat, zorder=3)
    _add_material_regression_lines(ax, xv, yv, mv, mats, colours)

    ax.set_xlabel('Rise Time to Peak Shock  (ns)', fontsize=FONT_AXIS)
    ax.set_ylabel('Peak Free-Surface Velocity  (m/s)', fontsize=FONT_AXIS)
    ax.set_title('Peak Free-Surface Velocity vs Rise Time to Peak Shock',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlim(0, xv.max() + 10)
    ax.set_ylim(0, yv.max() * 1.1)
    ax.tick_params(labelsize=FONT_TICK)

    mat_handles = [
        Line2D([0], [0], marker=mkrs[m], color='w',
               markerfacecolor=colours[m], markeredgecolor='black',
               markersize=8, label=f'{m} (n={int((mv == m).sum())})')
        for m in mats
    ]
    # Place legend outside the right edge so it does not overlap the Y label
    ax.legend(handles=mat_handles,
              fontsize=FONT_LEG,
              framealpha=0.9,
              title='Material',
              loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0)
    fig.tight_layout()
    save(fig, out_dir, 'PA_02b_peak_fsv_vs_risetime_scatter')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2c — Grady law log–log fit: log10(ε̇) vs log10(shock stress)
# ══════════════════════════════════════════════════════════════════════════════
def plot_grady_loglog_fit(df, out_dir):
    """
    Log–log Grady-law fits per material.

    Uses compressive strain rate (prefer Hugoniot-based `Compressive_StrainRate_Ufs_s^-1`
    if available, else fall back to `Compressive_StrainRate_Avg_s^-1`) and peak shock
    stress (GPa). For each material m, fits:

        log10(ε̇_m) = n_m * log10(σ_peak_m) + log10(A_m)

    so that ε̇_m = A_m * σ_peak_m^n_m, where σ_peak is in Pa (converted from GPa in meta_data).
    """
    stress_col = 'Peak_Shock_Stress_GPa'
    if stress_col not in df.columns:
        print("  [skip] Grady fit: missing Peak_Shock_Stress_GPa.")
        return

    if 'Compressive_StrainRate_Ufs_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Ufs_s^-1'
    elif 'Compressive_StrainRate_Avg_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Avg_s^-1'
    else:
        print("  [skip] Grady fit: no compressive strain-rate column found.")
        return

    df2 = df.copy()
    df2[stress_col] = pd.to_numeric(df2[stress_col], errors='coerce')
    df2[sr_col] = pd.to_numeric(df2[sr_col], errors='coerce')
    valid = df2.dropna(subset=[stress_col, sr_col])
    valid = valid[(valid[stress_col] > 0) & (valid[sr_col] > 0)]

    if valid.empty:
        print("  [skip] Grady fit: no positive stress/strain-rate pairs.")
        return

    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter points per material on log–log axes + per-material fits
    # Convert stress from GPa to Pa for Grady law (ε̇ = A σ^n in SI units)
    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        s_pa = s_gpa * 1e9  # GPa → Pa
        e = sub[sr_col].values.astype(float)
        x = np.log10(s_pa)
        y = np.log10(e)
        if len(x) < 2:
            print(f"  [skip] Grady fit for {mat}: too few points.")
            continue

        # Per-material linear regression on log–log data (compute A for legend)
        A_m = None
        if not (np.allclose(x, x[0]) or np.allclose(y, y[0])):
            try:
                n_m, logA_m = np.polyfit(x, y, 1)
                A_m = 10.0 ** logA_m
                print(f"  Grady-law fit for {mat} (ε̇ = A σ^n): n = {n_m:.3f}, A = {A_m:.3e} (σ in Pa)")
            except Exception as exc:
                print(f"  [skip] Grady fit for {mat}: regression failed: {exc}")

        leg_label = f'{mat}  (N={len(x)}, A={A_m:.2e}, n={n_m:.2f})' if A_m is not None else f'{mat}  (n={len(x)})'
        ax.scatter(x, y,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=leg_label)

        if A_m is not None:
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = n_m * x_fit + logA_m
            ax.plot(x_fit, y_fit, color=colours[mat], linewidth=2.0,
                    linestyle='--')

    ax.set_xlabel('log$_{10}$[ Shock Stress  (Pa) ]', fontsize=FONT_AXIS)
    ax.set_ylabel('log$_{10}$[ Compressive Strain Rate  (s$^{-1}$) ]', fontsize=FONT_AXIS)
    ax.set_title('Grady Law Log–Log Fit per Material\nlog$_{10}$(ε̇) vs log$_{10}$(Shock Stress in Pa)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best')
    fig.tight_layout()
    save(fig, out_dir, 'PA_02c_grady_loglog_fit')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2d — Grady law log–log with axes swapped (ε̇ on x, shock stress on y)
# ══════════════════════════════════════════════════════════════════════════════
def plot_grady_loglog_fit_swapped_axes(df, out_dir):
    """
    Same data and per-material fits as plot_grady_loglog_fit, but horizontal axis is
    log10(compressive strain rate) and vertical axis is log10(shock stress in Pa).

    Fit line: log10(σ) = (1/n) log10(ε̇) − log10(A)/n, from ε̇ = A σ^n.
    """
    stress_col = 'Peak_Shock_Stress_GPa'
    if stress_col not in df.columns:
        print("  [skip] Grady 2d: missing Peak_Shock_Stress_GPa.")
        return

    if 'Compressive_StrainRate_Ufs_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Ufs_s^-1'
    elif 'Compressive_StrainRate_Avg_s^-1' in df.columns:
        sr_col = 'Compressive_StrainRate_Avg_s^-1'
    else:
        print("  [skip] Grady 2d: no compressive strain-rate column found.")
        return

    df2 = df.copy()
    df2[stress_col] = pd.to_numeric(df2[stress_col], errors='coerce')
    df2[sr_col] = pd.to_numeric(df2[sr_col], errors='coerce')
    valid = df2.dropna(subset=[stress_col, sr_col])
    valid = valid[(valid[stress_col] > 0) & (valid[sr_col] > 0)]

    if valid.empty:
        print("  [skip] Grady 2d: no positive stress/strain-rate pairs.")
        return

    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        s_pa = s_gpa * 1e9
        e = sub[sr_col].values.astype(float)
        x_stress = np.log10(s_pa)
        y_sr = np.log10(e)
        if len(x_stress) < 2:
            print(f"  [skip] Grady 2d for {mat}: too few points.")
            continue

        A_m = None
        n_m = None
        logA_m = None
        if not (np.allclose(x_stress, x_stress[0]) or np.allclose(y_sr, y_sr[0])):
            try:
                n_m, logA_m = np.polyfit(x_stress, y_sr, 1)
                A_m = 10.0 ** logA_m
                print(f"  Grady 2d fit for {mat} (ε̇ = A σ^n): n = {n_m:.3f}, A = {A_m:.3e} (σ in Pa)")
            except Exception as exc:
                print(f"  [skip] Grady 2d for {mat}: regression failed: {exc}")

        leg_label = (
            f'{mat}  (N={len(x_stress)}, A={A_m:.2e}, n={n_m:.2f})'
            if A_m is not None and n_m is not None
            else f'{mat}  (n={len(x_stress)})')
        # x = strain rate, y = stress
        ax.scatter(y_sr, x_stress,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=leg_label)

        if A_m is not None and n_m is not None and abs(n_m) > 1e-12:
            x_fit_sr = np.linspace(y_sr.min(), y_sr.max(), 100)
            y_fit_stress = (1.0 / n_m) * x_fit_sr - logA_m / n_m
            ax.plot(x_fit_sr, y_fit_stress, color=colours[mat], linewidth=2.0,
                    linestyle='--')

    ax.set_xlabel('log$_{10}$[ Compressive Strain Rate  (s$^{-1}$) ]', fontsize=FONT_AXIS)
    ax.set_ylabel('log$_{10}$[ Shock Stress  (Pa) ]', fontsize=FONT_AXIS)
    ax.set_title(
        'Grady Law Log–Log (axes swapped)\n'
        'Shock Stress vs Compressive Strain Rate',
        fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best')
    fig.tight_layout()
    save(fig, out_dir, 'PA_02d_grady_loglog_swapped_axes')


#  Plastic strain-rate methods available for PA_02e / PA_02f: (column, description,
#  filename tag). All three are computed upstream in compute_spall_strengths()
#  (Binary_metal_analysis_config_version.py) directly from the raw velocity trace.
#  Fallback defaults if no config is available (mirrors compute_spall_strengths's own
#  defaults); build_plastic_sr_methods() below derives the real values from the
#  analysis config so labels/filenames track whatever percentages were actually used.
PLASTIC_SR_METHODS = [
    ('PlasticStrainRate_80_20_s^-1', '80%-20% backward walk', '80_20'),
    ('PlasticStrainRate_90_10_s^-1', '90%-10% backward walk', '90_10'),
    ('PlasticStrainRate_MaxSlope_s^-1', 'steepest 20%-of-peak window (max slope)', 'maxslope'),
]


def _cfg_pct(analysis_params, key, default_pct):
    """Read a plastic_sr_*_pct config value and normalize it to percent scale (0-100),
    accepting either a percent (e.g. 80) or a fraction (e.g. 0.8) like _as_fraction()
    in compute_spall_strengths() does."""
    try:
        v = float(analysis_params.get(key, default_pct))
    except (TypeError, ValueError):
        return default_pct
    if not np.isfinite(v) or v <= 0:
        return default_pct
    return v if v > 1.0 else v * 100.0


def _fmt_pct(v):
    return f'{v:.0f}' if abs(v - round(v)) < 0.05 else f'{v:.1f}'


def build_plastic_sr_methods(cfg):
    """Build the PA_02e/02f (column, description, filename-tag) list from the actual
    analysis config, so a change to plastic_sr_high_pct/low_pct (or the 90/10 and
    max-slope-window equivalents) in analysis_config.json is reflected in the plot
    titles and filenames instead of staying hardcoded at 80/20."""
    ap = cfg.get('analysis_parameters', {})
    hi80 = _cfg_pct(ap, 'plastic_sr_high_pct', 80.0)
    lo80 = _cfg_pct(ap, 'plastic_sr_low_pct', 20.0)
    hi90 = _cfg_pct(ap, 'plastic_sr90_high_pct', 90.0)
    lo90 = _cfg_pct(ap, 'plastic_sr90_low_pct', 10.0)
    ms_window = _cfg_pct(ap, 'plastic_sr_maxslope_window_pct', 20.0)
    return [
        ('PlasticStrainRate_80_20_s^-1',
         f'{_fmt_pct(hi80)}%-{_fmt_pct(lo80)}% backward walk',
         f'{_fmt_pct(hi80)}_{_fmt_pct(lo80)}'),
        ('PlasticStrainRate_90_10_s^-1',
         f'{_fmt_pct(hi90)}%-{_fmt_pct(lo90)}% backward walk',
         f'{_fmt_pct(hi90)}_{_fmt_pct(lo90)}'),
        ('PlasticStrainRate_MaxSlope_s^-1',
         f'steepest {_fmt_pct(ms_window)}%-of-peak window (max slope)',
         'maxslope'),
    ]


def _grady_plastic_strainrate_valid_df(df, sr_col, label='2e'):
    """Build rows with plastic strain rate (per `sr_col`) and positive stress; shared by
    PA_02e / PA_02f.

    Each strain-rate column is computed per-trace directly from the raw velocity trace
    (walk backward from peak to an upper %-of-peak crossing, then further back to a lower
    %-of-peak crossing — or, for the max-slope method, whichever %-of-peak window is
    steepest; slope between the two points, scaled by bulk wave speed). Unlike the old
    HEL→peak definition, this does not require HEL_OK, so coverage here matches PA_02/02b
    rather than being restricted to shots with a successful HEL detection.
    """
    stress_col = 'Peak_Shock_Stress_GPa'
    need = {stress_col, sr_col}
    if not need.issubset(df.columns):
        print(f"  [skip] Grady {label} (plastic): missing columns:", sorted(need - set(df.columns)))
        return None

    d = df.copy()
    d[stress_col] = pd.to_numeric(d[stress_col], errors='coerce')
    d['_eps_pl'] = pd.to_numeric(d[sr_col], errors='coerce')

    valid = d.dropna(subset=[stress_col, '_eps_pl'])
    valid = valid[(valid[stress_col] > 0) & (valid['_eps_pl'] > 0)]
    if valid.empty:
        print(f"  [skip] Grady {label} (plastic): no positive stress / plastic strain-rate pairs.")
        return None
    return valid


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2e — Grady log–log (same layout as 2d): plastic strain rate vs shock stress
# ══════════════════════════════════════════════════════════════════════════════
def plot_grady_loglog_plastic_strainrate(df, out_dir, sr_col='PlasticStrainRate_80_20_s^-1',
                                          method_desc='80%-20% backward walk', file_tag='80_20'):
    """
    Same Grady treatment as plot_grady_loglog_fit_swapped_axes (2d): scatter in
    log10(ε̇) vs log10(σ_Pa), per-material polyfit and dashed line; Zn: median ±1×scaled MAD; others: mean ±1σ.
    Strain rate ε̇ is plastic only, from `sr_col` (one of PLASTIC_SR_METHODS): either an
    upper%/lower%-of-peak backward-walk method ((Δu/Δt)×10⁹/(2 c_b)) or the steepest
    %-of-peak window (max-slope method).
    """
    valid = _grady_plastic_strainrate_valid_df(df, sr_col, label=f'2e_{file_tag}')
    if valid is None:
        return

    stress_col = 'Peak_Shock_Stress_GPa'
    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        s_pa = s_gpa * 1e9
        e = sub['_eps_pl'].values.astype(float)
        x_stress = np.log10(s_pa)
        y_sr = np.log10(e)
        if len(x_stress) < 2:
            print(f"  [skip] Grady 2e (plastic) for {mat}: too few points.")
            continue

        A_m = None
        n_m = None
        logA_m = None
        if not (np.allclose(x_stress, x_stress[0]) or np.allclose(y_sr, y_sr[0])):
            try:
                n_m, logA_m = np.polyfit(x_stress, y_sr, 1)
                A_m = 10.0 ** logA_m
                print(f"  Grady 2e (plastic) fit for {mat} (ε̇ = A σ^n): n = {n_m:.3f}, A = {A_m:.3e} (σ in Pa)")
            except Exception as exc:
                print(f"  [skip] Grady 2e (plastic) for {mat}: regression failed: {exc}")

        leg_label = (
            f'{mat}  (N={len(x_stress)}, A={A_m:.2e}, n={n_m:.2f})'
            if A_m is not None and n_m is not None
            else f'{mat}  (n={len(x_stress)})')
        ax.scatter(y_sr, x_stress,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=leg_label)

        if A_m is not None and n_m is not None and abs(n_m) > 1e-12:
            x_fit_sr = np.linspace(y_sr.min(), y_sr.max(), 100)
            y_fit_stress = (1.0 / n_m) * x_fit_sr - logA_m / n_m
            ax.plot(x_fit_sr, y_fit_stress, color=colours[mat], linewidth=2.0,
                    linestyle='--')

    ax.set_xlabel('log$_{10}$[ Shock Strain Rate  (s$^{-1}$) ]', fontsize=FONT_AXIS)
    ax.set_ylabel('log$_{10}$[ Shock Stress  (Pa) ]', fontsize=FONT_AXIS)
    ax.set_title(
        'Grady Law Log–Log (shock $\\dot{\\varepsilon}$)\n'
        f'Shock Stress vs Shock Strain Rate ({method_desc})',
        fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best')
    fig.tight_layout()
    save(fig, out_dir, f'PA_02e_grady_loglog_plastic_strainrate_{file_tag}')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2f — Same as 2e but axes flipped: shock stress on x, plastic strain rate on y
# ══════════════════════════════════════════════════════════════════════════════
def plot_grady_loglog_plastic_strainrate_flipped(df, out_dir, sr_col='PlasticStrainRate_80_20_s^-1',
                                                  method_desc='80%-20% backward walk', file_tag='80_20'):
    """PA_02f: log–log plastic Grady plot with x = stress (Pa), y = strain rate (s⁻¹)."""
    valid = _grady_plastic_strainrate_valid_df(df, sr_col, label=f'2f_{file_tag}')
    if valid is None:
        return

    stress_col = 'Peak_Shock_Stress_GPa'
    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        s_pa = s_gpa * 1e9
        e = sub['_eps_pl'].values.astype(float)
        x_stress = np.log10(s_pa)
        y_sr = np.log10(e)
        if len(x_stress) < 2:
            print(f"  [skip] Grady 2f (plastic) for {mat}: too few points.")
            continue

        A_m = None
        n_m = None
        logA_m = None
        if not (np.allclose(x_stress, x_stress[0]) or np.allclose(y_sr, y_sr[0])):
            try:
                n_m, logA_m = np.polyfit(x_stress, y_sr, 1)
                A_m = 10.0 ** logA_m
                print(f"  Grady 2f (plastic) fit for {mat} (ε̇ = A σ^n): n = {n_m:.3f}, A = {A_m:.3e} (σ in Pa)")
            except Exception as exc:
                print(f"  [skip] Grady 2f (plastic) for {mat}: regression failed: {exc}")

        leg_label = (
            f'{mat}  (N={len(x_stress)}, A={A_m:.2e}, n={n_m:.2f})'
            if A_m is not None and n_m is not None
            else f'{mat}  (n={len(x_stress)})')
        ax.scatter(x_stress, y_sr,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=leg_label)

        if A_m is not None and n_m is not None and abs(n_m) > 1e-12:
            x_fit_stress = np.linspace(x_stress.min(), x_stress.max(), 100)
            y_fit_sr = n_m * x_fit_stress + logA_m
            ax.plot(x_fit_stress, y_fit_sr, color=colours[mat], linewidth=2.0,
                    linestyle='--')

    ax.set_xlabel('log$_{10}$[ Shock Stress  (Pa) ]', fontsize=FONT_AXIS)
    ax.set_ylabel('log$_{10}$[ Shock Strain Rate  (s$^{-1}$) ]', fontsize=FONT_AXIS)
    ax.set_title(
        'Grady Law Log–Log (shock $\\dot{\\varepsilon}$, axes flipped)\n'
        f'Shock Strain Rate vs Shock Stress ({method_desc})',
        fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best')
    fig.tight_layout()
    save(fig, out_dir, f'PA_02f_grady_loglog_plastic_strainrate_flipped_{file_tag}')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2g — Rise time t_r vs Shock Stress on log–log axes (per-material power law)
# ══════════════════════════════════════════════════════════════════════════════
def plot_risetime_vs_stress_loglog(df, out_dir, pct_label='80–20'):
    """Rise time t_r vs peak shock stress on true log–log axes, per material.

    t_r is the configurable %-of-peak backward-walk rise time (`RiseTime_80_20_ns`,
    the same t_r used for the shock-front-width diagnostic); falls back to
    `RiseTime_ArrivalToPeak_ns` if that column is absent (older CSVs). `pct_label`
    names the actual high/low percentages used (e.g. '95–30'), passed in from the
    analysis config by main() so the title matches whatever was actually configured.

    Axes are log-scaled with real units (GPa, ns). For each material a power law
    t_r = A · σ^n is fit by least squares on log10(t_r) vs log10(σ); the exponent n
    is shown in the legend."""
    stress_col = 'Peak_Shock_Stress_GPa'
    tr_col = ('RiseTime_80_20_ns' if 'RiseTime_80_20_ns' in df.columns
              else 'RiseTime_ArrivalToPeak_ns' if 'RiseTime_ArrivalToPeak_ns' in df.columns
              else None)
    if stress_col not in df.columns or tr_col is None:
        print("  [skip] t_r vs stress: missing Peak_Shock_Stress_GPa or a rise-time column.")
        return
    tr_label = f'{pct_label}% rise time' if tr_col == 'RiseTime_80_20_ns' else 'arrival-to-peak rise time'

    df2 = df.copy()
    df2[stress_col] = pd.to_numeric(df2[stress_col], errors='coerce')
    df2[tr_col] = pd.to_numeric(df2[tr_col], errors='coerce')
    valid = df2.dropna(subset=[stress_col, tr_col])
    valid = valid[(valid[stress_col] > 0) & (valid[tr_col] > 0)]
    if valid.empty:
        print("  [skip] t_r vs stress: no positive stress / rise-time pairs.")
        return

    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        tr_ns = sub[tr_col].values.astype(float)
        lx = np.log10(s_gpa)
        ly = np.log10(tr_ns)
        x_raw, y_raw = 10.0 ** lx, 10.0 ** ly
        if len(lx) < 2:
            print(f"  [skip] t_r vs stress for {mat}: too few points.")
            continue

        n_m = None
        if not (np.allclose(lx, lx[0]) or np.allclose(ly, ly[0])):
            try:
                n_m, logA_m = np.polyfit(lx, ly, 1)
                print(f"  t_r vs stress fit for {mat} (t_r = A σ^n): n = {n_m:.3f}, A = {10.0**logA_m:.3e} (σ in GPa, t_r in ns)")
            except Exception as exc:
                print(f"  [skip] t_r vs stress for {mat}: regression failed: {exc}")

        leg_label = f'{mat}  (N={len(lx)}, n={n_m:.2f})' if n_m is not None else f'{mat}  (N={len(lx)})'
        ax.scatter(x_raw, y_raw,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=mat_alpha(mat, 0.8), label=leg_label, zorder=3)

        if n_m is not None:
            x_fit = np.linspace(lx.min(), lx.max(), 100)
            y_fit = n_m * x_fit + logA_m
            ax.plot(10.0 ** x_fit, 10.0 ** y_fit, color=colours[mat],
                    linewidth=2.0, linestyle='--', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_ylabel(f'Shock Rise Time  $t_r$  (ns)', fontsize=FONT_AXIS)
    ax.set_title(f'Rise Time vs Shock Stress (log–log, per-material power law)\n'
                 f'$t_r$ = {tr_label}',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    apply_full_axis_frame(ax)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best', title='Material')
    fig.tight_layout()
    save(fig, out_dir, 'PA_02g_risetime_vs_stress_loglog')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2h — Shock Stress vs Rise time t_r on log–log axes (axes flipped vs 2g)
# ══════════════════════════════════════════════════════════════════════════════
def plot_risetime_vs_stress_loglog_flipped(df, out_dir, pct_label='80–20'):
    """Same data as PA_02g with axes flipped: rise time t_r on x, shock stress on y,
    on true log–log axes, per material.

    The per-material power law is refit in this orientation as σ = A · t_r^n (least
    squares on log10(σ) vs log10(t_r)), so the exponent n differs from 2g's t_r = A·σ^n
    unless the log–log correlation is perfect. t_r is `RiseTime_80_20_ns` (falls back to
    `RiseTime_ArrivalToPeak_ns` for older CSVs). `pct_label` names the actual high/low
    percentages used (e.g. '95–30'), passed in from the analysis config by main()."""
    stress_col = 'Peak_Shock_Stress_GPa'
    tr_col = ('RiseTime_80_20_ns' if 'RiseTime_80_20_ns' in df.columns
              else 'RiseTime_ArrivalToPeak_ns' if 'RiseTime_ArrivalToPeak_ns' in df.columns
              else None)
    if stress_col not in df.columns or tr_col is None:
        print("  [skip] stress vs t_r: missing Peak_Shock_Stress_GPa or a rise-time column.")
        return
    tr_label = f'{pct_label}% rise time' if tr_col == 'RiseTime_80_20_ns' else 'arrival-to-peak rise time'

    df2 = df.copy()
    df2[stress_col] = pd.to_numeric(df2[stress_col], errors='coerce')
    df2[tr_col] = pd.to_numeric(df2[tr_col], errors='coerce')
    valid = df2.dropna(subset=[stress_col, tr_col])
    valid = valid[(valid[stress_col] > 0) & (valid[tr_col] > 0)]
    if valid.empty:
        print("  [skip] stress vs t_r: no positive stress / rise-time pairs.")
        return

    mats, colours, mkrs = material_groups(valid)
    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            continue
        s_gpa = sub[stress_col].values.astype(float)
        tr_ns = sub[tr_col].values.astype(float)
        # x = t_r, y = stress (flipped vs 2g)
        lx = np.log10(tr_ns)
        ly = np.log10(s_gpa)
        x_raw, y_raw = 10.0 ** lx, 10.0 ** ly
        if len(lx) < 2:
            print(f"  [skip] stress vs t_r for {mat}: too few points.")
            continue

        n_m = None
        if not (np.allclose(lx, lx[0]) or np.allclose(ly, ly[0])):
            try:
                n_m, logA_m = np.polyfit(lx, ly, 1)
                print(f"  stress vs t_r fit for {mat} (σ = A t_r^m, m = dlogσ/dlogt_r): m = {n_m:.3f}, A = {10.0**logA_m:.3e} (t_r in ns, σ in GPa)")
            except Exception as exc:
                print(f"  [skip] stress vs t_r for {mat}: regression failed: {exc}")

        leg_label = f'{mat}  (N={len(lx)}, m={n_m:.2f})' if n_m is not None else f'{mat}  (N={len(lx)})'
        ax.scatter(x_raw, y_raw,
                   c=colours[mat], marker=mkrs[mat], s=60,
                   edgecolors='black', linewidths=0.5,
                   alpha=mat_alpha(mat, 0.8), label=leg_label, zorder=3)

        if n_m is not None:
            x_fit = np.linspace(lx.min(), lx.max(), 100)
            y_fit = n_m * x_fit + logA_m
            ax.plot(10.0 ** x_fit, 10.0 ** y_fit, color=colours[mat],
                    linewidth=2.0, linestyle='--', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'Shock Rise Time  $t_r$  (ns)', fontsize=FONT_AXIS)
    ax.set_ylabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_title(f'Shock Stress vs Rise Time (log–log, axes flipped; power law $\\sigma = A\\,t_r^{{m}}$, '
                 f'$m = d\\log\\sigma/d\\log t_r$)\n'
                 f'$t_r$ = {tr_label}',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    # Log axes: label plain integer values instead of the default "2×10^0" notation.
    # A linear (additive) tick step still crowds near the top of each decade on a log
    # axis (e.g. 6,7,8,9 sit much closer together than 1,2,3), so ticks are chosen
    # from a log-friendly digit set (1,2,3,5,7 per decade) that stays roughly evenly
    # spaced in log space, then thinned further if still too dense.
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

    def _nice_log_ticks(lo, hi, max_ticks=6):
        lo = max(1, lo)
        hi = max(lo * 1.01, hi)
        nice_digits = (1, 2, 3, 5, 7)
        decade_lo = int(np.floor(np.log10(lo)))
        decade_hi = int(np.ceil(np.log10(hi)))
        candidates = sorted({
            d * 10.0 ** k
            for k in range(decade_lo, decade_hi + 1)
            for d in nice_digits
            if lo - 1e-9 <= d * 10.0 ** k <= hi + 1e-9
        })
        if len(candidates) > max_ticks:
            step = int(np.ceil(len(candidates) / max_ticks))
            candidates = candidates[::step]
        return [int(round(v)) for v in candidates]

    def _plain_fmt():
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        return fmt

    ymin, ymax = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator(_nice_log_ticks(ymin, ymax)))
    ax.yaxis.set_major_formatter(_plain_fmt())
    ax.yaxis.set_minor_formatter(NullFormatter())

    xmin, xmax = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator(_nice_log_ticks(xmin, xmax)))
    ax.xaxis.set_major_formatter(_plain_fmt())
    ax.xaxis.set_minor_formatter(NullFormatter())
    apply_full_axis_frame(ax)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='best', title='Material')
    fig.tight_layout()
    save(fig, out_dir, 'PA_02h_stress_vs_risetime_loglog_flipped')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 3 — HEL Strength distribution per material (violin + inner box)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_violin(df, out_dir):
    """Violin plot showing the full HEL GPa distribution for each material.
    A mini box-plot is overlaid to show median and IQR clearly.
    Violins reveal multi-modality that box-plots hide."""
    hel = df[df['HEL_OK'] == True].copy()
    hel = hel.dropna(subset=['HEL_GPa'])
    hel['_material'] = hel['_material'].astype(str)
    hel = hel[hel['_material'].str.strip().str.lower() != 'zn']
    hel = hel[hel['HEL_GPa'] > 0]
    mats, colours, _ = material_groups(hel)
    if not mats:
        print("  [skip] No HEL data for violin plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = range(len(mats))

    for i, mat in enumerate(mats):
        vals = hel[hel['_material'] == mat]['HEL_GPa'].values
        if len(vals) < 3:
            continue
        kde  = gaussian_kde(vals, bw_method=0.35)
        y_rng = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 300)
        x_kde = kde(y_rng)
        x_kde /= x_kde.max()          # normalise to half-width = 0.4
        x_kde *= 0.4
        ax.fill_betweenx(y_rng, i - x_kde, i + x_kde,
                         color=colours[mat], alpha=0.55, zorder=2)
        ax.plot(np.concatenate([i - x_kde, (i + x_kde)[::-1]]),
                np.concatenate([y_rng, y_rng[::-1]]),
                color=colours[mat], linewidth=0.8, zorder=3)
        # inner box
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        iqr_h = 0.05
        ax.add_patch(mpatches.FancyBboxPatch(
            (i - iqr_h, q25), 2 * iqr_h, q75 - q25,
            boxstyle='square,pad=0', fc='white', ec=colours[mat], lw=1.5, zorder=4))
        ax.plot([i], [med], 'o', color=colours[mat], ms=5, zorder=5)
        ax.text(i, hel[hel['_material'] == mat]['HEL_GPa'].max() + 0.02,
                f'n={len(vals)}', ha='center', fontsize=FONT_TICK - 2, color='#444')

    ax.set_xticks(list(positions))
    ax.set_xticklabels(mats, fontsize=FONT_TICK)
    ax.set_ylabel('HEL Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('HEL Strength Distribution by Material\n(violin + inner box)', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    save(fig, out_dir, 'PA_03_hel_violin')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 4 — Spall Strength distribution per material (violin + jitter)
# ══════════════════════════════════════════════════════════════════════════════
def plot_spall_violin(df, out_dir):
    """Violin + jitter for Spall Strength.  Jitter is used (instead of pure violin)
    because spall detection is rare (n≈77 total) — showing raw points prevents
    the violin from misrepresenting a small sample."""
    sp = df[df['Spall_OK'] == True].copy()
    sp['Spall_Strength_GPa'] = pd.to_numeric(sp['Spall_Strength_GPa'], errors='coerce')
    sp = sp.dropna(subset=['Spall_Strength_GPa'])
    sp = sp[sp['Spall_Strength_GPa'] > 0]
    mats, colours, _ = material_groups(sp)
    if not mats:
        print("  [skip] No Spall data for violin plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(42)

    for i, mat in enumerate(mats):
        vals = sp[sp['_material'] == mat]['Spall_Strength_GPa'].values
        n    = len(vals)
        if n == 0:
            continue

        if n >= 4:
            kde  = gaussian_kde(vals, bw_method=0.5)
            y_rng = np.linspace(max(vals.min() - 0.1, 0), vals.max() + 0.1, 200)
            x_kde = kde(y_rng); x_kde /= x_kde.max(); x_kde *= 0.38
            ax.fill_betweenx(y_rng, i - x_kde, i + x_kde,
                             color=colours[mat], alpha=0.4, zorder=2)

        jitter = rng.uniform(-0.12, 0.12, n)
        ax.scatter(i + jitter, vals, color=colours[mat],
                   s=40, alpha=0.8, edgecolors='black', linewidths=0.4, zorder=4)

        if n >= 2:
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.plot([i - 0.1, i + 0.1], [med, med], '-', lw=2, color='black', zorder=5)
            ax.plot([i, i], [q25, q75], '-', lw=1.5, color='black', zorder=5)

        ax.text(i, (vals.max() or 0) + 0.05, f'n={n}',
                ha='center', fontsize=FONT_TICK - 2, color='#444')

    ax.set_xticks(list(range(len(mats))))
    ax.set_xticklabels(mats, fontsize=FONT_TICK)
    ax.set_ylabel('Spall Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Spall Strength Distribution by Material\n(violin + jitter, median line)', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    save(fig, out_dir, 'PA_04_spall_violin_jitter')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 5 — Ridgeline: HEL GPa distribution stacked per material
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_ridgeline(df, out_dir):
    """Ridgeline (joy) plot: one overlapping KDE per material, stacked vertically.
    Ideal for comparing the shapes of 3–5 distributions along a shared X axis."""
    hel = df[df['HEL_OK'] == True].dropna(subset=['HEL_GPa'])
    hel = hel[hel['HEL_GPa'] > 0]
    mats, colours, _ = material_groups(hel)
    if not mats:
        print("  [skip] No HEL data for ridgeline plot.")
        return

    x_min = max(hel['HEL_GPa'].min() - 0.05, 0)
    x_max = hel['HEL_GPa'].max() + 0.05
    x_rng = np.linspace(x_min, x_max, 400)
    overlap = 0.7          # how much rows overlap (0 = no overlap, 1 = full)

    n_rows = len(mats)
    fig_h  = 2.0 + n_rows * 1.4
    fig, ax = plt.subplots(figsize=(9, fig_h))

    for i, mat in enumerate(reversed(mats)):   # bottom to top
        vals = hel[hel['_material'] == mat]['HEL_GPa'].values
        if len(vals) < 3:
            continue
        kde   = gaussian_kde(vals, bw_method=0.25)
        y_kde = kde(x_rng)
        y_kde /= y_kde.max()
        baseline = i * (1 - overlap)

        ax.fill_between(x_rng, baseline, baseline + y_kde,
                        color=colours[mat], alpha=0.65, zorder=i + 1)
        ax.plot(x_rng, baseline + y_kde, color=colours[mat], lw=1.5, zorder=i + 2)
        ax.axhline(baseline, color='white', lw=0.8, zorder=i)
        ax.text(x_min - 0.01, baseline + 0.02, f'{mat}\n(n={len(vals)})',
                ha='right', va='bottom', fontsize=FONT_AXIS - 2, color=colours[mat], fontweight='bold')

    ax.set_xlabel('HEL Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_yticks([])
    ax.set_title('HEL Strength — Ridgeline by Material', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.set_xlim(x_min, x_max)
    fig.tight_layout()
    save(fig, out_dir, 'PA_05_hel_ridgeline')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 6 — Detection rate lollipop  (HEL & Spall per material)
# ══════════════════════════════════════════════════════════════════════════════
def plot_detection_lollipop(df, out_dir):
    """Horizontal lollipop showing HEL and Spall detection counts and rates per
    material.  Lollipops are preferred over bar charts when bar heights are
    similar (avoids Moiré / visual clutter)."""
    mats, colours, _ = material_groups(df)
    records = []
    for mat in mats:
        sub  = df[df['_material'] == mat]
        n    = len(sub)
        n_hel   = int((sub['HEL_OK']   == True).sum()) if 'HEL_OK'   in sub.columns else 0
        n_spall = int((sub['Spall_OK'] == True).sum()) if 'Spall_OK' in sub.columns else 0
        records.append({'mat': mat, 'n': n,
                        'HEL':   n_hel,   'HEL_pct':   100 * n_hel   / max(n, 1),
                        'Spall': n_spall, 'Spall_pct': 100 * n_spall / max(n, 1)})
    rec = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, metric, title in zip(axes,
                                  ['HEL', 'Spall'],
                                  ['HEL Detection', 'Spall Detection']):
        y_pos = range(len(rec))
        vals  = rec[metric].values
        pcts  = rec[metric + '_pct'].values
        clrs  = [colours[m] for m in rec['mat']]

        ax.hlines(list(y_pos), 0, vals, colors=clrs, linewidths=2, alpha=0.7)
        ax.scatter(vals, list(y_pos), color=clrs, s=120, zorder=5,
                   edgecolors='black', linewidths=0.6)
        for j, (v, p, n_tot) in enumerate(zip(vals, pcts, rec['n'])):
            ax.text(v + max(vals.max() * 0.01, 1), j,
                    f'{v}  ({p:.0f}% of {n_tot})',
                    va='center', fontsize=FONT_TICK - 2)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(rec['mat'], fontsize=FONT_TICK)
        ax.set_xlabel('Count', fontsize=FONT_AXIS)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
        ax.set_xlim(0, vals.max() * 1.35)
        ax.tick_params(labelsize=FONT_TICK)

    fig.suptitle('Detection Rates per Material', fontsize=FONT_TITLE, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, out_dir, 'PA_06_detection_lollipop')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 7 — Pairwise scatter matrix (correlogram) of key physics variables
# ══════════════════════════════════════════════════════════════════════════════
def _plot_correlogram_generic(df, out_dir, var_dict, suptitle, out_name):
    """Shared scatter-matrix builder for the PA_07 correlogram variants.
    Diagonal panels show per-material KDE (distribution); off-diagonal are scatter
    coloured by material.  Reveals pairwise correlations and group separation."""
    labels   = list(var_dict.keys())
    col_keys = list(var_dict.values())

    # keep only rows with at least 2 of the 3 variables finite
    sub = df[col_keys].apply(pd.to_numeric, errors='coerce')
    sub['_material'] = df['_material']
    sub = sub[sub[col_keys].notna().sum(axis=1) >= 2]
    sub['_material'] = sub['_material'].astype(str)

    mats, colours, mkrs = material_groups(sub)
    # Darken Zn so it is visible against a white background
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'  # dark green
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(3 * n + 2, 3 * n))
    x_rng = np.linspace(0, 1, 200)

    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            ck_r = col_keys[r]
            ck_c = col_keys[c]
            if r == c:
                # Diagonal: KDE per material
                all_vals = pd.to_numeric(sub[ck_r], errors='coerce').dropna()
                if len(all_vals) < 2:
                    ax.set_visible(False); continue
                for mat in mats:
                    vals = pd.to_numeric(
                        sub.loc[sub['_material'] == mat, ck_r], errors='coerce').dropna()
                    if len(vals) < 3: continue
                    try:
                        kde = gaussian_kde(vals, bw_method=0.4)
                        vr  = np.linspace(all_vals.min(), all_vals.max(), 300)
                        ax.plot(vr, kde(vr), color=colours[mat], lw=1.5)
                        ax.fill_between(vr, kde(vr), alpha=0.2, color=colours[mat])
                    except Exception:
                        pass
                ax.set_xlabel(labels[r], fontsize=FONT_AXIS)
                ax.set_yticks([])
                ax.set_ylabel('Probability Density', fontsize=FONT_AXIS)
            else:
                # Off-diagonal: scatter
                for mat in mats:
                    mv = sub['_material'] == mat
                    xv = pd.to_numeric(sub.loc[mv, ck_c], errors='coerce')
                    yv = pd.to_numeric(sub.loc[mv, ck_r], errors='coerce')
                    ok = xv.notna() & yv.notna()
                    if ok.sum() < 1: continue
                    ax.scatter(xv[ok], yv[ok], c=colours[mat],
                               marker=mkrs[mat], s=9.4, alpha=0.5,
                               edgecolors='none')

            # Uniform tick size across all panels (diagonal and off-diagonal)
            ax.tick_params(labelsize=FONT_TICK)

            # Axis limits: HEL strain rate 0.5e5–4e5
            if ck_c == 'HEL_StrainRate_s^-1':
                ax.set_xlim(0.5e5, 4e5)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                ax.xaxis.offsetText.set_fontsize(9)
            if r != c and ck_r == 'HEL_StrainRate_s^-1':
                ax.set_ylim(0.5e5, 4e5)
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                ax.yaxis.offsetText.set_fontsize(9)

            if c == 0 and r != c: ax.set_ylabel(labels[r], fontsize=FONT_AXIS)
            if r == n - 1: ax.set_xlabel(labels[c], fontsize=FONT_AXIS)

    # Legend as a single row beneath the panel grid
    counts = sub['_material'].value_counts()
    handles = [Line2D([0], [0], marker=mkrs[m], color='w',
                       markerfacecolor=colours[m], markersize=20,
                       markeredgecolor='black', label=f'{m} (n={counts.get(m, 0)})')
               for m in mats]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure,
               ncol=len(handles), fontsize=13, framealpha=0.9, title='Material')
    fig.suptitle(suptitle, fontsize=FONT_TITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, out_dir, out_name)


def plot_correlogram(df, out_dir):
    """Scatter matrix: HEL, 1/elastic rise time, and Laser Energy."""
    _VARS = {
        'HEL\n(GPa)':                  'HEL_GPa',
        '1/Elastic\nRise Time ($s^{-1}$)': 'HEL_StrainRate_s^-1',
        'Laser Energy\n(mJ)':          'Laser_Target_Energy (mJ)',
    }
    _plot_correlogram_generic(df, out_dir, _VARS,
                               'Pairwise Scatter Matrix — Key Shock Physics Variables',
                               'PA_07_correlogram')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 7b — Pairwise scatter matrix (correlogram), Laser Energy replaced by
#  Peak Shock Stress (no dependence on the Laser_Target_Energy column)
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlogram_stress(df, out_dir):
    """Scatter matrix: HEL, 1/elastic rise time, and Peak Shock Stress.

    Same layout as PA_07 but substitutes Peak_Shock_Stress_GPa for Laser Energy —
    the drive stress is the physical loading variable behind the classic elastic
    precursor-decay relationship with HEL, and (unlike laser energy) is always
    populated regardless of which raw-file column layout (HAAPI vs LMI) produced
    a given shot."""
    _VARS = {
        'HEL\n(GPa)':                  'HEL_GPa',
        '1/Elastic\nRise Time ($s^{-1}$)': 'HEL_StrainRate_s^-1',
        'Peak Stress\n(GPa)':          'Peak_Shock_Stress_GPa',
    }
    _plot_correlogram_generic(df, out_dir, _VARS,
                               'Pairwise Scatter Matrix — HEL, Strain Rate & Peak Stress',
                               'PA_07b_correlogram_stress')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 7c — First-row detail from PA_07b: HEL distribution + HEL vs
#  1/Elastic Rise Time scatter only (no full correlogram), legend in-panel
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlogram_stress_top2(df, out_dir):
    """Two-panel excerpt of PA_07b's first row: per-material HEL KDE (top) and
    HEL vs 1/Rise Time scatter (bottom), each with its own in-panel legend."""
    col_hel, col_rate = 'HEL_GPa', 'HEL_StrainRate_s^-1'
    label_hel  = 'HEL (GPa)'
    label_rate = '1/Elastic Rise Time ($s^{-1}$)'

    sub = df[[col_hel, col_rate]].apply(pd.to_numeric, errors='coerce')
    sub['_material'] = df['_material']
    sub = sub[sub[[col_hel, col_rate]].notna().sum(axis=1) >= 2]
    sub['_material'] = sub['_material'].astype(str)

    mats, colours, mkrs = material_groups(sub)
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'

    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    counts = sub['_material'].value_counts()

    # Top panel: per-material KDE of HEL
    ax = axes[0]
    all_vals = pd.to_numeric(sub[col_hel], errors='coerce').dropna()
    # Extend the evaluation range below the data minimum so the KDE tail tapers
    # naturally instead of being cut off (trimmed) right at the lowest data point.
    vals_span = all_vals.max() - all_vals.min()
    pad = 0.15 * vals_span if vals_span > 0 else 1.0
    vr_lo = max(0.0, all_vals.min() - pad)
    vr = np.linspace(vr_lo, all_vals.max() + pad, 300)
    for mat in mats:
        vals = pd.to_numeric(sub.loc[sub['_material'] == mat, col_hel], errors='coerce').dropna()
        if len(vals) < 3:
            continue
        try:
            kde = gaussian_kde(vals, bw_method=0.4)
            ax.plot(vr, kde(vr), color=colours[mat], lw=1.5)
            ax.fill_between(vr, kde(vr), alpha=0.2, color=colours[mat])
        except Exception:
            pass
    ax.set_xlabel(label_hel, fontsize=FONT_AXIS)
    ax.set_xlim(vr_lo, all_vals.max() + pad)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_ylabel('Probability Density', fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK - 1)

    # Bottom panel: HEL vs 1/Rise Time scatter
    ax = axes[1]
    for mat in mats:
        mv = sub['_material'] == mat
        xv = pd.to_numeric(sub.loc[mv, col_rate], errors='coerce')
        yv = pd.to_numeric(sub.loc[mv, col_hel], errors='coerce')
        ok = xv.notna() & yv.notna()
        if ok.sum() < 1:
            continue
        ax.scatter(xv[ok], yv[ok], c=colours[mat], marker=mkrs[mat], s=45,
                   alpha=0.6, edgecolors='none')
    ax.set_xlim(0.5e5, 4e5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_major_formatter().set_useMathText(True)  # "×10^5" instead of "1e5"
    ax.xaxis.offsetText.set_fontsize(FONT_LEG)
    ax.set_xlabel('1/Rise Time ($s^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel(label_hel, fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK - 1)

    # In-panel legend (each subplot carries its own copy)
    handles = [Line2D([0], [0], marker=mkrs[m], color='w',
                       markerfacecolor=colours[m], markersize=7,
                       markeredgecolor='black', label=f'{m} (n={counts.get(m, 0)})')
               for m in mats]
    for ax in axes:
        ax.legend(handles=handles, loc='best', fontsize=FONT_LEG, framealpha=0.9, title='Material')

    fig.suptitle('HEL Distribution and HEL vs 1/Elastic Rise Time', fontsize=FONT_TITLE, fontweight='bold')
    fig.tight_layout()
    save(fig, out_dir, 'PA_07c_correlogram_stress_top2')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 8 — HEL GPa vs Shock Stress  (scatter, per material)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_peak_stress(df, out_dir):
    """Scatter: HEL strength vs Shock Stress, coloured by material.
    Traces the elastic-plastic boundary — shows whether HEL scales with drive stress."""
    hel = df[(df['HEL_OK'] == True)].dropna(subset=['HEL_GPa', 'Peak_Shock_Stress_GPa'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['Peak_Shock_Stress_GPa'] > 0)]
    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] No data for HEL vs Peak Stress plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for mat in mats:
        sub = hel[hel['_material'] == mat]
        ax.scatter(sub['Peak_Shock_Stress_GPa'], sub['HEL_GPa'],
                   c=colours[mat], marker=mkrs[mat], s=65,
                   edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=f'{mat}  (n={len(sub)})')

    ax.set_xlabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL Strength  (GPa)',       fontsize=FONT_AXIS)
    ax.set_title('HEL Strength vs Shock Stress\n(elastic–plastic boundary)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, 'PA_08_hel_vs_peak_stress')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 9 — HEL vs 1/Elastic Rise Time  (scatter, per material)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_hel_strain_rate(df, out_dir):
    """Scatter: HEL (GPa) vs HEL (elastic) strain rate (s⁻¹), coloured by material."""
    if not {'HEL_GPa', 'HEL_StrainRate_s^-1'}.issubset(df.columns):
        print("  [skip] Missing HEL_GPa or HEL_StrainRate_s^-1 for HEL vs HEL strain rate plot.")
        return

    hel = df[(df['HEL_OK'] == True)].copy()
    hel['HEL_GPa'] = pd.to_numeric(hel['HEL_GPa'], errors='coerce')
    hel['HEL_StrainRate_s^-1'] = pd.to_numeric(hel['HEL_StrainRate_s^-1'], errors='coerce')
    hel = hel.dropna(subset=['HEL_GPa', 'HEL_StrainRate_s^-1'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['HEL_StrainRate_s^-1'] > 0)]
    # Remove Zn from this plot
    if '_material' in hel.columns:
        hel['_material'] = hel['_material'].astype(str)
        hel = hel[hel['_material'].str.strip().str.lower() != 'zn']

    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] No data for HEL vs HEL strain rate plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    for mat in mats:
        sub = hel[hel['_material'] == mat]
        ax.scatter(sub['HEL_StrainRate_s^-1'], sub['HEL_GPa'],
                   c=colours[mat], marker=mkrs[mat], s=65,
                   edgecolors='black', linewidths=0.6,
                   alpha=0.85, label=f'{mat} (n={len(sub)})')

    ax.set_xlabel('1/Elastic Rise Time (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL (GPa)', fontsize=FONT_AXIS)
    ax.set_title('HEL vs 1/Elastic Rise Time — Multi-Material',
                 fontsize=FONT_TITLE, fontweight='bold')

    # X-axis: scientific notation, 0.5×10^5 … 4×10^5
    ax.set_xlim(0.5e5, 4e5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='upper right')
    fig.tight_layout()
    save(fig, out_dir, 'PA_09_hel_vs_hel_strain_rate')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 10 — HEL vs 1/Elastic Rise Time  (scatter + per-material regressions)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_hel_strain_rate_regression(df, out_dir):
    """Scatter + regression lines: HEL (GPa) vs HEL strain rate (s⁻¹) per material."""
    if not {'HEL_GPa', 'HEL_StrainRate_s^-1'}.issubset(df.columns):
        print("  [skip] Missing HEL_GPa or HEL_StrainRate_s^-1 for HEL vs HEL strain rate regression plot.")
        return

    hel = df[(df['HEL_OK'] == True)].copy()
    hel['HEL_GPa'] = pd.to_numeric(hel['HEL_GPa'], errors='coerce')
    hel['HEL_StrainRate_s^-1'] = pd.to_numeric(hel['HEL_StrainRate_s^-1'], errors='coerce')
    hel = hel.dropna(subset=['HEL_GPa', 'HEL_StrainRate_s^-1'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['HEL_StrainRate_s^-1'] > 0)]
    # Remove Zn from this plot (match filters used in Plot 9)
    if '_material' in hel.columns:
        hel['_material'] = hel['_material'].astype(str)
        hel = hel[hel['_material'].str.strip().str.lower() != 'zn']

    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] No data for HEL vs HEL strain rate regression plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = hel[hel['_material'] == mat]
        x = sub['HEL_StrainRate_s^-1'].values.astype(float)
        y = sub['HEL_GPa'].values.astype(float)

        # Always show all points in the scatter
        ax.scatter(x, y,
                   c=colours[mat], marker=mkrs[mat], s=65,
                   edgecolors='black', linewidths=0.6,
                   alpha=0.75, label=f'{mat} (n={len(sub)})')

        # Restrict regression to the lower 90th percentile of strain rate
        if len(x) < 2 or np.allclose(x, x[0]):
            continue
        try:
            p90 = np.nanpercentile(x, 90)
        except Exception:
            continue
        reg_mask = np.isfinite(x) & np.isfinite(y) & (x <= p90)

        # Apply a 2σ filter in HEL (Y) to remove strong outliers before fitting
        if np.any(reg_mask):
            y_all_finite = y[np.isfinite(y)]
            if y_all_finite.size >= 2:
                y_mean = float(np.mean(y_all_finite))
                y_std = float(np.std(y_all_finite))
                if y_std > 0:
                    ymin = y_mean - 2.0 * y_std
                    ymax = y_mean + 2.0 * y_std
                    reg_mask &= (y >= ymin) & (y <= ymax)

        if reg_mask.sum() < 2 or np.allclose(x[reg_mask], x[reg_mask][0]):
            continue

        x_reg = x[reg_mask]
        y_reg = y[reg_mask]

        # Simple linear regression in linear space on the central regime only
        try:
            m, b = np.polyfit(x_reg, y_reg, 1)
            x_fit = np.linspace(max(0.5e5, x_reg.min()), min(4e5, x_reg.max()), 100)
            y_fit = m * x_fit + b
            ax.plot(x_fit, y_fit, color=colours[mat],
                    linewidth=2.0, alpha=0.9)
        except Exception:
            # Fall back silently if regression fails numerically
            continue

    ax.set_xlabel('1/Elastic Rise Time (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL (GPa)', fontsize=FONT_AXIS)
    ax.set_title('HEL vs 1/Elastic Rise Time — Per-Material Linear Fits',
                 fontsize=FONT_TITLE, fontweight='bold')

    ax.set_xlim(0.5e5, 4e5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='upper right')
    fig.tight_layout()
    save(fig, out_dir, 'PA_10_hel_vs_hel_strain_rate_regression')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 11 — HEL vs 1/Elastic Rise Time  (per-material mean ±1σ vs strain rate)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_hel_strain_rate_mean(df, out_dir):
    """Per-material mean HEL vs strain rate, with ±1σ shaded band."""
    if not {'HEL_GPa', 'HEL_StrainRate_s^-1'}.issubset(df.columns):
        print("  [skip] Missing HEL_GPa or HEL_StrainRate_s^-1 for HEL vs HEL strain rate mean plot.")
        return

    hel = df[(df['HEL_OK'] == True)].copy()
    hel['HEL_GPa'] = pd.to_numeric(hel['HEL_GPa'], errors='coerce')
    hel['HEL_StrainRate_s^-1'] = pd.to_numeric(hel['HEL_StrainRate_s^-1'], errors='coerce')
    hel = hel.dropna(subset=['HEL_GPa', 'HEL_StrainRate_s^-1'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['HEL_StrainRate_s^-1'] > 0)]

    # Remove Zn from this plot, to match plots 9–10
    if '_material' in hel.columns:
        hel['_material'] = hel['_material'].astype(str)
        hel = hel[hel['_material'].str.strip().str.lower() != 'zn']

    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] No data for HEL vs HEL strain rate mean plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for mat in mats:
        sub = hel[hel['_material'] == mat]
        if sub.empty:
            continue

        # Sort by strain rate
        sub = sub.sort_values('HEL_StrainRate_s^-1')
        x_all = sub['HEL_StrainRate_s^-1'].values.astype(float)
        y_all = sub['HEL_GPa'].values.astype(float)

        window = 20
        x_bins = []
        y_mean = []
        y_std = []

        for start in range(0, len(x_all), window):
            end = min(start + window, len(x_all))
            x_w = x_all[start:end]
            y_w = y_all[start:end]
            if len(x_w) == 0:
                continue
            x_bins.append(np.mean(x_w))
            y_mean.append(np.mean(y_w))
            if len(y_w) > 1:
                y_std.append(np.std(y_w, ddof=1))
            else:
                y_std.append(0.0)

        if len(x_bins) < 2:
            continue

        x_bins = np.array(x_bins, dtype=float)
        y_mean = np.array(y_mean, dtype=float)
        y_std = np.array(y_std, dtype=float)

        order = np.argsort(x_bins)
        x_bins = x_bins[order]
        y_mean = y_mean[order]
        y_std = y_std[order]

        ax.plot(x_bins, y_mean, color=colours[mat],
                marker=mkrs[mat], linewidth=2.0, markersize=5,
                label=f'{mat} (mean of {window} pts)')
        ax.fill_between(x_bins,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=colours[mat],
                        alpha=0.18)

    ax.set_xlabel('1/Elastic Rise Time (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL (GPa)', fontsize=FONT_AXIS)
    ax.set_title('HEL vs 1/Elastic Rise Time — Per-Material Mean ±1σ',
                 fontsize=FONT_TITLE, fontweight='bold')

    ax.set_xlim(0.5e5, 4e5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='upper right')
    fig.tight_layout()
    save(fig, out_dir, 'PA_11_hel_vs_hel_strain_rate_mean')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 12 — HEL vs 1/Elastic Rise Time  (rolling mean ±1σ, stride=10)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_hel_strain_rate_mean_rolling(df, out_dir):
    """Per-material rolling mean HEL vs strain rate, with ±1σ shaded band.
    Uses overlapping windows (stride=10) for fewer edge effects."""
    if not {'HEL_GPa', 'HEL_StrainRate_s^-1'}.issubset(df.columns):
        print("  [skip] Missing HEL_GPa or HEL_StrainRate_s^-1 for HEL vs HEL strain rate rolling mean plot.")
        return

    hel = df[(df['HEL_OK'] == True)].copy()
    hel['HEL_GPa'] = pd.to_numeric(hel['HEL_GPa'], errors='coerce')
    hel['HEL_StrainRate_s^-1'] = pd.to_numeric(hel['HEL_StrainRate_s^-1'], errors='coerce')
    hel = hel.dropna(subset=['HEL_GPa', 'HEL_StrainRate_s^-1'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['HEL_StrainRate_s^-1'] > 0)]

    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] No data for HEL vs HEL strain rate rolling mean plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    window = 5  # points per window
    for mat in mats:
        sub = hel[hel['_material'] == mat]
        if sub.empty or len(sub) < window:
            continue

        # Sort by strain rate
        sub = sub.sort_values('HEL_StrainRate_s^-1')
        x_all = sub['HEL_StrainRate_s^-1'].values.astype(float)
        y_all = sub['HEL_GPa'].values.astype(float)

        # Rolling windows: stride=10, so [0..4], [10..14], [20..24], ...
        stride = 10
        x_bins = []
        y_mean = []
        y_std = []

        for start in range(0, len(x_all) - window + 1, stride):
            end = start + window
            x_w = x_all[start:end]
            y_w = y_all[start:end]
            x_bins.append(np.mean(x_w))
            y_mean.append(np.mean(y_w))
            if len(y_w) > 1:
                y_std.append(np.std(y_w, ddof=1))
            else:
                y_std.append(0.0)

        if len(x_bins) < 2:
            continue

        x_bins = np.array(x_bins, dtype=float)
        y_mean = np.array(y_mean, dtype=float)
        y_std = np.array(y_std, dtype=float)

        ax.plot(x_bins, y_mean, color=colours[mat],
                marker=mkrs[mat], linewidth=2.0, markersize=4,
                label=f'{mat} (rolling n={window})')
        ax.fill_between(x_bins,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=colours[mat],
                        alpha=0.18)

    ax.set_xlabel('1/Elastic Rise Time (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL (GPa)', fontsize=FONT_AXIS)
    ax.set_title('HEL vs 1/Elastic Rise Time — Rolling Mean ±1σ (stride=10)',
                 fontsize=FONT_TITLE, fontweight='bold')

    ax.set_xlim(0.5e5, 4e5)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc='upper right')
    fig.tight_layout()
    save(fig, out_dir, 'PA_12_hel_vs_hel_strain_rate_mean_rolling')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 15 — HEL vs 1/Elastic Rise Time  (per-material faceted, log x-axis)
# ══════════════════════════════════════════════════════════════════════════════
def plot_hel_vs_hel_strain_rate_faceted(df, out_dir):
    """Per-material faceted scatter: HEL (GPa) vs HEL strain rate with log x-axis.

    One subplot per material removes inter-material overlap so each trend is
    visible on its own scale.  The x-axis is log₁₀(HEL strain rate) to
    linearise the typically power-law relationship.  A least-squares line is
    fitted in log-x space and drawn with a ±1σ-residual shaded band so the
    direction and tightness of each material's trend are immediately obvious.
    """
    if not {'HEL_GPa', 'HEL_StrainRate_s^-1'}.issubset(df.columns):
        print("  [skip] PA_15: Missing HEL_GPa or HEL_StrainRate_s^-1.")
        return

    hel = df[df['HEL_OK'] == True].copy()
    hel['HEL_GPa'] = pd.to_numeric(hel['HEL_GPa'], errors='coerce')
    hel['HEL_StrainRate_s^-1'] = pd.to_numeric(hel['HEL_StrainRate_s^-1'], errors='coerce')
    hel = hel.dropna(subset=['HEL_GPa', 'HEL_StrainRate_s^-1'])
    hel = hel[(hel['HEL_GPa'] > 0) & (hel['HEL_StrainRate_s^-1'] > 0)]
    if '_material' in hel.columns:
        hel['_material'] = hel['_material'].astype(str)
        hel = hel[hel['_material'].str.strip().str.lower() != 'zn']

    mats, colours, mkrs = material_groups(hel)
    if not mats:
        print("  [skip] PA_15: No data after filtering.")
        return

    ncols = min(3, len(mats))
    nrows = (len(mats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 5 * nrows),
                              squeeze=False)

    for idx, mat in enumerate(mats):
        ax = axes.flat[idx]
        sub = hel[hel['_material'] == mat]
        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub['HEL_StrainRate_s^-1'].values.astype(float)
        y = sub['HEL_GPa'].values.astype(float)

        ax.scatter(x, y, c=colours[mat], marker=mkrs[mat], s=55,
                   edgecolors='black', linewidths=0.6, alpha=0.85, zorder=3)
        ax.set_xscale('log')

        log_x = np.log10(x)
        if len(log_x) >= 3 and not np.allclose(log_x, log_x[0]):
            try:
                m_coef, b_coef = np.polyfit(log_x, y, 1)
                log_x_fit = np.linspace(log_x.min(), log_x.max(), 300)
                x_fit = 10.0 ** log_x_fit
                y_fit = m_coef * log_x_fit + b_coef
                y_pred = m_coef * log_x + b_coef
                sigma = np.std(y - y_pred, ddof=1) if len(y) > 2 else 0.0
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

                ax.plot(x_fit, y_fit, color=colours[mat], linewidth=2.2, zorder=4)
                if sigma > 0:
                    ax.fill_between(x_fit, y_fit - sigma, y_fit + sigma,
                                    color=colours[mat], alpha=0.20, zorder=2)

                ann = (f'n={len(x)}\n'
                       f'slope={m_coef:.3f} GPa/decade\n'
                       f'R²={r2:.3f}')
                ax.text(0.04, 0.96, ann,
                        transform=ax.transAxes,
                        fontsize=FONT_TICK - 3, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#cccccc', alpha=0.85))
            except Exception:
                ax.text(0.04, 0.96, f'n={len(x)}',
                        transform=ax.transAxes, fontsize=FONT_TICK - 3, va='top')
        else:
            ax.text(0.04, 0.96, f'n={len(x)}',
                    transform=ax.transAxes, fontsize=FONT_TICK - 3, va='top')

        ax.set_title(mat, fontsize=FONT_TITLE, fontweight='bold', color=colours[mat])
        ax.set_xlabel('1/Elastic Rise Time  (s$^{-1}$)', fontsize=FONT_AXIS)
        ax.set_ylabel('HEL  (GPa)', fontsize=FONT_AXIS)
        ax.tick_params(labelsize=FONT_TICK, which='both')
        apply_full_axis_frame(ax)

    for j in range(len(mats), nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle(
        'HEL vs 1/Elastic Rise Time — Per-Material Faceted\n'
        '(log x-axis · regression line · ±1σ residual band)',
        fontsize=FONT_TITLE, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, out_dir, 'PA_15_hel_vs_hel_strain_rate_faceted')


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 13 — Shock Stress vs Flyer Row × Column (stratified by laser energy ±50 mJ)
# ══════════════════════════════════════════════════════════════════════════════
def _get_flyer_col(df, row_name, col_name):
    """Resolve Flyer_Row / Flyer_Column column names (allow case variants and common alternates)."""
    row_cand = next((c for c in df.columns if str(c).strip().lower() == row_name.lower()), None)
    col_cand = next((c for c in df.columns if str(c).strip().lower() == col_name.lower()), None)
    if row_cand is None:
        row_cand = next((c for c in df.columns if 'flyer' in str(c).lower() and 'row' in str(c).lower()), None)
    if col_cand is None:
        col_cand = next((c for c in df.columns if 'flyer' in str(c).lower() and 'col' in str(c).lower()), None)
    if row_cand is None:
        row_cand = next((c for c in df.columns if str(c).strip().lower() == 'row'), None)
    if col_cand is None:
        col_cand = next((c for c in df.columns if str(c).strip().lower() in ('column', 'col')), None)
    return row_cand, col_cand


def _get_energy_col(df):
    """Resolve Laser Energy (mJ) column. Prefer exact 'Laser_Target_Energy (mJ)'."""
    exact = 'Laser_Target_Energy (mJ)'
    if exact in df.columns:
        return exact
    for c in df.columns:
        s = str(c)
        if 'laser' in s.lower() and 'energy' in s.lower() and 'mj' in s.lower():
            return c
    return None


def _get_pdv_return_power_col(df):
    """Resolve PDV return power column. Prefer exact 'PDV_Return_Power (dBm)'."""
    exact = 'PDV_Return_Power (dBm)'
    if exact in df.columns:
        return exact
    for c in df.columns:
        s = str(c).lower()
        if 'pdv' in s and 'return' in s and 'power' in s:
            return c
    return None


def _get_hel_unc_col(df):
    """Resolve HEL uncertainty column (GPa)."""
    for name in ['HEL_Unc_GPa', 'HEL_Uncertainty_GPa', 'HEL_Error_GPa']:
        if name in df.columns:
            return name
    for c in df.columns:
        s = str(c).lower()
        if 'hel' in s and ('unc' in s or 'uncertainty' in s or 'error' in s):
            return c
    return None


# Nominal laser energies (mJ) for stratification; each band is E ± 50 mJ
_ENERGY_BANDS_MJ = [400, 800, 1200, 1600]
_ENERGY_TOL_MJ = 50


def plot_peak_stress_vs_flyer_row_column(df, out_dir):
    """Shock Stress vs (Flyer_Row, Flyer_Column) at similar laser energy (±50 mJ).
    One figure per material; subplots = energy bands. Fallback: one plot with all data if no band has data."""
    row_col = _get_flyer_col(df, 'Flyer_Row', 'Flyer_Column')
    flyer_row_col, flyer_col_col = row_col[0], row_col[1]
    energy_col = _get_energy_col(df)

    if flyer_row_col is None or flyer_col_col is None:
        print("  [skip] PA_13: Missing Flyer_Row or Flyer_Column. Available:", list(df.columns)[:15], "...")
        return

    df2 = df.copy()
    df2['Peak_Shock_Stress_GPa'] = pd.to_numeric(df2['Peak_Shock_Stress_GPa'], errors='coerce')
    df2['_row'] = pd.to_numeric(df2[flyer_row_col].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    df2['_col'] = pd.to_numeric(df2[flyer_col_col].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    if energy_col is not None:
        df2['_energy_mj'] = pd.to_numeric(df2[energy_col], errors='coerce')
    else:
        df2['_energy_mj'] = np.nan

    # Require row/col/stress; energy optional for stratification
    valid = df2.dropna(subset=['Peak_Shock_Stress_GPa', '_row', '_col'])
    valid = valid[valid['Peak_Shock_Stress_GPa'] > 0]
    valid_energy = valid.dropna(subset=['_energy_mj']) if energy_col is not None else pd.DataFrame()

    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] PA_13: No valid rows with Peak_Shock_Stress_GPa, Flyer_Row, Flyer_Column.")
        return

    any_saved = False

    for mat in mats:
        sub_mat = valid[valid['_material'] == mat]
        if sub_mat.empty:
            continue

        sub_mat_energy = valid_energy[valid_energy['_material'] == mat]
        bands_with_data = []
        for E in _ENERGY_BANDS_MJ:
            if sub_mat_energy.empty:
                break
            in_band = (sub_mat_energy['_energy_mj'] >= E - _ENERGY_TOL_MJ) & (sub_mat_energy['_energy_mj'] <= E + _ENERGY_TOL_MJ)
            if in_band.sum() >= 1:
                bands_with_data.append((E, sub_mat_energy.loc[in_band].copy()))

        if not bands_with_data:
            # Fallback: one panel with all data for this material (no energy filter)
            bands_with_data = [(None, sub_mat)]

        npan = len(bands_with_data)
        ncols = min(3, npan)
        nrows = (npan + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if npan == 1:
            axes = np.array([axes])
        axes = np.atleast_1d(axes).flat

        vmin = sub_mat['Peak_Shock_Stress_GPa'].min()
        vmax = sub_mat['Peak_Shock_Stress_GPa'].max()

        for idx, (E, sub_band) in enumerate(bands_with_data):
            ax = axes[idx]
            sc = ax.scatter(sub_band['_col'], sub_band['_row'],
                            c=sub_band['Peak_Shock_Stress_GPa'],
                            cmap='viridis', s=80, edgecolors='black', linewidths=0.5,
                            vmin=vmin, vmax=vmax)
            ax.set_xlabel('Flyer Column', fontsize=FONT_AXIS)
            ax.set_ylabel('Flyer Row', fontsize=FONT_AXIS)
            if E is not None:
                ax.set_title(f'{E}±{_ENERGY_TOL_MJ} mJ (n={len(sub_band)})', fontsize=FONT_TITLE, fontweight='bold')
            else:
                ax.set_title(f'All energies (n={len(sub_band)})', fontsize=FONT_TITLE, fontweight='bold')
            ax.tick_params(labelsize=FONT_TICK)
            ax.invert_yaxis()
            plt.colorbar(sc, ax=ax, label='Shock Stress (GPa)', shrink=0.8)

        for j in range(len(bands_with_data), len(axes)):
            axes[j].set_visible(False)

        title = f'Shock Stress vs Row × Column — {mat}'
        if bands_with_data[0][0] is not None:
            title += f' (same laser energy ±{_ENERGY_TOL_MJ} mJ)'
        fig.suptitle(title, fontsize=FONT_TITLE, fontweight='bold', y=1.02)
        fig.tight_layout()
        safe_mat = mat.replace(' ', '_')
        save(fig, out_dir, f'PA_13_peak_stress_vs_row_column_{safe_mat}')
        any_saved = True

    # Always write one file with fixed name so "PA_13" is easy to find
    if any_saved:
        return
    # If no material had bands (should not happen after fallback), make one overview plot
    nmat = len(mats)
    ncols = min(3, nmat)
    nrows = (nmat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nmat == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    vmin, vmax = valid['Peak_Shock_Stress_GPa'].min(), valid['Peak_Shock_Stress_GPa'].max()
    for idx, mat in enumerate(mats):
        ax = axes.flat[idx]
        sub = valid[valid['_material'] == mat]
        if sub.empty:
            ax.set_visible(False)
            continue
        sc = ax.scatter(sub['_col'], sub['_row'], c=sub['Peak_Shock_Stress_GPa'],
                        cmap='viridis', s=80, edgecolors='black', linewidths=0.5, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Flyer Column', fontsize=FONT_AXIS)
        ax.set_ylabel('Flyer Row', fontsize=FONT_AXIS)
        ax.set_title(f'{mat} (n={len(sub)})', fontsize=FONT_TITLE, fontweight='bold')
        ax.tick_params(labelsize=FONT_TICK)
        ax.invert_yaxis()
        plt.colorbar(sc, ax=ax, label='Shock Stress (GPa)', shrink=0.8)
    for j in range(len(mats), nrows * ncols):
        axes.flat[j].set_visible(False)
    fig.suptitle('Shock Stress vs Flyer Row × Column', fontsize=FONT_TITLE, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, out_dir, 'PA_13_peak_stress_vs_flyer_row_column')


# ══════════════════════════════════════════════════════════════════════════════
#  Spatial 3-D maps — Spall & HEL (uses meta_data.csv)
# ══════════════════════════════════════════════════════════════════════════════
def plot_spatial_3d_maps(df, out_dir):
    """3-D spatial maps for Spall Strength and HEL, **separately per material**.

    Uses flyer row/column to build X/Y positions, and laser energy (preferred) or
    waveplate angle for the Z axis. For HEL maps, each Z plane is coloured by the
    mean HEL value on that plane (consistent with the HEL colourbar), while
    individual points retain their per-shot values.
    """
    composition_str = df.get('Composition', pd.Series(dtype=str))
    comp_pct = composition_str.str.extract(r'\(([^%]+)%\)', expand=False).astype(float)

    flyer_row = df.get('Flyer_Row', pd.Series(dtype=str))
    flyer_column = df.get('Flyer_Column', pd.Series(dtype=str))

    spall = df.get('Spall_Strength_GPa', pd.Series(dtype=float))
    hel = df.get('HEL_GPa', pd.Series(dtype=float))

    spall_numeric = pd.to_numeric(spall, errors='coerce')
    hel_numeric = pd.to_numeric(hel, errors='coerce')

    # Build spatial X/Y positions (mm) using shared helper from main analysis code
    START_X = 8.0
    START_Y = 8.0
    SPACING = 6.0
    Y_MIN, Y_MAX = 8.0, 32.0
    x_pos, y_pos = _build_xy_positions(
        df,
        flyer_row,
        flyer_column,
        comp_pct,
        START_X=START_X,
        START_Y=START_Y,
        SPACING=SPACING,
        Y_MIN=Y_MIN,
        Y_MAX=Y_MAX,
    )

    # Z-axis: prefer laser energy; if absent, fall back to waveplate angle
    laser_col = _find_laser_energy_col(df)
    if laser_col is not None:
        raw_energy = pd.to_numeric(df[laser_col], errors='coerce')
        wp_angles = (raw_energy / 100.0).round() * 100.0  # bin to nearest 100 mJ
        z_axis_label = 'Laser Energy (mJ)'
    else:
        wp_col = _find_waveplate_col(df)
        if wp_col is not None:
            wp_angles = pd.to_numeric(df[wp_col], errors='coerce')
            z_axis_label = 'Waveplate Angle (°)'
        else:
            print("  [skip] Spatial 3-D: no laser-energy or waveplate column found in meta_data.csv.")
            return

    # Work per material so plots are not mixed across alloys
    if '_material' not in df.columns:
        print("  [skip] Spatial 3-D: missing '_material' column; cannot separate materials.")
        return

    mats = sorted(df['_material'].dropna().unique())
    if not mats:
        print("  [skip] Spatial 3-D: no materials found.")
        return

    for mat in mats:
        mask_mat = (df['_material'] == mat)
        if not mask_mat.any():
            continue

        x_m = x_pos[mask_mat]
        y_m = y_pos[mask_mat]
        wp_m = wp_angles[mask_mat]
        spall_m = spall_numeric[mask_mat]
        hel_m = hel_numeric[mask_mat]

        # Z levels for this material only
        unique_wp_m = sorted(pd.to_numeric(wp_m, errors='coerce').dropna().unique())
        if len(unique_wp_m) <= 1:
            print(f"  [skip] Spatial 3-D: {mat} has single (or no) Z-axis level; not generating 3-D planes.")
            continue

        valid_spall_m = ~x_m.isna() & ~y_m.isna() & ~spall_m.isna()
        valid_hel_m = ~x_m.isna() & ~y_m.isna() & ~hel_m.isna()

        n_spall_m = int(valid_spall_m.sum())
        n_hel_m = int(valid_hel_m.sum())
        if n_spall_m == 0 and n_hel_m == 0:
            print(f"  [skip] Spatial 3-D: no valid Spall or HEL data for {mat}.")
            continue

        print(f"  Spatial 3-D ({mat}): valid points — Spall={n_spall_m}, HEL={n_hel_m}")
        safe_mat = str(mat).replace(' ', '_')

        # Spall 3-D map for this material
        if n_spall_m > 0:
            wp_spall = wp_m[valid_spall_m]
            unique_wp_spall = sorted(pd.to_numeric(wp_spall, errors='coerce').dropna().unique())
            if len(unique_wp_spall) > 1:
                _plot_3d_contour_surface(
                    output_df=df[mask_mat][valid_spall_m],
                    x_pos=x_m[valid_spall_m],
                    y_pos=y_m[valid_spall_m],
                    values=spall_m[valid_spall_m],
                    unique_angles=unique_wp_spall,
                    wp_angles=wp_spall,
                    value_label='Spall Strength (GPa)',
                    cmap_name='viridis',
                    title_str=f'Spall Strength — {mat} — Spatial 3-D\n({len(unique_wp_spall)} {z_axis_label} levels)',
                    out_path_base=os.path.join(out_dir, f"PA_spatial_3d_spall_{safe_mat}"),
                    z_label=z_axis_label,
                )

        # HEL 3-D map for this material
        if n_hel_m > 0:
            wp_hel = wp_m[valid_hel_m]
            unique_wp_hel = sorted(pd.to_numeric(wp_hel, errors='coerce').dropna().unique())
            if len(unique_wp_hel) > 1:
                _plot_3d_contour_surface(
                    output_df=df[mask_mat][valid_hel_m],
                    x_pos=x_m[valid_hel_m],
                    y_pos=y_m[valid_hel_m],
                    values=hel_m[valid_hel_m],
                    unique_angles=unique_wp_hel,
                    wp_angles=wp_hel,
                    value_label='HEL (GPa)',
                    cmap_name='plasma',
                    title_str=f'HEL — {mat} — Spatial 3-D\n({len(unique_wp_hel)} {z_axis_label} levels)',
                    out_path_base=os.path.join(out_dir, f"PA_spatial_3d_hel_{safe_mat}"),
                    z_label=z_axis_label,
                )


def plot_spatial_2d_mean_maps(df, out_dir):
    """2-D spatial contour maps from Z-collapsed mean surfaces, per material.

    Uses the same X/Y position mapping as the spatial 3-D routine, then averages
    repeated points at the same X/Y across all Z levels (laser energy or waveplate)
    to produce one representative surface per property.
    """
    composition_str = df.get('Composition', pd.Series(dtype=str))
    comp_pct = composition_str.str.extract(r'\(([^%]+)%\)', expand=False).astype(float)

    flyer_row = df.get('Flyer_Row', pd.Series(dtype=str))
    flyer_column = df.get('Flyer_Column', pd.Series(dtype=str))

    spall = pd.to_numeric(df.get('Spall_Strength_GPa', pd.Series(dtype=float)), errors='coerce')
    hel = pd.to_numeric(df.get('HEL_GPa', pd.Series(dtype=float)), errors='coerce')

    START_X = 8.0
    START_Y = 8.0
    SPACING = 6.0
    Y_MIN, Y_MAX = 8.0, 32.0
    x_pos, y_pos = _build_xy_positions(
        df,
        flyer_row,
        flyer_column,
        comp_pct,
        START_X=START_X,
        START_Y=START_Y,
        SPACING=SPACING,
        Y_MIN=Y_MIN,
        Y_MAX=Y_MAX,
    )

    if '_material' not in df.columns:
        print("  [skip] Spatial 2-D mean: missing '_material' column; cannot separate materials.")
        return

    mats = sorted(df['_material'].dropna().unique())
    if not mats:
        print("  [skip] Spatial 2-D mean: no materials found.")
        return

    def _plot_one_property(mat_df, value_col, cmap_name, value_label, out_stub):
        sub = mat_df[['x_mm', 'y_mm', value_col]].dropna()
        if sub.empty:
            return False

        # Collapse all Z planes by averaging repeated X/Y locations
        avg_xy = (
            sub.groupby(['x_mm', 'y_mm'], as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: 'mean_value'})
        )
        if len(avg_xy) < 3:
            return False

        x = avg_xy['x_mm'].to_numpy(dtype=float)
        y = avg_xy['y_mm'].to_numpy(dtype=float)
        z = avg_xy['mean_value'].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        contour_done = False
        try:
            # Filled contour on irregular x/y positions
            tcf = ax.tricontourf(x, y, z, levels=14, cmap=cmap_name)
            ax.tricontour(x, y, z, levels=14, colors='k', linewidths=0.45, alpha=0.35)
            cbar = plt.colorbar(tcf, ax=ax, shrink=0.9)
            cbar.set_label(f'Mean {value_label}', fontsize=FONT_AXIS)
            contour_done = True
        except Exception:
            contour_done = False

        # Always overlay averaged points; used as fallback when contour fails
        sc = ax.scatter(
            x, y, c=z, cmap=cmap_name, s=85, edgecolors='black', linewidths=0.6, zorder=3
        )
        if not contour_done:
            cbar = plt.colorbar(sc, ax=ax, shrink=0.9)
            cbar.set_label(f'Mean {value_label}', fontsize=FONT_AXIS)

        ax.set_xlabel('Flyer X position (mm)', fontsize=FONT_AXIS)
        ax.set_ylabel('Flyer Y position (mm)', fontsize=FONT_AXIS)
        ax.set_title(
            f'Spatial 2-D Mean Map — {value_label}\n{mat_df["_material"].iloc[0]} (averaged across all Z levels)',
            fontsize=FONT_TITLE,
            fontweight='bold',
        )
        ax.tick_params(labelsize=FONT_TICK)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.20, linestyle='--', linewidth=0.5)
        fig.tight_layout()
        save(fig, out_dir, out_stub)
        return True

    for mat in mats:
        mask_mat = df['_material'] == mat
        if not mask_mat.any():
            continue

        mat_df = pd.DataFrame({
            '_material': df.loc[mask_mat, '_material'],
            'x_mm': x_pos[mask_mat],
            'y_mm': y_pos[mask_mat],
            'Spall_Strength_GPa': spall[mask_mat],
            'HEL_GPa': hel[mask_mat],
        })
        safe_mat = str(mat).replace(' ', '_')

        made_spall = _plot_one_property(
            mat_df=mat_df,
            value_col='Spall_Strength_GPa',
            cmap_name='viridis',
            value_label='Spall Strength (GPa)',
            out_stub=f'PA_spatial_2d_mean_spall_{safe_mat}',
        )
        made_hel = _plot_one_property(
            mat_df=mat_df,
            value_col='HEL_GPa',
            cmap_name='plasma',
            value_label='HEL (GPa)',
            out_stub=f'PA_spatial_2d_mean_hel_{safe_mat}',
        )

        if not made_spall and not made_hel:
            print(f"  [skip] Spatial 2-D mean: no valid HEL/Spall XY data for {mat}.")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 14 — PDV Return Power vs HEL Uncertainty  (scatter, per material)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pdv_return_vs_hel_uncertainty(df, out_dir):
    """Scatter: PDV Return Power (dBm) vs HEL Uncertainty (GPa), coloured by material."""
    pdv_col = _get_pdv_return_power_col(df)
    hel_unc_col = _get_hel_unc_col(df)
    if pdv_col is None:
        print("  [skip] PA_14: Missing PDV Return Power column. Available:", list(df.columns)[:20], "...")
        return
    if hel_unc_col is None:
        print("  [skip] PA_14: Missing HEL uncertainty column. Available:", list(df.columns)[:20], "...")
        return

    df2 = df.copy()
    df2['_pdv'] = pd.to_numeric(df2[pdv_col], errors='coerce')
    df2['_hel_unc'] = pd.to_numeric(df2[hel_unc_col], errors='coerce')
    valid = df2.dropna(subset=['_pdv', '_hel_unc'])
    valid = valid[valid['_hel_unc'] >= 0]  # keep zero; drop only negative if any

    # 3σ outlier removal per material: keep only points within 3 std of mean on both axes
    keep = np.zeros(len(valid), dtype=bool)
    for mat in valid['_material'].unique():
        sub = valid['_material'] == mat
        x = valid.loc[sub, '_pdv'].values.astype(float)
        y = valid.loc[sub, '_hel_unc'].values.astype(float)
        mx, sx = np.mean(x), np.std(x)
        my, sy = np.mean(y), np.std(y)
        if sx <= 0:
            sx = 1e-30
        if sy <= 0:
            sy = 1e-30
        in_x = np.abs(x - mx) <= 3.0 * sx
        in_y = np.abs(y - my) <= 3.0 * sy
        keep[sub] = in_x & in_y
    valid = valid[keep].reset_index(drop=True)

    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] PA_14: No valid rows with PDV Return Power and HEL Uncertainty (after 3σ filter).")
        return

    # Normalize y (HEL uncertainty) by global maximum
    y_max = valid['_hel_unc'].max()
    valid = valid.copy()
    valid['_hel_unc_norm'] = valid['_hel_unc'] / y_max if y_max > 0 else valid['_hel_unc']

    window = 15
    fig, ax = plt.subplots(figsize=(9, 7))
    for mat in mats:
        sub = valid[valid['_material'] == mat].copy()
        sub = sub.sort_values('_pdv').reset_index(drop=True)
        if len(sub) < window:
            ax.scatter(sub['_pdv'], sub['_hel_unc_norm'],
                      c=colours[mat], marker=mkrs[mat], s=65,
                      edgecolors='black', linewidths=0.5,
                      alpha=0.8, label=f'{mat}  (n={len(sub)})')
            continue
        # 5-point moving average and 1σ std (centered window)
        sub['_roll_mean'] = sub['_hel_unc_norm'].rolling(window, center=True).mean()
        sub['_roll_std'] = sub['_hel_unc_norm'].rolling(window, center=True).std(ddof=1)
        x = sub['_pdv'].values
        y_mean = sub['_roll_mean'].values
        y_std = sub['_roll_std'].values
        valid_mask = np.isfinite(y_mean) & np.isfinite(y_std)
        x_plot = x[valid_mask]
        y_lo = (y_mean - y_std)[valid_mask]
        y_hi = (y_mean + y_std)[valid_mask]
        ax.scatter(sub['_pdv'], sub['_hel_unc_norm'],
                  c=colours[mat], marker=mkrs[mat], s=65,
                  edgecolors='black', linewidths=0.5,
                  alpha=0.8, label=f'{mat}  (n={len(sub)})')
        if len(x_plot) > 0:
            ax.fill_between(x_plot, y_lo, y_hi, color=colours[mat], alpha=0.25)
            ax.plot(x_plot, y_mean[valid_mask], color=colours[mat], linewidth=2, zorder=5)

    ax.set_xlabel('PDV Return Power  (dBm)', fontsize=FONT_AXIS)
    ax.set_ylabel('HEL Uncertainty  (normalized by max)', fontsize=FONT_AXIS)
    ax.set_title('HEL Uncertainty vs PDV Return Power\n(per material, 3σ removed, 15-pt moving avg ± 1σ)', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, 'PA_14_pdv_return_power_vs_hel_uncertainty')


# ══════════════════════════════════════════════════════════════════════════════
#  TENSILE-ONLY PLOTS (no HEL, no compressive strain rate; use tensile = line 3 pullback slope)
#  Same script, extra plots with prefix PA_tensile_*
# ══════════════════════════════════════════════════════════════════════════════

def plot_tensile_stress_vs_tensile_strainrate_scatter(df, out_dir):
    """Shock Stress vs Tensile Strain Rate (Spall_StrainRate_s^-1 = line 3 pullback slope)."""
    valid = df.dropna(subset=['Peak_Shock_Stress_GPa', 'Spall_StrainRate_s^-1'])
    valid = valid[valid['Spall_StrainRate_s^-1'] > 0]
    valid = valid[valid['Peak_Shock_Stress_GPa'] > 0]
    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] No valid data for stress vs tensile strain rate scatter.")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    cu_mats = [m for m in mats if m.strip().lower().startswith('cu')]
    non_cu = [m for m in mats if not m.strip().lower().startswith('cu')]
    for mat in (non_cu + cu_mats):
        sub = valid[valid['_material'] == mat]
        alpha = 0.35 if mat.strip().lower().startswith('cu') else 0.85
        ax.scatter(sub['Spall_StrainRate_s^-1'], sub['Peak_Shock_Stress_GPa'],
                   c=colours[mat], marker=mkrs[mat], s=60, edgecolors='black', linewidths=0.5,
                   alpha=alpha, label=f'{mat}  (n={len(sub)})', zorder=3)
    ax.set_xlabel('Tensile Strain Rate  (s$^{-1}$)\n(line 3 pullback slope)', fontsize=FONT_AXIS)
    ax.set_ylabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Shock Stress vs Tensile Strain Rate', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_01_stress_vs_tensile_strainrate_scatter')


def plot_tensile_stress_vs_tensile_strainrate_bubble(df, out_dir):
    """Shock Stress vs Tensile Strain Rate, bubble = Laser Energy."""
    df2 = df.copy()
    for col in ['Peak_Shock_Stress_GPa', 'Spall_StrainRate_s^-1', 'Laser_Target_Energy (mJ)']:
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
    valid = df2.dropna(subset=['Peak_Shock_Stress_GPa', 'Spall_StrainRate_s^-1', 'Laser_Target_Energy (mJ)'])
    valid = valid[valid['Spall_StrainRate_s^-1'] > 0]
    valid = valid[valid['Peak_Shock_Stress_GPa'] > 0].reset_index(drop=True)
    mats, colours, mkrs = material_groups(valid)
    if not mats:
        print("  [skip] No valid data for stress vs tensile strain rate bubble.")
        return
    energy = pd.to_numeric(valid['Laser_Target_Energy (mJ)'], errors='coerce').fillna(0).clip(lower=1).values
    e_min, e_max = float(energy.min()), float(energy.max())
    sizes = np.clip(20 + 280 * (energy - e_min) / max(e_max - e_min, 1), 5, 400)
    xv = valid['Spall_StrainRate_s^-1'].values.astype(float)
    yv = valid['Peak_Shock_Stress_GPa'].values.astype(float)
    mv = valid['_material'].values
    fig, ax = plt.subplots(figsize=(10, 7))
    for mat in mats:
        idx = mv == mat
        ax.scatter(xv[idx], yv[idx], s=sizes[idx], c=colours[mat], marker=mkrs[mat],
                   edgecolors='black', linewidths=0.4, alpha=0.7, label=mat, zorder=3)
    size_handles = []
    for e_val, lbl in [(400, '400 mJ'), (900, '900 mJ'), (1600, '1600 mJ')]:
        r_pt = np.sqrt(20 + 280 * (e_val - e_min) / max(e_max - e_min, 1)) / 2
        size_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='black',
                   markersize=max(r_pt * 0.5, 4), alpha=0.6, label=lbl))
    mat_handles = [Line2D([0], [0], marker=mkrs[m], color='w', markerfacecolor=colours[m],
                          markeredgecolor='black', markersize=8, label=m) for m in mats]
    leg1 = ax.legend(handles=mat_handles, title='Material', loc='upper left', fontsize=FONT_LEG, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=size_handles, title='Laser Energy', loc='lower right', fontsize=FONT_LEG, framealpha=0.9)
    ax.set_xlabel('Tensile Strain Rate  (s$^{-1}$)\n(line 3 pullback slope)', fontsize=FONT_AXIS)
    ax.set_ylabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Shock Stress vs Tensile Strain Rate\n(bubble area ∝ Laser Energy)', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_02_stress_vs_tensile_strainrate_bubble')


def plot_tensile_correlogram(df, out_dir):
    """Correlogram: Tensile Strain Rate, Spall Strength, Shock Stress, Laser Energy (no HEL)."""
    _VARS = {
        'Tensile Strain\nRate (s⁻¹)':  'Spall_StrainRate_s^-1',
        'Spall\nStrength (GPa)':       'Spall_Strength_GPa',
        'Peak Shock\nStress (GPa)':    'Peak_Shock_Stress_GPa',
        'Laser Energy\n(mJ)':          'Laser_Target_Energy (mJ)',
    }
    labels   = list(_VARS.keys())
    col_keys = list(_VARS.values())

    sub = df[col_keys].apply(pd.to_numeric, errors='coerce')
    sub['_material'] = df['_material']
    sub = sub[sub[col_keys].notna().sum(axis=1) >= 2]
    sub['_material'] = sub['_material'].astype(str)

    mats, colours, mkrs = material_groups(sub)
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'

    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(3 * n + 2, 3 * n))

    for r in range(n):
        for c in range(n):
            ax = axes[r][c]
            ck_r, ck_c = col_keys[r], col_keys[c]
            if r == c:
                all_vals = pd.to_numeric(sub[ck_r], errors='coerce').dropna()
                if len(all_vals) < 2:
                    ax.set_visible(False)
                    continue
                for mat in mats:
                    vals = pd.to_numeric(sub.loc[sub['_material'] == mat, ck_r], errors='coerce').dropna()
                    if len(vals) < 3:
                        continue
                    try:
                        kde = gaussian_kde(vals, bw_method=0.4)
                        vr = np.linspace(all_vals.min(), all_vals.max(), 300)
                        ax.plot(vr, kde(vr), color=colours[mat], lw=1.5)
                        ax.fill_between(vr, kde(vr), alpha=0.2, color=colours[mat])
                    except Exception:
                        pass
                ax.set_xlabel(labels[r], fontsize=FONT_AXIS)
                ax.set_yticks([])
                ax.set_ylabel('Probability Density', fontsize=FONT_AXIS)
            else:
                for mat in mats:
                    mv = sub['_material'] == mat
                    xv = pd.to_numeric(sub.loc[mv, ck_c], errors='coerce')
                    yv = pd.to_numeric(sub.loc[mv, ck_r], errors='coerce')
                    ok = xv.notna() & yv.notna()
                    if ok.sum() < 1:
                        continue
                    ax.scatter(xv[ok], yv[ok], c=colours[mat], marker=mkrs[mat],
                               s=9.4, alpha=0.5, edgecolors='none')

            # Uniform tick size across all panels
            ax.tick_params(labelsize=FONT_TICK)

            # Scientific notation for tensile strain rate axis
            if ck_c == 'Spall_StrainRate_s^-1':
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            if r != c and ck_r == 'Spall_StrainRate_s^-1':
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            if c == 0 and r != c:
                ax.set_ylabel(labels[r], fontsize=FONT_AXIS)
            if r == n - 1:
                ax.set_xlabel(labels[c], fontsize=FONT_AXIS)

    counts = sub['_material'].value_counts()
    handles = [Line2D([0], [0], marker=mkrs[m], color='w',
                      markerfacecolor=colours[m], markersize=15.6,
                      markeredgecolor='black', label=f'{m} (n={counts.get(m, 0)})')
               for m in mats]
    fig.legend(handles=handles, loc='upper left',
               bbox_to_anchor=(0.92, 1.0), bbox_transform=fig.transFigure,
               fontsize=13, framealpha=0.9, title='Material')
    fig.suptitle('Pairwise Scatter Matrix — Tensile / Spall (no HEL)',
                 fontsize=FONT_TITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    save(fig, out_dir, 'PA_tensile_03_correlogram')


def plot_tensile_spall_vs_peak_stress(df, out_dir):
    """Spall Strength vs Shock Stress (tensile-only variant; no HEL)."""
    sp = df[df['Spall_OK'] == True].copy()
    sp['Spall_Strength_GPa'] = pd.to_numeric(sp['Spall_Strength_GPa'], errors='coerce')
    sp['Peak_Shock_Stress_GPa'] = pd.to_numeric(sp['Peak_Shock_Stress_GPa'], errors='coerce')
    sp = sp.dropna(subset=['Spall_Strength_GPa', 'Peak_Shock_Stress_GPa'])
    sp = sp[(sp['Spall_Strength_GPa'] > 0) & (sp['Peak_Shock_Stress_GPa'] > 0)]
    mats, colours, mkrs = material_groups(sp)
    if not mats:
        print("  [skip] No data for Spall vs Peak Stress (tensile).")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for mat in mats:
        sub = sp[sp['_material'] == mat]
        ax.scatter(sub['Peak_Shock_Stress_GPa'], sub['Spall_Strength_GPa'],
                   c=colours[mat], marker=mkrs[mat], s=65, edgecolors='black', linewidths=0.5,
                   alpha=0.8, label=f'{mat}  (n={len(sub)})')
    ax.set_xlabel('Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_ylabel('Spall Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Spall Strength vs Shock Stress\n(tensile/spall only)', fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_04_spall_vs_peak_stress')


def plot_tensile_spall_vs_strainrate_stress_bins(df, out_dir):
    """Spall Strength vs Tensile Strain Rate; fill color = Prior Shock Stress (continuous colorbar); edge = material."""
    VMIN, VMAX = 1.5, 8.0
    _cmap = plt.cm.get_cmap('plasma')

    required = ['Spall_StrainRate_s^-1', 'Spall_Strength_GPa', 'Peak_Shock_Stress_GPa']
    sp = df[df['Spall_OK'] == True].copy() if 'Spall_OK' in df.columns else df.copy()
    for col in required:
        if col in sp.columns:
            sp[col] = pd.to_numeric(sp[col], errors='coerce')
    sp['Peak_Shock_Stress_GPa'] = pd.to_numeric(sp.get('Peak_Shock_Stress_GPa', np.nan), errors='coerce')
    valid = sp.dropna(subset=required)
    valid = valid[(valid['Spall_StrainRate_s^-1'] > 0) &
                  (valid['Spall_Strength_GPa'] > 0) &
                  (valid['Peak_Shock_Stress_GPa'] >= 1.5)].copy()
    if len(valid) == 0:
        print("  [skip] PA_tensile_05: no valid data.")
        return

    mats, colours, mkrs = material_groups(valid)
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'

    norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
    S_MIN, S_MAX = 30, 300  # marker area range (points²)

    def _marker_size(stress_vals):
        """Scale marker area linearly with Prior Shock Stress."""
        s = np.clip(stress_vals.values, VMIN, VMAX)
        return S_MIN + (s - VMIN) / (VMAX - VMIN) * (S_MAX - S_MIN)

    MAT_EDGE_WIDTH = 1.8

    fig, ax = plt.subplots(figsize=(10, 7))

    cu_mats = [m for m in mats if m.strip().lower().startswith('cu')]
    non_cu  = [m for m in mats if not m.strip().lower().startswith('cu')]
    for mat in non_cu + cu_mats:
        mat_mask = valid['_material'] == mat
        edge_col = colours[mat]
        alpha = 0.55 if mat.strip().lower().startswith('cu') else 0.90
        subset = valid[mat_mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset['Spall_StrainRate_s^-1'], subset['Spall_Strength_GPa'],
                   c=subset['Peak_Shock_Stress_GPa'], cmap=_cmap, norm=norm,
                   marker='o', s=_marker_size(subset['Peak_Shock_Stress_GPa']),
                   edgecolors=edge_col, linewidths=MAT_EDGE_WIDTH,
                   alpha=alpha, zorder=3)

    # ── Colorbar: Prior Shock Stress ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Prior Shock Stress  (GPa)', fontsize=FONT_AXIS)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    # ── Legend: Material (edge color) + size reference ───────────────────────
    _MAT_DISPLAY = {'Al': '1100 Al', 'Cu': '110 Cu',
                    'Cu 1mm': '110 Cu 1mm', 'Cu 0.5mm': '110 Cu 0.5mm'}
    mat_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightgrey', markeredgecolor=colours[m],
               markeredgewidth=MAT_EDGE_WIDTH, markersize=12,
               label=_MAT_DISPLAY.get(m, m))
        for m in mats
    ]
    size_ref_vals = [2, 4, 6, 8]
    size_ref_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=np.sqrt(S_MIN + (v - VMIN) / (VMAX - VMIN) * (S_MAX - S_MIN)),
               label=f'{v} GPa')
        for v in size_ref_vals
    ]
    leg1 = ax.legend(handles=mat_handles, title='Material',
                     loc='upper left', fontsize=FONT_LEG, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=size_ref_handles, title='Prior Shock Stress',
                     loc='lower right', fontsize=FONT_LEG, framealpha=0.9)

    ax.set_xlabel('Tensile Strain Rate  (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('Spall Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Spall Strength vs Tensile Strain Rate\n(fill color = Prior Shock Stress; edge = material)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_05_spall_vs_strainrate_stress_bins')


def plot_tensile_spall_vs_stress_strainrate_bins(df, out_dir):
    """Spall Strength vs Prior Shock Stress; fill color = Tensile Strain Rate (continuous colorbar); size ∝ strain rate; edge = material."""
    VMIN, VMAX = 0.5e6, 2.0e6
    _cmap = plt.cm.get_cmap('viridis')

    required = ['Spall_StrainRate_s^-1', 'Spall_Strength_GPa', 'Peak_Shock_Stress_GPa']
    sp = df[df['Spall_OK'] == True].copy() if 'Spall_OK' in df.columns else df.copy()
    for col in required:
        if col in sp.columns:
            sp[col] = pd.to_numeric(sp[col], errors='coerce')
    valid = sp.dropna(subset=required)
    valid = valid[
        (valid['Spall_StrainRate_s^-1'] >= VMIN) &
        (valid['Spall_Strength_GPa'] > 0) &
        (valid['Peak_Shock_Stress_GPa'] >= 1.5)
    ].copy()
    if len(valid) == 0:
        print("  [skip] PA_tensile_06: no valid data.")
        return

    mats, colours, mkrs = material_groups(valid)
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'

    norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
    S_MIN, S_MAX = 30, 300

    def _marker_size(sr_vals):
        s = np.clip(sr_vals.values, VMIN, VMAX)
        return S_MIN + (s - VMIN) / (VMAX - VMIN) * (S_MAX - S_MIN)

    MAT_EDGE_WIDTH = 1.8

    fig, ax = plt.subplots(figsize=(10, 7))

    cu_mats = [m for m in mats if m.strip().lower().startswith('cu')]
    non_cu  = [m for m in mats if not m.strip().lower().startswith('cu')]
    for mat in non_cu + cu_mats:
        mat_mask = valid['_material'] == mat
        edge_col = colours[mat]
        alpha = 0.55 if mat.strip().lower().startswith('cu') else 0.90
        subset = valid[mat_mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset['Peak_Shock_Stress_GPa'], subset['Spall_Strength_GPa'],
                   c=subset['Spall_StrainRate_s^-1'], cmap=_cmap, norm=norm,
                   marker='o', s=_marker_size(subset['Spall_StrainRate_s^-1']),
                   edgecolors=edge_col, linewidths=MAT_EDGE_WIDTH,
                   alpha=alpha, zorder=3)

    # ── Colorbar: Tensile Strain Rate ─────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Tensile Strain Rate  (s$^{-1}$)', fontsize=FONT_AXIS)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.formatter = plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}×$10^6$')
    cbar.ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}×10⁶'))
    cbar.update_ticks()

    # ── Legend: Material (edge color) + size reference ───────────────────────
    _MAT_DISPLAY = {'Al': '1100 Al', 'Cu': '110 Cu',
                    'Cu 1mm': '110 Cu 1mm', 'Cu 0.5mm': '110 Cu 0.5mm'}
    mat_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightgrey', markeredgecolor=colours[m],
               markeredgewidth=MAT_EDGE_WIDTH, markersize=12,
               label=_MAT_DISPLAY.get(m, m))
        for m in mats
    ]
    size_ref_vals = [0.5e6, 1e6, 2e6]
    size_ref_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=np.sqrt(S_MIN + (v - VMIN) / (VMAX - VMIN) * (S_MAX - S_MIN)),
               label=f'${v/1e6:g}\\times10^6$ s$^{{-1}}$')
        for v in size_ref_vals
    ]
    leg1 = ax.legend(handles=mat_handles, title='Material',
                     loc='upper left', fontsize=FONT_LEG, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=size_ref_handles, title='Tensile Strain Rate',
              loc='lower right', fontsize=FONT_LEG, framealpha=0.9)

    ax.set_xlabel('Prior Shock Stress  (GPa)', fontsize=FONT_AXIS)
    ax.set_ylabel('Spall Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Spall Strength vs Prior Shock Stress\n(fill color = Tensile Strain Rate; edge = material)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_06_spall_vs_stress_strainrate_bins')


def plot_tensile_spall_strainrate_linear_fit(df, out_dir):
    """Spall Strength vs Tensile Strain Rate with per-material linear fits to compare rate dependence."""
    _MAT_DISPLAY = {'Al': '1100 Al', 'Cu': '110 Cu',
                    'Cu 1mm': '110 Cu 1mm', 'Cu 0.5mm': '110 Cu 0.5mm'}
    VMIN, VMAX = 1.5, 8.0
    _cmap = plt.cm.get_cmap('plasma')

    required = ['Spall_StrainRate_s^-1', 'Spall_Strength_GPa', 'Peak_Shock_Stress_GPa']
    sp = df[df['Spall_OK'] == True].copy() if 'Spall_OK' in df.columns else df.copy()
    for col in required:
        if col in sp.columns:
            sp[col] = pd.to_numeric(sp[col], errors='coerce')
    valid = sp.dropna(subset=required)
    valid = valid[
        (valid['Spall_StrainRate_s^-1'] > 0) &
        (valid['Spall_Strength_GPa'] > 0) &
        (valid['Peak_Shock_Stress_GPa'] >= 1.5)
    ].copy()
    if len(valid) == 0:
        print("  [skip] PA_tensile_07: no valid data.")
        return

    mats, colours, mkrs = material_groups(valid)
    for m in mats:
        if str(m).strip().lower() == 'zn':
            colours[m] = '#1B5E20'

    norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
    S_MIN, S_MAX = 30, 300

    def _marker_size(stress_vals):
        s = np.clip(stress_vals.values, VMIN, VMAX)
        return S_MIN + (s - VMIN) / (VMAX - VMIN) * (S_MAX - S_MIN)

    MAT_EDGE_WIDTH = 1.8
    FIT_COLORS = {'Al': '#1565C0', 'Cu': '#B71C1C',
                  'Cu 1mm': '#B71C1C', 'Cu 0.5mm': '#F4511E'}

    fig, ax = plt.subplots(figsize=(10, 7))

    cu_mats = [m for m in mats if m.strip().lower().startswith('cu')]
    non_cu  = [m for m in mats if not m.strip().lower().startswith('cu')]
    for mat in non_cu + cu_mats:
        mat_mask = valid['_material'] == mat
        edge_col = colours[mat]
        alpha = 0.45 if mat.strip().lower().startswith('cu') else 0.75
        subset = valid[mat_mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset['Spall_StrainRate_s^-1'], subset['Spall_Strength_GPa'],
                   c=subset['Peak_Shock_Stress_GPa'], cmap=_cmap, norm=norm,
                   marker='o', s=_marker_size(subset['Peak_Shock_Stress_GPa']),
                   edgecolors=edge_col, linewidths=MAT_EDGE_WIDTH,
                   alpha=alpha, zorder=3)

    # ── Per-material linear fits ──────────────────────────────────────────────
    fit_handles = []
    for mat in mats:
        subset = valid[valid['_material'] == mat]
        if len(subset) < 3:
            continue
        x = subset['Spall_StrainRate_s^-1'].values
        y = subset['Spall_Strength_GPa'].values
        slope, intercept, r, _, _ = linregress(x, y)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = slope * x_fit + intercept
        fc = FIT_COLORS.get(mat, colours[mat])
        sign = '+' if intercept >= 0 else '-'
        label = (f'{_MAT_DISPLAY.get(mat, mat)}  '
                 f'slope={slope*1e6:.3f} GPa/(10⁶ s⁻¹)\n'
                 f'  y={slope:.3e}x {sign} {abs(intercept):.2f},  R²={r**2:.3f}')
        line, = ax.plot(x_fit, y_fit, color=fc, lw=2.5, zorder=5, label=label)
        fit_handles.append(line)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Prior Shock Stress  (GPa)', fontsize=FONT_AXIS)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    # ── Legend: material markers ──────────────────────────────────────────────
    mat_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='lightgrey', markeredgecolor=colours[m],
               markeredgewidth=MAT_EDGE_WIDTH, markersize=12,
               label=_MAT_DISPLAY.get(m, m))
        for m in mats
    ]
    leg1 = ax.legend(handles=mat_handles, title='Material',
                     loc='upper left', fontsize=FONT_LEG, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=fit_handles, title='Linear Fit',
              loc='lower right', fontsize=FONT_LEG, framealpha=0.9)

    ax.set_xlabel('Tensile Strain Rate  (s$^{-1}$)', fontsize=FONT_AXIS)
    ax.set_ylabel('Spall Strength  (GPa)', fontsize=FONT_AXIS)
    ax.set_title('Spall Strength vs Tensile Strain Rate — Linear Fit Comparison\n'
                 '(fill color = Prior Shock Stress; edge = material)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, out_dir, 'PA_tensile_07_spall_strainrate_linear_fit')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='HELIX Toolbox v2 post-analysis plots (standalone)')
    parser.add_argument('--config', default=os.path.join(
        os.path.dirname(__file__), 'helix_master_config.yml'),
        help='Path to helix_master_config.yml (or .json)')
    parser.add_argument('--spade-dir', default=None,
        help='SPADE_analysis directory to read the master summary from and write plots '
             'into (overrides the directory derived from the config)')
    parser.add_argument('--csv', default=None,
        help='Explicit path to a consolidated *Data_Summary.csv (overrides discovery)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve the SPADE_analysis directory: explicit --spade-dir wins, else the
    # post_processing_config.spade_output_dir if set to a real path, else
    # <cli_settings.output_dir>/SPADE_analysis.
    if args.csv:
        csv_path = args.csv
    else:
        if args.spade_dir:
            spade_dir = args.spade_dir
        else:
            ppc = (cfg.get('post_processing_config') or {})
            spade_dir = ppc.get('spade_output_dir')
            if not spade_dir or str(spade_dir).startswith('/path/to'):
                out_dir = (cfg.get('cli_settings') or {}).get('output_dir', '')
                spade_dir = os.path.join(out_dir, 'SPADE_analysis')
        csv_path = find_summary_csv(spade_dir)

    # Plots are written next to the master summary (the SPADE_analysis directory).
    out_dir = os.path.dirname(os.path.abspath(csv_path))

    print(f"\nLoading:  {csv_path}")
    # Read bytes first so pandas never holds an open handle on the OneDrive path.
    # OneDrive's sync lock blocks open() calls on existing files; reading into
    # BytesIO first completes the I/O in one shot and parses from memory.
    with open(csv_path, 'rb') as _f:
        _raw = _f.read()
    df = pd.read_csv(io.BytesIO(_raw))
    print(f"Shape:    {df.shape[0]} rows × {df.shape[1]} columns")

    # Normalise material column to '_material' for internal use
    mat_col = next((c for c in df.columns
                    if c.lower() in ('sample material', 'material', 'composition')), None)
    if mat_col:
        df['_material'] = df[mat_col].astype(str).str.strip()
    elif 'Composition' in df.columns:
        df['_material'] = df['Composition'].astype(str).str.strip()
    else:
        df['_material'] = 'Unknown'

    df = split_material_by_thickness(df)

    # Cast boolean columns
    for col in ('HEL_OK', 'Spall_OK'):
        if col in df.columns:
            df[col] = df[col].map({True: True, False: False,
                                   'True': True, 'False': False,
                                   1: True, 0: False})

    print(f"\nMaterials: {df['_material'].value_counts().to_dict()}")
    print(f"Output:    {out_dir}\n")

    plastic_sr_methods = build_plastic_sr_methods(cfg)
    print("Plastic strain-rate methods (PA_02e/02f):")
    for col, desc, tag in plastic_sr_methods:
        print(f"  {tag}: {desc}  [{col}]")

    fns = [
        plot_stress_vs_strainrate_scatter,
        plot_stress_vs_strainrate_bubble,
        plot_peak_fsv_vs_risetime_scatter,
        plot_grady_loglog_fit,
        plot_grady_loglog_fit_swapped_axes,
        *[partial(plot_grady_loglog_plastic_strainrate, sr_col=col, method_desc=desc, file_tag=tag)
          for col, desc, tag in plastic_sr_methods],
        *[partial(plot_grady_loglog_plastic_strainrate_flipped, sr_col=col, method_desc=desc, file_tag=tag)
          for col, desc, tag in plastic_sr_methods],
        partial(plot_risetime_vs_stress_loglog, pct_label=plastic_sr_methods[0][2].replace('_', '–')),
        partial(plot_risetime_vs_stress_loglog_flipped, pct_label=plastic_sr_methods[0][2].replace('_', '–')),
        plot_hel_violin,
        plot_spall_violin,
        plot_hel_ridgeline,
        plot_detection_lollipop,
        plot_correlogram,
        plot_correlogram_stress,
        plot_correlogram_stress_top2,
        plot_hel_vs_peak_stress,
        plot_hel_vs_hel_strain_rate,
        plot_hel_vs_hel_strain_rate_regression,
        plot_hel_vs_hel_strain_rate_mean,
        plot_hel_vs_hel_strain_rate_mean_rolling,
        plot_hel_vs_hel_strain_rate_faceted,
        plot_peak_stress_vs_flyer_row_column,
        plot_pdv_return_vs_hel_uncertainty,
        plot_spatial_3d_maps,
        plot_spatial_2d_mean_maps,
    ]
    for fn in fns:
        fn_name = fn.func.__name__ if isinstance(fn, partial) else fn.__name__
        extra = f" ({fn.keywords.get('file_tag')})" if isinstance(fn, partial) and 'file_tag' in fn.keywords else ""
        print(f"[{fn_name}{extra}]")
        try:
            fn(df, out_dir)
        except Exception as exc:
            print(f"  ✗  {exc}")

    # Tensile-only plots (no HEL, no compressive strain rate; tensile = line 3 pullback slope)
    print("\n--- Tensile-only plots (PA_tensile_*) ---")
    tensile_fns = [
        plot_tensile_stress_vs_tensile_strainrate_scatter,
        plot_tensile_stress_vs_tensile_strainrate_bubble,
        plot_tensile_correlogram,
        plot_tensile_spall_vs_peak_stress,
        plot_tensile_spall_vs_strainrate_stress_bins,
        plot_tensile_spall_vs_stress_strainrate_bins,
        plot_tensile_spall_strainrate_linear_fit,
    ]
    for fn in tensile_fns:
        print(f"[{fn.__name__}]")
        try:
            fn(df, out_dir)
        except Exception as exc:
            print(f"  ✗  {exc}")

    print("\nDone.")


if __name__ == '__main__':
    main()
