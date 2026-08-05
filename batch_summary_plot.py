#!/usr/bin/env python3
"""
Aggregate batch SPADE results and produce combined Spall and HEL strength plots.

Reads helix_master_config_batch_process.json to locate the batch input directory,
collects enhanced_spall_summary.csv and velocity_shots_summary.csv from each
subfolder's SPADE_analysis output, then saves a two-panel figure to input_dir:
  Left panel  — Spall Strength (GPa) vs Spall Strain Rate (s⁻¹)
  Right panel — HEL Strength (GPa)   vs HEL Strain Rate   (s⁻¹)
Points are colour-coded by material; error bars are included where available.
Also saves a violin-plot figure (per material) of Spall Strength and HEL
Strength distributions; Spall Strength points are colour-coded by Peak Shock
Stress (GPa).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_CONFIG = os.path.join(REPO_ROOT, "helix_master_config_batch_process.json")


# ── Config ────────────────────────────────────────────────────────────────────

def _load_batch_settings():
    with open(BATCH_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cs = cfg["cli_settings"]
    return {
        "input_dir":         cs["input_dir"],
        "subfolder_pattern": cs.get("subfolder_pattern", "*"),
        "output_subdir":     cs.get("output_subdir_name", "Output"),
        "combined_output":   cs.get("combined_output_dir") or None,
        "igsn_material_map": cfg.get("igsn_material_map", {}),
    }


# ── Data collection ───────────────────────────────────────────────────────────

_INVALID_MATERIAL = {"", "nan", "None", "Unknown", "[]", "[ ]", "none", "unknown"}


def _material_from_igsn(igsn, igsn_material_map):
    """Match an IGSN against the config's igsn_material_map (longest key wins)."""
    if not igsn_material_map:
        return None
    igsn = str(igsn).strip().lower()
    if not igsn:
        return None
    for map_key in sorted(igsn_material_map, key=lambda k: len(str(k)), reverse=True):
        key_lower = str(map_key).strip().lower()
        if key_lower and igsn.startswith(key_lower):
            return str(igsn_material_map[map_key]).strip()
    return None


def _infer_material(df, igsn_material_map=None):
    """Resolve a per-row 'Material' column.

    Priority: igsn_material_map (config IGSN -> material) takes precedence
    over whatever is recorded in 'Sample material'/'Material' columns, so a
    known sample's IGSN always wins over a stale or missing parameter-file
    value. 'Flyer_material' is never used here — it describes the flyer, not
    the target sample.
    """
    source_col = next((c for c in ("Sample material", "Material", "sample_material") if c in df.columns), None)
    igsn_col = next((c for c in ("Sample_IGSN", "Sample IGSN", "IGSN") if c in df.columns), None)

    def resolve(row):
        if igsn_col is not None:
            mapped = _material_from_igsn(row[igsn_col], igsn_material_map)
            if mapped:
                return mapped
        if source_col is not None:
            val = str(row[source_col]).strip()
            if val not in _INVALID_MATERIAL:
                return val
        return "Unknown"

    df["Material"] = df.apply(resolve, axis=1)
    return df


# The consolidated master (<IGSN>-Data_Summary.csv) is now written with standardized
# column names (e.g. HEL_GPa, Peak_Shock_Stress_GPa). This module still refers to the
# legacy spellings throughout, so alias the standardized names back to the legacy ones
# on load -- keeps both old and new master files working with a single edit point.
_STANDARDIZED_TO_LEGACY = {
    'HEL_GPa': 'hel_strength_gpa',
    'HEL_Uncertainty_GPa': 'hel_uncertainty_gpa',
    'HEL_StrainRate_s^-1': 'hel_strain_rate_s^-1',
    'HEL_OK': 'hel_ok',
    'HEL_Segment_Time_ns': 'hel_segment_time_ns',
    'HEL_Consecutive_Points': 'hel_consecutive_points',
    'HEL_FreeSurface_Velocity_m_s': 'free_surface_velocity_ms',
    'Peak_Shock_Stress_GPa': 'Peak Shock Stress (GPa)',
    'Peak_Shock_Stress_Unc_GPa': 'Peak Shock Stress Uncertainty (GPa)',
    'Plateau_Mean_Velocity_m_s': 'Plateau Mean Velocity (m/s)',
}


def _alias_standardized_columns(df):
    """Ensure legacy column spellings exist for any standardized master columns."""
    if df is None or not hasattr(df, "columns"):
        return df
    for new, old in _STANDARDIZED_TO_LEGACY.items():
        if new in df.columns and old not in df.columns:
            df[old] = df[new]
    return df


def _load_spade_dir(spade_dir, label, igsn_material_map=None):
    """Return (spall_df, hel_df) from one SPADE_analysis folder, or (None, None).

    Looks for *-Data_Summary.csv files (new naming: IGSN-shotnum-Data_Summary.csv).
    Falls back to legacy fixed filenames if none are found.
    """
    spall_df = hel_df = None

    # New naming convention: *-Data_Summary.csv
    data_summary_files = sorted(glob.glob(os.path.join(spade_dir, "*-Data_Summary.csv")))

    if data_summary_files:
        frames = []
        for path in data_summary_files:
            df = pd.read_csv(path)
            df = _alias_standardized_columns(df)
            df = _infer_material(df, igsn_material_map)
            df["_subfolder"] = label
            frames.append(df)
            print(f"  [data]  {len(df):>4d} rows  ← {path}")
        combined = pd.concat(frames, ignore_index=True)
        # Feed the combined summary to both panels; column checks in plotting
        # functions determine what actually gets rendered.
        if "Spall_Strength_GPa" in combined.columns:
            spall_df = combined
        if "hel_strength_gpa" in combined.columns:
            hel_df = combined
    else:
        # Legacy fallback
        spall_path = os.path.join(spade_dir, "enhanced_spall_summary.csv")
        vel_path   = os.path.join(spade_dir, "velocity_shots_summary.csv")

        if os.path.exists(spall_path):
            df = pd.read_csv(spall_path)
            df = _infer_material(df, igsn_material_map)
            df["_subfolder"] = label
            spall_df = df
            print(f"  [spall] {len(df):>4d} rows  ← {spall_path}")
        else:
            print(f"  [WARN]  not found: {spall_path}")

        if os.path.exists(vel_path):
            df = pd.read_csv(vel_path)
            df = _infer_material(df, igsn_material_map)
            df["_subfolder"] = label
            hel_df = df
            print(f"  [hel]   {len(df):>4d} rows  ← {vel_path}")
        else:
            print(f"  [WARN]  not found: {vel_path}")

    if spall_df is None and hel_df is None:
        print(f"  [WARN]  no usable files in: {spade_dir}")

    return spall_df, hel_df


def collect_data(settings):
    parent   = settings["input_dir"]
    pattern  = settings["subfolder_pattern"]
    subdir   = settings["output_subdir"]
    combined = settings["combined_output"]
    igsn_material_map = settings.get("igsn_material_map", {})

    spall_frames, hel_frames, raw_frames = [], [], []

    if combined:
        spade_dir = os.path.join(combined, "SPADE_analysis")
        print(f"\nCombined output mode: {spade_dir}")
        s, h = _load_spade_dir(spade_dir, label=os.path.basename(parent), igsn_material_map=igsn_material_map)
        if s is not None: spall_frames.append(s); raw_frames.append(s)
        if h is not None and h is not s: hel_frames.append(h)
        elif h is not None: hel_frames.append(h)
    else:
        candidates = sorted(glob.glob(os.path.join(os.path.abspath(parent), pattern)))
        subfolders = [p for p in candidates if os.path.isdir(p)]
        print(f"\nFound {len(subfolders)} subfolder(s) under: {parent}")
        for sf in subfolders:
            label = os.path.basename(sf)
            print(f"\n  [{label}]")
            spade_dir = os.path.join(sf, subdir, "SPADE_analysis")
            s, h = _load_spade_dir(spade_dir, label=label, igsn_material_map=igsn_material_map)
            if s is not None:
                spall_frames.append(s)
                raw_frames.append(s)
            if h is not None:
                hel_frames.append(h)

    spall_df = pd.concat(spall_frames, ignore_index=True) if spall_frames else pd.DataFrame()
    hel_df   = pd.concat(hel_frames,  ignore_index=True) if hel_frames  else pd.DataFrame()
    raw_df   = pd.concat(raw_frames,   ignore_index=True) if raw_frames  else pd.DataFrame()
    return spall_df, hel_df, raw_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _material_palette(materials):
    """Assign a consistent colour to each unique material string."""
    unique = sorted(set(str(m) for m in materials))
    cmap   = plt.cm.tab10
    return {m: cmap(i % 10) for i, m in enumerate(unique)}


_THICKNESS_COL = "Flyer_Thickness (um)"


def _spall_yerr(gdf):
    if "Spall_Strength_Unc_GPa" in gdf.columns:
        v = pd.to_numeric(gdf["Spall_Strength_Unc_GPa"], errors="coerce")
        if v.notna().any():
            return v.fillna(0).values
    return None

def _spall_xerr(gdf):
    if "Strain_Rate_Uncertainty_s^-1" in gdf.columns:
        v = pd.to_numeric(gdf["Strain_Rate_Uncertainty_s^-1"], errors="coerce")
        if v.notna().any():
            return v.fillna(0).values
    return None

def _hel_yerr(gdf):
    if "hel_uncertainty_gpa" in gdf.columns:
        v = pd.to_numeric(gdf["hel_uncertainty_gpa"], errors="coerce")
        if v.notna().any():
            return v.fillna(0).values
    return None

def _thickness_palette(thicknesses):
    """Blue (thin) → Red (thick) gradient; 'Unknown' gets grey."""
    known = sorted({t for t in thicknesses if t != "Unknown"}, key=lambda x: float(x))
    n = len(known)
    if n == 0:
        palette = {}
    elif n == 1:
        palette = {known[0]: (0.20, 0.40, 0.80, 1.0)}       # single → blue
    elif n == 2:
        palette = {known[0]: (0.20, 0.40, 0.80, 1.0),        # thin  → blue
                   known[1]: (0.85, 0.15, 0.15, 1.0)}        # thick → red
    else:
        cmap = plt.cm.RdBu_r                                  # blue→red gradient
        palette = {t: cmap(i / (n - 1)) for i, t in enumerate(known)}
    palette["Unknown"] = (0.55, 0.55, 0.55, 1.0)
    return palette


def _normalise_thickness(df):
    """Return a Series of thickness labels (string µm values or 'Unknown')."""
    if _THICKNESS_COL not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    vals = pd.to_numeric(df[_THICKNESS_COL], errors="coerce")
    return vals.apply(lambda v: "Unknown" if pd.isna(v) else str(int(v)) if v == int(v) else str(v))


def _plot_spall_by_thickness(ax, df):
    if df.empty:
        ax.text(0.5, 0.5, "No spall data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    if "Spall_OK" in df.columns:
        df = df[df["Spall_OK"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]
    df["Spall_Strength_GPa"]    = pd.to_numeric(df["Spall_Strength_GPa"],    errors="coerce")
    df["Spall_StrainRate_s^-1"] = pd.to_numeric(df.get("Spall_StrainRate_s^-1", np.nan), errors="coerce")
    df = df.dropna(subset=["Spall_Strength_GPa", "Spall_StrainRate_s^-1"])
    df = df[df["Spall_Strength_GPa"] > 0]
    df = df[df["Spall_StrainRate_s^-1"] > 0]

    if df.empty:
        ax.text(0.5, 0.5, "No valid spall points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df["_thickness"] = _normalise_thickness(df)
    palette = _thickness_palette(df["_thickness"])

    for thick, gdf in df.groupby("_thickness"):
        marker = "x" if thick == "Unknown" else "o"
        lbl    = f"Unknown (n={len(gdf)})" if thick == "Unknown" else f"{thick} µm (n={len(gdf)})"
        ax.errorbar(
            gdf["Spall_StrainRate_s^-1"], gdf["Spall_Strength_GPa"],
            yerr=_spall_yerr(gdf), xerr=_spall_xerr(gdf),
            fmt=marker, color=palette[thick], label=lbl,
            capsize=3, elinewidth=0.8, markersize=6, alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Spall Strain Rate (s⁻¹)", fontsize=11)
    ax.set_ylabel("Spall Strength (GPa)", fontsize=11)
    ax.set_title("Spall Strength vs Strain Rate\n(by Flyer Thickness)", fontsize=12)
    ax.legend(fontsize=9, loc="best", framealpha=0.7, title="Flyer Thickness")
    ax.grid(True, which="both", alpha=0.3)
    print(f"\n  Spall-thickness plot: {len(df)} points")


def _plot_hel_by_thickness(ax, df):
    if df.empty or "hel_strength_gpa" not in df.columns:
        ax.text(0.5, 0.5, "No HEL data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    df["hel_strength_gpa"]     = pd.to_numeric(df["hel_strength_gpa"],     errors="coerce")
    df["hel_strain_rate_s^-1"] = pd.to_numeric(df.get("hel_strain_rate_s^-1", np.nan), errors="coerce")

    if "hel_ok" in df.columns:
        df = df[df["hel_ok"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]

    df = df.dropna(subset=["hel_strength_gpa", "hel_strain_rate_s^-1"])
    df = df[df["hel_strength_gpa"] > 0]
    df = df[df["hel_strain_rate_s^-1"] > 0]

    if df.empty:
        ax.text(0.5, 0.5, "No valid HEL points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df["_thickness"] = _normalise_thickness(df)
    palette = _thickness_palette(df["_thickness"])

    for thick, gdf in df.groupby("_thickness"):
        marker = "x" if thick == "Unknown" else "s"
        lbl    = f"Unknown (n={len(gdf)})" if thick == "Unknown" else f"{thick} µm (n={len(gdf)})"
        ax.errorbar(
            gdf["hel_strain_rate_s^-1"], gdf["hel_strength_gpa"],
            yerr=_hel_yerr(gdf),
            fmt=marker, color=palette[thick], label=lbl,
            capsize=3, elinewidth=0.8, markersize=6, alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("HEL Strain Rate (s⁻¹)", fontsize=11)
    ax.set_ylabel("HEL Strength (GPa)", fontsize=11)
    ax.set_title("HEL Strength vs HEL Strain Rate\n(by Flyer Thickness)", fontsize=12)
    ax.legend(fontsize=9, loc="best", framealpha=0.7, title="Flyer Thickness")
    ax.grid(True, which="both", alpha=0.3)
    print(f"  HEL-thickness plot:   {len(df)} points")


def _plot_spall(ax, df):
    if df.empty:
        ax.text(0.5, 0.5, "No spall data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    if "Spall_OK" in df.columns:
        df = df[df["Spall_OK"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]
    df["Spall_Strength_GPa"]   = pd.to_numeric(df["Spall_Strength_GPa"],   errors="coerce")
    df["Spall_StrainRate_s^-1"] = pd.to_numeric(df.get("Spall_StrainRate_s^-1", np.nan), errors="coerce")
    df = df.dropna(subset=["Spall_Strength_GPa", "Spall_StrainRate_s^-1"])
    df = df[df["Spall_Strength_GPa"] > 0]
    df = df[df["Spall_StrainRate_s^-1"] > 0]

    if df.empty:
        ax.text(0.5, 0.5, "No valid spall points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    palette = _material_palette(df["Material"])
    for mat, gdf in df.groupby("Material"):
        yerr = _spall_yerr(gdf)
        xerr = _spall_xerr(gdf)
        ax.errorbar(
            gdf["Spall_StrainRate_s^-1"], gdf["Spall_Strength_GPa"],
            yerr=yerr, xerr=xerr, fmt="o", color=palette[mat], label=f"{mat} (n={len(gdf)})",
            capsize=3, elinewidth=0.8, markersize=6, alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Spall Strain Rate (s⁻¹)", fontsize=11)
    ax.set_ylabel("Spall Strength (GPa)", fontsize=11)
    ax.set_title("Spall Strength vs Strain Rate", fontsize=12)
    ax.legend(fontsize=9, loc="best", framealpha=0.7)
    ax.grid(True, which="both", alpha=0.3)
    print(f"\n  Spall plot: {len(df)} points, {df['Material'].nunique()} material(s)")


def _plot_hel(ax, df):
    if df.empty or "hel_strength_gpa" not in df.columns:
        ax.text(0.5, 0.5, "No HEL data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    df["hel_strength_gpa"]    = pd.to_numeric(df["hel_strength_gpa"],    errors="coerce")
    df["hel_strain_rate_s^-1"] = pd.to_numeric(df.get("hel_strain_rate_s^-1", np.nan), errors="coerce")

    # Keep only shots where HEL was successfully detected
    if "hel_ok" in df.columns:
        df = df[df["hel_ok"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]

    df = df.dropna(subset=["hel_strength_gpa", "hel_strain_rate_s^-1"])
    df = df[df["hel_strength_gpa"] > 0]
    df = df[df["hel_strain_rate_s^-1"] > 0]

    if df.empty:
        ax.text(0.5, 0.5, "No valid HEL points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    palette = _material_palette(df["Material"])
    for mat, gdf in df.groupby("Material"):
        ax.errorbar(
            gdf["hel_strain_rate_s^-1"], gdf["hel_strength_gpa"],
            yerr=_hel_yerr(gdf),
            fmt="s", color=palette[mat], label=f"{mat} (n={len(gdf)})",
            capsize=3, elinewidth=0.8, markersize=6, alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("HEL Strain Rate (s⁻¹)", fontsize=11)
    ax.set_ylabel("HEL Strength (GPa)", fontsize=11)
    ax.set_title("HEL Strength vs HEL Strain Rate", fontsize=12)
    ax.legend(fontsize=9, loc="best", framealpha=0.7)
    ax.grid(True, which="both", alpha=0.3)
    print(f"  HEL plot:   {len(df)} points, {df['Material'].nunique()} material(s)")


_RATE_BINS = [
    (0,          1e6,  "0 – 1×10⁶",       40,  plt.cm.viridis(0.0)),
    (1e6,        3e6,  "1×10⁶ – 3×10⁶",   80,  plt.cm.viridis(0.25)),
    (3e6,        6e6,  "3×10⁶ – 6×10⁶",  130,  plt.cm.viridis(0.55)),
    (6e6,        1e7,  "6×10⁶ – 1×10⁷",  190,  plt.cm.viridis(0.80)),
    (1e7,   np.inf,    "> 1×10⁷",         260,  plt.cm.viridis(1.0)),
]


def _plot_spall_vs_shock_stress(ax, df):
    """Spall Strength vs Peak Shock Stress; colour & size = discrete strain-rate bins."""
    if df.empty:
        ax.text(0.5, 0.5, "No spall data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    if "Spall_OK" in df.columns:
        df = df[df["Spall_OK"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]

    shock_col = "Peak_Shock_Stress_GPa_Final" if "Peak_Shock_Stress_GPa_Final" in df.columns \
                else "Peak Shock Stress (GPa)"

    for col in [shock_col, "Spall_Strength_GPa", "Spall_StrainRate_s^-1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[shock_col, "Spall_Strength_GPa", "Spall_StrainRate_s^-1"])
    df = df[(df[shock_col] > 0) & (df["Spall_Strength_GPa"] > 0) & (df["Spall_StrainRate_s^-1"] > 0)]

    if df.empty:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    total = 0
    for lo, hi, label, size, color in _RATE_BINS:
        mask = (df["Spall_StrainRate_s^-1"] >= lo) & (df["Spall_StrainRate_s^-1"] < hi)
        sub  = df[mask]
        if sub.empty:
            continue
        n = len(sub)
        total += n
        ax.scatter(
            sub[shock_col], sub["Spall_Strength_GPa"],
            s=size, color=color, alpha=0.82,
            edgecolors="k", linewidths=0.4,
            label=f"{label} s⁻¹  (n={n})",
            zorder=3,
        )

    ax.legend(fontsize=8, loc="best", framealpha=0.75, title="Strain Rate Bin")
    ax.set_xlabel("Peak Shock Stress (GPa)", fontsize=11)
    ax.set_ylabel("Spall Strength (GPa)", fontsize=11)
    ax.set_title("Spall Strength vs Shock Stress\n(colour & size = strain rate bin)", fontsize=12)
    ax.grid(True, alpha=0.3)
    print(f"\n  Shock-stress plot: {total} points across {len(_RATE_BINS)} strain-rate bins")


def _shock_stress_col(df):
    return "Peak_Shock_Stress_GPa_Final" if "Peak_Shock_Stress_GPa_Final" in df.columns \
        else "Peak Shock Stress (GPa)"


def _style_violin_bodies(parts):
    for body in parts["bodies"]:
        body.set_facecolor("0.75")
        body.set_edgecolor("0.35")
        body.set_alpha(0.5)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("0.35")
            parts[key].set_linewidth(1.0)


def _jitter(n, width=0.12, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    return rng.uniform(-width, width, size=n) if n else np.array([])


_SIGMA_BAND_COLORS = {1: "#c6dbef", 2: "#6baed6", 3: "#2171b5"}   # light -> dark blue
_SIGMA_BAND_ALPHA  = {1: 0.45,      2: 0.40,      3: 0.35}


def _sigma_bands(ax, pos, vals, width=0.35):
    """Fit a normal distribution to vals, shade +/-1,2,3 sigma bands at x=pos
    (restricted to [pos-width, pos+width]), draw a dashed mean line, and
    return a boolean mask (same order as vals) flagging points beyond 3 sigma.
    """
    vals = np.asarray(vals, dtype=float)
    outliers = np.zeros(len(vals), dtype=bool)
    if len(vals) < 2:
        return outliers

    mu, sigma = vals.mean(), vals.std(ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        return outliers

    xspan = [pos - width, pos + width]
    prev_k = 0
    for k in (1, 2, 3):
        lo_out, hi_out = mu - k * sigma, mu + k * sigma
        lo_in, hi_in = mu - prev_k * sigma, mu + prev_k * sigma
        ax.fill_between(xspan, [hi_in, hi_in], [hi_out, hi_out],
                         color=_SIGMA_BAND_COLORS[k], alpha=_SIGMA_BAND_ALPHA[k], zorder=1, linewidth=0)
        ax.fill_between(xspan, [lo_out, lo_out], [lo_in, lo_in],
                         color=_SIGMA_BAND_COLORS[k], alpha=_SIGMA_BAND_ALPHA[k], zorder=1, linewidth=0)
        prev_k = k
    ax.plot(xspan, [mu, mu], color="0.2", linestyle="--", linewidth=1.0, zorder=2)

    return np.abs(vals - mu) > 3 * sigma


def _sigma_legend_handles():
    return [
        Patch(facecolor=_SIGMA_BAND_COLORS[1], alpha=_SIGMA_BAND_ALPHA[1], label="±1σ"),
        Patch(facecolor=_SIGMA_BAND_COLORS[2], alpha=_SIGMA_BAND_ALPHA[2], label="±2σ"),
        Patch(facecolor=_SIGMA_BAND_COLORS[3], alpha=_SIGMA_BAND_ALPHA[3], label="±3σ"),
        Line2D([0], [0], color="0.2", linestyle="--", label="mean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6",
               markeredgecolor="red", markeredgewidth=1.6, markersize=8, label="> 3σ (outlier)"),
    ]


def _prepare_spall_df(df):
    """Clean a raw spall dataframe: Spall_OK filter, numeric coercion, positive values only."""
    if df.empty:
        return df.copy()
    df = df.copy()
    if "Spall_OK" in df.columns:
        df = df[df["Spall_OK"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]

    shock_col = _shock_stress_col(df)
    df["Spall_Strength_GPa"] = pd.to_numeric(df["Spall_Strength_GPa"], errors="coerce")
    if shock_col in df.columns:
        df[shock_col] = pd.to_numeric(df[shock_col], errors="coerce")

    df = df.dropna(subset=["Spall_Strength_GPa"])
    df = df[df["Spall_Strength_GPa"] > 0]
    return df


def _draw_spall_violin_panel(fig, ax, df, shock_col, title, norm=None,
                              show_colorbar=True, show_legend=True):
    """Draw one Spall Strength violin panel (per material), with sigma bands,
    3-sigma outlier flagging, and points coloured by Peak Shock Stress.

    `df` must already be cleaned (see `_prepare_spall_df`). Returns the
    number of points flagged as beyond 3 sigma.
    """
    if df.empty:
        ax.text(0.5, 0.5, "No valid spall points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return 0

    materials = sorted(df["Material"].unique())
    positions = np.arange(1, len(materials) + 1)

    have_shock = shock_col in df.columns and df[shock_col].notna().any()
    cmap = plt.cm.viridis
    if have_shock and norm is None:
        norm = mcolors.Normalize(vmin=df[shock_col].min(), vmax=df[shock_col].max())

    rng = np.random.default_rng(0)
    n_outliers = 0
    for pos, mat in zip(positions, materials):
        gdf = df[df["Material"] == mat]
        vals = gdf["Spall_Strength_GPa"].values
        if len(vals) >= 2 and np.ptp(vals) > 0:
            parts = ax.violinplot([vals], positions=[pos], widths=0.7,
                                   showmeans=False, showmedians=True, showextrema=True)
            _style_violin_bodies(parts)

        outlier = _sigma_bands(ax, pos, vals)
        n_outliers += int(outlier.sum())
        edgecolors = np.where(outlier, "red", "k")
        linewidths = np.where(outlier, 1.6, 0.5)
        sizes      = np.where(outlier, 70, 45)

        jitter = _jitter(len(gdf), rng=rng)
        x = pos + jitter

        if have_shock:
            has_val = gdf[shock_col].notna().values
            shock_vals = gdf[shock_col].values
            if has_val.any():
                ax.scatter(x[has_val], vals[has_val],
                           c=shock_vals[has_val], cmap=cmap, norm=norm,
                           edgecolors=edgecolors[has_val], linewidths=linewidths[has_val],
                           s=sizes[has_val], zorder=3)
            if (~has_val).any():
                ax.scatter(x[~has_val], vals[~has_val],
                           color="0.6", edgecolors=edgecolors[~has_val],
                           linewidths=linewidths[~has_val], s=sizes[~has_val], zorder=3)
        else:
            ax.scatter(x, vals, color="0.25", edgecolors=edgecolors,
                       linewidths=linewidths, s=sizes, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{m}\n(n={(df['Material'] == m).sum()})" for m in materials], fontsize=9)
    ax.set_ylabel("Spall Strength (GPa)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    if show_legend:
        ax.legend(handles=_sigma_legend_handles(), fontsize=7, loc="upper right",
                  framealpha=0.75, title="Normal fit (per material)", title_fontsize=7.5)

    if have_shock and show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Peak Shock Stress (GPa)", fontsize=9)

    return n_outliers


def _plot_violin_spall(fig, ax, df):
    """Violin plot of Spall Strength per material; points coloured by Peak Shock Stress."""
    if df.empty:
        ax.text(0.5, 0.5, "No spall data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    shock_col = _shock_stress_col(df)
    clean = _prepare_spall_df(df)
    n_outliers = _draw_spall_violin_panel(
        fig, ax, clean, shock_col,
        title="Spall Strength Distribution\n(points coloured by Peak Shock Stress)",
    )
    print(f"\n  Spall violin: {len(clean)} points across {clean['Material'].nunique() if not clean.empty else 0} "
          f"material(s), {n_outliers} beyond 3σ")


_SHOCK_STRESS_BINS = [(1, 2), (2, 4), (4, 6), (6, 8)]


def _plot_spall_violin_by_shock_bins(fig, axes, df):
    """Same sigma-band violin plot as `_plot_violin_spall`, split into one
    subplot per Peak Shock Stress bin (GPa): 1-2, 2-4, 4-6, 6-8."""
    shock_col = _shock_stress_col(df)
    clean = _prepare_spall_df(df)

    for ax, (lo, hi) in zip(np.asarray(axes).flat, _SHOCK_STRESS_BINS):
        if clean.empty or shock_col not in clean.columns:
            ax.text(0.5, 0.5, "No spall data found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="grey")
            continue
        sub = clean[(clean[shock_col] >= lo) & (clean[shock_col] < hi)]
        n_outliers = _draw_spall_violin_panel(
            fig, ax, sub, shock_col,
            title=f"Shock Stress {lo}–{hi} GPa  (n={len(sub)})",
        )
        print(f"  Spall violin [{lo}-{hi} GPa]: {len(sub)} points, {n_outliers} beyond 3σ")


def _igsn_col(df):
    for c in ("Sample_IGSN", "Sample IGSN", "IGSN", "_subfolder"):
        if c in df.columns:
            return c
    return None


def spall_igsn_list(df):
    """Sorted list of IGSN labels present in a cleaned spall dataframe."""
    clean = _prepare_spall_df(df)
    igsn_col = _igsn_col(clean)
    if clean.empty or igsn_col is None:
        return []
    return sorted(clean[igsn_col].dropna().astype(str).unique())


def _plot_spall_violin_by_igsn(fig, axes, df):
    """Same sigma-band violin plot as `_plot_violin_spall`, split into one
    subplot per IGSN (sample/shot). `fig`/`axes` must already be sized to fit
    `len(spall_igsn_list(df))` panels (see `make_plot`)."""
    shock_col = _shock_stress_col(df)
    clean = _prepare_spall_df(df)
    igsn_col = _igsn_col(clean)

    axes_list = list(np.asarray(axes).flat)

    if clean.empty or igsn_col is None:
        for ax in axes_list:
            ax.axis("off")
        axes_list[0].axis("on")
        axes_list[0].text(0.5, 0.5, "No spall data found", ha="center", va="center",
                           transform=axes_list[0].transAxes, fontsize=11, color="grey")
        return

    igsns = sorted(clean[igsn_col].dropna().astype(str).unique())
    have_shock = shock_col in clean.columns and clean[shock_col].notna().any()
    norm = None
    if have_shock:
        norm = mcolors.Normalize(vmin=clean[shock_col].min(), vmax=clean[shock_col].max())

    y_min, y_max = clean["Spall_Strength_GPa"].min(), clean["Spall_Strength_GPa"].max()
    pad = (y_max - y_min) * 0.08 or 1.0
    ylim = (y_min - pad, y_max + pad)

    for ax, igsn in zip(axes_list, igsns):
        sub = clean[clean[igsn_col].astype(str) == igsn]
        n_outliers = _draw_spall_violin_panel(
            fig, ax, sub, shock_col,
            title=f"{igsn}  (n={len(sub)})",
            norm=norm, show_colorbar=False, show_legend=False,
        )
        ax.set_ylim(*ylim)
        print(f"  Spall violin [{igsn}]: {len(sub)} points, {n_outliers} beyond 3σ")

    for ax in axes_list[len(igsns):]:
        ax.axis("off")

    # Lay out the grid first; the colorbar/legend added below reserve extra
    # space from the *already-laid-out* axes, so nothing else may reflow the
    # figure after this point (a later plt.tight_layout() would overlap them).
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])

    fig.legend(handles=_sigma_legend_handles(), fontsize=9, loc="upper right",
               bbox_to_anchor=(0.99, 0.99), framealpha=0.85, title="Normal fit (per IGSN)")

    if have_shock:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_list, shrink=0.6, pad=0.03, fraction=0.04)
        cbar.set_label("Peak Shock Stress (GPa)", fontsize=10)


def _plot_violin_hel(ax, df):
    """Violin plot of HEL Strength per material."""
    if df.empty or "hel_strength_gpa" not in df.columns:
        ax.text(0.5, 0.5, "No HEL data found", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    df = df.copy()
    df["hel_strength_gpa"] = pd.to_numeric(df["hel_strength_gpa"], errors="coerce")
    if "hel_ok" in df.columns:
        df = df[df["hel_ok"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]

    df = df.dropna(subset=["hel_strength_gpa"])
    df = df[df["hel_strength_gpa"] > 0]

    if df.empty:
        ax.text(0.5, 0.5, "No valid HEL points", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="grey")
        return

    materials = sorted(df["Material"].unique())
    positions = np.arange(1, len(materials) + 1)

    rng = np.random.default_rng(1)
    n_outliers = 0
    for pos, mat in zip(positions, materials):
        gdf = df[df["Material"] == mat]
        vals = gdf["hel_strength_gpa"].values
        if len(vals) >= 2 and np.ptp(vals) > 0:
            parts = ax.violinplot([vals], positions=[pos], widths=0.7,
                                   showmeans=False, showmedians=True, showextrema=True)
            _style_violin_bodies(parts)

        outlier = _sigma_bands(ax, pos, vals)
        n_outliers += int(outlier.sum())
        edgecolors = np.where(outlier, "red", "k")
        linewidths = np.where(outlier, 1.6, 0.5)
        sizes      = np.where(outlier, 65, 40)

        jitter = _jitter(len(gdf), rng=rng)
        ax.scatter(pos + jitter, vals, marker="s", color="0.25",
                   edgecolors=edgecolors, linewidths=linewidths, s=sizes, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{m}\n(n={(df['Material'] == m).sum()})" for m in materials], fontsize=9)
    ax.set_ylabel("HEL Strength (GPa)", fontsize=11)
    ax.set_title("HEL Strength Distribution", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=_sigma_legend_handles(), fontsize=7.5, loc="upper right",
              framealpha=0.75, title="Normal fit (per material)", title_fontsize=8)
    print(f"  HEL violin:   {len(df)} points across {len(materials)} material(s), "
          f"{n_outliers} beyond 3σ")


def make_plot(spall_df, hel_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Figure 1 — by material
    fig1, (ax_spall, ax_hel) = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("Batch SPADE Summary — by Material", fontsize=14, fontweight="bold")
    _plot_spall(ax_spall, spall_df)
    _plot_hel(ax_hel, hel_df)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = os.path.join(save_dir, "batch_spall_hel_summary.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n✅ Saved: {out1}")

    # Figure 2 — by flyer thickness
    fig2, (ax_spall_t, ax_hel_t) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Batch SPADE Summary — by Flyer Thickness", fontsize=14, fontweight="bold")
    _plot_spall_by_thickness(ax_spall_t, spall_df)
    _plot_hel_by_thickness(ax_hel_t, hel_df)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = os.path.join(save_dir, "batch_spall_hel_by_thickness.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"✅ Saved: {out2}")

    # Figure 3 — spall strength vs shock stress (colour & size = strain rate)
    fig3, ax_ss = plt.subplots(1, 1, figsize=(8, 6))
    fig3.suptitle("Spall Strength vs Peak Shock Stress", fontsize=14, fontweight="bold")
    _plot_spall_vs_shock_stress(ax_ss, spall_df)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = os.path.join(save_dir, "batch_spall_vs_shock_stress.png")
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"✅ Saved: {out3}")

    # Figure 4 — violin plots: Spall Strength (coloured by shock stress) & HEL Strength
    fig4, (ax_violin_spall, ax_violin_hel) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle("Batch SPADE Summary — Strength Distributions", fontsize=14, fontweight="bold")
    _plot_violin_spall(fig4, ax_violin_spall, spall_df)
    _plot_violin_hel(ax_violin_hel, hel_df)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out4 = os.path.join(save_dir, "batch_spall_hel_violin.png")
    fig4.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close(fig4)
    print(f"✅ Saved: {out4}")

    # Figure 5 — Spall Strength violin, split into subplots per Peak Shock Stress bin
    fig5, axes5 = plt.subplots(2, 2, figsize=(14, 12))
    fig5.suptitle("Spall Strength Distribution by Peak Shock Stress Bin", fontsize=14, fontweight="bold")
    _plot_spall_violin_by_shock_bins(fig5, axes5, spall_df)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out5 = os.path.join(save_dir, "batch_spall_violin_by_shock_bin.png")
    fig5.savefig(out5, dpi=200, bbox_inches="tight")
    plt.close(fig5)
    print(f"✅ Saved: {out5}")

    # Figure 6 — Spall Strength violin, split into subplots per IGSN (sample/shot)
    igsn_list = spall_igsn_list(spall_df)
    ncols6 = 4
    nrows6 = max(int(np.ceil(len(igsn_list) / ncols6)), 1)
    fig6, axes6 = plt.subplots(nrows6, ncols6, figsize=(4.2 * ncols6, 4.2 * nrows6), squeeze=False)
    fig6.suptitle("Spall Strength Distribution by IGSN", fontsize=14, fontweight="bold")
    _plot_spall_violin_by_igsn(fig6, axes6, spall_df)
    out6 = os.path.join(save_dir, "batch_spall_violin_by_igsn.png")
    fig6.savefig(out6, dpi=200, bbox_inches="tight")
    plt.close(fig6)
    print(f"✅ Saved: {out6}")

    return out1, out2, out3, out4, out5, out6


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Config: {BATCH_CONFIG}")
    settings = _load_batch_settings()
    print(f"Input dir: {settings['input_dir']}")

    spall_df, hel_df, raw_df = collect_data(settings)
    print(f"\nTotal — spall rows: {len(spall_df)},  HEL rows: {len(hel_df)},  raw rows: {len(raw_df)}")

    save_dir = settings["combined_output"] or settings["input_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Save combined CSV of all raw rows from every subfolder
    csv_path = os.path.join(save_dir, "batch_combined_summary.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    # Report any rows whose material resolved to Unknown / []
    if not raw_df.empty and "Material" in raw_df.columns:
        bad = raw_df[raw_df["Material"].isin(_INVALID_MATERIAL) | (raw_df["Material"] == "Unknown")]
        if not bad.empty:
            igsn_col = "Sample_IGSN" if "Sample_IGSN" in bad.columns else "_subfolder"
            grouped = bad.groupby(igsn_col).size().reset_index(name="rows")
            print(f"\n⚠️  {len(bad)} row(s) with unresolved material across "
                  f"{grouped[igsn_col].nunique()} IGSN(s):")
            for _, row in grouped.iterrows():
                print(f"    {row[igsn_col]}  ({row['rows']} row(s))")
            if "Filename" in bad.columns:
                print("  Filenames:")
                for fn in bad["Filename"].dropna().unique():
                    print(f"    {fn}")
        else:
            print("\n✅ All rows have a recognised material.")

    # Report IGSNs with HEL strength below 0.7 GPa
    if not hel_df.empty and "hel_strength_gpa" in hel_df.columns:
        h = hel_df.copy()
        if "hel_ok" in h.columns:
            h = h[h["hel_ok"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]
        h["hel_strength_gpa"] = pd.to_numeric(h["hel_strength_gpa"], errors="coerce")
        low_hel = h[h["hel_strength_gpa"] < 0.7].dropna(subset=["hel_strength_gpa"])
        if not low_hel.empty:
            igsn_col = "Sample_IGSN" if "Sample_IGSN" in low_hel.columns else "_subfolder"
            grouped  = low_hel.groupby(igsn_col)["hel_strength_gpa"].agg(["count", "min", "max"])
            print(f"\n⚠️  HEL < 0.7 GPa — {len(low_hel)} shot(s) across "
                  f"{grouped.shape[0]} IGSN(s):")
            for igsn, row in grouped.iterrows():
                print(f"    {igsn}  n={int(row['count'])}  "
                      f"range [{row['min']:.3f} – {row['max']:.3f}] GPa")
        else:
            print("\n✅ No HEL values below 0.7 GPa.")

    make_plot(spall_df, hel_df, save_dir)


if __name__ == "__main__":
    main()
