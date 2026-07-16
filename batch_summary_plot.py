#!/usr/bin/env python3
"""
Aggregate batch SPADE results and produce combined Spall and HEL strength plots.

Reads helix_master_config_batch_process.json to locate the batch input directory,
collects enhanced_spall_summary.csv and velocity_shots_summary.csv from each
subfolder's SPADE_analysis output, then saves a two-panel figure to input_dir:
  Left panel  — Spall Strength (GPa) vs Spall Strain Rate (s⁻¹)
  Right panel — HEL Strength (GPa)   vs HEL Strain Rate   (s⁻¹)
Points are colour-coded by material; error bars are included where available.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    }


# ── Data collection ───────────────────────────────────────────────────────────

_INVALID_MATERIAL = {"", "nan", "None", "Unknown", "[]", "[ ]", "none", "unknown"}

def _infer_material(df):
    """Return the first non-empty material column found, standardised as 'Material'."""
    for col in ("Sample material", "Material", "Flyer_material", "sample_material"):
        if col in df.columns:
            vals = df[col].astype(str).str.strip()
            valid = vals[~vals.isin(_INVALID_MATERIAL)]
            if not valid.empty:
                df["Material"] = vals
                return df
    df["Material"] = "Unknown"
    return df


def _load_spade_dir(spade_dir, label):
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
            df = _infer_material(df)
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
            df = _infer_material(df)
            df["_subfolder"] = label
            spall_df = df
            print(f"  [spall] {len(df):>4d} rows  ← {spall_path}")
        else:
            print(f"  [WARN]  not found: {spall_path}")

        if os.path.exists(vel_path):
            df = pd.read_csv(vel_path)
            df = _infer_material(df)
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

    spall_frames, hel_frames, raw_frames = [], [], []

    if combined:
        spade_dir = os.path.join(combined, "SPADE_analysis")
        print(f"\nCombined output mode: {spade_dir}")
        s, h = _load_spade_dir(spade_dir, label=os.path.basename(parent))
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
            s, h = _load_spade_dir(spade_dir, label=label)
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

    return out1, out2, out3


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
