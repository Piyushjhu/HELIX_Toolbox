#!/usr/bin/env python3
# =============================================================================
# HELIX Toolbox — One-At-A-Time (OAT) Parameter Sensitivity Analysis
# =============================================================================
# Drives the HELIX CLI pipeline (ALPSS -> SPADE) repeatedly on a SINGLE PDV
# file, varying one key parameter at a time around a baseline configuration,
# and records how the physics outputs respond:
#
#     * Spall strength            (Spall_Strength_GPa_Final)
#     * Spall strain rate         (Strain_Rate_s1_Final)
#     * Peak shock stress         (Peak_Shock_Stress_GPa_Final)
#     * HEL                       (HEL_GPa)
#     * HEL strain rate           (HEL_StrainRate_s^-1)
#     * Intermediate velocities   (first maxima, plateau mean, pullback minima,
#                                  rise time, shock-front width, HEL free-surf)
#
# Method: local one-at-a-time (OAT) screening. Each parameter is swept across a
# physically sensible grid while every other parameter is held at its baseline
# value. This isolates the marginal effect of each knob and yields:
#   - a tidy long-format CSV of every run and its outputs (one 'trace' column),
#   - a per-(parameter, output) sensitivity summary CSV (swing %, elasticity),
#   - tornado plots, a sensitivity heatmap, and response curves.
#
# Multiple traces: pass several files (or a folder) and every trace is swept over
# the full parameter set. Each trace gets its own influence matrix + plots/<trace>/,
# and a cross-trace consistency CSV + heatmap show whether a parameter's influence
# is reproducible across materials/shots or is specific to one trace.
#
# Usage (from repo root, using the project venv):
#   # single trace (default 0001):
#   QT_QPA_PLATFORM=offscreen helix_toolbox_env/bin/python3 \
#       helix_sensitivity_analysis.py --config helix_master_config.yml
#   # several specific traces:
#   ... helix_sensitivity_analysis.py --input-files a.csv b.csv c.csv
#   # every file in a folder:
#   ... helix_sensitivity_analysis.py --input-dir /path/PDV --input-glob "*.csv"
#
# Useful flags:
#   --input-file PATH     single PDV file (default: 0001 from config input_dir)
#   --input-files A B ..  explicit list of PDV files; each swept over all params
#   --input-dir DIR       folder of PDV files (with --input-glob, default *.csv)
#   --param-folder DIR    experiment metadata folder (material/energy enrichment);
#                         'none' disables it. Overrides the config's param_folder so
#                         the harness can run on any dataset without editing the config.
#   --outdir DIR          where results/plots/CSVs go (default: sensitivity_analysis/<ts>)
#   --groups G[,G...]     restrict to: alpss, spade_spall, spade_hel
#   --params P[,P...]     restrict to specific parameter ids
#   --dry-run             list the runs that WOULD execute, then exit
#   --keep-run-outputs    keep each run's full toolbox output dir (large!)
#   --analyze-only DIR    recompute summary + plots from an existing runs CSV dir
#   --resume              skip runs already present in the runs CSV
# =============================================================================

import argparse
import copy
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO_ROOT, "helix_toolbox_env", "bin", "python3")
CLI = os.path.join(REPO_ROOT, "helix_cli_runner.py")
DEFAULT_TEST_FILE = "C1--20251022--00007.csv"

# -----------------------------------------------------------------------------
# Output metrics pulled from <prefix>-Data_Summary.csv.  (column, friendly, primary)
# "primary" metrics get tornado plots + heatmap columns; the rest are recorded
# and available in the runs CSV as supporting/intermediate signals.
# -----------------------------------------------------------------------------
OUTPUT_METRICS = [
    ("Spall_Strength_GPa_Final",   "Spall strength (GPa)",        True),
    ("Strain_Rate_s1_Final",       "Spall strain rate (1/s)",     True),
    ("Peak_Shock_Stress_GPa_Final","Peak shock stress (GPa)",     True),
    ("HEL_GPa",                    "HEL (GPa)",                   True),
    ("HEL_StrainRate_s^-1",        "HEL strain rate (1/s)",       True),
    ("First_Maxima_m_s",           "First maxima (m/s)",          False),
    ("Plateau_Mean_Velocity_m_s",  "Plateau mean vel (m/s)",      False),
    ("Minima_m_s",                 "Pullback minima (m/s)",       False),
    ("RiseTime_80_20_ns",          "Rise time 80-20 (ns)",        False),
    ("Shock_Front_Width_um",       "Shock front width (um)",      False),
    ("HEL_FreeSurface_Velocity_m_s","HEL free-surface vel (m/s)", False),
]
# Non-numeric status columns recorded for context.
STATUS_COLS = ["DNS_Classification", "Spall_OK", "HEL_OK", "Processing_Status", "Material"]

METRIC_COLS = [c for c, _, _ in OUTPUT_METRICS]
PRIMARY_METRICS = [c for c, _, p in OUTPUT_METRICS if p]
LABELS = {c: lbl for c, lbl, _ in OUTPUT_METRICS}


# -----------------------------------------------------------------------------
# Parameter specification.  Each entry sweeps ONE knob (possibly writing to
# several coupled config keys).  `baseline` is filled in from the loaded config.
#   id      : short identifier used in CSVs / filenames / --params
#   section : "alpss_config" or "spade_config"
#   keys    : config key(s) written with the swept value (list => coupled)
#   group   : alpss | spade_spall | spade_hel
#   values  : grid of levels to test (baseline is added automatically as the
#             reference point but is only *run* once, globally)
#   kind    : "num" (default) or "cat" (categorical; no elasticity computed)
# -----------------------------------------------------------------------------
def build_param_spec():
    E9 = 1e9
    spec = [
        # ---- ALPSS: velocity extraction (upstream of everything) ----
        dict(id="iq_threshold_factor", section="alpss_config", keys=["iq_threshold_factor"],
             group="alpss", values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
        dict(id="freq_min", section="alpss_config", keys=["freq_min"],
             group="alpss", values=[1.0*E9, 1.1*E9, 1.2*E9, 1.3*E9, 1.5*E9]),
        dict(id="freq_max", section="alpss_config", keys=["freq_max"],
             group="alpss", values=[2.8*E9, 3.0*E9, 3.3*E9, 3.6*E9, 4.0*E9]),
        dict(id="nperseg", section="alpss_config", keys=["nperseg"],
             group="alpss", values=[512, 768, 1024, 1536, 2048], couple="nperseg_noverlap"),
        dict(id="noverlap", section="alpss_config", keys=["noverlap"],
             group="alpss", values=[200, 300, 400, 460, 500]),
        dict(id="nfft", section="alpss_config", keys=["nfft"],
             group="alpss", values=[1280, 2048, 2560, 3072, 4096]),
        dict(id="blur_kernel", section="alpss_config", keys=["blur_kernel_x", "blur_kernel_y"],
             group="alpss", values=[1, 3, 5, 7]),
        dict(id="smoothing_window_ns", section="alpss_config", keys=["smoothing_window_ns"],
             group="alpss", values=[4, 6, 8, 10, 15, 20]),
        dict(id="savgol_polyorder", section="alpss_config", keys=["savgol_polyorder"],
             group="alpss", values=[2, 3, 4]),
        dict(id="t_before", section="alpss_config", keys=["t_before"],
             group="alpss", values=[2e-8, 4e-8, 6e-8, 8e-8, 1.2e-7]),
        dict(id="t_after", section="alpss_config", keys=["t_after"],
             group="alpss", values=[1e-7, 1.5e-7, 2e-7, 3e-7, 4e-7]),
        dict(id="time_to_take", section="alpss_config", keys=["time_to_take"],
             group="alpss", values=[3e-6, 4.5e-6, 6e-6, 7.5e-6]),
        dict(id="carrier_band_time", section="alpss_config", keys=["carrier_band_time"],
             group="alpss", values=[1.5e-7, 2e-7, 2.5e-7, 3e-7, 3.5e-7]),
        dict(id="uncert_mult", section="alpss_config", keys=["uncert_mult"],
             group="alpss", values=[5, 8, 10, 15, 20]),

        # ---- SPADE: spall P3/P4 detection & gating ----
        dict(id="spall_smoothing_sigma_ns", section="spade_config", keys=["spall_smoothing_sigma_ns"],
             group="spade_spall", values=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
        dict(id="prominence_factor", section="spade_config", keys=["prominence_factor"],
             group="spade_spall", values=[0.005, 0.0075, 0.01, 0.015, 0.02]),
        dict(id="peak_distance_ns", section="spade_config", keys=["peak_distance_ns"],
             group="spade_spall", values=[1.0, 2.0, 3.0, 4.0, 5.0]),
        dict(id="min_recomp_ratio", section="spade_config", keys=["min_recomp_ratio"],
             group="spade_spall", values=[0.01, 0.015, 0.025, 0.05, 0.08]),
        dict(id="min_recomp_velocity_ratio", section="spade_config", keys=["min_recomp_velocity_ratio"],
             group="spade_spall", values=[1.01, 1.015, 1.025, 1.05, 1.08]),
        dict(id="min_recomp_time_ns", section="spade_config", keys=["min_recomp_time_ns"],
             group="spade_spall", values=[1, 2, 3, 4, 5]),
        dict(id="spall_end_time_ns", section="spade_config", keys=["spall_end_time_ns"],
             group="spade_spall", values=[60, 80, 100, 120, 150]),
        dict(id="threshold_velocity_ms", section="spade_config", keys=["threshold_velocity_ms"],
             group="spade_spall", values=[2, 5, 8, 12, 20]),

        # ---- SPADE: HEL detection ----
        dict(id="hel_end_time_ns", section="spade_config", keys=["hel_end_time_ns"],
             group="spade_hel", values=[10, 15, 20, 30, 40]),
        dict(id="minimum_HEL_velocity_expected", section="spade_config", keys=["minimum_HEL_velocity_expected"],
             group="spade_hel", values=[5, 10, 15, 20]),
        dict(id="hel_detection_min_points", section="spade_config", keys=["hel_detection_min_points"],
             group="spade_hel", values=[5, 8, 10, 15, 20]),
        dict(id="hel_rdp_epsilon", section="spade_config", keys=["hel_rdp_epsilon"],
             group="spade_hel", values=[0.5, 1.0, 1.25, 2.0, 3.0]),
        dict(id="hel_slope_drop_ratio", section="spade_config", keys=["hel_slope_drop_ratio"],
             group="spade_hel", values=[0.8, 0.85, 0.9, 0.95]),
        dict(id="hel_min_plateau_duration", section="spade_config", keys=["hel_min_plateau_duration"],
             group="spade_hel", values=[0.25, 0.5, 1.0, 2.0]),
        dict(id="hel_t0_method", section="spade_config", keys=["hel_t0_method"], kind="cat",
             group="spade_hel", values=["alpss_signal_start", "signal_start", "first_positive"]),
    ]
    for s in spec:
        s.setdefault("kind", "num")
        s.setdefault("couple", None)
    return spec


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_couplings(cfg, couple, spec):
    """Fix hard constraints between coupled ALPSS keys after a value is set."""
    if couple == "nperseg_noverlap":
        a = cfg["alpss_config"]
        # noverlap must be strictly < nperseg; preserve overlap fraction if violated.
        if a.get("noverlap", 0) >= a.get("nperseg", 1):
            a["noverlap"] = int(round(a["nperseg"] * 0.78))


def fmt_val(v, kind):
    if kind == "cat":
        return str(v)
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:g}"
    return str(v)


def _safe(s):
    """Filesystem-safe token for filenames/run stubs."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))


def make_sweep_base_config(cfg):
    """Copy base config and disable per-run plotting for speed/disk."""
    c = copy.deepcopy(cfg)
    c.setdefault("alpss_config", {})["save_all_plots"] = "no"
    sp = c.setdefault("spade_config", {})
    sp["plot_individual"] = False
    sp["generate_all_velocity_plot"] = False
    sp["show_plots"] = False
    c.setdefault("post_processing_config", {})["enabled"] = False
    return c


def read_metrics(run_out_dir):
    """Parse the consolidated master summary produced by one run."""
    matches = glob.glob(os.path.join(run_out_dir, "SPADE_analysis", "*Data_Summary.csv"))
    if not matches:
        return None, "no_summary"
    try:
        df = pd.read_csv(matches[0])
    except Exception as e:  # pragma: no cover - defensive
        return None, f"read_error:{e}"
    if df.empty:
        return None, "empty_summary"
    row = df.iloc[0]
    out = {}
    for col in METRIC_COLS:
        out[col] = pd.to_numeric(row.get(col), errors="coerce") if col in df.columns else np.nan
    for col in STATUS_COLS:
        out[col] = row.get(col, "") if col in df.columns else ""
    return out, "ok"


def run_one(base_sweep_cfg, section, keys, value, couple, spec_couple,
            input_file, run_out_dir, cfg_path, log_path):
    """Write a temp config with the swept value and execute the CLI once."""
    cfg = copy.deepcopy(base_sweep_cfg)
    for k in keys:
        cfg[section][k] = value
    apply_couplings(cfg, couple, spec_couple)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    os.makedirs(run_out_dir, exist_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [PYTHON, CLI, "--config", cfg_path,
             "--input-files", input_file, "--output-dir", run_out_dir],
            stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT,
        )
    wall = time.time() - t0

    if proc.returncode != 0:
        return None, "cli_error", wall
    metrics, status = read_metrics(run_out_dir)
    return metrics, status, wall


# -----------------------------------------------------------------------------
# Sweep driver
# -----------------------------------------------------------------------------
RUN_FIELDS = (["trace", "run_id", "group", "param", "param_value", "param_value_num",
               "is_baseline", "status", "wall_time_s"] + METRIC_COLS + STATUS_COLS)


def append_run_row(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUN_FIELDS})


def existing_run_ids(csv_path):
    """Set of already-completed (trace||run_id) keys, for --resume."""
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={"trace": str})
        tr = df["trace"].astype(str) if "trace" in df.columns else pd.Series([""] * len(df))
        return set(tr + "||" + df["run_id"].astype(str))
    except Exception:
        return set()


def build_plan(spec):
    """Build the per-trace run list: baseline first, then each off-baseline level."""
    planned = [dict(run_id="baseline", group="baseline", param="baseline",
                    section=None, keys=None, value=None, num=np.nan,
                    is_baseline=True, kind="num", couple=None)]
    for s in spec:
        base = s["baseline"]
        for v in s["values"]:
            if s["kind"] == "num" and base is not None and np.isclose(float(v), float(base)):
                continue
            if s["kind"] == "cat" and str(v) == str(base):
                continue
            rid = f"{s['id']}={fmt_val(v, s['kind'])}"
            planned.append(dict(run_id=rid, group=s["group"], param=s["id"],
                                section=s["section"], keys=s["keys"], value=v,
                                num=(float(v) if s["kind"] == "num" else np.nan),
                                is_baseline=False, kind=s["kind"], couple=s["couple"]))
    return planned


def run_sweep(args, cfg, spec, outdir, traces):
    """Run the OAT sweep for every trace in `traces` (list of (label, path))."""
    runs_csv = os.path.join(outdir, "helix_sensitivity_runs.csv")
    logs_dir = os.path.join(outdir, "run_logs")
    tmp_dir = os.path.join(outdir, "_tmp_runs")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    base_sweep_cfg = make_sweep_base_config(cfg)
    done = existing_run_ids(runs_csv) if args.resume else set()
    planned = build_plan(spec)

    if args.dry_run:
        print(f"\nDRY RUN — {len(traces)} trace(s) x {len(planned)} runs "
              f"= {len(traces) * len(planned)} total (1 baseline + {len(planned)-1} OAT each):\n")
        for label, path in traces:
            print(f"  TRACE {label}  ({path})")
        cur = None
        for p in planned:
            if p["group"] != cur:
                cur = p["group"]
                print(f"     [{cur}]")
            print(f"        {p['run_id']}")
        est = len(traces) * len(planned) * 16
        print(f"\nEstimated wall time ~{est//60} min at ~16 s/run.")
        return runs_csv

    grand_total = len(traces) * len(planned)
    n = 0
    t_start = time.time()
    for label, path in traces:
        safe_tr = _safe(label)
        for p in planned:
            n += 1
            key = f"{label}||{p['run_id']}"
            if key in done:
                print(f"[{n}/{grand_total}] {label}: skip (resume): {p['run_id']}")
                continue
            stub = f"{safe_tr}__{_safe(p['run_id'])}"
            run_out = os.path.join(tmp_dir, stub)
            cfg_path = os.path.join(tmp_dir, f"cfg_{safe_tr}.yml")
            log_path = os.path.join(logs_dir, stub + ".log")

            if p["is_baseline"]:
                section, keys, value, couple = "alpss_config", ["iq_threshold_factor"], \
                    base_sweep_cfg["alpss_config"]["iq_threshold_factor"], None
            else:
                section, keys, value, couple = p["section"], p["keys"], p["value"], p["couple"]

            metrics, status, wall = run_one(
                base_sweep_cfg, section, keys, value, couple, p["couple"],
                path, run_out, cfg_path, log_path)

            row = dict(trace=label, run_id=p["run_id"], group=p["group"], param=p["param"],
                       param_value=fmt_val(p["value"], p["kind"]) if not p["is_baseline"] else "baseline",
                       param_value_num=p["num"], is_baseline=p["is_baseline"],
                       status=status, wall_time_s=round(wall, 2))
            if metrics:
                row.update(metrics)
            append_run_row(runs_csv, row)

            spall = row.get("Spall_Strength_GPa_Final", "")
            hel = row.get("HEL_GPa", "")
            elapsed = time.time() - t_start
            eta = elapsed / n * (grand_total - n)
            print(f"[{n}/{grand_total}] {label:>8s} {p['run_id']:38s} {status:9s} "
                  f"spall={spall!s:>8.8} HEL={hel!s:>8.8} "
                  f"({wall:4.1f}s, ETA {eta/60:5.1f}m)")

            if not args.keep_run_outputs:
                shutil.rmtree(run_out, ignore_errors=True)

    if not args.keep_run_outputs:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nSweep complete: {grand_total} runs ({len(traces)} trace(s)) "
          f"in {(time.time()-t_start)/60:.1f} min")
    print(f"Runs CSV: {runs_csv}")
    return runs_csv


# -----------------------------------------------------------------------------
# Analysis: per-(param, metric) sensitivity summary
# -----------------------------------------------------------------------------
def _valid_mask(df, metric):
    """A run is a valid data point for a metric only if its detection passed.

    HEL-family metrics (any column mentioning HEL, incl. HEL free-surface velocity)
    gate on HEL_OK; all spall/shock/velocity metrics gate on Spall_OK.
    """
    is_hel = "HEL" in metric
    ok = (df["HEL_OK"].astype(str) == "True") if is_hel else (df["Spall_OK"].astype(str) == "True")
    return ok & pd.to_numeric(df[metric], errors="coerce").notna()


def _summary_rows(df_trace, base, spec_by_id, trace_label):
    """Per-(param, metric) sensitivity rows for ONE trace.

    swing is computed over VALID runs only (failed detections excluded) so a
    breakdown does not masquerade as sensitivity; failures are recorded via
    n_failed and surfaced separately.
    """
    rows = []
    body = df_trace[df_trace["is_baseline"] != True]
    for param, g in body.groupby("param"):
        s = spec_by_id.get(param, {})
        base_num = s.get("baseline", np.nan)
        for metric in METRIC_COLS:
            base_m = pd.to_numeric(base.get(metric), errors="coerce")
            vm = _valid_mask(g, metric)
            gv = g[vm]
            vals = pd.to_numeric(gv[metric], errors="coerce").dropna()
            pts = list(vals.values)
            if pd.notna(base_m):
                pts.append(base_m)
            n_valid, n_failed = len(vals), int((~vm).sum())
            if len(pts) < 2 or pd.isna(base_m) or base_m == 0:
                mn = float(np.min(pts)) if pts else np.nan
                mx = float(np.max(pts)) if pts else np.nan
                rows.append(dict(trace=trace_label, param=param, group=s.get("group", ""),
                                 metric=metric, metric_label=LABELS[metric], baseline=base_m,
                                 min=mn, max=mx, swing_abs=(mx - mn if pts else np.nan),
                                 swing_pct=np.nan, max_abs_elasticity=np.nan,
                                 mean_abs_elasticity=np.nan, n_valid=n_valid,
                                 n_failed=n_failed, n_runs=len(g)))
                continue
            mn, mx = float(np.min(pts)), float(np.max(pts))
            swing_abs = mx - mn
            swing_pct = swing_abs / abs(base_m) * 100.0

            # Local elasticity relative to baseline (numeric params only): (dM/M0)/(dP/P0)
            elasticities = []
            if s.get("kind", "num") == "num" and pd.notna(base_num) and base_num != 0:
                for _, r in gv.iterrows():
                    pv = pd.to_numeric(r.get("param_value_num"), errors="coerce")
                    mv = pd.to_numeric(r.get(metric), errors="coerce")
                    if pd.isna(pv) or pd.isna(mv) or np.isclose(pv, base_num):
                        continue
                    dp = (pv - base_num) / base_num
                    if dp != 0:
                        elasticities.append(abs(((mv - base_m) / base_m) / dp))
            rows.append(dict(trace=trace_label, param=param, group=s.get("group", ""),
                             metric=metric, metric_label=LABELS[metric], baseline=base_m,
                             min=mn, max=mx, swing_abs=swing_abs, swing_pct=swing_pct,
                             max_abs_elasticity=(max(elasticities) if elasticities else np.nan),
                             mean_abs_elasticity=(float(np.mean(elasticities)) if elasticities else np.nan),
                             n_valid=n_valid, n_failed=n_failed, n_runs=len(g)))
    return rows


def _pivot(sdf):
    prim = sdf[sdf["metric"].isin(PRIMARY_METRICS)]
    pivot = prim.pivot_table(index="param", columns="metric", values="swing_pct")
    pivot = pivot.reindex(columns=PRIMARY_METRICS)
    pivot["max_swing_pct"] = pivot.max(axis=1)
    return pivot.sort_values("max_swing_pct", ascending=False)


def compute_summary(runs_csv, spec, outdir):
    """Compute per-trace sensitivity + (for >1 trace) a cross-trace consistency table.

    Returns {trace_label: (summary_df, pivot_df, base_row)}.
    """
    df = pd.read_csv(runs_csv, dtype={"trace": str})
    if "trace" not in df.columns:
        df["trace"] = "(single)"
    spec_by_id = {s["id"]: s for s in spec}
    traces = list(dict.fromkeys(df["trace"].astype(str)))

    bundles, all_rows = {}, []
    for tr in traces:
        dtr = df[df["trace"].astype(str) == tr]
        base = dtr[dtr["is_baseline"] == True]
        if base.empty:
            base = dtr[dtr["run_id"] == "baseline"]
        if base.empty:
            print(f"[warn] trace {tr}: no baseline run — skipping its summary")
            continue
        base = base.iloc[0]
        rows = _summary_rows(dtr, base, spec_by_id, tr)
        all_rows += rows
        sdf = pd.DataFrame(rows)
        pivot = _pivot(sdf)
        suffix = f"__{_safe(tr)}" if len(traces) > 1 else ""
        pivot.to_csv(os.path.join(outdir, f"helix_sensitivity_influence_matrix{suffix}.csv"))
        bundles[tr] = (sdf, pivot, base)

    if not bundles:
        raise SystemExit("No baseline run found in runs CSV — cannot compute sensitivity.")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(outdir, "helix_sensitivity_summary.csv"), index=False)
    print(f"Summary CSV: {os.path.join(outdir, 'helix_sensitivity_summary.csv')}")

    # Cross-trace consistency: does each parameter rank the same across traces?
    if len(bundles) > 1:
        cols = {tr: piv["max_swing_pct"] for tr, (sdf, piv, base) in bundles.items()}
        m = pd.DataFrame(cols)
        tr_cols = list(cols)
        m["mean_swing_pct"] = m[tr_cols].mean(axis=1)
        m["std_swing_pct"] = m[tr_cols].std(axis=1)
        m["min_swing_pct"] = m[tr_cols].min(axis=1)
        m["max_swing_pct"] = m[tr_cols].max(axis=1)
        # coefficient of variation: low => consistently (un)important across traces
        m["cv"] = m["std_swing_pct"] / m["mean_swing_pct"].replace(0, np.nan)
        m = m.sort_values("mean_swing_pct", ascending=False)
        m.to_csv(os.path.join(outdir, "helix_cross_trace_consistency.csv"))
        print(f"Cross-trace consistency CSV: {os.path.join(outdir, 'helix_cross_trace_consistency.csv')}")
    return bundles


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _cross_trace_plot(bundles, plots_dir, plt):
    """Heatmap of each parameter's max output swing % across traces (top 20)."""
    cols = {tr: piv["max_swing_pct"] for tr, (sdf, piv, base) in bundles.items()}
    m = pd.DataFrame(cols)
    tr_cols = list(cols)
    m["__mean"] = m[tr_cols].mean(axis=1)
    m = m.sort_values("__mean", ascending=False).drop(columns="__mean").head(20)
    if m.empty:
        return
    data = m.values.astype(float)
    vmax = np.nanpercentile(data, 95) if np.isfinite(data).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.2 * len(m.columns) + 4, 0.34 * len(m) + 2))
    im = ax.imshow(np.nan_to_num(data, nan=0.0), aspect="auto", cmap="magma",
                   vmin=0, vmax=max(vmax, 1e-6))
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels(m.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels(m.index, fontsize=8)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < max(vmax, 1e-6) * 0.6 else "black")
    ax.set_title("Cross-trace sensitivity — max output swing % per parameter", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="max swing % of baseline")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "cross_trace_sensitivity.png"))
    plt.close(fig)


def make_plots(runs_csv, bundles, spec, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25})
    df = pd.read_csv(runs_csv, dtype={"trace": str})
    if "trace" not in df.columns:
        df["trace"] = "(single)"
    spec_by_id = {s["id"]: s for s in spec}
    multi = len(bundles) > 1
    for tr, (summary, pivot, base) in bundles.items():
        plots_dir = os.path.join(outdir, "plots", _safe(tr)) if multi else os.path.join(outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        dtr = df[df["trace"].astype(str) == str(tr)]
        _render_trace_plots(dtr, summary, pivot, base, spec, spec_by_id, plots_dir, plt)
    if multi:
        _cross_trace_plot(bundles, os.path.join(outdir, "plots"), plt)
    print(f"Plots: {os.path.join(outdir, 'plots')}")


def _render_trace_plots(df, summary, pivot, base, spec, spec_by_id, plots_dir, plt):
    # 1) Tornado plot per primary metric: params ranked by swing %.
    for metric in PRIMARY_METRICS:
        sub = summary[(summary["metric"] == metric) & summary["swing_pct"].notna()]
        sub = sub[sub["swing_pct"] > 0].sort_values("swing_pct")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(sub) + 1)))
        colors = ["#4C78A8" if g == "alpss" else "#E45756" if g == "spade_spall"
                  else "#54A24B" for g in sub["group"]]
        ax.barh(sub["param"], sub["swing_pct"], color=colors)
        for y, (v, e) in enumerate(zip(sub["swing_pct"], sub["max_abs_elasticity"])):
            lbl = f"{v:.1f}%" + (f"  (E={e:.2f})" if pd.notna(e) else "")
            ax.text(v, y, "  " + lbl, va="center", fontsize=7.5)
        b = pd.to_numeric(base.get(metric), errors="coerce")
        ax.set_title(f"OAT sensitivity — {LABELS[metric]}\nbaseline = {b:.4g}", fontsize=10)
        ax.set_xlabel("Output swing across tested range (% of baseline)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#4C78A8", "#E45756", "#54A24B"]]
        ax.legend(handles, ["ALPSS", "SPADE spall", "SPADE HEL"], fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"tornado_{metric.replace('^','').replace('/','_')}.png"))
        plt.close(fig)

    # 2) Heatmap: parameter x primary metric, colored by swing %.
    hm = pivot.drop(columns=["max_swing_pct"], errors="ignore").copy()
    if not hm.empty:
        fig, ax = plt.subplots(figsize=(1.6 * len(hm.columns) + 3, 0.34 * len(hm) + 2))
        data = hm.values.astype(float)
        vmax = np.nanpercentile(data, 95) if np.isfinite(data).any() else 1.0
        im = ax.imshow(np.nan_to_num(data, nan=0.0), aspect="auto",
                       cmap="magma", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_xticks(range(len(hm.columns)))
        ax.set_xticklabels([LABELS[c] for c in hm.columns], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(hm.index)))
        ax.set_yticklabels(hm.index, fontsize=8)
        for i in range(len(hm.index)):
            for j in range(len(hm.columns)):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v < max(vmax, 1e-6) * 0.6 else "black")
        ax.set_title("Parameter sensitivity heatmap — output swing (% of baseline)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7, label="swing % of baseline")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "sensitivity_heatmap.png"))
        plt.close(fig)

    # 3) Response curves: each primary metric vs param value, small multiples,
    #    one figure per parameter group. Only numeric params.
    for group in ["alpss", "spade_spall", "spade_hel"]:
        params = [s["id"] for s in spec if s["group"] == group and s["kind"] == "num"]
        params = [p for p in params if (df["param"] == p).any()]
        if not params:
            continue
        ncol = 4
        nrow = int(np.ceil(len(params) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
        for idx, p in enumerate(params):
            ax = axes[idx // ncol][idx % ncol]
            s = spec_by_id[p]
            g = df[df["param"] == p].copy()
            g["pv"] = pd.to_numeric(g["param_value_num"], errors="coerce")
            g = g.sort_values("pv")
            base_num = s.get("baseline")
            # Plot the two headline metrics normalized to baseline (%) on shared axis.
            for metric, col in [("Spall_Strength_GPa_Final", "#E45756"),
                                ("HEL_GPa", "#54A24B"),
                                ("Peak_Shock_Stress_GPa_Final", "#4C78A8")]:
                b = pd.to_numeric(base.get(metric), errors="coerce")
                if pd.isna(b) or b == 0:
                    continue
                pv = list(g["pv"].values)
                mv = list(pd.to_numeric(g[metric], errors="coerce").values)
                if base_num is not None:
                    pv.append(base_num)
                    mv.append(b)
                order = np.argsort(pv)
                pv = np.array(pv)[order]
                mv = (np.array(mv)[order] / b - 1.0) * 100.0
                ax.plot(pv, mv, "o-", ms=3, lw=1.2, color=col, label=LABELS[metric].split(" (")[0])
            if base_num is not None:
                ax.axvline(base_num, color="0.5", ls="--", lw=0.8)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_title(p, fontsize=8)
            ax.tick_params(labelsize=7)
        for j in range(len(params), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        axes[0][0].legend(fontsize=6.5, loc="best")
        fig.suptitle(f"Response curves — {group}  (% change from baseline; dashed = baseline)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(plots_dir, f"response_curves_{group}.png"))
        plt.close(fig)


# -----------------------------------------------------------------------------
def trace_label(path):
    """Short, unique-ish label for a trace from its filename.

    Prefers the trailing run number (e.g. C1--20251022--00042.csv -> 00042);
    falls back to the full stem if no numeric tail is found.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.split("--")[-1].split("_")[-1]
    return tail if tail and any(c.isdigit() for c in tail) else stem


def resolve_input_files(args, cfg):
    """Return a list of (label, abspath) traces from the various input flags.

    Precedence: --input-files > --input-dir(+--input-glob) > --input-file >
    config input_dir + default 0001. Labels are de-duplicated so two files that
    share a run number stay distinct.
    """
    paths = []
    if args.input_files:
        paths = [os.path.abspath(p) for p in args.input_files]
    elif args.input_dir:
        base = os.path.abspath(args.input_dir)
        paths = sorted(glob.glob(os.path.join(base, args.input_glob or "*.csv")))
    elif args.input_file:
        paths = [os.path.abspath(args.input_file)]
    else:
        input_dir = cfg.get("cli_settings", {}).get("input_dir")
        cand = os.path.join(input_dir, DEFAULT_TEST_FILE) if input_dir else None
        if cand and os.path.exists(cand):
            paths = [cand]

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("Input file(s) not found:\n  " + "\n  ".join(missing))
    if not paths:
        raise SystemExit("No input traces resolved. Pass --input-file / --input-files / --input-dir.")

    traces, seen = [], {}
    for p in paths:
        lab = trace_label(p)
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab}#{seen[lab]}"
        else:
            seen[lab] = 0
        traces.append((lab, p))
    return traces


def main():
    ap = argparse.ArgumentParser(description="HELIX OAT parameter sensitivity analysis")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "helix_master_config.yml"))
    ap.add_argument("--input-file", default=None,
                    help="single PDV file to analyse (default: 0001 from config input_dir)")
    ap.add_argument("--input-files", nargs="+", default=None,
                    help="explicit list of PDV files; each is swept over all parameters")
    ap.add_argument("--input-dir", default=None,
                    help="directory of PDV files (combine with --input-glob)")
    ap.add_argument("--input-glob", default=None,
                    help="glob applied inside --input-dir (default '*.csv')")
    ap.add_argument("--param-folder", default=None,
                    help="experiment metadata folder for material/energy enrichment "
                         "(overrides config cli_settings.param_folder; pass 'none' to disable)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--groups", default=None, help="comma list: alpss,spade_spall,spade_hel")
    ap.add_argument("--params", default=None, help="comma list of parameter ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-run-outputs", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", default=None,
                    help="path to an existing outdir; recompute summary+plots and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("cli_settings", {})

    # Directory overrides — let the harness run against any dataset without
    # editing the master config. Input dir is consumed in resolve_input_files;
    # output dir below; the param (metadata) folder is injected into the config
    # each per-run pipeline reads.
    if args.input_dir:
        cfg["cli_settings"]["input_dir"] = os.path.abspath(args.input_dir)
    if args.param_folder is not None:
        if str(args.param_folder).strip().lower() in ("none", "null", ""):
            cfg["cli_settings"]["param_folder"] = None
            # With no metadata folder, don't drop traces whose material is unknown.
            cfg.setdefault("spade_config", {})["skip_unknown_material_traces"] = False
        else:
            pf = os.path.abspath(args.param_folder)
            if not os.path.isdir(pf):
                raise SystemExit(f"--param-folder not found: {pf}")
            cfg["cli_settings"]["param_folder"] = pf

    spec = build_param_spec()
    # Fill baselines from config.
    for s in spec:
        s["baseline"] = cfg.get(s["section"], {}).get(s["keys"][0])

    # Filters.
    if args.groups:
        keep = set(args.groups.split(","))
        spec = [s for s in spec if s["group"] in keep]
    if args.params:
        keep = set(args.params.split(","))
        spec = [s for s in spec if s["id"] in keep]

    if args.analyze_only:
        outdir = os.path.abspath(args.analyze_only)
        runs_csv = os.path.join(outdir, "helix_sensitivity_runs.csv")
        if not os.path.exists(runs_csv):
            raise SystemExit(f"No runs CSV in {outdir}")
        bundles = compute_summary(runs_csv, spec, outdir)
        make_plots(runs_csv, bundles, spec, outdir)
        return

    outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(
        REPO_ROOT, "sensitivity_analysis", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir, exist_ok=True)
    traces = resolve_input_files(args, cfg)

    print("=" * 70)
    print("HELIX OAT Parameter Sensitivity Analysis")
    print("=" * 70)
    print(f"Config     : {args.config}")
    print(f"Traces     : {len(traces)}  ->  {', '.join(l for l, _ in traces)}")
    print(f"Param folder: {cfg['cli_settings'].get('param_folder')}")
    print(f"Output dir : {outdir}")
    print(f"Parameters : {len(spec)}  groups={sorted({s['group'] for s in spec})}")
    print("=" * 70)

    runs_csv = run_sweep(args, cfg, spec, outdir, traces)
    if args.dry_run:
        return
    bundles = compute_summary(runs_csv, spec, outdir)
    make_plots(runs_csv, bundles, spec, outdir)

    for tr, (summary, pivot, base) in bundles.items():
        print(f"\nTop movers — trace {tr} (max swing % across primary metrics):")
        for name, v in pivot["max_swing_pct"].dropna().head(8).items():
            print(f"   {name:32s} {v:6.1f}%")
    print(f"\nAll outputs in: {outdir}")


if __name__ == "__main__":
    main()
