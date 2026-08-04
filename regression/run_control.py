#!/usr/bin/env python3
"""
run_control.py — regression control for HELIX Toolbox
=====================================================
Runs a fixed, known control trace through the pipeline (deterministic SPADE-only on
a bundled velocity file) and compares the resulting per-shot values against a stored
baseline. Every run is appended, with a timestamp and the current git commit, to
history.jsonl — so the file accumulates a timestamped version history of the control's
values as the code evolves.

Control trace: C1--20251022--00001 (Brass) — the shot whose spall strength
(2.892868603294747 GPa) was independently reproduced bit-for-bit.

Usage
-----
    # compare against the current baseline (normal use after a code change)
    helix_toolbox_env/bin/python3 regression/run_control.py

    # (re)create the baseline from the current code — do this only when a change is
    # intentional and reviewed; the previous baseline stays in history.jsonl
    helix_toolbox_env/bin/python3 regression/run_control.py --update-baseline

Options
-------
    --update-baseline   Write the current run's values as the new baseline.
    --rtol FLOAT        Relative tolerance for the comparison (default 1e-6).
    --atol FLOAT        Absolute tolerance for the comparison (default 1e-9).
    --keep-output       Keep the temporary run output directory (for debugging).

Exit code is 0 on PASS, 1 on FAIL (a metric drifted beyond tolerance, or a metric
went missing), so it can gate CI / a pre-commit check.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd
import yaml

# ── Locations ───────────────────────────────────────────────────────────────────
REG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(REG_DIR)
DATA_DIR = os.path.join(REG_DIR, "data")
PARAMS_DIR = os.path.join(DATA_DIR, "params")
CONTROL_CONFIG = os.path.join(REG_DIR, "control_config.yml")
BASELINE = os.path.join(REG_DIR, "baseline.json")
HISTORY = os.path.join(REG_DIR, "history.jsonl")

CONTROL_BASENAME = "C1--20251022--00001"

# Metrics tracked for the control shot. Context fields (Material/Density/...) guard
# against a silent material-resolution change; the rest are the physics outputs.
METRICS = [
    "Material", "Density_kg_m3", "Acoustic_Velocity_m_s",
    "Spall_Strength_GPa", "Spall_Strength_Unc_GPa", "Spall_StrainRate_s^-1",
    "First_Maxima_m_s", "Minima_m_s", "Plateau_Mean_Velocity_m_s",
    "Peak_Shock_Stress_GPa",
    "HEL_GPa", "HEL_StrainRate_s^-1", "HEL_Uncertainty_GPa",
    "Peak_Shock_Time_ns", "RiseTime_ArrivalToPeak_ns",
    "RiseTime_80_20_ns", "RiseTime_90_10_ns", "RiseTime_MaxSlope_ns",
    "PlasticStrainRate_80_20_s^-1", "PlasticStrainRate_90_10_s^-1",
    "PlasticStrainRate_MaxSlope_s^-1",
    "Compressive_StrainRate_Avg_s^-1", "Compressive_StrainRate_Ufs_s^-1",
    "Shock_Velocity_Us_m_s", "Shock_Front_Width_um",
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_pipeline(out_dir):
    """Resolve the frozen control config to absolute paths and run the CLI runner."""
    cfg = yaml.safe_load(open(CONTROL_CONFIG))

    def resolve(v):
        if not isinstance(v, str):
            return v
        return (v.replace("@DATA@", DATA_DIR)
                 .replace("@PARAMS@", PARAMS_DIR)
                 .replace("@OUTPUT@", out_dir))

    for section in ("cli_settings", "post_processing_config"):
        for k, v in (cfg.get(section) or {}).items():
            cfg[section][k] = resolve(v)

    os.makedirs(os.path.join(out_dir, "SPADE_analysis"), exist_ok=True)
    tmp_cfg = os.path.join(out_dir, "_control_config_resolved.yml")
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_DIR, "helix_cli_runner.py"), "--config", tmp_cfg],
        cwd=REPO_DIR, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise RuntimeError(f"pipeline run failed (exit {proc.returncode})")
    return proc


def extract_values(out_dir):
    """Pull the control shot's metric values from the produced master summary."""
    sp = os.path.join(out_dir, "SPADE_analysis")
    import glob
    masters = sorted(glob.glob(os.path.join(sp, "*Data_Summary.csv")))
    if not masters:
        # fall back to spall_summary if the master wasn't written for some reason
        masters = [p for p in [os.path.join(sp, "spall_summary.csv")] if os.path.exists(p)]
    if not masters:
        raise FileNotFoundError(f"no summary CSV produced in {sp}")
    df = pd.read_csv(masters[0])
    fname_col = df.columns[0]
    row = df[df[fname_col].astype(str).str.contains(CONTROL_BASENAME, na=False)]
    if row.empty:
        raise ValueError(f"control shot {CONTROL_BASENAME} not found in {masters[0]}")
    r = row.iloc[0]
    out = {}
    for m in METRICS:
        if m in df.columns:
            v = r[m]
            try:
                out[m] = None if pd.isna(v) else (float(v) if not isinstance(v, str) else v)
            except (TypeError, ValueError):
                out[m] = str(v)
        else:
            out[m] = "<absent>"
    return out, os.path.basename(masters[0])


def compare(current, baseline, rtol, atol):
    """Return (ok, rows) where rows = [(metric, base, cur, status, delta)]."""
    rows = []
    ok = True
    for m in METRICS:
        cur = current.get(m, "<absent>")
        base = baseline.get(m, "<absent>")
        status, delta = "ok", ""
        if base == "<absent>" and cur == "<absent>":
            status = "ok"
        elif isinstance(base, str) or isinstance(cur, str) or base is None or cur is None:
            same = (base == cur)
            status = "ok" if same else "CHANGED"
        else:
            if base == 0:
                same = abs(cur - base) <= atol
            else:
                same = math.isclose(cur, base, rel_tol=rtol, abs_tol=atol)
            if not same:
                delta = f"{(cur - base):+.6g} ({(cur - base) / base * 100:+.4f}%)" if base else f"{cur - base:+.6g}"
            status = "ok" if same else "DRIFT"
        if status != "ok":
            ok = False
        rows.append((m, base, cur, status, delta))
    return ok, rows


def main():
    ap = argparse.ArgumentParser(description="HELIX regression control runner")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the current run's values as the new baseline")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--keep-output", action="store_true")
    args = ap.parse_args()

    out_dir = tempfile.mkdtemp(prefix="helix_control_")
    try:
        print(f"[control] running SPADE-only on {CONTROL_BASENAME} ...")
        run_pipeline(out_dir)
        values, master_name = extract_values(out_dir)
    finally:
        if not args.keep_output:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            print(f"[control] kept output: {out_dir}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit = git_commit()

    if args.update_baseline or not os.path.exists(BASELINE):
        reason = "explicit --update-baseline" if args.update_baseline else "no baseline present"
        base_obj = {"created": ts, "git_commit": commit, "master": master_name,
                    "rtol": args.rtol, "atol": args.atol, "metrics": values}
        with open(BASELINE, "w") as f:
            json.dump(base_obj, f, indent=2)
        status = "BASELINE_SET"
        print(f"\n[control] baseline written ({reason}): {BASELINE}")
        for m in METRICS:
            print(f"    {m:32s} = {values[m]}")
        ok = True
        rows = []
    else:
        baseline = json.load(open(BASELINE))
        ok, rows = compare(values, baseline.get("metrics", {}), args.rtol, args.atol)
        status = "PASS" if ok else "FAIL"
        print(f"\n[control] baseline: {baseline.get('git_commit','?')} @ {baseline.get('created','?')}")
        print(f"{'metric':34s} {'baseline':>22s} {'current':>22s}  status")
        print("-" * 92)
        for m, base, cur, st, delta in rows:
            mark = "" if st == "ok" else f"  <-- {st} {delta}"
            print(f"{m:34s} {str(base):>22s} {str(cur):>22s}  {st}{mark}")
        print("-" * 92)
        print(f"[control] {status}")

    # Append a timestamped version entry to the history log (always).
    entry = {"timestamp": ts, "git_commit": commit, "status": status,
             "master": master_name, "values": values}
    if rows:
        entry["drift"] = {m: {"baseline": b, "current": c, "delta": d}
                          for m, b, c, st, d in rows if st != "ok"}
    with open(HISTORY, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[control] appended to {os.path.relpath(HISTORY, REPO_DIR)}")

    sys.exit(0 if status in ("PASS", "BASELINE_SET") else 1)


if __name__ == "__main__":
    main()
