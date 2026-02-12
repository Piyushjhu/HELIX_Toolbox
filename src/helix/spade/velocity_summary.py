"""Generate velocity shots summary from ALPSS output files.

Extracted from ``AnalysisThread.generate_velocity_shots_summary``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from alpss.analysis.hel import hel_detection, HELResult

logger = logging.getLogger("helix")


def _get_param_data_for_file(
    base_name: str, param_data: Optional[Dict[str, Dict]]
) -> Dict[str, Any]:
    """Smart filename→param matching (exact, date-shot, partial)."""
    if not param_data:
        return {}
    if base_name in param_data:
        return param_data[base_name]
    m = re.search(r"(\d{8}--\d{5})", base_name)
    if m:
        date_shot = m.group(1)
        for key, val in param_data.items():
            if date_shot in str(key):
                return val
    for key, val in param_data.items():
        if base_name in str(key) or str(key) in base_name:
            return val
    return {}


def _detect_pdv_column(columns: List[str]) -> Optional[str]:
    normalized = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in columns}
    known = [
        "pdvfilename", "pdvfile", "pdv_file", "pdv_file_name",
        "dvfilename", "dvfile", "filename", "file_name", "file",
    ]
    for variant in known:
        for col, norm in normalized.items():
            if norm == variant:
                return col
    for col in columns:
        low = col.lower()
        if "pdv" in low and "file" in low:
            return col
    return None


def generate_velocity_summary(
    output_dir: str,
    spade_params: Dict[str, Any],
    param_data: Optional[Dict[str, Dict]] = None,
    material_resolver=None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[pd.DataFrame]:
    """Build the velocity_shots_summary.csv from ALPSS output files.

    This collects peak velocity, material info, and optionally runs HEL
    detection on each trace.

    Returns the summary DataFrame (also saved to disk).
    """
    def _emit(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    vel_pattern = "*--vel-smooth-with-uncert.csv"
    import glob as _glob

    vel_files = sorted(_glob.glob(os.path.join(output_dir, vel_pattern)))
    if not vel_files:
        _emit("No velocity files found for summary generation.")
        return None

    hel_enabled = spade_params.get("hel_detection_enabled", False)
    hel_params = {
        "hel_start_ns": spade_params.get("hel_start_time_ns", -5.0),
        "hel_end_ns": spade_params.get("hel_end_time_ns", 8.0),
        "angle_threshold_deg": spade_params.get("hel_angle_threshold_deg", 45.0),
        "min_points": spade_params.get("hel_detection_min_points", 20),
        "min_velocity": spade_params.get("minimum_HEL_velocity_expected", 5.0),
    }

    threshold_vel = spade_params.get("threshold_velocity_ms", spade_params.get("align_velocity_threshold_ms", 10.0))
    mad_enabled = spade_params.get("mad_filter_enabled", True)
    mad_threshold = spade_params.get("mad_filter_threshold", 2.0)

    rows = []
    for vel_file in vel_files:
        fname = os.path.basename(vel_file)
        # strip suffixes to get base name
        base = fname
        for suffix in ("--vel-smooth-with-uncert.csv", "--velocity--smooth.csv", ".csv"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        try:
            df = pd.read_csv(vel_file, header=None)
            if len(df.columns) < 2:
                continue
            time_ns = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
            velocity = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
            uncertainty = (
                pd.to_numeric(df.iloc[:, 2], errors="coerce").values
                if len(df.columns) >= 3
                else np.ones_like(velocity) * 5.0
            )
        except Exception:
            continue

        # Clean NaN
        valid = ~np.isnan(velocity)
        if np.sum(valid) < 10:
            continue
        time_clean = time_ns[valid]
        vel_clean = velocity[valid]
        unc_clean = uncertainty[valid] if uncertainty is not None else np.ones_like(vel_clean) * 5.0

        # Uncertainty filter
        max_vel_abs = np.nanmax(np.abs(vel_clean))
        if max_vel_abs > 1e-9:
            rel_unc = np.abs(unc_clean) / max_vel_abs
            noise_mask = rel_unc < 1.0
            if np.sum(noise_mask) >= 10:
                time_clean = time_clean[noise_mask]
                vel_clean = vel_clean[noise_mask]
                unc_clean = unc_clean[noise_mask]

        peak_velocity = np.nanmax(vel_clean)

        # Material info from param data
        exp_info = _get_param_data_for_file(base, param_data)
        material_keys = [
            "Sample material", "Sample Material", "Sample_Material",
            "sample_material", "Material", "material",
            "Flyer_material", "Flyer Material",
        ]
        material = "Unknown"
        for mk in material_keys:
            val = exp_info.get(mk)
            if val and str(val).strip().lower() != "nan":
                material = str(val).strip()
                break

        # Resolve material properties
        mat_props = {"density": np.nan, "bulk_wave_speed": np.nan, "C_L": np.nan}
        if material_resolver:
            mat_props = material_resolver.resolve(material)

        row = {
            "Filename": base,
            "Material": material,
            "Peak_Velocity_m_s": peak_velocity,
            "Density_kg_m3": mat_props.get("density", np.nan),
            "Acoustic_Velocity_m_s": mat_props.get("bulk_wave_speed", np.nan),
        }

        # Peak shock stress
        rho = mat_props.get("density", np.nan)
        c0 = mat_props.get("bulk_wave_speed", np.nan)
        if np.isfinite(rho) and np.isfinite(c0):
            row["Peak_Shock_Stress_GPa"] = 0.5 * rho * c0 * peak_velocity / 1e9
        else:
            row["Peak_Shock_Stress_GPa"] = np.nan

        # HEL detection (delegated to ALPSS)
        if hel_enabled:
            hel_result = hel_detection(
                time_clean, vel_clean, unc_clean,
                density=rho if np.isfinite(rho) else None,
                acoustic_velocity=c0 if np.isfinite(c0) else None,
                C_L=mat_props.get("C_L") if np.isfinite(mat_props.get("C_L", np.nan)) else None,
                **hel_params,
            )
            row["HEL_Detected"] = hel_result.ok
            row["HEL_Strength_GPa"] = hel_result.strength_gpa
            row["HEL_Uncertainty_GPa"] = hel_result.uncertainty_gpa
            row["HEL_FSV_m_s"] = hel_result.free_surface_velocity
            row["HEL_Time_ns"] = hel_result.time_detection_ns
            row["HEL_Consecutive_Points"] = hel_result.consecutive_points
            row["HEL_Strain_Rate"] = hel_result.strain_rate
        else:
            row["HEL_Detected"] = False

        # Experiment metadata passthrough
        for meta_key in ("Laser_energy", "Laser Energy", "laser_energy",
                         "Waveplate_Angle", "Waveplate Angle", "waveplate_angle",
                         "Row", "Column"):
            val = exp_info.get(meta_key)
            if val is not None and str(val).strip().lower() != "nan":
                row[meta_key] = val

        rows.append(row)
        _emit(f"  Processed {base}: peak={peak_velocity:.1f} m/s, material={material}")

    if not rows:
        _emit("No valid traces for velocity summary.")
        return None

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "velocity_shots_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    _emit(f"Velocity summary saved: {summary_path} ({len(rows)} traces)")
    return summary_df
