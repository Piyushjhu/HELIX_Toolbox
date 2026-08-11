"""DNS (Did Not Spall) detection and spall parameter extraction.

Extracted from ``AnalysisThread.detect_dns_and_process_spall`` in the
original monolith.  Pure-function version with no Qt dependency.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger("helix")


def _default_result(base_name: str) -> Dict[str, Any]:
    return {
        "Filename": base_name,
        "Spall_Strength_GPa": np.nan,
        "Spall_Strength_Unc_GPa": np.nan,
        "Spall_OK": False,
        "Spall_StrainRate_s^-1": np.nan,
        "First_Maxima_m_s": np.nan,
        "Minima_m_s": np.nan,
        "Second_Maxima_m_s": np.nan,
        "Pullback_Velocity_m_s": np.nan,
        "Pullback_Velocity_Unc_m_s": np.nan,
        "Processing_Status": "Failed",
        "DNS_Classification": "Unknown",
    }


def detect_dns_and_process_spall(
    file_path: str,
    base_name: str,
    density: float,
    acoustic_velocity: float,
    threshold_velocity: float,
    spall_start_time: float,
    spall_end_time: float,
    analysis_model: str = "max_min",
    plot_path: Optional[str] = None,
    progress_callback=None,
    **spade_kwargs,
) -> Dict[str, Any]:
    """Detect DNS and compute spall parameters.

    Parameters
    ----------
    file_path : str
        Path to the smoothed velocity CSV (time_s, velocity, uncertainty).
    base_name : str
        Filename stem for logging/plotting.
    density : float
        Material density in kg/m³.
    acoustic_velocity : float
        Bulk wave speed in m/s.
    threshold_velocity : float
        Velocity threshold for shock arrival detection (m/s).
    spall_start_time, spall_end_time : float
        Spall analysis window in nanoseconds (relative to shock arrival).
    analysis_model : str
        SPADE analysis model (``"max_min"`` / ``"hybrid"`` / etc.).
    plot_path : str or None
        If given, save a diagnostic plot there.
    progress_callback : callable or None
        ``fn(msg: str)`` for progress/debug messages.

    Returns
    -------
    dict
        Result dictionary with spall strength, DNS classification, etc.
    """
    def _emit(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    results = _default_result(base_name)

    try:
        # Step 1: Load CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            results["Processing_Status"] = f"Failed: Could not load file: {exc}"
            return results

        # Detect header
        if len(df) > 0:
            first_row_str = " ".join(str(x).lower() for x in df.iloc[0].values[:3])
            if any(kw in first_row_str for kw in ("time", "velocity", "uncertainty")):
                df = pd.read_csv(file_path, header=0)
            else:
                df = pd.read_csv(file_path, header=None)

        if len(df.columns) < 3:
            results["Processing_Status"] = "Failed: Insufficient columns (< 3)"
            return results

        time_s = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        velocity = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        uncertainty = pd.to_numeric(df.iloc[:, 2], errors="coerce").values

        if len(time_s) < 20:
            results["Processing_Status"] = "Failed: Insufficient data points (< 20)"
            return results

        # Step 1b: Uncertainty filter
        max_vel = np.nanmax(np.abs(velocity))
        rel_unc = np.abs(uncertainty) / max(max_vel, 1e-9)
        vel_clean = velocity.copy()
        vel_clean[rel_unc >= 1.0] = np.nan

        # Step 2: Trace alignment
        valid_mask = ~np.isnan(vel_clean)
        if not np.any(valid_mask):
            results["Processing_Status"] = "Failed: No valid velocity data after filtering"
            return results

        vel_valid = vel_clean[valid_mask]
        time_valid = time_s[valid_mask]

        threshold_idx = np.where(vel_valid >= threshold_velocity)[0]
        if len(threshold_idx) == 0:
            results["Processing_Status"] = "Failed: No shock arrival detected"
            return results

        t0 = time_valid[threshold_idx[0]]
        t_aligned_ns = (time_s - t0) * 1e9

        # Step 3: Window extraction
        window_mask = (
            ~np.isnan(vel_clean)
            & (t_aligned_ns >= spall_start_time)
            & (t_aligned_ns <= spall_end_time)
        )
        if np.sum(window_mask) < 20:
            results["Processing_Status"] = "Failed: Insufficient data in spall window"
            return results

        time_window = t_aligned_ns[window_mask]
        vel_window = vel_clean[window_mask]
        uncert_window = uncertainty[window_mask]

        # Step 4: DNS structural check
        prominence = np.nanstd(vel_window) * 0.1
        peaks, _ = find_peaks(vel_window, prominence=prominence)
        valleys, _ = find_peaks(-vel_window, prominence=prominence)

        dns_reason = None
        valleys_after_peak = np.array([], dtype=int)
        peaks_after_valley = np.array([], dtype=int)

        if len(peaks) == 0 or len(valleys) == 0:
            dns_reason = "No clear peak/valley structure"
        else:
            first_peak_idx = peaks[0]
            valleys_after_peak = valleys[valleys > first_peak_idx]
            if len(valleys_after_peak) == 0:
                dns_reason = "No pullback after initial rise"
            else:
                first_valley_idx = valleys_after_peak[0]
                peaks_after_valley = peaks[peaks > first_valley_idx]
                if len(peaks_after_valley) == 0:
                    dns_reason = "No re-acceleration after pullback"

        if dns_reason:
            results["Spall_Strength_GPa"] = "DNS"
            results["Spall_OK"] = False
            results["Processing_Status"] = f"DNS: {dns_reason}"
            results["DNS_Classification"] = dns_reason
            if plot_path:
                _plot_dns(
                    plot_path, time_window, vel_window, uncert_window,
                    peaks[0] if len(peaks) > 0 else None,
                    valleys_after_peak[0] if len(valleys_after_peak) > 0 else None,
                    base_name, _emit,
                )
            return results

        # Step 5: Valid spall — extract key velocities
        first_peak_idx = peaks[0]
        first_valley_idx = valleys_after_peak[0]
        second_peak_idx = peaks_after_valley[0]

        first_peak_vel = vel_window[first_peak_idx]
        first_valley_vel = vel_window[first_valley_idx]
        second_peak_vel = vel_window[second_peak_idx]

        peak_unc = uncert_window[first_peak_idx] if first_peak_idx < len(uncert_window) else 0
        valley_unc = uncert_window[first_valley_idx] if first_valley_idx < len(uncert_window) else 0
        pullback_unc = (
            np.sqrt(peak_unc**2 + valley_unc**2)
            if np.isfinite(peak_unc) and np.isfinite(valley_unc)
            else np.nan
        )

        results["First_Maxima_m_s"] = first_peak_vel
        results["Minima_m_s"] = first_valley_vel
        results["Second_Maxima_m_s"] = second_peak_vel
        results["Pullback_Velocity_m_s"] = abs(first_peak_vel - first_valley_vel)
        results["Pullback_Velocity_Unc_m_s"] = pullback_unc

        # Step 6: SPADE analysis
        from SPADE.spall_analysis_release.spall_analysis.data_processing import (
            calculate_spall_parameters,
        )

        result_dict, spade_lines_info, spade_intersections = calculate_spall_parameters(
            time_ns=time_window,
            velocity_ms=vel_window,
            uncertainty_ms=uncert_window,
            density=density,
            acoustic_velocity=acoustic_velocity,
            analysis_model=analysis_model,
            plot_path=plot_path,
            **{k: v for k, v in spade_kwargs.items()
               if k not in ("density", "acoustic_velocity", "analysis_model")},
        )

        # Extract spall strength
        spall_strength = np.nan
        for key in result_dict:
            kl = key.lower()
            if "spall" in kl and "strength" in kl and "gpa" in kl and "unc" not in kl and "err" not in kl:
                val = result_dict[key]
                if isinstance(val, str) and val.upper() == "DNS":
                    spall_strength = "DNS"
                else:
                    spall_strength = float(val) if pd.notna(val) else np.nan
                break

        # Extract uncertainty
        spall_unc = np.nan
        for key in result_dict:
            kl = key.lower()
            if ("unc" in kl or "err" in kl) and "spall" in kl and "gpa" in kl:
                spall_unc = float(result_dict[key]) if pd.notna(result_dict[key]) else np.nan
                break

        # Uncertainty fallback
        if pd.isna(spall_unc) and np.isfinite(pullback_unc) and np.isfinite(density) and np.isfinite(acoustic_velocity):
            spall_unc = 0.5 * density * acoustic_velocity * pullback_unc / 1e9

        # Strain rate
        strain_rate = result_dict.get("Strain Rate (s^-1)", result_dict.get("Strain_Rate_s^-1", np.nan))

        # Classification
        if result_dict.get("Processing Status") == "Success":
            results["Spall_OK"] = True
            results["Processing_Status"] = "Success"
        else:
            results["Processing_Status"] = result_dict.get("Processing Status", "Failed: SPADE analysis failed")

        results["Spall_Strength_GPa"] = spall_strength
        results["Spall_Strength_Unc_GPa"] = spall_unc
        results["Spall_StrainRate_s^-1"] = strain_rate
        results["DNS_Classification"] = "Valid Spall" if results["Spall_OK"] else "Failed"

    except Exception as exc:
        import traceback
        results["Processing_Status"] = f"Failed: {exc}"
        results["DNS_Classification"] = "Error"
        _emit(f"Error in DNS detection for {base_name}: {exc}")
        _emit(traceback.format_exc())

    return results


def _plot_dns(plot_path, time_window, vel_window, uncert_window, peak_idx, valley_idx, base_name, emit):
    """Quick DNS diagnostic plot."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_window, vel_window, "b-", linewidth=1.5, alpha=0.7, label="Velocity")
        if uncert_window is not None and len(uncert_window) == len(vel_window):
            ax.fill_between(time_window, vel_window - uncert_window, vel_window + uncert_window,
                            alpha=0.2, color="blue", label="Uncertainty")
        if peak_idx is not None and peak_idx < len(time_window):
            ax.plot(time_window[peak_idx], vel_window[peak_idx], "ro", markersize=10,
                    label=f"Peak: {vel_window[peak_idx]:.1f} m/s")
        if valley_idx is not None and valley_idx < len(time_window):
            ax.plot(time_window[valley_idx], vel_window[valley_idx], "go", markersize=10,
                    label=f"Valley: {vel_window[valley_idx]:.1f} m/s")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(f"DNS Detection: {base_name}", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        emit(f"Warning: Could not generate DNS plot: {exc}")
