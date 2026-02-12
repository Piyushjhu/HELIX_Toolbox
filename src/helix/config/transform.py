"""Parameter transformation: raw config → analysis-ready dicts.

Bridges the gap between JSON config files (which store raw user values)
and what ALPSS / SPADE functions actually expect.
"""
from __future__ import annotations

from typing import Any, Dict


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


# ---------------------------------------------------------------------------
# ALPSS parameter transform
# ---------------------------------------------------------------------------

def transform_alpss_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw ALPSS config into the dict expected by ``alpss_main``."""
    params = raw.copy()

    # --- save_all_plots dropdown → flags ---
    if "save_all_plots" in params and "save_all_plots_enabled" not in params:
        dropdown = params["save_all_plots"]
        plots_enabled = dropdown in ("subfolder", "main_dir")
        params["save_all_plots_enabled"] = "yes" if plots_enabled else "no"
        params["save_plots_in_subfolder"] = dropdown == "subfolder"
    elif "save_all_plots_enabled" in params:
        plots_enabled = params["save_all_plots_enabled"] in ("yes", True)
        params.setdefault(
            "save_plots_in_subfolder",
            params.get("save_all_plots") == "subfolder",
        )
    else:
        plots_enabled = False
        params["save_all_plots_enabled"] = "no"
        params["save_plots_in_subfolder"] = False

    for flag, default in [("save_combined_plot", True), ("save_iq_start_time_plot", False)]:
        params[flag] = bool(params.get(flag, default)) and plots_enabled

    # --- blur_kernel tuple ---
    if "blur_kernel" not in params:
        params["blur_kernel"] = (
            int(params.pop("blur_kernel_x", 5)),
            int(params.pop("blur_kernel_y", 5)),
        )
    elif isinstance(params["blur_kernel"], list):
        params["blur_kernel"] = tuple(params["blur_kernel"])

    # --- defaults for keys ALPSS expects ---
    params.setdefault("plot_figsize", (30, 10))
    if isinstance(params["plot_figsize"], list):
        params["plot_figsize"] = tuple(params["plot_figsize"])
    params.setdefault("plot_dpi", 300)
    params.setdefault("smart_selection_enabled", False)

    # --- string coercion for certain fields ---
    for key, default in [
        ("start_time_user", "none"),
        ("window", "hann"),
        ("cmap", "viridis"),
        ("display_plots", "no"),
        ("spall_calculation", "yes"),
        ("save_data", "yes"),
    ]:
        val = params.get(key, default)
        if val is None:
            val = default
        params[key] = str(val) if not isinstance(val, str) else val

    # --- numeric type coercion ---
    _INT_PARAMS = {
        "header_lines", "pb_neighbors", "pb_idx_correction",
        "rc_neighbors", "rc_idx_correction", "nperseg", "noverlap",
        "nfft", "smoothing_window", "order", "plot_dpi",
    }
    for key in list(params):
        val = params[key]
        if isinstance(val, str):
            try:
                val = float(val)
                params[key] = val
            except ValueError:
                continue
        if key in _INT_PARAMS and isinstance(val, (int, float)):
            params[key] = int(val)

    # --- boolean coercion ---
    for key in [
        "use_notch_filter", "save_velocity_csv", "save_velocity_smooth_csv",
        "save_velocity_uncert_csv", "save_velocity_smooth_uncert_csv",
        "save_results_csv", "save_noise_csv",
    ]:
        if key in params:
            params[key] = _to_bool(params[key])

    return params


# ---------------------------------------------------------------------------
# SPADE parameter transform
# ---------------------------------------------------------------------------

def transform_spade_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw SPADE config into the dict expected by the pipeline."""
    params = (raw or {}).copy()

    # experiment toggles
    params["velocity_shots_enabled"] = _to_bool(
        params.get("velocity_shots_enabled", params.get("experiment_velocity_shots", True)),
        default=True,
    )
    params["spall_analysis_enabled"] = _to_bool(
        params.get("spall_analysis_enabled", params.get("experiment_spall_analysis", False)),
        default=False,
    )
    params["hel_detection_enabled"] = _to_bool(
        params.get("hel_detection_enabled", params.get("experiment_hel_detection", False)),
        default=False,
    )

    # summary table flag
    if "save_summary_table" not in params:
        params["save_summary_table"] = _to_bool(params.get("save_summary", True), default=True)

    # smooth window naming
    if "smooth_window" not in params and "spade_smooth_window" in params:
        params["smooth_window"] = params["spade_smooth_window"]

    # signal_length → signal_length_ns
    if "signal_length_ns" not in params:
        sl = params.pop("signal_length", None)
        if sl is None or (isinstance(sl, str) and sl.strip().lower().startswith("full")):
            params["signal_length_ns"] = None
        else:
            try:
                params["signal_length_ns"] = float(sl)
            except (ValueError, TypeError):
                params["signal_length_ns"] = None
    else:
        params.pop("signal_length", None)

    for key, default in [
        ("plot_individual", False),
        ("save_summary_table", True),
        ("show_plots", False),
        ("generate_all_velocity_plot", True),
        ("include_uncert_bands", True),
        ("auto_calculate_limits", True),
    ]:
        if key in params:
            params[key] = _to_bool(params[key], default=default)

    return params
