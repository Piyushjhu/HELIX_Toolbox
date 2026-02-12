"""Run SPADE spall analysis in batch mode over velocity files."""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from helix.spade.dns_detection import detect_dns_and_process_spall
from helix.materials.resolver import MaterialResolver

logger = logging.getLogger("helix")


def run_spade_batch(
    vel_files: List[str],
    output_dir: str,
    spade_params: Dict[str, Any],
    material_resolver: Optional[MaterialResolver] = None,
    param_data: Optional[Dict[str, Dict]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Run SPADE spall analysis on a list of velocity CSV files.

    Returns a DataFrame with one row per file.
    """
    def _emit(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    if material_resolver is None:
        material_resolver = MaterialResolver(
            default_density=spade_params.get("density"),
            default_acoustic_velocity=spade_params.get("acoustic_velocity"),
        )

    spade_dir = os.path.join(output_dir, "SPADE_analysis")
    os.makedirs(spade_dir, exist_ok=True)

    threshold_vel = spade_params.get("threshold_velocity_ms",
                                     spade_params.get("align_velocity_threshold_ms", 10.0))
    spall_start = spade_params.get("spall_start_time_ns", 10.0)
    spall_end = spade_params.get("spall_end_time_ns", 100.0)
    analysis_model = spade_params.get("analysis_model", "hybrid")
    plot_individual = spade_params.get("plot_individual", False)

    rows = []
    for vel_file in vel_files:
        fname = os.path.basename(vel_file)
        base = fname
        for suffix in ("--vel-smooth-with-uncert.csv", "--velocity--smooth.csv", ".csv"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        _emit(f"SPADE: {base}")

        # Resolve material
        material = "Unknown"
        if param_data:
            from helix.spade.velocity_summary import _get_param_data_for_file
            exp_info = _get_param_data_for_file(base, param_data)
            for mk in ("Sample material", "Sample Material", "Material", "material",
                        "Flyer_material", "Flyer Material"):
                val = exp_info.get(mk)
                if val and str(val).strip().lower() != "nan":
                    material = str(val).strip()
                    break

        mat = material_resolver.resolve(material)
        density = mat["density"]
        acoustic_vel = mat["bulk_wave_speed"]

        plot_path = os.path.join(spade_dir, f"{base}--spall.png") if plot_individual else None

        result = detect_dns_and_process_spall(
            file_path=vel_file,
            base_name=base,
            density=density,
            acoustic_velocity=acoustic_vel,
            threshold_velocity=threshold_vel,
            spall_start_time=spall_start,
            spall_end_time=spall_end,
            analysis_model=analysis_model,
            plot_path=plot_path,
            progress_callback=_emit,
        )
        result["Material"] = material
        rows.append(result)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(spade_dir, "spall_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    _emit(f"SPADE summary saved: {summary_path} ({len(rows)} files)")
    return summary_df
