"""Top-level pipeline orchestrator: ALPSS → velocity summary → SPADE."""
from __future__ import annotations

import logging
import os
import glob as _glob
from typing import Any, Callable, Dict, List, Optional

from helix.materials.resolver import MaterialResolver
from helix.pipeline.alpss_runner import run_alpss_batch
from helix.pipeline.spade_runner import run_spade_batch
from helix.spade.velocity_summary import generate_velocity_summary

logger = logging.getLogger("helix")


def run_pipeline(
    input_files: List[str],
    output_dir: str,
    alpss_params: Dict[str, Any],
    spade_params: Dict[str, Any],
    analysis_mode: str = "both",
    param_data: Optional[Dict[str, Dict]] = None,
    material_properties: Optional[Dict[str, Dict]] = None,
    spade_input_files: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the full HELIX pipeline.

    Parameters
    ----------
    input_files : list of str
        Raw PDV CSV files for ALPSS.
    output_dir : str
        Root output directory.
    alpss_params, spade_params : dict
        Already-transformed parameter dicts.
    analysis_mode : str
        ``"alpss_only"`` | ``"spade_only"`` | ``"both"``
    param_data : dict or None
        Filename → experiment-metadata mapping.
    material_properties : dict or None
        Config-level material overrides.
    spade_input_files : list or None
        Explicit velocity files for manual SPADE mode.
    progress_callback : callable or None
        ``fn(msg)`` for progress messages.

    Returns
    -------
    dict with keys ``successful_files``, ``failed_files``, ``velocity_summary``,
    ``spall_summary``.
    """
    def _emit(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    os.makedirs(output_dir, exist_ok=True)

    resolver = MaterialResolver(
        config_materials=material_properties,
        default_density=spade_params.get("density", alpss_params.get("density")),
        default_acoustic_velocity=spade_params.get("acoustic_velocity", alpss_params.get("C0")),
    )

    result: Dict[str, Any] = {
        "successful_files": [],
        "failed_files": [],
        "velocity_summary": None,
        "spall_summary": None,
    }

    # --- Phase 1: ALPSS ---
    if analysis_mode != "spade_only" and input_files:
        _emit("=" * 60)
        _emit("PHASE 1: ALPSS Velocity Processing")
        _emit("=" * 60)
        successful, failed = run_alpss_batch(
            input_files, output_dir, alpss_params,
            param_data=param_data,
            progress_callback=_emit,
        )
        result["successful_files"] = successful
        result["failed_files"] = failed

    # --- Phase 2: Velocity Summary ---
    if analysis_mode != "alpss_only":
        _emit("=" * 60)
        _emit("PHASE 2: Velocity Summary + HEL Detection")
        _emit("=" * 60)
        result["velocity_summary"] = generate_velocity_summary(
            output_dir, spade_params,
            param_data=param_data,
            material_resolver=resolver,
            progress_callback=_emit,
        )

    # --- Phase 3: SPADE Spall Analysis ---
    if analysis_mode != "alpss_only" and spade_params.get("spall_analysis_enabled", False):
        _emit("=" * 60)
        _emit("PHASE 3: SPADE Spall Analysis")
        _emit("=" * 60)

        if spade_input_files:
            vel_files = spade_input_files
        else:
            vel_files = sorted(
                _glob.glob(os.path.join(output_dir, "*--vel-smooth-with-uncert.csv"))
            )

        if vel_files:
            result["spall_summary"] = run_spade_batch(
                vel_files, output_dir, spade_params,
                material_resolver=resolver,
                param_data=param_data,
                progress_callback=_emit,
            )
        else:
            _emit("No velocity files found for SPADE analysis.")

    _emit("=" * 60)
    _emit("Pipeline complete.")
    _emit("=" * 60)
    return result
