"""Run ALPSS in batch mode over a list of PDV files."""
from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("helix")


def run_alpss_batch(
    input_files: List[str],
    output_dir: str,
    alpss_params: Dict[str, Any],
    param_data: Optional[Dict[str, Dict]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Run ``alpss_main`` on each file in *input_files*.

    Returns (successful_files, failed_files) where failed_files is a list
    of ``(path, error_message)`` tuples.
    """
    from alpss.alpss_main import alpss_main
    import re

    def _emit(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    os.makedirs(output_dir, exist_ok=True)
    successful: List[str] = []
    failed: List[Tuple[str, str]] = []
    total_time = 0.0

    for i, input_file in enumerate(input_files):
        basename = os.path.basename(input_file)
        _emit(f"ALPSS [{i + 1}/{len(input_files)}]: {basename}")
        gc.collect()

        params = alpss_params.copy()
        params["filepath"] = os.path.abspath(input_file)
        params["out_files_dir"] = output_dir

        # Attach experiment metadata if available
        if param_data:
            stem = os.path.splitext(basename)[0]
            exp_info = _match_param_data(stem, param_data)
            params["experiment_info"] = exp_info
        else:
            params["experiment_info"] = {}

        t0 = time.time()
        try:
            alpss_main(**params)
            successful.append(input_file)
            dt = time.time() - t0
            total_time += dt
            _emit(f"  done in {dt:.1f}s")
        except Exception as exc:
            failed.append((input_file, str(exc)))
            _emit(f"  FAILED: {exc}")

    avg = total_time / len(successful) if successful else 0
    _emit(
        f"ALPSS batch: {len(successful)}/{len(input_files)} succeeded "
        f"({total_time:.1f}s total, {avg:.1f}s avg)"
    )
    return successful, failed


def _match_param_data(stem: str, param_data: Dict[str, Dict]) -> Dict[str, Any]:
    """Smart filename → param-data matching."""
    import re as _re

    if stem in param_data:
        return param_data[stem]
    m = _re.search(r"(\d{8}--\d{5})", stem)
    if m:
        ds = m.group(1)
        for key, val in param_data.items():
            if ds in str(key):
                return val
    for key, val in param_data.items():
        if stem in str(key) or str(key) in stem:
            return val
    return {}
