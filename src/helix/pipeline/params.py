"""Parameter file loading: experiment metadata from CSV/Excel files."""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("helix")


def _detect_pdv_column(columns: List[str]) -> Optional[str]:
    normalized = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in columns}
    known = [
        "pdvfilename", "pdvfile", "pdvfilename", "dvfilename",
        "dvfile", "filename", "file_name", "file",
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


def load_parameter_folder(folder: str) -> Dict[str, Dict]:
    """Load experiment metadata from all CSV/Excel files in *folder*.

    Returns a dict mapping PDV filename → row-dict of metadata.
    """
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Parameter folder not found: {folder}")

    param_data: Dict[str, Dict] = {}
    for entry in os.listdir(folder):
        if not entry.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        fpath = os.path.join(folder, entry)
        try:
            if entry.lower().endswith(".csv"):
                df = pd.read_csv(fpath)
            else:
                df = pd.read_excel(fpath)
        except Exception as exc:
            logger.warning("Skipping parameter file %s: %s", entry, exc)
            continue

        if df.empty:
            continue

        pdv_col = _detect_pdv_column(list(df.columns))
        if not pdv_col:
            logger.warning("Could not detect PDV column in %s", entry)
            continue

        for _, row in df.iterrows():
            pdv_name = str(row.get(pdv_col, "")).strip()
            if not pdv_name or pdv_name.lower() == "nan":
                continue
            param_data[pdv_name] = row.to_dict()

    return param_data


def resolve_file_list(
    explicit_files: Optional[List[str]],
    directory: Optional[str],
    pattern: str = "*.csv",
) -> List[str]:
    """Resolve input files from explicit paths or directory + glob pattern."""
    import glob as _glob

    if explicit_files:
        return [os.path.abspath(p) for p in explicit_files if os.path.exists(p)]

    if directory:
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            return []
        return sorted(_glob.glob(os.path.join(directory, pattern)))

    return []
