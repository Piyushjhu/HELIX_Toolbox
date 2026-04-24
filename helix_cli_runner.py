#!/usr/bin/env python3
"""
Command-line runner for HELIX Toolbox analysis.

This script bypasses the GUI and executes the full ALPSS→SPADE pipeline
using a configuration file. It reuses the same AnalysisThread logic that
powers the GUI but drives it from a terminal, making it suitable for batch
runs and automated / headless workflows.

Config files are YAML (.yml / .yaml) or JSON (.json) — format is detected
automatically from the file extension. The fully-commented YAML templates
(helix_master_config.yml, alpss_config_default.yml, spade_config_default.yml)
are the recommended starting point.

Example usage (master config file — recommended):
    python helix_cli_runner.py --config ./helix_master_config.yml

Example usage (override config paths from the command line):
    python helix_cli_runner.py --config ./helix_master_config.yml \\
        --input-dir /path/to/pdv --output-dir /path/to/output

Example usage (separate config files):
    python helix_cli_runner.py \
        --alpss-config ./alpss_config_default.yml \
        --spade-config ./spade_config_default.yml \
        --input-dir /path/to/pdv \
        --input-pattern "*Velocity*.csv" \
        --output-dir /path/to/output \
        --param-folder /path/to/parameter_folder \
        --analysis-mode both \
        --spade-mode auto
"""
import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import pandas as pd
from PyQt5.QtCore import QCoreApplication

# Ensure repository root is importable no matter where the script is run
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from helix_analysis_toolbox import AnalysisThread, load_config_from_file


# Extensions probed (in order) when looking for a default master config next
# to the repo. YAML takes precedence over JSON so users can migrate simply by
# dropping a commented .yml next to the existing .json.
_DEFAULT_CONFIG_BASENAMES = (
    # "helix_master_config.yml",
    # "helix_master_config.yaml",
    "helix_master_config.json",
)


def _find_default_master_config(search_dir: str = REPO_ROOT) -> Optional[str]:
    """Return the first existing default master config path, or None."""
    for name in _DEFAULT_CONFIG_BASENAMES:
        candidate = os.path.join(search_dir, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_config_file(config_path: str) -> Dict:
    """Load a config file (YAML or JSON, auto-detected) and return the dict."""
    ok, data, message = load_config_from_file(config_path)
    if not ok:
        raise RuntimeError(f"Failed to load config {config_path}: {message}")
    return data


# Backwards-compatible alias: historically this helper was JSON-only.
_load_json_config = _load_config_file


def _load_master_config(config_path: str) -> Dict:
    """Load master config file containing CLI settings and nested ALPSS/SPADE configs."""
    master_config = _load_config_file(config_path)
    
    # Validate structure
    if "cli_settings" not in master_config:
        raise ValueError("Master config must contain 'cli_settings' section")
    if "alpss_config" not in master_config:
        raise ValueError("Master config must contain 'alpss_config' section")
    if "spade_config" not in master_config:
        raise ValueError("Master config must contain 'spade_config' section")
    
    return master_config


def _transform_alpss_params_for_analysis(alpss_params: Dict) -> Dict:
    """
    Transform ALPSS config parameters to match exactly what get_alpss_params() returns.
    This ensures CLI produces identical results to GUI.
    
    Handles both:
    - Raw config files (with blur_kernel_x/y, save_all_plots as dropdown value)
    - Transformed config files (with blur_kernel tuple, save_all_plots_enabled)
    """
    # Make a copy to avoid modifying the original
    params = alpss_params.copy()
    
    # Convert save_all_plots dropdown value to ALPSS-expected format
    # This matches get_alpss_params() lines 6402-6411
    # Handle both cases: config with raw dropdown value OR already transformed
    if 'save_all_plots' in params and 'save_all_plots_enabled' not in params:
        # Raw config format - need to transformface
        
        dropdown_value = params['save_all_plots']
        save_plots_value = 'yes' if dropdown_value in ['subfolder', 'main_dir'] else 'no'
        plots_enabled = (save_plots_value == 'yes')
        
        params['save_all_plots_enabled'] = save_plots_value
        params['save_plots_in_subfolder'] = (dropdown_value == 'subfolder')
    elif 'save_all_plots_enabled' in params:
        # Already transformed - just ensure plots_enabled is set correctly
        save_plots_value = params['save_all_plots_enabled']
        plots_enabled = (save_plots_value == 'yes')
        if 'save_plots_in_subfolder' not in params:
            # Infer from save_all_plots if available
            if 'save_all_plots' in params:
                params['save_plots_in_subfolder'] = (params['save_all_plots'] == 'subfolder')
            else:
                params['save_plots_in_subfolder'] = False
    else:
        # Neither present - default to no plots
        plots_enabled = False
        params['save_all_plots_enabled'] = 'no'
        params['save_plots_in_subfolder'] = False
    
    # Gate plot flags by plots_enabled (matches get_alpss_params() lines 6462-6469)
    plot_flags = {
        'save_combined_plot': True,
        'save_iq_start_time_plot': False,
    }
    for flag, default in plot_flags.items():
        if flag in params:
            params[flag] = bool(params[flag]) and plots_enabled
        else:
            params[flag] = (default and plots_enabled)
    
    # Convert blur_kernel_x and blur_kernel_y to blur_kernel tuple
    # This matches get_alpss_params() line 6438
    # Handle both cases: raw config (has x/y) OR already transformed (has tuple)
    if 'blur_kernel' not in params:
        if 'blur_kernel_x' in params and 'blur_kernel_y' in params:
            # Raw config format - convert to tuple
            params['blur_kernel'] = (int(params['blur_kernel_x']), int(params['blur_kernel_y']))
        else:
            # Neither present - use default
            params['blur_kernel'] = (5, 5)  # Default from GUI
    elif isinstance(params['blur_kernel'], list):
        # Convert list to tuple
        params['blur_kernel'] = tuple(params['blur_kernel'])
    
    # Ensure plot_figsize is a tuple (matches get_alpss_params() line 6459)
    if 'plot_figsize' not in params:
        params['plot_figsize'] = (30, 10)  # Default from GUI
    elif isinstance(params['plot_figsize'], list):
        params['plot_figsize'] = tuple(params['plot_figsize'])
    
    # Ensure plot_dpi exists (matches get_alpss_params() line 6460)
    if 'plot_dpi' not in params:
        params['plot_dpi'] = 300  # Default from GUI
    
    # Ensure smart_selection_enabled exists (matches get_alpss_params() line 6475)
    if 'smart_selection_enabled' not in params:
        params['smart_selection_enabled'] = False
    
    # Ensure start_time_user is a string (can be "none" or a number as string)
    if 'start_time_user' in params:
        if params['start_time_user'] is None:
            params['start_time_user'] = "none"
        elif not isinstance(params['start_time_user'], str):
            # Convert number to string if needed
            val_str = str(params['start_time_user']).lower()
            if val_str == "none" or val_str == "nan":
                params['start_time_user'] = "none"
            else:
                params['start_time_user'] = str(params['start_time_user'])
    else:
        params['start_time_user'] = "none"
    
    # Ensure window is a string (matches get_alpss_params() line 6437)
    if 'window' in params and not isinstance(params['window'], str):
        params['window'] = str(params['window'])
    elif 'window' not in params:
        params['window'] = "hann"  # Default
    
    # Ensure cmap is a string (matches get_alpss_params() line 6442)
    if 'cmap' in params and not isinstance(params['cmap'], str):
        params['cmap'] = str(params['cmap'])
    elif 'cmap' not in params:
        params['cmap'] = "viridis"  # Default
    
    # Ensure display_plots and spall_calculation are strings (matches get_alpss_params() lines 6455-6456)
    if 'display_plots' in params and not isinstance(params['display_plots'], str):
        params['display_plots'] = str(params['display_plots'])
    elif 'display_plots' not in params:
        params['display_plots'] = "yes"  # Default
    
    if 'spall_calculation' in params and not isinstance(params['spall_calculation'], str):
        params['spall_calculation'] = str(params['spall_calculation'])
    elif 'spall_calculation' not in params:
        params['spall_calculation'] = "yes"  # Default
    
    # Ensure save_data is a string (matches get_alpss_params() line 6410)
    if 'save_data' in params and not isinstance(params['save_data'], str):
        params['save_data'] = str(params['save_data'])
    elif 'save_data' not in params:
        params['save_data'] = "yes"  # Default
    
    # Ensure all numeric parameters are the correct type (matching GUI's .value() behavior)
    # GUI uses .value() which returns Python float/int, so we need to ensure JSON numbers are converted
    numeric_params = [
        'header_lines', 'time_to_skip', 'time_to_take', 't_before', 't_after',
        'start_time_correction', 'iq_threshold_factor', 'freq_min', 'freq_max',
        'smoothing_window', 'smoothing_wid', 'smoothing_amp', 'smoothing_sigma', 'smoothing_mu',
        'pb_neighbors', 'pb_idx_correction', 'rc_neighbors', 'rc_idx_correction',
        'sample_rate', 'nperseg', 'noverlap', 'nfft', 'carrier_band_time',
        'blur_sigx', 'blur_sigy', 'order', 'wid', 'uncert_mult',
        'lam', 'theta', 'C0', 'density', 'delta_rho', 'delta_C0', 'delta_lam', 'delta_theta'
    ]
    for param in numeric_params:
        if param in params:
            # Convert to float if it's a string representation of a number
            if isinstance(params[param], str):
                try:
                    params[param] = float(params[param])
                except ValueError:
                    pass  # Keep as string if conversion fails
            # Ensure integers are actually integers (header_lines, pb_neighbors, etc.)
            if param in ['header_lines', 'pb_neighbors', 'pb_idx_correction', 'rc_neighbors', 
                        'rc_idx_correction', 'nperseg', 'noverlap', 'nfft', 'smoothing_window',
                        'order', 'plot_dpi']:
                if isinstance(params[param], (int, float)):
                    params[param] = int(params[param])
            
            # Ensure float parameters for IQ detection are floats
            if param in ['iq_smoothing_window_ns', 'iq_skip_start_ns', 'iq_persistence_ns', 
                        'iq_threshold_factor', 'iq_baseline_fraction']:
                if isinstance(params[param], (int, float, str)):
                    try:
                        params[param] = float(params[param])
                    except (ValueError, TypeError):
                        pass  # Keep original if conversion fails
    
    # Ensure boolean parameters are actually booleans
    boolean_params = ['use_notch_filter', 'use_robust_iq_detection',
                     'save_velocity_csv', 'save_velocity_smooth_csv',
                     'save_velocity_uncert_csv', 'save_velocity_smooth_uncert_csv',
                     'save_results_csv', 'save_noise_csv']
    for param in boolean_params:
        if param in params:
            if isinstance(params[param], str):
                params[param] = params[param].lower() in ('true', 'yes', '1')
            else:
                params[param] = bool(params[param])
    
    return params


def _to_bool(value, default: bool = False) -> bool:
    """Convert config values to boolean."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _transform_spade_params_for_analysis(spade_params: Dict) -> Dict:
    """
    Transform SPADE config parameters to match exactly what get_spade_params() returns.
    Ensures CLI toggles (spall/HEL/velocity plots) behave like the GUI.
    """
    params = (spade_params or {}).copy()

    # Experiment toggles
    params['velocity_shots_enabled'] = _to_bool(
        params.get('velocity_shots_enabled', params.get('experiment_velocity_shots', True)),
        default=True,
    )
    params['spall_analysis_enabled'] = _to_bool(
        params.get('spall_analysis_enabled', params.get('experiment_spall_analysis', False)),
        default=False,
    )
    params['hel_detection_enabled'] = _to_bool(
        params.get('hel_detection_enabled', params.get('experiment_hel_detection', False)),
        default=False,
    )

    # Summary table flag
    if 'save_summary_table' not in params:
        params['save_summary_table'] = _to_bool(params.get('save_summary', True), default=True)

    # Smooth window naming (some configs used spade_smooth_window)
    if 'smooth_window' not in params and 'spade_smooth_window' in params:
        params['smooth_window'] = params['spade_smooth_window']

    # Signal length handling (GUI stores None for "Full Signal")
    if 'signal_length_ns' not in params:
        signal_length_value = params.get('signal_length')
        if signal_length_value is None:
            params['signal_length_ns'] = None
        elif isinstance(signal_length_value, str):
            if signal_length_value.strip().lower().startswith('full'):
                params['signal_length_ns'] = None
            else:
                try:
                    params['signal_length_ns'] = float(signal_length_value)
                except ValueError:
                    params['signal_length_ns'] = None
        else:
            params['signal_length_ns'] = signal_length_value
    # Remove legacy key to avoid confusion
    params.pop('signal_length', None)

    # Ensure known booleans are bool type
    for bool_key, default in [
        ('plot_individual', False),
        ('save_summary_table', True),
        ('show_plots', False),
        ('generate_all_velocity_plot', True),
        ('include_uncert_bands', True),
        ('auto_calculate_limits', True),
    ]:
        if bool_key in params:
            params[bool_key] = _to_bool(params[bool_key], default=default)
        else:
            # Set default value if key doesn't exist
            params[bool_key] = default

    return params


def _resolve_file_list(
    explicit_files: Optional[List[str]],
    directory: Optional[str],
    pattern: str,
) -> List[str]:
    """Resolve input files from explicit paths or directory/pattern."""
    if explicit_files:
        existing_files = [os.path.abspath(path) for path in explicit_files if os.path.exists(path)]
        missing_files = [os.path.abspath(path) for path in explicit_files if not os.path.exists(path)]
        if missing_files:
            print("[WARN] Some explicit input files do not exist:")
            for f in missing_files:
                print(f"  - {f}")
        return existing_files

    if directory:
        abs_directory = os.path.abspath(directory)
        if not os.path.isdir(abs_directory):
            print(f"[ERROR] Input directory does not exist: {abs_directory}")
            return []
        
        # Handle pattern - ensure it works with glob
        if not pattern:
            pattern = "*.csv"
        
        # Construct search glob pattern
        search_glob = os.path.join(abs_directory, pattern)
        
        # Debug output
        print(f"[INFO] Searching for files in: {abs_directory}")
        print(f"[INFO] Using pattern: {pattern}")
        print(f"[INFO] Full glob pattern: {search_glob}")
        
        matched_files = sorted(glob.glob(search_glob))
        
        if not matched_files:
            # Provide helpful diagnostics
            print(f"[WARN] No files matched pattern '{pattern}' in directory: {abs_directory}")
            # List what files actually exist in the directory
            try:
                existing_files = os.listdir(abs_directory)
                csv_files = [f for f in existing_files if f.lower().endswith('.csv')]
                if csv_files:
                    print(f"[INFO] Found {len(csv_files)} CSV files in directory (but pattern didn't match):")
                    for f in csv_files[:10]:  # Show first 10
                        print(f"  - {f}")
                    if len(csv_files) > 10:
                        print(f"  ... and {len(csv_files) - 10} more CSV files")
                else:
                    print(f"[INFO] No CSV files found in directory. Total files: {len(existing_files)}")
                    if existing_files:
                        print("[INFO] Sample files in directory:")
                        for f in existing_files[:10]:
                            print(f"  - {f}")
            except Exception as e:
                print(f"[WARN] Could not list directory contents: {e}")
        else:
            print(f"[INFO] Found {len(matched_files)} file(s) matching pattern")
        
        return matched_files

    return []


def _detect_pdv_column(columns: List[str]) -> Optional[str]:
    """Identify the PDV filename column by normalizing column names."""
    normalized_columns = {
        col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in columns
    }
    known_variants = [
        "pdvfilename",
        "pdvfile",
        "pdv_file",
        "pdv_file_name",
        "pdvfilename",
        "dvfilename",
        "dvfile",
        "filename",
        "file_name",
        "file",
    ]
    for variant in known_variants:
        if variant in normalized_columns.values():
            for col, normalized in normalized_columns.items():
                if normalized == variant:
                    return col

    # Fallback: any column containing both "pdv" and "file"
    for col in columns:
        lower = col.lower()
        if "pdv" in lower and "file" in lower:
            return col

    return None


def _normalize_pdv_filename(value: str) -> str:
    """
    Normalize PDV filename strings into canonical form.

    Canonical form used throughout HELIX is typically:
      C<digit>--YYYYMMDD--#####  (e.g., C1--20251023--00072)

    Some metadata files can contain Unicode dash characters (en-dash/em-dash/minus)
    or inconsistent single-dash separators (e.g., "C1–20251023--00072").
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""

    # Strip any directory prefix. Parameter files sometimes store the full path
    # (e.g. "C:\\Users\\Administrator\\Desktop\\PDV_DATA\\<name>" from the
    # acquisition PC, or a POSIX path). We only want the bare filename.
    # Handle both separator styles explicitly since os.path.basename on POSIX
    # hosts will not split on backslashes.
    if "\\" in s:
        s = s.rsplit("\\", 1)[-1]
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = s.strip().strip('"').strip("'")

    # Drop common data-file extensions if present
    for _ext in (".csv", ".txt", ".dat", ".bin", ".h5", ".hdf5", ".trc"):
        if s.lower().endswith(_ext):
            s = s[: -len(_ext)]
            break

    # Normalize common Unicode dash variants to ASCII hyphen
    # (en-dash U+2013, em-dash U+2014, minus U+2212)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # Canonicalize common PDV naming variants to C#--YYYYMMDD--#####.
    # Accept one/two hyphens between tokens after normalization.
    m = re.match(r"^(C\d+)--?(\d{8})--?(\d{5})$", s)
    if m:
        return f"{m.group(1)}--{m.group(2)}--{m.group(3)}"

    # If the string already contains the canonical pattern as a substring, return that.
    m2 = re.search(r"(C\d+--\d{8}--\d{5})", s)
    if m2:
        return m2.group(1)

    return s


def _load_parameter_folder(folder: str) -> Dict[str, Dict]:
    """Aggregate experiment metadata from all CSV/Excel files in a folder."""
    if not folder:
        return {}

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Parameter folder not found: {folder}")

    param_data: Dict[str, Dict] = {}
    all_entries = os.listdir(folder)
    param_files = [e for e in all_entries if e.lower().endswith((".csv", ".xlsx", ".xls"))]
    
    print(f"[INFO] Found {len(param_files)} parameter file(s) to process")
    
    for idx, entry in enumerate(param_files, 1):
        print(f"[INFO] Processing parameter file {idx}/{len(param_files)}: {entry}")
        file_path = os.path.join(folder, entry)
        try:
            if entry.lower().endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                try:
                    df = pd.read_excel(file_path)
                except ImportError as exc:
                    raise ImportError(
                        "openpyxl is required to read Excel parameter files."
                    ) from exc
        except Exception as exc:  # pragma: no cover - best-effort loader
            print(f"[WARN] Skipping parameter file {entry}: {exc}")
            continue

        if df.empty:
            print(f"[WARN] Parameter file {entry} is empty, skipping")
            continue

        pdv_col = _detect_pdv_column(list(df.columns))
        if not pdv_col:
            print(f"[WARN] Could not detect PDV filename column in {entry}")
            continue

        # Check for 'Sample material' column (with various name variations)
        sample_material_col = None
        material_col_variations = [
            'Sample material', 'Sample Material', 'Sample_Material',
            'sample_material', 'samplematerial', 'SampleMaterial',
            'Material', 'material', 'Target material', 'Target Material'
        ]
        for col in df.columns:
            if col in material_col_variations:
                sample_material_col = col
                break
        
        if sample_material_col:
            print(f"[INFO] Found 'Sample material' column: '{sample_material_col}'")
            # Count non-empty material values
            material_values = df[sample_material_col].dropna()
            material_values = material_values[material_values.astype(str).str.strip() != '']
            unique_materials = material_values.unique()
            print(f"[INFO]   Found {len(material_values)} entries with material values")
            print(f"[INFO]   Unique materials: {', '.join(sorted([str(m) for m in unique_materials]))}")
        else:
            print(f"[WARN] No 'Sample material' column found in {entry}")
            print(f"[WARN]   Available columns: {', '.join(df.columns.tolist())}")

        rows_loaded = 0
        rows_with_material = 0
        for _, row in df.iterrows():
            pdv_name_raw = row.get(pdv_col, "")
            pdv_name = _normalize_pdv_filename(pdv_name_raw)
            if not pdv_name:
                continue
            
            # Store all row data as dictionary
            row_dict = row.to_dict()

            # Normalize PDV_FileName inside the row dict too (helps downstream exact matching)
            if "PDV_FileName" in row_dict:
                row_dict["PDV_FileName"] = _normalize_pdv_filename(row_dict.get("PDV_FileName"))
            
            # Ensure 'Sample material' is accessible (normalize column name)
            if sample_material_col and sample_material_col in row_dict:
                # Also store under standard name for easier access
                material_value = str(row_dict[sample_material_col]).strip()
                if material_value and material_value.lower() not in ['nan', 'none', '']:
                    row_dict['Sample material'] = material_value
                    rows_with_material += 1
                else:
                    row_dict['Sample material'] = None
            
            # Store by PDV filename (from detected column, which should be PDV_FileName)
            param_data[pdv_name] = row_dict
            
            # Also store by PDV_FileName column value if it exists and differs from pdv_col
            # This ensures we can match using either the column name or the actual PDV_FileName value
            if 'PDV_FileName' in row_dict and pdv_col != 'PDV_FileName':
                pdv_file_name_value = _normalize_pdv_filename(row_dict.get('PDV_FileName'))
                if pdv_file_name_value and pdv_file_name_value != pdv_name:
                    # Store additional entry keyed by PDV_FileName column value
                    param_data[pdv_file_name_value] = row_dict
            
            rows_loaded += 1
        
        print(f"[INFO] Loaded {rows_loaded} entries from {entry}")
        if sample_material_col:
            print(f"[INFO]   {rows_with_material} entries have valid 'Sample material' values")

    # Summary: Report total parameter data loaded
    total_entries = len(param_data)
    if total_entries > 0:
        # Count entries with Sample material
        entries_with_material = sum(
            1 for entry in param_data.values()
            if isinstance(entry, dict) and 
            entry.get('Sample material') and 
            str(entry.get('Sample material')).strip().lower() not in ['nan', 'none', '']
        )
        
        # Get unique Exp_IDs and materials
        exp_id_materials = {}
        for entry in param_data.values():
            if isinstance(entry, dict):
                pdv_file = entry.get('PDV_FileName', '')
                if pdv_file:
                    # Extract Exp_ID from PDV_FileName
                    import re
                    exp_id_match = re.search(r'([A-Z]+\d+-\d+)', str(pdv_file))
                    if exp_id_match:
                        exp_id = exp_id_match.group(1)
                        material = entry.get('Sample material', 'Unknown')
                        if material and str(material).strip().lower() not in ['nan', 'none', '', 'unknown']:
                            exp_id_materials[exp_id] = str(material).strip()
        
        print(f"\n[INFO] Parameter data summary:")
        print(f"[INFO]   Total entries loaded: {total_entries}")
        print(f"[INFO]   Entries with 'Sample material': {entries_with_material}")
        print(f"[INFO]   Entries without 'Sample material': {total_entries - entries_with_material}")
        print(f"[INFO]   Unique Exp_IDs with materials: {len(exp_id_materials)}")
        if exp_id_materials:
            print(f"[INFO]   Exp_ID -> Material mapping:")
            for exp_id, material in sorted(exp_id_materials.items()):
                print(f"[INFO]     {exp_id} -> {material}")
        
        # Show sample entries with materials
        if entries_with_material > 0:
            sample_count = min(5, entries_with_material)
            print(f"[INFO]   Sample PDV_FileName entries (showing {sample_count}):")
            count = 0
            for pdv_name, entry in param_data.items():
                if isinstance(entry, dict):
                    material = entry.get('Sample material', '')
                    pdv_file = entry.get('PDV_FileName', pdv_name)
                    if material and str(material).strip().lower() not in ['nan', 'none', '']:
                        print(f"[INFO]     {pdv_file} -> Material: {material}")
                        count += 1
                        if count >= sample_count:
                            break
        print()

    return param_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI runner for HELIX Toolbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use master config file (recommended):
  python helix_cli_runner.py --config ./helix_master_config.yml

  # Override config file paths from the command line:
  python helix_cli_runner.py --config ./helix_master_config.yml \\
      --input-dir /different/path --output-dir /different/output

  # Use separate per-tool config files:
  python helix_cli_runner.py \\
      --alpss-config ./alpss_config_default.yml \\
      --spade-config ./spade_config_default.yml \\
      --input-dir /path/to/pdv \\
      --output-dir /path/to/output
        """
    )
    
    # Master config mode
    parser.add_argument(
        "--config",
        help="Path to master config file (contains cli_settings, alpss_config, spade_config). If provided, all settings come from this file (can be overridden by CLI args).",
    )
    
    # Separate config files (required only if --config is not provided)
    parser.add_argument(
        "--alpss-config",
        help="Path to ALPSS JSON config file (required if --config is not provided).",
    )
    parser.add_argument(
        "--spade-config",
        help="Path to SPADE JSON config file (required if --config is not provided).",
    )
    
    # CLI arguments (can override config file values)
    parser.add_argument(
        "--output-dir",
        help="Directory where all outputs will be saved (overrides config file).",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        help="Explicit list of PDV files for ALPSS processing (overrides config file).",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing PDV files for ALPSS processing (overrides config file).",
    )
    parser.add_argument(
        "--input-pattern",
        default="*.csv",
        help="Glob pattern applied within --input-dir (default: %(default)s, overrides config file).",
    )
    parser.add_argument(
        "--param-folder",
        help="Directory containing experiment metadata CSV/Excel files (overrides config file).",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["both", "alpss_only", "spade_only"],
        help="Which stages to run (overrides config file).",
    )
    parser.add_argument(
        "--spade-mode",
        choices=["auto", "manual"],
        help="Use ALPSS outputs (auto) or provide SPADE inputs manually (overrides config file).",
    )
    parser.add_argument(
        "--spade-input-files",
        nargs="+",
        help="Explicit list of velocity files for SPADE when --spade-mode=manual (overrides config file).",
    )
    parser.add_argument(
        "--spade-input-dir",
        help="Directory of velocity files for SPADE when --spade-mode=manual (overrides config file).",
    )
    parser.add_argument(
        "--spade-input-pattern",
        help="Pattern for SPADE manual mode (overrides config file).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # If no config specified, try to use default master config in current directory.
    # Probe order: .yml -> .yaml -> .json (first found wins).
    if not args.config and not args.alpss_config and not args.spade_config:
        default_config = _find_default_master_config(REPO_ROOT)
        if default_config:
            args.config = default_config
            print(f"Using default config: {default_config}")

    # Load configs - either master config or separate configs
    if args.config:
        # Master config mode
        config_path_abs = os.path.abspath(args.config)
        print(f"\n[DEBUG] Loading config from: {config_path_abs}")
        master_config = _load_master_config(config_path_abs)
        cli_settings = master_config.get("cli_settings", {})
        
        # Debug: Print what was loaded from config
        print(f"[DEBUG] Config file 'cli_settings' section:")
        print(f"  input_dir: {cli_settings.get('input_dir', 'NOT FOUND')}")
        print(f"  output_dir: {cli_settings.get('output_dir', 'NOT FOUND')}")
        print(f"  param_folder: {cli_settings.get('param_folder', 'NOT FOUND')}")
        
        alpss_params = master_config.get("alpss_config", {})
        spade_params = master_config.get("spade_config", {})
        material_properties = master_config.get("material_properties", {})  # Extract material properties
        post_processing_config = master_config.get("post_processing_config", {})  # Extract post-processing config
        spade_params = _transform_spade_params_for_analysis(spade_params)
        
        # Transform ALPSS parameters to match exactly what get_alpss_params() returns
        # This ensures CLI produces identical results to GUI
        alpss_params = _transform_alpss_params_for_analysis(alpss_params)
        
        # Extract CLI settings from config, allowing command-line overrides
        # Convert JSON null to None for optional fields
        def _get_or_none(value):
            return None if value is None else value
        
        input_dir = args.input_dir if args.input_dir else _get_or_none(cli_settings.get("input_dir"))
        input_files = args.input_files if args.input_files else _get_or_none(cli_settings.get("input_files"))
        input_pattern = args.input_pattern if args.input_pattern else cli_settings.get("input_pattern", "*.csv")
        output_dir = args.output_dir if args.output_dir else cli_settings.get("output_dir")
        param_folder = args.param_folder if args.param_folder else _get_or_none(cli_settings.get("param_folder"))
        analysis_mode = args.analysis_mode if args.analysis_mode else cli_settings.get("analysis_mode", "both")
        spade_mode = args.spade_mode if args.spade_mode else cli_settings.get("spade_mode", "auto")
        spade_input_files = args.spade_input_files if args.spade_input_files else _get_or_none(cli_settings.get("spade_input_files"))
        spade_input_dir = args.spade_input_dir if args.spade_input_dir else _get_or_none(cli_settings.get("spade_input_dir"))
        spade_input_pattern = args.spade_input_pattern if args.spade_input_pattern else cli_settings.get("spade_input_pattern", "*--vel-smooth-with-uncert.csv")
        
        # Debug: Print resolved values
        print(f"\n[DEBUG] Resolved CLI settings (after command-line overrides):")
        print(f"  input_dir: {input_dir}")
        print(f"  output_dir: {output_dir}")
        print(f"  param_folder: {param_folder}")
        print(f"  analysis_mode: {analysis_mode}")
        print(f"  spade_mode: {spade_mode}\n")
        
    else:
        # Separate configs mode (backward compatibility)
        if not args.alpss_config or not args.spade_config:
            default_config = _find_default_master_config(REPO_ROOT) or os.path.join(
                REPO_ROOT, "helix_master_config.yml"
            )
            error_msg = (
                "No configuration specified.\n\n"
                "Options:\n"
                f"  1. Use master config: --config ./helix_master_config.yml\n"
                f"     (or place helix_master_config.yml / helix_master_config.json in current directory)\n"
                "  2. Use separate configs: --alpss-config <path> --spade-config <path>\n\n"
                f"Default config location: {default_config}"
            )
            raise ValueError(error_msg)
        
        alpss_params = _load_json_config(os.path.abspath(args.alpss_config))
        spade_params = _load_json_config(os.path.abspath(args.spade_config))
        material_properties = {}  # Separate configs don't have material_properties section
        post_processing_config = {}  # Separate configs don't have post_processing_config section
        spade_params = _transform_spade_params_for_analysis(spade_params)
        
        # Transform ALPSS parameters to match exactly what get_alpss_params() returns
        # This ensures CLI produces identical results to GUI
        alpss_params = _transform_alpss_params_for_analysis(alpss_params)
        
        # Get CLI settings from command-line args only
        input_dir = args.input_dir
        input_files = args.input_files
        input_pattern = args.input_pattern if args.input_pattern else "*.csv"
        output_dir = args.output_dir
        param_folder = args.param_folder
        analysis_mode = args.analysis_mode if args.analysis_mode else "both"
        spade_mode = args.spade_mode if args.spade_mode else "auto"
        spade_input_files = args.spade_input_files
        spade_input_dir = args.spade_input_dir
        spade_input_pattern = args.spade_input_pattern if args.spade_input_pattern else "*--vel-smooth-with-uncert.csv"

    # Validate required settings
    if not output_dir:
        raise ValueError("--output-dir is required (either in config file or command line)")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Debug: Print output directory information
    print("\n" + "=" * 70)
    print("OUTPUT DIRECTORY CONFIGURATION")
    print("=" * 70)
    print(f"Output directory (resolved): {output_dir}")
    print(f"Output directory exists: {os.path.exists(output_dir)}")
    if os.path.exists(output_dir):
        print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")
    print("=" * 70 + "\n")

    # Resolve input files
    if analysis_mode != "spade_only":
        print("\n" + "=" * 70)
        print("RESOLVING PDV INPUT FILES")
        print("=" * 70)
        if input_files:
            print(f"Using explicit input files: {len(input_files)} file(s) specified")
        elif input_dir:
            print(f"Searching directory: {input_dir}")
            print(f"Using pattern: {input_pattern}")
        else:
            print("ERROR: No input files or input directory specified!")
        print("=" * 70 + "\n")
        
        resolved_input_files = _resolve_file_list(input_files, input_dir, input_pattern)
        if not resolved_input_files:
            error_msg = "No PDV input files found for ALPSS processing.\n\n"
            if input_files:
                error_msg += f"Explicit files specified: {input_files}\n"
                error_msg += "Please verify these file paths exist.\n"
            elif input_dir:
                error_msg += f"Directory: {input_dir}\n"
                error_msg += f"Pattern: {input_pattern}\n"
                error_msg += "Please verify:\n"
                error_msg += "  1. The directory path is correct\n"
                error_msg += "  2. The directory contains CSV files\n"
                error_msg += "  3. The pattern matches your file naming convention\n"
            else:
                error_msg += "No input files or input directory was specified.\n"
                error_msg += "Please provide either --input-files or --input-dir in config or command line.\n"
            raise RuntimeError(error_msg)
        
        print(f"\n✓ Successfully resolved {len(resolved_input_files)} input file(s)\n")
    else:
        resolved_input_files = []
        print("\n" + "=" * 70)
        print("SPADE-ONLY MODE: Skipping ALPSS input file resolution")
        print("=" * 70)
        print(f"analysis_mode: {analysis_mode}")
        print(f"resolved_input_files: [] (empty list)")
        print("=" * 70 + "\n")

    if analysis_mode == "spade_only" and spade_mode == "auto":
        raise ValueError("Auto SPADE mode requires ALPSS outputs from the same run. "
                         "Please use --analysis-mode both or switch to manual SPADE mode.")

    if spade_mode == "manual":
        resolved_spade_input_files = _resolve_file_list(
            spade_input_files,
            spade_input_dir,
            spade_input_pattern,
        )
        if not resolved_spade_input_files:
            raise RuntimeError("Manual SPADE mode requires --spade-input-files or --spade-input-dir.")
        spade_auto_mode = False
    else:
        resolved_spade_input_files = None
        spade_auto_mode = True

    # Load parameter folder with progress indication
    if param_folder:
        print(f"\n[INFO] Loading parameter data from: {param_folder}")
        param_data = _load_parameter_folder(param_folder)
    else:
        param_data = None
    
    # Inform user about parameter data loading for material color-coding
    if param_data:
        num_params = len(param_data)
        print(f"✓ Loaded parameter data for {num_params} experiments (will be used for material color-coding in velocity plots)")
        
        # Show sample PDV filenames and materials for verification
        sample_count = min(5, num_params)
        print(f"\n  Sample parameter file entries (showing {sample_count} of {num_params}):")
        sample_keys = list(param_data.keys())[:sample_count]
        for pdv_name in sample_keys:
            param_row = param_data[pdv_name]
            # Try to find sample material column
            material = 'Unknown'
            material_keys = [
                'Sample material', 'Sample Material', 'Sample_Material',
                'sample_material', 'samplematerial', 'SampleMaterial',
                'Material', 'material', 'Flyer_material', 'Flyer Material'
            ]
            for key in material_keys:
                if key in param_row:
                    material_val = str(param_row[key]).strip()
                    if material_val and material_val.lower() != 'nan':
                        material = material_val
                        break
            print(f"    {pdv_name} -> Material: {material}")
        if num_params > sample_count:
            print(f"    ... and {num_params - sample_count} more entries")
        print()
    elif param_folder:
        print(f"⚠️  Parameter folder specified but no data loaded: {param_folder}")
    else:
        print("ℹ️  No parameter folder specified - velocity traces will be color-coded as 'Unknown' material")

    # Validate critical ALPSS parameters
    critical_params = [
        'save_data', 'header_lines', 'time_to_skip', 'time_to_take',
        't_before', 't_after', 'start_time_user', 'start_time_correction',
        'iq_threshold_factor', 'freq_min', 'freq_max', 'smoothing_window',
        'smoothing_wid', 'smoothing_amp', 'smoothing_sigma', 'smoothing_mu',
        'sample_rate', 'nperseg', 'noverlap', 'nfft', 'window',
        'lam', 'theta', 'C0', 'density', 'blur_kernel', 'use_notch_filter'
    ]
    missing_params = [p for p in critical_params if p not in alpss_params]
    if missing_params:
        raise ValueError(f"Missing critical ALPSS parameters: {', '.join(missing_params)}")
    
    # Warn about suspicious parameter values that might cause incorrect results
    warnings = []
    if 'time_to_take' in alpss_params:
        time_to_take = alpss_params['time_to_take']
        if isinstance(time_to_take, (int, float)) and time_to_take < 5e-6:
            warnings.append(f"WARNING: time_to_take={time_to_take:.2e} is very short (<5μs). GUI default is 10e-6 (10μs). This may cause incorrect results!")
    if 'smoothing_window' in alpss_params:
        smoothing_window = alpss_params['smoothing_window']
        if isinstance(smoothing_window, (int, float)) and smoothing_window < 100:
            warnings.append(f"WARNING: smoothing_window={smoothing_window} is very small. Typical values are 500-1000.")
    if warnings:
        print("\n" + "=" * 70)
        print("⚠️  PARAMETER WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
        print("=" * 70 + "\n")
    
    # Debug: Print ALL ALPSS parameters for comprehensive comparison
    print("\n" + "=" * 70)
    print("=== ALL ALPSS PARAMETERS BEING PASSED TO ALPSS ===")
    print("=" * 70)
    # Sort keys for easier comparison
    sorted_keys = sorted(alpss_params.keys())
    for key in sorted_keys:
        value = alpss_params[key]
        # Format different types appropriately
        if isinstance(value, (int, float)):
            if abs(value) > 1e6 or (abs(value) < 1e-6 and value != 0):
                print(f"  {key:35s}: {value:.6e}")
            else:
                print(f"  {key:35s}: {value}")
        elif isinstance(value, tuple):
            print(f"  {key:35s}: {value}")
        elif isinstance(value, list):
            print(f"  {key:35s}: [list of {len(value)} items]")
        elif isinstance(value, dict):
            print(f"  {key:35s}: {{dict with {len(value)} keys}}")
        elif isinstance(value, str):
            # Truncate very long strings
            if len(value) > 100:
                print(f"  {key:35s}: {value[:97]}...")
            else:
                print(f"  {key:35s}: '{value}'")
        elif isinstance(value, bool):
            print(f"  {key:35s}: {value}")
        else:
            print(f"  {key:35s}: {type(value).__name__} = {value}")
    print("=" * 70 + "\n")

    # Check if post-processing mode is enabled (only for master config mode)
    # post_processing_config is now always defined (either from master config or set to {} for separate configs)
    if post_processing_config.get("enabled", False):
        # Post-processing mode: Generate plots from existing data
        print("\n" + "=" * 70)
        print("POST-PROCESSING MODE: Generating plots from existing data")
        print("=" * 70)
        print(f"SPADE output directory: {post_processing_config.get('spade_output_dir', output_dir)}")
        print(f"Enabled plots: {[k for k, v in post_processing_config.get('plots', {}).items() if v]}")
        print("=" * 70 + "\n")
        
        # Prepare Qt event loop for the AnalysisThread (which inherits QThread)
        app = QCoreApplication([])
        thread = AnalysisThread(
            alpss_params=alpss_params,
            spade_params=spade_params,
            input_files=[],  # Not needed for post-processing
            output_dir=output_dir,
            param_data=param_data,
            spade_auto_mode=spade_auto_mode,
            spade_input_files=None,
            analysis_mode="post_process_only",  # Special mode for post-processing
            material_properties=material_properties,
        )
        
        # Connect progress signal
        def on_progress(msg):
            print(msg)
        thread.progress_signal.connect(on_progress)
        
        # Run post-processing
        thread.start()
        success = thread.run_post_processing(post_processing_config)
        thread.wait()
        
        if success:
            print("\n✅ Post-processing complete!")
            sys.exit(0)
        else:
            print("\n❌ Post-processing failed. See error messages above.")
            sys.exit(1)
    
    # Normal analysis mode
    # Prepare Qt event loop for the AnalysisThread (which inherits QThread)
    app = QCoreApplication([])
    
    # Debug: Print output directory being passed to AnalysisThread
    print("\n" + "=" * 70)
    print("CREATING ANALYSIS THREAD")
    print("=" * 70)
    print(f"Output directory passed to AnalysisThread: {output_dir}")
    print(f"Output directory absolute path: {os.path.abspath(output_dir)}")
    print("=" * 70 + "\n")
    
    thread = AnalysisThread(
        alpss_params=alpss_params,
        spade_params=spade_params,
        input_files=resolved_input_files,
        output_dir=output_dir,
        param_data=param_data,
        spade_auto_mode=spade_auto_mode,
        spade_input_files=resolved_spade_input_files,
        analysis_mode=analysis_mode,
        material_properties=material_properties,
    )

    result = {"success": False}

    def _on_progress(message: str):
        print(message, flush=True)

    def _on_finished(success: bool, message: str):
        result["success"] = success
        if success:
            print("✅ Analysis completed successfully.", flush=True)
        else:
            print(f"❌ Analysis failed: {message}", flush=True)
        app.quit()

    thread.progress_signal.connect(_on_progress)
    thread.finished_signal.connect(_on_finished)
    thread.start()

    try:
        app.exec_()
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping analysis...", flush=True)
        thread.requestInterruption()
        thread.wait()

    # Ensure thread is fully finished before exiting to avoid "QThread: Destroyed while thread is still running"
    # This ensures proper cleanup and prevents the warning message
    if thread.isRunning():
        thread.wait(5000)  # Wait up to 5 seconds for thread to finish
    # Even if not running, ensure any pending cleanup completes
    thread.wait(1000)  # Brief wait to ensure cleanup is complete

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()

