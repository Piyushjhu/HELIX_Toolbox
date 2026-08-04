#!/usr/bin/env python3
"""
ALPSS + SPADE Combined GUI
A comprehensive GUI for running ALPSS analysis followed by SPADE spall analysis
"""
import sys
import os
import glob
import subprocess
import threading
import time
import json
import pandas as pd
import numpy as np
import matplotlib

# Ensure stdout/stderr use UTF-8 on Windows (default is cp1252/charmap which
# cannot encode characters like ✓ \u2713, causing UnicodeEncodeError in print()).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# Set non-interactive backend BEFORE importing pyplot or SPADE to avoid macOS aborts
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Excel support will be checked dynamically when needed

# Values that mean "no material recorded", including the "[]" empty-list
# artifact some LMI/AIM-DRC parameter exports leave in 'Sample material'
# when the field wasn't filled in upstream.
INVALID_MATERIAL_VALUES = {'', 'nan', 'none', 'unknown', '[]', '[ ]'}


def _is_valid_material_value(value):
    """True if value looks like a real material name rather than a blank/placeholder."""
    if value is None:
        return False
    return str(value).strip().lower() not in INVALID_MATERIAL_VALUES

# ---------------------------------------------------------------------------
# PyQt5 / display bootstrap
#
# In headless Linux environments (GitHub Codespaces, Docker containers, CI)
# two things can go wrong before PyQt5 can be imported:
#   1. libGL.so.1 is missing  → ImportError on PyQt5 import
#   2. No DISPLAY is set      → Qt: cannot connect to X server
#
# The block below attempts to fix both automatically, so users don't have to
# run any manual apt/Xvfb commands.
# ---------------------------------------------------------------------------
def _ensure_display_and_libgl():
    """Auto-fix missing libGL and headless display on Linux before Qt starts."""
    import platform
    if platform.system() != "Linux":
        return  # macOS / Windows handle this themselves

    # -- 1. Auto-install libGL if the shared library is missing ---------------
    import ctypes.util
    if ctypes.util.find_library("GL") is None:
        print("[HELIX] libGL.so.1 not found — attempting to install libgl1-mesa-glx …")
        _ok = False
        try:
            # Remove any broken apt sources (e.g. unsigned Yarn repo) so that
            # apt-get update succeeds.
            _broken_src = "/etc/apt/sources.list.d/yarn.list"
            if os.path.exists(_broken_src):
                subprocess.run(["sudo", "rm", "-f", _broken_src],
                               capture_output=True, check=False)
            subprocess.run(["sudo", "apt-get", "update", "-qq"],
                           capture_output=True, check=False)
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "-qq",
                 "libgl1-mesa-glx", "libglib2.0-0"],
                capture_output=True, check=False)
            if result.returncode == 0:
                print("[HELIX] libgl1-mesa-glx installed successfully.")
                _ok = True
            else:
                # Fallback: some newer Debian/Ubuntu renamed the package
                result2 = subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "-qq", "libgl1"],
                    capture_output=True, check=False)
                _ok = result2.returncode == 0
                if _ok:
                    print("[HELIX] libgl1 installed successfully (fallback).")
        except Exception as exc:
            print(f"[HELIX] Could not auto-install libGL: {exc}")
        if not _ok:
            print("[HELIX] WARNING: libGL installation failed. "
                  "Run: sudo apt-get install -y libgl1-mesa-glx")

    # -- 2. Set up a virtual framebuffer if no display is available -----------
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("[HELIX] No display detected — starting virtual framebuffer (Xvfb) …")
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "-qq", "xvfb"],
                           capture_output=True, check=False)
            # Launch Xvfb on :99 (ignore error if already running)
            subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.environ["DISPLAY"] = ":99"
            time.sleep(0.5)  # give Xvfb a moment to start
            print("[HELIX] Virtual display started on :99.")
        except Exception as exc:
            # Last resort: use Qt's offscreen platform (no window, but no crash)
            print(f"[HELIX] Could not start Xvfb ({exc}). "
                  "Falling back to offscreen Qt platform.")
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ensure_display_and_libgl()

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
        QLineEdit, QPushButton, QTextEdit, QPlainTextEdit, QProgressBar,
        QFileDialog, QCheckBox, QComboBox, QSpinBox, QRadioButton, QButtonGroup,
        QDoubleSpinBox, QGroupBox, QScrollArea, QMessageBox,
        QSplitter, QFrame, QStyleFactory, QTabBar, QListWidget)
    from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
    from PyQt5.QtGui import QFont, QValidator
    _GUI_AVAILABLE = True
except ImportError as _qt_err:
    _GUI_AVAILABLE = False
    print(f"\n[HELIX] WARNING: PyQt5 not found — running in headless CLI mode. ({_qt_err})")

    # --- Lightweight shims so AnalysisThread works without a Qt event loop ---

    class _CliSignal:
        """Drop-in for pyqtSignal: stores callbacks and calls them synchronously."""
        def __init__(self):
            self._cbs = []
        def connect(self, cb):
            self._cbs.append(cb)
        def emit(self, *args):
            for cb in self._cbs:
                cb(*args)

    def pyqtSignal(*types):
        """Class-level descriptor that creates one _CliSignal per instance."""
        class _Descriptor:
            def __set_name__(self, owner, name):
                self._attr = f'_clisig_{name}'
            def __get__(self, obj, cls=None):
                if obj is None:
                    return self
                if not hasattr(obj, self._attr):
                    object.__setattr__(obj, self._attr, _CliSignal())
                return getattr(obj, self._attr)
        return _Descriptor()

    class QThread:
        def __init__(self): pass
        def start(self): self.run()
        def wait(self, ms=None): pass
        def isRunning(self): return False
        def isInterruptionRequested(self): return False
        def requestInterruption(self): pass

    class Qt: pass
    class QObject: pass

    # GUI widget stubs — raise a clear error if GUI code is attempted without PyQt5
    class _QtWidgetStub:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "PyQt5 is required for GUI mode. "
                "Install it with:  pip install PyQt5"
            )
    QApplication = QMainWindow = QTabWidget = QWidget = _QtWidgetStub
    QVBoxLayout = QHBoxLayout = QGridLayout = _QtWidgetStub
    QLabel = QLineEdit = QPushButton = QTextEdit = QPlainTextEdit = _QtWidgetStub
    QProgressBar = QFileDialog = QCheckBox = QComboBox = _QtWidgetStub
    QSpinBox = QRadioButton = QButtonGroup = QDoubleSpinBox = _QtWidgetStub
    QGroupBox = QScrollArea = QMessageBox = QSplitter = QFrame = _QtWidgetStub
    QStyleFactory = QTabBar = QListWidget = _QtWidgetStub
    QFont = QValidator = _QtWidgetStub
from SPADE.spall_analysis_release.spall_analysis import (
    plot_combined_mean_traces,
    plot_spall_vs_strain_rate,
    plot_spall_vs_shock_stress,
    plot_shock_stress_vs_laser_energy,
)
from datetime import datetime
from material_properties import get_material_properties, list_available_materials
from helix_paper_plots import (
    generate_spall_vs_strain_rate_plot,
    generate_spall_vs_strain_rate_by_material_subplots,
)

def cleanup_matplotlib():
    """Clean up matplotlib figures to prevent memory leaks"""
    import matplotlib.pyplot as plt
    plt.close('all')  # Close all figures
    plt.clf()  # Clear current figure
    plt.cla()  # Clear current axes

# ---------------------------------------------------------------------------
# Configuration I/O
#
# Both YAML (.yml / .yaml) and JSON (.json) are supported. The format is chosen
# automatically from the file extension, so existing JSON configs keep working
# unchanged while users can migrate to commented YAML at their own pace.
#
# YAML support requires the optional PyYAML dependency. If it is not installed,
# JSON still works, and attempting to load/save a YAML file will return a clear
# error telling the user to `pip install pyyaml`.
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml  # PyYAML
except ImportError:  # pragma: no cover - optional dep
    _yaml = None


def _config_format(file_path):
    """Return 'yaml' or 'json' based on file extension (defaults to 'json')."""
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext in (".yml", ".yaml"):
        return "yaml"
    return "json"


def save_config_to_file(config_dict, file_path):
    """Save configuration dictionary to a JSON or YAML file.

    The output format is chosen from the file extension:
      * ``.json``          -> JSON (indent=4)
      * ``.yml`` / ``.yaml`` -> YAML (block style, keys preserved)

    Note: saving to YAML uses PyYAML's plain dumper, so any comments from a
    previously-loaded YAML template are **not** preserved. Edit the YAML
    template directly if you want to keep comments.
    """
    fmt = _config_format(file_path)
    try:
        if fmt == "yaml":
            if _yaml is None:
                return False, (
                    "YAML support requires PyYAML. Install it with "
                    "`pip install pyyaml`, or save as .json instead."
                )
            with open(file_path, "w", encoding='utf-8') as f:
                _yaml.safe_dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2,
                    allow_unicode=True,
                )
        else:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
        return True, f"Configuration saved to {file_path}"
    except Exception as e:
        return False, f"Error saving config: {str(e)}"


def load_config_from_file(file_path):
    """Load a configuration dictionary from a JSON or YAML file.

    Returns a ``(ok, config_dict, message)`` tuple. The parser is chosen from
    the file extension (``.yml`` / ``.yaml`` -> YAML, anything else -> JSON).
    """
    fmt = _config_format(file_path)
    try:
        if fmt == "yaml":
            if _yaml is None:
                return False, {}, (
                    "YAML support requires PyYAML. Install it with "
                    "`pip install pyyaml`, or use a .json config instead."
                )
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = _yaml.safe_load(f)
            if config_dict is None:
                config_dict = {}
            if not isinstance(config_dict, dict):
                return False, {}, (
                    f"YAML file {file_path} did not parse into a mapping "
                    f"(got {type(config_dict).__name__})."
                )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        return True, config_dict, f"Configuration loaded from {file_path}"
    except Exception as e:
        return False, {}, f"Error loading config: {str(e)}"

class ScientificSpinBox(QDoubleSpinBox):
    """Custom spin box that accepts scientific notation input"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(15)  # Allow high precision
        # Ensure value only updates when editing is finished (Enter/blur)
        self.setKeyboardTracking(False)

    def _strip_prefix_suffix(self, text: str) -> str:
        """Remove any prefix/suffix and normalize scientific notation markers."""
        if text is None:
            return ""
        cleaned = text.strip()
        # Remove suffix/prefix if present
        suf = self.suffix()
        pre = self.prefix()
        if suf:
            cleaned = cleaned.replace(suf, "")
        if pre:
            cleaned = cleaned.replace(pre, "")
        # Normalize scientific notation
        cleaned = cleaned.replace('E+', 'e+').replace('E-', 'e-').replace('E', 'e')
        return cleaned.strip()

    def textFromValue(self, value):
        """Convert value to scientific notation string with high precision"""
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            # For scientific notation, preserve more significant figures
            return f"{value:.9e}"
        else:
            # For regular numbers, preserve more significant figures
            return f"{value:.9g}"

    def valueFromText(self, text):
        """Convert scientific notation string to value"""
        try:
            cleaned = self._strip_prefix_suffix(text)
            # Empty field should not force-reset; keep current value
            if cleaned == "":
                return self.value()
            return float(cleaned)
        except ValueError:
            # Keep current value on parse error instead of resetting
            return self.value()

    def validate(self, text, pos):
        """Validate scientific notation input"""
        try:
            cleaned = self._strip_prefix_suffix(text)
            if cleaned == "":
                return (QValidator.Acceptable, text, pos)
            # Allow partial input during typing
            if cleaned.endswith('e'):
                return (QValidator.Intermediate, text, pos)
            if cleaned.endswith('e+') or cleaned.endswith('e-'):
                return (QValidator.Intermediate, text, pos)
            # Try to parse the value
            float(cleaned)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            return (QValidator.Invalid, text, pos)


def filter_3sigma_outliers(data_df, x_col, y_col, progress_callback=None):
    """
    Filter outliers using 3-sigma rule on both x and y axes.
    Returns filtered dataframe and list of removed filenames (if available).
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Input data
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data
    progress_callback : callable, optional
        Callback function for progress messages
    
    Returns:
    --------
    filtered_df : pd.DataFrame
        Data with outliers removed
    outliers_removed : list
        List of filenames or indices of removed outliers
    """
    if len(data_df) == 0:
        return data_df, []
    
    # Calculate mean and std for both axes
    x_data = pd.to_numeric(data_df[x_col], errors='coerce')
    y_data = pd.to_numeric(data_df[y_col], errors='coerce')
    
    x_mean = x_data.mean()
    x_std = x_data.std()
    y_mean = y_data.mean()
    y_std = y_data.std()
    
    # Identify outliers (beyond 3 sigma on either axis)
    x_outliers = np.abs(x_data - x_mean) > 3 * x_std
    y_outliers = np.abs(y_data - y_mean) > 3 * y_std
    outlier_mask = x_outliers | y_outliers
    
    # Get list of removed entries
    outliers_removed = []
    if 'Filename' in data_df.columns:
        outliers_removed = data_df[outlier_mask]['Filename'].tolist()
    else:
        outliers_removed = data_df[outlier_mask].index.tolist()
    
    # Filter data
    filtered_df = data_df[~outlier_mask].copy()
    
    if progress_callback and len(outliers_removed) > 0:
        progress_callback(f"   3-sigma filter: Removed {len(outliers_removed)} outlier(s) from {len(data_df)} data points")
        if len(outliers_removed) <= 10:
            for outlier in outliers_removed:
                progress_callback(f"     Removed: {outlier}")
    
    return filtered_df, outliers_removed


# ── Consolidated-summary column standardization ─────────────────────────────────
# The persisted master summary (<IGSN>-Data_Summary.csv) uses a single, standardized
# naming convention: underscore-separated names with a trailing unit token
# (e.g. Peak_Shock_Stress_GPa, HEL_GPa). Internal SPADE result dicts and matplotlib
# display labels keep their human-readable spaced forms on purpose -- only the
# columns written to / read back from the master CSV are standardized, via the two
# helpers below. `standardize_summary_columns` is applied just before writing the
# master; `normalize_summary_columns` is applied just after reading any summary CSV
# (master or legacy) so downstream code sees standardized names regardless of whether
# the file on disk was written by an older build.
SUMMARY_COLUMN_RENAME = {
    # Spall / shock (spaced -> underscore+unit)
    'Peak Shock Stress (GPa)': 'Peak_Shock_Stress_GPa',
    'Peak Shock Stress Uncertainty (GPa)': 'Peak_Shock_Stress_Unc_GPa',
    'Plateau Mean Velocity (m/s)': 'Plateau_Mean_Velocity_m_s',
    'Plateau Mean Velocity Uncertainty (m/s)': 'Plateau_Mean_Velocity_Unc_m_s',
    'Strain Rate Uncertainty (s^-1)': 'Strain_Rate_Unc_s^-1',
    'First Maxima (m/s)': 'First_Maxima_m_s',
    'Pullback Minima (m/s)': 'Minima_m_s',
    # HEL (lowercase -> HEL_*)
    'hel_ok': 'HEL_OK',
    'hel_strength_gpa': 'HEL_GPa',
    'hel_uncertainty_gpa': 'HEL_Uncertainty_GPa',
    'hel_strain_rate_s^-1': 'HEL_StrainRate_s^-1',
    'hel_segment_time_ns': 'HEL_Segment_Time_ns',
    'hel_consecutive_points': 'HEL_Consecutive_Points',
    'free_surface_velocity_ms': 'HEL_FreeSurface_Velocity_m_s',
}
# Reverse view for reader code that still needs a legacy spelling (not used to write).
SUMMARY_COLUMN_RENAME_INVERSE = {v: k for k, v in SUMMARY_COLUMN_RENAME.items()}


def standardize_summary_columns(df):
    """Rename master-summary columns to the standardized convention (see map above).

    Only renames a legacy column when its standardized target is not already present,
    so a df that is already standardized (or has both) is left unharmed.
    """
    if df is None or not hasattr(df, 'columns'):
        return df
    rename = {old: new for old, new in SUMMARY_COLUMN_RENAME.items()
              if old in df.columns and new not in df.columns}
    return df.rename(columns=rename) if rename else df


def normalize_summary_columns(df):
    """Prepare a just-loaded summary df so BOTH naming conventions resolve in memory.

    After reading a master (or legacy) summary CSV, this ensures the standardized name
    exists for every known column AND keeps a legacy-spelled alias alongside it. That
    way new code can use standardized names while the many existing GUI plot methods
    that still reference the spaced/lowercase spellings keep working, regardless of
    which build wrote the file. The persisted file itself is written single-named via
    standardize_summary_columns(); these duplicate aliases live only in memory.
    """
    if df is None or not hasattr(df, 'columns'):
        return df
    # 1) make sure the standardized spelling is present for each known legacy column
    for old, new in SUMMARY_COLUMN_RENAME.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    # 2) make sure the legacy spelling is present for each known standardized column
    for new, old in SUMMARY_COLUMN_RENAME_INVERSE.items():
        if new in df.columns and old not in df.columns:
            df[old] = df[new]
    return df


class AnalysisThread(QThread):
    """Thread for running ALPSS and SPADE analysis"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
    self,
    alpss_params,
    spade_params,
    input_files,
    output_dir,
    param_data=None,
    spade_auto_mode=True,
    spade_input_files=None,
     analysis_mode="both",
     material_properties=None,
     igsn_material_map=None):
        super().__init__()
        self.alpss_params = alpss_params
        self.spade_params = spade_params
        self.input_files = input_files
        self.output_dir = output_dir
        self.param_data = param_data  # Parameter file data mapping
        self.spade_auto_mode = spade_auto_mode
        self.spade_input_files = spade_input_files
        self.analysis_mode = analysis_mode  # "alpss_only", "spade_only", or "both"
        self.material_properties = material_properties or {}  # Material properties from config
        self.igsn_material_map = igsn_material_map or {}  # IGSN → material fallback mapping from config

        # Initialize trace counting for summary
        self.total_input_traces = 0
        self.traces_plotted = 0
        self.traces_rejected = 0
        self.rejection_reasons = {}  # Track reasons for rejection
        self._warned_skip_unknown_override = False
        # sample_rate is auto-detected per file inside alpss_main and never written
        # back to self.alpss_params, so the configured fallback alone doesn't tell
        # you what was actually used. Track the real per-file value here so
        # _save_run_config can report it accurately.
        self._alpss_effective_sample_rate_by_file = {}

    def _get_summary_filename(self):
        """Return the CSV filename for the consolidated data summary.

        Derives the IGSN prefix from the parent folder of output_dir.
        Example: .../JHAMAL00016-004/Output  →  JHAMAL00016-004-Data_Summary.csv
        Falls back to 'Data_Summary.csv' if the parent folder name is generic.
        """
        try:
            parent = os.path.basename(os.path.dirname(os.path.abspath(self.output_dir)))
            if parent and parent not in ('', '.', '..', 'Output', 'output', 'Results', 'results'):
                return f"{parent}-Data_Summary.csv"
        except Exception:
            pass
        return "Data_Summary.csv"

    def _save_run_config(self, spade_output_dir):
        """Save the ALPSS/SPADE parameters used for this run next to the summary CSV.

        Without this, tracing which config produced a given Data_Summary.csv
        requires reconstructing it from memory/git history, since the run
        doesn't otherwise leave a record of its own input parameters.
        """
        try:
            config_filename = self._get_summary_filename().replace('Data_Summary.csv', 'Run_Config.json')
            config_path = os.path.join(spade_output_dir, config_filename)
            config_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'analysis_mode': self.analysis_mode,
                'output_dir': self.output_dir,
                'input_files': self.input_files,
                'spade_auto_mode': self.spade_auto_mode,
                'spade_input_files': self.spade_input_files,
                'alpss_params': self.alpss_params,
                'spade_params': self.spade_params,
                'material_properties': self.material_properties,
                'igsn_material_map': self.igsn_material_map,
                # alpss_params['sample_rate'] is only the configured fallback;
                # this records what was actually detected/used per input file.
                'alpss_effective_sample_rate_by_file': self._alpss_effective_sample_rate_by_file,
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_snapshot, f, indent=2, default=str)
            self.progress_signal.emit(f"Saved run config to: {config_path}")
        except Exception as e:
            self.progress_signal.emit(f"Warning: Could not save run config: {e}")

    def _should_skip_unknown_materials(self):
        """
        Determine whether Unknown-material traces should be skipped.
        Automatically disables skipping when no parameter file is loaded so
        combined plots are not empty.
        """
        skip_requested = bool(self.spade_params.get('skip_unknown_material_traces', False))
        param_available = bool(self.param_data)
        if skip_requested and not param_available:
            if not self._warned_skip_unknown_override:
                self.progress_signal.emit(
                    "⚠️  Parameter file not selected; including 'Unknown' material traces in combined plots."
                )
                self._warned_skip_unknown_override = True
            return False
        return skip_requested

    def get_param_data_for_file(self, base_name):
        """
        Get parameter data for a filename by matching PDV_FileName column.
        Prioritizes exact PDV_FileName match, falls back to Exp_ID match since 
        material is the same for all shots in an experiment.
        
        Args:
            base_name: Base filename (without extension and suffixes)
            
        Returns:
            Dictionary of parameter data, or empty dict if no match found
        """
        import re
        
        if not self.param_data:
            return {}

        def _pdv_basename(raw):
            """Strip any directory prefix + common extensions from a PDV_FileName.

            Parameter files sometimes store the full Windows/POSIX path that the
            acquisition PC wrote (e.g. ``C:\\Users\\Administrator\\Desktop\\PDV_DATA\\<name>``).
            We normalise to the bare filename so matching works regardless of
            whether just the filename or the full path was recorded.
            """
            s = "" if raw is None else str(raw).strip()
            if not s or s.lower() == "nan":
                return ""
            if "\\" in s:
                s = s.rsplit("\\", 1)[-1]
            if "/" in s:
                s = s.rsplit("/", 1)[-1]
            s = s.strip().strip('"').strip("'")
            for _ext in (".csv", ".txt", ".dat", ".bin", ".h5", ".hdf5", ".trc"):
                if s.lower().endswith(_ext):
                    s = s[: -len(_ext)]
                    break
            return s

        base_name_norm = _pdv_basename(base_name)

        # Strategy 1: Try exact match first (if base_name is a key in param_data)
        if base_name in self.param_data:
            return self.param_data[base_name]
        if base_name_norm and base_name_norm in self.param_data:
            return self.param_data[base_name_norm]

        # Strategy 2a: Exact basename match against PDV_FileName / key
        # Handles rows whose PDV_FileName was saved as a full path on Windows.
        for key, param_entry in self.param_data.items():
            if not isinstance(param_entry, dict):
                continue
            key_base = _pdv_basename(key)
            param_pdv_base = _pdv_basename(param_entry.get('PDV_FileName', ''))
            if base_name_norm and (base_name_norm == key_base or base_name_norm == param_pdv_base):
                self.progress_signal.emit(
                    f"PDV_FileName basename match: {base_name} -> {param_pdv_base or key_base}"
                )
                return param_entry

        # Strategy 2b: Extract PDV filename and match to PDV_FileName column (exact match)
        pdv_pattern = re.search(r'(C\d+--\d{8}--\d{5})', base_name)
        if pdv_pattern:
            pdv_filename = pdv_pattern.group(1)
            for key, param_entry in self.param_data.items():
                if isinstance(param_entry, dict):
                    param_pdv = _pdv_basename(param_entry.get('PDV_FileName', ''))
                    if param_pdv == pdv_filename or _pdv_basename(key) == pdv_filename:
                        self.progress_signal.emit(f"PDV_FileName exact match: {base_name} -> {key}")
                        return param_entry

        # Strategy 3: Match by Exp_ID (material is same for all shots in an experiment)
        # This handles cases where shots from same experiment have different timestamps
        exp_id_pattern = re.search(r'([A-Z]+\d+-\d+)', base_name)
        if exp_id_pattern:
            exp_id = exp_id_pattern.group(1)
            # Search PDV_FileName column for any entry with this Exp_ID
            for key, param_entry in self.param_data.items():
                if isinstance(param_entry, dict):
                    param_pdv = _pdv_basename(param_entry.get('PDV_FileName', ''))
                    # Check if PDV_FileName contains the Exp_ID
                    if param_pdv and exp_id in param_pdv:
                        self.progress_signal.emit(f"Exp_ID match via PDV_FileName: {base_name} -> {param_pdv} (Exp_ID: {exp_id})")
                        return param_entry
                    # Also check Exp_ID column if available
                    param_exp_id = str(param_entry.get('Exp_ID', '')).strip()
                    if param_exp_id == exp_id:
                        self.progress_signal.emit(f"Exp_ID column match: {base_name} -> {key} (Exp_ID: {exp_id})")
                        return param_entry

        # Strategy 4: Partial match on key (use normalised basenames on both sides)
        for key in self.param_data.keys():
            key_base = _pdv_basename(key)
            if base_name_norm and key_base and (base_name_norm in key_base or key_base in base_name_norm):
                self.progress_signal.emit(f"Key partial match: {base_name} -> {key_base}")
                return self.param_data[key]

        return {}

    def get_material_from_igsn(self, base_name, matched_param=None):
        """
        Resolve a material name from the config igsn_material_map section.

        The IGSN is taken from the parameter data ('Sample_IGSN' column) when
        available, otherwise from the start of the filename
        (e.g. 'JHAMAL00016-004_2026-04-23_...' → matches 'JHAMAL00016-004').
        Map keys may be full IGSNs or parent-IGSN prefixes: the key
        'JHAMAL00016' matches 'JHAMAL00016-004', 'JHAMAL00016-005', etc.
        Longest matching key wins; matching is case-insensitive.

        Returns:
            Mapped material name, or None if no match.
        """
        if not self.igsn_material_map:
            return None

        # Collect candidate IGSN strings for this trace
        candidates = []
        if matched_param:
            for key in ('Sample_IGSN', 'Sample IGSN', 'IGSN'):
                val = str(matched_param.get(key, '')).strip()
                if val and val.lower() not in ('nan', 'none'):
                    candidates.append(val)
        if base_name:
            candidates.append(str(base_name).strip())

        # Longest key first so a full IGSN beats its parent prefix
        for map_key in sorted(self.igsn_material_map, key=lambda k: len(str(k)), reverse=True):
            key_lower = str(map_key).strip().lower()
            if not key_lower:
                continue
            for candidate in candidates:
                if candidate.lower().startswith(key_lower):
                    return str(self.igsn_material_map[map_key]).strip()
        return None

    def resolve_sample_material(self, base_name, matched_param=None):
        """
        Resolve the sample material for a trace.

        Priority:
        1. igsn_material_map from the config (IGSN → material), so a known
           sample's IGSN always wins over whatever is in the parameter file.
        2. 'Sample material' column in the parameter file (plus common variants)
        3. 'Unknown'

        Note: 'Flyer_material' is deliberately never used here — it describes
        the flyer, not the target sample being tested.
        """
        igsn_material = self.get_material_from_igsn(base_name, matched_param)
        if igsn_material and _is_valid_material_value(igsn_material):
            self.progress_signal.emit(
                f"  Material for {base_name} resolved via IGSN map: {igsn_material}")
            return igsn_material

        if matched_param:
            for key in ('Sample material', 'Sample Material', 'Sample_Material', 'Material', 'material'):
                if key in matched_param:
                    material_val = str(matched_param[key]).strip()
                    if _is_valid_material_value(material_val):
                        return material_val

        return 'Unknown'

    def refresh_material_column(self, df):
        """
        Re-resolve the 'Material' column of a summary dataframe loaded from disk.

        Summary CSVs from earlier runs can have stale/invalid Material values
        (e.g. '[]' from a parameter-file export bug, or values recorded before
        igsn_material_map existed in the config). Simply NaN-filling a stale
        column would leave those in place, so every row is re-resolved through
        resolve_sample_material — IGSN map first, parameter-file value as
        fallback — using the row's own Sample_IGSN/Material/Filename values.
        """
        if df is None or df.empty:
            return df

        igsn_col = next((c for c in ('Sample_IGSN', 'Sample IGSN', 'IGSN') if c in df.columns), None)
        material_source_col = 'Sample material' if 'Sample material' in df.columns else (
            'Material' if 'Material' in df.columns else None)
        name_col = 'Filename' if 'Filename' in df.columns else None

        def resolve_row(row):
            matched_param = {}
            if igsn_col is not None:
                matched_param['Sample_IGSN'] = row[igsn_col]
            if material_source_col is not None:
                matched_param['Sample material'] = row[material_source_col]
            base_name = row[name_col] if name_col is not None else ''
            return self.resolve_sample_material(base_name, matched_param)

        df['Material'] = df.apply(resolve_row, axis=1)
        return df

    def get_material_properties_from_config(self, material_name):
        """
        Resolve material properties strictly from configured sources -- no
        fallback/default values are substituted for an unrecognized material.

        Priority order:
        1. Config material_properties section (exact or case-insensitive match)
        2. Material properties database (material_properties.py)

        If neither source has the material, 'material_found' is False and
        'density'/'bulk_wave_speed'/'C_L' are all None. Callers must check
        'material_found' and treat an unresolved material as an error for
        that trace (e.g. record "Mat not found") rather than compute with a
        guessed density/velocity -- do not add a fallback here.

        Args:
            material_name: Name of the material to look up

        Returns:
            Dictionary with 'density', 'bulk_wave_speed', 'C_L', 'S',
            'material_found', 'material_name', 'source'

        'S' is the Hugoniot slope (Us = C0 + S*up). Like the other
        properties it has no default: if the config entry does not define
        it, 'S' is None and callers must flag the trace (e.g. "S missing")
        instead of substituting a value.
        """
        # Clean material name
        material_name = str(material_name).strip() if material_name else 'Unknown'

        def _config_entry(config_props, resolved_name):
            bulk_wave_speed = config_props.get('bulk_wave_speed', config_props.get('C0'))
            density = config_props.get('density')
            return {
                'density': float(density) if density is not None else None,
                'bulk_wave_speed': float(bulk_wave_speed) if bulk_wave_speed is not None else None,
                'C_L': float(config_props.get('C_L', bulk_wave_speed)) if config_props.get('C_L', bulk_wave_speed) is not None else None,
                'S': float(config_props['S']) if config_props.get('S') is not None else None,
                'material_found': True,
                'material_name': resolved_name,
                'source': 'config'
            }

        # Priority 1: Check config material_properties section (exact match)
        if self.material_properties and material_name in self.material_properties:
            return _config_entry(self.material_properties[material_name], material_name)

        # Try case-insensitive match in config
        if self.material_properties:
            for config_mat_name, config_props in self.material_properties.items():
                if config_mat_name.lower() == material_name.lower():
                    return _config_entry(config_props, config_mat_name)

        # Priority 2: Use the material_properties.py database -- also no fallback
        # (the database does not define Hugoniot slopes, so 'S' is None here)
        mat_props = get_material_properties(material_name)
        mat_props['source'] = 'database' if mat_props['material_found'] else None
        mat_props.setdefault('C_L', mat_props.get('bulk_wave_speed'))
        mat_props.setdefault('S', None)
        return mat_props

    def _compute_derived_shock_columns(self, t_aligned_ns, vel_clean, time_window, vel_window,
                                       threshold_velocity, acoustic_velocity, hugoniot_S):
        """Rise-time and strain-rate diagnostics from the aligned velocity trace.

        Ports the Binary_metal_analysis (HELIX v1) backward-walk methodology verbatim so
        the numbers stay identical across toolbox versions:
          - RiseTime_ArrivalToPeak_ns : sustained backward walk from peak to threshold.
          - RiseTime_/PlasticStrainRate_ {80_20, 90_10, MaxSlope} : %-of-peak backward walks.
          - Compressive_StrainRate_Avg/Ufs, Shock_Velocity_Us, Shock_Front_Width.
        The plastic/compressive-rate denominator uses the Hugoniot shock velocity
        Us = c_b + S*u_p when the material's Hugoniot slope S is available, else falls back
        to the bulk wave speed c_b (matching v1). Percentages come from spade_params
        (accepted as percent, e.g. 80, or fraction, e.g. 0.8). Returns a dict of
        column-name -> value (NaN where a quantity can't be formed).
        """
        out = {
            'Peak_Shock_Time_ns': np.nan,
            'RiseTime_ArrivalToPeak_ns': np.nan,
            'RiseTime_80_20_ns': np.nan,
            'RiseTime_90_10_ns': np.nan,
            'RiseTime_MaxSlope_ns': np.nan,
            'PlasticStrainRate_80_20_s^-1': np.nan,
            'PlasticStrainRate_90_10_s^-1': np.nan,
            'PlasticStrainRate_MaxSlope_s^-1': np.nan,
            'Compressive_StrainRate_Avg_s^-1': np.nan,
            'Compressive_StrainRate_Ufs_s^-1': np.nan,
            'Shock_Velocity_Us_m_s': np.nan,
            'Shock_Front_Width_um': np.nan,
        }
        try:
            if vel_window is None or len(vel_window) == 0 or not np.any(np.isfinite(vel_window)):
                return out

            def _as_fraction(val, default):
                # percent (80) or fraction (0.8) both accepted; matches v1 _as_fraction
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return default
                if not np.isfinite(f) or f <= 0:
                    return default
                return f / 100.0 if f > 1.0 else f

            sp = self.spade_params if isinstance(getattr(self, 'spade_params', None), dict) else {}
            hi80 = _as_fraction(sp.get('plastic_sr_high_pct'), 0.8)
            lo80 = _as_fraction(sp.get('plastic_sr_low_pct'), 0.2)
            hi90 = _as_fraction(sp.get('plastic_sr90_high_pct'), 0.9)
            lo90 = _as_fraction(sp.get('plastic_sr90_low_pct'), 0.1)
            ms_window = _as_fraction(sp.get('plastic_sr_maxslope_window_pct'), 0.2)
            ms_floor = _as_fraction(sp.get('plastic_sr_maxslope_floor_pct'), 0.05)
            ms_step = _as_fraction(sp.get('plastic_sr_maxslope_step_pct'), 0.05)

            # Peak = global max within the spall window (v1: not the first find_peaks hit,
            # to avoid latching onto a low-prominence pre-shock bump).
            First_Maxima = float(np.nanmax(vel_window))
            peak_t = float(time_window[int(np.nanargmax(vel_window))])
            out['Peak_Shock_Time_ns'] = peak_t
            # NaN-robust nearest-index: the aligned time array can carry a NaN in row 0
            # (detect_dns re-reads header'd velocity CSVs with header=None, pulling the
            # header text in as a NaN row), and plain argmin would latch onto that NaN
            # index and break every backward walk below.
            idx_peak = int(np.nanargmin(np.abs(t_aligned_ns - peak_t)))

            # RiseTime_ArrivalToPeak_ns: walk BACKWARD from peak until velocity drops to
            # threshold and STAYS there for a sustained 3 ns run (skips the known small
            # artifact dip between elastic precursor and plastic wave).
            dt_ns = float(np.nanmedian(np.abs(np.diff(t_aligned_ns)))) if len(t_aligned_ns) > 1 else np.nan
            if np.isfinite(dt_ns) and dt_ns > 0:
                n_sustain = max(3, int(round(3.0 / dt_ns)))
                t_low = np.nan
                run = 0
                for i in range(idx_peak, -1, -1):
                    v = vel_clean[i]
                    if np.isnan(v):
                        run = 0
                        continue
                    if v <= threshold_velocity:
                        run += 1
                        if run >= n_sustain:
                            t_low = t_aligned_ns[i + n_sustain - 1]
                            break
                    else:
                        run = 0
                if np.isfinite(t_low):
                    out['RiseTime_ArrivalToPeak_ns'] = peak_t - t_low

            # Shock velocity Us at this shot's peak particle velocity up = 0.5*v_peak_fs
            # (Hugoniot Us = C0 + S*up; C0 = bulk sound speed). Bulk-speed fallback if S
            # is undefined for the material -- same fallback v1 uses.
            v_peak_val = float(vel_clean[idx_peak])
            up_peak = 0.5 * v_peak_val
            if hugoniot_S is not None and np.isfinite(hugoniot_S) and np.isfinite(acoustic_velocity):
                Us_peak = acoustic_velocity + hugoniot_S * up_peak
            else:
                Us_peak = acoustic_velocity

            def _walk(frm, level):
                # first index at or below `level`, walking backward from `frm`
                if frm is None:
                    return None
                for i in range(frm, -1, -1):
                    v = vel_clean[i]
                    if np.isnan(v):
                        continue
                    if v <= level:
                        return i
                return None

            def _pct_pair(hi_frac, lo_frac):
                # backward walk to upper%, then further back to lower%; slope -> strain rate
                ih = _walk(idx_peak, hi_frac * v_peak_val)
                il = _walk(ih, lo_frac * v_peak_val)
                if ih is None or il is None:
                    return np.nan, np.nan
                dt = t_aligned_ns[ih] - t_aligned_ns[il]
                dv = float(vel_clean[ih]) - float(vel_clean[il])
                if dt > 0 and np.isfinite(Us_peak) and Us_peak > 0:
                    return dt, (dv / dt) * 1e9 / (2.0 * Us_peak)
                return (dt if dt > 0 else np.nan), np.nan

            out['RiseTime_80_20_ns'], out['PlasticStrainRate_80_20_s^-1'] = _pct_pair(hi80, lo80)
            out['RiseTime_90_10_ns'], out['PlasticStrainRate_90_10_s^-1'] = _pct_pair(hi90, lo90)

            # Max-slope: slide a fixed-width %-of-peak window down the rising edge, keep the
            # steepest position (largest dv/dt) rather than averaging one fixed span.
            best_slope = np.nan
            best_hi = None
            best_lo = None
            hi_frac = 1.0
            while hi_frac - ms_window >= ms_floor - 1e-9:
                lo_frac = hi_frac - ms_window
                ih = _walk(idx_peak, hi_frac * v_peak_val)
                il = _walk(ih, lo_frac * v_peak_val)
                if ih is not None and il is not None and ih != il:
                    dt_c = t_aligned_ns[ih] - t_aligned_ns[il]
                    dv_c = float(vel_clean[ih]) - float(vel_clean[il])
                    if dt_c > 0:
                        slope = dv_c / dt_c
                        if not np.isfinite(best_slope) or slope > best_slope:
                            best_slope = slope
                            best_hi = ih
                            best_lo = il
                hi_frac -= ms_step
            if best_hi is not None and best_lo is not None and np.isfinite(Us_peak) and Us_peak > 0:
                out['RiseTime_MaxSlope_ns'] = t_aligned_ns[best_hi] - t_aligned_ns[best_lo]
                out['PlasticStrainRate_MaxSlope_s^-1'] = best_slope * 1e9 / (2.0 * Us_peak)

            # Compressive strain rate: average (u_fs/t_peak scaled by 2*c_b) and the
            # Hugoniot free-surface variant (up / ((C0 + S*up) * t_r)); Ufs needs S.
            if np.isfinite(peak_t) and abs(peak_t) > 0.01 and np.isfinite(acoustic_velocity) and acoustic_velocity > 0:
                out['Compressive_StrainRate_Avg_s^-1'] = (First_Maxima / peak_t) * 1e9 / (2.0 * acoustic_velocity)
                if hugoniot_S is not None and np.isfinite(hugoniot_S):
                    up = First_Maxima / 2.0
                    denom = (acoustic_velocity + hugoniot_S * up) * (peak_t * 1e-9)
                    if denom > 0:
                        out['Compressive_StrainRate_Ufs_s^-1'] = up / denom

            # Shock-front width diagnostic: w = Us * t_r (t_r = RiseTime_80_20_ns).
            up_fs = First_Maxima / 2.0
            if hugoniot_S is not None and np.isfinite(hugoniot_S) and np.isfinite(acoustic_velocity):
                Us = acoustic_velocity + hugoniot_S * up_fs
            else:
                Us = acoustic_velocity
            out['Shock_Velocity_Us_m_s'] = Us
            tr_ns = out['RiseTime_80_20_ns']
            if np.isfinite(Us) and Us > 0 and np.isfinite(tr_ns) and tr_ns > 0:
                out['Shock_Front_Width_um'] = Us * (tr_ns * 1e-9) * 1e6
        except Exception as _e:
            try:
                self.progress_signal.emit(f"  [WARN] derived-column computation failed: {_e}")
            except Exception:
                pass
        return out

    def detect_dns_and_process_spall(self, file_path, base_name, density, acoustic_velocity,
                                     threshold_velocity, spall_start_time, spall_end_time,
                                     analysis_model='max_min', plot_path=None, plot_dir=None, sample_material='Unknown', **spade_kwargs):
        """
        Detect DNS (Did Not Spall) and process spall analysis following Binary_metal_analysis methodology.
        
        This function implements the detailed spall detection methodology:
        1. Loads and filters data
        2. Aligns trace to shock arrival
        3. Extracts time window
        4. Performs DNS detection (structural checks)
        5. Only calls SPADE for valid spall cases
        
        Returns:
            dict: Results dictionary with spall strength, DNS classification, and diagnostics
        """
        import pandas as pd
        
        results = {
            'Filename': base_name,
            'Spall_Strength_GPa': np.nan,
            'Spall_Strength_Unc_GPa': np.nan,
            'Spall_OK': False,
            'Spall_StrainRate_s^-1': np.nan,
            'First_Maxima_m_s': np.nan,
            'Minima_m_s': np.nan,
            'Second_Maxima_m_s': np.nan,
            'Pullback_Velocity_m_s': np.nan,
            'Pullback_Velocity_Unc_m_s': np.nan,
            'Processing_Status': 'Failed',
            'DNS_Classification': 'Unknown',
            'Analysis_Notes': '',
            # Derived shock-front diagnostics (populated below once the aligned trace and
            # spall window exist); pre-seeded so every row carries the same columns even
            # when a trace fails early.
            'Peak_Shock_Time_ns': np.nan,
            'RiseTime_ArrivalToPeak_ns': np.nan,
            'RiseTime_80_20_ns': np.nan,
            'RiseTime_90_10_ns': np.nan,
            'RiseTime_MaxSlope_ns': np.nan,
            'PlasticStrainRate_80_20_s^-1': np.nan,
            'PlasticStrainRate_90_10_s^-1': np.nan,
            'PlasticStrainRate_MaxSlope_s^-1': np.nan,
            'Compressive_StrainRate_Avg_s^-1': np.nan,
            'Compressive_StrainRate_Ufs_s^-1': np.nan,
            'Shock_Velocity_Us_m_s': np.nan,
            'Shock_Front_Width_um': np.nan,
        }

        # No material resolved (not in config material_properties, not in the
        # material_properties.py database, and no explicit Density_kg_m3/
        # Bulk_Wave_Speed_m_s override) -- do not guess; flag and stop here.
        if density is None or acoustic_velocity is None:
            results['Processing_Status'] = 'Mat not found'
            return results

        try:
            # Step 1: Load CSV and handle headers
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                results['Processing_Status'] = f'Failed: Could not load file: {str(e)}'
                return results
            
            # Detect header row
            has_header = False
            first_row = df.iloc[0] if len(df) > 0 else None
            if first_row is not None:
                first_row_str = ' '.join([str(x).lower() for x in first_row.values[:3]])
                if any(keyword in first_row_str for keyword in ['time', 'velocity', 'uncertainty']):
                    has_header = True
                    df = pd.read_csv(file_path, header=0)
                else:
                    df = pd.read_csv(file_path, header=None)
            
            # Extract first 3 columns
            if len(df.columns) < 3:
                results['Processing_Status'] = 'Failed: Insufficient columns (< 3)'
                return results
            
            time_s = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            velocity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            uncertainty = pd.to_numeric(df.iloc[:, 2], errors='coerce').values if len(df.columns) >= 3 else np.full_like(velocity, np.nan)
            
            if len(time_s) < 20:
                results['Processing_Status'] = 'Failed: Insufficient data points (< 20)'
                return results
            
            # Step 1b: Apply uncertainty filter
            max_vel = np.nanmax(np.abs(velocity))
            rel_unc = np.abs(uncertainty) / max(max_vel, 1e-9)
            vel_clean = velocity.copy()
            vel_clean[rel_unc >= 1.0] = np.nan
            
            # Step 2: Trace alignment to shock arrival using HEL t=0 method
            valid_mask = ~np.isnan(vel_clean)
            if not np.any(valid_mask):
                results['Processing_Status'] = 'Failed: No valid velocity data after filtering'
                return results
            
            vel_valid = vel_clean[valid_mask]
            time_valid = time_s[valid_mask]
            
            # Convert time to nanoseconds for HEL alignment
            time_valid_ns = time_valid * 1e9
            
            # Try HEL t=0 alignment first (same as HEL detection algorithm)
            use_hel_alignment = self.spade_params.get('use_hel_t0_alignment_for_plots', True)
            t0 = None
            alignment_method_used = None
            
            if use_hel_alignment:
                t0_method_plot = self.spade_params.get('hel_t0_method', 'alpss_signal_start')

                # Prefer the physical shock-arrival time ALPSS detected for this trace
                # (t_start_corrected), so this plot's t=0 matches the HEL detection and the
                # spall/binary-metal analysis rather than a per-trace velocity-domain foot.
                if t0_method_plot == 'alpss_signal_start':
                    alpss_t0_ns = self._load_alpss_signal_start_ns(file_path)
                    if alpss_t0_ns is not None:
                        t0 = alpss_t0_ns / 1e9  # ns -> seconds
                        alignment_method_used = "ALPSS"
                        t_aligned_ns = (time_s - t0) * 1e9

                if t0 is None:
                    # Fallback (or explicit velocity-domain methods): foot-of-rise detector.
                    min_velocity_threshold = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                    fallback_method = 'signal_start' if t0_method_plot == 'alpss_signal_start' else t0_method_plot
                    hel_t0, hel_t0_idx, time_aligned_hel = self.find_hel_t0_alignment(
                        time_valid_ns, vel_valid, min_velocity_threshold, method=fallback_method
                    )

                    if hel_t0 is not None and hel_t0_idx is not None:
                        # HEL alignment succeeded
                        t0 = hel_t0 / 1e9  # Convert back to seconds for consistency
                        alignment_method_used = "HEL"
                        t_aligned_ns = (time_s - t0) * 1e9
                    else:
                        # HEL alignment failed - fall back to threshold alignment
                        alignment_method_used = "threshold_fallback"
                        # Will fall through to threshold alignment below
            
            # Fallback to threshold alignment if HEL alignment not enabled or failed
            if t0 is None:
                threshold_idx = np.where(vel_valid >= threshold_velocity)[0]
                if len(threshold_idx) == 0:
                    results['Processing_Status'] = 'Failed: No shock arrival detected (threshold not reached)'
                    return results
                
                t0 = time_valid[threshold_idx[0]]
                t_aligned_ns = (time_s - t0) * 1e9
                if alignment_method_used is None:
                    alignment_method_used = "threshold"
            
            # Log alignment method used for debugging
            if alignment_method_used:
                self.progress_signal.emit(f"  [ALIGN] {base_name}: Using {alignment_method_used} alignment (t0={t0*1e9:.2f} ns)")
            
            # Step 3: Time window extraction
            window_mask = (~np.isnan(vel_clean)) & (t_aligned_ns >= spall_start_time) & (t_aligned_ns <= spall_end_time)
            if np.sum(window_mask) < 20:
                results['Processing_Status'] = 'Failed: Insufficient data points in spall window (< 20)'
                return results
            
            time_window = t_aligned_ns[window_mask]
            vel_window = vel_clean[window_mask]
            uncert_window = uncertainty[window_mask]

            # Derived shock-front diagnostics (rise times, plastic & compressive strain
            # rates, shock-front width) from the aligned trace. Ported verbatim from
            # Binary_metal_analysis (HELIX v1) so values match across toolbox versions;
            # denominator uses this material's Hugoniot slope S when available, else the
            # bulk wave speed. Failures here never abort spall processing.
            try:
                _mp = self.get_material_properties_from_config(sample_material)
                _hug_S = _mp.get('S') if isinstance(_mp, dict) else None
            except Exception:
                _hug_S = None
            results.update(self._compute_derived_shock_columns(
                t_aligned_ns, vel_clean, time_window, vel_window,
                threshold_velocity, acoustic_velocity, _hug_S))

            # Step 4: Build spall analysis configuration
            spall_config = {
                'min_recomp_ratio': spade_kwargs.get('min_recomp_ratio', self.spade_params.get('min_recomp_ratio', 0.1)),
                'min_recomp_velocity_ratio': spade_kwargs.get('min_recomp_velocity_ratio', self.spade_params.get('min_recomp_velocity_ratio', 1.05)),
                'min_recomp_time_ns': spade_kwargs.get('min_recomp_time_ns', self.spade_params.get('min_recomp_time_ns', 2.5)),
                'spall_smoothing_sigma_ns': spade_kwargs.get('spall_smoothing_sigma_ns', self.spade_params.get('spall_smoothing_sigma_ns', 1.5)),
            }

            # Initialize variables
            is_dns = False
            dns_reason = None

            # Step 4b: Gaussian smoothing for spall detection ONLY.
            # HEL detection and the plotted raw trace keep the ALPSS
            # Savitzky-Golay output. This second, more aggressive filter is
            # applied exactly once here; analyze_spall_horizontal_plateau
            # does no smoothing of its own. sigma is in ns of real time so
            # behaviour is independent of oscilloscope sample rate.
            # Set spall_smoothing_sigma_ns: 0 to disable.
            from scipy.ndimage import gaussian_filter1d
            sigma_ns = float(spall_config.get('spall_smoothing_sigma_ns', 1.5))
            dt_ns_window = float(np.median(np.diff(time_window))) if len(time_window) > 1 else 0.0
            if sigma_ns > 0 and dt_ns_window > 0:
                sigma_samples = sigma_ns / dt_ns_window
                vel_window_spall = gaussian_filter1d(vel_window.astype(float), sigma=sigma_samples, mode='nearest')
                self.progress_signal.emit(f"  [SMOOTH] {base_name}: Gaussian sigma={sigma_ns:.1f} ns ({sigma_samples:.1f} samples) applied for spall detection")
            else:
                vel_window_spall = vel_window.astype(float)

            # Step 5: Extract Key Velocities (basic max/min diagnostics; refined
            # by the horizontal-plateau fit results below when the fit succeeds)
            results['First_Maxima_m_s'] = np.nanmax(vel_window) if len(vel_window) > 0 else np.nan
            results['Minima_m_s'] = np.nanmin(vel_window) if len(vel_window) > 0 else np.nan
            results['Second_Maxima_m_s'] = np.nan
            results['Pullback_Velocity_m_s'] = np.nan
            results['Pullback_Velocity_Unc_m_s'] = np.nan
            pullback_unc = np.nan  # Initialize for later use in uncertainty calculation
            
            # Step 6: Spall Analysis - Horizontal Plateau 5-Segment
            spade_lines_info = None
            spade_intersections = None
            result_dict = None

            # Use horizontal plateau constraint method
            self.progress_signal.emit(f"  [HORIZ-PLAT] {base_name}: Using horizontal plateau 5-segment analysis")
            
            is_spall_plat, plat_reason, plat_results = self.analyze_spall_horizontal_plateau(
                time_window, vel_window_spall, uncert_window, density, acoustic_velocity, spall_config
            )
            
            if is_spall_plat:
                # Success - extract results
                result_dict = plat_results
                fits_dict = plat_results.get('fits', None)
                spade_intersections = plat_results.get('intersections', None)
                # Convert fits dictionary to list format expected by plotting function
                if fits_dict and isinstance(fits_dict, dict):
                    spade_lines_info = [
                        (fits_dict.get('seg1_rise', {}).get('m', 0), fits_dict.get('seg1_rise', {}).get('c', 0)),
                        (fits_dict.get('seg2_plateau', {}).get('m', 0), fits_dict.get('seg2_plateau', {}).get('c', 0)),
                        (fits_dict.get('seg3_release', {}).get('m', 0), fits_dict.get('seg3_release', {}).get('c', 0)),
                        (fits_dict.get('seg4_recomp', {}).get('m', 0), fits_dict.get('seg4_recomp', {}).get('c', 0)),
                        (fits_dict.get('seg5_tail', {}).get('m', 0), fits_dict.get('seg5_tail', {}).get('c', 0))
                    ]
                    print(f"  [HORIZ-PLAT] {base_name}: Success case - Converted fits dict to list: {len(spade_lines_info)} segments")
                    sys.stdout.flush()
                else:
                    print(f"  [HORIZ-PLAT] {base_name}: Success case - No fits dict found: fits_dict={fits_dict}")
                    sys.stdout.flush()
                    spade_lines_info = None
                
                spall_val = plat_results.get('Spall Strength (GPa)', 0)
                strain_val = plat_results.get('Strain Rate (s^-1)', 0)
                shock_val = plat_results.get('Peak Shock Stress (GPa)', 0)
                self.progress_signal.emit(f"  [HORIZ-PLAT] {base_name}: Spall={spall_val:.3f} GPa, Strain Rate={strain_val:.2e} s^-1, Shock={shock_val:.3f} GPa")
            else:
                # Horizontal plateau method detected DNS
                is_dns = True
                dns_reason = plat_reason
                results['DNS_Classification'] = dns_reason
                results['Spall_Strength_GPa'] = "DNS"
                self.progress_signal.emit(f"  [DNS] {base_name}: {plat_reason}")
                
                # Use results from method (now includes plateau velocity and shock stress)
                result_dict = plat_results if plat_results else {}
                # Ensure required fields are present even if method returned empty dict (fallback)
                if not result_dict or 'Plateau Mean Velocity (m/s)' not in result_dict:
                    plateau_vel = np.nanmax(vel_window[:len(vel_window)//3])
                    result_dict = {
                        'Processing Status': 'DNS',
                        'Plateau Mean Velocity (m/s)': plateau_vel,
                        'Peak Shock Stress (GPa)': density * acoustic_velocity * plateau_vel / 1e9,
                        'Peak Shock Stress Uncertainty (GPa)': 0.0
                    }
                else:
                    # Log what we got from the method
                    plat_vel = result_dict.get('Plateau Mean Velocity (m/s)', 'N/A')
                    shock_stress = result_dict.get('Peak Shock Stress (GPa)', 'N/A')
                    self.progress_signal.emit(f"  [HORIZ-PLAT] {base_name}: DNS case - Plateau={plat_vel} m/s, Shock={shock_stress} GPa")
                    # Extract fits and intersections for visualization (even for DNS cases)
                    fits_dict = result_dict.get('fits', None)
                    spade_intersections = result_dict.get('intersections', None)
                    # Convert fits dictionary to list format expected by plotting function
                    if fits_dict and isinstance(fits_dict, dict):
                        spade_lines_info = [
                            (fits_dict.get('seg1_rise', {}).get('m', 0), fits_dict.get('seg1_rise', {}).get('c', 0)),
                            (fits_dict.get('seg2_plateau', {}).get('m', 0), fits_dict.get('seg2_plateau', {}).get('c', 0)),
                            (fits_dict.get('seg3_release', {}).get('m', 0), fits_dict.get('seg3_release', {}).get('c', 0)),
                            (fits_dict.get('seg4_recomp', {}).get('m', 0), fits_dict.get('seg4_recomp', {}).get('c', 0)),
                            (fits_dict.get('seg5_tail', {}).get('m', 0), fits_dict.get('seg5_tail', {}).get('c', 0))
                        ]
                        print(f"  [HORIZ-PLAT] {base_name}: DNS case - Converted fits dict to list: {len(spade_lines_info)} segments, intersections: {len(spade_intersections) if spade_intersections else 0} points")
                        sys.stdout.flush()
                        if spade_intersections:
                            self.progress_signal.emit(f"  [HORIZ-PLAT] {base_name}: DNS case - Will show 5-segment lines for visualization")
                    else:
                        print(f"  [HORIZ-PLAT] {base_name}: DNS case - No fits dict found: fits_dict={fits_dict}, type={type(fits_dict)}")
                        sys.stdout.flush()
                        spade_lines_info = None
                        spade_intersections = None
            
            # DEBUG: Log processing status
            if result_dict:
                rdp_fit_status = result_dict.get('Processing Status', 'Unknown')
                if rdp_fit_status != 'Success' and rdp_fit_status != 'DNS':
                    self.progress_signal.emit(f"  [DEBUG] {base_name}: RDP-FIT Processing Status = {rdp_fit_status}")
                if 'Error Message' in result_dict:
                    self.progress_signal.emit(f"  [DEBUG]   RDP-FIT Error: {result_dict.get('Error Message', 'N/A')}")
            
            # Extract spall strength with flexible key matching
            spall_strength = np.nan
            if result_dict:
                for key in result_dict.keys():
                    if 'spall' in key.lower() and 'strength' in key.lower() and 'gpa' in key.lower() and 'unc' not in key.lower() and 'err' not in key.lower():
                        try:
                            val = result_dict[key]
                            if isinstance(val, str) and val.upper() == 'DNS':
                                spall_strength = "DNS"
                            else:
                                spall_strength = float(val) if pd.notna(val) else np.nan
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Extract uncertainty
            spall_unc = np.nan
            if result_dict:
                for key in result_dict.keys():
                    if ('unc' in key.lower() or 'err' in key.lower()) and 'spall' in key.lower() and 'gpa' in key.lower():
                        try:
                            spall_unc = float(result_dict[key]) if pd.notna(result_dict[key]) else np.nan
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Step 7: Uncertainty fallback calculation
            if pd.isna(spall_unc) and np.isfinite(pullback_unc) and np.isfinite(density) and np.isfinite(acoustic_velocity):
                spall_unc = 0.5 * density * acoustic_velocity * pullback_unc / 1e9
            
            # Extract strain rate
            strain_rate = np.nan
            if result_dict:
                strain_rate = result_dict.get('Strain Rate (s^-1)', np.nan)
                if pd.isna(strain_rate):
                    strain_rate = result_dict.get('Strain_Rate_s^-1', np.nan)
            
            # Extract strain rate uncertainty
            strain_rate_unc = np.nan
            if result_dict:
                strain_rate_unc = result_dict.get('Strain Rate Uncertainty (s^-1)', np.nan)
                if pd.isna(strain_rate_unc):
                    # Try alternative column names
                    for key in result_dict.keys():
                        if 'strain' in key.lower() and 'rate' in key.lower() and ('unc' in key.lower() or 'err' in key.lower()):
                            try:
                                strain_rate_unc = float(result_dict[key]) if pd.notna(result_dict[key]) else np.nan
                                if pd.notna(strain_rate_unc):
                                    break
                            except (ValueError, TypeError):
                                continue
            
            # Calculate Peak Shock Stress using EOS: U = c + S*u_p, then σ = ρ * U * u_p
            # Get Plateau Mean Velocity from RDP-FIT results (5-segment method)
            plateau_velocity = np.nan
            if result_dict:
                plateau_velocity = result_dict.get('Plateau Mean Velocity (m/s)', np.nan)
            if pd.isna(plateau_velocity):
                plateau_velocity = result_dict.get('Plateau_Mean_Velocity_ms', np.nan)
            
            # If plateau velocity not available from SPADE, shock stress = NaN (no fallback)
            peak_shock_stress = np.nan
            peak_shock_stress_unc = np.nan
            
            if pd.notna(plateau_velocity) and plateau_velocity > 0:
                # Get material properties for EOS calculation
                mat_props = self.get_material_properties_from_config(sample_material)

                # Hugoniot slope S must be defined for the material (config
                # material_properties section). No default/backup value is
                # ever substituted: without S the shock stress stays NaN and
                # the trace is flagged in Analysis_Notes.
                S = mat_props.get('S')
                if S is None:
                    note = f"ERROR: Hugoniot slope 'S' missing for material '{sample_material}' - Peak Shock Stress not calculated"
                    results['Analysis_Notes'] = note
                    self.progress_signal.emit(f"  [ERROR] {base_name}: {note}")
                    print(f"  [ERROR] {base_name}: {note}")
                    sys.stdout.flush()
                else:
                    # Calculate using EOS method
                    # plateau_velocity is free surface velocity (u_fs)
                    u_p = plateau_velocity / 2.0  # Particle velocity = free surface velocity / 2
                    shock_velocity = acoustic_velocity + S * u_p  # U = c + S*u_p
                    peak_shock_stress = density * shock_velocity * u_p * 1e-9  # σ = ρ * U * u_p (GPa)

                    # Calculate uncertainty if velocity uncertainty is available
                    velocity_unc = result_dict.get('Plateau Mean Velocity Uncertainty (m/s)', np.nan)
                    if pd.isna(velocity_unc):
                        velocity_unc = result_dict.get('Peak Velocity Uncertainty (m/s)', np.nan)

                    if pd.notna(velocity_unc) and velocity_unc > 0:
                        # Propagate uncertainty: δσ = ρ * (c + 2*S*u_p) * δu_p * 1e-9
                        u_p_unc = velocity_unc / 2.0  # Uncertainty in particle velocity
                        peak_shock_stress_unc = density * (acoustic_velocity + 2 * S * u_p) * u_p_unc * 1e-9
            
            # Step 8: Final classification
            # is_dns flag is already set by RDP check (Step 4) or SPADE verification (Step 6a)
            # dns_reason is set by RDP or SPADE check
            
            # For DNS cases, keep DNS status but update with plateau velocity if available
            if is_dns:
                # Keep DNS classification and status - DO NOT overwrite with SPADE values
                results['Spall_Strength_GPa'] = "DNS"
                results['Spall_Strength_Unc_GPa'] = np.nan
                results['Spall_OK'] = False
                results['Processing_Status'] = f'DNS: {dns_reason}' if dns_reason else 'DNS'
                results['DNS_Classification'] = dns_reason if dns_reason else 'DNS'  # Preserve the DNS reason
                # But still store plateau velocity if SPADE calculated it (from fallback)
                results['Plateau Mean Velocity (m/s)'] = plateau_velocity
                results['Peak Shock Stress (GPa)'] = peak_shock_stress
                results['Peak Shock Stress Uncertainty (GPa)'] = peak_shock_stress_unc
                results['Spall_StrainRate_s^-1'] = strain_rate  # May be NaN for DNS
                results['Strain_Rate_Uncertainty_s^-1'] = strain_rate_unc  # May be NaN for DNS
            else:
                # Valid spall case - use SPADE calculated values
                if result_dict.get('Processing Status') in ['Success', 'Success (Fallback)']:
                    results['Spall_OK'] = True
                    results['Processing_Status'] = 'Success'
                    results['DNS_Classification'] = 'Valid Spall'
                    # Use SPADE calculated spall strength for valid spall cases
                    results['Spall_Strength_GPa'] = spall_strength
                    results['Spall_Strength_Unc_GPa'] = spall_unc
                    results['Spall_StrainRate_s^-1'] = strain_rate
                    results['Strain_Rate_Uncertainty_s^-1'] = strain_rate_unc
                else:
                    results['Spall_OK'] = False
                    results['Processing_Status'] = result_dict.get('Processing Status', 'Failed: SPADE analysis failed')
                    results['DNS_Classification'] = 'Failed: SPADE analysis failed'
                    # For failed cases, still try to use SPADE values if available
                    results['Spall_Strength_GPa'] = spall_strength if pd.notna(spall_strength) else np.nan
                    results['Spall_Strength_Unc_GPa'] = spall_unc
                    results['Spall_StrainRate_s^-1'] = strain_rate
                    results['Strain_Rate_Uncertainty_s^-1'] = strain_rate_unc
            
            # Common fields for all cases
            results['Peak Shock Stress (GPa)'] = peak_shock_stress
            results['Peak Shock Stress Uncertainty (GPa)'] = peak_shock_stress_unc
            results['Plateau Mean Velocity (m/s)'] = plateau_velocity
            
            # Generate plot if plot_path or plot_dir is provided
            # When plot_dir is used, plots go into spalled/ or dns/ subfolders
            # Note: SPADE's calculate_spall_parameters now uses hybrid approach internally
            effective_plot_path = None
            if plot_dir:
                subfolder = 'spalled' if results.get('Spall_OK', False) else 'dns'
                out_subdir = os.path.join(plot_dir, subfolder)
                os.makedirs(out_subdir, exist_ok=True)
                effective_plot_path = os.path.join(out_subdir, f"{base_name}_spall_analysis.png")
            elif plot_path:
                effective_plot_path = plot_path

            if effective_plot_path:
                self.progress_signal.emit(f"  [PLOT] Generating individual spall plot for {base_name}: {effective_plot_path}")
                try:
                    self._plot_generic_spall_analysis(
                        effective_plot_path, time_window, vel_window, uncert_window,
                        results.get('Spall_Strength_GPa', spall_strength),
                        results.get('Spall_Strength_Unc_GPa', spall_unc),
                        base_name,
                        analysis_model='hybrid',  # Use 'hybrid' to indicate mixed approach
                        lines_info=spade_lines_info,
                        intersections=spade_intersections,
                        vel_smooth=vel_window_spall
                    )
                    # Verify plot was actually created
                    if os.path.exists(effective_plot_path):
                        self.progress_signal.emit(f"  [PLOT] ✓ Successfully generated plot for {base_name}: {effective_plot_path} (file exists)")
                    else:
                        self.progress_signal.emit(f"  [PLOT] ⚠ WARNING: Plot function returned without error, but file does not exist: {effective_plot_path}")
                except Exception as plot_error:
                    import traceback
                    self.progress_signal.emit(f"  [PLOT] ✗ ERROR: Could not generate spall plot for {base_name}: {str(plot_error)}")
                    self.progress_signal.emit(f"  [PLOT] Error traceback: {traceback.format_exc()}")
            else:
                print(f"  [PLOT] No plot_path provided for {base_name} (plot_individual may be disabled or spall_analysis not enabled)")
                sys.stdout.flush()
                self.progress_signal.emit(f"  [PLOT] No plot_path provided for {base_name} (plot_individual may be disabled or spall_analysis not enabled)")
        
        except Exception as e:
            import traceback
            results['Processing_Status'] = f'Failed: {str(e)}'
            results['DNS_Classification'] = 'Error'
            self.progress_signal.emit(f"Error in DNS detection for {base_name}: {str(e)}")
            self.progress_signal.emit(traceback.format_exc())
        
        return results


    def _plot_generic_spall_analysis(self, plot_path, time_window, vel_window, uncert_window,
                                     spall_strength, spall_unc, base_name,
                                     analysis_model='max_min', lines_info=None, intersections=None,
                                     vel_smooth=None):
        """Generate generic spall analysis plot for any analysis model.

        vel_smooth, when given, is the Gaussian-smoothed trace the spall
        detection actually ran on; it is overlaid on the raw trace.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot velocity trace (ALPSS output) and, if available, the
            # Gaussian-smoothed trace used for spall detection
            ax.plot(time_window, vel_window, 'b-', linewidth=1.0, alpha=0.5, label='Velocity (ALPSS)')
            if vel_smooth is not None and len(vel_smooth) == len(time_window):
                ax.plot(time_window, vel_smooth, 'k-', linewidth=1.3, alpha=0.9, label='Gaussian smoothed (spall)')
            
            # Plot uncertainty bands if available
            if uncert_window is not None and len(uncert_window) == len(vel_window):
                ax.fill_between(time_window, vel_window - uncert_window, vel_window + uncert_window,
                               alpha=0.2, color='blue', label='Uncertainty')
            
            # Overlay hybrid 5-segment fit if available (from strain rate calculation)
            # Note: We now always use hybrid_5_segment for strain rate, so lines_info should be available
            if lines_info and intersections:
                self._overlay_hybrid_segments(ax, time_window, lines_info, intersections)
            else:
                print(f"  [PLOT] {base_name}: No 5-segment lines to overlay - lines_info={lines_info}, intersections={intersections}")
                sys.stdout.flush()
            
            # Format title with uncertainty if available
            strength_val = None
            try:
                if spall_strength is not None and not isinstance(spall_strength, str):
                    strength_val = float(spall_strength)
                elif isinstance(spall_strength, str):
                    strength_val = float(spall_strength)
            except (TypeError, ValueError):
                strength_val = None
            
            if strength_val is not None and np.isfinite(strength_val):
                if spall_unc is not None and not np.isnan(spall_unc) and spall_unc > 0:
                    title = (
                        f'Spall Analysis: {base_name}\n'
                        f'Spall Strength: {strength_val:.3f} ± {spall_unc:.3f} GPa'
                    )
                else:
                    title = (
                        f'Spall Analysis: {base_name}\n'
                        f'Spall Strength: {strength_val:.3f} GPa'
                    )
            else:
                strength_label = str(spall_strength) if spall_strength is not None else "Unknown"
                title = (
                    f'Spall Analysis: {base_name}\n'
                    f'Spall Strength: {strength_label}'
                )
            
            ax.set_xlabel('Time (ns)', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add an explicit DNS stamp for clarity when DNS is detected.
            # (Some users rely on the visual DNS label rather than the title line.)
            try:
                is_dns = isinstance(spall_strength, str) and spall_strength.strip().upper() == "DNS"
            except Exception:
                is_dns = False
            if is_dns:
                ax.text(
                    0.98,
                    0.98,
                    "DNS\n(Did Not Spall)",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color="darkred",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="darkred", boxstyle="round,pad=0.3"),
                )
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.progress_signal.emit(f"  Saved spall plot: {os.path.basename(plot_path)}")
        except Exception as e:
            self.progress_signal.emit(f"  Warning: Could not generate spall plot: {str(e)}")

    def _overlay_hybrid_segments(self, ax, time_window, lines_info, intersections):
        """Overlay 5-segment hybrid model lines and key points on the velocity plot.
        
        For legacy 5-segment method:
        - Line 1: Average slope from start to peak (not forced through origin)
        - Line 3: Average downward slope to P3
        - Other lines: Use fitted equations
        """
        import numpy as np
        import pandas as pd
        
        if not lines_info or not intersections:
            print(f"  [PLOT] _overlay_hybrid_segments: Skipping - lines_info={lines_info}, intersections={intersections}")
            sys.stdout.flush()
            return
        
        print(f"  [PLOT] _overlay_hybrid_segments: Drawing 5 segments - lines_info has {len(lines_info)} segments, intersections has {len(intersections)} points")
        sys.stdout.flush()
        
        t_min = float(np.min(time_window)) if len(time_window) > 0 else 0.0
        t_max = float(np.max(time_window)) if len(time_window) > 0 else 0.0
        
        # Extract P1, P2, P3, P4 from intersections
        P1 = intersections[0] if len(intersections) > 0 else None
        P2 = intersections[1] if len(intersections) > 1 else None
        P3 = intersections[2] if len(intersections) > 2 else None
        P4 = intersections[3] if len(intersections) > 3 else None
        
        colors = ['blue', 'green', 'red', 'purple', 'brown']
        labels = ['Line 1 (Rise)', 'Line 2 (Plateau)', 'Line 3 (Pullback)', 'Line 4 (Recomp Rise)', 'Line 5 (Recomp Tail)']
        
        # Line 1: Rise from start to P1 - Use fitted slope (average slope from start to peak)
        if P1 and len(lines_info) > 0:
            if not any(pd.isna(coord) for coord in P1):
                m1, c1 = lines_info[0]
                if pd.notna(m1) and pd.notna(c1):
                    # Draw from start (t_min) to P1 using fitted line
                    t_line = np.array([t_min, float(P1[0])])
                    v_line = m1 * t_line + c1
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[0], label=f'{labels[0]} (m={m1:.2f})')
                else:
                    # Fallback: draw from (0,0) to P1
                    t_line = np.array([t_min, float(P1[0])])
                    v_line = np.array([0.0, float(P1[1])])
                    m1_fallback = (v_line[1] - v_line[0]) / (t_line[1] - t_line[0]) if t_line[1] != t_line[0] else 0
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[0], label=f'{labels[0]} (m={m1_fallback:.2f})')
        
        # Line 2: Plateau from P1 to P2 - Use fitted equation
        if P1 and P2 and len(lines_info) > 1:
            if not any(pd.isna(coord) for coord in P1) and not any(pd.isna(coord) for coord in P2):
                m2, c2 = lines_info[1]
                if pd.notna(m2) and pd.notna(c2):
                    t_line = np.linspace(float(P1[0]), float(P2[0]), 50)
                    v_line = m2 * t_line + c2
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[1], label=f'{labels[1]} (m={m2:.2f})')
        
        # Line 3: Pullback from P2 to P3 - Use fitted slope (average downward slope)
        if P2 and P3 and len(lines_info) > 2:
            if not any(pd.isna(coord) for coord in P2) and not any(pd.isna(coord) for coord in P3):
                m3, c3 = lines_info[2]
                if pd.notna(m3) and pd.notna(c3):
                    # Draw from P2 to P3 using fitted line
                    t_line = np.array([float(P2[0]), float(P3[0])])
                    v_line = m3 * t_line + c3
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[2], label=f'{labels[2]} (m={m3:.2f})')
                else:
                    # Fallback: draw forced through endpoints
                    t_line = np.array([float(P2[0]), float(P3[0])])
                    v_line = np.array([float(P2[1]), float(P3[1])])
                    m3_fallback = (v_line[1] - v_line[0]) / (t_line[1] - t_line[0]) if t_line[1] != t_line[0] else 0
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[2], label=f'{labels[2]} (m={m3_fallback:.2f})')
        
        # Line 4: Recomp Rise from P3 to P4 - Use fitted equation
        if P3 and P4 and len(lines_info) > 3:
            if not any(pd.isna(coord) for coord in P3) and not any(pd.isna(coord) for coord in P4):
                m4, c4 = lines_info[3]
                if pd.notna(m4) and pd.notna(c4):
                    t_line = np.linspace(float(P3[0]), float(P4[0]), 50)
                    v_line = m4 * t_line + c4
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[3], label=f'{labels[3]} (m={m4:.2f})')
        
        # Line 5: Recomp Tail from P4 to end - Use fitted equation
        if P4 and len(lines_info) > 4:
            if not any(pd.isna(coord) for coord in P4):
                m5, c5 = lines_info[4]
                if pd.notna(m5) and pd.notna(c5):
                    t_line = np.linspace(float(P4[0]), t_max, 50)
                    v_line = m5 * t_line + c5
                    ax.plot(t_line, v_line, '--', linewidth=2, color=colors[4], label=f'{labels[4]} (m={m5:.2f})')
        
        # Mark intersection points (P1, P2, P3, P4)
        point_colors = ['cyan', 'magenta', 'yellow', 'lime']
        point_labels = ['P1', 'P2', 'P3', 'P4']
        
        # Check if P3 and P4 overlap (same coordinates)
        p3_p4_overlap = False
        if P3 and P4 and not any(pd.isna(coord) for coord in P3) and not any(pd.isna(coord) for coord in P4):
            coord_tolerance = 0.1  # Consider overlapping if within 0.1 units
            p3_p4_overlap = (abs(float(P3[0]) - float(P4[0])) < coord_tolerance and 
                           abs(float(P3[1]) - float(P4[1])) < coord_tolerance)
        
        for idx, (point, color, label) in enumerate(zip([P1, P2, P3, P4], point_colors, point_labels)):
            if point and not any(pd.isna(coord) for coord in point):
                if idx == 2:  # P3
                    if p3_p4_overlap:
                        # P3 overlaps with P4 - use square marker and higher zorder to make it visible
                        ax.plot(point[0], point[1], 's', color=color, markersize=12,
                               markeredgewidth=3, markeredgecolor='black',
                               label=f'{label} ({point[0]:.1f}, {point[1]:.1f})', zorder=7)
                        # Add a small star marker slightly offset to make P3 even more visible
                        ax.plot(point[0] + 1.5, point[1] + 1.5, '*', color=color, markersize=10,
                               markeredgewidth=1.5, markeredgecolor='black', zorder=8,
                               label='_nolegend_')  # Don't add to legend
                    else:
                        # Normal P3 plotting
                        ax.plot(point[0], point[1], 's', color=color, markersize=10,
                               markeredgewidth=2, markeredgecolor='black',
                               label=f'{label} ({point[0]:.1f}, {point[1]:.1f})', zorder=6)
                elif idx == 3:  # P4
                    if p3_p4_overlap:
                        # P4 overlaps with P3 - draw with lower zorder and slight transparency
                        ax.plot(point[0], point[1], 'o', color=color, markersize=10,
                               markeredgewidth=2, markeredgecolor='black',
                               label=f'{label} ({point[0]:.1f}, {point[1]:.1f})', zorder=5, alpha=0.6)
                    else:
                        # Normal P4 plotting
                        ax.plot(point[0], point[1], 'o', color=color, markersize=10,
                               markeredgewidth=2, markeredgecolor='black',
                               label=f'{label} ({point[0]:.1f}, {point[1]:.1f})', zorder=5)
                else:
                    # Normal plotting for P1, P2
                    ax.plot(point[0], point[1], 'o', color=color, markersize=10,
                           markeredgewidth=2, markeredgecolor='black',
                           label=f'{label} ({point[0]:.1f}, {point[1]:.1f})', zorder=5)

    def run(self):
        try:
            # Add memory management
            import gc
            gc.collect()  # Force garbage collection before starting
            
            # Start timing the entire analysis
            self.start_time = time.time()

            # Import ALPSS and SPADE modules using absolute repo-rooted paths.
            # Relative sys.path entries can break in Colab/CLI when cwd changes.
            repo_root = os.path.dirname(os.path.abspath(__file__))
            alpss_path = os.path.join(repo_root, 'ALPSS')
            spade_path = os.path.join(repo_root, 'SPADE', 'spall_analysis_release')
            if alpss_path not in sys.path:
                sys.path.insert(0, alpss_path)
            if spade_path not in sys.path:
                sys.path.insert(0, spade_path)

            # Force-reload so code changes made while the GUI is open take effect
            # immediately on the next run (avoids Python module-cache stale reads).
            import importlib
            import alpss_main as _alpss_module
            importlib.reload(_alpss_module)
            alpss_main = _alpss_module.alpss_main
            detect_sample_rate = _alpss_module.detect_sample_rate

            import spall_analysis as _spade_module
            importlib.reload(_spade_module)
            from spall_analysis import process_velocity_files


            # Set matplotlib backend to Agg for thread safety on macOS
            # This prevents crashes when matplotlib runs in background thread
            import matplotlib
            matplotlib.use("Agg")
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Initialize successful_files list for both ALPSS and SPADE modes
            self.successful_files = []
            
            # Track total input files for summary
            if self.input_files:
                self.total_input_traces = len(self.input_files)
            else:
                self.total_input_traces = 0

            # Process ALPSS files if provided and not SPADE-only mode
            # Explicitly skip ALPSS in spade_only mode, even if input_files is set
            if self.analysis_mode == "spade_only":
                self.progress_signal.emit(f"Skipping ALPSS processing (analysis_mode = {self.analysis_mode}, input_files count = {len(self.input_files) if self.input_files else 0})")
            elif self.input_files:
                total_alpss_time = 0

                # Process all files
                files_to_process = self.input_files
                failed_files = []

                for i, input_file in enumerate(files_to_process):
                    self.progress_signal.emit(
                        f"ALPSS Processing file {i+1}/{len(files_to_process)}: {os.path.basename(input_file)}")

                    # Start timing for this file
                    file_start_time = time.time()

                    # Run ALPSS with error handling
                    # Memory management before ALPSS
                    gc.collect()
                    self.progress_signal.emit("Running ALPSS analysis...")

                    # Update ALPSS parameters with current file
                    alpss_params = self.alpss_params.copy()
                    alpss_params['filename'] = os.path.basename(input_file)
                    alpss_params['exp_data_dir'] = os.path.dirname(input_file)
                    alpss_params['out_files_dir'] = self.output_dir

                    # Resolve the actual sample rate up front so alpss_params (and the
                    # eventual run-config snapshot) reflects what alpss_main will really
                    # use, not just the configured fallback. alpss_main re-detects this
                    # itself internally, but never reports it back to the caller.
                    try:
                        detected_rate = detect_sample_rate(**alpss_params)
                        alpss_params['sample_rate'] = detected_rate
                        self._alpss_effective_sample_rate_by_file[alpss_params['filename']] = detected_rate
                    except Exception as rate_err:
                        self._alpss_effective_sample_rate_by_file[alpss_params['filename']] = {
                            'error': str(rate_err),
                            'fallback_used': self.alpss_params.get('sample_rate'),
                        }

                    # Debug: Print output directory being used for ALPSS
                    self.progress_signal.emit(f"[DEBUG] ALPSS output directory: {os.path.abspath(self.output_dir)}")

                    # Add experiment info if parameter data is available
                    if self.param_data:
                        base_name = os.path.splitext(
                            os.path.basename(input_file))[0]
                        # Use helper function for smart matching (exact, date-shot pattern, or partial)
                        exp_info = self.get_param_data_for_file(base_name)
                        if exp_info:
                            alpss_params['experiment_info'] = exp_info
                            # Handle different possible column names for experiment info
                            exp_id = exp_info.get('exp_id', exp_info.get('Exp_ID', 'Unknown'))
                            sample_material = exp_info.get('sample_material', exp_info.get('Flyer_material', 'Unknown'))
                            self.progress_signal.emit(
                                f"Linked to experiment: {exp_id} - {sample_material}")
                        else:
                            self.progress_signal.emit(
                                f"No experiment info found for {base_name}")
                            alpss_params['experiment_info'] = {}
                    else:
                        alpss_params['experiment_info'] = {}

                    # Pass image selection parameters to ALPSS
                    alpss_params['save_combined_plot'] = self.alpss_params.get(
                        'save_combined_plot', True)
                    alpss_params['save_iq_start_time_plot'] = self.alpss_params.get(
                        'save_iq_start_time_plot', False)
                    
                    # Handle smart selection for combined mode
                    smart_selection_enabled = self.alpss_params.get('smart_selection_enabled', False)
                    if smart_selection_enabled and self.analysis_mode == "both":
                        # For combined mode, save files needed for SPADE + enhanced analysis
                        alpss_params['save_velocity_csv'] = False
                        alpss_params['save_velocity_smooth_csv'] = False
                        alpss_params['save_velocity_uncert_csv'] = False
                        alpss_params['save_velocity_smooth_uncert_csv'] = True  # Main file needed for SPADE
                        alpss_params['save_results_csv'] = False
                        alpss_params['save_noise_csv'] = True  # Also save noise file for enhanced filtering
                    else:
                        # Pass output file selection parameters to ALPSS
                        alpss_params['save_velocity_csv'] = self.alpss_params.get('save_velocity_csv', True)
                        alpss_params['save_velocity_smooth_csv'] = self.alpss_params.get('save_velocity_smooth_csv', True)
                        alpss_params['save_velocity_uncert_csv'] = self.alpss_params.get('save_velocity_uncert_csv', True)
                        alpss_params['save_velocity_smooth_uncert_csv'] = self.alpss_params.get('save_velocity_smooth_uncert_csv', True)
                        alpss_params['save_results_csv'] = self.alpss_params.get('save_results_csv', True)
                        alpss_params['save_noise_csv'] = self.alpss_params.get('save_noise_csv', True)

                    try:
                        alpss_main(**alpss_params)

                        # Check if required files were generated based on analysis mode
                        base_name = os.path.splitext(
                            os.path.basename(input_file))[0]
                        
                        # Define required files based on analysis mode and smart selection
                        required_files = []
                        if smart_selection_enabled and self.analysis_mode == "both":
                            # For combined mode with smart selection, check for SPADE input file + noise file
                            required_files.append(os.path.join(
                                self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv"))
                            required_files.append(os.path.join(
                                self.output_dir, f"{base_name}--noise--frac.csv"))
                        else:
                            # Check all selected output files
                            if self.alpss_params.get('save_velocity_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--velocity.csv"))
                            if self.alpss_params.get('save_velocity_smooth_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--velocity--smooth.csv"))
                            if self.alpss_params.get('save_velocity_uncert_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--vel--uncert.csv"))
                            if self.alpss_params.get('save_velocity_smooth_uncert_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv"))
                            if self.alpss_params.get('save_results_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--results.csv"))
                            if self.alpss_params.get('save_noise_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--noise--frac.csv"))
                        
                        # Check if all required files exist
                        missing_files = [f for f in required_files if not os.path.exists(f)]
                        
                        if not missing_files:
                            self.successful_files.append(input_file)
                            self.progress_signal.emit(
                                f"✅ Successfully processed: {os.path.basename(input_file)}")
                        else:
                            missing_file_names = [os.path.basename(f) for f in missing_files]
                            failed_files.append(
                                (input_file, f"Missing files: {', '.join(missing_file_names)}"))
                            self.progress_signal.emit(
                                f"❌ Failed to generate required files: {os.path.basename(input_file)} - Missing: {', '.join(missing_file_names)}")

                    except Exception as e:
                        failed_files.append((input_file, str(e)))
                        self.progress_signal.emit(
                            f"❌ ALPSS processing failed for {os.path.basename(input_file)}: {str(e)}")
                        # Continue with next file instead of stopping
                        continue

                    # Calculate timing for this file
                    file_end_time = time.time()
                    file_time = file_end_time - file_start_time
                    total_alpss_time += file_time

                    self.progress_signal.emit(
                        f"Completed ALPSS analysis for {os.path.basename(input_file)} in {file_time:.2f} seconds")

                # Report ALPSS processing summary
                self.progress_signal.emit(f"=== ALPSS Processing Summary ===")
                self.progress_signal.emit(
                    f"Total files: {len(files_to_process)}")
                self.progress_signal.emit(
                    f"Successfully processed: {len(self.successful_files)}")
                self.progress_signal.emit(f"Failed: {len(failed_files)}")

                if failed_files:
                    self.progress_signal.emit(f"=== Failed Files ===")
                    for failed_file, error_msg in failed_files:
                        self.progress_signal.emit(
                            f"❌ {os.path.basename(failed_file)}: {error_msg}")

                # Report total ALPSS timing
                avg_time = total_alpss_time / \
                    len(self.successful_files) if self.successful_files else 0
                self.progress_signal.emit(
                    f"ALPSS Analysis Summary: {len(self.successful_files)} files processed in {total_alpss_time:.2f} seconds (avg: {avg_time:.2f}s per file)")
                
                # Save failed files list as CSV
                if failed_files:
                    try:
                        failed_files_data = []
                        for failed_file, error_msg in failed_files:
                            failed_files_data.append({
                                'input_file': os.path.basename(failed_file),
                                'input_file_path': failed_file,
                                'error_message': error_msg,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                        
                        failed_files_df = pd.DataFrame(failed_files_data)
                        failed_files_path = os.path.join(self.output_dir, 'failed_data_files.csv')
                        failed_files_df.to_csv(failed_files_path, index=False)
                        self.progress_signal.emit(f"Saved failed files list to: {failed_files_path}")
                    except Exception as e:
                        self.progress_signal.emit(f"Warning: Could not save failed files list: {str(e)}")

            # Run SPADE analysis if not ALPSS-only mode
            if self.analysis_mode != "alpss_only":
                spade_start_time = time.time()
                print(f"  [DEBUG] SPADE section: analysis_mode={self.analysis_mode}, spade_auto_mode={self.spade_auto_mode}")
                self.progress_signal.emit(f"  [DEBUG] SPADE section: analysis_mode={self.analysis_mode}, spade_auto_mode={self.spade_auto_mode}")
                if self.spade_auto_mode:
                    print(f"  [DEBUG] spade_auto_mode = True, successful_files count = {len(self.successful_files) if self.successful_files else 0}")
                    self.progress_signal.emit(f"  [DEBUG] spade_auto_mode = True, successful_files count = {len(self.successful_files) if self.successful_files else 0}")
                    # Automatic mode: use ALPSS output
                    self.progress_signal.emit(f"  [DEBUG] spade_auto_mode = True, successful_files count = {len(self.successful_files) if self.successful_files else 0}")
                    if self.successful_files:  # Use successful_files instead of self.input_files
                        print("Running SPADE analysis on ALPSS outputs...")
                        sys.stdout.flush()
                        self.progress_signal.emit(
                            "Running SPADE analysis on ALPSS outputs...")

                        # Find all velocity files generated by ALPSS
                        vel_files = []
                        missing_spade_files = []
                        for input_file in self.successful_files:  # Only check successful files
                            base_name = os.path.splitext(
                                os.path.basename(input_file))[0]
                            # Use the velocity with uncertainty file (contains
                            # smoothed velocity + uncertainty)
                            vel_file = os.path.join(
    self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                            if os.path.exists(vel_file):
                                vel_files.append(vel_file)
                            else:
                                missing_spade_files.append(base_name)

                        # Report SPADE file availability
                        self.progress_signal.emit(
                            f"=== SPADE File Availability ===")
                        self.progress_signal.emit(
                            f"ALPSS successful files: {len(self.successful_files)}")
                        self.progress_signal.emit(
                            f"SPADE input files found: {len(vel_files)}")
                        self.progress_signal.emit(
                            f"Missing SPADE input files: {len(missing_spade_files)}")

                        if missing_spade_files:
                            self.progress_signal.emit(
                                f"=== Missing SPADE Input Files ===")
                            for missing_file in missing_spade_files:
                                self.progress_signal.emit(
                                    f"❌ Missing: {missing_file}--vel-smooth-with-uncert.csv")

                        # Check if spall analysis is enabled (check both key names for compatibility)
                        spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False) or self.spade_params.get('experiment_spall_analysis', False)
                        experiment_spall_analysis = self.spade_params.get('experiment_spall_analysis', False)
                        self.progress_signal.emit(f"  [DEBUG] spall_analysis_enabled = {spall_analysis_enabled}, experiment_spall_analysis = {experiment_spall_analysis}, vel_files count = {len(vel_files) if vel_files else 0}")
                        print(f"  [DEBUG] spall_analysis_enabled = {spall_analysis_enabled}, experiment_spall_analysis = {experiment_spall_analysis}, vel_files count = {len(vel_files) if vel_files else 0}")
                        sys.stdout.flush()
                        
                        if vel_files and spall_analysis_enabled:
                            print(f"  [DEBUG] Entering spall detection loop: {len(vel_files)} files, spall_analysis_enabled={spall_analysis_enabled}")
                            sys.stdout.flush()
                            self.progress_signal.emit(f"  [DEBUG] Entering spall detection loop: {len(vel_files)} files, spall_analysis_enabled={spall_analysis_enabled}")
                            # Create SPADE output subdirectory
                            spade_output_dir = os.path.join(
                                self.output_dir, "SPADE_analysis")
                            os.makedirs(spade_output_dir, exist_ok=True)

                            # Debug: Check paths
                            self.progress_signal.emit(
                                f"Debug: output_dir = {self.output_dir}")
                            self.progress_signal.emit(
                                f"Debug: spade_output_dir = {spade_output_dir}")
                            self.progress_signal.emit(
                                f"Debug: Found {len(vel_files)} velocity files")

                            # Validate paths
                            if not os.path.exists(self.output_dir):
                                raise ValueError(
                                    f"Output directory does not exist: {self.output_dir}")
                            if not os.path.exists(spade_output_dir):
                                raise ValueError(
                                    f"SPADE output directory could not be created: {spade_output_dir}")

                            # Run SPADE with progress updates
                            self.progress_signal.emit(
                                f"SPADE Processing file 1/{len(vel_files)}: Starting SPADE analysis...")

                            spade_call_params = self.spade_params.copy()

                            # Add parameter data for enhanced legends if
                            # available
                            if self.param_data:
                                spade_call_params['param_data'] = self.param_data
                                self.progress_signal.emit(
                                    "Using parameter data for enhanced legends")
                            else:
                                spade_call_params['param_data'] = None

                            # Create subfolder for individual spall plots if plot_individual is enabled
                            plot_individual_enabled = self.spade_params.get('plot_individual', False)
                            print(f"  [SPALL-PLOT] Debug: plot_individual_enabled = {plot_individual_enabled}, spall_analysis_enabled = {spall_analysis_enabled}")
                            print(f"  [SPALL-PLOT] Config values: plot_individual={self.spade_params.get('plot_individual', 'NOT FOUND')}, experiment_spall_analysis={self.spade_params.get('experiment_spall_analysis', 'NOT FOUND')}")
                            sys.stdout.flush()
                            self.progress_signal.emit(f"  [SPALL-PLOT] Debug: plot_individual_enabled = {plot_individual_enabled}, spall_analysis_enabled = {spall_analysis_enabled}")
                            if plot_individual_enabled and spall_analysis_enabled:
                                spall_plots_dir = os.path.join(spade_output_dir, 'spall_plots')
                                os.makedirs(spall_plots_dir, exist_ok=True)
                                print(f"  [SPALL-PLOT] ✓ Individual spall plots will be saved to: {spall_plots_dir}/spalled/ or dns/")
                                sys.stdout.flush()
                                self.progress_signal.emit(f"  [SPALL-PLOT] ✓ Individual spall plots will be saved to: {spall_plots_dir}/spalled/ or dns/")
                            else:
                                spall_plots_dir = spade_output_dir
                                if not plot_individual_enabled:
                                    self.progress_signal.emit(f"  [SPALL-PLOT] ⚠ Individual spall plots disabled (plot_individual = {plot_individual_enabled})")
                                if not spall_analysis_enabled:
                                    self.progress_signal.emit(f"  [SPALL-PLOT] ⚠ Individual spall plots disabled (spall_analysis_enabled = {spall_analysis_enabled})")

                            try:
                                # Process files with DNS detection
                                results_list = []
                                
                                # Get spall detection parameters
                                threshold_velocity = self.spade_params.get('threshold_velocity_ms', 30.0)
                                spall_start_time = self.spade_params.get('spall_start_time_ns', 10.0)
                                spall_end_time = self.spade_params.get('spall_end_time_ns', 100.0)
                                # Note: analysis_model is no longer user-selectable
                                # Spall strength is always calculated using 'max_min' method
                                # Strain rate is always calculated using 'hybrid_5_segment' method
                                # The parameter is kept for backward compatibility with config files but ignored in calculations
                                analysis_model = 'hybrid'  # Placeholder - actual calculations use hybrid approach
                                
                                # Spall detection configuration
                                min_recomp_ratio = self.spade_params.get('min_recomp_ratio', 0.1)

                                spall_msg = f"  [SPALL] Detection Method: Horizontal Plateau 5-Segment Analysis"
                                self.progress_signal.emit(spall_msg)
                                spall_sigma_ns = self.spade_params.get('spall_smoothing_sigma_ns', 1.5)
                                prom_factor = self.spade_params.get('prominence_factor', 0.01)
                                peak_dist_ns = self.spade_params.get('peak_distance_ns', 3.0)
                                spall_msg2 = f"  [SPALL] smoothing_sigma={spall_sigma_ns:.1f} ns, prominence_factor={prom_factor:.3f}, peak_distance={peak_dist_ns:.1f} ns, min_recomp_ratio={min_recomp_ratio:.3f}"
                                self.progress_signal.emit(spall_msg2)
                                spall_msg3 = f"  [SPALL] Analysis window=[{spall_start_time:.1f}, {spall_end_time:.1f}] ns, threshold={threshold_velocity:.1f} m/s"
                                self.progress_signal.emit(spall_msg3)
                                print(spall_msg)  # Also print to terminal
                                print(spall_msg2)  # Also print to terminal
                                print(spall_msg3)  # Also print to terminal
                                
                                for i, vel_file in enumerate(vel_files):
                                    print(f"SPADE Processing file {i+1}/{len(vel_files)}: {os.path.basename(vel_file)}")
                                    self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(vel_files)}: {os.path.basename(vel_file)}")
                                    print(f"  [SPALL] Starting spall detection for {os.path.basename(vel_file)}")
                                    self.progress_signal.emit(f"  [SPALL] Starting spall detection for {os.path.basename(vel_file)}")
                                    
                                    # Get base name for material lookup
                                    base_name = os.path.splitext(os.path.basename(vel_file))[0]
                                    # Remove suffix if present
                                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                                        if base_name.endswith(suffix):
                                            base_name = base_name[:-len(suffix)]
                                            break
                                    
                                    # Get material properties
                                    # Priority: 'Sample material' column, then IGSN map fallback
                                    matched_param = self.get_param_data_for_file(base_name)
                                    sample_material = self.resolve_sample_material(base_name, matched_param)

                                    mat_props = self.get_material_properties_from_config(sample_material)
                                    density = matched_param.get('Density_kg_m3', mat_props['density']) if matched_param else mat_props['density']
                                    acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed']) if matched_param else mat_props['bulk_wave_speed']
                                    
                                    # Process with DNS detection
                                    # Generate plot dir if individual plots are enabled (plots go into spalled/ or dns/ subfolders)
                                    spall_plot_dir = None
                                    if plot_individual_enabled and spall_analysis_enabled:
                                        spall_plot_dir = spall_plots_dir
                                        print(f"  [PLOT] Will save individual spall plot to: {spall_plots_dir}/spalled/ or dns/")
                                        sys.stdout.flush()
                                        self.progress_signal.emit(f"  [PLOT] Will save individual spall plot to spalled/ or dns/ subfolder")
                                    else:
                                        print(f"  [PLOT] Individual plot disabled: plot_individual={plot_individual_enabled}, spall_analysis={spall_analysis_enabled}")
                                        sys.stdout.flush()
                                        self.progress_signal.emit(f"  [PLOT] Individual plot disabled: plot_individual={plot_individual_enabled}, spall_analysis={spall_analysis_enabled}")
                                    
                                    print(f"  [DEBUG] About to call detect_dns_and_process_spall for {base_name}, plot_dir={spall_plot_dir}")
                                    sys.stdout.flush()
                                    result = self.detect_dns_and_process_spall(
                                        file_path=vel_file,
                                        base_name=base_name,
                                        density=density,
                                        acoustic_velocity=acoustic_velocity,
                                        threshold_velocity=threshold_velocity,
                                        spall_start_time=spall_start_time,
                                        spall_end_time=spall_end_time,
                                        analysis_model=analysis_model,
                                        plot_dir=spall_plot_dir,
                                        sample_material=sample_material,  # Pass material for EOS calculation
                                        **{k: v for k, v in spade_call_params.items() if k not in ['plot_individual', 'density', 'acoustic_velocity', 'analysis_model']}
                                    )
                                    
                                    # Add material info
                                    result['Material'] = sample_material
                                    result['Density_kg_m3'] = density
                                    result['Acoustic_Velocity_m_s'] = acoustic_velocity
                                    
                                    results_list.append(result)
                                
                                # Save summary
                                if results_list:
                                    summary_df = pd.DataFrame(results_list)
                                    summary_path = os.path.join(spade_output_dir, "spall_summary.csv")
                                    
                                    # Check if file exists (from previous run)
                                    if os.path.exists(summary_path):
                                        self.progress_signal.emit(f"⚠ Overwriting existing spall_summary.csv with new RDP-based results")
                                    
                                    summary_df.to_csv(summary_path, index=False)
                                    print(f"✓ Saved spall summary with {len(results_list)} entries (RDP-based detection) to: {summary_path}")
                                    sys.stdout.flush()
                                    self.progress_signal.emit(f"✓ Saved spall summary with {len(results_list)} entries (RDP-based detection) to: {summary_path}")
                                    
                                    # Print detailed summary statistics
                                    valid_spall_mask = summary_df['Spall_Strength_GPa'].apply(
                                        lambda x: isinstance(x, (int, float)) and pd.notna(x) and not pd.isna(x)
                                    )
                                    valid_spall = valid_spall_mask.sum()
                                    dns_count = (summary_df['Spall_Strength_GPa'] == "DNS").sum()
                                    failed_count = len(results_list) - valid_spall - dns_count
                                    
                                    self.progress_signal.emit(f"")
                                    self.progress_signal.emit(f"=== Spall Analysis Summary (RDP-Based Detection) ===")
                                    self.progress_signal.emit(f"Total shots processed: {len(results_list)}")
                                    self.progress_signal.emit(f"Valid spall (numeric strength): {valid_spall} ({100*valid_spall/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"DNS (Did Not Spall): {dns_count} ({100*dns_count/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"Failed/Error: {failed_count} ({100*failed_count/len(results_list):.1f}%)")
                                    
                                    # Statistics for valid spall cases
                                    if valid_spall > 0:
                                        valid_df = summary_df[valid_spall_mask]
                                        strength_values = pd.to_numeric(valid_df['Spall_Strength_GPa'], errors='coerce')
                                        strength_values = strength_values[strength_values.notna()]
                                        
                                        if len(strength_values) > 0:
                                            self.progress_signal.emit(f"")
                                            self.progress_signal.emit(f"Valid Spall Statistics:")
                                            self.progress_signal.emit(f"  Mean strength: {strength_values.mean():.3f} GPa")
                                            self.progress_signal.emit(f"  Std strength: {strength_values.std():.3f} GPa")
                                            self.progress_signal.emit(f"  Min strength: {strength_values.min():.3f} GPa")
                                            self.progress_signal.emit(f"  Max strength: {strength_values.max():.3f} GPa")
                                            
                                            # Strain rate statistics if available
                                            strain_rates = pd.to_numeric(valid_df['Spall_StrainRate_s^-1'], errors='coerce')
                                            strain_rates = strain_rates[strain_rates.notna()]
                                            if len(strain_rates) > 0:
                                                self.progress_signal.emit(f"  Mean strain rate: {strain_rates.mean():.2e} s^-1")
                                                self.progress_signal.emit(f"  Std strain rate: {strain_rates.std():.2e} s^-1")
                                    
                                    # Comparison plots removed - hybrid velocity is now integrated into standard file
                            except Exception as e:
                                import traceback
                                self.progress_signal.emit(f"Error during SPADE spall analysis: {str(e)}")
                                self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

                            # Update progress after completion
                            for i in range(len(vel_files)):
                                self.progress_signal.emit(
                                    f"SPADE Processing file {i+1}/{len(vel_files)}: Completed")

                            spade_end_time = time.time()
                            spade_time = spade_end_time - spade_start_time
                            self.progress_signal.emit(
                                f"Completed SPADE analysis for {len(vel_files)} files in {spade_time:.2f} seconds")
                        elif vel_files and not spall_analysis_enabled:
                            print(f"Found {len(vel_files)} velocity files but spall analysis is disabled (spall_analysis_enabled={spall_analysis_enabled}, experiment_spall_analysis={self.spade_params.get('experiment_spall_analysis', 'not found')}) - skipping SPADE analysis")
                            self.progress_signal.emit(
                                f"Found {len(vel_files)} velocity files but spall analysis is disabled (spall_analysis_enabled={spall_analysis_enabled}, experiment_spall_analysis={self.spade_params.get('experiment_spall_analysis', 'not found')}) - skipping SPADE analysis")
                        elif not vel_files:
                            print(f"⚠ No velocity files found for spall detection (vel_files is empty). This may be normal if spall analysis is disabled or files are missing.")
                            self.progress_signal.emit(
                                f"⚠ No velocity files found for spall detection (vel_files is empty). This may be normal if spall analysis is disabled or files are missing.")
                        else:
                            print(f"⚠ Spall detection loop condition not met: vel_files={len(vel_files) if vel_files else 0}, spall_analysis_enabled={spall_analysis_enabled}")
                            self.progress_signal.emit(
                                f"⚠ Spall detection loop condition not met: vel_files={len(vel_files) if vel_files else 0}, spall_analysis_enabled={spall_analysis_enabled}")
                    else:
                        self.progress_signal.emit(
                            "No ALPSS files to process for automatic SPADE mode")
                else:
                    # Manual mode: use provided SPADE input files
                    if self.spade_input_files:
                        self.progress_signal.emit(
                            f"Running SPADE analysis on {len(self.spade_input_files)} manual input files...")

                        # Create SPADE output subdirectory
                        spade_output_dir = os.path.join(
                            self.output_dir, "SPADE_analysis")
                        os.makedirs(spade_output_dir, exist_ok=True)

                        # Run SPADE - for manual mode, we need to create a temporary directory with the files
                        # or use a different approach since SPADE expects
                        # input_folder and file_pattern
                        if len(self.spade_input_files) == 1:
                            # Single file - use its directory as input_folder
                            input_dir = os.path.dirname(
                                self.spade_input_files[0])
                            file_pattern = os.path.basename(
                                self.spade_input_files[0])
                        else:
                            # Multiple files - use the first file's directory
                            # and a pattern that matches all
                            input_dir = os.path.dirname(
                                self.spade_input_files[0])
                            # Always use standard velocity file (contains hybrid if enabled, STFT if not)
                            file_pattern = "*--vel-smooth-with-uncert.csv"

                        # Start SPADE processing
                        spade_start_time = time.time()
                        self.progress_signal.emit(
                            f"SPADE Processing file 1/{len(self.spade_input_files)}: Starting SPADE analysis...")

                        spade_call_params = self.spade_params.copy()

                        # Add parameter data for enhanced legends if available
                        if self.param_data:
                            spade_call_params['param_data'] = self.param_data
                            self.progress_signal.emit(
                                "Using parameter data for enhanced legends")
                        else:
                            spade_call_params['param_data'] = None

                        # Create subfolder for individual spall plots if plot_individual is enabled
                        plot_individual_enabled = self.spade_params.get('plot_individual', False)
                        spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False) or self.spade_params.get('experiment_spall_analysis', False)
                        self.progress_signal.emit(f"Debug: plot_individual_enabled = {plot_individual_enabled}, spall_analysis_enabled = {spall_analysis_enabled}")
                        
                        # Only run spall analysis if enabled
                        if spall_analysis_enabled:
                            if plot_individual_enabled:
                                spall_plots_dir = os.path.join(spade_output_dir, 'spall_plots')
                                os.makedirs(spall_plots_dir, exist_ok=True)
                                self.progress_signal.emit(f"Individual spall plots will be saved to: {spall_plots_dir}")
                            else:
                                spall_plots_dir = spade_output_dir
                                self.progress_signal.emit(f"⚠ Individual spall plots disabled (plot_individual = {plot_individual_enabled})")

                            try:
                                # Process files with DNS detection (same as automatic mode)
                                results_list = []
                                
                                # Get spall detection parameters
                                threshold_velocity = self.spade_params.get('threshold_velocity_ms', 30.0)
                                spall_start_time = self.spade_params.get('spall_start_time_ns', 10.0)
                                spall_end_time = self.spade_params.get('spall_end_time_ns', 100.0)
                                # Note: analysis_model is no longer user-selectable
                                # Spall strength is always calculated using 'max_min' method
                                # Strain rate is always calculated using 'hybrid_5_segment' method
                                # The parameter is kept for backward compatibility with config files but ignored in calculations
                                analysis_model = 'hybrid'  # Placeholder - actual calculations use hybrid approach
                                spall_msg = f"  [SPALL] Using hybrid approach: spall strength from 'max_min', strain rate from 'hybrid_5_segment', window=[{spall_start_time:.1f}, {spall_end_time:.1f}] ns, threshold={threshold_velocity:.1f} m/s"
                                self.progress_signal.emit(spall_msg)
                                print(spall_msg)  # Also print to terminal
                                
                                for i, vel_file in enumerate(self.spade_input_files):
                                    self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(self.spade_input_files)}: {os.path.basename(vel_file)}")
                                    
                                    # Get base name for material lookup
                                    base_name = os.path.splitext(os.path.basename(vel_file))[0]
                                    # Remove suffix if present
                                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                                        if base_name.endswith(suffix):
                                            base_name = base_name[:-len(suffix)]
                                            break
                                    
                                    # Get material properties
                                    # Priority: 'Sample material' column, then IGSN map fallback
                                    matched_param = self.get_param_data_for_file(base_name)
                                    sample_material = self.resolve_sample_material(base_name, matched_param)

                                    mat_props = self.get_material_properties_from_config(sample_material)
                                    density = matched_param.get('Density_kg_m3', mat_props['density']) if matched_param else mat_props['density']
                                    acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed']) if matched_param else mat_props['bulk_wave_speed']
                                    
                                    # Process with DNS detection
                                    # Generate plot dir if individual plots are enabled (plots go into spalled/ or dns/ subfolders)
                                    spall_plot_dir = None
                                    if plot_individual_enabled and spall_analysis_enabled:
                                        spall_plot_dir = spall_plots_dir
                                        print(f"  [PLOT] Will save individual spall plot to: {spall_plots_dir}/spalled/ or dns/")
                                        sys.stdout.flush()
                                        self.progress_signal.emit(f"  [PLOT] Will save individual spall plot to spalled/ or dns/ subfolder")
                                    else:
                                        print(f"  [PLOT] Individual plot disabled: plot_individual={plot_individual_enabled}, spall_analysis={spall_analysis_enabled}")
                                        sys.stdout.flush()
                                        self.progress_signal.emit(f"  [PLOT] Individual plot disabled: plot_individual={plot_individual_enabled}, spall_analysis={spall_analysis_enabled}")
                                    
                                    print(f"  [DEBUG] About to call detect_dns_and_process_spall for {base_name}, plot_dir={spall_plot_dir}")
                                    sys.stdout.flush()
                                    result = self.detect_dns_and_process_spall(
                                        file_path=vel_file,
                                        base_name=base_name,
                                        density=density,
                                        acoustic_velocity=acoustic_velocity,
                                        threshold_velocity=threshold_velocity,
                                        spall_start_time=spall_start_time,
                                        spall_end_time=spall_end_time,
                                        analysis_model=analysis_model,
                                        plot_dir=spall_plot_dir,
                                        sample_material=sample_material,  # Pass material for EOS calculation
                                        **{k: v for k, v in spade_call_params.items() if k not in ['plot_individual', 'density', 'acoustic_velocity', 'analysis_model']}
                                    )
                                    
                                    # Add material info
                                    result['Material'] = sample_material
                                    result['Density_kg_m3'] = density
                                    result['Acoustic_Velocity_m_s'] = acoustic_velocity
                                    
                                    results_list.append(result)
                                
                                # Save summary
                                if results_list:
                                    summary_df = pd.DataFrame(results_list)
                                    summary_path = os.path.join(spade_output_dir, "spall_summary.csv")
                                    
                                    # Check if file exists (from previous run)
                                    if os.path.exists(summary_path):
                                        self.progress_signal.emit(f"⚠ Overwriting existing spall_summary.csv with new RDP-based results")
                                    
                                    summary_df.to_csv(summary_path, index=False)
                                    print(f"✓ Saved spall summary with {len(results_list)} entries (RDP-based detection) to: {summary_path}")
                                    sys.stdout.flush()
                                    self.progress_signal.emit(f"✓ Saved spall summary with {len(results_list)} entries (RDP-based detection) to: {summary_path}")
                                    
                                    # Print detailed summary statistics
                                    valid_spall_mask = summary_df['Spall_Strength_GPa'].apply(
                                        lambda x: isinstance(x, (int, float)) and pd.notna(x) and not pd.isna(x)
                                    )
                                    valid_spall = valid_spall_mask.sum()
                                    dns_count = (summary_df['Spall_Strength_GPa'] == "DNS").sum()
                                    failed_count = len(results_list) - valid_spall - dns_count
                                    
                                    self.progress_signal.emit(f"")
                                    self.progress_signal.emit(f"=== Spall Analysis Summary (RDP-Based Detection) ===")
                                    self.progress_signal.emit(f"Total shots processed: {len(results_list)}")
                                    self.progress_signal.emit(f"Valid spall (numeric strength): {valid_spall} ({100*valid_spall/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"DNS (Did Not Spall): {dns_count} ({100*dns_count/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"Failed/Error: {failed_count} ({100*failed_count/len(results_list):.1f}%)")
                                    
                                    # Statistics for valid spall cases
                                    if valid_spall > 0:
                                        valid_df = summary_df[valid_spall_mask]
                                        strength_values = pd.to_numeric(valid_df['Spall_Strength_GPa'], errors='coerce')
                                        strength_values = strength_values[strength_values.notna()]
                                        
                                        if len(strength_values) > 0:
                                            self.progress_signal.emit(f"")
                                            self.progress_signal.emit(f"Valid Spall Statistics:")
                                            self.progress_signal.emit(f"  Mean strength: {strength_values.mean():.3f} GPa")
                                            self.progress_signal.emit(f"  Std strength: {strength_values.std():.3f} GPa")
                                            self.progress_signal.emit(f"  Min strength: {strength_values.min():.3f} GPa")
                                            self.progress_signal.emit(f"  Max strength: {strength_values.max():.3f} GPa")
                                            
                                            # Strain rate statistics if available
                                            strain_rates = pd.to_numeric(valid_df['Spall_StrainRate_s^-1'], errors='coerce')
                                            strain_rates = strain_rates[strain_rates.notna()]
                                            if len(strain_rates) > 0:
                                                self.progress_signal.emit(f"  Mean strain rate: {strain_rates.mean():.2e} s^-1")
                                                self.progress_signal.emit(f"  Std strain rate: {strain_rates.std():.2e} s^-1")
                            except Exception as e:
                                import traceback
                                self.progress_signal.emit(f"Error during SPADE spall analysis (manual mode): {str(e)}")
                                self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
                                
                                self.progress_signal.emit(f"")
                        else:
                            # Spall analysis is disabled
                            self.progress_signal.emit(f"⚠ Spall analysis is disabled (experiment_spall_analysis = false) - skipping spall analysis")

                        # Update progress after completion
                        for i in range(len(self.spade_input_files)):
                            self.progress_signal.emit(
                                f"SPADE Processing file {i+1}/{len(self.spade_input_files)}: Completed")

                        spade_end_time = time.time()
                        spade_time = spade_end_time - spade_start_time
                        self.progress_signal.emit(
                            f"Completed SPADE analysis for {len(self.spade_input_files)} files in {spade_time:.2f} seconds")
                    else:
                        self.progress_signal.emit(
                            "No SPADE input files provided")
            # After SPADE analysis, generate mean velocity file and combined
            # plots
            spade_output_dir = os.path.join(self.output_dir, "SPADE_analysis")
            os.makedirs(spade_output_dir, exist_ok=True)

            # Handle different experiment types (can be both)
            velocity_shots_enabled = self.spade_params.get('velocity_shots_enabled', True) or self.spade_params.get('experiment_velocity_shots', True)
            spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False) or self.spade_params.get('experiment_spall_analysis', False)

            if velocity_shots_enabled:
                self.progress_signal.emit("Running velocity shots analysis...")
                self.generate_velocity_shots_summary(spade_output_dir)
            
            if spall_analysis_enabled:
                self.progress_signal.emit("Running spall analysis...")
                self.generate_spall_analysis_summary(spade_output_dir)
            
            if not velocity_shots_enabled and not spall_analysis_enabled:
                self.progress_signal.emit("No experiment types selected, defaulting to velocity shots")
                self.generate_velocity_shots_summary(spade_output_dir)

            # Calculate total processing time
            total_end_time = time.time()
            total_time = total_end_time - \
                self.start_time if hasattr(self, 'start_time') else 0

            self.progress_signal.emit("All analysis completed successfully!")
            self.progress_signal.emit(
                f"Total processing time: {total_time:.2f} seconds")
            
            # If counters are still 0 (e.g. spade_only mode), try to populate from summary files
            if self.total_input_traces == 0:
                try:
                    enhanced_path = os.path.join(spade_output_dir, self._get_summary_filename())
                    vs_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
                    if os.path.exists(enhanced_path):
                        df = pd.read_csv(enhanced_path)
                        n_rows = len(df)
                        if n_rows > 0:
                            self.total_input_traces = n_rows
                            self.traces_plotted = n_rows
                            self.progress_signal.emit(f"[INFO] Summary populated from {self._get_summary_filename()} ({n_rows} traces)")
                    elif os.path.exists(vs_path):
                        vs_df = pd.read_csv(vs_path)
                        n_rows = len(vs_df)
                        if n_rows > 0:
                            self.total_input_traces = n_rows
                            self.traces_plotted = n_rows
                            self.progress_signal.emit(f"[INFO] Summary populated from velocity_shots_summary ({n_rows} traces)")
                except Exception as e:
                    self.progress_signal.emit(f"[DEBUG] Could not populate summary from files: {e}")

            # Print summary of trace analysis
            self.progress_signal.emit("")
            self.progress_signal.emit("=" * 70)
            self.progress_signal.emit("ANALYSIS SUMMARY")
            self.progress_signal.emit("=" * 70)
            self.progress_signal.emit(f"Total input traces: {self.total_input_traces}")
            self.progress_signal.emit(f"Traces plotted: {self.traces_plotted}")
            self.progress_signal.emit(f"Traces rejected: {self.traces_rejected}")
            if self.rejection_reasons:
                self.progress_signal.emit("")
                self.progress_signal.emit("Rejection reasons:")
                for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                    self.progress_signal.emit(f"  - {reason}: {count}")
            self.progress_signal.emit(f"Total analysis run time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            self.progress_signal.emit("=" * 70)
            
            self.finished_signal.emit(True, "Analysis completed successfully")
        except Exception as e:
            import traceback
            error_msg = str(e)
            # Avoid duplicate "Analysis failed:" prefix
            if error_msg.startswith("Analysis failed:"):
                error_msg = error_msg.replace("Analysis failed:", "").strip()
            self.progress_signal.emit(f"Error during analysis: {error_msg}")
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
            self.finished_signal.emit(False, error_msg)

    def _load_alpss_signal_start_ns(self, velocity_file_path):
        """
        Return the physical signal-start time (t=0) ALPSS detected for this trace, in
        nanoseconds, or None if it cannot be resolved.

        ALPSS writes ``t_start_corrected`` -- the IQ/spectrogram shock-arrival time that
        the spall (binary-metal) analysis uses as t=0 -- to the ``<base>--results.csv``
        sidecar as the row ``Signal Start Time`` (in seconds). The velocity CSV shares the
        same absolute time base (``time_f``), so subtracting this value zeroes the trace at
        the same physical arrival used everywhere else in the pipeline. Anchoring HEL to it
        keeps t=0 consistent across traces instead of re-deriving a per-trace velocity-domain
        foot of rise.
        """
        import os
        import numpy as np
        import pandas as pd

        if not velocity_file_path:
            return None

        # Derive the results sidecar path from the velocity filename.
        results_path = None
        for suffix in ('--vel-smooth-with-uncert.csv', '--velocity--smooth.csv',
                       '--velocity.csv'):
            if velocity_file_path.endswith(suffix):
                results_path = velocity_file_path[:-len(suffix)] + '--results.csv'
                break
        if results_path is None or not os.path.exists(results_path):
            return None

        try:
            res = pd.read_csv(results_path, header=None)
        except Exception:
            return None
        if res.shape[1] < 2:
            return None

        names = res.iloc[:, 0].astype(str).str.strip()
        match = res.loc[names == 'Signal Start Time']
        if match.empty:
            return None
        try:
            t_start_s = float(match.iloc[0, 1])
        except (ValueError, TypeError):
            return None
        if not np.isfinite(t_start_s):
            return None
        return t_start_s * 1e9  # seconds -> nanoseconds

    def find_hel_t0_alignment(self, time_data, velocity_data, min_velocity_threshold=10.0, method=None):
        """
        Find t=0 alignment at the "foot of the rise" — the point where the signal departs
        from baseline noise and the main rise begins.

        Algorithm ("signal_start", default):
            1. Estimate a robust peak velocity (95th percentile of |v|) to calibrate thresholds.
            2. Find the first sustained crossing of a "rise confirmation" threshold
               (max(0.20 * peak, min_velocity_threshold)) that stays high for ~1 ns.
               This locates a point unambiguously inside the rise, well above noise.
            3. Estimate baseline statistics (median + 1.4826·MAD) from the pre-rise region,
               backed off by a small margin to avoid contamination from the rise shoulder.
            4. Backtrack from the rise-confirmation point to the last sample where the
               velocity is still within `baseline_median + max(3·sigma, 2 m/s)`. That sample
               is the foot of the rise, i.e. the signal start.

        This is far more robust than scanning forward for the first v>0 sample, which
        locks onto noise excursions that happen to precede the real rise.

        Legacy algorithm ("first_positive") is preserved for comparison/backup.

        Parameters
        ----------
        time_data : array-like
            Time data in nanoseconds.
        velocity_data : array-like
            Velocity data in m/s.
        min_velocity_threshold : float, optional
            Minimum velocity floor used when scaling thresholds from peak (default: 10 m/s).
        method : {"signal_start", "first_positive", None}, optional
            Which algorithm to run. If None, reads ``self.spade_params['hel_t0_method']``
            (default: "signal_start"). "first_positive" selects the legacy implementation.

        Returns
        -------
        tuple
            (t0, t0_idx, time_aligned):
              - t0         : time at alignment point (ns), or None if not found
              - t0_idx     : index of alignment point, or None if not found
              - time_aligned : ``time_data - t0`` on success, original ``time_data`` otherwise
        """
        import numpy as np

        if len(time_data) == 0 or len(velocity_data) == 0:
            return None, None, time_data

        time_arr = np.asarray(time_data, dtype=float)
        vel_arr = np.asarray(velocity_data, dtype=float)

        # Resolve method (arg → spade_params → default)
        if method is None:
            try:
                method = self.spade_params.get('hel_t0_method', 'signal_start')
            except AttributeError:
                method = 'signal_start'
        method = str(method).lower()

        # Try the new robust algorithm first (unless explicitly forced to legacy)
        if method != 'first_positive':
            t0, t0_idx, aligned = self._find_t0_signal_start(
                time_arr, vel_arr, min_velocity_threshold=min_velocity_threshold
            )
            if t0 is not None:
                return t0, t0_idx, aligned
            # If signal_start fails, fall through to legacy as a last resort

        return self._find_t0_first_positive(
            time_arr, vel_arr, min_velocity_threshold=min_velocity_threshold
        )

    def _find_t0_signal_start(self, time_arr, vel_arr, min_velocity_threshold=10.0):
        """
        Robust signal-start detector: rise confirmation + baseline backtrack.

        See ``find_hel_t0_alignment`` for the algorithm description.
        """
        import numpy as np

        if len(time_arr) < 10:
            return None, None, time_arr

        finite_mask = np.isfinite(vel_arr)
        if np.sum(finite_mask) < 10:
            return None, None, time_arr

        # Robust peak estimate (ignore isolated spikes/outliers)
        peak_velocity = float(np.percentile(np.abs(vel_arr[finite_mask]), 95))
        peak_velocity = max(peak_velocity, float(min_velocity_threshold))

        # Rise-confirmation threshold: unambiguously above noise
        rise_threshold = max(0.20 * peak_velocity, float(min_velocity_threshold))

        # Sustain requirement: ~1 ns worth of samples (min 3)
        dts = np.diff(time_arr)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        dt_median = float(np.median(dts)) if dts.size > 0 else 1.0
        sustain_n = max(int(round(1.0 / dt_median)), 3) if dt_median > 0 else 3

        # Step 1: first sustained crossing of rise_threshold
        i_rise = None
        upper_limit = len(vel_arr) - sustain_n
        for i in range(max(0, sustain_n), upper_limit):
            window = vel_arr[i:i + sustain_n]
            if not np.all(np.isfinite(window)):
                continue
            # Require mean above threshold and no sample below 0.8 * threshold
            if np.mean(window) >= rise_threshold and np.min(window) >= 0.8 * rise_threshold:
                i_rise = i
                break

        if i_rise is None or i_rise < 3:
            return None, None, time_arr

        # Step 2: baseline from pre-rise region, with a small backoff
        # Use samples up to (i_rise - 2 * sustain_n), but keep at least 5 samples.
        backoff = max(2 * sustain_n, 3)
        baseline_end_idx = max(i_rise - backoff, 5)
        baseline_values = vel_arr[:baseline_end_idx]
        baseline_values = baseline_values[np.isfinite(baseline_values)]

        if len(baseline_values) < 5:
            return None, None, time_arr

        baseline_median = float(np.median(baseline_values))
        baseline_mad = float(np.median(np.abs(baseline_values - baseline_median)))
        baseline_sigma = 1.4826 * baseline_mad if baseline_mad > 0 else float(np.std(baseline_values))
        if not np.isfinite(baseline_sigma) or baseline_sigma <= 0:
            baseline_sigma = 1.0

        # Floor the "at baseline" tolerance so we don't over-constrain very quiet traces
        baseline_tol = max(3.0 * baseline_sigma, 2.0)
        baseline_upper = baseline_median + baseline_tol

        # Step 3: backtrack from i_rise to the last sample still within baseline
        i_foot = None
        for i in range(i_rise, 0, -1):
            v = vel_arr[i]
            if np.isfinite(v) and v <= baseline_upper:
                i_foot = i
                break

        if i_foot is None:
            # Rise sits above baseline for the whole pre-rise window (unlikely). Bail out.
            return None, None, time_arr

        t0 = float(time_arr[i_foot])
        time_aligned = time_arr - t0
        return t0, int(i_foot), time_aligned

    def _find_t0_first_positive(self, time_arr, vel_arr, min_velocity_threshold=10.0):
        """
        Legacy alignment: first sample where v>0 and v stays positive+rising for 10 ns,
        with adaptive thresholds (8% of peak, with floor = ``min_velocity_threshold``).
        """
        import numpy as np

        if len(time_arr) == 0 or len(vel_arr) == 0:
            return None, None, time_arr

        peak_velocity = np.nanmax(vel_arr) if len(vel_arr) > 0 else 0
        adaptive_threshold = max(peak_velocity * 0.08, min_velocity_threshold)
        adaptive_velocity_increase = max(peak_velocity * 0.04, min_velocity_threshold * 0.5)

        for candidate_idx in range(len(vel_arr)):
            if not (np.isfinite(vel_arr[candidate_idx]) and vel_arr[candidate_idx] > 0):
                continue

            candidate_time = time_arr[candidate_idx]
            window_end_time = candidate_time + 10.0
            window_mask = (time_arr >= candidate_time) & (time_arr <= window_end_time)
            window_indices = np.where(window_mask)[0]

            if len(window_indices) < 2:
                continue

            velocity_segment = vel_arr[window_indices]
            time_segment = time_arr[window_indices]

            if not np.all(velocity_segment > 0):
                continue

            time_diff = time_segment[-1] - time_segment[0]
            if time_diff <= 0:
                continue
            avg_slope = (velocity_segment[-1] - velocity_segment[0]) / time_diff

            initial_end_time = candidate_time + 1.0
            initial_mask = (time_arr >= candidate_time) & (time_arr <= initial_end_time)
            initial_indices = np.where(initial_mask)[0]
            if len(initial_indices) <= 1:
                continue

            initial_velocity_segment = vel_arr[initial_indices]
            initial_time_segment = time_arr[initial_indices]
            if len(initial_velocity_segment) <= 1:
                continue
            initial_time_diff = initial_time_segment[-1] - initial_time_segment[0]
            if initial_time_diff <= 0:
                continue

            initial_slope = (initial_velocity_segment[-1] - initial_velocity_segment[0]) / initial_time_diff

            # Continuous flat region check
            flat_threshold = 0.1
            flat_region_duration = 0.0
            max_flat_duration = 0.0
            for i in range(len(initial_velocity_segment) - 1):
                local_dt = initial_time_segment[i + 1] - initial_time_segment[i]
                if local_dt <= 0:
                    continue
                local_slope = (initial_velocity_segment[i + 1] - initial_velocity_segment[i]) / local_dt
                if abs(local_slope) < flat_threshold:
                    flat_region_duration += local_dt
                else:
                    max_flat_duration = max(max_flat_duration, flat_region_duration)
                    flat_region_duration = 0.0
            max_flat_duration = max(max_flat_duration, flat_region_duration)

            max_velocity_in_window = float(np.max(velocity_segment))
            velocity_increase = float(velocity_segment[-1] - velocity_segment[0])

            if (avg_slope > 0 and
                initial_slope >= 0.1 and
                max_flat_duration <= 1.0 and
                max_velocity_in_window >= adaptive_threshold and
                velocity_increase >= adaptive_velocity_increase):
                t0 = float(time_arr[candidate_idx])
                return t0, int(candidate_idx), time_arr - t0

        return None, None, time_arr

    def generate_velocity_shots_summary(self, spade_output_dir):
        """Generate velocity shots summary CSV with impact velocity calculations and combined velocity plot"""
        self.progress_signal.emit("Generating velocity shots summary...")
        
        # In SPADE-only mode, use the provided spade_input_files
        # In combined/automatic mode, use only files that were just processed (from successful_files)
        if self.analysis_mode == "spade_only" and self.spade_input_files:
            velocity_files = [f for f in self.spade_input_files if os.path.exists(f)]
            self.progress_signal.emit(f"SPADE-only mode: Using {len(velocity_files)} provided input files")
        else:
            # Use only files that were just processed by ALPSS (not all files in output_dir)
            velocity_files = []
            if hasattr(self, 'successful_files') and self.successful_files:
                for input_file in self.successful_files:
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    # Always use standard velocity file (contains hybrid if enabled, STFT if not)
                    vel_file = os.path.join(self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                    if os.path.exists(vel_file) and os.path.getsize(vel_file) > 0:
                        velocity_files.append(vel_file)
                self.progress_signal.emit(f"Using {len(velocity_files)} files from current ALPSS processing run")
            else:
                # Fallback: if successful_files not available, use glob but warn
                self.progress_signal.emit("Warning: Using all files in output directory (successful_files not available)")
                velocity_files = glob.glob(os.path.join(self.output_dir, '*--vel-smooth-with-uncert.csv'))
        
        # Filter out empty files
        valid_velocity_files = []
        for file_path in velocity_files:
            if os.path.getsize(file_path) > 0:
                valid_velocity_files.append(file_path)
            else:
                self.progress_signal.emit(f"Warning: Empty velocity file found: {os.path.basename(file_path)}")

        if not valid_velocity_files:
            self.progress_signal.emit(
                "No valid velocity files with uncertainty data found for velocity shots summary")
            return

        self.progress_signal.emit(f"Found {len(valid_velocity_files)} valid velocity files to process")
        velocity_files = valid_velocity_files

        # Debug: Report available parameter data
        if self.param_data:
            self.progress_signal.emit(f"Available parameter data keys: {list(self.param_data.keys())}")
            # Show sample of parameter data structure
            if self.param_data:
                sample_key = list(self.param_data.keys())[0]
                sample_data = self.param_data[sample_key]
                self.progress_signal.emit(f"Sample parameter data structure for '{sample_key}': {list(sample_data.keys())}")
        else:
            self.progress_signal.emit("No parameter data available")

        velocity_shots_data = []
        velocity_plot_data = []  # For combined velocity plot
        unaligned_entries = []

        for file_path in velocity_files:
            try:
                # Read velocity data with uncertainty
                df = pd.read_csv(file_path)
                if df.shape[1] < 3:  # Should have time, velocity, uncertainty
                    continue

                time_data = df.iloc[:, 0].values
                velocity_data = df.iloc[:, 1].values
                uncertainty_data = df.iloc[:, 2].values

                # Convert time to ns if needed
                if np.nanmax(time_data) < 1.0:
                    time_data = time_data * 1e9

                # Load noise fraction data and filter velocity data
                noise_fraction = None
                velocity_filtered = velocity_data.copy()

                # Try to load noise fraction data
                noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                if os.path.exists(noise_file):
                    try:
                        df_noise = pd.read_csv(noise_file)
                        if df_noise.shape[1] >= 1:
                            # Use last column
                            noise_fraction = df_noise.iloc[:, -1].values
                            if len(noise_fraction) == len(velocity_data):
                                # Filter out data points where noise fraction > 1
                                high_noise_mask = noise_fraction > 1.0
                                velocity_filtered[high_noise_mask] = np.nan
                                self.progress_signal.emit(
                                    f"Filtered {np.sum(high_noise_mask)} high-noise points from {os.path.basename(file_path)}")
                            else:
                                self.progress_signal.emit(
                                    f"Warning: Noise fraction length mismatch for {os.path.basename(file_path)}")
                        else:
                            self.progress_signal.emit(
                                f"Warning: Noise fraction file has insufficient columns: {os.path.basename(noise_file)}")
                    except Exception as e:
                        self.progress_signal.emit(
                            f"Warning: Could not read noise fraction for {os.path.basename(file_path)}: {e}")
                else:
                    self.progress_signal.emit(
                        f"Info: No noise fraction file found for {os.path.basename(file_path)}, using unfiltered data")

                # TRACE ALIGNMENT: Find t=0 when velocity reaches user-defined threshold
                velocity_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)  # m/s
                t0_idx = None
                t0 = None
                
                # Find first point where velocity exceeds threshold
                for i, vel in enumerate(velocity_filtered):
                    if not np.isnan(vel) and vel >= velocity_threshold:
                        t0_idx = i
                        break
                
                if t0_idx is None:
                    self.progress_signal.emit(
                        f"Warning: Could not find velocity threshold {velocity_threshold} m/s for {os.path.basename(file_path)}")
                    # Use original time data
                    time_aligned = time_data
                    aligned_ok = False
                    alignment_reason = f"velocity never reached threshold {velocity_threshold} m/s"
                else:
                    # Align time data to t=0 at velocity threshold
                    t0 = time_data[t0_idx]
                    time_aligned = time_data - t0
                    self.progress_signal.emit(
                        f"Aligned trace: t=0 at {t0:.2f} ns when velocity reached {velocity_threshold} m/s")
                    aligned_ok = True
                    alignment_reason = ""

                # Calculate mean velocity using aligned time and filtered data
                # First, determine the actual time range of the data
                time_range = np.max(time_aligned) - np.min(time_aligned)
                self.progress_signal.emit(f"Time range after alignment: {np.min(time_aligned):.1f} to {np.max(time_aligned):.1f} ns (span: {time_range:.1f} ns)")
                
                # Use adaptive time windows based on actual data range
                if time_range > 1000:  # Long time range (>1μs)
                    # Use windows relative to the middle of the data
                    mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
                    window_start = mid_time - 50  # 50ns window around middle
                    window_end = mid_time + 50
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
                elif time_range > 100:  # Medium time range (100ns-1μs)
                    # Use the middle 100ns of the data
                    mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
                    window_start = mid_time - 50
                    window_end = mid_time + 50
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
                else:  # Short time range (<100ns)
                    # Use the entire available range
                    window_start = np.min(time_aligned)
                    window_end = np.max(time_aligned)
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (full range)"
                
                # Calculate mean velocity in the adaptive window
                mask_window = (time_aligned >= window_start) & (time_aligned <= window_end)
                velocities_window = velocity_filtered[mask_window]
                velocities_window = velocities_window[~np.isnan(velocities_window)]
                
                if len(velocities_window) > 0:
                    mean_velocity_300_400 = np.mean(velocities_window)
                    self.progress_signal.emit(f"Using {len(velocities_window)} data points in window {time_window_used}")
                else:
                    # Fallback: use all available data
                    velocities_all = velocity_filtered[~np.isnan(velocity_filtered)]
                    if len(velocities_all) > 0:
                        mean_velocity_300_400 = np.mean(velocities_all)
                        time_window_used = f"All data ({len(velocities_all)} points)"
                        self.progress_signal.emit(f"Warning: No data in adaptive window, using all available data")
                    else:
                        mean_velocity_300_400 = np.nan
                        time_window_used = "No data available"
                        self.progress_signal.emit(f"Error: No velocity data available for {os.path.basename(file_path)}")

                # Get file base name
                base_name = os.path.splitext(os.path.basename(file_path))[0].replace(
                    '--vel-smooth-with-uncert', ''
                )

                # Track maximum observed velocity for diagnostics
                if np.any(~np.isnan(velocity_filtered)):
                    max_velocity_observed = float(np.nanmax(velocity_filtered))
                else:
                    max_velocity_observed = np.nan
                # Get parameter data if available using helper function
                param_info = self.get_param_data_for_file(base_name)
                if not param_info:
                    self.progress_signal.emit(f"No parameter match found for {base_name}")
                
                # Debug parameter data
                if param_info:
                    self.progress_signal.emit(
                        f"Found parameter data for {base_name}: {list(param_info.keys())}")
                else:
                    self.progress_signal.emit(
                        f"No parameter data found for {base_name}")

                # Create data row for velocity shots summary
                # HEL DETECTION
                hel_strength = np.nan
                hel_uncertainty = np.nan
                free_surface_velocity = np.nan
                hel_ok = False
                hel_time_detection = np.nan  # Time at HEL detection point (ns)
                C_L = np.nan  # Longitudinal wave velocity (will be set from material properties)
                hel_consecutive_points = 0  # Number of consecutive points in HEL segment
                hel_segment_time_ns = np.nan  # Time duration of HEL segment (ns)
                
                # Create HEL-aligned time using the configured t0 detection method.
                # Default ("signal_start") locates the foot of the main rise via a robust
                # baseline + backtrack algorithm; "first_positive" preserves legacy behavior.
                min_hel_velocity_for_t0 = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                t0_method = self.spade_params.get('hel_t0_method', 'alpss_signal_start')

                hel_t0 = None
                hel_t0_idx = None

                # Preferred: anchor HEL t=0 to the physical shock-arrival time ALPSS already
                # detected for this trace (t_start_corrected -- the same t=0 the spall/
                # binary-metal analysis uses). This keeps the HEL window consistent across
                # traces instead of re-deriving a per-trace velocity-domain foot of rise.
                if t0_method == 'alpss_signal_start':
                    alpss_t0_ns = self._load_alpss_signal_start_ns(file_path)
                    if alpss_t0_ns is not None:
                        hel_t0 = float(alpss_t0_ns)
                        hel_t0_idx = int(np.argmin(np.abs(time_data - hel_t0)))
                        time_aligned_iq = time_data - hel_t0
                        self.progress_signal.emit(
                            f"  [trace-t0] t=0 at {hel_t0:.2f} ns (ALPSS signal-start, matches spall analysis)"
                        )

                # Fallback (or explicit velocity-domain methods): foot-of-rise detector.
                if hel_t0 is None:
                    fallback_method = 'signal_start' if t0_method == 'alpss_signal_start' else t0_method
                    hel_t0, hel_t0_idx, time_aligned_iq_candidate = self.find_hel_t0_alignment(
                        time_data, velocity_filtered,
                        min_velocity_threshold=min_hel_velocity_for_t0,
                        method=fallback_method,
                    )
                    if hel_t0 is not None and hel_t0_idx is not None:
                        time_aligned_iq = time_aligned_iq_candidate
                        self.progress_signal.emit(
                            f"  [trace-t0] t=0 at {hel_t0:.2f} ns (method='{fallback_method}', for velocity alignment / plots)"
                        )
                    else:
                        # Last resort: use velocity threshold alignment if no valid point found
                        time_aligned_iq = time_aligned
                        hel_t0 = t0 if 't0' in locals() else (time_data[0] if len(time_data) > 0 else 0.0)
                        hel_t0_idx = t0_idx if 't0_idx' in locals() else 0
                        self.progress_signal.emit(
                            f"  [trace-t0] Warning: signal-start detection failed; falling back to velocity threshold alignment"
                        )
                
                hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                        self.spade_params.get('experiment_hel_detection', False))
                
                # Create HEL_plots folder early if plot_individual is enabled
                plot_individual_enabled = self.spade_params.get('plot_individual', False)
                if plot_individual_enabled and hel_detection_enabled:
                    hel_plots_dir = os.path.join(spade_output_dir, 'HEL_plots')
                    os.makedirs(hel_plots_dir, exist_ok=True)
                    self.progress_signal.emit(f"Individual HEL plots will be saved to: {hel_plots_dir}")
                
                if hel_detection_enabled:
                    try:
                        from scipy.ndimage import uniform_filter1d
                        
                        hel_start = self.spade_params.get('hel_start_time_ns', 0.0)
                        hel_end = self.spade_params.get('hel_end_time_ns', None)
                        angle_thresh_deg = self.spade_params.get('hel_angle_threshold_deg', 45.0)  # Kept for plotting compatibility
                        min_hel_velocity = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                        hel_rdp_epsilon = self.spade_params.get('hel_rdp_epsilon', 3.0)
                        hel_slope_drop_ratio = self.spade_params.get('hel_slope_drop_ratio', 0.2)
                        hel_min_plateau_duration = self.spade_params.get('hel_min_plateau_duration', 2.0)
                        t0_method_msg = self.spade_params.get('hel_t0_method', 'alpss_signal_start')
                        hel_msg = (f"  [HEL] Using RDP+Linear hybrid method: time window=[{hel_start:.1f}, {hel_end if hel_end is not None else 'None'}] ns "
                                  f"(aligned via t0_method='{t0_method_msg}'), min_velocity={min_hel_velocity:.1f} m/s, "
                                  f"rdp_epsilon={hel_rdp_epsilon:.1f} m/s, slope_drop_ratio={hel_slope_drop_ratio:.2f}, "
                                  f"min_plateau_duration={hel_min_plateau_duration:.1f} ns")
                        self.progress_signal.emit(hel_msg)
                        print(hel_msg)  # Also print to terminal
                        
                        # Step 1: Load data and filter by relative uncertainty
                        valid_mask = ~np.isnan(velocity_filtered)
                        if np.sum(valid_mask) > 5:
                            hel_time_all = time_aligned_iq[valid_mask]
                            hel_velocity_all = velocity_filtered[valid_mask]
                            hel_unc_all = uncertainty_data[valid_mask]
                            
                            # Filter by relative uncertainty: |uncertainty| / max(|velocity|, 1e-9) >= 1 -> NaN
                            max_vel = np.max(np.abs(hel_velocity_all))
                            rel_unc = np.abs(hel_unc_all) / max(max_vel, 1e-9)
                            noise_mask = rel_unc < 1.0
                            if np.sum(noise_mask) < 10:
                                noise_mask = valid_mask  # Fallback if too many points filtered
                            
                            hel_time_clean = hel_time_all[noise_mask]
                            hel_velocity_clean = hel_velocity_all[noise_mask]
                            hel_unc_clean = hel_unc_all[noise_mask]
                                
                            # Step 2: Build mask for HEL window
                            search_mask = hel_time_clean >= hel_start
                            if hel_end is not None and hel_end > hel_start:
                                search_mask &= (hel_time_clean <= hel_end)
                            if np.sum(search_mask) < 10:
                                search_mask = np.ones_like(hel_time_clean, dtype=bool)
                            
                            hel_time_window = hel_time_clean[search_mask]
                            hel_velocity_window = hel_velocity_clean[search_mask]
                            hel_unc_window = hel_unc_clean[search_mask]
                            
                            if len(hel_time_window) < 10:
                                self.progress_signal.emit(f"HEL: Insufficient data points in window for {base_name}")
                                hel_ok = False
                                
                                # Still create plot if plot_individual is enabled
                                if self.spade_params.get('plot_individual', False):
                                    try:
                                        # Create minimal gradient for plotting
                                        if len(hel_time_window) > 1:
                                            gradient_minimal = np.gradient(hel_velocity_window, hel_time_window)
                                            angles_deg_minimal = np.degrees(np.arctan(np.abs(gradient_minimal)))
                                        else:
                                            gradient_minimal = np.array([])
                                            angles_deg_minimal = np.array([])
                                        
                                        self._plot_individual_hel_detection(
                                            base_name,
                                            time_aligned_iq,
                                            velocity_filtered,
                                            hel_start,
                                            hel_end if hel_end not in [None] else np.max(time_aligned_iq),
                                            hel_time_window,
                                            hel_velocity_window,
                                            hel_strength,
                                            hel_uncertainty,
                                            sample_material,
                                            spade_output_dir,
                                            gradient=gradient_minimal if len(gradient_minimal) > 0 else None,
                                            angles_deg=angles_deg_minimal if len(angles_deg_minimal) > 0 else None,
                                            hel_segment_start=None,
                                            hel_segment_end=None,
                                            free_surface_velocity=None,
                                            angle_thresh_deg=angle_thresh_deg,
                                            U_0=None,
                                            t_0=None,
                                            t_hel=None,
                                            hel_ok=False,
                                            rdp_points=None,
                                            rise_slope_fit=None,
                                            plateau_slope_fit=None,
                                            t_rise=None,
                                            v_rise=None,
                                            t_plat=None,
                                            v_plat=None,
                                            rise_intercept=None,
                                            plateau_intercept=None,
                                            rdp_gradient_rise=None,
                                            rdp_gradient_plateau=None,
                                            rdp_angle_rise=None,
                                            rdp_angle_plateau=None,
                                            rdp_segment_gradients=None,
                                            rdp_segment_angles=None,
                                            slope_drop_ratio=self.spade_params.get('hel_slope_drop_ratio', 0.2),
                                            min_plateau_duration=self.spade_params.get('hel_min_plateau_duration', 2.0),
                                            duration_plat_rdp=None,
                                            min_hel_velocity=min_hel_velocity,
                                            hel_strain_rate=None,
                                        )
                                    except Exception as plot_error:
                                        self.progress_signal.emit(
                                            f"Warning: Could not create HEL plot for {base_name}: {str(plot_error)[:50]}")
                            else:
                                # Step 3 & 4: RDP+Linear Hybrid HEL Detection (Replaces gradient/angle method)
                                # RDP identifies candidate segments, then linear regression on raw data verifies slopes
                                # Prepare configuration for RDP detection
                                rdp_config = {
                                    'hel_rdp_epsilon': self.spade_params.get('hel_rdp_epsilon', 3.0),
                                    'hel_slope_drop_ratio': self.spade_params.get('hel_slope_drop_ratio', 0.2),
                                    'hel_min_plateau_duration': self.spade_params.get('hel_min_plateau_duration', 2.0),
                                    'hel_angle_threshold_deg': self.spade_params.get('hel_angle_threshold_deg', 30.0),
                                    'minimum_HEL_velocity_expected': self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                                }
                                
                                # Run RDP-based HEL detection
                                hel_found_rdp, hel_results_rdp = self.detect_hel_rdp(
                                    hel_time_window, hel_velocity_window, rdp_config
                                )
                                
                                # Initialize variables
                                hel_segment_start = None
                                hel_segment_end = None
                                hel_time_detection = np.nan
                                free_surface_velocity = np.nan
                                detection_used_gradient = False
                                
                                # Log RDP parameters (hybrid method)
                                self.progress_signal.emit(
                                    f"   [RDP+Linear] epsilon={rdp_config['hel_rdp_epsilon']:.1f} m/s, "
                                    f"slope_drop_ratio={rdp_config['hel_slope_drop_ratio']:.2f}, "
                                    f"min_plateau_duration={rdp_config['hel_min_plateau_duration']:.1f} ns")
                                
                                if hel_found_rdp and hel_results_rdp is not None:
                                    # Extract HEL detection results
                                    hel_time_detection = hel_results_rdp['hel_time_detection']
                                    free_surface_velocity = hel_results_rdp['free_surface_velocity']
                                    
                                    # Map RDP vertex back to original data indices for segment calculation
                                    # Find indices in original data closest to HEL detection time and segment end time
                                    hel_segment_start_time = hel_results_rdp.get('hel_segment_start_time', hel_time_detection)
                                    hel_segment_end_time = hel_results_rdp.get('hel_segment_end_time', hel_time_detection)
                                    
                                    # Find closest indices in original data
                                    hel_segment_start = np.argmin(np.abs(hel_time_window - hel_segment_start_time))
                                    hel_segment_end = np.argmin(np.abs(hel_time_window - hel_segment_end_time))
                                    
                                    # Ensure valid indices
                                    if hel_segment_start >= len(hel_velocity_window):
                                        hel_segment_start = 0
                                    if hel_segment_end >= len(hel_velocity_window):
                                        hel_segment_end = len(hel_velocity_window) - 1
                                    
                                    # Log detection results (hybrid method: RDP + linear regression)
                                    self.progress_signal.emit(
                                        f"   [RDP+Linear] HEL detected: t={hel_time_detection:.2f} ns, "
                                        f"v={free_surface_velocity:.2f} m/s, "
                                        f"rise_slope_fit={hel_results_rdp['rise_slope']:.2f} m/s/ns, "
                                        f"plateau_slope_fit={hel_results_rdp['plateau_slope']:.2f} m/s/ns")
                                    
                                    # Store RDP points for plotting (optional)
                                    rdp_points = hel_results_rdp.get('rdp_points', None)
                                    rdp_segment_gradients = hel_results_rdp.get('rdp_segment_gradients', None)
                                    rdp_segment_angles = hel_results_rdp.get('rdp_segment_angles', None)
                                else:
                                    # HEL not detected via RDP+Linear hybrid method
                                    error_msg = hel_results_rdp.get('error', 'No valid elastic-plastic transition found') if hel_results_rdp else 'RDP+Linear detection failed'
                                    self.progress_signal.emit(f"   [RDP+Linear] {error_msg}")
                                    # Still extract RDP points and gradients for visualization even when HEL not detected
                                    rdp_points = hel_results_rdp.get('rdp_points', None) if hel_results_rdp else None
                                    rdp_segment_gradients = hel_results_rdp.get('rdp_segment_gradients', None) if hel_results_rdp else None
                                    rdp_segment_angles = hel_results_rdp.get('rdp_segment_angles', None) if hel_results_rdp else None
                                
                                # For backward compatibility with plotting, create gradient arrays if needed
                                # (These may not be used if RDP detection succeeds, but needed for plotting)
                                gradient_smooth = None
                                angles_deg = None
                                if len(hel_time_window) > 1:
                                    gradient = np.gradient(hel_velocity_window, hel_time_window)
                                    # Minimal smoothing for plotting purposes only
                                    window_size = max(3, min(5, len(gradient) // 3))
                                    if window_size % 2 == 0:
                                        window_size += 1
                                    gradient_smooth = uniform_filter1d(gradient, size=window_size, mode='nearest')
                                    angles_deg = np.degrees(np.arctan(np.abs(gradient_smooth)))
                                sample_material = self.resolve_sample_material(base_name, param_info)
                                # Get material properties from config first, then database
                                mat_props = self.get_material_properties_from_config(sample_material)
                                density = param_info.get('Density_kg_m3', mat_props['density'])
                                acoustic_velocity = param_info.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed'])
                                # Get C_L from config (longitudinal wave velocity), fallback to acoustic_velocity if not specified
                                C_L = mat_props.get('C_L', acoustic_velocity)
                                
                                if mat_props['material_found']:
                                    source = mat_props.get('source', 'unknown')
                                    source_msg = f" (from {source})" if source != "unknown" else ""
                                    self.progress_signal.emit(
                                        f"Using {mat_props['material_name']} properties: ρ={density:.0f} kg/m³, c={acoustic_velocity:.0f} m/s{source_msg}")
                                else:
                                    self.progress_signal.emit(
                                        f"Mat not found for {base_name} (material='{sample_material}') -- HEL strain rate will not be calculated for this trace")
                                hel_plot_end = (
                                    hel_end if hel_end is not None and hel_end > hel_start else np.max(time_aligned_iq)
                                )

                                # Initialize U_0 and t_0 for plotting (used even when HEL detection fails)
                                U_0_for_plot = None
                                t_0_for_plot = None
                                
                                # Try to get U_0 and t_0 from HEL-aligned time (first velocity > 0 after 10 ns)
                                if hel_t0 is not None and hel_t0_idx is not None:
                                    if hel_t0_idx < len(velocity_filtered):
                                        U_0_for_plot = velocity_filtered[hel_t0_idx]
                                        t_0_for_plot = hel_t0
                                    else:
                                        # Fallback: find closest point to hel_t0
                                        closest_idx = np.argmin(np.abs(time_data - hel_t0))
                                        if closest_idx < len(velocity_filtered):
                                            U_0_for_plot = velocity_filtered[closest_idx]
                                            t_0_for_plot = time_data[closest_idx]
                                else:
                                    # Fallback: use first valid velocity point
                                    valid_idx = np.where(~np.isnan(velocity_filtered) & (velocity_filtered > 0))[0]
                                    if len(valid_idx) > 0:
                                        U_0_for_plot = velocity_filtered[valid_idx[0]]
                                        t_0_for_plot = time_data[valid_idx[0]] if valid_idx[0] < len(time_data) else 0.0
                                    else:
                                        U_0_for_plot = 0.0
                                        t_0_for_plot = 0.0

                                if hel_segment_start is not None and hel_segment_end is not None and not np.isnan(hel_time_detection):
                                    # RDP detected HEL - use RDP values for velocity and time
                                    # Calculate segment statistics from mapped indices
                                    hel_segment_indices = np.arange(hel_segment_start, hel_segment_end + 1)
                                    
                                    # Ensure indices are valid
                                    hel_segment_indices = hel_segment_indices[hel_segment_indices < len(hel_velocity_window)]
                                    
                                    # Calculate consecutive points count and segment time duration
                                    hel_consecutive_points = len(hel_segment_indices)
                                    if hel_segment_end < len(hel_time_window) and hel_segment_start < len(hel_time_window):
                                        hel_segment_time_ns = hel_time_window[hel_segment_end] - hel_time_window[hel_segment_start]
                                    else:
                                        hel_segment_time_ns = np.nan
                                    
                                    # Uncertainty: find closest point to HEL detection time in original data
                                    hel_time_idx = np.argmin(np.abs(hel_time_window - hel_time_detection))
                                    if hel_time_idx < len(hel_unc_window):
                                        u_unc = abs(hel_unc_window[hel_time_idx])
                                    else:
                                        # Fallback: use mean uncertainty in segment
                                        if len(hel_segment_indices) > 0:
                                            u_unc = np.mean(np.abs(hel_unc_window[hel_segment_indices]))
                                        else:
                                            u_unc = np.nan
                                    
                                    # Step 6: Check minimum HEL velocity constraint
                                    if np.isnan(free_surface_velocity) or abs(free_surface_velocity) < min_hel_velocity:
                                        # HEL velocity below threshold or NaN - reject this detection
                                        hel_ok = False
                                        hel_strength = np.nan
                                        hel_uncertainty = np.nan
                                        detected_vel = free_surface_velocity  # Save before setting to NaN
                                        free_surface_velocity = np.nan
                                        hel_time_detection = np.nan
                                        vel_str = f"{abs(detected_vel):.2f}" if not np.isnan(detected_vel) else "NaN"
                                        self.progress_signal.emit(
                                            f"HEL rejected for {base_name}: detected velocity {vel_str} m/s "
                                            f"is below minimum threshold of {min_hel_velocity:.1f} m/s")
                                    else:
                                        # Step 6: Compute HEL stress
                                        hel_strength = 0.5 * density * acoustic_velocity * abs(free_surface_velocity) / 1e9
                                        hel_uncertainty = 0.5 * density * acoustic_velocity * u_unc / 1e9
                                        hel_ok = True
                                        detection_used_gradient = True
                                        
                                        self.progress_signal.emit(
                                            f"HEL detected via gradient method: {hel_strength:.3f} GPa for {base_name} "
                                            f"(plateau at {hel_time_window[hel_segment_start]:.1f}-{hel_time_window[hel_segment_end]:.1f} ns, "
                                            f"{hel_consecutive_points} consecutive points, {hel_segment_time_ns:.3f} ns duration)")

                                        # Get U_0 and t_0 for strain rate slope calculation
                                        # Use HEL-aligned time (t=0 at first velocity > 0 after 10 ns increasing trend)
                                        if hel_t0 is not None:
                                            # Find velocity at t=0 in HEL-aligned time (hel_t0 point)
                                            iq_t0_idx_in_valid = np.argmin(np.abs(time_aligned_iq[valid_mask]))
                                            if iq_t0_idx_in_valid < len(hel_velocity_all):
                                                U_0_for_plot = hel_velocity_all[iq_t0_idx_in_valid]
                                                t_0_for_plot = hel_time_all[iq_t0_idx_in_valid]
                                            else:
                                                # Fallback: use first valid point
                                                U_0_for_plot = hel_velocity_all[0] if len(hel_velocity_all) > 0 else 0.0
                                                t_0_for_plot = hel_time_all[0] if len(hel_time_all) > 0 else 0.0
                                        else:
                                            # Fallback: use velocity threshold alignment
                                            if t0_idx is not None and t0_idx < len(velocity_filtered):
                                                U_0_for_plot = velocity_filtered[t0_idx]
                                                t_0_for_plot = time_aligned[t0_idx] if t0_idx < len(time_aligned) else 0.0
                                            else:
                                                valid_idx = np.where(~np.isnan(velocity_filtered))[0]
                                                if len(valid_idx) > 0:
                                                    U_0_for_plot = velocity_filtered[valid_idx[0]]
                                                    t_0_for_plot = time_aligned[valid_idx[0]] if valid_idx[0] < len(time_aligned) else 0.0
                                                else:
                                                    U_0_for_plot = 0.0
                                                    t_0_for_plot = 0.0

                                    # Always plot if plot_individual is enabled (even if HEL not detected)
                                    if self.spade_params.get('plot_individual', False):
                                        try:
                                            # Extract linear fit information for plotting
                                            # Now extracts best candidate values even if HEL not detected (for diagnostics)
                                            rise_slope_plot = hel_results_rdp.get('rise_slope', None) if hel_results_rdp else None
                                            plateau_slope_plot = hel_results_rdp.get('plateau_slope', None) if hel_results_rdp else None
                                            rise_intercept_plot = hel_results_rdp.get('rise_intercept', None) if hel_results_rdp else None
                                            plateau_intercept_plot = hel_results_rdp.get('plateau_intercept', None) if hel_results_rdp else None
                                            t_rise_plot = hel_results_rdp.get('t_rise', None) if hel_results_rdp else None
                                            v_rise_plot = hel_results_rdp.get('v_rise', None) if hel_results_rdp else None
                                            t_plat_plot = hel_results_rdp.get('t_plat', None) if hel_results_rdp else None
                                            v_plat_plot = hel_results_rdp.get('v_plat', None) if hel_results_rdp else None
                                            # Extract RDP segment gradients for plotting (candidate segment if HEL detected or best candidate)
                                            rdp_gradient_rise_plot = hel_results_rdp.get('rdp_gradient_rise', None) if hel_results_rdp else None
                                            rdp_gradient_plateau_plot = hel_results_rdp.get('rdp_gradient_plateau', None) if hel_results_rdp else None
                                            rdp_angle_rise_plot = hel_results_rdp.get('rdp_angle_rise', None) if hel_results_rdp else None
                                            rdp_angle_plateau_plot = hel_results_rdp.get('rdp_angle_plateau', None) if hel_results_rdp else None
                                            # Extract all RDP segment gradients for visualization (even when HEL not detected)
                                            rdp_segment_gradients_plot = hel_results_rdp.get('rdp_segment_gradients', None) if hel_results_rdp else None
                                            rdp_segment_angles_plot = hel_results_rdp.get('rdp_segment_angles', None) if hel_results_rdp else None
                                            # Extract RDP segment duration for diagnostics
                                            duration_plat_rdp_plot = hel_results_rdp.get('duration_plat_rdp', None) if hel_results_rdp else None
                                            
                                            self._plot_individual_hel_detection(
                                                base_name,
                                                time_aligned_iq,  # Use IQ-aligned time for HEL plot
                                                velocity_filtered,
                                                hel_start,
                                                hel_end if hel_end not in [None] else np.max(time_aligned_iq),
                                                hel_time_window,
                                                hel_velocity_window,
                                                hel_strength,
                                                hel_uncertainty,
                                                sample_material,
                                                spade_output_dir,
                                                gradient=gradient_smooth,
                                                angles_deg=angles_deg,
                                                hel_segment_start=hel_segment_start,
                                                hel_segment_end=hel_segment_end,
                                                free_surface_velocity=free_surface_velocity,
                                                angle_thresh_deg=angle_thresh_deg,
                                                U_0=U_0_for_plot,
                                                t_0=t_0_for_plot,
                                                t_hel=hel_time_detection,
                                                hel_ok=hel_ok,  # Pass hel_ok flag to plotting function
                                                rdp_points=rdp_points,
                                                rise_slope_fit=rise_slope_plot,
                                                plateau_slope_fit=plateau_slope_plot,
                                                t_rise=t_rise_plot,
                                                v_rise=v_rise_plot,
                                                t_plat=t_plat_plot,
                                                v_plat=v_plat_plot,
                                                rise_intercept=rise_intercept_plot,
                                                plateau_intercept=plateau_intercept_plot,
                                                rdp_gradient_rise=rdp_gradient_rise_plot,
                                                rdp_gradient_plateau=rdp_gradient_plateau_plot,
                                                rdp_angle_rise=rdp_angle_rise_plot,
                                                rdp_angle_plateau=rdp_angle_plateau_plot,
                                                rdp_segment_gradients=rdp_segment_gradients_plot,
                                                rdp_segment_angles=rdp_segment_angles_plot,
                                                slope_drop_ratio=rdp_config.get('hel_slope_drop_ratio', 0.2),
                                                min_plateau_duration=rdp_config.get('hel_min_plateau_duration', 2.0),
                                                duration_plat_rdp=duration_plat_rdp_plot,
                                                min_hel_velocity=min_hel_velocity,
                                                hel_strain_rate=hel_strain_rate if 'hel_strain_rate' in locals() else None,
                                            )
                                        except Exception as plot_error:
                                            self.progress_signal.emit(
                                                f"Warning: Could not create HEL plot for {base_name}: {str(plot_error)[:50]}")
                                else:
                                    # No valid HEL segment found - RDP+Linear detection failed
                                    if not hel_ok or hel_segment_start is None:
                                        self.progress_signal.emit(f"HEL: No elastic-plastic transition detected via RDP+Linear hybrid method in {base_name}")
                                        # Surface per-triplet diagnostics from the detector so the
                                        # user can see exactly which rule(s) each triplet failed.
                                        if isinstance(hel_results_rdp, dict):
                                            diag_msgs = hel_results_rdp.get('triplet_diagnostics', None)
                                            if diag_msgs:
                                                for _msg in diag_msgs:
                                                    self.progress_signal.emit(_msg)
                                        hel_consecutive_points = 0
                                        hel_segment_time_ns = np.nan
                                        
                                        # Calculate and report time spacing for reference
                                        if len(hel_time_window) > 1:
                                            time_diffs = np.diff(hel_time_window)
                                            mean_dt = np.mean(time_diffs)
                                            min_plateau_duration = rdp_config.get('hel_min_plateau_duration', 2.0)
                                            self.progress_signal.emit(
                                                f"   Time spacing: {mean_dt:.3f} ns/point, "
                                                f"min_plateau_duration: {min_plateau_duration:.1f} ns")
                                        
                                        # Always plot if plot_individual is enabled (even if HEL not detected)
                                        if self.spade_params.get('plot_individual', False):
                                            try:
                                                # Extract duration_plat_rdp from hel_results_rdp if available
                                                duration_plat_rdp_plot = hel_results_rdp.get('duration_plat_rdp', None) if hel_results_rdp else None
                                                
                                                self._plot_individual_hel_detection(
                                                    base_name,
                                                    time_aligned_iq,  # Use IQ-aligned time for HEL plot
                                                    velocity_filtered,
                                                    hel_start,
                                                    hel_end if hel_end not in [None] else np.max(time_aligned_iq),
                                                    hel_time_window,
                                                    hel_velocity_window,
                                                    hel_strength,
                                                    hel_uncertainty,
                                                    sample_material,
                                                    spade_output_dir,
                                                    gradient=gradient_smooth if 'gradient_smooth' in locals() else None,
                                                    angles_deg=angles_deg if 'angles_deg' in locals() else None,
                                                    hel_segment_start=None,
                                                    hel_segment_end=None,
                                                    free_surface_velocity=None,
                                                    angle_thresh_deg=angle_thresh_deg,
                                                    U_0=U_0_for_plot,
                                                    t_0=t_0_for_plot,
                                                    t_hel=None,
                                                    hel_ok=False,  # Pass hel_ok flag to plotting function
                                                    rdp_points=rdp_points,
                                                    rise_slope_fit=None,
                                                    plateau_slope_fit=None,
                                                    t_rise=None,
                                                    v_rise=None,
                                                    t_plat=None,
                                                    v_plat=None,
                                                    rise_intercept=None,
                                                    plateau_intercept=None,
                                                    rdp_gradient_rise=None,
                                                    rdp_gradient_plateau=None,
                                                    rdp_angle_rise=None,
                                                    rdp_angle_plateau=None,
                                                    rdp_segment_gradients=rdp_segment_gradients if 'rdp_segment_gradients' in locals() else None,
                                                    rdp_segment_angles=rdp_segment_angles if 'rdp_segment_angles' in locals() else None,
                                                    slope_drop_ratio=rdp_config.get('hel_slope_drop_ratio', 0.2),
                                                    min_plateau_duration=rdp_config.get('hel_min_plateau_duration', 2.0),
                                                    duration_plat_rdp=duration_plat_rdp_plot,
                                                    min_hel_velocity=min_hel_velocity,
                                                    hel_strain_rate=None,  # Not calculated if HEL not detected
                                                )
                                            except Exception as plot_error:
                                                self.progress_signal.emit(
                                                    f"Warning: Could not create HEL plot for {base_name}: {str(plot_error)[:50]}")
                    except Exception as hel_error:
                        import traceback
                        self.progress_signal.emit(f"HEL detection error for {base_name}: {str(hel_error)}")
                        self.progress_signal.emit(traceback.format_exc())
                
                # Calculate elastic shock strain rate if HEL was detected
                # HEL strain rate = slope from t=0 (in HEL plot) to HEL detection point
                # Both times are in HEL-aligned time frame (t=0 is HEL t=0 point)
                hel_strain_rate = np.nan
                hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                        self.spade_params.get('experiment_hel_detection', False))
                if hel_ok and hel_detection_enabled and np.isfinite(hel_time_detection) and np.isfinite(free_surface_velocity):
                    try:
                        # Get velocity at t=0 in HEL-aligned time frame
                        # In HEL-aligned time, t=0 is the HEL t=0 point (first velocity > 0 after 10 ns)
                        # hel_time_detection is already in HEL-aligned time frame
                        if time_aligned_iq is not None and len(time_aligned_iq) > 0:
                            # Find index closest to t=0 in HEL-aligned time
                            t0_idx_aligned = np.argmin(np.abs(time_aligned_iq))
                            if t0_idx_aligned < len(velocity_filtered):
                                U_0 = velocity_filtered[t0_idx_aligned]
                                t_0_ns = 0.0  # In HEL-aligned time, t=0 is the start
                            else:
                                U_0 = 0.0
                                t_0_ns = 0.0
                        else:
                            # Fallback: use velocity at HEL t=0 point from original data
                            if hel_t0 is not None and hel_t0_idx is not None and hel_t0_idx < len(velocity_filtered):
                                U_0 = velocity_filtered[hel_t0_idx]
                                t_0_ns = 0.0  # In HEL-aligned time, t=0 is the start
                            else:
                                U_0 = 0.0
                                t_0_ns = 0.0
                        
                        # Convert times from ns to seconds
                        # hel_time_detection is already in HEL-aligned time (relative to t=0)
                        t_hel_s = hel_time_detection * 1e-9
                        t_0_s = t_0_ns * 1e-9  # Should be 0.0 in HEL-aligned time
                        
                        # Calculate strain rate using C_L from material properties.
                        # C_L is None when the material couldn't be resolved -- no fallback.
                        if C_L is not None and np.isfinite(C_L) and np.isfinite(free_surface_velocity) and np.isfinite(U_0) and np.isfinite(t_hel_s) and np.isfinite(t_0_s):
                            hel_strain_rate = self.elastic_shock_strain_rate(
                                C_L=C_L,
                                U_hel=free_surface_velocity,
                                U_0=U_0,
                                t_hel=t_hel_s,
                                t_0=t_0_s
                            )
                            
                            # Check if strain rate is negative - reject HEL if so
                            if np.isfinite(hel_strain_rate) and hel_strain_rate < 0:
                                # Negative strain rate - reject this HEL detection
                                hel_ok = False
                                hel_strength = np.nan
                                hel_uncertainty = np.nan
                                free_surface_velocity = np.nan
                                hel_time_detection = np.nan
                                hel_strain_rate = np.nan
                                self.progress_signal.emit(
                                    f"HEL rejected for {base_name}: negative strain rate ({hel_strain_rate:.2e} s⁻¹)")
                            else:
                                # Strain rate calculated successfully
                                self.progress_signal.emit(
                                    f"   HEL strain rate for {base_name}: {hel_strain_rate:.2e} s⁻¹")
                        else:
                            hel_strain_rate = np.nan  # Ensure it's set to NaN if calculation fails
                            if C_L is None:
                                self.progress_signal.emit(f"   HEL strain rate for {base_name}: Not calculated (Mat not found -- no C_L)")
                            else:
                                self.progress_signal.emit(f"Warning: Invalid values for HEL strain rate calculation for {base_name} (C_L={C_L}, U_hel={free_surface_velocity}, U_0={U_0}, t_hel={t_hel_s}, t_0={t_0_s})")
                    except Exception as strain_error:
                        hel_strain_rate = np.nan  # Ensure it's set to NaN if exception occurs
                        self.progress_signal.emit(f"Warning: Could not calculate HEL strain rate for {base_name}: {str(strain_error)}")
                else:
                    # HEL not detected or not enabled - strain rate cannot be calculated
                    # Check hel_detection_enabled before hel_ok: when detection is off,
                    # hel_ok stays False and would wrongly imply "HEL not detected".
                    if not hel_detection_enabled:
                        self.progress_signal.emit(f"   HEL strain rate for {base_name}: Not calculated (HEL detection disabled)")
                    elif not hel_ok:
                        self.progress_signal.emit(f"   HEL strain rate for {base_name}: Not calculated (HEL not detected)")
                    elif not np.isfinite(hel_time_detection):
                        self.progress_signal.emit(f"   HEL strain rate for {base_name}: Not calculated (hel_time_detection is not finite)")
                    elif not np.isfinite(free_surface_velocity):
                        self.progress_signal.emit(f"   HEL strain rate for {base_name}: Not calculated (free_surface_velocity is not finite)")
                
                if not aligned_ok:
                    unaligned_entries.append({
                        'file_name': base_name,
                        'alignment_reason': alignment_reason,
                        'max_velocity_ms': max_velocity_observed,
                        'velocity_threshold_ms': velocity_threshold
                    })
                
                shot_data = {
                    'file_name': base_name,
                    'sample_material': sample_material,
                    'material_status': 'OK' if mat_props['material_found'] else 'Mat not found',
                    'mean_velocity_300_400ns_ms': mean_velocity_300_400,
                    'time_window_used': time_window_used,
                    'uncertainty_avg_ms': np.nanmean(uncertainty_data),
                    # Use HEL-based alignment (hel_t0) if available, otherwise fall back to threshold alignment (t0)
                    't0_ns': hel_t0 if hel_t0 is not None else (t0 if t0_idx is not None else np.nan),
                    'velocity_threshold_ms': velocity_threshold,
                    'max_velocity_ms': max_velocity_observed,
                    'aligned_ok': aligned_ok,
                    'alignment_reason': alignment_reason,
                    'hel_strength_gpa': hel_strength,
                    'hel_uncertainty_gpa': hel_uncertainty,
                    'hel_consecutive_points': hel_consecutive_points,
                    'hel_segment_time_ns': hel_segment_time_ns,
                    'free_surface_velocity_ms': free_surface_velocity,
                    'hel_ok': hel_ok,
                    'hel_strain_rate_s^-1': hel_strain_rate,
                }

                # Add ALL parameter file data as extra columns (without 'param_' prefix)
                for key, value in param_info.items():
                    shot_data[key] = value

                velocity_shots_data.append(shot_data)

                # Create data for combined velocity plot (use aligned time and filtered data)
                plot_data = {
                    'time_ns': time_aligned,
                    'velocity_ms': velocity_filtered,  # Use filtered velocity
                    'file_name': base_name,
                    'param_info': param_info
                }
                velocity_plot_data.append(plot_data)

            except Exception as e:
                self.progress_signal.emit(
                    f"Error processing {os.path.basename(file_path)}: {str(e)}")
                continue

        # Save velocity shots summary
        if velocity_shots_data:
            # Ensure all parameter columns from the parameter input file are included in the summary
            all_param_columns = set()
            # 1) Collect from parameter file(s) if available
            if self.param_data:
                try:
                    for _key, param_dict in self.param_data.items():
                        if isinstance(param_dict, dict):
                            all_param_columns.update(param_dict.keys())
                except Exception:
                    pass
            # 2) Also include any parameter keys already merged into rows
            for shot_data in velocity_shots_data:
                for key in shot_data.keys():
                    if key not in ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms',
                                   't0_ns', 'velocity_threshold_ms', 'max_velocity_ms', 'aligned_ok', 'alignment_reason']:
                        all_param_columns.add(key)

            # Add missing parameter columns with NaN values to each row
            for shot_data in velocity_shots_data:
                for param_col in all_param_columns:
                    if param_col not in shot_data:
                        shot_data[param_col] = np.nan

            velocity_shots_df = pd.DataFrame(velocity_shots_data)
            total_shots = len(velocity_shots_df)
            
            # Apply MAD filter to remove outlier peak velocities if enabled
            # Filter is applied per material and laser energy bracket
            mad_filter_enabled = self.spade_params.get('mad_filter_enabled', False)
            mad_filter_threshold = self.spade_params.get('mad_filter_threshold', 3.0)
            
            if mad_filter_enabled and 'max_velocity_ms' in velocity_shots_df.columns:
                # Initialize mad_filter_keep column
                velocity_shots_df['mad_filter_keep'] = True
                
                # Find material column
                material_col = None
                for col_name in velocity_shots_df.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                
                if material_col is None:
                    # Try common alternatives
                    for col_name in ['Material', 'material', 'Sample_Material', 'Sample Material']:
                        if col_name in velocity_shots_df.columns:
                            material_col = col_name
                            break
                
                if material_col is None:
                    self.progress_signal.emit("⚠️  MAD filter: Material column not found, applying to entire dataset")
                    material_col = 'Material'
                    velocity_shots_df[material_col] = 'Unknown'
                
                # Find laser energy column
                laser_energy_col = None
                if 'Laser_Target_Energy (mJ)' in velocity_shots_df.columns:
                    laser_energy_col = 'Laser_Target_Energy (mJ)'
                else:
                    possible_names = [
                        'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                        'Laser_Target_Energy', 'Laser Target Energy',
                        'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                        'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                    ]
                    for col_name in velocity_shots_df.columns:
                        col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                        for possible in possible_names:
                            possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                            if col_name == possible or possible_normalized in col_normalized:
                                laser_energy_col = col_name
                                break
                        if laser_energy_col:
                            break
                
                if laser_energy_col is None:
                    self.progress_signal.emit("⚠️  MAD filter: Laser energy column not found, applying per material only")
                    # Group by material only
                    total_outliers = 0
                    for material in velocity_shots_df[material_col].unique():
                        material_data = velocity_shots_df[velocity_shots_df[material_col] == material]
                        valid_velocities = material_data['max_velocity_ms'].dropna()
                        
                        if len(valid_velocities) >= 2:  # Need at least 2 points for MAD
                            keep_mask = self.mad_filter(valid_velocities.values, threshold=mad_filter_threshold)
                            valid_indices = valid_velocities.index
                            for idx, keep in zip(valid_indices, keep_mask):
                                velocity_shots_df.loc[idx, 'mad_filter_keep'] = keep
                            group_outliers = (~keep_mask).sum()
                            total_outliers += group_outliers
                            if group_outliers > 0:
                                self.progress_signal.emit(f"   {material}: {group_outliers} outlier(s) identified")
                else:
                    # Convert laser energy to numeric
                    velocity_shots_df[laser_energy_col] = pd.to_numeric(velocity_shots_df[laser_energy_col], errors='coerce')
                    
                    # Group by material, then create energy bins (±30 mJ), then apply MAD filter within each bin
                    total_outliers = 0
                    bins_processed = 0
                    energy_bin_width = 30.0  # ±30 mJ bins
                    
                    for material in velocity_shots_df[material_col].unique():
                        material_data = velocity_shots_df[velocity_shots_df[material_col] == material].copy()
                        laser_energies = material_data[laser_energy_col].dropna()
                        
                        if len(laser_energies) == 0:
                            continue
                        
                        # Create energy bins: group laser energies within ±30 mJ of bin mean
                        # Sort energies and create bins using iterative mean-based clustering
                        sorted_energies = sorted(laser_energies.unique())
                        energy_bins = []
                        
                        # Create bins by grouping energies that are within ±30 mJ of the bin's mean
                        # Iterate until all energies are assigned to bins
                        remaining_energies = sorted_energies.copy()
                        
                        while len(remaining_energies) > 0:
                            # Start a new bin with the smallest remaining energy
                            current_bin = [remaining_energies.pop(0)]
                            
                            # Iteratively add energies that are within ±30 mJ of current bin mean
                            # Continue until no more energies can be added
                            changed = True
                            while changed:
                                changed = False
                                bin_mean = np.mean(current_bin)
                                
                                # Check all remaining energies
                                to_remove = []
                                for energy in remaining_energies:
                                    if abs(energy - bin_mean) <= energy_bin_width:
                                        current_bin.append(energy)
                                        to_remove.append(energy)
                                        changed = True
                                
                                # Remove added energies from remaining list
                                for energy in to_remove:
                                    remaining_energies.remove(energy)
                            
                            # Finalize this bin
                            energy_bins.append(current_bin)
                        
                        # Log bin details
                        bin_details = []
                        for bin_idx, bin_energies in enumerate(energy_bins):
                            bin_mean = np.mean(bin_energies)
                            bin_min = min(bin_energies)
                            bin_max = max(bin_energies)
                            bin_details.append(f"Bin {bin_idx+1}: {len(bin_energies)} energy levels, "
                                             f"mean={bin_mean:.1f} mJ, range=[{bin_min:.1f}, {bin_max:.1f}] mJ")
                        
                        self.progress_signal.emit(f"   {material}: Created {len(energy_bins)} energy bin(s) from {len(sorted_energies)} unique energy levels")
                        for detail in bin_details:
                            self.progress_signal.emit(f"      {detail}")
                        
                        # Apply MAD filter within each energy bin
                        for bin_idx, energy_bin in enumerate(energy_bins):
                            # Get all data points within this energy bin (±30 mJ from bin mean)
                            bin_mean = np.mean(energy_bin)
                            bin_data = material_data[
                                (material_data[laser_energy_col] >= bin_mean - energy_bin_width) &
                                (material_data[laser_energy_col] <= bin_mean + energy_bin_width)
                            ]
                            
                            valid_velocities = bin_data['max_velocity_ms'].dropna()
                            
                            if len(valid_velocities) >= 2:  # Need at least 2 points for MAD
                                bins_processed += 1
                                keep_mask = self.mad_filter(valid_velocities.values, threshold=mad_filter_threshold)
                                valid_indices = valid_velocities.index
                                for idx, keep in zip(valid_indices, keep_mask):
                                    velocity_shots_df.loc[idx, 'mad_filter_keep'] = keep
                                bin_outliers = (~keep_mask).sum()
                                total_outliers += bin_outliers
                                if bin_outliers > 0:
                                    self.progress_signal.emit(
                                        f"      Bin {bin_idx+1} ({bin_mean:.1f}±{energy_bin_width} mJ): {bin_outliers} outlier(s), "
                                        f"n={len(valid_velocities)}"
                                    )
                    
                    self.progress_signal.emit(f"MAD filter: Applied to {bins_processed} material+energy_bin groups")
                    self.progress_signal.emit(f"MAD filter: {total_outliers} outlier(s) identified across all bins (threshold={mad_filter_threshold})")
                
                # Mark outliers in alignment status (so they're excluded from plots)
                outliers_count = (~velocity_shots_df['mad_filter_keep']).sum()
                if outliers_count > 0:
                    velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'aligned_ok'] = False
                    velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'alignment_reason'] = \
                        velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'alignment_reason'].apply(
                            lambda x: f"{x}; MAD outlier" if pd.notna(x) and x != "" else "MAD outlier"
                        )
            else:
                velocity_shots_df['mad_filter_keep'] = True  # Default: keep all if filter disabled
            
            if 'aligned_ok' in velocity_shots_df.columns:
                unaligned_count = int((~velocity_shots_df['aligned_ok']).sum())
            else:
                unaligned_count = 0
            self.progress_signal.emit(f"Alignment summary: {unaligned_count}/{total_shots} traces failed to reach the threshold")
            if unaligned_entries:
                unaligned_df = pd.DataFrame(unaligned_entries)
                unaligned_path = os.path.join(spade_output_dir, 'unaligned_traces.csv')
                unaligned_df.to_csv(unaligned_path, index=False)
                self.progress_signal.emit(f"Saved unaligned trace list ({unaligned_count} entries) to: {unaligned_path}")
            else:
                self.progress_signal.emit("All processed traces reached the alignment threshold")

            # Reorder columns: standard columns first, then all parameter columns (sorted)
            standard_cols = ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms',
                             't0_ns', 'velocity_threshold_ms', 'max_velocity_ms', 'aligned_ok', 'alignment_reason']
            param_cols = sorted([c for c in all_param_columns if c not in standard_cols])
            final_cols = standard_cols + param_cols
            # Include any unexpected columns at the end to avoid dropping
            remaining_cols = [c for c in velocity_shots_df.columns if c not in final_cols]
            velocity_shots_df = velocity_shots_df[final_cols + remaining_cols]
            
            unaligned_basenames = {entry['file_name'] for entry in unaligned_entries}
            
            # Add MAD-filtered traces to unaligned_basenames so they're excluded from all_velocity_traces_plot
            if 'mad_filter_keep' in velocity_shots_df.columns:
                mad_filtered_basenames = set(
                    velocity_shots_df[velocity_shots_df['mad_filter_keep'] == False]['file_name'].values
                )
                unaligned_basenames.update(mad_filtered_basenames)
                if len(mad_filtered_basenames) > 0:
                    self.progress_signal.emit(f"Added {len(mad_filtered_basenames)} MAD-filtered traces to exclusion list for all_velocity_traces_plot")
            
            velocity_shots_path = os.path.join(
    spade_output_dir, 'velocity_shots_summary.csv')
            # Write with high precision to avoid rounding issues
            velocity_shots_df.to_csv(velocity_shots_path, index=False, float_format='%.10f')
            self.progress_signal.emit(
                f"Generated velocity shots summary with {len(velocity_shots_data)} shots")
            self.progress_signal.emit(f"Saved to: {velocity_shots_path}")
            self.progress_signal.emit(f"Parameter columns included: {param_cols}")
            
            # Generate HEL consecutive points summary report
            hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                    self.spade_params.get('experiment_hel_detection', False))
            if hel_detection_enabled and 'hel_consecutive_points' in velocity_shots_df.columns:
                self._generate_hel_consecutive_points_report(velocity_shots_df, spade_output_dir)

            # Generate combined velocity plot - DISABLED (material classification now in all_velocity_traces.png)
            # The all_velocity_traces.png plot already includes material-wise classification
            # if velocity_plot_data:
            #     self.generate_combined_velocity_plot(velocity_plot_data, spade_output_dir)

            # Generate the comprehensive aligned plot using ALPSS outputs and GUI threshold
            # This plot includes material-wise classification (color coding by material)
            try:
                if hasattr(self, 'spade_params'):
                    # Use ALPSS output directory as input; same folder SPADE takes input from
                    input_path = self.output_dir
                    uncertainty_threshold = self.spade_params.get('uncertainty_threshold_ms', 50.0)
                    generate_all = self.spade_params.get('generate_all_velocity_plot', True)
                    self.progress_signal.emit(f"[DEBUG] generate_all_velocity_plot flag: {generate_all}")
                    self.progress_signal.emit(f"[DEBUG] input_path exists: {os.path.exists(input_path)}")
                    self.progress_signal.emit(f"[DEBUG] input_path: {input_path}")
                    if generate_all and os.path.exists(input_path):
                        self.progress_signal.emit(
                            f"Generating 'All Velocity Traces' aligned plot with material classification (threshold={uncertainty_threshold} m/s)")
                        self.generate_all_velocity_traces_plot(
                            input_path,
                            spade_output_dir,
                            uncertainty_threshold,
                            unaligned_basenames=unaligned_basenames
                        )
                    elif not generate_all:
                        self.progress_signal.emit("[WARNING] generate_all_velocity_plot is disabled - skipping all velocity traces plot")
                    elif not os.path.exists(input_path):
                        self.progress_signal.emit(f"[WARNING] Input path does not exist: {input_path} - skipping all velocity traces plot")
            except Exception as e:
                self.progress_signal.emit(f"Warning: Failed to create comprehensive aligned velocity plot: {e}")
                import traceback
                self.progress_signal.emit(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            # Create parameter mapping report for debugging
            self.create_parameter_mapping_report(velocity_shots_data, spade_output_dir)
            
            # Generate HEL vs Laser Energy plot if HEL detection was enabled
            hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                    self.spade_params.get('experiment_hel_detection', False))
            if hel_detection_enabled:
                self.generate_hel_vs_laser_energy_plot(spade_output_dir)
                # Generate HEL vs Peak Velocity plot
                self.generate_hel_vs_peak_velocity_plot(spade_output_dir)
                # Generate HEL vs HEL Strain Rate plot
                self.generate_hel_vs_hel_strain_rate_plot(spade_output_dir)
            
            # Generate Shock Stress vs Laser Energy plot
            self.generate_shock_stress_vs_laser_energy_plot(spade_output_dir)
            
            # Generate Shock Stress vs Waveplate Angle plot
            self.generate_shock_stress_vs_waveplate_angle_plot(spade_output_dir)
            
            # Generate Laser Energy vs Waveplate Angle plot
            self.generate_laser_energy_vs_waveplate_angle_plot(spade_output_dir)
            
            # Generate Shock Stress vs Peak Velocity plot
            self.generate_shock_stress_vs_peak_velocity_plot(spade_output_dir)
            
            # Generate positional plots (row/column vs metrics)
            self.generate_row_column_vs_peak_shock_stress_plots(spade_output_dir)
            self.generate_row_column_vs_peak_velocity_heatmap(spade_output_dir)
            self.generate_row_column_pair_vs_peak_velocity_plot(spade_output_dir)
            self.generate_row_column_pair_vs_peak_velocity_by_material_plot(spade_output_dir)
            self.generate_peak_velocity_pattern_analysis_plot(spade_output_dir)
        else:
            self.progress_signal.emit("No velocity shots data generated")

    def run_post_processing(self, post_processing_config):
        """
        Post-processing mode: Generate plots from existing velocity_shots_summary.csv
        without rerunning the full SPADE analysis.
        
        Args:
            post_processing_config: Dictionary with 'enabled', 'spade_output_dir', and 'plots' keys
        
        Returns:
            bool: True if successful, False if failed (e.g., missing files)
        """
        if not post_processing_config.get('enabled', False):
            self.progress_signal.emit("Post-processing is disabled in config")
            return False
        
        spade_output_dir = post_processing_config.get('spade_output_dir', self.output_dir)
        if not os.path.exists(spade_output_dir):
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("❌ POST-PROCESSING FAILED")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"⚠ SPADE output directory not found: {spade_output_dir}")
            self.progress_signal.emit("\nPlease check:")
            self.progress_signal.emit("  1. The 'spade_output_dir' path in post_processing_config")
            self.progress_signal.emit("  2. That the directory exists and contains velocity_shots_summary.csv")
            self.progress_signal.emit("=" * 60)
            return False
        
        velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
        if not os.path.exists(velocity_shots_path):
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("❌ POST-PROCESSING FAILED")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"⚠ velocity_shots_summary.csv not found in:")
            self.progress_signal.emit(f"   {spade_output_dir}")
            self.progress_signal.emit("\nThis file is required for post-processing.")
            self.progress_signal.emit("\nPlease:")
            self.progress_signal.emit("  1. Run SPADE analysis first to generate velocity_shots_summary.csv")
            self.progress_signal.emit("  2. Or check that 'spade_output_dir' points to the correct directory")
            self.progress_signal.emit(f"\nExpected file location: {velocity_shots_path}")
            self.progress_signal.emit("=" * 60)
            return False
        
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit("POST-PROCESSING MODE: Generating plots from existing data")
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit(f"Output directory: {spade_output_dir}")
        
        plots_config = post_processing_config.get('plots', {})
        
        # Check HEL detection status for HEL plots
        hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                self.spade_params.get('experiment_hel_detection', False))
        
        # Generate plots based on config
        plots_generated = 0
        
        if plots_config.get('hel_vs_peak_velocity', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs Peak Velocity plot...")
            self.generate_hel_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('hel_vs_laser_energy', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs Laser Energy plot...")
            self.generate_hel_vs_laser_energy_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('hel_vs_hel_strain_rate', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs HEL Strain Rate plot...")
            self.generate_hel_vs_hel_strain_rate_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_laser_energy', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Laser Energy plot...")
            self.generate_shock_stress_vs_laser_energy_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_waveplate_angle', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Waveplate Angle plot...")
            self.generate_shock_stress_vs_waveplate_angle_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('laser_energy_vs_waveplate_angle', False):
            self.progress_signal.emit("\n📊 Generating Laser Energy vs Waveplate Angle plot...")
            self.generate_laser_energy_vs_waveplate_angle_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_peak_velocity', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Peak Velocity plot...")
            self.generate_shock_stress_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('row_column_vs_peak_shock_stress', False):
            self.progress_signal.emit("\n📊 Generating Row/Column vs Peak Shock Stress plots...")
            self.generate_row_column_vs_peak_shock_stress_plots(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_peak_velocity_heatmap', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Peak Velocity Heatmap...")
            self.generate_row_column_vs_peak_velocity_heatmap(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_pair_peak_velocity', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Pair vs Peak Velocity plot...")
            self.generate_row_column_pair_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_pair_peak_velocity_by_material', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Pair vs Peak Velocity plot (by material, color-coded by laser energy)...")
            self.generate_row_column_pair_vs_peak_velocity_by_material_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('peak_velocity_pattern_analysis', False):
            self.progress_signal.emit("\n📊 Generating Peak Velocity Pattern Analysis plot...")
            self.generate_peak_velocity_pattern_analysis_plot(spade_output_dir)
            plots_generated += 1
        
        self.progress_signal.emit("\n" + "=" * 60)
        self.progress_signal.emit(f"✅ Post-processing complete! Generated {plots_generated} plot(s)")
        self.progress_signal.emit("=" * 60)
        return True

    def create_parameter_mapping_report(self, velocity_shots_data, spade_output_dir):
        """Create a detailed report of parameter mapping for debugging"""
        try:
            report_lines = []
            report_lines.append("=== PARAMETER MAPPING REPORT ===")
            report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            if self.param_data:
                report_lines.append(f"Total parameter entries available: {len(self.param_data)}")
                report_lines.append("Available parameter keys:")
                for key in sorted(self.param_data.keys()):
                    report_lines.append(f"  - {key}")
                report_lines.append("")
            else:
                report_lines.append("No parameter data available")
                report_lines.append("")
            
            report_lines.append("=== VELOCITY FILE MAPPING ===")
            for shot_data in velocity_shots_data:
                file_name = shot_data['file_name']
                param_keys = [k for k in shot_data.keys() if k not in ['file_name', 'mean_velocity_300_400ns_ms', 
                                                                      'time_window_used', 'uncertainty_avg_ms', 
                                                                      't0_ns', 'velocity_threshold_ms',
                                                                      'max_velocity_ms', 'aligned_ok', 'alignment_reason']]
                report_lines.append(f"File: {file_name}")
                report_lines.append(f"  Parameter columns: {param_keys}")
                if param_keys:
                    non_nan_params = [k for k in param_keys if not pd.isna(shot_data.get(k, np.nan))]
                    report_lines.append(f"  Non-NaN parameters: {non_nan_params}")
                else:
                    report_lines.append("  No parameters mapped")
                report_lines.append("")
            
            # Save report
            report_path = os.path.join(spade_output_dir, 'parameter_mapping_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.progress_signal.emit(f"Created parameter mapping report: {report_path}")
            
        except Exception as e:
            self.progress_signal.emit(f"Error creating parameter mapping report: {str(e)}")

    def _generate_hel_consecutive_points_report(self, velocity_shots_df, spade_output_dir):
        """
        Generate a summary report listing all traces with their HEL consecutive points count
        and corresponding time duration in nanoseconds.
        """
        try:
            import pandas as pd
            
            # Create report DataFrame with relevant columns
            report_data = []
            for idx, row in velocity_shots_df.iterrows():
                report_data.append({
                    'file_name': row.get('file_name', 'Unknown'),
                    'hel_consecutive_points': row.get('hel_consecutive_points', 0),
                    'hel_segment_time_ns': row.get('hel_segment_time_ns', np.nan),
                    'hel_ok': row.get('hel_ok', False),
                    'hel_strength_gpa': row.get('hel_strength_gpa', np.nan),
                    'aligned_ok': row.get('aligned_ok', False)
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Sort by consecutive points (descending), then by file name
            report_df = report_df.sort_values(
                by=['hel_consecutive_points', 'file_name'], 
                ascending=[False, True]
            )
            
            # Save report
            report_path = os.path.join(spade_output_dir, 'hel_consecutive_points_report.csv')
            report_df.to_csv(report_path, index=False)
            
            # Print summary statistics
            total_traces = len(report_df)
            traces_with_hel = report_df['hel_ok'].sum()
            traces_with_segment = (report_df['hel_consecutive_points'] > 0).sum()
            
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("HEL Consecutive Points Summary Report")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"Total traces processed: {total_traces}")
            min_points = self.spade_params.get('hel_detection_min_points', 3)
            self.progress_signal.emit(f"Traces with HEL segment (≥{min_points} points): {traces_with_segment}")
            self.progress_signal.emit(f"Traces with valid HEL detection: {traces_with_hel}")
            
            if traces_with_segment > 0:
                valid_segments = report_df[report_df['hel_consecutive_points'] > 0]
                self.progress_signal.emit(f"\nConsecutive Points Statistics:")
                self.progress_signal.emit(f"  Min: {valid_segments['hel_consecutive_points'].min()}")
                self.progress_signal.emit(f"  Max: {valid_segments['hel_consecutive_points'].max()}")
                self.progress_signal.emit(f"  Mean: {valid_segments['hel_consecutive_points'].mean():.1f}")
                self.progress_signal.emit(f"  Median: {valid_segments['hel_consecutive_points'].median():.1f}")
                
                self.progress_signal.emit(f"\nSegment Time Duration Statistics (ns):")
                valid_times = valid_segments['hel_segment_time_ns'].dropna()
                if len(valid_times) > 0:
                    self.progress_signal.emit(f"  Min: {valid_times.min():.3f} ns")
                    self.progress_signal.emit(f"  Max: {valid_times.max():.3f} ns")
                    self.progress_signal.emit(f"  Mean: {valid_times.mean():.3f} ns")
                    self.progress_signal.emit(f"  Median: {valid_times.median():.3f} ns")
            
            self.progress_signal.emit(f"\nDetailed report saved to: {report_path}")
            self.progress_signal.emit("=" * 60)
            
        except Exception as e:
            self.progress_signal.emit(f"Warning: Could not generate HEL consecutive points report: {str(e)}")

    def mad_filter(self, values, threshold=3.0):
        """
        Asymmetric statistical outlier filter using the Median Absolute Deviation (MAD).
        Uses MAD_lower (based on values below median) for all values.
        Removes points whose scaled deviation exceeds `threshold`.
        
        Parameters:
        -----------
        values : array-like
            Input values to filter
        threshold : float, default=3.0
            Threshold for modified z-score (typical: 3.0)
        
        Returns:
        --------
        numpy.ndarray
            Boolean mask: True for values to keep, False for outliers
        """
        v = np.array(values)
        median = np.median(v)
        
        # Calculate MAD_lower using only values below median
        lower_mask = v < median
        if np.sum(lower_mask) == 0:
            # If no values below median, use symmetric MAD as fallback
            abs_dev = np.abs(v - median)
            mad = np.median(abs_dev)
            if mad == 0:
                return np.ones_like(v, dtype=bool)
            modified_z = 0.6745 * abs_dev / mad
            return modified_z < threshold
        
        mad_lower = np.median(np.abs(v[lower_mask] - median))
        
        # Avoid division by zero
        if mad_lower == 0:
            return np.ones_like(v, dtype=bool)
        
        # Calculate M_i_lower for all values using MAD_lower
        M_i_lower = 0.6745 * np.abs(v - median) / mad_lower
        
        # Return mask: True for values to keep (M_i_lower < threshold)
        return M_i_lower < threshold
    
    def _get_material_color_mapping(self, materials):
        """
        Generate consistent color mapping for materials across all plots.
        Uses predefined colors for common materials, then colormap for others.
        Ensures same material always gets same color.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Predefined colors for common materials (consistent across all plots)
        predefined_colors = {
            'Cu': '#1f77b4',      # Blue
            'Copper': '#1f77b4',  # Blue
            'Zn': '#ff7f0e',      # Orange
            'Zinc': '#ff7f0e',    # Orange
            'Brass': '#2ca02c',   # Green
            'Al': '#d62728',      # Red
            'Aluminum': '#d62728', # Red
            'Ti': '#9467bd',      # Purple
            'Titanium': '#9467bd', # Purple
            'Steel': '#8c564b',   # Brown
            'Fe': '#e377c2',      # Pink
            'Iron': '#e377c2',    # Pink
        }
        
        # Use Set3 for <=12 materials, tab20 for more
        if len(materials) <= 12:
            cmap = plt.get_cmap('Set3')
        else:
            cmap = plt.get_cmap('tab20')
        
        color_mapping = {}
        predefined_used = set()
        
        # First, assign predefined colors
        for material in materials:
            material_key = str(material).strip()
            if material_key in predefined_colors:
                color_mapping[material] = predefined_colors[material_key]
                predefined_used.add(material_key)
        
        # Then assign colors from colormap for remaining materials
        # Use a deterministic order based on material name hash for consistency
        remaining_materials = sorted([m for m in materials if m not in color_mapping], 
                                     key=lambda x: hash(str(x)))
        
        for i, material in enumerate(remaining_materials):
            # Use hash to get consistent index, but map to colormap range
            material_hash = hash(str(material))
            # Map hash to 0-1 range for colormap
            color_idx = abs(material_hash) % 1000 / 1000.0
            color_mapping[material] = cmap(color_idx)
        
        return color_mapping

    def _find_parameter_column(self, df, candidate_names):
        """
        Find a parameter column in a dataframe by matching against candidate names.
        Comparison is case-insensitive and ignores spaces/underscores/hyphens.
        """
        if df is None or df.empty:
            return None
        
        normalized_columns = {
            col: ''.join(ch for ch in col.lower() if ch.isalnum())
            for col in df.columns
        }
        
        for candidate in candidate_names:
            normalized_candidate = ''.join(ch for ch in candidate.lower() if ch.isalnum())
            for col, normalized in normalized_columns.items():
                if normalized == normalized_candidate:
                    return col
                # Also allow candidate to be substring of column name
                if normalized_candidate in normalized:
                    return col
        return None

    def _convert_row_column_to_numeric(self, series):
        """
        Convert row/column labels (letters, numbers, or mixed) to numeric values for plotting.
        Returns tuple (numeric_series, tick_mapping) where tick_mapping maps numeric value -> label.
        """
        import numpy as np
        import pandas as pd
        
        if series is None or len(series) == 0:
            return pd.Series(dtype=float), {}
        
        series = pd.Series(series)
        
        # 1) Try direct numeric conversion
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() > 0:
            numeric_series = numeric.astype(float)
            mapping = {}
            for num, label in zip(numeric_series, series):
                if pd.notna(num) and pd.notna(label):
                    mapping[num] = str(label)
            return numeric_series, mapping
        
        # 2) Try alphabetic conversion (A=1, B=2, AA=27, etc.)
        def alpha_to_num(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return np.nan
            value_str = str(value).strip().upper()
            if not value_str:
                return np.nan
            # Keep only letters
            letters = ''.join(ch for ch in value_str if ch.isalpha())
            if not letters:
                return np.nan
            total = 0
            for ch in letters:
                if 'A' <= ch <= 'Z':
                    total = total * 26 + (ord(ch) - 64)
                else:
                    return np.nan
            return float(total)
        
        alpha_numeric = series.apply(alpha_to_num)
        if alpha_numeric.notna().sum() > 0:
            mapping = {}
            for num, label in zip(alpha_numeric, series):
                if pd.notna(num) and pd.notna(label):
                    mapping[num] = str(label)
            return alpha_numeric.astype(float), mapping
        
        # 3) Fallback: assign category codes in sorted order
        unique_values = [val for val in series.dropna().unique() if str(val).strip()]
        unique_values_sorted = sorted(unique_values, key=lambda x: str(x))
        value_to_code = {val: idx + 1 for idx, val in enumerate(unique_values_sorted)}
        numeric_series = series.map(value_to_code).astype(float)
        mapping = {code: str(val) for val, code in value_to_code.items()}
        return numeric_series, mapping

    def generate_hel_vs_laser_energy_plot(self, spade_output_dir):
        """Generate HEL vs Laser Energy plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs Laser Energy plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Expected path: {velocity_shots_path}")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:15])}...")
            
            # Check if HEL and laser energy columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist())}")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data")
            
            # Look for laser energy column - prioritize 'Laser_Target_Energy (mJ)' from parameter file
            laser_energy_col = None
            energy_in_mj = False
            
            # First, check for exact match with 'Laser_Target_Energy (mJ)'
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
                self.progress_signal.emit("✓ Using 'Laser_Target_Energy (mJ)' from parameter file")
            else:
                # Fallback to other possible names
                possible_names = [
                    'Laser_Target_Energy (mJ)',  # Try exact match first (case-sensitive)
                    'Laser Target Energy (mJ)',  # With space instead of underscore
                    'Laser_Target_Energy',  # Without unit
                    'Laser Target Energy',  # With space, without unit
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy', 
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        # Check for exact match first, then substring match
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            # Check if it's in mJ
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found in parameter file - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:10])}...")
                return
            
            # Filter data: only rows with valid HEL and laser energy
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df[laser_energy_col].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and laser energy data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Laser Energy data points - skipping plot")
                hel_count = df['hel_strength_gpa'].notna().sum()
                energy_count = df[laser_energy_col].notna().sum()
                self.progress_signal.emit(f"   HEL data points: {hel_count}, Laser energy data points: {energy_count}")
                return
            
            # Ensure laser energy is numeric (keep original units - mJ or J)
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            
            # Get material column (should be added by parameter file)
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                # No material column, create a default one
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    laser_energy_col, 
                    'hel_strength_gpa',
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors
            materials = valid_data[material_col].unique()
            cmap = plt.get_cmap('tab10' if len(materials) <= 10 else 'tab20')
            colors = {mat: cmap(i / max(len(materials), 1)) for i, mat in enumerate(materials)}
            
            # Plot data grouped by material
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Plot with error bars if HEL uncertainty is available
                if 'hel_uncertainty_gpa' in material_data.columns:
                    ax.errorbar(
                        material_data[laser_energy_col],
                        material_data['hel_strength_gpa'],
                        yerr=material_data['hel_uncertainty_gpa'],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=material
                    )
                else:
                    ax.scatter(
                        material_data[laser_energy_col],
                        material_data['hel_strength_gpa'],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=material
                    )
            
            # Set labels and title (use correct unit based on column)
            energy_unit = 'mJ' if energy_in_mj else 'J'
            ax.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Hugoniot Elastic Limit vs Laser Energy by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_laser_energy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs Laser Energy plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs Laser Energy plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_hel_vs_peak_velocity_plot(self, spade_output_dir):
        """Generate HEL vs Peak Velocity scatter plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs Peak Velocity plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check if HEL and peak velocity columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs Peak Velocity plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity (max_velocity_ms) not found - skipping HEL vs Peak Velocity plot")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data")
            
            # Filter data: only rows with valid HEL and peak velocity
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df['max_velocity_ms'].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and peak velocity data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Peak Velocity data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data['max_velocity_ms'] = pd.to_numeric(valid_data['max_velocity_ms'], errors='coerce')
            valid_data['hel_strength_gpa'] = pd.to_numeric(valid_data['hel_strength_gpa'], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['max_velocity_ms'].notna()) & (valid_data['hel_strength_gpa'].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    'max_velocity_ms', 
                    'hel_strength_gpa',
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Debug: Log HEL values being used (with file names for verification)
            if len(valid_data) > 0:
                hel_values = valid_data['hel_strength_gpa'].values
                file_names = valid_data['file_name'].values if 'file_name' in valid_data.columns else None
                self.progress_signal.emit(f"   [DEBUG] HEL vs Peak Velocity plot: Using {len(hel_values)} HEL values")
                self.progress_signal.emit(f"   [DEBUG] HEL range: {np.min(hel_values):.6f} to {np.max(hel_values):.6f} GPa")
                if len(hel_values) <= 5 and file_names is not None:
                    for fname, hel_val in zip(file_names, hel_values):
                        self.progress_signal.emit(f"   [DEBUG]   {fname}: HEL = {hel_val:.10f} GPa")
                elif len(hel_values) <= 5:
                    self.progress_signal.emit(f"   [DEBUG] All HEL values: {hel_values}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if HEL uncertainty is available
                if 'hel_uncertainty_gpa' in material_data.columns:
                    errorbar_handle = ax.errorbar(
                        material_data['max_velocity_ms'],
                        material_data['hel_strength_gpa'],
                        yerr=material_data['hel_uncertainty_gpa'],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data['max_velocity_ms'],
                        material_data['hel_strength_gpa'],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            ax.set_xlabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Hugoniot Elastic Limit vs Peak Velocity by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_peak_velocity_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs Peak Velocity plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
    
    def generate_spall_vs_strain_rate_plot(self, summary_df, spade_output_dir):
        """Generate Spall Strength vs Strain Rate plot matching HEL plot format"""
        self.progress_signal.emit("Generating Spall Strength vs Strain Rate plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Filter data: only rows with valid Spall Strength and Strain Rate
            valid_data = summary_df[
                (summary_df['Spall Strength (GPa)'].notna()) & 
                (summary_df['Strain Rate (s^-1)'].notna())
            ].copy()
            
            # Remove rows with non-positive values
            valid_data = valid_data[
                (pd.to_numeric(valid_data['Spall Strength (GPa)'], errors='coerce') > 0) &
                (pd.to_numeric(valid_data['Strain Rate (s^-1)'], errors='coerce') > 0)
            ].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Spall Strength vs Strain Rate data - skipping plot")
                return
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    'Strain Rate (s^-1)', 
                    'Spall Strength (GPa)',
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if col_name.lower() == 'material':
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data['Spall Strength (GPa)'] = pd.to_numeric(valid_data['Spall Strength (GPa)'], errors='coerce')
            valid_data['Strain Rate (s^-1)'] = pd.to_numeric(valid_data['Strain Rate (s^-1)'], errors='coerce')
            
            # Convert uncertainty to numeric (replace strings like "DNS" with NaN)
            if 'Spall Strength Uncertainty (GPa)' in valid_data.columns:
                valid_data['Spall Strength Uncertainty (GPa)'] = pd.to_numeric(
                    valid_data['Spall Strength Uncertainty (GPa)'], errors='coerce'
                )
            
            # Convert strain rate uncertainty to numeric
            if 'Strain Rate Uncertainty (s^-1)' in valid_data.columns:
                valid_data['Strain Rate Uncertainty (s^-1)'] = pd.to_numeric(
                    valid_data['Strain Rate Uncertainty (s^-1)'], errors='coerce'
                )
                # Debug: Check strain rate uncertainty values
                strain_unc_valid = valid_data['Strain Rate Uncertainty (s^-1)'].notna() & (valid_data['Strain Rate Uncertainty (s^-1)'] > 0)
                self.progress_signal.emit(f"DEBUG: Strain Rate Uncertainty - Valid values: {strain_unc_valid.sum()} out of {len(valid_data)}")
                if strain_unc_valid.sum() > 0:
                    sample_vals = valid_data.loc[strain_unc_valid, 'Strain Rate Uncertainty (s^-1)'].head(5)
                    self.progress_signal.emit(f"DEBUG:   Sample values: {sample_vals.tolist()}")
            else:
                # Try alternative column names
                alt_cols = ['Strain_Rate_Uncertainty_s^-1', 'Strain_Rate_Unc_s^-1', 'StrainRate_Unc_s^-1', 
                           'Strain_Rate_Uncertainty_s1_Final', 'Strain_Rate_Uncertainty_s1']
                for alt_col in alt_cols:
                    if alt_col in valid_data.columns:
                        valid_data['Strain Rate Uncertainty (s^-1)'] = pd.to_numeric(
                            valid_data[alt_col], errors='coerce'
                        )
                        self.progress_signal.emit(f"DEBUG: Found strain rate uncertainty in column '{alt_col}', mapped to 'Strain Rate Uncertainty (s^-1)'")
                        break
                else:
                    self.progress_signal.emit("DEBUG: WARNING - 'Strain Rate Uncertainty (s^-1)' column not found. Available columns:")
                    unc_cols = [c for c in valid_data.columns if 'uncertainty' in c.lower() or 'unc' in c.lower() or 'error' in c.lower()]
                    self.progress_signal.emit(f"DEBUG:   Uncertainty-related columns: {unc_cols}")
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[
                (valid_data['Spall Strength (GPa)'].notna()) & 
                (valid_data['Strain Rate (s^-1)'].notna())
            ].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                if len(material_data) == 0:
                    continue
                
                marker = markers[i % len(markers)]
                color = colors[material]
                n_points = len(material_data)
                
                # Get uncertainty values for both x and y axes
                # Initialize with NaN arrays - matplotlib will skip error bars for NaN values
                yerr = np.full(len(material_data), np.nan)
                xerr = np.full(len(material_data), np.nan)
                
                # Y-error bars: Spall Strength Uncertainty
                if 'Spall Strength Uncertainty (GPa)' in material_data.columns:
                    yerr_series = pd.to_numeric(material_data['Spall Strength Uncertainty (GPa)'], errors='coerce')
                    # Replace NaN values in yerr array with actual uncertainty values where available
                    valid_mask = yerr_series.notna() & (yerr_series > 0)
                    yerr[valid_mask] = yerr_series[valid_mask].values
                
                # X-error bars: Strain Rate Uncertainty
                if 'Strain Rate Uncertainty (s^-1)' in material_data.columns:
                    xerr_series = pd.to_numeric(material_data['Strain Rate Uncertainty (s^-1)'], errors='coerce')
                    # Replace NaN values in xerr array with actual uncertainty values where available
                    valid_mask = xerr_series.notna() & (xerr_series > 0)
                    xerr[valid_mask] = xerr_series[valid_mask].values
                else:
                    # Try alternative column names
                    alt_cols = ['Strain_Rate_Uncertainty_s^-1', 'Strain_Rate_Unc_s^-1', 'StrainRate_Unc_s^-1',
                               'Strain_Rate_Uncertainty_s1_Final', 'Strain_Rate_Uncertainty_s1']
                    for alt_col in alt_cols:
                        if alt_col in material_data.columns:
                            xerr_series = pd.to_numeric(material_data[alt_col], errors='coerce')
                            valid_mask = xerr_series.notna() & (xerr_series > 0)
                            xerr[valid_mask] = xerr_series[valid_mask].values
                            break
                
                # Always use errorbar for consistency - matplotlib handles NaN values gracefully
                # If no uncertainty data, error bars will just be zero/not visible
                errorbar_handle = ax.errorbar(
                    material_data['Strain Rate (s^-1)'],
                    material_data['Spall Strength (GPa)'],
                    xerr=xerr,  # X-error bars for strain rate uncertainty
                    yerr=yerr,  # Y-error bars for spall strength uncertainty
                    fmt=marker,
                    color=color,
                    markersize=10,
                    linewidth=0,
                    elinewidth=2.0,  # Thicker error bars for better visibility
                    capsize=5,  # Larger caps for better visibility
                    capthick=2.0,  # Thicker caps
                    alpha=0.7,
                    label=f"{material} (n={n_points})"
                )
                legend_handles.append(errorbar_handle[0])
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            ax.set_xlabel('Strain Rate (s^-1)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Spall Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Spall Strength vs Strain Rate by Material', fontsize=16, fontweight='bold')
            ax.set_xscale('log')  # Use log scale for strain rate
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'spall_vs_strain_rate.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Spall Strength vs Strain Rate plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Spall Strength vs Strain Rate plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
    
    def generate_spall_vs_shock_stress_plot(self, summary_df, spade_output_dir):
        """Generate Spall Strength vs Shock Stress plot matching HEL plot format"""
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit("DEBUG: Starting Spall Strength vs Shock Stress plot generation")
        self.progress_signal.emit("=" * 60)
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Debug: Show data source and shape
            self.progress_signal.emit(f"DEBUG: Data source: summary_df passed to function")
            self.progress_signal.emit(f"DEBUG: summary_df shape: {summary_df.shape}")
            self.progress_signal.emit(f"DEBUG: summary_df type: {type(summary_df)}")
            
            # Debug: Check available columns
            available_cols = list(summary_df.columns)
            self.progress_signal.emit(f"DEBUG: Total columns in summary_df: {len(available_cols)}")
            self.progress_signal.emit(f"DEBUG: All columns: {available_cols}")
            
            # Try multiple possible column names for Spall Strength
            spall_strength_col = None
            possible_spall_cols = ['Spall_Strength_GPa', 'Spall Strength (GPa)', 'Spall_Strength_GPa_Final', 
                                  'ALPSS_Spall_Strength_GPa']
            self.progress_signal.emit(f"DEBUG: Checking for Spall Strength column in: {possible_spall_cols}")
            
            for col_name in possible_spall_cols:
                if col_name in summary_df.columns:
                    spall_strength_col = col_name
                    self.progress_signal.emit(f"DEBUG: ✓ Found '{spall_strength_col}' for Spall Strength")
                    # Show sample values
                    spall_values = summary_df[spall_strength_col].dropna()
                self.progress_signal.emit(f"   - Non-null values: {len(spall_values)}")
                if len(spall_values) > 0:
                    self.progress_signal.emit(f"   - Sample values: {spall_values.head(5).tolist()}")
                    self.progress_signal.emit(f"   - Min: {spall_values.min()}, Max: {spall_values.max()}")
                    break
            
            if spall_strength_col is None:
                self.progress_signal.emit("DEBUG: 'Spall Strength' column NOT FOUND")
                self.progress_signal.emit("⚠ Spall Strength column not found - checking available columns")
                spall_candidates = [c for c in available_cols if 'spall' in c.lower() and 'strength' in c.lower()]
                self.progress_signal.emit(f"   Available spall strength columns: {spall_candidates}")
                if len(spall_candidates) == 0:
                    self.progress_signal.emit("   ERROR: No spall strength columns found at all!")
                return
            
            # Check for shock stress columns
            shock_cols = [c for c in available_cols if 'shock' in c.lower() or 'stress' in c.lower()]
            self.progress_signal.emit(f"DEBUG: Columns containing 'shock' or 'stress': {shock_cols}")
            for col in shock_cols:
                values = summary_df[col].dropna()
                self.progress_signal.emit(f"DEBUG: '{col}':")
                self.progress_signal.emit(f"   - Non-null values: {len(values)}")
                if len(values) > 0:
                    self.progress_signal.emit(f"   - Sample values: {values.head(5).tolist()}")
                    self.progress_signal.emit(f"   - Min: {values.min()}, Max: {values.max()}")
            
            # Try multiple possible column names for Peak Shock Stress
            shock_stress_col = None
            possible_cols = ['Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa', 'Peak_Shock_Stress_GPa_Final', 
                            'Shock Stress (GPa)', 'Shock_Stress_GPa', 'Peak Shock Stress', 'Plateau Mean Velocity (m/s)']
            self.progress_signal.emit(f"DEBUG: Checking for Peak Shock Stress column in: {possible_cols}")
            
            for col_name in possible_cols:
                if col_name in summary_df.columns:
                    shock_stress_col = col_name
                    self.progress_signal.emit(f"DEBUG: ✓ Found '{shock_stress_col}' for Peak Shock Stress")
                    # Show sample values
                    sample_vals = summary_df[shock_stress_col].dropna().head(5)
                    self.progress_signal.emit(f"DEBUG:   Sample values: {sample_vals.tolist()}")
                    break
            
            if shock_stress_col is None:
                self.progress_signal.emit("⚠ Peak Shock Stress column not found - checking available columns")
                shock_stress_candidates = [c for c in available_cols if 'shock' in c.lower() or 'stress' in c.lower() or 'plateau' in c.lower()]
                self.progress_signal.emit(f"   Available shock/stress/plateau columns: {shock_stress_candidates}")
                if len(shock_stress_candidates) == 0:
                    self.progress_signal.emit("   ERROR: No shock/stress columns found at all!")
                return
            
            # Filter data: only rows with valid Spall Strength and Shock Stress
            self.progress_signal.emit(f"DEBUG: Filtering data...")
            self.progress_signal.emit(f"   - Rows with non-null '{spall_strength_col}': {summary_df[spall_strength_col].notna().sum()}")
            self.progress_signal.emit(f"   - Rows with non-null '{shock_stress_col}': {summary_df[shock_stress_col].notna().sum()}")
            
            valid_data = summary_df[
                (summary_df[spall_strength_col].notna()) & 
                (summary_df[shock_stress_col].notna())
            ].copy()
            
            self.progress_signal.emit(f"DEBUG: Found {len(valid_data)} rows with both Spall Strength and {shock_stress_col}")
            
            if len(valid_data) > 0:
                # Show sample of valid data
                self.progress_signal.emit(f"DEBUG: Sample of valid data (first 3 rows):")
                for idx, row in valid_data.head(3).iterrows():
                    # Check if values are numeric before formatting
                    spall_val = row[spall_strength_col]
                    shock_val = row[shock_stress_col]
                    # Try to convert to numeric, if fails use string representation
                    try:
                        spall_num = pd.to_numeric(spall_val, errors='raise')
                        spall_str = f"{spall_num:.3f}"
                    except (ValueError, TypeError):
                        spall_str = str(spall_val)
                    try:
                        shock_num = pd.to_numeric(shock_val, errors='raise')
                        shock_str = f"{shock_num:.3f}"
                    except (ValueError, TypeError):
                        shock_str = str(shock_val)
                    self.progress_signal.emit(f"   Row {idx}: Spall={spall_str}, Shock={shock_str}")
            
            # Remove rows with non-positive values
            before_filter = len(valid_data)
            valid_data = valid_data[
                (pd.to_numeric(valid_data[spall_strength_col], errors='coerce') > 0) &
                (pd.to_numeric(valid_data[shock_stress_col], errors='coerce') > 0)
            ].copy()
            
            self.progress_signal.emit(f"DEBUG: After filtering non-positive values: {len(valid_data)} rows (removed {before_filter - len(valid_data)})")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Spall Strength vs Shock Stress data - skipping plot")
                return
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    shock_stress_col, 
                    spall_strength_col,
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if col_name.lower() == 'material':
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[spall_strength_col] = pd.to_numeric(valid_data[spall_strength_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Convert uncertainty to numeric (replace strings like "DNS" with NaN)
            # Check for multiple possible uncertainty column names
            spall_unc_col = None
            for unc_col in ['Spall_Strength_Unc_GPa', 'Spall Strength Uncertainty (GPa)', 'Spall_Strength_Uncertainty_GPa']:
                if unc_col in valid_data.columns:
                    spall_unc_col = unc_col
                    break
            
            if spall_unc_col:
                valid_data[spall_unc_col] = pd.to_numeric(valid_data[spall_unc_col], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[
                (valid_data[spall_strength_col].notna()) & 
                (valid_data[shock_stress_col].notna())
            ].copy()
            
            self.progress_signal.emit(f"DEBUG: After numeric conversion: {len(valid_data)} rows")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors
            materials = valid_data[material_col].unique()
            self.progress_signal.emit(f"DEBUG: Materials found: {materials}")
            self.progress_signal.emit(f"DEBUG: Number of materials: {len(materials)}")
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            self.progress_signal.emit(f"DEBUG: Starting to plot {len(materials)} materials...")
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                if len(material_data) == 0:
                    self.progress_signal.emit(f"DEBUG: Skipping {material} - no data")
                    continue
                
                marker = markers[i % len(markers)]
                color = colors[material]
                n_points = len(material_data)
                
                self.progress_signal.emit(f"DEBUG: Plotting {material}: {n_points} points")
                self.progress_signal.emit(f"   - X values ({shock_stress_col}): min={material_data[shock_stress_col].min():.3f}, max={material_data[shock_stress_col].max():.3f}")
                self.progress_signal.emit(f"   - Y values (Spall Strength): min={material_data[spall_strength_col].min():.3f}, max={material_data[spall_strength_col].max():.3f}")
                
                # Get uncertainty values for both x and y axes
                yerr = None
                xerr = None
                
                # Y-error bars: Spall Strength Uncertainty
                if spall_unc_col and spall_unc_col in material_data.columns:
                    yerr = pd.to_numeric(material_data[spall_unc_col], errors='coerce')
                    # Keep NaN values - matplotlib will handle them (no error bar for NaN points)
                    if (yerr > 0).any():  # At least one valid uncertainty value
                        yerr = yerr.values  # Convert to numpy array
                    else:
                        yerr = None
                
                # X-error bars: Shock Stress Uncertainty
                if 'Peak Shock Stress Uncertainty (GPa)' in material_data.columns:
                    xerr = pd.to_numeric(material_data['Peak Shock Stress Uncertainty (GPa)'], errors='coerce')
                    # Keep NaN values - matplotlib will handle them (no error bar for NaN points)
                    if (xerr > 0).any():  # At least one valid uncertainty value
                        xerr = xerr.values  # Convert to numpy array
                    else:
                        xerr = None
                
                # Always use errorbar for consistency - matplotlib handles NaN values gracefully
                # If no uncertainty data, error bars will just be zero/not visible
                errorbar_handle = ax.errorbar(
                    material_data[shock_stress_col],
                    material_data[spall_strength_col],
                    xerr=xerr,  # X-error bars for shock stress uncertainty
                    yerr=yerr,  # Y-error bars for spall strength uncertainty
                    fmt=marker,
                    color=color,
                    markersize=10,
                    linewidth=0,
                    elinewidth=2.0,  # Thicker error bars for better visibility
                    capsize=5,  # Larger caps for better visibility
                    capthick=2.0,  # Thicker caps
                    alpha=0.7,
                    label=f"{material} (n={n_points})"
                )
                legend_handles.append(errorbar_handle[0])
                self.progress_signal.emit(f"DEBUG:   Plotted {material} with error bars (x={xerr is not None}, y={yerr is not None})")
                
                legend_labels.append(f"{material} (n={n_points})")
            
            self.progress_signal.emit(f"DEBUG: Finished plotting. Total legend handles: {len(legend_handles)}, labels: {len(legend_labels)}")
            
            # Set labels and title
            ax.set_xlabel('Peak Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Spall Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Spall Strength vs Shock Stress by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'spall_vs_shock_stress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"✅ Generated Spall Strength vs Shock Stress plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            self.progress_signal.emit("=" * 60)
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Spall Strength vs Shock Stress plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_hel_vs_hel_strain_rate_plot(self, spade_output_dir):
        """Generate HEL vs HEL Strain Rate scatter plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs HEL Strain Rate plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs HEL Strain Rate plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check if HEL and strain rate columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs HEL Strain Rate plot")
                return
            
            if 'hel_strain_rate_s^-1' not in df.columns:
                self.progress_signal.emit("⚠ HEL strain rate not found in velocity shots - skipping HEL vs HEL Strain Rate plot")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            strain_rate_count = df['hel_strain_rate_s^-1'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data, {strain_rate_count} rows with strain rate data")
            
            # Filter data: only rows with valid HEL and strain rate
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df['hel_strain_rate_s^-1'].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and strain rate data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Strain Rate data points - skipping plot")
                return
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    'hel_strain_rate_s^-1', 
                    'hel_strength_gpa',
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data['hel_strength_gpa'] = pd.to_numeric(valid_data['hel_strength_gpa'], errors='coerce')
            valid_data['hel_strain_rate_s^-1'] = pd.to_numeric(valid_data['hel_strain_rate_s^-1'], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['hel_strength_gpa'].notna()) & (valid_data['hel_strain_rate_s^-1'].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Debug: Log HEL values being used (with file names for verification)
            if len(valid_data) > 0:
                hel_values = valid_data['hel_strength_gpa'].values
                file_names = valid_data['file_name'].values if 'file_name' in valid_data.columns else None
                self.progress_signal.emit(f"   [DEBUG] HEL vs HEL Strain Rate plot: Using {len(hel_values)} HEL values")
                self.progress_signal.emit(f"   [DEBUG] HEL range: {np.min(hel_values):.6f} to {np.max(hel_values):.6f} GPa")
                if len(hel_values) <= 5 and file_names is not None:
                    for fname, hel_val in zip(file_names, hel_values):
                        self.progress_signal.emit(f"   [DEBUG]   {fname}: HEL = {hel_val:.10f} GPa")
                elif len(hel_values) <= 5:
                    self.progress_signal.emit(f"   [DEBUG] All HEL values: {hel_values}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter
                scatter_handle = ax.scatter(
                    material_data['hel_strain_rate_s^-1'],
                    material_data['hel_strength_gpa'],
                    marker=marker,
                    c=[color],
                    s=100,
                    alpha=0.7,
                    label=f"{material} (n={n_points})",
                    edgecolors='black',
                    linewidths=0.5
                )
                legend_handles.append(scatter_handle)
                legend_labels.append(f"{material} (n={n_points})")
            
            ax.set_xlabel('HEL Strain Rate (s⁻¹)', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('HEL vs HEL Strain Rate', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Format x-axis in scientific notation
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best', fontsize=10, title='Material')
            
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_hel_strain_rate.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs HEL Strain Rate plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs HEL Strain Rate plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_laser_energy_plot(self, spade_output_dir):
        """Generate Shock Stress vs Laser Energy plot grouped by material (matching HEL plot format)"""
        self.progress_signal.emit("Generating Shock Stress vs Laser Energy plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Load velocity shots summary which contains shock stress data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Laser Energy plot")
                self.progress_signal.emit(f"   Expected path: {velocity_shots_path}")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:15])}...")
            
            # Check if shock stress column exists - use ONLY EOS-calculated value
            if 'Peak Shock Stress (GPa)' not in df.columns:
                self.progress_signal.emit("⚠ Peak Shock Stress (GPa) not found in velocity shots - skipping Shock Stress vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist())}")
                return
            
            # Check how many rows have shock stress data
            shock_stress_count = df['Peak Shock Stress (GPa)'].notna().sum()
            self.progress_signal.emit(f"   Found {shock_stress_count} rows with Peak Shock Stress data")
            
            # Look for laser energy column - prioritize 'Laser_Target_Energy (mJ)' from parameter file
            laser_energy_col = None
            energy_in_mj = False
            
            # First, check for exact match with 'Laser_Target_Energy (mJ)'
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
                self.progress_signal.emit("✓ Using 'Laser_Target_Energy (mJ)' from parameter file")
            else:
                # Fallback to other possible names
                possible_names = [
                    'Laser_Target_Energy (mJ)',  # Try exact match first (case-sensitive)
                    'Laser Target Energy (mJ)',  # With space instead of underscore
                    'Laser_Target_Energy',  # Without unit
                    'Laser Target Energy',  # With space, without unit
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        # Check for exact match first, then substring match
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            # Check if it's in mJ
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found in parameter file - skipping Shock Stress vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:10])}...")
                return
            
            # Filter data: only rows with valid shock stress and laser energy
            valid_data = df[(df['Peak Shock Stress (GPa)'].notna()) & (df[laser_energy_col].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both Peak Shock Stress and laser energy data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Peak Shock Stress + Laser Energy data points - skipping plot")
                shock_count = df['Peak Shock Stress (GPa)'].notna().sum()
                energy_count = df[laser_energy_col].notna().sum()
                self.progress_signal.emit(f"   Peak Shock Stress data points: {shock_count}, Laser energy data points: {energy_count}")
                return
                
            # Ensure laser energy is numeric (keep original units - mJ or J)
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data['Peak Shock Stress (GPa)'] = pd.to_numeric(valid_data['Peak Shock Stress (GPa)'], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data[laser_energy_col].notna()) & (valid_data['Peak Shock Stress (GPa)'].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Get material column (should be added by parameter file)
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                # No material column, create a default one
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (same as HEL plot)
            materials = valid_data[material_col].unique()
            cmap = plt.get_cmap('tab10' if len(materials) <= 10 else 'tab20')
            colors = {mat: cmap(i / max(len(materials), 1)) for i, mat in enumerate(materials)}
            
            # Plot data grouped by material
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Plot with error bars if shock stress uncertainty is available
                # Y-error bars: Shock Stress Uncertainty (laser energy is controlled, no x-error bars)
                yerr = None
                if 'Peak Shock Stress Uncertainty (GPa)' in material_data.columns:
                    yerr = pd.to_numeric(material_data['Peak Shock Stress Uncertainty (GPa)'], errors='coerce')
                    yerr = yerr.fillna(0)
                    if (yerr > 0).any():
                        yerr = yerr.values  # Convert to numpy array
                    else:
                        yerr = None
                
                if yerr is not None:
                    ax.errorbar(
                        material_data[laser_energy_col],
                        material_data['Peak Shock Stress (GPa)'],
                        yerr=yerr,
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=material
                    )
                else:
                    ax.scatter(
                        material_data[laser_energy_col],
                        material_data['Peak Shock Stress (GPa)'],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=material
                    )
                
            # Set labels and title (use correct unit based on column)
            energy_unit = 'mJ' if energy_in_mj else 'J'
            ax.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_ylabel('Peak Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Peak Shock Stress vs Laser Energy by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_laser_energy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Laser Energy plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Laser Energy plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_waveplate_angle_plot(self, spade_output_dir):
        """Generate Shock Stress vs Waveplate Angle scatter plot grouped by material"""
        self.progress_signal.emit("Generating Shock Stress vs Waveplate Angle plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Waveplate Angle plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Look for waveplate angle column
            waveplate_angle_col = None
            
            # Check for waveplate angle column
            if 'Waveplate_Angle (Degrees)' in df.columns:
                waveplate_angle_col = 'Waveplate_Angle (Degrees)'
            else:
                possible_names = [
                    'Waveplate_Angle (Degrees)', 'Waveplate Angle (Degrees)',
                    'Waveplate_Angle', 'Waveplate Angle', 'WaveplateAngle',
                    'waveplate_angle', 'waveplate angle', 'Waveplate',
                    'Angle (Degrees)', 'Angle', 'angle'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            waveplate_angle_col = col_name
                            break
                    if waveplate_angle_col:
                        break
            
            if waveplate_angle_col is None:
                self.progress_signal.emit("⚠ Waveplate angle column not found - skipping Shock Stress vs Waveplate Angle plot")
                return
            
            self.progress_signal.emit(f"✓ Found waveplate angle column: '{waveplate_angle_col}'")
            
            # Get shock stress - use ONLY EOS-calculated value from spall analysis (no fallbacks)
            shock_stress_col = None
            shock_stress_unc_col = None
            
            if 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
                shock_stress_unc_col = 'Peak Shock Stress Uncertainty (GPa)' if 'Peak Shock Stress Uncertainty (GPa)' in df.columns else None
            else:
                self.progress_signal.emit("⚠ Peak Shock Stress not available - skipping plot")
                return
            
            # Filter data: only rows with valid shock stress and waveplate angle
            valid_data = df[(df[shock_stress_col].notna()) & (df[waveplate_angle_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Shock Stress + Waveplate Angle data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column first
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[waveplate_angle_col] = pd.to_numeric(valid_data[waveplate_angle_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data[waveplate_angle_col].notna()) & (valid_data[shock_stress_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if uncertainty is available
                # Y-error bars: Shock Stress Uncertainty (waveplate angle is controlled, no x-error bars)
                yerr = None
                if shock_stress_unc_col and shock_stress_unc_col in material_data.columns:
                    yerr = pd.to_numeric(material_data[shock_stress_unc_col], errors='coerce')
                    yerr = yerr.fillna(0)
                    if (yerr > 0).any():
                        yerr = yerr.values  # Convert to numpy array
                    else:
                        yerr = None
                
                if yerr is not None:
                    errorbar_handle = ax.errorbar(
                        material_data[waveplate_angle_col],
                        material_data[shock_stress_col],
                        yerr=yerr,
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data[waveplate_angle_col],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            ax.set_xlabel('Waveplate Angle (Degrees)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Shock Stress vs Waveplate Angle by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_waveplate_angle_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Waveplate Angle plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Waveplate Angle plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_laser_energy_vs_waveplate_angle_plot(self, spade_output_dir):
        """Generate Laser Energy vs Waveplate Angle scatter plot to show energy variation at each angle"""
        self.progress_signal.emit("Generating Laser Energy vs Waveplate Angle plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Look for waveplate angle column
            waveplate_angle_col = None
            if 'Waveplate_Angle (Degrees)' in df.columns:
                waveplate_angle_col = 'Waveplate_Angle (Degrees)'
            else:
                possible_names = [
                    'Waveplate_Angle (Degrees)', 'Waveplate Angle (Degrees)',
                    'Waveplate_Angle', 'Waveplate Angle', 'WaveplateAngle',
                    'waveplate_angle', 'waveplate angle', 'Waveplate',
                    'Angle (Degrees)', 'Angle', 'angle'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            waveplate_angle_col = col_name
                            break
                    if waveplate_angle_col:
                        break
            
            if waveplate_angle_col is None:
                self.progress_signal.emit("⚠ Waveplate angle column not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            # Look for laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            self.progress_signal.emit(f"✓ Found waveplate angle column: '{waveplate_angle_col}'")
            self.progress_signal.emit(f"✓ Found laser energy column: '{laser_energy_col}'")
            
            # Filter to only include traces that were successfully plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            if 'aligned_ok' in df.columns:
                valid_data = df[df['aligned_ok'] == True].copy()
            else:
                valid_data = df.copy()
            
            # Skip Unknown material if configured
            skip_unknown = self.spade_params.get('skip_unknown_material_traces', False)
            if skip_unknown:
                material_col = None
                for col_name in valid_data.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                if material_col:
                    valid_data = valid_data[valid_data[material_col].str.lower() != 'unknown'].copy()
            
            # Convert to numeric
            valid_data[waveplate_angle_col] = pd.to_numeric(valid_data[waveplate_angle_col], errors='coerce')
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            
            # Remove rows with NaN values
            valid_data = valid_data.dropna(subset=[waveplate_angle_col, laser_energy_col])
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data points - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get material column for color coding
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col and material_col in valid_data.columns:
                materials = valid_data[material_col].unique()
                colors = self._get_material_color_mapping(materials)
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
                legend_handles = []
                legend_labels = []
                
                for i, material in enumerate(materials):
                    material_data = valid_data[valid_data[material_col] == material]
                    
                    if len(material_data) == 0:
                        continue
                    
                    marker = markers[i % len(markers)]
                    color = colors[material]
                    n_points = len(material_data)
                    
                    scatter_handle = ax.scatter(
                        material_data[waveplate_angle_col],
                        material_data[laser_energy_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                    legend_labels.append(f"{material} (n={n_points})")
            else:
                # No material column, plot all points in one color
                ax.scatter(
                    valid_data[waveplate_angle_col],
                    valid_data[laser_energy_col],
                    marker='o',
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"All data (n={len(valid_data)})"
                )
            
            # Set labels and title
            ax.set_xlabel('Waveplate Angle (Degrees)', fontsize=14, fontweight='bold')
            energy_unit = "mJ" if energy_in_mj else "J"
            ax.set_ylabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_title('Laser Energy vs Waveplate Angle\n(Showing Energy Variation at Each Angle)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if material_col and material_col in valid_data.columns:
                ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            else:
                ax.legend(loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'laser_energy_vs_waveplate_angle.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Laser Energy vs Waveplate Angle plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Laser Energy vs Waveplate Angle plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_peak_velocity_plot(self, spade_output_dir):
        """Generate Shock Stress vs Peak Velocity scatter plot grouped by material"""
        self.progress_signal.emit("Generating Shock Stress vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Peak Velocity plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check for peak velocity column (max_velocity_ms)
            peak_velocity_col = 'max_velocity_ms'
            if peak_velocity_col not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping Shock Stress vs Peak Velocity plot")
                return
            
            # Get or calculate shock stress (same logic as other plots)
            # Get shock stress - use ONLY EOS-calculated value from spall analysis (no fallbacks)
            shock_stress_col = None
            shock_stress_unc_col = None
            
            if 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
                shock_stress_unc_col = 'Peak Shock Stress Uncertainty (GPa)' if 'Peak Shock Stress Uncertainty (GPa)' in df.columns else None
            else:
                self.progress_signal.emit("⚠ Peak Shock Stress not available - skipping plot")
                return
            
            # Filter data: only rows with valid shock stress and peak velocity
            valid_data = df[(df[shock_stress_col].notna()) & (df[peak_velocity_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Shock Stress + Peak Velocity data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column first
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[peak_velocity_col] = pd.to_numeric(valid_data[peak_velocity_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Convert free surface velocity (peak velocity) to particle velocity for plotting
            # u_p = u_fs / 2 (free surface velocity is twice the particle velocity)
            valid_data['particle_velocity_ms'] = valid_data[peak_velocity_col] / 2.0
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['particle_velocity_ms'].notna()) & (valid_data[shock_stress_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Apply 3-sigma outlier filter to remove extreme points
            if len(valid_data) > 3:
                valid_data, outliers = filter_3sigma_outliers(
                    valid_data, 
                    'particle_velocity_ms', 
                    shock_stress_col,
                    progress_callback=self.progress_signal.emit
                )
                
                if len(valid_data) == 0:
                    self.progress_signal.emit("⚠ No valid data after 3-sigma filtering - skipping plot")
                    return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if uncertainty is available
                # Use particle velocity on x-axis (not free surface velocity)
                # Y-error bars: Shock Stress Uncertainty
                # X-error bars: Peak Velocity Uncertainty (if available)
                yerr = None
                xerr = None
                
                if shock_stress_unc_col and shock_stress_unc_col in material_data.columns:
                    yerr = pd.to_numeric(material_data[shock_stress_unc_col], errors='coerce')
                    yerr = yerr.fillna(0)
                    if (yerr > 0).any():
                        yerr = yerr.values  # Convert to numpy array
                    else:
                        yerr = None
                
                # Check for peak velocity uncertainty (if available)
                if 'max_velocity_unc_ms' in material_data.columns:
                    xerr = pd.to_numeric(material_data['max_velocity_unc_ms'], errors='coerce')
                    xerr = xerr.fillna(0)
                    if (xerr > 0).any():
                        xerr = xerr.values  # Convert to numpy array
                    else:
                        xerr = None
                
                if yerr is not None or xerr is not None:
                    errorbar_handle = ax.errorbar(
                        material_data['particle_velocity_ms'],
                        material_data[shock_stress_col],
                        xerr=xerr,  # X-error bars for peak velocity uncertainty (if available)
                        yerr=yerr,  # Y-error bars for shock stress uncertainty
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data['particle_velocity_ms'],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            # Note: x-axis shows particle velocity (u_p = u_fs/2) to show the quadratic relationship
            ax.set_xlabel('Particle Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Shock Stress vs Particle Velocity by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_peak_velocity_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Peak Velocity plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_vs_peak_shock_stress_plots(self, spade_output_dir):
        """Plot Flyer Column/Row versus peak shock stress with material grouping"""
        self.progress_signal.emit("Generating Flyer Column/Row vs Peak Shock Stress plots...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Flyer Row/Column plots")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping Flyer Row/Column plots")
                return
            
            # Identify required columns
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None and col_col is None:
                self.progress_signal.emit("⚠ Flyer Row/Column columns not found - skipping positional plots")
                return
            
            # Determine shock stress column
            # Use ONLY EOS-calculated value from spall analysis (no fallbacks)
            shock_stress_col = None
            if 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
            else:
                self.progress_signal.emit("⚠ Peak Shock Stress not available - skipping Flyer Row/Column plots")
                return
            
            required_columns = [shock_stress_col]
            if col_col:
                required_columns.append(col_col)
            if row_col:
                required_columns.append(row_col)
            
            subset_columns = required_columns + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_columns].copy()
            valid_data = valid_data[valid_data[shock_stress_col].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid shock stress data for Flyer Row/Column plots")
                return
            
            # Filter aligned traces only
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No aligned traces available for Flyer Row/Column plots")
                return
            
            # Skip Unknown material if configured
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            else:
                valid_data[material_col] = df.loc[valid_data.index, material_col]
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping positional plots")
                return
            
            plots_to_generate = []
            if col_col is not None:
                col_numeric, col_ticks = self._convert_row_column_to_numeric(valid_data[col_col])
                valid_data['Column_numeric'] = col_numeric
                column_valid = valid_data[valid_data['Column_numeric'].notna()].copy()
                if len(column_valid) > 0:
                    plots_to_generate.append(('Flyer Column', 'Column_numeric', col_ticks, column_valid))
                else:
                    self.progress_signal.emit("⚠ No valid numeric Flyer Column values - skipping column plot")
            
            if row_col is not None:
                row_numeric, row_ticks = self._convert_row_column_to_numeric(valid_data[row_col])
                valid_data['Row_numeric'] = row_numeric
                row_valid = valid_data[valid_data['Row_numeric'].notna()].copy()
                if len(row_valid) > 0:
                    plots_to_generate.append(('Flyer Row', 'Row_numeric', row_ticks, row_valid))
                else:
                    self.progress_signal.emit("⚠ No valid numeric Flyer Row values - skipping row plot")
            
            if not plots_to_generate:
                self.progress_signal.emit("⚠ No positional plots generated due to missing row/column data")
                return
            
            # Get materials and colors
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            
            fig, axes = plt.subplots(1, len(plots_to_generate), figsize=(9 * len(plots_to_generate), 7), sharey=True)
            if len(plots_to_generate) == 1:
                axes = [axes]
            
            legend_handles = []
            legend_labels = []
            
            for axis_idx, (axis_label, numeric_col, tick_mapping, plot_data) in enumerate(plots_to_generate):
                ax = axes[axis_idx]
                for i, material in enumerate(materials):
                    material_data = plot_data[plot_data[material_col] == material]
                    if len(material_data) == 0:
                        continue
                    marker = markers[i % len(markers)]
                    color = colors[material]
                    scatter_handle = ax.scatter(
                        material_data[numeric_col],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=90,
                        alpha=0.8,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{material} (n={len(material_data)})"
                    )
                    if axis_idx == 0:
                        legend_handles.append(scatter_handle)
                        legend_labels.append(f"{material} (n={len(material_data)})")
                
                ax.set_xlabel(axis_label, fontsize=14, fontweight='bold')
                if axis_idx == 0:
                    ax.set_ylabel('Peak Shock Stress (GPa)', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if tick_mapping:
                    ticks = sorted(tick_mapping.keys())
                    labels = [tick_mapping[t] for t in ticks]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
            
            title = 'Flyer Column/Row vs Peak Shock Stress'
            fig.suptitle(title, fontsize=18, fontweight='bold')
            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            plt.tight_layout(rect=[0, 0, 0.98, 0.95])
            
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_vs_peak_shock_stress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column vs Peak Shock Stress plots: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column vs Peak Shock Stress plots: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_vs_peak_velocity_heatmap(self, spade_output_dir):
        """Create heatmap of Flyer Row/Column vs Peak Velocity with energy-bin subplots"""
        self.progress_signal.emit("Generating Flyer Row/Column vs Peak Velocity heatmap...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from math import ceil, sqrt
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping heatmap")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping heatmap")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping heatmap")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping heatmap")
                return
            
            # Find laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
                self.progress_signal.emit(f"   Found laser energy column: '{laser_energy_col}'")
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
                if laser_energy_col:
                    self.progress_signal.emit(f"   Found laser energy column: '{laser_energy_col}'")
                else:
                    self.progress_signal.emit("   ⚠ Laser energy column not found - will show only combined heatmap")
            
            # Prepare base data
            subset_columns = [row_col, col_col, 'max_velocity_ms'] + [c for c in ['aligned_ok'] if c in df.columns]
            if laser_energy_col:
                subset_columns.append(laser_energy_col)
            
            valid_data = df[subset_columns].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms'])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for Flyer Row/Column heatmap")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping heatmap")
                return
            
            # Helper function to create energy bins (same algorithm as MAD filter)
            def create_energy_bins(energies, energy_bin_width=30.0):
                """Create energy bins using iterative mean-based clustering"""
                sorted_energies = sorted(energies.unique())
                energy_bins = []
                remaining_energies = sorted_energies.copy()
                
                while len(remaining_energies) > 0:
                    current_bin = [remaining_energies.pop(0)]
                    changed = True
                    while changed:
                        changed = False
                        bin_mean = np.mean(current_bin)
                        to_remove = []
                        for energy in remaining_energies:
                            if abs(energy - bin_mean) <= energy_bin_width:
                                current_bin.append(energy)
                                to_remove.append(energy)
                                changed = True
                        for energy in to_remove:
                            remaining_energies.remove(energy)
                    energy_bins.append(current_bin)
                
                return energy_bins
            
            # Helper function to create a single heatmap
            def create_heatmap_subplot(ax, data_subset, row_col, col_col, title, cmap, vmin=None, vmax=None):
                """Create a heatmap subplot from data subset with local color range
                Returns: (heatmap, data_points_used, total_data_points)
                """
                total_data_points = len(data_subset)
                if total_data_points == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    return None, 0, 0
                
                # Filter to only rows with valid row, column, and velocity data
                valid_for_grouping = data_subset.dropna(subset=[row_col, col_col, 'max_velocity_ms'])
                data_points_used = len(valid_for_grouping)
                
                if data_points_used == 0:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    return None, 0, total_data_points
                
                grouped = valid_for_grouping.groupby([row_col, col_col])['max_velocity_ms']
                mean_table = grouped.mean().unstack()
                count_table = grouped.count().unstack().reindex_like(mean_table)
                max_table = grouped.max().unstack().reindex_like(mean_table)
                min_table = grouped.min().unstack().reindex_like(mean_table)
                
                if mean_table.empty:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    return None, 0, total_data_points
                
                rows = list(mean_table.index.astype(str))
                columns = list(mean_table.columns.astype(str))
                data = mean_table.values.astype(float)
                
                # Calculate local min/max for this subplot if not provided
                if vmin is None or vmax is None:
                    valid_data_values = data[~np.isnan(data)]
                    if len(valid_data_values) > 0:
                        local_vmin = np.nanmin(data)
                        local_vmax = np.nanmax(data)
                    else:
                        local_vmin = None
                        local_vmax = None
                else:
                    local_vmin = vmin
                    local_vmax = vmax
                
                masked_data = np.ma.masked_invalid(data)
                heatmap = ax.imshow(masked_data, cmap=cmap, aspect='auto', origin='lower', vmin=local_vmin, vmax=local_vmax)
                
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        value = data[i, j]
                        if not np.isnan(value):
                            count_val = count_table.values[i, j] if not np.isnan(count_table.values[i, j]) else np.nan
                            max_val = max_table.values[i, j] if not np.isnan(max_table.values[i, j]) else np.nan
                            min_val = min_table.values[i, j] if not np.isnan(min_table.values[i, j]) else np.nan
                            text_color = 'white' if (local_vmax is not None and value > 0.8 * local_vmax) or (local_vmax is None and np.nanmax(data) > 0 and value > 0.8 * np.nanmax(data)) else 'black'
                            label_lines = [f"{value:.1f}"]
                            if not np.isnan(count_val):
                                label_lines.append(f"n={int(count_val)}")
                            ax.text(j, i, "\n".join(label_lines), ha='center', va='center', color=text_color, fontsize=7)
                
                ax.set_xticks(np.arange(len(columns)))
                ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=8)
                ax.set_yticks(np.arange(len(rows)))
                ax.set_yticklabels(rows, fontsize=8)
                ax.set_xlabel('Flyer Column', fontsize=10, fontweight='bold')
                ax.set_ylabel('Flyer Row', fontsize=10, fontweight='bold')
                ax.set_title(title, fontsize=11, fontweight='bold')
                
                return heatmap, data_points_used, total_data_points
            
            # Create energy bins if laser energy column exists
            energy_bins = []
            if laser_energy_col and laser_energy_col in valid_data.columns:
                valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
                laser_energies = valid_data[laser_energy_col].dropna()
                
                if len(laser_energies) > 0:
                    energy_bin_width = 30.0  # ±30 mJ bins
                    energy_bins = create_energy_bins(laser_energies, energy_bin_width)
                    self.progress_signal.emit(f"   Created {len(energy_bins)} energy bin(s) from {len(laser_energies.unique())} unique energy levels")
                    for bin_idx, energy_bin in enumerate(energy_bins):
                        bin_mean = np.mean(energy_bin)
                        bin_min = min(energy_bin)
                        bin_max = max(energy_bin)
                        self.progress_signal.emit(f"      Bin {bin_idx+1}: mean={bin_mean:.1f} mJ, range=[{bin_min:.1f}, {bin_max:.1f}] mJ")
                else:
                    self.progress_signal.emit("   ⚠ No valid laser energy values found - will show only combined heatmap")
            else:
                if laser_energy_col:
                    self.progress_signal.emit(f"   ⚠ Laser energy column '{laser_energy_col}' not in valid data - will show only combined heatmap")
            
            # Determine number of subplots: 1 (combined) + number of energy bins with data
            num_subplots = 1 + len(energy_bins)
            self.progress_signal.emit(f"   Creating {num_subplots} subplot(s): 1 combined + {len(energy_bins)} energy-bin heatmap(s)")
            
            # Calculate grid dimensions
            if num_subplots == 1:
                nrows, ncols = 1, 1
            else:
                ncols = min(3, ceil(sqrt(num_subplots)))
                nrows = ceil(num_subplots / ncols)
            
            # Create figure with subplots
            fig_width = max(12, ncols * 4)
            fig_height = max(10, nrows * 4)
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
            
            # Handle single subplot case
            if num_subplots == 1:
                axes = [axes]
            else:
                # Convert to list of axes for consistent indexing
                if nrows == 1:
                    # Single row: axes is already a 1D array
                    axes = [axes] if ncols == 1 else list(axes) if hasattr(axes, '__iter__') else [axes]
                else:
                    # Multiple rows: axes is 2D, need to flatten
                    axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
            
            self.progress_signal.emit(f"   Grid layout: {nrows} rows × {ncols} columns, {len(axes)} axes available")
            
            # Use colormap that matches other scatter plots (plasma is similar aesthetic to tab10/tab20)
            # Alternative options: 'plasma', 'inferno', 'magma', 'cividis'
            cmap = plt.get_cmap('plasma')
            
            # Subplot (1,1): Combined heatmap (all energy levels) - uses its own local range
            ax_combined = axes[0]
            total_input_data = len(valid_data)
            heatmap_combined, data_used_combined, total_combined = create_heatmap_subplot(
                ax_combined, valid_data, row_col, col_col,
                'Peak Velocity Heatmap (All Energy Levels)', cmap, None, None
            )
            
            # Update title with actual data usage
            if heatmap_combined is not None and data_used_combined > 0:
                if data_used_combined != total_combined:
                    title_combined = f'Peak Velocity Heatmap (All Energy Levels)\nUsed: {data_used_combined}/{total_combined} data points'
                else:
                    title_combined = f'Peak Velocity Heatmap (All Energy Levels)\n({data_used_combined} data points)'
                ax_combined.set_title(title_combined, fontsize=11, fontweight='bold')
            
            if data_used_combined != total_combined:
                excluded = total_combined - data_used_combined
                self.progress_signal.emit(f"   ⚠ Combined heatmap: {excluded} data point(s) excluded (missing row/column/velocity data)")
            
            # Additional subplots: One per energy bin
            subplot_idx = 1
            bins_with_data = 0
            for bin_idx, energy_bin in enumerate(energy_bins):
                if subplot_idx >= len(axes):
                    self.progress_signal.emit(f"   ⚠ Reached maximum subplot limit ({len(axes)}), skipping remaining bins")
                    break
                
                bin_mean = np.mean(energy_bin)
                energy_bin_width = 30.0
                
                # Filter data for this energy bin
                bin_data = valid_data[
                    (valid_data[laser_energy_col] >= bin_mean - energy_bin_width) &
                    (valid_data[laser_energy_col] <= bin_mean + energy_bin_width)
                ].copy()
                
                if len(bin_data) > 0:
                    bins_with_data += 1
                    energy_unit = 'mJ' if energy_in_mj else 'J'
                    total_bin_data = len(bin_data)
                    title = f'Energy: {bin_mean:.0f}±{energy_bin_width:.0f} {energy_unit} (n={total_bin_data})'
                    ax_bin = axes[subplot_idx]
                    self.progress_signal.emit(f"   Creating heatmap for bin {bin_idx+1}: {title}")
                    # Each bin uses its own local color range (pass None, None)
                    heatmap_bin, data_used_bin, total_bin = create_heatmap_subplot(
                        ax_bin, bin_data, row_col, col_col,
                        title, cmap, None, None
                    )
                    
                    # Update title with actual data usage
                    if heatmap_bin is not None and data_used_bin > 0:
                        if data_used_bin != total_bin:
                            title_updated = f'Energy: {bin_mean:.0f}±{energy_bin_width:.0f} {energy_unit}\nUsed: {data_used_bin}/{total_bin} data points'
                        else:
                            title_updated = f'Energy: {bin_mean:.0f}±{energy_bin_width:.0f} {energy_unit}\n({data_used_bin} data points)'
                        ax_bin.set_title(title_updated, fontsize=11, fontweight='bold')
                    
                    if data_used_bin != total_bin:
                        excluded = total_bin - data_used_bin
                        self.progress_signal.emit(f"      ⚠ Bin {bin_idx+1}: {excluded} data point(s) excluded (missing row/column/velocity data)")
                    
                    subplot_idx += 1
                else:
                    self.progress_signal.emit(f"   ⚠ Bin {bin_idx+1} ({bin_mean:.0f}±{energy_bin_width:.0f} mJ) has no data - skipping")
            
            self.progress_signal.emit(f"   Created {bins_with_data} energy-bin heatmap(s) with data")
            
            # Hide unused subplots
            for idx in range(subplot_idx, len(axes)):
                axes[idx].axis('off')
            
            # Add individual colorbars for each subplot (since each has its own range)
            # We'll add colorbars to all subplots that have data
            heatmaps_created = []
            if heatmap_combined is not None:
                heatmaps_created.append((axes[0], heatmap_combined))
            
            # Collect all heatmaps from energy bins
            for idx in range(1, subplot_idx):
                if idx < len(axes):
                    # Get the heatmap from the axes (it's stored in the axes' images)
                    ax_imgs = axes[idx].images
                    if len(ax_imgs) > 0:
                        heatmaps_created.append((axes[idx], ax_imgs[0]))
            
            # Add colorbar to each subplot
            for ax, heatmap in heatmaps_created:
                cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
                cbar.set_label('Peak Velocity (m/s)', fontsize=10, fontweight='bold')
            
            # Add legend/explanation
            legend_text = (
                "Cell labels show:\n"
                " - Avg peak velocity (m/s)\n"
                " - n = traces contributing"
            )
            fig.text(0.99, 0.01, legend_text, ha='right', va='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_peak_velocity_heatmap.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column vs Peak Velocity heatmap: {plot_path}")
            self.progress_signal.emit(f"   Created {subplot_idx} subplot(s): 1 combined + {subplot_idx-1} energy-bin heatmap(s)")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column vs Peak Velocity heatmap: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_pair_vs_peak_velocity_plot(self, spade_output_dir):
        """Plot peak velocity for each Flyer Row/Column pair, grouped by material"""
        self.progress_signal.emit("Generating Flyer Row/Column pair vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping row/column pair plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping row/column pair plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping row/column pair plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping row/column pair plot")
                return
            
            subset_cols = [row_col, col_col, 'max_velocity_ms'] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms'])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for row/column pair plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping row/column pair plot")
                return
            
            # Attach material info
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            else:
                valid_data[material_col] = df.loc[valid_data.index, material_col]
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping row/column pair plot")
                return
            
            # Convert rows/columns to numeric order and build pair labels
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid numeric row/column data - skipping row/column pair plot")
                return
            
            # Determine sorting order (ascending row, then column)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            def format_label(row_value, col_value, row_raw, col_raw):
                # Format row/column labels: remove .0 if present (e.g., R1.0 C1.0 -> R1C1)
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            # Build ordered pair list
            ordered_pairs = []
            pair_labels = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                for col_val in unique_cols:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    ordered_pairs.append((row_val, col_val, label))
                    pair_labels.append(label)
            
            if not ordered_pairs:
                self.progress_signal.emit("⚠ No valid row/column combinations to plot")
                return
            
            pair_to_position = {pair[2]: idx for idx, pair in enumerate(ordered_pairs)}
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            valid_data['RowCol_Pos'] = valid_data['RowCol_Label'].map(pair_to_position)
            
            # Prepare figure
            fig, ax = plt.subplots(figsize=(max(12, len(ordered_pairs) * 0.7), 8))
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            
            legend_handles = []
            legend_labels = []
            jitter = 0.05
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                if len(material_data) == 0:
                    continue
                marker = markers[i % len(markers)]
                color = colors[material]
                x_positions = material_data['RowCol_Pos'] + np.random.uniform(-jitter, jitter, size=len(material_data))
                scatter_handle = ax.scatter(
                    x_positions,
                    material_data['max_velocity_ms'],
                    marker=marker,
                    c=[color],
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"{material} (n={len(material_data)})"
                )
                legend_handles.append(scatter_handle)
                legend_labels.append(f"{material} (n={len(material_data)})")
            
            ax.set_xticks(range(len(pair_to_position)))
            ax.set_xticklabels(pair_labels, rotation=45, ha='right')
            ax.set_xlabel('Flyer Row/Column Pair', fontsize=14, fontweight='bold')
            ax.set_ylabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_title('Peak Velocity by Flyer Row/Column Pair', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best', title='Material')
            
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_pair_peak_velocity.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column pair vs Peak Velocity plot: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column pair vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_pair_vs_peak_velocity_by_material_plot(self, spade_output_dir):
        """Plot peak velocity for each Flyer Row/Column pair, with 3 subplots (one per material), markers color-coded by laser energy"""
        self.progress_signal.emit("Generating Flyer Row/Column pair vs Peak Velocity plot (by material, color-coded by laser energy)...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping row/column pair by material plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping row/column pair by material plot")
                return
            
            # Find laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping row/column pair by material plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping row/column pair by material plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping row/column pair by material plot")
                return
            
            # Get material column
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                df['Material'] = 'Unknown'
                material_col = 'Material'
            
            subset_cols = [row_col, col_col, 'max_velocity_ms', laser_energy_col, material_col] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms', laser_energy_col])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for row/column pair by material plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping row/column pair by material plot")
                return
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping row/column pair by material plot")
                return
            
            # Convert rows/columns to numeric order and build pair labels
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid numeric row/column data - skipping row/column pair by material plot")
                return
            
            # Determine sorting order (ascending row, then column)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            def format_label(row_value, col_value, row_raw, col_raw):
                # Format row/column labels: remove .0 if present (e.g., R1.0 C1.0 -> R1C1)
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            # Build ordered pair list with zigzag pattern:
            # Odd rows (1, 3, 5): columns in reverse order (C5, C4, C3, C2, C1)
            # Even rows (2, 4): columns in normal order (C1, C2, C3, C4, C5)
            ordered_pairs = []
            pair_labels = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                
                # Determine column order: reverse for odd rows, normal for even rows
                # Row 1, 3, 5 are odd -> reverse; Row 2, 4 are even -> normal
                is_odd_row = (int(row_val) % 2 == 1)
                col_order = reversed(unique_cols) if is_odd_row else unique_cols
                
                for col_val in col_order:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    ordered_pairs.append((row_val, col_val, label))
                    pair_labels.append(label)
            
            if not ordered_pairs:
                self.progress_signal.emit("⚠ No valid row/column combinations to plot")
                return
            
            pair_to_position = {pair[2]: idx for idx, pair in enumerate(ordered_pairs)}
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            valid_data['RowCol_Pos'] = valid_data['RowCol_Label'].map(pair_to_position)
            
            # Ensure laser energy is numeric
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data = valid_data[valid_data[laser_energy_col].notna()].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid laser energy data - skipping row/column pair by material plot")
                return
            
            # Get unique materials (limit to 3 for 3 subplots)
            materials = sorted(valid_data[material_col].unique())[:3]
            if len(materials) == 0:
                self.progress_signal.emit("⚠ No materials found - skipping row/column pair by material plot")
                return
            
            # Get global laser energy range for consistent colormap across all subplots
            global_energy_min = valid_data[laser_energy_col].min()
            global_energy_max = valid_data[laser_energy_col].max()
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(max(12, len(ordered_pairs) * 0.7), 12))
            if len(materials) < 3:
                # Hide unused subplots
                for i in range(len(materials), 3):
                    axes[i].axis('off')
            
            # Use a colormap for laser energy
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=global_energy_min, vmax=global_energy_max)
            jitter = 0.05
            
            for subplot_idx, material in enumerate(materials):
                ax = axes[subplot_idx]
                material_data = valid_data[valid_data[material_col] == material].copy()
                
                if len(material_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {material}', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_ylabel('Peak Velocity (m/s)', fontsize=12, fontweight='bold')
                    continue
                
                x_positions = material_data['RowCol_Pos'] + np.random.uniform(-jitter, jitter, size=len(material_data))
                laser_energies = material_data[laser_energy_col].values
                
                # Color-code by laser energy
                scatter = ax.scatter(
                    x_positions,
                    material_data['max_velocity_ms'],
                    c=laser_energies,
                    cmap=cmap,
                    norm=norm,
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                ax.set_xticks(range(len(pair_to_position)))
                ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Peak Velocity (m/s)', fontsize=12, fontweight='bold')
                ax.set_title(f'{material} (n={len(material_data)})', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Set x-axis label on bottom subplot only
            axes[-1].set_xlabel('Flyer Row/Column Pair', fontsize=14, fontweight='bold')
            
            # Overall title
            fig.suptitle('Peak Velocity by Flyer Row/Column Pair (Color-coded by Laser Energy)', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            # Adjust layout first to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.92, 0.99])  # Leave space for colorbar on the right
            
            # Add colorbar for laser energy (shared across all subplots) - positioned to the right of plot area
            energy_unit = 'mJ' if energy_in_mj else 'J'
            # Create colorbar in a separate axes to the right of the plot
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coordinates
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label(f'Laser Energy ({energy_unit})', fontsize=12, fontweight='bold')
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_pair_peak_velocity_by_material_laser_energy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column pair vs Peak Velocity plot (by material, color-coded by laser energy): {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column pair vs Peak Velocity plot (by material): {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_peak_velocity_pattern_analysis_plot(self, spade_output_dir):
        """Generate diagnostic plots to analyze patterns: 1) Effect of laser energy on peak velocity, 2) Effect of test location (RxCy) on peak velocity"""
        self.progress_signal.emit("Generating Peak Velocity Pattern Analysis plots...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping pattern analysis plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping pattern analysis plot")
                return
            
            # Find laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping pattern analysis plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping pattern analysis plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping pattern analysis plot")
                return
            
            # Get material column
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                df['Material'] = 'Unknown'
                material_col = 'Material'
            
            subset_cols = [row_col, col_col, 'max_velocity_ms', laser_energy_col, material_col] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms', laser_energy_col])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for pattern analysis plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after filtering - skipping pattern analysis plot")
                return
            
            # Ensure laser energy is numeric
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data = valid_data[valid_data[laser_energy_col].notna()].copy()
            
            # Convert rows/columns to numeric
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            # Create row/column pair labels
            def format_label(row_val, col_val, row_raw, col_raw):
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            
            # Get unique materials
            materials = sorted(valid_data[material_col].unique())
            colors = self._get_material_color_mapping(materials)
            energy_unit = 'mJ' if energy_in_mj else 'J'
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # ===== SUBPLOT 1: Laser Energy vs Peak Velocity =====
            ax1 = axes[0]
            for material in materials:
                # Filter by material (case-insensitive matching for robustness)
                material_mask = valid_data[material_col].astype(str).str.strip().str.lower() == str(material).strip().lower()
                material_data = valid_data[material_mask].copy()
                
                if len(material_data) == 0:
                    continue
                
                # Ensure numeric and remove any remaining NaN values
                material_data[laser_energy_col] = pd.to_numeric(material_data[laser_energy_col], errors='coerce')
                material_data['max_velocity_ms'] = pd.to_numeric(material_data['max_velocity_ms'], errors='coerce')
                
                # Remove NaN and invalid values
                material_data = material_data.dropna(subset=[laser_energy_col, 'max_velocity_ms'])
                material_data = material_data[
                    (material_data[laser_energy_col] > 0) & 
                    (material_data['max_velocity_ms'] > 0) &
                    (np.isfinite(material_data[laser_energy_col])) &
                    (np.isfinite(material_data['max_velocity_ms']))
                ].copy()
                
                if len(material_data) < 2:
                    # Not enough data points for regression
                    if len(material_data) > 0:
                        x = material_data[laser_energy_col].values
                        y = material_data['max_velocity_ms'].values
                        color = colors.get(material, 'gray')
                        ax1.scatter(x, y, c=[color], s=80, alpha=0.7, edgecolors='black', linewidths=0.5, 
                                  label=f"{material} (n={len(material_data)})")
                    continue
                
                x = material_data[laser_energy_col].values
                y = material_data['max_velocity_ms'].values
                color = colors.get(material, 'gray')
                
                # Scatter plot
                ax1.scatter(x, y, c=[color], s=80, alpha=0.7, edgecolors='black', linewidths=0.5, 
                          label=f"{material} (n={len(material_data)})")
            
            ax1.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax1.set_title('Effect of Laser Energy on Peak Velocity', fontsize=16, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.legend(loc='best', fontsize=10)
            
            # ===== SUBPLOT 2: Test Location (RxCy) vs Peak Velocity =====
            ax2 = axes[1]
            
            # Build ordered pair list with zigzag pattern (same as other plot):
            # Odd rows (1, 3, 5): columns in reverse order (C5, C4, C3, C2, C1)
            # Even rows (2, 4): columns in normal order (C1, C2, C3, C4, C5)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            ordered_pairs = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                
                # Determine column order: reverse for odd rows, normal for even rows
                is_odd_row = (int(row_val) % 2 == 1)
                col_order = reversed(unique_cols) if is_odd_row else unique_cols
                
                for col_val in col_order:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    
                    # Only add if this pair exists in the data
                    if label in valid_data['RowCol_Label'].values:
                        ordered_pairs.append(label)
            
            # Get unique row/column pairs that exist in data and calculate mean velocity for each
            unique_pairs = ordered_pairs  # Use zigzag-ordered pairs
            pair_means = []
            pair_stds = []
            pair_counts = []
            
            for pair in unique_pairs:
                pair_data = valid_data[valid_data['RowCol_Label'] == pair]
                if len(pair_data) > 0:
                    pair_means.append(pair_data['max_velocity_ms'].mean())
                    pair_stds.append(pair_data['max_velocity_ms'].std() if len(pair_data) > 1 else 0)
                    pair_counts.append(len(pair_data))
                else:
                    pair_means.append(np.nan)
                    pair_stds.append(0)
                    pair_counts.append(0)
            
            # Create bar plot with error bars
            x_pos = np.arange(len(unique_pairs))
            bars = ax2.bar(x_pos, pair_means, yerr=pair_stds, capsize=5, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Color bars by row (consistent color for all locations in the same row)
            # Extract row number from pair label (e.g., "R4C1" -> 4)
            for i, pair in enumerate(unique_pairs):
                pair_data = valid_data[valid_data['RowCol_Label'] == pair]
                if len(pair_data) > 0:
                    # Extract row number from the pair label
                    try:
                        # Parse row number from label like "R4C1" or "R4.0C1.0"
                        row_str = pair.split('C')[0].replace('R', '').replace('.0', '')
                        row_num = int(float(row_str))
                        
                        # Use a colormap to assign colors by row (consistent across all columns in same row)
                        # Use a simple color scheme: different shade for each row
                        row_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rows)))
                        row_idx = unique_rows.index(row_num) if row_num in unique_rows else 0
                        bars[i].set_facecolor(row_colors[row_idx])
                    except (ValueError, IndexError):
                        # Fallback: use most common material if row parsing fails
                        most_common_material = pair_data[material_col].mode()[0] if len(pair_data[material_col].mode()) > 0 else materials[0]
                        bars[i].set_facecolor(colors.get(most_common_material, 'gray'))
            
            ax2.set_xlabel('Test Location (Row/Column Pair)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Mean Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax2.set_title('Effect of Test Location on Peak Velocity', fontsize=16, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(unique_pairs, rotation=45, ha='right', fontsize=9)
            ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Add count annotations on bars
            for i, (bar, count) in enumerate(zip(bars, pair_counts)):
                if count > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + pair_stds[i] + 5,
                           f'n={count}', ha='center', va='bottom', fontsize=8)
            
            # Overall title
            fig.suptitle('Peak Velocity Pattern Analysis', fontsize=18, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            plot_path = os.path.join(spade_output_dir, 'peak_velocity_pattern_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Peak Velocity Pattern Analysis plot: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Peak Velocity Pattern Analysis plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def ramer_douglas_peucker(self, points, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for polyline simplification.
        
        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (N, 2) where columns are [time, velocity]
        epsilon : float
            Maximum distance a point can deviate from the line (m/s)
            
        Returns
        -------
        numpy.ndarray
            Simplified array of points preserving critical shape changes
        """
        if len(points) <= 2:
            return points
        
        dmax = 0.0
        index = 0
        end = len(points) - 1
        
        # Calculate perpendicular distance from each point to the line segment
        line_vec = points[end] - points[0]
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            # All points are the same, return endpoints
            d = np.linalg.norm(points[1:-1] - points[0], axis=1) if len(points) > 2 else np.array([])
            if len(d) > 0:
                dmax = np.max(d)
                index = np.argmax(d) + 1
        else:
            # Calculate perpendicular distance using cross product
            # For 2D: cross product = |(x2-x1)*(y-y1) - (y2-y1)*(x-x1)| / length
            vec_pt_start = points[1:end] - points[0]
            cross_prod = np.abs(line_vec[0] * vec_pt_start[:, 1] - line_vec[1] * vec_pt_start[:, 0])
            d = cross_prod / line_len
            
            if len(d) > 0:
                dmax = np.max(d)
                index = np.argmax(d) + 1
        
        # Recursively simplify if maximum deviation exceeds epsilon
        if dmax > epsilon:
            rec_results1 = self.ramer_douglas_peucker(points[:index+1], epsilon)
            rec_results2 = self.ramer_douglas_peucker(points[index:], epsilon)
            # Combine results, avoiding duplicate point at index
            return np.vstack((rec_results1[:-1], rec_results2))
        else:
            # Return only endpoints
            return np.vstack((points[0], points[end]))
    
    def ramer_douglas_peucker_indices(self, time, velocity, epsilon):
        """
        Ramer-Douglas-Peucker algorithm that returns INDICES of simplified points.
        This is used for the hybrid approach to map back to raw data.
        
        Parameters
        ----------
        time : numpy.ndarray
            Time array (ns)
        velocity : numpy.ndarray
            Velocity array (m/s)
        epsilon : float
            Maximum distance a point can deviate from the line (m/s)
            
        Returns
        -------
        numpy.ndarray
            Array of indices pointing to the simplified vertices in the original data
        """
        if len(time) <= 2 or len(velocity) <= 2:
            return np.arange(len(time))
        
        points = np.column_stack((time, velocity))
        
        def rdp_recursive(start_idx, end_idx):
            """Recursive RDP that works with indices"""
            if end_idx - start_idx <= 1:
                return [start_idx, end_idx]
            
            # Get points for this segment
            seg_points = points[start_idx:end_idx+1]
            seg_indices = np.arange(start_idx, end_idx + 1)
            
            if len(seg_points) <= 2:
                return [start_idx, end_idx]
            
            # Calculate perpendicular distances
            line_vec = seg_points[-1] - seg_points[0]
            line_len = np.linalg.norm(line_vec)
            
            dmax = 0.0
            index = start_idx
            
            if line_len == 0:
                # All points are the same
                d = np.linalg.norm(seg_points[1:-1] - seg_points[0], axis=1)
                if len(d) > 0:
                    max_idx = np.argmax(d) + 1
                    index = seg_indices[max_idx]
                    dmax = np.max(d)
            else:
                # Calculate perpendicular distance using cross product
                vec_pt_start = seg_points[1:-1] - seg_points[0]
                cross_prod = np.abs(line_vec[0] * vec_pt_start[:, 1] - line_vec[1] * vec_pt_start[:, 0])
                d = cross_prod / line_len
                
                if len(d) > 0:
                    max_idx = np.argmax(d)
                    index = seg_indices[max_idx + 1]  # +1 because we skipped first point
                    dmax = np.max(d)
            
            # Recursively simplify if maximum deviation exceeds epsilon
            if dmax > epsilon:
                left = rdp_recursive(start_idx, index)
                right = rdp_recursive(index, end_idx)
                # Combine, avoiding duplicate at index
                return left[:-1] + right
            else:
                # Return only endpoints
                return [start_idx, end_idx]
        
        # Run recursive RDP
        indices = rdp_recursive(0, len(time) - 1)
        return np.array(sorted(set(indices)))  # Ensure sorted and unique
    
    def analyze_spall_horizontal_plateau(self, time, velocity, uncert, density, acoustic_velocity, config):
        """
        Spall analysis with Horizontal Plateau Constraint to prevent P1/P2 shifting.
        
        Logic:
        1. Find global peak (maximum velocity)
        2. Find all points >= 95% of peak
        3. Calculate mean velocity of all those points
        4. Line 2 (Plateau): Horizontal line (0 slope) at this mean velocity
        5. Line 1 (Rise): Line from (0,0) to first point (earliest time) in 95% region
        6. P3: First minimum after the maximum (velocity arrives Gaussian-smoothed)
        7. Line 3 (Pullback): Line joining P2 (last plateau point) to P3
        8. P1 = first plateau point, P2 = last plateau point
        
        Returns:
            is_spall (bool): True if valid spall detected
            reason (str): Success or DNS reason
            results (dict): Contains fits, intersections, and physics results
        """
        import numpy as np
        from scipy.signal import find_peaks
        
        # 1. Find Global Peak
        idx_max = np.argmax(velocity)
        v_max = velocity[idx_max]
        t_max = time[idx_max]

        # Spall feature-detection knobs (config "Spall detection filters"):
        #   prominence_factor — a candidate velocity dip must be at least this
        #       fraction of the plateau velocity deep (relative to its shoulders)
        #       to count as a valley. Rejects shallow noise wiggles.
        #   peak_distance_ns  — two accepted valleys must be at least this far
        #       apart in time; de-duplicates a single broad minimum. Converted
        #       to samples here so behaviour is sample-rate independent.
        # A small absolute floor (1 m/s) keeps the prominence bar sane on
        # low-velocity shots where prominence_factor*plateau would be sub-noise.
        prominence_factor = float(config.get('prominence_factor', 0.01))
        peak_distance_ns = float(config.get('peak_distance_ns', 3.0))
        _dt_ns = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
        _min_dist_samples = max(3, int(round(peak_distance_ns / _dt_ns))) if _dt_ns > 0 else 3
        _prom_floor = 1.0  # m/s absolute floor

        # 2. Find all points >= 95% of peak
        threshold_ratio = config.get('plateau_threshold', 0.95)
        threshold_val = v_max * threshold_ratio
        
        # Get all indices where velocity >= 95% of peak
        plateau_mask = velocity >= threshold_val
        plateau_indices = np.where(plateau_mask)[0]
        
        if len(plateau_indices) == 0:
            # Fallback: use points within 2 indices of peak
            plateau_indices = np.array([max(0, idx_max-2), idx_max, min(len(velocity)-1, idx_max+2)])
        
        # 3. Calculate mean velocity of all points >= 95% of peak
        v_plateau_mean = np.mean(velocity[plateau_indices])
        
        # 4. Find first (earliest) and last (latest) points in the 95% region
        idx_first_plateau = plateau_indices[0]  # Earliest time point >= 95%
        idx_last_plateau = plateau_indices[-1]  # Latest time point >= 95%
        t_first_plateau = time[idx_first_plateau]
        v_first_plateau = velocity[idx_first_plateau]
        t_last_plateau = time[idx_last_plateau]
        v_last_plateau = velocity[idx_last_plateau]
        
        print(f"  [HORIZ-PLAT] Global peak: idx={idx_max}, v={v_max:.2f} m/s at t={t_max:.2f} ns")
        print(f"  [HORIZ-PLAT] Found {len(plateau_indices)} points >= 95% of peak (threshold={threshold_val:.2f} m/s)")
        print(f"  [HORIZ-PLAT] Plateau mean velocity: {v_plateau_mean:.2f} m/s")
        print(f"  [HORIZ-PLAT] First plateau point: idx={idx_first_plateau}, t={t_first_plateau:.2f} ns, v={v_first_plateau:.2f} m/s")
        print(f"  [HORIZ-PLAT] Last plateau point: idx={idx_last_plateau}, t={t_last_plateau:.2f} ns, v={v_last_plateau:.2f} m/s")
        sys.stdout.flush()
        
        # 5. Line 1: From (0,0) to first point in 95% region
        # Force line through (0,0) and first plateau point
        if t_first_plateau > 1e-6:  # Avoid division by zero
            m1 = v_first_plateau / t_first_plateau
            c1 = 0.0  # Line passes through origin
        else:
            m1 = 0.0
            c1 = v_first_plateau
        
        print(f"  [HORIZ-PLAT] Line 1: m={m1:.2f}, c={c1:.2f} (from (0,0) to first plateau point)")
        sys.stdout.flush()
        
        # Calculate shock stress from plateau (needed even for DNS cases)
        peak_shock_stress = density * acoustic_velocity * v_plateau_mean / 1e9
        
        # 6. Find Pullback Minimum (to locate the valley, then find first max after it)
        # Search after the last plateau point
        post_plateau_data = velocity[idx_last_plateau:]
        if len(post_plateau_data) == 0:
            # Calculate intersections and fits even for DNS (for visualization)
            # P1: First plateau point
            t_p1 = t_first_plateau
            v_p1 = v_plateau_mean
            
            # P2: Last plateau point
            t_p2 = t_last_plateau
            v_p2 = v_plateau_mean
            
            # Construct minimal fits and intersections for visualization
            fits = {
                'seg1_rise': {'m': m1, 'c': c1, 't_range': (time[0], t_p1), 'start_idx': 0, 'end_idx': idx_first_plateau},
                'seg2_plateau': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p1, t_p2), 'start_idx': idx_first_plateau, 'end_idx': idx_last_plateau},
                'seg3_release': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p2, time[-1]), 'start_idx': idx_last_plateau, 'end_idx': len(time)-1},
                'seg4_recomp': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time)-1, 'end_idx': len(time)-1},
                'seg5_tail': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time)-1, 'end_idx': len(time)-1}
            }
            
            intersections = [
                (t_p1, v_p1),            # P1: First plateau point
                (t_p2, v_p2),            # P2: Last plateau point
                (time[-1], v_plateau_mean),  # P3 (fallback)
                (time[-1], v_plateau_mean)   # P4 (fallback)
            ]
            
            # Return DNS but include plateau velocity, shock stress, and fits for visualization
            return False, "DNS: No data after peak", {
                'Processing Status': 'DNS',
                'Plateau Mean Velocity (m/s)': v_plateau_mean,
                'Peak Shock Stress (GPa)': peak_shock_stress,
                'Peak Shock Stress Uncertainty (GPa)': 0.0,
                'First Maxima (m/s)': v_plateau_mean,
                'Minima (m/s)': np.nan,
                'Second Maxima (m/s)': np.nan,
                'fits': fits,
                'intersections': intersections
            }
        
        # Find pullback minimum P3.
        #
        # Strategy: find the FIRST local minimum after the plateau using
        # prominence-based peak detection on the inverted signal.
        # The physical spall pullback is the first velocity dip after the
        # shock peak, not necessarily the global minimum. Secondary oscillations
        # (reverberations) can produce deeper but later minima that are not the
        # primary spall signal.
        #
        # The velocity passed into this function is already Gaussian-smoothed
        # (spall_smoothing_sigma_ns, applied once by the caller), so no
        # further smoothing happens here.
        #
        # No fallback: if no local valley clears the prominence threshold,
        # the trace is classified DNS immediately. Substituting the global
        # minimum (typically just the last sample of a still-decaying trace)
        # would report a P3 that isn't a real pullback point.

        from scipy.signal import find_peaks as _find_peaks

        # --- First local minimum with sufficient prominence ---
        # Threshold: 1% of plateau mean velocity (floor 1 m/s). Prominence is
        # measured against the plateau level rather than the raw peak so the
        # bar reflects the height actually being pulled back from.
        inverted  = -post_plateau_data.astype(float)
        min_prom  = max(v_plateau_mean * prominence_factor, _prom_floor)
        min_dist  = _min_dist_samples
        valleys, _ = _find_peaks(inverted, prominence=min_prom, distance=min_dist)

        if len(valleys) == 0:
            # No candidate valley cleared the prominence threshold - classify
            # as DNS immediately rather than defaulting to the global minimum.
            print(f"  [HORIZ-PLAT] No local pullback minimum found after plateau "
                  f"(min_prom={min_prom:.2f} m/s) - classifying as DNS")
            sys.stdout.flush()

            t_p1, v_p1 = t_first_plateau, v_plateau_mean
            t_p2, v_p2 = t_last_plateau, v_plateau_mean

            fits = {
                'seg1_rise': {'m': m1, 'c': c1, 't_range': (time[0], t_p1), 'start_idx': 0, 'end_idx': idx_first_plateau},
                'seg2_plateau': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p1, t_p2), 'start_idx': idx_first_plateau, 'end_idx': idx_last_plateau},
                'seg3_release': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p2, time[-1]), 'start_idx': idx_last_plateau, 'end_idx': len(time) - 1},
                'seg4_recomp': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time) - 1, 'end_idx': len(time) - 1},
                'seg5_tail': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time) - 1, 'end_idx': len(time) - 1}
            }

            intersections = [
                (t_p1, v_p1),
                (t_p2, v_p2),
                (np.nan, np.nan),
                (np.nan, np.nan)
            ]

            return False, f"DNS: No local pullback minimum found (prominence threshold {min_prom:.2f} m/s not met)", {
                'Processing Status': 'DNS',
                'Plateau Mean Velocity (m/s)': v_plateau_mean,
                'Peak Shock Stress (GPa)': peak_shock_stress,
                'Peak Shock Stress Uncertainty (GPa)': 0.0,
                'First Maxima (m/s)': v_plateau_mean,
                'Minima (m/s)': np.nan,
                'Second Maxima (m/s)': np.nan,
                'fits': fits,
                'intersections': intersections
            }

        # --- Improved P3/P4 selection ---
        # A spall is a prominent pullback valley FOLLOWED BY a prominent
        # recompression peak. Enumerate prominent peaks too, then take the first
        # valley (in time) whose immediate recompression clears the gates. P4 is
        # the FIRST prominent peak after that valley (local), never the global
        # maximum after P3 -- that avoids latching onto a late / end-of-window
        # minimum on a trace that merely releases without recompressing (the old
        # "first valley + global-max P4" rule fabricated a spurious pullback from
        # the end-of-window decay in exactly that situation).
        peaks_pp, _ = _find_peaks(post_plateau_data.astype(float),
                                  prominence=min_prom, distance=min_dist)

        min_recomp_velocity_ratio = config.get('min_recomp_velocity_ratio', 1.05)
        min_recomp_time_ns = config.get('min_recomp_time_ns', 2.5)
        min_recomp_ratio = config.get('min_recomp_ratio', 0.1)

        sel_p3 = sel_p4 = None      # accepted pair (passes all gates)
        fb_p3 = fb_p4 = None        # last evaluated pair (kept for DNS visualization)
        for _vi in valleys:
            c_p3 = idx_last_plateau + int(_vi)
            if abs(float(velocity[c_p3])) <= 10.0:
                continue            # decayed to baseline -- not a real pullback
            _later = peaks_pp[peaks_pp > _vi]
            if len(_later) == 0:
                continue            # no recompression peak after this valley
            c_p4 = idx_last_plateau + int(_later[0])
            _v3, _v4 = float(velocity[c_p3]), float(velocity[c_p4])
            _pull = v_plateau_mean - _v3
            _reb = _v4 - _v3
            _dt = float(time[c_p4]) - float(time[c_p3])
            fb_p3, fb_p4 = c_p3, c_p4
            if (_v4 > _v3 and _v4 >= _v3 * min_recomp_velocity_ratio
                    and _dt >= min_recomp_time_ns
                    and not (_pull > 1e-6 and _reb < _pull * min_recomp_ratio)):
                sel_p3, sel_p4 = c_p3, c_p4
                break

        if sel_p3 is None and fb_p3 is None:
            # No valley is followed by a prominent recompression peak: the surface
            # released without reforming an internal surface -> DNS.
            print(f"  [HORIZ-PLAT] No valley with a prominent recompression peak - classifying as DNS")
            sys.stdout.flush()
            t_p1, v_p1 = t_first_plateau, v_plateau_mean
            t_p2, v_p2 = t_last_plateau, v_plateau_mean
            fits = {
                'seg1_rise': {'m': m1, 'c': c1, 't_range': (time[0], t_p1), 'start_idx': 0, 'end_idx': idx_first_plateau},
                'seg2_plateau': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p1, t_p2), 'start_idx': idx_first_plateau, 'end_idx': idx_last_plateau},
                'seg3_release': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p2, time[-1]), 'start_idx': idx_last_plateau, 'end_idx': len(time) - 1},
                'seg4_recomp': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time) - 1, 'end_idx': len(time) - 1},
                'seg5_tail': {'m': 0.0, 'c': v_plateau_mean, 't_range': (time[-1], time[-1]), 'start_idx': len(time) - 1, 'end_idx': len(time) - 1}
            }
            intersections = [(t_p1, v_p1), (t_p2, v_p2), (np.nan, np.nan), (np.nan, np.nan)]
            return False, "DNS: No valley with a prominent recompression peak", {
                'Processing Status': 'DNS',
                'Plateau Mean Velocity (m/s)': v_plateau_mean,
                'Peak Shock Stress (GPa)': peak_shock_stress,
                'Peak Shock Stress Uncertainty (GPa)': 0.0,
                'First Maxima (m/s)': v_plateau_mean,
                'Minima (m/s)': np.nan,
                'Second Maxima (m/s)': np.nan,
                'fits': fits,
                'intersections': intersections
            }

        # Accepted pair, or -- if none passed the gates -- the last evaluated pair
        # (kept so DNS Check 2 below reports the gate failure with a fitted plot).
        idx_p3, idx_p4 = (sel_p3, sel_p4) if sel_p3 is not None else (fb_p3, fb_p4)
        v_p3 = float(velocity[idx_p3])
        t_p3 = float(time[idx_p3])
        v_p4 = float(velocity[idx_p4])
        t_p4 = float(time[idx_p4])
        pullback_amp = v_max - v_p3
        print(f"  [HORIZ-PLAT] P3 (pullback min): t={t_p3:.2f} ns, v={v_p3:.2f} m/s "
              f"(pullback={pullback_amp:.1f} m/s); P4 (recompression): t={t_p4:.2f} ns, v={v_p4:.2f} m/s")
        sys.stdout.flush()
        
        # 7. Line 3: From P2 to P3
        # Line joining P2 (last plateau point) to P3 (first minimum after peak)
        # Use P2 and P3 as endpoints to define the line
        if abs(t_last_plateau - t_p3) > 1e-6:  # Avoid division by zero
            # Calculate slope and intercept for line through P2 and P3
            m3 = (v_p3 - v_plateau_mean) / (t_p3 - t_last_plateau)
            c3 = v_plateau_mean - m3 * t_last_plateau
        else:
            # P2 and P3 are at same time (shouldn't happen, but handle it)
            m3 = 0.0
            c3 = v_plateau_mean
        
        print(f"  [HORIZ-PLAT] Line 3: from P2 (t={t_last_plateau:.2f}, v={v_plateau_mean:.2f}) to P3 (t={t_p3:.2f}, v={v_p3:.2f})")
        print(f"  [HORIZ-PLAT] Line 3 fitted: m={m3:.2f}, c={c3:.2f}")
        sys.stdout.flush()
        
        # 8. Recompression peak P4 was selected together with P3 above
        # (the first prominent recompression peak after the pullback valley).

        # 9. Find next minimum after P4 (for Line 5 endpoint)
        # Input is already Gaussian-smoothed; same prominence method as P3.
        post_p4_data = velocity[idx_p4+1:]
        if len(post_p4_data) > 0:
            inverted_p4_signal = -post_p4_data
            min_prominence_p4 = max(v_plateau_mean * prominence_factor, _prom_floor)
            min_distance_p4 = _min_dist_samples
            valleys_p4, _ = find_peaks(inverted_p4_signal, prominence=min_prominence_p4, distance=min_distance_p4)
            
            if len(valleys_p4) > 0:
                # Use first local minimum after P4
                idx_next_min_local = valleys_p4[0]
                idx_next_min = idx_p4 + 1 + idx_next_min_local
                t_next_min = time[idx_next_min]
                v_next_min = velocity[idx_next_min]
                print(f"  [HORIZ-PLAT] Next minimum after P4: idx={idx_next_min}, t={t_next_min:.2f} ns, v={v_next_min:.2f} m/s")
                sys.stdout.flush()
            else:
                # No clear minimum found, use end of data
                idx_next_min = len(velocity) - 1
                t_next_min = time[idx_next_min]
                v_next_min = velocity[idx_next_min]
                print(f"  [HORIZ-PLAT] No minimum found after P4, using end of data: t={t_next_min:.2f} ns, v={v_next_min:.2f} m/s")
                sys.stdout.flush()
        else:
            # No data after P4, use P4 as endpoint
            idx_next_min = idx_p4
            t_next_min = t_p4
            v_next_min = v_p4
            print(f"  [HORIZ-PLAT] No data after P4, using P4 as endpoint")
            sys.stdout.flush()
        
        # DNS Check 2: Physical P3 and recompression validation.
        # P3 is the free-surface pullback minimum, so it must remain strictly
        # positive.  A zero or negative P3 is not a valid spall measurement,
        # even if a subsequent noise rebound would otherwise pass the P4 gates.
        # Require: (a) P3 > 0, (b) P4 > P3,
        #          (c) v_p4 >= v_p3 * min_recomp_velocity_ratio (default 1.05 = 5%),
        #          (d) t_p4 - t_p3 >= min_recomp_time_ns (default 2.5 ns),
        #          (e) rebound_mag >= pullback_mag * min_recomp_ratio (default 10% of drop)
        min_recomp_velocity_ratio = config.get('min_recomp_velocity_ratio', 1.05)
        min_recomp_time_ns = config.get('min_recomp_time_ns', 2.5)
        min_recomp_ratio = config.get('min_recomp_ratio', 0.1)
        pullback_mag = v_plateau_mean - v_p3
        rebound_mag = v_p4 - v_p3
        t_p3_to_p4_ns = t_p4 - t_p3

        # Build DNS reason for first failing check (priority order)
        dns_reason = None
        if not np.isfinite(v_p3) or v_p3 <= 0:
            dns_reason = f"DNS: P3 velocity ({v_p3:.2f} m/s) must be > 0 m/s"
        elif v_p4 <= v_p3:
            dns_reason = f"DNS: P4 ({v_p4:.2f}) <= P3 ({v_p3:.2f}) - no re-acceleration"
        elif v_p4 < v_p3 * min_recomp_velocity_ratio:
            dns_reason = f"DNS: Recompression too weak - v_p4 ({v_p4:.2f}) < v_p3*{min_recomp_velocity_ratio} ({v_p3 * min_recomp_velocity_ratio:.2f} m/s). Need clear spall rebound."
        elif t_p3_to_p4_ns < min_recomp_time_ns:
            dns_reason = f"DNS: Recompression too early - Δt={t_p3_to_p4_ns:.2f} ns < {min_recomp_time_ns} ns. Need sustained spall signature."
        elif pullback_mag > 1e-6 and rebound_mag < pullback_mag * min_recomp_ratio:
            req_rebound = pullback_mag * min_recomp_ratio
            dns_reason = f"DNS: Rebound too small - {rebound_mag:.2f} m/s < {min_recomp_ratio*100:.0f}% of pullback ({req_rebound:.2f} m/s). No clear spall signature."

        if dns_reason is not None:
            # Calculate intersections and fits even for DNS (for visualization)
            # P1: First plateau point
            t_p1 = t_first_plateau
            v_p1 = v_plateau_mean
            
            # P2: Last plateau point
            t_p2 = t_last_plateau
            v_p2 = v_plateau_mean
            
            # Line 4: From P3 to P4 (recompression)
            if abs(t_p4 - t_p3) > 1e-6:
                m4 = (v_p4 - v_p3) / (t_p4 - t_p3)
                c4 = v_p3 - m4 * t_p3
            else:
                m4 = 0.0
                c4 = v_p3
            
            # Line 5: From P4 to next minimum
            if abs(t_next_min - t_p4) > 1e-6:
                m5 = (v_next_min - v_p4) / (t_next_min - t_p4)
                c5 = v_p4 - m5 * t_p4
            else:
                m5 = 0.0
                c5 = v_p4
            
            # Construct fits and intersections for visualization
            # Line 1: From (0,0) to P1
            # Line 2: Horizontal plateau from P1 to P2
            # Line 3: From P2 to P3
            # Line 4: From P3 to P4
            # Line 5: From P4 to next minimum
            fits = {
                'seg1_rise': {'m': m1, 'c': c1, 't_range': (time[0], t_p1), 'start_idx': 0, 'end_idx': idx_first_plateau},
                'seg2_plateau': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p1, t_p2), 'start_idx': idx_first_plateau, 'end_idx': idx_last_plateau},
                'seg3_release': {'m': m3, 'c': c3, 't_range': (t_p2, t_p3), 'start_idx': idx_last_plateau, 'end_idx': idx_p3},
                'seg4_recomp': {'m': m4, 'c': c4, 't_range': (t_p3, t_p4), 'start_idx': idx_p3, 'end_idx': idx_p4},
                'seg5_tail': {'m': m5, 'c': c5, 't_range': (t_p4, t_next_min), 'start_idx': idx_p4, 'end_idx': idx_next_min}
            }
            
            intersections = [
                (t_p1, v_p1),            # P1: First plateau point
                (t_p2, v_p2),            # P2: Last plateau point
                (t_p3, v_p3),            # P3: First minimum after peak
                (t_p4, v_p4)             # P4: Global maximum after P3
            ]
            
            # Calculate strain rate from Line 3 (P2 to P3) even for DNS
            strain_rate = abs(m3) * 1e9 / (2.0 * acoustic_velocity) if m3 != 0 else 0.0  # e_dot = |du_fs/dt| / (2*c_b): free-surface factor of 2
            
            # Return DNS but include plateau velocity, shock stress, strain rate, and fits for visualization
            return False, dns_reason, {
                'Processing Status': 'DNS',
                'Plateau Mean Velocity (m/s)': v_plateau_mean,
                'Peak Shock Stress (GPa)': peak_shock_stress,
                'Peak Shock Stress Uncertainty (GPa)': 0.0,
                'First Maxima (m/s)': v_plateau_mean,
                'Minima (m/s)': v_p3,
                'Second Maxima (m/s)': v_p4,
                'Strain Rate (s^-1)': strain_rate,
                'Strain Rate Uncertainty (s^-1)': 0.0,
                'fits': fits,
                'intersections': intersections
            }
        
        # 8. Calculate Intersection Points
        # P1: First plateau point (where Line 1 meets plateau)
        t_p1 = t_first_plateau
        v_p1 = v_plateau_mean
        
        # P2: Last plateau point (where plateau meets Line 3)
        t_p2 = t_last_plateau
        v_p2 = v_plateau_mean
        
        # 10. Fit Line 4 (from P3 to P4) and Line 5 (from P4 to next minimum)
        # Line 4: From P3 to P4 (recompression)
        if abs(t_p4 - t_p3) > 1e-6:
            m4 = (v_p4 - v_p3) / (t_p4 - t_p3)
            c4 = v_p3 - m4 * t_p3
        else:
            m4 = 0.0
            c4 = v_p3
        
        # Line 5: From P4 to next minimum
        if abs(t_next_min - t_p4) > 1e-6:
            m5 = (v_next_min - v_p4) / (t_next_min - t_p4)
            c5 = v_p4 - m5 * t_p4
        else:
            m5 = 0.0
            c5 = v_p4
        
        print(f"  [HORIZ-PLAT] Line 4 (P3 to P4): m={m4:.2f}, c={c4:.2f}")
        print(f"  [HORIZ-PLAT] Line 5 (P4 to next min): m={m5:.2f}, c={c5:.2f}")
        sys.stdout.flush()
        
        # 11. Compile Results
        # Line 1: From (0,0) to P1
        # Line 2: Horizontal plateau from P1 to P2
        # Line 3: From P2 to P3
        # Line 4: From P3 to P4
        # Line 5: From P4 to next minimum
        fits = {
            'seg1_rise': {'m': m1, 'c': c1, 't_range': (time[0], t_p1), 'start_idx': 0, 'end_idx': idx_first_plateau},
            'seg2_plateau': {'m': 0.0, 'c': v_plateau_mean, 't_range': (t_p1, t_p2), 'start_idx': idx_first_plateau, 'end_idx': idx_last_plateau},
            'seg3_release': {'m': m3, 'c': c3, 't_range': (t_p2, t_p3), 'start_idx': idx_last_plateau, 'end_idx': idx_p3},
            'seg4_recomp': {'m': m4, 'c': c4, 't_range': (t_p3, t_p4), 'start_idx': idx_p3, 'end_idx': idx_p4},
            'seg5_tail': {'m': m5, 'c': c5, 't_range': (t_p4, t_next_min), 'start_idx': idx_p4, 'end_idx': idx_next_min}
        }
        
        intersections = [
            (t_p1, v_p1),            # P1: First plateau point
            (t_p2, v_p2),            # P2: Last plateau point
            (t_p3, v_p3),            # P3
            (t_p4, v_p4)             # P4
        ]
        
        # 9. Calculate Physics
        pullback_velocity = v_plateau_mean - v_p3
        spall_strength_gpa = 0.5 * density * acoustic_velocity * pullback_velocity / 1e9

        # Strain rate from pullback slope
        strain_rate = abs(m3) * 1e9 / (2.0 * acoustic_velocity) if m3 != 0 else 0.0  # e_dot = |du_fs/dt| / (2*c_b): free-surface factor of 2

        # Shock stress from plateau
        peak_shock_stress = density * acoustic_velocity * v_plateau_mean / 1e9

        # Uncertainty from velocity uncertainty array at P2 (plateau end) and P3 (minimum)
        plateau_unc = float(uncert[idx_last_plateau]) if uncert is not None and idx_last_plateau < len(uncert) and np.isfinite(uncert[idx_last_plateau]) else 0.0
        p3_unc = float(uncert[idx_p3]) if uncert is not None and idx_p3 < len(uncert) and np.isfinite(uncert[idx_p3]) else 0.0
        pullback_velocity_unc = np.sqrt(plateau_unc**2 + p3_unc**2)
        spall_strength_unc_gpa = 0.5 * density * acoustic_velocity * pullback_velocity_unc / 1e9
        delta_t_release_ns = abs(t_p3 - t_last_plateau)
        strain_rate_unc = pullback_velocity_unc * 1e9 / (2.0 * acoustic_velocity * delta_t_release_ns) if delta_t_release_ns > 0 else 0.0

        results = {
            'Processing Status': 'Success',
            'First Maxima (m/s)': v_plateau_mean,
            'Minima (m/s)': v_p3,
            'Second Maxima (m/s)': v_p4,
            'Pullback Velocity (m/s)': pullback_velocity,
            'Pullback Velocity Uncertainty (m/s)': pullback_velocity_unc,
            'Strain Rate (s^-1)': strain_rate,
            'Strain Rate Uncertainty (s^-1)': strain_rate_unc,
            'Spall Strength (GPa)': spall_strength_gpa,
            'Spall Strength Uncertainty (GPa)': spall_strength_unc_gpa,
            'Plateau Mean Velocity (m/s)': v_plateau_mean,
            'Peak Shock Stress (GPa)': peak_shock_stress,
            'Peak Shock Stress Uncertainty (GPa)': 0.0,
            'fits': fits,
            'intersections': intersections
        }
        
        return True, "Success: Horizontal plateau constraint", results
    
    def detect_hel_rdp(self, time, velocity, config):
        """
        Hybrid HEL Detection: RDP for segmentation (The Scout), Linear Regression for verification (The Verifier).
        
        This approach uses RDP to identify candidate corners, then goes back to raw data
        to fit lines and verify slopes using linear regression. This eliminates errors
        introduced by RDP epsilon parameter and handles ramping plateaus robustly.
        
        Parameters
        ----------
        time : numpy.ndarray
            Time array (ns)
        velocity : numpy.ndarray
            Velocity array (m/s)
        config : dict
            Configuration parameters
            
        Returns
        -------
        tuple
            (hel_found: bool, hel_results: dict or None)
        """
        if len(time) < 3 or len(velocity) < 3:
            return False, {"error": "Insufficient data points"}
        
        # Parameters
        epsilon = config.get('hel_rdp_epsilon', 3.0)
        slope_drop_ratio = config.get('hel_slope_drop_ratio', 0.2)
        min_duration = config.get('hel_min_plateau_duration', 2.0)
        angle_threshold_deg = config.get('hel_angle_threshold_deg', 30.0)
        # Minimum plateau velocity. Used as a 5th gating condition inside the search
        # loop so the detector does not lock onto noise-floor triplets (near v=0)
        # before the real elastic-plastic knee. Set to 0 to disable.
        min_hel_velocity = config.get('minimum_HEL_velocity_expected', 0.0)
        
        # Step 1: RDP Simplification (The Scout) - Get indices of simplified vertices
        rdp_indices = self.ramer_douglas_peucker_indices(time, velocity, epsilon)
        
        # Always create RDP points for visualization (even if HEL not detected)
        rdp_points = np.column_stack((time[rdp_indices], velocity[rdp_indices])) if len(rdp_indices) >= 2 else None
        
        # Calculate gradients for ALL RDP segments (for visualization even when HEL not detected)
        rdp_segment_gradients = []
        rdp_segment_angles = []
        if len(rdp_indices) >= 2:
            for j in range(len(rdp_indices) - 1):
                dt_seg = time[rdp_indices[j+1]] - time[rdp_indices[j]]
                dv_seg = velocity[rdp_indices[j+1]] - velocity[rdp_indices[j]]
                if dt_seg > 0:
                    grad_seg = dv_seg / dt_seg  # m/s per ns
                    angle_seg = np.degrees(np.arctan(np.abs(grad_seg)))
                else:
                    grad_seg = np.nan
                    angle_seg = np.nan
                rdp_segment_gradients.append(grad_seg)
                rdp_segment_angles.append(angle_seg)
        
        if len(rdp_indices) < 3:
            return False, {
                "error": "Trace too simple (linear) after RDP simplification", 
                "rdp_points": rdp_points,
                "rdp_segment_gradients": rdp_segment_gradients,
                "rdp_segment_angles": rdp_segment_angles
            }
        
        # Step 2: Iterate through candidate segments
        hel_found = False
        hel_results = None
        best_candidate = None   # Highest-scoring candidate for diagnostics
        best_score = -1

        # Precompute RDP vertex time/velocity once (used for fallback slopes & durations)
        rdp_times_all = time[rdp_indices]
        rdp_vels_all = velocity[rdp_indices]

        # Keep per-triplet diagnostics for the progress log (shown only when HEL not found)
        triplet_diag_messages = []
        rdp_points_for_candidate = np.column_stack((rdp_times_all, rdp_vels_all))

        def _safe_polyfit(t_seg, v_seg):
            """Return (slope, intercept, ok) — suppress polyfit warnings, detect NaN results."""
            if len(t_seg) < 2 or len(v_seg) < 2:
                return np.nan, np.nan, False
            try:
                with np.errstate(all='ignore'):
                    import warnings as _warnings
                    with _warnings.catch_warnings():
                        _warnings.simplefilter('ignore')
                        m, c = np.polyfit(t_seg, v_seg, 1)
                if not (np.isfinite(m) and np.isfinite(c)):
                    return np.nan, np.nan, False
                return float(m), float(c), True
            except (np.linalg.LinAlgError, ValueError, TypeError):
                return np.nan, np.nan, False

        for i in range(len(rdp_indices) - 2):
            # Indices for the start, knee, and end of the potential HEL sequence
            idx_start = int(rdp_indices[i])
            idx_knee = int(rdp_indices[i + 1])   # Potential HEL point
            idx_end = int(rdp_indices[i + 2])

            # Step 3: Extract raw data segments
            t_rise = time[idx_start : idx_knee + 1]
            v_rise = velocity[idx_start : idx_knee + 1]
            t_plat = time[idx_knee : idx_end + 1]
            v_plat = velocity[idx_knee : idx_end + 1]

            if len(t_plat) < 2:
                triplet_diag_messages.append(
                    f"    [HEL] triplet i={i}: plateau has <2 raw samples -> skipped"
                )
                continue
            duration_plat = float(t_plat[-1] - t_plat[0])  # raw-data duration (reference)

            # RDP segment duration (the value that is actually checked)
            duration_plat_rdp = float(rdp_times_all[i + 2] - rdp_times_all[i + 1])

            if not np.isfinite(duration_plat_rdp) or duration_plat_rdp < min_duration:
                triplet_diag_messages.append(
                    f"    [HEL] triplet i={i}: plateau duration {duration_plat_rdp:.2f} ns "
                    f"< min {min_duration:.2f} ns -> skipped"
                )
                continue

            # Step 4: Raw-data linear fits for rise and plateau. If either fit fails,
            # fall back to the RDP segment slope so we still get a usable slope (and a
            # full candidate_info) for diagnostics instead of discarding the triplet.
            m_rise_fit, c_rise_fit, rise_fit_ok = _safe_polyfit(t_rise, v_rise)
            m_plat_fit, c_plat_fit, plat_fit_ok = _safe_polyfit(t_plat, v_plat)

            # Step 5: RDP segment slopes (always finite when dt > 0)
            dt_rise_rdp = float(rdp_times_all[i + 1] - rdp_times_all[i])
            dv_rise_rdp = float(rdp_vels_all[i + 1] - rdp_vels_all[i])
            if dt_rise_rdp > 0:
                gradient_rise_rdp = dv_rise_rdp / dt_rise_rdp
                angle_rise_rdp = float(np.degrees(np.arctan(np.abs(gradient_rise_rdp))))
            else:
                gradient_rise_rdp = np.nan
                angle_rise_rdp = np.nan

            dt_plat_rdp = float(rdp_times_all[i + 2] - rdp_times_all[i + 1])
            dv_plat_rdp = float(rdp_vels_all[i + 2] - rdp_vels_all[i + 1])
            if dt_plat_rdp > 0:
                gradient_plat_rdp = dv_plat_rdp / dt_plat_rdp
                angle_plat_rdp = float(np.degrees(np.arctan(np.abs(gradient_plat_rdp))))
            else:
                gradient_plat_rdp = np.nan
                angle_plat_rdp = np.nan

            # Effective slopes used for rule checks: prefer polyfit, fall back to RDP slope
            if rise_fit_ok:
                m_rise = m_rise_fit
                rise_slope_src = 'polyfit'
            else:
                m_rise = gradient_rise_rdp
                rise_slope_src = 'rdp_fallback'

            if plat_fit_ok:
                m_plat = m_plat_fit
                plat_slope_src = 'polyfit'
            else:
                m_plat = gradient_plat_rdp
                plat_slope_src = 'rdp_fallback'

            # Intercept for the line equations (only meaningful if fit succeeded)
            c_rise = c_rise_fit if rise_fit_ok else np.nan
            c_plat = c_plat_fit if plat_fit_ok else np.nan

            # Step 6: Condition checks (always run; slopes guaranteed to be numbers or NaN)
            v_plat_mean = float(np.mean(v_plat)) if len(v_plat) > 0 else np.nan

            m_rise_finite = np.isfinite(m_rise)
            m_plat_finite = np.isfinite(m_plat)

            rise_slope_ok = bool(m_rise_finite and m_rise > 0)
            if rise_slope_ok and m_plat_finite:
                plateau_slope_ok = bool(m_plat < m_rise * slope_drop_ratio)
            else:
                plateau_slope_ok = False

            rdp_angle_ok = bool(np.isfinite(angle_plat_rdp) and angle_plat_rdp <= angle_threshold_deg)
            duration_ok = bool(np.isfinite(duration_plat_rdp) and duration_plat_rdp >= min_duration)
            velocity_ok = bool(
                (min_hel_velocity <= 0.0) or
                (np.isfinite(v_plat_mean) and abs(v_plat_mean) >= min_hel_velocity)
            )

            conditions_met = {
                'duration_ok': duration_ok,
                'rise_slope_ok': rise_slope_ok,
                'plateau_slope_ok': plateau_slope_ok,
                'rdp_angle_ok': rdp_angle_ok,
                'velocity_ok': velocity_ok,
            }
            score = sum(1 for v in conditions_met.values() if v)

            # Always build a full candidate_info with whatever values we have
            candidate_info = {
                'hel_time_detection': float(time[idx_knee]),
                'free_surface_velocity': v_plat_mean,
                'hel_segment_end_time': float(time[idx_end]),
                'hel_segment_start_time': float(time[idx_start]),
                'rise_slope': m_rise,
                'plateau_slope': m_plat,
                'rise_slope_source': rise_slope_src,
                'plateau_slope_source': plat_slope_src,
                'rise_slope_polyfit_ok': rise_fit_ok,
                'plateau_slope_polyfit_ok': plat_fit_ok,
                'rise_intercept': c_rise,
                'plateau_intercept': c_plat,
                'rdp_vertex_index': i + 1,
                'hel_segment_start_idx': idx_start,
                'hel_segment_knee_idx': idx_knee,
                'hel_segment_end_idx': idx_end,
                't_rise': t_rise,
                'v_rise': v_rise,
                't_plat': t_plat,
                'v_plat': v_plat,
                'duration_plat': duration_plat,
                'duration_plat_rdp': duration_plat_rdp,
                'rdp_points': rdp_points_for_candidate,
                'rdp_gradient_rise': gradient_rise_rdp,
                'rdp_gradient_plateau': gradient_plat_rdp,
                'rdp_angle_rise': angle_rise_rdp,
                'rdp_angle_plateau': angle_plat_rdp,
                'rdp_segment_gradients': rdp_segment_gradients,
                'rdp_segment_angles': rdp_segment_angles,
                'score': score,
                'conditions_met': conditions_met,
            }

            # Record per-triplet diagnostic line (shown on failure)
            fail_reasons = [name.replace('_ok', '') for name, ok in conditions_met.items() if not ok]
            triplet_diag_messages.append(
                f"    [HEL] triplet i={i}: m_rise={m_rise:.2f} ({rise_slope_src}), "
                f"m_plat={m_plat:.2f} ({plat_slope_src}), "
                f"plateau_angle={angle_plat_rdp:.1f}°, "
                f"plateau_dur={duration_plat_rdp:.2f} ns, "
                f"v_plat_mean={v_plat_mean:.1f} m/s, score={score}/5"
                + (f", fail=[{', '.join(fail_reasons)}]" if fail_reasons else ", PASS")
            )

            # Track best candidate for diagnostics
            if score > best_score:
                best_score = score
                best_candidate = candidate_info

            # Short-circuit: all five rules must pass to accept
            if not (rise_slope_ok and plateau_slope_ok and duration_ok and
                    rdp_angle_ok and velocity_ok):
                continue

            hel_found = True
            hel_results = candidate_info
            break  # Stop at the first valid elastic-plastic transition

        # If HEL not found, return best candidate for diagnostics
        if not hel_found:
            if best_candidate is not None:
                hel_results = best_candidate.copy()
                hel_results['error'] = 'No valid elastic-plastic transition found'
            else:
                hel_results = {
                    'error': 'No valid elastic-plastic transition found',
                    'rdp_points': rdp_points,
                    'rdp_segment_gradients': rdp_segment_gradients,
                    'rdp_segment_angles': rdp_segment_angles,
                }
            if triplet_diag_messages:
                hel_results['triplet_diagnostics'] = triplet_diag_messages

        return hel_found, hel_results

    def elastic_shock_strain_rate(self, C_L, U_hel, U_0, t_hel, t_0):
        """
        Compute elastic shock strain rate.

        Parameters
        ----------
        C_L : float
            Longitudinal wave velocity of the material (m/s).
        U_hel : float
            Free surface velocity at HEL (m/s).
        U_0 : float
            Free surface velocity at t = 0 (m/s).
        t_hel : float
            Time at which U_hel is measured (s).
        t_0 : float
            Initial time (s).

        Returns
        -------
        float
            Elastic shock strain rate (1/s).
        """
        dU = U_hel - U_0
        dt = t_hel - t_0
        
        if dt <= 0:
            return np.nan
        
        return (1 / (2 * C_L)) * (dU / dt)

    def _plot_individual_hel_detection(self, base_name, time_aligned, velocity_filtered,
                                       hel_start, hel_end, hel_time_clean, hel_velocity_clean,
                                       hel_strength, hel_uncertainty, sample_material,
                                       spade_output_dir, gradient=None, angles_deg=None,
                                       hel_segment_start=None, hel_segment_end=None,
                                       free_surface_velocity=None, angle_thresh_deg=None,
                                       U_0=None, t_0=None, t_hel=None, hel_ok=True, rdp_points=None,
                                       rise_slope_fit=None, plateau_slope_fit=None,
                                       t_rise=None, v_rise=None, t_plat=None, v_plat=None,
                                       rise_intercept=None, plateau_intercept=None,
                                       rdp_gradient_rise=None, rdp_gradient_plateau=None,
                                       rdp_angle_rise=None, rdp_angle_plateau=None,
                                       rdp_segment_gradients=None, rdp_segment_angles=None,
                                       slope_drop_ratio=0.2, min_plateau_duration=2.0,
                                       duration_plat_rdp=None, min_hel_velocity=None, hel_strain_rate=None):
        """Generate individual HEL detection plot showing detection results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Create figure with four subplots (added diagnostic table subplot)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        
        # Top subplot: Full velocity trace with HEL window highlighted (limited to 60 ns)
        # Filter data to only show up to 60 ns
        time_mask = time_aligned <= 60.0
        time_plot = time_aligned[time_mask]
        velocity_plot = velocity_filtered[time_mask]
        
        ax1.plot(time_plot, velocity_plot, 'b-', linewidth=1.5, alpha=0.7, label='Velocity')
        
        # Highlight HEL detection window (only if within 60 ns range)
        if hel_start <= 60.0:
            hel_end_plot = min(hel_end, 60.0) if hel_end is not None else 60.0
            ax1.axvspan(hel_start, hel_end_plot, alpha=0.2, color='yellow', label='HEL Window')
            if hel_end_plot < 60.0:
                ax1.axvline(hel_end_plot, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axvline(hel_start, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'HEL Detection - {base_name}\nMaterial: {sample_material}', fontsize=14, fontweight='bold')
        ax1.set_xlim([min(time_plot) if len(time_plot) > 0 else 0, 60.0])  # Limit x-axis to 60 ns
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Middle subplot: Zoomed HEL window with velocity and detection overlays
        if len(hel_time_clean) > 0 and len(hel_velocity_clean) > 0:
            ax2.plot(hel_time_clean, hel_velocity_clean, 'b-', linewidth=2, alpha=0.6, label='Velocity in HEL window')
        else:
            # No data in HEL window
            ax2.text(0.5, 0.5, 'No data in HEL window', transform=ax2.transAxes,
                    fontsize=12, horizontalalignment='center', verticalalignment='center',
                    style='italic', color='gray')
        
        # RDP+Linear Hybrid Visualization
        if rdp_points is not None and len(rdp_points) > 0:
            rdp_times = rdp_points[:, 0]
            rdp_velocities = rdp_points[:, 1]
            
            # Plot RDP simplified line
            ax2.plot(rdp_times, rdp_velocities, 'r-', linewidth=2.5, alpha=0.9, 
                    label='RDP Simplified', zorder=3)
            
            # Mark all RDP vertices
            ax2.plot(rdp_times, rdp_velocities, 'ro', markersize=6, alpha=0.7, 
                    label='RDP Vertices', zorder=4)
            
            # Highlight HEL detection point if found
            if hel_ok and t_hel is not None and np.isfinite(t_hel):
                # Find the RDP vertex closest to HEL detection time
                hel_rdp_idx = np.argmin(np.abs(rdp_times - t_hel))
                if hel_rdp_idx < len(rdp_times):
                    ax2.plot(rdp_times[hel_rdp_idx], rdp_velocities[hel_rdp_idx], 
                            'g*', markersize=15, markeredgecolor='darkgreen', 
                            markeredgewidth=2, label='HEL Detection Point', zorder=5)
            
            # Only show RDP segment annotations if HEL was detected (simplified - remove individual segment annotations)
            # Keep only the key annotations: RDP Rise and RDP Plateau (if HEL detected)
            if hel_ok and rdp_gradient_rise is not None and np.isfinite(rdp_gradient_rise):
                # Find the rise segment (from first to second RDP vertex, or to HEL point)
                if len(rdp_times) >= 2:
                    if hel_ok and t_hel is not None:
                        # Find indices for rise segment
                        hel_rdp_idx = np.argmin(np.abs(rdp_times - t_hel))
                        if hel_rdp_idx > 0:
                            # Rise segment: from vertex before HEL to HEL
                            t_rise_mid = (rdp_times[hel_rdp_idx-1] + rdp_times[hel_rdp_idx]) / 2
                            v_rise_mid = (rdp_velocities[hel_rdp_idx-1] + rdp_velocities[hel_rdp_idx]) / 2
                        else:
                            t_rise_mid = (rdp_times[0] + rdp_times[1]) / 2 if len(rdp_times) > 1 else rdp_times[0]
                            v_rise_mid = (rdp_velocities[0] + rdp_velocities[1]) / 2 if len(rdp_velocities) > 1 else rdp_velocities[0]
                    else:
                        t_rise_mid = (rdp_times[0] + rdp_times[1]) / 2 if len(rdp_times) > 1 else rdp_times[0]
                        v_rise_mid = (rdp_velocities[0] + rdp_velocities[1]) / 2 if len(rdp_velocities) > 1 else rdp_velocities[0]
                    
                    # Annotate RDP rise gradient (simplified - no angle text to reduce clutter)
                    ax2.annotate(f'RDP Rise: {rdp_gradient_rise:.2f} m/s/ns', 
                               xy=(t_rise_mid, v_rise_mid),
                               xytext=(5, 15), textcoords='offset points',
                               fontsize=8, color='darkred', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='darkred', linewidth=1.5))
            
            if hel_ok and rdp_gradient_plateau is not None and np.isfinite(rdp_gradient_plateau):
                # Find the plateau segment (from HEL point to next RDP vertex)
                if len(rdp_times) >= 2:
                    if hel_ok and t_hel is not None:
                        # Find indices for plateau segment
                        hel_rdp_idx = np.argmin(np.abs(rdp_times - t_hel))
                        if hel_rdp_idx < len(rdp_times) - 1:
                            # Plateau segment: from HEL to next vertex
                            t_plat_mid = (rdp_times[hel_rdp_idx] + rdp_times[hel_rdp_idx+1]) / 2
                            v_plat_mid = (rdp_velocities[hel_rdp_idx] + rdp_velocities[hel_rdp_idx+1]) / 2
                        else:
                            t_plat_mid = (rdp_times[-2] + rdp_times[-1]) / 2 if len(rdp_times) > 1 else rdp_times[-1]
                            v_plat_mid = (rdp_velocities[-2] + rdp_velocities[-1]) / 2 if len(rdp_velocities) > 1 else rdp_velocities[-1]
                    else:
                        t_plat_mid = (rdp_times[-2] + rdp_times[-1]) / 2 if len(rdp_times) > 1 else rdp_times[-1]
                        v_plat_mid = (rdp_velocities[-2] + rdp_velocities[-1]) / 2 if len(rdp_velocities) > 1 else rdp_velocities[-1]
                    
                    # Annotate RDP plateau gradient (simplified - show angle and threshold status)
                    angle_text = f" ({rdp_angle_plateau:.1f}°)" if rdp_angle_plateau is not None and np.isfinite(rdp_angle_plateau) else ""
                    threshold_text = ""
                    if angle_thresh_deg is not None and rdp_angle_plateau is not None and np.isfinite(rdp_angle_plateau):
                        if rdp_angle_plateau <= angle_thresh_deg:
                            threshold_text = f" ✓"
                        else:
                            threshold_text = f" ✗"
                    
                    ax2.annotate(f'RDP Plateau: {rdp_gradient_plateau:.2f} m/s/ns{angle_text}{threshold_text}', 
                               xy=(t_plat_mid, v_plat_mid),
                               xytext=(5, -20), textcoords='offset points',
                               fontsize=8, color='darkred', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='darkred', linewidth=1.5))
        
        # Linear Regression Fits on Raw Data (Hybrid Method - RDP+Linear)
        if (t_rise is not None and v_rise is not None and len(t_rise) > 0 and 
            rise_slope_fit is not None and np.isfinite(rise_slope_fit)):
            # Plot raw rise segment data points
            ax2.plot(t_rise, v_rise, 'c.', markersize=4, alpha=0.5, 
                    label='Rise Segment (raw)', zorder=2)
            
            # Plot linear fit on rise segment
            if len(t_rise) > 1:
                # Use actual intercept if available, otherwise calculate from first point
                if rise_intercept is not None and np.isfinite(rise_intercept):
                    c_rise = rise_intercept
                else:
                    c_rise = v_rise[0] - rise_slope_fit * t_rise[0]  # Approximate intercept
                
                v_rise_fit = rise_slope_fit * t_rise + c_rise
                ax2.plot(t_rise, v_rise_fit, 'c--', linewidth=2.5, alpha=0.9, 
                        label=f'Rise Linear Fit: {rise_slope_fit:.2f} m/s/ns', zorder=3)
                
                # Remove annotation for rise slope (reduce clutter - slope is already in legend)
        
        if (t_plat is not None and v_plat is not None and len(t_plat) > 0 and 
            plateau_slope_fit is not None and np.isfinite(plateau_slope_fit)):
            # Plot raw plateau segment data points
            ax2.plot(t_plat, v_plat, 'm.', markersize=4, alpha=0.5, 
                    label='Plateau Segment (raw)', zorder=2)
            
            # Plot linear fit on plateau segment
            if len(t_plat) > 1:
                # Use actual intercept if available, otherwise calculate from first point
                if plateau_intercept is not None and np.isfinite(plateau_intercept):
                    c_plat = plateau_intercept
                else:
                    c_plat = v_plat[0] - plateau_slope_fit * t_plat[0]  # Approximate intercept
                
                v_plat_fit = plateau_slope_fit * t_plat + c_plat
                ax2.plot(t_plat, v_plat_fit, 'm--', linewidth=2.5, alpha=0.9, 
                        label=f'Plateau Linear Fit: {plateau_slope_fit:.2f} m/s/ns', zorder=3)
                
                # Remove annotation for plateau slope (reduce clutter - slope is already in legend)
        
        # Highlight HEL plateau region: only show from HEL detection point to plateau end
        # (not from hel_segment_start which might be earlier)
        if hel_ok and t_hel is not None and np.isfinite(t_hel) and free_surface_velocity is not None:
            # Use HEL detection time as plateau start (not hel_segment_start)
            plateau_start_time = t_hel
            
            # Find plateau end: use hel_segment_end if available, otherwise use end of plateau data
            if hel_segment_end is not None and hel_segment_end < len(hel_time_clean):
                plateau_end_time = hel_time_clean[hel_segment_end]
            elif t_plat is not None and len(t_plat) > 0:
                # Use end of plateau raw data segment
                plateau_end_time = t_plat[-1]
            else:
                # Fallback: use hel_segment_end if available
                plateau_end_time = hel_time_clean[hel_segment_end] if hel_segment_end is not None and hel_segment_end < len(hel_time_clean) else t_hel + 10.0
            
            # Only show plateau region if it's valid
            if plateau_end_time > plateau_start_time:
                ax2.axvspan(plateau_start_time, plateau_end_time, alpha=0.15, color='orange', 
                           label=f'HEL Plateau Region', zorder=1)
            ax2.axhline(free_surface_velocity, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                           label=f'Mean Plateau Velocity: {free_surface_velocity:.1f} m/s', zorder=2)
            # Mark segment boundaries
            ax2.axvline(plateau_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
            ax2.axvline(plateau_end_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)
        
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title(f'HEL Window Detail - Velocity (RDP+Linear Hybrid Method)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        # Simplify legend - only show key elements
        handles, labels = ax2.get_legend_handles_labels()
        # Keep only essential legend entries: RDP Simplified, HEL Detection Point, Rise/Plateau fits, HEL Plateau Region, Mean Plateau Velocity
        essential_labels = ['RDP Simplified', 'HEL Detection Point', 'Rise Linear Fit', 'Plateau Linear Fit', 'HEL Plateau Region', 'Mean Plateau Velocity']
        filtered_handles = []
        filtered_labels = []
        for handle, label in zip(handles, labels):
            # Keep if it's an essential label or if it's not a duplicate
            if any(essential in label for essential in essential_labels):
                # Check if we already have this type
                label_type = None
                if 'Rise Linear Fit' in label:
                    label_type = 'Rise'
                elif 'Plateau Linear Fit' in label:
                    label_type = 'Plateau'
                elif 'RDP Simplified' in label:
                    label_type = 'RDP'
                elif 'HEL Detection Point' in label:
                    label_type = 'HEL'
                elif 'HEL Plateau Region' in label:
                    label_type = 'PlateauRegion'
                elif 'Mean Plateau Velocity' in label:
                    label_type = 'MeanVel'
                
                if label_type is None or not any(label_type in existing for existing in filtered_labels):
                    filtered_handles.append(handle)
                    filtered_labels.append(label)
        
        if len(filtered_handles) > 0:
            ax2.legend(filtered_handles, filtered_labels, loc='best', fontsize=9, ncol=2)
        
        # Bottom subplot: Gradient vs Time (RDP and Raw gradients)
        # Plot RDP gradient (piecewise constant between RDP vertices)
        if rdp_points is not None and len(rdp_points) > 1 and len(hel_time_clean) > 0:
            rdp_times_grad = rdp_points[:, 0]
            rdp_velocities_grad = rdp_points[:, 1]
            
            # Create piecewise constant gradient for RDP line
            rdp_gradient_time = []
            rdp_gradient_value = []
            
            for seg_idx in range(len(rdp_times_grad) - 1):
                t_start = rdp_times_grad[seg_idx]
                t_end = rdp_times_grad[seg_idx + 1]
                v_start = rdp_velocities_grad[seg_idx]
                v_end = rdp_velocities_grad[seg_idx + 1]
                
                if t_end > t_start:
                    grad_seg = (v_end - v_start) / (t_end - t_start)
                    # Add points at start and end of segment (for step plot)
                    rdp_gradient_time.extend([t_start, t_end])
                    rdp_gradient_value.extend([grad_seg, grad_seg])
            
            if len(rdp_gradient_time) > 0:
                # Plot as step function to show piecewise constant nature
                ax3.plot(rdp_gradient_time, rdp_gradient_value, 'r-', linewidth=2.5, alpha=0.9, 
                        label='RDP Gradient', zorder=3)
                # Mark segment boundaries
                for t_seg in rdp_times_grad[1:-1]:  # Skip first and last
                    ax3.axvline(t_seg, color='red', linestyle=':', linewidth=1, alpha=0.5)
        
        # Plot raw gradient (for reference)
        if gradient is not None and len(gradient) > 0 and len(hel_time_clean) == len(gradient):
            ax3.plot(hel_time_clean, gradient, 'g-', linewidth=1, alpha=0.4, label='Raw Gradient (dv/dt)', zorder=1)
            ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Highlight HEL plateau region in gradient plot: only from HEL detection point to plateau end
        if hel_ok and t_hel is not None and np.isfinite(t_hel):
            # Use HEL detection time as plateau start (not hel_segment_start)
            plateau_start_time = t_hel
            
            # Find plateau end: use hel_segment_end if available, otherwise use end of plateau data
            if hel_segment_end is not None and hel_segment_end < len(hel_time_clean):
                plateau_end_time = hel_time_clean[hel_segment_end]
            elif t_plat is not None and len(t_plat) > 0:
                # Use end of plateau raw data segment
                plateau_end_time = t_plat[-1]
            else:
                # Fallback: use hel_segment_end if available
                plateau_end_time = hel_time_clean[hel_segment_end] if hel_segment_end is not None and hel_segment_end < len(hel_time_clean) else t_hel + 10.0
            
            # Only show plateau region if it's valid
            if plateau_end_time > plateau_start_time:
                ax3.axvspan(plateau_start_time, plateau_end_time, alpha=0.3, color='orange', 
                           label='HEL Plateau Region')
            ax3.axvline(plateau_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax3.axvline(plateau_end_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Compute sensible y-limits from real gradient data so the threshold line
        # (which blows up to ~1e16 near 90°) doesn't flatten the plot to zero.
        _grad_vals = []
        if rdp_points is not None and len(rdp_points) > 1:
            rdp_t = rdp_points[:, 0]
            rdp_v = rdp_points[:, 1]
            for _i in range(len(rdp_t) - 1):
                if rdp_t[_i + 1] > rdp_t[_i]:
                    _grad_vals.append((rdp_v[_i + 1] - rdp_v[_i]) / (rdp_t[_i + 1] - rdp_t[_i]))
        if gradient is not None and len(gradient) > 0:
            _g = np.asarray(gradient)
            _g = _g[np.isfinite(_g)]
            if _g.size:
                _grad_vals.extend(_g.tolist())
        if _grad_vals:
            _g_arr = np.asarray(_grad_vals, dtype=float)
            _g_arr = _g_arr[np.isfinite(_g_arr)]
            if _g_arr.size:
                _g_max = float(np.nanmax(np.abs(_g_arr)))
                _pad = max(_g_max * 1.3, 5.0)  # at least ±5 m/s/ns so near-flat traces still render
                ax3.set_ylim(-_pad, _pad)
                _y_lim_abs = _pad
            else:
                _y_lim_abs = None
        else:
            _y_lim_abs = None

        # Plot angle threshold line
        if angle_thresh_deg is not None:
            # Convert angle threshold to gradient (slope). tan(90°) is effectively infinite
            # so clamp the angle used for plotting to keep the threshold line on-scale.
            _ang_for_plot = min(float(angle_thresh_deg), 89.0)
            angle_thresh_rad = np.radians(_ang_for_plot)
            gradient_thresh = np.tan(angle_thresh_rad)
            # Only draw the horizontal threshold lines when they fit inside the
            # visible y-range; otherwise they just rescale the axis to ~1e16.
            _draw_thresh = _y_lim_abs is None or gradient_thresh <= 5.0 * _y_lim_abs
            if _draw_thresh and np.isfinite(gradient_thresh):
                ax3.axhline(gradient_thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                           label=f'Angle Threshold ({angle_thresh_deg}°)', zorder=2)
                ax3.axhline(-gradient_thresh, color='red', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)
            else:
                # Threshold is effectively vertical (≥89°) – note it instead of drawing it.
                ax3.text(0.02, 0.95,
                         f'Angle Threshold {angle_thresh_deg}° '
                         f'(|slope| ≥ {gradient_thresh:.1e} — off-scale)',
                         transform=ax3.transAxes, fontsize=9, color='red',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='red', alpha=0.8))

            # Annotate RDP segments that pass/fail threshold
            if rdp_segment_gradients is not None and rdp_segment_angles is not None and rdp_points is not None:
                rdp_times_annot = rdp_points[:, 0]
                for seg_idx in range(len(rdp_segment_gradients)):
                    if seg_idx < len(rdp_times_annot) - 1 and np.isfinite(rdp_segment_gradients[seg_idx]):
                        t_seg_mid = (rdp_times_annot[seg_idx] + rdp_times_annot[seg_idx + 1]) / 2
                        grad_seg = rdp_segment_gradients[seg_idx]
                        angle_seg = rdp_segment_angles[seg_idx] if seg_idx < len(rdp_segment_angles) else np.nan
                        
                        # Check threshold
                        if np.isfinite(angle_seg):
                            if angle_seg <= angle_thresh_deg:
                                marker = '✓'
                                color_annot = 'green'
                            else:
                                marker = '✗'
                                color_annot = 'orange'
                            
                            # Annotate on gradient plot
                            ax3.plot(t_seg_mid, grad_seg, 'o', color=color_annot, markersize=8, 
                                    zorder=4, markeredgecolor='black', markeredgewidth=1)
                            ax3.annotate(f'Seg{seg_idx}\n{marker}', 
                                       xy=(t_seg_mid, grad_seg),
                                       xytext=(0, 10), textcoords='offset points',
                                       fontsize=7, color=color_annot, weight='bold',
                                       ha='center', va='bottom',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color_annot, 
                                               alpha=0.2, edgecolor=color_annot))
        else:
            # No gradient data available
            if rdp_points is None or len(rdp_points) < 2:
                ax3.text(0.5, 0.5, 'No gradient data available', transform=ax3.transAxes,
                        fontsize=12, horizontalalignment='center', verticalalignment='center',
                        style='italic', color='gray')
        
        ax3.set_xlabel('Time (ns)', fontsize=12)
        ax3.set_ylabel('Gradient (m/s per ns)', fontsize=12)
        ax3.set_title(f'Gradient vs Time (Reference Only - Not Used for Detection)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        # Fourth subplot: Diagnostic table showing HEL detection conditions
        ax4.axis('off')  # Turn off axes for table display
        
        # Prepare diagnostic data with detailed explanations
        conditions = []
        statuses = []
        explanations = []
        
        # Condition 1: RDP plateau segment angle threshold check.
        # Report the PLATEAU angle of the chosen candidate (matches detector logic),
        # not the minimum angle across all RDP segments (which was misleading).
        if angle_thresh_deg is not None:
            conditions.append(f'RDP plateau angle ≤ {angle_thresh_deg:.0f}°')
            if rdp_angle_plateau is not None and np.isfinite(rdp_angle_plateau):
                if rdp_angle_plateau <= angle_thresh_deg:
                    statuses.append('✓')
                    explanations.append(f'Candidate plateau angle: {rdp_angle_plateau:.1f}° (passes threshold)')
                else:
                    statuses.append('✗')
                    explanations.append(f'Candidate plateau angle: {rdp_angle_plateau:.1f}° > {angle_thresh_deg:.0f}° (fails threshold)')
            else:
                # No candidate triplet was available (trace too simple, etc.)
                statuses.append('?')
                if rdp_segment_angles is not None and len(rdp_segment_angles) > 0:
                    finite_angles = [a for a in rdp_segment_angles if np.isfinite(a)]
                    if finite_angles:
                        explanations.append(f'No candidate triplet; segment angle range: {min(finite_angles):.1f}°–{max(finite_angles):.1f}°')
                    else:
                        explanations.append('No candidate triplet found')
                else:
                    explanations.append('No candidate triplet found')
        
        # Condition 2: Rise slope > 0 (from raw data fit)
        if rise_slope_fit is not None and np.isfinite(rise_slope_fit):
            conditions.append('Rise slope > 0 (from raw data fit)')
            if rise_slope_fit > 0:
                statuses.append('✓')
                explanations.append(f'Rise slope: {rise_slope_fit:.2f} m/s/ns (positive)')
            else:
                statuses.append('✗')
                explanations.append(f'Rise slope: {rise_slope_fit:.2f} m/s/ns (not positive)')
        else:
            conditions.append('Rise slope > 0 (from raw data fit)')
            statuses.append('?')
            explanations.append('No candidate triplet found to test')
        
        # Condition 3: Plateau slope < rise × slope_drop_ratio
        slope_drop_pct = slope_drop_ratio * 100
        if (rise_slope_fit is not None and plateau_slope_fit is not None and 
            np.isfinite(rise_slope_fit) and np.isfinite(plateau_slope_fit) and rise_slope_fit > 0):
            threshold_slope = rise_slope_fit * slope_drop_ratio
            conditions.append(f'Plateau slope < rise × {slope_drop_pct:.0f}%')
            if plateau_slope_fit < threshold_slope:
                statuses.append('✓')
                explanations.append(f'Plateau: {plateau_slope_fit:.2f} < {threshold_slope:.2f} m/s/ns (passes)')
            else:
                statuses.append('✗')
                explanations.append(f'Plateau: {plateau_slope_fit:.2f} ≥ {threshold_slope:.2f} m/s/ns (fails)')
        else:
            conditions.append(f'Plateau slope < rise × {slope_drop_pct:.0f}%')
            statuses.append('?')
            if rise_slope_fit is None or not np.isfinite(rise_slope_fit):
                explanations.append('Rise slope not available')
            elif plateau_slope_fit is None or not np.isfinite(plateau_slope_fit):
                explanations.append('Plateau slope not available')
            else:
                explanations.append('No valid candidate triplet found')
        
        # Condition 4: Plateau duration ≥ min_duration (using RDP segment duration)
        # Get RDP segment duration (this is what we check)
        duration_rdp = duration_plat_rdp  # Use parameter passed to function
        
        # Also get raw data duration for reference
        duration_raw = None
        if t_plat is not None and len(t_plat) > 1:
            duration_raw = t_plat[-1] - t_plat[0]
        
        if duration_rdp is not None and np.isfinite(duration_rdp):
            conditions.append(f'Plateau duration (RDP segment) ≥ {min_plateau_duration:.1f} ns')
            if duration_rdp >= min_plateau_duration:
                statuses.append('✓')
                if duration_raw is not None and np.isfinite(duration_raw):
                    explanations.append(f'RDP segment: {duration_rdp:.2f} ns (raw data: {duration_raw:.2f} ns) - meets minimum')
                else:
                    explanations.append(f'RDP segment: {duration_rdp:.2f} ns (meets minimum)')
            else:
                statuses.append('✗')
                if duration_raw is not None and np.isfinite(duration_raw):
                    explanations.append(f'RDP segment: {duration_rdp:.2f} ns < {min_plateau_duration:.1f} ns (raw data: {duration_raw:.2f} ns)')
                else:
                    explanations.append(f'RDP segment: {duration_rdp:.2f} ns < {min_plateau_duration:.1f} ns (too short)')
        else:
            conditions.append(f'Plateau duration (RDP segment) ≥ {min_plateau_duration:.1f} ns')
            statuses.append('?')
            explanations.append('No candidate plateau segment found (RDP duration not available)')
        
        # Condition 5: Minimum HEL velocity check
        if min_hel_velocity is not None and free_surface_velocity is not None:
            conditions.append(f'HEL velocity ≥ {min_hel_velocity:.1f} m/s')
            if np.isfinite(free_surface_velocity) and abs(free_surface_velocity) >= min_hel_velocity:
                statuses.append('✓')
                explanations.append(f'HEL velocity: {abs(free_surface_velocity):.2f} m/s (meets minimum)')
            else:
                statuses.append('✗')
                if np.isfinite(free_surface_velocity):
                    explanations.append(f'HEL velocity: {abs(free_surface_velocity):.2f} m/s < {min_hel_velocity:.1f} m/s (too low)')
                else:
                    explanations.append(f'HEL velocity: NaN (invalid)')
        else:
            conditions.append('HEL velocity ≥ threshold')
            statuses.append('?')
            explanations.append('Velocity threshold check not available')
        
        # Condition 6: Strain rate must be positive
        if hel_strain_rate is not None:
            conditions.append('Strain rate ≥ 0')
            if np.isfinite(hel_strain_rate) and hel_strain_rate >= 0:
                statuses.append('✓')
                explanations.append(f'Strain rate: {hel_strain_rate:.2e} s⁻¹ (positive)')
            else:
                statuses.append('✗')
                if np.isfinite(hel_strain_rate):
                    explanations.append(f'Strain rate: {hel_strain_rate:.2e} s⁻¹ (negative - invalid)')
                else:
                    explanations.append(f'Strain rate: NaN (invalid)')
        else:
            conditions.append('Strain rate ≥ 0')
            statuses.append('?')
            explanations.append('Strain rate calculation not available')
        
        # Create table
        if len(conditions) > 0:
            # Title
            ax4.text(0.5, 0.95, 'HEL Detection Conditions', transform=ax4.transAxes,
                    fontsize=14, fontweight='bold', ha='center', va='top')
            
            # Table header
            ax4.text(0.1, 0.85, 'Condition', transform=ax4.transAxes,
                    fontsize=11, fontweight='bold', ha='left', va='top')
            ax4.text(0.9, 0.85, 'Status', transform=ax4.transAxes,
                    fontsize=11, fontweight='bold', ha='right', va='top')
            
            # Draw horizontal line
            ax4.plot([0.05, 0.95], [0.82, 0.82], 'k-', linewidth=1, transform=ax4.transAxes)
            
            # Table rows with explanations
            y_start = 0.75
            y_spacing = 0.14  # Increased spacing to fit explanations
            for i, (condition, status, explanation) in enumerate(zip(conditions, statuses, explanations)):
                y_pos = y_start - i * y_spacing
                
                # Condition text (bold)
                ax4.text(0.1, y_pos, condition, transform=ax4.transAxes,
                        fontsize=10, fontweight='bold', ha='left', va='top')
                
                # Explanation text (smaller, italic, below condition)
                if explanation:
                    ax4.text(0.1, y_pos - 0.04, explanation, transform=ax4.transAxes,
                            fontsize=8, style='italic', ha='left', va='top', color='gray')
                
                # Status symbol with color
                if status == '✓':
                    color = 'green'
                elif status == '✗':
                    color = 'red'
                else:
                    color = 'gray'
                
                ax4.text(0.9, y_pos, status, transform=ax4.transAxes,
                        fontsize=16, fontweight='bold', ha='right', va='top', color=color)
            
            # Overall result
            if hel_ok:
                result_text = 'HEL DETECTED'
                result_color = 'green'
            else:
                result_text = 'NO HEL DETECTED'
                result_color = 'red'
            
            ax4.text(0.5, 0.15, result_text, transform=ax4.transAxes,
                    fontsize=16, fontweight='bold', ha='center', va='bottom', color=result_color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=result_color, linewidth=2))
        else:
            ax4.text(0.5, 0.5, 'No diagnostic data available', transform=ax4.transAxes,
                    fontsize=12, ha='center', va='center', style='italic', color='gray')
        
        # Add simple line from t=0 to HEL detection point
        if hel_ok and t_hel is not None and np.isfinite(t_hel) and rdp_points is not None and len(rdp_points) > 0:
            # Get HEL detection point coordinates (from RDP vertex at HEL)
            rdp_times = rdp_points[:, 0]
            rdp_velocities = rdp_points[:, 1]
            hel_rdp_idx = np.argmin(np.abs(rdp_times - t_hel))
            if hel_rdp_idx < len(rdp_times):
                hel_point_time = rdp_times[hel_rdp_idx]
                hel_point_velocity = rdp_velocities[hel_rdp_idx]
                
                # Draw line from t=0 to HEL detection point
                ax2.plot([0, hel_point_time], [0, hel_point_velocity], 
                        'g--', linewidth=2, alpha=0.7, 
                        label='t=0 to HEL Point', zorder=4)
        
        # Add text box with HEL results (on velocity subplot)
        if hel_ok and np.isfinite(hel_strength) and np.isfinite(hel_uncertainty):
            result_text = f'HEL Strength: {hel_strength:.3f} ± {hel_uncertainty:.3f} GPa'
            ax2.text(0.02, 0.98, result_text, transform=ax2.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        # NO HEL label removed - diagnostic table in 4th subplot shows all information
        
        plt.tight_layout()
        
        # Create HEL_plots subfolder if plot_individual is enabled
        plot_individual_enabled = self.spade_params.get('plot_individual', False)
        if plot_individual_enabled:
            hel_plots_dir = os.path.join(spade_output_dir, 'HEL_plots')
            os.makedirs(hel_plots_dir, exist_ok=True)
            plot_dir = hel_plots_dir
        else:
            plot_dir = spade_output_dir
        
        # Save plot in appropriate folder
        plot_filename = f'{base_name}--hel_detection.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.progress_signal.emit(f"  Saved HEL plot: {plot_filename}")

    def generate_combined_velocity_plot(self, velocity_plot_data, spade_output_dir):
        """Generate combined velocity plot with color coding based on material information"""
        self.progress_signal.emit("Generating combined velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Helper to extract material from parameter info using explicit 'Sample material' column
            def extract_material(param_info_dict):
                if not param_info_dict or not isinstance(param_info_dict, dict):
                    return 'Unknown'
                # Normalize keys once
                normalized = {}
                for k, v in param_info_dict.items():
                    key_norm = ''.join(ch for ch in k.lower() if ch.isalnum())
                    normalized[key_norm] = v
                # Preferred explicit columns
                preferred_keys = [
                    'samplematerial', 'samplemat', 'sample', 'material', 'flyermaterial'
                ]
                for pk in preferred_keys:
                    if pk in normalized and str(normalized[pk]).strip() != '':
                        return str(normalized[pk]).strip()
                # Fallback: scan values for common material tokens
                for v in param_info_dict.values():
                    val = str(v).strip()
                    if not val:
                        continue
                    low = val.lower()
                    if any(tok in low for tok in ['al', 'aluminum', 'aluminium']):
                        return val
                    if any(tok in low for tok in ['cu', 'copper']):
                        return val
                    if any(tok in low for tok in ['ti', 'titanium']):
                        return val
                    if 'steel' in low:
                        return val
                return 'Unknown'

            # Build list of (time, velocity, material)
            plotted = []
            for plot_data in velocity_plot_data:
                time_data = plot_data['time_ns']
                velocity_data = plot_data['velocity_ms']
                param_info = plot_data.get('param_info')
                material = extract_material(param_info)
                plotted.append((time_data, velocity_data, material))

            # Assign colors per unique material using a colormap
            unique_materials = []
            seen = set()
            for _t, _v, m in plotted:
                if m not in seen:
                    seen.add(m)
                    unique_materials.append(m)
            cmap = plt.get_cmap('Set3') if len(unique_materials) <= 12 else plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(unique_materials))))
            material_to_color = {m: colors[i] for i, m in enumerate(unique_materials)}

            # Plot traces grouped by material color
            for time_data, velocity_data, material in plotted:
                color = material_to_color.get(material, 'gray')
                ax.plot(time_data, velocity_data, color=color, alpha=0.7, linewidth=1)
            
            # Customize plot
            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            ax.set_xlabel(f'Time (ns) - Aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title('Combined Velocity Traces - Aligned (Color-coded by Material)', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Apply axis limits from SPADE params if not auto
            try:
                auto_calc = self.spade_params.get('auto_calculate_limits', True)
                self.progress_signal.emit(f"Combined plot - auto_calculate_limits: {auto_calc}")
                
                if not auto_calc:
                    x_min = float(self.spade_params.get('x_min_main', 0))
                    x_max = float(self.spade_params.get('x_max_main', 100))
                    y_min = float(self.spade_params.get('y_min_main', 0))
                    y_max = float(self.spade_params.get('y_max_main', 600))
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    self.progress_signal.emit(f"Applied axis limits: X({x_min}-{x_max}), Y({y_min}-{y_max})")
                else:
                    self.progress_signal.emit("Using auto-calculated axis limits")
            except Exception as e:
                self.progress_signal.emit(f"Error applying axis limits: {str(e)}")
                pass
            
            # Create legend with material colors
            legend_elements = [mpatches.Patch(color=material_to_color[m], label=m) for m in unique_materials]
            
            ax.legend(handles=legend_elements, title='Material', loc='upper right')
            
            # Save plot
            plot_path = os.path.join(spade_output_dir, "combined_velocity_plot.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress_signal.emit(f"Saved combined velocity plot to {plot_path}")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating combined velocity plot: {e}")

    def generate_all_velocity_traces_plot(self, input_path, spade_output_dir, uncertainty_threshold, unaligned_basenames=None):
        """Generate an all-traces plot aligned at 30 m/s using ALPSS output files in input_path.
        Applies noise fraction filtering (>1) and removes points with uncertainty > threshold.
        Color-codes traces by sample material from parameter files.
        Skips traces that never aligned (t0 not found) and saves PNG to both main output_dir and SPADE output dir."""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import re

            skip_unaligned = set(unaligned_basenames or [])

            pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
            self.progress_signal.emit(f"[DEBUG] Searching for velocity files with pattern: {pattern}")
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            self.progress_signal.emit(f"[DEBUG] Found {len(files)} velocity files matching pattern")
            if not files:
                self.progress_signal.emit(f"⚠️ No '*--vel-smooth-with-uncert.csv' files found for all-traces plot in {input_path}")
                # List what files are actually in the directory for debugging
                if os.path.exists(input_path):
                    all_csv_files = glob.glob(os.path.join(input_path, '**/*.csv'), recursive=True)
                    self.progress_signal.emit(f"[DEBUG] Found {len(all_csv_files)} total CSV files in {input_path}")
                    if all_csv_files:
                        sample_files = [os.path.basename(f) for f in all_csv_files[:5]]
                        self.progress_signal.emit(f"[DEBUG] Sample CSV files: {sample_files}")
                return
            
            # Initialize counters for this plot
            # We'll count actual traces processed, not just files found
            traces_plotted_local = 0
            traces_rejected_local = 0
            traces_skipped_initial = 0  # Traces skipped before processing loop
            rejection_reasons_local = {}

            # Collect traces with their materials
            trace_data = []
            for file_path in sorted(files):
                try:
                    # Extract base filename for parameter matching
                    base_filename = os.path.basename(file_path)
                    
                    # Extract PDV filename for matching to PDV_FileName column in parameter files
                    # Strategy 1: Extract PDV filename pattern (C1--YYYYMMDD--NNNNN)
                    pdv_filename_pattern = None
                    pattern_match = re.search(r'(C\d+--\d{8}--\d{5})', base_filename)
                    if pattern_match:
                        pdv_filename_pattern = pattern_match.group(1)
                        self.progress_signal.emit(f"  Extracted PDV filename: {pdv_filename_pattern}")
                    else:
                        # Strategy 2: Remove suffixes to get base name for matching
                        temp_name = base_filename
                        for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel', '--smooth']:
                            if temp_name.endswith(suffix):
                                temp_name = temp_name[:-len(suffix)]
                                break
                        pdv_filename_pattern = os.path.splitext(temp_name)[0]
                        self.progress_signal.emit(f"  Using base filename for matching: {pdv_filename_pattern}")
                    
                    if skip_unaligned and pdv_filename_pattern in skip_unaligned:
                        traces_skipped_initial += 1
                        reason = 'Unaligned trace (from SPADE)'
                        rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                        self.progress_signal.emit(f"Skipping {pdv_filename_pattern} in all-traces plot (unaligned trace)")
                        continue

                    # Get parameter data by matching PDV_FileName column
                    param_info = {}
                    material = 'Unknown'
                    
                    if hasattr(self, 'param_data') and self.param_data:
                        # Try exact match first (if pdv_filename_pattern is a key in param_data)
                        if pdv_filename_pattern in self.param_data:
                            param_info = self.param_data[pdv_filename_pattern]
                            self.progress_signal.emit(f"✓ Exact match: {pdv_filename_pattern}")
                        else:
                            # Use get_param_data_for_file to match by PDV_FileName column
                            param_info = self.get_param_data_for_file(pdv_filename_pattern)
                            if not param_info:
                                self.progress_signal.emit(f"⚠️  No parameter match for {pdv_filename_pattern}")
                                # Debug: show sample PDV_FileName values from param_data
                                if len(self.param_data) > 0:
                                    sample_pdv_files = []
                                    for key, param_entry in list(self.param_data.items())[:5]:
                                        if isinstance(param_entry, dict):
                                            pdv_file = param_entry.get('PDV_FileName', '')
                                            if pdv_file:
                                                sample_pdv_files.append(str(pdv_file).strip())
                                    if sample_pdv_files:
                                        self.progress_signal.emit(f"  Sample PDV_FileName values: {', '.join(set(sample_pdv_files))}")
                                    sample_keys = list(self.param_data.keys())[:3]
                                    self.progress_signal.emit(f"  Sample param_data keys: {sample_keys}")
                        
                        # Resolve material: igsn_material_map takes precedence over the
                        # parameter file's 'Sample material' column (see resolve_sample_material).
                        material = self.resolve_sample_material(base_filename, param_info)
                        self.progress_signal.emit(f"  Material for {base_filename}: '{material}'")

                    self.progress_signal.emit(f"File: {base_filename} -> Pattern: {pdv_filename_pattern} -> Material: {material}")
                    
                    trace_data.append({
                        'file_path': file_path,
                        'base_name': pdv_filename_pattern,
                        'material': material
                    })
                except Exception as e:
                    traces_skipped_initial += 1
                    reason = f'Error in initial processing: {str(e)[:50]}'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Error processing {file_path}: {e}")
                    import traceback
                    self.progress_signal.emit(traceback.format_exc())
                    continue

            # Group by material and assign colors
            unique_materials = []
            seen = set()
            material_counts = {}
            for trace in trace_data:
                mat = trace['material']
                material_counts[mat] = material_counts.get(mat, 0) + 1
                if mat not in seen:
                    seen.add(mat)
                    unique_materials.append(mat)
            
            # Report material grouping summary
            self.progress_signal.emit(f"\n=== Material Grouping Summary ===")
            self.progress_signal.emit(f"Found {len(unique_materials)} unique material(s):")
            for mat in sorted(unique_materials):
                count = material_counts[mat]
                self.progress_signal.emit(f"  {mat}: {count} trace(s)")
            self.progress_signal.emit("=" * 40 + "\n")
            
            # Use Set3 colormap for <=12 materials, tab20 for more
            if len(unique_materials) <= 12:
                cmap = plt.get_cmap('Set3')
            else:
                cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(unique_materials))))
            material_to_color = {mat: colors[i] for i, mat in enumerate(unique_materials)}

            # Use fixed 2920 x 1824 px per subplot (≈1.6:1 aspect ratio)
            # Total figure: 2920 x 3648 px (2 subplots stacked vertically)
            target_width_px = 2920
            target_height_px = 3648  # 1824 * 2 for two subplots
            target_dpi = 300
            fig_width_in = target_width_px / target_dpi
            fig_height_in = target_height_px / target_dpi
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(fig_width_in, fig_height_in),
                dpi=target_dpi,
            )

            # Option to skip Unknown material traces
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                has_labeled_material = any(trace['material'] != 'Unknown' for trace in trace_data)
                if not has_labeled_material:
                    skip_unknown = False
                    self.progress_signal.emit("⚠️  No parameter data matched current files; rendering 'Unknown' traces to avoid empty combined plot.")
            
            # List to track IQ detection failures
            iq_detection_failures = []

            traces_plotted = 0
            for trace in trace_data:
                file_path = trace['file_path']
                material = trace['material']
                
                # Skip Unknown material traces if requested
                if skip_unknown and material == 'Unknown':
                    traces_rejected_local += 1
                    reason = 'Unknown material'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Skipping {trace['base_name']} in all-traces plot (Unknown material)")
                    continue
                
                color = material_to_color.get(material, 'gray')
                
                try:
                    # Check IQ detection start time - skip if too close to end of time window
                    results_file = file_path.replace('--vel-smooth-with-uncert.csv', '--results.csv')
                    if os.path.exists(results_file):
                        try:
                            # Read ALPSS results - saved without header, has Name and Value columns
                            results_df = pd.read_csv(results_file, header=None, names=['Name', 'Value'])
                            
                            # Find Signal Start Time row
                            signal_start_time_s = None
                            signal_start_row = results_df[results_df['Name'] == 'Signal Start Time']
                            if not signal_start_row.empty:
                                signal_start_time_s = signal_start_row['Value'].iloc[0]
                                # Convert to float if it's a string
                                try:
                                    signal_start_time_s = float(signal_start_time_s)
                                except (ValueError, TypeError):
                                    signal_start_time_s = None
                            
                            if signal_start_time_s is not None and not pd.isna(signal_start_time_s):
                                # Get time_to_take from ALPSS config
                                time_to_take_s = self.alpss_params.get('time_to_take', 1e-6)
                                time_to_skip_s = self.alpss_params.get('time_to_skip', 2e-6)
                                
                                # Calculate end of time window (absolute time from file start)
                                time_window_end_s = time_to_skip_s + time_to_take_s
                                
                                # Check if start time is too close to end (within 5% of time_to_take)
                                threshold_fraction = 0.05  # 5% of time window
                                min_required_time_after_start = time_to_take_s * threshold_fraction
                                
                                # Calculate time available after signal start
                                time_after_start = time_window_end_s - signal_start_time_s
                                
                                if time_after_start < min_required_time_after_start:
                                    failure_info = {
                                        'base_name': trace['base_name'],
                                        'material': material,
                                        'signal_start_time_us': signal_start_time_s * 1e6,
                                        'time_window_end_us': time_window_end_s * 1e6,
                                        'time_after_start_us': time_after_start * 1e6,
                                        'min_required_us': min_required_time_after_start * 1e6,
                                        'reason': 'IQ detection failed: start time too close to end of time window'
                                    }
                                    iq_detection_failures.append(failure_info)
                                    traces_rejected_local += 1
                                    reason = 'IQ detection failed'
                                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                                    self.progress_signal.emit(
                                        f"Skipping {trace['base_name']} in all-traces plot "
                                        f"(IQ detection failed: start time {signal_start_time_s*1e6:.2f} μs too close to end "
                                        f"{time_window_end_s*1e6:.2f} μs, only {time_after_start*1e6:.2f} μs after start, "
                                        f"need at least {min_required_time_after_start*1e6:.2f} μs)")
                                    continue
                        except Exception as e:
                            # If we can't read results, continue anyway (don't fail on this check)
                            # Uncomment for debugging: self.progress_signal.emit(f"Warning: Could not read results for {trace['base_name']}: {e}")
                            pass
                    
                    # Load STFT velocity data
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_stft = df.iloc[:, 0].values
                    velocity_stft = df.iloc[:, 1].values
                    uncertainty_stft = df.iloc[:, 2].values

                    # Convert time to ns if likely in s/us
                    if np.nanmax(time_stft) < 1e-3:
                        time_stft = time_stft * 1e9
                    elif np.nanmax(time_stft) < 1.0:
                        time_stft = time_stft * 1e3
                    
                    # Standard file contains hybrid velocity if enabled, STFT if not
                    # No need to recreate hybrid - it's already in the file
                    base_file_path = file_path.replace('--vel-smooth-with-uncert.csv', '')
                    time_data = time_stft
                    velocity_data = velocity_stft
                    uncertainty_data = uncertainty_stft

                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass

                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    # Uncertainty threshold filtering
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)

                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                    if len(time_clean) == 0:
                        traces_rejected_local += 1
                        reason = 'No valid data after filtering'
                        rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                        continue

                    # Try HEL t=0 alignment first (if enabled), then fall back to threshold alignment
                    use_hel_alignment = self.spade_params.get('use_hel_t0_alignment_for_plots', True)
                    alignment_method = None
                    t0 = None
                    t0_idx = None
                    fallback_reason = None
                    
                    if use_hel_alignment:
                        # Try HEL t=0 alignment (velocity > 0 and increasing for 10 ns)
                        min_velocity_threshold = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                        hel_t0, hel_t0_idx, time_aligned_hel = self.find_hel_t0_alignment(
                            time_clean, velocity_clean, min_velocity_threshold
                        )
                        
                        if hel_t0 is not None and hel_t0_idx is not None:
                            t0 = hel_t0
                            t0_idx = hel_t0_idx
                            time_clean = time_aligned_hel
                            alignment_method = "HEL t=0 (velocity > 0, increasing 10 ns)"
                        else:
                            # Fall back to threshold alignment
                            alignment_method = "threshold (HEL t=0 failed)"
                            fallback_reason = "HEL t=0 alignment failed, using threshold alignment"
                    
                    # Use threshold alignment if HEL alignment not enabled or failed
                    if t0 is None:
                        # Get alignment threshold from config file
                        align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
                        tolerance = 0.01  # 0.01 m/s tolerance for floating point comparison
                        
                        # Find the first point where velocity reaches or exceeds threshold
                        t0_idx = None
                        for j, v in enumerate(velocity_clean):
                            if not np.isnan(v) and v >= (align_threshold - tolerance):
                                t0_idx = j
                                break
                        
                        if t0_idx is None:
                            # Fall back to earliest valid point
                            t0_idx = 0
                            max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                            min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                            fallback_reason = (
                                f"No threshold crossing at {align_threshold} m/s "
                                f"(velocity range {min_vel:.1f}-{max_vel:.1f} m/s); aligning to earliest valid point."
                            )
                        else:
                            has_point_below = any(
                                (not np.isnan(velocity_clean[j])) and velocity_clean[j] < (align_threshold - tolerance)
                                for j in range(max(t0_idx, 1))
                            )
                            if not has_point_below:
                                fallback_reason = (
                                    f"Trace starts at/above {align_threshold} m/s; aligning to first available sample."
                                )
                                # Prefer earliest point < threshold if it exists elsewhere
                                below_candidates = np.where(~np.isnan(velocity_clean) & (velocity_clean < (align_threshold - tolerance)))[0]
                                if len(below_candidates) > 0:
                                    t0_idx = below_candidates[0]
                    
                        if alignment_method is None:
                            alignment_method = "threshold"
                    
                    # Align the trace
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0
                    
                    if fallback_reason:
                        self.progress_signal.emit(f"[All Traces Plot] {trace['base_name']}: {fallback_reason}")

                    # Verify alignment: check that t=0 is actually at the alignment point
                    if abs(time_clean[t0_idx]) > 1e-6:  # Should be very close to 0 after alignment
                        self.progress_signal.emit(f"Warning: {trace['base_name']} alignment may be incorrect (t0={time_clean[t0_idx]:.3f} ns)")

                    ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1.5, label=material if traces_plotted == 0 or material not in [t['material'] for t in trace_data[:traces_plotted]] else '')
                    # Optional uncertainty bands
                    if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None and len(uncert_clean) == len(velocity_clean):
                        alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                        ax1.fill_between(time_clean,
                                         velocity_clean - uncert_clean,
                                         velocity_clean + uncert_clean,
                                         color=color, alpha=alpha)

                    # Bottom zoom window
                    zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                    mask_1000 = time_clean <= zoom_ns
                    if np.any(mask_1000):
                        ax2.plot(time_clean[mask_1000], velocity_clean[mask_1000], color=color, alpha=0.7, linewidth=1.5)
                        if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None and len(uncert_clean) == len(velocity_clean):
                            alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                            ax2.fill_between(time_clean[mask_1000],
                                             (velocity_clean - uncert_clean)[mask_1000],
                                             (velocity_clean + uncert_clean)[mask_1000],
                                             color=color, alpha=alpha)

                    traces_plotted += 1
                    traces_plotted_local += 1
                except Exception as e:
                    traces_rejected_local += 1
                    reason = f'Processing error: {str(e)[:50]}'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Error processing {trace['base_name']} for all-traces plot: {str(e)}")
                    import traceback
                    self.progress_signal.emit(traceback.format_exc())
                    continue

            # Determine alignment method label
            # Always define align_threshold for use in labels
            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            use_hel_alignment = self.spade_params.get('use_hel_t0_alignment_for_plots', True)
            if use_hel_alignment:
                alignment_label = "aligned to t=0 (HEL method: velocity > 0, increasing 10 ns)"
            else:
                alignment_label = f"aligned to t=0 at {align_threshold} m/s (threshold)"
            ax1.set_xlabel(f'Time (ns) - {alignment_label}', fontsize=12)
            ax1.set_ylabel('Velocity (m/s)', fontsize=12)
            ax1.set_title(f'All Velocity Traces (Aligned, Color by Material) - {traces_plotted} traces', fontsize=14)
            ax1.grid(False)
            
            # Add legend for materials (only show unique materials)
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                # Remove duplicates while preserving order
                seen_labels = set()
                unique_handles = []
                unique_labels = []
                for h, l in zip(handles, labels):
                    if l not in seen_labels:
                        seen_labels.add(l)
                        unique_handles.append(h)
                        unique_labels.append(l)
                # Set legend line width to 2
                for handle in unique_handles:
                    handle.set_linewidth(2)
                ax1.legend(unique_handles, unique_labels, loc='best', fontsize=12, title='Sample Material', title_fontsize=13)

            # Use same alignment label for ax2
            ax2.set_xlabel(f'Time (ns) - {alignment_label}', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.grid(False)

            # Apply axis limits
            try:
                if not self.spade_params.get('auto_calculate_limits', True):
                    x_min_main = float(self.spade_params.get('x_min_main', 0))
                    x_max_main = float(self.spade_params.get('x_max_main', 100))
                    y_min_main = float(self.spade_params.get('y_min_main', 0))
                    y_max_main = float(self.spade_params.get('y_max_main', 600))
                    # Top subplot should match combined plot main limits
                    ax1.set_xlim(x_min_main, x_max_main)
                    ax1.set_ylim(y_min_main, y_max_main)

                    x_min_zoom = float(self.spade_params.get('x_min_zoom', 0))
                    x_max_zoom = float(self.spade_params.get('x_max_zoom', self.spade_params.get('zoom_window_ns', 1000)))
                    y_min_zoom = float(self.spade_params.get('y_min_zoom', y_min_main))
                    y_max_zoom = float(self.spade_params.get('y_max_zoom', y_max_main))
                    ax2.set_xlim(x_min_zoom, x_max_zoom)
                    ax2.set_ylim(y_min_zoom, y_max_zoom)
                    ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)} to {int(x_max_zoom)} ns)', fontsize=14)
                else:
                    # Default zoom: 0 to zoom_window_ns
                    zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                    ax2.set_xlim(0, zoom_ns)
                    ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)
            except Exception:
                # Fallback to default behavior
                zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)

            plt.tight_layout()

            # Save to main output_dir and SPADE output dir
            out_main = os.path.join(self.output_dir, 'all_velocity_traces.png')
            out_spade = os.path.join(spade_output_dir, 'all_velocity_traces.png')
            try:
                fig.savefig(out_main, dpi=target_dpi, bbox_inches='tight')
            except Exception:
                pass
            try:
                fig.savefig(out_spade, dpi=target_dpi, bbox_inches='tight')
            except Exception:
                pass
            plt.close(fig)

            # Save IQ detection failures list
            if iq_detection_failures:
                failure_file = os.path.join(spade_output_dir, 'IQ_based_detection_failure.csv')
                try:
                    failure_df = pd.DataFrame(iq_detection_failures)
                    failure_df.to_csv(failure_file, index=False)
                    self.progress_signal.emit(f"Saved IQ detection failure list: {failure_file} ({len(iq_detection_failures)} traces)")
                except Exception as e:
                    self.progress_signal.emit(f"Error saving IQ detection failure list: {e}")

            # Update global counters
            # Total input traces = files found (some may be skipped before processing)
            self.total_input_traces = len(files)
            self.traces_plotted = traces_plotted_local
            # Total rejected = rejected during processing + skipped in initial loop
            self.traces_rejected = traces_rejected_local + traces_skipped_initial
            self.rejection_reasons = rejection_reasons_local

            self.progress_signal.emit(f"✅ Successfully saved aligned all-traces velocity plot to: {out_spade}")
            self.progress_signal.emit(f"   Total traces plotted: {traces_plotted_local}")
            self.progress_signal.emit(f"   Total traces rejected/skipped: {self.traces_rejected}")
            if self.rejection_reasons:
                self.progress_signal.emit(f"   Rejection reasons: {dict(self.rejection_reasons)}")
        except Exception as e:
            self.progress_signal.emit(f"❌ Error generating all-traces velocity plot: {e}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_zn_traces_diagnostic_plot(self, input_path, spade_output_dir, uncertainty_threshold):
        """Temporary diagnostic plot showing only Zn traces with labels (last 5 digits of filename)"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import re

            pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            if not files:
                self.progress_signal.emit("No '*--vel-smooth-with-uncert.csv' files found for Zn diagnostic plot")
                return

            # Collect Zn traces
            zn_traces = []
            for file_path in sorted(files):
                try:
                    base_filename = os.path.basename(file_path)
                    
                    # Extract C1--XXXXXXXX--XXYYY pattern from filename
                    pattern_match = re.search(r'(C\d+--\d{8}--\d{5})', base_filename)
                    if pattern_match:
                        pdv_filename_pattern = pattern_match.group(1)
                    else:
                        pdv_filename_pattern = re.sub(r'--.*$', '', base_filename)
                        pdv_filename_pattern = os.path.splitext(pdv_filename_pattern)[0]
                    
                    # Extract last 5 digits from filename
                    last_5_digits = pdv_filename_pattern[-5:] if len(pdv_filename_pattern) >= 5 else pdv_filename_pattern
                    
                    # Get material
                    material = 'Unknown'
                    if hasattr(self, 'param_data') and self.param_data:
                        if pdv_filename_pattern in self.param_data:
                            param_info = self.param_data[pdv_filename_pattern]
                            material_keys = [
                                'Sample material', 'Sample Material', 'Sample_Material',
                                'sample_material', 'samplematerial', 'SampleMaterial',
                                'Material', 'material', 'Flyer_material', 'Flyer Material'
                            ]
                            for key in material_keys:
                                if key in param_info:
                                    material_val = str(param_info[key]).strip()
                                    if material_val and material_val.lower() != 'nan':
                                        material = material_val
                                        break
                    
                    # Only include Zn traces
                    if material.lower() in ['zn', 'zinc']:
                        zn_traces.append({
                            'file_path': file_path,
                            'base_name': pdv_filename_pattern,
                            'label': last_5_digits,
                            'material': material
                        })
                except Exception as e:
                    continue
            
            if len(zn_traces) == 0:
                self.progress_signal.emit("No Zn traces found for diagnostic plot")
                return
            
            self.progress_signal.emit(f"Found {len(zn_traces)} Zn traces for diagnostic plot")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            traces_plotted = 0
            
            for trace in zn_traces:
                file_path = trace['file_path']
                label = trace['label']
                
                try:
                    # Check IQ detection start time - skip if too close to end of time window
                    results_file = file_path.replace('--vel-smooth-with-uncert.csv', '--results.csv')
                    if os.path.exists(results_file):
                        try:
                            # Read ALPSS results - saved without header, has Name and Value columns
                            results_df = pd.read_csv(results_file, header=None, names=['Name', 'Value'])
                            
                            # Find Signal Start Time row
                            signal_start_time_s = None
                            signal_start_row = results_df[results_df['Name'] == 'Signal Start Time']
                            if not signal_start_row.empty:
                                signal_start_time_s = signal_start_row['Value'].iloc[0]
                                # Convert to float if it's a string
                                try:
                                    signal_start_time_s = float(signal_start_time_s)
                                except (ValueError, TypeError):
                                    signal_start_time_s = None
                            
                            if signal_start_time_s is not None and not pd.isna(signal_start_time_s):
                                time_to_take_s = self.alpss_params.get('time_to_take', 1e-6)
                                time_to_skip_s = self.alpss_params.get('time_to_skip', 2e-6)
                                time_window_end_s = time_to_skip_s + time_to_take_s
                                threshold_fraction = 0.05  # 5% of time window
                                min_required_time_after_start = time_to_take_s * threshold_fraction
                                time_after_start = time_window_end_s - signal_start_time_s
                                
                                if time_after_start < min_required_time_after_start:
                                    self.progress_signal.emit(
                                        f"Skipping {trace['base_name']} (IQ detection failed: start time {signal_start_time_s*1e6:.2f} μs too close to end)")
                                    continue
                        except Exception:
                            pass
                    
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    uncertainty_data = df.iloc[:, 2].values

                    # Convert time to ns if likely in s/us
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3

                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass

                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)

                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    if len(time_clean) == 0:
                        continue

                    # Get alignment threshold and check if trace crosses threshold from below
                    # This threshold is defined by the user in helix_master_config.json as "align_velocity_threshold_ms"
                    tolerance = 0.01  # 0.01 m/s tolerance for floating point comparison
                    
                    # Find the first point where velocity reaches or exceeds threshold
                    # This must be a rising crossing (trace starts below threshold and crosses it)
                    t0_idx = None
                    for j, v in enumerate(velocity_clean):
                        if not np.isnan(v) and v >= (align_threshold - tolerance):
                            t0_idx = j
                            break
                    
                    # Check if trace crosses threshold from below (not starting above and decreasing)
                    if t0_idx is None or t0_idx == 0:
                        # No threshold crossing found, or trace starts at threshold (likely bad data)
                        max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                        min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                        self.progress_signal.emit(f"Skipping {trace['base_name']} (no threshold crossing from below at {align_threshold} m/s, velocity range: {min_vel:.1f} to {max_vel:.1f} m/s)")
                        continue
                    
                    # Verify that trace started below threshold (crossing from below)
                    # Check if there's at least one point before t0_idx that's below threshold
                    has_point_below = False
                    for j in range(t0_idx):
                        if not np.isnan(velocity_clean[j]) and velocity_clean[j] < (align_threshold - tolerance):
                            has_point_below = True
                            break
                    
                    if not has_point_below:
                        # Trace starts at or above threshold - likely bad data (e.g., IQ detection failed, starts late)
                        max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                        min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                        self.progress_signal.emit(f"Skipping {trace['base_name']} (trace starts at/above threshold {align_threshold} m/s, does not cross from below, velocity range: {min_vel:.1f} to {max_vel:.1f} m/s)")
                        continue
                    
                    # Align the trace
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0
                    
                    # Filter negative time if requested
                    filter_negative_time = self.spade_params.get('filter_negative_time', False)
                    if filter_negative_time:
                        mask_t_positive = time_clean >= 0
                        time_clean = time_clean[mask_t_positive]
                        velocity_clean = velocity_clean[mask_t_positive]
                        if len(time_clean) == 0:
                            continue
                    
                    # Plot with label
                    ax.plot(time_clean, velocity_clean, alpha=0.7, linewidth=1.5, label=label)
                    
                    # Add text label at peak or end of trace
                    if len(velocity_clean) > 0:
                        peak_idx = np.nanargmax(velocity_clean)
                        if peak_idx < len(time_clean) and peak_idx < len(velocity_clean):
                            ax.text(time_clean[peak_idx], velocity_clean[peak_idx], 
                                   f' {label}', fontsize=8, alpha=0.8, 
                                   verticalalignment='bottom')
                    
                    traces_plotted += 1
                except Exception as e:
                    self.progress_signal.emit(f"Error processing {trace['base_name']}: {e}")
                    continue

            ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title(f'Zn Traces Diagnostic Plot - {traces_plotted} traces (labeled with last 5 digits)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=3)
            
            plt.tight_layout()

            # Save to SPADE output dir
            out_path = os.path.join(spade_output_dir, 'zn_traces_diagnostic.png')
            try:
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                self.progress_signal.emit(f"Saved Zn diagnostic plot to: {out_path}")
            except Exception as e:
                self.progress_signal.emit(f"Error saving Zn diagnostic plot: {e}")
            plt.close(fig)

        except Exception as e:
            self.progress_signal.emit(f"Error generating Zn diagnostic plot: {e}")
            import traceback
            self.progress_signal.emit(traceback.format_exc())

    def generate_spall_analysis_summary(self, spade_output_dir):
        """Generate enhanced spall analysis summary with comprehensive parameter file data"""
        self.progress_signal.emit("Generating enhanced spall analysis summary...")

        # Check if spall summary already exists
        spall_summary_path = os.path.join(spade_output_dir, 'spall_summary.csv')
        if os.path.exists(spall_summary_path):
            # Check file modification time to see if it was just created
            import time
            file_age = time.time() - os.path.getmtime(spall_summary_path)
            if file_age < 60:  # Created within last minute (likely from this run)
                self.progress_signal.emit(f"✓ Found spall_summary.csv created {file_age:.1f} seconds ago (from current run)")
            else:
                self.progress_signal.emit(f"⚠ Reading from existing spall_summary.csv (created {file_age/60:.1f} minutes ago)")
                self.progress_signal.emit(f"   To see RDP detection in action, delete {os.path.basename(spall_summary_path)} and re-run analysis")
        else:
            self.progress_signal.emit("No spall summary found - SPADE analysis may not have completed")
            return

        # Read existing spall summary
        spall_df = pd.read_csv(spall_summary_path)
        self.progress_signal.emit(f"Found {len(spall_df)} entries in spall summary")
        
        # Debug: Check for traces with P4 <= P3 that should be DNS
        if 'DNS_Classification' in spall_df.columns:
            dns_count = spall_df['DNS_Classification'].notna().sum()
            p4_p3_dns = spall_df['DNS_Classification'].astype(str).str.contains('No re-acceleration after pullback', case=False, na=False).sum()
            self.progress_signal.emit(f"DEBUG: DNS classifications found: {dns_count} total, {p4_p3_dns} with 'No re-acceleration after pullback'")
            if p4_p3_dns > 0:
                p4_p3_traces = spall_df[spall_df['DNS_Classification'].astype(str).str.contains('No re-acceleration after pullback', case=False, na=False)]
                self.progress_signal.emit(f"DEBUG: Traces with P4 <= P3 (should be DNS): {list(p4_p3_traces['Filename'].head(5))}")
        
        # Debug: Check what columns SPADE output contains
        self.progress_signal.emit(f"DEBUG: Columns in SPADE spall_summary.csv: {list(spall_df.columns)}")
        shock_stress_cols_in_spade = [c for c in spall_df.columns if 'shock' in c.lower() or 'stress' in c.lower()]
        self.progress_signal.emit(f"DEBUG: Shock/stress columns in SPADE output: {shock_stress_cols_in_spade}")
        if 'Plateau Mean Velocity (m/s)' in spall_df.columns:
            plateau_non_null = spall_df['Plateau Mean Velocity (m/s)'].notna().sum()
            self.progress_signal.emit(f"DEBUG: Rows with Plateau Mean Velocity: {plateau_non_null} out of {len(spall_df)}")

        # Enhance with parameter file data
        enhanced_spall_data = []
        no_alpss_count = 0

        for idx, row in spall_df.iterrows():
            filename = row.get('Filename', '')
            if not filename or filename == 'data' or not isinstance(filename, str):
                # Skip invalid filenames
                if filename:
                    self.progress_signal.emit(f"Skipping invalid filename: {repr(filename)}")
                continue

            # Get file base name for parameter matching
            # Handle cases where filename might be a full path or just a name
            base_name = os.path.splitext(os.path.basename(str(filename)))[0]
            
            # Strip velocity-specific suffixes that might be in the filename
            # These suffixes are added by ALPSS/SPADE processing
            velocity_suffixes = [
                '--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel',
                '--smooth', '--results', '--noise', '--frac'
            ]
            for suffix in velocity_suffixes:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break

            # Skip if base_name is still invalid
            if not base_name or base_name == 'data':
                self.progress_signal.emit(f"Skipping invalid base_name from filename: {repr(filename)}")
                continue

            # Get parameter data if available using helper function
            param_info = self.get_param_data_for_file(base_name)

            # Debug parameter data with more detail
            if param_info:
                self.progress_signal.emit(f"Found parameter data for {base_name}: {list(param_info.keys())}")
                # Check if Sample material exists
                if 'Sample material' in param_info:
                    sample_mat_val = str(param_info['Sample material']).strip()
                    self.progress_signal.emit(f"  Sample material value: '{sample_mat_val}'")
            else:
                self.progress_signal.emit(f"No parameter data found for {base_name}")
                # Debug: show what keys are available in param_data (first few)
                if self.param_data:
                    sample_keys = list(self.param_data.keys())[:5]
                    self.progress_signal.emit(f"  Sample param_data keys (showing first 5): {sample_keys}")
                    # Try to find a partial match
                    for key in self.param_data.keys():
                        if base_name in str(key) or str(key) in base_name:
                            self.progress_signal.emit(f"  Potential match found: {key} (but get_param_data_for_file didn't return it)")
                            break

            # Create enhanced row with all original SPADE data
            enhanced_row = row.copy()
            
            # Debug: Check if Peak Shock Stress exists in SPADE row
            if 'Peak Shock Stress (GPa)' in row:
                self.progress_signal.emit(f"  DEBUG: Found 'Peak Shock Stress (GPa)' in SPADE row for {base_name}: {row['Peak Shock Stress (GPa)']}")
            elif 'Peak_Shock_Stress_GPa' in row:
                self.progress_signal.emit(f"  DEBUG: Found 'Peak_Shock_Stress_GPa' in SPADE row for {base_name}: {row['Peak_Shock_Stress_GPa']}")
                # Normalize to standard column name
                enhanced_row['Peak Shock Stress (GPa)'] = row['Peak_Shock_Stress_GPa']
            else:
                self.progress_signal.emit(f"  DEBUG: Peak Shock Stress NOT found in SPADE row for {base_name}. Available columns: {list(row.keys())}")

            # Add parameter file data as extra columns (preserve original names only)
            for key, value in param_info.items():
                enhanced_row[key] = value

            # Resolve material: igsn_material_map takes precedence over the parameter
            # file's 'Sample material' column (see resolve_sample_material). Falls back
            # to whatever Material value is already on the row if neither resolves.
            sample_material = self.resolve_sample_material(base_name, param_info)
            if sample_material == 'Unknown' and 'Material' in row:
                existing_material = str(row['Material']).strip()
                if _is_valid_material_value(existing_material):
                    sample_material = existing_material
            enhanced_row['Material'] = sample_material
            self.progress_signal.emit(f"  Material for {base_name}: '{sample_material}'")

            # Try to find corresponding ALPSS results file for additional data
            # Search both current output_dir and standard ALPSS output location
            candidate_results = [
                os.path.join(self.output_dir, f"{base_name}--results.csv"),
                os.path.join("ALPSS", "output_data", f"{base_name}--results.csv"),
                os.path.join("output", f"{base_name}--results.csv"),
            ]
            alpss_results_file = next((p for p in candidate_results if os.path.exists(p)), None)
            if alpss_results_file and os.path.exists(alpss_results_file):
                try:
                    # Read ALPSS results
                    alpss_results = pd.read_csv(alpss_results_file, header=None, names=['Name', 'Value'])
                    alpss_dict = dict(zip(alpss_results['Name'], alpss_results['Value']))
                    
                    # Add ALPSS results to enhanced summary
                    # ALPSS saves spall strength in Pa, convert to GPa
                    alpss_spall_pa = alpss_dict.get('Spall Strength', np.nan)
                    if pd.notna(alpss_spall_pa):
                        try:
                            enhanced_row['ALPSS_Spall_Strength_GPa'] = float(alpss_spall_pa) / 1e9
                        except (ValueError, TypeError):
                            enhanced_row['ALPSS_Spall_Strength_GPa'] = np.nan
                    else:
                        enhanced_row['ALPSS_Spall_Strength_GPa'] = np.nan
                    
                    alpss_spall_unc_pa = alpss_dict.get('Spall Strength Uncertainty', np.nan)
                    if pd.notna(alpss_spall_unc_pa):
                        try:
                            enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = float(alpss_spall_unc_pa) / 1e9
                        except (ValueError, TypeError):
                            enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = np.nan
                    else:
                        enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = np.nan
                    enhanced_row['ALPSS_Strain_Rate_s1'] = alpss_dict.get('Strain Rate', np.nan)
                    enhanced_row['ALPSS_Strain_Rate_Uncertainty_s1'] = alpss_dict.get('Strain Rate Uncertainty', np.nan)
                    # Note: ALPSS no longer calculates Peak Shock Stress (removed as part of cleanup)
                    enhanced_row['ALPSS_Velocity_at_Max_Compression_ms'] = alpss_dict.get('Velocity at Max Compression', np.nan)
                    enhanced_row['ALPSS_Velocity_at_Max_Tension_ms'] = alpss_dict.get('Velocity at Max Tension', np.nan)
                    enhanced_row['ALPSS_Velocity_at_Recompression_ms'] = alpss_dict.get('Velocity at Recompression', np.nan)
                    enhanced_row['ALPSS_Time_at_Max_Compression_ns'] = alpss_dict.get('Time at Max Compression', np.nan)
                    enhanced_row['ALPSS_Time_at_Max_Tension_ns'] = alpss_dict.get('Time at Max Tension', np.nan)
                    enhanced_row['ALPSS_Time_at_Recompression_ns'] = alpss_dict.get('Time at Recompression', np.nan)
                    enhanced_row['ALPSS_Carrier_Frequency_Hz'] = alpss_dict.get('Carrier Frequency', np.nan)
                    enhanced_row['ALPSS_Signal_Start_Time_s'] = alpss_dict.get('Signal Start Time', np.nan)
                    enhanced_row['ALPSS_Smoothing_Characteristic_Time_s'] = alpss_dict.get('Smoothing Characteristic Time', np.nan)
                    
                    self.progress_signal.emit(f"Added ALPSS results for {base_name}")
                except Exception as e:
                    self.progress_signal.emit(f"Warning: Could not read ALPSS results for {base_name}: {e}")
            else:
                no_alpss_count += 1
            
            # Peak Shock Stress is already calculated using EOS in detect_dns_and_process_spall
            # Just ensure it's in the standard column name
            if 'Peak Shock Stress (GPa)' not in enhanced_row:
                # Try alternative column names
                for alt_col in ['Peak_Shock_Stress_GPa', 'Peak Shock Stress', 'Shock Stress (GPa)']:
                    if alt_col in enhanced_row:
                        enhanced_row['Peak Shock Stress (GPa)'] = enhanced_row[alt_col]
                        break
                else:
                    enhanced_row['Peak Shock Stress (GPa)'] = np.nan

            enhanced_spall_data.append(enhanced_row)

        if no_alpss_count > 0:
            self.progress_signal.emit(f"ℹ No ALPSS results for {no_alpss_count} trace(s) (SPADE-only or ALPSS not run - using SPADE data)")

        # Save enhanced spall summary (include all parameter file columns, drop redundant columns)
        if enhanced_spall_data:
            enhanced_spall_df = pd.DataFrame(enhanced_spall_data)

            # Ensure all parameter file columns are present in the output
            all_param_columns = set()
            if self.param_data:
                try:
                    for _key, param_dict in self.param_data.items():
                        if isinstance(param_dict, dict):
                            all_param_columns.update(param_dict.keys())
                except Exception:
                    pass

            # Add any missing param columns with NaN
            for col in sorted(all_param_columns):
                if col not in enhanced_spall_df.columns:
                    enhanced_spall_df[col] = np.nan

            # Remove redundant columns
            try:
                import re
                # 1) Drop columns that normalize to the same token (keep first occurrence)
                seen_norm = set()
                cols_to_drop = []
                for col in enhanced_spall_df.columns:
                    norm = re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_").lower()
                    if norm in seen_norm:
                        cols_to_drop.append(col)
                    else:
                        seen_norm.add(norm)
                if cols_to_drop:
                    enhanced_spall_df = enhanced_spall_df.drop(columns=cols_to_drop, errors='ignore')

                # 2) Drop exact-duplicate columns (identical values across all rows)
                enhanced_spall_df = enhanced_spall_df.T.drop_duplicates().T

                # 3) Drop entirely-NaN columns, but always keep the derived shock-front
                #    diagnostics so the master schema is stable across runs. A single-trace
                #    run where a backward-walk quantity is NaN must not make the column
                #    vanish -- the standalone post-analysis plots rely on it existing.
                _keep_always = {
                    'Peak_Shock_Time_ns', 'RiseTime_ArrivalToPeak_ns', 'RiseTime_80_20_ns',
                    'RiseTime_90_10_ns', 'RiseTime_MaxSlope_ns', 'PlasticStrainRate_80_20_s^-1',
                    'PlasticStrainRate_90_10_s^-1', 'PlasticStrainRate_MaxSlope_s^-1',
                    'Compressive_StrainRate_Avg_s^-1', 'Compressive_StrainRate_Ufs_s^-1',
                    'Shock_Velocity_Us_m_s', 'Shock_Front_Width_um',
                }
                _allnan = [c for c in enhanced_spall_df.columns
                           if c not in _keep_always and enhanced_spall_df[c].isna().all()]
                if _allnan:
                    enhanced_spall_df = enhanced_spall_df.drop(columns=_allnan)
            except Exception:
                pass

            # Reorganize columns in the specified order
            # Order: filename, metadata, experimental params, spall analysis results, then ALPSS values
            priority_columns = [
                # 1. Basic identifiers
                'Filename',
                'Timestamp',
                'Exp_ID',
                # 2. Flyer information
                'Flyer_material',
                'Flyer_ID',
                'Flyer_Thickness (um)',
                'flyer_thickness',  # Alternative naming
                # 3. Sample information
                'Sample_ID',
                'Material',
                'Density_kg_m3',
                'density',  # Alternative naming
                'Acoustic_Velocity_m_s',
                'acoustic_velocity',  # Alternative naming
                # 4. Experimental parameters
                'Spacing (um)',
                'Waveplate_Angle (Degrees)',
                'PDV_Target_Wavelength (m)',
                'PDV_Target_Power (dBm)',
                'PDV_Ref_Wavelength (m)',
                'PDV_Ref_Power (dBm)',
                'PDV_Return_Power (dBm)',
                'Flyer_Row',
                'Flyer_Column',
                'Flyer_X_Position_Desired (mm)',
                'Flyer_Y_Position_desired (mm)',
                'Flyer_X_Position_Corrected (mm)',
                'Flyer_Y_Position_Corrected (mm)',
                'Laser_Ref_Energy (mJ)',
                'Laser_Target_Energy (mJ)',
                'Exp_Time (seconds)',
                'Notes',
                # 5. SPADE analysis results
                'DNS_Classification',
                'Processing_Status',
                'Analysis_Notes',
                'Spall_OK',
                'First_Maxima_m_s',
                'Minima_m_s',
                'Second_Maxima_m_s',
                'Pullback_Velocity_m_s',
                'Pullback_Velocity_Unc_m_s',
                'Plateau Mean Velocity (m/s)',
                'Peak Shock Stress (GPa)',
                'Peak Shock Stress Uncertainty (GPa)',
                'Spall_Strength_GPa',
                'Spall_Strength_Unc_GPa',
                'Spall_StrainRate_s^-1',
                'Spall_StrainRate_UNCERTAINITY',  # As specified by user
                'Strain_Rate_Uncertainty_s^-1',  # Alternative naming
                'Strain Rate Uncertainty (s^-1)',  # Alternative naming
                'Strain_Rate_Unc_s^-1',  # Alternative naming
                'StrainRate_Unc_s^-1',   # Alternative naming
                # 5b. Derived shock-front diagnostics (computed in detect_dns_and_process_spall)
                'Peak_Shock_Time_ns',
                'RiseTime_ArrivalToPeak_ns',
                'RiseTime_80_20_ns',
                'RiseTime_90_10_ns',
                'RiseTime_MaxSlope_ns',
                'PlasticStrainRate_80_20_s^-1',
                'PlasticStrainRate_90_10_s^-1',
                'PlasticStrainRate_MaxSlope_s^-1',
                'Compressive_StrainRate_Avg_s^-1',
                'Compressive_StrainRate_Ufs_s^-1',
                'Shock_Velocity_Us_m_s',
                'Shock_Front_Width_um',
                # 6. HEL analysis results
                'hel_ok',
                'hel_strength_gpa',
                'hel_uncertainty_gpa',
                'hel_strain_rate_s^-1',
                'hel_segment_time_ns',
                'hel_consecutive_points',
                'free_surface_velocity_ms',
                # Note: ALPSS columns will be added at the end automatically
            ]

            # Merge HEL data from velocity_shots_summary.csv (these columns are not in spall_summary)
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if os.path.exists(velocity_shots_path):
                try:
                    vel_df = pd.read_csv(velocity_shots_path)
                    hel_cols = [c for c in [
                        'hel_ok', 'hel_strength_gpa', 'hel_uncertainty_gpa',
                        'hel_strain_rate_s^-1', 'hel_segment_time_ns',
                        'hel_consecutive_points', 'free_surface_velocity_ms'
                    ] if c in vel_df.columns]
                    if hel_cols and 'file_name' in vel_df.columns:
                        hel_merge_df = vel_df[['file_name'] + hel_cols].rename(columns={'file_name': 'Filename'})
                        # Drop columns already present to avoid conflicts
                        existing_hel = [c for c in hel_cols if c in enhanced_spall_df.columns]
                        if existing_hel:
                            enhanced_spall_df = enhanced_spall_df.drop(columns=existing_hel)
                        enhanced_spall_df = enhanced_spall_df.merge(hel_merge_df, on='Filename', how='left')
                        self.progress_signal.emit(f"Merged HEL data ({len(hel_cols)} columns) from velocity_shots_summary.csv")
                except Exception as e:
                    self.progress_signal.emit(f"Warning: Could not merge HEL data from velocity_shots_summary: {e}")

            # Get all columns from the DataFrame
            all_columns = list(enhanced_spall_df.columns)

            # Start with priority columns (only those that exist)
            reordered_columns = [col for col in priority_columns if col in all_columns]

            # Separate remaining columns into ALPSS and other columns
            remaining_columns = [col for col in all_columns if col not in reordered_columns]

            # Separate ALPSS columns (they start with 'ALPSS_')
            alpss_columns = [col for col in remaining_columns if col.startswith('ALPSS_')]
            other_columns = [col for col in remaining_columns if not col.startswith('ALPSS_')]

            # Add other columns first, then ALPSS columns at the end
            reordered_columns.extend(other_columns)
            reordered_columns.extend(sorted(alpss_columns))  # Sort ALPSS columns for consistency

            # Reorder the DataFrame
            enhanced_spall_df = enhanced_spall_df[reordered_columns]

            enhanced_spall_path = os.path.join(spade_output_dir, self._get_summary_filename())
            # Persist the master with standardized column names (single naming
            # convention on disk); the in-memory enhanced_spall_df keeps its original
            # names for the downstream GUI plotting calls below.
            standardize_summary_columns(enhanced_spall_df.copy()).to_csv(enhanced_spall_path, index=False)
            self.progress_signal.emit(f"Generated enhanced spall summary with {len(enhanced_spall_data)} entries")
            self.progress_signal.emit(f"Saved to: {enhanced_spall_path}")
            self._save_run_config(spade_output_dir)
            
            # Add note about filtering criteria to a separate notes file
            notes_file = os.path.join(spade_output_dir, 'spall_analysis_notes.txt')
            with open(notes_file, 'w', encoding='utf-8') as f:
                f.write("SPALL ANALYSIS FILTERING NOTES\n")
                f.write("=" * 60 + "\n\n")
                f.write("The following filtering criteria were applied during spall detection:\n\n")
                f.write("1. Low-velocity flat signal filter:\n")
                f.write("   - Traces are discarded (marked as DNS) if all 4 key points\n")
                f.write("     (first peak, first valley, second peak, initial velocity)\n")
                f.write("     are within 30 m/s of each other AND peak velocity < 100 m/s.\n")
                f.write("   - This filters out signals that are essentially flat or have\n")
                f.write("     very low velocity variation, which are not valid spall signals.\n\n")
                f.write("2. Structural requirements:\n")
                f.write("   - Traces must have clear peak/valley structure\n")
                f.write("   - Must show pullback after initial rise\n")
                f.write("   - Must show re-acceleration after pullback\n\n")
                f.write("Traces that do not meet these criteria are marked as 'DNS' (Did Not Spall)\n")
                f.write(f"in the DNS_Classification column of {self._get_summary_filename()}\n")
            self.progress_signal.emit(f"Saved filtering notes to: {notes_file}")

            # Merge EOS-calculated shock stress back into velocity_shots_summary.csv
            # This ensures HEL plots and spall plots use the same shock stress values
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if os.path.exists(velocity_shots_path) and 'Peak Shock Stress (GPa)' in enhanced_spall_df.columns:
                try:
                    velocity_shots_df = pd.read_csv(velocity_shots_path)
                    
                    # Match by filename (remove extensions and suffixes)
                    def normalize_filename(fname):
                        # Remove common suffixes and extensions
                        fname = str(fname).replace('--vel-smooth-with-uncert', '')
                        fname = os.path.splitext(fname)[0]
                        return fname.strip()
                    
                    # Create mapping from enhanced_spall_df
                    spall_shock_stress_map = {}
                    spall_shock_stress_unc_map = {}
                    
                    if 'Filename' in enhanced_spall_df.columns:
                        for idx, row in enhanced_spall_df.iterrows():
                            filename = normalize_filename(row['Filename'])
                            if pd.notna(row.get('Peak Shock Stress (GPa)')):
                                spall_shock_stress_map[filename] = row['Peak Shock Stress (GPa)']
                            if 'Peak Shock Stress Uncertainty (GPa)' in enhanced_spall_df.columns:
                                if pd.notna(row.get('Peak Shock Stress Uncertainty (GPa)')):
                                    spall_shock_stress_unc_map[filename] = row['Peak Shock Stress Uncertainty (GPa)']
                    
                    # Update velocity_shots_summary with EOS-calculated shock stress
                    updated_count = 0
                    for idx, row in velocity_shots_df.iterrows():
                        filename = normalize_filename(row.get('file_name', ''))
                        if filename in spall_shock_stress_map:
                            velocity_shots_df.loc[idx, 'Peak Shock Stress (GPa)'] = spall_shock_stress_map[filename]
                            if filename in spall_shock_stress_unc_map:
                                velocity_shots_df.loc[idx, 'Peak Shock Stress Uncertainty (GPa)'] = spall_shock_stress_unc_map[filename]
                            updated_count += 1
                    
                    if updated_count > 0:
                        # Write with high precision to avoid rounding issues
                        velocity_shots_df.to_csv(velocity_shots_path, index=False, float_format='%.10f')
                        self.progress_signal.emit(f"Merged EOS-calculated shock stress into velocity_shots_summary.csv ({updated_count} entries updated)")
                    else:
                        self.progress_signal.emit("No matching filenames found to merge shock stress values")
                except Exception as e:
                    self.progress_signal.emit(f"Warning: Could not merge shock stress into velocity_shots_summary: {e}")

            # Retain spall_summary.csv alongside the consolidated master for safety /
            # back-compat (it is a strict subset of the master). Previously this file was
            # deleted as redundant; it is now kept as a legacy export.
            try:
                spall_summary_path = os.path.join(spade_output_dir, 'spall_summary.csv')
                if os.path.exists(spall_summary_path):
                    self.progress_signal.emit(f"Kept spall_summary.csv as a legacy export (master: {self._get_summary_filename()})")
            except Exception:
                pass
            
            # SANITY CHECK: Verify that every trace with spall strength has shock stress
            self._validate_spall_shock_stress_consistency(enhanced_spall_df)
            
            # Generate additional analysis plots if parameter data is available
            self.generate_spall_analysis_plots(enhanced_spall_df, spade_output_dir)
        else:
            self.progress_signal.emit("No enhanced spall data generated")

    def _validate_spall_shock_stress_consistency(self, enhanced_spall_df):
        """Sanity check: Verify that every trace with spall strength has shock stress"""
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit("SANITY CHECK: Validating spall strength vs shock stress consistency")
        self.progress_signal.emit("=" * 60)
        
        try:
            # Identify columns for spall strength and shock stress
            spall_strength_col = None
            shock_stress_col = None
            
            # Find spall strength column (prioritize ALPSS value, then SPADE)
            for col in ['ALPSS_Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall_Strength_GPa', 'Spall Strength (GPa)']:
                if col in enhanced_spall_df.columns:
                    spall_strength_col = col
                    break
            
            # Find shock stress column
            for col in ['Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa_Final', 'Peak_Shock_Stress_GPa']:
                if col in enhanced_spall_df.columns:
                    shock_stress_col = col
                    break
            
            if spall_strength_col is None:
                self.progress_signal.emit("⚠ WARNING: Could not find spall strength column for validation")
                return
            
            if shock_stress_col is None:
                self.progress_signal.emit("⚠ WARNING: Could not find shock stress column for validation")
                return
            
            # Convert to numeric, handling string values like "DNS"
            spall_strength = pd.to_numeric(enhanced_spall_df[spall_strength_col], errors='coerce')
            shock_stress = pd.to_numeric(enhanced_spall_df[shock_stress_col], errors='coerce')
            
            # Identify traces with valid spall strength
            # PRIMARY METHOD: Use Spall_OK column if available (most reliable)
            if 'Spall_OK' in enhanced_spall_df.columns:
                valid_spall_mask = enhanced_spall_df['Spall_OK'] == True
                self.progress_signal.emit(f"Using 'Spall_OK' column to determine valid spall traces")
            else:
                # FALLBACK: Use spall strength value and DNS classification
                # Valid spall = non-NaN, positive spall strength
                valid_spall_mask = (spall_strength.notna()) & (spall_strength > 0)
                
                # Exclude DNS cases if DNS_Classification column is available
                if 'DNS_Classification' in enhanced_spall_df.columns:
                    dns_reasons = [
                        'DNS', 'Did Not Spall', 'No Spall', 
                        'No clear peak/valley structure', 
                        'No pullback after initial rise', 
                        'No re-acceleration after pullback',
                        'Low-velocity flat signal: all 4 key points within 30 m/s and peak velocity < 100 m/s'
                    ]
                    dns_mask = enhanced_spall_df['DNS_Classification'].isin(dns_reasons)
                    valid_spall_mask = valid_spall_mask & (~dns_mask)
                    self.progress_signal.emit(f"Using 'Spall_Strength_GPa' and 'DNS_Classification' columns to determine valid spall traces")
                else:
                    self.progress_signal.emit(f"Using 'Spall_Strength_GPa' column only (DNS_Classification not available)")
            
            # Count traces with valid spall
            traces_with_spall = valid_spall_mask.sum()
            self.progress_signal.emit(f"Total traces with valid spall strength: {traces_with_spall}")
            
            if traces_with_spall == 0:
                self.progress_signal.emit("ℹ No traces with valid spall strength found - skipping validation")
                return
            
            # Check which traces with spall are missing shock stress
            missing_shock_stress_mask = valid_spall_mask & (shock_stress.isna() | (shock_stress <= 0))
            missing_count = missing_shock_stress_mask.sum()
            
            if missing_count > 0:
                self.progress_signal.emit("=" * 60)
                self.progress_signal.emit(f"⚠ WARNING: Found {missing_count} trace(s) with spall strength but missing shock stress:")
                self.progress_signal.emit("=" * 60)
                
                # Show details of problematic traces
                problematic_traces = enhanced_spall_df[missing_shock_stress_mask]
                
                if 'Filename' in problematic_traces.columns:
                    for idx, row in problematic_traces.iterrows():
                        filename = row.get('Filename', f'Row {idx}')
                        spall_val = row.get(spall_strength_col, 'N/A')
                        shock_val = row.get(shock_stress_col, 'N/A')
                        plateau_vel = row.get('Plateau Mean Velocity (m/s)', 'N/A')
                        
                        self.progress_signal.emit(f"  - {filename}:")
                        self.progress_signal.emit(f"      Spall Strength: {spall_val}")
                        self.progress_signal.emit(f"      Shock Stress: {shock_val}")
                        self.progress_signal.emit(f"      Plateau Velocity: {plateau_vel}")
                        
                        # Check if plateau velocity is available
                        if pd.notna(plateau_vel) and plateau_vel > 0:
                            self.progress_signal.emit(f"      ⚠ Plateau velocity available but shock stress not calculated!")
                        else:
                            self.progress_signal.emit(f"      ℹ No plateau velocity available (expected for shock stress calculation)")
                
                self.progress_signal.emit("=" * 60)
                self.progress_signal.emit("RECOMMENDATION: Check why shock stress was not calculated for these traces")
                self.progress_signal.emit("  Expected: If spall is detected, plateau velocity should be available,")
                self.progress_signal.emit("  and shock stress should be calculated using EOS method.")
                self.progress_signal.emit("=" * 60)
            else:
                self.progress_signal.emit("✓ PASS: All traces with spall strength have shock stress values")
            
            # Additional check: traces with shock stress but no spall (this is OK)
            valid_shock_mask = (shock_stress.notna()) & (shock_stress > 0)
            shock_without_spall = valid_shock_mask & (~valid_spall_mask)
            shock_without_spall_count = shock_without_spall.sum()
            
            if shock_without_spall_count > 0:
                self.progress_signal.emit(f"ℹ INFO: {shock_without_spall_count} trace(s) have shock stress but no spall (this is OK)")
            
            # Summary statistics
            total_traces = len(enhanced_spall_df)
            traces_with_both = ((valid_spall_mask) & (valid_shock_mask)).sum()
            
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit("SUMMARY:")
            self.progress_signal.emit(f"  Total traces: {total_traces}")
            self.progress_signal.emit(f"  Traces with spall: {traces_with_spall}")
            self.progress_signal.emit(f"  Traces with shock stress: {valid_shock_mask.sum()}")
            self.progress_signal.emit(f"  Traces with BOTH spall and shock stress: {traces_with_both}")
            self.progress_signal.emit(f"  Traces with spall but NO shock stress: {missing_count}")
            self.progress_signal.emit("=" * 60)
            
        except Exception as e:
            self.progress_signal.emit(f"⚠ ERROR during sanity check: {e}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_spall_analysis_plots(self, enhanced_spall_df, spade_output_dir):
        """Generate additional analysis plots for spall data with parameter information"""
        import pandas as pd  # Import at function level to avoid scoping issues
        import numpy as np
        
        self.progress_signal.emit("Generating spall analysis plots...")
        self.progress_signal.emit(f"[DEBUG] generate_spall_analysis_plots called with enhanced_spall_df: {enhanced_spall_df is not None}, empty: {enhanced_spall_df.empty if enhanced_spall_df is not None else 'N/A'}")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Check if we have material information for color coding
            material_cols = [col for col in enhanced_spall_df.columns if 'material' in col.lower()]
            color_col = material_cols[0] if material_cols else None
            
            # Create color mapping for materials if available
            if color_col and len(enhanced_spall_df[color_col].unique()) > 1:
                materials = enhanced_spall_df[color_col].unique()
                colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(materials)))
                color_map = dict(zip(materials, colors))
                
                # Create legend patches
                legend_patches = [mpatches.Patch(color=color_map[mat], label=mat) for mat in materials]
            else:
                color_map = None
                legend_patches = []
            
            # Plot 1: Spall Strength vs Strain Rate
            if 'Spall Strength (GPa)' in enhanced_spall_df.columns and 'Strain Rate (s^-1)' in enhanced_spall_df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for idx, row in enhanced_spall_df.iterrows():
                    spall_strength = row.get('Spall Strength (GPa)', np.nan)
                    strain_rate = row.get('Strain Rate (s^-1)', np.nan)
                    
                    if not pd.isna(spall_strength) and not pd.isna(strain_rate):
                        if color_col and color_map:
                            color = color_map.get(row[color_col], 'blue')
                            ax.scatter(strain_rate, spall_strength, c=[color], s=100, alpha=0.7)
                        else:
                            ax.scatter(strain_rate, spall_strength, c='blue', s=100, alpha=0.7)
                
                ax.set_xlabel('Strain Rate (s⁻¹)', fontsize=14)
                ax.set_ylabel('Spall Strength (GPa)', fontsize=14)
                ax.set_title('Spall Strength vs Strain Rate', fontsize=16)
                ax.grid(True, alpha=0.3)
                
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper left')
                
                plt.tight_layout()
                plot_path = os.path.join(spade_output_dir, 'spall_strength_vs_strain_rate_enhanced.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.progress_signal.emit(f"Generated enhanced spall strength vs strain rate plot")
            
            # Plot 2: Spall Strength vs Shock Stress
            if 'Spall Strength (GPa)' in enhanced_spall_df.columns and 'Peak Shock Stress (GPa)' in enhanced_spall_df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for idx, row in enhanced_spall_df.iterrows():
                    spall_strength = row.get('Spall Strength (GPa)', np.nan)
                    shock_stress = row.get('Peak Shock Stress (GPa)', np.nan)
                    
                    if not pd.isna(spall_strength) and not pd.isna(shock_stress):
                        if color_col and color_map:
                            color = color_map.get(row[color_col], 'red')
                            ax.scatter(shock_stress, spall_strength, c=[color], s=100, alpha=0.7)
                        else:
                            ax.scatter(shock_stress, spall_strength, c='red', s=100, alpha=0.7)
                
                ax.set_xlabel('Peak Shock Stress (GPa)', fontsize=14)
                ax.set_ylabel('Spall Strength (GPa)', fontsize=14)
                ax.set_title('Spall Strength vs Shock Stress', fontsize=16)
                ax.grid(True, alpha=0.3)
                
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper left')
                
                plt.tight_layout()
                plot_path = os.path.join(spade_output_dir, 'spall_strength_vs_shock_stress_enhanced.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.progress_signal.emit(f"Generated enhanced spall strength vs shock stress plot")
            
            # Plot 3: Material comparison (if material data available)
            if color_col and len(enhanced_spall_df[color_col].unique()) > 1:
                # Check if we have valid data for both plots
                has_spall_data = 'Spall Strength (GPa)' in enhanced_spall_df.columns
                has_strain_data = 'Strain Rate (s^-1)' in enhanced_spall_df.columns
                
                if has_spall_data or has_strain_data:
                    # Determine number of subplots needed
                    n_plots = sum([has_spall_data, has_strain_data])
                    if n_plots == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                        axes = [ax]
                    else:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        axes = [ax1, ax2]
                
                    plot_idx = 0
                    materials = enhanced_spall_df[color_col].unique()
                    
                    # Box plot of spall strength by material
                    if has_spall_data:
                        spall_data = [
                            enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Spall Strength (GPa)'].dropna()
                            for mat in materials
                        ]
                        valid_spall_data = [(data, mat) for data, mat in zip(spall_data, materials) if len(data) > 0]
                        if valid_spall_data:
                            spall_data_clean, materials_clean = zip(*valid_spall_data)
                            bp1 = axes[plot_idx].boxplot(spall_data_clean, labels=materials_clean, patch_artist=True)
                            for patch, mat in zip(bp1['boxes'], materials_clean):
                                patch.set_facecolor(color_map.get(mat, 'gray'))
                                patch.set_alpha(0.7)
                            axes[plot_idx].set_ylabel('Spall Strength (GPa)', fontsize=14)
                            axes[plot_idx].set_title('Spall Strength by Material', fontsize=16)
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
                    
                    # Box plot of strain rate by material
                    if has_strain_data:
                        strain_data = [
                            enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Strain Rate (s^-1)'].dropna()
                            for mat in materials
                        ]
                        valid_strain_data = [(data, mat) for data, mat in zip(strain_data, materials) if len(data) > 0]
                        if valid_strain_data:
                            strain_data_clean, materials_clean = zip(*valid_strain_data)
                            bp2 = axes[plot_idx].boxplot(strain_data_clean, labels=materials_clean, patch_artist=True)
                            for patch, mat in zip(bp2['boxes'], materials_clean):
                                patch.set_facecolor(color_map.get(mat, 'gray'))
                                patch.set_alpha(0.7)
                            axes[plot_idx].set_ylabel('Strain Rate (s^-1)', fontsize=14)
                            axes[plot_idx].set_title('Strain Rate by Material', fontsize=16)
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
                    
                    plt.tight_layout()
                    comparison_path = os.path.join(spade_output_dir, 'spall_material_comparison.png')
                    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.progress_signal.emit(f"Generated material comparison plot: {comparison_path}")
                else:
                    self.progress_signal.emit("Skipping material comparison plot - insufficient data for spall or strain rate")
            
            self.progress_signal.emit("Completed spall analysis plots generation")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating spall analysis plots: {str(e)}")
            # Continue with downstream plot generation best-effort.

        # 1. Find all ALPSS velocity files (raw, not smooth)
        velocity_files = glob.glob(os.path.join(self.output_dir, '*--velocity.csv'))
        if velocity_files:
            # 2. Read and align all velocity files by time
            dfs = []
            for f in velocity_files:
                try:
                    # Try reading with headers first
                    df = pd.read_csv(f)
                    if 'Time' in df.columns and 'Velocity' in df.columns:
                        dfs.append(df[['Time', 'Velocity']].rename(
                            columns={'Velocity': os.path.basename(f)}))
                    else:
                        # No headers, assume first column is time, second is velocity
                        df = pd.read_csv(f, header=None)
                        if df.shape[1] >= 2:
                            df.columns = ['Time', 'Velocity']
                            dfs.append(df[['Time', 'Velocity']].rename(
                                columns={'Velocity': os.path.basename(f)}))
                except Exception as e:
                    print(f"[WARNING] Could not read velocity file {f}: {e}")
                    continue
            if dfs:
                # Merge on Time
                merged = dfs[0]
                for d in dfs[1:]:
                    merged = pd.merge(merged, d, on='Time', how='outer')
                merged = merged.sort_values('Time').reset_index(drop=True)
                # Compute mean and std dev
                velocity_cols = [
    col for col in merged.columns if col != 'Time']
                
                # Convert velocity columns to numeric, coercing errors to NaN
                for col in velocity_cols:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce')
                
                # Filter to only numeric columns (exclude any that became all NaN)
                numeric_velocity_cols = [col for col in velocity_cols 
                                        if merged[col].dtype in ['float64', 'int64'] 
                                        and not merged[col].isna().all()]
                
                if numeric_velocity_cols:
                    merged['Mean Velocity (m/s)'] = merged[numeric_velocity_cols].mean(axis=1)
                    merged['Std Dev Velocity (m/s)'] = merged[numeric_velocity_cols].std(axis=1)
                else:
                    # No valid numeric columns found
                    merged['Mean Velocity (m/s)'] = np.nan
                    merged['Std Dev Velocity (m/s)'] = np.nan
                    self.progress_signal.emit("Warning: No valid numeric velocity columns found for mean calculation")

                # Create a properly named file that matches SPADE's expected pattern
                # Use a generic name that will work with the plotting function
                mean_vel_file = os.path.join(
    spade_output_dir, 'combined_mean_raw_velocity.csv')
                merged[['Time',
    'Mean Velocity (m/s)',
    'Std Dev Velocity (m/s)']].to_csv(mean_vel_file,
     index=False)

                # 3. Plot combined mean velocity - create a simple plot since
                # SPADE's function expects specific naming
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    time_data = merged['Time']
                    mean_velocity = merged['Mean Velocity (m/s)']
                    velocity_std = merged['Std Dev Velocity (m/s)']

                    # Plot mean line
                    ax.plot(
    time_data,
    mean_velocity,
    'b-',
    linewidth=2,
     label='Mean Velocity')

                    # Plot shaded uncertainty if available
                    if not velocity_std.isna().all():
                        ax.fill_between(time_data, mean_velocity - velocity_std, mean_velocity + velocity_std,
                                      alpha=0.3, color='blue', label='±1σ')

                    ax.set_xlabel('Time (ns)', fontsize=14)
                    ax.set_ylabel(
    'Mean Free Surface Velocity (m/s)', fontsize=14)
                    ax.set_title('Combined Mean Velocity Traces', fontsize=16)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 700)

                    plt.tight_layout()
                    # Save to both spade_output_dir and main output_dir
                    # NOTE: this is the "overall mean" plot (not the material-subplot plot).
                    # Use a different name to avoid clobbering the by-material figure.
                    spade_plot_path = os.path.join(spade_output_dir, 'combined_mean_velocity_overall.png')
                    main_plot_path = os.path.join(self.output_dir, 'combined_mean_velocity_overall.png')

                    # Save atomically to avoid partially-written (blank) PNGs.
                    # IMPORTANT: the temp file must still end with a supported extension (e.g. ".png"),
                    # otherwise matplotlib infers format="tmp" and raises: "Format 'tmp' is not supported".
                    for final_path in (spade_plot_path, main_plot_path):
                        _root, _ext = os.path.splitext(final_path)
                        tmp_path = _root + ".tmp" + (_ext if _ext else ".png")
                        try:
                            fig.savefig(tmp_path, dpi=300, bbox_inches='tight', format='png')
                            os.replace(tmp_path, final_path)
                        finally:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                    plt.close(fig)
                    self.progress_signal.emit(
                        f"Generated combined_mean_velocity_overall.png in both locations")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate combined_mean_velocity.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)

        # --- ENHANCED: Plot all smoothed velocity traces with material and waveplate angle information ---
        # Accept either classic smoothed file or the smoothed-with-uncertainty file (used in smart selection)
        smoothed_files_velocity = glob.glob(os.path.join(self.output_dir, '*--velocity--smooth.csv'))
        smoothed_files_uncert = glob.glob(os.path.join(self.output_dir, '*--vel-smooth-with-uncert.csv'))
        smoothed_files = list(set(smoothed_files_velocity + smoothed_files_uncert))

        # Report combined plotting file availability
        self.progress_signal.emit(
            f"=== Enhanced Combined Velocity Plotting ===")
        self.progress_signal.emit(
            f"Found {len(smoothed_files)} velocity files for combined plotting")

        if len(smoothed_files) != len(self.successful_files):
            missing_plot_files = []
            for input_file in self.successful_files:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                # Check primary and fallback patterns
                expected_primary = os.path.join(self.output_dir, f"{base_name}--velocity--smooth.csv")
                expected_fallback = os.path.join(self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                if not (os.path.exists(expected_primary) or os.path.exists(expected_fallback)):
                    missing_plot_files.append(base_name)

            if missing_plot_files:
                self.progress_signal.emit(
                    f"Missing velocity files for plotting: {len(missing_plot_files)}")
                for missing_file in missing_plot_files:
                    self.progress_signal.emit(
                        f"❌ Missing: {missing_file} (no smoothed CSV found: --velocity--smooth.csv or --vel-smooth-with-uncert.csv)")

        

        # 4. Spall Strength vs. Strain Rate and Shock Stress (only if spall analysis is enabled)
        # Check if spall analysis is enabled (from config flag)
        spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False) or \
                                 self.spade_params.get('experiment_spall_analysis', False)
        
        if spall_analysis_enabled:
            # Define paths for summary files
            enhanced_summary_csv = os.path.join(spade_output_dir, self._get_summary_filename())
            summary_csv = os.path.join(spade_output_dir, 'spall_summary.csv')
            
            # Try to use enhanced_spall_df if available (passed to this function), 
            # otherwise try to load from enhanced_spall_summary.csv or spall_summary.csv
            if enhanced_spall_df is not None and not enhanced_spall_df.empty:
                # Use the enhanced dataframe that's already loaded
                summary_df = enhanced_spall_df.copy()
                summary_df = self.refresh_material_column(summary_df)
                # Ensure required columns exist with correct names for SPADE plotting function
                if 'Spall_Strength_GPa_Final' in summary_df.columns:
                    summary_df['Spall Strength (GPa)'] = summary_df['Spall_Strength_GPa_Final']
                if 'Spall_Strength_Uncertainty_GPa_Final' in summary_df.columns:
                    summary_df['Spall Strength Uncertainty (GPa)'] = summary_df['Spall_Strength_Uncertainty_GPa_Final']
                if 'Strain_Rate_s1_Final' in summary_df.columns:
                    summary_df['Strain Rate (s^-1)'] = summary_df['Strain_Rate_s1_Final']
                if 'Peak_Shock_Stress_GPa_Final' in summary_df.columns:
                    summary_df['Peak Shock Stress (GPa)'] = summary_df['Peak_Shock_Stress_GPa_Final']
                if 'Peak_Shock_Stress_Uncertainty_GPa_Final' in summary_df.columns:
                    summary_df['Peak Shock Stress Uncertainty (GPa)'] = summary_df['Peak_Shock_Stress_Uncertainty_GPa_Final']
            elif os.path.exists(enhanced_summary_csv):
                summary_df = pd.read_csv(enhanced_summary_csv)
                # Master CSV is written with standardized names; expose both standardized
                # and legacy spellings in memory so the mapping below (and any GUI plot
                # consumer) resolves regardless of which build wrote the file.
                summary_df = normalize_summary_columns(summary_df)
                summary_df = self.refresh_material_column(summary_df)
                # Map column names if needed
                if 'Spall_Strength_GPa_Final' in summary_df.columns:
                    summary_df['Spall Strength (GPa)'] = summary_df['Spall_Strength_GPa_Final']
                if 'Spall_Strength_Uncertainty_GPa_Final' in summary_df.columns:
                    summary_df['Spall Strength Uncertainty (GPa)'] = summary_df['Spall_Strength_Uncertainty_GPa_Final']
                if 'Strain_Rate_s1_Final' in summary_df.columns:
                    summary_df['Strain Rate (s^-1)'] = summary_df['Strain_Rate_s1_Final']
                if 'Peak_Shock_Stress_GPa_Final' in summary_df.columns:
                    summary_df['Peak Shock Stress (GPa)'] = summary_df['Peak_Shock_Stress_GPa_Final']
                if 'Peak_Shock_Stress_Uncertainty_GPa_Final' in summary_df.columns:
                    summary_df['Peak Shock Stress Uncertainty (GPa)'] = summary_df['Peak_Shock_Stress_Uncertainty_GPa_Final']
            elif os.path.exists(summary_csv):
                summary_df = pd.read_csv(summary_csv)
                summary_df = self.refresh_material_column(summary_df)
            else:
                self.progress_signal.emit(f"[WARNING] No spall summary file found (neither {self._get_summary_filename()} nor spall_summary.csv)")
                summary_df = None
            
            if summary_df is not None and not summary_df.empty:
                # Check if we already have enhanced data (from enhanced_spall_df or enhanced_spall_summary.csv)
                # If so, skip the enhancement step and go straight to plotting
                already_enhanced = ('Spall_Strength_GPa_Final' in summary_df.columns or 
                                  'ALPSS_Spall_Strength_GPa' in summary_df.columns or
                                  'Spall Strength (GPa)' in summary_df.columns)
                
                if not already_enhanced:
                    # Use parameter file data passed to AnalysisThread
                    param_data = self.param_data if self.param_data else {}
                    
                    # Enhance SPADE summary with ALPSS results and additional calculations
                    enhanced_summary = []
                    
                    for idx, row in summary_df.iterrows():
                        enhanced_row = row.copy()
                        filename = row.get('Filename', '')
                        
                        # Get material-specific properties if available
                        base_name = filename
                        for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                            if base_name.endswith(suffix):
                                base_name = base_name[:-len(suffix)]
                                break
                    
                        # Try to get material from parameter file, falling back to the IGSN map
                        matched_param = self.get_param_data_for_file(base_name)
                        sample_material = self.resolve_sample_material(base_name, matched_param)
                    
                        # Get material-specific properties from config first, then database
                        mat_props = self.get_material_properties_from_config(sample_material)
                        # Parameter file can override (highest priority)
                        if matched_param:
                            density = matched_param.get('Density_kg_m3', mat_props['density'])
                            acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed'])
                        else:
                            density = mat_props['density']
                            acoustic_velocity = mat_props['bulk_wave_speed']
                        # Add material information to enhanced row
                        enhanced_row['Material'] = sample_material
                        enhanced_row['Density_kg_m3'] = density
                        enhanced_row['Acoustic_Velocity_m_s'] = acoustic_velocity
                        enhanced_row['Material_Found_In_Database'] = mat_props['material_found']
                        enhanced_row['Material_Properties_Source'] = mat_props.get('source', 'unknown')
                        enhanced_row['Material_Status'] = 'OK' if mat_props['material_found'] or density is not None else 'Mat not found'
                    
                        # Only merge ALPSS results if spall_calculation is enabled
                        # When spall_calculation is "no", we only use SPADE data
                        use_alpss_data = (self.alpss_params.get('spall_calculation', 'yes').lower() == 'yes')
                        
                        if not use_alpss_data:
                            # Skip ALPSS data - only use SPADE results
                            pass  # Continue without ALPSS data
                        
                        # Try to find corresponding ALPSS results file (only if ALPSS spall calculation is enabled)
                        alpss_results_file = os.path.join(self.output_dir, f"{filename}--results.csv")
                        if use_alpss_data and os.path.exists(alpss_results_file):
                            try:
                                # Read ALPSS results
                                alpss_results = pd.read_csv(alpss_results_file, header=None, names=['Name', 'Value'])
                                alpss_dict = dict(zip(alpss_results['Name'], alpss_results['Value']))
                            
                                # Add ALPSS results to enhanced summary
                                # ALPSS saves spall strength in Pa, convert to GPa
                                alpss_spall_pa = alpss_dict.get('Spall Strength', np.nan)
                                if pd.notna(alpss_spall_pa):
                                    try:
                                        enhanced_row['ALPSS_Spall_Strength_GPa'] = float(alpss_spall_pa) / 1e9
                                    except (ValueError, TypeError):
                                        enhanced_row['ALPSS_Spall_Strength_GPa'] = np.nan
                                else:
                                    enhanced_row['ALPSS_Spall_Strength_GPa'] = np.nan
                                
                                alpss_spall_unc_pa = alpss_dict.get('Spall Strength Uncertainty', np.nan)
                                if pd.notna(alpss_spall_unc_pa):
                                    try:
                                        enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = float(alpss_spall_unc_pa) / 1e9
                                    except (ValueError, TypeError):
                                        enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = np.nan
                                else:
                                    enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = np.nan
                                enhanced_row['ALPSS_Strain_Rate_s1'] = alpss_dict.get('Strain Rate', np.nan)
                                enhanced_row['ALPSS_Strain_Rate_Uncertainty_s1'] = alpss_dict.get('Strain Rate Uncertainty', np.nan)
                                # Note: ALPSS no longer calculates Peak Shock Stress (removed as part of cleanup)
                                enhanced_row['ALPSS_Peak_Velocity_Uncertainty_ms'] = alpss_dict.get('Peak Velocity Uncertainty', np.nan)
                                enhanced_row['ALPSS_Pullback_Velocity_Uncertainty_ms'] = alpss_dict.get('Pullback Velocity Uncertainty', np.nan)
                                enhanced_row['ALPSS_Velocity_at_Max_Compression_ms'] = alpss_dict.get('Velocity at Max Compression', np.nan)
                                enhanced_row['ALPSS_Velocity_at_Max_Tension_ms'] = alpss_dict.get('Velocity at Max Tension', np.nan)
                                enhanced_row['ALPSS_Velocity_at_Recompression_ms'] = alpss_dict.get('Velocity at Recompression', np.nan)
                                enhanced_row['ALPSS_Time_at_Max_Compression_ns'] = alpss_dict.get('Time at Max Compression', np.nan)
                                enhanced_row['ALPSS_Time_at_Max_Tension_ns'] = alpss_dict.get('Time at Max Tension', np.nan)
                                enhanced_row['ALPSS_Time_at_Recompression_ns'] = alpss_dict.get('Time at Recompression', np.nan)
                                enhanced_row['ALPSS_Carrier_Frequency_Hz'] = alpss_dict.get('Carrier Frequency', np.nan)
                                enhanced_row['ALPSS_Signal_Start_Time_s'] = alpss_dict.get('Signal Start Time', np.nan)
                                enhanced_row['ALPSS_Smoothing_Characteristic_Time_s'] = alpss_dict.get('Smoothing Characteristic Time', np.nan)
                            
                            except Exception as e:
                                print(f"[WARNING] Could not read ALPSS results for {filename}: {e}")
                    
                        # Use SPADE values (5-segment method) for spall strength - this is the authoritative source
                        # ALPSS values are kept for reference but not used for final calculations
                        spade_spall = row.get('Spall_Strength_GPa', row.get('Spall Strength (GPa)', np.nan))
                        # Convert DNS strings to NaN
                        if isinstance(spade_spall, str) and spade_spall.upper() in ['DNS', 'NO SPALL', 'DID NOT SPALL']:
                            spade_spall = np.nan
                        enhanced_row['Spall_Strength_GPa_Final'] = pd.to_numeric(spade_spall, errors='coerce')
                        enhanced_row['Spall_Strength_Uncertainty_GPa_Final'] = pd.to_numeric(
                            row.get('Spall_Strength_Unc_GPa', row.get('Spall_Strength_Uncertainty_GPa', 
                            row.get('Spall Strength Uncertainty (GPa)', np.nan))), errors='coerce')
                        
                        # Use SPADE strain rate (5-segment method) - this is the authoritative source
                        # ALPSS strain rate is kept for reference but not used for final calculations
                        spade_strain = row.get('Spall_StrainRate_s^-1', row.get('Strain_Rate_s^-1', 
                                              row.get('Strain_Rate_s1', row.get('Strain Rate (s^-1)', np.nan))))
                        enhanced_row['Strain_Rate_s1_Final'] = pd.to_numeric(spade_strain, errors='coerce')
                        enhanced_row['Strain_Rate_Uncertainty_s1_Final'] = pd.to_numeric(
                            row.get('Strain_Rate_Uncertainty_s^-1', row.get('Strain Rate Uncertainty (s^-1)', np.nan)), errors='coerce')
                        
                        # Peak Shock Stress comes ONLY from HELIX EOS calculation (no ALPSS fallback)
                        peak_shock_stress = row.get('Peak Shock Stress (GPa)', np.nan)
                        if pd.isna(peak_shock_stress):
                            # Try alternative column names
                            for alt_col in ['Peak_Shock_Stress_GPa', 'Peak Shock Stress', 'Shock Stress (GPa)']:
                                if alt_col in row:
                                    peak_shock_stress = row.get(alt_col, np.nan)
                                    if pd.notna(peak_shock_stress):
                                        break
                        
                        enhanced_row['Peak_Shock_Stress_GPa_Final'] = peak_shock_stress
                        
                        # Peak Shock Stress Uncertainty
                        peak_shock_stress_unc = row.get('Peak Shock Stress Uncertainty (GPa)', np.nan)
                        if pd.isna(peak_shock_stress_unc):
                            for alt_col in ['Peak_Shock_Stress_Uncertainty_GPa', 'Peak Shock Stress Uncertainty']:
                                if alt_col in row:
                                    peak_shock_stress_unc = row.get(alt_col, np.nan)
                                    if pd.notna(peak_shock_stress_unc):
                                        break
                        enhanced_row['Peak_Shock_Stress_Uncertainty_GPa_Final'] = peak_shock_stress_unc
                    
                        enhanced_summary.append(enhanced_row)
                    
                    # Create enhanced summary DataFrame (outside the loop)
                    if enhanced_summary:
                        enhanced_summary_df = pd.DataFrame(enhanced_summary)
                    else:
                        # If no enhanced summary was created, use summary_df as fallback
                        self.progress_signal.emit("[WARNING] No enhanced summary data created, using original summary_df")
                        enhanced_summary_df = summary_df.copy()
                    
                    # Reorganize columns in the specified order
                    # Order: filename, metadata, experimental params, spall analysis results, then ALPSS values
                    priority_columns = [
                        # 1. Basic identifiers
                        'Filename',
                        'Timestamp',
                        'Exp_ID',
                        # 2. Flyer information
                        'Flyer_material',
                        'Flyer_ID',
                        'Flyer_Thickness (um)',
                        'flyer_thickness',  # Alternative naming
                        # 3. Sample information
                        'Sample_ID',
                        'Material',
                        'Density_kg_m3',
                        'density',  # Alternative naming
                        'Acoustic_Velocity_m_s',
                        'acoustic_velocity',  # Alternative naming
                        # 4. Experimental parameters
                        'Spacing (um)',
                        'Waveplate_Angle (Degrees)',
                        'PDV_Target_Wavelength (m)',
                        'PDV_Target_Power (dBm)',
                        'PDV_Ref_Wavelength (m)',
                        'PDV_Ref_Power (dBm)',
                        'PDV_Return_Power (dBm)',
                        'Flyer_Row',
                        'Flyer_Column',
                        'Flyer_X_Position_Desired (mm)',
                        'Flyer_Y_Position_desired (mm)',
                        'Flyer_X_Position_Corrected (mm)',
                        'Flyer_Y_Position_Corrected (mm)',
                        'Laser_Ref_Energy (mJ)',
                        'Laser_Target_Energy (mJ)',
                        'Exp_Time (seconds)',
                        'Notes',
                        # 5. SPADE analysis results
                        'DNS_Classification',
                        'Processing_Status',
                        'Analysis_Notes',
                        'Spall_OK',
                        'First_Maxima_m_s',
                        'Minima_m_s',
                        'Second_Maxima_m_s',
                        'Pullback_Velocity_m_s',
                        'Pullback_Velocity_Unc_m_s',
                        'Plateau Mean Velocity (m/s)',
                        'Peak Shock Stress (GPa)',
                        'Peak Shock Stress Uncertainty (GPa)',
                        'Spall_Strength_GPa',
                        'Spall_Strength_Unc_GPa',
                        'Spall_StrainRate_s^-1',
                        'Spall_StrainRate_UNCERTAINITY',  # As specified by user
                        'Strain_Rate_Uncertainty_s^-1',  # Alternative naming
                        'Strain Rate Uncertainty (s^-1)',  # Alternative naming
                        'Strain_Rate_Unc_s^-1',  # Alternative naming
                        'StrainRate_Unc_s^-1',   # Alternative naming
                        # 5b. Derived shock-front diagnostics (computed in detect_dns_and_process_spall)
                        'Peak_Shock_Time_ns',
                        'RiseTime_ArrivalToPeak_ns',
                        'RiseTime_80_20_ns',
                        'RiseTime_90_10_ns',
                        'RiseTime_MaxSlope_ns',
                        'PlasticStrainRate_80_20_s^-1',
                        'PlasticStrainRate_90_10_s^-1',
                        'PlasticStrainRate_MaxSlope_s^-1',
                        'Compressive_StrainRate_Avg_s^-1',
                        'Compressive_StrainRate_Ufs_s^-1',
                        'Shock_Velocity_Us_m_s',
                        'Shock_Front_Width_um',
                        # 6. HEL analysis results
                        'hel_ok',
                        'hel_strength_gpa',
                        'hel_uncertainty_gpa',
                        'hel_strain_rate_s^-1',
                        'hel_segment_time_ns',
                        'hel_consecutive_points',
                        'free_surface_velocity_ms',
                        # Note: ALPSS columns will be added at the end automatically
                    ]

                    # Merge HEL data from velocity_shots_summary.csv
                    velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
                    if os.path.exists(velocity_shots_path):
                        try:
                            vel_df = pd.read_csv(velocity_shots_path)
                            hel_cols = [c for c in [
                                'hel_ok', 'hel_strength_gpa', 'hel_uncertainty_gpa',
                                'hel_strain_rate_s^-1', 'hel_segment_time_ns',
                                'hel_consecutive_points', 'free_surface_velocity_ms'
                            ] if c in vel_df.columns]
                            if hel_cols and 'file_name' in vel_df.columns:
                                hel_merge_df = vel_df[['file_name'] + hel_cols].rename(columns={'file_name': 'Filename'})
                                existing_hel = [c for c in hel_cols if c in enhanced_summary_df.columns]
                                if existing_hel:
                                    enhanced_summary_df = enhanced_summary_df.drop(columns=existing_hel)
                                enhanced_summary_df = enhanced_summary_df.merge(hel_merge_df, on='Filename', how='left')
                        except Exception as e:
                            self.progress_signal.emit(f"Warning: Could not merge HEL data: {e}")

                    # Get all columns from the DataFrame
                    all_columns = list(enhanced_summary_df.columns)

                    # Start with priority columns (only those that exist)
                    reordered_columns = [col for col in priority_columns if col in all_columns]

                    # Separate remaining columns into ALPSS and other columns
                    remaining_columns = [col for col in all_columns if col not in reordered_columns]

                    # Separate ALPSS columns (they start with 'ALPSS_')
                    alpss_columns = [col for col in remaining_columns if col.startswith('ALPSS_')]
                    other_columns = [col for col in remaining_columns if not col.startswith('ALPSS_')]

                    # Add other columns first, then ALPSS columns at the end
                    reordered_columns.extend(other_columns)
                    reordered_columns.extend(sorted(alpss_columns))  # Sort ALPSS columns for consistency

                    # Reorder the DataFrame
                    enhanced_summary_df = enhanced_summary_df[reordered_columns]

                    # Save enhanced summary (single consolidated master file) with
                    # standardized column names; keep the in-memory df unrenamed for the
                    # GUI plotting/consumers that run after this.
                    enhanced_summary_path = os.path.join(spade_output_dir, self._get_summary_filename())
                    standardize_summary_columns(enhanced_summary_df.copy()).to_csv(enhanced_summary_path, index=False)
                    self._save_run_config(spade_output_dir)

                    # Check if ALPSS data was included
                    use_alpss_data = (self.alpss_params.get('spall_calculation', 'yes').lower() == 'yes')
                    if use_alpss_data:
                        self.progress_signal.emit("Enhanced SPADE summary with ALPSS results and uncertainty calculations")
                    else:
                        self.progress_signal.emit("Enhanced SPADE summary (ALPSS data excluded - using SPADE data only)")
                    
                    # Ensure Material column exists for plotting
                    if 'Material' not in summary_df.columns:
                        summary_df['Material'] = enhanced_summary_df.get('Material', 'Unknown')
                    # Fill NaN Material values with 'Unknown'
                    if 'Material' in summary_df.columns:
                        summary_df['Material'] = summary_df['Material'].fillna('Unknown')
            else:
                self.progress_signal.emit("Using already-enhanced spall data for plotting")
                # Set enhanced_summary_df to summary_df for consistency
                enhanced_summary_df = summary_df.copy()
                
                # Debug: Log available columns
                self.progress_signal.emit(f"DEBUG: DataFrame columns: {list(summary_df.columns)}")
                self.progress_signal.emit(f"DEBUG: DataFrame shape: {summary_df.shape}")
                
                # Ensure required columns exist for plotting (handle both enhanced and non-enhanced data)
                # Check for Spall Strength column (try multiple possible names, prioritize ALPSS)
                if 'Spall Strength (GPa)' not in summary_df.columns:
                    # Priority 1: Use Final values (which use SPADE 5-segment method)
                    if 'Spall_Strength_GPa_Final' in summary_df.columns:
                        summary_df['Spall Strength (GPa)'] = pd.to_numeric(summary_df['Spall_Strength_GPa_Final'], errors='coerce')
                        self.progress_signal.emit("Using SPADE 5-segment method Spall Strength values")
                    # Priority 2: Use SPADE column directly
                    elif 'Spall_Strength_GPa' in summary_df.columns:
                        summary_df['Spall Strength (GPa)'] = pd.to_numeric(summary_df['Spall_Strength_GPa'], errors='coerce')
                        self.progress_signal.emit("Using SPADE Spall Strength values")
                    # ALPSS fallback removed - only use SPADE data
                    # (ALPSS data is only included when spall_calculation is enabled)
                    # Priority 3: Use SPADE's original column (convert DNS strings to NaN)
                    elif 'Spall_Strength_GPa' in summary_df.columns:
                        spall_col = summary_df['Spall_Strength_GPa'].copy()
                        # Replace "DNS" and other non-numeric strings with NaN
                        spall_col = spall_col.replace(['DNS', 'NO SPALL', 'Did Not Spall', 'dns', 'no spall'], np.nan)
                        summary_df['Spall Strength (GPa)'] = pd.to_numeric(spall_col, errors='coerce')
                        self.progress_signal.emit("Using SPADE Spall Strength values (converted DNS to NaN)")
                    else:
                        self.progress_signal.emit("WARNING: Could not find Spall Strength column")
                
                # Check for Spall Strength Uncertainty
                if 'Spall Strength Uncertainty (GPa)' not in summary_df.columns:
                        if 'Spall_Strength_Uncertainty_GPa_Final' in summary_df.columns:
                            summary_df['Spall Strength Uncertainty (GPa)'] = summary_df['Spall_Strength_Uncertainty_GPa_Final']
                        elif 'Spall_Strength_Uncertainty_GPa' in summary_df.columns:
                            summary_df['Spall Strength Uncertainty (GPa)'] = pd.to_numeric(summary_df['Spall_Strength_Uncertainty_GPa'], errors='coerce')
                        # ALPSS fallback removed - only use SPADE data
                    
                # Check for Strain Rate column (prioritize ALPSS)
                if 'Strain Rate (s^-1)' not in summary_df.columns:
                    # Priority 1: Use Final values (which use SPADE 5-segment method)
                    if 'Strain_Rate_s1_Final' in summary_df.columns:
                        summary_df['Strain Rate (s^-1)'] = pd.to_numeric(summary_df['Strain_Rate_s1_Final'], errors='coerce')
                    # Priority 2: Use SPADE column names directly
                    elif 'Spall_StrainRate_s^-1' in summary_df.columns:
                        summary_df['Strain Rate (s^-1)'] = pd.to_numeric(summary_df['Spall_StrainRate_s^-1'], errors='coerce')
                    elif 'Strain_Rate_s^-1' in summary_df.columns:
                        summary_df['Strain Rate (s^-1)'] = pd.to_numeric(summary_df['Strain_Rate_s^-1'], errors='coerce')
                    elif 'Strain_Rate_s1' in summary_df.columns:
                        summary_df['Strain Rate (s^-1)'] = pd.to_numeric(summary_df['Strain_Rate_s1'], errors='coerce')
                    # ALPSS fallback removed - only use SPADE data
                    else:
                        self.progress_signal.emit("WARNING: Could not find Strain Rate column")
                    
                # Check for Peak Shock Stress column (no ALPSS fallback - only EOS calculation)
                if 'Peak Shock Stress (GPa)' not in summary_df.columns:
                    if 'Peak_Shock_Stress_GPa_Final' in summary_df.columns:
                        summary_df['Peak Shock Stress (GPa)'] = summary_df['Peak_Shock_Stress_GPa_Final']
                    elif 'Peak_Shock_Stress_GPa' in summary_df.columns:
                        summary_df['Peak Shock Stress (GPa)'] = pd.to_numeric(summary_df['Peak_Shock_Stress_GPa'], errors='coerce')
                    
                # Check for Peak Shock Stress Uncertainty
                if 'Peak Shock Stress Uncertainty (GPa)' not in summary_df.columns:
                    if 'Peak_Shock_Stress_Uncertainty_GPa_Final' in summary_df.columns:
                        summary_df['Peak Shock Stress Uncertainty (GPa)'] = summary_df['Peak_Shock_Stress_Uncertainty_GPa_Final']
                    elif 'Peak_Shock_Stress_Uncertainty_GPa' in summary_df.columns:
                        summary_df['Peak Shock Stress Uncertainty (GPa)'] = pd.to_numeric(summary_df['Peak_Shock_Stress_Uncertainty_GPa'], errors='coerce')
                    
                # Ensure Material column exists for plotting
                if 'Material' not in summary_df.columns:
                    # Try to get from parameter data or set default
                    summary_df['Material'] = 'Unknown'
                    
                # Debug: Check data availability
                if 'Spall Strength (GPa)' in summary_df.columns:
                    valid_spall = summary_df['Spall Strength (GPa)'].notna().sum()
                    self.progress_signal.emit(f"DEBUG: Valid Spall Strength values: {valid_spall} out of {len(summary_df)}")
                    if valid_spall == 0:
                        # Try to see if ALPSS values exist but weren't used
                        if 'ALPSS_Spall_Strength_GPa' in summary_df.columns:
                            alpss_valid = summary_df['ALPSS_Spall_Strength_GPa'].notna().sum()
                            self.progress_signal.emit(f"DEBUG: ALPSS Spall Strength values available: {alpss_valid} out of {len(summary_df)}")
                            # Note: ALPSS values are for reference only - we use SPADE 5-segment method values
                            # SPADE values should already be in 'Spall Strength (GPa)' column from above
                if 'Strain Rate (s^-1)' in summary_df.columns:
                    valid_strain = summary_df['Strain Rate (s^-1)'].notna().sum()
                    self.progress_signal.emit(f"DEBUG: Valid Strain Rate values: {valid_strain} out of {len(summary_df)}")
                    
                # Ensure Material column exists and is not all NaN
                if 'Material' not in summary_df.columns:
                    summary_df['Material'] = 'Unknown'
                    self.progress_signal.emit("DEBUG: Material column was missing, set to 'Unknown'")
                else:
                    # Check if Material column has any non-NaN values
                    material_valid = summary_df['Material'].notna().sum()
                    if material_valid == 0:
                        # All Material values are NaN, set to 'Unknown'
                        summary_df['Material'] = 'Unknown'
                        self.progress_signal.emit("DEBUG: Material column was all NaN, set to 'Unknown'")
                    else:
                        # Fill remaining NaN Material values with 'Unknown'
                            summary_df['Material'] = summary_df['Material'].fillna('Unknown')
                            self.progress_signal.emit(f"DEBUG: Material values: {summary_df['Material'].value_counts().to_dict()}")
            
            # Generate plots (runs regardless of whether data was already enhanced or not)
            # Check how many rows have BOTH Spall Strength AND Strain Rate (required for plotting)
            if 'Spall Strength (GPa)' in summary_df.columns and 'Strain Rate (s^-1)' in summary_df.columns:
                both_valid = summary_df[['Spall Strength (GPa)', 'Strain Rate (s^-1)', 'Material']].notna().all(axis=1).sum()
                self.progress_signal.emit(f"DEBUG: Rows with BOTH Spall Strength AND Strain Rate AND Material: {both_valid} out of {len(summary_df)}")
                if both_valid == 0:
                    self.progress_signal.emit("WARNING: No rows have both Spall Strength and Strain Rate values. Plots will be empty.")
            
            # Log available outputs for spall analysis
            self.progress_signal.emit("Available outputs (Spall Analysis):")
            self.progress_signal.emit(f"  - {self._get_summary_filename()}: Complete results with spall, HEL, and ALPSS data")
            self.progress_signal.emit("  - spall_vs_strain_rate.png: Spall strength vs strain rate plot")
            self.progress_signal.emit("  - spall_vs_shock_stress.png: Spall strength vs shock stress plot")
            self.progress_signal.emit("  - shock_stress_vs_laser_energy.png: Peak shock stress vs laser energy")
            self.progress_signal.emit("  - Individual ALPSS files: *--results.csv, *--velocity.csv, etc.")
            self.progress_signal.emit("  - Individual SPADE analysis plots (if enabled)")
            self.progress_signal.emit("  - ALPSS velocity files: 4 columns (Time, Velocity, Uncertainty, Velocity+Uncertainty)")
            self.progress_signal.emit("  - SPADE uses ALPSS uncertainty data for error bars and analysis")
            
            # Spall Strength vs. Strain Rate (custom plot matching HEL format)
            try:
                generate_spall_vs_strain_rate_plot(summary_df, spade_output_dir, progress_callback=self.progress_signal.emit)
            except Exception as e:
                msg = f"[WARNING] Failed to generate spall_vs_strain_rate.png: {e}"
                print(msg)
                self.progress_signal.emit(msg)
            
            # Spall Strength vs. Strain Rate by Material (Cu/Al subplots with flyer thickness color coding)
            try:
                generate_spall_vs_strain_rate_by_material_subplots(summary_df, spade_output_dir, progress_callback=self.progress_signal.emit)
            except Exception as e:
                msg = f"[WARNING] Failed to generate spall_vs_strain_rate_by_material_subplots.png: {e}"
                print(msg)
                self.progress_signal.emit(msg)
            # Spall Strength vs. Shock Stress (custom plot matching HEL format)
            try:
                self.generate_spall_vs_shock_stress_plot(summary_df, spade_output_dir)
            except Exception as e:
                msg = f"[WARNING] Failed to generate spall_vs_shock_stress.png: {e}"
                print(msg)
                self.progress_signal.emit(msg)
            try:
                # This plot reads from velocity_shots_summary.csv under spade_output_dir
                self.generate_shock_stress_vs_laser_energy_plot(spade_output_dir)
                self.progress_signal.emit("✅ Generated shock_stress_vs_laser_energy.png")
            except Exception as e:
                msg = f"[WARNING] Failed to generate shock_stress_vs_laser_energy.png: {e}"
                print(msg)
                self.progress_signal.emit(msg)

            # Generate combined velocity traces plot with subplots per material, color-coded by laser energy
            # NOTE: This plot does NOT require spall summary data - it only needs velocity files and parameter data
            # So we generate it regardless of whether summary_df exists
            self.progress_signal.emit("[DEBUG] Starting combined_mean_velocity.png generation...")
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                from matplotlib.cm import viridis
                from collections import defaultdict
                
                # Reuse SPADE's alignment results to avoid per-trace alignment searches.
                # This is MUCH faster and ensures consistency with the rest of the pipeline.
                t0_map = {}
                aligned_ok_map = {}
                try:
                    vs_path = os.path.join(spade_output_dir, "velocity_shots_summary.csv")
                    if os.path.exists(vs_path):
                        vs = pd.read_csv(vs_path, usecols=["file_name", "t0_ns", "aligned_ok"])
                        for _, r in vs.iterrows():
                            bn = str(r.get("file_name", "")).strip()
                            if not bn:
                                continue
                            t0_map[bn] = r.get("t0_ns", np.nan)
                            aligned_ok_map[bn] = bool(r.get("aligned_ok", False))
                except Exception:
                    # If anything goes wrong, we fall back to the old alignment logic below.
                    t0_map = {}
                    aligned_ok_map = {}

                # Find velocity files (avoid scanning SPADE_analysis and plot folders)
                # Accept both --vel-smooth-with-uncert.csv and --velocity--smooth.csv patterns
                vel_files_by_base = {}  # base_name -> best path
                for root, dirs, files in os.walk(self.output_dir):
                    # prune directories to reduce work
                    dirs[:] = [
                        d for d in dirs
                        if d not in {"SPADE_analysis", "__pycache__"} and not d.endswith("_plots")
                    ]
                    for file in files:
                        # Use EXACT same pattern as working function: only --vel-smooth-with-uncert.csv
                        if not file.endswith("--vel-smooth-with-uncert.csv"):
                            continue
                        full_path = os.path.join(root, file)
                        base_name = os.path.splitext(file)[0]
                        for suffix in ['--vel-smooth-with-uncert', '--velocity--smooth', '--vel-smooth', '--velocity', '--vel']:
                            if base_name.endswith(suffix):
                                base_name = base_name[:-len(suffix)]
                                break
                        # Prefer shortest path (usually top-level output dir) to avoid duplicates
                        prev = vel_files_by_base.get(base_name)
                        if prev is None or len(full_path) < len(prev):
                            vel_files_by_base[base_name] = full_path
                vel_files = list(vel_files_by_base.values())
                
                self.progress_signal.emit(f"[DEBUG] Found {len(vel_files)} velocity files for combined_mean_velocity.png")
                if vel_files:
                    self.progress_signal.emit(f"Generating combined_mean_velocity.png from {len(vel_files)} velocity files...")
                    
                    # Organize traces by material and energy
                    traces_by_material = defaultdict(list)  # material -> list of (vel_file, energy, base_name)
                    
                    for vel_file in vel_files:
                        try:
                            # Extract base name for parameter lookup
                            filename = os.path.basename(vel_file)
                            base_name = os.path.splitext(filename)[0]
                            # Remove velocity file suffix (EXACT same as working function)
                            if base_name.endswith('--vel-smooth-with-uncert'):
                                base_name = base_name[:-len('--vel-smooth-with-uncert')]
                            
                            # Get parameter data for this file
                            param_data = self.get_param_data_for_file(base_name)
                            
                            # Extract material
                            material = 'Unknown'
                            if param_data:
                                # Try various material column names
                                for col in ['Sample material', 'Sample_material', 'Material', 'material', 'Sample Material']:
                                    if col in param_data:
                                        material = str(param_data[col]).strip()
                                        break
                            
                            # Extract laser energy
                            laser_energy = None
                            if param_data:
                                # Try various energy column names
                                for col in ['Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 
                                           'Laser_Target_Energy', 'Laser Target Energy',
                                           'Laser energy (J)', 'Laser_energy_J', 'laser_energy']:
                                    if col in param_data:
                                        energy_val = param_data[col]
                                        if energy_val is not None:
                                            try:
                                                energy_val = float(energy_val)
                                                # Convert to mJ if in J
                                                if 'J' in col and '(mJ)' not in col and energy_val < 1000:
                                                    energy_val = energy_val * 1000
                                                laser_energy = energy_val
                                                break
                                            except (ValueError, TypeError):
                                                continue
                            
                            if laser_energy is None:
                                laser_energy = 0  # Default to 0 if not found
                            
                            traces_by_material[material].append((vel_file, laser_energy, base_name))
                            
                        except Exception as e:
                            self.progress_signal.emit(f"  Warning: Could not process {os.path.basename(vel_file)}: {e}")
                            continue
                    
                    if not traces_by_material:
                        self.progress_signal.emit(f"[WARNING] No traces organized by material - skipping combined_mean_velocity.png (processed {len(vel_files)} files)")
                    else:
                        # Create subplots - one per material
                        # Skip Unknown material traces for this summary plot
                        materials = [
                            m for m in sorted(traces_by_material.keys())
                            if str(m).strip().lower() not in ["unknown", "nan", ""]
                        ]
                        n_materials = len(materials)
                        self.progress_signal.emit(f"[DEBUG] Organized traces by material: {dict((m, len(traces_by_material[m])) for m in traces_by_material.keys())}")
                        
                        if n_materials == 0:
                            self.progress_signal.emit(f"[WARNING] No materials found after filtering - skipping combined_mean_velocity.png (all materials were Unknown/nan/empty)")
                        else:
                            # Determine subplot layout
                            n_cols = min(3, n_materials)  # Max 3 columns
                            n_rows = (n_materials + n_cols - 1) // n_cols  # Ceiling division
                            
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
                            if n_materials == 1:
                                axes = [axes]
                            else:
                                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
                            
                            # Get energy range across all materials for consistent colormap
                            all_energies = []
                            for material_traces in traces_by_material.values():
                                for _, energy, _ in material_traces:
                                    all_energies.append(energy)

                            if all_energies:
                                min_energy = min(all_energies)
                                max_energy = max(all_energies)
                                # Round to nearest 100 mJ for binning
                                min_energy_bin = (int(min_energy) // 100) * 100
                                max_energy_bin = ((int(max_energy) // 100) + 1) * 100
                                energy_range = max_energy_bin - min_energy_bin
                            else:
                                min_energy_bin = 0
                                max_energy_bin = 1000
                                energy_range = 1000
                            
                            # Normalize colormap
                            norm = plt.Normalize(vmin=min_energy_bin, vmax=max_energy_bin)
                            cmap = viridis
                            
                            total_traces_plotted = 0
                            used_axes = []
                            
                            # Track alignment methods for debugging
                            alignment_method_counts = {}
                            
                            # Track velocity ranges per material for y-axis limits
                            material_velocity_ranges = {mat: [] for mat in materials}
                            # Track global maximum across all materials for uniform y-axis
                            global_max_velocity = 0.0

                            for idx, material in enumerate(materials):
                                ax = axes[idx]
                                material_traces = traces_by_material[material]

                                traces_plotted = 0
                                energy_bins_used = set()
                                
                                for vel_file, laser_energy, base_name in material_traces:
                                    try:
                                        # Read CSV - EXACT same method as working generate_all_velocity_traces_plot
                                        df = pd.read_csv(vel_file)
                                        if df.shape[1] < 3:
                                            self.progress_signal.emit(f"  [DEBUG] Skipping {base_name}: only {df.shape[1]} columns (need 3)")
                                            continue
                                        time_data = df.iloc[:, 0].values
                                        velocity_data = df.iloc[:, 1].values
                                        uncertainty_data = df.iloc[:, 2].values

                                        # Convert time to ns if likely in s/us (EXACT same as working function)
                                        if np.nanmax(time_data) < 1e-3:
                                            time_data = time_data * 1e9
                                        elif np.nanmax(time_data) < 1.0:
                                            time_data = time_data * 1e3

                                        # Noise fraction filtering (EXACT same as working function)
                                        noise_file = vel_file.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                                        high_noise_mask = None
                                        if os.path.exists(noise_file):
                                            try:
                                                df_noise = pd.read_csv(noise_file)
                                                if df_noise.shape[1] >= 1:
                                                    noise_fraction = df_noise.iloc[:, -1].values
                                                    if len(noise_fraction) == len(velocity_data):
                                                        high_noise_mask = noise_fraction > 1.0
                                            except Exception:
                                                pass

                                        # Build valid mask (EXACT same as working function)
                                        valid_mask = ~np.isnan(velocity_data)
                                        if high_noise_mask is not None:
                                            valid_mask &= (~high_noise_mask)
                                        if uncertainty_data is not None:
                                            uncertainty_threshold = self.spade_params.get('uncertainty_threshold_ms', 50.0)
                                            valid_mask &= (uncertainty_data <= uncertainty_threshold)
                                        else:
                                            # If no uncertainty data, that's OK - just use all valid velocity data
                                            pass

                                        time_clean = time_data[valid_mask]
                                        velocity_clean = velocity_data[valid_mask]
                                        if len(time_clean) == 0:
                                            self.progress_signal.emit(f"  [DEBUG] Skipping {base_name}: no valid data after filtering")
                                            continue

                                        # ALIGNMENT: Try HEL alignment first (if enabled), then threshold alignment (same logic as working function)
                                        use_hel_alignment = self.spade_params.get('use_hel_t0_alignment_for_plots', True)
                                        alignment_method_used = None
                                        
                                        if use_hel_alignment:
                                            # Try HEL-based alignment first
                                            min_velocity_threshold = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                                            hel_t0, hel_t0_idx, time_aligned_hel = self.find_hel_t0_alignment(
                                                time_clean, velocity_clean, min_velocity_threshold
                                            )
                                            if hel_t0 is not None and hel_t0_idx is not None:
                                                time_clean = time_aligned_hel
                                                alignment_method_used = "HEL"
                                            else:
                                                # HEL alignment failed - log diagnostic info
                                                if total_traces_plotted < 3:
                                                    # Find why it failed - check if velocity ever > 0
                                                    has_positive_vel = np.any(velocity_clean > 0)
                                                    max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                                                    if has_positive_vel:
                                                        # Check if velocity reaches threshold in 10 ns window
                                                        first_positive_idx = np.where(velocity_clean > 0)[0]
                                                        if len(first_positive_idx) > 0:
                                                            first_pos_idx = first_positive_idx[0]
                                                            window_end_time = time_clean[first_pos_idx] + 10.0
                                                            window_mask = (time_clean >= time_clean[first_pos_idx]) & (time_clean <= window_end_time)
                                                            if np.any(window_mask):
                                                                window_vel = velocity_clean[window_mask]
                                                                window_time = time_clean[window_mask]
                                                                if len(window_vel) > 1:
                                                                    avg_slope_check = (window_vel[-1] - window_vel[0]) / (window_time[-1] - window_time[0]) if (window_time[-1] - window_time[0]) > 0 else 0
                                                                    max_vel_in_window = np.max(window_vel)
                                                                    peak_vel = np.nanmax(velocity_clean)
                                                                    adaptive_thresh = max(peak_vel * 0.08, min_velocity_threshold)
                                                                    self.progress_signal.emit(f"  [DEBUG] {base_name}: HEL t=0 failed - first v>0 at t={time_clean[first_pos_idx]:.1f} ns, max_vel_in_10ns={max_vel_in_window:.1f} m/s (need >={adaptive_thresh:.1f} m/s, 8% of peak={peak_vel:.1f} m/s), avg_slope={avg_slope_check:.3f} m/s/ns")
                                                    else:
                                                        self.progress_signal.emit(f"  [DEBUG] {base_name}: HEL t=0 failed - no positive velocities found (max_vel={max_vel:.1f} m/s)")
                                        
                                        if alignment_method_used is None:
                                            # Fallback to threshold alignment (EXACT same as working function)
                                            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
                                            tolerance = 0.01
                                            
                                            # Find the first point where velocity reaches or exceeds threshold
                                            t0_idx = None
                                            for j, v in enumerate(velocity_clean):
                                                if not np.isnan(v) and v >= (align_threshold - tolerance):
                                                    t0_idx = j
                                                    break
                                            
                                            # Check if trace crosses threshold from below
                                            if t0_idx is None or t0_idx == 0:
                                                self.progress_signal.emit(f"  [DEBUG] Skipping {base_name}: threshold alignment failed (t0_idx={t0_idx})")
                                                continue  # Skip this trace (same as working function)
                                            
                                            # Verify that trace started below threshold
                                            has_point_below = False
                                            for j in range(t0_idx):
                                                if not np.isnan(velocity_clean[j]) and velocity_clean[j] < (align_threshold - tolerance):
                                                    has_point_below = True
                                                    break
                                            
                                            if not has_point_below:
                                                self.progress_signal.emit(f"  [DEBUG] Skipping {base_name}: threshold alignment failed (no point below threshold)")
                                                continue  # Skip this trace (same as working function)
                                            
                                            # Align the trace (EXACT same as working function)
                                            t0 = time_clean[t0_idx]
                                            time_clean = time_clean - t0
                                            alignment_method_used = "threshold"
                                        
                                        # Filter negative time if requested (EXACT same as working function)
                                        filter_negative_time = self.spade_params.get('filter_negative_time', False)
                                        if filter_negative_time:
                                            mask_t_positive = time_clean >= 0
                                            time_clean = time_clean[mask_t_positive]
                                            velocity_clean = velocity_clean[mask_t_positive]
                                            if len(time_clean) == 0:
                                                continue
                                        
                                        # Check if trace is properly aligned - velocity at t=0 should be < 10% of peak velocity
                                        # If velocity at t=0 is too high relative to peak, the alignment likely failed
                                        peak_velocity_trace = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                                        if peak_velocity_trace > 0:
                                            # Check velocity near t=0 (within ±1 ns)
                                            alignment_check_window = 1.0  # Check within ±1 ns of t=0
                                            alignment_check_mask = (time_clean >= -alignment_check_window) & (time_clean <= alignment_check_window)
                                            if np.any(alignment_check_mask):
                                                velocities_near_zero = velocity_clean[alignment_check_mask]
                                                if len(velocities_near_zero) > 0:
                                                    # Use minimum velocity near t=0 to be conservative
                                                    min_vel_at_zero = np.nanmin(velocities_near_zero)
                                                    velocity_fraction = (min_vel_at_zero / peak_velocity_trace) * 100
                                                    
                                                    # If velocity at t=0 is > 10% of peak, skip trace (alignment likely failed)
                                                    if velocity_fraction > 10.0:
                                                        self.progress_signal.emit(f"  [DEBUG] Skipping {base_name}: unaligned trace (velocity at t=0: {min_vel_at_zero:.1f} m/s = {velocity_fraction:.1f}% of peak {peak_velocity_trace:.1f} m/s, threshold: 10%)")
                                                        continue
                                        
                                        # Filter to x-axis range for plotting (but keep all data if needed)
                                        time_mask = (time_clean >= -20) & (time_clean <= 150)
                                        time_plot = time_clean[time_mask]
                                        velocity_plot = velocity_clean[time_mask]
                                        
                                        # If no data in range, use all data
                                        if len(time_plot) == 0:
                                            time_plot = time_clean
                                            velocity_plot = velocity_clean

                                        # Group energy into 100 mJ bins (defensive against NaN/None)
                                        try:
                                            le = float(laser_energy)
                                            if not np.isfinite(le):
                                                le = 0.0
                                        except Exception:
                                            le = 0.0
                                        energy_bin = (int(le) // 100) * 100
                                        energy_bins_used.add(energy_bin)

                                        # Get color from viridis colormap based on energy bin
                                        color = cmap(norm(energy_bin))

                                        # Plot (EXACT same style as working function)
                                        ax.plot(time_plot, velocity_plot, alpha=0.7, linewidth=1.5, color=color)
                                        traces_plotted += 1
                                        total_traces_plotted += 1
                                        
                                        # Track velocity range for y-axis limits
                                        if len(velocity_plot) > 0:
                                            valid_vel = velocity_plot[~np.isnan(velocity_plot)]
                                            if len(valid_vel) > 0:
                                                material_velocity_ranges[material].extend(valid_vel.tolist())
                                                # Update global maximum
                                                global_max_velocity = max(global_max_velocity, np.nanmax(valid_vel))
                                        
                                        # Track noise statistics for diagnostics (first trace per material)
                                        if traces_plotted == 1 and len(velocity_plot) > 10:
                                            # Calculate noise level as std dev of velocity differences
                                            vel_diff = np.diff(velocity_plot)
                                            noise_level = np.nanstd(vel_diff) if len(vel_diff) > 0 else 0
                                            mean_vel = np.nanmean(velocity_plot)
                                            relative_noise = (noise_level / mean_vel * 100) if mean_vel > 0 else 0
                                            if uncertainty_data is not None and len(uncertainty_data[valid_mask]) > 0:
                                                mean_uncertainty = np.nanmean(uncertainty_data[valid_mask])
                                                self.progress_signal.emit(f"  [DEBUG] {material} ({base_name}): mean_vel={mean_vel:.1f} m/s, noise_std={noise_level:.2f} m/s ({relative_noise:.1f}%), mean_uncertainty={mean_uncertainty:.2f} m/s")
                                        
                                        # Track alignment method
                                        if alignment_method_used:
                                            alignment_method_counts[alignment_method_used] = alignment_method_counts.get(alignment_method_used, 0) + 1
                                    except Exception as e:
                                        self.progress_signal.emit(f"  Warning: Could not plot {os.path.basename(vel_file)}: {e}")
                                        continue

                                # Format subplot once per material (even if some traces failed)
                                ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
                                ax.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
                                ax.set_title(f'{material} (n={traces_plotted})', fontsize=14, fontweight='bold')
                                ax.grid(False)
                                ax.set_xlim(-20, 150)
                                
                                # Y-axis limits will be set uniformly after all materials are processed

                                # Hide subplot completely if nothing plotted for this material
                                if traces_plotted == 0:
                                    ax.set_visible(False)
                                else:
                                    used_axes.append(ax)

                                # Remove any existing legend (if matplotlib auto-created one)
                                legend = ax.get_legend()
                                if legend:
                                    legend.remove()

                            # Finalize figure once (after all materials plotted)
                            # Hide unused subplots
                            try:
                                for j in range(n_materials, len(axes)):
                                    axes[j].set_visible(False)
                            except Exception:
                                pass

                            # Ensure no legends are created on any subplot
                            axes_list = used_axes if used_axes else (
                                list(axes)[:n_materials] if hasattr(axes, "__len__") else [axes]
                            )
                            for _ax in axes_list:
                                _legend = _ax.get_legend()
                                if _legend is not None:
                                    _legend.remove()
                            
                            # Set uniform y-axis limits for all subplots based on global maximum
                            # This ensures all subplots use the same scale for easy comparison
                            if global_max_velocity > 0:
                                # Add 10% padding above the maximum
                                y_max = global_max_velocity * 1.1
                                # Round up to nearest 10, 50, or 100 for cleaner axis labels
                                if y_max <= 100:
                                    y_max = np.ceil(y_max / 10) * 10
                                elif y_max <= 500:
                                    y_max = np.ceil(y_max / 50) * 50
                                else:
                                    y_max = np.ceil(y_max / 100) * 100
                                
                                # Apply same y-axis limits to ALL visible subplots
                                for ax in axes_list:
                                    ax.set_ylim(0, y_max)
                            else:
                                # Fallback: use default range if no data
                                for ax in axes_list:
                                    ax.set_ylim(0, 600)

                            # Adjust subplot layout to prevent label overlap and make room for colorbar
                            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on right for colorbar, prevent overlap

                            # Add colorbar on the right side in vertical orientation
                            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            sm.set_array([])
                            if axes_list:
                                cbar = fig.colorbar(
                                    sm,
                                    ax=axes_list,
                                    orientation='vertical',
                                    pad=0.05,
                                    aspect=25,
                                )
                                cbar.set_label(
                                    'Laser Energy (mJ, binned to 100 mJ)',
                                    fontsize=12,
                                    fontweight='bold',
                                    rotation=270,
                                    labelpad=25,
                                )

                            # Save atomically to avoid viewers showing a partially-written (blank) PNG.
                            # IMPORTANT: the temp file must still end with ".png" so matplotlib doesn't infer
                            # format="tmp" (which is unsupported).
                            plot_path = os.path.join(spade_output_dir, 'combined_mean_velocity.png')
                            _root, _ext = os.path.splitext(plot_path)
                            tmp_plot_path = _root + ".tmp" + (_ext if _ext else ".png")
                            try:
                                fig.savefig(tmp_plot_path, dpi=300, bbox_inches='tight', format='png')
                                os.replace(tmp_plot_path, plot_path)
                            finally:
                                try:
                                    if os.path.exists(tmp_plot_path):
                                        os.remove(tmp_plot_path)
                                except Exception:
                                    pass
                            plt.close(fig)
                            if total_traces_plotted == 0:
                                self.progress_signal.emit(
                                    f"[WARNING] Generated combined_mean_velocity.png but NO traces were plotted! ({n_materials} materials found, but all traces were filtered out)")
                                self.progress_signal.emit(f"[DEBUG] Alignment method counts: {alignment_method_counts}")
                            else:
                                self.progress_signal.emit(
                                    f"✅ Generated combined_mean_velocity.png ({total_traces_plotted} traces, {n_materials} materials)")
                                if alignment_method_counts:
                                    self.progress_signal.emit(f"[DEBUG] Alignment methods used: {dict(alignment_method_counts)}")
                else:
                    msg = "[INFO] No velocity files for combined plot (spade_only - ALPSS not run)" if self.analysis_mode == "spade_only" else "[WARNING] No velocity files found for combined plot"
                    self.progress_signal.emit(msg)
            except Exception as e:
                msg = f"[ERROR] Failed to generate combined_mean_velocity.png: {e}"
                print(msg)
                self.progress_signal.emit(msg)
                import traceback
                tb_str = traceback.format_exc()
                print(f"[ERROR] Traceback: {tb_str}")
                self.progress_signal.emit(f"[ERROR] Traceback: {tb_str}")
            
            # Generate combined velocity plot even if summary_df is empty (it doesn't depend on spall data)
            # This plot only needs velocity files and parameter data, not spall summary
            if summary_df is None or summary_df.empty:
                self.progress_signal.emit("[WARNING] Spall analysis selected but no spall_summary.csv found")
                self.progress_signal.emit("[INFO] Attempting to generate combined_mean_velocity.png anyway (doesn't require spall data)...")
                # Still try to generate combined velocity plot - it doesn't need spall data
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import pandas as pd
                    from matplotlib.cm import viridis
                    from collections import defaultdict
                    
                    # Reuse SPADE's alignment results
                    t0_map = {}
                    aligned_ok_map = {}
                    try:
                        vs_path = os.path.join(spade_output_dir, "velocity_shots_summary.csv")
                        if os.path.exists(vs_path):
                            vs = pd.read_csv(vs_path, usecols=["file_name", "t0_ns", "aligned_ok"])
                            for _, r in vs.iterrows():
                                bn = str(r.get("file_name", "")).strip()
                                if not bn:
                                    continue
                                t0_map[bn] = r.get("t0_ns", np.nan)
                                aligned_ok_map[bn] = bool(r.get("aligned_ok", False))
                    except Exception:
                        t0_map = {}
                        aligned_ok_map = {}
                    
                    # Find velocity files
                    vel_files_by_base = {}
                    for root, dirs, files in os.walk(self.output_dir):
                        dirs[:] = [
                            d for d in dirs
                            if d not in {"SPADE_analysis", "__pycache__"} and not d.endswith("_plots")
                        ]
                        for file in files:
                            if not (file.endswith("--vel-smooth-with-uncert.csv") or file.endswith("--velocity--smooth.csv")):
                                continue
                            full_path = os.path.join(root, file)
                            base_name = os.path.splitext(file)[0]
                            for suffix in ['--vel-smooth-with-uncert', '--velocity--smooth', '--vel-smooth', '--velocity', '--vel']:
                                if base_name.endswith(suffix):
                                    base_name = base_name[:-len(suffix)]
                                    break
                            prev = vel_files_by_base.get(base_name)
                            if prev is None or len(full_path) < len(prev):
                                vel_files_by_base[base_name] = full_path
                    vel_files = list(vel_files_by_base.values())
                    
                    if vel_files:
                        self.progress_signal.emit(f"[INFO] Found {len(vel_files)} velocity files, generating combined_mean_velocity.png...")
                        # Call the same logic as in the main block (we'll need to extract it to a helper function)
                        # For now, just emit a message that we would generate it
                        self.progress_signal.emit("[INFO] Combined velocity plot generation would proceed here (requires refactoring to extract logic)")
                    else:
                        msg = "[INFO] No velocity files for combined plot" if self.analysis_mode == "spade_only" else "[WARNING] No velocity files found for combined_mean_velocity.png"
                        self.progress_signal.emit(msg)
                except Exception as e:
                    self.progress_signal.emit(f"[ERROR] Failed to generate combined_mean_velocity.png (fallback): {e}")
                    import traceback
                    self.progress_signal.emit(f"[ERROR] Traceback: {traceback.format_exc()}")
        else:
            # Velocity shots mode - only log velocity-related outputs
            self.progress_signal.emit("Available outputs (Velocity Shots):")
            self.progress_signal.emit("  - Individual ALPSS files: *--results.csv, *--velocity.csv, etc.")
            self.progress_signal.emit("  - ALPSS velocity files: 4 columns (Time, Velocity, Uncertainty, Velocity+Uncertainty)")
            self.progress_signal.emit("  - Combined velocity plots (if enabled)")
        
        # Check for missing plots and warn (only check velocity plots for velocity mode, all plots for spall mode)
        # Skip velocity-trace plots in spade_only mode (no ALPSS velocity files exist)
        velocity_plot_optional = (self.analysis_mode == "spade_only")
        if spall_analysis_enabled:
            expected_plots = [
                'combined_mean_velocity.png',
                'spall_vs_strain_rate.png',
                'spall_vs_shock_stress.png',
                'shock_stress_vs_laser_energy.png',
                'all_smoothed_velocity_traces.png'
            ]
        else:
            expected_plots = [
                'combined_mean_velocity.png',
                'all_smoothed_velocity_traces.png'
            ]
        
        for plot_name in expected_plots:
            if velocity_plot_optional and plot_name in ('combined_mean_velocity.png', 'all_smoothed_velocity_traces.png'):
                continue  # These require ALPSS velocity files; skip warning in spade_only
            plot_path = os.path.join(spade_output_dir, plot_name)
            if not os.path.exists(plot_path):
                msg = f"[WARNING] Expected plot missing: {plot_name}"
                print(msg)
                self.progress_signal.emit(msg)

class PostProcessingWorker(QObject):
    """Worker thread for regenerating plots in post-processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, output_dir, spade_params, param_folder):
        super().__init__()
        self.output_dir = output_dir
        self.spade_params = spade_params or {}
        self.param_folder = param_folder
    
    def regenerate_plots(self, spade_params):
        """Regenerate velocity plots with current axis settings"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            self.progress.emit("Collecting velocity files...")
            
            # Apply limits from post-processing settings
            current_params = self.spade_params.copy()
            current_params.update(spade_params)
            # Debug: Log parameter updates
            self.progress.emit(f"[WORKER] Received parameters in regenerate_plots:")
            self.progress.emit(f"  auto_calc_limits: {current_params.get('auto_calculate_limits')}")
            self.progress.emit(f"  x_min/max_main: {current_params.get('x_min_main')}/{current_params.get('x_max_main')}")
            self.progress.emit(f"  y_min/max_main: {current_params.get('y_min_main')}/{current_params.get('y_max_main')}")
            
            pattern = os.path.join(self.output_dir, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            
            if not files:
                self.progress.emit("❌ No velocity files found")
                self.finished.emit()
                return
            
            self.progress.emit(f"Found {len(files)} velocity files")
            
            # Load parameter files to get material info
            self.progress.emit("Loading parameter files...")
            param_data = self._load_param_files()
            if param_data:
                self.progress.emit(f"✓ Loaded {len(param_data)} parameter entries")
                # Debug: show first few entries
                for i, (key, val) in enumerate(list(param_data.items())[:3]):
                    material = val.get('Sample material', 'Unknown') if isinstance(val, dict) else 'Unknown'
                    self.progress.emit(f"  - {key}: {material}")
                # Show first velocity filename for comparison
                if files:
                    first_vel_file = os.path.basename(files[0])
                    first_vel_base = os.path.splitext(first_vel_file)[0]
                    self.progress.emit(f"First velocity file: {first_vel_file}")
                    self.progress.emit(f"First velocity basename: {first_vel_base}")
                    # Check if it exists in param_data
                    if first_vel_base in param_data:
                        self.progress.emit(f"✓ MATCH found in param_data")
                    else:
                        self.progress.emit(f"✗ NO MATCH in param_data")
                        # Try to find similar keys
                        similar = [k for k in param_data.keys() if first_vel_base[:10] in k or k[:10] in first_vel_base]
                        if similar:
                            self.progress.emit(f"Similar keys in param_data: {similar[:3]}")
            else:
                self.progress.emit("⚠ No parameter data loaded - material will be 'Unknown'")
            
            spade_out = os.path.join(self.output_dir, "SPADE_analysis")
            os.makedirs(spade_out, exist_ok=True)
            
            # Generate plot
            self.progress.emit("Generating plot...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Use colormap for material differentiation
            cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(files))))
            
            traces_plotted = 0
            align_threshold = current_params.get('align_velocity_threshold_ms', 30.0)
            zoom_ns = int(current_params.get('zoom_window_ns', 1000))
            
            # Determine color coding mode
            use_material_colors = current_params.get('use_material_colors', True)
            color_by_waveplate = current_params.get('color_by_waveplate', False)
            color_by_laser_energy = current_params.get('color_by_laser_energy', False)
            
            # Track grouping parameter and colors for legend
            group_colors = {}
            file_group_map = {}
            color_label = "Material"  # Default label
            
            # Determine which parameter to use for grouping
            if color_by_waveplate:
                color_param = 'Waveplate_Angle (Degrees)'
                color_label = "Waveplate Angle (°)"
            elif color_by_laser_energy:
                color_param = 'Laser_Target_Energy (mJ)'
                color_label = "Laser Energy (mJ)"
            else:
                color_param = 'Sample material'
                color_label = "Material"
            
            # First pass: collect all unique values of the grouping parameter
            self.progress.emit(f"Scanning {color_param}...")
            unique_values = set()
            for i, file_path in enumerate(sorted(files)):
                try:
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    group_value = "Unknown"
                    if param_data:
                        # Try exact match first
                        if base_name in param_data:
                            group_value = param_data[base_name].get(color_param, 'Unknown')
                        else:
                            # Try date-shot pattern matching (YYYYMMDD--NNNNN)
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        group_value = param_data[key].get(color_param, 'Unknown')
                                        break
                        
                        # Convert to string and clean up
                        if isinstance(group_value, (int, float)):
                            group_value = str(group_value)
                        elif isinstance(group_value, str):
                            group_value = group_value.strip()
                    
                    unique_values.add(group_value)
                except Exception:
                    pass
            
            # Assign distinct colors to each unique value
            # Sort numerically for laser energy, alphabetically for others
            if color_by_laser_energy:
                # Convert to float for numeric sorting, handle 'Unknown'
                def safe_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float('inf')  # Put non-numeric values at the end
                unique_values_sorted = sorted(list(unique_values), key=safe_float)
                # Use viridis colormap for laser energy
                cmap = plt.get_cmap('viridis')
            else:
                unique_values_sorted = sorted(list(unique_values))
            cmap = plt.get_cmap('tab20')
            
            for i, value in enumerate(unique_values_sorted):
                group_colors[value] = cmap(i / max(len(unique_values_sorted), 1))
            
            self.progress.emit(f"Found {len(unique_values_sorted)} unique {color_label} values: {unique_values_sorted[:10]}")
            
            for i, file_path in enumerate(sorted(files)):
                try:
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    uncertainty_data = df.iloc[:, 2].values
                    
                    # Convert time to ns if needed
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3
                    
                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass
                    
                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    # Uncertainty threshold filtering
                    uncertainty_threshold = current_params.get('uncertainty_threshold_ms', 50.0)
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)
                    
                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                    if len(time_clean) == 0:
                        continue
                    
                    # Align at first >= threshold
                    t0_idx = None
                    for j, v in enumerate(velocity_clean):
                        if not np.isnan(v) and v >= align_threshold:
                            t0_idx = j
                            break
                    if t0_idx is not None:
                        t0 = time_clean[t0_idx]
                        time_clean = time_clean - t0
                    
                    # Extract grouping parameter value from parameter file
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes from base_name to match parameter file PDV_FileName
                    # E.g., "C1--20251023--00099--vel-smooth-with-uncert" -> "C1--20251023--00099"
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    group_value = "Unknown"
                    
                    # Try to get grouping parameter value from param_data
                    if param_data:
                        if base_name in param_data:
                            group_value = param_data[base_name].get(color_param, 'Unknown')
                    else:
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        group_value = param_data[key].get(color_param, 'Unknown')
                                        break
                    # Convert to string and clean up
                    if isinstance(group_value, (int, float)):
                        group_value = str(group_value)
                    elif isinstance(group_value, str):
                        group_value = group_value.strip()
                    
                    if group_value == "Unknown" and param_data:
                        if i < 5:
                            self.progress.emit(f"  No match for: {base_name}")
                        elif i == 5:
                            self.progress.emit("  ... (more non-matching files)")
                    
                    # Assign color based on grouping parameter value
                    if group_value not in group_colors:
                        group_colors[group_value] = colors[len(group_colors) % len(colors)]
                    
                    color = group_colors[group_value]
                    file_group_map[i] = group_value
                    
                    ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                    # Optional uncertainty bands
                    if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                        alpha = float(current_params.get('uncert_alpha', 0.2))
                        ax1.fill_between(time_clean,
                                         velocity_clean - uncert_clean,
                                         velocity_clean + uncert_clean,
                                         color=color, alpha=alpha)
                    
                    # Bottom zoom window
                    mask_zoom = time_clean <= zoom_ns
                    if np.any(mask_zoom):
                        ax2.plot(time_clean[mask_zoom], velocity_clean[mask_zoom], color=color, alpha=0.7, linewidth=1, label=group_value if group_value not in [file_group_map.get(j) for j in range(i)] else "")
                        if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                            alpha = float(current_params.get('uncert_alpha', 0.2))
                            ax2.fill_between(time_clean[mask_zoom],
                                             (velocity_clean - uncert_clean)[mask_zoom],
                                             (velocity_clean + uncert_clean)[mask_zoom],
                                             color=color, alpha=alpha)
                    
                    traces_plotted += 1
                    if traces_plotted % 5 == 0:
                        self.progress.emit(f"Processed {traces_plotted} traces...")
                except Exception as e:
                    continue
            
            # Labels and axis limits
            ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax1.set_ylabel('Velocity (m/s)', fontsize=12)
            ax1.set_title(f'All Velocity Traces (Aligned - Full Length, Color by {color_label}) - {traces_plotted} traces', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Apply axis limits from post-processing settings
            try:
                if not current_params.get('auto_calculate_limits', True):
                    # User has specified custom limits for both subplots
                    
                    # TOP SUBPLOT (ax1): Apply main subplot limits
                    x_min_main = float(current_params.get('x_min_main', 0))
                    x_max_main = float(current_params.get('x_max_main', 100))
                    y_min_main = float(current_params.get('y_min_main', 0))
                    y_max_main = float(current_params.get('y_max_main', 600))
                    ax1.set_xlim(x_min_main, x_max_main)
                    ax1.set_ylim(y_min_main, y_max_main)
                    self.progress.emit(f"Top subplot: Using user main limits X({x_min_main}-{x_max_main}), Y({y_min_main}-{y_max_main})")
                    
                    # BOTTOM SUBPLOT (ax2): Apply zoom subplot limits
                    x_min_zoom = float(current_params.get('x_min_zoom', 0))
                    x_max_zoom = float(current_params.get('x_max_zoom', zoom_ns))
                    y_min_zoom = float(current_params.get('y_min_zoom', 0))
                    y_max_zoom = float(current_params.get('y_max_zoom', 300))
                    ax2.set_xlim(x_min_zoom, x_max_zoom)
                    ax2.set_ylim(y_min_zoom, y_max_zoom)
                    ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)}-{int(x_max_zoom)} ns)', fontsize=14)
                    self.progress.emit(f"Bottom subplot: Using user zoom limits X({x_min_zoom}-{x_max_zoom}), Y({y_min_zoom}-{y_max_zoom})")
                else:
                    # Use auto-calculated limits
                    self.progress.emit(f"Using auto-calculated limits for both subplots")
                    ax2.set_xlim(0, zoom_ns)
                    ax2.set_title(f'Zoomed Velocity Traces (First {zoom_ns} ns)', fontsize=14)
            except Exception as e:
                self.progress.emit(f"⚠ Axis limits error: {str(e)[:80]}")
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'Zoomed Velocity Traces (First {zoom_ns} ns)', fontsize=14)
            
            # Add legend for grouping parameter (using the already sorted order)
            if group_colors:
                legend_patches = [mpatches.Patch(color=group_colors[val], label=val) for val in unique_values_sorted]
                ax1.legend(handles=legend_patches, loc='upper right', title=color_label, fontsize=9, title_fontsize=10)
                ax2.legend(handles=legend_patches, loc='upper right', title=color_label, fontsize=9, title_fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            self.progress.emit("Saving plot...")
            plot_path = os.path.join(spade_out, 'all_velocity_traces.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved: {plot_path}")
            self.progress.emit(f"✓ Plotted {traces_plotted} traces")
            if group_colors:
                self.progress.emit(f"✓ {color_label} values: {', '.join(sorted(group_colors.keys()))}")
            
            # Generate per-material subplots (if using material colors)
            if current_params.get('use_material_colors', True):
                self.progress.emit("Generating per-material subplot plot...")
                self._generate_material_subplots(
                    files,
                    param_data,
                    spade_out,
                    current_params,
                    group_colors,
                    align_threshold,
                )
            
            # Generate laser energy vs impact velocity plot (if enabled)
            if current_params.get('laser_energy_vs_velocity', False):
                self.progress.emit("Generating Laser Energy vs Impact Velocity plot...")
                self._generate_energy_vs_velocity_plot(files, param_data, spade_out, current_params)
            
        except Exception as e:
            self.progress.emit(f"❌ Error: {str(e)}")
            import traceback
            self.progress.emit(traceback.format_exc()[:200])
        finally:
            self.finished.emit()
    
    def _generate_material_subplots(self, files, param_data, spade_out, current_params, material_colors, align_threshold):
        """Generate a multi-subplot figure with one subplot per material"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Group files by material
            material_files = {}
            for file_path in sorted(files):
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                
                # Strip velocity-specific suffixes
                for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                material = "Unknown"
                if param_data:
                    if base_name in param_data:
                        material = param_data[base_name].get('Sample material', 'Unknown')
                    else:
                        import re
                        date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                        if date_shot_pattern:
                            date_shot = date_shot_pattern.group(1)
                            for key in param_data.keys():
                                if date_shot in str(key):
                                    material = param_data[key].get('Sample material', 'Unknown')
                                    break
                    if isinstance(material, str):
                        material = material.strip()
                
                if material not in material_files:
                    material_files[material] = []
                material_files[material].append(file_path)
            
            # Create subplots
            num_materials = len(material_files)
            if num_materials == 0:
                return
            
            # Arrange subplots in a grid (prefer rows over columns)
            ncols = min(2, num_materials)
            nrows = (num_materials + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows))
            if num_materials == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot each material
            for idx, (material, file_paths) in enumerate(sorted(material_files.items())):
                ax = axes[idx]
                color = material_colors.get(material, 'blue')
                
                traces_in_material = 0
                for file_path in file_paths:
                    try:
                        df = pd.read_csv(file_path)
                        if df.shape[1] < 3:
                            continue
                        
                        time_data = df.iloc[:, 0].values
                        velocity_data = df.iloc[:, 1].values
                        uncertainty_data = df.iloc[:, 2].values
                        
                        # Convert time to ns if needed
                        if np.nanmax(time_data) < 1e-3:
                            time_data = time_data * 1e9
                        elif np.nanmax(time_data) < 1.0:
                            time_data = time_data * 1e3
                        
                        # Noise fraction filtering
                        noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                        high_noise_mask = None
                        if os.path.exists(noise_file):
                            try:
                                df_noise = pd.read_csv(noise_file)
                                if df_noise.shape[1] >= 1:
                                    noise_fraction = df_noise.iloc[:, -1].values
                                    if len(noise_fraction) == len(velocity_data):
                                        high_noise_mask = noise_fraction > 1.0
                            except Exception:
                                pass
                        
                        valid_mask = ~np.isnan(velocity_data)
                        if high_noise_mask is not None:
                            valid_mask &= (~high_noise_mask)
                        
                        uncertainty_threshold = current_params.get('uncertainty_threshold_ms', 50.0)
                        if uncertainty_data is not None:
                            valid_mask &= (uncertainty_data <= uncertainty_threshold)
                        
                        time_clean = time_data[valid_mask]
                        velocity_clean = velocity_data[valid_mask]
                        uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                        
                        if len(time_clean) == 0:
                            continue
                        
                        # Align at first >= threshold
                        align_threshold_ms = current_params.get('align_velocity_threshold_ms', 30.0)
                        t0_idx = None
                        for j, v in enumerate(velocity_clean):
                            if not np.isnan(v) and v >= align_threshold_ms:
                                t0_idx = j
                                break
                        
                        if t0_idx is not None:
                            t0 = time_clean[t0_idx]
                            time_clean = time_clean - t0
                        
                        # Plot with material color
                        ax.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                        
                        # Optional uncertainty bands
                        if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                            alpha = float(current_params.get('uncert_alpha', 0.2))
                            ax.fill_between(time_clean,
                                           velocity_clean - uncert_clean,
                                           velocity_clean + uncert_clean,
                                           color=color, alpha=alpha)
                        
                        traces_in_material += 1
                    except Exception:
                        continue
                
                # Format subplot
                ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold_ms} m/s', fontsize=10)
                ax.set_ylabel('Velocity (m/s)', fontsize=10)
                ax.set_title(f'{material} ({traces_in_material} traces)', fontsize=12, color=color, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Apply axis limits from current_params
                if not current_params.get("auto_calculate_limits", True):
                    x_min = current_params.get("x_min_main", 0.0)
                    x_max = current_params.get("x_max_main", 100.0)
                    y_min = current_params.get("y_min_main", 0.0)
                    y_max = current_params.get("y_max_main", 600.0)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
            
            # Hide unused subplots
            for idx in range(num_materials, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(spade_out, 'velocity_traces_by_material.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved material subplots: {plot_path}")
            self.progress.emit(f"✓ Created {num_materials} subplots (one per material)")
            
        except Exception as e:
            self.progress.emit(f"⚠ Error generating material subplots: {str(e)[:80]}")
    
    def _generate_energy_vs_velocity_plot(self, files, param_data, spade_out, current_params):
        """Generate laser energy vs impact velocity scatter plot colored by waveplate angle"""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Collect data: list of (energy, velocity, waveplate_angle) tuples
            data_points = []
            align_threshold = current_params.get('align_velocity_threshold_ms', 30.0)
            
            self.progress.emit("Processing files for laser energy vs velocity...")
            
            for file_path in sorted(files):
                try:
                    # Read velocity data
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 2:
                        continue
                    
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    
                    # Convert time to ns if needed
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3
                    
                    # Filter NaN values
                    valid_mask = ~np.isnan(velocity_data) & ~np.isnan(time_data)
                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    
                    if len(time_clean) == 0:
                        continue
                    
                    # ALIGN TRACE: Try HEL t=0 alignment first (if enabled), then fall back to threshold alignment
                    use_hel_alignment = current_params.get('use_hel_t0_alignment_for_plots', True)
                    time_aligned = None
                    
                    if use_hel_alignment:
                        # Try HEL t=0 alignment (velocity > 0 and increasing for 10 ns)
                        min_velocity_threshold = current_params.get('minimum_HEL_velocity_expected', 10.0)
                        hel_t0, hel_t0_idx, time_aligned_hel = self.find_hel_t0_alignment(
                            time_clean, velocity_clean, min_velocity_threshold
                        )
                        
                        if hel_t0 is not None and hel_t0_idx is not None:
                            time_aligned = time_aligned_hel
                    
                    # Fall back to threshold alignment if HEL alignment not enabled or failed
                    if time_aligned is None:
                        threshold_mask = velocity_clean >= align_threshold
                        if np.sum(threshold_mask) > 0:
                            t0_idx = np.where(threshold_mask)[0][0]
                            t0_time = time_clean[t0_idx]
                            # Shift time so t=0 is at threshold crossing
                            time_aligned = time_clean - t0_time
                        else:
                            # No alignment possible - skip this trace
                            max_vel = np.max(velocity_clean)
                            self.progress.emit(f"  Warning: Velocity never reached {align_threshold} m/s for {os.path.basename(file_path)} (max: {max_vel:.1f} m/s)")
                            continue
                    
                    # Calculate mean velocity in 250-300ns window AFTER alignment (relative to t=0)
                    mask_window = (time_aligned >= 250) & (time_aligned <= 300)
                    if np.sum(mask_window) > 0:
                        mean_velocity = np.mean(velocity_clean[mask_window])
                        self.progress.emit(f"  ✓ Impact velocity 250-300ns after t=0 for {os.path.basename(file_path)}")
                    else:
                        # Fallback: try 300-320ns window
                        mask_window = (time_aligned >= 300) & (time_aligned <= 320)
                        if np.sum(mask_window) > 0:
                            mean_velocity = np.mean(velocity_clean[mask_window])
                            self.progress.emit(f"  Note: Using 300-320ns after t=0 for {os.path.basename(file_path)}")
                        else:
                            # Check what time range is actually available
                            time_range = f"{np.min(time_aligned):.0f} to {np.max(time_aligned):.0f}ns"
                            self.progress.emit(f"  Warning: No data in 250-300ns or 300-320ns after t=0 for {os.path.basename(file_path)} (available: {time_range})")
                            continue
                    
                    # Get laser energy and waveplate angle from parameter file
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    laser_energy = None
                    waveplate_angle = None
                    
                    if param_data:
                        # Try exact match first
                        if base_name in param_data:
                            laser_energy = param_data[base_name].get('Laser_Target_Energy (mJ)', None)
                            waveplate_angle = param_data[base_name].get('Waveplate_Angle (Degrees)', None)
                        else:
                            # Try date-shot pattern matching
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        laser_energy = param_data[key].get('Laser_Target_Energy (mJ)', None)
                                        waveplate_angle = param_data[key].get('Waveplate_Angle (Degrees)', None)
                                        break
                    
                    if laser_energy is not None and waveplate_angle is not None:
                        try:
                            energy_float = float(laser_energy)
                            angle_float = float(waveplate_angle)
                            data_points.append((energy_float, mean_velocity, angle_float))
                        except (ValueError, TypeError):
                            self.progress.emit(f"  Warning: Invalid energy/angle value for {base_name}")
                    else:
                        if laser_energy is None:
                            self.progress.emit(f"  Warning: No laser energy found for {base_name}")
                        if waveplate_angle is None:
                            self.progress.emit(f"  Warning: No waveplate angle found for {base_name}")
                
                except Exception as e:
                    self.progress.emit(f"  Error processing {os.path.basename(file_path)}: {str(e)[:50]}")
                    continue
            
            if not data_points:
                self.progress.emit("⚠ No valid laser energy vs velocity data found")
                return
            
            # Extract unique waveplate angles and assign colors
            unique_angles = sorted(list(set([dp[2] for dp in data_points])))
            self.progress.emit(f"Found {len(data_points)} data points with {len(unique_angles)} unique waveplate angles")
            
            # Use tab10 colormap for distinct colors
            cmap = plt.get_cmap('tab10')
            angle_colors = {}
            for i, angle in enumerate(unique_angles):
                angle_colors[angle] = cmap(i / max(len(unique_angles), 1))
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot each data point colored by waveplate angle
            for energy, velocity, angle in data_points:
                ax.scatter(energy, velocity, 
                          color=angle_colors[angle], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Laser Target Energy (mJ)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Flyer Impact Velocity (m/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Laser Energy vs Flyer Impact Velocity\n(Mean velocity 250-300ns after t=0 at {align_threshold:.0f} m/s, Color-coded by Waveplate Angle)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Create legend for waveplate angles
            legend_patches = [mpatches.Patch(color=angle_colors[angle], label=f'{angle}°') 
                            for angle in unique_angles]
            ax.legend(handles=legend_patches, title='Waveplate Angle', 
                     loc='best', fontsize=10, title_fontsize=11, framealpha=0.9)
            
            # Add text box with statistics
            textstr = f'Total points: {len(data_points)}\n'
            textstr += f'Alignment: t=0 at {align_threshold:.0f} m/s\n'
            textstr += f'Time window: 250-300ns after t=0\n'
            textstr += f'Energy range: {min(dp[0] for dp in data_points):.1f} - {max(dp[0] for dp in data_points):.1f} mJ\n'
            textstr += f'Velocity range: {min(dp[1] for dp in data_points):.1f} - {max(dp[1] for dp in data_points):.1f} m/s'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(spade_out, 'laser_energy_vs_impact_velocity.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved: {plot_path}")
            self.progress.emit(f"✓ Plotted {len(data_points)} data points across {len(unique_angles)} waveplate angles")
            
        except Exception as e:
            self.progress.emit(f"⚠ Error generating energy vs velocity plot: {str(e)[:80]}")
            import traceback
            self.progress.emit(traceback.format_exc()[:200])
    
    def _load_param_files(self):
        """Load parameter files using same logic as get_param_file_data"""
        param_data = {}
        
        if not self.param_folder or not os.path.exists(self.param_folder):
            return param_data
        
        try:
            import pandas as pd
            import re
            
            # Find all Excel and CSV files in the folder
            param_files = []
            for file in os.listdir(self.param_folder):
                if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                    param_files.append(os.path.join(self.param_folder, file))
            
            if not param_files:
                return param_data
            
            # Combine data from all parameter files
            combined_param_data = {}
            
            for param_file_path in param_files:
                if not os.path.exists(param_file_path):
                    continue
                
                try:
                    # Try to read the file (Excel or CSV)
                    if param_file_path.lower().endswith('.csv'):
                        df = pd.read_csv(param_file_path)
                    else:
                        try:
                            import openpyxl
                        except ImportError:
                            continue
                        df = pd.read_excel(param_file_path)
                    
                    # Handle different possible column names for PDV file name
                    pdv_col = None
                    # First pass: exact-ish known variants, ignoring spaces/underscores/dashes
                    normalized_columns = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in df.columns}
                    known_variants = [
                        'pdvfilename', 'pdvfile', 'pdv_file', 'pdv_file_name', 'pdv file name',
                        'dvfilename', 'dv_file', 'dvfile', 'filename', 'file_name', 'file name'
                    ]
                    normalized_variants = {re.sub(r"[^a-z0-9]", "", v): v for v in known_variants}
                    for col, norm in normalized_columns.items():
                        if norm in normalized_variants:
                            pdv_col = col
                            break
                    # Second pass: heuristic containing tokens 'pdv' and ('file' or 'name')
                    if pdv_col is None:
                        for col in df.columns:
                            col_lower = col.lower()
                            if ('pdv' in col_lower or 'dv' in col_lower) and ('file' in col_lower or 'name' in col_lower):
                                pdv_col = col
                                break
                    # Final fallback: a standalone 'filename' or 'file name' column
                    if pdv_col is None:
                        for col in df.columns:
                            if col.strip().lower() in ['filename', 'file name', 'file_name']:
                                pdv_col = col
                                break
                    
                    if pdv_col is None:
                        continue
                    
                    # Create mapping for each experiment
                    for idx, row in df.iterrows():
                        pdv_file = row[pdv_col]
                        if pd.isna(pdv_file) or pdv_file == 0:
                            continue
                        
                        # Convert PDV file name to string to ensure consistency
                        pdv_file_str = str(pdv_file).strip()
                        
                        # Clean the filename for better matching
                        # Remove common extensions and clean up the name
                        clean_pdv_file = pdv_file_str
                        for ext in ['.csv', '.txt', '.dat', '.xlsx', '.xls']:
                            if clean_pdv_file.lower().endswith(ext):
                                clean_pdv_file = clean_pdv_file[:-len(ext)]
                        
                        # Extract ALL columns from the row (except the PDV filename column itself)
                        exp_info = {}
                        for col in df.columns:
                            if col != pdv_col:  # Skip the PDV filename column
                                value = row.get(col)
                                if not pd.isna(value):  # Only include non-NaN values
                                    exp_info[col] = value
                        
                        # Store both original and cleaned versions for better matching
                        combined_param_data[pdv_file_str] = exp_info
                        if clean_pdv_file != pdv_file_str:
                            combined_param_data[clean_pdv_file] = exp_info
                        
                        # Also store with common variations for better matching
                        # Remove date patterns if present
                        date_cleaned = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', '', clean_pdv_file)
                        if date_cleaned != clean_pdv_file:
                            combined_param_data[date_cleaned] = exp_info
                
                except Exception:
                    continue
            
            param_data = combined_param_data
        except Exception:
            pass
        
        return param_data

class HELIXAnalysisToolbox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = 'light'
        self.config_file = os.path.join(os.path.expanduser('~'), '.helix_analysis_toolbox_config.json')
        self.spade_params = {}  # Initialize spade_params dict
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("HELIX Analysis Toolbox")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)  # Add margins around the main layout
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Increase tab size for better text fit (increased font size by 25%)
        # Enhanced checkbox styling for better usability
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                min-width: 200px;
                min-height: 40px;
                font-size: 16px;
                padding: 10px 20px;
            }
            
            QCheckBox {
                spacing: 10px;
                font-size: 14px;
                padding: 5px;
                min-height: 25px;
            }
            
            QCheckBox:hover {
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 3px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid #999;
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #555;
                background-color: #f0f0f0;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #2196F3;
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #1976D2;
                border: 2px solid #1976D2;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: white;
            }
            
            QCheckBox:disabled {
                color: #999;
            }
        """)
        
        # Create tabs
        self.create_file_selection_tab()
        self.create_analysis_mode_tab()
        self.create_alpss_params_tab()
        self.create_spade_params_tab()
        self.create_post_processing_tab()
        self.create_control_tab()
        self.create_documentation_tab()
        
        # Theme switcher at bottom
        theme_layout = QHBoxLayout()
        theme_layout.addStretch()
        self.theme_switch = QCheckBox("Dark Theme")
        self.theme_switch.stateChanged.connect(self.toggle_theme)
        theme_layout.addWidget(self.theme_switch)
        main_layout.addLayout(theme_layout)
        
        # Set initial theme
        self.apply_theme('light')
        
        # Initialize analysis thread
        self.analysis_thread = None
        
    def save_settings(self):
        """Save current settings to configuration file"""
        try:
            settings = {
                'file_paths': {
                    'input_files': self.get_input_files(),
                    'output_dir': self.output_path.text() if hasattr(self, 'output_path') else '',
                    'param_folder': self.param_folder_path.text() if hasattr(self, 'param_folder_path') else '',
                    'spade_input': self.spade_input_path.text() if hasattr(self, 'spade_input_path') else ''
                },
                'alpss_params': self.get_alpss_params() if hasattr(self, 'get_alpss_params') else {},
                'spade_params': self.get_spade_params() if hasattr(self, 'get_spade_params') else {},
                'ui_settings': {
                    'theme': self.current_theme,
                    'file_mode': self.file_mode_combo.currentText() if hasattr(self, 'file_mode_combo') else 'Single File',
                    'analysis_mode': self.mode_alpss_only.isChecked() if hasattr(self, 'mode_alpss_only') else True,
                    'spade_input_mode': self.spade_auto_mode.isChecked() if hasattr(self, 'spade_auto_mode') else True
                }
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding="utf-8") as f:
                    settings = json.load(f)
                
                # Load file paths
                if 'file_paths' in settings:
                    file_paths = settings['file_paths']
                    if hasattr(self, 'output_path') and 'output_dir' in file_paths:
                        self.output_path.setText(file_paths['output_dir'])
                    if hasattr(self, 'param_folder_path') and 'param_folder' in file_paths:
                        self.param_folder_path.setText(file_paths['param_folder'])
                    if hasattr(self, 'spade_input_path') and 'spade_input' in file_paths:
                        self.spade_input_path.setText(file_paths['spade_input'])
                
                # Load UI settings
                if 'ui_settings' in settings:
                    ui_settings = settings['ui_settings']
                    if 'theme' in ui_settings:
                        self.current_theme = ui_settings['theme']
                        if ui_settings['theme'] == 'dark':
                            self.theme_switch.setChecked(True)
                        self.apply_theme(ui_settings['theme'])
                    
                    if hasattr(self, 'file_mode_combo') and 'file_mode' in ui_settings:
                        index = self.file_mode_combo.findText(ui_settings['file_mode'])
                        if index >= 0:
                            self.file_mode_combo.setCurrentIndex(index)
                    
                    if hasattr(self, 'mode_alpss_only') and 'analysis_mode' in ui_settings:
                        if ui_settings['analysis_mode']:
                            self.mode_alpss_only.setChecked(True)
                        else:
                            self.mode_both.setChecked(True)
                    
                    if hasattr(self, 'spade_auto_mode') and 'spade_input_mode' in ui_settings:
                        if ui_settings['spade_input_mode']:
                            self.spade_auto_mode.setChecked(True)
                        else:
                            self.spade_manual_mode.setChecked(True)
                
                # Load ALPSS parameters
                if 'alpss_params' in settings and hasattr(self, 'get_alpss_params'):
                    alpss_params = settings['alpss_params']
                    # Apply ALPSS parameters to UI widgets
                    self.apply_alpss_params(alpss_params)
                
                # Load SPADE parameters
                if 'spade_params' in settings and hasattr(self, 'get_spade_params'):
                    spade_params = settings['spade_params']
                    # Apply SPADE parameters to UI widgets
                    self.apply_spade_params(spade_params)
                    
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def apply_alpss_params(self, params):
        """Apply ALPSS parameters to UI widgets"""
        try:
            if hasattr(self, 'carrier_frequency') and 'carrier_frequency' in params:
                self.carrier_frequency.setValue(params['carrier_frequency'])
            if hasattr(self, 'signal_start_time') and 'signal_start_time' in params:
                self.signal_start_time.setValue(params['signal_start_time'])
            if hasattr(self, 'smoothing_characteristic_time') and 'smoothing_characteristic_time' in params:
                self.smoothing_characteristic_time.setValue(params['smoothing_characteristic_time'])
            if hasattr(self, 'save_combined_plot') and 'save_combined_plot' in params:
                if hasattr(self.save_combined_plot, 'setChecked'):
                    self.save_combined_plot.setChecked(params['save_combined_plot'])
            if hasattr(self, 'save_iq_start_time_plot') and 'save_iq_start_time_plot' in params:
                if hasattr(self.save_iq_start_time_plot, 'setChecked'):
                    self.save_iq_start_time_plot.setChecked(params['save_iq_start_time_plot'])
            if hasattr(self, 'uncert_mult') and 'uncert_mult' in params:
                self.uncert_mult.setValue(params['uncert_mult'])
            if hasattr(self, 'spall_calculation') and 'spall_calculation' in params:
                if hasattr(self.spall_calculation, 'setChecked'):
                    self.spall_calculation.setChecked(params['spall_calculation'] == 'yes')
            # Apply output file selection parameters
            if hasattr(self, 'save_velocity_csv') and 'save_velocity_csv' in params:
                self.save_velocity_csv.setChecked(params['save_velocity_csv'])
            if hasattr(self, 'save_velocity_smooth_csv') and 'save_velocity_smooth_csv' in params:
                self.save_velocity_smooth_csv.setChecked(params['save_velocity_smooth_csv'])
            if hasattr(self, 'save_velocity_uncert_csv') and 'save_velocity_uncert_csv' in params:
                self.save_velocity_uncert_csv.setChecked(params['save_velocity_uncert_csv'])
            if hasattr(self, 'save_velocity_smooth_uncert_csv') and 'save_velocity_smooth_uncert_csv' in params:
                self.save_velocity_smooth_uncert_csv.setChecked(params['save_velocity_smooth_uncert_csv'])
            if hasattr(self, 'save_results_csv') and 'save_results_csv' in params:
                self.save_results_csv.setChecked(params['save_results_csv'])
            if hasattr(self, 'save_noise_csv') and 'save_noise_csv' in params:
                self.save_noise_csv.setChecked(params['save_noise_csv'])
            if hasattr(self, 'smart_selection_checkbox') and 'smart_selection_enabled' in params:
                self.smart_selection_checkbox.setChecked(params['smart_selection_enabled'])
            # IQ threshold factor
            if hasattr(self, 'iq_threshold_factor') and 'iq_threshold_factor' in params:
                try:
                    self.iq_threshold_factor.setValue(float(params['iq_threshold_factor']))
                except Exception:
                    pass
            # Common PDV/material parameters
            if hasattr(self, 'lam') and 'lam' in params:
                self.lam.setValue(params['lam'])
            if hasattr(self, 'C0') and 'C0' in params:
                self.C0.setValue(params['C0'])
            if hasattr(self, 'density') and 'density' in params:
                self.density.setValue(params['density'])
            if hasattr(self, 'delta_rho') and 'delta_rho' in params:
                self.delta_rho.setValue(params['delta_rho'])
            if hasattr(self, 'delta_C0') and 'delta_C0' in params:
                self.delta_C0.setValue(params['delta_C0'])
            if hasattr(self, 'delta_lam') and 'delta_lam' in params:
                self.delta_lam.setValue(params['delta_lam'])
            if hasattr(self, 'theta') and 'theta' in params:
                self.theta.setValue(params['theta'])
            if hasattr(self, 'delta_theta') and 'delta_theta' in params:
                self.delta_theta.setValue(params['delta_theta'])
        except Exception as e:
            print(f"Error applying ALPSS parameters: {e}")
    
    def apply_spade_params(self, params):
        """Apply SPADE parameters to UI widgets"""
        try:
            if hasattr(self, 'spade_density') and 'density' in params:
                self.spade_density.setValue(params['density'])
            if hasattr(self, 'spade_acoustic_velocity') and 'acoustic_velocity' in params:
                self.spade_acoustic_velocity.setValue(params['acoustic_velocity'])
            # Note: analysis_model is no longer user-selectable
            # Calculations always use hybrid approach (max_min for spall strength, hybrid_5_segment for strain rate)
            if hasattr(self, 'prominence_factor') and 'prominence_factor' in params:
                self.prominence_factor.setValue(params['prominence_factor'])
            if hasattr(self, 'peak_distance_ns') and 'peak_distance_ns' in params:
                self.peak_distance_ns.setValue(params['peak_distance_ns'])
            if hasattr(self, 'plot_individual') and 'plot_individual' in params:
                if hasattr(self.plot_individual, 'setChecked'):
                    self.plot_individual.setChecked(params['plot_individual'])
            if hasattr(self, 'save_summary') and 'save_summary_table' in params:
                if hasattr(self.save_summary, 'setChecked'):
                    self.save_summary.setChecked(params['save_summary_table'])
            if hasattr(self, 'show_plots') and 'show_plots' in params:
                if hasattr(self.show_plots, 'setChecked'):
                    self.show_plots.setChecked(params['show_plots'])
            if hasattr(self, 'experiment_velocity_shots') and 'experiment_type' in params:
                if params['experiment_type'] == 'velocity_shots':
                    if hasattr(self.experiment_velocity_shots, 'setChecked'):
                        self.experiment_velocity_shots.setChecked(True)
                elif params['experiment_type'] == 'spall_analysis':
                    if hasattr(self.experiment_spall_analysis, 'setChecked'):
                        self.experiment_spall_analysis.setChecked(True)
        except Exception as e:
            print(f"Error applying SPADE parameters: {e}")
    
    def closeEvent(self, event):
        """Save settings when closing the application"""
        self.save_settings()
        event.accept()
        
    def toggle_theme(self, state):
        if state == Qt.Checked:
            self.apply_theme('dark')
        else:
            self.apply_theme('light')
        
    def apply_theme(self, theme):
        self.current_theme = theme
        if theme == 'dark':
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #181a1b;
                }
                QWidget {
                    background-color: #181a1b;
                }
                QTabWidget::pane {
                    border: 1px solid #444;
                    background-color: #181a1b;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #232629;
                    color: #f0f0f0;
                    min-width: 200px;
                    min-height: 40px;
                    font-size: 16px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: #181a1b;
                    border-bottom: 2px solid #0078d4;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 15px;
                    border: 2px solid #444;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #f0f0f0;
                    background-color: #232629;
                }
                QGroupBox::title {
                    color: #f0f0f0;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    font-size: 14px;
                    color: #f0f0f0;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 14px;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border: 2px solid #0078d4;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                QPushButton:disabled {
                    background-color: #444;
                    color: #888;
                }
                QCheckBox {
                    spacing: 10px;
                    font-size: 14px;
                    color: #f0f0f0;
                    padding: 5px;
                    min-height: 25px;
                    background: transparent;
                }
                QCheckBox:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 3px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 3px;
                    border: 2px solid #666;
                    background-color: #232629;
                }
                QCheckBox::indicator:hover {
                    border: 2px solid #0078d4;
                    background-color: #2a2d30;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border: 2px solid #0078d4;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #106ebe;
                    border: 2px solid #106ebe;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #232629;
                }
                QCheckBox:disabled {
                    color: #666;
                }
                QRadioButton {
                    font-size: 14px;
                    color: #f0f0f0;
                    spacing: 8px;
                    background: transparent;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #888;
                    border-radius: 8px;
                    background: transparent;
                }
                QRadioButton::indicator:checked {
                    border: 1px solid #0078d4;
                    background-color: #0078d4;
                }
                QRadioButton::indicator:focus {
                    border: 2px solid #0078d4;
                }
                QTextEdit {
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 14px;
                }
                QScrollArea {
                    border: none;
                    background: #181a1b;
                }
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 4px;
                    text-align: center;
                    font-weight: bold;
                    color: #f0f0f0;
                    background: #232629;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f5f5f5;
                }
                QWidget {
                    background-color: #f5f5f5;
                }
                QTabWidget::pane {
                    border: 1px solid #c0c0c0;
                    background-color: white;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #e0e0e0;
                    color: #2c2c2c;
                    min-width: 200px;
                    min-height: 40px;
                    font-size: 16px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: white;
                    border-bottom: 2px solid #0078d4;
                    color: #2c2c2c;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 15px;
                    border: 2px solid #c0c0c0;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #2c2c2c;
                    background-color: white;
                }
                QGroupBox::title {
                    color: #2c2c2c;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    font-size: 14px;
                    color: #2c2c2c;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 14px;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border: 2px solid #0078d4;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                QPushButton:disabled {
                    background-color: #c0c0c0;
                    color: #666666;
                }
                QCheckBox {
                    spacing: 10px;
                    font-size: 14px;
                    color: #2c2c2c;
                    padding: 5px;
                    min-height: 25px;
                    background: transparent;
                }
                QCheckBox:hover {
                    background-color: rgba(0, 0, 0, 0.05);
                    border-radius: 3px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 3px;
                    border: 2px solid #999;
                    background-color: white;
                }
                QCheckBox::indicator:hover {
                    border: 2px solid #555;
                    background-color: #f0f0f0;
                }
                QCheckBox::indicator:checked {
                    background-color: #2196F3;
                    border: 2px solid #2196F3;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #1976D2;
                    border: 2px solid #1976D2;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                }
                QCheckBox:disabled {
                    color: #999;
                }
                QRadioButton {
                    font-size: 14px;
                    color: #2c2c2c;
                    spacing: 8px;
                    background: transparent;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #888;
                    border-radius: 8px;
                    background: transparent;
                }
                QRadioButton::indicator:checked {
                    border: 1px solid #0078d4;
                    background-color: #0078d4;
                }
                QRadioButton::indicator:focus {
                    border: 2px solid #0078d4;
                }
                QTextEdit {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 14px;
                }
                QScrollArea {
                    border: none;
                    background: #f5f5f5;
                }
                QProgressBar {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    text-align: center;
                    font-weight: bold;
                    color: #2c2c2c;
                    background: white;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)
        
    def create_file_selection_tab(self):
        """Create file selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Manual file selection (pick multiple files individually)
        single_file_layout = QHBoxLayout()
        self.single_file_radio = QCheckBox("Manual Select")
        self.single_file_radio.setChecked(True)
        self.single_file_radio.toggled.connect(self.on_file_mode_changed)
        single_file_layout.addWidget(self.single_file_radio)
        
        self.single_file_path = QLineEdit()
        self.single_file_path.setPlaceholderText("Select input file(s)...")
        single_file_layout.addWidget(self.single_file_path)
        
        self.single_file_btn = QPushButton("Browse")
        self.single_file_btn.clicked.connect(self.select_single_file)
        single_file_layout.addWidget(self.single_file_btn)
        
        file_layout.addLayout(single_file_layout)
        
        # Multiple files selection
        multi_file_layout = QHBoxLayout()
        self.multi_file_radio = QCheckBox("Multiple Files")
        self.multi_file_radio.toggled.connect(self.on_file_mode_changed)
        multi_file_layout.addWidget(self.multi_file_radio)
        
        self.multi_file_path = QLineEdit()
        self.multi_file_path.setPlaceholderText("Select input directory...")
        multi_file_layout.addWidget(self.multi_file_path)
        
        self.multi_file_btn = QPushButton("Browse")
        self.multi_file_btn.clicked.connect(self.select_multi_file_dir)
        multi_file_layout.addWidget(self.multi_file_btn)
        
        file_layout.addLayout(multi_file_layout)
        
        # File pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File Pattern:"))
        self.file_pattern = QLineEdit("*.csv")
        pattern_layout.addWidget(self.file_pattern)
        file_layout.addLayout(pattern_layout)
        
        layout.addWidget(file_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        
        self.output_path = QLineEdit()
        # Use a more robust default output path
        try:
            default_output = os.path.join(os.path.expanduser("~"), "ALPSS_SPADE_output")
        except:
            default_output = "output"
        self.output_path.setText(default_output)
        output_layout.addWidget(self.output_path)
        
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_btn)
        
        layout.addWidget(output_group)
        
        # Parameter file selection
        param_group = QGroupBox("Experiment Parameter Files (Optional)")
        param_layout = QVBoxLayout(param_group)
        
        # Description
        desc_label = QLabel("Select a folder containing parameter files (.xlsx/.xls/.csv) to link experiment data with processing results.")
        desc_label.setWordWrap(True)
        param_layout.addWidget(desc_label)
        
        # Parameter folder selection
        param_folder_layout = QHBoxLayout()
        self.param_folder_path = QLineEdit()
        self.param_folder_path.setPlaceholderText("Select parameter folder...")
        param_folder_layout.addWidget(self.param_folder_path)
        
        self.param_folder_btn = QPushButton("Browse")
        self.param_folder_btn.clicked.connect(self.select_param_folder)
        param_folder_layout.addWidget(self.param_folder_btn)
        
        param_layout.addLayout(param_folder_layout)
        
        # Parameter file info
        self.param_file_info = QTextEdit()
        self.param_file_info.setMaximumHeight(100)
        self.param_file_info.setPlaceholderText("Parameter folder information will appear here...")
        self.param_file_info.setReadOnly(True)
        param_layout.addWidget(QLabel("Parameter Folder Info:"))
        param_layout.addWidget(self.param_file_info)
        
        layout.addWidget(param_group)
        
        # File list display
        self.file_list = QPlainTextEdit()
        self.file_list.setMaximumHeight(200)
        self.file_list.setPlaceholderText("Selected files will appear here...")
        self.file_list.setReadOnly(True)
        layout.addWidget(QLabel("Selected Files:"))
        layout.addWidget(self.file_list)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "File Selection")
        
    def create_analysis_mode_tab(self):
        """Create analysis mode selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Analysis mode group
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        # Radio buttons for different modes (mutually exclusive)
        self.mode_button_group = QButtonGroup(mode_group)
        self.mode_button_group.setExclusive(True)
        
        self.mode_alpss_only = QRadioButton("ALPSS Only")
        self.mode_button_group.addButton(self.mode_alpss_only)
        mode_layout.addWidget(self.mode_alpss_only)
        
        self.mode_spade_only = QRadioButton("SPADE Only")
        self.mode_button_group.addButton(self.mode_spade_only)
        mode_layout.addWidget(self.mode_spade_only)
        
        self.mode_both = QRadioButton("ALPSS + SPADE (Combined)")
        self.mode_both.setChecked(True)
        self.mode_button_group.addButton(self.mode_both)
        mode_layout.addWidget(self.mode_both)
        
        self.mode_button_group.buttonToggled.connect(self.on_analysis_mode_changed)
        
        # Description text
        desc_text = QPlainTextEdit()
        desc_text.setMaximumHeight(150)
        desc_text.setReadOnly(True)
        desc_text.setPlainText(
            "ALPSS Only: Run ALPSS analysis on input files and save results.\n\n"
            "SPADE Only: Run SPADE analysis on manually selected velocity files.\n\n"
            "ALPSS + SPADE: Run ALPSS first, then automatically run SPADE on ALPSS outputs."
        )
        mode_layout.addWidget(desc_text)
        
        layout.addWidget(mode_group)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Analysis Mode")

    def create_post_processing_tab(self):
        """Create Post-Processing tab for quick plot edits after analysis"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        # Source selection
        src_group = QGroupBox("Source (ALPSS Output)")
        src_layout = QGridLayout(src_group)
        src_layout.setSpacing(10)

        src_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.pp_output_dir = QLineEdit()
        self.pp_output_dir.setPlaceholderText("Select ALPSS output directory...")
        src_layout.addWidget(self.pp_output_dir, 0, 1)
        self.pp_browse_output = QPushButton("Browse")
        self.pp_browse_output.clicked.connect(self.pp_select_output_dir)
        src_layout.addWidget(self.pp_browse_output, 0, 2)

        # Plot options
        opt_group = QGroupBox("Plot Options")
        opt_layout = QGridLayout(opt_group)
        opt_layout.setSpacing(10)

        self.pp_regen_combined = QCheckBox("Regenerate combined aligned velocity plot")
        self.pp_regen_combined.setChecked(True)
        opt_layout.addWidget(self.pp_regen_combined, 0, 0, 1, 2)

        self.pp_use_material_colors = QCheckBox("Color by Sample material (if available)")
        self.pp_use_material_colors.setChecked(True)
        self.pp_use_material_colors.stateChanged.connect(self.pp_on_material_color_changed)
        opt_layout.addWidget(self.pp_use_material_colors, 1, 0, 1, 2)

        self.pp_color_by_waveplate = QCheckBox("Color by Waveplate Angle (if available)")
        self.pp_color_by_waveplate.setChecked(False)
        self.pp_color_by_waveplate.stateChanged.connect(self.pp_on_waveplate_color_changed)
        opt_layout.addWidget(self.pp_color_by_waveplate, 2, 0, 1, 2)

        self.pp_color_by_laser_energy = QCheckBox("Color by Laser Target Energy (if available)")
        self.pp_color_by_laser_energy.setChecked(False)
        self.pp_color_by_laser_energy.stateChanged.connect(self.pp_on_laser_energy_color_changed)
        opt_layout.addWidget(self.pp_color_by_laser_energy, 3, 0, 1, 2)

        self.pp_laser_energy_vs_velocity = QCheckBox("Generate Laser Energy vs Impact Velocity plot")
        self.pp_laser_energy_vs_velocity.setChecked(False)
        self.pp_laser_energy_vs_velocity.setToolTip("Plot laser energy vs mean velocity (250-300 ns)")
        opt_layout.addWidget(self.pp_laser_energy_vs_velocity, 4, 0, 1, 2)

        opt_layout.addWidget(QLabel("Zoom Window (ns):"), 5, 0)
        self.pp_zoom_ns = QSpinBox()
        self.pp_zoom_ns.setRange(10, 10000)
        self.pp_zoom_ns.setValue(1000)
        opt_layout.addWidget(self.pp_zoom_ns, 5, 1)

        opt_layout.addWidget(QLabel("Alignment Threshold (m/s):"), 6, 0)
        self.pp_align_threshold = QDoubleSpinBox()
        self.pp_align_threshold.setRange(0.0, 1000.0)
        self.pp_align_threshold.setDecimals(2)
        self.pp_align_threshold.setValue(30.0)
        self.pp_align_threshold.setSuffix(" m/s")
        self.pp_align_threshold.setToolTip("Align traces to first velocity ≥ threshold (t=0)")
        opt_layout.addWidget(self.pp_align_threshold, 6, 1)

        # Axis limits
        axis_group = QGroupBox("Axis Limits")
        axis_layout = QGridLayout(axis_group)
        axis_layout.setSpacing(10)
        self.pp_auto_limits = QCheckBox("Auto")
        self.pp_auto_limits.setChecked(False)  # Default to custom limits
        axis_layout.addWidget(self.pp_auto_limits, 0, 0)

        # Main (top) subplot limits
        axis_layout.addWidget(QLabel("Main Subplot:"), 0, 1)
        axis_layout.addWidget(QLabel("X min (ns):"), 1, 0)
        self.pp_xmin = QDoubleSpinBox()
        self.pp_xmin.setRange(-1e6, 1e6)
        self.pp_xmin.setDecimals(2)
        self.pp_xmin.setValue(0.0)
        axis_layout.addWidget(self.pp_xmin, 1, 1)

        axis_layout.addWidget(QLabel("X max (ns):"), 1, 2)
        self.pp_xmax = QDoubleSpinBox()
        self.pp_xmax.setRange(-1e6, 1e6)
        self.pp_xmax.setDecimals(2)
        self.pp_xmax.setValue(100.0)
        axis_layout.addWidget(self.pp_xmax, 1, 3)

        axis_layout.addWidget(QLabel("Y min (m/s):"), 2, 0)
        self.pp_ymin = QDoubleSpinBox()
        self.pp_ymin.setRange(-1e6, 1e6)
        self.pp_ymin.setDecimals(2)
        self.pp_ymin.setValue(0.0)
        axis_layout.addWidget(self.pp_ymin, 2, 1)

        axis_layout.addWidget(QLabel("Y max (m/s):"), 2, 2)
        self.pp_ymax = QDoubleSpinBox()
        self.pp_ymax.setRange(-1e6, 1e6)
        self.pp_ymax.setDecimals(2)
        self.pp_ymax.setValue(600.0)
        axis_layout.addWidget(self.pp_ymax, 2, 3)

        # Zoom (bottom) subplot limits
        axis_layout.addWidget(QLabel("Zoom Subplot:"), 3, 1)
        axis_layout.addWidget(QLabel("X min (ns):"), 4, 0)
        self.pp_zoom_xmin = QDoubleSpinBox()
        self.pp_zoom_xmin.setRange(-1e6, 1e6)
        self.pp_zoom_xmin.setDecimals(2)
        self.pp_zoom_xmin.setValue(0.0)
        axis_layout.addWidget(self.pp_zoom_xmin, 4, 1)

        axis_layout.addWidget(QLabel("X max (ns):"), 4, 2)
        self.pp_zoom_xmax = QDoubleSpinBox()
        self.pp_zoom_xmax.setRange(-1e6, 1e6)
        self.pp_zoom_xmax.setDecimals(2)
        self.pp_zoom_xmax.setValue(50.0)
        axis_layout.addWidget(self.pp_zoom_xmax, 4, 3)

        axis_layout.addWidget(QLabel("Y min (m/s):"), 5, 0)
        self.pp_zoom_ymin = QDoubleSpinBox()
        self.pp_zoom_ymin.setRange(-1e6, 1e6)
        self.pp_zoom_ymin.setDecimals(2)
        self.pp_zoom_ymin.setValue(0.0)
        axis_layout.addWidget(self.pp_zoom_ymin, 5, 1)

        axis_layout.addWidget(QLabel("Y max (m/s):"), 5, 2)
        self.pp_zoom_ymax = QDoubleSpinBox()
        self.pp_zoom_ymax.setRange(-1e6, 1e6)
        self.pp_zoom_ymax.setDecimals(2)
        self.pp_zoom_ymax.setValue(300.0)
        axis_layout.addWidget(self.pp_zoom_ymax, 5, 3)

        # Actions
        action_layout = QHBoxLayout()
        self.pp_preview_btn = QPushButton("Preview")
        self.pp_preview_btn.clicked.connect(self.pp_preview_plots)
        self.pp_save_btn = QPushButton("Save")
        self.pp_save_btn.clicked.connect(self.pp_save_plots)
        action_layout.addStretch()
        action_layout.addWidget(self.pp_preview_btn)
        action_layout.addWidget(self.pp_save_btn)

        # Preview area
        self.pp_preview = QPlainTextEdit()
        self.pp_preview.setReadOnly(True)
        self.pp_preview.setMaximumHeight(120)
        self.pp_preview.setPlaceholderText("Preview log will appear here...")

        layout.addWidget(src_group)
        layout.addWidget(opt_group)
        layout.addWidget(axis_group)
        layout.addLayout(action_layout)
        layout.addWidget(self.pp_preview)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Post-Processing")

    def pp_on_material_color_changed(self, state):
        """Handle material color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_color_by_waveplate.blockSignals(True)
            self.pp_color_by_laser_energy.blockSignals(True)
            self.pp_color_by_waveplate.setChecked(False)
            self.pp_color_by_laser_energy.setChecked(False)
            self.pp_color_by_waveplate.blockSignals(False)
            self.pp_color_by_laser_energy.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_color_by_waveplate.isChecked() and not self.pp_color_by_laser_energy.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_use_material_colors.blockSignals(True)
                self.pp_use_material_colors.setChecked(True)
                self.pp_use_material_colors.blockSignals(False)

    def pp_on_waveplate_color_changed(self, state):
        """Handle waveplate angle color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_use_material_colors.blockSignals(True)
            self.pp_color_by_laser_energy.blockSignals(True)
            self.pp_use_material_colors.setChecked(False)
            self.pp_color_by_laser_energy.setChecked(False)
            self.pp_use_material_colors.blockSignals(False)
            self.pp_color_by_laser_energy.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_use_material_colors.isChecked() and not self.pp_color_by_laser_energy.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_color_by_waveplate.blockSignals(True)
                self.pp_color_by_waveplate.setChecked(True)
                self.pp_color_by_waveplate.blockSignals(False)

    def pp_on_laser_energy_color_changed(self, state):
        """Handle laser energy color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_use_material_colors.blockSignals(True)
            self.pp_color_by_waveplate.blockSignals(True)
            self.pp_use_material_colors.setChecked(False)
            self.pp_color_by_waveplate.setChecked(False)
            self.pp_use_material_colors.blockSignals(False)
            self.pp_color_by_waveplate.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_use_material_colors.isChecked() and not self.pp_color_by_waveplate.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_color_by_laser_energy.blockSignals(True)
                self.pp_color_by_laser_energy.setChecked(True)
                self.pp_color_by_laser_energy.blockSignals(False)

    def pp_select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select ALPSS Output Directory")
        if dir_path:
            self.pp_output_dir.setText(dir_path)

    def pp_apply_limits_to_spade_params(self):
        # Ensure we have a params dict to work with
        try:
            current_params = self.get_spade_params()
        except Exception:
            current_params = {}
        if not hasattr(self, 'spade_params') or not isinstance(getattr(self, 'spade_params'), dict):
            self.spade_params = current_params.copy()
        else:
            # Refresh base values from current UI before applying overrides
            self.spade_params.update(current_params)

        # Temporarily override SPADE params for plotting routines
        self.spade_params['auto_calculate_limits'] = self.pp_auto_limits.isChecked()
        if not self.pp_auto_limits.isChecked():
            # Main (top) subplot limits
            self.spade_params['x_min_main'] = self.pp_xmin.value()
            self.spade_params['x_max_main'] = self.pp_xmax.value()
            self.spade_params['y_min_main'] = self.pp_ymin.value()
            self.spade_params['y_max_main'] = self.pp_ymax.value()
            
            # Zoom (bottom) subplot limits
            self.spade_params['x_min_zoom'] = self.pp_zoom_xmin.value()
            self.spade_params['x_max_zoom'] = self.pp_zoom_xmax.value()
            self.spade_params['y_min_zoom'] = self.pp_zoom_ymin.value()
            self.spade_params['y_max_zoom'] = self.pp_zoom_ymax.value()
        
        self.spade_params['zoom_window_ns'] = self.pp_zoom_ns.value()
        self.spade_params['align_velocity_threshold_ms'] = self.pp_align_threshold.value()
        
        # Color coding options
        self.spade_params['use_material_colors'] = self.pp_use_material_colors.isChecked()
        self.spade_params['color_by_waveplate'] = self.pp_color_by_waveplate.isChecked()
        self.spade_params['color_by_laser_energy'] = self.pp_color_by_laser_energy.isChecked()
        
        # Additional plot options
        self.spade_params['laser_energy_vs_velocity'] = self.pp_laser_energy_vs_velocity.isChecked()

        # Debug: Log applied parameters
        self.progress_text.appendPlainText(f"[POST-PROCESSING] Parameters applied:")
        self.progress_text.appendPlainText(f"  auto_calc_limits: {self.spade_params.get('auto_calculate_limits')}")
        self.progress_text.appendPlainText(f"  x_min/max_main: {self.spade_params.get('x_min_main')}/{self.spade_params.get('x_max_main')}")
        self.progress_text.appendPlainText(f"  y_min/max_main: {self.spade_params.get('y_min_main')}/{self.spade_params.get('y_max_main')}")

    def pp_preview_plots(self):
        try:
            out_dir = self.pp_output_dir.text().strip() or self.output_path.text().strip()
            if not out_dir or not os.path.exists(out_dir):
                QMessageBox.warning(self, "Invalid Output Directory", "Please select a valid ALPSS output directory.")
                return
            
            self.pp_preview.appendPlainText("Starting preview generation in background...")
            self.pp_preview_btn.setEnabled(False)
            self.pp_save_btn.setEnabled(False)
            
            # Build spade_params from current settings
            self.pp_apply_limits_to_spade_params()
            
            # Get parameter folder from File Selection tab
            param_folder = self.param_folder_path.text() if hasattr(self, 'param_folder_path') else ""
            
            # Run in background thread
            self.pp_worker = PostProcessingWorker(out_dir, self.spade_params, param_folder)
            self.pp_thread = QThread()
            self.pp_worker.moveToThread(self.pp_thread)
            self.pp_worker.progress.connect(self.pp_on_progress)
            self.pp_worker.finished.connect(self.pp_on_finished)
            self.pp_worker.finished.connect(self.pp_thread.quit)
            self.pp_thread.started.connect(lambda: self.pp_worker.regenerate_plots(self.spade_params))
            self.pp_thread.start()
        except Exception as e:
            self.pp_preview.appendPlainText(f"Error: {e}")
            self.pp_preview_btn.setEnabled(True)
            self.pp_save_btn.setEnabled(True)

    def pp_save_plots(self):
        try:
            out_dir = self.pp_output_dir.text().strip() or self.output_path.text().strip()
            if not out_dir or not os.path.exists(out_dir):
                QMessageBox.warning(self, "Invalid Output Directory", "Please select a valid ALPSS output directory.")
                return
            
            self.pp_preview.appendPlainText("Starting plot save in background...")
            self.pp_preview_btn.setEnabled(False)
            self.pp_save_btn.setEnabled(False)
            
            # Build spade_params from current settings
            self.pp_apply_limits_to_spade_params()
            
            # Get parameter folder from File Selection tab
            param_folder = self.param_folder_path.text() if hasattr(self, 'param_folder_path') else ""
            
            # Run in background thread (same as preview, just a label difference)
            self.pp_worker = PostProcessingWorker(out_dir, self.spade_params, param_folder)
            self.pp_thread = QThread()
            self.pp_worker.moveToThread(self.pp_thread)
            self.pp_worker.progress.connect(self.pp_on_progress)
            self.pp_worker.finished.connect(self.pp_on_finished)
            self.pp_worker.finished.connect(self.pp_thread.quit)
            self.pp_thread.started.connect(lambda: self.pp_worker.regenerate_plots(self.spade_params))
            self.pp_thread.start()
        except Exception as e:
            self.pp_preview.appendPlainText(f"Error: {e}")
            self.pp_preview_btn.setEnabled(True)
            self.pp_save_btn.setEnabled(True)

    def pp_on_progress(self, msg):
        """Receive progress updates from post-processing worker"""
        self.pp_preview.appendPlainText(msg)
        QApplication.processEvents()

    def pp_on_finished(self):
        """Re-enable buttons when post-processing finishes"""
        self.pp_preview_btn.setEnabled(True)
        self.pp_save_btn.setEnabled(True)
        self.pp_preview.appendPlainText("✓ Complete!")

    def pp_regenerate_velocity_plot(self, input_path, spade_output_dir):
        """Regenerate all velocity traces plot with current axis settings for post-processing."""
        import glob
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.getsize(f) > 0]
        if not files:
            self.pp_preview.appendPlainText("No velocity files found for regeneration")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, max(1, len(files))))

        traces_plotted = 0
        align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
        zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))

        for i, file_path in enumerate(sorted(files)):
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] < 3:
                    continue
                time_data = df.iloc[:, 0].values
                velocity_data = df.iloc[:, 1].values
                uncertainty_data = df.iloc[:, 2].values

                # Convert time to ns if needed
                if np.nanmax(time_data) < 1e-3:
                    time_data = time_data * 1e9
                elif np.nanmax(time_data) < 1.0:
                    time_data = time_data * 1e3

                # Noise fraction filtering
                noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                high_noise_mask = None
                if os.path.exists(noise_file):
                    try:
                        df_noise = pd.read_csv(noise_file)
                        if df_noise.shape[1] >= 1:
                            noise_fraction = df_noise.iloc[:, -1].values
                            if len(noise_fraction) == len(velocity_data):
                                high_noise_mask = noise_fraction > 1.0
                    except Exception:
                        pass

                valid_mask = ~np.isnan(velocity_data)
                if high_noise_mask is not None:
                    valid_mask &= (~high_noise_mask)
                # Uncertainty threshold filtering
                uncertainty_threshold = self.spade_params.get('uncertainty_threshold_ms', 50.0)
                if uncertainty_data is not None:
                    valid_mask &= (uncertainty_data <= uncertainty_threshold)

                time_clean = time_data[valid_mask]
                velocity_clean = velocity_data[valid_mask]
                uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                if len(time_clean) == 0:
                    continue

                # Align at first >= threshold
                t0_idx = None
                for j, v in enumerate(velocity_clean):
                    if not np.isnan(v) and v >= align_threshold:
                        t0_idx = j
                        break
                if t0_idx is not None:
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0

                color = colors[i % len(colors)]
                ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                # Optional uncertainty bands
                if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None:
                    alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                    ax1.fill_between(time_clean,
                                     velocity_clean - uncert_clean,
                                     velocity_clean + uncert_clean,
                                     color=color, alpha=alpha)

                # Bottom zoom window
                mask_zoom = time_clean <= zoom_ns
                if np.any(mask_zoom):
                    ax2.plot(time_clean[mask_zoom], velocity_clean[mask_zoom], color=color, alpha=0.7, linewidth=1)
                    if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None:
                        alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                        ax2.fill_between(time_clean[mask_zoom],
                                         (velocity_clean - uncert_clean)[mask_zoom],
                                         (velocity_clean + uncert_clean)[mask_zoom],
                                         color=color, alpha=alpha)

                traces_plotted += 1
            except Exception as e:
                self.pp_preview.appendPlainText(f"Warning: Could not process {os.path.basename(file_path)}: {str(e)[:50]}")
                continue

        # Labels and axis limits
        ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'All Velocity Traces (Aligned) - {traces_plotted} traces', fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Apply axis limits from post-processing settings
        try:
            if not self.spade_params.get('auto_calculate_limits', True):
                x_min_main = float(self.spade_params.get('x_min_main', 0))
                x_max_main = float(self.spade_params.get('x_max_main', 100))
                y_min_main = float(self.spade_params.get('y_min_main', 0))
                y_max_main = float(self.spade_params.get('y_max_main', 600))
                ax1.set_xlim(x_min_main, x_max_main)
                ax1.set_ylim(y_min_main, y_max_main)

                x_min_zoom = float(self.spade_params.get('x_min_zoom', 0))
                x_max_zoom = float(self.spade_params.get('x_max_zoom', zoom_ns))
                y_min_zoom = float(self.spade_params.get('y_min_zoom', y_min_main))
                y_max_zoom = float(self.spade_params.get('y_max_zoom', y_max_main))
                ax2.set_xlim(x_min_zoom, x_max_zoom)
                ax2.set_ylim(y_min_zoom, y_max_zoom)
                ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)} to {int(x_max_zoom)} ns)', fontsize=14)
            else:
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)
        except Exception as e:
            self.pp_preview.appendPlainText(f"Warning: Could not apply axis limits: {str(e)[:50]}")
            ax2.set_xlim(0, zoom_ns)
            ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(spade_output_dir, 'all_velocity_traces.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.pp_preview.appendPlainText(f"Saved to: {plot_path}")

    def create_alpss_params_tab(self):
        """Create ALPSS parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
        # Config File Management
        config_group = QGroupBox("Configuration File Management")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        
        # Mode selection
        config_layout.addWidget(QLabel("Parameter Entry Mode:"), 0, 0)
        self.alpss_config_mode_group = QButtonGroup()
        self.alpss_manual_mode = QRadioButton("Manual Entry (use GUI controls)")
        self.alpss_config_mode = QRadioButton("Use Config File")
        self.alpss_config_mode_group.addButton(self.alpss_manual_mode)
        self.alpss_config_mode_group.addButton(self.alpss_config_mode)
        self.alpss_manual_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.alpss_manual_mode)
        mode_layout.addWidget(self.alpss_config_mode)
        config_layout.addLayout(mode_layout, 0, 1, 1, 3)
        
        # Config file path
        config_layout.addWidget(QLabel("Config File Path:"), 1, 0)
        self.alpss_config_path = QLineEdit()
        self.alpss_config_path.setPlaceholderText("Select a config file to load ALPSS parameters")
        self.alpss_config_path.setEnabled(False)
        config_layout.addWidget(self.alpss_config_path, 1, 1, 1, 2)
        
        self.alpss_config_browse_btn = QPushButton("Browse")
        self.alpss_config_browse_btn.clicked.connect(self.browse_alpss_config)
        self.alpss_config_browse_btn.setEnabled(False)
        config_layout.addWidget(self.alpss_config_browse_btn, 1, 3)
        
        # Load and Save buttons
        self.alpss_config_load_btn = QPushButton("Load Config")
        self.alpss_config_load_btn.clicked.connect(self.load_alpss_config)
        self.alpss_config_load_btn.setEnabled(False)
        config_layout.addWidget(self.alpss_config_load_btn, 2, 0)
        
        self.alpss_config_save_btn = QPushButton("Save Current Settings to Config")
        self.alpss_config_save_btn.clicked.connect(self.save_alpss_config)
        config_layout.addWidget(self.alpss_config_save_btn, 2, 1, 1, 3)
        
        # Connect mode change
        self.alpss_config_mode.toggled.connect(self.on_alpss_config_mode_changed)
        
        layout.addWidget(config_group)
        
        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        basic_layout.addWidget(QLabel("Save Data:"), 0, 0)
        self.save_data = QComboBox()
        self.save_data.addItems(["yes", "no"])
        self.save_data.setCurrentText("yes")
        basic_layout.addWidget(self.save_data, 0, 1)
        
        basic_layout.addWidget(QLabel("Display Plots:"), 0, 2)
        self.display_plots = QComboBox()
        self.display_plots.addItems(["yes", "no"])
        self.display_plots.setCurrentText("no")
        basic_layout.addWidget(self.display_plots, 0, 3)
        
        # Row 1
        basic_layout.addWidget(QLabel("Save ALPSS Plots:"), 1, 0)
        self.save_all_plots = QComboBox()
        self.save_all_plots.addItems(["no", "subfolder", "main_dir"])
        self.save_all_plots.setCurrentText("no")
        self.save_all_plots.setToolTip("'no': Only save CSV data files (unless individual plots are selected below). 'subfolder': Save plots in individual subfolders. 'main_dir': Save plots in main output directory.")
        basic_layout.addWidget(self.save_all_plots, 1, 1)
        
        # Row 2
        basic_layout.addWidget(QLabel("Spall Calculation:"), 2, 0)
        self.spall_calculation = QComboBox()
        self.spall_calculation.addItems(["yes", "no"])
        self.spall_calculation.setCurrentText("yes")
        basic_layout.addWidget(self.spall_calculation, 2, 1)
        
        basic_layout.addWidget(QLabel("Header Lines:"), 2, 2)
        self.header_lines = QSpinBox()
        self.header_lines.setRange(0, 100)
        self.header_lines.setValue(5)
        basic_layout.addWidget(self.header_lines, 2, 3)
        
        # Row 3
        basic_layout.addWidget(QLabel("Start Time User:"), 3, 0)
        self.start_time_user = QLineEdit("none")
        basic_layout.addWidget(self.start_time_user, 3, 1)
        
        basic_layout.addWidget(QLabel("Start Time Correction (s):"), 3, 2)
        self.start_time_correction = ScientificSpinBox()
        self.start_time_correction.setRange(-1e-3, 1e-3)
        self.start_time_correction.setValue(0e-9)
        basic_layout.addWidget(self.start_time_correction, 3, 3)
        
        layout.addWidget(basic_group)
        
        # Time parameters
        time_group = QGroupBox("Time Parameters")
        time_layout = QGridLayout(time_group)
        time_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        time_layout.addWidget(QLabel("Time to Skip (s):"), 0, 0)
        self.time_to_skip = ScientificSpinBox()
        self.time_to_skip.setRange(0, 1e-3)
        self.time_to_skip.setValue(0e-6)
        time_layout.addWidget(self.time_to_skip, 0, 1)
        
        time_layout.addWidget(QLabel("Time to Take (s):"), 0, 2)
        self.time_to_take = ScientificSpinBox()
        self.time_to_take.setRange(1e-9, 1e-3)
        self.time_to_take.setValue(10e-6)
        time_layout.addWidget(self.time_to_take, 0, 3)
        
        # Row 1
        time_layout.addWidget(QLabel("t_before (s):"), 1, 0)
        self.t_before = ScientificSpinBox()
        self.t_before.setRange(1e-12, 1e-6)
        self.t_before.setValue(10e-9)
        time_layout.addWidget(self.t_before, 1, 1)
        
        time_layout.addWidget(QLabel("t_after (s):"), 1, 2)
        self.t_after = ScientificSpinBox()
        self.t_after.setRange(1e-12, 1e-6)
        self.t_after.setValue(60e-9)
        time_layout.addWidget(self.t_after, 1, 3)

        # Row 2
        time_layout.addWidget(QLabel("IQ Threshold Factor:"), 2, 0)
        self.iq_threshold_factor = QDoubleSpinBox()
        self.iq_threshold_factor.setRange(0.0, 2.0)
        self.iq_threshold_factor.setDecimals(3)
        self.iq_threshold_factor.setSingleStep(0.05)
        self.iq_threshold_factor.setValue(0.4)
        self.iq_threshold_factor.setToolTip("Fraction of initial IQ amplitude used to detect start time (default 0.4)")
        time_layout.addWidget(self.iq_threshold_factor, 2, 1)
        
        layout.addWidget(time_group)
        
        # Frequency parameters
        freq_group = QGroupBox("Frequency Parameters")
        freq_layout = QGridLayout(freq_group)
        freq_layout.setSpacing(10)  # Increase spacing between elements
        
        freq_layout.addWidget(QLabel("Freq Min (Hz):"), 0, 0)
        self.freq_min = ScientificSpinBox()
        self.freq_min.setRange(1e6, 10e9)
        self.freq_min.setValue(1e9)
        freq_layout.addWidget(self.freq_min, 0, 1)
        
        freq_layout.addWidget(QLabel("Freq Max (Hz):"), 0, 2)
        self.freq_max = ScientificSpinBox()
        self.freq_max.setRange(1e6, 10e9)
        self.freq_max.setValue(3.5e9)
        freq_layout.addWidget(self.freq_max, 0, 3)
        
        layout.addWidget(freq_group)
        
        # Smoothing parameters
        smooth_group = QGroupBox("Smoothing Parameters")
        smooth_layout = QGridLayout(smooth_group)
        smooth_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        smooth_layout.addWidget(QLabel("Smoothing Window:"), 0, 0)
        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(1, 10000)
        self.smoothing_window.setValue(601)
        smooth_layout.addWidget(self.smoothing_window, 0, 1)
        
        smooth_layout.addWidget(QLabel("Smoothing Width:"), 0, 2)
        self.smoothing_wid = QDoubleSpinBox()
        self.smoothing_wid.setRange(0.1, 100)
        self.smoothing_wid.setValue(3)
        self.smoothing_wid.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_wid, 0, 3)
        
        # Row 1
        smooth_layout.addWidget(QLabel("Smoothing Amp:"), 1, 0)
        self.smoothing_amp = QDoubleSpinBox()
        self.smoothing_amp.setRange(0.1, 10)
        self.smoothing_amp.setValue(1)
        self.smoothing_amp.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_amp, 1, 1)
        
        smooth_layout.addWidget(QLabel("Smoothing Sigma:"), 1, 2)
        self.smoothing_sigma = QDoubleSpinBox()
        self.smoothing_sigma.setRange(0.1, 10)
        self.smoothing_sigma.setValue(1)
        self.smoothing_sigma.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_sigma, 1, 3)
        
        # Row 2
        smooth_layout.addWidget(QLabel("Smoothing Mu:"), 2, 0)
        self.smoothing_mu = QDoubleSpinBox()
        self.smoothing_mu.setRange(-10, 10)
        self.smoothing_mu.setValue(0)
        self.smoothing_mu.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_mu, 2, 1)
        
        layout.addWidget(smooth_group)
        
        # Peak detection parameters
        peak_group = QGroupBox("Peak Detection Parameters")
        peak_layout = QGridLayout(peak_group)
        peak_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        peak_layout.addWidget(QLabel("PB Neighbors:"), 0, 0)
        self.pb_neighbors = QSpinBox()
        self.pb_neighbors.setRange(1, 1000)  # Minimum value is 1, not 0
        self.pb_neighbors.setValue(400)
        self.pb_neighbors.setToolTip("Number of neighbors to compare when searching for pullback local minimum. Must be >= 1 (scipy requirement).")
        peak_layout.addWidget(self.pb_neighbors, 0, 1)
        
        peak_layout.addWidget(QLabel("PB Index Correction:"), 0, 2)
        self.pb_idx_correction = QSpinBox()
        self.pb_idx_correction.setRange(-100, 100)
        self.pb_idx_correction.setValue(0)
        peak_layout.addWidget(self.pb_idx_correction, 0, 3)
        
        # Row 1
        peak_layout.addWidget(QLabel("RC Neighbors:"), 1, 0)
        self.rc_neighbors = QSpinBox()
        self.rc_neighbors.setRange(1, 1000)  # Minimum value is 1, not 0
        self.rc_neighbors.setValue(400)
        self.rc_neighbors.setToolTip("Number of neighbors to compare when searching for recompression local maximum. Must be >= 1 (scipy requirement).")
        peak_layout.addWidget(self.rc_neighbors, 1, 1)
        
        peak_layout.addWidget(QLabel("RC Index Correction:"), 1, 2)
        self.rc_idx_correction = QSpinBox()
        self.rc_idx_correction.setRange(-100, 100)
        self.rc_idx_correction.setValue(0)
        peak_layout.addWidget(self.rc_idx_correction, 1, 3)
        
        layout.addWidget(peak_group)
        
        # STFT parameters
        stft_group = QGroupBox("STFT Parameters")
        stft_layout = QGridLayout(stft_group)
        stft_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        stft_layout.addWidget(QLabel("Sample Rate (Hz):"), 0, 0)
        self.sample_rate = ScientificSpinBox()
        self.sample_rate.setRange(1e6, 1e12)
        self.sample_rate.setValue(128e9)
        stft_layout.addWidget(self.sample_rate, 0, 1)
        
        stft_layout.addWidget(QLabel("Nperseg:"), 0, 2)
        self.nperseg = QSpinBox()
        self.nperseg.setRange(64, 4096)
        self.nperseg.setValue(512)
        stft_layout.addWidget(self.nperseg, 0, 3)
        
        # Row 1
        stft_layout.addWidget(QLabel("Noverlap:"), 1, 0)
        self.noverlap = QSpinBox()
        self.noverlap.setRange(0, 4096)
        self.noverlap.setValue(400)
        stft_layout.addWidget(self.noverlap, 1, 1)
        
        stft_layout.addWidget(QLabel("NFFT:"), 1, 2)
        self.nfft = QSpinBox()
        self.nfft.setRange(64, 8192)
        self.nfft.setValue(5120)
        stft_layout.addWidget(self.nfft, 1, 3)
        
        # Row 2
        stft_layout.addWidget(QLabel("Window:"), 2, 0)
        self.window = QComboBox()
        self.window.addItems(["hann", "hamming", "blackman", "bartlett"])
        self.window.setCurrentText("hann")
        stft_layout.addWidget(self.window, 2, 1)
        
        stft_layout.addWidget(QLabel("Carrier Band Time (s):"), 2, 2)
        self.carrier_band_time = ScientificSpinBox()
        self.carrier_band_time.setRange(1e-12, 1e-6)
        self.carrier_band_time.setValue(250e-9)
        stft_layout.addWidget(self.carrier_band_time, 2, 3)
        
        layout.addWidget(stft_group)
        
        # Blur parameters
        blur_group = QGroupBox("Blur Parameters")
        blur_layout = QGridLayout(blur_group)
        blur_layout.setSpacing(10)  # Increase spacing between elements
        
        blur_layout.addWidget(QLabel("Blur Kernel X:"), 0, 0)
        self.blur_kernel_x = QSpinBox()
        self.blur_kernel_x.setRange(1, 20)
        self.blur_kernel_x.setValue(5)
        blur_layout.addWidget(self.blur_kernel_x, 0, 1)
        
        blur_layout.addWidget(QLabel("Blur Kernel Y:"), 0, 2)
        self.blur_kernel_y = QSpinBox()
        self.blur_kernel_y.setRange(1, 20)
        self.blur_kernel_y.setValue(5)
        blur_layout.addWidget(self.blur_kernel_y, 0, 3)
        
        # Row 1
        blur_layout.addWidget(QLabel("Blur Sigma X:"), 1, 0)
        self.blur_sigx = QDoubleSpinBox()
        self.blur_sigx.setRange(0, 10)
        self.blur_sigx.setValue(0)
        self.blur_sigx.setDecimals(1)
        blur_layout.addWidget(self.blur_sigx, 1, 1)
        
        blur_layout.addWidget(QLabel("Blur Sigma Y:"), 1, 2)
        self.blur_sigy = QDoubleSpinBox()
        self.blur_sigy.setRange(0, 10)
        self.blur_sigy.setValue(0)
        self.blur_sigy.setDecimals(1)
        blur_layout.addWidget(self.blur_sigy, 1, 3)
        
        layout.addWidget(blur_group)
        
        # Filter parameters
        filter_group = QGroupBox("Filter Parameters")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0 - Add notch filter toggle
        filter_layout.addWidget(QLabel("Use Gaussian Notch Filter:"), 0, 0)
        self.use_notch_filter = QCheckBox("Enable carrier frequency removal")
        self.use_notch_filter.setChecked(True)
        self.use_notch_filter.setToolTip("Remove carrier frequency using Gaussian notch filter. Disable if signal is weak or carrier/signal frequencies are close.")
        filter_layout.addWidget(self.use_notch_filter, 0, 1)
        
        filter_layout.addWidget(QLabel("Order:"), 0, 2)
        self.order = QSpinBox()
        self.order.setRange(1, 20)
        self.order.setValue(6)
        filter_layout.addWidget(self.order, 0, 3)
        
        # Row 1
        filter_layout.addWidget(QLabel("Width:"), 1, 0)
        self.wid = ScientificSpinBox()
        self.wid.setRange(1e3, 1e10)
        self.wid.setValue(15e4)
        filter_layout.addWidget(self.wid, 1, 1)
        
        filter_layout.addWidget(QLabel("Uncertainty Multiplier:"), 1, 2)
        self.uncert_mult = QDoubleSpinBox()
        self.uncert_mult.setRange(0.1, 100)
        self.uncert_mult.setValue(10)
        self.uncert_mult.setDecimals(1)
        filter_layout.addWidget(self.uncert_mult, 1, 3)
        
        # Row 2
        filter_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "plasma", "inferno", "magma", "jet"])
        self.cmap.setCurrentText("viridis")
        filter_layout.addWidget(self.cmap, 2, 1)
        
        layout.addWidget(filter_group)
        
        # Material parameters
        material_group = QGroupBox("Material Parameters")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        material_layout.addWidget(QLabel("Bulk Wavespeed (m/s):"), 0, 0)
        self.C0 = QDoubleSpinBox()
        self.C0.setRange(1000, 10000)
        self.C0.setValue(3950)
        self.C0.setDecimals(0)
        material_layout.addWidget(self.C0, 0, 1)
        
        # Row 1
        material_layout.addWidget(QLabel("Density (kg/m³):"), 1, 0)
        self.density = QDoubleSpinBox()
        self.density.setRange(100, 20000)
        self.density.setValue(8960)
        self.density.setDecimals(0)
        material_layout.addWidget(self.density, 1, 1)
        
        layout.addWidget(material_group)
        
        # PDV parameters
        pdv_group = QGroupBox("PDV Parameters")
        pdv_layout = QGridLayout(pdv_group)
        pdv_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        pdv_layout.addWidget(QLabel("Target Wavelength:"), 0, 0)
        self.lam = ScientificSpinBox()
        self.lam.setRange(1e-12, 1e-3)  # Wider range to allow more wavelengths
        self.lam.setValue(1550.016e-9)
        self.lam.setDecimals(15)  # Allow high precision for significant figures
        self.lam.setSingleStep(1e-12)  # Allow fine control with arrow keys
        self.lam.setSuffix(" m")  # Add units suffix
        pdv_layout.addWidget(self.lam, 0, 1)
        
        pdv_layout.addWidget(QLabel("Angle of Incidence (deg):"), 0, 2)
        self.theta = QDoubleSpinBox()
        self.theta.setRange(-90, 90)
        self.theta.setValue(0)
        self.theta.setDecimals(1)
        pdv_layout.addWidget(self.theta, 0, 3)
        
        layout.addWidget(pdv_group)
        
        # Uncertainty parameters
        uncert_group = QGroupBox("Uncertainty Parameters")
        uncert_layout = QGridLayout(uncert_group)
        uncert_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        uncert_layout.addWidget(QLabel("Delta Density (kg/m³):"), 0, 0)
        self.delta_rho = QDoubleSpinBox()
        self.delta_rho.setRange(0, 1000)
        self.delta_rho.setValue(9)
        self.delta_rho.setDecimals(0)
        uncert_layout.addWidget(self.delta_rho, 0, 1)
        
        uncert_layout.addWidget(QLabel("Delta C0 (m/s):"), 0, 2)
        self.delta_C0 = QDoubleSpinBox()
        self.delta_C0.setRange(0, 1000)
        self.delta_C0.setValue(23)
        self.delta_C0.setDecimals(0)
        uncert_layout.addWidget(self.delta_C0, 0, 3)
        
        # Row 1
        uncert_layout.addWidget(QLabel("Delta Wavelength (m):"), 1, 0)
        self.delta_lam = QDoubleSpinBox()
        self.delta_lam.setRange(0, 1e-15)
        self.delta_lam.setValue(8e-18)
        self.delta_lam.setDecimals(18)
        uncert_layout.addWidget(self.delta_lam, 1, 1)
        
        uncert_layout.addWidget(QLabel("Delta Theta (deg):"), 1, 2)
        self.delta_theta = QDoubleSpinBox()
        self.delta_theta.setRange(0, 90)
        self.delta_theta.setValue(5)
        self.delta_theta.setDecimals(1)
        uncert_layout.addWidget(self.delta_theta, 1, 3)
        
        layout.addWidget(uncert_group)
        
        # Plot parameters
        plot_group = QGroupBox("Plot Parameters")
        plot_layout = QGridLayout(plot_group)
        plot_layout.setSpacing(10)  # Increase spacing between elements
        
        plot_layout.addWidget(QLabel("Figure Width:"), 0, 0)
        self.plot_width = QSpinBox()
        self.plot_width.setRange(5, 100)
        self.plot_width.setValue(30)
        plot_layout.addWidget(self.plot_width, 0, 1)
        
        plot_layout.addWidget(QLabel("Figure Height:"), 0, 2)
        self.plot_height = QSpinBox()
        self.plot_height.setRange(5, 100)
        self.plot_height.setValue(10)
        plot_layout.addWidget(self.plot_height, 0, 3)
        
        # Row 1
        plot_layout.addWidget(QLabel("DPI:"), 1, 0)
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(50, 600)
        self.plot_dpi.setValue(300)
        plot_layout.addWidget(self.plot_dpi, 1, 1)
        
        layout.addWidget(plot_group)
        
        # Image selection group
        image_group = QGroupBox("ALPSS Output Images")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(10)
        
        # Description
        desc_label = QLabel("Only the combined ALPSS summary plot and optional IQ start-time diagnostic are available in the current fast-plot workflow.")
        desc_label.setWordWrap(True)
        image_layout.addWidget(desc_label)
        
        self.save_combined_plot = QCheckBox("Combined ALPSS Summary Plot")
        self.save_combined_plot.setChecked(True)
        self.save_combined_plot.setToolTip("Generate the 5-panel summary plot (velocity, noise, ROI spectrograms, IQ analysis).")
        image_layout.addWidget(self.save_combined_plot)
        
        self.save_iq_start_time_plot = QCheckBox("IQ Start Time Detection Plot")
        self.save_iq_start_time_plot.setChecked(False)
        self.save_iq_start_time_plot.setToolTip("Generate IQ analysis start time detection plot with threshold overlays.")
        image_layout.addWidget(self.save_iq_start_time_plot)
        
        # Select all/none buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_images = QPushButton("Select All")
        self.select_all_images.clicked.connect(self.select_all_alpss_images)
        select_buttons_layout.addWidget(self.select_all_images)
        
        self.deselect_all_images = QPushButton("Deselect All")
        self.deselect_all_images.clicked.connect(self.deselect_all_alpss_images)
        select_buttons_layout.addWidget(self.deselect_all_images)
        
        image_layout.addLayout(select_buttons_layout)
        
        layout.addWidget(image_group)
        
        # Output file selection group
        output_files_group = QGroupBox("ALPSS Output Files")
        output_files_layout = QVBoxLayout(output_files_group)
        output_files_layout.setSpacing(10)
        
        # Description
        output_desc_label = QLabel("Select which ALPSS output files to save:")
        output_desc_label.setWordWrap(True)
        output_files_layout.addWidget(output_desc_label)
        
        # Output file checkboxes
        self.save_velocity_csv = QCheckBox("Velocity CSV (*--velocity.csv)")
        self.save_velocity_csv.setChecked(True)
        self.save_velocity_csv.setToolTip("Save raw velocity data")
        output_files_layout.addWidget(self.save_velocity_csv)
        
        self.save_velocity_smooth_csv = QCheckBox("Smoothed Velocity CSV (*--velocity--smooth.csv)")
        self.save_velocity_smooth_csv.setChecked(True)
        self.save_velocity_smooth_csv.setToolTip("Save smoothed velocity data")
        output_files_layout.addWidget(self.save_velocity_smooth_csv)
        
        self.save_velocity_uncert_csv = QCheckBox("Velocity with Uncertainty (*--vel--uncert.csv)")
        self.save_velocity_uncert_csv.setChecked(True)
        self.save_velocity_uncert_csv.setToolTip("Save velocity data with uncertainty")
        output_files_layout.addWidget(self.save_velocity_uncert_csv)
        
        self.save_velocity_smooth_uncert_csv = QCheckBox("Smoothed Velocity with Uncertainty (*--vel-smooth-with-uncert.csv)")
        self.save_velocity_smooth_uncert_csv.setChecked(True)
        self.save_velocity_smooth_uncert_csv.setToolTip("Save smoothed velocity with uncertainty (required for SPADE)")
        output_files_layout.addWidget(self.save_velocity_smooth_uncert_csv)
        
        self.save_results_csv = QCheckBox("Results CSV (*--results.csv)")
        self.save_results_csv.setChecked(True)
        self.save_results_csv.setToolTip("Save analysis results with uncertainties")
        output_files_layout.addWidget(self.save_results_csv)
        
        self.save_noise_csv = QCheckBox("Noise Fraction CSV (*--noise--frac.csv)")
        self.save_noise_csv.setChecked(True)
        self.save_noise_csv.setToolTip("Save noise fraction data")
        output_files_layout.addWidget(self.save_noise_csv)
        
        # Smart selection for combined mode
        self.smart_selection_checkbox = QCheckBox("Smart Selection for Combined Mode")
        self.smart_selection_checkbox.setChecked(True)
        self.smart_selection_checkbox.setToolTip("When ALPSS+SPADE mode is selected, automatically save files needed for SPADE analysis + enhanced filtering")
        output_files_layout.addWidget(self.smart_selection_checkbox)
        
        # Select all/none buttons for output files
        output_select_buttons_layout = QHBoxLayout()
        self.select_all_output_files = QPushButton("Select All")
        self.select_all_output_files.clicked.connect(self.select_all_output_files_func)
        output_select_buttons_layout.addWidget(self.select_all_output_files)
        
        self.deselect_all_output_files = QPushButton("Deselect All")
        self.deselect_all_output_files.clicked.connect(self.deselect_all_output_files_func)
        output_select_buttons_layout.addWidget(self.deselect_all_output_files)
        
        self.smart_select_output_files = QPushButton("Smart Select")
        self.smart_select_output_files.clicked.connect(self.smart_select_output_files_func)
        self.smart_select_output_files.setToolTip("Select files needed for SPADE analysis + enhanced filtering")
        output_select_buttons_layout.addWidget(self.smart_select_output_files)
        
        output_files_layout.addLayout(output_select_buttons_layout)
        
        layout.addWidget(output_files_group)
        
        scroll.setWidget(scroll_widget)
        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "ALPSS Parameters")
        
    def select_all_alpss_images(self):
        """Select all ALPSS output images"""
        self.save_combined_plot.setChecked(True)
        self.save_iq_start_time_plot.setChecked(True)
        
    def deselect_all_alpss_images(self):
        """Deselect all ALPSS output images"""
        self.save_combined_plot.setChecked(False)
        self.save_iq_start_time_plot.setChecked(False)
        
    def select_all_output_files_func(self):
        """Select all ALPSS output files"""
        self.save_velocity_csv.setChecked(True)
        self.save_velocity_smooth_csv.setChecked(True)
        self.save_velocity_uncert_csv.setChecked(True)
        self.save_velocity_smooth_uncert_csv.setChecked(True)
        self.save_results_csv.setChecked(True)
        self.save_noise_csv.setChecked(True)
        
    def deselect_all_output_files_func(self):
        """Deselect all ALPSS output files"""
        self.save_velocity_csv.setChecked(False)
        self.save_velocity_smooth_csv.setChecked(False)
        self.save_velocity_uncert_csv.setChecked(False)
        self.save_velocity_smooth_uncert_csv.setChecked(False)
        self.save_results_csv.setChecked(False)
        self.save_noise_csv.setChecked(False)
        
    def smart_select_output_files_func(self):
        """Smart select files needed for SPADE analysis + enhanced filtering"""
        # For SPADE analysis, we need the smoothed velocity with uncertainty file + noise file
        self.save_velocity_csv.setChecked(False)
        self.save_velocity_smooth_csv.setChecked(False)
        self.save_velocity_uncert_csv.setChecked(False)
        self.save_velocity_smooth_uncert_csv.setChecked(True)  # Main file SPADE needs
        self.save_results_csv.setChecked(False)
        self.save_noise_csv.setChecked(True)  # Also save noise file for enhanced filtering
        
    def create_spade_params_tab(self):
        """Create SPADE parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
        # Config File Management
        config_group = QGroupBox("Configuration File Management")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        
        # Mode selection
        config_layout.addWidget(QLabel("Parameter Entry Mode:"), 0, 0)
        self.spade_config_mode_group = QButtonGroup()
        self.spade_manual_mode = QRadioButton("Manual Entry (use GUI controls)")
        self.spade_config_mode = QRadioButton("Use Config File")
        self.spade_config_mode_group.addButton(self.spade_manual_mode)
        self.spade_config_mode_group.addButton(self.spade_config_mode)
        self.spade_manual_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.spade_manual_mode)
        mode_layout.addWidget(self.spade_config_mode)
        config_layout.addLayout(mode_layout, 0, 1, 1, 3)
        
        # Config file path
        config_layout.addWidget(QLabel("Config File Path:"), 1, 0)
        self.spade_config_path = QLineEdit()
        self.spade_config_path.setPlaceholderText("Select a config file to load SPADE parameters")
        self.spade_config_path.setEnabled(False)
        config_layout.addWidget(self.spade_config_path, 1, 1, 1, 2)
        
        self.spade_config_browse_btn = QPushButton("Browse")
        self.spade_config_browse_btn.clicked.connect(self.browse_spade_config)
        self.spade_config_browse_btn.setEnabled(False)
        config_layout.addWidget(self.spade_config_browse_btn, 1, 3)
        
        # Load and Save buttons
        self.spade_config_load_btn = QPushButton("Load Config")
        self.spade_config_load_btn.clicked.connect(self.load_spade_config)
        self.spade_config_load_btn.setEnabled(False)
        config_layout.addWidget(self.spade_config_load_btn, 2, 0)
        
        self.spade_config_save_btn = QPushButton("Save Current Settings to Config")
        self.spade_config_save_btn.clicked.connect(self.save_spade_config)
        config_layout.addWidget(self.spade_config_save_btn, 2, 1, 1, 3)
        
        # Connect mode change
        self.spade_config_mode.toggled.connect(self.on_spade_config_mode_changed)
        
        layout.addWidget(config_group)
        
        # Experiment type selection
        experiment_group = QGroupBox("Experiment Type")
        experiment_layout = QVBoxLayout(experiment_group)
        experiment_layout.setSpacing(10)
        
        # Checkboxes for experiment type (can select both)
        self.experiment_velocity_shots = QCheckBox("Velocity Shots")
        self.experiment_velocity_shots.setChecked(True)
        experiment_layout.addWidget(self.experiment_velocity_shots)
        
        self.experiment_spall_analysis = QCheckBox("Spall Analysis")
        self.experiment_spall_analysis.setChecked(False)
        experiment_layout.addWidget(self.experiment_spall_analysis)

        self.experiment_hel_detection = QCheckBox("HEL Detection")
        self.experiment_hel_detection.setChecked(False)
        self.experiment_hel_detection.toggled.connect(self.on_hel_detection_toggled)
        experiment_layout.addWidget(self.experiment_hel_detection)
        
        # Description text for experiment types
        experiment_desc = QPlainTextEdit()
        experiment_desc.setMaximumHeight(120)
        experiment_desc.setReadOnly(True)
        experiment_desc.setPlainText(
            "Velocity Shots: Generate velocity shot summary with impact velocity calculations and combined velocity plot.\n"
            "Spall Analysis: Generate spall summary with spall strength and strain rate analysis.\n"
            "Note: You can select both options to run both analyses."
        )
        experiment_layout.addWidget(experiment_desc)
        
        layout.addWidget(experiment_group)
        
        # Material properties
        material_group = QGroupBox("Material Properties")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(10)  # Increase spacing between elements
        
        # Density
        material_layout.addWidget(QLabel("Default Density (kg/m³):"), 0, 0)
        self.spade_density = QDoubleSpinBox()
        self.spade_density.setRange(1000, 25000)
        self.spade_density.setDecimals(2)
        self.spade_density.setValue(8960)  # Default: Copper
        self.spade_density.setToolTip("Fallback value if material not found in database")
        material_layout.addWidget(self.spade_density, 0, 1)
        
        # Acoustic velocity
        material_layout.addWidget(QLabel("Default Acoustic Velocity (m/s):"), 0, 2)
        self.spade_acoustic_velocity = QDoubleSpinBox()
        self.spade_acoustic_velocity.setRange(1000, 10000)
        self.spade_acoustic_velocity.setDecimals(2)
        self.spade_acoustic_velocity.setValue(3950)  # Default: Copper
        self.spade_acoustic_velocity.setToolTip("Fallback value if material not found in database")
        material_layout.addWidget(self.spade_acoustic_velocity, 0, 3)
        
        # Add help text explaining material properties priority
        material_help = QPlainTextEdit()
        material_help.setMaximumHeight(80)
        material_help.setReadOnly(True)
        material_help.setPlainText(
            "📌 Material Properties Priority:\n"
            "1. Parameter file explicit columns (Density_kg_m3, Bulk_Wave_Speed_m_s) - HIGHEST\n"
            "2. Material Database lookup via 'Sample material' column - OVERRIDES GUI values\n"
            "3. GUI values above - FALLBACK only when material not found\n"
            "💡 Tip: Leave defaults as-is. System auto-detects from parameter file!"
        )
        material_layout.addWidget(material_help, 1, 0, 1, 4)
        
        layout.addWidget(material_group)
        
        # Analysis method info (no longer user-selectable)
        # Note: Calculations now use hybrid approach:
        # - Spall strength: always calculated using 'max_min' method
        # - Strain rate: always calculated using 'hybrid_5_segment' method
        info_group = QGroupBox("Spall Analysis Method")
        info_layout = QVBoxLayout(info_group)
        info_label = QLabel("ℹ️ Calculations use hybrid approach:\n• Spall strength: calculated using 'max_min' method (peak/valley detection)\n• Strain rate: calculated using 'hybrid_5_segment' method (5-segment line fitting)")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)

        # HEL Detection parameters
        self.hel_group = QGroupBox("HEL Detection Parameters")
        hel_layout = QGridLayout(self.hel_group)
        hel_layout.setSpacing(10)

        hel_layout.addWidget(QLabel("HEL Start Time (ns):"), 0, 0)
        self.hel_start_time_ns = QDoubleSpinBox()
        self.hel_start_time_ns.setRange(-1000.0, 1000.0)
        self.hel_start_time_ns.setDecimals(2)
        self.hel_start_time_ns.setValue(0.0)
        self.hel_start_time_ns.setToolTip("Relative to t=0 (after alignment). Start HEL analysis window.")
        hel_layout.addWidget(self.hel_start_time_ns, 0, 1)

        hel_layout.addWidget(QLabel("HEL End Time (ns):"), 0, 2)
        self.hel_end_time_ns = QDoubleSpinBox()
        self.hel_end_time_ns.setRange(-1000.0, 5000.0)
        self.hel_end_time_ns.setDecimals(2)
        self.hel_end_time_ns.setValue(12.0)
        self.hel_end_time_ns.setToolTip("Relative to t=0 (after alignment). End HEL analysis window.")
        hel_layout.addWidget(self.hel_end_time_ns, 0, 3)

        hel_layout.addWidget(QLabel("Angle Threshold (deg):"), 1, 0)
        self.hel_angle_threshold_deg = QDoubleSpinBox()
        self.hel_angle_threshold_deg.setRange(1.0, 89.0)
        self.hel_angle_threshold_deg.setDecimals(1)
        self.hel_angle_threshold_deg.setValue(45.0)
        hel_layout.addWidget(self.hel_angle_threshold_deg, 1, 1)

        hel_layout.addWidget(QLabel("Min HEL Velocity (m/s):"), 1, 2)
        self.minimum_hel_velocity = QDoubleSpinBox()
        self.minimum_hel_velocity.setRange(0.0, 2000.0)
        self.minimum_hel_velocity.setDecimals(2)
        self.minimum_hel_velocity.setValue(15.0)
        self.minimum_hel_velocity.setToolTip("Reject HEL detections below this free-surface velocity.")
        hel_layout.addWidget(self.minimum_hel_velocity, 1, 3)

        hel_layout.addWidget(QLabel("Min Consecutive Points:"), 2, 0)
        self.hel_detection_min_points = QSpinBox()
        self.hel_detection_min_points.setRange(1, 10000)
        self.hel_detection_min_points.setValue(50)
        self.hel_detection_min_points.setToolTip("Minimum number of consecutive low-slope points required to accept HEL plateau.")
        hel_layout.addWidget(self.hel_detection_min_points, 2, 1)

        # Initially hidden unless HEL Detection is enabled
        self.hel_group.setVisible(self.experiment_hel_detection.isChecked())
        layout.addWidget(self.hel_group)
        
        # Signal length
        signal_group = QGroupBox("Signal Length")
        signal_layout = QGridLayout(signal_group)
        signal_layout.setSpacing(10)  # Increase spacing between elements
        
        signal_layout.addWidget(QLabel("Signal Length:"), 0, 0)
        self.signal_length_combo = QComboBox()
        self.signal_length_combo.addItems(["Full Signal (None)", "Custom..."])
        self.signal_length_combo.currentIndexChanged.connect(self.toggle_signal_length_spin)
        signal_layout.addWidget(self.signal_length_combo, 0, 1)
        
        signal_layout.addWidget(QLabel("Custom Length (ns):"), 1, 0)
        self.signal_length_spin = QDoubleSpinBox()
        self.signal_length_spin.setRange(0, 10000)
        self.signal_length_spin.setValue(20.0)
        self.signal_length_spin.setSuffix(" ns")
        self.signal_length_spin.setEnabled(False)
        signal_layout.addWidget(self.signal_length_spin, 1, 1)
        
        layout.addWidget(signal_group)
        
        # Filtering parameters
        filter_group = QGroupBox("Peak Detection Parameters")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(10)  # Increase spacing between elements
        
        filter_layout.addWidget(QLabel("Prominence Factor:"), 0, 0)
        self.prominence_factor = QDoubleSpinBox()
        self.prominence_factor.setRange(0, 1)
        self.prominence_factor.setSingleStep(0.01)
        self.prominence_factor.setValue(0.01)
        self.prominence_factor.setDecimals(3)
        self.prominence_factor.setSuffix(" (fraction)")
        filter_layout.addWidget(self.prominence_factor, 0, 1)
        
        filter_layout.addWidget(QLabel("Peak Distance (ns):"), 0, 2)
        self.peak_distance_ns = QDoubleSpinBox()
        self.peak_distance_ns.setRange(0, 1000)
        self.peak_distance_ns.setValue(5.0)
        self.peak_distance_ns.setSuffix(" ns")
        filter_layout.addWidget(self.peak_distance_ns, 0, 3)
        
        layout.addWidget(filter_group)

        # Spall detection window parameters
        spall_group = QGroupBox("Spall Window & Thresholds")
        spall_layout = QGridLayout(spall_group)
        spall_layout.setSpacing(10)

        spall_layout.addWidget(QLabel("Spall Start Time (ns):"), 0, 0)
        self.spall_start_time_ns = QDoubleSpinBox()
        self.spall_start_time_ns.setRange(-1000.0, 10000.0)
        self.spall_start_time_ns.setDecimals(2)
        self.spall_start_time_ns.setValue(10.0)
        self.spall_start_time_ns.setToolTip("Lower bound of the velocity window (relative to t=0) for spall fitting.")
        spall_layout.addWidget(self.spall_start_time_ns, 0, 1)

        spall_layout.addWidget(QLabel("Spall End Time (ns):"), 0, 2)
        self.spall_end_time_ns = QDoubleSpinBox()
        self.spall_end_time_ns.setRange(-1000.0, 10000.0)
        self.spall_end_time_ns.setDecimals(2)
        self.spall_end_time_ns.setValue(100.0)
        self.spall_end_time_ns.setToolTip("Upper bound of the velocity window (relative to t=0) for spall fitting.")
        spall_layout.addWidget(self.spall_end_time_ns, 0, 3)

        spall_layout.addWidget(QLabel("Threshold Velocity (m/s):"), 1, 0)
        self.threshold_velocity_ms = QDoubleSpinBox()
        self.threshold_velocity_ms.setRange(0.0, 1000.0)
        self.threshold_velocity_ms.setDecimals(2)
        self.threshold_velocity_ms.setValue(10.0)
        self.threshold_velocity_ms.setToolTip("Minimum velocity required before searching for spall peaks (also used for alignment warnings).")
        spall_layout.addWidget(self.threshold_velocity_ms, 1, 1)

        layout.addWidget(spall_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        output_layout.setSpacing(10)  # Increase spacing between elements
        
        self.plot_individual = QCheckBox("Generate Individual Plots")
        self.plot_individual.setChecked(True)
        self.plot_individual.setToolTip("Generate individual plots for each file:\n"
                                       "• Spall analysis: spall detection plots saved in 'spall_plots/spalled/' or 'spall_plots/dns/'\n"
                                       "• HEL detection: HEL window plots with peak/valley markers saved in 'HEL_plots' subfolder\n"
                                       "Plots organized in separate subfolders within SPADE_analysis for easy review")
        output_layout.addWidget(self.plot_individual, 0, 0)
        
        self.save_summary = QCheckBox("Save Summary Table")
        self.save_summary.setChecked(True)
        output_layout.addWidget(self.save_summary, 0, 1)
        
        self.show_plots = QCheckBox("Show Plots (if possible)")
        self.show_plots.setChecked(False)
        output_layout.addWidget(self.show_plots, 0, 2)
        
        layout.addWidget(output_group)
        
        # Plot axis limits
        axis_group = QGroupBox("Combined Plot Axis Limits")
        axis_layout = QGridLayout(axis_group)
        axis_layout.setSpacing(10)  # Increase spacing between elements
        
        # X-axis limits for main plot
        axis_layout.addWidget(QLabel("Main Plot X-Limits (ns):"), 0, 0)
        x_limits_layout = QHBoxLayout()
        self.x_min_main = QDoubleSpinBox()
        self.x_min_main.setRange(-1000, 1000)
        self.x_min_main.setValue(0)
        self.x_min_main.setSuffix(" ns")
        x_limits_layout.addWidget(self.x_min_main)
        
        x_limits_layout.addWidget(QLabel("to"))
        
        self.x_max_main = QDoubleSpinBox()
        self.x_max_main.setRange(-1000, 1000)
        self.x_max_main.setValue(100)
        self.x_max_main.setSuffix(" ns")
        x_limits_layout.addWidget(self.x_max_main)
        axis_layout.addLayout(x_limits_layout, 0, 1)
        
        # Y-axis limits for main plot
        axis_layout.addWidget(QLabel("Main Plot Y-Limits (m/s):"), 1, 0)
        y_limits_layout = QHBoxLayout()
        self.y_min_main = QDoubleSpinBox()
        self.y_min_main.setRange(-1000, 1000)
        self.y_min_main.setValue(0)
        self.y_min_main.setSuffix(" m/s")
        y_limits_layout.addWidget(self.y_min_main)
        
        y_limits_layout.addWidget(QLabel("to"))
        
        self.y_max_main = QDoubleSpinBox()
        self.y_max_main.setRange(0, 2000)
        self.y_max_main.setValue(700)
        self.y_max_main.setSuffix(" m/s")
        y_limits_layout.addWidget(self.y_max_main)
        axis_layout.addLayout(y_limits_layout, 1, 1)
        
        # X-axis limits for zoomed plot
        axis_layout.addWidget(QLabel("Zoomed Plot X-Limits (ns):"), 2, 0)
        x_zoom_layout = QHBoxLayout()
        self.x_min_zoom = QDoubleSpinBox()
        self.x_min_zoom.setRange(-1000, 1000)
        self.x_min_zoom.setValue(0)
        self.x_min_zoom.setSuffix(" ns")
        x_zoom_layout.addWidget(self.x_min_zoom)
        
        x_zoom_layout.addWidget(QLabel("to"))
        
        self.x_max_zoom = QDoubleSpinBox()
        self.x_max_zoom.setRange(-1000, 1000)
        self.x_max_zoom.setValue(20)
        self.x_max_zoom.setSuffix(" ns")
        x_zoom_layout.addWidget(self.x_max_zoom)
        axis_layout.addLayout(x_zoom_layout, 2, 1)
        
        # Y-axis limits for zoomed plot
        axis_layout.addWidget(QLabel("Zoomed Plot Y-Limits (m/s):"), 3, 0)
        y_zoom_layout = QHBoxLayout()
        self.y_min_zoom = QDoubleSpinBox()
        self.y_min_zoom.setRange(-1000, 1000)
        self.y_min_zoom.setValue(0)
        self.y_min_zoom.setSuffix(" m/s")
        y_zoom_layout.addWidget(self.y_min_zoom)
        
        y_zoom_layout.addWidget(QLabel("to"))
        
        self.y_max_zoom = QDoubleSpinBox()
        self.y_max_zoom.setRange(0, 2000)
        self.y_max_zoom.setValue(700)
        self.y_max_zoom.setSuffix(" m/s")
        y_zoom_layout.addWidget(self.y_max_zoom)
        axis_layout.addLayout(y_zoom_layout, 3, 1)
        
        # Auto-calculate checkbox
        self.auto_calculate_limits = QCheckBox("Auto-calculate limits from data")
        self.auto_calculate_limits.setChecked(True)
        self.auto_calculate_limits.setToolTip("Automatically calculate axis limits from the data. Uncheck to use custom limits above.")
        axis_layout.addWidget(self.auto_calculate_limits, 4, 0, 1, 2)
        
        layout.addWidget(axis_group)

        # Combined Velocity Plot controls
        combined_group = QGroupBox("Combined Velocity Plot (Aligned)")
        combined_layout = QGridLayout(combined_group)
        combined_layout.setSpacing(10)

        self.generate_all_velocity_plot = QCheckBox("Generate combined aligned velocity plot")
        self.generate_all_velocity_plot.setChecked(True)
        self.generate_all_velocity_plot.setToolTip("Generate combined plot of all velocity traces aligned at t=0 (threshold), with noise fraction > 1 filtered, and uncertainty threshold applied.")
        combined_layout.addWidget(self.generate_all_velocity_plot, 0, 0, 1, 2)

        # Alignment threshold
        combined_layout.addWidget(QLabel("Alignment threshold:"), 1, 0)
        self.align_velocity_threshold = QDoubleSpinBox()
        self.align_velocity_threshold.setRange(0.0, 1000.0)
        self.align_velocity_threshold.setDecimals(2)
        self.align_velocity_threshold.setValue(30.0)
        self.align_velocity_threshold.setSuffix(" m/s")
        self.align_velocity_threshold.setToolTip("Set the velocity threshold for alignment (t=0 at first time ≥ threshold).")
        combined_layout.addWidget(self.align_velocity_threshold, 1, 1)

        # Uncertainty threshold
        combined_layout.addWidget(QLabel("Uncertainty threshold:"), 2, 0)
        self.uncertainty_threshold = QDoubleSpinBox()
        self.uncertainty_threshold.setRange(0.0, 5000.0)
        self.uncertainty_threshold.setDecimals(2)
        self.uncertainty_threshold.setValue(50.0)
        self.uncertainty_threshold.setSuffix(" m/s")
        self.uncertainty_threshold.setToolTip("Remove points with uncertainty > this threshold from the combined plot.")
        combined_layout.addWidget(self.uncertainty_threshold, 2, 1)

        # Include uncertainty bands
        self.include_uncert_bands = QCheckBox("Include uncertainty bands (±)")
        self.include_uncert_bands.setChecked(True)
        self.include_uncert_bands.setToolTip("Shade ± uncertainty around each trace after filtering and alignment.")
        combined_layout.addWidget(self.include_uncert_bands, 3, 0, 1, 2)

        # Uncertainty band alpha
        combined_layout.addWidget(QLabel("Uncertainty band alpha:"), 4, 0)
        self.uncert_alpha = QDoubleSpinBox()
        self.uncert_alpha.setRange(0.0, 1.0)
        self.uncert_alpha.setDecimals(2)
        self.uncert_alpha.setSingleStep(0.05)
        self.uncert_alpha.setValue(0.2)
        combined_layout.addWidget(self.uncert_alpha, 4, 1)

        # Zoom window (ns)
        combined_layout.addWidget(QLabel("Zoom window (ns):"), 5, 0)
        self.zoom_window_ns = QSpinBox()
        self.zoom_window_ns.setRange(0, 100000)
        self.zoom_window_ns.setValue(1000)
        self.zoom_window_ns.setToolTip("Length of time window shown in the bottom (zoom) plot, starting at t=0.")
        combined_layout.addWidget(self.zoom_window_ns, 5, 1)

        layout.addWidget(combined_group)
        
        # SPADE input mode
        spade_input_group = QGroupBox("SPADE Input Mode")
        spade_input_layout = QVBoxLayout(spade_input_group)
        
        self.spade_auto_radio = QCheckBox("Automatic: Use ALPSS outputs")
        self.spade_auto_radio.setChecked(True)
        self.spade_auto_radio.toggled.connect(self.on_spade_input_mode_changed)
        spade_input_layout.addWidget(self.spade_auto_radio)
        
        self.spade_manual_radio = QCheckBox("Manual: Select SPADE input files")
        self.spade_manual_radio.toggled.connect(self.on_spade_input_mode_changed)
        spade_input_layout.addWidget(self.spade_manual_radio)
        
        # Manual input selection
        manual_input_layout = QHBoxLayout()
        self.spade_input_path = QLineEdit()
        self.spade_input_path.setPlaceholderText("Select SPADE input files or directory...")
        self.spade_input_path.setEnabled(False)
        manual_input_layout.addWidget(self.spade_input_path)
        
        self.spade_input_btn = QPushButton("Browse")
        self.spade_input_btn.clicked.connect(self.select_spade_input)
        self.spade_input_btn.setEnabled(False)
        manual_input_layout.addWidget(self.spade_input_btn)
        
        spade_input_layout.addLayout(manual_input_layout)
        
        # File pattern for manual mode
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File Pattern:"))
        self.spade_file_pattern = QLineEdit("*--vel-smooth-with-uncert.csv")
        self.spade_file_pattern.setEnabled(False)
        pattern_layout.addWidget(self.spade_file_pattern)
        spade_input_layout.addLayout(pattern_layout)
        
        layout.addWidget(spade_input_group)
        
        layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "SPADE Parameters")
        
    def create_documentation_tab(self):
        """Create documentation tab with ALPSS parameter key"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Create scroll area for documentation
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        doc_layout = QVBoxLayout(scroll_widget)
        doc_layout.setSpacing(15)  # Increase spacing between groups
        
        # ALPSS Parameter Key
        alpss_doc_group = QGroupBox("ALPSS Parameter Key")
        alpss_doc_layout = QVBoxLayout(alpss_doc_group)
        
        alpss_doc_text = QPlainTextEdit()
        alpss_doc_text.setReadOnly(True)
        alpss_doc_text.setMaximumHeight(400)
        alpss_doc_text.setPlainText("""
ALPSS Parameter Key:

filename:                   str; filename for the data to run
save_data:                  str; 'yes' or 'no' to save output data
save_all_plots:             str; 'no', 'subfolder', or 'main_dir' to control plot saving location (CSV data files are always saved when save_data='yes')
start_time_user:            str or float; if 'none' the program will attempt to find the
                                             signal start time automatically. if float then
                                             the program will use that as the signal start time
header_lines:               int; number of header lines to skip in the data file
time_to_skip:               float; the amount of time to skip in the full data file before beginning to read in data
time_to_take:               float; the amount of time to take in the data file after skipping time_to_skip
t_before:                   float; amount of time before the signal start time to include in the velocity calculation
t_after:                    float; amount of time after the signal start time to include in the velocity calculation
start_time_correction:      float; amount of time to adjust the signal start time by
freq_min:                   float; minimum frequency for the region of interest
freq_max:                   float; maximum frequency for the region of interest
smoothing_window:           int; number of points to use for the smoothing window. must be an odd number
smoothing_wid:              float; half the width of the normal distribution used
                                   to calculate the smoothing weights (recommend 3)
smoothing_amp:              float; amplitude of the normal distribution used to calculate
                                   the smoothing weights (recommend 1)
smoothing_sigma:            float; standard deviation of the normal distribution used
                                   to calculate the smoothing weights (recommend 1)
smoothing_mu:               float; mean of the normal distribution used to calculate
                                   the smoothing weights (recommend 0)
pb_neighbors:               int; number of neighbors to compare to when searching
                                     for the pullback local minimum (must be >= 1)
pb_idx_correction:          int; number of local minima to adjust by if the program grabs the wrong one
rc_neighbors:               int; number of neighbors to compare to when searching
                                     for the recompression local maximum (must be >= 1)
rc_idx_correction:          int; number of local maxima to adjust by if the program grabs the wrong one
sample_rate:                float; sample rate of the oscilloscope used in the experiment
nperseg:                    int; number of points to use per segment of the stft
noverlap:                   int; number of points to overlap per segment of the stft
nfft:                       int; number of points to zero pad per segment of the stft
window:                     str or tuple or array_like; window function to use for the stft (recommend 'hann')
blur_kernel:                tuple; kernel size for gaussian blur smoothing (recommend (5, 5))
blur_sigx:                  float; standard deviation of the gaussian blur kernel in the x direction (recommend 0)
blur_sigy:                  float; standard deviation of the gaussian blur kernel in the y direction (recommend 0)
carrier_band_time:          float; length of time from the beginning of the imported data window to average
                                   the frequency of the top of the carrier band in the thresholded spectrogram
cmap:                       str; colormap for the spectrograms (recommend 'viridis')
uncert_mult:                float; factor to multiply the velocity uncertainty by when plotting - allows for easier
                                   visulaization when uncertainties are small
use_notch_filter:           bool; whether to use the gaussian notch filter to remove the carrier band (recommend True for strong signals, False for weak signals)
order:                      int; order for the gaussian notch filter used to remove the carrier band (recommend 6)
wid:                        float; width of the gaussian notch filter used to remove the carrier band (recommend 1e8)
lam:                        float; wavelength of the target laser
C0:                         float; bulk wavespeed of the sample
density:                    float; density of the sample
delta_rho:                  float; uncertainty in density of the sample
delta_C0:                   float; uncertainty in the bulk wavespeed of the sample
delta_lam:                  float; uncertainty in the wavelength of the target laser
theta:                      float; angle of incidence of the PDV probe
delta_theta:                float; uncertainty in the angle of incidence of the PDV probe
exp_data_dir:               str; directory from which to read the experimental data file
out_files_dir:              str; directory to save output data to
display_plots:              str; 'yes' to display the final plots and 'no' to not display them. if save_data='yes'
                                     and and display_plots='no' the plots will be saved but not displayed
spall_calculation:          str; 'yes' to run the calculations for the spall analysis and 'no' to extract the velocity
                                  without doing the spall analysis
plot_figsize:               tuple; figure size for the final plots
plot_dpi:                   float; dpi for the final plots
        """)
        alpss_doc_layout.addWidget(alpss_doc_text)
        doc_layout.addWidget(alpss_doc_group)
        
        # SPADE Documentation
        spade_doc_group = QGroupBox("SPADE Analysis Information")
        spade_doc_layout = QVBoxLayout(spade_doc_group)
        
        spade_doc_text = QPlainTextEdit()
        spade_doc_text.setReadOnly(True)
        spade_doc_text.setMaximumHeight(300)
        spade_doc_text.setPlainText("""
SPADE (Spall Analysis Toolkit) Parameters:

Material Properties:
- density: Material density in kg/m³
- acoustic_velocity: Acoustic velocity in m/s

Analysis Model:
- NOTE: The system now uses a hybrid approach by default:
  * Spall strength: calculated using 'max_min' method (peak/valley detection)
  * Strain rate: calculated using 'hybrid_5_segment' method (5-segment line fitting)
- The analysis_model dropdown is kept for backward compatibility but no longer controls the actual calculations

Signal Processing:
- signal_length_ns: Custom signal length in nanoseconds (None for full signal)
- smooth_window: Smoothing window size (odd number)
- polyorder: Polynomial order for Savitzky-Golay filter

Peak Detection:
- prominence_factor: Minimum peak prominence as fraction of signal
- peak_distance_ns: Minimum distance between peaks in nanoseconds

Output Options:
- plot_individual: Generate individual analysis plots
  * For spall analysis: individual spall detection plots
  * For HEL detection: individual HEL detection plots (shows full trace + zoomed HEL window with peak/valley markers)
- save_summary_table: Save summary CSV with results
- show_plots: Display plots during analysis (if possible)

Input Requirements:
- Velocity files must be in CSV format with 'Time' and 'Velocity' columns
- Time should be in nanoseconds
- Velocity should be in m/s
        """)
        spade_doc_layout.addWidget(spade_doc_text)
        doc_layout.addWidget(spade_doc_group)
        
        # Usage Instructions
        usage_group = QGroupBox("Usage Instructions")
        usage_layout = QVBoxLayout(usage_group)
        
        usage_text = QPlainTextEdit()
        usage_text.setReadOnly(True)
        usage_text.setMaximumHeight(200)
        usage_text.setPlainText("""
How to Use This GUI:

1. File Selection Tab:
   - Choose Manual Select (pick files individually) or Multiple Files (directory)
   - Select input files or directory
   - Set output directory

2. Analysis Mode Tab:
   - ALPSS Only: Process raw data to velocity traces
   - SPADE Only: Analyze existing velocity files
   - Combined: Full pipeline from raw data to spall analysis

3. ALPSS Parameters Tab:
   - Configure all ALPSS processing parameters
   - Use recommended values for most parameters
   - Adjust based on your experimental setup
   - Gaussian Notch Filter: Enable to remove carrier frequency (recommended for strong signals)
     Disable if signal is weak or carrier/signal frequencies are close together
   - PB/RC Neighbors: Must be ≥ 1 (scipy requirement for peak detection)

4. SPADE Parameters Tab:
   - Set material properties (density, acoustic velocity)
   - Choose analysis model and parameters
   - Configure output options
   - Smooth Window: Only used when SPADE performs its own smoothing (not in combined mode)

5. Control Tab:
   - Run analysis with current settings
   - Monitor progress
   - View output directory

Gaussian Notch Filter Guidelines:
- ENABLE when: Strong carrier signal masks Doppler-shifted signal, clear frequency separation
- DISABLE when: Weak signal relative to noise, carrier and signal frequencies are close
- Effects: Removes carrier frequency but may introduce ringing or phase distortion
- Default: Enabled (True) for backward compatibility

Parameter Constraints:
- PB Neighbors and RC Neighbors: Must be ≥ 1 (required by scipy's peak detection functions)
- Smooth Window: Only applies when SPADE performs smoothing (automatically skipped in combined mode)

Output Files:
- ALPSS outputs: CSV files with velocity data, PNG plots, results with uncertainties
- SPADE outputs: Analysis plots, summary CSV in SPADE_analysis subfolder
- Enhanced outputs: Complete results combining ALPSS and SPADE with all uncertainties
- Key outputs include:
  * spall_summary.csv: Basic SPADE results
  * enhanced_spall_summary.csv: Complete results with ALPSS data and uncertainties
  * spall_vs_strain_rate.png: Spall strength vs strain rate plot
  * spall_vs_shock_stress.png: Spall strength vs shock stress plot
  * all_smoothed_velocity_traces.png: Combined velocity traces
  * Individual ALPSS files: *--results.csv (with uncertainties), *--velocity.csv, etc.
        """)
        usage_layout.addWidget(usage_text)
        doc_layout.addWidget(usage_group)
        
        doc_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "Documentation")
        
    def create_control_tab(self):
        """Create control and progress tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.stop_btn)
        
        self.open_output_btn = QPushButton("Open Output Directory")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.open_output_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.open_output_btn)
        
        layout.addLayout(control_layout)
        
        # Progress bars
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # ALPSS Progress
        alpss_progress_layout = QHBoxLayout()
        alpss_progress_layout.addWidget(QLabel("ALPSS:"))
        self.alpss_progress_bar = QProgressBar()
        self.alpss_progress_bar.setVisible(False)
        alpss_progress_layout.addWidget(self.alpss_progress_bar)
        progress_layout.addLayout(alpss_progress_layout)
        
        # SPADE Progress
        spade_progress_layout = QHBoxLayout()
        spade_progress_layout.addWidget(QLabel("SPADE:"))
        self.spade_progress_bar = QProgressBar()
        self.spade_progress_bar.setVisible(False)
        spade_progress_layout.addWidget(self.spade_progress_bar)
        progress_layout.addLayout(spade_progress_layout)
        
        layout.addWidget(progress_group)
        
        # Progress text (QPlainTextEdit scales better for large logs)
        self.progress_text = QPlainTextEdit()
        self.progress_text.setMaximumHeight(300)
        self.progress_text.setPlaceholderText("Analysis progress will appear here...")
        self.progress_text.setReadOnly(True)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_text)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Control & Progress")
        
    def on_file_mode_changed(self):
        """Handle file mode radio button changes"""
        if self.single_file_radio.isChecked():
            self.single_file_path.setEnabled(True)
            self.single_file_btn.setEnabled(True)
            self.multi_file_path.setEnabled(False)
            self.multi_file_btn.setEnabled(False)
            self.file_pattern.setEnabled(False)
        else:
            self.single_file_path.setEnabled(False)
            self.single_file_btn.setEnabled(False)
            self.multi_file_path.setEnabled(True)
            self.multi_file_btn.setEnabled(True)
            self.file_pattern.setEnabled(True)
            
        self.update_file_list()
        
    def on_analysis_mode_changed(self):
        """Handle analysis mode radio button changes"""
        # Update UI based on selected mode
        self.update_ui_for_analysis_mode()
        
    def update_ui_for_analysis_mode(self):
        """Update UI elements based on selected analysis mode"""
        if self.mode_alpss_only.isChecked():
            # ALPSS only: Enable ALPSS tab, disable SPADE tab
            self.tab_widget.setTabEnabled(2, True)  # ALPSS params tab
            self.tab_widget.setTabEnabled(3, False)  # SPADE params tab
            # Force SPADE input mode to manual for SPADE-only mode
            self.spade_auto_radio.setChecked(False)
            self.spade_manual_radio.setChecked(True)
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
            
        elif self.mode_spade_only.isChecked():
            # SPADE only: Disable ALPSS tab, enable SPADE tab
            self.tab_widget.setTabEnabled(2, False)  # ALPSS params tab
            self.tab_widget.setTabEnabled(3, True)   # SPADE params tab
            # Force SPADE input mode to manual
            self.spade_auto_radio.setChecked(False)
            self.spade_manual_radio.setChecked(True)
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
            
        else:  # Both modes
            # Both: Enable both tabs
            self.tab_widget.setTabEnabled(2, True)   # ALPSS params tab
            self.tab_widget.setTabEnabled(3, True)   # SPADE params tab
            # Allow SPADE input mode selection
            self.spade_auto_radio.setEnabled(True)
            self.spade_manual_radio.setEnabled(True)
            
    def on_spade_input_mode_changed(self):
        """Handle SPADE input mode radio button changes"""
        if self.spade_auto_radio.isChecked():
            self.spade_input_path.setEnabled(False)
            self.spade_input_btn.setEnabled(False)
            self.spade_file_pattern.setEnabled(False)
        else:
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
    
    def on_hel_detection_toggled(self, checked):
        """Show/Hide HEL parameters group based on checkbox."""
        try:
            self.hel_group.setVisible(bool(checked))
        except Exception:
            pass
    
    # ========== Config File Management Functions ==========
    
    def on_alpss_config_mode_changed(self, checked):
        """Enable/disable config file controls when ALPSS config mode changes"""
        if checked:  # Config file mode selected
            self.alpss_config_path.setEnabled(True)
            self.alpss_config_browse_btn.setEnabled(True)
            self.alpss_config_load_btn.setEnabled(True)
        else:  # Manual mode selected
            self.alpss_config_path.setEnabled(False)
            self.alpss_config_browse_btn.setEnabled(False)
            self.alpss_config_load_btn.setEnabled(False)
    
    def on_spade_config_mode_changed(self, checked):
        """Enable/disable config file controls when SPADE config mode changes"""
        if checked:  # Config file mode selected
            self.spade_config_path.setEnabled(True)
            self.spade_config_browse_btn.setEnabled(True)
            self.spade_config_load_btn.setEnabled(True)
        else:  # Manual mode selected
            self.spade_config_path.setEnabled(False)
            self.spade_config_browse_btn.setEnabled(False)
            self.spade_config_load_btn.setEnabled(False)
    
    def browse_alpss_config(self):
        """Browse for ALPSS config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ALPSS Config File", "",
            "Config Files (*.yml *.yaml *.json);;YAML Files (*.yml *.yaml);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.alpss_config_path.setText(file_path)
    
    def browse_spade_config(self):
        """Browse for SPADE config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SPADE Config File", "",
            "Config Files (*.yml *.yaml *.json);;YAML Files (*.yml *.yaml);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.spade_config_path.setText(file_path)
    
    def load_alpss_config(self):
        """Load ALPSS parameters from config file"""
        config_path = self.alpss_config_path.text()
        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, "Error", "Please select a valid config file")
            return
        
        success, config_dict, message = load_config_from_file(config_path)
        if not success:
            QMessageBox.critical(self, "Error", message)
            return
        
        # Apply loaded config to GUI controls
        try:
            self.apply_alpss_config(config_dict)
            QMessageBox.information(self, "Success", f"{message}\n\nALPSS parameters loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying config: {str(e)}")
    
    def load_spade_config(self):
        """Load SPADE parameters from config file"""
        config_path = self.spade_config_path.text()
        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, "Error", "Please select a valid config file")
            return
        
        success, config_dict, message = load_config_from_file(config_path)
        if not success:
            QMessageBox.critical(self, "Error", message)
            return
        
        # Apply loaded config to GUI controls
        try:
            self.apply_spade_config(config_dict)
            QMessageBox.information(self, "Success", f"{message}\n\nSPADE parameters loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying config: {str(e)}")
    
    def save_alpss_config(self):
        """Save current ALPSS parameters to config file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ALPSS Config File", "alpss_config.yml",
            "YAML Files (*.yml *.yaml);;JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        # Get current parameters
        alpss_params = self.get_alpss_params()
        
        success, message = save_config_to_file(alpss_params, file_path)
        if success:
            QMessageBox.information(self, "Success", message)
            self.alpss_config_path.setText(file_path)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def save_spade_config(self):
        """Save current SPADE parameters to config file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save SPADE Config File", "spade_config.yml",
            "YAML Files (*.yml *.yaml);;JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        # Get current parameters
        spade_params = self.get_spade_params()
        
        success, message = save_config_to_file(spade_params, file_path)
        if success:
            QMessageBox.information(self, "Success", message)
            self.spade_config_path.setText(file_path)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def apply_alpss_config(self, config_dict):
        """Apply loaded ALPSS config to GUI controls"""
        # Basic parameters
        if 'save_data' in config_dict:
            self.save_data.setCurrentText(config_dict['save_data'])
        if 'display_plots' in config_dict:
            self.display_plots.setCurrentText(config_dict['display_plots'])
        if 'save_all_plots' in config_dict:
            self.save_all_plots.setCurrentText(config_dict['save_all_plots'])
        if 'spall_calculation' in config_dict:
            self.spall_calculation.setCurrentText(config_dict['spall_calculation'])
        if 'header_lines' in config_dict:
            self.header_lines.setValue(config_dict['header_lines'])
        if 'start_time_user' in config_dict:
            self.start_time_user.setText(config_dict['start_time_user'])
        if 'start_time_correction' in config_dict:
            self.start_time_correction.setValue(config_dict['start_time_correction'])
        
        # Time parameters
        if 'time_to_skip' in config_dict:
            self.time_to_skip.setValue(config_dict['time_to_skip'])
        if 'time_to_take' in config_dict:
            self.time_to_take.setValue(config_dict['time_to_take'])
        if 't_before' in config_dict:
            self.t_before.setValue(config_dict['t_before'])
        if 't_after' in config_dict:
            self.t_after.setValue(config_dict['t_after'])
        if 'iq_threshold_factor' in config_dict:
            self.iq_threshold_factor.setValue(config_dict['iq_threshold_factor'])
        
        # Frequency parameters
        if 'freq_min' in config_dict:
            self.freq_min.setValue(config_dict['freq_min'])
        if 'freq_max' in config_dict:
            self.freq_max.setValue(config_dict['freq_max'])
        
        # Smoothing parameters
        if 'smoothing_window' in config_dict:
            self.smoothing_window.setValue(config_dict['smoothing_window'])
        if 'smoothing_wid' in config_dict:
            self.smoothing_wid.setValue(config_dict['smoothing_wid'])
        if 'smoothing_amp' in config_dict:
            self.smoothing_amp.setValue(config_dict['smoothing_amp'])
        if 'smoothing_sigma' in config_dict:
            self.smoothing_sigma.setValue(config_dict['smoothing_sigma'])
        if 'smoothing_mu' in config_dict:
            self.smoothing_mu.setValue(config_dict['smoothing_mu'])
        
        # Peak detection parameters
        if 'pb_neighbors' in config_dict:
            self.pb_neighbors.setValue(config_dict['pb_neighbors'])
        if 'pb_idx_correction' in config_dict:
            self.pb_idx_correction.setValue(config_dict['pb_idx_correction'])
        if 'rc_neighbors' in config_dict:
            self.rc_neighbors.setValue(config_dict['rc_neighbors'])
        if 'rc_idx_correction' in config_dict:
            self.rc_idx_correction.setValue(config_dict['rc_idx_correction'])
        
        # STFT parameters
        if 'sample_rate' in config_dict:
            self.sample_rate.setValue(config_dict['sample_rate'])
        if 'nperseg' in config_dict:
            self.nperseg.setValue(config_dict['nperseg'])
        if 'noverlap' in config_dict:
            self.noverlap.setValue(config_dict['noverlap'])
        if 'nfft' in config_dict:
            self.nfft.setValue(config_dict['nfft'])
        if 'window' in config_dict:
            self.window.setCurrentText(config_dict['window'])
        if 'carrier_band_time' in config_dict:
            self.carrier_band_time.setValue(config_dict['carrier_band_time'])
        
        # Blur parameters
        if 'blur_kernel_x' in config_dict:
            self.blur_kernel_x.setValue(config_dict['blur_kernel_x'])
        if 'blur_kernel_y' in config_dict:
            self.blur_kernel_y.setValue(config_dict['blur_kernel_y'])
        if 'blur_sigx' in config_dict:
            self.blur_sigx.setValue(config_dict['blur_sigx'])
        if 'blur_sigy' in config_dict:
            self.blur_sigy.setValue(config_dict['blur_sigy'])
        
        # Signal processing
        if 'use_notch_filter' in config_dict:
            self.use_notch_filter.setChecked(config_dict['use_notch_filter'])
        if 'order' in config_dict:
            self.order.setValue(config_dict['order'])
        if 'wid' in config_dict:
            self.wid.setValue(config_dict['wid'])
        if 'uncert_mult' in config_dict:
            self.uncert_mult.setValue(config_dict['uncert_mult'])
        if 'cmap' in config_dict:
            self.cmap.setCurrentText(config_dict['cmap'])
        
        # PDV wavelength
        if 'lam' in config_dict:
            self.lam.setValue(config_dict['lam'])
        if 'theta' in config_dict:
            self.theta.setValue(config_dict['theta'])
        
        # Uncertainty deltas
        if 'delta_rho' in config_dict:
            self.delta_rho.setValue(config_dict['delta_rho'])
        if 'delta_C0' in config_dict:
            self.delta_C0.setValue(config_dict['delta_C0'])
        if 'delta_lam' in config_dict:
            self.delta_lam.setValue(config_dict['delta_lam'])
        if 'delta_theta' in config_dict:
            self.delta_theta.setValue(config_dict['delta_theta'])
        
        # Material properties
        if 'C0' in config_dict:
            self.C0.setValue(config_dict['C0'])
        if 'density' in config_dict:
            self.density.setValue(config_dict['density'])
        
        # Output file selection checkboxes
        if 'save_velocity_csv' in config_dict:
            self.save_velocity_csv.setChecked(config_dict['save_velocity_csv'])
        if 'save_velocity_smooth_csv' in config_dict:
            self.save_velocity_smooth_csv.setChecked(config_dict['save_velocity_smooth_csv'])
        if 'save_velocity_uncert_csv' in config_dict:
            self.save_velocity_uncert_csv.setChecked(config_dict['save_velocity_uncert_csv'])
        if 'save_velocity_smooth_uncert_csv' in config_dict:
            self.save_velocity_smooth_uncert_csv.setChecked(config_dict['save_velocity_smooth_uncert_csv'])
        if 'save_results_csv' in config_dict:
            self.save_results_csv.setChecked(config_dict['save_results_csv'])
        if 'save_noise_csv' in config_dict:
            self.save_noise_csv.setChecked(config_dict['save_noise_csv'])
    
    def apply_spade_config(self, config_dict):
        """Apply loaded SPADE config to GUI controls"""
        # Experiment type
        if 'experiment_velocity_shots' in config_dict:
            self.experiment_velocity_shots.setChecked(config_dict['experiment_velocity_shots'])
        if 'experiment_spall_analysis' in config_dict:
            self.experiment_spall_analysis.setChecked(config_dict['experiment_spall_analysis'])
        if 'experiment_hel_detection' in config_dict:
            self.experiment_hel_detection.setChecked(config_dict['experiment_hel_detection'])
        
        # Material properties
        if 'density' in config_dict:
            self.spade_density.setValue(config_dict['density'])
        if 'acoustic_velocity' in config_dict:
            self.spade_acoustic_velocity.setValue(config_dict['acoustic_velocity'])
        
        # Velocity shots parameters
        if 'impact_velocity_window_start' in config_dict:
            self.impact_velocity_window_start.setValue(config_dict['impact_velocity_window_start'])
        if 'impact_velocity_window_end' in config_dict:
            self.impact_velocity_window_end.setValue(config_dict['impact_velocity_window_end'])
        if 'align_velocity_threshold_ms' in config_dict:
            self.align_velocity_threshold.setValue(config_dict['align_velocity_threshold_ms'])
        
        # Spall analysis parameters
        if 'smoothing_method' in config_dict:
            self.smoothing_method.setCurrentText(config_dict['smoothing_method'])
        if 'smoothing_window_length' in config_dict:
            self.smoothing_window_length.setValue(config_dict['smoothing_window_length'])
        if 'derivative_smoothing_window_length' in config_dict:
            self.derivative_smoothing_window_length.setValue(config_dict['derivative_smoothing_window_length'])
        if 'pullback_threshold_fraction' in config_dict:
            self.pullback_threshold_fraction.setValue(config_dict['pullback_threshold_fraction'])
        if 'uncertainty_threshold_ms' in config_dict:
            self.uncertainty_threshold.setValue(config_dict['uncertainty_threshold_ms'])
        if 'include_uncert_bands' in config_dict:
            self.include_uncert_bands.setChecked(config_dict['include_uncert_bands'])
        
        # Combined plot parameters
        if 'auto_calculate_limits' in config_dict:
            self.auto_calculate_limits.setChecked(config_dict['auto_calculate_limits'])
        if 'x_min_main' in config_dict:
            self.x_min_main.setValue(config_dict['x_min_main'])
        if 'x_max_main' in config_dict:
            self.x_max_main.setValue(config_dict['x_max_main'])
        if 'y_min_main' in config_dict:
            self.y_min_main.setValue(config_dict['y_min_main'])
        if 'y_max_main' in config_dict:
            self.y_max_main.setValue(config_dict['y_max_main'])
        if 'x_min_zoom' in config_dict:
            self.x_min_zoom.setValue(config_dict['x_min_zoom'])
        if 'x_max_zoom' in config_dict:
            self.x_max_zoom.setValue(config_dict['x_max_zoom'])
        if 'y_min_zoom' in config_dict:
            self.y_min_zoom.setValue(config_dict['y_min_zoom'])
        if 'y_max_zoom' in config_dict:
            self.y_max_zoom.setValue(config_dict['y_max_zoom'])
        
        # HEL parameters
        if 'hel_time_window_start_ns' in config_dict:
            self.hel_time_window_start_ns.setValue(config_dict['hel_time_window_start_ns'])
        if 'hel_time_window_end_ns' in config_dict:
            self.hel_time_window_end_ns.setValue(config_dict['hel_time_window_end_ns'])
        if 'hel_derivative_threshold' in config_dict:
            self.hel_derivative_threshold.setValue(config_dict['hel_derivative_threshold'])
        if 'hel_smoothing_window' in config_dict:
            self.hel_smoothing_window.setValue(config_dict['hel_smoothing_window'])
        if 'minimum_HEL_velocity_expected' in config_dict:
            self.minimum_hel_velocity.setValue(config_dict['minimum_HEL_velocity_expected'])
        if 'hel_detection_min_points' in config_dict:
            self.hel_detection_min_points.setValue(int(config_dict['hel_detection_min_points']))

        # Spall window parameters
        if 'spall_start_time_ns' in config_dict:
            self.spall_start_time_ns.setValue(config_dict['spall_start_time_ns'])
        if 'spall_end_time_ns' in config_dict:
            self.spall_end_time_ns.setValue(config_dict['spall_end_time_ns'])
        if 'threshold_velocity_ms' in config_dict:
            self.threshold_velocity_ms.setValue(config_dict['threshold_velocity_ms'])
    
    # ========== End Config File Management ==========
            
    def toggle_signal_length_spin(self):
        """Toggle signal length spin box based on combo selection"""
        if self.signal_length_combo.currentText() == "Custom...":
            self.signal_length_spin.setEnabled(True)
        else:
            self.signal_length_spin.setEnabled(False)
    
    # Experiment type change handler removed - now allows both options to be selected
            
    def select_single_file(self):
        """Select one or more input files (manual selection)"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input File(s)", "", 
            "Data Files (*.csv *.txt);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if file_paths:
            self.single_file_path.setText(";".join(file_paths))
            self.update_file_list()
            
    def select_multi_file_dir(self):
        """Select directory for multiple files"""
        try:
            print("Opening directory dialog...")
            
            # Try with different options for macOS compatibility
            try:
                dir_path = QFileDialog.getExistingDirectory(
                    self, 
                    "Select Input Directory",
                    os.getcwd(),  # Start from current working directory
                    QFileDialog.ShowDirsOnly
                )
            except Exception as dialog_error:
                print(f"First dialog attempt failed: {dialog_error}")
                # Fallback to simpler dialog
                dir_path = QFileDialog.getExistingDirectory(
                    self, 
                    "Select Input Directory"
                )
            
            print(f"Directory dialog result: {dir_path}")
            if dir_path:
                self.multi_file_path.setText(dir_path)
                print(f"Set directory path: {dir_path}")
                self.update_file_list()
            else:
                print("No directory selected")
        except Exception as e:
            print(f"Error in select_multi_file_dir: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to select directory: {e}")
            
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)
            
    def select_param_folder(self):
        """Select parameter folder containing Excel files"""
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Parameter Folder", 
            ""
        )
        if folder_path:
            self.param_folder_path.setText(folder_path)
            self.load_param_folder_info()
            
    def load_param_folder_info(self):
        """Load and display information from all Excel/CSV files in the parameter folder"""
        folder_path = self.param_folder_path.text()
        if not folder_path or not os.path.exists(folder_path):
            self.param_file_info.setText("No parameter folder selected")
            return
            
        # Find all Excel/CSV files in the folder
        excel_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                excel_files.append(os.path.join(folder_path, file))
                
        if not excel_files:
            self.param_file_info.setText(f"No parameter files (.xlsx/.xls/.csv) found in {os.path.basename(folder_path)}")
            return
            
        total_experiments = 0
        total_pdv_files = 0
        all_materials = set()
        all_pdv_files = []
        
        info_text = f"Parameter Folder: {os.path.basename(folder_path)}\n"
        # Summarize by type
        num_xlsx = sum(1 for f in excel_files if f.lower().endswith(('.xlsx', '.xls')))
        num_csv = sum(1 for f in excel_files if f.lower().endswith('.csv'))
        info_text += f"Parameter Files Found: {len(excel_files)} (Excel: {num_xlsx}, CSV: {num_csv})\n"
        
        for file_path in excel_files:
            try:
                # Try to read the file depending on extension
                if file_path.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        info_text += f"\n{os.path.basename(file_path)}: Error reading CSV - {str(e)}"
                        continue
                else:
                    try:
                        import openpyxl
                    except ImportError:
                        self.param_file_info.setText("Error: openpyxl not installed. Please install with: pip install openpyxl")
                        return
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as e:
                        info_text += f"\n{os.path.basename(file_path)}: Error reading Excel - {str(e)}"
                        continue
                    
                # Display basic information for this file
                file_experiments = len(df)
                total_experiments += file_experiments
                info_text += f"\n{os.path.basename(file_path)}: {file_experiments} experiments"
                
                # Check for required columns
                if 'PDV_FileName' in df.columns:
                    pdv_files = df['PDV_FileName'].dropna().astype(str).tolist()
                    total_pdv_files += len(pdv_files)
                    all_pdv_files.extend(pdv_files)
                    info_text += f", {len(pdv_files)} PDV files"
                else:
                    info_text += ", no PDV_FileName column"
                    
                # Check for sample material column
                
                # Look for sample material column with various possible names
                sample_material_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    # Check for various possible column names including spaces, underscores, and different formats
                    if any(name in col_lower for name in [
                        'sample_material', 'samplematerial', 'sample material', 'sample-material',
                        'material', 'sample', 'sample_mat', 'sample mat', 'sample-mat'
                    ]):
                        sample_material_col = col
                        break
                
                if sample_material_col:
                    materials = df[sample_material_col].dropna().unique()
                    all_materials.update(materials)
                else:
                    pass
                    
            except Exception as e:
                info_text += f"\n{os.path.basename(file_path)}: Error - {str(e)}"
                
        # Add summary information
        info_text += f"\n\nSummary: {total_experiments} total experiments, {total_pdv_files} PDV files"
        if all_materials:
            info_text += f"\nSample Materials: {', '.join(sorted(all_materials))}"
        if all_pdv_files:
            # Ensure all PDV files are strings and handle any remaining non-string values
            pdv_files_display = []
            for pdv_file in all_pdv_files[:5]:
                try:
                    pdv_files_display.append(str(pdv_file))
                except:
                    pdv_files_display.append("Unknown")
            info_text += f"\nSample PDV Files: {', '.join(pdv_files_display)}"
            if len(all_pdv_files) > 5:
                info_text += f" ... and {len(all_pdv_files) - 5} more"
                
        self.param_file_info.setText(info_text)
            
    def select_spade_input(self):
        """Select SPADE input files or directory"""
        # Provide clear options: Individual Files or Directory
        dialog = QMessageBox(self)
        dialog.setWindowTitle("SPADE Input Selection")
        dialog.setText("Choose how to select SPADE input:")
        btn_files = dialog.addButton("Individual Files", QMessageBox.AcceptRole)
        btn_dir = dialog.addButton("Directory", QMessageBox.AcceptRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked == btn_files:
            # Select one or multiple individual files
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Velocity File(s)", "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file_paths:
                self.spade_input_path.setText(";".join(file_paths))
        elif clicked == btn_dir:
            # Select a directory
            dir_path = QFileDialog.getExistingDirectory(self, "Select Velocity Files Directory")
            if dir_path:
                self.spade_input_path.setText(dir_path)
            
    def update_file_list(self):
        """Update the file list display"""
        try:
            print("Updating file list...")
            self.file_list.clear()
            
            if self.single_file_radio.isChecked():
                file_text = self.single_file_path.text()
                if file_text:
                    paths = [p for p in file_text.split(";") if p]
                    valid_paths = [p for p in paths if os.path.exists(p)]
                    if valid_paths:
                        self.file_list.appendPlainText(f"Manual Select: {len(valid_paths)} file(s)")
                        for p in valid_paths:
                            self.file_list.appendPlainText(f"  • {os.path.basename(p)}")
            else:
                dir_path = self.multi_file_path.text()
                pattern = self.file_pattern.text()
                print(f"Multi-file mode - Directory: {dir_path}, Pattern: {pattern}")
                
                if dir_path and os.path.exists(dir_path):
                    try:
                        print(f"Scanning directory: {dir_path}")
                        files = glob.glob(os.path.join(dir_path, pattern))
                        print(f"Found {len(files)} files matching pattern")
                        if files:
                            self.file_list.appendPlainText(f"Found {len(files)} files in {dir_path}:")
                            for file_path in sorted(files):
                                self.file_list.appendPlainText(f"  • {os.path.basename(file_path)}")
                        else:
                            self.file_list.appendPlainText(f"No files found matching pattern '{pattern}' in {dir_path}")
                    except Exception as e:
                        print(f"Error scanning directory: {e}")
                        self.file_list.appendPlainText(f"Error scanning directory: {e}")
                else:
                    print(f"Invalid directory: {dir_path}")
                    self.file_list.appendPlainText("No valid directory selected")
            print("File list update complete")
        except Exception as e:
            print(f"Error in update_file_list: {e}")
            import traceback
            traceback.print_exc()
            self.file_list.appendPlainText(f"Error updating file list: {e}")
                    
    def get_input_files(self):
        """Get list of input files based on current selection"""
        if self.single_file_radio.isChecked():
            file_text = self.single_file_path.text()
            if file_text:
                paths = [p for p in file_text.split(";") if p]
                return [p for p in paths if os.path.exists(p)]
        else:
            dir_path = self.multi_file_path.text()
            pattern = self.file_pattern.text()
            
            if dir_path and os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, pattern))
                return sorted(files)
        
        return []
        
    def get_param_file_data(self):
        """Get parameter file data from all Excel files in the selected parameter folder"""
        folder_path = self.param_folder_path.text()
        if not folder_path or not os.path.exists(folder_path):
            return None
            
        # Find all Excel files in the folder
        excel_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                excel_files.append(os.path.join(folder_path, file))
                
        if not excel_files:
            return None
            
        # Combine data from all parameter files
        combined_param_data = {}
        
        for param_file_path in excel_files:
            if not os.path.exists(param_file_path):
                continue
                
            try:
                # Try to read the file (Excel or CSV)
                try:
                    if param_file_path.lower().endswith('.csv'):
                        df = pd.read_csv(param_file_path)
                    else:
                        try:
                            import openpyxl
                        except ImportError:
                            print(f"Error: openpyxl not installed for {param_file_path}")
                            continue
                        df = pd.read_excel(param_file_path)
                except Exception as e:
                    print(f"Error reading file {param_file_path}: {str(e)}")
                    continue
                    
                # Create a mapping from PDV_FileName to experiment data
                param_data = {}
                
                # Handle different possible column names for PDV file name
                pdv_col = None
                # First pass: exact-ish known variants, ignoring spaces/underscores/dashes
                import re
                normalized_columns = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in df.columns}
                known_variants = [
                    'pdvfilename', 'pdvfile', 'pdv_file', 'pdv_file_name', 'pdv file name',
                    'dvfilename', 'dv_file', 'dvfile', 'filename', 'file_name', 'file name'
                ]
                normalized_variants = {re.sub(r"[^a-z0-9]", "", v): v for v in known_variants}
                for col, norm in normalized_columns.items():
                    if norm in normalized_variants:
                        pdv_col = col
                        break
                # Second pass: heuristic containing tokens 'pdv' and ('file' or 'name')
                if pdv_col is None:
                    for col in df.columns:
                        col_lower = col.lower()
                        if ('pdv' in col_lower or 'dv' in col_lower) and ('file' in col_lower or 'name' in col_lower):
                            pdv_col = col
                            break
                # Final fallback: a standalone 'filename' or 'file name' column
                if pdv_col is None:
                    for col in df.columns:
                        if col.strip().lower() in ['filename', 'file name', 'file_name']:
                            pdv_col = col
                            break
                        
                if pdv_col is None:
                    continue
                    
                # Create mapping for each experiment
                for idx, row in df.iterrows():
                    pdv_file = row[pdv_col]
                    if pd.isna(pdv_file) or pdv_file == 0:
                        continue
                        
                    # Convert PDV file name to string to ensure consistency
                    pdv_file_str = str(pdv_file).strip()
                    
                    # Clean the filename for better matching
                    # Remove common extensions and clean up the name
                    clean_pdv_file = pdv_file_str
                    for ext in ['.csv', '.txt', '.dat', '.xlsx', '.xls']:
                        if clean_pdv_file.lower().endswith(ext):
                            clean_pdv_file = clean_pdv_file[:-len(ext)]
                    
                    # Extract ALL columns from the row (except the PDV filename column itself)
                    exp_info = {}
                    for col in df.columns:
                        if col != pdv_col:  # Skip the PDV filename column
                            value = row.get(col)
                            if not pd.isna(value):  # Only include non-NaN values
                                exp_info[col] = value
                    
                    # Store both original and cleaned versions for better matching
                    combined_param_data[pdv_file_str] = exp_info
                    if clean_pdv_file != pdv_file_str:
                        combined_param_data[clean_pdv_file] = exp_info
                    
                    # Also store with common variations for better matching
                    # Remove date patterns if present
                    date_cleaned = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', '', clean_pdv_file)
                    if date_cleaned != clean_pdv_file:
                        combined_param_data[date_cleaned] = exp_info
                
            except Exception as e:
                print(f"Error loading parameter file {param_file_path}: {e}")
                continue
                
        return combined_param_data if combined_param_data else None
        
    def get_alpss_params(self):
        """Get ALPSS parameters from GUI"""
        # save_all_plots is the master switch: "no" suppresses every per-file plot;
        # "subfolder"/"main_dir" enable plotting and pick the destination folder.
        dropdown_value = self.save_all_plots.currentText()
        save_plots_value = 'yes' if dropdown_value in ['subfolder', 'main_dir'] else 'no'
        plots_enabled = (save_plots_value == 'yes')

        return {
            'filename': 'example_file.csv',  # Will be updated per file in thread
            'save_data': self.save_data.currentText(),
            'save_all_plots': dropdown_value,  # Save actual dropdown value for config files
            'save_all_plots_enabled': save_plots_value,  # Converted yes/no for ALPSS
            'save_plots_in_subfolder': dropdown_value == 'subfolder',
            'start_time_user': self.start_time_user.text(),
            'header_lines': self.header_lines.value(),
            'time_to_skip': self.time_to_skip.value(),
            'time_to_take': self.time_to_take.value(),
            't_before': self.t_before.value(),
            't_after': self.t_after.value(),
            'start_time_correction': self.start_time_correction.value(),
            'iq_threshold_factor': self.iq_threshold_factor.value(),
            'freq_min': self.freq_min.value(),
            'freq_max': self.freq_max.value(),
            'smoothing_window': self.smoothing_window.value(),
            'smoothing_wid': self.smoothing_wid.value(),
            'smoothing_amp': self.smoothing_amp.value(),
            'smoothing_sigma': self.smoothing_sigma.value(),
            'smoothing_mu': self.smoothing_mu.value(),
            'pb_neighbors': self.pb_neighbors.value(),
            'pb_idx_correction': self.pb_idx_correction.value(),
            'rc_neighbors': self.rc_neighbors.value(),
            'rc_idx_correction': self.rc_idx_correction.value(),
            'sample_rate': self.sample_rate.value(),
            'nperseg': self.nperseg.value(),
            'noverlap': self.noverlap.value(),
            'nfft': self.nfft.value(),
            'window': self.window.currentText(),
            'blur_kernel': (self.blur_kernel_x.value(), self.blur_kernel_y.value()),
            'blur_sigx': self.blur_sigx.value(),
            'blur_sigy': self.blur_sigy.value(),
            'carrier_band_time': self.carrier_band_time.value(),
            'cmap': self.cmap.currentText(),
            'uncert_mult': self.uncert_mult.value(),
            'use_notch_filter': self.use_notch_filter.isChecked(),
            'order': self.order.value(),
            'wid': self.wid.value(),
            'lam': self.lam.value(),
            'C0': self.C0.value(),
            'density': self.density.value(),
            'delta_rho': self.delta_rho.value(),
            'delta_C0': self.delta_C0.value(),
            'delta_lam': self.delta_lam.value(),
            'theta': self.theta.value(),
            'delta_theta': self.delta_theta.value(),
            'exp_data_dir': '',  # Will be updated per file in thread
            'out_files_dir': '',  # Will be updated per file in thread
            'display_plots': self.display_plots.currentText(),
            'spall_calculation': self.spall_calculation.currentText(),
            'plot_figsize': (self.plot_width.value(), self.plot_height.value()),
            'plot_dpi': self.plot_dpi.value(),
            # Image selection parameters (globally gated by Save ALPSS Plots dropdown)
            'save_combined_plot': self.save_combined_plot.isChecked() and plots_enabled,
            'save_iq_start_time_plot': self.save_iq_start_time_plot.isChecked() and plots_enabled,
            # Output file selection parameters
            'save_velocity_csv': self.save_velocity_csv.isChecked(),
            'save_velocity_smooth_csv': self.save_velocity_smooth_csv.isChecked(),
            'save_velocity_uncert_csv': self.save_velocity_uncert_csv.isChecked(),
            'save_velocity_smooth_uncert_csv': self.save_velocity_smooth_uncert_csv.isChecked(),
            'save_results_csv': self.save_results_csv.isChecked(),
            'save_noise_csv': self.save_noise_csv.isChecked(),
            'smart_selection_enabled': self.smart_selection_checkbox.isChecked(),
        }
        
    def get_spade_params(self):
        """Get SPADE parameters from GUI"""
        # Get signal length
        if self.signal_length_combo.currentText() == "Full Signal (None)":
            signal_length_ns = None
        else:
            signal_length_ns = self.signal_length_spin.value()
        
        # Get experiment types (can be both)
        velocity_shots_enabled = self.experiment_velocity_shots.isChecked()
        spall_analysis_enabled = self.experiment_spall_analysis.isChecked()
        
        # Default to velocity_shots if neither is selected
        if not velocity_shots_enabled and not spall_analysis_enabled:
            velocity_shots_enabled = True
            
        return {
            'density': self.spade_density.value(),
            'acoustic_velocity': self.spade_acoustic_velocity.value(),
            # Note: analysis_model is no longer user-selectable - always uses hybrid approach
            # (max_min for spall strength, hybrid_5_segment for strain rate)
            'analysis_model': 'hybrid',  # Placeholder for backward compatibility with config files
            'signal_length_ns': signal_length_ns,
            'prominence_factor': self.prominence_factor.value(),
            'peak_distance_ns': self.peak_distance_ns.value(),
            'plot_individual': self.plot_individual.isChecked(),
            'save_summary_table': self.save_summary.isChecked(),
            'show_plots': self.show_plots.isChecked(),
            # Experiment types
            'velocity_shots_enabled': velocity_shots_enabled,
            'spall_analysis_enabled': spall_analysis_enabled,
            'hel_detection_enabled': self.experiment_hel_detection.isChecked(),
            # HEL parameters
            'hel_start_time_ns': self.hel_start_time_ns.value(),
            'hel_end_time_ns': self.hel_end_time_ns.value(),
            'hel_angle_threshold_deg': self.hel_angle_threshold_deg.value(),
            'minimum_HEL_velocity_expected': self.minimum_hel_velocity.value(),
            'hel_detection_min_points': self.hel_detection_min_points.value(),
            # Spall window parameters
            'spall_start_time_ns': self.spall_start_time_ns.value(),
            'spall_end_time_ns': self.spall_end_time_ns.value(),
            'threshold_velocity_ms': self.threshold_velocity_ms.value(),
            # Axis limits for combined plots
            'auto_calculate_limits': self.auto_calculate_limits.isChecked(),
            'x_min_main': self.x_min_main.value(),
            'x_max_main': self.x_max_main.value(),
            'y_min_main': self.y_min_main.value(),
            'y_max_main': self.y_max_main.value(),
            'x_min_zoom': self.x_min_zoom.value(),
            'x_max_zoom': self.x_max_zoom.value(),
            'y_min_zoom': self.y_min_zoom.value(),
            'y_max_zoom': self.y_max_zoom.value(),
            # Combined velocity plot options
            'generate_all_velocity_plot': self.generate_all_velocity_plot.isChecked(),
            'uncertainty_threshold_ms': self.uncertainty_threshold.value(),
            'align_velocity_threshold_ms': self.align_velocity_threshold.value(),
            'include_uncert_bands': self.include_uncert_bands.isChecked(),
            'uncert_alpha': self.uncert_alpha.value(),
            'zoom_window_ns': self.zoom_window_ns.value(),
        }

    def _apply_runtime_spade_overrides(self, spade_params):
        """Ensure critical SPADE settings respect the current GUI selections."""
        overrides = {
            # Note: analysis_model is no longer user-selectable - always uses hybrid approach
            'spall_start_time_ns': self.spall_start_time_ns.value(),
            'spall_end_time_ns': self.spall_end_time_ns.value(),
            'threshold_velocity_ms': self.threshold_velocity_ms.value(),
            'hel_start_time_ns': self.hel_start_time_ns.value(),
            'hel_end_time_ns': self.hel_end_time_ns.value(),
        }
        spade_params.update(overrides)
        if hasattr(self, 'progress_text') and self.progress_text:
            try:
                self.progress_text.appendPlainText(
                    "✓ Applied GUI overrides to SPADE parameters: "
                    + ", ".join(f"{k}={v}" for k, v in overrides.items())
                )
            except Exception:
                pass
        return spade_params
        
    def run_analysis(self):
        """Run the analysis"""
        # Get output directory
        output_dir = self.output_path.text()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return
            
        # Get parameters - from config file or GUI
        # ALPSS Parameters
        if self.alpss_config_mode.isChecked():
            # Use config file
            config_path = self.alpss_config_path.text()
            if not config_path or not os.path.exists(config_path):
                QMessageBox.warning(self, "Error", "Config file mode selected but no valid ALPSS config file specified.\nPlease select a config file or switch to manual mode.")
                return
            success, alpss_params, message = load_config_from_file(config_path)
            if not success:
                QMessageBox.critical(self, "Error", f"Failed to load ALPSS config:\n{message}")
                return
            self.progress_text.appendPlainText(f"✓ Using ALPSS parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            alpss_params = self.get_alpss_params()
            self.progress_text.appendPlainText("✓ Using ALPSS parameters from GUI")
        
        # SPADE Parameters
        if self.spade_config_mode.isChecked():
            # Use config file
            config_path = self.spade_config_path.text()
            if not config_path or not os.path.exists(config_path):
                QMessageBox.warning(self, "Error", "Config file mode selected but no valid SPADE config file specified.\nPlease select a config file or switch to manual mode.")
                return
            success, spade_params, message = load_config_from_file(config_path)
            if not success:
                QMessageBox.critical(self, "Error", f"Failed to load SPADE config:\n{message}")
                return
            self.progress_text.appendPlainText(f"✓ Using SPADE parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            spade_params = self.get_spade_params()
            self.progress_text.appendPlainText("✓ Using SPADE parameters from GUI")
        
        spade_params = self._apply_runtime_spade_overrides(spade_params)
        self.spade_params = spade_params.copy()
        
        # Get parameter file data if available
        param_data = self.get_param_file_data()
        if param_data:
            # Count parameter files (Excel/CSV) in the parameter folder
            folder_path = self.param_folder_path.text()
            total_count = 0
            num_xlsx = 0
            num_csv = 0
            if folder_path and os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                        total_count += 1
                        if file.lower().endswith('.csv'):
                            num_csv += 1
                        else:
                            num_xlsx += 1
            self.progress_text.appendPlainText(f"Loaded {total_count} parameter files (Excel: {num_xlsx}, CSV: {num_csv}) with {len(param_data)} total experiments")
        else:
            self.progress_text.appendPlainText("No parameter files provided - using default file names")
        
        # Determine analysis mode
        if self.mode_alpss_only.isChecked():
            # ALPSS only mode
            input_files = self.get_input_files()
            if not input_files:
                QMessageBox.warning(self, "No Input Files", "Please select input files for ALPSS analysis.")
                return
                
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(True)
            self.alpss_progress_bar.setRange(0, len(input_files))
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(False)
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start ALPSS-only analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, input_files, output_dir, param_data,
                spade_auto_mode=False, spade_input_files=None, analysis_mode="alpss_only",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
            
        elif self.mode_spade_only.isChecked():
            # SPADE only mode
            spade_input_path = self.spade_input_path.text()
            if not spade_input_path:
                QMessageBox.warning(self, "No SPADE Input", "Please select SPADE input files or directory.")
                return
                
            # Get SPADE input files
            spade_input_files = None
            if ";" in spade_input_path:
                # Multiple individual files
                spade_input_files = spade_input_path.split(";")
            elif os.path.isdir(spade_input_path):
                # Directory with pattern
                pattern = self.spade_file_pattern.text()
                spade_input_files = glob.glob(os.path.join(spade_input_path, pattern))
            elif os.path.isfile(spade_input_path):
                # Single file
                spade_input_files = [spade_input_path]
            else:
                QMessageBox.warning(self, "Invalid SPADE Input", "Please select valid SPADE input files or directory.")
                return
                
            if not spade_input_files:
                QMessageBox.warning(self, "No SPADE Files", "No files found matching the specified pattern.")
                return
                
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(False)
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(True)
            self.spade_progress_bar.setRange(0, len(spade_input_files))
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start SPADE-only analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, [], output_dir, param_data,
                spade_auto_mode=False, spade_input_files=spade_input_files, analysis_mode="spade_only",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
            
        else:
            # Combined ALPSS + SPADE mode
            input_files = self.get_input_files()
            if not input_files:
                QMessageBox.warning(self, "No Input Files", "Please select input files for ALPSS analysis.")
                return
                
            # Get SPADE input mode and files
            spade_auto_mode = self.spade_auto_radio.isChecked()
            spade_input_files = None
            
            if not spade_auto_mode:
                spade_input_path = self.spade_input_path.text()
                if ";" in spade_input_path:
                    # Multiple individual files
                    spade_input_files = spade_input_path.split(";")
                elif os.path.isdir(spade_input_path):
                    # Directory with pattern
                    pattern = self.spade_file_pattern.text()
                    spade_input_files = glob.glob(os.path.join(spade_input_path, pattern))
                elif os.path.isfile(spade_input_path):
                    # Single file
                    spade_input_files = [spade_input_path]
                else:
                    QMessageBox.warning(self, "Invalid SPADE Input", "Please select valid SPADE input files or directory.")
                    return
            
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(True)
            self.alpss_progress_bar.setRange(0, len(input_files))
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(False)
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start combined analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, input_files, output_dir, param_data,
                spade_auto_mode=spade_auto_mode, spade_input_files=spade_input_files, analysis_mode="both",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
        
    def stop_analysis(self):
        """Stop the analysis"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            self.progress_text.appendPlainText("Analysis stopped by user.")
            
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.spade_progress_bar.setVisible(False)
        
    def update_progress(self, message):
        """Update progress display without overloading the GUI"""
        # Append to UI with newline
        self.progress_text.appendPlainText(message)
        # Truncate to last N lines to avoid massive widget growth
        max_lines = 2000
        current_text = self.progress_text.toPlainText()
        if current_text.count("\n") > max_lines + 100:
            lines = current_text.splitlines()
            trimmed = "\n".join(lines[-max_lines:])
            self.progress_text.setPlainText(trimmed)
            self.progress_text.moveCursor(self.progress_text.textCursor().End)
        
        # Also write to a rolling log file
        try:
            log_dir = os.path.join(self.output_path.text() or os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"helix_analysis_{datetime.now().strftime('%Y%m%d')}.log")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        except Exception:
            pass
        
        def _safe_int_from_token(token):
            if not token:
                return None
            token_digits = ''.join(ch for ch in token if ch.isdigit())
            return int(token_digits) if token_digits.isdigit() else None
        
        # Update the correct progress bar
        if "ALPSS" in message and "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.alpss_progress_bar.maximum() or 100
                    self.alpss_progress_bar.setMaximum(total)
                self.alpss_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        elif "SPADE" in message and "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.spade_progress_bar.maximum() or 100
                    self.spade_progress_bar.setMaximum(total)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        elif "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.spade_progress_bar.maximum() or 100
                    self.spade_progress_bar.setMaximum(total)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        # Handle completion messages - set progress to 100%
        elif "All analysis completed successfully" in message or "Analysis completed successfully" in message:
            if self.alpss_progress_bar.isVisible():
                max_val = self.alpss_progress_bar.maximum()
                if max_val > 0:
                    self.alpss_progress_bar.setValue(max_val)
            if self.spade_progress_bar.isVisible():
                max_val = self.spade_progress_bar.maximum()
                if max_val > 0:
                    self.spade_progress_bar.setValue(max_val)
            QApplication.processEvents()  # Force immediate update
                
    def analysis_finished(self, success, message):
        """Handle analysis completion"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Set progress bars to 100% when analysis completes
        if self.alpss_progress_bar.isVisible():
            max_val = self.alpss_progress_bar.maximum()
            if max_val > 0:
                self.alpss_progress_bar.setValue(max_val)
        if self.spade_progress_bar.isVisible():
            max_val = self.spade_progress_bar.maximum()
            if max_val > 0:
                self.spade_progress_bar.setValue(max_val)
        
        self.spade_progress_bar.setVisible(False)
        
        if success:
            self.progress_text.appendPlainText("Analysis completed successfully!")
            QMessageBox.information(self, "Success", "Analysis completed successfully!")
        else:
            self.progress_text.appendPlainText(f"Analysis failed: {message}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {message}")
            
    def open_output_directory(self):
        """Open the output directory in file explorer"""
        output_dir = self.output_path.text()
        if output_dir and os.path.exists(output_dir):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", output_dir])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "Output directory does not exist.")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    # Set application font with larger size and better readability (increased by 25%)
    # Use system-appropriate font to avoid Qt warnings
    import platform
    if platform.system() == "Darwin":  # macOS
        font = QFont("Helvetica", 14)  # macOS system font (always available)
    elif platform.system() == "Windows":
        font = QFont("Segoe UI", 14)  # Windows system font
    else:  # Linux and others
        font = QFont("Arial", 14)  # Universal fallback
    font.setWeight(QFont.Normal)
    app.setFont(font)
    
    # Set application style sheet for modern look (increased font sizes by 25%)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QTabWidget::pane {
            border: 1px solid #c0c0c0;
            background-color: white;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-weight: bold;
            font-size: 16px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #0078d4;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 15px;
            border: 2px solid #c0c0c0;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #2c2c2c;
        }
        QLabel {
            font-size: 14px;
            color: #2c2c2c;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 6px;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #0078d4;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #c0c0c0;
            color: #666666;
        }
        QCheckBox {
            font-size: 14px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QTextEdit {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
        }
        QScrollArea {
            border: none;
        }
        QProgressBar {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
    """)
    
    window = HELIXAnalysisToolbox()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
