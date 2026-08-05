# HELIX Toolbox

**A Comprehensive Analysis Platform for Single Point PDV Data — GUI and Config-Driven CLI**

**Author:** Piyush Wanchoo  
**GitHub:** [@Piyushjhu](https://github.com/Piyushjhu)  
**Institution:** Johns Hopkins University  
**Year:** 2026

---

## What's New in v3

Version 3 consolidates outputs, adds per-shot dynamic diagnostics, and ships a
standalone plot generator and a regression guard.

- **Single consolidated master summary.** Each run writes one canonical
  `<IGSN>-Data_Summary.csv` (spall + HEL + parameters + derived quantities) with a
  single standardized column-naming convention (`Peak_Shock_Stress_GPa`, `HEL_GPa`,
  `RiseTime_80_20_ns`, …). The legacy `spall_summary.csv` / `velocity_shots_summary.csv`
  are still written for back-compat.
- **Derived shock-front diagnostics, computed in the engine** and added to every
  summary row: `Peak_Shock_Time_ns`, `RiseTime_ArrivalToPeak_ns`,
  `RiseTime_{80_20,90_10,MaxSlope}_ns`, `PlasticStrainRate_{80_20,90_10,MaxSlope}_s^-1`,
  `Compressive_StrainRate_{Avg,Ufs}_s^-1`, `Shock_Velocity_Us_m_s`, `Shock_Front_Width_um`.
  Denominators use the material Hugoniot slope `S` where available, falling back to the
  bulk wave speed.
- **Standalone post-analysis plot generator** — `helix_post_analysis_plots.py` reads the
  consolidated master (auto-discovered by `*Data_Summary.csv`) and produces the full
  publication plot suite (Grady log–log, rise-time, HEL, spall, spatial 3-D/2-D, tensile)
  as PNG + PDF, with no dependency on the older Binary-metal module:
  `python helix_post_analysis_plots.py --config helix_master_config.yml`
- **Regression control** — `regression/run_control.py` re-analyzes a fixed control shot
  (SPADE-only, deterministic), compares ~25 metrics against a stored baseline, and appends
  a timestamped, commit-stamped entry to `regression/history.jsonl`. Exit 0 = PASS, 1 = FAIL,
  so it can gate a pre-commit hook or CI. See `regression/README.md`.
- **CLI fix:** `--input-pattern` no longer silently overrides the config's `input_pattern`
  (its argparse default was clobbering the config value), so per-file/glob selection from
  the config now works as documented.

---

## Run Online — No Installation Required

Try HELIX Toolbox directly in your browser using any of the platforms below.
All example notebooks open with bundled sample PDV data pre-loaded; bring your own CSV files by uploading them in the first notebook cell.

| Platform | What you get | Launch |
|---|---|---|
| **Binder** | Full Jupyter environment, all deps pre-installed, no account needed | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Piyushjhu/HELIX_Toolbox/main?labpath=examples%2F01_full_pipeline_cli.ipynb) |
| **Google Colab** | Google-hosted GPU/CPU, persistent Drive storage, free tier | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyushjhu/HELIX_Toolbox/blob/main/examples/01_full_pipeline_cli.ipynb) |
| **GitHub Codespaces** | Full VS Code in the browser, drag-and-drop file upload, CLI + GUI | [![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Piyushjhu/HELIX_Toolbox) |

### Quick-start per notebook

| Notebook | Binder | Colab |
|---|---|---|
| 01 — Full pipeline CLI | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Piyushjhu/HELIX_Toolbox/main?labpath=examples%2F01_full_pipeline_cli.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyushjhu/HELIX_Toolbox/blob/main/examples/01_full_pipeline_cli.ipynb) |
| 02 — ALPSS signal processing | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Piyushjhu/HELIX_Toolbox/main?labpath=examples%2F02_alpss_signal_processing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyushjhu/HELIX_Toolbox/blob/main/examples/02_alpss_signal_processing.ipynb) |
| 03 — SPADE spall & HEL analysis | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Piyushjhu/HELIX_Toolbox/main?labpath=examples%2F03_spade_spall_hel_analysis.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyushjhu/HELIX_Toolbox/blob/main/examples/03_spade_spall_hel_analysis.ipynb) |
| 04 — Post-processing & paper plots | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Piyushjhu/HELIX_Toolbox/main?labpath=examples%2F04_postprocessing_paper_plots.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Piyushjhu/HELIX_Toolbox/blob/main/examples/04_postprocessing_paper_plots.ipynb) |

> **Using your own data on Binder/Colab?** The first cell of each notebook detects the cloud environment and offers an upload widget. On **Codespaces**, drag-and-drop your CSV files directly into the VS Code file explorer then update the path variables.

---

## Table of Contents

0. [Run Online — No Installation Required](#run-online--no-installation-required)
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Command-Line Interface (CLI)](#command-line-interface-cli)
6. [Configuration Files](#configuration-files)
7. [Post-Processing Mode](#post-processing-mode)
8. [Paper Plot Generation](#paper-plot-generation)
9. [Physical Parameter Calculations](#physical-parameter-calculations)
10. [Output Files](#output-files)
11. [HEL Detection](#hel-detection)
12. [Spall Detection](#spall-detection)
13. [MAD Filter](#mad-filter)
14. [Algorithm Reference Documents](#algorithm-reference-documents)
15. [Examples](#examples)
16. [Troubleshooting](#troubleshooting)
17. [Credits](#credits)
18. [Citation](#citation)

---

## Overview

HELIX Toolbox integrates ALPSS (Automated Laser Photonic Doppler Velocimetry Signal Processing) and SPADE (Spall Analysis Toolkit) into a unified platform for single-point PDV (Photonic Doppler Velocimetry) data analysis. It provides two complementary ways to run the same analysis pipeline:

- **GUI** (`helix_analysis_toolbox.py`) — interactive desktop application for exploratory work, parameter tuning, and real-time feedback.
- **CLI** (`helix_cli_runner.py`) — config-file-driven command-line runner for batch processing, automation, and headless / HPC environments. All settings live in a human-readable, commented YAML file (`helix_master_config.yml`).

Both modes share identical analysis logic and produce the same outputs, from raw PDV signals through to complete spall strength, strain rate, HEL, and shock-stress analysis with full uncertainty quantification.

**Latest Updates (v2.1.0):**
- **Batch processing mode**: process a parent directory of per-shot subfolders in one CLI run (`batch_mode` + `subfolder_pattern` in `cli_settings`; see `helix_master_config_batch_process.json`)
- **Batch summary plotting**: `batch_summary_plot.py` aggregates spall and HEL results across all subfolders of a batch run into combined strength-vs-strain-rate figures
- **ALPSS noise-fraction filter**: `noise_filter_enabled` / `noise_filter_threshold` replace high-noise velocity samples with linear interpolation before plotting and saving
- **Consolidated data summary**: results now save to `<prefix>-Data_Summary.csv` (IGSN-prefixed from the parent folder of the output directory), replacing `enhanced_spall_summary.csv`
- **Run traceability**: every run saves a `<prefix>-Run_Config.json` next to the summary CSV recording all ALPSS/SPADE parameters and material properties used
- **First-local-minimum spall pullback (P3) detection**: prominence-based detection of the true pullback dip; traces with no qualifying valley, or with `P3 <= 0 m/s`, are classified DNS
- **RDP + Linear Hybrid HEL detection**: Robust elastic-plastic transition detection using Ramer–Douglas–Peucker simplification combined with linear regression on raw segments (see [HEL_DETECTION_ALGORITHM.md](HEL_DETECTION_ALGORITHM.md))
- **Horizontal-plateau 5-segment spall analysis**: Direct plateau → pullback → recompression qualification, with 5-segment fits for diagnostic plots and derived quantities
- **Paper-quality plotting suite**: `helix_paper_plots.py` (library, used at runtime by the GUI) plus standalone post-processing scripts in [`supplementary/paper_plots/`](supplementary/paper_plots/)
- **Robust IQ start-time detection**: New `use_robust_iq_detection` pipeline with configurable smoothing and persistence windows
- **Expanded material library**: Added Ti (CP-Ti / Grade 2), Ti-6Al-4V, Vanadium, and Magnesium alongside Cu, Zn, Brass, and Al
- **Configuration file support** for ALPSS and SPADE parameters with GUI override behaviour
- **Post-processing mode** with selective plot generation from existing SPADE outputs
- **MAD (Median Absolute Deviation)** statistical outlier filtering per material / laser-energy bracket
- **Command-line interface** for batch processing without the GUI

---

## Features

### 🔬 **Single Point PDV Analysis**
- Process raw PDV signals from single point measurements
- Automated carrier frequency removal with optional Gaussian notch filter
- Velocity extraction with uncertainty quantification
- Real-time signal processing and visualization

### 📊 **Comprehensive Analysis Pipeline**
- **ALPSS Integration**: Raw signal processing to velocity traces
- **SPADE Integration**: Spall strength, strain rate, and HEL analysis
- **Combined Mode**: Full pipeline from raw data to complete analysis
- **Individual Modes**: Run ALPSS or SPADE independently

### 🎛️ **Advanced Processing Options**
- **Gaussian Notch Filter**: Optional carrier frequency removal
- **Robust IQ Start-Time Detection**: Noise-tolerant detection of signal onset with configurable smoothing and persistence
- **Smoothing Parameters**: Configurable signal smoothing (Savitzky–Golay and Gaussian)
- **Peak Detection**: Automated feature detection with user controls
- **Uncertainty Propagation**: Complete error analysis throughout pipeline
- **MAD Filter**: Statistical outlier removal for peak velocities (per material / laser-energy bracket)
- **HEL Detection**: RDP + Linear Hybrid detection of the Hugoniot Elastic Limit (see [HEL_DETECTION_ALGORITHM.md](HEL_DETECTION_ALGORITHM.md))
- **Spall Detection**: Horizontal-plateau 5-segment analysis with P3/P4 DNS qualification (see [Spall Detection](#spall-detection))
- **Noise-Fraction Filter**: Optional ALPSS filter that replaces velocity samples whose instantaneous noise fraction exceeds a threshold with linear interpolation

### 📈 **Rich Output Generation**
- Velocity traces with uncertainty bands (aggregate and per-material)
- Spall strength vs. strain rate plots (per material + combined)
- Spall strength vs. shock stress plots and 3-D SVR surfaces (Al, Cu)
- Shock stress vs. laser energy, waveplate angle, and peak velocity
- HEL vs. peak velocity and elastic strain rate plots
- Row/column spatial analysis heatmaps and pair plots
- Velocity traces grouped by waveplate angle (full + HEL-focused)
- Velocity traces coloured by laser energy (2-D and 3-D views)
- Failure-detection summary table (CSV)
- Enhanced summary tables with all uncertainties

### 🖥️ **Two Ways to Run — Same Results**
- **GUI** (`helix_analysis_toolbox.py`): interactive desktop application on Windows, macOS, and Linux
- **CLI** (`helix_cli_runner.py`): config-file-driven batch runner — no display required, ideal for HPC / server use
- **YAML config templates**: fully commented `.yml` files make parameters self-documenting and version-controllable
- **Identical output**: both modes run the same `AnalysisThread` logic and produce the same CSVs and plots

---

## Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8 | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB |
| Disk space | 1 GB | 2 GB |
| OS | Windows 10 / macOS 10.15 / Ubuntu 20.04 | latest stable |

> **Important — always use a virtual environment.**  
> Modern macOS and Linux distributions mark their system Python as "externally managed" and will refuse a bare `pip install`. A virtual environment avoids all of these conflicts and keeps your system clean.

---

### Step 1 — Get the code

```bash
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox
```

---

### Step 2 — Create and activate a virtual environment

Pick **Option A (venv)** or **Option B (Conda)** — you only need one.

#### Option A: Python venv (recommended)

**macOS / Linux**
```bash
# Create the environment (do this once)
python3 -m venv helix_toolbox_env

# Activate it (do this every new terminal session)
source helix_toolbox_env/bin/activate
```

**Windows — PowerShell**
```powershell
# Create the environment (do this once)
python -m venv helix_toolbox_env

# If the 'python' command is not found, try:
py -3 -m venv helix_toolbox_env

# Activate it
.\helix_toolbox_env\Scripts\Activate.ps1

# If you get an execution-policy error, run this first, then re-activate:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows — Command Prompt (CMD)**
```cmd
python -m venv helix_toolbox_env
helix_toolbox_env\Scripts\activate.bat
```

Once activated you will see `(helix_toolbox_env)` at the start of your prompt — every `python` and `pip` command now operates inside the environment.

#### Option B: Conda

```bash
# Create the environment (do this once)
conda create -n helix_toolbox python=3.10 -y

# Activate it (do this every new terminal session)
conda activate helix_toolbox
```

---

### Step 3 — Install dependencies

With your environment active, upgrade pip then install all packages from `requirements.txt`:

**macOS / Linux**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell or CMD)**
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS note:** If you see errors related to `PyQt5` or `cv2` during install, first install the Qt5 system libraries:
> ```bash
> brew install qt@5
> ```
>
> **Ubuntu / Debian note:** If you see missing shared-library errors for PyQt5 or OpenCV, install the system dependencies first:
> ```bash
> sudo apt-get install -y libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxtst6 libxi6 libopencv-dev
> ```
>
> **Headless / server environments:** If there is no display, set the Qt platform before running:
> ```bash
> export QT_QPA_PLATFORM=offscreen
> ```

---

### Step 4 — Verify the installation

Run this one-liner to confirm that every key dependency imported successfully:

**macOS / Linux**
```bash
python -c "
import PyQt5, numpy, scipy, pandas, matplotlib, sklearn, cv2, findiff, yaml, openpyxl
print('Python    :', __import__('sys').version.split()[0])
print('NumPy     :', numpy.__version__)
print('SciPy     :', scipy.__version__)
print('Pandas    :', pandas.__version__)
print('Matplotlib:', matplotlib.__version__)
print('scikit-learn:', sklearn.__version__)
print('PyYAML    :', yaml.__version__)
print()
print('All dependencies OK.')
"
```

**Windows (PowerShell or CMD)**
```cmd
python -c "import PyQt5, numpy, scipy, pandas, matplotlib, sklearn, cv2, findiff, yaml, openpyxl; print('All dependencies OK.')"
```

Expected output ends with `All dependencies OK.` If any import fails, re-run `pip install -r requirements.txt` and check for error messages.

---

### Step 5 — Run HELIX Toolbox

**GUI (interactive desktop application)**

```bash
python helix_analysis_toolbox.py
```

On Windows you can also double-click `run_helix_toolbox.bat` — it activates the venv and launches the GUI automatically.

**CLI (batch / headless processing)**

```bash
python helix_cli_runner.py --config helix_master_config.yml
```

See the [Command-Line Interface (CLI)](#command-line-interface-cli) section for full options.

---

### Reactivating the environment in future sessions

You must activate the environment every time you open a new terminal before running HELIX Toolbox:

| Shell | Command |
|-------|---------|
| macOS / Linux (venv) | `source helix_toolbox_env/bin/activate` |
| Windows PowerShell (venv) | `.\helix_toolbox_env\Scripts\Activate.ps1` |
| Windows CMD (venv) | `helix_toolbox_env\Scripts\activate.bat` |
| Any OS (Conda) | `conda activate helix_toolbox` |

To deactivate when you are done: run `deactivate` (venv) or `conda deactivate` (Conda).

---

## Usage

### GUI Mode

1. **File Selection**
   - Choose single file or batch processing mode
   - Select input PDV data files (CSV format)
   - Set output directory for results

2. **Analysis Mode**
   - **ALPSS Only**: Process raw PDV data to velocity traces
   - **SPADE Only**: Analyze existing velocity files
   - **Combined**: Full pipeline from raw data to spall analysis

3. **Parameter Configuration**
   - **ALPSS Parameters**: Signal processing, filtering, and smoothing options
   - **SPADE Parameters**: Material properties and analysis models
   - **Advanced Options**: Gaussian notch filter, uncertainty multipliers
   - **GUI Overrides**: GUI parameter selections automatically override config file values for:
     - HEL time window (`hel_start_time_ns`, `hel_end_time_ns`)
     - SPADE analysis model (`hybrid_5_segment` vs `max_min`)
     - Spall analysis window (`spall_start_time_ns`, `spall_end_time_ns`)
     - Threshold velocity (`threshold_velocity_ms`)

4. **Run Analysis**
   - Monitor real-time progress with debug messages showing active parameters
   - View generated plots and results
   - Access comprehensive output files
   - Debug messages confirm which parameters are being used (e.g., `[HEL] Using time window=[X, Y] ns`, `[SPALL] Detection Method: Horizontal Plateau 5-Segment Analysis`)

---

## Command-Line Interface (CLI)

The `helix_cli_runner.py` script allows you to run HELIX Toolbox analysis from the command line without the GUI, which is much faster for batch processing.

### Basic Requirements

1. **Activated environment**: Make sure you have activated your virtual environment (see [Installation](#installation)) before running any commands.

2. **Config file**: Two options:
   - **Master Config (recommended)**: Single `helix_master_config.yml` (or `.json`) containing all settings
   - **Separate Configs**: Individual `alpss_config_default.yml` and `spade_config_default.yml` files

### Quick Start Examples

#### Example 1: Using Master Config File (Simplest — Recommended)

Edit `helix_master_config.yml` with your paths and settings, then:

```bash
python helix_cli_runner.py --config ./helix_master_config.yml
```

#### Example 2: Override Config File Settings

You can override any setting from the command line:

```bash
python helix_cli_runner.py \
    --config ./helix_master_config.yml \
    --input-dir /different/path/to/files \
    --output-dir /different/output/path
```

#### Example 3: ALPSS Only

```bash
python helix_cli_runner.py \
    --config ./helix_master_config.yml \
    --analysis-mode alpss_only
```

#### Example 4: SPADE Only (Using Existing ALPSS Outputs)

Edit `helix_master_config.yml` to set:
- `analysis_mode: spade_only`
- `spade_mode: manual`
- `spade_input_dir: /path/to/alpss/outputs`

Then run:
```bash
python helix_cli_runner.py --config ./helix_master_config.yml
```

### Input Selection (Choose One Method)

**Method 1: Explicit file list**
```bash
--input-files file1.csv file2.csv file3.csv
```

**Method 2: Directory + pattern (recommended for batch processing)**
```bash
--input-dir /path/to/files --input-pattern "C1--*.csv"
```

### Batch Mode (Multiple Shot Folders in One Run)

When your data is organised as one subfolder per shot group, the CLI can process all of them in a single run. Enable it in the master config's `cli_settings` (see `helix_master_config_batch_process.json` for a complete example):

```json
"cli_settings": {
    "batch_mode": true,
    "input_dir": "/path/to/parent_folder",
    "subfolder_pattern": "*",
    "output_subdir_name": "Output",
    "combined_output_dir": null,
    ...
}
```

- `batch_mode`: set `true` to treat `input_dir` as a parent folder containing one subfolder per shot group
- `subfolder_pattern`: glob to select which subfolders to process (default `*`)
- `output_subdir_name`: name of the output folder created inside each subfolder (default `Output`)
- `combined_output_dir`: optionally route all outputs to one shared directory instead of per-subfolder outputs

Each subfolder gets its own full analysis (ALPSS + SPADE) and its own `<prefix>-Data_Summary.csv` / `<prefix>-Run_Config.json`. A pass/fail summary for all subfolders is printed at the end.

After a batch run, aggregate results across all subfolders with:

```bash
python batch_summary_plot.py
```

which reads `helix_master_config_batch_process.json` and produces combined Spall Strength and HEL Strength vs strain-rate figures, colour-coded by material.

### Command-Line Arguments

**Master Config Mode** (when using `--config`):
- `--config`: Path to master config file (required)
- All other arguments are optional and override config file values

**Separate Configs Mode** (when using `--alpss-config` and `--spade-config`):
- `--alpss-config`: Path to ALPSS JSON config file
- `--spade-config`: Path to SPADE JSON config file
- `--output-dir`: Directory where all outputs will be saved
- `--input-dir`: Directory containing PDV files
- `--input-pattern`: Glob pattern to match files (default: `*.csv`)
- `--input-files`: Space-separated list of specific files
- `--param-folder`: Directory containing experiment metadata (CSV/Excel files)
- `--analysis-mode`: `both`, `alpss_only`, or `spade_only`
- `--spade-mode`: `auto` (use ALPSS outputs) or `manual` (provide SPADE inputs explicitly)

### Performance Tips

1. **Use directory + pattern** instead of listing individual files for large batches
2. **Disable plot saving** in config files if you only need CSV outputs (faster)
3. **Use `alpss_only` mode** first, then run SPADE separately if you need to iterate on SPADE parameters
4. **Run in background** for long batches:
   ```bash
   nohup python3 helix_cli_runner.py [args] > run.log 2>&1 &
   ```

---

## Configuration Files

The HELIX Toolbox supports loading analysis parameters from configuration files (JSON format). This feature allows you to:
- **Save time**: Reuse the same parameter sets across multiple analysis sessions
- **Ensure consistency**: Use identical parameters for reproducible results
- **Easy sharing**: Share configuration files with collaborators
- **Version control**: Track parameter changes over time

### Master Config File Structure

The master config file (`helix_master_config.json`) contains three main sections:

1. **`cli_settings`**: Command-line arguments and file paths
2. **`alpss_config`**: All ALPSS processing parameters
3. **`spade_config`**: All SPADE processing parameters
4. **`post_processing_config`**: Post-processing plot generation settings
5. **`material_properties`**: Material-specific properties (density, wave speeds, etc.)

### Example Master Config File

```json
{
    "cli_settings": {
        "input_dir": "/path/to/pdv/files",
        "input_files": null,
        "input_pattern": "*.csv",
        "output_dir": "/path/to/output",
        "param_folder": "/path/to/parameter/folder",
        "analysis_mode": "both",
        "spade_mode": "auto"
    },
    "alpss_config": {
        "save_data": "yes",
        "display_plots": "no",
        "save_all_plots": "no",
        "header_lines": 22,
        "time_to_take": 4e-06,
        "sample_rate": 128000000000.0,
        "use_notch_filter": false,
        "use_robust_iq_detection": true,
        "iq_threshold_factor": 0.8,
        "iq_smoothing_window_ns": 5.0,
        "iq_persistence_ns": 0.5,
        "smoothing_type": "savgol",
        "smoothing_window_ns": 6.0,
        "savgol_polyorder": 3,
        "noise_filter_enabled": true,
        "noise_filter_threshold": 1,
        "C0": 3950.0,
        "density": 8960.0
    },
    "spade_config": {
        "experiment_velocity_shots": true,
        "experiment_spall_analysis": true,
        "experiment_hel_detection": true,
        "spall_start_time_ns": 0.0,
        "spall_end_time_ns": 90.0,
        "threshold_velocity_ms": 5.0,
        "spall_smoothing_sigma_ns": 1.5,
        "min_recomp_ratio": 0.03,
        "min_recomp_velocity_ratio": 1.1,
        "min_recomp_time_ns": 2.5,
        "hel_start_time_ns": 0,
        "hel_end_time_ns": 20,
        "minimum_HEL_velocity_expected": 40.0,
        "hel_detection_min_points": 10,
        "hel_rdp_epsilon": 1.25,
        "hel_slope_drop_ratio": 0.9,
        "hel_min_plateau_duration": 0.5,
        "use_hel_t0_alignment_for_plots": true,
        "hel_t0_method": "signal_start",
        "align_velocity_threshold_ms": 10.0,
        "mad_filter_enabled": true,
        "mad_filter_threshold": 2.0,
        "skip_unknown_material_traces": true
    },
    "material_properties": {
        "Cu":   { "density": 8960.0, "bulk_wave_speed": 3950.0, "C0": 3950.0, "C_L": 4700.0 },
        "Zn":   { "density": 7140.0, "bulk_wave_speed": 3700.0, "C0": 3700.0, "C_L": 4200.0 },
        "Brass":{ "density": 8520.0, "bulk_wave_speed": 3800.0, "C0": 3800.0, "C_L": 4500.0 },
        "Al":   { "density": 2700.0, "bulk_wave_speed": 5240.0, "C0": 5240.0, "C_L": 6000.0 },
        "Ti":   { "density": 4510.0, "bulk_wave_speed": 5020.0, "C0": 5020.0, "C_L": 6070.0 },
        "Ti64": { "density": 4430.0, "bulk_wave_speed": 5130.0, "C0": 5130.0, "C_L": 6130.0 },
        "V":    { "density": 6100.0, "bulk_wave_speed": 5200.0, "C0": 5200.0, "C_L": 6000.0 },
        "Mg":   { "density": 1738.0, "bulk_wave_speed": 4490.0, "C0": 4490.0, "C_L": 5770.0 }
    }
}
```

The shipped `helix_master_config.json` contains a full list of materials including
`Copper`, `Zinc`, `Aluminum`, `Titanium`, `Ti_Grade2`, `CP-Ti`, `Ti-6Al-4V`, `Ti6Al4V`,
`Vanadium`, and `Magnesium` aliases.

**Noise-fraction filter** (`alpss_config`): when `noise_filter_enabled` is `true`, velocity samples whose instantaneous noise fraction exceeds `noise_filter_threshold` are replaced with linear interpolation from the surrounding valid samples, before plotting and CSV saving. This suppresses spurious velocity spikes in low-signal regions without altering clean portions of the trace.

### ALPSS Sample Rate and Smoothing

`sample_rate` is a **fallback**, not a fixed per-run sampling rate. For every raw
trace, ALPSS measures the mean spacing of the trace's `Time` column and uses
`fs = 1 / mean(diff(Time))`. The configured value is used only if that detection
cannot be completed. The effective rate used for each trace is included in the
automatically saved `<prefix>-Run_Config.json`.

When `smoothing_window_ns` is positive, it takes precedence over the legacy
sample-count setting `smoothing_window`. The code converts the physical window
to samples for each trace as `int(smoothing_window_ns * fs * 1e-9)`, then makes
the result odd (and at least three samples) for the smoothing algorithms. For
example, a 6 ns window becomes 769 samples at 128 GS/s and 481 samples at 80
GS/s. Set `smoothing_window_ns: null` to use `smoothing_window` directly.

### Using Configuration Files in GUI

1. **Saving Current Settings**: Click "Save Current Settings to Config" button
2. **Loading Settings**: Select "Use Config File" radio button, browse and load config file
3. **Mixing Modes**: You can use ALPSS config file + Manual SPADE parameters (or vice versa)

### Key SPADE Config Parameters

**Experiment toggles**
- `experiment_velocity_shots`: Enable velocity shots analysis
- `experiment_spall_analysis`: Enable spall strength analysis
- `experiment_hel_detection`: Enable HEL detection

**Spall analysis**
- `spall_start_time_ns`, `spall_end_time_ns`: Spall analysis window (ns, relative to aligned t=0)
- `threshold_velocity_ms`: Velocity threshold for shock-arrival detection (m/s)
- `spall_smoothing_sigma_ns`: Gaussian smoothing sigma applied once inside the spall window (ns; set `0` to disable)
- `min_recomp_ratio`: Required fraction of the pullback recovered by P4
- `min_recomp_velocity_ratio`: Required P4/P3 velocity factor
- `min_recomp_time_ns`: Minimum P3-to-P4 duration (ns)

The active spall calculation is always horizontal-plateau 5-segment analysis.
`analysis_model`, `spall_detection_method`, `spall_rdp_epsilon`, and
`min_pullback_velocity` remain accepted for compatibility with older config
files, but do not change this active path. Likewise, the active path currently
uses its internal P3-search defaults (`prominence_factor=0.01` and
`peak_distance_ns=3.0`) rather than values supplied through the config.

**HEL detection**
- `hel_start_time_ns`, `hel_end_time_ns`: HEL analysis window (ns, relative to aligned t=0)
- `minimum_HEL_velocity_expected`: Minimum HEL velocity to accept (m/s)
- `hel_detection_min_points`: Minimum consecutive raw-data points in the plateau segment
- `hel_rdp_epsilon`: RDP simplification tolerance for knee detection (m/s)
- `hel_slope_drop_ratio`: Required fractional drop between rise slope and plateau slope (`0.9` → plateau < 10% of rise)
- `hel_min_plateau_duration`: Minimum plateau duration in ns
- `hel_angle_threshold_deg`: Legacy angle-based fallback threshold (degrees)
- `hel_t0_method`: `"signal_start"` (default) uses first positive velocity with sustained increase; falls back to velocity threshold
- `use_hel_t0_alignment_for_plots`: Use HEL t=0 alignment for plot x-axes

**Outlier filtering & bookkeeping**
- `mad_filter_enabled`: Enable MAD outlier filter
- `mad_filter_threshold`: MAD filter threshold (typically 2.0–3.5)
- `skip_unknown_material_traces`: Skip traces with unknown material

**Note:** When using the GUI, the following parameters will override config file values:
- `spall_start_time_ns` and `spall_end_time_ns` (spall analysis window)
- `threshold_velocity_ms` (shock arrival threshold)
- `hel_start_time_ns` and `hel_end_time_ns` (HEL analysis window)

This ensures your GUI selections are always respected, even when loading a config file.

---

## Post-Processing Mode

Post-processing mode allows you to generate plots from existing SPADE analysis results **without rerunning the entire analysis**. This is useful when you want to:
- Regenerate plots with different settings
- Create new plots that weren't generated initially
- Quickly update plots after modifying config settings

### Configuration

In `helix_master_config.json`, set:

```json
{
    "post_processing_config": {
        "enabled": true,
        "spade_output_dir": "/path/to/your/spade/output",
        "plots": {
            "hel_vs_peak_velocity": true,
            "hel_vs_hel_strain_rate": true,
            "flyer_row_column_peak_velocity_heatmap": true,
            "flyer_row_column_pair_peak_velocity": true,
            "flyer_row_column_pair_peak_velocity_by_material_laser_energy": true,
            "peak_velocity_pattern_analysis": true,
            "shock_stress_vs_laser_energy": true,
            "shock_stress_vs_waveplate_angle": true,
            "shock_stress_vs_peak_velocity": true,
            "row_column_vs_peak_shock_stress": true,
            "laser_energy_vs_waveplate_angle": true
        }
    }
}
```

**Important:** The `spade_output_dir` must contain `velocity_shots_summary.csv` (generated by SPADE analysis).

### Running Post-Processing

```bash
python helix_cli_runner.py --config helix_master_config.json
```

The CLI runner will:
1. Detect `post_processing_config.enabled = true`
2. **Skip** ALPSS and SPADE analysis
3. Read existing `velocity_shots_summary.csv`
4. Generate only the plots you've enabled
5. Save plots to the `spade_output_dir`

### Important Notes

- **HEL Plots**: Require `experiment_hel_detection: true` and HEL data in summary CSV
- **Row/Column Plots**: Require `Flyer_Row` and `Flyer_Column` columns from parameter files
- **Other Config Sections**: Still used (e.g., `skip_unknown_material_traces`, material properties)

---

## Paper Plot Generation

The repository ships a set of standalone publication-quality plotting scripts that operate on the summary CSVs produced by SPADE. They do **not** re-run ALPSS or SPADE and can be invoked any time after a successful analysis. These scripts live in [`supplementary/paper_plots/`](supplementary/paper_plots/) because they are optional post-processing tools and are not needed to run the core toolbox.

The only paper-plot module that stays at the repo root is `helix_paper_plots.py`, which is imported at runtime by `helix_analysis_toolbox.py`.

### `supplementary/paper_plots/generate_paper_plots_standalone.py`

Generates the full set of paper figures from the consolidated data summary CSV (`<prefix>-Data_Summary.csv`; and `velocity_shots_summary.csv` if available). It reads the output directory from `helix_master_config.json` by default.

```bash
# Uses helix_master_config.json for the output path
python supplementary/paper_plots/generate_paper_plots_standalone.py

# Or pass an explicit summary CSV
python supplementary/paper_plots/generate_paper_plots_standalone.py /path/to/SPADE_analysis/JHAMAL00016-004-Data_Summary.csv

# With a custom config
python supplementary/paper_plots/generate_paper_plots_standalone.py /path/to/summary.csv /path/to/custom_config.json
```

Outputs (written to the SPADE analysis directory):
- `spall_vs_strain_rate.png` and per-material subplot version
- `spall_vs_shock_stress.png` and per-material subplot version
- `peak_velocity_vs_time.png`, `laser_energy_vs_time.png`
- `laser_energy_stability_table.csv`
- `failure_detection_summary.csv`
- `spall_svr_surface_3d.png` (Al & Cu 3-D SVR regression surfaces)
- `velocity_traces_by_waveplate_angle.png` and HEL-focused variant (±20 ns)

### `supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py`

Produces velocity-trace plots grouped by material with line colour mapped to laser energy (both 2-D and 3-D views).

```bash
python supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py \
    /path/to/SPADE_analysis/JHAMAL00016-004-Data_Summary.csv \
    --output-dir /path/to/SPADE_analysis \
    --plot-3d

# Or let the script pick up paths from helix_master_config.json
python supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py --config ./helix_master_config.json
```

### `helix_paper_plots.py` (repo root)

Library module at the repo root that provides the plotting primitives used by both the toolbox itself and the standalone scripts in `supplementary/paper_plots/` (publication "Data to Viz" styling, material colour/marker mappings, 3σ outlier filtering, etc.). Import it directly to build custom figures:

```python
from helix_paper_plots import (
    apply_data_to_viz_poster_style,
    generate_spall_vs_strain_rate_by_material_subplots,
    generate_spall_vs_shock_stress_plot,
    generate_all_plots_from_summary_files,
)
```

### `supplementary/paper_plots/run_plot_all_traces.sh`

Convenience bash wrapper that loops over materials (Zn, Cu, Brass) and generates per-material velocity trace plots in batches. Edit the `PYTHON`, `SCRIPT`, `SUMMARY`, and directory variables at the top of the file before running.

```bash
./supplementary/paper_plots/run_plot_all_traces.sh
```

---

## Physical Parameter Calculations

### 1. Free Surface Velocity Extraction

**Method**: Phase unwrapping and differentiation of PDV signal

The free surface velocity is extracted from the PDV signal using:

```
v(t) = (λ/2) × f_Doppler(t)
```

Where:
- `v(t)` = free surface velocity (m/s)
- `λ` = laser wavelength (typically 1550 nm)
- `f_Doppler(t)` = instantaneous Doppler shift frequency (Hz)

**Process**:
1. Signal demodulation: Extract In-phase (I) and Quadrature (Q) components via IQ analysis
2. Phase calculation: `φ(t) = arctan2(Q, I)`
3. Phase unwrapping: Remove 2π discontinuities
4. Frequency extraction: `f(t) = (1/2π) × dφ/dt`
5. Velocity conversion: `v(t) = (λ/2) × f(t)`
6. Smoothing: Apply Gaussian window for noise reduction

**Implementation**: `velocity_calculation()` in `ALPSS/alpss_main.py`

### 2. Velocity Uncertainty Calculation

**Method**: Instantaneous noise analysis with time-frequency uncertainty principle

```
Δv(t) = (λ/2) × Δf(t)
```

Where the frequency uncertainty is:

```
Δf(t) = η(t) × (1/π) × √[6 / (f_s × τ³)]
```

**Parameters**:
- `η(t)` = instantaneous noise fraction = `std(noise) / [A(t)/2]`
- `A(t)` = instantaneous signal amplitude
- `f_s` = sampling frequency (Hz)
- `τ` = characteristic time = FWHM of Gaussian smoothing window (s)

**Reference**: [Fratanduono et al., Review of Scientific Instruments 91, 051501 (2020)](https://doi.org/10.1063/12.0000870)

### 3. Spall Strength Calculation

**Method**: Acoustic approximation from pullback velocity

```
σ_spall = (1/2) × ρ₀ × c_b × Δv_pullback
```

Where:
- `σ_spall` = spall strength (GPa)
- `ρ₀` = initial material density (kg/m³)
- `c_b` = bulk sound speed (m/s)
- `Δv_pullback` = velocity pullback magnitude = |v_peak - v_min| (m/s)

**Uncertainty Propagation**:
```
Δσ_spall = (1/2) × ρ₀ × c_b × √(Δv²_peak + Δv²_min)
```

### 4. Shock Stress Calculation

**Method**: Hugoniot Equation of State (EOS)

```
U = c + S × u_p
σ_shock = ρ × U × u_p
```

Where:
- `U` = shock velocity (m/s)
- `c` = bulk wave speed (m/s)
- `S` = material-specific parameter
- `u_p` = particle velocity = `u_fs / 2` (m/s)
- `u_fs` = free surface velocity (peak velocity from trace) (m/s)
- `σ_shock` = shock stress (GPa)
- `ρ` = material density (kg/m³)

### 5. HEL (Hugoniot Elastic Limit) Calculation

**Method**: Gradient-based detection of low-slope plateaus

The HEL strength is determined from the elastic wave amplitude:

```
σ_HEL = (1/2) × ρ₀ × c_b × |free_surface_velocity| / 1e9
```

Where:
- `σ_HEL` = Hugoniot Elastic Limit (GPa)
- `free_surface_velocity` = mean velocity of HEL plateau segment (m/s)
- `ρ₀` = material density (kg/m³)
- `c_b` = bulk wave speed (m/s)

**Elastic Shock Strain Rate**:
```
ε̇_elastic = (1 / (2 × C_L)) × (dU / dt)
```

Where:
- `C_L` = longitudinal wave velocity (m/s)
- `dU` = change in free surface velocity (U_hel - U_0)
- `dt` = time duration (t_hel - t_0)

### 6. Summary Table of Calculations

| Parameter | Formula | Units | Uncertainty Method |
|-----------|---------|-------|-------------------|
| Velocity | v = (λ/2) × f | m/s | Time-frequency uncertainty |
| Spall Strength | σ = (1/2) × ρ × c × Δv | GPa | Propagate velocity uncertainties |
| Strain Rate | ε̇ = \|dv/dt\|/c | s⁻¹ | Linear fit residuals |
| HEL | σ_HEL = (1/2) × ρ × c × \|v_hel\| | GPa | Velocity uncertainty at HEL |
| Shock Stress | σ = ρ × U × u_p | GPa | EOS with particle velocity |
| Elastic Strain Rate | ε̇ = (1/(2×C_L)) × (dU/dt) | s⁻¹ | Time derivative |

---

## Output Files

### ALPSS Output Files

| File | Description | Columns |
|------|-------------|---------|
| `*--velocity.csv` | Raw velocity data | `Time_s`, `Velocity_m_s` |
| `*--velocity--smooth.csv` | Smoothed velocity data | `Time_s`, `Velocity_Smooth_m_s` |
| `*--vel--uncert.csv` | Velocity uncertainty | `Time_s`, `Velocity_Uncertainty_m_s` |
| `*--vel-smooth-with-uncert.csv` ⭐ | Smoothed velocity with uncertainty (main file for SPADE) | `Time_s`, `Velocity_Smooth_m_s`, `Velocity_Uncertainty_m_s`, `Velocity_Plus_Uncertainty_m_s` |
| `*--noise--frac.csv` | Noise fraction data | `Time_s`, `Noise_Fraction` |
| `*--voltage.csv` | Filtered voltage signal | `Time_s`, `Voltage_Real_V`, `Voltage_Imag_V` |
| `*--results.csv` | Analysis results summary | Various parameters and results |
| `*--inputs.csv` | Input parameters | Parameter names and values |

### SPADE Output Files

All SPADE outputs are written to `<output_dir>/SPADE_analysis/`.

| File | Description |
|------|-------------|
| `velocity_shots_summary.csv` | Complete velocity shots analysis summary (main output) |
| `<prefix>-Data_Summary.csv` | Consolidated summary with spall, HEL, and ALPSS results plus DNS classification. `<prefix>` is derived from the parent folder of the output directory (e.g. `JHAMAL00016-004-Data_Summary.csv`); plain `Data_Summary.csv` if the parent name is generic. Replaces the former `enhanced_spall_summary.csv` |
| `<prefix>-Run_Config.json` | Snapshot of every ALPSS/SPADE parameter, material property, input file, and analysis mode used for the run — saved automatically for traceability |
| `all_velocity_traces.png` | Combined velocity traces plot |
| `shock_stress_vs_laser_energy_by_material.png` | Shock stress vs laser energy |
| `shock_stress_vs_waveplate_angle_by_material.png` | Shock stress vs waveplate angle |
| `shock_stress_vs_peak_velocity_by_material.png` | Shock stress vs peak velocity |
| `hel_vs_peak_velocity_by_material.png` | HEL vs peak velocity (if HEL enabled) |
| `hel_vs_hel_strain_rate_by_material.png` | HEL vs HEL strain rate (if HEL enabled) |
| `flyer_row_column_peak_velocity_heatmap.png` | Row/column heatmap of peak velocity |
| `flyer_row_column_pair_peak_velocity.png` | Row/column pair scatter plot |
| `flyer_row_column_pair_peak_velocity_by_material_laser_energy.png` | Row/column by material with laser energy color coding |
| `peak_velocity_pattern_analysis.png` | Pattern analysis (laser energy and location effects) |
| `laser_energy_vs_waveplate_angle.png` | Laser energy vs waveplate angle |
| `row_column_vs_peak_shock_stress.png` | Row/column vs shock stress plots |
| `spall_plots/` | Per-trace spall detection plots (horizontal-plateau 5-segment fits) |
| `HEL_plots/` | Per-trace HEL detection plots with RDP knee and regression segments |

### Paper Plot Outputs (from standalone scripts)

Generated by `generate_paper_plots_standalone.py` and `plot_velocity_traces_by_laser_energy.py`:

| File | Description |
|------|-------------|
| `spall_vs_strain_rate.png` | Combined spall strength vs strain rate (all materials) |
| `spall_vs_strain_rate_by_material_subplots.png` | Per-material subplots |
| `spall_vs_shock_stress.png` | Combined spall vs shock stress |
| `spall_vs_shock_stress_by_material_subplots.png` | Per-material subplots |
| `peak_velocity_vs_time.png` | Peak velocity trend over experiment time |
| `laser_energy_vs_time.png` | Laser energy stability over time |
| `laser_energy_stability_table.csv` | Laser energy statistics summary |
| `failure_detection_summary.csv` | Failure reasons and counts |
| `spall_svr_surface_3d.png` | 3-D SVR regression surface (spall vs strain rate vs shock stress) for Al & Cu |
| `velocity_traces_by_waveplate_angle.png` | Velocity traces grouped by waveplate angle |
| `velocity_traces_by_waveplate_angle_hel.png` | HEL-focused (±20 ns) velocity traces by waveplate angle |
| `velocity_traces_by_laser_energy.png` | Velocity traces coloured by laser energy (2-D) |
| `velocity_traces_by_laser_energy_3d.png` | 3-D velocity-traces surface coloured by laser energy |
| `velocity_traces_by_shock_stress_3d.png` | 3-D velocity-traces surface coloured by shock stress |

### ALPSS Plot Files (if enabled)

When `save_all_plots: "subfolder"` is enabled, ALPSS creates a subfolder `{filename}_plots/` containing:
- `--velocity_with_uncertainty.png`: Velocity with uncertainty bands
- `--iq_analysis.png`: IQ signal components and magnitude
- `--IQ_start_time_detection.png`: IQ start time detection
- `--velocity_comparison.png`: Raw vs smoothed velocity
- `--imported_spectrogram.png`: Original signal spectrogram
- `--noise_analysis.png`: Noise analysis (2 panels)
- `--peak_detection.png`: Peak detection results
- And more (see config for full list)

---

## HEL Detection

HEL (Hugoniot Elastic Limit) detection uses an **RDP + Linear Hybrid** method that combines geometric simplification with linear regression on raw data for a noise-robust detection of the elastic–plastic transition.

> **Full algorithm reference:** [HEL_DETECTION_ALGORITHM.md](HEL_DETECTION_ALGORITHM.md)

### Detection Method

1. **Time-Zero Alignment**: Find the first point where velocity > 0 and sustained/increasing over a configurable window to establish `t=0` (`hel_t0_method: "signal_start"`, with velocity-threshold fallback).
2. **Uncertainty Filtering**: Exclude points where `relative_uncertainty >= 1.0`.
3. **Window Extraction**: Clip data to `[hel_start_time_ns, hel_end_time_ns]`.
4. **RDP Simplification**: Apply Ramer–Douglas–Peucker with `hel_rdp_epsilon` to extract candidate knee points.
5. **Linear Regression on Raw Segments**: For each candidate knee, fit linear regressions to the rise segment (before the knee) and plateau segment (after the knee) using the **raw data**, not the RDP vertices.
6. **Physics Validation**:
   - Rise slope must be positive and significantly larger than plateau slope (`hel_slope_drop_ratio`, default 0.9 → plateau slope < 10% of rise slope).
   - Plateau must last at least `hel_min_plateau_duration` ns.
   - Detected velocity must exceed `minimum_HEL_velocity_expected`.
   - Elastic strain rate must be positive.
7. **HEL Plateau Velocity**: Mean velocity of the validated plateau segment.
8. **HEL Strength**: `σ_HEL = 0.5 × ρ × c_b × |free_surface_velocity| / 1e9` (GPa)

### Elastic Shock Strain Rate

```
ε̇_elastic = (1 / (2 × C_L)) × (dU / dt)
```

Where `C_L` is the longitudinal wave velocity, `dU = U_hel - U_0`, and `dt = t_hel - t_0`.

### Configuration

In `spade_config`:
```json
{
    "experiment_hel_detection": true,
    "hel_start_time_ns": 0,
    "hel_end_time_ns": 20,
    "minimum_HEL_velocity_expected": 40.0,
    "hel_detection_min_points": 10,
    "hel_rdp_epsilon": 1.25,
    "hel_slope_drop_ratio": 0.9,
    "hel_min_plateau_duration": 0.5,
    "hel_angle_threshold_deg": 86.0,
    "use_hel_t0_alignment_for_plots": true,
    "hel_t0_method": "signal_start"
}
```

**Note:** When using the GUI, the HEL time window parameters (`hel_start_time_ns` and `hel_end_time_ns`) set in the GUI override any values from the config file.

### Output

HEL values are stored in `velocity_shots_summary.csv`:
- `hel_strength_gpa`: HEL strength (GPa)
- `hel_velocity_ms`: HEL free surface velocity (m/s)
- `hel_strain_rate_s^-1`: Elastic shock strain rate (1/s)
- `hel_ok`: Boolean indicating if HEL was successfully detected
- `hel_consecutive_points`: Number of consecutive points in the validated plateau segment
- `hel_segment_time_ns`: Time duration of the plateau segment (ns)

Per-trace HEL detection plots are saved to the `SPADE_analysis/HEL_plots/` subfolder when `plot_individual: true`.

---

## Spall Detection

Spall strength and strain rate are extracted from the characteristic
**Plateau → Pullback → Recompression** signature using the horizontal-plateau
5-segment procedure. The active HELIX spall path does not perform RDP topology
selection; RDP is used by the HEL detector.

### Active Detection Procedure

1. Remove velocity samples whose relative uncertainty is at least one, align the
   trace to the ALPSS/HEL arrival time when available (otherwise use the velocity
   threshold), and clip to `[spall_start_time_ns, spall_end_time_ns]`.
2. Apply Gaussian smoothing once in that window using
   `spall_smoothing_sigma_ns`; this smoothing is separate from the ALPSS velocity
   smoothing and is disabled by setting the sigma to `0`.
3. Find the global maximum, then define the plateau as all smoothed samples at or
   above 95% of that maximum. Their mean is the horizontal plateau velocity.
4. Use the plateau end as P2, detect post-plateau local valleys (P3 candidates),
   and pair each with the first prominent local peak after it (P4 candidate).
5. Accept only a P3/P4 pair that passes the physical P3 and recompression gates
   below. The P2-to-P3 drop provides spall strength; the P2-to-P3 line slope
   provides strain rate.

### Pullback Minimum (P3) Detection

The pullback minimum is the first viable local minimum after the plateau, found
with prominence-based peak detection on the inverted, smoothed post-plateau
signal. The prominence threshold is `max(0.01 * plateau_velocity, 1 m/s)` and
candidate valleys are separated by at least 3 ns (converted to samples from the
trace time spacing). Valleys with `|P3| <= 10 m/s` are treated as decayed
baseline candidates and are not used. If no viable valley is followed by a
prominent P4 candidate, the trace is classified **DNS** rather than substituting
a global minimum.

P3 must also be **strictly positive**. For the selected P3/P4 pair, final DNS
validation checks `P3 <= 0 m/s` before reporting any recompression result.
HELIX retains the fitted diagnostic plot for that trace, but does not report a
spall strength.

### Analysis Model Compatibility

`analysis_model` is retained for compatibility with older configuration files,
but it is not a run-time selector in the active HELIX spall path. Strength is
calculated from the horizontal plateau-to-P3 velocity drop; strain rate is
calculated from the P2-to-P3 release slope.

### Rebound / Recompression Gating

After a positive P3 is found, HELIX pairs it with the first prominent local P4
peak after the valley. To prevent post-shock ringing from being reported as
spall, the pair must satisfy every configured gate:

- `P4 > P3` (a real re-acceleration);
- `min_recomp_velocity_ratio`: P4 must exceed P3 by the configured factor;
- `min_recomp_time_ns`: P4 must occur at least the configured duration after P3;
- `min_recomp_ratio`: the P3-to-P4 rebound must recover the configured fraction of the plateau-to-P3 pullback.

### Configuration

In `spade_config`:
```json
{
    "experiment_spall_analysis": true,
    "spall_start_time_ns": 0.0,
    "spall_end_time_ns": 90.0,
    "threshold_velocity_ms": 5.0,
    "spall_smoothing_sigma_ns": 1.5,
    "min_recomp_ratio": 0.03,
    "min_recomp_velocity_ratio": 1.1,
    "min_recomp_time_ns": 2.5
}
```

### Output

Spall results appear in `velocity_shots_summary.csv` and `<prefix>-Data_Summary.csv`:
- `Spall_Strength_GPa`, `Spall_Strength_Uncertainty_GPa`
- `Strain_Rate_s^-1`, `Strain_Rate_Uncertainty_s^-1`
- `Peak_Velocity_ms`, `Min_Velocity_ms` (post-peak valley)
- `Shock_Stress_GPa` (from Hugoniot EOS)
- `DNS_Classification` / `Processing_Status` (spall vs. DNS vs. failure reason)

Per-trace spall plots are saved to `SPADE_analysis/spall_plots/`.

---

## MAD Filter

The MAD (Median Absolute Deviation) filter is a statistical outlier removal method applied to peak velocities.

### Method

1. **Group Data**: Group by material and laser energy brackets (±30 mJ from mean)
2. **Calculate MAD**: For each group, calculate median and MAD
3. **Asymmetric MAD**: Use `MAD_lower` for values below median
4. **Modified Z-Score**: `M_i = 0.6745 × |value - median| / MAD_lower`
5. **Filter**: Remove points where `M_i >= threshold`

### Configuration

In `spade_config`:
```json
{
    "mad_filter_enabled": true,
    "mad_filter_threshold": 2.0
}
```

**Threshold Guidelines**:
- `2.0`: More aggressive (removes more outliers)
- `2.5`: Moderate
- `3.0`: Standard (removes extreme outliers)
- `3.5`: Conservative (removes only very extreme outliers)

### Application

- Applied **per material** and **per laser energy bracket**
- Only affects peak velocity values
- Filtered traces are excluded from plots and analysis
- Filtered trace basenames are logged

The MAD filter is applied per material and per laser energy bracket (±30 mJ from mean).

---

## Algorithm Reference Documents

Detailed step-by-step algorithm documentation lives alongside this README:

| Document | Description |
|----------|-------------|
| [HEL_DETECTION_ALGORITHM.md](HEL_DETECTION_ALGORITHM.md) | Full RDP + Linear Hybrid HEL detection pipeline: time-zero alignment, uncertainty filtering, RDP simplification, raw-segment linear regression, physics validation |
| [Spall Detection](#spall-detection) (this README) | Active horizontal-plateau 5-segment spall analysis and P3/P4 DNS qualification |
| [HELIX_spall_detection_methodology.pdf](HELIX_spall_detection_methodology.pdf) | Spall detection and DNS qualification overview, including the mandatory positive-P3 criterion |
| [supplementary/references/SPALL_STRENGTH_CALCULATION.tex](supplementary/references/SPALL_STRENGTH_CALCULATION.tex) | Derivation and uncertainty propagation for the acoustic spall-strength formula |
| [CHANGELOG.md](CHANGELOG.md) | Release history and feature changes |
| [supplementary/README.md](supplementary/README.md) | Description of optional / non-runtime files |

---

## Examples

The [`examples/`](examples/) folder contains four Jupyter notebooks that walk
through common HELIX Toolbox workflows, from a full end-to-end run to
fine-grained post-processing:

| Notebook | What it shows |
|----------|---------------|
| [01_full_pipeline_cli.ipynb](examples/01_full_pipeline_cli.ipynb) | Raw PDV CSVs → velocity → spall strength via CLI + YAML config |
| [02_alpss_signal_processing.ipynb](examples/02_alpss_signal_processing.ipynb) | ALPSS only: IQ extraction, velocity trace, uncertainty bands |
| [03_spade_spall_hel_analysis.ipynb](examples/03_spade_spall_hel_analysis.ipynb) | SPADE only: spall strength, strain rate, shock stress, HEL detection |
| [04_postprocessing_paper_plots.ipynb](examples/04_postprocessing_paper_plots.ipynb) | Regenerate paper plots from existing SPADE summary CSVs |

**Quick start:**
```bash
pip install jupyter
jupyter notebook examples/
```

Place any reference figures in [`examples/figures/`](examples/figures/) to
display them in the notebooks alongside your own results.

---

## Troubleshooting

### Common Issues

**Error: "No PDV input files found"**
- Check that `--input-dir` exists and contains files matching `--input-pattern`
- Or use `--input-files` to specify files explicitly

**Error: "Failed to load config"**
- Verify config file paths are correct
- Check that config files are valid JSON or YAML
- "YAML support requires PyYAML": your environment is missing PyYAML — run `pip install pyyaml` (or `pip install -r requirements.txt`), or switch to a `.json` config

**Error: "ModuleNotFoundError"**
- Install missing dependencies: `pip install -r requirements.txt`
- Or individually: `pip install pandas numpy matplotlib scipy PyQt5 openpyxl pyyaml`

**Error: "velocity_shots_summary.csv not found" (Post-processing)**
- Check that `spade_output_dir` points to the correct directory
- Verify the file exists: `ls /path/to/spade/output/velocity_shots_summary.csv`

**HEL Detection Not Working**
- Ensure `experiment_hel_detection: true` in config
- Check that `minimum_HEL_velocity_expected` is not too high
- Verify sufficient data points in HEL analysis window

**MAD Filter Not Removing Data**
- Check that `mad_filter_enabled: true` in config
- Try lowering `mad_filter_threshold` (e.g., 2.0 instead of 3.0)
- Verify data is grouped correctly by material and laser energy

**Plots Not Generating**
- Check that required columns exist in `velocity_shots_summary.csv`
- For HEL plots, ensure HEL detection was enabled during SPADE analysis
- For row/column plots, ensure parameter files contain `Flyer_Row` and `Flyer_Column`

**Progress Not Showing (CLI)**
- The script prints progress to stdout in real-time
- For background runs, redirect to a log file: `> run.log 2>&1`

### Diagnostic Tools

- `diagnose_high_laser_energy.py`: Find files with laser energy anomalies
  ```bash
  python diagnose_high_laser_energy.py --spade-output-dir /path/to/output
  ```

---

## Credits

### ALPSS (Automated Laser Photonic Doppler Velocimetry Signal Processing)
**Author:** Jake Diamond  
**GitHub:** [@Jake-Diamond-9](https://github.com/Jake-Diamond-9)  
**Description:** Original ALPSS package for PDV signal processing and velocity extraction

### SPADE (Spall Analysis Toolkit)
**Author:** Piyush Wanchoo  
**GitHub:** [@Piyushjhu](https://github.com/Piyushjhu)  
**Description:** Spall strength and strain rate analysis toolkit

### HELIX Toolbox
**Author:** Piyush Wanchoo  
**Institution:** Johns Hopkins University  
**Description:** Integration and GUI for ALPSS and SPADE

---

## Citation

If you use HELIX Toolbox in your research, please cite:

```bibtex
@software{helix_toolbox_2026,
  title={HELIX Toolbox: A Comprehensive GUI for Single Point PDV Data Analysis},
  author={Wanchoo, Piyush},
  year={2026},
  url={https://github.com/Piyushjhu/HELIX_Toolbox}
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

For questions, issues, or feature requests, please:
1. Check this README and the technical documentation files
2. Search existing [Issues](https://github.com/Piyushjhu/HELIX_Toolbox/issues)
3. Create a new issue with detailed information

---

## Acknowledgments

- **Jake Diamond** for the original ALPSS package
- **Johns Hopkins University** for research support
- The scientific community for PDV and spall analysis methodology

---

**HELIX Toolbox** - Advancing single point PDV data analysis for shock physics research across all platforms. 🖥️💻📱
