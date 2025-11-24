# HELIX Toolbox

**A Comprehensive GUI for Single Point PDV Data Analysis**

**Author:** Piyush Wanchoo  
**GitHub:** [@Piyushjhu](https://github.com/Piyushjhu)  
**Institution:** Johns Hopkins University  
**Year:** 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Command-Line Interface (CLI)](#command-line-interface-cli)
6. [Configuration Files](#configuration-files)
7. [Post-Processing Mode](#post-processing-mode)
8. [Physical Parameter Calculations](#physical-parameter-calculations)
9. [Output Files](#output-files)
10. [HEL Detection](#hel-detection)
11. [MAD Filter](#mad-filter)
12. [Troubleshooting](#troubleshooting)
13. [Credits](#credits)
14. [Citation](#citation)

---

## Overview

HELIX Toolbox is a comprehensive graphical user interface (GUI) that combines ALPSS (Automated Laser Photonic Doppler Velocimetry Signal Processing) and SPADE (Spall Analysis Toolkit) for single point PDV (Photonic Doppler Velocimetry) data analysis. This tool provides an integrated workflow from raw PDV signals to complete spall strength analysis with uncertainty quantification.

**Latest Updates:**
- Configuration file support for ALPSS and SPADE parameters
- Enhanced post-processing with selective plot generation
- HEL (Hugoniot Elastic Limit) detection with strain rate calculation
- MAD (Median Absolute Deviation) statistical outlier filtering
- Command-line interface for batch processing
- Comprehensive diagnostic tools
- **GUI parameter overrides**: GUI selections now properly override config file values for HEL time windows and SPADE analysis models
- **Improved plot generation**: Fixed all velocity traces plot and streamlined ALPSS plot options
- **Enhanced debugging**: Added parameter logging to verify GUI selections are being used

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
- **Smoothing Parameters**: Configurable signal smoothing
- **Peak Detection**: Automated feature detection with user controls
- **Uncertainty Propagation**: Complete error analysis throughout pipeline
- **MAD Filter**: Statistical outlier removal for peak velocities
- **HEL Detection**: Gradient-based Hugoniot Elastic Limit detection

### 📈 **Rich Output Generation**
- Velocity traces with uncertainty bands
- Spall strength vs. strain rate plots
- Shock stress analysis plots
- HEL vs. peak velocity and strain rate plots
- Row/column spatial analysis heatmaps
- Enhanced summary tables with all uncertainties

### 🖥️ **Cross-Platform Compatibility**
- **Windows**: Native Windows GUI with Explorer integration
- **macOS**: Optimized for macOS with native file dialogs
- **Linux**: Full Linux support with X11 integration
- **Unified Interface**: Same features across all platforms

---

## Installation

### System Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space

### Quick Start

#### Windows
```cmd
# Method 1: Using batch file (easiest)
# Double-click run_helix_toolbox.bat

# Method 2: Command line
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox
pip install -r requirements.txt
python helix_analysis_toolbox.py
```

#### macOS/Linux
```bash
# Clone the repository
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox

# Install dependencies
pip install -r requirements.txt

# Run the GUI
python helix_analysis_toolbox.py
```

### Environment Setup (Recommended)

Create an isolated Python environment to avoid dependency conflicts. This is highly recommended to prevent conflicts with other Python packages.

#### Option A: Python venv (Recommended)

**Step 1: Create Virtual Environment**

**Windows (PowerShell)**
```powershell
# Navigate to HELIX Toolbox directory
cd path\to\HELIX_Toolbox_v_2

# Create virtual environment
py -3 -m venv helix_toolbox_env

# Or if py command doesn't work, try:
python -m venv helix_toolbox_env
```

**Windows (Command Prompt / CMD)**
```cmd
# Navigate to HELIX Toolbox directory
cd path\to\HELIX_Toolbox_v_2

# Create virtual environment
py -3 -m venv helix_toolbox_env

# Or if py command doesn't work, try:
python -m venv helix_toolbox_env
```

**macOS/Linux (Terminal)**
```bash
# Navigate to HELIX Toolbox directory
cd path/to/HELIX_Toolbox_v_2

# Create virtual environment
python3 -m venv helix_toolbox_env
```

**Step 2: Activate Virtual Environment**

**Windows (PowerShell)**
```powershell
# Activate the virtual environment
.\helix_toolbox_env\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt / CMD)**
```cmd
# Activate the virtual environment
helix_toolbox_env\Scripts\activate.bat
```

**macOS/Linux (Terminal)**
```bash
# Activate the virtual environment
source helix_toolbox_env/bin/activate
```

**Step 3: Install Dependencies**

Once activated, you should see `(helix_toolbox_env)` at the beginning of your command prompt. Then install dependencies:

**Windows (PowerShell or CMD)**
```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**macOS/Linux (Terminal)**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Step 4: Verify Installation**

**Windows (PowerShell or CMD)**
```cmd
# Check Python version (should show Python 3.7+)
python --version

# Check if packages are installed
pip list
```

**macOS/Linux (Terminal)**
```bash
# Check Python version (should show Python 3.7+)
python --version

# Check if packages are installed
pip list
```

**Deactivate Virtual Environment**

When you're done working, deactivate the environment:

**Windows (PowerShell or CMD)**
```cmd
deactivate
```

**macOS/Linux (Terminal)**
```bash
deactivate
```

**Note:** After deactivating, you'll need to reactivate the environment each time you open a new terminal session to use HELIX Toolbox.

#### Option B: Conda

**Step 1: Create Conda Environment**

**Windows (Anaconda Prompt or PowerShell)**
```cmd
conda create -n helix_toolbox python=3.10 -y
```

**macOS/Linux (Terminal)**
```bash
conda create -n helix_toolbox python=3.10 -y
```

**Step 2: Activate Conda Environment**

**Windows (Anaconda Prompt or PowerShell)**
```cmd
conda activate helix_toolbox
```

**macOS/Linux (Terminal)**
```bash
conda activate helix_toolbox
```

**Step 3: Install Dependencies**

**Windows (Anaconda Prompt or PowerShell)**
```cmd
pip install -r requirements.txt
```

**macOS/Linux (Terminal)**
```bash
pip install -r requirements.txt
```

**Deactivate Conda Environment**

**Windows (Anaconda Prompt or PowerShell)**
```cmd
conda deactivate
```

**macOS/Linux (Terminal)**
```bash
conda deactivate
```

### System Packages (if needed)
- macOS: `brew install qt5`
- Ubuntu/Debian: `sudo apt-get install -y libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxtst6 libxi6`

Headless environments may require:
```bash
export QT_QPA_PLATFORM=offscreen
```

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
   - Debug messages confirm which parameters are being used (e.g., `[HEL] Using time window=[X, Y] ns`, `[SPALL] Using analysis_model='hybrid_5_segment'`)

---

## Command-Line Interface (CLI)

The `helix_cli_runner.py` script allows you to run HELIX Toolbox analysis from the command line without the GUI, which is much faster for batch processing.

### Basic Requirements

1. **Config Files**: You have two options:
   - **Master Config (Recommended)**: Single `helix_master_config.json` file containing all settings
   - **Separate Configs**: Individual `alpss_config_default.json` and `spade_config_default.json` files

2. **Python Environment**: Make sure you have all dependencies installed

### Quick Start Examples

#### Example 1: Using Master Config File (Simplest - Recommended)

Edit `helix_master_config.json` with your paths and settings, then:

```bash
python3 helix_cli_runner.py --config ./helix_master_config.json
```

#### Example 2: Override Config File Settings

You can override any setting from the command line:

```bash
python3 helix_cli_runner.py \
    --config ./helix_master_config.json \
    --input-dir /different/path/to/files \
    --output-dir /different/output/path
```

#### Example 3: ALPSS Only

```bash
python3 helix_cli_runner.py \
    --config ./helix_master_config.json \
    --analysis-mode alpss_only
```

#### Example 4: SPADE Only (Using Existing ALPSS Outputs)

Edit `helix_master_config.json` to set:
- `"analysis_mode": "spade_only"`
- `"spade_mode": "manual"`
- `"spade_input_dir": "/path/to/alpss/outputs"`

Then run:
```bash
python3 helix_cli_runner.py --config ./helix_master_config.json
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
        "spall_calculation": "no",
        "header_lines": 5,
        "time_to_take": 1e-05,
        "use_notch_filter": true,
        "carrier_freq": 1500000000.0,
        "smoothing_window_size": 51
    },
    "spade_config": {
        "experiment_velocity_shots": true,
        "experiment_spall_analysis": false,
        "experiment_hel_detection": true,
        "align_velocity_threshold_ms": 30.0,
        "minimum_HEL_velocity_expected": 15.0,
        "hel_detection_min_points": 3,
        "mad_filter_enabled": true,
        "mad_filter_threshold": 2.0,
        "skip_unknown_material_traces": true
    },
    "material_properties": {
        "Cu": {
            "density": 8960.0,
            "bulk_wave_speed": 3940.0,
            "C_L": 4760.0
        }
    }
}
```

### Using Configuration Files in GUI

1. **Saving Current Settings**: Click "Save Current Settings to Config" button
2. **Loading Settings**: Select "Use Config File" radio button, browse and load config file
3. **Mixing Modes**: You can use ALPSS config file + Manual SPADE parameters (or vice versa)

### Key SPADE Config Parameters

- `experiment_velocity_shots`: Enable velocity shots analysis
- `experiment_spall_analysis`: Enable spall strength analysis
- `experiment_hel_detection`: Enable HEL detection
- `analysis_model`: Spall analysis method - `"hybrid_5_segment"` (5-segment strain computation) or `"max_min"` (peak/valley detection)
- `spall_start_time_ns`: Start time for spall analysis window (ns, relative to t=0 after alignment)
- `spall_end_time_ns`: End time for spall analysis window (ns, relative to t=0 after alignment)
- `threshold_velocity_ms`: Velocity threshold for shock arrival detection (m/s)
- `align_velocity_threshold_ms`: Velocity threshold for trace alignment (m/s)
- `minimum_HEL_velocity_expected`: Minimum HEL velocity to accept (m/s)
- `hel_detection_min_points`: Minimum consecutive points for HEL detection
- `hel_start_time_ns`: Start time for HEL analysis window (ns, relative to t=0 after alignment)
- `hel_end_time_ns`: End time for HEL analysis window (ns, relative to t=0 after alignment)
- `mad_filter_enabled`: Enable MAD outlier filter
- `mad_filter_threshold`: MAD filter threshold (typically 2.0-3.5)
- `skip_unknown_material_traces`: Skip traces with unknown material

**Note:** When using the GUI, the following parameters will override config file values:
- `analysis_model` (SPADE analysis method selection)
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

| File | Description |
|------|-------------|
| `velocity_shots_summary.csv` | Complete velocity shots analysis summary (main output) |
| `spall_summary.csv` | Spall analysis summary (if spall analysis enabled) |
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

HEL (Hugoniot Elastic Limit) detection uses a gradient-based method to identify low-slope plateaus in velocity-time data.

### Detection Method

1. **Gradient Calculation**: `gradient = d(velocity)/d(time)` using `np.gradient()`
2. **Angle Conversion**: `angle = arctan(|gradient|)` in degrees
3. **Low-Slope Segment Detection**: Find consecutive points where `angle < angle_threshold_deg`
4. **Minimum Segment Length**: At least `hel_detection_min_points` consecutive points (default: 3, configurable)
5. **HEL Plateau Velocity**: Mean velocity of the low-slope segment
6. **HEL Strength**: `σ_HEL = 0.5 × ρ × c_b × |free_surface_velocity| / 1e9` (GPa)

### Constraints

- **Minimum HEL Velocity**: `minimum_HEL_velocity_expected` (default: 15.0 m/s, configurable)
  - If detected HEL velocity is below this, HEL is rejected
- **Relative Uncertainty Filter**: Points with `relative_uncertainty >= 1.0` are excluded
- **HEL Analysis Window**: Configurable via `hel_start_time_ns` and `hel_end_time_ns`
- **Negative Strain Rate**: HEL detections with negative strain rate are rejected

### Elastic Shock Strain Rate

Calculated as:
```
ε̇_elastic = (1 / (2 × C_L)) × (dU / dt)
```

Where:
- `C_L` = longitudinal wave velocity (from material properties)
- `dU = U_hel - U_0` (change in free surface velocity)
- `dt = t_hel - t_0` (time duration)

### Configuration

In `spade_config`:
```json
{
    "experiment_hel_detection": true,
    "minimum_HEL_velocity_expected": 15.0,
    "hel_detection_min_points": 3,
    "hel_angle_threshold_deg": 45.0,
    "hel_start_time_ns": 0.0,
    "hel_end_time_ns": null
}
```

**Note:** When using the GUI, the HEL time window parameters (`hel_start_time_ns` and `hel_end_time_ns`) set in the GUI will override any values from the config file. This ensures your GUI selections are always respected.

### Output

HEL values are stored in `velocity_shots_summary.csv`:
- `hel_strength_gpa`: HEL strength (GPa)
- `hel_velocity_ms`: HEL free surface velocity (m/s)
- `hel_strain_rate_s^-1`: Elastic shock strain rate (1/s)
- `hel_ok`: Boolean indicating if HEL was successfully detected
- `hel_consecutive_points`: Number of consecutive points in HEL segment
- `hel_segment_time_ns`: Time duration of HEL segment (ns)

**Key Constraints:**
- Minimum HEL velocity: `minimum_HEL_velocity_expected` (default: 15.0 m/s)
- Minimum consecutive points: `hel_detection_min_points` (default: 3)
- Relative uncertainty filter: Points with `relative_uncertainty >= 1.0` are excluded
- Negative strain rate: HEL detections with negative strain rate are rejected

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

## Troubleshooting

### Common Issues

**Error: "No PDV input files found"**
- Check that `--input-dir` exists and contains files matching `--input-pattern`
- Or use `--input-files` to specify files explicitly

**Error: "Failed to load config"**
- Verify config file paths are correct
- Check that config files are valid JSON

**Error: "ModuleNotFoundError"**
- Install missing dependencies: `pip install pandas numpy matplotlib scipy PyQt5 openpyxl`

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
@software{helix_toolbox_2025,
  title={HELIX Toolbox: A Comprehensive GUI for Single Point PDV Data Analysis},
  author={Wanchoo, Piyush},
  year={2025},
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
