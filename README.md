# HELIX Toolbox

**A Comprehensive GUI for Single Point PDV Data Analysis**

**Author:** Piyush Wanchoo  
**GitHub:** [@Piyushjhu](https://github.com/Piyushjhu)  
**Institution:** Johns Hopkins University  
**Year:** 2025  

## Overview

HELIX Toolbox is a comprehensive graphical user interface (GUI) that combines ALPSS (Automated Laser Photonic Doppler Velocimetry Signal Processing) and SPADE (Spall Analysis Toolkit) for single point PDV (Photonic Doppler Velocimetry) data analysis. This tool provides an integrated workflow from raw PDV signals to complete spall strength analysis with uncertainty quantification.

## Features

### 🔬 **Single Point PDV Analysis**
- Process raw PDV signals from single point measurements
- Automated carrier frequency removal with optional Gaussian notch filter
- Velocity extraction with uncertainty quantification
- Real-time signal processing and visualization

### 📊 **Comprehensive Analysis Pipeline**
- **ALPSS Integration**: Raw signal processing to velocity traces
- **SPADE Integration**: Spall strength and strain rate analysis
- **Combined Mode**: Full pipeline from raw data to spall analysis
- **Individual Modes**: Run ALPSS or SPADE independently

### 🎛️ **Advanced Processing Options**
- **Gaussian Notch Filter**: Optional carrier frequency removal
- **Smoothing Parameters**: Configurable signal smoothing
- **Peak Detection**: Automated feature detection with user controls
- **Uncertainty Propagation**: Complete error analysis throughout pipeline

### 📈 **Rich Output Generation**
- Velocity traces with uncertainty bands
- Spall strength vs. strain rate plots
- Spall strength vs. shock stress analysis
- Enhanced summary tables with all uncertainties
- Individual and combined analysis plots

### 🖥️ **Cross-Platform Compatibility**
- **Windows**: Native Windows GUI with Explorer integration
- **macOS**: Optimized for macOS with native file dialogs
- **Linux**: Full Linux support with X11 integration
- **Unified Interface**: Same features across all platforms

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

### Platform-Specific Installation

- **[Windows Installation Guide](supplementary/docs_archive/docs/WINDOWS_INSTALLATION.md)** - Detailed Windows setup and troubleshooting
- **macOS**: Install Xcode Command Line Tools if needed
- **Linux**: Install system dependencies: `sudo apt-get install python3-dev python3-pip`

## Environment Setup (recommended)

Create an isolated Python environment to avoid dependency conflicts.

### Option A: Python venv (recommended)

#### Windows (PowerShell or CMD)
```cmd
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS/Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Deactivate with:
```bash
deactivate
```

### Option B: Conda
```bash
conda create -n helix_toolbox python=3.10 -y
conda activate helix_toolbox
pip install -r requirements.txt
```

### System packages (if needed)
- macOS: `brew install qt5`
- Ubuntu/Debian: `sudo apt-get install -y libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxtst6 libxi6`

Headless environments may require:
```bash
export QT_QPA_PLATFORM=offscreen
```

## Usage

### 1. **File Selection**
- Choose single file or batch processing mode
- Select input PDV data files (CSV format)
- Set output directory for results

### 2. **Analysis Mode**
- **ALPSS Only**: Process raw PDV data to velocity traces
- **SPADE Only**: Analyze existing velocity files
- **Combined**: Full pipeline from raw data to spall analysis

### 3. **Parameter Configuration**
- **ALPSS Parameters**: Signal processing, filtering, and smoothing options
- **SPADE Parameters**: Material properties and analysis models
- **Advanced Options**: Gaussian notch filter, uncertainty multipliers

### 4. **Run Analysis**
- Monitor real-time progress
- View generated plots and results
- Access comprehensive output files

## Physical Parameter Calculations

### 1. Free Surface Velocity Extraction

**Method**: Phase unwrapping and differentiation of PDV signal

The free surface velocity is extracted from the PDV (Photonic Doppler Velocimetry) signal using the fundamental relationship:

```
v(t) = (λ/2) × f_Doppler(t)
```

Where:
- `v(t)` = free surface velocity (m/s)
- `λ` = laser wavelength (typically 1550 nm)
- `f_Doppler(t)` = instantaneous Doppler shift frequency (Hz)

**Process**:
1. **Signal Demodulation**: Extract In-phase (I) and Quadrature (Q) components via IQ analysis
2. **Phase Calculation**: `φ(t) = arctan2(Q, I)`
3. **Phase Unwrapping**: Remove 2π discontinuities
4. **Frequency Extraction**: `f(t) = (1/2π) × dφ/dt`
5. **Velocity Conversion**: `v(t) = (λ/2) × f(t)`
6. **Smoothing**: Apply Gaussian window for noise reduction

**Implementation**: `velocity_calculation()` in `ALPSS/alpss_main.py`

---

### 2. Velocity Uncertainty Calculation

**Method**: Instantaneous noise analysis with time-frequency uncertainty principle

The velocity uncertainty accounts for signal-to-noise ratio and temporal resolution:

```
Δv(t) = (λ/2) × Δf(t)
```

Where the frequency uncertainty is:

```
Δf(t) = η(t) × (1/π) × √[6 / (f_s × τ³)]
```

**Parameters**:
- `η(t)` = instantaneous noise fraction = `std(noise) / [A(t)/2]`
- `A(t)` = instantaneous signal amplitude (from envelope detection)
- `f_s` = sampling frequency (Hz)
- `τ` = characteristic time = FWHM of Gaussian smoothing window (s)

**Process**:
1. **Noise Estimation**: Fit sinusoid to pre-event signal, calculate residuals
2. **Envelope Detection**: Extract upper and lower signal envelopes
3. **Instantaneous Amplitude**: `A(t)` = envelope_max - envelope_min
4. **Noise Fraction**: `η(t) = std(noise) / [A(t)/2]`
5. **Characteristic Time**: Calculate FWHM of smoothing window
6. **Frequency Uncertainty**: Apply uncertainty formula
7. **Velocity Uncertainty**: Convert using `Δv = (λ/2) × Δf`

**Reference**: [Fratanduono et al., Review of Scientific Instruments 91, 051501 (2020)](https://doi.org/10.1063/12.0000870)

**Implementation**: `instantaneous_uncertainty_analysis()` in `ALPSS/alpss_main.py`

---

### 3. Spall Strength Calculation

**Method**: Acoustic approximation from pullback velocity

Spall strength is calculated from the velocity pullback using the acoustic approximation:

```
σ_spall = (1/2) × ρ₀ × c_b × Δv_pullback
```

Where:
- `σ_spall` = spall strength (GPa)
- `ρ₀` = initial material density (kg/m³)
- `c_b` = bulk sound speed (m/s)
- `Δv_pullback` = velocity pullback magnitude = |v_peak - v_min| (m/s)

**Process**:
1. **Peak Detection**: Find maximum velocity (peak) after shock arrival
2. **Minimum Detection**: Find minimum velocity (valley) after peak
3. **Pullback Calculation**: `Δv = |v_peak - v_min|`
4. **Material Properties**: Get ρ₀ and c_b from database or user input
5. **Spall Strength**: Apply formula, convert Pa → GPa (÷10⁹)

**Uncertainty Propagation**:
```
Δσ_spall = (1/2) × ρ₀ × c_b × √(Δv²_peak + Δv²_min)
```

**Implementation**: `calculate_spall_strength()` in `SPADE/spall_analysis_release/spall_analysis/data_processing.py`

---

### 4. Strain Rate Calculation

**Method**: Time derivative of velocity during pullback

The strain rate during spalling is estimated from the rate of velocity change:

```
ε̇ = (1/c_b) × |dv/dt|_pullback
```

Where:
- `ε̇` = strain rate (s⁻¹)
- `c_b` = bulk sound speed (m/s)
- `|dv/dt|` = velocity change rate during pullback (m/s²)

**Process**:
1. **Identify Pullback Region**: Time between peak and minimum velocity
2. **Linear Fit**: Fit line to velocity vs. time in pullback region
3. **Velocity Rate**: Extract slope `dv/dt`
4. **Strain Rate**: `ε̇ = |dv/dt| / c_b`

**Alternative Method** (if linear fit fails):
```
ε̇ ≈ Δv_pullback / (c_b × Δt_pullback)
```

**Implementation**: `calculate_strain_rate()` in SPADE data processing module

---

### 5. HEL (Hugoniot Elastic Limit) Calculation

**Method**: Peak-valley analysis with material properties

The HEL strength is determined from the elastic wave amplitude:

```
σ_HEL = (1/2) × ρ₀ × c_b × (v_peak - v_valley)
```

Where:
- `σ_HEL` = Hugoniot Elastic Limit (GPa)
- `v_peak` = first peak velocity in elastic wave (m/s)
- `v_valley` = first valley velocity after peak (m/s)
- `ρ₀` = material density (kg/m³)
- `c_b` = bulk sound speed (m/s)

**Process**:
1. **Elastic Wave Detection**: Identify oscillations in early-time velocity
2. **Peak Finding**: Detect first maximum with prominence threshold
3. **Valley Finding**: Detect first minimum after peak
4. **Material Lookup**: Get ρ₀ and c_b from `material_properties.py` database
5. **HEL Calculation**: Apply formula, convert to GPa

**Uncertainty Propagation**:
```
Δσ_HEL = (1/2) × ρ₀ × c_b × √(Δv²_peak + Δv²_valley)
```

**Implementation**: `generate_velocity_shots_summary()` in `helix_analysis_toolbox.py` (lines ~810-840)

---

### 6. Shock Stress Calculation

**Method**: Impedance matching with flyer impact velocity

Shock stress is calculated from the impact conditions:

```
σ_shock = (1/2) × ρ₀ × c_b × v_impact
```

Or using the measured free surface velocity:

```
σ_shock = (1/2) × ρ₀ × c_b × (2 × v_fs)
```

Where:
- `σ_shock` = shock stress (GPa)
- `v_impact` = flyer impact velocity (m/s)
- `v_fs` = free surface velocity (m/s)
- Factor of 2 accounts for free surface approximation

**Implementation**: User-provided impact velocity or extracted from peak velocity

---

### 7. Peak Velocity and Time Parameters

**Peak Velocity** (`v_peak`):
- Maximum velocity in the velocity trace
- Detected using `scipy.signal.find_peaks()` with prominence threshold
- Represents the maximum particle velocity reached during loading

**Pullback Velocity** (`v_min`):
- Minimum velocity after the peak
- Indicates onset of tension/spall damage
- Used for spall strength calculation

**Recompression Velocity** (`v_rc`):
- Velocity increase after minimum (if present)
- Indicates shock wave reflection and recompression
- Used for damage evolution analysis

**Time Parameters**:
- `t_10%`: Time when velocity reaches 10% of peak
- `t_peak`: Time of maximum velocity
- `t_min`: Time of minimum velocity (pullback)
- `t_rc`: Time of recompression (if detected)
- `Δt_pullback`: Duration of pullback = `t_min - t_peak`

**Implementation**: `spall_analysis()` and peak detection routines in ALPSS

---

### 8. Material Properties Database

**Source**: `material_properties.py`

The toolbox includes a comprehensive database of material properties:

**Properties Stored**:
- `density` (ρ₀): Initial density (kg/m³)
- `bulk_wave_speed` (c_b): Longitudinal sound speed (m/s)

**Supported Materials** (48 materials):
- **Metals**: Cu, Al, Fe, Ti, Ni, Ta, W, Au, Ag, Pb, Mg, Zn
- **Polymers**: PMMA, Polycarbonate, Teflon, Polyethylene
- **Ceramics**: Sapphire, Silicon, SiC, Glass, Fused Silica
- **Others**: Diamond, Graphite, Water

**Usage**:
```python
from material_properties import get_material_properties
props = get_material_properties('Copper')
# Returns: {'density': 8960, 'bulk_wave_speed': 3940, 'material_found': True}
```

**Fallback**: If material not found, uses Copper properties as default or user-specified values

---

### 9. Noise Fraction

**Method**: Ratio of noise to signal amplitude

```
η(t) = std(noise) / [A(t)/2]
```

Where:
- `noise` = residuals from sinusoidal fit to pre-event signal
- `A(t)` = instantaneous signal amplitude

**Purpose**: 
- Quantifies signal quality at each time point
- Used in velocity uncertainty calculation
- Helps identify regions of poor signal quality

**Output**: Saved in `*--noise--frac.csv`

---

### Summary Table of Calculations

| Parameter | Formula | Units | Uncertainty Method |
|-----------|---------|-------|-------------------|
| Velocity | v = (λ/2) × f | m/s | Time-frequency uncertainty |
| Spall Strength | σ = (1/2) × ρ × c × Δv | GPa | Propagate velocity uncertainties |
| Strain Rate | ε̇ = \|dv/dt\|/c | s⁻¹ | Linear fit residuals |
| HEL | σ_HEL = (1/2) × ρ × c × Δv_elastic | GPa | Peak-valley uncertainties |
| Shock Stress | σ = (1/2) × ρ × c × v_impact | GPa | Impact velocity uncertainty |
| Noise Fraction | η = std(noise)/(A/2) | - | Statistical (std) |

**References**:
- ALPSS methodology: Diamond et al. (methodology paper if available)
- Uncertainty quantification: [Fratanduono et al., RSI 91, 051501 (2020)](https://doi.org/10.1063/12.0000870)
- Spall strength theory: Antoun et al., "Spall Fracture" (2003)

---

## Output Files

### ALPSS Outputs
- `*--velocity.csv`: Raw velocity data (Time_s, Velocity_m_s)
- `*--velocity--smooth.csv`: Smoothed velocity data (Time_s, Velocity_Smooth_m_s)
- `*--vel--uncert.csv`: Velocity uncertainty data (Time_s, Velocity_Uncertainty_m_s)
- `*--vel-smooth-with-uncert.csv`: Smoothed velocity with uncertainty (Time_s, Velocity_Smooth_m_s, Velocity_Uncertainty_m_s, Velocity_Plus_Uncertainty_m_s)
- `*--noise--frac.csv`: Noise fraction data (Time_s, Noise_Fraction)
- `*--voltage.csv`: Filtered voltage signal (Time_s, Voltage_Real_V, Voltage_Imag_V)
- `*--results.csv`: Analysis results with uncertainties
- `*--plots.png`: Individual analysis plots

**See [CSV_FILE_FORMATS.md](CSV_FILE_FORMATS.md) for detailed column descriptions**

### SPADE Outputs
- `enhanced_spall_summary.csv`: Complete results with ALPSS data (supersedes the older `spall_summary.csv`)
- `spall_vs_strain_rate.png`: Spall strength vs strain rate plot
- `spall_vs_shock_stress.png`: Spall strength vs shock stress plot
- `all_smoothed_velocity_traces.png`: Combined velocity traces

## Key Parameters

### Gaussian Notch Filter
- **Enable**: Remove carrier frequency (recommended for strong signals)
- **Disable**: When signal is weak or carrier/signal frequencies are close
- **Effects**: May introduce ringing or phase distortion if misused

### Peak Detection
- **PB Neighbors**: Must be ≥ 1 (scipy requirement for pullback detection)
- **RC Neighbors**: Must be ≥ 1 (scipy requirement for recompression detection)

### Smoothing
- **ALPSS Smoothing**: Applied to raw velocity data
- **SPADE Smoothing**: Automatically skipped in combined mode (uses ALPSS smoothed data)

## Platform-Specific Features

### Windows
- **Native Explorer Integration**: "Open Output Directory" opens Windows Explorer
- **Segoe UI Font**: Native Windows styling
- **Batch File Launcher**: Easy one-click startup
- **High DPI Support**: Optimized for modern displays

### macOS
- **Native Finder Integration**: File dialogs use macOS Finder
- **Dark Mode Support**: Automatic theme switching
- **Retina Display**: High-resolution graphics support

### Linux
- **X11 Integration**: Native Linux desktop integration
- **Package Manager Support**: Easy installation via pip
- **Terminal Friendly**: Full command-line interface

## Credits

### ALPSS (Automated Laser Photonic Doppler Velocimetry Signal Processing)
**Author:** Jake Diamond  
**GitHub:** [@Jake-Diamond-9](https://github.com/Jake-Diamond-9)  
**Description:** Original ALPSS package for PDV signal processing and velocity extraction

### SPADE (Spall Analysis Toolkit)
**Author:** Piyush Wanchoo  
**GitHub:** [@Piyushjhu](https://github.com/Piyushjhu)  
**Description:** Spall strength and strain rate analysis toolkit

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests, please:
1. Check the [Documentation](docs/) folder
2. Search existing [Issues](https://github.com/Piyushjhu/HELIX_Toolbox/issues)
3. Create a new issue with detailed information

## Acknowledgments

- **Jake Diamond** for the original ALPSS package
- **Johns Hopkins University** for research support
- The scientific community for PDV and spall analysis methodology

---

**HELIX Toolbox** - Advancing single point PDV data analysis for shock physics research across all platforms. 🖥️💻📱 