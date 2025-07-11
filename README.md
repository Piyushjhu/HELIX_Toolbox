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

## Installation

### Prerequisites
- Python 3.7+
- PyQt5
- Required scientific packages (see requirements.txt)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox

# Install dependencies
pip install -r requirements.txt

# Run the GUI
python alpss_spade_gui.py
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

## Output Files

### ALPSS Outputs
- `*--velocity.csv`: Raw velocity data
- `*--velocity--smooth.csv`: Smoothed velocity data
- `*--vel--uncert.csv`: Velocity uncertainty data
- `*--vel-smooth-with-uncert.csv`: Smoothed velocity with uncertainty
- `*--results.csv`: Analysis results with uncertainties
- `*--plots.png`: Individual analysis plots

### SPADE Outputs
- `spall_summary.csv`: Basic spall analysis results
- `enhanced_spall_summary.csv`: Complete results with ALPSS data
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

**HELIX Toolbox** - Advancing single point PDV data analysis for shock physics research. 