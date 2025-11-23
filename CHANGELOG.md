# Changelog

All notable changes to HELIX Toolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-22

### Added
- **Command-Line Interface (CLI)**: New `helix_cli_runner.py` for batch processing without GUI
- **Master Configuration File**: Single `helix_master_config.json` for all settings
- **Post-Processing Mode**: Generate plots from existing analysis results without rerunning full analysis
- **HEL (Hugoniot Elastic Limit) Detection**: Gradient-based method with configurable parameters
  - Elastic shock strain rate calculation
  - Minimum HEL velocity constraint
  - Configurable consecutive points threshold
  - Individual HEL detection plots with strain rate slope marking
- **MAD (Median Absolute Deviation) Filter**: Statistical outlier removal for peak velocities
  - Applied per material and laser energy bracket
  - Asymmetric MAD calculation
  - Configurable threshold
- **Enhanced Plotting**: Multiple new analysis plots
  - Shock stress vs laser energy (by material)
  - Shock stress vs waveplate angle (by material)
  - Shock stress vs peak velocity (by material)
  - HEL vs peak velocity (by material)
  - HEL vs HEL strain rate (by material)
  - Row/column peak velocity heatmap
  - Row/column pair scatter plots
  - Peak velocity pattern analysis
  - Laser energy vs waveplate angle
- **Material Properties Enhancement**: Added `C_L` (longitudinal wave velocity) to material properties
- **IQ Detection Filtering**: Filter traces with poor IQ detection start times
- **Trace Alignment Improvements**: Enhanced threshold crossing detection and alignment
- **Diagnostic Tools**: `diagnose_high_laser_energy.py` for identifying data anomalies

### Changed
- **Shock Stress Calculation**: Updated to use Hugoniot EOS (`U = c + S*u_p`, `σ = ρ * U * u_p`)
- **Particle Velocity**: Corrected to use `u_p = u_fs / 2` in shock stress calculations
- **HEL Detection**: Removed peak-valley fallback method, now uses gradient-based method only
- **Configuration Structure**: Consolidated into master config file format
- **Plot Styling**: Improved line widths, legend formatting, and color consistency across plots

### Fixed
- Fixed trace alignment to require crossing from below threshold
- Fixed IQ detection failure filtering
- Fixed spall analysis running when disabled in config
- Fixed post-processing mode error handling
- Fixed heatmap plot aspect ratio and white space issues
- Fixed material color consistency across all plots

### Improved
- Enhanced error handling and user feedback
- Improved trace counting and rejection reporting
- Better summary output at end of CLI runs
- Optimized plot generation and layout
- Streamlined documentation (consolidated into single README)

### Technical Details
- Added `hel_detection_min_points` config parameter (default: 3)
- Added `minimum_HEL_velocity_expected` config parameter (default: 15.0 m/s)
- Added `mad_filter_enabled` and `mad_filter_threshold` config parameters
- Added `skip_unknown_material_traces` config parameter
- Post-processing config with selective plot generation
- Improved material property retrieval with fallback options

---

## [1.0.0] - 2025-01-XX

### Added
- Initial release of HELIX Toolbox
- Comprehensive GUI for single point PDV data analysis
- Integration of ALPSS and SPADE packages
- Three analysis modes: ALPSS Only, SPADE Only, and Combined
- Optional Gaussian notch filter for carrier frequency removal
- Complete uncertainty propagation throughout analysis pipeline
- Batch processing capabilities for multiple files
- Real-time progress monitoring
- Dark/light theme support
- Comprehensive parameter configuration options
- Rich output generation including plots and summary tables

### Features
- **ALPSS Integration**: Raw PDV signal processing to velocity traces
- **SPADE Integration**: Spall strength and strain rate analysis
- **Gaussian Notch Filter**: Optional carrier frequency removal with user control
- **Uncertainty Analysis**: Complete error propagation from velocity to spall strength
- **Peak Detection**: Automated feature detection with configurable parameters
- **Material Properties**: Support for various materials with customizable properties
- **Output Formats**: CSV data files, PNG plots, and enhanced summary tables

### Technical Details
- Built with PyQt5 for cross-platform compatibility
- Scientific notation support for high-precision parameters
- Parameter validation and constraint enforcement
- Modular architecture for easy maintenance and extension
- Comprehensive error handling and user feedback

### Credits
- **ALPSS**: Original package by Jake Diamond (@Jake-Diamond-9)
- **SPADE**: Spall analysis toolkit by Piyush Wanchoo (@Piyushjhu)
- **HELIX Toolbox**: Integration and GUI by Piyush Wanchoo (@Piyushjhu)

---

## Version History

### Version 2.0.0
- Major feature release with CLI, HEL detection, MAD filtering, and enhanced plotting
- Comprehensive post-processing capabilities
- Improved configuration management

### Version 1.0.0
- Initial release with full ALPSS and SPADE integration
- Complete GUI with all analysis modes
- Comprehensive documentation and user guides
