# Changelog

All notable changes to HELIX Toolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Version 1.0.0
- Initial release with full ALPSS and SPADE integration
- Complete GUI with all analysis modes
- Comprehensive documentation and user guides 