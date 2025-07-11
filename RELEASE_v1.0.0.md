# HELIX Toolbox v1.0.0 Release

## Release Date: 2025-07-11

### 🎉 Major Features

#### Combined ALPSS + SPADE Analysis Pipeline
- **Unified GUI**: Modern PyQt5 interface with dark/light themes
- **Three Analysis Modes**: ALPSS-only, SPADE-only, or combined pipeline
- **Batch Processing**: Support for single files or multiple files with pattern matching
- **Real-time Progress**: Dual progress bars for ALPSS and SPADE processing

#### Enhanced ALPSS Features
- **Optional Gaussian Notch Filter**: Toggle carrier frequency removal
- **Improved Uncertainty Handling**: 4-column velocity output with uncertainty
- **Scientific Notation Support**: High-precision wavelength input
- **Enhanced Plotting**: All uncertainty plots show multiplier information
- **Parameter Validation**: Proper constraints for peak detection parameters

#### Enhanced SPADE Features
- **Uncertainty Propagation**: Full uncertainty calculations for all outputs
- **Peak Shock Stress**: Calculation with uncertainty from velocity data
- **Enhanced Summary**: Complete results combining ALPSS and SPADE data
- **Skip Smoothing Option**: Avoid double smoothing when using ALPSS outputs
- **Multiple Analysis Models**: Support for hybrid_5_segment and max_min models

#### Advanced Outputs
- **Combined Velocity Traces**: Aligned and uncertainty-shaded plots
- **Spall Strength Analysis**: vs. strain rate and shock stress plots
- **Enhanced Summary CSV**: Complete results with all uncertainties
- **Individual Analysis Plots**: Detailed analysis for each file
- **Literature Comparison**: Built-in literature data for comparison

### 🔧 Technical Improvements

#### Code Quality
- **Modular Architecture**: Clean separation between ALPSS and SPADE
- **Error Handling**: Comprehensive error handling and user feedback
- **Documentation**: Extensive inline documentation and user guides
- **Testing**: Automated test suite for all major functions

#### User Experience
- **Modern GUI**: Large fonts, better spacing, responsive design
- **Input Validation**: Real-time validation with helpful error messages
- **Progress Tracking**: Detailed progress updates during analysis
- **Output Organization**: Structured output directories with clear naming

#### Performance
- **Optimized Processing**: Efficient batch processing capabilities
- **Memory Management**: Proper cleanup and resource management
- **Parallel Processing**: Threaded analysis to prevent GUI freezing

### 📁 File Structure

```
HELIX_Toolbox/
├── ALPSS/                    # ALPSS analysis package
│   ├── alpss_main.py        # Main ALPSS processing
│   ├── alpss_auto_run.py    # Automated ALPSS runner
│   └── requirements.txt     # ALPSS dependencies
├── SPADE/                    # SPADE analysis package
│   └── spall_analysis_release/
│       ├── spall_analysis/  # SPADE analysis modules
│       └── requirements.txt # SPADE dependencies
├── alpss_spade_gui.py       # Main GUI application
├── setup.py                 # Package installation
├── requirements.txt         # Main dependencies
├── README.md               # Comprehensive documentation
└── docs/                   # User guides and documentation
```

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox

# Install dependencies
pip install -r requirements.txt

# Run the GUI
python alpss_spade_gui.py
```

### 📋 System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for installation

### 🎯 Key Features Summary

1. **Unified Interface**: Single GUI for both ALPSS and SPADE analysis
2. **Flexible Input**: Support for single files, multiple files, or directories
3. **Advanced Processing**: Optional notch filtering, uncertainty propagation
4. **Comprehensive Outputs**: Multiple plot types, summary tables, enhanced results
5. **User-Friendly**: Modern interface with themes, validation, and progress tracking
6. **Well-Documented**: Extensive documentation and user guides

### 🔬 Scientific Applications

- **Spallation Experiments**: Complete analysis from raw PDV data to spall strength
- **Material Characterization**: Velocity and stress analysis for material properties
- **Research Workflows**: Streamlined processing for high-throughput experiments
- **Data Validation**: Uncertainty quantification and literature comparison

### 👥 Credits

- **Author**: Piyush Wanchoo
- **ALPSS Credits**: Jake Diamond (original ALPSS development)
- **SPADE Credits**: SPADE development team
- **GUI Development**: Enhanced PyQt5 interface with modern design

### 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/Piyushjhu/HELIX_Toolbox/issues
- **Documentation**: See README.md and docs/ directory
- **Examples**: Check the examples/ directory for usage examples

---

**HELIX Toolbox v1.0.0** - A comprehensive solution for spallation experiment analysis combining ALPSS and SPADE capabilities in a modern, user-friendly interface.
