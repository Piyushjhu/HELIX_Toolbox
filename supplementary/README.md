# Supplementary Tools and Utilities

This directory contains various supplementary tools, utilities, and debugging scripts for the ALPSS-SPADE analysis pipeline. These tools are designed to help with troubleshooting, data processing, monitoring, and enhanced analysis capabilities.

## 📁 File Categories

### 🔧 **Debugging & Fix Tools**

#### `alpss_fix_script.py`
- **Purpose**: Provides fixes for identified performance and broadcasting issues in ALPSS
- **Features**: 
  - Fixes array broadcasting issues in velocity data processing
  - Optimizes analysis parameters for better performance
  - Provides safe array trimming functions
- **Usage**: Run to apply fixes to ALPSS code

 

#### `diagnose_velocity_summary.py`
- **Purpose**: Diagnoses issues with velocity summary data processing
- **Features**:
  - Tests velocity calculation functions
  - Validates data formats and structures
  - Provides debugging information for velocity summary generation
- **Usage**: Run to test and diagnose velocity summary processing

### 📊 **Monitoring & Analysis Tools**

#### `analysis_monitor.py`
- **Purpose**: Comprehensive monitoring tool for analysis failures and diagnostics
- **Features**:
  - Scans output directories for ALPSS and SPADE results
  - Checks for missing files and incomplete analyses
  - Provides detailed summaries of analysis status
  - Identifies failed files and potential issues
- **Usage**: Run to get a complete overview of analysis status

#### `realtime_monitor.py`
- **Purpose**: Real-time monitoring of analysis progress and GUI status
- **Features**:
  - Monitors file creation in real-time
  - Tracks GUI process status and resource usage
  - Provides live updates on analysis progress
  - Detects new files and changes
- **Usage**: Run for continuous monitoring during analysis

#### `analysis_performance_debug.py`
- **Purpose**: Debug performance issues in analysis pipeline
- **Features**:
  - Analyzes processing times and bottlenecks
  - Identifies memory usage patterns
  - Provides performance optimization recommendations
- **Usage**: Run to diagnose performance issues

### 📈 **Enhanced Plotting & Visualization**

#### `enhanced_plotting.py`
- **Purpose**: Advanced plotting capabilities for velocity analysis data
- **Features**:
  - 6 different figure types with customizable options
  - Noise filtering and trace alignment
  - Material and waveplate angle color coding
  - Spread analysis and statistical plots
  - CSV data export for further analysis
- **Usage**: `python enhanced_plotting.py --input_dir /path/to/velocity/files --output_dir /path/to/output`

#### `plot_all_velocity_data.py`
- **Purpose**: Creates comprehensive plots of all velocity data files
- **Features**:
  - Plots all velocity smooth data files in a single plot
  - Analyzes velocity data statistics
  - Creates velocity summaries
  - Handles uncertainty thresholds
- **Usage**: Run to generate overview plots of all velocity data

### 📋 **Data Processing & Management**

#### `velocity_summary_post_processor.py`
- **Purpose**: Post-processes velocity shots summary data
- **Features**:
  - Creates box plots of shot time vs material
  - Creates box plots of PDV Return Power vs material
  - Handles different material column names
  - Saves plots in high resolution
  - Generates statistics summary
- **Usage**: `python velocity_summary_post_processor.py --input velocity_shots_summary.csv --output output_directory`

#### `create_velocity_summary.py`
- **Purpose**: Creates velocity summary from processed data
- **Features**:
  - Aggregates velocity data from multiple files
  - Calculates statistics and summaries
  - Exports summary data for further analysis
- **Usage**: Run to create velocity summaries from processed data

### 🖥️ **GUI Integration Note**

- The standalone legacy GUI `alpss_spade_gui.py` and helper scripts for copying or fixing arrays have been removed. The GUI functionality is now integrated into `helix_analysis_toolbox.py` with:
  - Combined aligned velocity plot (t=0 at threshold) with noise and uncertainty filtering
  - User-configurable alignment threshold, uncertainty threshold, uncertainty bands, and zoom window
  - Inclusion of all parameter-file columns in both velocity and spall summaries

 

### 📚 **Documentation Archive**

#### `docs_archive/`
- **Purpose**: Contains archived documentation and release notes
- **Contents**:
  - Installation guides
  - User guides
  - Release notes
  - Troubleshooting documentation
  - Feature summaries
  - Bug fix documentation

## 🚀 **Quick Start Guide**

### For Monitoring Analysis:
```bash
python analysis_monitor.py          # Check analysis status
python realtime_monitor.py         # Monitor in real-time
```

### For Enhanced Plotting:
```bash
python enhanced_plotting.py --input_dir ./output --output_dir ./plots
```

### For Data Processing:
```bash
python velocity_summary_post_processor.py --input summary.csv --output ./results
python plot_all_velocity_data.py
```

### For Debugging:
```bash
python alpss_fix_script.py         # Apply fixes
python diagnose_velocity_summary.py # Test velocity processing
```

## 📝 **Notes**

- These tools are supplementary and not required for basic ALPSS-SPADE operation
- Some tools may require specific data formats or directory structures
- Always backup your data before running diagnostic or fix tools
- The primary GUI is `helix_analysis_toolbox.py`.

## 🔧 **Dependencies**

Most tools require:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn` (for enhanced plotting)
- `psutil` (for real-time monitoring)

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn psutil
```
