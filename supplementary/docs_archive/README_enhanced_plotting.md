# Enhanced Plotting Module

## Overview

The `enhanced_plotting.py` module provides standalone enhanced plotting capabilities for ALPSS-SPADE velocity analysis data. This module can be run independently of the main SPADE workflow and offers 6 different figure types with customizable options.

## Features

- **6 Different Figure Types**: Individual file legends, color meaning legends, spread analysis, velocity vs waveplate angle, shot time vs material, and PDV power vs material
- **Noise Filtering**: Automatically removes data points where noise fraction > 1.0
- **Trace Alignment**: Aligns time data to t=0 when velocity reaches 30 m/s threshold
- **Material and Waveplate Angle Color Coding**: Consistent color schemes across all plots
- **Spread Analysis**: Statistical analysis with min/max bounds and mean traces
- **CSV Data Export**: Comprehensive data export for further analysis
- **Parameter File Integration**: Supports Excel parameter files for enhanced legends

## Usage

### Command Line Interface

```bash
# Basic usage with all figures enabled
python enhanced_plotting.py --input_dir /path/to/velocity/files --output_dir /path/to/output

# With parameter file
python enhanced_plotting.py --input_dir /path/to/velocity/files --output_dir /path/to/output --param_file /path/to/parameters.xlsx

# Selective plotting (only specific figures)
python enhanced_plotting.py --input_dir /path/to/velocity/files --output_dir /path/to/output --plot_options Figure1 Figure4 Figure5
```

### Python API

```python
from enhanced_plotting import EnhancedPlotting

# Create plotting instance
plotting = EnhancedPlotting(
    input_dir="/path/to/velocity/files",
    output_dir="/path/to/output",
    param_file="/path/to/parameters.xlsx",  # optional
    plot_options={
        'plot_individual_legends': True,    # Figure 1
        'plot_color_meaning': True,         # Figure 2
        'plot_spread_analysis': True,       # Figure 3
        'plot_velocity_vs_angle': True,     # Figure 4
        'plot_shot_time_vs_material': True, # Figure 5
        'plot_pdv_power_vs_material': True  # Figure 6
    }
)

# Run enhanced plotting
plotting.run_enhanced_plotting()
```

## Figure Types

### Figure 1: Individual File Legends
- **File**: `all_smoothed_velocity_traces_with_legends.png`
- **Description**: Three subplots showing velocity traces with individual file legends
- **Subplots**: Material-based, waveplate angle-based, and zoomed region (0-20 ns)

### Figure 2: Color Meaning Legends
- **File**: `all_smoothed_velocity_traces_color_meaning.png`
- **Description**: Three subplots with color-coded legends only
- **Subplots**: Material-based, waveplate angle-based, and zoomed region (0-20 ns)

### Figure 3: Spread Analysis
- **File**: `all_smoothed_velocity_traces_spread.png`
- **Description**: Statistical spread analysis with min/max bounds and mean traces
- **Subplots**: Material-based and waveplate angle-based spread plots

### Figure 4: Velocity vs Waveplate Angle
- **File**: `max_velocity_vs_waveplate_angle.png`
- **Description**: Scatter plot of maximum velocity vs waveplate angle by material
- **Data**: Mean velocity between 300-400ns time window

### Figure 5: Shot Time vs Material
- **File**: `shot_time_vs_material.png`
- **Description**: Box plot of shot time vs material with outliers
- **Data**: Shot time from parameter files

### Figure 6: PDV Power vs Material
- **File**: `pdv_power_vs_material.png`
- **Description**: Scatter plot of PDV return power vs material
- **Data**: Calculated from velocity signal power

## Data Files

### Input Files
- **Velocity Files**: `*--velocity--smooth.csv` (time, velocity columns)
- **Noise Files**: `*--noise--frac.csv` (noise fraction for filtering)
- **Parameter Files**: Excel files with experiment metadata

### Output Files
- **Analysis Data**: `analysis_data.csv` (comprehensive data export)
- **Plots**: 6 PNG files (300 DPI, high quality)

## Data Processing

### Noise Filtering
- Loads `*--noise--frac.csv` files
- Filters out data points where noise fraction > 1.0
- Sets filtered points to `np.nan`

### Trace Alignment
- Finds t=0 when velocity first reaches 30 m/s threshold
- Shifts all time data by this offset
- Ensures consistent time alignment across all traces

### Velocity Calculation
- Calculates mean velocity in 300-400ns window
- Uses aligned time data and filtered velocity
- Provides fallback windows (200-300ns, 400-500ns) if primary window is empty

## Parameter File Format

The parameter file should be an Excel file with columns including:
- `exp_id`: Experiment identifier (used as key)
- `sample_material`: Material type (Al, Ti, Cu, etc.)
- `waveplate_angle`: Waveplate angle in degrees
- `shot_time`: Shot time in seconds
- Additional metadata columns

## Dependencies

- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `openpyxl`: Excel file reading (optional)

## Integration with Main Workflow

This module is designed to be independent of the main ALPSS-SPADE GUI workflow. The enhanced plotting functionality has been removed from the main GUI to:

1. **Reduce Complexity**: Simplify the main GUI workflow
2. **Improve Performance**: Avoid memory issues with large datasets
3. **Enable Flexibility**: Allow standalone execution and customization
4. **Maintain Focus**: Keep the main GUI focused on core ALPSS-SPADE analysis

## Error Handling

The module includes comprehensive error handling for:
- Missing or corrupted input files
- Parameter file loading issues
- Data processing errors
- Plot generation failures

All errors are logged with detailed messages for debugging.

## Performance Considerations

- **Memory Efficient**: Processes files one at a time
- **Progress Reporting**: Updates progress every 10 files
- **Selective Plotting**: Can enable/disable specific figures
- **High-Quality Output**: 300 DPI PNG files

## Future Enhancements

Potential improvements include:
- Additional plot types
- Interactive plotting options
- Batch processing capabilities
- Integration with other analysis tools
- Custom color schemes
- Export to additional formats (PDF, SVG) 