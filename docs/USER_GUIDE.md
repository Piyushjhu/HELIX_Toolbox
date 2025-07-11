# User Guide

## Getting Started

### Launching HELIX Toolbox

1. **Start the application**
   ```bash
   python alpss_spade_gui.py
   ```

2. **The GUI will open with 6 main tabs:**
   - File Selection
   - Analysis Mode
   - ALPSS Parameters
   - SPADE Parameters
   - Control & Progress
   - Documentation

## Step-by-Step Workflow

### Step 1: File Selection

1. **Choose Input Mode**
   - **Single File**: Process one PDV data file
   - **Multiple Files**: Process all files in a directory

2. **Select Input Files**
   - Click "Browse" to select files or directory
   - For multiple files, set file pattern (default: `*.csv`)

3. **Set Output Directory**
   - Choose where to save results
   - Results will be organized in subdirectories

### Step 2: Analysis Mode

Choose your analysis approach:

- **ALPSS Only**: Process raw PDV data to velocity traces
- **SPADE Only**: Analyze existing velocity files
- **Combined**: Full pipeline from raw data to spall analysis

### Step 3: Parameter Configuration

#### ALPSS Parameters

**Basic Parameters:**
- **Save Data**: Choose whether to save output files
- **Display Plots**: Show plots during processing
- **Spall Calculation**: Enable spall analysis in ALPSS

**Time Parameters:**
- **Time to Skip**: Initial time to skip in data
- **Time to Take**: Duration of data to analyze
- **t_before/t_after**: Time around signal start

**Filter Parameters:**
- **Gaussian Notch Filter**: Enable/disable carrier frequency removal
- **Order**: Filter order (recommended: 6)
- **Width**: Filter width (recommended: 1e8)

**Peak Detection:**
- **PB Neighbors**: Must be ≥ 1 (pullback detection)
- **RC Neighbors**: Must be ≥ 1 (recompression detection)

#### SPADE Parameters

**Material Properties:**
- **Density**: Material density in kg/m³
- **Acoustic Velocity**: Sound speed in m/s

**Analysis Model:**
- **hybrid_5_segment**: Advanced 5-segment analysis
- **max_min**: Simple maximum/minimum analysis

### Step 4: Run Analysis

1. **Click "Run Analysis"**
2. **Monitor Progress**: Watch real-time progress updates
3. **View Results**: Check output directory for results

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

## Advanced Features

### Gaussian Notch Filter

**When to Enable:**
- Strong carrier signal masks Doppler-shifted signal
- Clear frequency separation between carrier and signal

**When to Disable:**
- Weak signal relative to noise
- Carrier and signal frequencies are close together

**Effects:**
- Removes carrier frequency
- May introduce ringing or phase distortion if misused

### Uncertainty Analysis

The toolbox provides comprehensive uncertainty analysis:

- **Velocity Uncertainty**: Propagated through all calculations
- **Spall Strength Uncertainty**: Includes material property uncertainties
- **Strain Rate Uncertainty**: Based on time and velocity uncertainties
- **Shock Stress Uncertainty**: Derived from peak velocity uncertainty

### Batch Processing

For multiple files:

1. **Select Directory**: Choose folder containing all PDV files
2. **Set Pattern**: Use file pattern to match specific files
3. **Run Analysis**: Process all files automatically
4. **Combined Results**: Get summary plots and tables

## Troubleshooting

### Common Issues

**GUI Not Starting:**
- Check PyQt5 installation
- Verify Python version (≥3.7)

**No Output Files:**
- Check output directory permissions
- Verify input file format (CSV)

**Analysis Fails:**
- Check parameter values
- Verify input data quality
- Review error messages in progress window

**Slow Performance:**
- Reduce batch size for multiple files
- Close other applications
- Check available memory

### Error Messages

**"No peaks found in smoothed signal"**
- Adjust prominence factor
- Check signal quality
- Verify smoothing parameters

**"Not enough data for smoothing"**
- Increase time window
- Check data length
- Verify time parameters

**"File not found"**
- Check file paths
- Verify file permissions
- Ensure files exist

## Tips and Best Practices

### Data Preparation

1. **File Format**: Use CSV format with time in first column, voltage in second
2. **Data Quality**: Ensure clean signals with minimal noise
3. **Time Units**: Verify time is in seconds or nanoseconds
4. **Header Lines**: Set correct number of header lines to skip

### Parameter Selection

1. **Start with Defaults**: Use recommended parameter values
2. **Adjust Gradually**: Make small changes and test results
3. **Check Plots**: Always review generated plots for quality
4. **Document Changes**: Keep notes of parameter modifications

### Workflow Optimization

1. **Test on Single File**: Verify settings before batch processing
2. **Use Virtual Environment**: Isolate dependencies
3. **Backup Data**: Keep original files safe
4. **Organize Outputs**: Use descriptive output directory names

## Support

For additional help:

1. **Check Documentation**: Review this guide and README
2. **Search Issues**: Look for similar problems on GitHub
3. **Create Issue**: Report bugs with detailed information
4. **Contact Author**: Reach out for specific questions 