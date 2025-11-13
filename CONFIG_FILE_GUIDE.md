# Configuration File Guide

## Overview

The HELIX Toolbox now supports loading analysis parameters from configuration files (JSON format). This feature allows you to:

**📁 Default Config Files Available:**
- `alpss_config_default.json` - Standard ALPSS parameters for typical PDV analysis
- `spade_config_default.json` - Standard SPADE parameters for velocity shots analysis

These files are included in the HELIX_Toolbox_v_2 directory and can be used as:
- Starting templates for your own configurations
- Quick-start configs for standard analysis
- Reference examples for parameter formatting

## Benefits

- **Save time**: Reuse the same parameter sets across multiple analysis sessions
- **Ensure consistency**: Use identical parameters for reproducible results
- **Easy sharing**: Share configuration files with collaborators
- **Version control**: Track parameter changes over time

## Using Configuration Files

### 1. Parameter Entry Modes

Both **ALPSS Parameters** and **SPADE Parameters** tabs have two modes:

- **Manual Entry (use GUI controls)**: Enter parameters directly in the GUI (default)
- **Use Config File**: Load parameters from a JSON configuration file

### 2. Saving Current Settings to Config File

To save your current GUI settings:

1. Configure all parameters in the GUI as desired
2. Click **"Save Current Settings to Config"** button
3. Choose a filename (e.g., `alpss_config.json` or `spade_config.json`)
4. Click **Save**

Your current settings are now saved and can be reused later!

### 3. Loading Parameters from Config File

To use a saved configuration:

1. Select **"Use Config File"** radio button
2. Click **"Browse"** to select your config file
3. Click **"Load Config"** to load the parameters into the GUI

The GUI will update to show all loaded parameters, allowing you to verify them before running analysis.

### 4. Running Analysis with Config Files

When you run analysis:

- If **"Manual Entry"** is selected: Uses current GUI values
- If **"Use Config File"** is selected: Uses parameters from the config file
  - The config file path must be valid
  - If missing, you'll get an error message

**Note**: You can mix modes! For example:
- Use ALPSS config file + Manual SPADE parameters
- Or vice versa

### 5. Overriding Config File Parameters

If you need to modify a single parameter from a config file:

1. Load the config file (parameters populate the GUI)
2. Switch back to **"Manual Entry"** mode
3. Modify the specific parameters you want to change
4. Run analysis (uses modified GUI values)
5. Optionally save as a new config file

## Config File Format

Configuration files are saved in JSON format. Example structure:

### ALPSS Config File (`alpss_config.json`)

```json
{
    "save_data": "yes",
    "display_plots": "no",
    "save_all_plots": "no",
    "spall_calculation": "yes",
    "header_lines": 5,
    "start_time_user": "none",
    "start_time_correction": 0.0,
    "time_to_skip": 0.0,
    "time_to_take": 1e-05,
    "t_before": 5e-07,
    "t_after": 1e-06,
    "use_notch_filter": true,
    "carrier_freq": 1500000000.0,
    "bandwidth_notch": 50000000.0,
    "smoothing_window_size": 51,
    "noise_window_duration": 1e-07,
    "noise_threshold_multiplier": 3.0,
    "use_advanced_noise_model": true,
    "lambda_laser": 1.55e-06,
    "velocity_per_fringe": 775.0,
    "save_voltage_csv": false,
    "save_velocity_csv": false,
    "save_velocity_smooth_csv": false,
    "save_velocity_uncert_csv": false,
    "save_velocity_smooth_uncert_csv": true,
    "save_results_csv": false,
    "save_noise_csv": true
}
```

### SPADE Config File (`spade_config.json`)

```json
{
    "experiment_velocity_shots": true,
    "experiment_spall_analysis": false,
    "experiment_hel_detection": false,
    "density": 8960.0,
    "acoustic_velocity": 3950.0,
    "impact_velocity_window_start": 250.0,
    "impact_velocity_window_end": 300.0,
    "align_velocity_threshold_ms": 30.0,
    "smoothing_method": "savgol",
    "smoothing_window_length": 51,
    "derivative_smoothing_window_length": 101,
    "pullback_threshold_fraction": 0.05,
    "min_pullback_velocity_ms": 5.0,
    "uncertainty_threshold_ms": 50.0,
    "include_uncert_bands": true,
    "auto_calculate_limits": false,
    "x_min_main": -40.0,
    "x_max_main": 500.0,
    "y_min_main": 0.0,
    "y_max_main": 1000.0,
    "x_min_zoom": -10.0,
    "x_max_zoom": 300.0,
    "y_min_zoom": 0.0,
    "y_max_zoom": 1000.0,
    "hel_time_window_start_ns": 0.0,
    "hel_time_window_end_ns": 100.0,
    "hel_derivative_threshold": 1.0,
    "hel_smoothing_window": 11
}
```

## Best Practices

### 1. **Naming Convention**
Use descriptive filenames:
- `alpss_copper_1550nm.json` - For copper samples with 1550nm laser
- `spade_aluminum_highrate.json` - For high strain rate aluminum spall analysis
- `alpss_default.json` - Your standard/default configuration

### 2. **Version Control**
- Store config files in a `configs/` directory
- Use git to track changes to config files
- Document what each config file is for

### 3. **Validation**
- Always load and review parameters before running analysis
- Test new config files on a small dataset first
- Keep backup copies of working configurations

### 4. **Parameter Documentation**
Consider adding a comment file alongside your config:
```
alpss_copper_config.json       <- The config file
alpss_copper_config_notes.txt  <- Notes about when/why to use this config
```

## Troubleshooting

### Config File Won't Load
- **Check file path**: Ensure the file exists and path is correct
- **Check JSON syntax**: Use a JSON validator if you manually edited the file
- **Check parameter names**: Must match exactly (case-sensitive)

### Parameters Don't Apply
- **Verify mode**: Ensure "Use Config File" is selected
- **Check load confirmation**: You should see a success message after loading
- **Review values**: Switch to Manual mode to see what was loaded

### Mixed Parameters
If you need some params from config and some manual:
1. Load config file
2. Switch to Manual mode
3. Modify specific parameters
4. Run analysis

## Example Workflow

### Setting Up Standard Configurations

```bash
# 1. Configure ALPSS parameters for copper samples in GUI
# 2. Click "Save Current Settings to Config"
# 3. Save as: configs/alpss_copper_standard.json

# 4. Configure SPADE parameters for velocity shots in GUI
# 5. Click "Save Current Settings to Config"
# 6. Save as: configs/spade_velocity_shots.json
```

### Running Batch Analysis with Config Files

```bash
# For each analysis session:
# 1. Select "Use Config File" for ALPSS
# 2. Browse and load: configs/alpss_copper_standard.json
# 3. Select "Use Config File" for SPADE
# 4. Browse and load: configs/spade_velocity_shots.json
# 5. Select input files and run analysis
```

## Support

For questions or issues with configuration files:
- Check the main README.md for general usage
- Review CSV_FILE_FORMATS.md for output file specifications
- Contact the developer: [@Piyushjhu](https://github.com/Piyushjhu)

