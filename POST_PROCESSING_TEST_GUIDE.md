# Post-Processing Debugging Guide

## Quick Test Procedure

### 1. Start the GUI
```bash
python3 helix_analysis_toolbox.py
```

### 2. Complete an Analysis
- Select input files
- Run ALPSS or ALPSS+SPADE analysis
- Ensure velocity traces are generated

### 3. Go to Post-Processing Tab
1. Click **Post-Processing** tab
2. **Select Output Directory** - Choose the folder from step 2
3. **Verify "Auto Calculate Limits" is UNCHECKED**
4. Set custom limits:
   - Main X: 0 to 100 ns
   - Main Y: 0 to 600 m/s
   - Zoom X: 0 to 50 ns
   - Zoom Y: 0 to 300 m/s
5. Click **Preview**

### 4. Check the Debug Output
Look for these three log sections in the Preview text area:

#### Expected Log Output:
```
[POST-PROCESSING] Parameters applied:
  auto_calc_limits: False
  x_min/max_main: 0.0/100.0
  y_min/max_main: 0.0/600.0

[WORKER] Received parameters in regenerate_plots:
  auto_calc_limits: False
  x_min/max_main: 0.0/100.0
  y_min/max_main: 0.0/600.0

Applied top limits: X(0.0-100.0), Y(0.0-600.0)
Applied zoom limits: X(0.0-50.0), Y(0.0-300.0)
```

## Interpreting Results

### ✅ All three log sections appear with correct values
- **Status**: Parameters flow correctly through the system
- **Next Check**: Look at generated plots
  - Open `SPADE_analysis/all_velocity_traces.png`
  - Verify X-axis: 0-100 ns, Y-axis: 0-600 m/s
- **If plots correct**: Bug is FIXED ✅
- **If plots wrong**: Issue is in plot generation, not parameters

### ❌ First log shows values, second log shows None
- **Status**: Parameters lost during threading
- **Cause**: Likely shallow copy issue
- **Fix Needed**: Use deepcopy or Queue for parameter passing

### ❌ First log shows wrong/missing values
- **Status**: Parameters not set from UI
- **Cause**: UI controls might not be initialized
- **Fix Needed**: Check UI widget initialization

### ❌ No log output at all
- **Status**: Debug code not running
- **Cause**: Code might not have been updated
- **Fix Needed**: Verify git pull and syntax check

## Debug Log Locations

1. **pp_apply_limits_to_spade_params()** → Line ~3235
   - Logs when parameters are set from UI

2. **PostProcessingWorker.regenerate_plots()** → Line ~1847
   - Logs when worker receives parameters

3. **Axis limit application** → Line ~2068
   - Logs when limits are applied to matplotlib axes

## Troubleshooting

### Plots not updating
- Ensure you're looking in the correct output folder
- Check the Preview tab for error messages
- Verify "Auto Calculate Limits" is UNCHECKED

### GUI freezes
- This shouldn't happen (worker runs in separate thread)
- If it does, check CPU usage
- Try with fewer files

### No preview output
- Check if output directory has velocity files
- Look for `*--vel-smooth-with-uncert.csv` files
- Verify parameter files are in the input folder

## What to Report

When sharing debug output with developers, include:
1. All three log sections (or note which are missing)
2. The actual parameter values shown
3. Whether plots respect the limits or not
4. Any error messages in the Preview output
