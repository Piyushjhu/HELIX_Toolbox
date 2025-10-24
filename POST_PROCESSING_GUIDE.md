# Post-Processing Guide

## Quick Start

1. **Launch GUI**: `python3 helix_analysis_toolbox.py`
2. **Go to**: Post-Processing Tab
3. **Select**: Output directory from ALPSS analysis
4. **Configure**: Plot options and axis limits
5. **Click**: "Preview" or "Save"

## Features

### Plot Options

- **Regenerate combined aligned velocity plot**: Generate all_velocity_traces.png
- **Color by Sample material**: Use material colors from parameter file
- **Zoom Window (ns)**: Size of zoom region in bottom subplot
- **Alignment Threshold (m/s)**: First velocity ≥ this is t=0 (NEW!)

### Axis Limits

- **Auto**: Automatically calculate limits from data
- **Manual**: Specify exact ranges
  - Main Subplot: X and Y ranges for top plot
  - Zoom Subplot: X and Y ranges for bottom plot

### Material Subplots

- Separate plot with one subplot per material
- Each material in its unique color
- Respects your axis limit settings
- Shows trace count per material

## Workflow Examples

### Example 1: Different Alignment Threshold

Problem: Want to align traces at 50 m/s instead of 30 m/s

Solution:
1. Change "Alignment Threshold" to 50 m/s
2. Click "Save"
3. New plots generated instantly!

### Example 2: Focus on Specific Time Range

Problem: Only care about 0-50 ns window

Solution:
1. Uncheck "Auto" under Axis Limits
2. Set Main X min to 0, X max to 50
3. Adjust zoom window similarly
4. Click "Save"
5. All plots zoomed to your range!

### Example 3: Analyze One Material

Problem: Want to focus on just Copper traces

Solution:
1. Go to velocity_traces_by_material.png
2. Look at Cu subplot
3. Adjust axis limits to zoom in
4. Regenerate to see other materials too

## Files Generated

After clicking "Save", check output directory for:

- **all_velocity_traces.png** - 2 subplots (main + zoom)
- **velocity_traces_by_material.png** - 4 subplots (one per material)

## Parameters

All adjustable parameters:

| Parameter | Range | Default | Unit |
|-----------|-------|---------|------|
| Zoom Window | 10-10000 | 1000 | ns |
| Alignment Threshold | 0-1000 | 30 | m/s |
| X min (main) | Any | 0 | ns |
| X max (main) | Any | 100 | ns |
| Y min (main) | Any | 0 | m/s |
| Y max (main) | Any | 600 | m/s |
| X min (zoom) | Any | 0 | ns |
| X max (zoom) | Any | 50 | ns |
| Y min (zoom) | Any | 0 | m/s |
| Y max (zoom) | Any | 300 | m/s |

## Troubleshooting

**No files found?**
- Check output directory path
- Verify SPADE_analysis folder exists
- Check file permissions

**Plots not updating?**
- Click "Preview" first (testing mode)
- Then click "Save" to generate final plots
- Check that parameters are not set to "Auto"

**Memory issues with 923 files?**
- Processing should take <10 seconds
- If slow, reduce zoom window or adjust limits

## Tips

✓ Always preview before saving final plots
✓ Alignment threshold affects all traces - use wisely
✓ Material colors are consistent across all plots
✓ Auto limits are fast for exploration
✓ Custom limits give you full control

---

**Status**: ✅ Production Ready

All 923 velocity traces can be processed and visualized quickly!
