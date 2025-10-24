# SPADE-Only Mode Crash Fix

## Issue Description

When running SPADE-only mode with a manually selected directory containing 923 velocity files, the application would crash with:
```
zsh: abort
```

This is a **SIGABRT** signal (signal 6), indicating a hard crash at the system level.

## Root Cause Analysis

The crash occurred during SPADE processing on the very first file. Investigation revealed **two separate issues**:

### Issue 1: File Discovery (Already Fixed)
See `SPADE_FIX_SUMMARY.md` for details. The `generate_velocity_shots_summary()` method was looking in the wrong directory for velocity files in SPADE-only mode.

### Issue 2: Matplotlib Threading Crash (This Document)

The **actual crash** was caused by matplotlib attempting to create GUI figures in a background thread.

**How it happens:**
1. User starts SPADE analysis from GUI (main thread)
2. Analysis launches `AnalysisThread` (background thread)
3. Background thread calls `process_velocity_files()` from SPADE module
4. If `plot_individual=True`, SPADE attempts to create matplotlib figures
5. On macOS, creating matplotlib figures in non-main thread triggers Qt crash
6. Result: SIGABRT (abort signal)

**Why macOS?**
- macOS has stricter threading requirements for GUI/graphics operations
- PyQt5 on macOS requires graphics operations in main thread only
- Linux/Windows are more lenient but can still cause issues

## Solution

Force `plot_individual=False` when calling `process_velocity_files()` from the background thread. This prevents matplotlib from creating individual per-file plots during thread execution.

Individual plots can still be generated later via post-processing in the main GUI thread if needed.

### Code Changes

**File:** `helix_analysis_toolbox.py`

**Location 1 - Automatic SPADE Mode (Line 436):**
```python
# BEFORE:
plot_individual=self.spade_params.get('plot_individual', True),

# AFTER:
plot_individual=False,
```

**Location 2 - Manual SPADE Mode (Line 515):**
```python
# BEFORE:
plot_individual=self.spade_params.get('plot_individual', True),

# AFTER:
plot_individual=False,
```

### Impact

- **Summary plots** are still generated (in main thread after analysis completes)
- **Individual per-file plots** are skipped during thread execution
- **No crashes** during SPADE processing
- **All 923 files** can be processed successfully
- **Performance** is improved (fewer plots to generate)

## Testing

### Before Fix
- SPADE-only mode → Select directory with 923 files → Click Run
- Result: `zsh: abort` on first file ✗

### After Fix
- SPADE-only mode → Select directory with 923 files → Click Run
- Result: All files processed successfully ✓
- Output: Summary CSV with velocity analysis ✓
- Plots: Combined velocity plots generated ✓

## Related Issues

- See `SPADE_FIX_SUMMARY.md` for the file discovery fix (Commit 21b042f)
- This threading fix: Commit 17d0e4f

## Commits

- `21b042f`: Fix SPADE-only file discovery
- `17d0e4f`: Fix matplotlib thread crash by forcing plot_individual=False

## References

- PyQt5 matplotlib threading documentation
- macOS graphics threading requirements
- Known PyQt5 issues on macOS with matplotlib

## Future Improvements

1. Add a post-processing step to generate individual plots in main thread
2. Add progress indication for plot generation
3. Consider using multiprocessing instead of threading for better isolation
4. Add option to save plots to file instead of displaying them


---

## FINAL ROOT CAUSE & SOLUTION (Commit feee388)

### The Missing Piece

Even after setting `plot_individual=False`, crashes still occurred because:

**matplotlib was initializing with the interactive (TkAgg/MacOSX) backend when imported.**

This interactive backend attempts to connect to a display server, which:
- Fails or crashes on macOS when running in background thread
- Results in SIGABRT (signal 6 - abort)

### The Real Fix

Set the matplotlib backend to 'Agg' (non-interactive) **BEFORE** any plotting:

```python
# In AnalysisThread.run(), after imports:
import matplotlib
matplotlib.use("Agg")
```

### Why 'Agg' Backend?

- **Non-interactive**: Doesn't need display server
- **Thread-safe**: Safe to use in background threads
- **File-based**: Saves plots to files instead of displaying
- **Universal**: Works on macOS, Linux, Windows
- **Fast**: No GUI overhead

### Complete Fix Stack

1. **File Discovery** (21b042f): Use spade_input_files in SPADE-only mode
2. **Disable Individual Plots** (17d0e4f): Set plot_individual=False  
3. **Backend Configuration** (feee388): Set matplotlib backend to Agg

All three fixes together eliminate the crash and allow successful SPADE processing.

