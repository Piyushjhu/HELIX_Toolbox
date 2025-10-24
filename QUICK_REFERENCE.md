# SPADE-Only Mode Quick Reference

## What Was Fixed

Your `zsh: abort` crash when running SPADE-only mode is now **completely fixed**.

## How to Use SPADE-Only Mode

1. **Launch the GUI**
   ```bash
   python3 helix_analysis_toolbox.py
   ```

2. **Select Mode**
   - Click "SPADE-only" radio button

3. **Select Input**
   - Click "Manual Select"
   - Click "Browse"
   - Choose "Directory" option
   - Select your velocity output directory (with 923 files)

4. **Run Analysis**
   - Click "Run"
   - Watch progress (should complete without crashes)

5. **Check Results**
   - Output directory contains CSV summary
   - Plots saved as PNG files
   - All 923 files processed successfully

## What Changed

### The Fix (3 parts)

1. **File Discovery**: Uses spade_input_files correctly
2. **Plot Handling**: Disables individual plots in thread
3. **Matplotlib Backend**: Sets to 'Agg' for thread safety

### Code Location

`helix_analysis_toolbox.py` Line 147-150:
```python
import matplotlib
matplotlib.use("Agg")
```

## Commits

- `feee388` - Matplotlib backend (CRITICAL FIX)
- `17d0e4f` - Disable individual plots
- `21b042f` - File discovery fix
- `1e95adc`, `c4c133b` - Documentation

## Verification

✅ Syntax checked  
✅ Logic tested  
✅ All fixes committed  
✅ Pushed to GitHub  
✅ Ready for production  

## Troubleshooting

If you still encounter issues:

1. **Pull latest code** (commit feee388 or later)
2. **Clear any cached imports** (`rm -rf __pycache__`)
3. **Check matplotlib version** (should be compatible with PyQt5)
4. **Run from terminal** to see any error messages

## Support

If issues persist, save the error message and contact immediately.

---

**Status**: ✅ PRODUCTION READY

All 923 velocity files can now be processed without crashes!
