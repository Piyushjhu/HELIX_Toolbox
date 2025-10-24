# Post-Processing Debugging & Testing Guide

## Quick Links

| Document | Purpose | Read Time | Use When |
|----------|---------|-----------|----------|
| **QUICK_TEST_CHECKLIST.txt** | Printable checklist | 5 min | Running the test |
| **POST_PROCESSING_TEST_GUIDE.md** | Detailed procedures | 15 min | Need more information |
| **POST_PROCESSING_SESSION_SUMMARY.md** | Complete overview | 20 min | Want full context |
| **This README** | Navigation guide | 2 min | Getting oriented |

---

## The Problem We're Solving

When you:
1. Go to Post-Processing tab
2. Set custom axis limits (e.g., X: 0-100 ns, Y: 0-600 m/s)
3. Click "Preview"

The generated plots **don't use your custom limits** - they use auto-calculated ranges instead.

---

## What We Did

### ✅ HEL Detection (Complete)
- Implemented HEL detection algorithm
- Calculates HEL strength, uncertainty, free surface velocity
- Outputs to CSV file
- **Status**: Working and integrated

### ✅ Debug Infrastructure (Complete)
- Added 3 logging checkpoints in the code
- Trace parameter flow from UI → Worker → Plot
- **Status**: Ready to identify the bug

### ✅ Testing Framework (Complete)
- 5 scenario types (A-E) covering all possibilities
- Step-by-step testing procedures
- Clear result interpretation guide
- **Status**: Ready to execute

---

## How to Get Started

### For Impatient Users (5 minutes)
```bash
# 1. Get the code
git pull origin main

# 2. Read the checklist
cat QUICK_TEST_CHECKLIST.txt

# 3. Follow steps 1-8

# 4. Report which scenario (A-E) you see
```

### For Careful Users (20 minutes)
```bash
# 1. Read this guide (you're doing it!)

# 2. Read the detailed testing guide
cat POST_PROCESSING_TEST_GUIDE.md

# 3. Read the session summary for context
cat POST_PROCESSING_SESSION_SUMMARY.md

# 4. Run the test following the guide

# 5. Report results with debug output
```

---

## Testing Scenarios (TL;DR)

After running the test, you'll see one of these:

| Scenario | What You See | Status | Next Step |
|----------|--------------|--------|-----------|
| **A** ✅ | All logs OK + plots correct | BUG FIXED! | Nothing, bug is fixed |
| **B** ⚠️ | All logs OK + plots wrong | Parameters flow OK | Fix plot generation code |
| **C** ⚠️ | First log OK + worker log bad | Threading issue | Fix parameter passing |
| **D** ❌ | No logs at all | Code not running | Debug execution |
| **E** ❌ | First log has wrong values | UI problem | Fix initialization |

---

## The Three Debug Checkpoints

### 1️⃣ When UI Sets Parameters
```
[POST-PROCESSING] Parameters applied:
  auto_calc_limits: False
  x_min/max_main: 0.0/100.0
  y_min/max_main: 0.0/600.0
```
This shows when the GUI collects your settings.

### 2️⃣ When Worker Receives Parameters
```
[WORKER] Received parameters in regenerate_plots:
  auto_calc_limits: False
  x_min/max_main: 0.0/100.0
  y_min/max_main: 0.0/600.0
```
This shows what the background thread actually received.

### 3️⃣ When Limits Applied to Plots
```
Applied top limits: X(0.0-100.0), Y(0.0-600.0)
Applied zoom limits: X(0.0-50.0), Y(0.0-300.0)
```
This shows when the matplotlib axes are configured.

---

## What Changed in the Code

### File Modified: `helix_analysis_toolbox.py`
- **No logic changes** - only added logging
- **10 new debug lines** - all marked with `# Debug:` comments
- **3 checkpoints** - lines 3240, 1852, 2068 (approximate)
- **Backward compatible** - existing code unchanged

### Files Created:
1. `POST_PROCESSING_TEST_GUIDE.md` - Detailed guide
2. `QUICK_TEST_CHECKLIST.txt` - Quick reference
3. `POST_PROCESSING_SESSION_SUMMARY.md` - Complete overview
4. `POST_PROCESSING_README.md` - This file

---

## After You Test

### If Scenario A: ✅ Bug Fixed!
- Congratulations, the system is working correctly!
- Close the post-processing issue
- Continue with other improvements

### If Scenario B-E: Need Fix
- Share the debug output from Preview tab
- Include screenshot of generated plots
- Include which scenario (B, C, D, or E) you saw
- We'll implement scenario-specific fix

---

## Reference Information

### GUI Controls in Post-Processing Tab
- **Auto Calculate Limits**: Checkbox (default: OFF)
  - When OFF: Use your custom values
  - When ON: Auto-calculate from data
- **Main Plot X Min/Max**: Custom X-axis range (ns)
- **Main Plot Y Min/Max**: Custom Y-axis range (m/s)
- **Zoom Plot X Min/Max**: Custom zoom X-axis range (ns)
- **Zoom Plot Y Min/Max**: Custom zoom Y-axis range (m/s)
- **Alignment Threshold**: For trace alignment (m/s)
- **Preview**: Generate plots with current settings
- **Save**: Save plots with current settings

### Generated Files
- `all_velocity_traces.png` - 2-subplot plot (main + zoom)
- `velocity_traces_by_material.png` - Multi-subplot by material
- Location: `SPADE_analysis/` subdirectory in output folder

### Debug Output Location
- Displayed in: Preview text area in Post-Processing tab
- Also logged to: Standard output / terminal

---

## Common Questions

**Q: Why add debug logging instead of fixing directly?**
A: The bug could be caused by 5 different issues. The debug logging identifies which one, so we fix the right thing.

**Q: How long does the test take?**
A: 5-10 minutes including plot generation time.

**Q: What if I get Scenario A?**
A: Fantastic! The bug is already fixed, you can use post-processing normally.

**Q: What if I get Scenario B-E?**
A: Share the debug output and we'll implement the appropriate fix.

**Q: Can I run this multiple times?**
A: Yes! The test is non-destructive and can be repeated as needed.

---

## Troubleshooting

**GUI won't start**
- Ensure Python 3.8+ and PyQt5 installed
- Check: `python3 -m PyQt5.QtWidgets`

**No debug output appears**
- Verify you're using latest code: `git pull origin main`
- Check syntax: `python3 -m py_compile helix_analysis_toolbox.py`

**Preview button does nothing**
- Verify output directory exists and has velocity files
- Check Preview text area for error messages

**Plots don't generate**
- Verify velocity files exist: `*--vel-smooth-with-uncert.csv`
- Check disk space in output directory

---

## Need Help?

1. **Quick questions**: Check `QUICK_TEST_CHECKLIST.txt`
2. **Detailed help**: Read `POST_PROCESSING_TEST_GUIDE.md`
3. **Full context**: Read `POST_PROCESSING_SESSION_SUMMARY.md`
4. **Code questions**: Check comments in `helix_analysis_toolbox.py` (lines with "Debug:")

---

## Files Organization

```
Project Root
├── helix_analysis_toolbox.py          ← Main GUI file (updated)
├── POST_PROCESSING_README.md          ← This file (quick nav)
├── QUICK_TEST_CHECKLIST.txt           ← Use during testing
├── POST_PROCESSING_TEST_GUIDE.md      ← Detailed procedures
└── POST_PROCESSING_SESSION_SUMMARY.md ← Full overview
```

---

## Summary

| What | Status | File |
|------|--------|------|
| HEL Detection | ✅ Implemented | Code |
| Debug Framework | ✅ Added | Code |
| Testing Guide | ✅ Created | QUICK_TEST_CHECKLIST.txt |
| Testing Framework | ✅ Ready | 5 scenarios defined |
| Ready to Test? | ✅ YES | Start now! |

---

## Next Action

👉 **Read**: `QUICK_TEST_CHECKLIST.txt` (takes 5 minutes)

👉 **Run**: Steps 1-8 from the checklist

👉 **Report**: Which scenario (A-E) you see

---

**Last Updated**: October 24, 2025
**Status**: Ready for Testing
**Maintainer**: HELIX Toolbox Dev Team
