# HEL Detection Timing Reference

## Quick Answer

**HEL start/end times are RELATIVE TO THE ALIGNED TIME (t=0), NOT the original time.**

## Detailed Explanation

### The Timing Sequence

1. **Load Data** (Step 1)
   - Time is in seconds (or converted to ns)
   - Velocity data is raw
   - No alignment yet

2. **Align Traces** (Step 2)
   - Find first time when velocity ≥ alignment threshold (default: 30 m/s)
   - Call this time `t0`
   - Create aligned time: `t_aligned = t_original - t0`
   - **Now t=0 is at the point where velocity reached threshold**

3. **HEL Analysis Window** (Step 3)
   - HEL start time: User specified (default: 0.0 ns)
   - HEL end time: User specified (default: 12.0 ns)
   - **These are relative to aligned time (t=0)**
   - Analysis window: `t_aligned >= hel_start_time AND t_aligned <= hel_end_time`

### Example Timeline

```
Original Time:  900 ns  1000 ns (velocity=30m/s)  1012 ns
Aligned Time:   -100 ns  0 ns ← t=0 alignment     12 ns

HEL Analysis with defaults (start=0, end=12):
├─ Starts at: 0 ns (aligned) = 1000 ns (original)
└─ Ends at: 12 ns (aligned) = 1012 ns (original)

HEL Analysis with custom (start=-10, end=20):
├─ Starts at: -10 ns (aligned) = 990 ns (original)
└─ Ends at: 20 ns (aligned) = 1020 ns (original)
```

## GUI Parameters

| Parameter | Unit | Default | Reference |
|-----------|------|---------|-----------|
| HEL Start Time | ns | 0.0 | Relative to t=0 |
| HEL End Time | ns | 12.0 | Relative to t=0 |
| Alignment Threshold | m/s | 30.0 | Where t=0 is set |
| Angle Threshold | degrees | 45.0 | For HEL detection |

## How to Use

### Scenario 1: Analyze Early Response After Shock

Want to see HEL detection immediately after shock arrival:
- Set HEL Start Time: 0 ns (right at alignment)
- Set HEL End Time: 5 ns (5 ns after shock)
- **Analysis window**: t=0 to 5 ns (after shock)

### Scenario 2: Skip Shock Rise and Analyze Plateau

Want to skip the shock rise and analyze elastic plateau:
- Set HEL Start Time: 5 ns (after shock rise)
- Set HEL End Time: 20 ns (plateau region)
- **Analysis window**: 5-20 ns (after shock)

### Scenario 3: Look Before Shock Alignment

Want to include data before shock reaches threshold:
- Set HEL Start Time: -50 ns (50 ns before shock)
- Set HEL End Time: 10 ns (10 ns after shock)
- **Note**: Start/end times can be negative!
- **Analysis window**: 50 ns before to 10 ns after shock

## Code Implementation

```python
# Step 1: Load velocity data
time_original = df.iloc[:, 0].values  # seconds
velocity = df.iloc[:, 1].values       # m/s

# Step 2: Find and apply alignment
alignment_threshold = 30.0  # m/s
idx = first(velocity >= alignment_threshold)
t0 = time_original[idx]
time_aligned = (time_original - t0) * 1e9  # convert to ns

# Step 3: HEL analysis window (relative to t=0)
hel_start = 0.0    # ns, relative to t=0
hel_end = 12.0     # ns, relative to t=0

# Step 4: Crop to HEL window
mask = (time_aligned >= hel_start) & (time_aligned <= hel_end)
hel_data = velocity[mask]
hel_time = time_aligned[mask]

# Step 5: Perform HEL detection
hel_strength = detect_hel(hel_time, hel_data)
```

## Important Notes

⚠️ **Relative vs Absolute Times**
- All HEL times in GUI are RELATIVE to aligned t=0
- Not relative to file start or acquisition time
- Useful for focusing on shock response region

✓ **Negative Times Are Valid**
- You can set HEL Start Time to negative values
- This includes data BEFORE shock alignment
- Range: -1000 to +5000 ns supported

✓ **Default Settings**
- Start: 0 ns (right at shock arrival)
- End: 12 ns (elastic response window)
- Good for typical shock experiments

## Commit

- **Commit**: 474f83c
- **Message**: Clarify HEL detection timing
- **Changes**: Updated GUI tooltips for clarity

---

**Remember**: Think in terms of "time after shock arrival", not "original time"!
