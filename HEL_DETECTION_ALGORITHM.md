# HEL (Hugoniot Elastic Limit) Detection Algorithm - Complete Implementation

## Overview

The HEL detection algorithm identifies the Hugoniot Elastic Limit in velocity-time traces from PDV (Photonic Doppler Velocimetry) data. The algorithm uses a **RDP+Linear Hybrid Method** that combines geometric simplification (Ramer-Douglas-Peucker algorithm) with linear regression on raw data to robustly detect the elastic-plastic transition.

**Key Innovation**: Instead of relying on gradient-based detection (which is sensitive to noise), the algorithm:
1. Uses RDP to identify candidate "knee" points (geometric simplification)
2. Performs linear regression on **raw data segments** to verify slopes
3. Validates physics-based criteria (rise must be positive, plateau must be significantly flatter)

This approach is more robust to noise and handles ramping plateaus correctly.

---

## Algorithm Steps

### Step 0: Time Zero Alignment

**Purpose**: Establish a reliable time zero (t=0) for HEL detection.

**Method**: 
1. Find the first point where `velocity > 0`
2. Verify that velocity remains non-zero and is **increasing on average** for **2 ns** after this point
3. If conditions are met, set this point as `t=0` (called `hel_t0`)
4. Create HEL-aligned time array: `time_aligned_iq = time_data - hel_t0`

**Validation Criteria**:
- All velocities in the 2 ns window must be > 0
- Average slope (trend) over 2 ns must be positive: `(v_end - v_start) / (t_end - t_start) > 0`

**Fallback**: If no valid point is found, use velocity threshold alignment (30 m/s default).

**Code Location**: Lines 1839-1890

---

### Step 1: Data Loading and Uncertainty Filtering

**Purpose**: Load velocity data and filter out high-uncertainty points.

**Process**:
1. Extract valid data points (non-NaN velocities)
2. Calculate relative uncertainty: `rel_unc = |uncertainty| / max(|velocity|, 1e-9)`
3. Filter points where `rel_unc < 1.0` (uncertainty < 100% of max velocity)
4. If too many points are filtered (< 10 remaining), use all valid points as fallback

**Output**: 
- `hel_time_clean`: Filtered time array
- `hel_velocity_clean`: Filtered velocity array  
- `hel_unc_clean`: Filtered uncertainty array

**Code Location**: Lines 1915-1931

---

### Step 2: HEL Window Selection

**Purpose**: Extract data within the user-specified HEL detection window.

**Parameters** (from config):
- `hel_start_time_ns`: Start of HEL window (default: 0.0 ns)
- `hel_end_time_ns`: End of HEL window (default: 25.0 ns)

**Process**:
1. Create mask: `hel_start ≤ time ≤ hel_end` (if `hel_end` is specified)
2. Extract data within window: `hel_time_window`, `hel_velocity_window`, `hel_unc_window`
3. If < 10 points in window, use all clean data as fallback

**Code Location**: Lines 1933-1942

---

### Step 3 & 4: RDP+Linear Hybrid HEL Detection

**Purpose**: Detect the elastic-plastic transition using geometric simplification (RDP) and linear regression verification.

**Overview**: This is a two-stage approach:
- **Stage 1 (RDP - The Scout)**: Use Ramer-Douglas-Peucker algorithm to simplify the velocity trace and identify candidate "knee" points
- **Stage 2 (Linear Regression - The Verifier)**: Extract raw data segments around candidate knees, fit lines, and verify physics-based criteria

**Parameters** (from config):
- `hel_rdp_epsilon`: RDP tolerance (default: 3.0 m/s)
- `hel_slope_drop_ratio`: Minimum slope drop ratio (default: 0.2)
- `hel_min_plateau_duration`: Minimum plateau duration (default: 2.0 ns)

#### Step 3.1: RDP Simplification (The Scout)

**Process**:
1. Apply Ramer-Douglas-Peucker algorithm to the velocity trace:
   ```python
   rdp_indices = ramer_douglas_peucker_indices(time, velocity, epsilon)
   ```
2. RDP reduces the trace to key vertices (corners/knees) that capture the main shape
3. The `epsilon` parameter controls simplification:
   - **Higher epsilon** → Fewer vertices (more simplification, may miss features)
   - **Lower epsilon** → More vertices (less simplification, may include noise)
4. Extract RDP simplified points: `rdp_points = (time[rdp_indices], velocity[rdp_indices])`

**What RDP Does**:
- RDP finds the minimum number of points needed to represent the trace within `epsilon` tolerance
- It identifies "important" points where the trace changes direction significantly
- These points include potential HEL transition points (knees)

**Code Location**: Lines 6275-6279, Function `ramer_douglas_peucker_indices()` (lines 6136-6210)

#### Step 3.2: Candidate Segment Iteration

**Process**:
1. Iterate through RDP vertices in groups of three: `(start, knee, end)`
2. For each candidate:
   - `idx_start = rdp_indices[i]` (start of rise segment)
   - `idx_knee = rdp_indices[i+1]` (potential HEL point)
   - `idx_end = rdp_indices[i+2]` (end of plateau segment)

**Code Location**: Lines 6288-6292

#### Step 3.3: Extract Raw Data Segments

**Process**:
1. Extract **raw data** (not RDP simplified) for each segment:
   ```python
   # Rise segment: from start to knee
   t_rise = time[idx_start : idx_knee + 1]
   v_rise = velocity[idx_start : idx_knee + 1]
   
   # Plateau segment: from knee to end
   t_plat = time[idx_knee : idx_end + 1]
   v_plat = velocity[idx_knee : idx_end + 1]
   ```

**Why Raw Data?**: 
- RDP simplification introduces errors based on `epsilon`
- Linear regression on raw data gives accurate slopes without RDP artifacts
- This is the "hybrid" aspect: RDP finds candidates, raw data verifies them

**Code Location**: Lines 6294-6301

#### Step 3.4: Duration Check (Fast Filter)

**Process**:
1. Calculate plateau duration: `duration_plat = t_plat[-1] - t_plat[0]`
2. If `duration_plat < hel_min_plateau_duration`, reject candidate (too short, likely noise)
3. This is a cheap check to filter out obviously invalid candidates early

**Code Location**: Lines 6303-6308

#### Step 3.5: Linear Regression on Raw Data (The Verifier)

**Process**:
1. Fit linear models to both segments using `np.polyfit`:
   ```python
   # Rise segment: v = m_rise * t + c_rise
   m_rise, c_rise = np.polyfit(t_rise, v_rise, 1)
   
   # Plateau segment: v = m_plat * t + c_plat
   m_plat, c_plat = np.polyfit(t_plat, v_plat, 1)
   ```

2. **Slope Interpretation**:
   - `m_rise`: Slope of rise segment (m/s per ns) - should be positive
   - `m_plat`: Slope of plateau segment (m/s per ns) - should be much smaller than `m_rise`

**Code Location**: Lines 6310-6324

#### Step 3.6: Physics-Based Verification

**Validation Rules**:

1. **Rule 1: Rise Must Be Rising**
   ```python
   if m_rise <= 0:
       continue  # Reject candidate
   ```
   - The rise segment must have positive slope (velocity increasing)
   - This ensures we're looking at the correct phase (elastic loading)

2. **Rule 2: Plateau Must Be Significantly Flatter**
   ```python
   if m_plat >= (m_rise * hel_slope_drop_ratio):
       continue  # Reject candidate
   ```
   - Plateau slope must be < `(rise_slope × hel_slope_drop_ratio)`
   - Example: If `m_rise = 5.0 m/s/ns` and `hel_slope_drop_ratio = 0.2`, then `m_plat < 1.0 m/s/ns`
   - **This handles ramping plateaus**: Plateau can still have positive slope, just much smaller

**Why This Works**:
- RDP identifies geometric "knees" (candidate transitions)
- Linear regression on raw data gives accurate slopes
- Physics rules ensure we detect the correct transition (elastic → plastic)
- Works for both flat and ramping plateaus

**Code Location**: Lines 6326-6336

#### Step 3.7: HEL Detection Success

**Process**:
1. If all validation rules pass, HEL is detected
2. Calculate mean plateau velocity from raw data: `mean_plateau_velocity = np.mean(v_plat)`
3. Store detection results:
   - `hel_time_detection = time[idx_knee]` (RDP knee point time)
   - `free_surface_velocity = mean_plateau_velocity` (mean of raw plateau data)
   - `rise_slope`, `plateau_slope`, `rise_intercept`, `plateau_intercept` (for plotting)
   - `t_rise`, `v_rise`, `t_plat`, `v_plat` (raw data segments for plotting)
   - `rdp_points` (RDP simplified points for plotting)

**Code Location**: Lines 6338-6362

**Note**: If no candidate passes all validation rules, HEL is not detected (`hel_found = False`), but `rdp_points` are still returned for visualization.

---

### Step 5: HEL Strength Calculation

**Purpose**: Calculate HEL stress from the plateau velocity.

**Material Properties** (from config or database):
- `density` (ρ): Material density (kg/m³)
- `acoustic_velocity` (c): Bulk wave speed (m/s)
- `C_L`: Longitudinal wave velocity (m/s), fallback to acoustic_velocity

**HEL Stress Formula**:
```
σ_HEL = 0.5 × ρ × c × |U_HEL| / 1e9
```

Where:
- `σ_HEL`: HEL stress (GPa)
- `ρ`: Material density (kg/m³)
- `c`: Acoustic velocity (m/s)
- `U_HEL`: Free surface velocity at HEL plateau (m/s)
- Division by 1e9 converts Pa to GPa

**Uncertainty Calculation**:
```
δσ_HEL = 0.5 × ρ × c × δU / 1e9
```

Where `δU` is the velocity uncertainty at the HEL detection point (interpolated from uncertainty array).

**Code Location**: Lines 2156-2158

---

### Step 6: Validation Checks

**Purpose**: Validate HEL detection before accepting it.

**Validation Criteria**:

1. **Minimum Velocity Check**:
   ```python
   if abs(free_surface_velocity) < min_hel_velocity:
       # Reject HEL
   ```
   - Parameter: `minimum_HEL_velocity_expected` (default: 10.0 m/s)
   - Rejects HEL if detected velocity is below threshold

2. **Strain Rate Check** (if HEL passes velocity check):
   - Calculate elastic shock strain rate (see Step 7)
   - If strain rate < 0, reject HEL (unphysical)

**Code Location**: Lines 2142-2154, 2292-2302

---

### Step 7: Elastic Shock Strain Rate Calculation

**Purpose**: Calculate the strain rate during the elastic shock phase.

**Formula**:
```
ε̇_elastic = (1 / (2 × C_L)) × (U_HEL - U_0) / (t_HEL - t_0)
```

Where:
- `ε̇_elastic`: Elastic shock strain rate (s⁻¹)
- `C_L`: Longitudinal wave velocity (m/s)
- `U_HEL`: Free surface velocity at HEL (m/s)
- `U_0`: Free surface velocity at t=0 (m/s)
- `t_HEL`: Time at HEL detection (s)
- `t_0`: Time at t=0 (s)

**Time Zero for Strain Rate**:
- Uses the HEL-aligned time zero (`hel_t0`) from Step 0
- `U_0` = velocity at `hel_t0`
- `t_0` = `hel_t0` (in seconds)

**Validation**:
- If `t_HEL ≤ t_0`, return NaN (invalid time interval)
- If strain rate < 0, reject HEL (unphysical)

**Code Location**: Lines 2277-2305, Function `elastic_shock_strain_rate()` (lines 6364-6392)

---

## Configuration Parameters

All parameters are defined in `helix_master_config.json` under the `spade_params` section:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hel_detection_enabled` | `false` | Enable/disable HEL detection |
| `experiment_hel_detection` | `false` | Alternative flag for HEL detection |
| `hel_start_time_ns` | `0.0` | Start of HEL detection window (ns, relative to t=0) |
| `hel_end_time_ns` | `25.0` | End of HEL detection window (ns, relative to t=0) |
| `hel_rdp_epsilon` | `3.0` | RDP tolerance (m/s) - controls simplification level |
| `hel_slope_drop_ratio` | `0.2` | Minimum slope drop ratio - plateau must be < 20% of rise slope |
| `hel_min_plateau_duration` | `2.0` | Minimum plateau duration (ns) - filters short features |
| `minimum_HEL_velocity_expected` | `10.0` | Minimum velocity (m/s) for valid HEL |
| `plot_individual` | `false` | Generate individual HEL plots for each trace |

### Parameter Tuning Guide

**`hel_rdp_epsilon` (m/s)**:
- **Too high** (e.g., > 10 m/s): May miss subtle transitions, too much simplification
- **Too low** (e.g., < 1 m/s): May include noise, too many vertices
- **Recommended**: 2.0 - 5.0 m/s for most materials

**`hel_slope_drop_ratio` (dimensionless)**:
- **Too high** (e.g., > 0.5): Allows plateaus with high slopes, may detect false positives
- **Too low** (e.g., < 0.1): Requires very flat plateaus, may miss ramping plateaus
- **Recommended**: 0.15 - 0.25 for most materials

**`hel_min_plateau_duration` (ns)**:
- **Too high** (e.g., > 5 ns): May miss short but valid HEL plateaus
- **Too low** (e.g., < 1 ns): May detect noise spikes as HEL
- **Recommended**: 1.5 - 3.0 ns depending on data sampling rate

---

## Output Variables

### HEL Detection Results:
- `hel_ok`: Boolean indicating if HEL was successfully detected
- `hel_strength`: HEL stress (GPa)
- `hel_uncertainty`: HEL stress uncertainty (GPa)
- `hel_time_detection`: Time at HEL detection (ns, relative to HEL t=0)
- `hel_consecutive_points`: Number of points in HEL plateau segment
- `hel_segment_time_ns`: Duration of HEL plateau segment (ns)
- `hel_strain_rate`: Elastic shock strain rate (s⁻¹)
- `free_surface_velocity`: Mean velocity at HEL plateau (m/s)

### RDP+Linear Detection Details (for plotting):
- `rdp_points`: RDP simplified points (N×2 array: [time, velocity])
- `rise_slope`: Linear fit slope of rise segment (m/s per ns)
- `plateau_slope`: Linear fit slope of plateau segment (m/s per ns)
- `rise_intercept`: Linear fit intercept of rise segment (m/s)
- `plateau_intercept`: Linear fit intercept of plateau segment (m/s)
- `t_rise`, `v_rise`: Raw data points for rise segment
- `t_plat`, `v_plat`: Raw data points for plateau segment

### Time Alignment:
- `hel_t0`: Time zero point (ns, original time scale)
- `hel_t0_idx`: Index of time zero point
- `time_aligned_iq`: Time array aligned to HEL t=0

---

## Algorithm Flow Summary

```
1. Time Zero Alignment
   └─> Find first velocity > 0 with 2 ns increasing trend
       └─> Set as t=0

2. Data Filtering
   └─> Filter by relative uncertainty (< 100%)
       └─> Extract HEL window

3. RDP Simplification (The Scout)
   └─> Apply Ramer-Douglas-Peucker algorithm
       └─> Get simplified vertices (candidate knees)
           └─> Extract RDP points for visualization

4. Candidate Iteration
   └─> For each (start, knee, end) triplet:
       ├─> Extract raw data segments (rise, plateau)
       ├─> Check minimum plateau duration
       ├─> Fit linear models to raw data
       ├─> Verify: rise slope > 0
       └─> Verify: plateau slope < (rise_slope × drop_ratio)
           └─> If all pass: HEL detected!

5. HEL Strength Calculation
   └─> σ_HEL = 0.5 × ρ × c × |U_HEL| / 1e9

6. Validation
   └─> Check minimum velocity threshold
       └─> Calculate strain rate
           └─> Reject if strain rate < 0

7. Output
   └─> Save results and generate plots (if enabled)
       └─> Plot includes: RDP simplified line, RDP vertices, raw data segments, linear fits
```

---

## Visualization in Plots

When `plot_individual = true`, the HEL plots show:

### Top Subplot: Full Velocity Trace
- Blue line: Full velocity trace
- Yellow shaded region: HEL detection window
- Orange dashed lines: HEL window boundaries
- X-axis limited to 60 ns

### Middle Subplot: HEL Window Detail (RDP+Linear Hybrid Method)
- Blue line: Velocity in HEL window
- **Red line**: RDP simplified line (geometric simplification)
- **Red circles**: RDP vertices (key points from simplification)
- **Green star**: HEL detection point (if detected)
- **Cyan dots**: Raw data points for rise segment
- **Cyan dashed line**: Linear fit to rise segment with slope annotation
- **Magenta dots**: Raw data points for plateau segment
- **Magenta dashed line**: Linear fit to plateau segment with slope annotation
- **"NO HEL" label**: Red box if HEL not detected

### Bottom Subplot: Gradient vs Time (Reference Only)
- Green line: Gradient (dv/dt) - **not used for detection**, shown for reference only
- Red dashed lines: Angle thresholds (if applicable)
- Title: "Gradient vs Time (Reference Only - Not Used for Detection)"

---

## Advantages of RDP+Linear Hybrid Method

1. **Robust to Noise**: RDP simplification filters noise while preserving important features
2. **Accurate Slopes**: Linear regression on raw data gives precise slopes without RDP artifacts
3. **Handles Ramping Plateaus**: Works for both flat and ramping plateaus (positive but small plateau slope)
4. **Physics-Based**: Validation rules ensure correct elastic-plastic transition
5. **Visualizable**: RDP points and linear fits can be plotted for verification
6. **No Gradient Smoothing Issues**: Avoids problems with gradient smoothing window size

---

## Notes and Limitations

1. **RDP Epsilon Sensitivity**: The `hel_rdp_epsilon` parameter controls simplification. Too high may miss transitions, too low may include noise. Tune based on your data quality.

2. **Time Zero**: The algorithm uses a robust method to find t=0 by requiring velocity to be increasing for 2 ns, avoiding noise spikes.

3. **Plateau Detection**: The algorithm finds the **earliest** valid elastic-plastic transition that passes all validation rules.

4. **Material Properties**: HEL strength calculation requires accurate material properties (density and acoustic velocity). These are loaded from config or material database.

5. **Uncertainty Propagation**: Velocity uncertainty is propagated to HEL stress uncertainty using the same formula with uncertainty values.

6. **No HEL Detected**: If no candidate passes validation, HEL is not detected, but RDP visualization is still shown in plots (with "NO HEL" label).

7. **Multiple Candidates**: The algorithm stops at the first valid candidate (earliest in time). This ensures we detect the elastic response before plastic deformation.

---

## References

- Hugoniot Elastic Limit (HEL) is the maximum stress a material can sustain in elastic response under shock loading
- The HEL corresponds to the transition from elastic to plastic deformation
- HEL stress is calculated using the acoustic approximation: σ = 0.5 × ρ × c × U
- Ramer-Douglas-Peucker algorithm: A line simplification algorithm that reduces the number of points while preserving shape

---

**Last Updated**: Based on RDP+Linear Hybrid implementation in `helix_analysis_toolbox.py` (lines 2000-2305, 6244-6362)
