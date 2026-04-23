# SPALL Detection Algorithm - Complete Implementation

## Overview

The SPALL (Spallation) detection algorithm identifies spallation events in velocity-time traces from PDV (Photonic Doppler Velocimetry) data. The algorithm uses multiple detection methods to robustly identify the characteristic "checkmark" signature: **Plateau → Drop (Pullback) → Rebound (Recompression)**.

**Key Innovation**: The algorithm combines:
1. **RDP Topology Detection**: Uses Ramer-Douglas-Peucker simplification to identify the geometric "checkmark" shape
2. **5-Segment Linear Analysis**: Fits linear segments to extract spall strength and strain rate
3. **Horizontal Plateau Constraint**: Prevents P1/P2 shifting by enforcing a horizontal plateau

The algorithm supports two detection methods:
- **RDP Method**: Uses RDP-guided 5-segment fitting (recommended)
- **Horizontal Plateau Method**: Uses horizontal plateau constraint with 5-segment fitting (legacy)

---

## Algorithm Flow

### Main Entry Point: `detect_dns_and_process_spall`

**Purpose**: Main function that orchestrates the complete spall detection and analysis pipeline.

**Input Parameters**:
- `file_path`: Path to velocity CSV file
- `base_name`: Base filename for logging
- `density`: Material density (kg/m³)
- `acoustic_velocity`: Material acoustic velocity (m/s)
- `threshold_velocity`: Velocity threshold for shock arrival (m/s)
- `spall_start_time`: Start of spall analysis window (ns)
- `spall_end_time`: End of spall analysis window (ns)
- `analysis_model`: Analysis model ('max_min', 'hybrid', etc.)
- `plot_path`: Optional path to save analysis plot
- `sample_material`: Material name for EOS calculations

**Returns**: Dictionary with spall strength, DNS classification, strain rate, shock stress, and diagnostics

**Code Location**: Lines 369-986 in `helix_analysis_toolbox.py`

---

## Step-by-Step Algorithm

### Step 1: Data Loading and Header Detection

**Purpose**: Load velocity CSV and handle various header formats.

**Process**:
1. Read CSV file using pandas
2. Detect header row by checking if first row contains keywords: 'time', 'velocity', 'uncertainty'
3. If header detected, read with `header=0`; otherwise read as headerless
4. Extract first 3 columns:
   - Column 0: Time (seconds)
   - Column 1: Velocity (m/s)
   - Column 2: Uncertainty (m/s) - optional

**Validation**:
- Require at least 3 columns
- Require at least 20 data points
- Convert to numeric arrays, handling NaN values

**Code Location**: Lines 404-433

---

### Step 1b: Uncertainty Filtering

**Purpose**: Filter out high-uncertainty data points.

**Process**:
1. Calculate maximum velocity: `max_vel = max(|velocity|)`
2. Calculate relative uncertainty: `rel_unc = |uncertainty| / max(max_vel, 1e-9)`
3. Filter points where `rel_unc >= 1.0` (uncertainty ≥ 100% of max velocity)
4. Set filtered points to NaN in velocity array

**Output**: `vel_clean` - velocity array with high-uncertainty points set to NaN

**Code Location**: Lines 435-439

---

### Step 2: Trace Alignment to Shock Arrival

**Purpose**: Align trace to shock arrival (t=0) using the same method as HEL detection for consistency.

**Method**: Two-stage alignment with fallback

#### Stage 2a: HEL t=0 Alignment (Primary Method)

**Process**:
1. Extract valid (non-NaN) velocity and time data
2. Convert time to nanoseconds: `time_valid_ns = time_valid * 1e9`
3. Call `find_hel_t0_alignment()` with:
   - `time_valid_ns`: Time array in nanoseconds
   - `vel_valid`: Valid velocity array
   - `min_velocity_threshold`: Minimum expected HEL velocity (default: 10.0 m/s)
4. HEL alignment algorithm:
   - Find first point where `velocity > min_velocity_threshold`
   - Verify velocity remains positive and increasing for **10 ns** window
   - Set this point as `t=0` (called `hel_t0`)
5. If HEL alignment succeeds:
   - `t0 = hel_t0 / 1e9` (convert back to seconds)
   - `alignment_method_used = "HEL"`
   - Create aligned time: `t_aligned_ns = (time_s - t0) * 1e9`

**Code Location**: Lines 441-473

#### Stage 2b: Threshold Alignment (Fallback)

**Process** (if HEL alignment fails or is disabled):
1. Find first point where `velocity >= threshold_velocity`
2. Set `t0 = time_valid[threshold_idx[0]]`
3. Create aligned time: `t_aligned_ns = (time_s - t0) * 1e9`
4. `alignment_method_used = "threshold"` or `"threshold_fallback"`

**Validation**:
- Require at least one point above threshold
- If no threshold point found, return error: "No shock arrival detected"

**Code Location**: Lines 475-485

---

### Step 3: Time Window Extraction

**Purpose**: Extract data within the spall analysis window.

**Parameters** (from config):
- `spall_start_time`: Start of spall window (ns, default: typically 10-20 ns)
- `spall_end_time`: End of spall window (ns, default: typically 100-200 ns)

**Process**:
1. Create mask: `(~np.isnan(vel_clean)) & (t_aligned_ns >= spall_start_time) & (t_aligned_ns <= spall_end_time)`
2. Extract windowed data:
   - `time_window`: Time array in spall window (ns)
   - `vel_window`: Velocity array in spall window (m/s)
   - `uncert_window`: Uncertainty array in spall window (m/s)

**Validation**:
- Require at least 20 points in window
- If insufficient points, return error: "Insufficient data points in spall window"

**Code Location**: Lines 491-499

---

### Step 4: RDP Topology Check (The Scanner)

**Purpose**: Use RDP simplification to scan for the "checkmark" signature (Plateau → Drop → Rebound).

**Note**: This step is only executed when `spall_detection_method == 'rdp'`. For legacy `'5-segment'` method, this step is skipped.

**Parameters** (from config):
- `spall_rdp_epsilon`: RDP tolerance (default: 5.0 m/s) - larger than HEL epsilon
- `min_pullback_velocity`: Minimum pullback magnitude (default: 10.0 m/s)
- `min_recomp_ratio`: Minimum rebound ratio (default: 0.1, i.e., rebound must be ≥10% of drop)

**Process**:
1. Generate RDP simplified points for visualization (even if detection fails):
   ```python
   rdp_indices = ramer_douglas_peucker_indices(time_window, vel_window, rdp_epsilon)
   rdp_points = np.column_stack((time_window[rdp_indices], vel_window[rdp_indices]))
   ```

2. If using RDP method, call `detect_spall_rdp()`:
   - Returns: `(is_spall_rdp, rdp_reason, rdp_keys)`
   - `is_spall_rdp`: Boolean indicating valid spall signature
   - `rdp_reason`: Classification reason ("Valid Spall" or DNS reason)
   - `rdp_keys`: Dictionary with detected key points

3. If RDP detection fails:
   - Set `is_dns = True`
   - Set `dns_reason = rdp_reason`
   - Mark `Spall_Strength_GPa = "DNS"`

**Code Location**: Lines 501-562

---

### Step 4a: RDP-Based Spall Detection (`detect_spall_rdp`)

**Purpose**: Detect spall signature using RDP topology analysis.

**Algorithm**:
1. **RDP Simplification**:
   - Apply RDP with `epsilon = 5.0 m/s` (larger than HEL epsilon)
   - Extract simplified vertices: `rdp_vel = velocity[rdp_indices]`, `rdp_time = time[rdp_indices]`
   - Require at least 3 vertices after simplification

2. **Find Global Maximum (Shock Peak)**:
   - `max_idx_local = argmax(rdp_vel)` - index in RDP array
   - `max_idx_global = rdp_indices[max_idx_local]` - index in original window
   - Validate: Peak must not be at the end (require at least 2 points after peak)

3. **Topology Search: "Drop → Rebound"**:
   - Iterate through triplets after peak: `[Start, Min, End]`
   - For each triplet `(v_start, v_min, v_end)`:
     - **Check 1 (Geometry)**: Must be a valley: `v_min < v_start` AND `v_end > v_min`
     - **Check 2 (Pullback Magnitude)**: `pullback_mag = v_start - v_min` must be ≥ `min_pullback_velocity` (default: 10.0 m/s)
     - **Check 3 (Rebound Magnitude)**: `rebound_mag = v_end - v_min` must be ≥ `pullback_mag * min_recomp_ratio` (default: ≥10% of drop)
   - Stop at first valid spall signature

4. **Return Results**:
   - If valid signature found:
     ```python
     {
         'shock_peak_idx': max_idx_global,
         'plateau_idx': post_peak_indices[i],
         'min_idx': post_peak_indices[i+1],
         'recomp_idx': post_peak_indices[i+2],
         'pullback_velocity': pullback_mag,
         'rebound_velocity': rebound_mag,
         'plateau_velocity': v_start,
         'min_velocity': v_min,
         'recomp_velocity': v_end
     }
     ```
   - If no valid signature: Return `(False, "DNS: No valid Pullback-Recompression signature found (RDP)", None)`

**Code Location**: Lines 7887-7996

---

### Step 5: Extract Key Velocities

**Purpose**: Extract diagnostic velocities from RDP-detected features (if available).

**Process**:
1. If `rdp_keys` is available (RDP method succeeded):
   - Extract indices: `idx_peak`, `idx_plat`, `idx_min`, `idx_recomp`
   - Extract velocities from raw window data at these indices:
     - `v_peak = vel_window[idx_peak]` (First Maxima)
     - `v_plat = vel_window[idx_plat]` (Plateau velocity)
     - `v_min = vel_window[idx_min]` (Minima)
     - `v_recomp = vel_window[idx_recomp]` (Second Maxima)
   - Store in results dictionary
   - Calculate pullback velocity uncertainty from plateau and valley uncertainties

2. If `rdp_keys` is None (DNS case or legacy method):
   - Fallback: Use basic max/min:
     - `First_Maxima_m_s = max(vel_window)`
     - `Minima_m_s = min(vel_window)`
     - `Second_Maxima_m_s = NaN`
     - `Pullback_Velocity_m_s = NaN`

**Code Location**: Lines 564-599

---

### Step 6: Spall Analysis - Choose Method

**Purpose**: Perform detailed spall analysis using selected method.

**Methods Available**:
1. **RDP Method** (`spall_detection_method == 'rdp'`): RDP-guided 5-segment fitting
2. **Horizontal Plateau Method** (`spall_detection_method == '5-segment'`): Horizontal plateau constraint with 5-segment fitting

**Code Location**: Lines 601-772

---

### Step 6a: RDP-Guided 5-Segment Analysis (`analyze_spall_rdp_5_segment`)

**Purpose**: Perform 5-segment linear analysis using RDP vertices to define segment boundaries.

**Algorithm**:
1. **RDP Simplification**:
   - Apply RDP with `epsilon = 5.0 m/s`
   - Require at least 4 vertices after simplification

2. **Feature Mapping** (Identify P1, P2, P3, P4):
   - **P1 (Shock Peak)**: Global maximum in RDP array
     - `idx_p1_loc = argmax(rdp_vel)`
     - `idx_p1_global = rdp_indices[idx_p1_loc]`
   
   - **P3 (Pullback Minimum)**: Deepest valley after P1
     - `post_peak_indices = rdp_indices[idx_p1_loc:]`
     - `post_peak_vel = velocity[post_peak_indices]`
     - `idx_min_loc = argmin(post_peak_vel)`
     - `idx_p3_global = post_peak_indices[idx_min_loc]`
     - **Validation**: P3 must not be the last point (require recompression)
   
   - **P2 (Plateau Knee)**: Last vertex between P1 and P3
     - `intermediate_indices = [i for i in rdp_indices if idx_p1_global < i < idx_p3_global]`
     - If no intermediate vertices: `idx_p2_global = idx_p1_global` (triangular wave case)
     - Otherwise: `idx_p2_global = intermediate_indices[-1]` (last vertex before drop)
   
   - **P4 (Recompression Peak)**: Maximum velocity after P3
     - `post_p3_indices = rdp_indices[p3_loc_in_rdp+1:]`
     - `post_p3_vel = velocity[post_p3_indices]`
     - `idx_max_recomp_loc = argmax(post_p3_vel)`
     - `idx_p4_global = post_p3_indices[idx_max_recomp_loc]`
   
   - **Recompression Validation**:
     - `pullback_mag = v_p2 - v_p3`
     - `rebound_mag = v_p4 - v_p3`
     - Require: `rebound_mag >= pullback_mag * min_recomp_ratio` (default: ≥10%)
     - If validation fails: Return DNS

3. **Segment-wise Linear Fits**:
   - **Segment 1 (Rise)**: From start (index 0) to P1
   - **Segment 2 (Plateau)**: From P1 to P2 (may be zero-length if P1==P2)
   - **Segment 3 (Release)**: From P2 to P3 (pullback)
   - **Segment 4 (Recompression)**: From P3 to P4 (rebound)
   - **Segment 5 (Tail)**: From P4 to end
   
   Each segment fit:
   ```python
   t_seg = time[start_idx:end_idx+1]
   v_seg = velocity[start_idx:end_idx+1]
   m, c = np.polyfit(t_seg, v_seg, 1)  # Linear regression
   ```

4. **Calculate Physics Results**:
   - **Pullback Velocity**: `spall_strength_velocity = v_p2 - v_p3` (m/s)
   - **Spall Strength**: `σ_spall = 0.5 * ρ * c * Δu_p / 1e9` (GPa)
     - Where: `ρ` = density (kg/m³), `c` = acoustic velocity (m/s), `Δu_p` = pullback velocity (m/s)
   - **Plateau Mean Velocity**: Average velocity between P1 and P2 (or just P1 if P1==P2)
   - **Strain Rate**: `ε̇ = |release_slope| / c` (s⁻¹)
     - Where: `release_slope` = slope of Segment 3 (m/s per ns), converted to m/s per s
   - **Peak Shock Stress**: `σ_shock = ρ * c * u_plateau / 1e9` (GPa)

**Returns**:
```python
{
    'is_spall': True,
    'indices': {'P1': idx_p1, 'P2': idx_p2, 'P3': idx_p3, 'P4': idx_p4},
    'velocities': {'P1': v_p1, 'P2': v_p2, 'P3': v_p3, 'P4': v_p4},
    'times': {'P1': t_p1, 'P2': t_p2, 'P3': t_p3, 'P4': t_p4},
    'fits': {
        'seg1_rise': {'m': m1, 'c': c1, ...},
        'seg2_plateau': {'m': m2, 'c': c2, ...},
        'seg3_release': {'m': m3, 'c': c3, ...},
        'seg4_recomp': {'m': m4, 'c': c4, ...},
        'seg5_tail': {'m': m5, 'c': c5, ...}
    },
    'pullback_velocity': spall_strength_velocity,
    'pullback_mag': pullback_mag,
    'rebound_mag': rebound_mag
}
```

**Code Location**: Lines 7142-7264

---

### Step 6b: Horizontal Plateau 5-Segment Analysis (`analyze_spall_horizontal_plateau`)

**Purpose**: Perform 5-segment analysis with horizontal plateau constraint to prevent P1/P2 shifting.

**Algorithm**:
1. **Find Global Peak**:
   - `idx_max = argmax(velocity)`
   - `v_max = velocity[idx_max]`
   - `t_max = time[idx_max]`

2. **Identify Plateau Region**:
   - Threshold: `threshold_val = v_max * plateau_threshold` (default: 95% of peak)
   - Find all points where `velocity >= threshold_val`
   - Calculate **mean plateau velocity**: `v_plateau_mean = mean(velocity[plateau_indices])`
   - Identify:
     - `idx_first_plateau`: First (earliest) point in plateau region
     - `idx_last_plateau`: Last (latest) point in plateau region

3. **Line 1 (Rise)**: From origin (0,0) to first plateau point
   - Slope: `m1 = v_first_plateau / t_first_plateau`
   - Intercept: `c1 = 0.0` (line passes through origin)

4. **Line 2 (Plateau)**: Horizontal line at mean plateau velocity
   - Slope: `m2 = 0.0`
   - Intercept: `c2 = v_plateau_mean`

5. **Find P3 (Pullback Minimum)**:
   - Search after last plateau point: `post_plateau_data = velocity[idx_last_plateau:]`
   - Apply smoothing: `smooth_window = min(10, len(post_plateau_data) // 5)`
   - Use `scipy.signal.find_peaks` on inverted signal to find valleys
   - **Validation Steps**:
     - **Step 1**: Check ±1 ns window around candidate minimum
       - Verify candidate is actually the minimum in the ±1 ns window
       - Require velocity range in window > threshold (0.5 m/s or 1% of value)
     - **Step 2**: Check if velocity continues dropping after minimum
       - Check velocity at t+10 ns
       - If velocity drops > threshold (2 m/s or 5% of value), reject (still on downward slope)
   - Accept first validated minimum as P3
   - If no validated minimum found, use fallback: first turning point where velocity stops decreasing

6. **Line 3 (Release/Pullback)**: From P2 (last plateau point) to P3
   - Slope: `m3 = (v_p3 - v_plateau_mean) / (t_p3 - t_last_plateau)`
   - Intercept: `c3 = v_plateau_mean - m3 * t_last_plateau`

7. **DNS Check 1: P3 Too Close to Zero**:
   - If `|v_p3| <= 10.0 m/s`: Classify as DNS
   - Still calculate fits and intersections for visualization

8. **Find P4 (Recompression Peak)**:
   - Search after P3: `recomp_search = velocity[idx_p3+1:]`
   - `idx_p4_local = argmax(recomp_search)`
   - `idx_p4 = idx_p3 + 1 + idx_p4_local`
   - `v_p4 = velocity[idx_p4]`
   - `t_p4 = time[idx_p4]`

9. **DNS Check 2: Recompression Validation**:
   - **P3 Validation**: Require a subsequent maximum (recompression) at least **2.5 ns** after P3
   - Require recompression velocity ≥ **10% higher** than P3: `v_p4 >= v_p3 * 1.1`
   - If validation fails: Classify as DNS ("Did Not Spall")

10. **Line 4 (Recompression)**: From P3 to P4
    - Slope: `m4 = (v_p4 - v_p3) / (t_p4 - t_p3)`
    - Intercept: `c4 = v_p3 - m4 * t_p3`

11. **Find Next Minimum After P4** (for tail segment):
    - Search after P4: `post_p4 = velocity[idx_p4+1:]`
    - Apply smoothing and find valleys using `find_peaks` on inverted signal
    - If no valley found, use last point as fallback

12. **Line 5 (Tail)**: From P4 to next minimum
    - Slope: `m5 = (v_next_min - v_p4) / (t_next_min - t_p4)`
    - Intercept: `c5 = v_p4 - m5 * t_p4`

13. **Calculate Physics Results**:
    - **Pullback Velocity**: `Δu_p = v_plateau_mean - v_p3` (m/s)
    - **Spall Strength**: `σ_spall = 0.5 * ρ * c * Δu_p / 1e9` (GPa)
    - **Plateau Mean Velocity**: `v_plateau_mean` (m/s)
    - **Peak Shock Stress**: `σ_shock = ρ * c * v_plateau_mean / 1e9` (GPa)
    - **Strain Rate**: `ε̇ = |m3| * 1e9 / c` (s⁻¹)
      - Where: `m3` = slope of Line 3 (m/s per ns), converted to m/s per s

**Returns**:
```python
{
    'Processing Status': 'Success' or 'DNS',
    'Spall Strength (GPa)': σ_spall,
    'Strain Rate (s^-1)': strain_rate,
    'Peak Shock Stress (GPa)': σ_shock,
    'Plateau Mean Velocity (m/s)': v_plateau_mean,
    'First Maxima (m/s)': v_plateau_mean,
    'Minima (m/s)': v_p3,
    'Second Maxima (m/s)': v_p4,
    'Pullback Velocity (m/s)': Δu_p,
    'fits': {
        'seg1_rise': {'m': m1, 'c': c1, ...},
        'seg2_plateau': {'m': m2, 'c': c2, ...},
        'seg3_release': {'m': m3, 'c': c3, ...},
        'seg4_recomp': {'m': m4, 'c': c4, ...},
        'seg5_tail': {'m': m5, 'c': c5, ...}
    },
    'intersections': [(t_p1, v_p1), (t_p2, v_p2), (t_p3, v_p3), (t_p4, v_p4)]
}
```

**Code Location**: Lines 7266-7665

---

### Step 7: Uncertainty Calculation

**Purpose**: Calculate uncertainties for spall strength and other parameters.

**Process**:
1. **Pullback Velocity Uncertainty**:
   - If available from RDP keys: `pullback_unc = sqrt(peak_unc² + valley_unc²)`
   - Otherwise: Use uncertainty from result dictionary or NaN

2. **Spall Strength Uncertainty**:
   - If available from result dictionary: Use directly
   - Otherwise, propagate from pullback velocity:
     - `spall_unc = 0.5 * density * acoustic_velocity * pullback_unc / 1e9` (GPa)

3. **Strain Rate Uncertainty**:
   - Extract from result dictionary if available
   - Otherwise: NaN

**Code Location**: Lines 821-845

---

### Step 8: Peak Shock Stress Calculation (EOS Method)

**Purpose**: Calculate peak shock stress using Hugoniot EOS (Equation of State).

**Method**: Uses Hugoniot EOS: `U = c + S * u_p`, then `σ = ρ * U * u_p`

**Process**:
1. Extract plateau velocity from results
2. Get material properties (S parameter) from config or defaults:
   - Cu/Copper: S = 1.49
   - Zn/Zinc: S = 1.30
   - Al/Aluminum: S = 1.34
   - Brass: S = 1.43
   - Default: S = 1.49

3. Calculate:
   - **Particle velocity**: `u_p = plateau_velocity / 2.0` (free surface velocity / 2)
   - **Shock velocity**: `U = acoustic_velocity + S * u_p`
   - **Peak shock stress**: `σ_shock = ρ * U * u_p * 1e-9` (GPa)

4. **Uncertainty Propagation** (if velocity uncertainty available):
   - `u_p_unc = velocity_unc / 2.0`
   - `σ_shock_unc = ρ * (c + 2 * S * u_p) * u_p_unc * 1e-9` (GPa)

**Code Location**: Lines 847-893

---

### Step 9: Final Classification

**Purpose**: Set final DNS classification and status flags.

**Process**:
1. **DNS Cases** (`is_dns == True`):
   - `Spall_Strength_GPa = "DNS"`
   - `Spall_Strength_Unc_GPa = NaN`
   - `Spall_OK = False`
   - `Processing_Status = f'DNS: {dns_reason}'`
   - Still store plateau velocity and shock stress (if available)

2. **Valid Spall Cases** (`is_dns == False`):
   - `Spall_OK = True`
   - `Processing_Status = 'Success'`
   - `DNS_Classification = 'Valid Spall'`
   - Store SPADE-calculated spall strength, strain rate, shock stress

**Code Location**: Lines 895-937

---

### Step 10: Plot Generation

**Purpose**: Generate visualization plot with analysis results.

**Process** (if `plot_path` is provided):
1. Call `_plot_generic_spall_analysis()` with:
   - `time_window`, `vel_window`, `uncert_window`
   - `peak_idx`, `valley_idx` (from RDP keys if available)
   - `spall_strength`, `spall_unc`
   - `lines_info`: 5-segment fits
   - `intersections`: P1, P2, P3, P4 points
   - `rdp_keys`: RDP-detected key points (for visualization)
   - `rdp_points`: RDP simplified trace (for visualization)

2. Plot includes:
   - Raw velocity trace (blue line)
   - Uncertainty bands (shaded region)
   - RDP simplified points (purple markers, if RDP method used)
   - RDP-detected key points (red star: peak, magenta circle: plateau end, green circle: minimum, cyan circle: recompression)
   - 5-segment fitted lines (colored lines)
   - Intersection points (P1, P2, P3, P4)
   - Text annotations with spall strength and strain rate

**Code Location**: Lines 939-977

---

## Key Points and Features

### DNS (Did Not Spall) Classification Reasons

1. **RDP Method DNS Reasons**:
   - `"DNS: Trace too short/simple after RDP"` - Insufficient vertices after simplification
   - `"DNS: Trace ends before spall signature"` - Peak is at the end, no data after
   - `"DNS: No valid Pullback-Recompression signature found (RDP)"` - No valid checkmark shape found
   - `"DNS: Insufficient RDP vertices (need at least 4)"` - Not enough vertices for 5-segment analysis
   - `"DNS: No data after peak"` - No data after shock peak
   - `"DNS: No recompression features (Minima is at end)"` - P3 is at the end
   - `"DNS: Rebound too small"` - Recompression magnitude < 10% of pullback

2. **Horizontal Plateau Method DNS Reasons**:
   - `"DNS: No data after peak"` - No data after plateau
   - `"DNS: P3 too close to zero"` - Pullback minimum velocity ≤ 10 m/s
   - `"DNS: Did Not Spall"` - P3 validation failed (no recompression ≥10% higher than P3, or recompression < 2.5 ns after P3)

### Configuration Parameters

**RDP Detection Parameters**:
- `spall_rdp_epsilon`: RDP tolerance (default: 5.0 m/s)
- `min_pullback_velocity`: Minimum pullback magnitude (default: 10.0 m/s)
- `min_recomp_ratio`: Minimum rebound ratio (default: 0.1, i.e., 10%)

**Horizontal Plateau Parameters**:
- `plateau_threshold`: Plateau threshold ratio (default: 0.95, i.e., 95% of peak)
- `min_recomp_ratio`: Minimum recompression ratio (default: 0.1, i.e., 10%)
- `min_recomp_time_ns`: Minimum time between P3 and P4 (default: 2.5 ns)
- `min_recomp_velocity_ratio`: Minimum recompression velocity ratio (default: 1.1, i.e., 10% higher)

**Alignment Parameters**:
- `use_hel_t0_alignment_for_plots`: Use HEL t=0 alignment (default: True)
- `minimum_HEL_velocity_expected`: Minimum velocity threshold for HEL alignment (default: 10.0 m/s)

**Window Parameters**:
- `spall_start_time`: Start of spall analysis window (ns)
- `spall_end_time`: End of spall analysis window (ns)

### Physics Formulas

1. **Spall Strength**:
   ```
   σ_spall = 0.5 * ρ * c * Δu_p / 1e9  (GPa)
   ```
   Where:
   - `ρ` = density (kg/m³)
   - `c` = acoustic velocity (m/s)
   - `Δu_p` = pullback velocity (m/s) = `v_plateau - v_minimum`

2. **Strain Rate**:
   ```
   ε̇ = |release_slope| / c  (s⁻¹)
   ```
   Where:
   - `release_slope` = slope of release segment (m/s per ns), converted to m/s per s
   - `c` = acoustic velocity (m/s)

3. **Peak Shock Stress (EOS Method)**:
   ```
   U = c + S * u_p
   σ_shock = ρ * U * u_p / 1e9  (GPa)
   ```
   Where:
   - `U` = shock velocity (m/s)
   - `c` = acoustic velocity (m/s)
   - `S` = Hugoniot slope parameter (dimensionless)
   - `u_p` = particle velocity (m/s) = `u_fs / 2`
   - `u_fs` = free surface velocity (m/s) = plateau velocity

---

## Code Locations

- **Main Entry Point**: `detect_dns_and_process_spall()` - Lines 369-986
- **RDP Detection**: `detect_spall_rdp()` - Lines 7887-7996
- **RDP-Guided 5-Segment**: `analyze_spall_rdp_5_segment()` - Lines 7142-7264
- **Horizontal Plateau 5-Segment**: `analyze_spall_horizontal_plateau()` - Lines 7266-7665
- **Plotting**: `_plot_generic_spall_analysis()` - Lines 989-1080
- **RDP Algorithm**: `ramer_douglas_peucker_indices()` - (referenced, implementation in HEL detection code)

---

## Example Usage

```python
# Configuration
config = {
    'spall_detection_method': 'rdp',  # or '5-segment'
    'spall_rdp_epsilon': 5.0,
    'min_pullback_velocity': 10.0,
    'min_recomp_ratio': 0.1,
    'spall_start_time': 10.0,
    'spall_end_time': 200.0,
    'use_hel_t0_alignment_for_plots': True,
    'minimum_HEL_velocity_expected': 10.0
}

# Call detection
results = detect_dns_and_process_spall(
    file_path='velocity_data.csv',
    base_name='shot_001',
    density=8960.0,  # Cu density (kg/m³)
    acoustic_velocity=3940.0,  # Cu acoustic velocity (m/s)
    threshold_velocity=30.0,  # m/s
    spall_start_time=10.0,  # ns
    spall_end_time=200.0,  # ns
    analysis_model='hybrid',
    plot_path='spall_plot.png',
    sample_material='Cu',
    **config
)

# Check results
if results['Spall_OK']:
    print(f"Spall Strength: {results['Spall_Strength_GPa']:.3f} GPa")
    print(f"Strain Rate: {results['Spall_StrainRate_s^-1']:.2e} s⁻¹")
else:
    print(f"DNS: {results['DNS_Classification']}")
```

---

## References

- Ramer-Douglas-Peucker Algorithm: Line simplification algorithm for geometric feature detection
- Hugoniot EOS: Equation of State for shock wave propagation
- Impedance Matching: Method for calculating spall strength from pullback velocity
- PDV (Photonic Doppler Velocimetry): Experimental technique for measuring free surface velocity

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Codebase**: HELIX Toolbox v2
