# SPALL Detection Algorithm — 5-Segment Only (No RDP)

## Overview

This document describes the spall detection flow when **`"spall_detection_method": "5-segment"`** is set in the configuration. In this mode:

- **No RDP (Ramer–Douglas–Peucker)** is used: no topology check, no RDP-guided segment boundaries, no RDP overlay on plots.
- **Spall vs DNS** is decided only by the **Horizontal Plateau 5-Segment** method (`analyze_spall_horizontal_plateau`).
- The same t=0 alignment, data loading, windowing, uncertainty, shock stress, and final classification as the full algorithm still apply.

**Config**: In `helix_master_config.json` (or equivalent), set:
```json
"spall_detection_method": "5-segment"
```

---

## Algorithm Flow (5-Segment Only)

### Main Entry: `detect_dns_and_process_spall`

Same as the full algorithm: same inputs (e.g. `file_path`, `base_name`, `density`, `acoustic_velocity`, `threshold_velocity`, `spall_start_time`, `spall_end_time`, etc.) and same return structure (spall strength, DNS classification, strain rate, shock stress, etc.).

**Code location**: Lines 369–986 in `helix_analysis_toolbox.py`.

---

## Step-by-Step (5-Segment Path Only)

### Step 1: Data Loading and Header Detection

- Same as full algorithm: load CSV, detect header (time/velocity/uncertainty), take first three columns (time in s, velocity in m/s, uncertainty in m/s).
- Require ≥ 3 columns and ≥ 20 rows; convert to numeric with NaN handling.

**Code location**: Lines 404–433.

---

### Step 1b: Uncertainty Filtering

- Same as full algorithm: `rel_unc = |uncertainty| / max(|velocity|, 1e-9)`; set `velocity[i] = NaN` where `rel_unc >= 1.0`.

**Code location**: Lines 435–439.

---

### Step 2: Trace Alignment to Shock Arrival (t=0)

- Same as full algorithm:
  - **Primary**: HEL t=0 alignment (`find_hel_t0_alignment`) — first point with velocity above threshold and sustained increase over a 10 ns window.
  - **Fallback**: First point where `velocity >= threshold_velocity`.
- Produces `t_aligned_ns = (time_s - t0) * 1e9`.

**Code location**: Lines 441–485.

---

### Step 3: Time Window Extraction

- Same as full algorithm: mask by `t_aligned_ns` in `[spall_start_time, spall_end_time]` and non-NaN velocity; extract `time_window`, `vel_window`, `uncert_window`.
- Require ≥ 20 points in window.

**Code location**: Lines 491–499.

---

### Step 4 (5-Segment Only): No RDP — Skip to Key Velocities

When `spall_detection_method == '5-segment'`:

- **RDP is not run**: no `ramer_douglas_peucker_indices`, no `detect_spall_rdp`.
- Logic sets: `is_spall_rdp = True`, `rdp_reason = "Skipped (using legacy 5-segment)"`, `rdp_keys = None`, `rdp_points = None`.
- **No pre-classification** from RDP; DNS is decided only later by the Horizontal Plateau method.

**Code location**: Lines 519–562 (branch where `spall_detection_method != 'rdp'`).

---

### Step 5: Extract Key Velocities (Fallback Only)

Because `rdp_keys` is always `None` in 5-segment mode:

- **First_Maxima_m_s** = `max(vel_window)`
- **Minima_m_s** = `min(vel_window)`
- **Second_Maxima_m_s** = NaN
- **Pullback_Velocity_m_s** = NaN  
(These are diagnostics only; actual spall/DNS and pullback come from the Horizontal Plateau step.)

**Code location**: Lines 564–599 (else branch when `rdp_keys` is None).

---

### Step 6: Spall Analysis — Horizontal Plateau 5-Segment Only

With `spall_detection_method == '5-segment'`, only **Horizontal Plateau 5-Segment** runs:

- Call: `analyze_spall_horizontal_plateau(time_window, vel_window, uncert_window, density, acoustic_velocity, config)`.
- This function both **detects** spall vs DNS and **fits** the 5 segments. No RDP-guided fit is used.

**Code location**: Lines 695–772 (branch `spall_detection_method == '5-segment'`).

---

### Step 6b: Horizontal Plateau 5-Segment Algorithm (Full Detail)

This is the only detection/fitting method in 5-segment mode.

1. **Global peak**
   - `idx_max = argmax(velocity)`, `v_max = velocity[idx_max]`, `t_max = time[idx_max]`.

2. **Plateau region**
   - Threshold: `threshold_val = v_max * plateau_threshold` (default 0.95).
   - All points with `velocity >= threshold_val`; mean velocity `v_plateau_mean` over these points.
   - `idx_first_plateau` = first index in this set, `idx_last_plateau` = last index.

3. **Line 1 (Rise)**
   - From (0, 0) to first plateau point: `m1 = v_first_plateau / t_first_plateau`, `c1 = 0`.

4. **Line 2 (Plateau)**
   - Horizontal: `m2 = 0`, `c2 = v_plateau_mean`.

5. **P3 (pullback minimum)**
   - Search after `idx_last_plateau`: smooth post-plateau velocity, find valleys (e.g. `find_peaks` on `-velocity`).
   - Validation:
     - In a ±1 ns window, candidate must be the minimum and range must exceed a small threshold.
     - At t+10 ns, velocity must not drop by more than 2 m/s or 5% of P3 (else reject as still on downward slope).
   - First validated minimum = P3; fallback: first turning point where velocity stops decreasing.

6. **Line 3 (Release)**
   - From last plateau point (P2) to P3: `m3 = (v_p3 - v_plateau_mean) / (t_p3 - t_last_plateau)`, `c3 = v_plateau_mean - m3 * t_last_plateau`.

7. **DNS check 1: P3 too close to zero**
   - If `|v_p3| <= 10.0` m/s → classify DNS. Still compute fits/intersections for plotting.

8. **P4 (recompression peak)**
   - After P3: `idx_p4 = idx_p3 + 1 + argmax(velocity[idx_p3+1:])`, `v_p4`, `t_p4`.

9. **DNS check 2: Recompression**
   - Require a maximum at least **2.5 ns** after P3 and **v_p4 >= v_p3 * 1.1** (10% higher).
   - If not satisfied → DNS ("Did Not Spall").

10. **Line 4 (Recompression)**  
    - From P3 to P4: `m4 = (v_p4 - v_p3) / (t_p4 - t_p3)`, `c4 = v_p3 - m4 * t_p3`.

11. **Next minimum after P4**  
    - For tail: search after P4 (smoothed, valleys); if none, use last point.

12. **Line 5 (Tail)**  
    - From P4 to that next minimum: `m5`, `c5` from the two endpoints.

13. **Physics**
    - Pullback: `Δu_p = v_plateau_mean - v_p3`.
    - Spall strength: `σ_spall = 0.5 * ρ * c * Δu_p / 1e9` (GPa).
    - Plateau mean velocity: `v_plateau_mean`.
    - Peak shock stress (simple): `σ_shock = ρ * c * v_plateau_mean / 1e9` (GPa).
    - Strain rate: `ε̇ = |m3| * 1e9 / c` (s⁻¹).

**Returns**: Same result dictionary as in the main doc (Processing Status, Spall Strength, Strain Rate, Peak Shock Stress, Plateau Mean Velocity, First Maxima, Minima, Second Maxima, Pullback Velocity, fits, intersections).

**Code location**: Lines 7266–7665.

---

### Step 7: Uncertainty Calculation

- Same as full algorithm: pullback uncertainty from result dict or NaN; spall strength uncertainty from result or propagation from pullback; strain rate uncertainty from result or NaN.
- In 5-segment mode there are no RDP key indices, so pullback uncertainty typically comes from the Horizontal Plateau result if provided.

**Code location**: Lines 821–845.

---

### Step 8: Peak Shock Stress (EOS)

- Same as full algorithm: use plateau velocity, material S parameter, `u_p = plateau_velocity/2`, `U = c + S*u_p`, `σ_shock = ρ*U*u_p*1e-9` (GPa), with optional uncertainty propagation.

**Code location**: Lines 847–893.

---

### Step 9: Final Classification

- **If Horizontal Plateau returned DNS** (`is_dns` set from `plat_reason`):  
  `Spall_Strength_GPa = "DNS"`, `Spall_OK = False`, `Processing_Status = 'DNS: {reason}'`, etc.; plateau/shock stress still stored if available.
- **If Horizontal Plateau returned valid spall**:  
  `Spall_OK = True`, `Processing_Status = 'Success'`, `DNS_Classification = 'Valid Spall'`, SPADE spall strength, strain rate, shock stress stored.

**Code location**: Lines 895–937.

---

### Step 10: Plot Generation

- Same `_plot_generic_spall_analysis()` call, but with **`rdp_keys=None`** and **`rdp_points=None`**.
- So the plot shows:
  - Raw velocity and uncertainty band
  - **No** RDP simplified trace
  - **No** RDP key-point markers (peak/plateau/min/recomp from RDP)
- It still shows the 5-segment lines and P1–P4 intersections from the Horizontal Plateau fit, and any peak/valley from basic max/min if used for fallback drawing.

**Code location**: Lines 939–977, 989–1136 (RDP branches are skipped when `rdp_keys`/`rdp_points` are None).

---

## What Does *Not* Run in 5-Segment Mode

- **RDP simplification** on the spall window (no `ramer_douglas_peucker_indices` for spall).
- **`detect_spall_rdp()`** — no topology “checkmark” scan.
- **`analyze_spall_rdp_5_segment()`** — no RDP-guided P1–P4 or segment fits.
- **RDP overlay** on the plot (no purple RDP curve, no RDP peak/plateau/min/recomp markers).

---

## DNS Reasons (5-Segment Only)

Only the Horizontal Plateau method can set DNS:

- `"DNS: No data after peak"` — no data after the plateau.
- `"DNS: P3 too close to zero ({v_p3} m/s, threshold=±10 m/s)"` — pullback minimum ≤ 10 m/s.
- `"DNS: Did Not Spall"` — recompression validation failed (P4 not ≥ 2.5 ns after P3 or `v_p4 < v_p3 * 1.1`).

There are no RDP-based DNS messages (e.g. no “Trace too short after RDP”, “No valid Pullback-Recompression signature (RDP)”, etc.).

---

## Configuration (5-Segment Only)

Relevant config when using **only** 5-segment:

- **Method**: `"spall_detection_method": "5-segment"`.
- **Horizontal plateau**: `plateau_threshold` (default 0.95), `min_recomp_ratio`, `min_recomp_time_ns` (e.g. 2.5 ns), `min_recomp_velocity_ratio` (e.g. 1.1).
- **Alignment**: `use_hel_t0_alignment_for_plots`, `minimum_HEL_velocity_expected`.
- **Window**: `spall_start_time`, `spall_end_time`.

RDP-related keys (`spall_rdp_epsilon`, `min_pullback_velocity`, etc.) are **not** used in this path.

---

## Physics Formulas (Unchanged)

- Spall strength: `σ_spall = 0.5 * ρ * c * Δu_p / 1e9` (GPa).
- Strain rate: `ε̇ = |m3| * 1e9 / c` (s⁻¹).
- Peak shock stress (EOS): `U = c + S*u_p`, `σ_shock = ρ*U*u_p*1e-9` (GPa).

---

## Example Config and Usage

**Config** (e.g. in `helix_master_config.json`):

```json
"spall_detection_method": "5-segment"
```

Optional (same as full algorithm where applicable):

```json
"plateau_threshold": 0.95,
"spall_start_time": 10.0,
"spall_end_time": 200.0,
"use_hel_t0_alignment_for_plots": true,
"minimum_HEL_velocity_expected": 10.0
```

**Usage**: Same as full pipeline — run with the above config; spall detection and fitting are done only by the Horizontal Plateau 5-segment method, with no RDP.

---

**Document version**: 1.0  
**Variant**: 5-segment only (no RDP)  
**Codebase**: HELIX Toolbox v2
