# spade_analysis/data_processing.py
"""
Functions for processing raw velocity-time data.
Supports multiple analysis models and user-defined material properties.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
import os
import warnings
import traceback
import logging
import glob
from . import utils

# Setup logger for this module
logger = logging.getLogger(__name__)


def _load_and_clean_data(file_path):
    """
    Loads a CSV file, cleans it, and automatically converts time from s to ns if needed.
    
    Returns a clean DataFrame with 'Time', 'Velocity', and 'Uncertainty' columns (if available).
    """
    try:
        df = pd.read_csv(file_path, header=None, on_bad_lines='skip', engine='python')
        if df.shape[1] < 2:
            logger.warning(f"Skipping file {os.path.basename(file_path)}: Needs at least 2 columns (Time, Velocity).")
            return None
        
        # Handle different column configurations
        if df.shape[1] >= 4:
            # Has 4 columns: Time, Velocity, Uncertainty, Velocity+Uncertainty
            df.columns = ['Time', 'Velocity', 'Uncertainty', 'Velocity_Plus_Uncertainty'] + [f'col_{i}' for i in range(4, df.shape[1])]
            df = df[['Time', 'Velocity', 'Uncertainty']]  # Keep only what we need
        elif df.shape[1] == 3:
            # Has 3 columns: Time, Velocity, Uncertainty (legacy format)
            df.columns = ['Time', 'Velocity', 'Uncertainty'] + [f'col_{i}' for i in range(3, df.shape[1])]
            df = df[['Time', 'Velocity', 'Uncertainty']]
        else:
            # Only has Time and Velocity
            df.columns = ['Time', 'Velocity']
            df['Uncertainty'] = np.nan  # Add empty uncertainty column
        
        # Convert to numeric, coercing errors will turn non-numeric headers into NaT/NaN
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['Velocity'] = pd.to_numeric(df['Velocity'], errors='coerce')
        df['Uncertainty'] = pd.to_numeric(df['Uncertainty'], errors='coerce')
        df.dropna(subset=['Time', 'Velocity'], inplace=True)  # Keep rows with valid time/velocity even if uncertainty is NaN
        
        # Check if time is in seconds (very small values) and convert to nanoseconds.
        # If the maximum time value is less than 1.0, it's almost certainly in seconds.
        if not df.empty and df['Time'].max() < 1.0:
            logger.debug(f"Time data for {os.path.basename(file_path)} appears to be in seconds, converting to ns.")
            df['Time'] *= 1e9

        return df.sort_values('Time').reset_index(drop=True)
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Critical error loading file {file_path}: {e}")
        return None


# --- Helper Functions ---
def _get_interp_y(x_data, y_data, x_target, kind='linear'):
    """ Safely interpolates y value at x_target from x_data, y_data. """
    if not isinstance(x_data, (pd.Series, np.ndarray)) or not isinstance(y_data, (pd.Series, np.ndarray)): return np.nan
    if len(x_data) < 2: return np.nan
    try:
        x_vals = x_data.values if isinstance(x_data, pd.Series) else np.asarray(x_data)
        y_vals = y_data.values if isinstance(y_data, pd.Series) else np.asarray(y_data)
        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if finite_mask.sum() < 2: return np.nan

        x_vals = x_vals[finite_mask]
        y_vals = y_vals[finite_mask]
        sort_idx = np.argsort(x_vals)
        x_data_sorted = x_vals[sort_idx]
        y_data_sorted = y_vals[sort_idx]

        unique_x_mask = np.concatenate(([True], np.diff(x_data_sorted) > 1e-9))
        if unique_x_mask.sum() < 2: return np.nan
        x_unique = x_data_sorted[unique_x_mask]
        y_unique = y_data_sorted[unique_x_mask]

        interp_func = interp1d(x_unique, y_unique, kind=kind, bounds_error=False, fill_value="extrapolate")
        y_target = interp_func(x_target)
        if isinstance(y_target, np.ndarray): y_target = y_target.item(0)
        return float(y_target) if pd.notna(y_target) else np.nan
    except (ValueError, Exception): return np.nan

def _fit_line_to_range(x_data, y_data, x_start, x_end):
    """ Fits a line to data within a specified x range. """
    if not isinstance(x_data, pd.Series): x_data = pd.Series(x_data)
    if not isinstance(y_data, pd.Series): y_data = pd.Series(y_data)
    if pd.isna(x_start) or pd.isna(x_end) or x_start >= x_end: return np.nan, np.nan
    x_min_data, x_max_data = x_data.min(), x_data.max()
    x_start = max(x_start, x_min_data); x_end = min(x_end, x_max_data)
    if x_start >= x_end: return np.nan, np.nan
    mask = (x_data >= x_start) & (x_data <= x_end)
    x_subset = x_data[mask]; y_subset = y_data[mask]
    valid_subset = x_subset.notna() & y_subset.notna()
    if valid_subset.sum() < 2: return np.nan, np.nan
    try:
        coeffs = np.polyfit(x_subset[valid_subset], y_subset[valid_subset], 1)
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)): return np.nan, np.nan
        return coeffs[0], coeffs[1]
    except (np.linalg.LinAlgError, ValueError, Exception): return np.nan, np.nan

def _find_intersection(m1, c1, m2, c2):
    """ Finds the intersection point (x, y) of two lines y=m1*x+c1 and y=m2*x+c2. """
    if np.isnan(m1) or np.isnan(c1) or np.isnan(m2) or np.isnan(c2): return np.nan, np.nan
    if np.isinf(m1) and np.isinf(m2): return np.nan, np.nan
    if not np.isinf(m1) and not np.isinf(m2) and np.isclose(m1, m2): return np.nan, np.nan
    if np.isinf(m1): x_intersect = c1; y_intersect = m2 * x_intersect + c2 if not np.isinf(m2) else np.nan
    elif np.isinf(m2): x_intersect = c2; y_intersect = m1 * x_intersect + c1 if not np.isinf(m1) else np.nan
    else: x_intersect = (c2 - c1) / (m1 - m2); y_intersect = m1 * x_intersect + c1
    if not (-500 < x_intersect < 1000): return np.nan, np.nan
    return x_intersect, y_intersect

# --- Plotting Function ---
def _plot_analysis_results(data_dict, lines_info, intersections, output_path):
    """ Plots trace, smoothed data, fitted lines, and intersections. """
    if not output_path:
        return
    import matplotlib.pyplot as plt
    x_trace = data_dict.get('x_shifted', pd.Series(dtype=float))
    y_trace = data_dict.get('y_original', pd.Series(dtype=float))
    y_smooth = data_dict.get('y_smooth', pd.Series(dtype=float))
    uncertainty = data_dict.get('uncertainty', pd.Series(dtype=float))
    filename = data_dict.get('filename', 'Unknown Filename')
    base_filename = os.path.splitext(filename)[0]

    if x_trace.empty or y_trace.empty:
        logger.warning(f"Skipping plot for {filename}: No valid trace data.")
        return

    logger.debug(f"    Generating model plot for {base_filename}")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Plot uncertainty bands if available
        if not uncertainty.empty and not uncertainty.isna().all():
            ax.fill_between(x_trace, y_trace - uncertainty, y_trace + uncertainty, 
                          alpha=0.3, color='lightblue', label='Uncertainty')
        
        ax.plot(x_trace, y_trace, label='Original Data', color='grey', alpha=0.6, lw=1.0)
        if not y_smooth.empty:
            ax.plot(x_trace, y_smooth, label='Smoothed Data', color='black', alpha=0.8, lw=1.0, ls=':')

        if lines_info:
            colors = ['blue', 'green', 'red', 'purple', 'brown']
            labels = ['Line 1 (Rise)', 'Line 2 (Plateau)', 'Line 3 (Pullback)', 'Line 4 (Recomp Rise)', 'Line 5 (Recomp Tail)']
            for i, (m, c) in enumerate(lines_info):
                if pd.notna(m) and pd.notna(c):
                    x_start, x_end = ax.get_xlim()
                    x_line = np.linspace(x_start, x_end, 10)
                    y_line = m * x_line + c
                    ax.plot(x_line, y_line, color=colors[i], linestyle='--', lw=2, label=f'{labels[i]} (m={m:.2f})')
        
        if intersections:
            is_max_min_plot = not lines_info
            point_labels = ['Peak', 'Valley'] if is_max_min_plot else ['P1', 'P2', 'P3', 'P4']
            colors = ['#ff7f0e', '#1f77b4'] if is_max_min_plot else ['cyan', 'magenta', 'orange', 'lime']

            for i, (px, py) in enumerate(intersections):
                if pd.notna(px) and pd.notna(py) and i < len(point_labels):
                    ax.scatter([px], [py], label=f'{point_labels[i]} ({px:.1f}, {py:.1f})', 
                               s=100, zorder=5, edgecolors='black', color=colors[i])

        ax.set_xlabel('Time (ns)', fontsize=14)
        ax.set_ylabel('Velocity (m/s)', fontsize=14)
        ax.set_title(f'Analysis for: {base_filename}', fontsize=16)
        ax.legend(loc='best')
        ax.grid(True, linestyle=':')
        
        all_y = y_trace[np.isfinite(y_trace)]
        if not all_y.empty:
            min_y, max_y = np.nanmin(all_y), np.nanmax(all_y)
            ax.set_ylim(min_y - 50, max_y + 100)
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.debug(f"    Successfully saved plot: {os.path.basename(output_path)}")

    except Exception as plot_err:
        logger.exception(f"    ERROR during plot generation for {base_filename}: {plot_err}")
    finally:
        plt.close(fig)

# --- Feature Calculation Models ---

def _calculate_max_min_features(time_shifted, velocity_smoothed, density, acoustic_velocity, uncertainty_smoothed=None, **kwargs):
    """
    Calculates spall parameters by finding the first peak and the subsequent global minimum.
    """
    results = {}
    peaks, _ = find_peaks(velocity_smoothed)
    if not peaks.any():
        raise ValueError("No peaks found in smoothed signal.")
    idx_peak = peaks[0]

    signal_after_peak = velocity_smoothed[idx_peak + 1:]
    if len(signal_after_peak) == 0:
        raise ValueError("No signal data found after the initial peak.")
    
    relative_idx_min = np.argmin(signal_after_peak)
    idx_pullback = idx_peak + 1 + relative_idx_min

    peak_coords = (time_shifted[idx_peak], velocity_smoothed[idx_peak])
    valley_coords = (time_shifted[idx_pullback], velocity_smoothed[idx_pullback])
    
    results['First Maxima (m/s)'] = peak_coords[1]
    results['Pullback Minima (m/s)'] = valley_coords[1]
    results['Plateau Mean Velocity (m/s)'] = peak_coords[1]  # For max_min, plateau = peak
    
    delta_v = results['First Maxima (m/s)'] - results['Pullback Minima (m/s)']
    results['Spall Strength (GPa)'] = 0.5 * density * acoustic_velocity * delta_v * 1e-9
    
    # Calculate peak shock stress from plateau velocity
    results['Peak Shock Stress (GPa)'] = 0.5 * density * acoustic_velocity * results['Plateau Mean Velocity (m/s)'] * 1e-9
    
    time_diff_s = (time_shifted[idx_pullback] - time_shifted[idx_peak]) * 1e-9
    
    if time_diff_s <= 0:
        raise ValueError(f"Time difference for strain rate is non-positive ({time_diff_s}s). This may indicate an issue with the time data.")
        
    results['Strain Rate (s^-1)'] = (delta_v / time_diff_s) / (2 * acoustic_velocity)
    
    # Calculate uncertainties if available
    if uncertainty_smoothed is not None and not uncertainty_smoothed.isna().all():
        try:
            # Get uncertainties at peak and pullback points
            peak_uncertainty = uncertainty_smoothed.iloc[idx_peak] if idx_peak < len(uncertainty_smoothed) else np.nan
            pullback_uncertainty = uncertainty_smoothed.iloc[idx_pullback] if idx_pullback < len(uncertainty_smoothed) else np.nan
            
            # Propagate velocity uncertainty to spall strength uncertainty
            # Spall Strength = 0.5 * density * acoustic_velocity * delta_v * 1e-9
            # Uncertainty in delta_v = sqrt(peak_uncertainty^2 + pullback_uncertainty^2)
            if not pd.isna(peak_uncertainty) and not pd.isna(pullback_uncertainty):
                delta_v_uncertainty = np.sqrt(peak_uncertainty**2 + pullback_uncertainty**2)
                results['Spall Strength Uncertainty (GPa)'] = 0.5 * density * acoustic_velocity * delta_v_uncertainty * 1e-9
            else:
                results['Spall Strength Uncertainty (GPa)'] = np.nan
            
            # Propagate velocity uncertainty to strain rate uncertainty
            # Strain Rate = (delta_v / time_diff_s) / (2 * acoustic_velocity)
            # Uncertainty in strain rate = (delta_v_uncertainty / time_diff_s) / (2 * acoustic_velocity)
            if not pd.isna(peak_uncertainty) and not pd.isna(pullback_uncertainty):
                results['Strain Rate Uncertainty (s^-1)'] = (delta_v_uncertainty / time_diff_s) / (2 * acoustic_velocity)
            else:
                results['Strain Rate Uncertainty (s^-1)'] = np.nan
            
            # Propagate velocity uncertainty to peak shock stress uncertainty
            # Peak Shock Stress = 0.5 * density * acoustic_velocity * plateau_velocity * 1e-9
            # Uncertainty in peak shock stress = 0.5 * density * acoustic_velocity * peak_uncertainty * 1e-9
            if not pd.isna(peak_uncertainty):
                results['Peak Shock Stress Uncertainty (GPa)'] = 0.5 * density * acoustic_velocity * peak_uncertainty * 1e-9
            else:
                results['Peak Shock Stress Uncertainty (GPa)'] = np.nan
                
        except Exception as e:
            logger.warning(f"Could not calculate uncertainties: {e}")
            results['Spall Strength Uncertainty (GPa)'] = np.nan
            results['Strain Rate Uncertainty (s^-1)'] = np.nan
            results['Peak Shock Stress Uncertainty (GPa)'] = np.nan
    else:
        results['Spall Strength Uncertainty (GPa)'] = np.nan
        results['Strain Rate Uncertainty (s^-1)'] = np.nan
        results['Peak Shock Stress Uncertainty (GPa)'] = np.nan
    
    return results, [], [peak_coords, valley_coords]

def _calculate_hybrid_5_segment_features(time_shifted, velocity_smoothed, density, acoustic_velocity, uncertainty_smoothed=None, **kwargs):
    """
    Calculates spall parameters using a robust automated 5-segment line model.
    Line 1 is now fit from the detected initial rise (not always the first point) to the first peak.
    """
    prominence_factor = kwargs.get('prominence_factor', 0.05)
    peak_distance_ns = kwargs.get('peak_distance_ns', 5.0)

    if len(time_shifted) > 1:
        time_step = np.mean(np.diff(time_shifted))
        distance_samples = int(peak_distance_ns / time_step) if time_step > 0 else 1
    else:
        distance_samples = 1
    
    velocity_range = np.ptp(velocity_smoothed)
    prominence = velocity_range * prominence_factor
    
    logging.debug(f"Hybrid model: prominence={prominence:.2f}, distance_samples={distance_samples}")

    peaks, _ = find_peaks(velocity_smoothed, prominence=prominence, distance=distance_samples)
    if len(peaks) < 2:
        logging.warning("Hybrid model failed: Could not find at least two significant peaks.")
        raise ValueError("Could not find at least two significant peaks (initial and recompaction).")
    
    valleys, _ = find_peaks(-velocity_smoothed, prominence=prominence, distance=distance_samples)
    
    idx_peak1 = peaks[0]
    valleys_after_peak1 = valleys[valleys > idx_peak1]
    if not valleys_after_peak1.any():
        raise ValueError("No pullback minimum found after initial peak.")
    idx_pullback = valleys_after_peak1[0]
    
    peaks_after_pullback = peaks[peaks > idx_pullback]
    if not peaks_after_pullback.any():
        raise ValueError("No recompaction peak found after pullback.")
    idx_peak2 = peaks_after_pullback[0]

    # --- Improved Initial Rise Detection ---
    N_baseline = min(10, len(velocity_smoothed)//5)
    baseline = np.median(velocity_smoothed[:N_baseline])
    peak_val = velocity_smoothed[idx_peak1]
    threshold = baseline + max(0.05 * (peak_val - baseline), 10.0)  # 5% of peak or 10 m/s above baseline
    initial_rise_indices = np.where(velocity_smoothed > threshold)[0]
    if len(initial_rise_indices) > 0:
        idx_rise = initial_rise_indices[0]
    else:
        idx_rise = 0  # fallback to first point if no clear rise
    t_rise = time_shifted.iloc[idx_rise]
    t_peak1 = time_shifted.loc[idx_peak1]
    t_pullback = time_shifted.loc[idx_pullback]
    t_peak2 = time_shifted.loc[idx_peak2]

    # --- Line Fitting ---
    m1, c1 = _fit_line_to_range(time_shifted, velocity_smoothed, t_rise, t_peak1)
    v_peak1 = velocity_smoothed.loc[idx_peak1]
    m2, c2 = 0.0, v_peak1
    pullback_fit_start_t = t_peak1 + (t_pullback - t_peak1) * 0.1
    m3, c3 = _fit_line_to_range(time_shifted, velocity_smoothed, pullback_fit_start_t, t_pullback)
    m4, c4 = _fit_line_to_range(time_shifted, velocity_smoothed, t_pullback, t_peak2)
    v_peak2 = velocity_smoothed.loc[idx_peak2]
    m5, c5 = 0.0, v_peak2
    
    lines_info = [(m1, c1), (m2, c2), (m3, c3), (m4, c4), (m5, c5)]
    if any(pd.isna(val) for line in lines_info for val in line):
        raise ValueError("Failed to fit one or more of the 5 required line segments.")
        
    P1 = _find_intersection(m1, c1, m2, c2)
    P2 = _find_intersection(m2, c2, m3, c3)
    P3 = _find_intersection(m3, c3, m4, c4)
    P4 = _find_intersection(m4, c4, m5, c5)
    intersections = [P1, P2, P3, P4]
    
    if any(p is None or pd.isna(p[0]) for p in intersections):
        raise ValueError("Failed to find all four critical intersection points.")

    results = {}
    delta_u_fs = abs(P2[1] - P3[1])
    results['Spall Strength (GPa)'] = 0.5 * density * acoustic_velocity * delta_u_fs * 1e-9
    
    pullback_slope_ns = m3
    results['Strain Rate (s^-1)'] = abs(0.5 * (pullback_slope_ns * 1e9) / acoustic_velocity)

    results['First Maxima (m/s)'] = v_peak1
    results['Plateau Mean Velocity (m/s)'] = v_peak1
    results['Pullback Minima (m/s)'] = P3[1]
    
    # Calculate peak shock stress from plateau velocity
    results['Peak Shock Stress (GPa)'] = 0.5 * density * acoustic_velocity * results['Plateau Mean Velocity (m/s)'] * 1e-9
    
    # Calculate uncertainties if available
    if uncertainty_smoothed is not None and not uncertainty_smoothed.isna().all():
        try:
            # For hybrid model, we need to estimate uncertainties at intersection points
            # Get uncertainties at the key points used in calculations
            peak1_uncertainty = uncertainty_smoothed.iloc[idx_peak1] if idx_peak1 < len(uncertainty_smoothed) else np.nan
            pullback_uncertainty = uncertainty_smoothed.iloc[idx_pullback] if idx_pullback < len(uncertainty_smoothed) else np.nan
            
            # Estimate uncertainties at intersection points P2 and P3
            # This is a simplified approach - in practice, you might want more sophisticated uncertainty propagation
            if not pd.isna(peak1_uncertainty) and not pd.isna(pullback_uncertainty):
                # Estimate uncertainty in delta_u_fs (difference between P2 and P3)
                # Use average uncertainty as approximation
                avg_uncertainty = (peak1_uncertainty + pullback_uncertainty) / 2
                delta_u_fs_uncertainty = np.sqrt(2) * avg_uncertainty  # Assuming independent uncertainties
                
                # Propagate to spall strength uncertainty
                results['Spall Strength Uncertainty (GPa)'] = 0.5 * density * acoustic_velocity * delta_u_fs_uncertainty * 1e-9
                
                # For strain rate, estimate uncertainty in slope
                # This is more complex, but we can use a simplified approach
                time_range = t_pullback - pullback_fit_start_t
                if time_range > 0:
                    # Estimate slope uncertainty based on velocity uncertainty over time range
                    slope_uncertainty = avg_uncertainty / (time_range * 1e9)  # Convert to s^-1
                    results['Strain Rate Uncertainty (s^-1)'] = abs(0.5 * slope_uncertainty / acoustic_velocity)
                else:
                    results['Strain Rate Uncertainty (s^-1)'] = np.nan
                
                # Propagate velocity uncertainty to peak shock stress uncertainty
                # Peak Shock Stress = 0.5 * density * acoustic_velocity * plateau_velocity * 1e-9
                # Uncertainty in peak shock stress = 0.5 * density * acoustic_velocity * peak_uncertainty * 1e-9
                if not pd.isna(peak1_uncertainty):
                    results['Peak Shock Stress Uncertainty (GPa)'] = 0.5 * density * acoustic_velocity * peak1_uncertainty * 1e-9
                else:
                    results['Peak Shock Stress Uncertainty (GPa)'] = np.nan
            else:
                results['Spall Strength Uncertainty (GPa)'] = np.nan
                results['Strain Rate Uncertainty (s^-1)'] = np.nan
                results['Peak Shock Stress Uncertainty (GPa)'] = np.nan
                
        except Exception as e:
            logger.warning(f"Could not calculate uncertainties for hybrid model: {e}")
            results['Spall Strength Uncertainty (GPa)'] = np.nan
            results['Strain Rate Uncertainty (s^-1)'] = np.nan
    else:
        results['Spall Strength Uncertainty (GPa)'] = np.nan
        results['Strain Rate Uncertainty (s^-1)'] = np.nan
    
    return results, lines_info, intersections


# --- Main Data Processing Entry Point ---

def calculate_spall_parameters(
    time_ns, velocity_ms, density=None, acoustic_velocity=None,
    plot_path=None, smooth_window=101, polyorder=3,
    analysis_model='hybrid_5_segment', signal_length_ns=None, uncertainty_ms=None, skip_smoothing=False, **kwargs
):
    """
    Main function to process a single velocity trace.
    """
    base_name = os.path.splitext(os.path.basename(plot_path))[0] if plot_path else 'data'
    results = {'Filename': base_name}
    status = 'Failed'
    error_message = ''
    lines_info, intersections = [], []
    current_data_dict = {'filename': base_name}
    
    try:
        if density is None or acoustic_velocity is None:
            raise ValueError("Material 'density' and 'acoustic_velocity' must be provided.")
        
        df = pd.DataFrame({'time': time_ns, 'velocity': velocity_ms})
        if uncertainty_ms is not None:
            df['uncertainty'] = uncertainty_ms
        else:
            df['uncertainty'] = np.nan
            
        if len(df) < 20: raise ValueError("Not enough valid data points for analysis.")

        if signal_length_ns and signal_length_ns > 0:
            original_len = len(df)
            df = df[df['time'] <= signal_length_ns].reset_index(drop=True)
            logger.debug(f"Signal cropped to {signal_length_ns} ns. Kept {len(df)} of {original_len} points.")
            if df.empty:
                raise ValueError(f"Signal cropping to {signal_length_ns} ns resulted in an empty dataset. Check signal length or data.")
        
        # Robust Time Shifting: Find the first significant rise to set t=0
        condition = df['velocity'] > (0.1 * df['velocity'].max())
        if condition.any():
            initial_rise_idx = condition.idxmax()
        else:
            initial_rise_idx = df.index[0]
            logger.debug("Could not find significant initial rise; using start of signal as t=0 reference.")

        t_shift = df['time'][initial_rise_idx]
        df['time_shifted'] = df['time'] - t_shift

        df_final = df[df['time_shifted'] >= 0].reset_index(drop=True)
        
        # Apply smoothing only if not skipped (for ALPSS pre-smoothed data)
        if skip_smoothing:
            logger.debug("Skipping SPADE smoothing - using pre-smoothed data from ALPSS")
            df_final['velocity_smoothed'] = df_final['velocity']
        else:
            if len(df_final) < smooth_window: 
                raise ValueError("Not enough data after time shifting for smoothing.")
            df_final['velocity_smoothed'] = savgol_filter(df_final['velocity'], window_length=smooth_window, polyorder=polyorder)
        
        current_data_dict.update({
            'x_shifted': df_final['time_shifted'],
            'y_original': df_final['velocity'],
            'y_smooth': df_final['velocity_smoothed'],
            'uncertainty': df_final['uncertainty']
        })

        logger.debug(f"Calling analysis model: '{analysis_model}'")

        if analysis_model == 'hybrid_5_segment':
            model_kwargs = {k: v for k, v in kwargs.items() if k in ['prominence_factor', 'peak_distance_ns']}
            results_model, lines_info, intersections = _calculate_hybrid_5_segment_features(
                df_final['time_shifted'], df_final['velocity_smoothed'], density, acoustic_velocity, uncertainty_smoothed=df_final['uncertainty'], **model_kwargs)
        
        elif analysis_model == 'max_min':
            results_model, lines_info, intersections = _calculate_max_min_features(
                df_final['time_shifted'], df_final['velocity_smoothed'], density, acoustic_velocity, uncertainty_smoothed=df_final['uncertainty'])
        
        else:
            raise ValueError(f"Unknown analysis_model: '{analysis_model}'. Must be 'hybrid_5_segment' or 'max_min'.")
            
        results.update(results_model)
        status = 'Success'

    except Exception as e:
        error_message = str(e)
        logger.warning(f"  Could not process {base_name}: {e}")
    
    results['Processing Status'] = status
    results['Error Message'] = error_message
    
    if status == 'Success' and plot_path:
        _plot_analysis_results(current_data_dict, lines_info, intersections, plot_path)
            
    return results, lines_info, intersections

def process_velocity_files(
    input_folder, file_pattern, output_folder,
    save_summary_table=True, summary_table_name="spall_summary.csv", **kwargs
):
    """
    Processes all matching velocity files in a folder.
    """
    files_to_process = sorted(glob.glob(os.path.join(input_folder, file_pattern)))
    if not files_to_process:
        logger.warning(f"No files found for pattern '{file_pattern}' in '{input_folder}'.")
        return pd.DataFrame()

    logger.info(f"Processing {len(files_to_process)} files from '{input_folder}'...")
    
    results_list = []
    subfolder_name = os.path.basename(os.path.normpath(input_folder))
    material_label = utils.extract_legend_info(subfolder_name, utils.MATERIAL_MAPPING, utils.ENERGY_VELOCITY_MAPPING)[0] or "Unknown"

    calc_args = {k: v for k, v in kwargs.items() if k not in ['input_folder', 'file_pattern', 'output_folder', 'save_summary_table', 'summary_table_name']}
    if 'density' not in calc_args or 'acoustic_velocity' not in calc_args:
        raise ValueError("Critical: 'density' and 'acoustic_velocity' must be provided via kwargs.")

    for file_path in files_to_process:
        filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"-> Processing: {filename_no_ext}")
        
        try:
            data = _load_and_clean_data(file_path)
            if data is None or data.empty:
                logger.warning(f"Skipping {filename_no_ext} due to loading/cleaning error or empty data.")
                continue
            
            plot_path = os.path.join(output_folder, f"{filename_no_ext}_analysis.png") if kwargs.get('plot_individual') else None
            
            # Pass uncertainty data if available
            uncertainty_data = data['Uncertainty'] if 'Uncertainty' in data.columns else None
            
            result_dict, _, _ = calculate_spall_parameters(
                time_ns=data['Time'], 
                velocity_ms=data['Velocity'], 
                uncertainty_ms=uncertainty_data,
                plot_path=plot_path, 
                **calc_args
            )
            result_dict['Material'] = material_label
            results_list.append(result_dict)

        except Exception as e:
            logger.error(f"Critical error processing file {filename_no_ext}: {e}", exc_info=True)
            results_list.append({
                'Filename': filename_no_ext, 'Material': material_label, 'Processing Status': 'Failed: Critical Error'
            })

    if not results_list:
        logger.warning("No results were generated.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(results_list)
    
    if save_summary_table:
        table_output_dir = os.path.dirname(summary_table_name)
        os.makedirs(table_output_dir, exist_ok=True)
        summary_df.to_csv(summary_table_name, index=False, float_format='%.4e')
        logger.info(f"Summary table saved to: {summary_table_name}")
        
    return summary_df