# spade_analysis/data_processing.py
"""
Functions for processing raw velocity-time data.
Supports multiple analysis models and user-defined material properties.
"""

import pandas as pd
import numpy as np
import matplotlib
# Force non-interactive backend to avoid GUI usage in threads/background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, find_peaks
import os
import warnings
import traceback
import logging
import glob
# Support both package and script execution
try:
    from . import utils  # type: ignore
except Exception:
    try:
        # If run as a script, add this file's directory to sys.path and import utils
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent))
        import utils  # type: ignore
    except Exception as _e:
        raise ImportError(
            "Could not import 'utils'. Run via package (python -m SPADE.spall_analysis_release.spall_analysis.data_processing) "
            "or ensure the module directory is on PYTHONPATH."
        ) from _e

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
def _plot_analysis_results(data_dict, lines_info, intersections, output_path, axis_limits=None, plateau_velocity=None, plateau_window=None, dns_classification=None):
    """ Plots trace, smoothed data, fitted lines, and intersections.

    axis_limits: Optional dict with keys:
        - auto_calculate_limits: bool
        - x_min_main, x_max_main, y_min_main, y_max_main: floats
    plateau_velocity: Optional float indicating if plateau velocity was found (for subplot annotation)
    plateau_window: Optional tuple (t_start, t_end, v_mean) for fallback method visualization
    dns_classification: Optional string indicating DNS classification (e.g., "Did Not Spall", "Valid Spall", etc.)
    """
    if not output_path:
        return
    import matplotlib.pyplot as plt
    # Prefer full-length aligned trace if provided; fall back to cropped
    x_trace = data_dict.get('x_shifted_full', data_dict.get('x_shifted', pd.Series(dtype=float)))
    y_trace = data_dict.get('y_original_full', data_dict.get('y_original', pd.Series(dtype=float)))
    y_smooth = data_dict.get('y_smooth', pd.Series(dtype=float))
    x_shifted = data_dict.get('x_shifted', pd.Series(dtype=float))  # Post-t0 data only
    uncertainty = data_dict.get('uncertainty', pd.Series(dtype=float))
    filename = data_dict.get('filename', 'Unknown Filename')
    base_filename = os.path.splitext(filename)[0]

    if x_trace.empty or y_trace.empty:
        logger.warning(f"Skipping plot for {filename}: No valid trace data.")
        return

    logger.debug(f"    Generating model plot for {base_filename}")
    # Create figure with two subplots: main plot on top, zoomed subplot on bottom
    fig = plt.figure(figsize=(10, 8))
    ax_main = plt.subplot(2, 1, 1)  # Main plot (top)
    ax_sub = plt.subplot(2, 1, 2)  # Subplot for first 70 ns (bottom)
    
    # Helper function to plot data on an axis
    def _plot_on_axis(ax, x_data, y_data, y_smooth_data, uncertainty_data, x_shifted_data, lines_info_data, intersections_data):
        """Helper function to plot data on a given axis"""
        # Plot uncertainty bands if available
        if not uncertainty_data.empty and not uncertainty_data.isna().all():
            if len(uncertainty_data) == len(x_data):
                ax.fill_between(x_data, y_data - uncertainty_data, y_data + uncertainty_data, 
                              alpha=0.3, color='lightblue', label='Uncertainty')
            elif len(uncertainty_data) == len(x_shifted_data) and not x_shifted_data.empty:
                uncertainty_full = pd.Series(index=x_data.index, dtype=float)
                uncertainty_values = uncertainty_data.values
                if len(uncertainty_values) > 0:
                    uncertainty_full.loc[x_shifted_data.index[:len(uncertainty_values)]] = uncertainty_values
                    if not uncertainty_full.isna().all():
                        ax.fill_between(x_data, y_data - uncertainty_full.fillna(0), 
                                      y_data + uncertainty_full.fillna(0), 
                                      alpha=0.3, color='lightblue', label='Uncertainty')
        
        ax.plot(x_data, y_data, label='Original Data', color='grey', alpha=0.6, lw=1.0)
        
        # Plot smoothed data
        if not y_smooth_data.empty:
            if len(y_smooth_data) == len(x_data):
                ax.plot(x_data, y_smooth_data, label='Smoothed Data', color='black', alpha=0.8, lw=1.0, ls=':')
            elif len(y_smooth_data) == len(x_shifted_data) and not x_shifted_data.empty:
                ax.plot(x_shifted_data, y_smooth_data, label='Smoothed Data', color='black', alpha=0.8, lw=1.0, ls=':')
            else:
                min_len = min(len(x_data), len(y_smooth_data))
                if min_len > 0:
                    ax.plot(x_data.iloc[:min_len], y_smooth_data.iloc[:min_len], 
                          label='Smoothed Data', color='black', alpha=0.8, lw=1.0, ls=':')

        # Plot fitted lines - draw only between intersection points
        if lines_info_data and intersections_data:
            colors = ['blue', 'green', 'red', 'purple', 'brown']
            labels = ['Line 1 (Rise)', 'Line 2 (Plateau)', 'Line 3 (Pullback)', 'Line 4 (Recomp Rise)', 'Line 5 (Recomp Tail)']
            
            # Get time range for drawing
            if not x_shifted_data.empty:
                t_min = float(x_shifted_data.min())
                t_max = float(x_shifted_data.max())
            else:
                t_min = float(x_data.min()) if len(x_data) > 0 else 0.0
                t_max = float(x_data.max()) if len(x_data) > 0 else 100.0
            
            # Line 1: from start of data to P1
            if len(intersections_data) > 0 and intersections_data[0] is not None and not pd.isna(intersections_data[0][0]):
                P1 = intersections_data[0]
                m1, c1 = lines_info_data[0]
                if pd.notna(m1) and pd.notna(c1):
                    x_line1 = np.linspace(t_min, float(P1[0]), 50)
                    y_line1 = m1 * x_line1 + c1
                    ax.plot(x_line1, y_line1, color=colors[0], linestyle='--', lw=2, label=f'{labels[0]} (m={m1:.2f})')
            
            # Line 2: from P1 to P2
            if len(intersections_data) > 1 and intersections_data[0] is not None and intersections_data[1] is not None:
                P1 = intersections_data[0]
                P2 = intersections_data[1]
                if not pd.isna(P1[0]) and not pd.isna(P2[0]):
                    m2, c2 = lines_info_data[1]
                    if pd.notna(m2) and pd.notna(c2):
                        x_line2 = np.linspace(float(P1[0]), float(P2[0]), 50)
                        y_line2 = m2 * x_line2 + c2
                        ax.plot(x_line2, y_line2, color=colors[1], linestyle='--', lw=2, label=f'{labels[1]} (m={m2:.2f})')
            
            # Line 3: from P2 to P3
            if len(intersections_data) > 2 and intersections_data[1] is not None and intersections_data[2] is not None:
                P2 = intersections_data[1]
                P3 = intersections_data[2]
                if not pd.isna(P2[0]) and not pd.isna(P3[0]):
                    m3, c3 = lines_info_data[2]
                    if pd.notna(m3) and pd.notna(c3):
                        x_line3 = np.linspace(float(P2[0]), float(P3[0]), 50)
                        y_line3 = m3 * x_line3 + c3
                        ax.plot(x_line3, y_line3, color=colors[2], linestyle='--', lw=2, label=f'{labels[2]} (m={m3:.2f})')
            
            # Line 4: from P3 to P4
            if len(intersections_data) > 3 and intersections_data[2] is not None and intersections_data[3] is not None:
                P3 = intersections_data[2]
                P4 = intersections_data[3]
                if not pd.isna(P3[0]) and not pd.isna(P4[0]):
                    m4, c4 = lines_info_data[3]
                    if pd.notna(m4) and pd.notna(c4):
                        x_line4 = np.linspace(float(P3[0]), float(P4[0]), 50)
                        y_line4 = m4 * x_line4 + c4
                        ax.plot(x_line4, y_line4, color=colors[3], linestyle='--', lw=2, label=f'{labels[3]} (m={m4:.2f})')
            
            # Line 5: from P4 to end of data
            if len(intersections_data) > 3 and intersections_data[3] is not None:
                P4 = intersections_data[3]
                if not pd.isna(P4[0]):
                    m5, c5 = lines_info_data[4]
                    if pd.notna(m5) and pd.notna(c5):
                        x_line5 = np.linspace(float(P4[0]), t_max, 50)
                        y_line5 = m5 * x_line5 + c5
                        ax.plot(x_line5, y_line5, color=colors[4], linestyle='--', lw=2, label=f'{labels[4]} (m={m5:.2f})')
        
        # Plot intersection points
        if intersections_data:
            is_max_min_plot = not lines_info_data
            point_labels = ['Peak', 'Valley'] if is_max_min_plot else ['P1', 'P2', 'P3', 'P4']
            colors = ['#ff7f0e', '#1f77b4'] if is_max_min_plot else ['cyan', 'magenta', 'orange', 'lime']
            for i, (px, py) in enumerate(intersections_data):
                if pd.notna(px) and pd.notna(py) and i < len(point_labels):
                    ax.scatter([px], [py], label=f'{point_labels[i]} ({px:.1f}, {py:.1f})', 
                               s=100, zorder=5, edgecolors='black', color=colors[i])
    
    try:
        # Plot on main axis (full range)
        _plot_on_axis(ax_main, x_trace, y_trace, y_smooth, uncertainty, x_shifted, lines_info, intersections)
        
        ax_main.set_xlabel('Time (ns)', fontsize=14)
        ax_main.set_ylabel('Velocity (m/s)', fontsize=14)
        ax_main.set_title(f'Analysis for: {base_filename}', fontsize=16)
        ax_main.legend(loc='best', fontsize=9)
        ax_main.grid(True, linestyle=':')
        
        # Add DNS label if classification indicates Did Not Spall
        # Check if DNS classification exists and indicates DNS (not valid spall)
        is_dns = False
        if dns_classification and isinstance(dns_classification, str):
            dns_upper = dns_classification.upper()
            # Check for explicit DNS indicators
            if ('DNS' in dns_upper or 'DID NOT SPALL' in dns_upper or 'NO SPALL' in dns_upper or
                'RE-ACCELERATION' in dns_upper or 'NO PULLBACK' in dns_upper or 
                'NO CLEAR PEAK' in dns_upper or 'LOW-VELOCITY' in dns_upper or
                'NO RE-ACCELERATION' in dns_upper):
                is_dns = True
            # Also check if it's NOT a valid spall classification
            # If DNS_Classification exists and is not "Valid Spall" or "Unknown", it's likely DNS
            elif dns_classification.upper() not in ['UNKNOWN', 'VALID SPALL', '']:
                is_dns = True
        
        if is_dns:
            # Add DNS label in the upper right corner of the main plot
            ax_main.text(0.98, 0.98, 'DNS\n(Did Not Spall)', 
                       transform=ax_main.transAxes,
                       fontsize=14, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=2),
                       color='white', zorder=10)
        
        # Calculate y-axis limits from main plot
        all_y = y_trace[np.isfinite(y_trace)]
        if not all_y.empty:
            min_y, max_y = np.nanmin(all_y), np.nanmax(all_y)
            y_lim_main = (min_y - 50, max_y + 100)
            ax_main.set_ylim(y_lim_main)
        else:
            y_lim_main = ax_main.get_ylim()
        
        # Apply axis limits if provided
        try:
            if isinstance(axis_limits, dict) and not axis_limits.get('auto_calculate_limits', True):
                x_min = float(axis_limits.get('x_min_main', np.nan))
                x_max = float(axis_limits.get('x_max_main', np.nan))
                y_min = float(axis_limits.get('y_min_main', np.nan))
                y_max = float(axis_limits.get('y_max_main', np.nan))
                if np.isfinite(x_min) and np.isfinite(x_max):
                    ax_main.set_xlim(x_min, x_max)
                if np.isfinite(y_min) and np.isfinite(y_max):
                    ax_main.set_ylim(y_min, y_max)
                    y_lim_main = (y_min, y_max)
        except Exception:
            pass
        
        # Plot on subplot (first 70 ns only)
        # Filter data for first 70 ns
        if not x_trace.empty:
            x_min_sub = x_trace.min()
            x_max_sub = x_min_sub + 70.0  # 70 ns window
            mask_sub = (x_trace >= x_min_sub) & (x_trace <= x_max_sub)
            x_trace_sub = x_trace[mask_sub]
            y_trace_sub = y_trace[mask_sub]
            
            # Filter smoothed data
            x_shifted_sub = x_shifted  # Default
            if not y_smooth.empty:
                if len(y_smooth) == len(x_trace):
                    y_smooth_sub = y_smooth[mask_sub]
                elif len(y_smooth) == len(x_shifted) and not x_shifted.empty:
                    mask_smooth = (x_shifted >= x_min_sub) & (x_shifted <= x_max_sub)
                    y_smooth_sub = y_smooth[mask_smooth]
                    x_shifted_sub = x_shifted[mask_smooth]
                else:
                    y_smooth_sub = pd.Series(dtype=float)
            else:
                y_smooth_sub = pd.Series(dtype=float)
            
            # Filter uncertainty
            uncertainty_sub = pd.Series(dtype=float)
            if not uncertainty.empty and not uncertainty.isna().all():
                if len(uncertainty) == len(x_trace):
                    uncertainty_sub = uncertainty[mask_sub]
                elif len(uncertainty) == len(x_shifted) and not x_shifted.empty:
                    mask_unc = (x_shifted >= x_min_sub) & (x_shifted <= x_max_sub)
                    uncertainty_sub = uncertainty[mask_unc]
            
            # Filter intersections for subplot range
            intersections_sub = []
            if intersections:
                for px, py in intersections:
                    if pd.notna(px) and pd.notna(py) and x_min_sub <= px <= x_max_sub:
                        intersections_sub.append((px, py))
            
            # Plot on subplot
            _plot_on_axis(ax_sub, x_trace_sub, y_trace_sub, y_smooth_sub, uncertainty_sub, 
                         x_shifted_sub, lines_info, intersections_sub)
            
            # Set subplot limits: same y-axis as main, x-axis limited to 70 ns
            ax_sub.set_xlim(x_min_sub, x_max_sub)
            ax_sub.set_ylim(y_lim_main)
            ax_sub.set_xlabel('Time (ns)', fontsize=12)
            ax_sub.set_ylabel('Velocity (m/s)', fontsize=12)
            ax_sub.grid(True, linestyle=':')
            
            # Add annotation for plateau velocity status
            plateau_found = pd.notna(plateau_velocity) and plateau_velocity > 0
            if plateau_found:
                status_text = f'✓ Plateau Velocity Found: {plateau_velocity:.1f} m/s'
                status_color = 'green'
            else:
                status_text = '✗ Plateau Velocity NOT Found'
                status_color = 'red'
            
            # Add text box with plateau velocity status
            ax_sub.text(0.02, 0.98, status_text, transform=ax_sub.transAxes,
                       fontsize=11, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=status_color, linewidth=2),
                       color=status_color, weight='bold')
            
            # If plateau velocity found, mark it on the plot
            if plateau_found:
                # Check if this is from fallback method (peak-based)
                if plateau_window is not None:
                    # Fallback method: show ±1 ns window and mean plateau line
                    t_window_start, t_window_end, v_mean = plateau_window
                    
                    # Draw vertical lines marking the ±1 ns window
                    ax_sub.axvline(x=t_window_start, color='green', linestyle='--', linewidth=2, alpha=0.7, label='±1 ns Window')
                    ax_sub.axvline(x=t_window_end, color='green', linestyle='--', linewidth=2, alpha=0.7)
                    
                    # Shade the window region
                    ax_sub.axvspan(t_window_start, t_window_end, alpha=0.2, color='green', label='Plateau Window')
                    
                    # Draw horizontal line at mean plateau velocity
                    ax_sub.axhline(y=v_mean, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean Plateau: {v_mean:.1f} m/s')
                    
                    # Mark the peak point
                    if not x_trace_sub.empty and not y_trace_sub.empty:
                        # Find peak in subplot range
                        mask_peak = (x_trace_sub >= t_window_start) & (x_trace_sub <= t_window_end)
                        if mask_peak.any():
                            peak_in_window = y_trace_sub[mask_peak].idxmax()
                            ax_sub.scatter([x_trace_sub.loc[peak_in_window]], [y_trace_sub.loc[peak_in_window]], 
                                         s=150, zorder=10, color='green', marker='*', 
                                         edgecolors='darkgreen', linewidths=2, label='Peak')
                    
                    ax_sub.legend(loc='best', fontsize=9)
                else:
                    # 5-segment method: mark plateau velocity point
                    if not y_trace_sub.empty:
                        # Find where velocity is closest to plateau_velocity
                        vel_diff = np.abs(y_trace_sub.values - plateau_velocity)
                        closest_idx = np.argmin(vel_diff)
                        if closest_idx < len(x_trace_sub):
                            ax_sub.scatter([x_trace_sub.iloc[closest_idx]], [y_trace_sub.iloc[closest_idx]], 
                                         s=150, zorder=10, color='green', marker='*', 
                                         edgecolors='darkgreen', linewidths=2, label='Plateau Velocity')
                            ax_sub.legend(loc='best', fontsize=9)
                
                # Also mark on main plot
                if plateau_window is not None:
                    # Fallback method: show window on main plot too
                    t_window_start, t_window_end, v_mean = plateau_window
                    ax_main.axvline(x=t_window_start, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax_main.axvline(x=t_window_end, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
                    ax_main.axvspan(t_window_start, t_window_end, alpha=0.15, color='green')
                    ax_main.axhline(y=v_mean, color='green', linestyle='-', linewidth=1.5, alpha=0.6, label=f'Mean Plateau (fallback): {v_mean:.1f} m/s')

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

def _calculate_peak_based_plateau_velocity(time_shifted, velocity_smoothed, density, acoustic_velocity, uncertainty_smoothed=None, **kwargs):
    """
    Fallback method: Calculate plateau velocity using peak ±1 ns window.
    Used when 5-segment method fails.
    
    Returns:
        results: dict with plateau velocity and related parameters
        plateau_window: tuple of (t_start, t_end, v_mean) for plotting
    """
    results = {}
    
    # Find peak velocity (global maximum)
    idx_peak = int(np.argmax(velocity_smoothed.values))
    t_peak = float(time_shifted.iloc[idx_peak])
    v_peak = float(velocity_smoothed.iloc[idx_peak])
    
    # Define ±1 ns window around peak
    t_window_start = t_peak - 1.0
    t_window_end = t_peak + 1.0
    
    # Find indices within the window
    mask_window = (time_shifted >= t_window_start) & (time_shifted <= t_window_end)
    velocities_in_window = velocity_smoothed[mask_window]
    
    if len(velocities_in_window) == 0:
        # If window is empty, just use peak velocity
        plateau_velocity = v_peak
        logging.warning(f"Peak-based method: No data in ±1 ns window, using peak velocity directly")
    else:
        # Calculate mean velocity in the window
        plateau_velocity = float(velocities_in_window.mean())
        logging.info(f"Peak-based method: Plateau velocity = {plateau_velocity:.2f} m/s (mean of {len(velocities_in_window)} points in ±1 ns window)")
    
    # Store results
    results['Plateau Mean Velocity (m/s)'] = plateau_velocity
    results['First Maxima (m/s)'] = v_peak
    
    # Calculate uncertainty if available
    if uncertainty_smoothed is not None and not uncertainty_smoothed.isna().all():
        if len(velocities_in_window) > 0:
            uncertainties_in_window = uncertainty_smoothed[mask_window]
            if not uncertainties_in_window.isna().all():
                # Mean uncertainty in the window
                plateau_uncertainty = float(uncertainties_in_window.mean())
                results['Plateau Mean Velocity Uncertainty (m/s)'] = plateau_uncertainty
            else:
                results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
        else:
            # Use peak uncertainty
            if idx_peak < len(uncertainty_smoothed):
                results['Plateau Mean Velocity Uncertainty (m/s)'] = float(uncertainty_smoothed.iloc[idx_peak])
            else:
                results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
    else:
        results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
    
    # Store window info for plotting
    plateau_window = (t_window_start, t_window_end, plateau_velocity)
    
    return results, plateau_window

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
    
    logging.debug(f"5-segment model: prominence={prominence:.2f}, distance_samples={distance_samples}")

    # Try to find peaks with decreasing prominence if initial attempt fails
    peaks = None
    valleys = None
    prominence_attempts = [prominence_factor, prominence_factor * 0.5, prominence_factor * 0.25, 0.01]
    
    for prom_factor in prominence_attempts:
        current_prominence = velocity_range * prom_factor
        peaks, _ = find_peaks(velocity_smoothed, prominence=current_prominence, distance=distance_samples)
        valleys, _ = find_peaks(-velocity_smoothed, prominence=current_prominence, distance=distance_samples)
        
        if len(peaks) >= 2 and len(valleys) >= 1:
            logging.debug(f"Found {len(peaks)} peaks and {len(valleys)} valleys with prominence_factor={prom_factor}")
            break
    
    if len(peaks) < 2:
        # Last resort: find any peaks without prominence requirement
        peaks, _ = find_peaks(velocity_smoothed, distance=distance_samples)
        if len(peaks) < 2:
            logging.warning(f"5-segment model: Could not find at least two peaks even with relaxed criteria. Found {len(peaks)} peaks.")
            raise ValueError(f"Could not find at least two peaks (found {len(peaks)}). Signal may not have clear spall signature.")
    
    if len(valleys) < 1:
        # Last resort: find any valleys without prominence requirement
        valleys, _ = find_peaks(-velocity_smoothed, distance=distance_samples)
        if len(valleys) < 1:
            logging.warning(f"5-segment model: Could not find any valleys even with relaxed criteria.")
            raise ValueError("Could not find any pullback minimum. Signal may not have clear spall signature.")
    
    # Choose the global maximum peak to anchor P1/P2 near the true maxima
    try:
        peak_vals = velocity_smoothed.iloc[peaks]
        idx_peak1 = int(peaks[int(np.argmax(peak_vals.values))])
    except Exception:
        # Fallback to first detected peak
        idx_peak1 = int(peaks[0])
    valleys_after_peak1 = valleys[valleys > idx_peak1]
    if not valleys_after_peak1.any():
        # Fallback: find the global minimum after peak1
        signal_after_peak = velocity_smoothed.iloc[idx_peak1+1:]
        if len(signal_after_peak) == 0:
            raise ValueError("No signal data found after initial peak.")
        relative_idx_min = np.argmin(signal_after_peak.values)
        idx_pullback = idx_peak1 + 1 + relative_idx_min
        logging.info(f"Using global minimum after peak1 as pullback (index {idx_pullback})")
    else:
        idx_pullback = valleys_after_peak1[0]
    
    peaks_after_pullback = peaks[peaks > idx_pullback]
    if not peaks_after_pullback.any():
        # Fallback: if no recompaction peak found, use the last point or a point near the end
        # This handles cases where recompaction is weak or signal ends early
        if idx_pullback < len(velocity_smoothed) - 1:
            # Use a point near the end of the signal as a proxy for recompaction
            idx_peak2 = len(velocity_smoothed) - 1
            logging.info(f"No recompaction peak found, using end of signal as proxy (index {idx_peak2})")
        else:
            raise ValueError("No recompaction peak found after pullback and signal too short.")
    else:
        idx_peak2 = peaks_after_pullback[0]

    # --- Improved Initial Rise Detection anchored to global maximum ---
    N_baseline = min(10, len(velocity_smoothed)//5)
    baseline = np.median(velocity_smoothed[:N_baseline])
    peak_val = float(velocity_smoothed.iloc[idx_peak1])
    threshold = baseline + max(0.05 * (peak_val - baseline), 10.0)  # 5% of peak or 10 m/s above baseline
    initial_rise_indices = np.where(velocity_smoothed.values > threshold)[0]
    if len(initial_rise_indices) > 0:
        idx_rise = initial_rise_indices[0]
    else:
        idx_rise = 0  # fallback to first point if no clear rise
    t_rise = time_shifted.iloc[idx_rise]
    t_peak1 = time_shifted.loc[idx_peak1]
    t_pullback = time_shifted.loc[idx_pullback]
    t_peak2 = time_shifted.loc[idx_peak2]

    # --- Line Fitting ---
    # Line 1: Rise from initial rise to peak1
    m1, c1 = _fit_line_to_range(time_shifted, velocity_smoothed, t_rise, t_peak1)
    v_peak1 = velocity_smoothed.loc[idx_peak1]
    
    # Line 2: Plateau (horizontal at peak velocity)
    m2, c2 = 0.0, v_peak1
    
    # Initial fit for Line 3 to find approximate P2
    pullback_fit_start_t = t_peak1 + (t_pullback - t_peak1) * 0.1
    m3_initial, c3_initial = _fit_line_to_range(time_shifted, velocity_smoothed, pullback_fit_start_t, t_pullback)
    
    # Calculate initial P2 (intersection of Line 2 and initial Line 3)
    P2_initial = _find_intersection(m2, c2, m3_initial, c3_initial)
    if P2_initial is None or pd.isna(P2_initial[0]):
        # Fallback: use pullback_fit_start_t
        P2_initial = (float(pullback_fit_start_t), float(v_peak1))
    
    # Now refit Line 3 from P2 to pullback to ensure it starts exactly at P2
    t_p2 = float(P2_initial[0])
    # Ensure P2 is not after pullback
    if t_p2 >= float(t_pullback):
        t_p2 = float(t_pullback) - 0.1 * (float(t_pullback) - float(t_peak1))
        P2_initial = (t_p2, float(v_peak1))
    
    # Refit Line 3 from P2 to pullback
    m3, c3 = _fit_line_to_range(time_shifted, velocity_smoothed, t_p2, t_pullback)
    
    # Line 4: Recompression rise from pullback to peak2
    m4, c4 = _fit_line_to_range(time_shifted, velocity_smoothed, t_pullback, t_peak2)
    v_peak2 = velocity_smoothed.loc[idx_peak2]
    
    # Line 5: Recompression tail (horizontal at peak2)
    m5, c5 = 0.0, v_peak2
    
    lines_info = [(m1, c1), (m2, c2), (m3, c3), (m4, c4), (m5, c5)]
    if any(pd.isna(val) for line in lines_info for val in line):
        raise ValueError("Failed to fit one or more of the 5 required line segments.")
        
    # Calculate intersection points
    P1 = _find_intersection(m1, c1, m2, c2)
    P2 = _find_intersection(m2, c2, m3, c3)  # This should now be very close to P2_initial
    P3_intersection = _find_intersection(m3, c3, m4, c4)
    P4 = _find_intersection(m4, c4, m5, c5)
    
    # Fallback for missing intersections: use actual data points
    if P1 is None or pd.isna(P1[0]):
        # Use point at t_peak1 (start of plateau)
        P1 = (float(t_peak1), float(v_peak1))
        logging.info("P1 intersection failed, using peak1 point")
    
    if P2 is None or pd.isna(P2[0]):
        # Use point at P2_initial (end of plateau, start of pullback)
        P2 = P2_initial if P2_initial is not None and not pd.isna(P2_initial[0]) else (float(t_p2), float(v_peak1))
        logging.info("P2 intersection failed, using initial P2 estimate as plateau end")
    
    # CRITICAL FIX: P3 should be at the actual global minimum, not just the line intersection
    # Find the actual global minimum between P2 (or peak1) and peak2
    try:
        t_p2_float = float(P2[0]) if P2 is not None and not pd.isna(P2[0]) else float(t_peak1)
        t_peak2_float = float(t_peak2)
        
        # Find indices in the range from P2 to peak2
        mask = (time_shifted.values >= t_p2_float) & (time_shifted.values <= t_peak2_float)
        if not mask.any():
            # Fallback: use range from peak1 to peak2
            mask = (time_shifted.values >= float(t_peak1)) & (time_shifted.values <= t_peak2_float)
        
        if mask.any():
            # Find the actual global minimum in this range
            velocity_in_range = velocity_smoothed.values[mask]
            time_in_range = time_shifted.values[mask]
            idx_min_in_range = np.argmin(velocity_in_range)
            idx_global_min = np.where(mask)[0][idx_min_in_range]
            
            t_actual_min = float(time_shifted.iloc[idx_global_min])
            v_actual_min = float(velocity_smoothed.iloc[idx_global_min])
            
            # Use the actual minimum as P3
            P3 = (t_actual_min, v_actual_min)
            logging.info(f"P3 set to actual global minimum: ({t_actual_min:.2f} ns, {v_actual_min:.2f} m/s)")
            
            # Validate that the line intersection is reasonable (within 5% of velocity range)
            if P3_intersection is not None and not pd.isna(P3_intersection[0]):
                v_range = float(np.ptp(velocity_smoothed)) if len(velocity_smoothed) > 0 else 0.0
                v_tol = max(0.05 * v_range, 5.0)  # 5% of range or 5 m/s
                if abs(P3_intersection[1] - v_actual_min) > v_tol:
                    logging.info(f"P3 line intersection ({P3_intersection[1]:.2f} m/s) differs significantly from actual minimum ({v_actual_min:.2f} m/s), using actual minimum")
        else:
            # Fallback: use pullback minimum point
            P3 = (float(t_pullback), float(velocity_smoothed.loc[idx_pullback]))
            logging.warning("Could not find minimum in range, using pullback minimum point")
    except Exception as e:
        # On any failure, fall back to the pullback minima
        P3 = (float(t_pullback), float(velocity_smoothed.loc[idx_pullback]))
        logging.warning(f"Error finding actual minimum for P3: {e}. Using pullback minimum point")

    # Handle P4 if missing (less critical for spall strength calculation)
    if P4 is None or pd.isna(P4[0]):
        # Use point at t_peak2 or end of signal
        if idx_peak2 < len(velocity_smoothed):
            P4 = (float(t_peak2), float(v_peak2))
        else:
            # Use last point of signal
            P4 = (float(time_shifted.iloc[-1]), float(velocity_smoothed.iloc[-1]))
        logging.info("P4 intersection failed, using peak2 or end point")
    
    # CRITICAL: Ensure temporal ordering P1 < P2 < P3 < P4
    # P1 should be at the start of plateau (where rise ends), P2 at the end of plateau (where pullback begins)
    try:
        t_p1 = float(P1[0]) if P1 is not None and not pd.isna(P1[0]) else float(t_peak1)
        t_p2 = float(P2[0]) if P2 is not None and not pd.isna(P2[0]) else float(t_p2)
        
        # P1 should be at or before t_peak1 (start of plateau)
        if t_p1 > float(t_peak1):
            # Clamp P1 to t_peak1
            P1 = (float(t_peak1), float(v_peak1))
            t_p1 = float(t_peak1)
            logging.info("Adjusted P1 to t_peak1 to ensure it's at plateau start")
        
        # P2 should be at or after t_peak1 (end of plateau, start of pullback)
        # Since Line 3 is now fitted from P2, P2 should be valid
        if t_p2 < float(t_peak1):
            # Clamp P2 to be after P1
            t_p2 = float(t_peak1) + 0.1 * (float(t_pullback) - float(t_peak1))
            P2 = (t_p2, float(v_peak1))
            logging.info("Adjusted P2 to be after P1")
        
        # Ensure P1 < P2 (critical: P1 must come before P2)
        if t_p1 >= t_p2:
            # If they're too close or swapped, use default positions
            # P1 at t_peak1, P2 slightly after
            t_p2 = float(t_peak1) + 0.1 * (float(t_pullback) - float(t_peak1))
            P1 = (float(t_peak1), float(v_peak1))
            P2 = (t_p2, float(v_peak1))
            logging.warning(f"P1 and P2 were swapped or too close (P1={t_p1:.2f}, P2={t_p2:.2f}). Using default positions: P1 at t_peak1, P2 slightly after.")
        
        # Ensure P2 < P3
        t_p3 = float(P3[0]) if P3 is not None and not pd.isna(P3[0]) else float(t_pullback)
        if t_p2 >= t_p3:
            # Adjust P2 to be before P3
            t_p2_new = float(t_pullback) - 0.1 * (float(t_pullback) - float(t_peak1))
            P2 = (t_p2_new, float(v_peak1))
            logging.warning(f"P2 was after P3 (P2={t_p2:.2f}, P3={t_p3:.2f}). Adjusted P2 to {t_p2_new:.2f}")
        
        # Ensure P3 < P4
        t_p4 = float(P4[0]) if P4 is not None and not pd.isna(P4[0]) else float(t_peak2)
        if t_p3 >= t_p4:
            # Adjust P3 to be before P4
            t_p3_new = float(t_peak2) - 0.1 * (float(t_peak2) - float(t_pullback))
            idx_near = int(np.argmin(np.abs(time_shifted.values - t_p3_new)))
            P3 = (float(time_shifted.iloc[idx_near]), float(velocity_smoothed.iloc[idx_near]))
            logging.warning(f"P3 was after P4 (P3={t_p3:.2f}, P4={t_p4:.2f}). Adjusted P3 to {t_p3_new:.2f}")
            
    except Exception as e:
        logging.warning(f"Error during temporal ordering check: {e}. Using default positions.")
        # Use safe defaults
        P1 = (float(t_peak1), float(v_peak1))
        t_p2_default = float(t_peak1) + 0.1 * (float(t_pullback) - float(t_peak1))
        P2 = (t_p2_default, float(v_peak1))
    
    # Critical intersections (P1, P2, P3) must be valid for spall strength calculation
    if P1 is None or pd.isna(P1[0]) or P2 is None or pd.isna(P2[0]) or P3 is None or pd.isna(P3[0]):
        raise ValueError("Failed to find critical intersection points (P1, P2, or P3) required for spall strength calculation.")

    # Final validation and enforcement: ensure P1 < P2 < P3 < P4
    # This MUST be enforced - temporal order is critical for correct spall analysis
    # This is a critical requirement - if violated, adjust points to maintain order
    try:
        t_p1 = float(P1[0])
        t_p2 = float(P2[0])
        t_p3 = float(P3[0])
        t_p4 = float(P4[0])
        
        # Check if ordering is violated
        ordering_violated = not (t_p1 < t_p2 < t_p3 < t_p4)
        
        if ordering_violated:
            logging.warning(f"Temporal ordering violation detected: P1={t_p1:.2f}, P2={t_p2:.2f}, P3={t_p3:.2f}, P4={t_p4:.2f}")
            logging.warning("Enforcing temporal order: P1 < P2 < P3 < P4")
            
            # Get reference times for proper positioning
            t_peak1_float = float(t_peak1)
            t_pullback_float = float(t_pullback)
            t_peak2_float = float(t_peak2)
            
            # Calculate minimum spacing between points (1% of total time range or 0.5 ns, whichever is larger)
            time_range = t_peak2_float - t_peak1_float if t_peak2_float > t_peak1_float else 100.0
            min_spacing = max(0.01 * time_range, 0.5)
            
            # Enforce P1 < P2: P1 should be at or before peak1, P2 should be after P1
            if t_p1 >= t_p2:
                t_p1 = t_peak1_float
                t_p2 = max(t_p1 + min_spacing, t_peak1_float + 0.1 * (t_pullback_float - t_peak1_float))
                P1 = (t_p1, float(v_peak1))
                P2 = (t_p2, float(v_peak1))
                logging.warning(f"Adjusted P1={t_p1:.2f} and P2={t_p2:.2f} to ensure P1 < P2")
            
            # Enforce P2 < P3: P3 should be at or after pullback minimum
            if t_p2 >= t_p3:
                t_p2 = min(t_p3 - min_spacing, t_pullback_float - 0.1 * (t_pullback_float - t_peak1_float))
                if t_p2 <= t_p1:
                    t_p2 = t_p1 + min_spacing
                P2 = (t_p2, float(v_peak1))
                logging.warning(f"Adjusted P2={t_p2:.2f} to ensure P2 < P3")
            
            # Ensure P3 is at or after pullback minimum
            if t_p3 < t_pullback_float:
                t_p3 = t_pullback_float
                v_p3 = float(velocity_smoothed.loc[idx_pullback])
                P3 = (t_p3, v_p3)
                logging.warning(f"Adjusted P3={t_p3:.2f} to be at or after pullback minimum")
            
            # Enforce P3 < P4: P4 should be at or after peak2
            if t_p3 >= t_p4:
                t_p3 = min(t_p4 - min_spacing, t_peak2_float - 0.1 * (t_peak2_float - t_pullback_float))
                if t_p3 <= t_p2:
                    t_p3 = t_p2 + min_spacing
                # Find nearest point on curve for P3
                idx_near = int(np.argmin(np.abs(time_shifted.values - t_p3)))
                if idx_near < len(velocity_smoothed):
                    P3 = (float(time_shifted.iloc[idx_near]), float(velocity_smoothed.iloc[idx_near]))
                else:
                    P3 = (t_p3, float(velocity_smoothed.loc[idx_pullback]))
                logging.warning(f"Adjusted P3={t_p3:.2f} to ensure P3 < P4")
            
            # Ensure P4 is at or after peak2
            if t_p4 < t_peak2_float:
                t_p4 = t_peak2_float
                v_p4 = float(v_peak2)
                P4 = (t_p4, v_p4)
                logging.warning(f"Adjusted P4={t_p4:.2f} to be at or after peak2")
            
            # Final check - if still violated, use strict defaults
            t_p1_final = float(P1[0])
            t_p2_final = float(P2[0])
            t_p3_final = float(P3[0])
            t_p4_final = float(P4[0])
            
            if not (t_p1_final < t_p2_final < t_p3_final < t_p4_final):
                logging.error(f"CRITICAL: Temporal ordering still violated after adjustments: P1={t_p1_final:.2f}, P2={t_p2_final:.2f}, P3={t_p3_final:.2f}, P4={t_p4_final:.2f}")
                logging.error("Using strict default positions to enforce order")
                # Use strict defaults with guaranteed spacing
                t_p1_final = t_peak1_float
                t_p2_final = t_peak1_float + min_spacing
                t_p3_final = t_pullback_float
                t_p4_final = max(t_peak2_float, t_p3_final + min_spacing)
                P1 = (t_p1_final, float(v_peak1))
                P2 = (t_p2_final, float(v_peak1))
                # Find P3 on curve
                idx_p3 = int(np.argmin(np.abs(time_shifted.values - t_p3_final)))
                if idx_p3 < len(velocity_smoothed):
                    P3 = (float(time_shifted.iloc[idx_p3]), float(velocity_smoothed.iloc[idx_p3]))
                else:
                    P3 = (t_p3_final, float(velocity_smoothed.loc[idx_pullback]))
                P4 = (t_p4_final, float(v_peak2))
            
            # Update intersections list with corrected values
            intersections = [P1, P2, P3, P4]
            
            # Verify final ordering
            t_p1_verify = float(P1[0])
            t_p2_verify = float(P2[0])
            t_p3_verify = float(P3[0])
            t_p4_verify = float(P4[0])
            if t_p1_verify < t_p2_verify < t_p3_verify < t_p4_verify:
                logging.info(f"✓ Temporal order enforced: P1={t_p1_verify:.2f} < P2={t_p2_verify:.2f} < P3={t_p3_verify:.2f} < P4={t_p4_verify:.2f}")
            else:
                raise ValueError(f"CRITICAL: Failed to enforce temporal order. Final: P1={t_p1_verify:.2f}, P2={t_p2_verify:.2f}, P3={t_p3_verify:.2f}, P4={t_p4_verify:.2f}")
        else:
            logging.debug(f"✓ Temporal order verified: P1={t_p1:.2f} < P2={t_p2:.2f} < P3={t_p3:.2f} < P4={t_p4:.2f}")
        
        # Always update intersections list after temporal ordering check
        intersections = [P1, P2, P3, P4]
            
    except Exception as e:
        logging.error(f"CRITICAL ERROR during temporal ordering enforcement: {e}")
        logging.error(traceback.format_exc())
        # Use safe defaults that guarantee order
        try:
            t_peak1_float = float(t_peak1)
            t_pullback_float = float(t_pullback)
            t_peak2_float = float(t_peak2)
            min_spacing = 0.5
            P1 = (t_peak1_float, float(v_peak1))
            P2 = (t_peak1_float + min_spacing, float(v_peak1))
            idx_p3 = int(np.argmin(np.abs(time_shifted.values - t_pullback_float)))
            if idx_p3 < len(velocity_smoothed):
                P3 = (float(time_shifted.iloc[idx_p3]), float(velocity_smoothed.iloc[idx_p3]))
            else:
                P3 = (t_pullback_float, float(velocity_smoothed.loc[idx_pullback]))
            P4 = (max(t_peak2_float, t_pullback_float + min_spacing), float(v_peak2))
            intersections = [P1, P2, P3, P4]
            # Verify emergency defaults maintain order
            t_p1_em = float(P1[0])
            t_p2_em = float(P2[0])
            t_p3_em = float(P3[0])
            t_p4_em = float(P4[0])
            if not (t_p1_em < t_p2_em < t_p3_em < t_p4_em):
                raise ValueError(f"Emergency defaults failed to maintain order: P1={t_p1_em:.2f}, P2={t_p2_em:.2f}, P3={t_p3_em:.2f}, P4={t_p4_em:.2f}")
            logging.warning(f"Used emergency defaults to ensure temporal order: P1={t_p1_em:.2f} < P2={t_p2_em:.2f} < P3={t_p3_em:.2f} < P4={t_p4_em:.2f}")
        except Exception as e2:
            logging.error(f"CRITICAL: Even emergency defaults failed: {e2}")
            raise ValueError(f"Failed to enforce temporal order P1 < P2 < P3 < P4. Original error: {e}, Emergency error: {e2}")

    results = {}
    # Spall strength: plateau velocity (P2) as max, minima after plateau (P3) as min
    plateau_velocity = P2[1]  # Plateau velocity from intersection P2
    minima_velocity = P3[1]   # Minima after plateau from intersection P3
    delta_u_fs = abs(plateau_velocity - minima_velocity)
    results['Spall Strength (GPa)'] = 0.5 * density * acoustic_velocity * delta_u_fs * 1e-9
    
    # Strain rate: keep using slope of line 3 (pullback slope)
    pullback_slope_ns = m3
    results['Strain Rate (s^-1)'] = abs(0.5 * (pullback_slope_ns * 1e9) / acoustic_velocity)

    results['First Maxima (m/s)'] = v_peak1
    results['Plateau Mean Velocity (m/s)'] = plateau_velocity  # Use P2[1] (plateau velocity), not v_peak1
    results['Pullback Minima (m/s)'] = minima_velocity
    
    # Note: Shock stress will be calculated in HELIX using EOS (ρ * U * u_p)
    # We don't calculate it here, just provide plateau velocity
    
    # Calculate uncertainties if available
    if uncertainty_smoothed is not None and not uncertainty_smoothed.isna().all():
        try:
            # Get uncertainties at intersection points P2 (plateau) and P3 (minima)
            # Find indices closest to P2 and P3 times
            idx_p2 = int(np.argmin(np.abs(time_shifted.values - P2[0]))) if P2 is not None and not pd.isna(P2[0]) else None
            idx_p3 = int(np.argmin(np.abs(time_shifted.values - P3[0]))) if P3 is not None and not pd.isna(P3[0]) else None
            
            p2_uncertainty = uncertainty_smoothed.iloc[idx_p2] if idx_p2 is not None and idx_p2 < len(uncertainty_smoothed) else np.nan
            p3_uncertainty = uncertainty_smoothed.iloc[idx_p3] if idx_p3 is not None and idx_p3 < len(uncertainty_smoothed) else np.nan
            
            # Calculate uncertainties for spall strength (P2 - P3)
            if not pd.isna(p2_uncertainty) and not pd.isna(p3_uncertainty):
                # Uncertainty in delta_u_fs (difference between P2 and P3)
                delta_u_fs_uncertainty = np.sqrt(p2_uncertainty**2 + p3_uncertainty**2)  # Assuming independent uncertainties
                
                # Propagate to spall strength uncertainty
                results['Spall Strength Uncertainty (GPa)'] = 0.5 * density * acoustic_velocity * delta_u_fs_uncertainty * 1e-9
            else:
                results['Spall Strength Uncertainty (GPa)'] = np.nan
            
            # For strain rate, estimate uncertainty in slope of line 3
            # Line 3 starts at P2, so use t_p2 for time range
            time_range = t_pullback - t_p2
            if time_range > 0 and not pd.isna(p3_uncertainty):
                # Estimate slope uncertainty based on velocity uncertainty over time range
                slope_uncertainty = p3_uncertainty / (time_range * 1e9)  # Convert to s^-1
                results['Strain Rate Uncertainty (s^-1)'] = abs(0.5 * slope_uncertainty / acoustic_velocity)
            else:
                results['Strain Rate Uncertainty (s^-1)'] = np.nan
            
            # Plateau velocity uncertainty (for shock stress calculation in HELIX)
            if not pd.isna(p2_uncertainty):
                results['Plateau Mean Velocity Uncertainty (m/s)'] = p2_uncertainty
            else:
                results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
                
        except Exception as e:
            logger.warning(f"Could not calculate uncertainties for 5-segment model: {e}")
            results['Spall Strength Uncertainty (GPa)'] = np.nan
            results['Strain Rate Uncertainty (s^-1)'] = np.nan
            results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
    else:
        results['Spall Strength Uncertainty (GPa)'] = np.nan
        results['Strain Rate Uncertainty (s^-1)'] = np.nan
        results['Plateau Mean Velocity Uncertainty (m/s)'] = np.nan
    
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

        # Preserve full-length shifted series for plotting (to show pre-t0 data)
        df_full = df.copy()
        # Keep full aligned series for potential plotting
        current_data_dict.update({
            'x_shifted_full': df_full['time_shifted'],
            'y_original_full': df_full['velocity'],
        })

        # Analysis uses post-t0 data only
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

        # 5-SEGMENT-ONLY BEHAVIOR: Use hybrid_5_segment for all calculations
        # - Spall strength: from plateau velocity (P2) as max and minima after plateau (P3) as min
        # - Strain rate: from slope of line 3 (pullback slope)
        # - Shock stress: calculated in HELIX using EOS (ρ * U * u_p) with plateau velocity
        # Matches spall_detection_method: "5-segment" in config and SPALL_DETECTION_ALGORITHM_5SEGMENT_ONLY.md
        
        logger.debug(f"Using 5-segment method for all calculations")
        
        plateau_window = None  # For fallback method visualization (will be set if fallback is used)
        try:
            model_kwargs = {k: v for k, v in kwargs.items() if k in ['prominence_factor', 'peak_distance_ns']}
            results_5seg, lines_info_5seg, intersections_5seg = _calculate_hybrid_5_segment_features(
                df_final['time_shifted'], df_final['velocity_smoothed'], density, acoustic_velocity, uncertainty_smoothed=df_final['uncertainty'], **model_kwargs)
            logger.debug(f"5-segment method succeeded: Plateau Mean Velocity = {results_5seg.get('Plateau Mean Velocity (m/s)', 'N/A')}")
            
            # Use results directly from 5-segment method
            results.update(results_5seg)
            lines_info = lines_info_5seg
            intersections = intersections_5seg
            # A physical spall pullback minimum must remain strictly above
            # zero velocity.  Preserve the fitted trace for diagnostics, but
            # do not report a spall strength for zero/negative P3 values.
            p3_velocity = np.nan
            try:
                if intersections and len(intersections) >= 3 and intersections[2] is not None:
                    p3_velocity = float(intersections[2][1])
            except (IndexError, TypeError, ValueError):
                pass

            if not np.isfinite(p3_velocity) or p3_velocity <= 0:
                dns_reason = f"DNS: P3 velocity ({p3_velocity:.2f} m/s) must be > 0 m/s"
                results['Processing Status'] = 'DNS'
                results['Spall Strength (GPa)'] = 'DNS'
                results['Spall_OK'] = False
                results['DNS_Classification'] = dns_reason
                status = 'DNS'
                logger.info("%s: %s", base_name, dns_reason)
            else:
                status = 'Success'
        except Exception as e:
            logger.warning(f"Failed to calculate using 5-segment method: {e}")
            # Fallback: Use peak-based plateau velocity method
            logger.info("Attempting fallback: peak-based plateau velocity method (±1 ns window)")
            try:
                results_fallback, plateau_window = _calculate_peak_based_plateau_velocity(
                    df_final['time_shifted'], df_final['velocity_smoothed'], density, acoustic_velocity,
                    uncertainty_smoothed=df_final['uncertainty'], **kwargs)
                
                # Store fallback results
                results.update(results_fallback)
                results['Processing Method'] = 'Peak-based (fallback)'
                results['Error Message'] = f"5-segment failed: {str(e)}. Used peak-based method."
                
                # For fallback, we don't have spall strength or strain rate (would need minima)
                # But we have plateau velocity for shock stress calculation
                lines_info = []
                intersections = []
                status = 'Success (Fallback)'
                logger.info(f"Fallback method succeeded: Plateau Mean Velocity = {results_fallback.get('Plateau Mean Velocity (m/s)', 'N/A')} m/s")
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                results = {}
                lines_info = []
                intersections = []
                status = 'Failed'
                error_message = f"5-segment method failed: {str(e)}. Fallback also failed: {str(fallback_error)}"

    except Exception as e:
        error_message = str(e)
        logger.warning(f"  Could not process {base_name}: {e}")
    
    results['Processing Status'] = status
    results['Error Message'] = error_message
    
    if status in ['Success', 'Success (Fallback)', 'DNS'] and plot_path:
        # Pass through axis limits if provided in kwargs
        axis_limits = {
            'auto_calculate_limits': kwargs.get('auto_calculate_limits', True),
            'x_min_main': kwargs.get('x_min_main'),
            'x_max_main': kwargs.get('x_max_main'),
            'y_min_main': kwargs.get('y_min_main'),
            'y_max_main': kwargs.get('y_max_main'),
        }
        # Pass plateau velocity information for subplot annotation
        plateau_velocity = results.get('Plateau Mean Velocity (m/s)', np.nan)
        # Get DNS classification if available (from helix_analysis_toolbox or results)
        dns_classification = results.get('DNS_Classification', kwargs.get('dns_classification', None))
        # Also check multiple indicators of DNS:
        # 1. Spall_Strength_GPa being "DNS" string
        spall_strength = results.get('Spall Strength (GPa)', results.get('Spall_Strength_GPa', None))
        if isinstance(spall_strength, str) and spall_strength.upper() == 'DNS':
            if not dns_classification:
                dns_classification = "Did Not Spall (Spall_Strength=DNS)"
        # 2. Spall_OK flag being False
        spall_ok = results.get('Spall_OK', None)
        if not dns_classification and spall_ok is False:
            dns_classification = "Did Not Spall (Spall_OK=False)"
        # 3. Check if P4 <= P3 (no re-acceleration) - this is a strong DNS indicator
        # This check happens in helix_analysis_toolbox after SPADE, but we can check the intersections here
        if not dns_classification and intersections and len(intersections) >= 4:
            try:
                P3 = intersections[2]  # P3 is the pullback minimum
                P4 = intersections[3]  # P4 is the recompression tail intersection
                # Check if P3 and P4 are valid (not None and not NaN)
                p3_valid = (P3 is not None and not pd.isna(P3[0]) and not pd.isna(P3[1]))
                p4_valid = (P4 is not None and not pd.isna(P4[0]) and not pd.isna(P4[1]))
                if p3_valid and p4_valid:
                    v_P3 = float(P3[1])
                    v_P4 = float(P4[1])
                    if v_P4 <= v_P3:
                        dns_classification = f"No re-acceleration after pullback: P4 velocity ({v_P4:.2f} m/s) <= P3 velocity ({v_P3:.2f} m/s)"
            except (IndexError, ValueError, TypeError):
                pass
        # Debug logging
        if dns_classification:
            logger.debug(f"DNS classification found for plotting: {dns_classification}")
        # plateau_window is set in the try/except block above if fallback method is used
        _plot_analysis_results(current_data_dict, lines_info, intersections, plot_path, 
                              axis_limits=axis_limits, plateau_velocity=plateau_velocity, 
                              plateau_window=plateau_window, dns_classification=dns_classification)
            
    return results, lines_info, intersections

def process_velocity_files(
    input_folder, file_pattern, output_folder,
    save_summary_table=True, summary_table_name="enhanced_spall_summary.csv", files_list=None, **kwargs
):
    """
    Processes all matching velocity files in a folder.
    Also generates STFT vs Phase Velocity comparison plots when both files are available.
    """
    # Either use explicit provided list or collect by pattern (supports recursive "**")
    if files_list:
        files_to_process = [f for f in files_list if os.path.isfile(f)]
    else:
        recursive = '**' in file_pattern
        files_to_process = sorted(glob.glob(os.path.join(input_folder, file_pattern), recursive=recursive))
    if not files_to_process:
        logger.warning(f"No files found for pattern '{file_pattern}' in '{input_folder}'.")
        return pd.DataFrame()

    logger.info(f"Processing {len(files_to_process)} files from '{input_folder}'...")
    
    results_list = []
    subfolder_name = os.path.basename(os.path.normpath(input_folder))
    material_label = utils.extract_legend_info(subfolder_name, utils.MATERIAL_MAPPING, utils.ENERGY_VELOCITY_MAPPING)[0] or "Unknown"

    calc_args = {k: v for k, v in kwargs.items() if k not in ['input_folder', 'file_pattern', 'output_folder', 'save_summary_table', 'summary_table_name', 'plot_stft_phase_comparison', 'comparison_x_min', 'comparison_x_max', 'comparison_y_min', 'comparison_y_max']}
    if 'density' not in calc_args or 'acoustic_velocity' not in calc_args:
        raise ValueError("Critical: 'density' and 'acoustic_velocity' must be provided via kwargs.")

    # Track STFT files and their corresponding Phase Velocity files for comparison plots
    stft_files_map = {}  # Maps base filename to STFT file path
    
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
            
            # Track STFT files for comparison plots
            if '--vel-smooth-with-uncert' in filename_no_ext:
                # Extract base filename (remove --vel-smooth-with-uncert suffix)
                base_name = filename_no_ext.replace('--vel-smooth-with-uncert', '')
                stft_files_map[base_name] = file_path

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
    
    # Generate STFT vs Phase Velocity comparison plots
    if kwargs.get('plot_stft_phase_comparison', True) and stft_files_map:
        logger.info("Generating STFT vs Phase Velocity comparison plots...")
        from . import plotting
        
        comparison_x_min = kwargs.get('comparison_x_min', -10.0)
        comparison_x_max = kwargs.get('comparison_x_max', 60.0)
        comparison_y_min = kwargs.get('comparison_y_min', None)
        comparison_y_max = kwargs.get('comparison_y_max', None)
        
        for base_name, stft_file_path in stft_files_map.items():
            # Look for corresponding Phase Velocity file
            phase_velocity_file = os.path.join(input_folder, f"{base_name}--velocity--phase.csv")
            
            # Check for original STFT file (if hybrid velocity was used)
            stft_original_file = os.path.join(input_folder, f"{base_name}--vel-smooth-stft-original.csv")
            if os.path.exists(stft_original_file):
                logger.info(f"  Found original STFT velocity file (hybrid mode was used)")
                stft_file_path = stft_original_file  # Use original STFT instead of hybrid
            
            if os.path.exists(phase_velocity_file):
                comparison_plot_path = os.path.join(output_folder, f"{base_name}_stft_vs_phase_comparison.png")
                logger.info(f"  Creating comparison plot: {os.path.basename(comparison_plot_path)}")
                
                try:
                    plotting.plot_stft_vs_phase_velocity_comparison(
                        stft_file_path=stft_file_path,
                        phase_velocity_file_path=phase_velocity_file,
                        output_filename=comparison_plot_path,
                        title=f"STFT vs Phase Velocity Comparison: {base_name}",
                        x_min=comparison_x_min,
                        x_max=comparison_x_max,
                        y_min=comparison_y_min,
                        y_max=comparison_y_max,
                        stft_color='blue',
                        phase_color='red',
                        stft_label='STFT Velocity (Smooth)',
                        phase_label='Phase Velocity (IQ)',
                        include_uncertainty=True,
                        uncertainty_alpha=0.2
                    )
                except Exception as e:
                    logger.error(f"Error creating comparison plot for {base_name}: {e}", exc_info=True)
            else:
                logger.debug(f"  Phase Velocity file not found for {base_name}, skipping comparison plot")
        
    return summary_df
