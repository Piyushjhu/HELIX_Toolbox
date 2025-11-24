#!/usr/bin/env python3
"""
ALPSS + SPADE Combined GUI
A comprehensive GUI for running ALPSS analysis followed by SPADE spall analysis
"""
import sys
import os
import glob
import subprocess
import threading
import time
import json
import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend BEFORE importing pyplot or SPADE to avoid macOS aborts
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Excel support will be checked dynamically when needed
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QPlainTextEdit, QProgressBar,
    QFileDialog, QCheckBox, QComboBox, QSpinBox, QRadioButton, QButtonGroup,
    QDoubleSpinBox, QGroupBox, QScrollArea, QMessageBox,
    QSplitter, QFrame, QStyleFactory, QTabBar, QListWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
from PyQt5.QtGui import QFont, QValidator
from SPADE.spall_analysis_release.spall_analysis import (
    plot_combined_mean_traces,
    plot_spall_vs_strain_rate,
    plot_spall_vs_shock_stress,
    plot_shock_stress_vs_laser_energy,
)
from datetime import datetime
from material_properties import get_material_properties, list_available_materials

def cleanup_matplotlib():
    """Clean up matplotlib figures to prevent memory leaks"""
    import matplotlib.pyplot as plt
    plt.close('all')  # Close all figures
    plt.clf()  # Clear current figure
    plt.cla()  # Clear current axes

def save_config_to_file(config_dict, file_path):
    """Save configuration dictionary to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        return True, f"Configuration saved to {file_path}"
    except Exception as e:
        return False, f"Error saving config: {str(e)}"

def load_config_from_file(file_path):
    """Load configuration dictionary from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return True, config_dict, f"Configuration loaded from {file_path}"
    except Exception as e:
        return False, {}, f"Error loading config: {str(e)}"

class ScientificSpinBox(QDoubleSpinBox):
    """Custom spin box that accepts scientific notation input"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(15)  # Allow high precision
        # Ensure value only updates when editing is finished (Enter/blur)
        self.setKeyboardTracking(False)

    def _strip_prefix_suffix(self, text: str) -> str:
        """Remove any prefix/suffix and normalize scientific notation markers."""
        if text is None:
            return ""
        cleaned = text.strip()
        # Remove suffix/prefix if present
        suf = self.suffix()
        pre = self.prefix()
        if suf:
            cleaned = cleaned.replace(suf, "")
        if pre:
            cleaned = cleaned.replace(pre, "")
        # Normalize scientific notation
        cleaned = cleaned.replace('E+', 'e+').replace('E-', 'e-').replace('E', 'e')
        return cleaned.strip()

    def textFromValue(self, value):
        """Convert value to scientific notation string with high precision"""
        if abs(value) >= 1e6 or (abs(value) < 1e-3 and value != 0):
            # For scientific notation, preserve more significant figures
            return f"{value:.9e}"
        else:
            # For regular numbers, preserve more significant figures
            return f"{value:.9g}"

    def valueFromText(self, text):
        """Convert scientific notation string to value"""
        try:
            cleaned = self._strip_prefix_suffix(text)
            # Empty field should not force-reset; keep current value
            if cleaned == "":
                return self.value()
            return float(cleaned)
        except ValueError:
            # Keep current value on parse error instead of resetting
            return self.value()

    def validate(self, text, pos):
        """Validate scientific notation input"""
        try:
            cleaned = self._strip_prefix_suffix(text)
            if cleaned == "":
                return (QValidator.Acceptable, text, pos)
            # Allow partial input during typing
            if cleaned.endswith('e'):
                return (QValidator.Intermediate, text, pos)
            if cleaned.endswith('e+') or cleaned.endswith('e-'):
                return (QValidator.Intermediate, text, pos)
            # Try to parse the value
            float(cleaned)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            return (QValidator.Invalid, text, pos)


class AnalysisThread(QThread):
    """Thread for running ALPSS and SPADE analysis"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
    self,
    alpss_params,
    spade_params,
    input_files,
    output_dir,
    param_data=None,
    spade_auto_mode=True,
    spade_input_files=None,
     analysis_mode="both",
     material_properties=None):
        super().__init__()
        self.alpss_params = alpss_params
        self.spade_params = spade_params
        self.input_files = input_files
        self.output_dir = output_dir
        self.param_data = param_data  # Parameter file data mapping
        self.spade_auto_mode = spade_auto_mode
        self.spade_input_files = spade_input_files
        self.analysis_mode = analysis_mode  # "alpss_only", "spade_only", or "both"
        self.material_properties = material_properties or {}  # Material properties from config

        # Initialize trace counting for summary
        self.total_input_traces = 0
        self.traces_plotted = 0
        self.traces_rejected = 0
        self.rejection_reasons = {}  # Track reasons for rejection
        self._warned_skip_unknown_override = False

    def _should_skip_unknown_materials(self):
        """
        Determine whether Unknown-material traces should be skipped.
        Automatically disables skipping when no parameter file is loaded so
        combined plots are not empty.
        """
        skip_requested = bool(self.spade_params.get('skip_unknown_material_traces', False))
        param_available = bool(self.param_data)
        if skip_requested and not param_available:
            if not self._warned_skip_unknown_override:
                self.progress_signal.emit(
                    "⚠️  Parameter file not selected; including 'Unknown' material traces in combined plots."
                )
                self._warned_skip_unknown_override = True
            return False
        return skip_requested

    def get_param_data_for_file(self, base_name):
        """
        Get parameter data for a filename using multiple matching strategies.
        Tries: 1) Exact match, 2) Date-shot pattern match, 3) Partial match
        
        Args:
            base_name: Base filename (without extension and suffixes)
            
        Returns:
            Dictionary of parameter data, or empty dict if no match found
        """
        import re
        
        if not self.param_data:
            return {}
        
        # Try exact match first
        if base_name in self.param_data:
            return self.param_data[base_name]
        
        # Extract date-shot pattern (YYYYMMDD--NNNNN) for matching
        date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
        
        if date_shot_pattern:
            date_shot = date_shot_pattern.group(1)
            # Try matching with just date and shot number
            for key in self.param_data.keys():
                if date_shot in str(key):
                    self.progress_signal.emit(f"Date-shot match: {base_name} -> {key} (using {date_shot})")
                    return self.param_data[key]
        
        # Try partial match as fallback
        for key in self.param_data.keys():
            if base_name in str(key) or str(key) in base_name:
                self.progress_signal.emit(f"Partial match: {base_name} -> {key}")
                return self.param_data[key]
        
        return {}

    def get_material_properties_from_config(self, material_name, default_density=None, default_acoustic_velocity=None):
        """
        Get material properties from config first, then fall back to database.
        
        Priority order:
        1. Config material_properties section (if material found)
        2. Parameter file explicit columns (Density_kg_m3, Bulk_Wave_Speed_m_s)
        3. Material properties database (material_properties.py)
        4. Config defaults (spade_params density/acoustic_velocity or alpss_params density/C0)
        5. Hardcoded defaults (Copper properties)
        
        Args:
            material_name: Name of the material to look up
            default_density: Default density to use if not found (kg/m³)
            default_acoustic_velocity: Default bulk wave speed to use if not found (m/s)
            
        Returns:
            Dictionary with 'density', 'bulk_wave_speed', 'material_found', 'material_name', 'source'
        """
        # Clean material name
        material_name = str(material_name).strip() if material_name else 'Unknown'
        
        # Get defaults from config if not provided
        if default_density is None:
            default_density = self.spade_params.get('density') or self.alpss_params.get('density', 8960)
        if default_acoustic_velocity is None:
            default_acoustic_velocity = (self.spade_params.get('acoustic_velocity') or 
                                       self.alpss_params.get('C0') or 3950)
        
        # Priority 1: Check config material_properties section
        if self.material_properties and material_name in self.material_properties:
            config_props = self.material_properties[material_name]
            # Default C_L to bulk_wave_speed if not specified
            default_C_L = config_props.get('bulk_wave_speed', config_props.get('C0', default_acoustic_velocity))
            return {
                'density': float(config_props.get('density', default_density)),
                'bulk_wave_speed': float(config_props.get('bulk_wave_speed', 
                                                          config_props.get('C0', default_acoustic_velocity))),
                'C_L': float(config_props.get('C_L', default_C_L)),
                'material_found': True,
                'material_name': material_name,
                'source': 'config'
            }
        
        # Try case-insensitive match in config
        if self.material_properties:
            for config_mat_name, config_props in self.material_properties.items():
                if config_mat_name.lower() == material_name.lower():
                    # Default C_L to bulk_wave_speed if not specified
                    default_C_L = config_props.get('bulk_wave_speed', config_props.get('C0', default_acoustic_velocity))
                    return {
                        'density': float(config_props.get('density', default_density)),
                        'bulk_wave_speed': float(config_props.get('bulk_wave_speed',
                                                                  config_props.get('C0', default_acoustic_velocity))),
                        'C_L': float(config_props.get('C_L', default_C_L)),
                        'material_found': True,
                        'material_name': config_mat_name,
                        'source': 'config'
                    }
        
        # Priority 2 & 3: Use database function (which also checks parameter file if passed)
        mat_props = get_material_properties(material_name, default_density, default_acoustic_velocity)
        mat_props['source'] = 'database' if mat_props['material_found'] else 'default'
        # Add C_L if not present (default to bulk_wave_speed for database materials)
        if 'C_L' not in mat_props:
            mat_props['C_L'] = mat_props.get('bulk_wave_speed', default_acoustic_velocity)
        return mat_props

    def detect_dns_and_process_spall(self, file_path, base_name, density, acoustic_velocity, 
                                     threshold_velocity, spall_start_time, spall_end_time,
                                     analysis_model='max_min', plot_path=None, **spade_kwargs):
        """
        Detect DNS (Did Not Spall) and process spall analysis following Binary_metal_analysis methodology.
        
        This function implements the detailed spall detection methodology:
        1. Loads and filters data
        2. Aligns trace to shock arrival
        3. Extracts time window
        4. Performs DNS detection (structural checks)
        5. Only calls SPADE for valid spall cases
        
        Returns:
            dict: Results dictionary with spall strength, DNS classification, and diagnostics
        """
        import pandas as pd
        from scipy.signal import find_peaks
        from scipy.ndimage import uniform_filter1d
        
        results = {
            'Filename': base_name,
            'Spall_Strength_GPa': np.nan,
            'Spall_Strength_Unc_GPa': np.nan,
            'Spall_OK': False,
            'Spall_StrainRate_s^-1': np.nan,
            'First_Maxima_m_s': np.nan,
            'Minima_m_s': np.nan,
            'Second_Maxima_m_s': np.nan,
            'Pullback_Velocity_m_s': np.nan,
            'Pullback_Velocity_Unc_m_s': np.nan,
            'Processing_Status': 'Failed',
            'DNS_Classification': 'Unknown'
        }
        
        try:
            # Step 1: Load CSV and handle headers
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                results['Processing_Status'] = f'Failed: Could not load file: {str(e)}'
                return results
            
            # Detect header row
            has_header = False
            first_row = df.iloc[0] if len(df) > 0 else None
            if first_row is not None:
                first_row_str = ' '.join([str(x).lower() for x in first_row.values[:3]])
                if any(keyword in first_row_str for keyword in ['time', 'velocity', 'uncertainty']):
                    has_header = True
                    df = pd.read_csv(file_path, header=0)
                else:
                    df = pd.read_csv(file_path, header=None)
            
            # Extract first 3 columns
            if len(df.columns) < 3:
                results['Processing_Status'] = 'Failed: Insufficient columns (< 3)'
                return results
            
            time_s = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            velocity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            uncertainty = pd.to_numeric(df.iloc[:, 2], errors='coerce').values if len(df.columns) >= 3 else np.full_like(velocity, np.nan)
            
            if len(time_s) < 20:
                results['Processing_Status'] = 'Failed: Insufficient data points (< 20)'
                return results
            
            # Step 1b: Apply uncertainty filter
            max_vel = np.nanmax(np.abs(velocity))
            rel_unc = np.abs(uncertainty) / max(max_vel, 1e-9)
            vel_clean = velocity.copy()
            vel_clean[rel_unc >= 1.0] = np.nan
            
            # Step 2: Trace alignment to shock arrival
            valid_mask = ~np.isnan(vel_clean)
            if not np.any(valid_mask):
                results['Processing_Status'] = 'Failed: No valid velocity data after filtering'
                return results
            
            vel_valid = vel_clean[valid_mask]
            time_valid = time_s[valid_mask]
            
            # Find first index where velocity >= threshold
            threshold_idx = np.where(vel_valid >= threshold_velocity)[0]
            if len(threshold_idx) == 0:
                results['Processing_Status'] = 'Failed: No shock arrival detected (threshold not reached)'
                return results
            
            t0 = time_valid[threshold_idx[0]]
            t_aligned_ns = (time_s - t0) * 1e9
            
            # Step 3: Time window extraction
            window_mask = (~np.isnan(vel_clean)) & (t_aligned_ns >= spall_start_time) & (t_aligned_ns <= spall_end_time)
            if np.sum(window_mask) < 20:
                results['Processing_Status'] = 'Failed: Insufficient data points in spall window (< 20)'
                return results
            
            time_window = t_aligned_ns[window_mask]
            vel_window = vel_clean[window_mask]
            uncert_window = uncertainty[window_mask]
            
            # Step 4: DNS Detection - Structural Requirements
            # 4a: Peak and valley detection (standard detection for structure validation)
            # Note: Expected times are used in max_min calculation, not in DNS detection
            # DNS detection checks overall structure to determine if spall signature exists
            prominence = np.nanstd(vel_window) * 0.1
            peaks, _ = find_peaks(vel_window, prominence=prominence)
            valleys, _ = find_peaks(-vel_window, prominence=prominence)
            
            # 4b: Structural requirements for valid spall
            dns_reason = None
            
            if len(peaks) == 0 or len(valleys) == 0:
                dns_reason = "No clear peak/valley structure"
            else:
                first_peak_idx = peaks[0]
                valleys_after_peak = valleys[valleys > first_peak_idx]
                
                if len(valleys_after_peak) == 0:
                    dns_reason = "No pullback after initial rise"
                else:
                    first_valley_idx = valleys_after_peak[0]
                    peaks_after_valley = peaks[peaks > first_valley_idx]
                    
                    if len(peaks_after_valley) == 0:
                        dns_reason = "No re-acceleration after pullback"
            
            # 4c: DNS Classification
            if dns_reason:
                results['Spall_Strength_GPa'] = "DNS"
                results['Spall_Strength_Unc_GPa'] = np.nan
                results['Spall_OK'] = False
                results['Processing_Status'] = f'DNS: {dns_reason}'
                results['DNS_Classification'] = dns_reason
                if plot_path:
                    try:
                        self._plot_generic_spall_analysis(
                            plot_path, time_window, vel_window, uncert_window,
                            peaks[0] if len(peaks) > 0 else None,
                            valleys_after_peak[0] if 'valleys_after_peak' in locals() and len(valleys_after_peak) > 0 else None,
                            "DNS", np.nan, base_name, analysis_model=analysis_model
                        )
                    except Exception as plot_error:
                        import traceback
                        self.progress_signal.emit(f"  Warning: Could not generate DNS plot: {str(plot_error)}")
                        self.progress_signal.emit(traceback.format_exc())
                return results
            
            # Step 5: Valid Spall Case - Extract Key Velocities
            first_peak_idx = peaks[0]
            first_valley_idx = valleys_after_peak[0]
            second_peak_idx = peaks_after_valley[0]
            
            first_peak_vel = vel_window[first_peak_idx]
            first_valley_vel = vel_window[first_valley_idx]
            second_peak_vel = vel_window[second_peak_idx]
            
            # Compute pullback velocity uncertainty
            peak_unc = uncert_window[first_peak_idx] if first_peak_idx < len(uncert_window) else 0
            valley_unc = uncert_window[first_valley_idx] if first_valley_idx < len(uncert_window) else 0
            pullback_unc = np.sqrt(peak_unc**2 + valley_unc**2) if np.isfinite(peak_unc) and np.isfinite(valley_unc) else np.nan
            
            # Store diagnostic velocities
            results['First_Maxima_m_s'] = first_peak_vel
            results['Minima_m_s'] = first_valley_vel
            results['Second_Maxima_m_s'] = second_peak_vel
            results['Pullback_Velocity_m_s'] = abs(first_peak_vel - first_valley_vel)
            results['Pullback_Velocity_Unc_m_s'] = pullback_unc
            
            # Step 6: SPADE Analysis (only for valid spall cases)
            from SPADE.spall_analysis_release.spall_analysis.data_processing import calculate_spall_parameters
            spade_lines_info = None
            spade_intersections = None

            result_dict, spade_lines_info, spade_intersections = calculate_spall_parameters(
                time_ns=time_window,
                velocity_ms=vel_window,
                uncertainty_ms=uncert_window,
                density=density,
                acoustic_velocity=acoustic_velocity,
                analysis_model=analysis_model,
                plot_path=plot_path,
                **{k: v for k, v in spade_kwargs.items() if k not in ['density', 'acoustic_velocity', 'analysis_model']}
            )
            
            # Extract spall strength with flexible key matching
            spall_strength = np.nan
            for key in result_dict.keys():
                if 'spall' in key.lower() and 'strength' in key.lower() and 'gpa' in key.lower() and 'unc' not in key.lower() and 'err' not in key.lower():
                    try:
                        val = result_dict[key]
                        if isinstance(val, str) and val.upper() == 'DNS':
                            spall_strength = "DNS"
                        else:
                            spall_strength = float(val) if pd.notna(val) else np.nan
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Extract uncertainty
            spall_unc = np.nan
            for key in result_dict.keys():
                if ('unc' in key.lower() or 'err' in key.lower()) and 'spall' in key.lower() and 'gpa' in key.lower():
                    try:
                        spall_unc = float(result_dict[key]) if pd.notna(result_dict[key]) else np.nan
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Step 7: Uncertainty fallback calculation
            if pd.isna(spall_unc) and np.isfinite(pullback_unc) and np.isfinite(density) and np.isfinite(acoustic_velocity):
                spall_unc = 0.5 * density * acoustic_velocity * pullback_unc / 1e9
            
            # Extract strain rate
            strain_rate = result_dict.get('Strain Rate (s^-1)', np.nan)
            if pd.isna(strain_rate):
                strain_rate = result_dict.get('Strain_Rate_s^-1', np.nan)
            
            # Step 8: Final classification
            if result_dict.get('Processing Status') == 'Success':
                results['Spall_OK'] = True
                results['Processing_Status'] = 'Success'
            else:
                results['Spall_OK'] = False
                results['Processing_Status'] = result_dict.get('Processing Status', 'Failed: SPADE analysis failed')
            
            results['Spall_Strength_GPa'] = spall_strength
            results['Spall_Strength_Unc_GPa'] = spall_unc
            results['Spall_StrainRate_s^-1'] = strain_rate
            results['DNS_Classification'] = 'Valid Spall' if results['Spall_OK'] else 'Failed'
            
            # Generate plot if plot_path is provided
            if plot_path:
                if analysis_model == 'hybrid_5_segment':
                    self.progress_signal.emit("  Hybrid 5-segment plot saved directly by SPADE (GUI overlay skipped to preserve SPADE styling).")
                else:
                    self.progress_signal.emit(f"  [DEBUG] Plot path provided for {base_name}: {plot_path}")
                    try:
                        self._plot_generic_spall_analysis(
                            plot_path, time_window, vel_window, uncert_window,
                            first_peak_idx, first_valley_idx,
                            results.get('Spall_Strength_GPa', spall_strength),
                            results.get('Spall_Strength_Unc_GPa', spall_unc),
                            base_name,
                            analysis_model=analysis_model,
                            lines_info=spade_lines_info,
                            intersections=spade_intersections
                        )
                        self.progress_signal.emit(f"  [DEBUG] Generated plot for {base_name}")
                    except Exception as plot_error:
                        import traceback
                        self.progress_signal.emit(f"  Warning: Could not generate spall plot: {str(plot_error)}")
                        self.progress_signal.emit(f"  [DEBUG] Plot error traceback: {traceback.format_exc()}")
            else:
                self.progress_signal.emit(f"  [DEBUG] No plot_path provided for {base_name}")
        
        except Exception as e:
            import traceback
            results['Processing_Status'] = f'Failed: {str(e)}'
            results['DNS_Classification'] = 'Error'
            self.progress_signal.emit(f"Error in DNS detection for {base_name}: {str(e)}")
            self.progress_signal.emit(traceback.format_exc())
        
        return results


    def _plot_generic_spall_analysis(self, plot_path, time_window, vel_window, uncert_window,
                                     peak_idx, valley_idx, spall_strength, spall_unc, base_name,
                                     analysis_model='max_min', lines_info=None, intersections=None):
        """Generate generic spall analysis plot for any analysis model"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot velocity trace
            ax.plot(time_window, vel_window, 'b-', linewidth=1.5, alpha=0.7, label='Velocity')
            
            # Plot uncertainty bands if available
            if uncert_window is not None and len(uncert_window) == len(vel_window):
                ax.fill_between(time_window, vel_window - uncert_window, vel_window + uncert_window,
                               alpha=0.2, color='blue', label='Uncertainty')
            
            # Mark peak and valley (max_min diagnostics)
            if peak_idx is not None and peak_idx < len(time_window):
                ax.plot(time_window[peak_idx], vel_window[peak_idx], 'ro', markersize=10, 
                       label=f'Peak: {vel_window[peak_idx]:.1f} m/s')
            if valley_idx is not None and valley_idx < len(time_window):
                ax.plot(time_window[valley_idx], vel_window[valley_idx], 'go', markersize=10,
                       label=f'Valley: {vel_window[valley_idx]:.1f} m/s')
            
            # Draw line between peak and valley if both exist
            if (peak_idx is not None and valley_idx is not None and 
                peak_idx < len(time_window) and valley_idx < len(time_window)):
                ax.plot([time_window[peak_idx], time_window[valley_idx]], 
                       [vel_window[peak_idx], vel_window[valley_idx]], 
                       'r--', linewidth=2, alpha=0.5, label='Pullback')

            # Overlay hybrid 5-segment fit if requested
            if analysis_model == 'hybrid_5_segment' and lines_info and intersections:
                self._overlay_hybrid_segments(ax, time_window, lines_info, intersections)
            
            # Format title with uncertainty if available
            strength_val = None
            try:
                if spall_strength is not None and not isinstance(spall_strength, str):
                    strength_val = float(spall_strength)
                elif isinstance(spall_strength, str):
                    strength_val = float(spall_strength)
            except (TypeError, ValueError):
                strength_val = None
            
            if strength_val is not None and np.isfinite(strength_val):
                if spall_unc is not None and not np.isnan(spall_unc) and spall_unc > 0:
                    title = (
                        f'Spall Analysis: {base_name}\n'
                        f'Spall Strength: {strength_val:.3f} ± {spall_unc:.3f} GPa'
                    )
                else:
                    title = (
                        f'Spall Analysis: {base_name}\n'
                        f'Spall Strength: {strength_val:.3f} GPa'
                    )
            else:
                strength_label = str(spall_strength) if spall_strength is not None else "Unknown"
                title = (
                    f'Spall Analysis: {base_name}\n'
                    f'Spall Strength: {strength_label}'
                )
            
            ax.set_xlabel('Time (ns)', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.progress_signal.emit(f"  Saved spall plot: {os.path.basename(plot_path)}")
        except Exception as e:
            self.progress_signal.emit(f"  Warning: Could not generate spall plot: {str(e)}")

    def _overlay_hybrid_segments(self, ax, time_window, lines_info, intersections):
        """Overlay 5-segment hybrid model lines and key points on the velocity plot."""
        import numpy as np
        
        if not lines_info:
            return
        
        t_min = float(np.min(time_window)) if len(time_window) > 0 else 0.0
        t_max = float(np.max(time_window)) if len(time_window) > 0 else 0.0
        
        # Build boundary times using intersections (P1..P4)
        boundary_times = [t_min]
        for point in intersections or []:
            if point and len(point) >= 2 and not any(pd.isna(coord) for coord in point):
                boundary_times.append(float(point[0]))
        boundary_times.append(t_max)
        
        # Ensure we have exactly len(lines)+1 boundaries
        while len(boundary_times) <= len(lines_info):
            boundary_times.append(boundary_times[-1] + 1.0)
        if len(boundary_times) > len(lines_info) + 1:
            boundary_times = boundary_times[:len(lines_info) + 1]
        
        colors = ['#FF8C00', '#B22222', '#228B22', '#1E90FF', '#8A2BE2']
        colors = (colors * ((len(lines_info) // len(colors)) + 1))[:len(lines_info)]
        
        for idx, ((m, c), color) in enumerate(zip(lines_info, colors)):
            t_start = boundary_times[idx]
            t_end = boundary_times[idx + 1]
            if t_end <= t_start:
                continue
            t_vals = np.linspace(t_start, t_end, 100)
            y_vals = m * t_vals + c
            ax.plot(t_vals, y_vals, '--', linewidth=1.2, color=color, label=f'Segment {idx + 1}')
        
        # Mark intersection points
        if intersections:
            for idx, point in enumerate(intersections, start=1):
                if not point or any(pd.isna(coord) for coord in point):
                    continue
                ax.plot(point[0], point[1], marker='X', color='black', markersize=7)
                ax.text(point[0], point[1], f"P{idx}", fontsize=9, color='black',
                        ha='left', va='bottom')

    def run(self):
        try:
            # Add memory management
            import gc
            gc.collect()  # Force garbage collection before starting
            
            # Start timing the entire analysis
            self.start_time = time.time()

            # Import ALPSS and SPADE modules
            sys.path.append('ALPSS')
            sys.path.append('SPADE/spall_analysis_release')

            from alpss_main import alpss_main
            from spall_analysis import process_velocity_files


            # Set matplotlib backend to Agg for thread safety on macOS
            # This prevents crashes when matplotlib runs in background thread
            import matplotlib
            matplotlib.use("Agg")
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Initialize successful_files list for both ALPSS and SPADE modes
            self.successful_files = []
            
            # Track total input files for summary
            if self.input_files:
                self.total_input_traces = len(self.input_files)
            else:
                self.total_input_traces = 0

            # Process ALPSS files if provided and not SPADE-only mode
            if self.analysis_mode != "spade_only" and self.input_files:
                total_alpss_time = 0

                # Process all files
                files_to_process = self.input_files
                failed_files = []

                for i, input_file in enumerate(files_to_process):
                    self.progress_signal.emit(
                        f"ALPSS Processing file {i+1}/{len(files_to_process)}: {os.path.basename(input_file)}")

                    # Start timing for this file
                    file_start_time = time.time()

                    # Run ALPSS with error handling
                    # Memory management before ALPSS
                    gc.collect()
                    self.progress_signal.emit("Running ALPSS analysis...")

                    # Update ALPSS parameters with current file
                    alpss_params = self.alpss_params.copy()
                    alpss_params['filename'] = os.path.basename(input_file)
                    alpss_params['exp_data_dir'] = os.path.dirname(input_file)
                    alpss_params['out_files_dir'] = self.output_dir

                    # Add experiment info if parameter data is available
                    if self.param_data:
                        base_name = os.path.splitext(
                            os.path.basename(input_file))[0]
                        # Use helper function for smart matching (exact, date-shot pattern, or partial)
                        exp_info = self.get_param_data_for_file(base_name)
                        if exp_info:
                            alpss_params['experiment_info'] = exp_info
                            # Handle different possible column names for experiment info
                            exp_id = exp_info.get('exp_id', exp_info.get('Exp_ID', 'Unknown'))
                            sample_material = exp_info.get('sample_material', exp_info.get('Flyer_material', 'Unknown'))
                            self.progress_signal.emit(
                                f"Linked to experiment: {exp_id} - {sample_material}")
                        else:
                            self.progress_signal.emit(
                                f"No experiment info found for {base_name}")
                            alpss_params['experiment_info'] = {}
                    else:
                        alpss_params['experiment_info'] = {}

                    # Pass image selection parameters to ALPSS
                    alpss_params['save_combined_plot'] = self.alpss_params.get(
                        'save_combined_plot', True)
                    alpss_params['save_iq_start_time_plot'] = self.alpss_params.get(
                        'save_iq_start_time_plot', False)
                    
                    # Handle smart selection for combined mode
                    smart_selection_enabled = self.alpss_params.get('smart_selection_enabled', False)
                    if smart_selection_enabled and self.analysis_mode == "both":
                        # For combined mode, save files needed for SPADE + enhanced analysis
                        alpss_params['save_velocity_csv'] = False
                        alpss_params['save_velocity_smooth_csv'] = False
                        alpss_params['save_velocity_uncert_csv'] = False
                        alpss_params['save_velocity_smooth_uncert_csv'] = True  # Main file needed for SPADE
                        alpss_params['save_results_csv'] = False
                        alpss_params['save_noise_csv'] = True  # Also save noise file for enhanced filtering
                    else:
                        # Pass output file selection parameters to ALPSS
                        alpss_params['save_velocity_csv'] = self.alpss_params.get('save_velocity_csv', True)
                        alpss_params['save_velocity_smooth_csv'] = self.alpss_params.get('save_velocity_smooth_csv', True)
                        alpss_params['save_velocity_uncert_csv'] = self.alpss_params.get('save_velocity_uncert_csv', True)
                        alpss_params['save_velocity_smooth_uncert_csv'] = self.alpss_params.get('save_velocity_smooth_uncert_csv', True)
                        alpss_params['save_results_csv'] = self.alpss_params.get('save_results_csv', True)
                        alpss_params['save_noise_csv'] = self.alpss_params.get('save_noise_csv', True)

                    try:
                        alpss_main(**alpss_params)

                        # Check if required files were generated based on analysis mode
                        base_name = os.path.splitext(
                            os.path.basename(input_file))[0]
                        
                        # Define required files based on analysis mode and smart selection
                        required_files = []
                        if smart_selection_enabled and self.analysis_mode == "both":
                            # For combined mode with smart selection, check for SPADE input file + noise file
                            required_files.append(os.path.join(
                                self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv"))
                            required_files.append(os.path.join(
                                self.output_dir, f"{base_name}--noise--frac.csv"))
                        else:
                            # Check all selected output files
                            if self.alpss_params.get('save_velocity_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--velocity.csv"))
                            if self.alpss_params.get('save_velocity_smooth_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--velocity--smooth.csv"))
                            if self.alpss_params.get('save_velocity_uncert_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--vel--uncert.csv"))
                            if self.alpss_params.get('save_velocity_smooth_uncert_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv"))
                            if self.alpss_params.get('save_results_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--results.csv"))
                            if self.alpss_params.get('save_noise_csv', True):
                                required_files.append(os.path.join(
                                    self.output_dir, f"{base_name}--noise--frac.csv"))
                        
                        # Check if all required files exist
                        missing_files = [f for f in required_files if not os.path.exists(f)]
                        
                        if not missing_files:
                            self.successful_files.append(input_file)
                            self.progress_signal.emit(
                                f"✅ Successfully processed: {os.path.basename(input_file)}")
                        else:
                            missing_file_names = [os.path.basename(f) for f in missing_files]
                            failed_files.append(
                                (input_file, f"Missing files: {', '.join(missing_file_names)}"))
                            self.progress_signal.emit(
                                f"❌ Failed to generate required files: {os.path.basename(input_file)} - Missing: {', '.join(missing_file_names)}")

                    except Exception as e:
                        failed_files.append((input_file, str(e)))
                        self.progress_signal.emit(
                            f"❌ ALPSS processing failed for {os.path.basename(input_file)}: {str(e)}")
                        # Continue with next file instead of stopping
                        continue

                    # Calculate timing for this file
                    file_end_time = time.time()
                    file_time = file_end_time - file_start_time
                    total_alpss_time += file_time

                    self.progress_signal.emit(
                        f"Completed ALPSS analysis for {os.path.basename(input_file)} in {file_time:.2f} seconds")

                # Report ALPSS processing summary
                self.progress_signal.emit(f"=== ALPSS Processing Summary ===")
                self.progress_signal.emit(
                    f"Total files: {len(files_to_process)}")
                self.progress_signal.emit(
                    f"Successfully processed: {len(self.successful_files)}")
                self.progress_signal.emit(f"Failed: {len(failed_files)}")

                if failed_files:
                    self.progress_signal.emit(f"=== Failed Files ===")
                    for failed_file, error_msg in failed_files:
                        self.progress_signal.emit(
                            f"❌ {os.path.basename(failed_file)}: {error_msg}")

                # Report total ALPSS timing
                avg_time = total_alpss_time / \
                    len(self.successful_files) if self.successful_files else 0
                self.progress_signal.emit(
                    f"ALPSS Analysis Summary: {len(self.successful_files)} files processed in {total_alpss_time:.2f} seconds (avg: {avg_time:.2f}s per file)")
                
                # Save failed files list as CSV
                if failed_files:
                    try:
                        failed_files_data = []
                        for failed_file, error_msg in failed_files:
                            failed_files_data.append({
                                'input_file': os.path.basename(failed_file),
                                'input_file_path': failed_file,
                                'error_message': error_msg,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                        
                        failed_files_df = pd.DataFrame(failed_files_data)
                        failed_files_path = os.path.join(self.output_dir, 'failed_data_files.csv')
                        failed_files_df.to_csv(failed_files_path, index=False)
                        self.progress_signal.emit(f"Saved failed files list to: {failed_files_path}")
                    except Exception as e:
                        self.progress_signal.emit(f"Warning: Could not save failed files list: {str(e)}")

            # Run SPADE analysis if not ALPSS-only mode
            if self.analysis_mode != "alpss_only":
                spade_start_time = time.time()
                if self.spade_auto_mode:
                    # Automatic mode: use ALPSS output
                    if self.successful_files:  # Use successful_files instead of self.input_files
                        self.progress_signal.emit(
                            "Running SPADE analysis on ALPSS outputs...")

                        # Find all velocity files generated by ALPSS
                        vel_files = []
                        missing_spade_files = []
                        for input_file in self.successful_files:  # Only check successful files
                            base_name = os.path.splitext(
                                os.path.basename(input_file))[0]
                            # Use the velocity with uncertainty file (contains
                            # smoothed velocity + uncertainty)
                            vel_file = os.path.join(
    self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                            if os.path.exists(vel_file):
                                vel_files.append(vel_file)
                            else:
                                missing_spade_files.append(base_name)

                        # Report SPADE file availability
                        self.progress_signal.emit(
                            f"=== SPADE File Availability ===")
                        self.progress_signal.emit(
                            f"ALPSS successful files: {len(self.successful_files)}")
                        self.progress_signal.emit(
                            f"SPADE input files found: {len(vel_files)}")
                        self.progress_signal.emit(
                            f"Missing SPADE input files: {len(missing_spade_files)}")

                        if missing_spade_files:
                            self.progress_signal.emit(
                                f"=== Missing SPADE Input Files ===")
                            for missing_file in missing_spade_files:
                                self.progress_signal.emit(
                                    f"❌ Missing: {missing_file}--vel-smooth-with-uncert.csv")

                        # Check if spall analysis is enabled
                        spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False)
                        
                        if vel_files and spall_analysis_enabled:
                            # Create SPADE output subdirectory
                            spade_output_dir = os.path.join(
                                self.output_dir, "SPADE_analysis")
                            os.makedirs(spade_output_dir, exist_ok=True)

                            # Debug: Check paths
                            self.progress_signal.emit(
                                f"Debug: output_dir = {self.output_dir}")
                            self.progress_signal.emit(
                                f"Debug: spade_output_dir = {spade_output_dir}")
                            self.progress_signal.emit(
                                f"Debug: Found {len(vel_files)} velocity files")

                            # Validate paths
                            if not os.path.exists(self.output_dir):
                                raise ValueError(
                                    f"Output directory does not exist: {self.output_dir}")
                            if not os.path.exists(spade_output_dir):
                                raise ValueError(
                                    f"SPADE output directory could not be created: {spade_output_dir}")

                            # Run SPADE with progress updates
                            self.progress_signal.emit(
                                f"SPADE Processing file 1/{len(vel_files)}: Starting SPADE analysis...")

                            # Add skip_smoothing parameter to avoid double
                            # smoothing
                            spade_params_with_skip = self.spade_params.copy()
                            # Skip SPADE smoothing since ALPSS already smoothed
                            spade_params_with_skip['skip_smoothing'] = True

                            # Remove smooth_window and polyorder when skipping
                            # smoothing to avoid confusion
                            if spade_params_with_skip.get(
                                'skip_smoothing', False):
                                spade_params_with_skip.pop(
                                    'smooth_window', None)
                                spade_params_with_skip.pop('polyorder', None)

                            # Add parameter data for enhanced legends if
                            # available
                            if self.param_data:
                                spade_params_with_skip['param_data'] = self.param_data
                                self.progress_signal.emit(
                                    "Using parameter data for enhanced legends")
                            else:
                                spade_params_with_skip['param_data'] = None

                            # Create subfolder for individual spall plots if plot_individual is enabled
                            plot_individual_enabled = self.spade_params.get('plot_individual', False)
                            self.progress_signal.emit(f"Debug: plot_individual_enabled = {plot_individual_enabled}, spall_analysis_enabled = {spall_analysis_enabled}")
                            if plot_individual_enabled and spall_analysis_enabled:
                                spall_plots_dir = os.path.join(spade_output_dir, 'spall_plots')
                                os.makedirs(spall_plots_dir, exist_ok=True)
                                self.progress_signal.emit(f"Individual spall plots will be saved to: {spall_plots_dir}")
                            else:
                                spall_plots_dir = spade_output_dir
                                if not plot_individual_enabled:
                                    self.progress_signal.emit(f"⚠ Individual spall plots disabled (plot_individual = {plot_individual_enabled})")
                                if not spall_analysis_enabled:
                                    self.progress_signal.emit(f"⚠ Individual spall plots disabled (spall_analysis_enabled = {spall_analysis_enabled})")

                            try:
                                # Process files with DNS detection
                                results_list = []
                                
                                # Get spall detection parameters
                                threshold_velocity = self.spade_params.get('threshold_velocity_ms', 30.0)
                                spall_start_time = self.spade_params.get('spall_start_time_ns', 10.0)
                                spall_end_time = self.spade_params.get('spall_end_time_ns', 100.0)
                                analysis_model = self.spade_params.get('analysis_model', 'max_min')
                                spall_msg = f"  [SPALL] Using analysis_model='{analysis_model}', window=[{spall_start_time:.1f}, {spall_end_time:.1f}] ns, threshold={threshold_velocity:.1f} m/s"
                                self.progress_signal.emit(spall_msg)
                                print(spall_msg)  # Also print to terminal
                                
                                # Get material properties for each file
                                default_density = self.spade_params.get('density', 8960)
                                default_acoustic_velocity = self.spade_params.get('acoustic_velocity', 3950)
                                
                                for i, vel_file in enumerate(vel_files):
                                    self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(vel_files)}: {os.path.basename(vel_file)}")
                                    
                                    # Get base name for material lookup
                                    base_name = os.path.splitext(os.path.basename(vel_file))[0]
                                    # Remove suffix if present
                                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                                        if base_name.endswith(suffix):
                                            base_name = base_name[:-len(suffix)]
                                            break
                                    
                                    # Get material properties
                                    sample_material = 'Unknown'
                                    matched_param = self.get_param_data_for_file(base_name)
                                    if matched_param:
                                        sample_material = matched_param.get('Sample material', 'Unknown')
                                    
                                    mat_props = self.get_material_properties_from_config(sample_material, default_density, default_acoustic_velocity)
                                    density = matched_param.get('Density_kg_m3', mat_props['density']) if matched_param else mat_props['density']
                                    acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed']) if matched_param else mat_props['bulk_wave_speed']
                                    
                                    # Process with DNS detection
                                    # Generate plot path if individual plots are enabled
                                    individual_plot_path = None
                                    if plot_individual_enabled and spall_analysis_enabled:
                                        individual_plot_path = os.path.join(spall_plots_dir, f"{base_name}_spall_analysis.png")
                                        self.progress_signal.emit(f"  Will save individual spall plot to: {os.path.basename(individual_plot_path)}")
                                    
                                    result = self.detect_dns_and_process_spall(
                                        file_path=vel_file,
                                        base_name=base_name,
                                        density=density,
                                        acoustic_velocity=acoustic_velocity,
                                        threshold_velocity=threshold_velocity,
                                        spall_start_time=spall_start_time,
                                        spall_end_time=spall_end_time,
                                        analysis_model=analysis_model,
                                        plot_path=individual_plot_path,
                                        **{k: v for k, v in spade_params_with_skip.items() if k not in ['plot_individual', 'density', 'acoustic_velocity', 'analysis_model']}
                                    )
                                    
                                    # Add material info
                                    result['Material'] = sample_material
                                    result['Density_kg_m3'] = density
                                    result['Acoustic_Velocity_m_s'] = acoustic_velocity
                                    
                                    results_list.append(result)
                                
                                # Save summary
                                if results_list:
                                    summary_df = pd.DataFrame(results_list)
                                    summary_path = os.path.join(spade_output_dir, "spall_summary.csv")
                                    summary_df.to_csv(summary_path, index=False)
                                    self.progress_signal.emit(f"Saved spall summary with {len(results_list)} entries to: {summary_path}")
                                    
                                    # Print detailed summary statistics
                                    valid_spall_mask = summary_df['Spall_Strength_GPa'].apply(
                                        lambda x: isinstance(x, (int, float)) and pd.notna(x) and not pd.isna(x)
                                    )
                                    valid_spall = valid_spall_mask.sum()
                                    dns_count = (summary_df['Spall_Strength_GPa'] == "DNS").sum()
                                    failed_count = len(results_list) - valid_spall - dns_count
                                    
                                    self.progress_signal.emit(f"")
                                    self.progress_signal.emit(f"=== Spall Analysis Summary ===")
                                    self.progress_signal.emit(f"Total shots processed: {len(results_list)}")
                                    self.progress_signal.emit(f"Valid spall (numeric strength): {valid_spall} ({100*valid_spall/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"DNS (Did Not Spall): {dns_count} ({100*dns_count/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"Failed/Error: {failed_count} ({100*failed_count/len(results_list):.1f}%)")
                                    
                                    # Statistics for valid spall cases
                                    if valid_spall > 0:
                                        valid_df = summary_df[valid_spall_mask]
                                        strength_values = pd.to_numeric(valid_df['Spall_Strength_GPa'], errors='coerce')
                                        strength_values = strength_values[strength_values.notna()]
                                        
                                        if len(strength_values) > 0:
                                            self.progress_signal.emit(f"")
                                            self.progress_signal.emit(f"Valid Spall Statistics:")
                                            self.progress_signal.emit(f"  Mean strength: {strength_values.mean():.3f} GPa")
                                            self.progress_signal.emit(f"  Std strength: {strength_values.std():.3f} GPa")
                                            self.progress_signal.emit(f"  Min strength: {strength_values.min():.3f} GPa")
                                            self.progress_signal.emit(f"  Max strength: {strength_values.max():.3f} GPa")
                                            
                                            # Strain rate statistics if available
                                            strain_rates = pd.to_numeric(valid_df['Spall_StrainRate_s^-1'], errors='coerce')
                                            strain_rates = strain_rates[strain_rates.notna()]
                                            if len(strain_rates) > 0:
                                                self.progress_signal.emit(f"  Mean strain rate: {strain_rates.mean():.2e} s^-1")
                                                self.progress_signal.emit(f"  Std strain rate: {strain_rates.std():.2e} s^-1")
                            except Exception as e:
                                import traceback
                                self.progress_signal.emit(f"Error during SPADE spall analysis: {str(e)}")
                                self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

                            # Update progress after completion
                            for i in range(len(vel_files)):
                                self.progress_signal.emit(
                                    f"SPADE Processing file {i+1}/{len(vel_files)}: Completed")

                            spade_end_time = time.time()
                            spade_time = spade_end_time - spade_start_time
                            self.progress_signal.emit(
                                f"Completed SPADE analysis for {len(vel_files)} files in {spade_time:.2f} seconds")
                        elif vel_files and not spall_analysis_enabled:
                            self.progress_signal.emit(
                                f"Found {len(vel_files)} velocity files but spall analysis is disabled - skipping SPADE analysis")
                        else:
                            self.progress_signal.emit(
                                "Warning: No velocity files found for SPADE analysis")
                    else:
                        self.progress_signal.emit(
                            "No ALPSS files to process for automatic SPADE mode")
                else:
                    # Manual mode: use provided SPADE input files
                    if self.spade_input_files:
                        self.progress_signal.emit(
                            f"Running SPADE analysis on {len(self.spade_input_files)} manual input files...")

                        # Create SPADE output subdirectory
                        spade_output_dir = os.path.join(
                            self.output_dir, "SPADE_analysis")
                        os.makedirs(spade_output_dir, exist_ok=True)

                        # Run SPADE - for manual mode, we need to create a temporary directory with the files
                        # or use a different approach since SPADE expects
                        # input_folder and file_pattern
                        if len(self.spade_input_files) == 1:
                            # Single file - use its directory as input_folder
                            input_dir = os.path.dirname(
                                self.spade_input_files[0])
                            file_pattern = os.path.basename(
                                self.spade_input_files[0])
                        else:
                            # Multiple files - use the first file's directory
                            # and a pattern that matches all
                            input_dir = os.path.dirname(
                                self.spade_input_files[0])
                            file_pattern = "*--vel-smooth-with-uncert.csv"

                        # Start SPADE processing
                        spade_start_time = time.time()
                        self.progress_signal.emit(
                            f"SPADE Processing file 1/{len(self.spade_input_files)}: Starting SPADE analysis...")

                        # Add skip_smoothing parameter to avoid double
                        # smoothing
                        spade_params_with_skip = self.spade_params.copy()
                        # Skip SPADE smoothing since ALPSS already smoothed
                        spade_params_with_skip['skip_smoothing'] = True

                        # Remove smooth_window and polyorder when skipping
                        # smoothing to avoid confusion
                        if spade_params_with_skip.get('skip_smoothing', False):
                            spade_params_with_skip.pop('smooth_window', None)
                            spade_params_with_skip.pop('polyorder', None)

                        # Add parameter data for enhanced legends if available
                        if self.param_data:
                            spade_params_with_skip['param_data'] = self.param_data
                            self.progress_signal.emit(
                                "Using parameter data for enhanced legends")
                        else:
                            spade_params_with_skip['param_data'] = None

                        # Create subfolder for individual spall plots if plot_individual is enabled
                        plot_individual_enabled = self.spade_params.get('plot_individual', False)
                        spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False)
                        self.progress_signal.emit(f"Debug: plot_individual_enabled = {plot_individual_enabled}, spall_analysis_enabled = {spall_analysis_enabled}")
                        
                        # Only run spall analysis if enabled
                        if spall_analysis_enabled:
                            if plot_individual_enabled:
                                spall_plots_dir = os.path.join(spade_output_dir, 'spall_plots')
                                os.makedirs(spall_plots_dir, exist_ok=True)
                                self.progress_signal.emit(f"Individual spall plots will be saved to: {spall_plots_dir}")
                            else:
                                spall_plots_dir = spade_output_dir
                                self.progress_signal.emit(f"⚠ Individual spall plots disabled (plot_individual = {plot_individual_enabled})")

                            try:
                                # Process files with DNS detection (same as automatic mode)
                                results_list = []
                                
                                # Get spall detection parameters
                                threshold_velocity = self.spade_params.get('threshold_velocity_ms', 30.0)
                                spall_start_time = self.spade_params.get('spall_start_time_ns', 10.0)
                                spall_end_time = self.spade_params.get('spall_end_time_ns', 100.0)
                                analysis_model = self.spade_params.get('analysis_model', 'max_min')
                                spall_msg = f"  [SPALL] Using analysis_model='{analysis_model}', window=[{spall_start_time:.1f}, {spall_end_time:.1f}] ns, threshold={threshold_velocity:.1f} m/s"
                                self.progress_signal.emit(spall_msg)
                                print(spall_msg)  # Also print to terminal
                                
                                # Get material properties for each file
                                default_density = self.spade_params.get('density', 8960)
                                default_acoustic_velocity = self.spade_params.get('acoustic_velocity', 3950)
                                
                                for i, vel_file in enumerate(self.spade_input_files):
                                    self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(self.spade_input_files)}: {os.path.basename(vel_file)}")
                                    
                                    # Get base name for material lookup
                                    base_name = os.path.splitext(os.path.basename(vel_file))[0]
                                    # Remove suffix if present
                                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                                        if base_name.endswith(suffix):
                                            base_name = base_name[:-len(suffix)]
                                            break
                                    
                                    # Get material properties
                                    sample_material = 'Unknown'
                                    matched_param = self.get_param_data_for_file(base_name)
                                    if matched_param:
                                        sample_material = matched_param.get('Sample material', 'Unknown')
                                    
                                    mat_props = self.get_material_properties_from_config(sample_material, default_density, default_acoustic_velocity)
                                    density = matched_param.get('Density_kg_m3', mat_props['density']) if matched_param else mat_props['density']
                                    acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed']) if matched_param else mat_props['bulk_wave_speed']
                                    
                                    # Process with DNS detection
                                    # Generate plot path if individual plots are enabled
                                    individual_plot_path = None
                                    if plot_individual_enabled and spall_analysis_enabled:
                                        individual_plot_path = os.path.join(spall_plots_dir, f"{base_name}_spall_analysis.png")
                                        self.progress_signal.emit(f"  Will save individual spall plot to: {os.path.basename(individual_plot_path)}")
                                    
                                    result = self.detect_dns_and_process_spall(
                                        file_path=vel_file,
                                        base_name=base_name,
                                        density=density,
                                        acoustic_velocity=acoustic_velocity,
                                        threshold_velocity=threshold_velocity,
                                        spall_start_time=spall_start_time,
                                        spall_end_time=spall_end_time,
                                        analysis_model=analysis_model,
                                        plot_path=individual_plot_path,
                                        **{k: v for k, v in spade_params_with_skip.items() if k not in ['plot_individual', 'density', 'acoustic_velocity', 'analysis_model']}
                                    )
                                    
                                    # Add material info
                                    result['Material'] = sample_material
                                    result['Density_kg_m3'] = density
                                    result['Acoustic_Velocity_m_s'] = acoustic_velocity
                                    
                                    results_list.append(result)
                                
                                # Save summary
                                if results_list:
                                    summary_df = pd.DataFrame(results_list)
                                    summary_path = os.path.join(spade_output_dir, "spall_summary.csv")
                                    summary_df.to_csv(summary_path, index=False)
                                    self.progress_signal.emit(f"Saved spall summary with {len(results_list)} entries to: {summary_path}")
                                    
                                    # Print detailed summary statistics
                                    valid_spall_mask = summary_df['Spall_Strength_GPa'].apply(
                                        lambda x: isinstance(x, (int, float)) and pd.notna(x) and not pd.isna(x)
                                    )
                                    valid_spall = valid_spall_mask.sum()
                                    dns_count = (summary_df['Spall_Strength_GPa'] == "DNS").sum()
                                    failed_count = len(results_list) - valid_spall - dns_count
                                    
                                    self.progress_signal.emit(f"")
                                    self.progress_signal.emit(f"=== Spall Analysis Summary ===")
                                    self.progress_signal.emit(f"Total shots processed: {len(results_list)}")
                                    self.progress_signal.emit(f"Valid spall (numeric strength): {valid_spall} ({100*valid_spall/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"DNS (Did Not Spall): {dns_count} ({100*dns_count/len(results_list):.1f}%)")
                                    self.progress_signal.emit(f"Failed/Error: {failed_count} ({100*failed_count/len(results_list):.1f}%)")
                                    
                                    # Statistics for valid spall cases
                                    if valid_spall > 0:
                                        valid_df = summary_df[valid_spall_mask]
                                        strength_values = pd.to_numeric(valid_df['Spall_Strength_GPa'], errors='coerce')
                                        strength_values = strength_values[strength_values.notna()]
                                        
                                        if len(strength_values) > 0:
                                            self.progress_signal.emit(f"")
                                            self.progress_signal.emit(f"Valid Spall Statistics:")
                                            self.progress_signal.emit(f"  Mean strength: {strength_values.mean():.3f} GPa")
                                            self.progress_signal.emit(f"  Std strength: {strength_values.std():.3f} GPa")
                                            self.progress_signal.emit(f"  Min strength: {strength_values.min():.3f} GPa")
                                            self.progress_signal.emit(f"  Max strength: {strength_values.max():.3f} GPa")
                                            
                                            # Strain rate statistics if available
                                            strain_rates = pd.to_numeric(valid_df['Spall_StrainRate_s^-1'], errors='coerce')
                                            strain_rates = strain_rates[strain_rates.notna()]
                                            if len(strain_rates) > 0:
                                                self.progress_signal.emit(f"  Mean strain rate: {strain_rates.mean():.2e} s^-1")
                                                self.progress_signal.emit(f"  Std strain rate: {strain_rates.std():.2e} s^-1")
                            except Exception as e:
                                import traceback
                                self.progress_signal.emit(f"Error during SPADE spall analysis (manual mode): {str(e)}")
                                self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
                                
                                self.progress_signal.emit(f"")
                        else:
                            # Spall analysis is disabled
                            self.progress_signal.emit(f"⚠ Spall analysis is disabled (experiment_spall_analysis = false) - skipping spall analysis")

                        # Update progress after completion
                        for i in range(len(self.spade_input_files)):
                            self.progress_signal.emit(
                                f"SPADE Processing file {i+1}/{len(self.spade_input_files)}: Completed")

                        spade_end_time = time.time()
                        spade_time = spade_end_time - spade_start_time
                        self.progress_signal.emit(
                            f"Completed SPADE analysis for {len(self.spade_input_files)} files in {spade_time:.2f} seconds")
                    else:
                        self.progress_signal.emit(
                            "No SPADE input files provided")
            # After SPADE analysis, generate mean velocity file and combined
            # plots
            spade_output_dir = os.path.join(self.output_dir, "SPADE_analysis")
            os.makedirs(spade_output_dir, exist_ok=True)

            # Handle different experiment types (can be both)
            velocity_shots_enabled = self.spade_params.get('velocity_shots_enabled', True)
            spall_analysis_enabled = self.spade_params.get('spall_analysis_enabled', False)

            if velocity_shots_enabled:
                self.progress_signal.emit("Running velocity shots analysis...")
                self.generate_velocity_shots_summary(spade_output_dir)
            
            if spall_analysis_enabled:
                self.progress_signal.emit("Running spall analysis...")
                self.generate_spall_analysis_summary(spade_output_dir)
            
            if not velocity_shots_enabled and not spall_analysis_enabled:
                self.progress_signal.emit("No experiment types selected, defaulting to velocity shots")
                self.generate_velocity_shots_summary(spade_output_dir)

            # Calculate total processing time
            total_end_time = time.time()
            total_time = total_end_time - \
                self.start_time if hasattr(self, 'start_time') else 0

            self.progress_signal.emit("All analysis completed successfully!")
            self.progress_signal.emit(
                f"Total processing time: {total_time:.2f} seconds")
            
            # Print summary of trace analysis
            self.progress_signal.emit("")
            self.progress_signal.emit("=" * 70)
            self.progress_signal.emit("ANALYSIS SUMMARY")
            self.progress_signal.emit("=" * 70)
            self.progress_signal.emit(f"Total input traces: {self.total_input_traces}")
            self.progress_signal.emit(f"Traces plotted: {self.traces_plotted}")
            self.progress_signal.emit(f"Traces rejected: {self.traces_rejected}")
            if self.rejection_reasons:
                self.progress_signal.emit("")
                self.progress_signal.emit("Rejection reasons:")
                for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                    self.progress_signal.emit(f"  - {reason}: {count}")
            self.progress_signal.emit(f"Total analysis run time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            self.progress_signal.emit("=" * 70)
            
            self.finished_signal.emit(True, "Analysis completed successfully")
        except Exception as e:
            import traceback
            error_msg = str(e)
            # Avoid duplicate "Analysis failed:" prefix
            if error_msg.startswith("Analysis failed:"):
                error_msg = error_msg.replace("Analysis failed:", "").strip()
            self.progress_signal.emit(f"Error during analysis: {error_msg}")
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")
            self.finished_signal.emit(False, error_msg)

    def generate_velocity_shots_summary(self, spade_output_dir):
        """Generate velocity shots summary CSV with impact velocity calculations and combined velocity plot"""
        self.progress_signal.emit("Generating velocity shots summary...")
        
        # In SPADE-only mode, use the provided spade_input_files
        # In combined/automatic mode, use only files that were just processed (from successful_files)
        if self.analysis_mode == "spade_only" and self.spade_input_files:
            velocity_files = [f for f in self.spade_input_files if os.path.exists(f)]
            self.progress_signal.emit(f"SPADE-only mode: Using {len(velocity_files)} provided input files")
        else:
            # Use only files that were just processed by ALPSS (not all files in output_dir)
            velocity_files = []
            if hasattr(self, 'successful_files') and self.successful_files:
                for input_file in self.successful_files:
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    vel_file = os.path.join(self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                    if os.path.exists(vel_file) and os.path.getsize(vel_file) > 0:
                        velocity_files.append(vel_file)
                self.progress_signal.emit(f"Using {len(velocity_files)} files from current ALPSS processing run")
            else:
                # Fallback: if successful_files not available, use glob but warn
                self.progress_signal.emit("Warning: Using all files in output directory (successful_files not available)")
                velocity_files = glob.glob(os.path.join(self.output_dir, '*--vel-smooth-with-uncert.csv'))
        
        # Filter out empty files
        valid_velocity_files = []
        for file_path in velocity_files:
            if os.path.getsize(file_path) > 0:
                valid_velocity_files.append(file_path)
            else:
                self.progress_signal.emit(f"Warning: Empty velocity file found: {os.path.basename(file_path)}")

        if not valid_velocity_files:
            self.progress_signal.emit(
                "No valid velocity files with uncertainty data found for velocity shots summary")
            return

        self.progress_signal.emit(f"Found {len(valid_velocity_files)} valid velocity files to process")
        velocity_files = valid_velocity_files

        # Debug: Report available parameter data
        if self.param_data:
            self.progress_signal.emit(f"Available parameter data keys: {list(self.param_data.keys())}")
            # Show sample of parameter data structure
            if self.param_data:
                sample_key = list(self.param_data.keys())[0]
                sample_data = self.param_data[sample_key]
                self.progress_signal.emit(f"Sample parameter data structure for '{sample_key}': {list(sample_data.keys())}")
        else:
            self.progress_signal.emit("No parameter data available")

        velocity_shots_data = []
        velocity_plot_data = []  # For combined velocity plot
        unaligned_entries = []

        for file_path in velocity_files:
            try:
                # Read velocity data with uncertainty
                df = pd.read_csv(file_path)
                if df.shape[1] < 3:  # Should have time, velocity, uncertainty
                    continue

                time_data = df.iloc[:, 0].values
                velocity_data = df.iloc[:, 1].values
                uncertainty_data = df.iloc[:, 2].values

                # Convert time to ns if needed
                if np.nanmax(time_data) < 1.0:
                    time_data = time_data * 1e9

                # Load noise fraction data and filter velocity data
                noise_fraction = None
                velocity_filtered = velocity_data.copy()

                # Try to load noise fraction data
                noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                if os.path.exists(noise_file):
                    try:
                        df_noise = pd.read_csv(noise_file)
                        if df_noise.shape[1] >= 1:
                            # Use last column
                            noise_fraction = df_noise.iloc[:, -1].values
                            if len(noise_fraction) == len(velocity_data):
                                # Filter out data points where noise fraction > 1
                                high_noise_mask = noise_fraction > 1.0
                                velocity_filtered[high_noise_mask] = np.nan
                                self.progress_signal.emit(
                                    f"Filtered {np.sum(high_noise_mask)} high-noise points from {os.path.basename(file_path)}")
                            else:
                                self.progress_signal.emit(
                                    f"Warning: Noise fraction length mismatch for {os.path.basename(file_path)}")
                        else:
                            self.progress_signal.emit(
                                f"Warning: Noise fraction file has insufficient columns: {os.path.basename(noise_file)}")
                    except Exception as e:
                        self.progress_signal.emit(
                            f"Warning: Could not read noise fraction for {os.path.basename(file_path)}: {e}")
                else:
                    self.progress_signal.emit(
                        f"Info: No noise fraction file found for {os.path.basename(file_path)}, using unfiltered data")

                # TRACE ALIGNMENT: Find t=0 when velocity reaches user-defined threshold
                velocity_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)  # m/s
                t0_idx = None
                t0 = None
                
                # Find first point where velocity exceeds threshold
                for i, vel in enumerate(velocity_filtered):
                    if not np.isnan(vel) and vel >= velocity_threshold:
                        t0_idx = i
                        break
                
                if t0_idx is None:
                    self.progress_signal.emit(
                        f"Warning: Could not find velocity threshold {velocity_threshold} m/s for {os.path.basename(file_path)}")
                    # Use original time data
                    time_aligned = time_data
                    aligned_ok = False
                    alignment_reason = f"velocity never reached threshold {velocity_threshold} m/s"
                else:
                    # Align time data to t=0 at velocity threshold
                    t0 = time_data[t0_idx]
                    time_aligned = time_data - t0
                    self.progress_signal.emit(
                        f"Aligned trace: t=0 at {t0:.2f} ns when velocity reached {velocity_threshold} m/s")
                    aligned_ok = True
                    alignment_reason = ""

                # Calculate mean velocity using aligned time and filtered data
                # First, determine the actual time range of the data
                time_range = np.max(time_aligned) - np.min(time_aligned)
                self.progress_signal.emit(f"Time range after alignment: {np.min(time_aligned):.1f} to {np.max(time_aligned):.1f} ns (span: {time_range:.1f} ns)")
                
                # Use adaptive time windows based on actual data range
                if time_range > 1000:  # Long time range (>1μs)
                    # Use windows relative to the middle of the data
                    mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
                    window_start = mid_time - 50  # 50ns window around middle
                    window_end = mid_time + 50
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
                elif time_range > 100:  # Medium time range (100ns-1μs)
                    # Use the middle 100ns of the data
                    mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
                    window_start = mid_time - 50
                    window_end = mid_time + 50
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
                else:  # Short time range (<100ns)
                    # Use the entire available range
                    window_start = np.min(time_aligned)
                    window_end = np.max(time_aligned)
                    time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (full range)"
                
                # Calculate mean velocity in the adaptive window
                mask_window = (time_aligned >= window_start) & (time_aligned <= window_end)
                velocities_window = velocity_filtered[mask_window]
                velocities_window = velocities_window[~np.isnan(velocities_window)]
                
                if len(velocities_window) > 0:
                    mean_velocity_300_400 = np.mean(velocities_window)
                    self.progress_signal.emit(f"Using {len(velocities_window)} data points in window {time_window_used}")
                else:
                    # Fallback: use all available data
                    velocities_all = velocity_filtered[~np.isnan(velocity_filtered)]
                    if len(velocities_all) > 0:
                        mean_velocity_300_400 = np.mean(velocities_all)
                        time_window_used = f"All data ({len(velocities_all)} points)"
                        self.progress_signal.emit(f"Warning: No data in adaptive window, using all available data")
                    else:
                        mean_velocity_300_400 = np.nan
                        time_window_used = "No data available"
                        self.progress_signal.emit(f"Error: No velocity data available for {os.path.basename(file_path)}")

                # Get file base name
                base_name = os.path.splitext(os.path.basename(file_path))[0].replace(
                    '--vel-smooth-with-uncert', ''
                )

                # Track maximum observed velocity for diagnostics
                if np.any(~np.isnan(velocity_filtered)):
                    max_velocity_observed = float(np.nanmax(velocity_filtered))
                else:
                    max_velocity_observed = np.nan
                # Get parameter data if available using helper function
                param_info = self.get_param_data_for_file(base_name)
                if not param_info:
                    self.progress_signal.emit(f"No parameter match found for {base_name}")
                
                # Debug parameter data
                if param_info:
                    self.progress_signal.emit(
                        f"Found parameter data for {base_name}: {list(param_info.keys())}")
                else:
                    self.progress_signal.emit(
                        f"No parameter data found for {base_name}")

                # Create data row for velocity shots summary
                # HEL DETECTION
                hel_strength = np.nan
                hel_uncertainty = np.nan
                free_surface_velocity = np.nan
                hel_ok = False
                hel_time_detection = np.nan  # Time at HEL detection point (ns)
                C_L = np.nan  # Longitudinal wave velocity (will be set from material properties)
                hel_consecutive_points = 0  # Number of consecutive points in HEL segment
                hel_segment_time_ns = np.nan  # Time duration of HEL segment (ns)
                
                hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                        self.spade_params.get('experiment_hel_detection', False))
                if hel_detection_enabled:
                    try:
                        from scipy.ndimage import uniform_filter1d
                        
                        hel_start = self.spade_params.get('hel_start_time_ns', 0.0)
                        hel_end = self.spade_params.get('hel_end_time_ns', None)
                        angle_thresh_deg = self.spade_params.get('hel_angle_threshold_deg', 45.0)
                        min_hel_velocity = self.spade_params.get('minimum_HEL_velocity_expected', 10.0)
                        hel_detection_min_points = self.spade_params.get('hel_detection_min_points', 3)
                        hel_msg = f"  [HEL] Using time window=[{hel_start:.1f}, {hel_end if hel_end is not None else 'None'}] ns, min_velocity={min_hel_velocity:.1f} m/s, min_points={hel_detection_min_points}"
                        self.progress_signal.emit(hel_msg)
                        print(hel_msg)  # Also print to terminal
                        
                        # Step 1: Load data and filter by relative uncertainty
                        valid_mask = ~np.isnan(velocity_filtered)
                        if np.sum(valid_mask) > 5:
                            hel_time_all = time_aligned[valid_mask]
                            hel_velocity_all = velocity_filtered[valid_mask]
                            hel_unc_all = uncertainty_data[valid_mask]
                            
                            # Filter by relative uncertainty: |uncertainty| / max(|velocity|, 1e-9) >= 1 -> NaN
                            max_vel = np.max(np.abs(hel_velocity_all))
                            rel_unc = np.abs(hel_unc_all) / max(max_vel, 1e-9)
                            noise_mask = rel_unc < 1.0
                            if np.sum(noise_mask) < 10:
                                noise_mask = valid_mask  # Fallback if too many points filtered
                            
                            hel_time_clean = hel_time_all[noise_mask]
                            hel_velocity_clean = hel_velocity_all[noise_mask]
                            hel_unc_clean = hel_unc_all[noise_mask]
                                
                            # Step 2: Build mask for HEL window
                            search_mask = hel_time_clean >= hel_start
                            if hel_end is not None and hel_end > hel_start:
                                search_mask &= (hel_time_clean <= hel_end)
                            if np.sum(search_mask) < 10:
                                search_mask = np.ones_like(hel_time_clean, dtype=bool)
                            
                            hel_time_window = hel_time_clean[search_mask]
                            hel_velocity_window = hel_velocity_clean[search_mask]
                            hel_unc_window = hel_unc_clean[search_mask]
                            
                            if len(hel_time_window) < 10:
                                self.progress_signal.emit(f"HEL: Insufficient data points in window for {base_name}")
                                hel_ok = False
                            else:
                                # Step 3: Compute gradients and convert to angles
                                gradient = np.gradient(hel_velocity_window, hel_time_window)
                                
                                # Smooth gradient with small uniform filter
                                window_size = max(3, min(5, len(gradient) // 3))
                                if window_size % 2 == 0:
                                    window_size += 1  # Make odd
                                gradient_smooth = uniform_filter1d(gradient, size=window_size, mode='nearest')
                                
                                # Convert gradients to angles (degrees)
                                angles_deg = np.degrees(np.arctan(np.abs(gradient_smooth)))
                                
                                # Step 4: Find consecutive low-slope segments (|angle| < angle_thresh_deg, ≥hel_detection_min_points points)
                                low_slope_mask = angles_deg < angle_thresh_deg
                                
                                # Find consecutive segments
                                hel_segment_start = None
                                hel_segment_end = None
                                
                                # Calculate time spacing for reference (min_points consecutive points = (min_points-1) intervals)
                                if len(hel_time_window) > 1:
                                    time_diffs = np.diff(hel_time_window)
                                    mean_dt = np.mean(time_diffs)
                                    min_points_time = (hel_detection_min_points - 1) * mean_dt  # min_points consecutive points = (min_points-1) intervals
                                    self.progress_signal.emit(
                                        f"   HEL time spacing: {mean_dt:.3f} ns/point, "
                                        f"{hel_detection_min_points} consecutive points = {min_points_time:.3f} ns")
                                
                                # Group consecutive True values
                                in_segment = False
                                segment_start = None
                                for i, is_low in enumerate(low_slope_mask):
                                    if is_low and not in_segment:
                                        segment_start = i
                                        in_segment = True
                                    elif not is_low and in_segment:
                                        segment_length = i - segment_start
                                        if segment_length >= hel_detection_min_points:  # At least min_points
                                            if hel_segment_start is None:  # First valid segment
                                                hel_segment_start = segment_start
                                                hel_segment_end = i - 1
                                        in_segment = False
                                        segment_start = None
                                
                                # Handle segment that extends to end of array
                                if in_segment and segment_start is not None:
                                    segment_length = len(low_slope_mask) - segment_start
                                    if segment_length >= hel_detection_min_points:
                                        if hel_segment_start is None:
                                            hel_segment_start = segment_start
                                            hel_segment_end = len(low_slope_mask) - 1
                                # Step 5: Use earliest segment as HEL plateau
                                detection_used_gradient = False
                                sample_material = param_info.get('Sample material', 'Unknown')
                                # Get material properties from config first, then database
                                mat_props = self.get_material_properties_from_config(sample_material)
                                density = param_info.get('Density_kg_m3', mat_props['density'])
                                acoustic_velocity = param_info.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed'])
                                # Get C_L from config (longitudinal wave velocity), fallback to acoustic_velocity if not specified
                                C_L = mat_props.get('C_L', acoustic_velocity)
                                
                                if mat_props['material_found']:
                                    source = mat_props.get('source', 'unknown')
                                    source_msg = f" (from {source})" if source != "unknown" else ""
                                    self.progress_signal.emit(
                                        f"Using {mat_props['material_name']} properties: ρ={density:.0f} kg/m³, c={acoustic_velocity:.0f} m/s{source_msg}")
                                hel_plot_end = (
                                    hel_end if hel_end is not None and hel_end > hel_start else np.max(time_aligned)
                                )

                                if hel_segment_start is not None and hel_segment_end is not None:
                                    # Mean velocity of the HEL plateau segment
                                    hel_segment_indices = np.arange(hel_segment_start, hel_segment_end + 1)
                                    free_surface_velocity = np.mean(hel_velocity_window[hel_segment_indices])
                                    
                                    # Time at HEL detection (start of HEL segment)
                                    hel_time_detection = hel_time_window[hel_segment_start]
                                    
                                    # Calculate consecutive points count and segment time duration
                                    hel_consecutive_points = len(hel_segment_indices)
                                    hel_segment_time_ns = hel_time_window[hel_segment_end] - hel_time_window[hel_segment_start]
                                    
                                    # Uncertainty: interpolate at segment (use closest point to mean)
                                    mean_idx = hel_segment_indices[np.argmin(np.abs(hel_velocity_window[hel_segment_indices] - free_surface_velocity))]
                                    u_unc = abs(hel_unc_window[mean_idx])
                                    
                                    # Step 6: Check minimum HEL velocity constraint
                                    if abs(free_surface_velocity) < min_hel_velocity:
                                        # HEL velocity below threshold - reject this detection
                                        hel_ok = False
                                        hel_strength = np.nan
                                        hel_uncertainty = np.nan
                                        free_surface_velocity = np.nan
                                        hel_time_detection = np.nan
                                        self.progress_signal.emit(
                                            f"HEL rejected for {base_name}: detected velocity {abs(free_surface_velocity):.2f} m/s "
                                            f"is below minimum threshold of {min_hel_velocity:.1f} m/s")
                                    else:
                                        # Step 6: Compute HEL stress
                                        hel_strength = 0.5 * density * acoustic_velocity * abs(free_surface_velocity) / 1e9
                                        hel_uncertainty = 0.5 * density * acoustic_velocity * u_unc / 1e9
                                        hel_ok = True
                                        detection_used_gradient = True
                                        
                                        self.progress_signal.emit(
                                            f"HEL detected via gradient method: {hel_strength:.3f} GPa for {base_name} "
                                            f"(plateau at {hel_time_window[hel_segment_start]:.1f}-{hel_time_window[hel_segment_end]:.1f} ns, "
                                            f"{hel_consecutive_points} consecutive points, {hel_segment_time_ns:.3f} ns duration)")

                                        # Get U_0 and t_0 for strain rate slope calculation
                                        if t0_idx is not None and t0_idx < len(velocity_filtered):
                                            U_0_for_plot = velocity_filtered[t0_idx]
                                            t_0_for_plot = time_aligned[t0_idx] if t0_idx < len(time_aligned) else 0.0
                                        else:
                                            valid_idx = np.where(~np.isnan(velocity_filtered))[0]
                                            if len(valid_idx) > 0:
                                                U_0_for_plot = velocity_filtered[valid_idx[0]]
                                                t_0_for_plot = time_aligned[valid_idx[0]] if valid_idx[0] < len(time_aligned) else 0.0
                                            else:
                                                U_0_for_plot = 0.0
                                                t_0_for_plot = 0.0

                                    if hel_ok and self.spade_params.get('plot_individual', False):
                                        try:
                                            self._plot_individual_hel_detection(
                                                base_name,
                                                time_aligned,
                                                velocity_filtered,
                                                hel_start,
                                                hel_end if hel_end not in [None] else np.max(time_aligned),
                                                hel_time_window,
                                                hel_velocity_window,
                                                hel_strength,
                                                hel_uncertainty,
                                                sample_material,
                                                spade_output_dir,
                                                gradient=gradient_smooth,
                                                angles_deg=angles_deg,
                                                hel_segment_start=hel_segment_start,
                                                hel_segment_end=hel_segment_end,
                                                free_surface_velocity=free_surface_velocity,
                                                angle_thresh_deg=angle_thresh_deg,
                                                U_0=U_0_for_plot,
                                                t_0=t_0_for_plot,
                                                t_hel=hel_time_detection,
                                            )
                                        except Exception as plot_error:
                                            self.progress_signal.emit(
                                                f"Warning: Could not create HEL plot for {base_name}: {str(plot_error)[:50]}")
                                else:
                                    # No valid low-slope segment found - HEL detection failed
                                    if not hel_ok:
                                        self.progress_signal.emit(f"HEL: No gradient plateau detected in {base_name}")
                                        hel_consecutive_points = 0
                                        hel_segment_time_ns = np.nan
                                        
                                        # Calculate and report time spacing for reference
                                        if len(hel_time_window) > 1:
                                            time_diffs = np.diff(hel_time_window)
                                            mean_dt = np.mean(time_diffs)
                                            min_points_time = (hel_detection_min_points - 1) * mean_dt  # min_points consecutive points = (min_points-1) intervals
                                            self.progress_signal.emit(
                                                f"   Time spacing: {mean_dt:.3f} ns/point, "
                                                f"{hel_detection_min_points} consecutive points = {min_points_time:.3f} ns")
                    except Exception as hel_error:
                        import traceback
                        self.progress_signal.emit(f"HEL detection error for {base_name}: {str(hel_error)}")
                        self.progress_signal.emit(traceback.format_exc())
                
                # Calculate elastic shock strain rate if HEL was detected
                hel_strain_rate = np.nan
                hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                        self.spade_params.get('experiment_hel_detection', False))
                if hel_ok and hel_detection_enabled and np.isfinite(hel_time_detection):
                    try:
                        # Get velocity at t=0 (after alignment, t0 should be at 0 or first valid point)
                        if t0_idx is not None and t0_idx < len(velocity_filtered):
                            U_0 = velocity_filtered[t0_idx]
                            t_0_ns = time_aligned[t0_idx] if t0_idx < len(time_aligned) else 0.0
                        else:
                            # Use first valid velocity point
                            valid_idx = np.where(~np.isnan(velocity_filtered))[0]
                            if len(valid_idx) > 0:
                                U_0 = velocity_filtered[valid_idx[0]]
                                t_0_ns = time_aligned[valid_idx[0]] if valid_idx[0] < len(time_aligned) else 0.0
                            else:
                                U_0 = 0.0
                                t_0_ns = 0.0
                        
                        # Convert times from ns to seconds
                        t_hel_s = hel_time_detection * 1e-9
                        t_0_s = t_0_ns * 1e-9
                        
                        # Calculate strain rate using C_L from material properties
                        if np.isfinite(C_L) and np.isfinite(free_surface_velocity) and np.isfinite(U_0) and np.isfinite(t_hel_s) and np.isfinite(t_0_s):
                            hel_strain_rate = self.elastic_shock_strain_rate(
                                C_L=C_L,
                                U_hel=free_surface_velocity,
                                U_0=U_0,
                                t_hel=t_hel_s,
                                t_0=t_0_s
                            )
                            
                            # Check if strain rate is negative - reject HEL if so
                            if np.isfinite(hel_strain_rate) and hel_strain_rate < 0:
                                # Negative strain rate - reject this HEL detection
                                hel_ok = False
                                hel_strength = np.nan
                                hel_uncertainty = np.nan
                                free_surface_velocity = np.nan
                                hel_time_detection = np.nan
                                hel_strain_rate = np.nan
                                self.progress_signal.emit(
                                    f"HEL rejected for {base_name}: negative strain rate ({hel_strain_rate:.2e} s⁻¹)")
                        else:
                            self.progress_signal.emit(f"Warning: Invalid values for HEL strain rate calculation for {base_name} (C_L={C_L}, U_hel={free_surface_velocity}, U_0={U_0}, t_hel={t_hel_s}, t_0={t_0_s})")
                    except Exception as strain_error:
                        self.progress_signal.emit(f"Warning: Could not calculate HEL strain rate for {base_name}: {str(strain_error)}")
                
                if not aligned_ok:
                    unaligned_entries.append({
                        'file_name': base_name,
                        'alignment_reason': alignment_reason,
                        'max_velocity_ms': max_velocity_observed,
                        'velocity_threshold_ms': velocity_threshold
                    })
                
                shot_data = {
                    'file_name': base_name,
                    'mean_velocity_300_400ns_ms': mean_velocity_300_400,
                    'time_window_used': time_window_used,
                    'uncertainty_avg_ms': np.nanmean(uncertainty_data),
                    't0_ns': t0 if t0_idx is not None else np.nan,
                    'velocity_threshold_ms': velocity_threshold,
                    'max_velocity_ms': max_velocity_observed,
                    'aligned_ok': aligned_ok,
                    'alignment_reason': alignment_reason,
                    'hel_strength_gpa': hel_strength,
                    'hel_uncertainty_gpa': hel_uncertainty,
                    'hel_consecutive_points': hel_consecutive_points,
                    'hel_segment_time_ns': hel_segment_time_ns,
                    'free_surface_velocity_ms': free_surface_velocity,
                    'hel_ok': hel_ok,
                    'hel_strain_rate_s^-1': hel_strain_rate,
                }

                # Add ALL parameter file data as extra columns (without 'param_' prefix)
                for key, value in param_info.items():
                    shot_data[key] = value

                velocity_shots_data.append(shot_data)

                # Create data for combined velocity plot (use aligned time and filtered data)
                plot_data = {
                    'time_ns': time_aligned,
                    'velocity_ms': velocity_filtered,  # Use filtered velocity
                    'file_name': base_name,
                    'param_info': param_info
                }
                velocity_plot_data.append(plot_data)

            except Exception as e:
                self.progress_signal.emit(
                    f"Error processing {os.path.basename(file_path)}: {str(e)}")
                continue

        # Save velocity shots summary
        if velocity_shots_data:
            # Ensure all parameter columns from the parameter input file are included in the summary
            all_param_columns = set()
            # 1) Collect from parameter file(s) if available
            if self.param_data:
                try:
                    for _key, param_dict in self.param_data.items():
                        if isinstance(param_dict, dict):
                            all_param_columns.update(param_dict.keys())
                except Exception:
                    pass
            # 2) Also include any parameter keys already merged into rows
            for shot_data in velocity_shots_data:
                for key in shot_data.keys():
                    if key not in ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms',
                                   't0_ns', 'velocity_threshold_ms', 'max_velocity_ms', 'aligned_ok', 'alignment_reason']:
                        all_param_columns.add(key)

            # Add missing parameter columns with NaN values to each row
            for shot_data in velocity_shots_data:
                for param_col in all_param_columns:
                    if param_col not in shot_data:
                        shot_data[param_col] = np.nan

            velocity_shots_df = pd.DataFrame(velocity_shots_data)
            total_shots = len(velocity_shots_df)
            
            # Apply MAD filter to remove outlier peak velocities if enabled
            # Filter is applied per material and laser energy bracket
            mad_filter_enabled = self.spade_params.get('mad_filter_enabled', False)
            mad_filter_threshold = self.spade_params.get('mad_filter_threshold', 3.0)
            
            if mad_filter_enabled and 'max_velocity_ms' in velocity_shots_df.columns:
                # Initialize mad_filter_keep column
                velocity_shots_df['mad_filter_keep'] = True
                
                # Find material column
                material_col = None
                for col_name in velocity_shots_df.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                
                if material_col is None:
                    # Try common alternatives
                    for col_name in ['Material', 'material', 'Sample_Material', 'Sample Material']:
                        if col_name in velocity_shots_df.columns:
                            material_col = col_name
                            break
                
                if material_col is None:
                    self.progress_signal.emit("⚠️  MAD filter: Material column not found, applying to entire dataset")
                    material_col = 'Material'
                    velocity_shots_df[material_col] = 'Unknown'
                
                # Find laser energy column
                laser_energy_col = None
                if 'Laser_Target_Energy (mJ)' in velocity_shots_df.columns:
                    laser_energy_col = 'Laser_Target_Energy (mJ)'
                else:
                    possible_names = [
                        'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                        'Laser_Target_Energy', 'Laser Target Energy',
                        'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                        'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                    ]
                    for col_name in velocity_shots_df.columns:
                        col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                        for possible in possible_names:
                            possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                            if col_name == possible or possible_normalized in col_normalized:
                                laser_energy_col = col_name
                                break
                        if laser_energy_col:
                            break
                
                if laser_energy_col is None:
                    self.progress_signal.emit("⚠️  MAD filter: Laser energy column not found, applying per material only")
                    # Group by material only
                    total_outliers = 0
                    for material in velocity_shots_df[material_col].unique():
                        material_data = velocity_shots_df[velocity_shots_df[material_col] == material]
                        valid_velocities = material_data['max_velocity_ms'].dropna()
                        
                        if len(valid_velocities) >= 2:  # Need at least 2 points for MAD
                            keep_mask = self.mad_filter(valid_velocities.values, threshold=mad_filter_threshold)
                            valid_indices = valid_velocities.index
                            for idx, keep in zip(valid_indices, keep_mask):
                                velocity_shots_df.loc[idx, 'mad_filter_keep'] = keep
                            group_outliers = (~keep_mask).sum()
                            total_outliers += group_outliers
                            if group_outliers > 0:
                                self.progress_signal.emit(f"   {material}: {group_outliers} outlier(s) identified")
                else:
                    # Convert laser energy to numeric
                    velocity_shots_df[laser_energy_col] = pd.to_numeric(velocity_shots_df[laser_energy_col], errors='coerce')
                    
                    # Group by material, then create energy bins (±30 mJ), then apply MAD filter within each bin
                    total_outliers = 0
                    bins_processed = 0
                    energy_bin_width = 30.0  # ±30 mJ bins
                    
                    for material in velocity_shots_df[material_col].unique():
                        material_data = velocity_shots_df[velocity_shots_df[material_col] == material].copy()
                        laser_energies = material_data[laser_energy_col].dropna()
                        
                        if len(laser_energies) == 0:
                            continue
                        
                        # Create energy bins: group laser energies within ±30 mJ of bin mean
                        # Sort energies and create bins using iterative mean-based clustering
                        sorted_energies = sorted(laser_energies.unique())
                        energy_bins = []
                        
                        # Create bins by grouping energies that are within ±30 mJ of the bin's mean
                        # Iterate until all energies are assigned to bins
                        remaining_energies = sorted_energies.copy()
                        
                        while len(remaining_energies) > 0:
                            # Start a new bin with the smallest remaining energy
                            current_bin = [remaining_energies.pop(0)]
                            
                            # Iteratively add energies that are within ±30 mJ of current bin mean
                            # Continue until no more energies can be added
                            changed = True
                            while changed:
                                changed = False
                                bin_mean = np.mean(current_bin)
                                
                                # Check all remaining energies
                                to_remove = []
                                for energy in remaining_energies:
                                    if abs(energy - bin_mean) <= energy_bin_width:
                                        current_bin.append(energy)
                                        to_remove.append(energy)
                                        changed = True
                                
                                # Remove added energies from remaining list
                                for energy in to_remove:
                                    remaining_energies.remove(energy)
                            
                            # Finalize this bin
                            energy_bins.append(current_bin)
                        
                        # Log bin details
                        bin_details = []
                        for bin_idx, bin_energies in enumerate(energy_bins):
                            bin_mean = np.mean(bin_energies)
                            bin_min = min(bin_energies)
                            bin_max = max(bin_energies)
                            bin_details.append(f"Bin {bin_idx+1}: {len(bin_energies)} energy levels, "
                                             f"mean={bin_mean:.1f} mJ, range=[{bin_min:.1f}, {bin_max:.1f}] mJ")
                        
                        self.progress_signal.emit(f"   {material}: Created {len(energy_bins)} energy bin(s) from {len(sorted_energies)} unique energy levels")
                        for detail in bin_details:
                            self.progress_signal.emit(f"      {detail}")
                        
                        # Apply MAD filter within each energy bin
                        for bin_idx, energy_bin in enumerate(energy_bins):
                            # Get all data points within this energy bin (±30 mJ from bin mean)
                            bin_mean = np.mean(energy_bin)
                            bin_data = material_data[
                                (material_data[laser_energy_col] >= bin_mean - energy_bin_width) &
                                (material_data[laser_energy_col] <= bin_mean + energy_bin_width)
                            ]
                            
                            valid_velocities = bin_data['max_velocity_ms'].dropna()
                            
                            if len(valid_velocities) >= 2:  # Need at least 2 points for MAD
                                bins_processed += 1
                                keep_mask = self.mad_filter(valid_velocities.values, threshold=mad_filter_threshold)
                                valid_indices = valid_velocities.index
                                for idx, keep in zip(valid_indices, keep_mask):
                                    velocity_shots_df.loc[idx, 'mad_filter_keep'] = keep
                                bin_outliers = (~keep_mask).sum()
                                total_outliers += bin_outliers
                                if bin_outliers > 0:
                                    self.progress_signal.emit(
                                        f"      Bin {bin_idx+1} ({bin_mean:.1f}±{energy_bin_width} mJ): {bin_outliers} outlier(s), "
                                        f"n={len(valid_velocities)}"
                                    )
                    
                    self.progress_signal.emit(f"MAD filter: Applied to {bins_processed} material+energy_bin groups")
                    self.progress_signal.emit(f"MAD filter: {total_outliers} outlier(s) identified across all bins (threshold={mad_filter_threshold})")
                
                # Mark outliers in alignment status (so they're excluded from plots)
                outliers_count = (~velocity_shots_df['mad_filter_keep']).sum()
                if outliers_count > 0:
                    velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'aligned_ok'] = False
                    velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'alignment_reason'] = \
                        velocity_shots_df.loc[~velocity_shots_df['mad_filter_keep'], 'alignment_reason'].apply(
                            lambda x: f"{x}; MAD outlier" if pd.notna(x) and x != "" else "MAD outlier"
                        )
            else:
                velocity_shots_df['mad_filter_keep'] = True  # Default: keep all if filter disabled
            
            if 'aligned_ok' in velocity_shots_df.columns:
                unaligned_count = int((~velocity_shots_df['aligned_ok']).sum())
            else:
                unaligned_count = 0
            self.progress_signal.emit(f"Alignment summary: {unaligned_count}/{total_shots} traces failed to reach the threshold")
            if unaligned_entries:
                unaligned_df = pd.DataFrame(unaligned_entries)
                unaligned_path = os.path.join(spade_output_dir, 'unaligned_traces.csv')
                unaligned_df.to_csv(unaligned_path, index=False)
                self.progress_signal.emit(f"Saved unaligned trace list ({unaligned_count} entries) to: {unaligned_path}")
            else:
                self.progress_signal.emit("All processed traces reached the alignment threshold")

            # Reorder columns: standard columns first, then all parameter columns (sorted)
            standard_cols = ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms',
                             't0_ns', 'velocity_threshold_ms', 'max_velocity_ms', 'aligned_ok', 'alignment_reason']
            param_cols = sorted([c for c in all_param_columns if c not in standard_cols])
            final_cols = standard_cols + param_cols
            # Include any unexpected columns at the end to avoid dropping
            remaining_cols = [c for c in velocity_shots_df.columns if c not in final_cols]
            velocity_shots_df = velocity_shots_df[final_cols + remaining_cols]
            
            unaligned_basenames = {entry['file_name'] for entry in unaligned_entries}
            
            # Add MAD-filtered traces to unaligned_basenames so they're excluded from all_velocity_traces_plot
            if 'mad_filter_keep' in velocity_shots_df.columns:
                mad_filtered_basenames = set(
                    velocity_shots_df[velocity_shots_df['mad_filter_keep'] == False]['file_name'].values
                )
                unaligned_basenames.update(mad_filtered_basenames)
                if len(mad_filtered_basenames) > 0:
                    self.progress_signal.emit(f"Added {len(mad_filtered_basenames)} MAD-filtered traces to exclusion list for all_velocity_traces_plot")
            
            velocity_shots_path = os.path.join(
    spade_output_dir, 'velocity_shots_summary.csv')
            velocity_shots_df.to_csv(velocity_shots_path, index=False)
            self.progress_signal.emit(
                f"Generated velocity shots summary with {len(velocity_shots_data)} shots")
            self.progress_signal.emit(f"Saved to: {velocity_shots_path}")
            self.progress_signal.emit(f"Parameter columns included: {param_cols}")
            
            # Generate HEL consecutive points summary report
            hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                    self.spade_params.get('experiment_hel_detection', False))
            if hel_detection_enabled and 'hel_consecutive_points' in velocity_shots_df.columns:
                self._generate_hel_consecutive_points_report(velocity_shots_df, spade_output_dir)

            # Generate combined velocity plot - DISABLED (material classification now in all_velocity_traces.png)
            # The all_velocity_traces.png plot already includes material-wise classification
            # if velocity_plot_data:
            #     self.generate_combined_velocity_plot(velocity_plot_data, spade_output_dir)

            # Generate the comprehensive aligned plot using ALPSS outputs and GUI threshold
            # This plot includes material-wise classification (color coding by material)
            try:
                if hasattr(self, 'spade_params'):
                    # Use ALPSS output directory as input; same folder SPADE takes input from
                    input_path = self.output_dir
                    uncertainty_threshold = self.spade_params.get('uncertainty_threshold_ms', 50.0)
                    generate_all = self.spade_params.get('generate_all_velocity_plot', True)
                    if generate_all and os.path.exists(input_path):
                        self.progress_signal.emit(
                            f"Generating 'All Velocity Traces' aligned plot with material classification (threshold={uncertainty_threshold} m/s)")
                        self.generate_all_velocity_traces_plot(
                            input_path,
                            spade_output_dir,
                            uncertainty_threshold,
                            unaligned_basenames=unaligned_basenames
                        )
                        
                        # Generate diagnostic plot for Zn traces (disabled)
                        # self.progress_signal.emit("Generating Zn traces diagnostic plot...")
                        # self.generate_zn_traces_diagnostic_plot(
                        #     input_path,
                        #     spade_output_dir,
                        #     uncertainty_threshold
                        # )
            except Exception as e:
                self.progress_signal.emit(f"Warning: Failed to create comprehensive aligned velocity plot: {e}")
            
            # Create parameter mapping report for debugging
            self.create_parameter_mapping_report(velocity_shots_data, spade_output_dir)
            
            # Generate HEL vs Laser Energy plot if HEL detection was enabled
            hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                    self.spade_params.get('experiment_hel_detection', False))
            if hel_detection_enabled:
                self.generate_hel_vs_laser_energy_plot(spade_output_dir)
                # Generate HEL vs Peak Velocity plot
                self.generate_hel_vs_peak_velocity_plot(spade_output_dir)
                # Generate HEL vs HEL Strain Rate plot
                self.generate_hel_vs_hel_strain_rate_plot(spade_output_dir)
            
            # Generate Shock Stress vs Laser Energy plot
            self.generate_shock_stress_vs_laser_energy_plot(spade_output_dir)
            
            # Generate Shock Stress vs Waveplate Angle plot
            self.generate_shock_stress_vs_waveplate_angle_plot(spade_output_dir)
            
            # Generate Laser Energy vs Waveplate Angle plot
            self.generate_laser_energy_vs_waveplate_angle_plot(spade_output_dir)
            
            # Generate Shock Stress vs Peak Velocity plot
            self.generate_shock_stress_vs_peak_velocity_plot(spade_output_dir)
            
            # Generate positional plots (row/column vs metrics)
            self.generate_row_column_vs_peak_shock_stress_plots(spade_output_dir)
            self.generate_row_column_vs_peak_velocity_heatmap(spade_output_dir)
            self.generate_row_column_pair_vs_peak_velocity_plot(spade_output_dir)
            self.generate_row_column_pair_vs_peak_velocity_by_material_plot(spade_output_dir)
            self.generate_peak_velocity_pattern_analysis_plot(spade_output_dir)
        else:
            self.progress_signal.emit("No velocity shots data generated")

    def run_post_processing(self, post_processing_config):
        """
        Post-processing mode: Generate plots from existing velocity_shots_summary.csv
        without rerunning the full SPADE analysis.
        
        Args:
            post_processing_config: Dictionary with 'enabled', 'spade_output_dir', and 'plots' keys
        
        Returns:
            bool: True if successful, False if failed (e.g., missing files)
        """
        if not post_processing_config.get('enabled', False):
            self.progress_signal.emit("Post-processing is disabled in config")
            return False
        
        spade_output_dir = post_processing_config.get('spade_output_dir', self.output_dir)
        if not os.path.exists(spade_output_dir):
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("❌ POST-PROCESSING FAILED")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"⚠ SPADE output directory not found: {spade_output_dir}")
            self.progress_signal.emit("\nPlease check:")
            self.progress_signal.emit("  1. The 'spade_output_dir' path in post_processing_config")
            self.progress_signal.emit("  2. That the directory exists and contains velocity_shots_summary.csv")
            self.progress_signal.emit("=" * 60)
            return False
        
        velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
        if not os.path.exists(velocity_shots_path):
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("❌ POST-PROCESSING FAILED")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"⚠ velocity_shots_summary.csv not found in:")
            self.progress_signal.emit(f"   {spade_output_dir}")
            self.progress_signal.emit("\nThis file is required for post-processing.")
            self.progress_signal.emit("\nPlease:")
            self.progress_signal.emit("  1. Run SPADE analysis first to generate velocity_shots_summary.csv")
            self.progress_signal.emit("  2. Or check that 'spade_output_dir' points to the correct directory")
            self.progress_signal.emit(f"\nExpected file location: {velocity_shots_path}")
            self.progress_signal.emit("=" * 60)
            return False
        
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit("POST-PROCESSING MODE: Generating plots from existing data")
        self.progress_signal.emit("=" * 60)
        self.progress_signal.emit(f"Output directory: {spade_output_dir}")
        
        plots_config = post_processing_config.get('plots', {})
        
        # Check HEL detection status for HEL plots
        hel_detection_enabled = (self.spade_params.get('hel_detection_enabled', False) or 
                                self.spade_params.get('experiment_hel_detection', False))
        
        # Generate plots based on config
        plots_generated = 0
        
        if plots_config.get('hel_vs_peak_velocity', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs Peak Velocity plot...")
            self.generate_hel_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('hel_vs_laser_energy', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs Laser Energy plot...")
            self.generate_hel_vs_laser_energy_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('hel_vs_hel_strain_rate', False) and hel_detection_enabled:
            self.progress_signal.emit("\n📊 Generating HEL vs HEL Strain Rate plot...")
            self.generate_hel_vs_hel_strain_rate_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_laser_energy', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Laser Energy plot...")
            self.generate_shock_stress_vs_laser_energy_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_waveplate_angle', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Waveplate Angle plot...")
            self.generate_shock_stress_vs_waveplate_angle_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('laser_energy_vs_waveplate_angle', False):
            self.progress_signal.emit("\n📊 Generating Laser Energy vs Waveplate Angle plot...")
            self.generate_laser_energy_vs_waveplate_angle_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('shock_stress_vs_peak_velocity', False):
            self.progress_signal.emit("\n📊 Generating Shock Stress vs Peak Velocity plot...")
            self.generate_shock_stress_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('row_column_vs_peak_shock_stress', False):
            self.progress_signal.emit("\n📊 Generating Row/Column vs Peak Shock Stress plots...")
            self.generate_row_column_vs_peak_shock_stress_plots(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_peak_velocity_heatmap', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Peak Velocity Heatmap...")
            self.generate_row_column_vs_peak_velocity_heatmap(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_pair_peak_velocity', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Pair vs Peak Velocity plot...")
            self.generate_row_column_pair_vs_peak_velocity_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('flyer_row_column_pair_peak_velocity_by_material', False):
            self.progress_signal.emit("\n📊 Generating Flyer Row/Column Pair vs Peak Velocity plot (by material, color-coded by laser energy)...")
            self.generate_row_column_pair_vs_peak_velocity_by_material_plot(spade_output_dir)
            plots_generated += 1
        
        if plots_config.get('peak_velocity_pattern_analysis', False):
            self.progress_signal.emit("\n📊 Generating Peak Velocity Pattern Analysis plot...")
            self.generate_peak_velocity_pattern_analysis_plot(spade_output_dir)
            plots_generated += 1
        
        self.progress_signal.emit("\n" + "=" * 60)
        self.progress_signal.emit(f"✅ Post-processing complete! Generated {plots_generated} plot(s)")
        self.progress_signal.emit("=" * 60)
        return True

    def create_parameter_mapping_report(self, velocity_shots_data, spade_output_dir):
        """Create a detailed report of parameter mapping for debugging"""
        try:
            report_lines = []
            report_lines.append("=== PARAMETER MAPPING REPORT ===")
            report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            if self.param_data:
                report_lines.append(f"Total parameter entries available: {len(self.param_data)}")
                report_lines.append("Available parameter keys:")
                for key in sorted(self.param_data.keys()):
                    report_lines.append(f"  - {key}")
                report_lines.append("")
            else:
                report_lines.append("No parameter data available")
                report_lines.append("")
            
            report_lines.append("=== VELOCITY FILE MAPPING ===")
            for shot_data in velocity_shots_data:
                file_name = shot_data['file_name']
                param_keys = [k for k in shot_data.keys() if k not in ['file_name', 'mean_velocity_300_400ns_ms', 
                                                                      'time_window_used', 'uncertainty_avg_ms', 
                                                                      't0_ns', 'velocity_threshold_ms',
                                                                      'max_velocity_ms', 'aligned_ok', 'alignment_reason']]
                report_lines.append(f"File: {file_name}")
                report_lines.append(f"  Parameter columns: {param_keys}")
                if param_keys:
                    non_nan_params = [k for k in param_keys if not pd.isna(shot_data.get(k, np.nan))]
                    report_lines.append(f"  Non-NaN parameters: {non_nan_params}")
                else:
                    report_lines.append("  No parameters mapped")
                report_lines.append("")
            
            # Save report
            report_path = os.path.join(spade_output_dir, 'parameter_mapping_report.txt')
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            self.progress_signal.emit(f"Created parameter mapping report: {report_path}")
            
        except Exception as e:
            self.progress_signal.emit(f"Error creating parameter mapping report: {str(e)}")

    def _generate_hel_consecutive_points_report(self, velocity_shots_df, spade_output_dir):
        """
        Generate a summary report listing all traces with their HEL consecutive points count
        and corresponding time duration in nanoseconds.
        """
        try:
            import pandas as pd
            
            # Create report DataFrame with relevant columns
            report_data = []
            for idx, row in velocity_shots_df.iterrows():
                report_data.append({
                    'file_name': row.get('file_name', 'Unknown'),
                    'hel_consecutive_points': row.get('hel_consecutive_points', 0),
                    'hel_segment_time_ns': row.get('hel_segment_time_ns', np.nan),
                    'hel_ok': row.get('hel_ok', False),
                    'hel_strength_gpa': row.get('hel_strength_gpa', np.nan),
                    'aligned_ok': row.get('aligned_ok', False)
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Sort by consecutive points (descending), then by file name
            report_df = report_df.sort_values(
                by=['hel_consecutive_points', 'file_name'], 
                ascending=[False, True]
            )
            
            # Save report
            report_path = os.path.join(spade_output_dir, 'hel_consecutive_points_report.csv')
            report_df.to_csv(report_path, index=False)
            
            # Print summary statistics
            total_traces = len(report_df)
            traces_with_hel = report_df['hel_ok'].sum()
            traces_with_segment = (report_df['hel_consecutive_points'] > 0).sum()
            
            self.progress_signal.emit("\n" + "=" * 60)
            self.progress_signal.emit("HEL Consecutive Points Summary Report")
            self.progress_signal.emit("=" * 60)
            self.progress_signal.emit(f"Total traces processed: {total_traces}")
            min_points = self.spade_params.get('hel_detection_min_points', 3)
            self.progress_signal.emit(f"Traces with HEL segment (≥{min_points} points): {traces_with_segment}")
            self.progress_signal.emit(f"Traces with valid HEL detection: {traces_with_hel}")
            
            if traces_with_segment > 0:
                valid_segments = report_df[report_df['hel_consecutive_points'] > 0]
                self.progress_signal.emit(f"\nConsecutive Points Statistics:")
                self.progress_signal.emit(f"  Min: {valid_segments['hel_consecutive_points'].min()}")
                self.progress_signal.emit(f"  Max: {valid_segments['hel_consecutive_points'].max()}")
                self.progress_signal.emit(f"  Mean: {valid_segments['hel_consecutive_points'].mean():.1f}")
                self.progress_signal.emit(f"  Median: {valid_segments['hel_consecutive_points'].median():.1f}")
                
                self.progress_signal.emit(f"\nSegment Time Duration Statistics (ns):")
                valid_times = valid_segments['hel_segment_time_ns'].dropna()
                if len(valid_times) > 0:
                    self.progress_signal.emit(f"  Min: {valid_times.min():.3f} ns")
                    self.progress_signal.emit(f"  Max: {valid_times.max():.3f} ns")
                    self.progress_signal.emit(f"  Mean: {valid_times.mean():.3f} ns")
                    self.progress_signal.emit(f"  Median: {valid_times.median():.3f} ns")
            
            self.progress_signal.emit(f"\nDetailed report saved to: {report_path}")
            self.progress_signal.emit("=" * 60)
            
        except Exception as e:
            self.progress_signal.emit(f"Warning: Could not generate HEL consecutive points report: {str(e)}")

    def mad_filter(self, values, threshold=3.0):
        """
        Asymmetric statistical outlier filter using the Median Absolute Deviation (MAD).
        Uses MAD_lower (based on values below median) for all values.
        Removes points whose scaled deviation exceeds `threshold`.
        
        Parameters:
        -----------
        values : array-like
            Input values to filter
        threshold : float, default=3.0
            Threshold for modified z-score (typical: 3.0)
        
        Returns:
        --------
        numpy.ndarray
            Boolean mask: True for values to keep, False for outliers
        """
        v = np.array(values)
        median = np.median(v)
        
        # Calculate MAD_lower using only values below median
        lower_mask = v < median
        if np.sum(lower_mask) == 0:
            # If no values below median, use symmetric MAD as fallback
            abs_dev = np.abs(v - median)
            mad = np.median(abs_dev)
            if mad == 0:
                return np.ones_like(v, dtype=bool)
            modified_z = 0.6745 * abs_dev / mad
            return modified_z < threshold
        
        mad_lower = np.median(np.abs(v[lower_mask] - median))
        
        # Avoid division by zero
        if mad_lower == 0:
            return np.ones_like(v, dtype=bool)
        
        # Calculate M_i_lower for all values using MAD_lower
        M_i_lower = 0.6745 * np.abs(v - median) / mad_lower
        
        # Return mask: True for values to keep (M_i_lower < threshold)
        return M_i_lower < threshold
    
    def _get_material_color_mapping(self, materials):
        """
        Generate consistent color mapping for materials across all plots.
        Uses predefined colors for common materials, then colormap for others.
        Ensures same material always gets same color.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Predefined colors for common materials (consistent across all plots)
        predefined_colors = {
            'Cu': '#1f77b4',      # Blue
            'Copper': '#1f77b4',  # Blue
            'Zn': '#ff7f0e',      # Orange
            'Zinc': '#ff7f0e',    # Orange
            'Brass': '#2ca02c',   # Green
            'Al': '#d62728',      # Red
            'Aluminum': '#d62728', # Red
            'Ti': '#9467bd',      # Purple
            'Titanium': '#9467bd', # Purple
            'Steel': '#8c564b',   # Brown
            'Fe': '#e377c2',      # Pink
            'Iron': '#e377c2',    # Pink
        }
        
        # Use Set3 for <=12 materials, tab20 for more
        if len(materials) <= 12:
            cmap = plt.get_cmap('Set3')
        else:
            cmap = plt.get_cmap('tab20')
        
        color_mapping = {}
        predefined_used = set()
        
        # First, assign predefined colors
        for material in materials:
            material_key = str(material).strip()
            if material_key in predefined_colors:
                color_mapping[material] = predefined_colors[material_key]
                predefined_used.add(material_key)
        
        # Then assign colors from colormap for remaining materials
        # Use a deterministic order based on material name hash for consistency
        remaining_materials = sorted([m for m in materials if m not in color_mapping], 
                                     key=lambda x: hash(str(x)))
        
        for i, material in enumerate(remaining_materials):
            # Use hash to get consistent index, but map to colormap range
            material_hash = hash(str(material))
            # Map hash to 0-1 range for colormap
            color_idx = abs(material_hash) % 1000 / 1000.0
            color_mapping[material] = cmap(color_idx)
        
        return color_mapping

    def _find_parameter_column(self, df, candidate_names):
        """
        Find a parameter column in a dataframe by matching against candidate names.
        Comparison is case-insensitive and ignores spaces/underscores/hyphens.
        """
        if df is None or df.empty:
            return None
        
        normalized_columns = {
            col: ''.join(ch for ch in col.lower() if ch.isalnum())
            for col in df.columns
        }
        
        for candidate in candidate_names:
            normalized_candidate = ''.join(ch for ch in candidate.lower() if ch.isalnum())
            for col, normalized in normalized_columns.items():
                if normalized == normalized_candidate:
                    return col
                # Also allow candidate to be substring of column name
                if normalized_candidate in normalized:
                    return col
        return None

    def _convert_row_column_to_numeric(self, series):
        """
        Convert row/column labels (letters, numbers, or mixed) to numeric values for plotting.
        Returns tuple (numeric_series, tick_mapping) where tick_mapping maps numeric value -> label.
        """
        import numpy as np
        import pandas as pd
        
        if series is None or len(series) == 0:
            return pd.Series(dtype=float), {}
        
        series = pd.Series(series)
        
        # 1) Try direct numeric conversion
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() > 0:
            numeric_series = numeric.astype(float)
            mapping = {}
            for num, label in zip(numeric_series, series):
                if pd.notna(num) and pd.notna(label):
                    mapping[num] = str(label)
            return numeric_series, mapping
        
        # 2) Try alphabetic conversion (A=1, B=2, AA=27, etc.)
        def alpha_to_num(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return np.nan
            value_str = str(value).strip().upper()
            if not value_str:
                return np.nan
            # Keep only letters
            letters = ''.join(ch for ch in value_str if ch.isalpha())
            if not letters:
                return np.nan
            total = 0
            for ch in letters:
                if 'A' <= ch <= 'Z':
                    total = total * 26 + (ord(ch) - 64)
                else:
                    return np.nan
            return float(total)
        
        alpha_numeric = series.apply(alpha_to_num)
        if alpha_numeric.notna().sum() > 0:
            mapping = {}
            for num, label in zip(alpha_numeric, series):
                if pd.notna(num) and pd.notna(label):
                    mapping[num] = str(label)
            return alpha_numeric.astype(float), mapping
        
        # 3) Fallback: assign category codes in sorted order
        unique_values = [val for val in series.dropna().unique() if str(val).strip()]
        unique_values_sorted = sorted(unique_values, key=lambda x: str(x))
        value_to_code = {val: idx + 1 for idx, val in enumerate(unique_values_sorted)}
        numeric_series = series.map(value_to_code).astype(float)
        mapping = {code: str(val) for val, code in value_to_code.items()}
        return numeric_series, mapping

    def generate_hel_vs_laser_energy_plot(self, spade_output_dir):
        """Generate HEL vs Laser Energy plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs Laser Energy plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Expected path: {velocity_shots_path}")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:15])}...")
            
            # Check if HEL and laser energy columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist())}")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data")
            
            # Look for laser energy column - prioritize 'Laser_Target_Energy (mJ)' from parameter file
            laser_energy_col = None
            energy_in_mj = False
            
            # First, check for exact match with 'Laser_Target_Energy (mJ)'
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
                self.progress_signal.emit("✓ Using 'Laser_Target_Energy (mJ)' from parameter file")
            else:
                # Fallback to other possible names
                possible_names = [
                    'Laser_Target_Energy (mJ)',  # Try exact match first (case-sensitive)
                    'Laser Target Energy (mJ)',  # With space instead of underscore
                    'Laser_Target_Energy',  # Without unit
                    'Laser Target Energy',  # With space, without unit
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy', 
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        # Check for exact match first, then substring match
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            # Check if it's in mJ
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found in parameter file - skipping HEL vs Laser Energy plot")
                self.progress_signal.emit(f"   Available columns: {', '.join(df.columns.tolist()[:10])}...")
                return
            
            # Filter data: only rows with valid HEL and laser energy
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df[laser_energy_col].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and laser energy data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Laser Energy data points - skipping plot")
                hel_count = df['hel_strength_gpa'].notna().sum()
                energy_count = df[laser_energy_col].notna().sum()
                self.progress_signal.emit(f"   HEL data points: {hel_count}, Laser energy data points: {energy_count}")
                return
            
            # Ensure laser energy is numeric (keep original units - mJ or J)
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            
            # Get material column (should be added by parameter file)
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                # No material column, create a default one
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors
            materials = valid_data[material_col].unique()
            cmap = plt.get_cmap('tab10' if len(materials) <= 10 else 'tab20')
            colors = {mat: cmap(i / max(len(materials), 1)) for i, mat in enumerate(materials)}
            
            # Plot data grouped by material
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Plot with error bars if HEL uncertainty is available
                if 'hel_uncertainty_gpa' in material_data.columns:
                    ax.errorbar(
                        material_data[laser_energy_col],
                        material_data['hel_strength_gpa'],
                        yerr=material_data['hel_uncertainty_gpa'],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=material
                    )
                else:
                    ax.scatter(
                        material_data[laser_energy_col],
                        material_data['hel_strength_gpa'],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=material
                    )
            
            # Set labels and title (use correct unit based on column)
            energy_unit = 'mJ' if energy_in_mj else 'J'
            ax.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Hugoniot Elastic Limit vs Laser Energy by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_laser_energy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs Laser Energy plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs Laser Energy plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_hel_vs_peak_velocity_plot(self, spade_output_dir):
        """Generate HEL vs Peak Velocity scatter plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs Peak Velocity plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check if HEL and peak velocity columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs Peak Velocity plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity (max_velocity_ms) not found - skipping HEL vs Peak Velocity plot")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data")
            
            # Filter data: only rows with valid HEL and peak velocity
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df['max_velocity_ms'].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and peak velocity data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Peak Velocity data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data['max_velocity_ms'] = pd.to_numeric(valid_data['max_velocity_ms'], errors='coerce')
            valid_data['hel_strength_gpa'] = pd.to_numeric(valid_data['hel_strength_gpa'], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['max_velocity_ms'].notna()) & (valid_data['hel_strength_gpa'].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if HEL uncertainty is available
                if 'hel_uncertainty_gpa' in material_data.columns:
                    errorbar_handle = ax.errorbar(
                        material_data['max_velocity_ms'],
                        material_data['hel_strength_gpa'],
                        yerr=material_data['hel_uncertainty_gpa'],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data['max_velocity_ms'],
                        material_data['hel_strength_gpa'],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
                
                # Add linear regression line if we have at least 2 points
                if len(material_data) >= 2:
                    x_vals = material_data['max_velocity_ms'].values.astype(float)
                    y_vals = material_data['hel_strength_gpa'].values.astype(float)
                    
                    # Only attempt fit if x range is non-zero
                    if np.nanmax(x_vals) - np.nanmin(x_vals) > 1e-6:
                        slope, intercept = np.polyfit(x_vals, y_vals, 1)
                        x_fit = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
                        y_fit = slope * x_fit + intercept
                        ax.plot(
                            x_fit,
                            y_fit,
                            color=color,
                            linewidth=2,
                            alpha=0.85,
                            linestyle='-',
                        )
            
            # Set labels and title
            ax.set_xlabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Hugoniot Elastic Limit vs Peak Velocity by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_peak_velocity_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs Peak Velocity plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_hel_vs_hel_strain_rate_plot(self, spade_output_dir):
        """Generate HEL vs HEL Strain Rate scatter plot grouped by material"""
        self.progress_signal.emit("Generating HEL vs HEL Strain Rate plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary which contains HEL data
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping HEL vs HEL Strain Rate plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check if HEL and strain rate columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs HEL Strain Rate plot")
                return
            
            if 'hel_strain_rate_s^-1' not in df.columns:
                self.progress_signal.emit("⚠ HEL strain rate not found in velocity shots - skipping HEL vs HEL Strain Rate plot")
                return
            
            # Check how many rows have HEL data
            hel_data_count = df['hel_strength_gpa'].notna().sum()
            strain_rate_count = df['hel_strain_rate_s^-1'].notna().sum()
            self.progress_signal.emit(f"   Found {hel_data_count} rows with HEL data, {strain_rate_count} rows with strain rate data")
            
            # Filter data: only rows with valid HEL and strain rate
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df['hel_strain_rate_s^-1'].notna())].copy()
            
            self.progress_signal.emit(f"   Found {len(valid_data)} rows with both HEL and strain rate data")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Strain Rate data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data['hel_strength_gpa'] = pd.to_numeric(valid_data['hel_strength_gpa'], errors='coerce')
            valid_data['hel_strain_rate_s^-1'] = pd.to_numeric(valid_data['hel_strain_rate_s^-1'], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['hel_strength_gpa'].notna()) & (valid_data['hel_strain_rate_s^-1'].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter
                scatter_handle = ax.scatter(
                    material_data['hel_strain_rate_s^-1'],
                    material_data['hel_strength_gpa'],
                    marker=marker,
                    c=[color],
                    s=100,
                    alpha=0.7,
                    label=f"{material} (n={n_points})",
                    edgecolors='black',
                    linewidths=0.5
                )
                legend_handles.append(scatter_handle)
                legend_labels.append(f"{material} (n={n_points})")
            
            ax.set_xlabel('HEL Strain Rate (s⁻¹)', fontsize=14, fontweight='bold')
            ax.set_ylabel('HEL Strength (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('HEL vs HEL Strain Rate', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Format x-axis in scientific notation
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best', fontsize=10, title='Material')
            
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'hel_vs_hel_strain_rate.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated HEL vs HEL Strain Rate plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating HEL vs HEL Strain Rate plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_laser_energy_plot(self, spade_output_dir):
        """Generate Shock Stress vs Laser Energy scatter plot grouped by material"""
        self.progress_signal.emit("Generating Shock Stress vs Laser Energy plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Laser Energy plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Look for laser energy column
            laser_energy_col = None
            energy_in_mj = False
            
            # Check for laser energy column (same logic as HEL plot)
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping Shock Stress vs Laser Energy plot")
                return
            
            # Get or calculate shock stress
            shock_stress_col = None
            shock_stress_unc_col = None
            
            # First, try to get from ALPSS results (if available)
            if 'ALPSS_Peak_Shock_Stress_GPa' in df.columns:
                shock_stress_col = 'ALPSS_Peak_Shock_Stress_GPa'
                shock_stress_unc_col = 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' if 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' in df.columns else None
            elif 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
                shock_stress_unc_col = 'Peak Shock Stress Uncertainty (GPa)' if 'Peak Shock Stress Uncertainty (GPa)' in df.columns else None
            else:
                # Calculate shock stress from max velocity using material properties
                self.progress_signal.emit("   Calculating shock stress from max velocity...")
                
                # Get material column
                material_col = None
                for col_name in df.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                
                if material_col is None or 'max_velocity_ms' not in df.columns:
                    self.progress_signal.emit("⚠ Cannot calculate shock stress - missing material or velocity data")
                    return
                
                # Calculate shock stress for each row
                shock_stress_values = []
                for idx, row in df.iterrows():
                    material = row.get(material_col, 'Unknown')
                    max_velocity = row.get('max_velocity_ms', np.nan)
                    
                    if pd.isna(max_velocity):
                        shock_stress_values.append(np.nan)
                        continue
                    
                    # Get material properties
                    if self.material_properties and material in self.material_properties:
                        props = self.material_properties[material]
                        density = props.get('density', 8960)  # Default to Cu density (kg/m³)
                        acoustic_velocity = props.get('bulk_wave_speed', props.get('C0', 3950))  # Default to Cu (m/s)
                        # S parameter for Hugoniot EOS: U = c + S*u_p
                        # Default S values based on common materials
                        S = props.get('S', props.get('slope_parameter', None))
                        if S is None:
                            # Use default S values for common materials
                            material_lower = str(material).lower()
                            if 'cu' in material_lower or 'copper' in material_lower:
                                S = 1.49
                            elif 'zn' in material_lower or 'zinc' in material_lower:
                                S = 1.30
                            elif 'brass' in material_lower:
                                S = 1.43
                            else:
                                S = 1.49  # Default to Cu value
                    else:
                        # Use defaults (Copper)
                        density = 8960  # kg/m³
                        acoustic_velocity = 3950  # m/s
                        S = 1.49  # Cu slope parameter
                    
                    # Calculate shock stress using EOS: U = c + S*u_p, then σ = ρ * U * u_p * 1e-9 (GPa)
                    # max_velocity is free surface velocity (u_fs), particle velocity u_p = u_fs / 2
                    u_p = max_velocity / 2.0  # Particle velocity = free surface velocity / 2
                    shock_velocity = acoustic_velocity + S * u_p  # U = c + S*u_p
                    shock_stress = density * shock_velocity * u_p * 1e-9  # σ = ρ * U * u_p (GPa)
                    shock_stress_values.append(shock_stress)
                
                df['Calculated_Shock_Stress_GPa'] = shock_stress_values
                shock_stress_col = 'Calculated_Shock_Stress_GPa'
            
            # Filter data: only rows with valid shock stress and laser energy
            valid_data = df[(df[shock_stress_col].notna()) & (df[laser_energy_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Shock Stress + Laser Energy data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column first
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data[laser_energy_col].notna()) & (valid_data[shock_stress_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if uncertainty is available
                if shock_stress_unc_col and shock_stress_unc_col in material_data.columns:
                    errorbar_handle = ax.errorbar(
                        material_data[laser_energy_col],
                        material_data[shock_stress_col],
                        yerr=material_data[shock_stress_unc_col],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data[laser_energy_col],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            energy_unit = 'mJ' if energy_in_mj else 'J'
            ax.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_ylabel('Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Shock Stress vs Laser Energy by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_laser_energy_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Laser Energy plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Laser Energy plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_waveplate_angle_plot(self, spade_output_dir):
        """Generate Shock Stress vs Waveplate Angle scatter plot grouped by material"""
        self.progress_signal.emit("Generating Shock Stress vs Waveplate Angle plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Waveplate Angle plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Look for waveplate angle column
            waveplate_angle_col = None
            
            # Check for waveplate angle column
            if 'Waveplate_Angle (Degrees)' in df.columns:
                waveplate_angle_col = 'Waveplate_Angle (Degrees)'
            else:
                possible_names = [
                    'Waveplate_Angle (Degrees)', 'Waveplate Angle (Degrees)',
                    'Waveplate_Angle', 'Waveplate Angle', 'WaveplateAngle',
                    'waveplate_angle', 'waveplate angle', 'Waveplate',
                    'Angle (Degrees)', 'Angle', 'angle'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            waveplate_angle_col = col_name
                            break
                    if waveplate_angle_col:
                        break
            
            if waveplate_angle_col is None:
                self.progress_signal.emit("⚠ Waveplate angle column not found - skipping Shock Stress vs Waveplate Angle plot")
                return
            
            self.progress_signal.emit(f"✓ Found waveplate angle column: '{waveplate_angle_col}'")
            
            # Get or calculate shock stress (same logic as laser energy plot)
            shock_stress_col = None
            shock_stress_unc_col = None
            
            # First, try to get from ALPSS results (if available)
            if 'ALPSS_Peak_Shock_Stress_GPa' in df.columns:
                shock_stress_col = 'ALPSS_Peak_Shock_Stress_GPa'
                shock_stress_unc_col = 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' if 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' in df.columns else None
            elif 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
                shock_stress_unc_col = 'Peak Shock Stress Uncertainty (GPa)' if 'Peak Shock Stress Uncertainty (GPa)' in df.columns else None
            else:
                # Calculate shock stress from max velocity using material properties
                self.progress_signal.emit("   Calculating shock stress from max velocity...")
                
                # Get material column
                material_col = None
                for col_name in df.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                
                if material_col is None or 'max_velocity_ms' not in df.columns:
                    self.progress_signal.emit("⚠ Cannot calculate shock stress - missing material or velocity data")
                    return
                
                # Calculate shock stress for each row
                shock_stress_values = []
                for idx, row in df.iterrows():
                    material = row.get(material_col, 'Unknown')
                    max_velocity = row.get('max_velocity_ms', np.nan)
                    
                    if pd.isna(max_velocity):
                        shock_stress_values.append(np.nan)
                        continue
                    
                    # Get material properties
                    if self.material_properties and material in self.material_properties:
                        props = self.material_properties[material]
                        density = props.get('density', 8960)  # Default to Cu density (kg/m³)
                        acoustic_velocity = props.get('bulk_wave_speed', props.get('C0', 3950))  # Default to Cu (m/s)
                        # S parameter for Hugoniot EOS: U = c + S*u_p
                        # Default S values based on common materials
                        S = props.get('S', props.get('slope_parameter', None))
                        if S is None:
                            # Use default S values for common materials
                            material_lower = str(material).lower()
                            if 'cu' in material_lower or 'copper' in material_lower:
                                S = 1.49
                            elif 'zn' in material_lower or 'zinc' in material_lower:
                                S = 1.30
                            elif 'brass' in material_lower:
                                S = 1.43
                            else:
                                S = 1.49  # Default to Cu value
                    else:
                        # Use defaults (Copper)
                        density = 8960  # kg/m³
                        acoustic_velocity = 3950  # m/s
                        S = 1.49  # Cu slope parameter
                    
                    # Calculate shock stress using EOS: U = c + S*u_p, then σ = ρ * U * u_p * 1e-9 (GPa)
                    # max_velocity is free surface velocity (u_fs), particle velocity u_p = u_fs / 2
                    u_p = max_velocity / 2.0  # Particle velocity = free surface velocity / 2
                    shock_velocity = acoustic_velocity + S * u_p  # U = c + S*u_p
                    shock_stress = density * shock_velocity * u_p * 1e-9  # σ = ρ * U * u_p (GPa)
                    shock_stress_values.append(shock_stress)
                
                df['Calculated_Shock_Stress_GPa'] = shock_stress_values
                shock_stress_col = 'Calculated_Shock_Stress_GPa'
            
            # Filter data: only rows with valid shock stress and waveplate angle
            valid_data = df[(df[shock_stress_col].notna()) & (df[waveplate_angle_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Shock Stress + Waveplate Angle data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column first
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[waveplate_angle_col] = pd.to_numeric(valid_data[waveplate_angle_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data[waveplate_angle_col].notna()) & (valid_data[shock_stress_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if uncertainty is available
                if shock_stress_unc_col and shock_stress_unc_col in material_data.columns:
                    errorbar_handle = ax.errorbar(
                        material_data[waveplate_angle_col],
                        material_data[shock_stress_col],
                        yerr=material_data[shock_stress_unc_col],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data[waveplate_angle_col],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            ax.set_xlabel('Waveplate Angle (Degrees)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Shock Stress vs Waveplate Angle by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_waveplate_angle_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Waveplate Angle plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Waveplate Angle plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_laser_energy_vs_waveplate_angle_plot(self, spade_output_dir):
        """Generate Laser Energy vs Waveplate Angle scatter plot to show energy variation at each angle"""
        self.progress_signal.emit("Generating Laser Energy vs Waveplate Angle plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Look for waveplate angle column
            waveplate_angle_col = None
            if 'Waveplate_Angle (Degrees)' in df.columns:
                waveplate_angle_col = 'Waveplate_Angle (Degrees)'
            else:
                possible_names = [
                    'Waveplate_Angle (Degrees)', 'Waveplate Angle (Degrees)',
                    'Waveplate_Angle', 'Waveplate Angle', 'WaveplateAngle',
                    'waveplate_angle', 'waveplate angle', 'Waveplate',
                    'Angle (Degrees)', 'Angle', 'angle'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            waveplate_angle_col = col_name
                            break
                    if waveplate_angle_col:
                        break
            
            if waveplate_angle_col is None:
                self.progress_signal.emit("⚠ Waveplate angle column not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            # Look for laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            self.progress_signal.emit(f"✓ Found waveplate angle column: '{waveplate_angle_col}'")
            self.progress_signal.emit(f"✓ Found laser energy column: '{laser_energy_col}'")
            
            # Filter to only include traces that were successfully plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            if 'aligned_ok' in df.columns:
                valid_data = df[df['aligned_ok'] == True].copy()
            else:
                valid_data = df.copy()
            
            # Skip Unknown material if configured
            skip_unknown = self.spade_params.get('skip_unknown_material_traces', False)
            if skip_unknown:
                material_col = None
                for col_name in valid_data.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                if material_col:
                    valid_data = valid_data[valid_data[material_col].str.lower() != 'unknown'].copy()
            
            # Convert to numeric
            valid_data[waveplate_angle_col] = pd.to_numeric(valid_data[waveplate_angle_col], errors='coerce')
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            
            # Remove rows with NaN values
            valid_data = valid_data.dropna(subset=[waveplate_angle_col, laser_energy_col])
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data points - skipping Laser Energy vs Waveplate Angle plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get material column for color coding
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col and material_col in valid_data.columns:
                materials = valid_data[material_col].unique()
                colors = self._get_material_color_mapping(materials)
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
                legend_handles = []
                legend_labels = []
                
                for i, material in enumerate(materials):
                    material_data = valid_data[valid_data[material_col] == material]
                    
                    if len(material_data) == 0:
                        continue
                    
                    marker = markers[i % len(markers)]
                    color = colors[material]
                    n_points = len(material_data)
                    
                    scatter_handle = ax.scatter(
                        material_data[waveplate_angle_col],
                        material_data[laser_energy_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                    legend_labels.append(f"{material} (n={n_points})")
            else:
                # No material column, plot all points in one color
                ax.scatter(
                    valid_data[waveplate_angle_col],
                    valid_data[laser_energy_col],
                    marker='o',
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"All data (n={len(valid_data)})"
                )
            
            # Set labels and title
            ax.set_xlabel('Waveplate Angle (Degrees)', fontsize=14, fontweight='bold')
            energy_unit = "mJ" if energy_in_mj else "J"
            ax.set_ylabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax.set_title('Laser Energy vs Waveplate Angle\n(Showing Energy Variation at Each Angle)', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if material_col and material_col in valid_data.columns:
                ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            else:
                ax.legend(loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'laser_energy_vs_waveplate_angle.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Laser Energy vs Waveplate Angle plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Laser Energy vs Waveplate Angle plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_shock_stress_vs_peak_velocity_plot(self, spade_output_dir):
        """Generate Shock Stress vs Peak Velocity scatter plot grouped by material"""
        self.progress_signal.emit("Generating Shock Stress vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Load velocity shots summary
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Shock Stress vs Peak Velocity plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            self.progress_signal.emit(f"   Loaded velocity shots summary with {len(df)} rows")
            
            # Check for peak velocity column (max_velocity_ms)
            peak_velocity_col = 'max_velocity_ms'
            if peak_velocity_col not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping Shock Stress vs Peak Velocity plot")
                return
            
            # Get or calculate shock stress (same logic as other plots)
            shock_stress_col = None
            shock_stress_unc_col = None
            
            # First, try to get from ALPSS results (if available)
            if 'ALPSS_Peak_Shock_Stress_GPa' in df.columns:
                shock_stress_col = 'ALPSS_Peak_Shock_Stress_GPa'
                shock_stress_unc_col = 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' if 'ALPSS_Peak_Shock_Stress_Uncertainty_GPa' in df.columns else None
            elif 'Peak Shock Stress (GPa)' in df.columns:
                shock_stress_col = 'Peak Shock Stress (GPa)'
                shock_stress_unc_col = 'Peak Shock Stress Uncertainty (GPa)' if 'Peak Shock Stress Uncertainty (GPa)' in df.columns else None
            else:
                # Calculate shock stress from max velocity using material properties
                self.progress_signal.emit("   Calculating shock stress from max velocity...")
                
                # Get material column
                material_col = None
                for col_name in df.columns:
                    if 'material' in col_name.lower() and 'sample' in col_name.lower():
                        material_col = col_name
                        break
                
                if material_col is None:
                    self.progress_signal.emit("⚠ Cannot calculate shock stress - missing material data")
                    return
                
                # Calculate shock stress for each row
                shock_stress_values = []
                for idx, row in df.iterrows():
                    material = row.get(material_col, 'Unknown')
                    max_velocity = row.get(peak_velocity_col, np.nan)
                    
                    if pd.isna(max_velocity):
                        shock_stress_values.append(np.nan)
                        continue
                    
                    # Get material properties
                    if self.material_properties and material in self.material_properties:
                        props = self.material_properties[material]
                        density = props.get('density', 8960)  # Default to Cu density (kg/m³)
                        acoustic_velocity = props.get('bulk_wave_speed', props.get('C0', 3950))  # Default to Cu (m/s)
                        # S parameter for Hugoniot EOS: U = c + S*u_p
                        # Default S values based on common materials
                        S = props.get('S', props.get('slope_parameter', None))
                        if S is None:
                            # Use default S values for common materials
                            material_lower = str(material).lower()
                            if 'cu' in material_lower or 'copper' in material_lower:
                                S = 1.49
                            elif 'zn' in material_lower or 'zinc' in material_lower:
                                S = 1.30
                            elif 'brass' in material_lower:
                                S = 1.43
                            else:
                                S = 1.49  # Default to Cu value
                    else:
                        # Use defaults (Copper)
                        density = 8960  # kg/m³
                        acoustic_velocity = 3950  # m/s
                        S = 1.49  # Cu slope parameter
                    
                    # Calculate shock stress using EOS: U = c + S*u_p, then σ = ρ * U * u_p * 1e-9 (GPa)
                    # max_velocity is free surface velocity (u_fs), particle velocity u_p = u_fs / 2
                    u_p = max_velocity / 2.0  # Particle velocity = free surface velocity / 2
                    shock_velocity = acoustic_velocity + S * u_p  # U = c + S*u_p
                    shock_stress = density * shock_velocity * u_p * 1e-9  # σ = ρ * U * u_p (GPa)
                    shock_stress_values.append(shock_stress)
                
                df['Calculated_Shock_Stress_GPa'] = shock_stress_values
                shock_stress_col = 'Calculated_Shock_Stress_GPa'
            
            # Filter data: only rows with valid shock stress and peak velocity
            valid_data = df[(df[shock_stress_col].notna()) & (df[peak_velocity_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid Shock Stress + Peak Velocity data points - skipping plot")
                return
            
            # Filter to only include traces that would be plotted in all_velocity_traces_plot
            # (i.e., only accepted traces)
            
            # 1. Must have aligned successfully (reached threshold)
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True].copy()
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            # 2. Skip Unknown material if configured (same as all_velocity_traces_plot)
            skip_unknown = self._should_skip_unknown_materials()
            
            # Get material column first
            material_col = None
            for col_name in valid_data.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            
            # Filter out Unknown material if configured
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after filtering - skipping plot")
                return
            
            # Ensure numeric
            valid_data[peak_velocity_col] = pd.to_numeric(valid_data[peak_velocity_col], errors='coerce')
            valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
            
            # Convert free surface velocity (peak velocity) to particle velocity for plotting
            # u_p = u_fs / 2 (free surface velocity is twice the particle velocity)
            valid_data['particle_velocity_ms'] = valid_data[peak_velocity_col] / 2.0
            
            # Remove any rows that became NaN after conversion
            valid_data = valid_data[(valid_data['particle_velocity_ms'].notna()) & (valid_data[shock_stress_col].notna())].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after numeric conversion - skipping plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique materials and assign colors (consistent across all plots)
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            
            # Plot data grouped by material (scatter plot)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            legend_handles = []
            legend_labels = []
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                
                # Skip if no data points for this material
                if len(material_data) == 0:
                    continue
                
                # Get marker for this material
                marker = markers[i % len(markers)]
                color = colors[material]
                
                # Count data points for this material
                n_points = len(material_data)
                
                # Plot scatter with error bars if uncertainty is available
                # Use particle velocity on x-axis (not free surface velocity)
                if shock_stress_unc_col and shock_stress_unc_col in material_data.columns:
                    errorbar_handle = ax.errorbar(
                        material_data['particle_velocity_ms'],
                        material_data[shock_stress_col],
                        yerr=material_data[shock_stress_unc_col],
                        fmt=marker,
                        color=color,
                        markersize=10,
                        linewidth=0,
                        elinewidth=1.5,
                        capsize=4,
                        alpha=0.7,
                        label=f"{material} (n={n_points})"
                    )
                    # errorbar returns a container, use the first element (line) for legend
                    legend_handles.append(errorbar_handle[0])
                else:
                    scatter_handle = ax.scatter(
                        material_data['particle_velocity_ms'],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=100,
                        alpha=0.7,
                        label=f"{material} (n={n_points})",
                        edgecolors='black',
                        linewidths=0.5
                    )
                    legend_handles.append(scatter_handle)
                
                legend_labels.append(f"{material} (n={n_points})")
            
            # Set labels and title
            # Note: x-axis shows particle velocity (u_p = u_fs/2) to show the quadratic relationship
            ax.set_xlabel('Particle Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Shock Stress (GPa)', fontsize=14, fontweight='bold')
            ax.set_title('Shock Stress vs Particle Velocity by Material', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=11)
            
            # Tight layout and save
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'shock_stress_vs_peak_velocity_by_material.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Shock Stress vs Peak Velocity plot: {plot_path}")
            self.progress_signal.emit(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating Shock Stress vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_vs_peak_shock_stress_plots(self, spade_output_dir):
        """Plot Flyer Column/Row versus peak shock stress with material grouping"""
        self.progress_signal.emit("Generating Flyer Column/Row vs Peak Shock Stress plots...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping Flyer Row/Column plots")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping Flyer Row/Column plots")
                return
            
            # Identify required columns
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None and col_col is None:
                self.progress_signal.emit("⚠ Flyer Row/Column columns not found - skipping positional plots")
                return
            
            # Determine shock stress column
            shock_stress_col = None
            for candidate in ['ALPSS_Peak_Shock_Stress_GPa', 'Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa_Final',
                              'Calculated_Shock_Stress_GPa']:
                if candidate in df.columns:
                    shock_stress_col = candidate
                    break
            
            if shock_stress_col is None:
                self.progress_signal.emit("⚠ Peak shock stress column not found - skipping Flyer Row/Column plots")
                return
            
            required_columns = [shock_stress_col]
            if col_col:
                required_columns.append(col_col)
            if row_col:
                required_columns.append(row_col)
            
            subset_columns = required_columns + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_columns].copy()
            valid_data = valid_data[valid_data[shock_stress_col].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid shock stress data for Flyer Row/Column plots")
                return
            
            # Filter aligned traces only
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No aligned traces available for Flyer Row/Column plots")
                return
            
            # Skip Unknown material if configured
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            else:
                valid_data[material_col] = df.loc[valid_data.index, material_col]
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces (removed {before_filter - after_filter} Unknown)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping positional plots")
                return
            
            plots_to_generate = []
            if col_col is not None:
                col_numeric, col_ticks = self._convert_row_column_to_numeric(valid_data[col_col])
                valid_data['Column_numeric'] = col_numeric
                column_valid = valid_data[valid_data['Column_numeric'].notna()].copy()
                if len(column_valid) > 0:
                    plots_to_generate.append(('Flyer Column', 'Column_numeric', col_ticks, column_valid))
                else:
                    self.progress_signal.emit("⚠ No valid numeric Flyer Column values - skipping column plot")
            
            if row_col is not None:
                row_numeric, row_ticks = self._convert_row_column_to_numeric(valid_data[row_col])
                valid_data['Row_numeric'] = row_numeric
                row_valid = valid_data[valid_data['Row_numeric'].notna()].copy()
                if len(row_valid) > 0:
                    plots_to_generate.append(('Flyer Row', 'Row_numeric', row_ticks, row_valid))
                else:
                    self.progress_signal.emit("⚠ No valid numeric Flyer Row values - skipping row plot")
            
            if not plots_to_generate:
                self.progress_signal.emit("⚠ No positional plots generated due to missing row/column data")
                return
            
            # Get materials and colors
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            
            fig, axes = plt.subplots(1, len(plots_to_generate), figsize=(9 * len(plots_to_generate), 7), sharey=True)
            if len(plots_to_generate) == 1:
                axes = [axes]
            
            legend_handles = []
            legend_labels = []
            
            for axis_idx, (axis_label, numeric_col, tick_mapping, plot_data) in enumerate(plots_to_generate):
                ax = axes[axis_idx]
                for i, material in enumerate(materials):
                    material_data = plot_data[plot_data[material_col] == material]
                    if len(material_data) == 0:
                        continue
                    marker = markers[i % len(markers)]
                    color = colors[material]
                    scatter_handle = ax.scatter(
                        material_data[numeric_col],
                        material_data[shock_stress_col],
                        marker=marker,
                        c=[color],
                        s=90,
                        alpha=0.8,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{material} (n={len(material_data)})"
                    )
                    if axis_idx == 0:
                        legend_handles.append(scatter_handle)
                        legend_labels.append(f"{material} (n={len(material_data)})")
                
                ax.set_xlabel(axis_label, fontsize=14, fontweight='bold')
                if axis_idx == 0:
                    ax.set_ylabel('Peak Shock Stress (GPa)', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if tick_mapping:
                    ticks = sorted(tick_mapping.keys())
                    labels = [tick_mapping[t] for t in ticks]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels, rotation=45, ha='right')
            
            title = 'Flyer Column/Row vs Peak Shock Stress'
            fig.suptitle(title, fontsize=18, fontweight='bold')
            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            plt.tight_layout(rect=[0, 0, 0.98, 0.95])
            
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_vs_peak_shock_stress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column vs Peak Shock Stress plots: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column vs Peak Shock Stress plots: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_vs_peak_velocity_heatmap(self, spade_output_dir):
        """Create heatmap of Flyer Row/Column vs Peak Velocity"""
        self.progress_signal.emit("Generating Flyer Row/Column vs Peak Velocity heatmap...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping heatmap")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping heatmap")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping heatmap")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping heatmap")
                return
            
            subset_columns = [row_col, col_col, 'max_velocity_ms'] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_columns].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms'])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for Flyer Row/Column heatmap")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping heatmap")
                return
            
            grouped = valid_data.groupby([row_col, col_col])['max_velocity_ms']
            mean_table = grouped.mean().unstack()
            count_table = grouped.count().unstack().reindex_like(mean_table)
            max_table = grouped.max().unstack().reindex_like(mean_table)
            min_table = grouped.min().unstack().reindex_like(mean_table)
            
            if mean_table.empty:
                self.progress_signal.emit("⚠ Pivot table is empty - skipping heatmap")
                return
            
            rows = list(mean_table.index.astype(str))
            columns = list(mean_table.columns.astype(str))
            data = mean_table.values.astype(float)
            
            fig = plt.figure(figsize=(9, 9))
            # Main heatmap axes - optimized to reduce white space
            ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
            cmap = plt.get_cmap('viridis')
            masked_data = np.ma.masked_invalid(data)
            heatmap = ax.imshow(masked_data, cmap=cmap, aspect='auto', origin='lower')
            
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    value = data[i, j]
                    if not np.isnan(value):
                        count_val = count_table.values[i, j] if not np.isnan(count_table.values[i, j]) else np.nan
                        max_val = max_table.values[i, j] if not np.isnan(max_table.values[i, j]) else np.nan
                        min_val = min_table.values[i, j] if not np.isnan(min_table.values[i, j]) else np.nan
                        text_color = 'white' if np.nanmax(data) > 0 and value > 0.8 * np.nanmax(data) else 'black'
                        label_lines = [f"{value:.1f} m/s"]
                        if not np.isnan(count_val):
                            label_lines.append(f"n={int(count_val)}")
                        if not np.isnan(max_val):
                            label_lines.append(f"max={max_val:.1f}")
                        if not np.isnan(min_val):
                            label_lines.append(f"min={min_val:.1f}")
                        ax.text(j, i, "\n".join(label_lines), ha='center', va='center', color=text_color, fontsize=9)
            
            ax.set_xticks(np.arange(len(columns)))
            ax.set_xticklabels(columns, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels(rows)
            ax.set_xlabel('Flyer Column', fontsize=14, fontweight='bold')
            ax.set_ylabel('Flyer Row', fontsize=14, fontweight='bold')
            ax.set_title('Peak Velocity Heatmap by Flyer Row/Column', fontsize=16, fontweight='bold')
            
            # Colorbar positioned right after heatmap
            cbar = fig.colorbar(heatmap, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label('Peak Velocity (m/s)', fontsize=12)
            
            # Legend/explanation axes on right, positioned after colorbar (compact)
            legend_text_ax = fig.add_axes([0.82, 0.75, 0.16, 0.20])
            legend_text_ax.axis('off')
            legend_text = (
                "Cell labels show:\\n"
                " - Avg peak velocity (m/s)\\n"
                " - n = traces contributing\\n"
                " - max/min = velocity bounds\\n"
                "Example format:\\n"
                "  R1C1 -> \\\"5200 m/s\\\\n"
                "          n=3\\\\n"
                "          max=5450\\\\n"
                "          min=4980\\\"\\n"
                "(values update per cell)"
            )
            legend_text_ax.text(0, 1, legend_text, va='top', fontsize=10.5)
            
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_peak_velocity_heatmap.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column vs Peak Velocity heatmap: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column vs Peak Velocity heatmap: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_pair_vs_peak_velocity_plot(self, spade_output_dir):
        """Plot peak velocity for each Flyer Row/Column pair, grouped by material"""
        self.progress_signal.emit("Generating Flyer Row/Column pair vs Peak Velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping row/column pair plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping row/column pair plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping row/column pair plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping row/column pair plot")
                return
            
            subset_cols = [row_col, col_col, 'max_velocity_ms'] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms'])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for row/column pair plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping row/column pair plot")
                return
            
            # Attach material info
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                valid_data['Material'] = 'Unknown'
                material_col = 'Material'
            else:
                valid_data[material_col] = df.loc[valid_data.index, material_col]
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping row/column pair plot")
                return
            
            # Convert rows/columns to numeric order and build pair labels
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid numeric row/column data - skipping row/column pair plot")
                return
            
            # Determine sorting order (ascending row, then column)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            def format_label(row_value, col_value, row_raw, col_raw):
                # Format row/column labels: remove .0 if present (e.g., R1.0 C1.0 -> R1C1)
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            # Build ordered pair list
            ordered_pairs = []
            pair_labels = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                for col_val in unique_cols:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    ordered_pairs.append((row_val, col_val, label))
                    pair_labels.append(label)
            
            if not ordered_pairs:
                self.progress_signal.emit("⚠ No valid row/column combinations to plot")
                return
            
            pair_to_position = {pair[2]: idx for idx, pair in enumerate(ordered_pairs)}
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            valid_data['RowCol_Pos'] = valid_data['RowCol_Label'].map(pair_to_position)
            
            # Prepare figure
            fig, ax = plt.subplots(figsize=(max(12, len(ordered_pairs) * 0.7), 8))
            materials = valid_data[material_col].unique()
            colors = self._get_material_color_mapping(materials)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
            
            legend_handles = []
            legend_labels = []
            jitter = 0.05
            
            for i, material in enumerate(materials):
                material_data = valid_data[valid_data[material_col] == material]
                if len(material_data) == 0:
                    continue
                marker = markers[i % len(markers)]
                color = colors[material]
                x_positions = material_data['RowCol_Pos'] + np.random.uniform(-jitter, jitter, size=len(material_data))
                scatter_handle = ax.scatter(
                    x_positions,
                    material_data['max_velocity_ms'],
                    marker=marker,
                    c=[color],
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"{material} (n={len(material_data)})"
                )
                legend_handles.append(scatter_handle)
                legend_labels.append(f"{material} (n={len(material_data)})")
            
            ax.set_xticks(range(len(pair_to_position)))
            ax.set_xticklabels(pair_labels, rotation=45, ha='right')
            ax.set_xlabel('Flyer Row/Column Pair', fontsize=14, fontweight='bold')
            ax.set_ylabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax.set_title('Peak Velocity by Flyer Row/Column Pair', fontsize=16, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc='best', title='Material')
            
            plt.tight_layout()
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_pair_peak_velocity.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column pair vs Peak Velocity plot: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column pair vs Peak Velocity plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_row_column_pair_vs_peak_velocity_by_material_plot(self, spade_output_dir):
        """Plot peak velocity for each Flyer Row/Column pair, with 3 subplots (one per material), markers color-coded by laser energy"""
        self.progress_signal.emit("Generating Flyer Row/Column pair vs Peak Velocity plot (by material, color-coded by laser energy)...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping row/column pair by material plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping row/column pair by material plot")
                return
            
            # Find laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping row/column pair by material plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping row/column pair by material plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping row/column pair by material plot")
                return
            
            # Get material column
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                df['Material'] = 'Unknown'
                material_col = 'Material'
            
            subset_cols = [row_col, col_col, 'max_velocity_ms', laser_energy_col, material_col] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms', laser_energy_col])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
                self.progress_signal.emit(f"   After alignment filter: {len(valid_data)} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for row/column pair by material plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            before_2sigma = len(valid_data)
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            after_2sigma = len(valid_data)
            self.progress_signal.emit(f"   After 2-sigma filter: {after_2sigma} traces (removed {before_2sigma - after_2sigma} outliers)")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after 2-sigma filtering - skipping row/column pair by material plot")
                return
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                before_filter = len(valid_data)
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
                after_filter = len(valid_data)
                if before_filter != after_filter:
                    self.progress_signal.emit(f"   After Unknown material filter: {after_filter} traces")
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No traces remaining after material filtering - skipping row/column pair by material plot")
                return
            
            # Convert rows/columns to numeric order and build pair labels
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid numeric row/column data - skipping row/column pair by material plot")
                return
            
            # Determine sorting order (ascending row, then column)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            def format_label(row_value, col_value, row_raw, col_raw):
                # Format row/column labels: remove .0 if present (e.g., R1.0 C1.0 -> R1C1)
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            # Build ordered pair list with zigzag pattern:
            # Odd rows (1, 3, 5): columns in reverse order (C5, C4, C3, C2, C1)
            # Even rows (2, 4): columns in normal order (C1, C2, C3, C4, C5)
            ordered_pairs = []
            pair_labels = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                
                # Determine column order: reverse for odd rows, normal for even rows
                # Row 1, 3, 5 are odd -> reverse; Row 2, 4 are even -> normal
                is_odd_row = (int(row_val) % 2 == 1)
                col_order = reversed(unique_cols) if is_odd_row else unique_cols
                
                for col_val in col_order:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    ordered_pairs.append((row_val, col_val, label))
                    pair_labels.append(label)
            
            if not ordered_pairs:
                self.progress_signal.emit("⚠ No valid row/column combinations to plot")
                return
            
            pair_to_position = {pair[2]: idx for idx, pair in enumerate(ordered_pairs)}
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            valid_data['RowCol_Pos'] = valid_data['RowCol_Label'].map(pair_to_position)
            
            # Ensure laser energy is numeric
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data = valid_data[valid_data[laser_energy_col].notna()].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid laser energy data - skipping row/column pair by material plot")
                return
            
            # Get unique materials (limit to 3 for 3 subplots)
            materials = sorted(valid_data[material_col].unique())[:3]
            if len(materials) == 0:
                self.progress_signal.emit("⚠ No materials found - skipping row/column pair by material plot")
                return
            
            # Get global laser energy range for consistent colormap across all subplots
            global_energy_min = valid_data[laser_energy_col].min()
            global_energy_max = valid_data[laser_energy_col].max()
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(3, 1, figsize=(max(12, len(ordered_pairs) * 0.7), 12))
            if len(materials) < 3:
                # Hide unused subplots
                for i in range(len(materials), 3):
                    axes[i].axis('off')
            
            # Use a colormap for laser energy
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=global_energy_min, vmax=global_energy_max)
            jitter = 0.05
            
            for subplot_idx, material in enumerate(materials):
                ax = axes[subplot_idx]
                material_data = valid_data[valid_data[material_col] == material].copy()
                
                if len(material_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {material}', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_ylabel('Peak Velocity (m/s)', fontsize=12, fontweight='bold')
                    continue
                
                x_positions = material_data['RowCol_Pos'] + np.random.uniform(-jitter, jitter, size=len(material_data))
                laser_energies = material_data[laser_energy_col].values
                
                # Color-code by laser energy
                scatter = ax.scatter(
                    x_positions,
                    material_data['max_velocity_ms'],
                    c=laser_energies,
                    cmap=cmap,
                    norm=norm,
                    s=80,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                ax.set_xticks(range(len(pair_to_position)))
                ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Peak Velocity (m/s)', fontsize=12, fontweight='bold')
                ax.set_title(f'{material} (n={len(material_data)})', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Set x-axis label on bottom subplot only
            axes[-1].set_xlabel('Flyer Row/Column Pair', fontsize=14, fontweight='bold')
            
            # Overall title
            fig.suptitle('Peak Velocity by Flyer Row/Column Pair (Color-coded by Laser Energy)', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            # Adjust layout first to make room for colorbar
            plt.tight_layout(rect=[0, 0, 0.92, 0.99])  # Leave space for colorbar on the right
            
            # Add colorbar for laser energy (shared across all subplots) - positioned to the right of plot area
            energy_unit = 'mJ' if energy_in_mj else 'J'
            # Create colorbar in a separate axes to the right of the plot
            cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coordinates
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
            cbar.set_label(f'Laser Energy ({energy_unit})', fontsize=12, fontweight='bold')
            plot_path = os.path.join(spade_output_dir, 'flyer_row_column_pair_peak_velocity_by_material_laser_energy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Flyer Row/Column pair vs Peak Velocity plot (by material, color-coded by laser energy): {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Flyer Row/Column pair vs Peak Velocity plot (by material): {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def generate_peak_velocity_pattern_analysis_plot(self, spade_output_dir):
        """Generate diagnostic plots to analyze patterns: 1) Effect of laser energy on peak velocity, 2) Effect of test location (RxCy) on peak velocity"""
        self.progress_signal.emit("Generating Peak Velocity Pattern Analysis plots...")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            velocity_shots_path = os.path.join(spade_output_dir, 'velocity_shots_summary.csv')
            if not os.path.exists(velocity_shots_path):
                self.progress_signal.emit("⚠ Velocity shots summary not found - skipping pattern analysis plot")
                return
            
            df = pd.read_csv(velocity_shots_path)
            if df.empty:
                self.progress_signal.emit("⚠ Velocity shots summary is empty - skipping pattern analysis plot")
                return
            
            # Find laser energy column
            laser_energy_col = None
            energy_in_mj = False
            if 'Laser_Target_Energy (mJ)' in df.columns:
                laser_energy_col = 'Laser_Target_Energy (mJ)'
                energy_in_mj = True
            else:
                possible_names = [
                    'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)',
                    'Laser_Target_Energy', 'Laser Target Energy',
                    'Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy',
                    'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power'
                ]
                for col_name in df.columns:
                    col_normalized = col_name.lower().replace('_', " ").replace('-', " ")
                    for possible in possible_names:
                        possible_normalized = possible.lower().replace('_', " ").replace('-', " ")
                        if col_name == possible or possible_normalized in col_normalized:
                            laser_energy_col = col_name
                            if "mj" in col_name.lower() or "(mj)" in col_name.lower():
                                energy_in_mj = True
                            break
                    if laser_energy_col:
                        break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found - skipping pattern analysis plot")
                return
            
            row_col_candidates = ['Flyer_Row', 'Flyer Row', 'row', 'Row', 'FlyerRow']
            col_col_candidates = ['Flyer_Column', 'Flyer Column', 'column', 'Column', 'FlyerColumn']
            row_col = self._find_parameter_column(df, row_col_candidates)
            col_col = self._find_parameter_column(df, col_col_candidates)
            
            if row_col is None or col_col is None:
                self.progress_signal.emit("⚠ Flyer Row or Column column not found - skipping pattern analysis plot")
                return
            
            if 'max_velocity_ms' not in df.columns:
                self.progress_signal.emit("⚠ Peak velocity column (max_velocity_ms) not found - skipping pattern analysis plot")
                return
            
            # Get material column
            material_col = None
            for col_name in df.columns:
                if 'material' in col_name.lower() and 'sample' in col_name.lower():
                    material_col = col_name
                    break
            if material_col is None:
                df['Material'] = 'Unknown'
                material_col = 'Material'
            
            subset_cols = [row_col, col_col, 'max_velocity_ms', laser_energy_col, material_col] + [c for c in ['aligned_ok'] if c in df.columns]
            valid_data = df[subset_cols].copy()
            valid_data = valid_data.dropna(subset=[row_col, col_col, 'max_velocity_ms', laser_energy_col])
            
            if 'aligned_ok' in valid_data.columns:
                valid_data = valid_data[valid_data['aligned_ok'] == True]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data for pattern analysis plot")
                return
            
            # Apply 2-sigma filtering to peak velocity
            velocity_values = valid_data['max_velocity_ms'].values
            velocity_mean = np.nanmean(velocity_values)
            velocity_std = np.nanstd(velocity_values)
            lower_bound = velocity_mean - 2 * velocity_std
            upper_bound = velocity_mean + 2 * velocity_std
            
            valid_data = valid_data[
                (valid_data['max_velocity_ms'] >= lower_bound) & 
                (valid_data['max_velocity_ms'] <= upper_bound)
            ].copy()
            
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                valid_data = valid_data[valid_data[material_col] != 'Unknown'].copy()
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid data after filtering - skipping pattern analysis plot")
                return
            
            # Ensure laser energy is numeric
            valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
            valid_data = valid_data[valid_data[laser_energy_col].notna()].copy()
            
            # Convert rows/columns to numeric
            row_numeric, _ = self._convert_row_column_to_numeric(valid_data[row_col])
            col_numeric, _ = self._convert_row_column_to_numeric(valid_data[col_col])
            valid_data['Row_numeric'] = row_numeric
            valid_data['Col_numeric'] = col_numeric
            valid_data = valid_data[valid_data['Row_numeric'].notna() & valid_data['Col_numeric'].notna()]
            
            # Create row/column pair labels
            def format_label(row_val, col_val, row_raw, col_raw):
                row_label = str(row_raw).replace('.0', '').strip()
                col_label = str(col_raw).replace('.0', '').strip()
                row_fmt = row_label if row_label.lower().startswith('r') else f"R{row_label}"
                col_fmt = col_label if col_label.lower().startswith('c') else f"C{col_label}"
                return f"{row_fmt}{col_fmt}"
            
            valid_data['RowCol_Label'] = valid_data.apply(
                lambda row: format_label(
                    row['Row_numeric'],
                    row['Col_numeric'],
                    row[row_col],
                    row[col_col]
                ),
                axis=1
            )
            
            # Get unique materials
            materials = sorted(valid_data[material_col].unique())
            colors = self._get_material_color_mapping(materials)
            energy_unit = 'mJ' if energy_in_mj else 'J'
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # ===== SUBPLOT 1: Laser Energy vs Peak Velocity =====
            ax1 = axes[0]
            for material in materials:
                # Filter by material (case-insensitive matching for robustness)
                material_mask = valid_data[material_col].astype(str).str.strip().str.lower() == str(material).strip().lower()
                material_data = valid_data[material_mask].copy()
                
                if len(material_data) == 0:
                    continue
                
                # Ensure numeric and remove any remaining NaN values
                material_data[laser_energy_col] = pd.to_numeric(material_data[laser_energy_col], errors='coerce')
                material_data['max_velocity_ms'] = pd.to_numeric(material_data['max_velocity_ms'], errors='coerce')
                
                # Remove NaN and invalid values
                material_data = material_data.dropna(subset=[laser_energy_col, 'max_velocity_ms'])
                material_data = material_data[
                    (material_data[laser_energy_col] > 0) & 
                    (material_data['max_velocity_ms'] > 0) &
                    (np.isfinite(material_data[laser_energy_col])) &
                    (np.isfinite(material_data['max_velocity_ms']))
                ].copy()
                
                if len(material_data) < 2:
                    # Not enough data points for regression
                    if len(material_data) > 0:
                        x = material_data[laser_energy_col].values
                        y = material_data['max_velocity_ms'].values
                        color = colors.get(material, 'gray')
                        ax1.scatter(x, y, c=[color], s=80, alpha=0.7, edgecolors='black', linewidths=0.5, 
                                  label=f"{material} (n={len(material_data)})")
                    continue
                
                x = material_data[laser_energy_col].values
                y = material_data['max_velocity_ms'].values
                color = colors.get(material, 'gray')
                
                # Scatter plot
                ax1.scatter(x, y, c=[color], s=80, alpha=0.7, edgecolors='black', linewidths=0.5, 
                          label=f"{material} (n={len(material_data)})")
            
            ax1.set_xlabel(f'Laser Energy ({energy_unit})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax1.set_title('Effect of Laser Energy on Peak Velocity', fontsize=16, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.legend(loc='best', fontsize=10)
            
            # ===== SUBPLOT 2: Test Location (RxCy) vs Peak Velocity =====
            ax2 = axes[1]
            
            # Build ordered pair list with zigzag pattern (same as other plot):
            # Odd rows (1, 3, 5): columns in reverse order (C5, C4, C3, C2, C1)
            # Even rows (2, 4): columns in normal order (C1, C2, C3, C4, C5)
            unique_rows = sorted(valid_data['Row_numeric'].unique())
            unique_cols = sorted(valid_data['Col_numeric'].unique())
            
            ordered_pairs = []
            for row_val in unique_rows:
                row_subset = valid_data[valid_data['Row_numeric'] == row_val]
                if row_subset.empty:
                    continue
                row_raw = row_subset[row_col].iloc[0]
                
                # Determine column order: reverse for odd rows, normal for even rows
                is_odd_row = (int(row_val) % 2 == 1)
                col_order = reversed(unique_cols) if is_odd_row else unique_cols
                
                for col_val in col_order:
                    col_subset = row_subset[row_subset['Col_numeric'] == col_val]
                    if col_subset.empty:
                        continue
                    col_raw = col_subset[col_col].iloc[0]
                    label = format_label(row_val, col_val, row_raw, col_raw)
                    
                    # Only add if this pair exists in the data
                    if label in valid_data['RowCol_Label'].values:
                        ordered_pairs.append(label)
            
            # Get unique row/column pairs that exist in data and calculate mean velocity for each
            unique_pairs = ordered_pairs  # Use zigzag-ordered pairs
            pair_means = []
            pair_stds = []
            pair_counts = []
            
            for pair in unique_pairs:
                pair_data = valid_data[valid_data['RowCol_Label'] == pair]
                if len(pair_data) > 0:
                    pair_means.append(pair_data['max_velocity_ms'].mean())
                    pair_stds.append(pair_data['max_velocity_ms'].std() if len(pair_data) > 1 else 0)
                    pair_counts.append(len(pair_data))
                else:
                    pair_means.append(np.nan)
                    pair_stds.append(0)
                    pair_counts.append(0)
            
            # Create bar plot with error bars
            x_pos = np.arange(len(unique_pairs))
            bars = ax2.bar(x_pos, pair_means, yerr=pair_stds, capsize=5, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Color bars by row (consistent color for all locations in the same row)
            # Extract row number from pair label (e.g., "R4C1" -> 4)
            for i, pair in enumerate(unique_pairs):
                pair_data = valid_data[valid_data['RowCol_Label'] == pair]
                if len(pair_data) > 0:
                    # Extract row number from the pair label
                    try:
                        # Parse row number from label like "R4C1" or "R4.0C1.0"
                        row_str = pair.split('C')[0].replace('R', '').replace('.0', '')
                        row_num = int(float(row_str))
                        
                        # Use a colormap to assign colors by row (consistent across all columns in same row)
                        # Use a simple color scheme: different shade for each row
                        row_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rows)))
                        row_idx = unique_rows.index(row_num) if row_num in unique_rows else 0
                        bars[i].set_facecolor(row_colors[row_idx])
                    except (ValueError, IndexError):
                        # Fallback: use most common material if row parsing fails
                        most_common_material = pair_data[material_col].mode()[0] if len(pair_data[material_col].mode()) > 0 else materials[0]
                        bars[i].set_facecolor(colors.get(most_common_material, 'gray'))
            
            ax2.set_xlabel('Test Location (Row/Column Pair)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Mean Peak Velocity (m/s)', fontsize=14, fontweight='bold')
            ax2.set_title('Effect of Test Location on Peak Velocity', fontsize=16, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(unique_pairs, rotation=45, ha='right', fontsize=9)
            ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
            
            # Add count annotations on bars
            for i, (bar, count) in enumerate(zip(bars, pair_counts)):
                if count > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + pair_stds[i] + 5,
                           f'n={count}', ha='center', va='bottom', fontsize=8)
            
            # Overall title
            fig.suptitle('Peak Velocity Pattern Analysis', fontsize=18, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            plot_path = os.path.join(spade_output_dir, 'peak_velocity_pattern_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.progress_signal.emit(f"✅ Generated Peak Velocity Pattern Analysis plot: {plot_path}")
        
        except Exception as e:
            self.progress_signal.emit(f"Error generating Peak Velocity Pattern Analysis plot: {str(e)}")
            import traceback
            self.progress_signal.emit(f"Traceback: {traceback.format_exc()}")

    def elastic_shock_strain_rate(self, C_L, U_hel, U_0, t_hel, t_0):
        """
        Compute elastic shock strain rate.

        Parameters
        ----------
        C_L : float
            Longitudinal wave velocity of the material (m/s).
        U_hel : float
            Free surface velocity at HEL (m/s).
        U_0 : float
            Free surface velocity at t = 0 (m/s).
        t_hel : float
            Time at which U_hel is measured (s).
        t_0 : float
            Initial time (s).

        Returns
        -------
        float
            Elastic shock strain rate (1/s).
        """
        dU = U_hel - U_0
        dt = t_hel - t_0
        
        if dt <= 0:
            return np.nan
        
        return (1 / (2 * C_L)) * (dU / dt)

    def _plot_individual_hel_detection(self, base_name, time_aligned, velocity_filtered,
                                       hel_start, hel_end, hel_time_clean, hel_velocity_clean,
                                       hel_strength, hel_uncertainty, sample_material,
                                       spade_output_dir, gradient=None, angles_deg=None,
                                       hel_segment_start=None, hel_segment_end=None,
                                       free_surface_velocity=None, angle_thresh_deg=None,
                                       U_0=None, t_0=None, t_hel=None):
        """Generate individual HEL detection plot showing detection results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Create figure with three subplots for gradient-based detection
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Top subplot: Full velocity trace with HEL window highlighted
        ax1.plot(time_aligned, velocity_filtered, 'b-', linewidth=1.5, alpha=0.7, label='Velocity')
        
        # Highlight HEL detection window
        ax1.axvspan(hel_start, hel_end, alpha=0.2, color='yellow', label='HEL Window')
        ax1.axvline(hel_start, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        ax1.axvline(hel_end, color='orange', linestyle='--', linewidth=1, alpha=0.7)
        
        ax1.set_xlabel('Time (ns)', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'HEL Detection - {base_name}\nMaterial: {sample_material}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Middle subplot: Zoomed HEL window with velocity and detection overlays
        ax2.plot(hel_time_clean, hel_velocity_clean, 'b-', linewidth=2, label='Velocity in HEL window')
        
        # Gradient-based detection: highlight HEL plateau region
        if hel_segment_start is not None and hel_segment_end is not None and free_surface_velocity is not None:
            plateau_start_time = hel_time_clean[hel_segment_start]
            plateau_end_time = hel_time_clean[hel_segment_end]
            ax2.axvspan(plateau_start_time, plateau_end_time, alpha=0.3, color='orange', 
                       label=f'HEL Plateau ({free_surface_velocity:.1f} m/s)')
            ax2.axhline(free_surface_velocity, color='orange', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'Mean Plateau Velocity: {free_surface_velocity:.1f} m/s')
            # Mark segment boundaries
            ax2.axvline(plateau_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.axvline(plateau_end_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title(f'HEL Window Detail - Velocity', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Bottom subplot: Gradient vs Time
        ax3.plot(hel_time_clean, gradient, 'g-', linewidth=1.5, alpha=0.6, label='Gradient (dv/dt)')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Highlight HEL plateau region in gradient plot
        if hel_segment_start is not None and hel_segment_end is not None:
            plateau_start_time = hel_time_clean[hel_segment_start]
            plateau_end_time = hel_time_clean[hel_segment_end]
            ax3.axvspan(plateau_start_time, plateau_end_time, alpha=0.3, color='orange', 
                       label='HEL Plateau Region')
            ax3.axvline(plateau_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax3.axvline(plateau_end_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Plot angle threshold line
        if angle_thresh_deg is not None:
            # Convert angle threshold to gradient (slope)
            angle_thresh_rad = np.radians(angle_thresh_deg)
            gradient_thresh = np.tan(angle_thresh_rad)
            ax3.axhline(gradient_thresh, color='red', linestyle='--', linewidth=1, alpha=0.7,
                       label=f'Angle Threshold ({angle_thresh_deg}°)')
            ax3.axhline(-gradient_thresh, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        ax3.set_xlabel('Time (ns)', fontsize=12)
        ax3.set_ylabel('Gradient (m/s per ns)', fontsize=12)
        ax3.set_title(f'Gradient vs Time - HEL Detection', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        
        # Add strain rate slope line if U_0, t_0, and t_hel are provided
        if (U_0 is not None and t_0 is not None and t_hel is not None and 
            free_surface_velocity is not None and
            np.isfinite(U_0) and np.isfinite(t_0) and np.isfinite(t_hel) and np.isfinite(free_surface_velocity)):
            # Calculate slope: dU/dt = (U_hel - U_0) / (t_hel - t_0)
            dU = free_surface_velocity - U_0
            dt = t_hel - t_0
            if dt > 0:
                slope = dU / dt  # m/s per ns
                # Draw line from (t_0, U_0) to (t_hel, U_hel)
                ax2.plot([t_0, t_hel], [U_0, free_surface_velocity], 
                        'r--', linewidth=2, alpha=0.8, 
                        label=f'Strain Rate Slope: {slope:.2e} m/s/ns')
                # Mark the points
                ax2.plot(t_0, U_0, 'go', markersize=8, label=f'U₀: {U_0:.1f} m/s @ {t_0:.1f} ns', zorder=5)
                ax2.plot(t_hel, free_surface_velocity, 'ro', markersize=8, 
                        label=f'U_HEL: {free_surface_velocity:.1f} m/s @ {t_hel:.1f} ns', zorder=5)
        
        # Add text box with HEL results (on velocity subplot)
        result_text = f'HEL Strength: {hel_strength:.3f} ± {hel_uncertainty:.3f} GPa'
        ax2.text(0.02, 0.98, result_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Create HEL_plots subfolder if plot_individual is enabled
        plot_individual_enabled = self.spade_params.get('plot_individual', False)
        if plot_individual_enabled:
            hel_plots_dir = os.path.join(spade_output_dir, 'HEL_plots')
            os.makedirs(hel_plots_dir, exist_ok=True)
            plot_dir = hel_plots_dir
        else:
            plot_dir = spade_output_dir
        
        # Save plot in appropriate folder
        plot_filename = f'{base_name}--hel_detection.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.progress_signal.emit(f"  Saved HEL plot: {plot_filename}")

    def generate_combined_velocity_plot(self, velocity_plot_data, spade_output_dir):
        """Generate combined velocity plot with color coding based on material information"""
        self.progress_signal.emit("Generating combined velocity plot...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Helper to extract material from parameter info using explicit 'Sample material' column
            def extract_material(param_info_dict):
                if not param_info_dict or not isinstance(param_info_dict, dict):
                    return 'Unknown'
                # Normalize keys once
                normalized = {}
                for k, v in param_info_dict.items():
                    key_norm = ''.join(ch for ch in k.lower() if ch.isalnum())
                    normalized[key_norm] = v
                # Preferred explicit columns
                preferred_keys = [
                    'samplematerial', 'samplemat', 'sample', 'material', 'flyermaterial'
                ]
                for pk in preferred_keys:
                    if pk in normalized and str(normalized[pk]).strip() != '':
                        return str(normalized[pk]).strip()
                # Fallback: scan values for common material tokens
                for v in param_info_dict.values():
                    val = str(v).strip()
                    if not val:
                        continue
                    low = val.lower()
                    if any(tok in low for tok in ['al', 'aluminum', 'aluminium']):
                        return val
                    if any(tok in low for tok in ['cu', 'copper']):
                        return val
                    if any(tok in low for tok in ['ti', 'titanium']):
                        return val
                    if 'steel' in low:
                        return val
                return 'Unknown'

            # Build list of (time, velocity, material)
            plotted = []
            for plot_data in velocity_plot_data:
                time_data = plot_data['time_ns']
                velocity_data = plot_data['velocity_ms']
                param_info = plot_data.get('param_info')
                material = extract_material(param_info)
                plotted.append((time_data, velocity_data, material))

            # Assign colors per unique material using a colormap
            unique_materials = []
            seen = set()
            for _t, _v, m in plotted:
                if m not in seen:
                    seen.add(m)
                    unique_materials.append(m)
            cmap = plt.get_cmap('Set3') if len(unique_materials) <= 12 else plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(unique_materials))))
            material_to_color = {m: colors[i] for i, m in enumerate(unique_materials)}

            # Plot traces grouped by material color
            for time_data, velocity_data, material in plotted:
                color = material_to_color.get(material, 'gray')
                ax.plot(time_data, velocity_data, color=color, alpha=0.7, linewidth=1)
            
            # Customize plot
            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            ax.set_xlabel(f'Time (ns) - Aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title('Combined Velocity Traces - Aligned (Color-coded by Material)', fontsize=14)
            ax.grid(True, alpha=0.3)

            # Apply axis limits from SPADE params if not auto
            try:
                auto_calc = self.spade_params.get('auto_calculate_limits', True)
                self.progress_signal.emit(f"Combined plot - auto_calculate_limits: {auto_calc}")
                
                if not auto_calc:
                    x_min = float(self.spade_params.get('x_min_main', 0))
                    x_max = float(self.spade_params.get('x_max_main', 100))
                    y_min = float(self.spade_params.get('y_min_main', 0))
                    y_max = float(self.spade_params.get('y_max_main', 600))
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    self.progress_signal.emit(f"Applied axis limits: X({x_min}-{x_max}), Y({y_min}-{y_max})")
                else:
                    self.progress_signal.emit("Using auto-calculated axis limits")
            except Exception as e:
                self.progress_signal.emit(f"Error applying axis limits: {str(e)}")
                pass
            
            # Create legend with material colors
            legend_elements = [mpatches.Patch(color=material_to_color[m], label=m) for m in unique_materials]
            
            ax.legend(handles=legend_elements, title='Material', loc='upper right')
            
            # Save plot
            plot_path = os.path.join(spade_output_dir, "combined_velocity_plot.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress_signal.emit(f"Saved combined velocity plot to {plot_path}")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating combined velocity plot: {e}")

    def generate_all_velocity_traces_plot(self, input_path, spade_output_dir, uncertainty_threshold, unaligned_basenames=None):
        """Generate an all-traces plot aligned at 30 m/s using ALPSS output files in input_path.
        Applies noise fraction filtering (>1) and removes points with uncertainty > threshold.
        Color-codes traces by sample material from parameter files.
        Skips traces that never aligned (t0 not found) and saves PNG to both main output_dir and SPADE output dir."""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import re

            skip_unaligned = set(unaligned_basenames or [])

            pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            if not files:
                self.progress_signal.emit("No '*--vel-smooth-with-uncert.csv' files found for all-traces plot")
                return
            
            # Initialize counters for this plot
            # We'll count actual traces processed, not just files found
            traces_plotted_local = 0
            traces_rejected_local = 0
            traces_skipped_initial = 0  # Traces skipped before processing loop
            rejection_reasons_local = {}

            # Helper function to extract material from parameter info
            def extract_material(param_info_dict):
                if not param_info_dict or not isinstance(param_info_dict, dict):
                    return 'Unknown'
                # Normalize keys once
                normalized = {}
                for k, v in param_info_dict.items():
                    key_norm = ''.join(ch for ch in k.lower() if ch.isalnum())
                    normalized[key_norm] = v
                # Preferred explicit columns
                preferred_keys = [
                    'samplematerial', 'samplemat', 'sample', 'material', 'flyermaterial'
                ]
                for pk in preferred_keys:
                    if pk in normalized and str(normalized[pk]).strip() != '':
                        return str(normalized[pk]).strip()
                # Fallback: scan values for common material tokens
                for v in param_info_dict.values():
                    val = str(v).strip()
                    if not val:
                        continue
                    low = val.lower()
                    if any(tok in low for tok in ['al', 'aluminum', 'aluminium']):
                        return val
                    if any(tok in low for tok in ['cu', 'copper']):
                        return val
                    if any(tok in low for tok in ['ti', 'titanium']):
                        return val
                    if 'steel' in low:
                        return val
                return 'Unknown'

            # Collect traces with their materials
            trace_data = []
            for file_path in sorted(files):
                try:
                    # Extract base filename for parameter matching
                    base_filename = os.path.basename(file_path)
                    
                    # Extract C1--XXXXXXXX--XXYYY pattern from filename
                    # Pattern: C1--YYYYMMDD--NNNNN (e.g., C1--20251023--00924)
                    pattern_match = re.search(r'(C\d+--\d{8}--\d{5})', base_filename)
                    if pattern_match:
                        pdv_filename_pattern = pattern_match.group(1)
                    else:
                        # Fallback: remove suffixes like '--vel-smooth-with-uncert.csv'
                        pdv_filename_pattern = re.sub(r'--.*$', '', base_filename)
                        pdv_filename_pattern = os.path.splitext(pdv_filename_pattern)[0]
                    
                    if skip_unaligned and pdv_filename_pattern in skip_unaligned:
                        traces_skipped_initial += 1
                        reason = 'Unaligned trace (from SPADE)'
                        rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                        self.progress_signal.emit(f"Skipping {pdv_filename_pattern} in all-traces plot (unaligned trace)")
                        continue

                    # Get parameter data for this file by matching PDV_FileName
                    param_info = {}
                    material = 'Unknown'
                    
                    if hasattr(self, 'param_data') and self.param_data:
                        # Try exact match first with the extracted pattern
                        if pdv_filename_pattern in self.param_data:
                            param_info = self.param_data[pdv_filename_pattern]
                            self.progress_signal.emit(f"✓ Matched {pdv_filename_pattern} to parameter file")
                        else:
                            # Try matching with get_param_data_for_file as fallback
                            param_info = self.get_param_data_for_file(pdv_filename_pattern)
                            if param_info:
                                self.progress_signal.emit(f"✓ Matched {pdv_filename_pattern} via fallback matching")
                            else:
                                self.progress_signal.emit(f"⚠️  No parameter match for {pdv_filename_pattern}")
                        
                        # Extract material from "Sample material" column
                        if param_info:
                            # Try various column name variations for sample material
                            material_keys = [
                                'Sample material', 'Sample Material', 'Sample_Material',
                                'sample_material', 'samplematerial', 'SampleMaterial',
                                'Material', 'material', 'Flyer_material', 'Flyer Material'
                            ]
                            for key in material_keys:
                                if key in param_info:
                                    material_val = str(param_info[key]).strip()
                                    if material_val and material_val.lower() != 'nan':
                                        material = material_val
                                        self.progress_signal.emit(f"  Found material '{material}' from column '{key}'")
                                        break
                    
                    # If still Unknown, try the extract_material helper (but don't use it if we found a match)
                    if material == 'Unknown' and param_info:
                        material = extract_material(param_info)
                        if material != 'Unknown':
                            self.progress_signal.emit(f"  Found material '{material}' via extract_material helper")
                    
                    self.progress_signal.emit(f"File: {base_filename} -> Pattern: {pdv_filename_pattern} -> Material: {material}")
                    
                    trace_data.append({
                        'file_path': file_path,
                        'base_name': pdv_filename_pattern,
                        'material': material
                    })
                except Exception as e:
                    traces_skipped_initial += 1
                    reason = f'Error in initial processing: {str(e)[:50]}'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Error processing {file_path}: {e}")
                    import traceback
                    self.progress_signal.emit(traceback.format_exc())
                    continue

            # Group by material and assign colors
            unique_materials = []
            seen = set()
            material_counts = {}
            for trace in trace_data:
                mat = trace['material']
                material_counts[mat] = material_counts.get(mat, 0) + 1
                if mat not in seen:
                    seen.add(mat)
                    unique_materials.append(mat)
            
            # Report material grouping summary
            self.progress_signal.emit(f"\n=== Material Grouping Summary ===")
            self.progress_signal.emit(f"Found {len(unique_materials)} unique material(s):")
            for mat in sorted(unique_materials):
                count = material_counts[mat]
                self.progress_signal.emit(f"  {mat}: {count} trace(s)")
            self.progress_signal.emit("=" * 40 + "\n")
            
            # Use Set3 colormap for <=12 materials, tab20 for more
            if len(unique_materials) <= 12:
                cmap = plt.get_cmap('Set3')
            else:
                cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(unique_materials))))
            material_to_color = {mat: colors[i] for i, mat in enumerate(unique_materials)}

            # Use fixed 2920 x 1824 px per subplot (≈1.6:1 aspect ratio)
            # Total figure: 2920 x 3648 px (2 subplots stacked vertically)
            target_width_px = 2920
            target_height_px = 3648  # 1824 * 2 for two subplots
            target_dpi = 300
            fig_width_in = target_width_px / target_dpi
            fig_height_in = target_height_px / target_dpi
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(fig_width_in, fig_height_in),
                dpi=target_dpi,
            )

            # Option to skip Unknown material traces
            skip_unknown = self._should_skip_unknown_materials()
            if skip_unknown:
                has_labeled_material = any(trace['material'] != 'Unknown' for trace in trace_data)
                if not has_labeled_material:
                    skip_unknown = False
                    self.progress_signal.emit("⚠️  No parameter data matched current files; rendering 'Unknown' traces to avoid empty combined plot.")
            
            # List to track IQ detection failures
            iq_detection_failures = []

            traces_plotted = 0
            for trace in trace_data:
                file_path = trace['file_path']
                material = trace['material']
                
                # Skip Unknown material traces if requested
                if skip_unknown and material == 'Unknown':
                    traces_rejected_local += 1
                    reason = 'Unknown material'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Skipping {trace['base_name']} in all-traces plot (Unknown material)")
                    continue
                
                color = material_to_color.get(material, 'gray')
                
                try:
                    # Check IQ detection start time - skip if too close to end of time window
                    results_file = file_path.replace('--vel-smooth-with-uncert.csv', '--results.csv')
                    if os.path.exists(results_file):
                        try:
                            # Read ALPSS results - saved without header, has Name and Value columns
                            results_df = pd.read_csv(results_file, header=None, names=['Name', 'Value'])
                            
                            # Find Signal Start Time row
                            signal_start_time_s = None
                            signal_start_row = results_df[results_df['Name'] == 'Signal Start Time']
                            if not signal_start_row.empty:
                                signal_start_time_s = signal_start_row['Value'].iloc[0]
                                # Convert to float if it's a string
                                try:
                                    signal_start_time_s = float(signal_start_time_s)
                                except (ValueError, TypeError):
                                    signal_start_time_s = None
                            
                            if signal_start_time_s is not None and not pd.isna(signal_start_time_s):
                                # Get time_to_take from ALPSS config
                                time_to_take_s = self.alpss_params.get('time_to_take', 1e-6)
                                time_to_skip_s = self.alpss_params.get('time_to_skip', 2e-6)
                                
                                # Calculate end of time window (absolute time from file start)
                                time_window_end_s = time_to_skip_s + time_to_take_s
                                
                                # Check if start time is too close to end (within 5% of time_to_take)
                                threshold_fraction = 0.05  # 5% of time window
                                min_required_time_after_start = time_to_take_s * threshold_fraction
                                
                                # Calculate time available after signal start
                                time_after_start = time_window_end_s - signal_start_time_s
                                
                                if time_after_start < min_required_time_after_start:
                                    failure_info = {
                                        'base_name': trace['base_name'],
                                        'material': material,
                                        'signal_start_time_us': signal_start_time_s * 1e6,
                                        'time_window_end_us': time_window_end_s * 1e6,
                                        'time_after_start_us': time_after_start * 1e6,
                                        'min_required_us': min_required_time_after_start * 1e6,
                                        'reason': 'IQ detection failed: start time too close to end of time window'
                                    }
                                    iq_detection_failures.append(failure_info)
                                    traces_rejected_local += 1
                                    reason = 'IQ detection failed'
                                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                                    self.progress_signal.emit(
                                        f"Skipping {trace['base_name']} in all-traces plot "
                                        f"(IQ detection failed: start time {signal_start_time_s*1e6:.2f} μs too close to end "
                                        f"{time_window_end_s*1e6:.2f} μs, only {time_after_start*1e6:.2f} μs after start, "
                                        f"need at least {min_required_time_after_start*1e6:.2f} μs)")
                                    continue
                        except Exception as e:
                            # If we can't read results, continue anyway (don't fail on this check)
                            # Uncomment for debugging: self.progress_signal.emit(f"Warning: Could not read results for {trace['base_name']}: {e}")
                            pass
                    
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    uncertainty_data = df.iloc[:, 2].values

                    # Convert time to ns if likely in s/us
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3

                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass

                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    # Uncertainty threshold filtering
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)

                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                    if len(time_clean) == 0:
                        traces_rejected_local += 1
                        reason = 'No valid data after filtering'
                        rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                        continue

                    # Get alignment threshold from config file and check if trace crosses threshold from below
                    # This threshold is defined by the user in helix_master_config.json as "align_velocity_threshold_ms"
                    align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
                    tolerance = 0.01  # 0.01 m/s tolerance for floating point comparison
                    
                    # Find the first point where velocity reaches or exceeds threshold
                    t0_idx = None
                    for j, v in enumerate(velocity_clean):
                        if not np.isnan(v) and v >= (align_threshold - tolerance):
                            t0_idx = j
                            break
                    
                    fallback_reason = None
                    if t0_idx is None:
                        # Fall back to earliest valid point
                        t0_idx = 0
                        max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                        min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                        fallback_reason = (
                            f"No threshold crossing at {align_threshold} m/s "
                            f"(velocity range {min_vel:.1f}-{max_vel:.1f} m/s); aligning to earliest valid point."
                        )
                    else:
                        has_point_below = any(
                            (not np.isnan(velocity_clean[j])) and velocity_clean[j] < (align_threshold - tolerance)
                            for j in range(max(t0_idx, 1))
                        )
                        if not has_point_below:
                            fallback_reason = (
                                f"Trace starts at/above {align_threshold} m/s; aligning to first available sample."
                            )
                            # Prefer earliest point < threshold if it exists elsewhere
                            below_candidates = np.where(~np.isnan(velocity_clean) & (velocity_clean < (align_threshold - tolerance)))[0]
                            if len(below_candidates) > 0:
                                t0_idx = below_candidates[0]
                    
                    if fallback_reason:
                        self.progress_signal.emit(f"[Combined Plot] {trace['base_name']}: {fallback_reason}")
                    
                    # Align the trace
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0

                    # Verify alignment: check that t=0 is actually at the alignment point
                    if abs(time_clean[t0_idx]) > 1e-6:  # Should be very close to 0 after alignment
                        self.progress_signal.emit(f"Warning: {trace['base_name']} alignment may be incorrect (t0={time_clean[t0_idx]:.3f} ns)")

                    ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1.5, label=material if traces_plotted == 0 or material not in [t['material'] for t in trace_data[:traces_plotted]] else '')
                    # Optional uncertainty bands
                    if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None and len(uncert_clean) == len(velocity_clean):
                        alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                        ax1.fill_between(time_clean,
                                         velocity_clean - uncert_clean,
                                         velocity_clean + uncert_clean,
                                         color=color, alpha=alpha)

                    # Bottom zoom window
                    zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                    mask_1000 = time_clean <= zoom_ns
                    if np.any(mask_1000):
                        ax2.plot(time_clean[mask_1000], velocity_clean[mask_1000], color=color, alpha=0.7, linewidth=1.5)
                        if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None and len(uncert_clean) == len(velocity_clean):
                            alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                            ax2.fill_between(time_clean[mask_1000],
                                             (velocity_clean - uncert_clean)[mask_1000],
                                             (velocity_clean + uncert_clean)[mask_1000],
                                             color=color, alpha=alpha)

                    traces_plotted += 1
                    traces_plotted_local += 1
                except Exception as e:
                    traces_rejected_local += 1
                    reason = f'Processing error: {str(e)[:50]}'
                    rejection_reasons_local[reason] = rejection_reasons_local.get(reason, 0) + 1
                    self.progress_signal.emit(f"Error processing {trace['base_name']} for all-traces plot: {str(e)}")
                    import traceback
                    self.progress_signal.emit(traceback.format_exc())
                    continue

            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax1.set_ylabel('Velocity (m/s)', fontsize=12)
            ax1.set_title(f'All Velocity Traces (Aligned, Color by Material) - {traces_plotted} traces', fontsize=14)
            ax1.grid(False)
            
            # Add legend for materials (only show unique materials)
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                # Remove duplicates while preserving order
                seen_labels = set()
                unique_handles = []
                unique_labels = []
                for h, l in zip(handles, labels):
                    if l not in seen_labels:
                        seen_labels.add(l)
                        unique_handles.append(h)
                        unique_labels.append(l)
                # Set legend line width to 2
                for handle in unique_handles:
                    handle.set_linewidth(2)
                ax1.legend(unique_handles, unique_labels, loc='best', fontsize=12, title='Sample Material', title_fontsize=13)

            ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.grid(False)

            # Apply axis limits
            try:
                if not self.spade_params.get('auto_calculate_limits', True):
                    x_min_main = float(self.spade_params.get('x_min_main', 0))
                    x_max_main = float(self.spade_params.get('x_max_main', 100))
                    y_min_main = float(self.spade_params.get('y_min_main', 0))
                    y_max_main = float(self.spade_params.get('y_max_main', 600))
                    # Top subplot should match combined plot main limits
                    ax1.set_xlim(x_min_main, x_max_main)
                    ax1.set_ylim(y_min_main, y_max_main)

                    x_min_zoom = float(self.spade_params.get('x_min_zoom', 0))
                    x_max_zoom = float(self.spade_params.get('x_max_zoom', self.spade_params.get('zoom_window_ns', 1000)))
                    y_min_zoom = float(self.spade_params.get('y_min_zoom', y_min_main))
                    y_max_zoom = float(self.spade_params.get('y_max_zoom', y_max_main))
                    ax2.set_xlim(x_min_zoom, x_max_zoom)
                    ax2.set_ylim(y_min_zoom, y_max_zoom)
                    ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)} to {int(x_max_zoom)} ns)', fontsize=14)
                else:
                    # Default zoom: 0 to zoom_window_ns
                    zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                    ax2.set_xlim(0, zoom_ns)
                    ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)
            except Exception:
                # Fallback to default behavior
                zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)

            plt.tight_layout()

            # Save to main output_dir and SPADE output dir
            out_main = os.path.join(self.output_dir, 'all_velocity_traces.png')
            out_spade = os.path.join(spade_output_dir, 'all_velocity_traces.png')
            try:
                fig.savefig(out_main, dpi=target_dpi, bbox_inches='tight')
            except Exception:
                pass
            try:
                fig.savefig(out_spade, dpi=target_dpi, bbox_inches='tight')
            except Exception:
                pass
            plt.close(fig)

            # Save IQ detection failures list
            if iq_detection_failures:
                failure_file = os.path.join(spade_output_dir, 'IQ_based_detection_failure.csv')
                try:
                    failure_df = pd.DataFrame(iq_detection_failures)
                    failure_df.to_csv(failure_file, index=False)
                    self.progress_signal.emit(f"Saved IQ detection failure list: {failure_file} ({len(iq_detection_failures)} traces)")
                except Exception as e:
                    self.progress_signal.emit(f"Error saving IQ detection failure list: {e}")

            # Update global counters
            # Total input traces = files found (some may be skipped before processing)
            self.total_input_traces = len(files)
            self.traces_plotted = traces_plotted_local
            # Total rejected = rejected during processing + skipped in initial loop
            self.traces_rejected = traces_rejected_local + traces_skipped_initial
            self.rejection_reasons = rejection_reasons_local

            self.progress_signal.emit(f"Saved aligned all-traces velocity plot to: {out_spade}")
        except Exception as e:
            self.progress_signal.emit(f"Error generating all-traces velocity plot: {e}")

    def generate_zn_traces_diagnostic_plot(self, input_path, spade_output_dir, uncertainty_threshold):
        """Temporary diagnostic plot showing only Zn traces with labels (last 5 digits of filename)"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import re

            pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            if not files:
                self.progress_signal.emit("No '*--vel-smooth-with-uncert.csv' files found for Zn diagnostic plot")
                return

            # Collect Zn traces
            zn_traces = []
            for file_path in sorted(files):
                try:
                    base_filename = os.path.basename(file_path)
                    
                    # Extract C1--XXXXXXXX--XXYYY pattern from filename
                    pattern_match = re.search(r'(C\d+--\d{8}--\d{5})', base_filename)
                    if pattern_match:
                        pdv_filename_pattern = pattern_match.group(1)
                    else:
                        pdv_filename_pattern = re.sub(r'--.*$', '', base_filename)
                        pdv_filename_pattern = os.path.splitext(pdv_filename_pattern)[0]
                    
                    # Extract last 5 digits from filename
                    last_5_digits = pdv_filename_pattern[-5:] if len(pdv_filename_pattern) >= 5 else pdv_filename_pattern
                    
                    # Get material
                    material = 'Unknown'
                    if hasattr(self, 'param_data') and self.param_data:
                        if pdv_filename_pattern in self.param_data:
                            param_info = self.param_data[pdv_filename_pattern]
                            material_keys = [
                                'Sample material', 'Sample Material', 'Sample_Material',
                                'sample_material', 'samplematerial', 'SampleMaterial',
                                'Material', 'material', 'Flyer_material', 'Flyer Material'
                            ]
                            for key in material_keys:
                                if key in param_info:
                                    material_val = str(param_info[key]).strip()
                                    if material_val and material_val.lower() != 'nan':
                                        material = material_val
                                        break
                    
                    # Only include Zn traces
                    if material.lower() in ['zn', 'zinc']:
                        zn_traces.append({
                            'file_path': file_path,
                            'base_name': pdv_filename_pattern,
                            'label': last_5_digits,
                            'material': material
                        })
                except Exception as e:
                    continue
            
            if len(zn_traces) == 0:
                self.progress_signal.emit("No Zn traces found for diagnostic plot")
                return
            
            self.progress_signal.emit(f"Found {len(zn_traces)} Zn traces for diagnostic plot")
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            traces_plotted = 0
            
            for trace in zn_traces:
                file_path = trace['file_path']
                label = trace['label']
                
                try:
                    # Check IQ detection start time - skip if too close to end of time window
                    results_file = file_path.replace('--vel-smooth-with-uncert.csv', '--results.csv')
                    if os.path.exists(results_file):
                        try:
                            # Read ALPSS results - saved without header, has Name and Value columns
                            results_df = pd.read_csv(results_file, header=None, names=['Name', 'Value'])
                            
                            # Find Signal Start Time row
                            signal_start_time_s = None
                            signal_start_row = results_df[results_df['Name'] == 'Signal Start Time']
                            if not signal_start_row.empty:
                                signal_start_time_s = signal_start_row['Value'].iloc[0]
                                # Convert to float if it's a string
                                try:
                                    signal_start_time_s = float(signal_start_time_s)
                                except (ValueError, TypeError):
                                    signal_start_time_s = None
                            
                            if signal_start_time_s is not None and not pd.isna(signal_start_time_s):
                                time_to_take_s = self.alpss_params.get('time_to_take', 1e-6)
                                time_to_skip_s = self.alpss_params.get('time_to_skip', 2e-6)
                                time_window_end_s = time_to_skip_s + time_to_take_s
                                threshold_fraction = 0.05  # 5% of time window
                                min_required_time_after_start = time_to_take_s * threshold_fraction
                                time_after_start = time_window_end_s - signal_start_time_s
                                
                                if time_after_start < min_required_time_after_start:
                                    self.progress_signal.emit(
                                        f"Skipping {trace['base_name']} (IQ detection failed: start time {signal_start_time_s*1e6:.2f} μs too close to end)")
                                    continue
                        except Exception:
                            pass
                    
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    uncertainty_data = df.iloc[:, 2].values

                    # Convert time to ns if likely in s/us
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3

                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass

                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)

                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    if len(time_clean) == 0:
                        continue

                    # Get alignment threshold and check if trace crosses threshold from below
                    # This threshold is defined by the user in helix_master_config.json as "align_velocity_threshold_ms"
                    tolerance = 0.01  # 0.01 m/s tolerance for floating point comparison
                    
                    # Find the first point where velocity reaches or exceeds threshold
                    # This must be a rising crossing (trace starts below threshold and crosses it)
                    t0_idx = None
                    for j, v in enumerate(velocity_clean):
                        if not np.isnan(v) and v >= (align_threshold - tolerance):
                            t0_idx = j
                            break
                    
                    # Check if trace crosses threshold from below (not starting above and decreasing)
                    if t0_idx is None or t0_idx == 0:
                        # No threshold crossing found, or trace starts at threshold (likely bad data)
                        max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                        min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                        self.progress_signal.emit(f"Skipping {trace['base_name']} (no threshold crossing from below at {align_threshold} m/s, velocity range: {min_vel:.1f} to {max_vel:.1f} m/s)")
                        continue
                    
                    # Verify that trace started below threshold (crossing from below)
                    # Check if there's at least one point before t0_idx that's below threshold
                    has_point_below = False
                    for j in range(t0_idx):
                        if not np.isnan(velocity_clean[j]) and velocity_clean[j] < (align_threshold - tolerance):
                            has_point_below = True
                            break
                    
                    if not has_point_below:
                        # Trace starts at or above threshold - likely bad data (e.g., IQ detection failed, starts late)
                        max_vel = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
                        min_vel = np.nanmin(velocity_clean) if len(velocity_clean) > 0 else 0
                        self.progress_signal.emit(f"Skipping {trace['base_name']} (trace starts at/above threshold {align_threshold} m/s, does not cross from below, velocity range: {min_vel:.1f} to {max_vel:.1f} m/s)")
                        continue
                    
                    # Align the trace
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0
                    
                    # Filter negative time if requested
                    filter_negative_time = self.spade_params.get('filter_negative_time', False)
                    if filter_negative_time:
                        mask_t_positive = time_clean >= 0
                        time_clean = time_clean[mask_t_positive]
                        velocity_clean = velocity_clean[mask_t_positive]
                        if len(time_clean) == 0:
                            continue
                    
                    # Plot with label
                    ax.plot(time_clean, velocity_clean, alpha=0.7, linewidth=1.5, label=label)
                    
                    # Add text label at peak or end of trace
                    if len(velocity_clean) > 0:
                        peak_idx = np.nanargmax(velocity_clean)
                        if peak_idx < len(time_clean) and peak_idx < len(velocity_clean):
                            ax.text(time_clean[peak_idx], velocity_clean[peak_idx], 
                                   f' {label}', fontsize=8, alpha=0.8, 
                                   verticalalignment='bottom')
                    
                    traces_plotted += 1
                except Exception as e:
                    self.progress_signal.emit(f"Error processing {trace['base_name']}: {e}")
                    continue

            ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax.set_ylabel('Velocity (m/s)', fontsize=12)
            ax.set_title(f'Zn Traces Diagnostic Plot - {traces_plotted} traces (labeled with last 5 digits)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8, ncol=3)
            
            plt.tight_layout()

            # Save to SPADE output dir
            out_path = os.path.join(spade_output_dir, 'zn_traces_diagnostic.png')
            try:
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                self.progress_signal.emit(f"Saved Zn diagnostic plot to: {out_path}")
            except Exception as e:
                self.progress_signal.emit(f"Error saving Zn diagnostic plot: {e}")
            plt.close(fig)

        except Exception as e:
            self.progress_signal.emit(f"Error generating Zn diagnostic plot: {e}")
            import traceback
            self.progress_signal.emit(traceback.format_exc())

    def generate_spall_analysis_summary(self, spade_output_dir):
        """Generate enhanced spall analysis summary with comprehensive parameter file data"""
        self.progress_signal.emit("Generating enhanced spall analysis summary...")

        # Check if spall summary already exists
        spall_summary_path = os.path.join(spade_output_dir, 'spall_summary.csv')
        if not os.path.exists(spall_summary_path):
            self.progress_signal.emit("No spall summary found - SPADE analysis may not have completed")
            return

        # Read existing spall summary
        spall_df = pd.read_csv(spall_summary_path)
        self.progress_signal.emit(f"Found {len(spall_df)} entries in spall summary")

        # Enhance with parameter file data
        enhanced_spall_data = []

        for idx, row in spall_df.iterrows():
            filename = row.get('Filename', '')
            if not filename or filename == 'data' or not isinstance(filename, str):
                # Skip invalid filenames
                if filename:
                    self.progress_signal.emit(f"Skipping invalid filename: {repr(filename)}")
                continue

            # Get file base name for parameter matching
            # Handle cases where filename might be a full path or just a name
            base_name = os.path.splitext(os.path.basename(str(filename)))[0]

            # Skip if base_name is still invalid
            if not base_name or base_name == 'data':
                self.progress_signal.emit(f"Skipping invalid base_name from filename: {repr(filename)}")
                continue

            # Get parameter data if available using helper function
            param_info = self.get_param_data_for_file(base_name)

            # Debug parameter data
            if param_info:
                self.progress_signal.emit(f"Found parameter data for {base_name}: {list(param_info.keys())}")
            else:
                self.progress_signal.emit(f"No parameter data found for {base_name}")

            # Create enhanced row with all original SPADE data
            enhanced_row = row.copy()

            # Add parameter file data as extra columns (preserve original names only)
            for key, value in param_info.items():
                enhanced_row[key] = value

            # Try to find corresponding ALPSS results file for additional data
            # Search both current output_dir and standard ALPSS output location
            candidate_results = [
                os.path.join(self.output_dir, f"{base_name}--results.csv"),
                os.path.join("ALPSS", "output_data", f"{base_name}--results.csv"),
                os.path.join("output", f"{base_name}--results.csv"),
            ]
            alpss_results_file = next((p for p in candidate_results if os.path.exists(p)), None)
            if alpss_results_file and os.path.exists(alpss_results_file):
                try:
                    # Read ALPSS results
                    alpss_results = pd.read_csv(alpss_results_file, header=None, names=['Name', 'Value'])
                    alpss_dict = dict(zip(alpss_results['Name'], alpss_results['Value']))
                    
                    # Add ALPSS results to enhanced summary
                    enhanced_row['ALPSS_Spall_Strength_GPa'] = alpss_dict.get('Spall Strength', np.nan)
                    enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = alpss_dict.get('Spall Strength Uncertainty', np.nan)
                    enhanced_row['ALPSS_Strain_Rate_s1'] = alpss_dict.get('Strain Rate', np.nan)
                    enhanced_row['ALPSS_Strain_Rate_Uncertainty_s1'] = alpss_dict.get('Strain Rate Uncertainty', np.nan)
                    enhanced_row['ALPSS_Peak_Shock_Stress_GPa'] = alpss_dict.get('Peak Shock Stress', np.nan)
                    enhanced_row['ALPSS_Velocity_at_Max_Compression_ms'] = alpss_dict.get('Velocity at Max Compression', np.nan)
                    enhanced_row['ALPSS_Velocity_at_Max_Tension_ms'] = alpss_dict.get('Velocity at Max Tension', np.nan)
                    enhanced_row['ALPSS_Velocity_at_Recompression_ms'] = alpss_dict.get('Velocity at Recompression', np.nan)
                    enhanced_row['ALPSS_Time_at_Max_Compression_ns'] = alpss_dict.get('Time at Max Compression', np.nan)
                    enhanced_row['ALPSS_Time_at_Max_Tension_ns'] = alpss_dict.get('Time at Max Tension', np.nan)
                    enhanced_row['ALPSS_Time_at_Recompression_ns'] = alpss_dict.get('Time at Recompression', np.nan)
                    enhanced_row['ALPSS_Carrier_Frequency_Hz'] = alpss_dict.get('Carrier Frequency', np.nan)
                    enhanced_row['ALPSS_Signal_Start_Time_s'] = alpss_dict.get('Signal Start Time', np.nan)
                    enhanced_row['ALPSS_Smoothing_Characteristic_Time_s'] = alpss_dict.get('Smoothing Characteristic Time', np.nan)
                    
                    self.progress_signal.emit(f"Added ALPSS results for {base_name}")
                except Exception as e:
                    self.progress_signal.emit(f"Warning: Could not read ALPSS results for {base_name}: {e}")
            else:
                self.progress_signal.emit(f"No ALPSS results file found for {base_name}")

            enhanced_spall_data.append(enhanced_row)

        # Save enhanced spall summary (include all parameter file columns, drop redundant columns)
        if enhanced_spall_data:
            enhanced_spall_df = pd.DataFrame(enhanced_spall_data)

            # Ensure all parameter file columns are present in the output
            all_param_columns = set()
            if self.param_data:
                try:
                    for _key, param_dict in self.param_data.items():
                        if isinstance(param_dict, dict):
                            all_param_columns.update(param_dict.keys())
                except Exception:
                    pass

            # Add any missing param columns with NaN
            for col in sorted(all_param_columns):
                if col not in enhanced_spall_df.columns:
                    enhanced_spall_df[col] = np.nan

            # Remove redundant columns
            try:
                import re
                # 1) Drop columns that normalize to the same token (keep first occurrence)
                seen_norm = set()
                cols_to_drop = []
                for col in enhanced_spall_df.columns:
                    norm = re.sub(r"[^a-zA-Z0-9]+", "_", str(col)).strip("_").lower()
                    if norm in seen_norm:
                        cols_to_drop.append(col)
                    else:
                        seen_norm.add(norm)
                if cols_to_drop:
                    enhanced_spall_df = enhanced_spall_df.drop(columns=cols_to_drop, errors='ignore')

                # 2) Drop exact-duplicate columns (identical values across all rows)
                enhanced_spall_df = enhanced_spall_df.T.drop_duplicates().T

                # 3) Optionally drop columns that are entirely NaN
                enhanced_spall_df = enhanced_spall_df.dropna(axis=1, how='all')
            except Exception:
                pass

            enhanced_spall_path = os.path.join(spade_output_dir, 'enhanced_spall_summary.csv')
            enhanced_spall_df.to_csv(enhanced_spall_path, index=False)
            self.progress_signal.emit(f"Generated enhanced spall summary with {len(enhanced_spall_data)} entries")
            self.progress_signal.emit(f"Saved to: {enhanced_spall_path}")

            # Remove the basic spall summary file as it's redundant
            try:
                spall_summary_path = os.path.join(spade_output_dir, 'spall_summary.csv')
                if os.path.exists(spall_summary_path):
                    os.remove(spall_summary_path)
                    self.progress_signal.emit("Removed redundant spall_summary.csv (superseded by enhanced_spall_summary.csv)")
            except Exception:
                pass
            
            # Generate additional analysis plots if parameter data is available
            self.generate_spall_analysis_plots(enhanced_spall_df, spade_output_dir)
        else:
            self.progress_signal.emit("No enhanced spall data generated")

    def generate_spall_analysis_plots(self, enhanced_spall_df, spade_output_dir):
        """Generate additional analysis plots for spall data with parameter information"""
        self.progress_signal.emit("Generating spall analysis plots...")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Check if we have material information for color coding
            material_cols = [col for col in enhanced_spall_df.columns if 'material' in col.lower()]
            color_col = material_cols[0] if material_cols else None
            
            # Create color mapping for materials if available
            if color_col and len(enhanced_spall_df[color_col].unique()) > 1:
                materials = enhanced_spall_df[color_col].unique()
                colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(materials)))
                color_map = dict(zip(materials, colors))
                
                # Create legend patches
                legend_patches = [mpatches.Patch(color=color_map[mat], label=mat) for mat in materials]
            else:
                color_map = None
                legend_patches = []
            
            # Plot 1: Spall Strength vs Strain Rate
            if 'Spall Strength (GPa)' in enhanced_spall_df.columns and 'Strain Rate (s^-1)' in enhanced_spall_df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for idx, row in enhanced_spall_df.iterrows():
                    spall_strength = row.get('Spall Strength (GPa)', np.nan)
                    strain_rate = row.get('Strain Rate (s^-1)', np.nan)
                    
                    if not pd.isna(spall_strength) and not pd.isna(strain_rate):
                        if color_col and color_map:
                            color = color_map.get(row[color_col], 'blue')
                            ax.scatter(strain_rate, spall_strength, c=[color], s=100, alpha=0.7)
                        else:
                            ax.scatter(strain_rate, spall_strength, c='blue', s=100, alpha=0.7)
                
                ax.set_xlabel('Strain Rate (s⁻¹)', fontsize=14)
                ax.set_ylabel('Spall Strength (GPa)', fontsize=14)
                ax.set_title('Spall Strength vs Strain Rate', fontsize=16)
                ax.grid(True, alpha=0.3)
                
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper left')
                
                plt.tight_layout()
                plot_path = os.path.join(spade_output_dir, 'spall_strength_vs_strain_rate_enhanced.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.progress_signal.emit(f"Generated enhanced spall strength vs strain rate plot")
            
            # Plot 2: Spall Strength vs Shock Stress
            if 'Spall Strength (GPa)' in enhanced_spall_df.columns and 'Peak Shock Stress (GPa)' in enhanced_spall_df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for idx, row in enhanced_spall_df.iterrows():
                    spall_strength = row.get('Spall Strength (GPa)', np.nan)
                    shock_stress = row.get('Peak Shock Stress (GPa)', np.nan)
                    
                    if not pd.isna(spall_strength) and not pd.isna(shock_stress):
                        if color_col and color_map:
                            color = color_map.get(row[color_col], 'red')
                            ax.scatter(shock_stress, spall_strength, c=[color], s=100, alpha=0.7)
                        else:
                            ax.scatter(shock_stress, spall_strength, c='red', s=100, alpha=0.7)
                
                ax.set_xlabel('Peak Shock Stress (GPa)', fontsize=14)
                ax.set_ylabel('Spall Strength (GPa)', fontsize=14)
                ax.set_title('Spall Strength vs Shock Stress', fontsize=16)
                ax.grid(True, alpha=0.3)
                
                if legend_patches:
                    ax.legend(handles=legend_patches, loc='upper left')
                
                plt.tight_layout()
                plot_path = os.path.join(spade_output_dir, 'spall_strength_vs_shock_stress_enhanced.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.progress_signal.emit(f"Generated enhanced spall strength vs shock stress plot")
            
            # Plot 3: Material comparison (if material data available)
            if color_col and len(enhanced_spall_df[color_col].unique()) > 1:
                # Check if we have valid data for both plots
                has_spall_data = 'Spall Strength (GPa)' in enhanced_spall_df.columns
                has_strain_data = 'Strain Rate (s^-1)' in enhanced_spall_df.columns
                
                if has_spall_data or has_strain_data:
                    # Determine number of subplots needed
                    n_plots = sum([has_spall_data, has_strain_data])
                    if n_plots == 1:
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                        axes = [ax]
                    else:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        axes = [ax1, ax2]
                
                    plot_idx = 0
                    materials = enhanced_spall_df[color_col].unique()
                    
                    # Box plot of spall strength by material
                    if has_spall_data:
                        spall_data = [
                            enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Spall Strength (GPa)'].dropna()
                            for mat in materials
                        ]
                        valid_spall_data = [(data, mat) for data, mat in zip(spall_data, materials) if len(data) > 0]
                        if valid_spall_data:
                            spall_data_clean, materials_clean = zip(*valid_spall_data)
                            bp1 = axes[plot_idx].boxplot(spall_data_clean, labels=materials_clean, patch_artist=True)
                            for patch, mat in zip(bp1['boxes'], materials_clean):
                                patch.set_facecolor(color_map.get(mat, 'gray'))
                                patch.set_alpha(0.7)
                            axes[plot_idx].set_ylabel('Spall Strength (GPa)', fontsize=14)
                            axes[plot_idx].set_title('Spall Strength by Material', fontsize=16)
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
                    
                    # Box plot of strain rate by material
                    if has_strain_data:
                        strain_data = [
                            enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Strain Rate (s^-1)'].dropna()
                            for mat in materials
                        ]
                        valid_strain_data = [(data, mat) for data, mat in zip(strain_data, materials) if len(data) > 0]
                        if valid_strain_data:
                            strain_data_clean, materials_clean = zip(*valid_strain_data)
                            bp2 = axes[plot_idx].boxplot(strain_data_clean, labels=materials_clean, patch_artist=True)
                            for patch, mat in zip(bp2['boxes'], materials_clean):
                                patch.set_facecolor(color_map.get(mat, 'gray'))
                                patch.set_alpha(0.7)
                            axes[plot_idx].set_ylabel('Strain Rate (s^-1)', fontsize=14)
                            axes[plot_idx].set_title('Strain Rate by Material', fontsize=16)
                            axes[plot_idx].grid(True, alpha=0.3)
                            plot_idx += 1
                    
                    plt.tight_layout()
                    comparison_path = os.path.join(spade_output_dir, 'spall_material_comparison.png')
                    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    self.progress_signal.emit(f"Generated material comparison plot: {comparison_path}")
                else:
                    self.progress_signal.emit("Skipping material comparison plot - insufficient data for spall or strain rate")
            
            self.progress_signal.emit("Completed spall analysis plots generation")
            
        except Exception as e:
            self.progress_signal.emit(f"Error generating spall analysis plots: {str(e)}")

        # 1. Find all ALPSS velocity files (raw, not smooth)
        velocity_files = glob.glob(os.path.join(self.output_dir, '*--velocity.csv'))
        if velocity_files:
            # 2. Read and align all velocity files by time
            dfs = []
            for f in velocity_files:
                try:
                    # Try reading with headers first
                    df = pd.read_csv(f)
                    if 'Time' in df.columns and 'Velocity' in df.columns:
                        dfs.append(df[['Time', 'Velocity']].rename(
                            columns={'Velocity': os.path.basename(f)}))
                    else:
                        # No headers, assume first column is time, second is velocity
                        df = pd.read_csv(f, header=None)
                        if df.shape[1] >= 2:
                            df.columns = ['Time', 'Velocity']
                            dfs.append(df[['Time', 'Velocity']].rename(
                                columns={'Velocity': os.path.basename(f)}))
                except Exception as e:
                    print(f"[WARNING] Could not read velocity file {f}: {e}")
                    continue
            if dfs:
                # Merge on Time
                merged = dfs[0]
                for d in dfs[1:]:
                    merged = pd.merge(merged, d, on='Time', how='outer')
                merged = merged.sort_values('Time').reset_index(drop=True)
                # Compute mean and std dev
                velocity_cols = [
    col for col in merged.columns if col != 'Time']
                
                # Convert velocity columns to numeric, coercing errors to NaN
                for col in velocity_cols:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce')
                
                # Filter to only numeric columns (exclude any that became all NaN)
                numeric_velocity_cols = [col for col in velocity_cols 
                                        if merged[col].dtype in ['float64', 'int64'] 
                                        and not merged[col].isna().all()]
                
                if numeric_velocity_cols:
                    merged['Mean Velocity (m/s)'] = merged[numeric_velocity_cols].mean(axis=1)
                    merged['Std Dev Velocity (m/s)'] = merged[numeric_velocity_cols].std(axis=1)
                else:
                    # No valid numeric columns found
                    merged['Mean Velocity (m/s)'] = np.nan
                    merged['Std Dev Velocity (m/s)'] = np.nan
                    self.progress_signal.emit("Warning: No valid numeric velocity columns found for mean calculation")

                # Create a properly named file that matches SPADE's expected pattern
                # Use a generic name that will work with the plotting function
                mean_vel_file = os.path.join(
    spade_output_dir, 'combined_mean_raw_velocity.csv')
                merged[['Time',
    'Mean Velocity (m/s)',
    'Std Dev Velocity (m/s)']].to_csv(mean_vel_file,
     index=False)

                # 3. Plot combined mean velocity - create a simple plot since
                # SPADE's function expects specific naming
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    time_data = merged['Time']
                    mean_velocity = merged['Mean Velocity (m/s)']
                    velocity_std = merged['Std Dev Velocity (m/s)']

                    # Plot mean line
                    ax.plot(
    time_data,
    mean_velocity,
    'b-',
    linewidth=2,
     label='Mean Velocity')

                    # Plot shaded uncertainty if available
                    if not velocity_std.isna().all():
                        ax.fill_between(time_data, mean_velocity - velocity_std, mean_velocity + velocity_std,
                                      alpha=0.3, color='blue', label='±1σ')

                    ax.set_xlabel('Time (ns)', fontsize=14)
                    ax.set_ylabel(
    'Mean Free Surface Velocity (m/s)', fontsize=14)
                    ax.set_title('Combined Mean Velocity Traces', fontsize=16)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 700)

                    plt.tight_layout()
                    # Save to both spade_output_dir and main output_dir
                    spade_plot_path = os.path.join(spade_output_dir, 'combined_mean_velocity.png')
                    main_plot_path = os.path.join(self.output_dir, 'combined_mean_velocity.png')
                    
                    plt.savefig(spade_plot_path, dpi=300)
                    plt.savefig(main_plot_path, dpi=300)
                    plt.close(fig)
                    self.progress_signal.emit(
                        f"Generated combined_mean_velocity.png in both locations")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate combined_mean_velocity.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)

        # --- ENHANCED: Plot all smoothed velocity traces with material and waveplate angle information ---
        # Accept either classic smoothed file or the smoothed-with-uncertainty file (used in smart selection)
        smoothed_files_velocity = glob.glob(os.path.join(self.output_dir, '*--velocity--smooth.csv'))
        smoothed_files_uncert = glob.glob(os.path.join(self.output_dir, '*--vel-smooth-with-uncert.csv'))
        smoothed_files = list(set(smoothed_files_velocity + smoothed_files_uncert))

        # Report combined plotting file availability
        self.progress_signal.emit(
            f"=== Enhanced Combined Velocity Plotting ===")
        self.progress_signal.emit(
            f"Found {len(smoothed_files)} velocity files for combined plotting")

        if len(smoothed_files) != len(self.successful_files):
            missing_plot_files = []
            for input_file in self.successful_files:
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                # Check primary and fallback patterns
                expected_primary = os.path.join(self.output_dir, f"{base_name}--velocity--smooth.csv")
                expected_fallback = os.path.join(self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                if not (os.path.exists(expected_primary) or os.path.exists(expected_fallback)):
                    missing_plot_files.append(base_name)

            if missing_plot_files:
                self.progress_signal.emit(
                    f"Missing velocity files for plotting: {len(missing_plot_files)}")
                for missing_file in missing_plot_files:
                    self.progress_signal.emit(
                        f"❌ Missing: {missing_file} (no smoothed CSV found: --velocity--smooth.csv or --vel-smooth-with-uncert.csv)")

        

        # 4. Spall Strength vs. Strain Rate and Shock Stress (only if spall analysis is enabled)
        experiment_type = self.spade_params.get('experiment_type', 'velocity_shots')
        
        if experiment_type == "spall_analysis":
            summary_csv = os.path.join(spade_output_dir, 'spall_summary.csv')
            if os.path.exists(summary_csv):
                summary_df = pd.read_csv(summary_csv)
                # Get default density and acoustic velocity from GUI/spade_params (used as fallback)
                default_density = self.spade_params.get('density', 8960)
                default_acoustic_velocity = self.spade_params.get('acoustic_velocity', 3950)
                
                # Use parameter file data passed to AnalysisThread
                param_data = self.param_data if self.param_data else {}
                
                # Enhance SPADE summary with ALPSS results and additional calculations
                enhanced_summary = []
                
                for idx, row in summary_df.iterrows():
                    enhanced_row = row.copy()
                    filename = row.get('Filename', '')
                    
                    # Get material-specific properties if available
                    base_name = filename
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    # Try to get material from parameter file using helper function
                    sample_material = 'Unknown'
                    matched_param = self.get_param_data_for_file(base_name)
                    if matched_param:
                        sample_material = matched_param.get('Sample material', 'Unknown')
                    
                    # Get material-specific properties from config first, then database
                    mat_props = self.get_material_properties_from_config(sample_material, default_density, default_acoustic_velocity)
                    # Parameter file can override (highest priority)
                    if matched_param:
                        density = matched_param.get('Density_kg_m3', mat_props['density'])
                        acoustic_velocity = matched_param.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed'])
                    else:
                        density = mat_props['density']
                        acoustic_velocity = mat_props['bulk_wave_speed']
                    # Add material information to enhanced row
                    enhanced_row['Material'] = sample_material
                    enhanced_row['Density_kg_m3'] = density
                    enhanced_row['Acoustic_Velocity_m_s'] = acoustic_velocity
                    enhanced_row['Material_Found_In_Database'] = mat_props['material_found']
                    enhanced_row['Material_Properties_Source'] = mat_props.get('source', 'unknown')
                    
                    # Try to find corresponding ALPSS results file
                    alpss_results_file = os.path.join(self.output_dir, f"{filename}--results.csv")
                    if os.path.exists(alpss_results_file):
                        try:
                            # Read ALPSS results
                            alpss_results = pd.read_csv(alpss_results_file, header=None, names=['Name', 'Value'])
                            alpss_dict = dict(zip(alpss_results['Name'], alpss_results['Value']))
                            
                            # Add ALPSS results to enhanced summary
                            enhanced_row['ALPSS_Spall_Strength_GPa'] = alpss_dict.get('Spall Strength', np.nan)
                            enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa'] = alpss_dict.get('Spall Strength Uncertainty', np.nan)
                            enhanced_row['ALPSS_Strain_Rate_s1'] = alpss_dict.get('Strain Rate', np.nan)
                            enhanced_row['ALPSS_Strain_Rate_Uncertainty_s1'] = alpss_dict.get('Strain Rate Uncertainty', np.nan)
                            enhanced_row['ALPSS_Peak_Shock_Stress_GPa'] = alpss_dict.get('Peak Shock Stress', np.nan)
                            enhanced_row['ALPSS_Peak_Velocity_Uncertainty_ms'] = alpss_dict.get('Peak Velocity Uncertainty', np.nan)
                            enhanced_row['ALPSS_Pullback_Velocity_Uncertainty_ms'] = alpss_dict.get('Pullback Velocity Uncertainty', np.nan)
                            enhanced_row['ALPSS_Velocity_at_Max_Compression_ms'] = alpss_dict.get('Velocity at Max Compression', np.nan)
                            enhanced_row['ALPSS_Velocity_at_Max_Tension_ms'] = alpss_dict.get('Velocity at Max Tension', np.nan)
                            enhanced_row['ALPSS_Velocity_at_Recompression_ms'] = alpss_dict.get('Velocity at Recompression', np.nan)
                            enhanced_row['ALPSS_Time_at_Max_Compression_ns'] = alpss_dict.get('Time at Max Compression', np.nan)
                            enhanced_row['ALPSS_Time_at_Max_Tension_ns'] = alpss_dict.get('Time at Max Tension', np.nan)
                            enhanced_row['ALPSS_Time_at_Recompression_ns'] = alpss_dict.get('Time at Recompression', np.nan)
                            enhanced_row['ALPSS_Carrier_Frequency_Hz'] = alpss_dict.get('Carrier Frequency', np.nan)
                            enhanced_row['ALPSS_Signal_Start_Time_s'] = alpss_dict.get('Signal Start Time', np.nan)
                            enhanced_row['ALPSS_Smoothing_Characteristic_Time_s'] = alpss_dict.get('Smoothing Characteristic Time', np.nan)
                            
                        except Exception as e:
                            print(f"[WARNING] Could not read ALPSS results for {filename}: {e}")
                    
                    # Calculate shock stress uncertainty using ALPSS-calculated velocity uncertainty
                    # Only use rigorously calculated uncertainty - no ad-hoc estimates
                    vel_uncertainty = enhanced_row.get('ALPSS_Peak_Velocity_Uncertainty_ms', np.nan)
                    
                    if not pd.isna(vel_uncertainty) and not pd.isna(density) and not pd.isna(acoustic_velocity):
                        enhanced_row['ALPSS_Peak_Shock_Stress_Uncertainty_GPa'] = 0.5 * density * acoustic_velocity * vel_uncertainty * 1e-9
                    else:
                        enhanced_row['ALPSS_Peak_Shock_Stress_Uncertainty_GPa'] = np.nan
                    
                    # Use ALPSS values if available, otherwise use SPADE values
                    if 'ALPSS_Spall_Strength_GPa' in enhanced_row and not pd.isna(enhanced_row['ALPSS_Spall_Strength_GPa']):
                        enhanced_row['Spall_Strength_GPa_Final'] = enhanced_row['ALPSS_Spall_Strength_GPa']
                        enhanced_row['Spall_Strength_Uncertainty_GPa_Final'] = enhanced_row['ALPSS_Spall_Strength_Uncertainty_GPa']
                    else:
                        enhanced_row['Spall_Strength_GPa_Final'] = row.get('Spall Strength (GPa)', np.nan)
                        enhanced_row['Spall_Strength_Uncertainty_GPa_Final'] = row.get('Spall Strength Uncertainty (GPa)', np.nan)
                    
                    if 'ALPSS_Strain_Rate_s1' in enhanced_row and not pd.isna(enhanced_row['ALPSS_Strain_Rate_s1']):
                        enhanced_row['Strain_Rate_s1_Final'] = enhanced_row['ALPSS_Strain_Rate_s1']
                        enhanced_row['Strain_Rate_Uncertainty_s1_Final'] = enhanced_row['ALPSS_Strain_Rate_Uncertainty_s1']
                    else:
                        enhanced_row['Strain_Rate_s1_Final'] = row.get('Strain Rate (s^-1)', np.nan)
                        enhanced_row['Strain_Rate_Uncertainty_s1_Final'] = row.get('Strain Rate Uncertainty (s^-1)', np.nan)
                    
                    if 'ALPSS_Peak_Shock_Stress_GPa' in enhanced_row and not pd.isna(enhanced_row['ALPSS_Peak_Shock_Stress_GPa']):
                        enhanced_row['Peak_Shock_Stress_GPa_Final'] = enhanced_row['ALPSS_Peak_Shock_Stress_GPa']
                        enhanced_row['Peak_Shock_Stress_Uncertainty_GPa_Final'] = enhanced_row['ALPSS_Peak_Shock_Stress_Uncertainty_GPa']
                    else:
                        # Calculate from SPADE's Plateau Mean Velocity
                        if 'Plateau Mean Velocity (m/s)' in row and not pd.isna(row['Plateau Mean Velocity (m/s)']):
                            enhanced_row['Peak_Shock_Stress_GPa_Final'] = row.get('Peak Shock Stress (GPa)', np.nan)
                            enhanced_row['Peak_Shock_Stress_Uncertainty_GPa_Final'] = row.get('Peak Shock Stress Uncertainty (GPa)', np.nan)
                        else:
                            enhanced_row['Peak_Shock_Stress_GPa_Final'] = np.nan
                            enhanced_row['Peak_Shock_Stress_Uncertainty_GPa_Final'] = np.nan
                    
                    enhanced_summary.append(enhanced_row)
                
                # Create enhanced summary DataFrame
                enhanced_summary_df = pd.DataFrame(enhanced_summary)
                
                # Save enhanced summary
                enhanced_summary_path = os.path.join(spade_output_dir, 'enhanced_spall_summary.csv')
                enhanced_summary_df.to_csv(enhanced_summary_path, index=False)
                
                # Update the original summary with key columns for plotting
                summary_df['Peak Shock Stress (GPa)'] = enhanced_summary_df['Peak_Shock_Stress_GPa_Final']
                summary_df['Peak Shock Stress Uncertainty (GPa)'] = enhanced_summary_df['Peak_Shock_Stress_Uncertainty_GPa_Final']
                summary_df['Spall Strength (GPa)'] = enhanced_summary_df['Spall_Strength_GPa_Final']
                summary_df['Spall Strength Uncertainty (GPa)'] = enhanced_summary_df['Spall_Strength_Uncertainty_GPa_Final']
                summary_df['Strain Rate Uncertainty (s^-1)'] = enhanced_summary_df['Strain_Rate_Uncertainty_s1_Final']
                summary_df.to_csv(summary_csv, index=False)
                
                self.progress_signal.emit("Enhanced SPADE summary with ALPSS results and uncertainty calculations")
                
                # Log available outputs for spall analysis
                self.progress_signal.emit("Available outputs (Spall Analysis):")
                self.progress_signal.emit("  - spall_summary.csv: Basic SPADE results")
                self.progress_signal.emit("  - enhanced_spall_summary.csv: Complete results with ALPSS data and uncertainties")
                self.progress_signal.emit("  - spall_vs_strain_rate.png: Spall strength vs strain rate plot")
                self.progress_signal.emit("  - spall_vs_shock_stress.png: Spall strength vs shock stress plot")
                self.progress_signal.emit("  - shock_stress_vs_laser_energy.png: Peak shock stress vs laser energy")
                self.progress_signal.emit("  - Individual ALPSS files: *--results.csv, *--velocity.csv, etc.")
                self.progress_signal.emit("  - Individual SPADE analysis plots (if enabled)")
                self.progress_signal.emit("  - ALPSS velocity files: 4 columns (Time, Velocity, Uncertainty, Velocity+Uncertainty)")
                self.progress_signal.emit("  - SPADE uses ALPSS uncertainty data for error bars and analysis")
                
                # Spall Strength vs. Strain Rate
                try:
                    plot_spall_vs_strain_rate(
                        df=summary_df,
                        output_filename=os.path.join(spade_output_dir, 'spall_vs_strain_rate.png'),
                        literature_data_file=None,  # Disable literature data to avoid column mismatch errors
                        spall_unc_col='Spall Strength Uncertainty (GPa)'
                    )
                    self.progress_signal.emit("✅ Generated spall_vs_strain_rate.png")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate spall_vs_strain_rate.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)
                try:
                    plot_spall_vs_shock_stress(
                        df=summary_df,
                        output_filename=os.path.join(spade_output_dir, 'spall_vs_shock_stress.png'),
                        literature_data_file=None,  # Disable literature data to avoid column mismatch errors
                        spall_unc_col='Spall Strength Uncertainty (GPa)'
                    )
                    self.progress_signal.emit("✅ Generated spall_vs_shock_stress.png")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate spall_vs_shock_stress.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)
                try:
                    plot_shock_stress_vs_laser_energy(
                        df=summary_df,
                        output_filename=os.path.join(spade_output_dir, 'shock_stress_vs_laser_energy.png')
                    )
                    self.progress_signal.emit("✅ Generated shock_stress_vs_laser_energy.png")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate shock_stress_vs_laser_energy.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)
                
                # Generate combined velocity traces plot
                try:
                    # Find all velocity files
                    vel_files = []
                    for root, dirs, files in os.walk(self.output_dir):
                        for file in files:
                            if file.endswith("--vel-smooth-with-uncert.csv"):
                                vel_files.append(os.path.join(root, file))
                    
                    if vel_files:
                        plot_combined_mean_traces(
                            mean_trace_files=vel_files,
                            output_filename=os.path.join(spade_output_dir, 'combined_mean_velocity.png'),
                            title="Combined Mean Velocity Traces"
                        )
                        self.progress_signal.emit("✅ Generated combined_mean_velocity.png")
                    else:
                        self.progress_signal.emit("[WARNING] No velocity files found for combined plot")
                except Exception as e:
                    msg = f"[WARNING] Failed to generate combined_mean_velocity.png: {e}"
                    print(msg)
                    self.progress_signal.emit(msg)
            else:
                self.progress_signal.emit("[WARNING] Spall analysis selected but no spall_summary.csv found")
        else:
            # Velocity shots mode - only log velocity-related outputs
            self.progress_signal.emit("Available outputs (Velocity Shots):")
            self.progress_signal.emit("  - Individual ALPSS files: *--results.csv, *--velocity.csv, etc.")
            self.progress_signal.emit("  - ALPSS velocity files: 4 columns (Time, Velocity, Uncertainty, Velocity+Uncertainty)")
            self.progress_signal.emit("  - Combined velocity plots (if enabled)")
        
        # Check for missing plots and warn (only check velocity plots for velocity mode, all plots for spall mode)
        if experiment_type == "spall_analysis":
            expected_plots = [
                'combined_mean_velocity.png',
                'spall_vs_strain_rate.png',
                'spall_vs_shock_stress.png',
                'shock_stress_vs_laser_energy.png',
                'all_smoothed_velocity_traces.png'
            ]
        else:
            expected_plots = [
                'combined_mean_velocity.png',
                'all_smoothed_velocity_traces.png'
            ]
        
        for plot_name in expected_plots:
            plot_path = os.path.join(spade_output_dir, plot_name)
            if not os.path.exists(plot_path):
                msg = f"[WARNING] Expected plot missing: {plot_name}"
                print(msg)
                self.progress_signal.emit(msg)

class PostProcessingWorker(QObject):
    """Worker thread for regenerating plots in post-processing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, output_dir, spade_params, param_folder):
        super().__init__()
        self.output_dir = output_dir
        self.spade_params = spade_params or {}
        self.param_folder = param_folder
    
    def regenerate_plots(self, spade_params):
        """Regenerate velocity plots with current axis settings"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            self.progress.emit("Collecting velocity files...")
            
            # Apply limits from post-processing settings
            current_params = self.spade_params.copy()
            current_params.update(spade_params)
            # Debug: Log parameter updates
            self.progress.emit(f"[WORKER] Received parameters in regenerate_plots:")
            self.progress.emit(f"  auto_calc_limits: {current_params.get('auto_calculate_limits')}")
            self.progress.emit(f"  x_min/max_main: {current_params.get('x_min_main')}/{current_params.get('x_max_main')}")
            self.progress.emit(f"  y_min/max_main: {current_params.get('y_min_main')}/{current_params.get('y_max_main')}")
            
            pattern = os.path.join(self.output_dir, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            
            if not files:
                self.progress.emit("❌ No velocity files found")
                self.finished.emit()
                return
            
            self.progress.emit(f"Found {len(files)} velocity files")
            
            # Load parameter files to get material info
            self.progress.emit("Loading parameter files...")
            param_data = self._load_param_files()
            if param_data:
                self.progress.emit(f"✓ Loaded {len(param_data)} parameter entries")
                # Debug: show first few entries
                for i, (key, val) in enumerate(list(param_data.items())[:3]):
                    material = val.get('Sample material', 'Unknown') if isinstance(val, dict) else 'Unknown'
                    self.progress.emit(f"  - {key}: {material}")
                # Show first velocity filename for comparison
                if files:
                    first_vel_file = os.path.basename(files[0])
                    first_vel_base = os.path.splitext(first_vel_file)[0]
                    self.progress.emit(f"First velocity file: {first_vel_file}")
                    self.progress.emit(f"First velocity basename: {first_vel_base}")
                    # Check if it exists in param_data
                    if first_vel_base in param_data:
                        self.progress.emit(f"✓ MATCH found in param_data")
                    else:
                        self.progress.emit(f"✗ NO MATCH in param_data")
                        # Try to find similar keys
                        similar = [k for k in param_data.keys() if first_vel_base[:10] in k or k[:10] in first_vel_base]
                        if similar:
                            self.progress.emit(f"Similar keys in param_data: {similar[:3]}")
            else:
                self.progress.emit("⚠ No parameter data loaded - material will be 'Unknown'")
            
            spade_out = os.path.join(self.output_dir, "SPADE_analysis")
            os.makedirs(spade_out, exist_ok=True)
            
            # Generate plot
            self.progress.emit("Generating plot...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Use colormap for material differentiation
            cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, max(1, len(files))))
            
            traces_plotted = 0
            align_threshold = current_params.get('align_velocity_threshold_ms', 30.0)
            zoom_ns = int(current_params.get('zoom_window_ns', 1000))
            
            # Determine color coding mode
            use_material_colors = current_params.get('use_material_colors', True)
            color_by_waveplate = current_params.get('color_by_waveplate', False)
            color_by_laser_energy = current_params.get('color_by_laser_energy', False)
            
            # Track grouping parameter and colors for legend
            group_colors = {}
            file_group_map = {}
            color_label = "Material"  # Default label
            
            # Determine which parameter to use for grouping
            if color_by_waveplate:
                color_param = 'Waveplate_Angle (Degrees)'
                color_label = "Waveplate Angle (°)"
            elif color_by_laser_energy:
                color_param = 'Laser_Target_Energy (mJ)'
                color_label = "Laser Energy (mJ)"
            else:
                color_param = 'Sample material'
                color_label = "Material"
            
            # First pass: collect all unique values of the grouping parameter
            self.progress.emit(f"Scanning {color_param}...")
            unique_values = set()
            for i, file_path in enumerate(sorted(files)):
                try:
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    group_value = "Unknown"
                    if param_data:
                        # Try exact match first
                        if base_name in param_data:
                            group_value = param_data[base_name].get(color_param, 'Unknown')
                        else:
                            # Try date-shot pattern matching (YYYYMMDD--NNNNN)
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        group_value = param_data[key].get(color_param, 'Unknown')
                                        break
                        
                        # Convert to string and clean up
                        if isinstance(group_value, (int, float)):
                            group_value = str(group_value)
                        elif isinstance(group_value, str):
                            group_value = group_value.strip()
                    
                    unique_values.add(group_value)
                except Exception:
                    pass
            
            # Assign distinct colors to each unique value
            # Sort numerically for laser energy, alphabetically for others
            if color_by_laser_energy:
                # Convert to float for numeric sorting, handle 'Unknown'
                def safe_float(val):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float('inf')  # Put non-numeric values at the end
                unique_values_sorted = sorted(list(unique_values), key=safe_float)
                # Use viridis colormap for laser energy
                cmap = plt.get_cmap('viridis')
            else:
                unique_values_sorted = sorted(list(unique_values))
            cmap = plt.get_cmap('tab20')
            
            for i, value in enumerate(unique_values_sorted):
                group_colors[value] = cmap(i / max(len(unique_values_sorted), 1))
            
            self.progress.emit(f"Found {len(unique_values_sorted)} unique {color_label} values: {unique_values_sorted[:10]}")
            
            for i, file_path in enumerate(sorted(files)):
                try:
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 3:
                        continue
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    uncertainty_data = df.iloc[:, 2].values
                    
                    # Convert time to ns if needed
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3
                    
                    # Noise fraction filtering
                    noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                    high_noise_mask = None
                    if os.path.exists(noise_file):
                        try:
                            df_noise = pd.read_csv(noise_file)
                            if df_noise.shape[1] >= 1:
                                noise_fraction = df_noise.iloc[:, -1].values
                                if len(noise_fraction) == len(velocity_data):
                                    high_noise_mask = noise_fraction > 1.0
                        except Exception:
                            pass
                    
                    valid_mask = ~np.isnan(velocity_data)
                    if high_noise_mask is not None:
                        valid_mask &= (~high_noise_mask)
                    # Uncertainty threshold filtering
                    uncertainty_threshold = current_params.get('uncertainty_threshold_ms', 50.0)
                    if uncertainty_data is not None:
                        valid_mask &= (uncertainty_data <= uncertainty_threshold)
                    
                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                    if len(time_clean) == 0:
                        continue
                    
                    # Align at first >= threshold
                    t0_idx = None
                    for j, v in enumerate(velocity_clean):
                        if not np.isnan(v) and v >= align_threshold:
                            t0_idx = j
                            break
                    if t0_idx is not None:
                        t0 = time_clean[t0_idx]
                        time_clean = time_clean - t0
                    
                    # Extract grouping parameter value from parameter file
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes from base_name to match parameter file PDV_FileName
                    # E.g., "C1--20251023--00099--vel-smooth-with-uncert" -> "C1--20251023--00099"
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    group_value = "Unknown"
                    
                    # Try to get grouping parameter value from param_data
                    if param_data:
                        if base_name in param_data:
                            group_value = param_data[base_name].get(color_param, 'Unknown')
                    else:
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        group_value = param_data[key].get(color_param, 'Unknown')
                                        break
                    # Convert to string and clean up
                    if isinstance(group_value, (int, float)):
                        group_value = str(group_value)
                    elif isinstance(group_value, str):
                        group_value = group_value.strip()
                    
                    if group_value == "Unknown" and param_data:
                        if i < 5:
                            self.progress.emit(f"  No match for: {base_name}")
                        elif i == 5:
                            self.progress.emit("  ... (more non-matching files)")
                    
                    # Assign color based on grouping parameter value
                    if group_value not in group_colors:
                        group_colors[group_value] = colors[len(group_colors) % len(colors)]
                    
                    color = group_colors[group_value]
                    file_group_map[i] = group_value
                    
                    ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                    # Optional uncertainty bands
                    if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                        alpha = float(current_params.get('uncert_alpha', 0.2))
                        ax1.fill_between(time_clean,
                                         velocity_clean - uncert_clean,
                                         velocity_clean + uncert_clean,
                                         color=color, alpha=alpha)
                    
                    # Bottom zoom window
                    mask_zoom = time_clean <= zoom_ns
                    if np.any(mask_zoom):
                        ax2.plot(time_clean[mask_zoom], velocity_clean[mask_zoom], color=color, alpha=0.7, linewidth=1, label=group_value if group_value not in [file_group_map.get(j) for j in range(i)] else "")
                        if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                            alpha = float(current_params.get('uncert_alpha', 0.2))
                            ax2.fill_between(time_clean[mask_zoom],
                                             (velocity_clean - uncert_clean)[mask_zoom],
                                             (velocity_clean + uncert_clean)[mask_zoom],
                                             color=color, alpha=alpha)
                    
                    traces_plotted += 1
                    if traces_plotted % 5 == 0:
                        self.progress.emit(f"Processed {traces_plotted} traces...")
                except Exception as e:
                    continue
            
            # Labels and axis limits
            ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax1.set_ylabel('Velocity (m/s)', fontsize=12)
            ax1.set_title(f'All Velocity Traces (Aligned - Full Length, Color by {color_label}) - {traces_plotted} traces', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Apply axis limits from post-processing settings
            try:
                if not current_params.get('auto_calculate_limits', True):
                    # User has specified custom limits for both subplots
                    
                    # TOP SUBPLOT (ax1): Apply main subplot limits
                    x_min_main = float(current_params.get('x_min_main', 0))
                    x_max_main = float(current_params.get('x_max_main', 100))
                    y_min_main = float(current_params.get('y_min_main', 0))
                    y_max_main = float(current_params.get('y_max_main', 600))
                    ax1.set_xlim(x_min_main, x_max_main)
                    ax1.set_ylim(y_min_main, y_max_main)
                    self.progress.emit(f"Top subplot: Using user main limits X({x_min_main}-{x_max_main}), Y({y_min_main}-{y_max_main})")
                    
                    # BOTTOM SUBPLOT (ax2): Apply zoom subplot limits
                    x_min_zoom = float(current_params.get('x_min_zoom', 0))
                    x_max_zoom = float(current_params.get('x_max_zoom', zoom_ns))
                    y_min_zoom = float(current_params.get('y_min_zoom', 0))
                    y_max_zoom = float(current_params.get('y_max_zoom', 300))
                    ax2.set_xlim(x_min_zoom, x_max_zoom)
                    ax2.set_ylim(y_min_zoom, y_max_zoom)
                    ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)}-{int(x_max_zoom)} ns)', fontsize=14)
                    self.progress.emit(f"Bottom subplot: Using user zoom limits X({x_min_zoom}-{x_max_zoom}), Y({y_min_zoom}-{y_max_zoom})")
                else:
                    # Use auto-calculated limits
                    self.progress.emit(f"Using auto-calculated limits for both subplots")
                    ax2.set_xlim(0, zoom_ns)
                    ax2.set_title(f'Zoomed Velocity Traces (First {zoom_ns} ns)', fontsize=14)
            except Exception as e:
                self.progress.emit(f"⚠ Axis limits error: {str(e)[:80]}")
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'Zoomed Velocity Traces (First {zoom_ns} ns)', fontsize=14)
            
            # Add legend for grouping parameter (using the already sorted order)
            if group_colors:
                legend_patches = [mpatches.Patch(color=group_colors[val], label=val) for val in unique_values_sorted]
                ax1.legend(handles=legend_patches, loc='upper right', title=color_label, fontsize=9, title_fontsize=10)
                ax2.legend(handles=legend_patches, loc='upper right', title=color_label, fontsize=9, title_fontsize=10)
            
            plt.tight_layout()
            
            # Save plot
            self.progress.emit("Saving plot...")
            plot_path = os.path.join(spade_out, 'all_velocity_traces.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved: {plot_path}")
            self.progress.emit(f"✓ Plotted {traces_plotted} traces")
            if group_colors:
                self.progress.emit(f"✓ {color_label} values: {', '.join(sorted(group_colors.keys()))}")
            
            # Generate per-material subplots (if using material colors)
            if current_params.get('use_material_colors', True):
                self.progress.emit("Generating per-material subplot plot...")
                self._generate_material_subplots(
                    files,
                    param_data,
                    spade_out,
                    current_params,
                    group_colors,
                    align_threshold,
                )
            
            # Generate laser energy vs impact velocity plot (if enabled)
            if current_params.get('laser_energy_vs_velocity', False):
                self.progress.emit("Generating Laser Energy vs Impact Velocity plot...")
                self._generate_energy_vs_velocity_plot(files, param_data, spade_out, current_params)
            
        except Exception as e:
            self.progress.emit(f"❌ Error: {str(e)}")
            import traceback
            self.progress.emit(traceback.format_exc()[:200])
        finally:
            self.finished.emit()
    
    def _generate_material_subplots(self, files, param_data, spade_out, current_params, material_colors, align_threshold):
        """Generate a multi-subplot figure with one subplot per material"""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Group files by material
            material_files = {}
            for file_path in sorted(files):
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0]
                
                # Strip velocity-specific suffixes
                for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                material = "Unknown"
                if param_data:
                    if base_name in param_data:
                        material = param_data[base_name].get('Sample material', 'Unknown')
                    else:
                        import re
                        date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                        if date_shot_pattern:
                            date_shot = date_shot_pattern.group(1)
                            for key in param_data.keys():
                                if date_shot in str(key):
                                    material = param_data[key].get('Sample material', 'Unknown')
                                    break
                    if isinstance(material, str):
                        material = material.strip()
                
                if material not in material_files:
                    material_files[material] = []
                material_files[material].append(file_path)
            
            # Create subplots
            num_materials = len(material_files)
            if num_materials == 0:
                return
            
            # Arrange subplots in a grid (prefer rows over columns)
            ncols = min(2, num_materials)
            nrows = (num_materials + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows))
            if num_materials == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot each material
            for idx, (material, file_paths) in enumerate(sorted(material_files.items())):
                ax = axes[idx]
                color = material_colors.get(material, 'blue')
                
                traces_in_material = 0
                for file_path in file_paths:
                    try:
                        df = pd.read_csv(file_path)
                        if df.shape[1] < 3:
                            continue
                        
                        time_data = df.iloc[:, 0].values
                        velocity_data = df.iloc[:, 1].values
                        uncertainty_data = df.iloc[:, 2].values
                        
                        # Convert time to ns if needed
                        if np.nanmax(time_data) < 1e-3:
                            time_data = time_data * 1e9
                        elif np.nanmax(time_data) < 1.0:
                            time_data = time_data * 1e3
                        
                        # Noise fraction filtering
                        noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                        high_noise_mask = None
                        if os.path.exists(noise_file):
                            try:
                                df_noise = pd.read_csv(noise_file)
                                if df_noise.shape[1] >= 1:
                                    noise_fraction = df_noise.iloc[:, -1].values
                                    if len(noise_fraction) == len(velocity_data):
                                        high_noise_mask = noise_fraction > 1.0
                            except Exception:
                                pass
                        
                        valid_mask = ~np.isnan(velocity_data)
                        if high_noise_mask is not None:
                            valid_mask &= (~high_noise_mask)
                        
                        uncertainty_threshold = current_params.get('uncertainty_threshold_ms', 50.0)
                        if uncertainty_data is not None:
                            valid_mask &= (uncertainty_data <= uncertainty_threshold)
                        
                        time_clean = time_data[valid_mask]
                        velocity_clean = velocity_data[valid_mask]
                        uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                        
                        if len(time_clean) == 0:
                            continue
                        
                        # Align at first >= threshold
                        align_threshold_ms = current_params.get('align_velocity_threshold_ms', 30.0)
                        t0_idx = None
                        for j, v in enumerate(velocity_clean):
                            if not np.isnan(v) and v >= align_threshold_ms:
                                t0_idx = j
                                break
                        
                        if t0_idx is not None:
                            t0 = time_clean[t0_idx]
                            time_clean = time_clean - t0
                        
                        # Plot with material color
                        ax.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                        
                        # Optional uncertainty bands
                        if current_params.get('include_uncert_bands', True) and uncert_clean is not None:
                            alpha = float(current_params.get('uncert_alpha', 0.2))
                            ax.fill_between(time_clean,
                                           velocity_clean - uncert_clean,
                                           velocity_clean + uncert_clean,
                                           color=color, alpha=alpha)
                        
                        traces_in_material += 1
                    except Exception:
                        continue
                
                # Format subplot
                ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold_ms} m/s', fontsize=10)
                ax.set_ylabel('Velocity (m/s)', fontsize=10)
                ax.set_title(f'{material} ({traces_in_material} traces)', fontsize=12, color=color, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Apply axis limits from current_params
                if not current_params.get("auto_calculate_limits", True):
                    x_min = current_params.get("x_min_main", 0.0)
                    x_max = current_params.get("x_max_main", 100.0)
                    y_min = current_params.get("y_min_main", 0.0)
                    y_max = current_params.get("y_max_main", 600.0)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
            
            # Hide unused subplots
            for idx in range(num_materials, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(spade_out, 'velocity_traces_by_material.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved material subplots: {plot_path}")
            self.progress.emit(f"✓ Created {num_materials} subplots (one per material)")
            
        except Exception as e:
            self.progress.emit(f"⚠ Error generating material subplots: {str(e)[:80]}")
    
    def _generate_energy_vs_velocity_plot(self, files, param_data, spade_out, current_params):
        """Generate laser energy vs impact velocity scatter plot colored by waveplate angle"""
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Collect data: list of (energy, velocity, waveplate_angle) tuples
            data_points = []
            align_threshold = current_params.get('align_velocity_threshold_ms', 30.0)
            
            self.progress.emit("Processing files for laser energy vs velocity...")
            
            for file_path in sorted(files):
                try:
                    # Read velocity data
                    df = pd.read_csv(file_path)
                    if df.shape[1] < 2:
                        continue
                    
                    time_data = df.iloc[:, 0].values
                    velocity_data = df.iloc[:, 1].values
                    
                    # Convert time to ns if needed
                    if np.nanmax(time_data) < 1e-3:
                        time_data = time_data * 1e9
                    elif np.nanmax(time_data) < 1.0:
                        time_data = time_data * 1e3
                    
                    # Filter NaN values
                    valid_mask = ~np.isnan(velocity_data) & ~np.isnan(time_data)
                    time_clean = time_data[valid_mask]
                    velocity_clean = velocity_data[valid_mask]
                    
                    if len(time_clean) == 0:
                        continue
                    
                    # ALIGN TRACE: Find t=0 when velocity first exceeds threshold
                    threshold_mask = velocity_clean >= align_threshold
                    if np.sum(threshold_mask) > 0:
                        t0_idx = np.where(threshold_mask)[0][0]
                        t0_time = time_clean[t0_idx]
                        # Shift time so t=0 is at threshold crossing
                        time_aligned = time_clean - t0_time
                    else:
                        # No alignment possible - skip this trace
                        max_vel = np.max(velocity_clean)
                        self.progress.emit(f"  Warning: Velocity never reached {align_threshold} m/s for {os.path.basename(file_path)} (max: {max_vel:.1f} m/s)")
                        continue
                    
                    # Calculate mean velocity in 250-300ns window AFTER alignment (relative to t=0)
                    mask_window = (time_aligned >= 250) & (time_aligned <= 300)
                    if np.sum(mask_window) > 0:
                        mean_velocity = np.mean(velocity_clean[mask_window])
                        self.progress.emit(f"  ✓ Impact velocity 250-300ns after t=0 for {os.path.basename(file_path)}")
                    else:
                        # Fallback: try 300-320ns window
                        mask_window = (time_aligned >= 300) & (time_aligned <= 320)
                        if np.sum(mask_window) > 0:
                            mean_velocity = np.mean(velocity_clean[mask_window])
                            self.progress.emit(f"  Note: Using 300-320ns after t=0 for {os.path.basename(file_path)}")
                        else:
                            # Check what time range is actually available
                            time_range = f"{np.min(time_aligned):.0f} to {np.max(time_aligned):.0f}ns"
                            self.progress.emit(f"  Warning: No data in 250-300ns or 300-320ns after t=0 for {os.path.basename(file_path)} (available: {time_range})")
                            continue
                    
                    # Get laser energy and waveplate angle from parameter file
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0]
                    
                    # Strip velocity-specific suffixes
                    for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                        if base_name.endswith(suffix):
                            base_name = base_name[:-len(suffix)]
                            break
                    
                    laser_energy = None
                    waveplate_angle = None
                    
                    if param_data:
                        # Try exact match first
                        if base_name in param_data:
                            laser_energy = param_data[base_name].get('Laser_Target_Energy (mJ)', None)
                            waveplate_angle = param_data[base_name].get('Waveplate_Angle (Degrees)', None)
                        else:
                            # Try date-shot pattern matching
                            import re
                            date_shot_pattern = re.search(r'(\d{8}--\d{5})', base_name)
                            if date_shot_pattern:
                                date_shot = date_shot_pattern.group(1)
                                for key in param_data.keys():
                                    if date_shot in str(key):
                                        laser_energy = param_data[key].get('Laser_Target_Energy (mJ)', None)
                                        waveplate_angle = param_data[key].get('Waveplate_Angle (Degrees)', None)
                                        break
                    
                    if laser_energy is not None and waveplate_angle is not None:
                        try:
                            energy_float = float(laser_energy)
                            angle_float = float(waveplate_angle)
                            data_points.append((energy_float, mean_velocity, angle_float))
                        except (ValueError, TypeError):
                            self.progress.emit(f"  Warning: Invalid energy/angle value for {base_name}")
                    else:
                        if laser_energy is None:
                            self.progress.emit(f"  Warning: No laser energy found for {base_name}")
                        if waveplate_angle is None:
                            self.progress.emit(f"  Warning: No waveplate angle found for {base_name}")
                
                except Exception as e:
                    self.progress.emit(f"  Error processing {os.path.basename(file_path)}: {str(e)[:50]}")
                    continue
            
            if not data_points:
                self.progress.emit("⚠ No valid laser energy vs velocity data found")
                return
            
            # Extract unique waveplate angles and assign colors
            unique_angles = sorted(list(set([dp[2] for dp in data_points])))
            self.progress.emit(f"Found {len(data_points)} data points with {len(unique_angles)} unique waveplate angles")
            
            # Use tab10 colormap for distinct colors
            cmap = plt.get_cmap('tab10')
            angle_colors = {}
            for i, angle in enumerate(unique_angles):
                angle_colors[angle] = cmap(i / max(len(unique_angles), 1))
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot each data point colored by waveplate angle
            for energy, velocity, angle in data_points:
                ax.scatter(energy, velocity, 
                          color=angle_colors[angle], 
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Laser Target Energy (mJ)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Flyer Impact Velocity (m/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'Laser Energy vs Flyer Impact Velocity\n(Mean velocity 250-300ns after t=0 at {align_threshold:.0f} m/s, Color-coded by Waveplate Angle)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Create legend for waveplate angles
            legend_patches = [mpatches.Patch(color=angle_colors[angle], label=f'{angle}°') 
                            for angle in unique_angles]
            ax.legend(handles=legend_patches, title='Waveplate Angle', 
                     loc='best', fontsize=10, title_fontsize=11, framealpha=0.9)
            
            # Add text box with statistics
            textstr = f'Total points: {len(data_points)}\n'
            textstr += f'Alignment: t=0 at {align_threshold:.0f} m/s\n'
            textstr += f'Time window: 250-300ns after t=0\n'
            textstr += f'Energy range: {min(dp[0] for dp in data_points):.1f} - {max(dp[0] for dp in data_points):.1f} mJ\n'
            textstr += f'Velocity range: {min(dp[1] for dp in data_points):.1f} - {max(dp[1] for dp in data_points):.1f} m/s'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(spade_out, 'laser_energy_vs_impact_velocity.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.progress.emit(f"✓ Saved: {plot_path}")
            self.progress.emit(f"✓ Plotted {len(data_points)} data points across {len(unique_angles)} waveplate angles")
            
        except Exception as e:
            self.progress.emit(f"⚠ Error generating energy vs velocity plot: {str(e)[:80]}")
            import traceback
            self.progress.emit(traceback.format_exc()[:200])
    
    def _load_param_files(self):
        """Load parameter files using same logic as get_param_file_data"""
        param_data = {}
        
        if not self.param_folder or not os.path.exists(self.param_folder):
            return param_data
        
        try:
            import pandas as pd
            import re
            
            # Find all Excel and CSV files in the folder
            param_files = []
            for file in os.listdir(self.param_folder):
                if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                    param_files.append(os.path.join(self.param_folder, file))
            
            if not param_files:
                return param_data
            
            # Combine data from all parameter files
            combined_param_data = {}
            
            for param_file_path in param_files:
                if not os.path.exists(param_file_path):
                    continue
                
                try:
                    # Try to read the file (Excel or CSV)
                    if param_file_path.lower().endswith('.csv'):
                        df = pd.read_csv(param_file_path)
                    else:
                        try:
                            import openpyxl
                        except ImportError:
                            continue
                        df = pd.read_excel(param_file_path)
                    
                    # Handle different possible column names for PDV file name
                    pdv_col = None
                    # First pass: exact-ish known variants, ignoring spaces/underscores/dashes
                    normalized_columns = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in df.columns}
                    known_variants = [
                        'pdvfilename', 'pdvfile', 'pdv_file', 'pdv_file_name', 'pdv file name',
                        'dvfilename', 'dv_file', 'dvfile', 'filename', 'file_name', 'file name'
                    ]
                    normalized_variants = {re.sub(r"[^a-z0-9]", "", v): v for v in known_variants}
                    for col, norm in normalized_columns.items():
                        if norm in normalized_variants:
                            pdv_col = col
                            break
                    # Second pass: heuristic containing tokens 'pdv' and ('file' or 'name')
                    if pdv_col is None:
                        for col in df.columns:
                            col_lower = col.lower()
                            if ('pdv' in col_lower or 'dv' in col_lower) and ('file' in col_lower or 'name' in col_lower):
                                pdv_col = col
                                break
                    # Final fallback: a standalone 'filename' or 'file name' column
                    if pdv_col is None:
                        for col in df.columns:
                            if col.strip().lower() in ['filename', 'file name', 'file_name']:
                                pdv_col = col
                                break
                    
                    if pdv_col is None:
                        continue
                    
                    # Create mapping for each experiment
                    for idx, row in df.iterrows():
                        pdv_file = row[pdv_col]
                        if pd.isna(pdv_file) or pdv_file == 0:
                            continue
                        
                        # Convert PDV file name to string to ensure consistency
                        pdv_file_str = str(pdv_file).strip()
                        
                        # Clean the filename for better matching
                        # Remove common extensions and clean up the name
                        clean_pdv_file = pdv_file_str
                        for ext in ['.csv', '.txt', '.dat', '.xlsx', '.xls']:
                            if clean_pdv_file.lower().endswith(ext):
                                clean_pdv_file = clean_pdv_file[:-len(ext)]
                        
                        # Extract ALL columns from the row (except the PDV filename column itself)
                        exp_info = {}
                        for col in df.columns:
                            if col != pdv_col:  # Skip the PDV filename column
                                value = row.get(col)
                                if not pd.isna(value):  # Only include non-NaN values
                                    exp_info[col] = value
                        
                        # Store both original and cleaned versions for better matching
                        combined_param_data[pdv_file_str] = exp_info
                        if clean_pdv_file != pdv_file_str:
                            combined_param_data[clean_pdv_file] = exp_info
                        
                        # Also store with common variations for better matching
                        # Remove date patterns if present
                        date_cleaned = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', '', clean_pdv_file)
                        if date_cleaned != clean_pdv_file:
                            combined_param_data[date_cleaned] = exp_info
                
                except Exception:
                    continue
            
            param_data = combined_param_data
        except Exception:
            pass
        
        return param_data

class HELIXAnalysisToolbox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = 'light'
        self.config_file = os.path.join(os.path.expanduser('~'), '.helix_analysis_toolbox_config.json')
        self.spade_params = {}  # Initialize spade_params dict
        self.init_ui()
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("HELIX Analysis Toolbox")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)  # Add margins around the main layout
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Increase tab size for better text fit (increased font size by 25%)
        # Enhanced checkbox styling for better usability
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                min-width: 200px;
                min-height: 40px;
                font-size: 16px;
                padding: 10px 20px;
            }
            
            QCheckBox {
                spacing: 10px;
                font-size: 14px;
                padding: 5px;
                min-height: 25px;
            }
            
            QCheckBox:hover {
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 3px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 2px solid #999;
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #555;
                background-color: #f0f0f0;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 2px solid #2196F3;
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #1976D2;
                border: 2px solid #1976D2;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: white;
            }
            
            QCheckBox:disabled {
                color: #999;
            }
        """)
        
        # Create tabs
        self.create_file_selection_tab()
        self.create_analysis_mode_tab()
        self.create_alpss_params_tab()
        self.create_spade_params_tab()
        self.create_post_processing_tab()
        self.create_control_tab()
        self.create_documentation_tab()
        
        # Theme switcher at bottom
        theme_layout = QHBoxLayout()
        theme_layout.addStretch()
        self.theme_switch = QCheckBox("Dark Theme")
        self.theme_switch.stateChanged.connect(self.toggle_theme)
        theme_layout.addWidget(self.theme_switch)
        main_layout.addLayout(theme_layout)
        
        # Set initial theme
        self.apply_theme('light')
        
        # Initialize analysis thread
        self.analysis_thread = None
        
    def save_settings(self):
        """Save current settings to configuration file"""
        try:
            settings = {
                'file_paths': {
                    'input_files': self.get_input_files(),
                    'output_dir': self.output_path.text() if hasattr(self, 'output_path') else '',
                    'param_folder': self.param_folder_path.text() if hasattr(self, 'param_folder_path') else '',
                    'spade_input': self.spade_input_path.text() if hasattr(self, 'spade_input_path') else ''
                },
                'alpss_params': self.get_alpss_params() if hasattr(self, 'get_alpss_params') else {},
                'spade_params': self.get_spade_params() if hasattr(self, 'get_spade_params') else {},
                'ui_settings': {
                    'theme': self.current_theme,
                    'file_mode': self.file_mode_combo.currentText() if hasattr(self, 'file_mode_combo') else 'Single File',
                    'analysis_mode': self.mode_alpss_only.isChecked() if hasattr(self, 'mode_alpss_only') else True,
                    'spade_input_mode': self.spade_auto_mode.isChecked() if hasattr(self, 'spade_auto_mode') else True
                }
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def load_settings(self):
        """Load settings from configuration file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                
                # Load file paths
                if 'file_paths' in settings:
                    file_paths = settings['file_paths']
                    if hasattr(self, 'output_path') and 'output_dir' in file_paths:
                        self.output_path.setText(file_paths['output_dir'])
                    if hasattr(self, 'param_folder_path') and 'param_folder' in file_paths:
                        self.param_folder_path.setText(file_paths['param_folder'])
                    if hasattr(self, 'spade_input_path') and 'spade_input' in file_paths:
                        self.spade_input_path.setText(file_paths['spade_input'])
                
                # Load UI settings
                if 'ui_settings' in settings:
                    ui_settings = settings['ui_settings']
                    if 'theme' in ui_settings:
                        self.current_theme = ui_settings['theme']
                        if ui_settings['theme'] == 'dark':
                            self.theme_switch.setChecked(True)
                        self.apply_theme(ui_settings['theme'])
                    
                    if hasattr(self, 'file_mode_combo') and 'file_mode' in ui_settings:
                        index = self.file_mode_combo.findText(ui_settings['file_mode'])
                        if index >= 0:
                            self.file_mode_combo.setCurrentIndex(index)
                    
                    if hasattr(self, 'mode_alpss_only') and 'analysis_mode' in ui_settings:
                        if ui_settings['analysis_mode']:
                            self.mode_alpss_only.setChecked(True)
                        else:
                            self.mode_both.setChecked(True)
                    
                    if hasattr(self, 'spade_auto_mode') and 'spade_input_mode' in ui_settings:
                        if ui_settings['spade_input_mode']:
                            self.spade_auto_mode.setChecked(True)
                        else:
                            self.spade_manual_mode.setChecked(True)
                
                # Load ALPSS parameters
                if 'alpss_params' in settings and hasattr(self, 'get_alpss_params'):
                    alpss_params = settings['alpss_params']
                    # Apply ALPSS parameters to UI widgets
                    self.apply_alpss_params(alpss_params)
                
                # Load SPADE parameters
                if 'spade_params' in settings and hasattr(self, 'get_spade_params'):
                    spade_params = settings['spade_params']
                    # Apply SPADE parameters to UI widgets
                    self.apply_spade_params(spade_params)
                    
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def apply_alpss_params(self, params):
        """Apply ALPSS parameters to UI widgets"""
        try:
            if hasattr(self, 'carrier_frequency') and 'carrier_frequency' in params:
                self.carrier_frequency.setValue(params['carrier_frequency'])
            if hasattr(self, 'signal_start_time') and 'signal_start_time' in params:
                self.signal_start_time.setValue(params['signal_start_time'])
            if hasattr(self, 'smoothing_characteristic_time') and 'smoothing_characteristic_time' in params:
                self.smoothing_characteristic_time.setValue(params['smoothing_characteristic_time'])
            if hasattr(self, 'save_combined_plot') and 'save_combined_plot' in params:
                if hasattr(self.save_combined_plot, 'setChecked'):
                    self.save_combined_plot.setChecked(params['save_combined_plot'])
            if hasattr(self, 'save_iq_start_time_plot') and 'save_iq_start_time_plot' in params:
                if hasattr(self.save_iq_start_time_plot, 'setChecked'):
                    self.save_iq_start_time_plot.setChecked(params['save_iq_start_time_plot'])
            if hasattr(self, 'uncert_mult') and 'uncert_mult' in params:
                self.uncert_mult.setValue(params['uncert_mult'])
            if hasattr(self, 'spall_calculation') and 'spall_calculation' in params:
                if hasattr(self.spall_calculation, 'setChecked'):
                    self.spall_calculation.setChecked(params['spall_calculation'] == 'yes')
            # Apply output file selection parameters
            if hasattr(self, 'save_velocity_csv') and 'save_velocity_csv' in params:
                self.save_velocity_csv.setChecked(params['save_velocity_csv'])
            if hasattr(self, 'save_velocity_smooth_csv') and 'save_velocity_smooth_csv' in params:
                self.save_velocity_smooth_csv.setChecked(params['save_velocity_smooth_csv'])
            if hasattr(self, 'save_velocity_uncert_csv') and 'save_velocity_uncert_csv' in params:
                self.save_velocity_uncert_csv.setChecked(params['save_velocity_uncert_csv'])
            if hasattr(self, 'save_velocity_smooth_uncert_csv') and 'save_velocity_smooth_uncert_csv' in params:
                self.save_velocity_smooth_uncert_csv.setChecked(params['save_velocity_smooth_uncert_csv'])
            if hasattr(self, 'save_results_csv') and 'save_results_csv' in params:
                self.save_results_csv.setChecked(params['save_results_csv'])
            if hasattr(self, 'save_noise_csv') and 'save_noise_csv' in params:
                self.save_noise_csv.setChecked(params['save_noise_csv'])
            if hasattr(self, 'smart_selection_checkbox') and 'smart_selection_enabled' in params:
                self.smart_selection_checkbox.setChecked(params['smart_selection_enabled'])
            # IQ threshold factor
            if hasattr(self, 'iq_threshold_factor') and 'iq_threshold_factor' in params:
                try:
                    self.iq_threshold_factor.setValue(float(params['iq_threshold_factor']))
                except Exception:
                    pass
            # Common PDV/material parameters
            if hasattr(self, 'lam') and 'lam' in params:
                self.lam.setValue(params['lam'])
            if hasattr(self, 'C0') and 'C0' in params:
                self.C0.setValue(params['C0'])
            if hasattr(self, 'density') and 'density' in params:
                self.density.setValue(params['density'])
            if hasattr(self, 'delta_rho') and 'delta_rho' in params:
                self.delta_rho.setValue(params['delta_rho'])
            if hasattr(self, 'delta_C0') and 'delta_C0' in params:
                self.delta_C0.setValue(params['delta_C0'])
            if hasattr(self, 'delta_lam') and 'delta_lam' in params:
                self.delta_lam.setValue(params['delta_lam'])
            if hasattr(self, 'theta') and 'theta' in params:
                self.theta.setValue(params['theta'])
            if hasattr(self, 'delta_theta') and 'delta_theta' in params:
                self.delta_theta.setValue(params['delta_theta'])
        except Exception as e:
            print(f"Error applying ALPSS parameters: {e}")
    
    def apply_spade_params(self, params):
        """Apply SPADE parameters to UI widgets"""
        try:
            if hasattr(self, 'spade_density') and 'density' in params:
                self.spade_density.setValue(params['density'])
            if hasattr(self, 'spade_acoustic_velocity') and 'acoustic_velocity' in params:
                self.spade_acoustic_velocity.setValue(params['acoustic_velocity'])
            if hasattr(self, 'analysis_model') and 'analysis_model' in params:
                index = self.analysis_model.findText(params['analysis_model'])
                if index >= 0:
                    self.analysis_model.setCurrentIndex(index)
            if hasattr(self, 'prominence_factor') and 'prominence_factor' in params:
                self.prominence_factor.setValue(params['prominence_factor'])
            if hasattr(self, 'peak_distance_ns') and 'peak_distance_ns' in params:
                self.peak_distance_ns.setValue(params['peak_distance_ns'])
            if hasattr(self, 'spade_smooth_window') and 'smooth_window' in params:
                self.spade_smooth_window.setValue(params['smooth_window'])
            if hasattr(self, 'polyorder') and 'polyorder' in params:
                self.polyorder.setValue(params['polyorder'])
            if hasattr(self, 'plot_individual') and 'plot_individual' in params:
                if hasattr(self.plot_individual, 'setChecked'):
                    self.plot_individual.setChecked(params['plot_individual'])
            if hasattr(self, 'save_summary') and 'save_summary_table' in params:
                if hasattr(self.save_summary, 'setChecked'):
                    self.save_summary.setChecked(params['save_summary_table'])
            if hasattr(self, 'show_plots') and 'show_plots' in params:
                if hasattr(self.show_plots, 'setChecked'):
                    self.show_plots.setChecked(params['show_plots'])
            if hasattr(self, 'experiment_velocity_shots') and 'experiment_type' in params:
                if params['experiment_type'] == 'velocity_shots':
                    if hasattr(self.experiment_velocity_shots, 'setChecked'):
                        self.experiment_velocity_shots.setChecked(True)
                elif params['experiment_type'] == 'spall_analysis':
                    if hasattr(self.experiment_spall_analysis, 'setChecked'):
                        self.experiment_spall_analysis.setChecked(True)
        except Exception as e:
            print(f"Error applying SPADE parameters: {e}")
    
    def closeEvent(self, event):
        """Save settings when closing the application"""
        self.save_settings()
        event.accept()
        
    def toggle_theme(self, state):
        if state == Qt.Checked:
            self.apply_theme('dark')
        else:
            self.apply_theme('light')
        
    def apply_theme(self, theme):
        self.current_theme = theme
        if theme == 'dark':
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #181a1b;
                }
                QWidget {
                    background-color: #181a1b;
                }
                QTabWidget::pane {
                    border: 1px solid #444;
                    background-color: #181a1b;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #232629;
                    color: #f0f0f0;
                    min-width: 200px;
                    min-height: 40px;
                    font-size: 16px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: #181a1b;
                    border-bottom: 2px solid #0078d4;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 15px;
                    border: 2px solid #444;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #f0f0f0;
                    background-color: #232629;
                }
                QGroupBox::title {
                    color: #f0f0f0;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    font-size: 14px;
                    color: #f0f0f0;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 14px;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border: 2px solid #0078d4;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                QPushButton:disabled {
                    background-color: #444;
                    color: #888;
                }
                QCheckBox {
                    spacing: 10px;
                    font-size: 14px;
                    color: #f0f0f0;
                    padding: 5px;
                    min-height: 25px;
                    background: transparent;
                }
                QCheckBox:hover {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 3px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 3px;
                    border: 2px solid #666;
                    background-color: #232629;
                }
                QCheckBox::indicator:hover {
                    border: 2px solid #0078d4;
                    background-color: #2a2d30;
                }
                QCheckBox::indicator:checked {
                    background-color: #0078d4;
                    border: 2px solid #0078d4;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #106ebe;
                    border: 2px solid #106ebe;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #232629;
                }
                QCheckBox:disabled {
                    color: #666;
                }
                QRadioButton {
                    font-size: 14px;
                    color: #f0f0f0;
                    spacing: 8px;
                    background: transparent;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #888;
                    border-radius: 8px;
                    background: transparent;
                }
                QRadioButton::indicator:checked {
                    border: 1px solid #0078d4;
                    background-color: #0078d4;
                }
                QRadioButton::indicator:focus {
                    border: 2px solid #0078d4;
                }
                QTextEdit {
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 14px;
                }
                QScrollArea {
                    border: none;
                    background: #181a1b;
                }
                QProgressBar {
                    border: 1px solid #444;
                    border-radius: 4px;
                    text-align: center;
                    font-weight: bold;
                    color: #f0f0f0;
                    background: #232629;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #f5f5f5;
                }
                QWidget {
                    background-color: #f5f5f5;
                }
                QTabWidget::pane {
                    border: 1px solid #c0c0c0;
                    background-color: white;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #e0e0e0;
                    color: #2c2c2c;
                    min-width: 200px;
                    min-height: 40px;
                    font-size: 16px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: white;
                    border-bottom: 2px solid #0078d4;
                    color: #2c2c2c;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 15px;
                    border: 2px solid #c0c0c0;
                    border-radius: 6px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #2c2c2c;
                    background-color: white;
                }
                QGroupBox::title {
                    color: #2c2c2c;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
                QLabel {
                    font-size: 14px;
                    color: #2c2c2c;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 14px;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border: 2px solid #0078d4;
                }
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
                QPushButton:disabled {
                    background-color: #c0c0c0;
                    color: #666666;
                }
                QCheckBox {
                    spacing: 10px;
                    font-size: 14px;
                    color: #2c2c2c;
                    padding: 5px;
                    min-height: 25px;
                    background: transparent;
                }
                QCheckBox:hover {
                    background-color: rgba(0, 0, 0, 0.05);
                    border-radius: 3px;
                }
                QCheckBox::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 3px;
                    border: 2px solid #999;
                    background-color: white;
                }
                QCheckBox::indicator:hover {
                    border: 2px solid #555;
                    background-color: #f0f0f0;
                }
                QCheckBox::indicator:checked {
                    background-color: #2196F3;
                    border: 2px solid #2196F3;
                }
                QCheckBox::indicator:checked:hover {
                    background-color: #1976D2;
                    border: 2px solid #1976D2;
                }
                QCheckBox::indicator:unchecked {
                    background-color: white;
                }
                QCheckBox:disabled {
                    color: #999;
                }
                QRadioButton {
                    font-size: 14px;
                    color: #2c2c2c;
                    spacing: 8px;
                    background: transparent;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #888;
                    border-radius: 8px;
                    background: transparent;
                }
                QRadioButton::indicator:checked {
                    border: 1px solid #0078d4;
                    background-color: #0078d4;
                }
                QRadioButton::indicator:focus {
                    border: 2px solid #0078d4;
                }
                QTextEdit {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 14px;
                }
                QScrollArea {
                    border: none;
                    background: #f5f5f5;
                }
                QProgressBar {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    text-align: center;
                    font-weight: bold;
                    color: #2c2c2c;
                    background: white;
                }
                QProgressBar::chunk {
                    background-color: #0078d4;
                    border-radius: 3px;
                }
            """)
        
    def create_file_selection_tab(self):
        """Create file selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Manual file selection (pick multiple files individually)
        single_file_layout = QHBoxLayout()
        self.single_file_radio = QCheckBox("Manual Select")
        self.single_file_radio.setChecked(True)
        self.single_file_radio.toggled.connect(self.on_file_mode_changed)
        single_file_layout.addWidget(self.single_file_radio)
        
        self.single_file_path = QLineEdit()
        self.single_file_path.setPlaceholderText("Select input file(s)...")
        single_file_layout.addWidget(self.single_file_path)
        
        self.single_file_btn = QPushButton("Browse")
        self.single_file_btn.clicked.connect(self.select_single_file)
        single_file_layout.addWidget(self.single_file_btn)
        
        file_layout.addLayout(single_file_layout)
        
        # Multiple files selection
        multi_file_layout = QHBoxLayout()
        self.multi_file_radio = QCheckBox("Multiple Files")
        self.multi_file_radio.toggled.connect(self.on_file_mode_changed)
        multi_file_layout.addWidget(self.multi_file_radio)
        
        self.multi_file_path = QLineEdit()
        self.multi_file_path.setPlaceholderText("Select input directory...")
        multi_file_layout.addWidget(self.multi_file_path)
        
        self.multi_file_btn = QPushButton("Browse")
        self.multi_file_btn.clicked.connect(self.select_multi_file_dir)
        multi_file_layout.addWidget(self.multi_file_btn)
        
        file_layout.addLayout(multi_file_layout)
        
        # File pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File Pattern:"))
        self.file_pattern = QLineEdit("*.csv")
        pattern_layout.addWidget(self.file_pattern)
        file_layout.addLayout(pattern_layout)
        
        layout.addWidget(file_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        
        self.output_path = QLineEdit()
        # Use a more robust default output path
        try:
            default_output = os.path.join(os.path.expanduser("~"), "ALPSS_SPADE_output")
        except:
            default_output = "output"
        self.output_path.setText(default_output)
        output_layout.addWidget(self.output_path)
        
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_btn)
        
        layout.addWidget(output_group)
        
        # Parameter file selection
        param_group = QGroupBox("Experiment Parameter Files (Optional)")
        param_layout = QVBoxLayout(param_group)
        
        # Description
        desc_label = QLabel("Select a folder containing parameter files (.xlsx/.xls/.csv) to link experiment data with processing results.")
        desc_label.setWordWrap(True)
        param_layout.addWidget(desc_label)
        
        # Parameter folder selection
        param_folder_layout = QHBoxLayout()
        self.param_folder_path = QLineEdit()
        self.param_folder_path.setPlaceholderText("Select parameter folder...")
        param_folder_layout.addWidget(self.param_folder_path)
        
        self.param_folder_btn = QPushButton("Browse")
        self.param_folder_btn.clicked.connect(self.select_param_folder)
        param_folder_layout.addWidget(self.param_folder_btn)
        
        param_layout.addLayout(param_folder_layout)
        
        # Parameter file info
        self.param_file_info = QTextEdit()
        self.param_file_info.setMaximumHeight(100)
        self.param_file_info.setPlaceholderText("Parameter folder information will appear here...")
        self.param_file_info.setReadOnly(True)
        param_layout.addWidget(QLabel("Parameter Folder Info:"))
        param_layout.addWidget(self.param_file_info)
        
        layout.addWidget(param_group)
        
        # File list display
        self.file_list = QPlainTextEdit()
        self.file_list.setMaximumHeight(200)
        self.file_list.setPlaceholderText("Selected files will appear here...")
        self.file_list.setReadOnly(True)
        layout.addWidget(QLabel("Selected Files:"))
        layout.addWidget(self.file_list)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "File Selection")
        
    def create_analysis_mode_tab(self):
        """Create analysis mode selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Analysis mode group
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        # Radio buttons for different modes (mutually exclusive)
        self.mode_button_group = QButtonGroup(mode_group)
        self.mode_button_group.setExclusive(True)
        
        self.mode_alpss_only = QRadioButton("ALPSS Only")
        self.mode_button_group.addButton(self.mode_alpss_only)
        mode_layout.addWidget(self.mode_alpss_only)
        
        self.mode_spade_only = QRadioButton("SPADE Only")
        self.mode_button_group.addButton(self.mode_spade_only)
        mode_layout.addWidget(self.mode_spade_only)
        
        self.mode_both = QRadioButton("ALPSS + SPADE (Combined)")
        self.mode_both.setChecked(True)
        self.mode_button_group.addButton(self.mode_both)
        mode_layout.addWidget(self.mode_both)
        
        self.mode_button_group.buttonToggled.connect(self.on_analysis_mode_changed)
        
        # Description text
        desc_text = QPlainTextEdit()
        desc_text.setMaximumHeight(150)
        desc_text.setReadOnly(True)
        desc_text.setPlainText(
            "ALPSS Only: Run ALPSS analysis on input files and save results.\n\n"
            "SPADE Only: Run SPADE analysis on manually selected velocity files.\n\n"
            "ALPSS + SPADE: Run ALPSS first, then automatically run SPADE on ALPSS outputs."
        )
        mode_layout.addWidget(desc_text)
        
        layout.addWidget(mode_group)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Analysis Mode")

    def create_post_processing_tab(self):
        """Create Post-Processing tab for quick plot edits after analysis"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        # Source selection
        src_group = QGroupBox("Source (ALPSS Output)")
        src_layout = QGridLayout(src_group)
        src_layout.setSpacing(10)

        src_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.pp_output_dir = QLineEdit()
        self.pp_output_dir.setPlaceholderText("Select ALPSS output directory...")
        src_layout.addWidget(self.pp_output_dir, 0, 1)
        self.pp_browse_output = QPushButton("Browse")
        self.pp_browse_output.clicked.connect(self.pp_select_output_dir)
        src_layout.addWidget(self.pp_browse_output, 0, 2)

        # Plot options
        opt_group = QGroupBox("Plot Options")
        opt_layout = QGridLayout(opt_group)
        opt_layout.setSpacing(10)

        self.pp_regen_combined = QCheckBox("Regenerate combined aligned velocity plot")
        self.pp_regen_combined.setChecked(True)
        opt_layout.addWidget(self.pp_regen_combined, 0, 0, 1, 2)

        self.pp_use_material_colors = QCheckBox("Color by Sample material (if available)")
        self.pp_use_material_colors.setChecked(True)
        self.pp_use_material_colors.stateChanged.connect(self.pp_on_material_color_changed)
        opt_layout.addWidget(self.pp_use_material_colors, 1, 0, 1, 2)

        self.pp_color_by_waveplate = QCheckBox("Color by Waveplate Angle (if available)")
        self.pp_color_by_waveplate.setChecked(False)
        self.pp_color_by_waveplate.stateChanged.connect(self.pp_on_waveplate_color_changed)
        opt_layout.addWidget(self.pp_color_by_waveplate, 2, 0, 1, 2)

        self.pp_color_by_laser_energy = QCheckBox("Color by Laser Target Energy (if available)")
        self.pp_color_by_laser_energy.setChecked(False)
        self.pp_color_by_laser_energy.stateChanged.connect(self.pp_on_laser_energy_color_changed)
        opt_layout.addWidget(self.pp_color_by_laser_energy, 3, 0, 1, 2)

        self.pp_laser_energy_vs_velocity = QCheckBox("Generate Laser Energy vs Impact Velocity plot")
        self.pp_laser_energy_vs_velocity.setChecked(False)
        self.pp_laser_energy_vs_velocity.setToolTip("Plot laser energy vs mean velocity (250-300 ns)")
        opt_layout.addWidget(self.pp_laser_energy_vs_velocity, 4, 0, 1, 2)

        opt_layout.addWidget(QLabel("Zoom Window (ns):"), 5, 0)
        self.pp_zoom_ns = QSpinBox()
        self.pp_zoom_ns.setRange(10, 10000)
        self.pp_zoom_ns.setValue(1000)
        opt_layout.addWidget(self.pp_zoom_ns, 5, 1)

        opt_layout.addWidget(QLabel("Alignment Threshold (m/s):"), 6, 0)
        self.pp_align_threshold = QDoubleSpinBox()
        self.pp_align_threshold.setRange(0.0, 1000.0)
        self.pp_align_threshold.setDecimals(2)
        self.pp_align_threshold.setValue(30.0)
        self.pp_align_threshold.setSuffix(" m/s")
        self.pp_align_threshold.setToolTip("Align traces to first velocity ≥ threshold (t=0)")
        opt_layout.addWidget(self.pp_align_threshold, 6, 1)

        # Axis limits
        axis_group = QGroupBox("Axis Limits")
        axis_layout = QGridLayout(axis_group)
        axis_layout.setSpacing(10)
        self.pp_auto_limits = QCheckBox("Auto")
        self.pp_auto_limits.setChecked(False)  # Default to custom limits
        axis_layout.addWidget(self.pp_auto_limits, 0, 0)

        # Main (top) subplot limits
        axis_layout.addWidget(QLabel("Main Subplot:"), 0, 1)
        axis_layout.addWidget(QLabel("X min (ns):"), 1, 0)
        self.pp_xmin = QDoubleSpinBox()
        self.pp_xmin.setRange(-1e6, 1e6)
        self.pp_xmin.setDecimals(2)
        self.pp_xmin.setValue(0.0)
        axis_layout.addWidget(self.pp_xmin, 1, 1)

        axis_layout.addWidget(QLabel("X max (ns):"), 1, 2)
        self.pp_xmax = QDoubleSpinBox()
        self.pp_xmax.setRange(-1e6, 1e6)
        self.pp_xmax.setDecimals(2)
        self.pp_xmax.setValue(100.0)
        axis_layout.addWidget(self.pp_xmax, 1, 3)

        axis_layout.addWidget(QLabel("Y min (m/s):"), 2, 0)
        self.pp_ymin = QDoubleSpinBox()
        self.pp_ymin.setRange(-1e6, 1e6)
        self.pp_ymin.setDecimals(2)
        self.pp_ymin.setValue(0.0)
        axis_layout.addWidget(self.pp_ymin, 2, 1)

        axis_layout.addWidget(QLabel("Y max (m/s):"), 2, 2)
        self.pp_ymax = QDoubleSpinBox()
        self.pp_ymax.setRange(-1e6, 1e6)
        self.pp_ymax.setDecimals(2)
        self.pp_ymax.setValue(600.0)
        axis_layout.addWidget(self.pp_ymax, 2, 3)

        # Zoom (bottom) subplot limits
        axis_layout.addWidget(QLabel("Zoom Subplot:"), 3, 1)
        axis_layout.addWidget(QLabel("X min (ns):"), 4, 0)
        self.pp_zoom_xmin = QDoubleSpinBox()
        self.pp_zoom_xmin.setRange(-1e6, 1e6)
        self.pp_zoom_xmin.setDecimals(2)
        self.pp_zoom_xmin.setValue(0.0)
        axis_layout.addWidget(self.pp_zoom_xmin, 4, 1)

        axis_layout.addWidget(QLabel("X max (ns):"), 4, 2)
        self.pp_zoom_xmax = QDoubleSpinBox()
        self.pp_zoom_xmax.setRange(-1e6, 1e6)
        self.pp_zoom_xmax.setDecimals(2)
        self.pp_zoom_xmax.setValue(50.0)
        axis_layout.addWidget(self.pp_zoom_xmax, 4, 3)

        axis_layout.addWidget(QLabel("Y min (m/s):"), 5, 0)
        self.pp_zoom_ymin = QDoubleSpinBox()
        self.pp_zoom_ymin.setRange(-1e6, 1e6)
        self.pp_zoom_ymin.setDecimals(2)
        self.pp_zoom_ymin.setValue(0.0)
        axis_layout.addWidget(self.pp_zoom_ymin, 5, 1)

        axis_layout.addWidget(QLabel("Y max (m/s):"), 5, 2)
        self.pp_zoom_ymax = QDoubleSpinBox()
        self.pp_zoom_ymax.setRange(-1e6, 1e6)
        self.pp_zoom_ymax.setDecimals(2)
        self.pp_zoom_ymax.setValue(300.0)
        axis_layout.addWidget(self.pp_zoom_ymax, 5, 3)

        # Actions
        action_layout = QHBoxLayout()
        self.pp_preview_btn = QPushButton("Preview")
        self.pp_preview_btn.clicked.connect(self.pp_preview_plots)
        self.pp_save_btn = QPushButton("Save")
        self.pp_save_btn.clicked.connect(self.pp_save_plots)
        action_layout.addStretch()
        action_layout.addWidget(self.pp_preview_btn)
        action_layout.addWidget(self.pp_save_btn)

        # Preview area
        self.pp_preview = QPlainTextEdit()
        self.pp_preview.setReadOnly(True)
        self.pp_preview.setMaximumHeight(120)
        self.pp_preview.setPlaceholderText("Preview log will appear here...")

        layout.addWidget(src_group)
        layout.addWidget(opt_group)
        layout.addWidget(axis_group)
        layout.addLayout(action_layout)
        layout.addWidget(self.pp_preview)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Post-Processing")

    def pp_on_material_color_changed(self, state):
        """Handle material color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_color_by_waveplate.blockSignals(True)
            self.pp_color_by_laser_energy.blockSignals(True)
            self.pp_color_by_waveplate.setChecked(False)
            self.pp_color_by_laser_energy.setChecked(False)
            self.pp_color_by_waveplate.blockSignals(False)
            self.pp_color_by_laser_energy.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_color_by_waveplate.isChecked() and not self.pp_color_by_laser_energy.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_use_material_colors.blockSignals(True)
                self.pp_use_material_colors.setChecked(True)
                self.pp_use_material_colors.blockSignals(False)

    def pp_on_waveplate_color_changed(self, state):
        """Handle waveplate angle color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_use_material_colors.blockSignals(True)
            self.pp_color_by_laser_energy.blockSignals(True)
            self.pp_use_material_colors.setChecked(False)
            self.pp_color_by_laser_energy.setChecked(False)
            self.pp_use_material_colors.blockSignals(False)
            self.pp_color_by_laser_energy.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_use_material_colors.isChecked() and not self.pp_color_by_laser_energy.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_color_by_waveplate.blockSignals(True)
                self.pp_color_by_waveplate.setChecked(True)
                self.pp_color_by_waveplate.blockSignals(False)

    def pp_on_laser_energy_color_changed(self, state):
        """Handle laser energy color checkbox - ensure mutual exclusivity"""
        if state == 2:  # Qt.Checked - uncheck others
            self.pp_use_material_colors.blockSignals(True)
            self.pp_color_by_waveplate.blockSignals(True)
            self.pp_use_material_colors.setChecked(False)
            self.pp_color_by_waveplate.setChecked(False)
            self.pp_use_material_colors.blockSignals(False)
            self.pp_color_by_waveplate.blockSignals(False)
        elif state == 0:  # Unchecked - prevent if it's the only one checked
            if not self.pp_use_material_colors.isChecked() and not self.pp_color_by_waveplate.isChecked():
                # Re-check this one since at least one must be checked
                self.pp_color_by_laser_energy.blockSignals(True)
                self.pp_color_by_laser_energy.setChecked(True)
                self.pp_color_by_laser_energy.blockSignals(False)

    def pp_select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select ALPSS Output Directory")
        if dir_path:
            self.pp_output_dir.setText(dir_path)

    def pp_apply_limits_to_spade_params(self):
        # Ensure we have a params dict to work with
        try:
            current_params = self.get_spade_params()
        except Exception:
            current_params = {}
        if not hasattr(self, 'spade_params') or not isinstance(getattr(self, 'spade_params'), dict):
            self.spade_params = current_params.copy()
        else:
            # Refresh base values from current UI before applying overrides
            self.spade_params.update(current_params)

        # Temporarily override SPADE params for plotting routines
        self.spade_params['auto_calculate_limits'] = self.pp_auto_limits.isChecked()
        if not self.pp_auto_limits.isChecked():
            # Main (top) subplot limits
            self.spade_params['x_min_main'] = self.pp_xmin.value()
            self.spade_params['x_max_main'] = self.pp_xmax.value()
            self.spade_params['y_min_main'] = self.pp_ymin.value()
            self.spade_params['y_max_main'] = self.pp_ymax.value()
            
            # Zoom (bottom) subplot limits
            self.spade_params['x_min_zoom'] = self.pp_zoom_xmin.value()
            self.spade_params['x_max_zoom'] = self.pp_zoom_xmax.value()
            self.spade_params['y_min_zoom'] = self.pp_zoom_ymin.value()
            self.spade_params['y_max_zoom'] = self.pp_zoom_ymax.value()
        
        self.spade_params['zoom_window_ns'] = self.pp_zoom_ns.value()
        self.spade_params['align_velocity_threshold_ms'] = self.pp_align_threshold.value()
        
        # Color coding options
        self.spade_params['use_material_colors'] = self.pp_use_material_colors.isChecked()
        self.spade_params['color_by_waveplate'] = self.pp_color_by_waveplate.isChecked()
        self.spade_params['color_by_laser_energy'] = self.pp_color_by_laser_energy.isChecked()
        
        # Additional plot options
        self.spade_params['laser_energy_vs_velocity'] = self.pp_laser_energy_vs_velocity.isChecked()

        # Debug: Log applied parameters
        self.progress_text.appendPlainText(f"[POST-PROCESSING] Parameters applied:")
        self.progress_text.appendPlainText(f"  auto_calc_limits: {self.spade_params.get('auto_calculate_limits')}")
        self.progress_text.appendPlainText(f"  x_min/max_main: {self.spade_params.get('x_min_main')}/{self.spade_params.get('x_max_main')}")
        self.progress_text.appendPlainText(f"  y_min/max_main: {self.spade_params.get('y_min_main')}/{self.spade_params.get('y_max_main')}")

    def pp_preview_plots(self):
        try:
            out_dir = self.pp_output_dir.text().strip() or self.output_path.text().strip()
            if not out_dir or not os.path.exists(out_dir):
                QMessageBox.warning(self, "Invalid Output Directory", "Please select a valid ALPSS output directory.")
                return
            
            self.pp_preview.appendPlainText("Starting preview generation in background...")
            self.pp_preview_btn.setEnabled(False)
            self.pp_save_btn.setEnabled(False)
            
            # Build spade_params from current settings
            self.pp_apply_limits_to_spade_params()
            
            # Get parameter folder from File Selection tab
            param_folder = self.param_folder_path.text() if hasattr(self, 'param_folder_path') else ""
            
            # Run in background thread
            self.pp_worker = PostProcessingWorker(out_dir, self.spade_params, param_folder)
            self.pp_thread = QThread()
            self.pp_worker.moveToThread(self.pp_thread)
            self.pp_worker.progress.connect(self.pp_on_progress)
            self.pp_worker.finished.connect(self.pp_on_finished)
            self.pp_worker.finished.connect(self.pp_thread.quit)
            self.pp_thread.started.connect(lambda: self.pp_worker.regenerate_plots(self.spade_params))
            self.pp_thread.start()
        except Exception as e:
            self.pp_preview.appendPlainText(f"Error: {e}")
            self.pp_preview_btn.setEnabled(True)
            self.pp_save_btn.setEnabled(True)

    def pp_save_plots(self):
        try:
            out_dir = self.pp_output_dir.text().strip() or self.output_path.text().strip()
            if not out_dir or not os.path.exists(out_dir):
                QMessageBox.warning(self, "Invalid Output Directory", "Please select a valid ALPSS output directory.")
                return
            
            self.pp_preview.appendPlainText("Starting plot save in background...")
            self.pp_preview_btn.setEnabled(False)
            self.pp_save_btn.setEnabled(False)
            
            # Build spade_params from current settings
            self.pp_apply_limits_to_spade_params()
            
            # Get parameter folder from File Selection tab
            param_folder = self.param_folder_path.text() if hasattr(self, 'param_folder_path') else ""
            
            # Run in background thread (same as preview, just a label difference)
            self.pp_worker = PostProcessingWorker(out_dir, self.spade_params, param_folder)
            self.pp_thread = QThread()
            self.pp_worker.moveToThread(self.pp_thread)
            self.pp_worker.progress.connect(self.pp_on_progress)
            self.pp_worker.finished.connect(self.pp_on_finished)
            self.pp_worker.finished.connect(self.pp_thread.quit)
            self.pp_thread.started.connect(lambda: self.pp_worker.regenerate_plots(self.spade_params))
            self.pp_thread.start()
        except Exception as e:
            self.pp_preview.appendPlainText(f"Error: {e}")
            self.pp_preview_btn.setEnabled(True)
            self.pp_save_btn.setEnabled(True)

    def pp_on_progress(self, msg):
        """Receive progress updates from post-processing worker"""
        self.pp_preview.appendPlainText(msg)
        QApplication.processEvents()

    def pp_on_finished(self):
        """Re-enable buttons when post-processing finishes"""
        self.pp_preview_btn.setEnabled(True)
        self.pp_save_btn.setEnabled(True)
        self.pp_preview.appendPlainText("✓ Complete!")

    def pp_regenerate_velocity_plot(self, input_path, spade_output_dir):
        """Regenerate all velocity traces plot with current axis settings for post-processing."""
        import glob
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
        files = glob.glob(pattern, recursive=True)
        files = [f for f in files if os.path.getsize(f) > 0]
        if not files:
            self.pp_preview.appendPlainText("No velocity files found for regeneration")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, max(1, len(files))))

        traces_plotted = 0
        align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
        zoom_ns = int(self.spade_params.get('zoom_window_ns', 1000))

        for i, file_path in enumerate(sorted(files)):
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] < 3:
                    continue
                time_data = df.iloc[:, 0].values
                velocity_data = df.iloc[:, 1].values
                uncertainty_data = df.iloc[:, 2].values

                # Convert time to ns if needed
                if np.nanmax(time_data) < 1e-3:
                    time_data = time_data * 1e9
                elif np.nanmax(time_data) < 1.0:
                    time_data = time_data * 1e3

                # Noise fraction filtering
                noise_file = file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
                high_noise_mask = None
                if os.path.exists(noise_file):
                    try:
                        df_noise = pd.read_csv(noise_file)
                        if df_noise.shape[1] >= 1:
                            noise_fraction = df_noise.iloc[:, -1].values
                            if len(noise_fraction) == len(velocity_data):
                                high_noise_mask = noise_fraction > 1.0
                    except Exception:
                        pass

                valid_mask = ~np.isnan(velocity_data)
                if high_noise_mask is not None:
                    valid_mask &= (~high_noise_mask)
                # Uncertainty threshold filtering
                uncertainty_threshold = self.spade_params.get('uncertainty_threshold_ms', 50.0)
                if uncertainty_data is not None:
                    valid_mask &= (uncertainty_data <= uncertainty_threshold)

                time_clean = time_data[valid_mask]
                velocity_clean = velocity_data[valid_mask]
                uncert_clean = uncertainty_data[valid_mask] if uncertainty_data is not None else None
                if len(time_clean) == 0:
                    continue

                # Align at first >= threshold
                t0_idx = None
                for j, v in enumerate(velocity_clean):
                    if not np.isnan(v) and v >= align_threshold:
                        t0_idx = j
                        break
                if t0_idx is not None:
                    t0 = time_clean[t0_idx]
                    time_clean = time_clean - t0

                color = colors[i % len(colors)]
                ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1)
                # Optional uncertainty bands
                if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None:
                    alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                    ax1.fill_between(time_clean,
                                     velocity_clean - uncert_clean,
                                     velocity_clean + uncert_clean,
                                     color=color, alpha=alpha)

                # Bottom zoom window
                mask_zoom = time_clean <= zoom_ns
                if np.any(mask_zoom):
                    ax2.plot(time_clean[mask_zoom], velocity_clean[mask_zoom], color=color, alpha=0.7, linewidth=1)
                    if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None:
                        alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                        ax2.fill_between(time_clean[mask_zoom],
                                         (velocity_clean - uncert_clean)[mask_zoom],
                                         (velocity_clean + uncert_clean)[mask_zoom],
                                         color=color, alpha=alpha)

                traces_plotted += 1
            except Exception as e:
                self.pp_preview.appendPlainText(f"Warning: Could not process {os.path.basename(file_path)}: {str(e)[:50]}")
                continue

        # Labels and axis limits
        ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'All Velocity Traces (Aligned) - {traces_plotted} traces', fontsize=14)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Apply axis limits from post-processing settings
        try:
            if not self.spade_params.get('auto_calculate_limits', True):
                x_min_main = float(self.spade_params.get('x_min_main', 0))
                x_max_main = float(self.spade_params.get('x_max_main', 100))
                y_min_main = float(self.spade_params.get('y_min_main', 0))
                y_max_main = float(self.spade_params.get('y_max_main', 600))
                ax1.set_xlim(x_min_main, x_max_main)
                ax1.set_ylim(y_min_main, y_max_main)

                x_min_zoom = float(self.spade_params.get('x_min_zoom', 0))
                x_max_zoom = float(self.spade_params.get('x_max_zoom', zoom_ns))
                y_min_zoom = float(self.spade_params.get('y_min_zoom', y_min_main))
                y_max_zoom = float(self.spade_params.get('y_max_zoom', y_max_main))
                ax2.set_xlim(x_min_zoom, x_max_zoom)
                ax2.set_ylim(y_min_zoom, y_max_zoom)
                ax2.set_title(f'Zoomed Velocity Traces ({int(x_min_zoom)} to {int(x_max_zoom)} ns)', fontsize=14)
            else:
                ax2.set_xlim(0, zoom_ns)
                ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)
        except Exception as e:
            self.pp_preview.appendPlainText(f"Warning: Could not apply axis limits: {str(e)[:50]}")
            ax2.set_xlim(0, zoom_ns)
            ax2.set_title(f'First {zoom_ns} ns Velocity Traces', fontsize=14)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(spade_output_dir, 'all_velocity_traces.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.pp_preview.appendPlainText(f"Saved to: {plot_path}")

    def create_alpss_params_tab(self):
        """Create ALPSS parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
        # Config File Management
        config_group = QGroupBox("Configuration File Management")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        
        # Mode selection
        config_layout.addWidget(QLabel("Parameter Entry Mode:"), 0, 0)
        self.alpss_config_mode_group = QButtonGroup()
        self.alpss_manual_mode = QRadioButton("Manual Entry (use GUI controls)")
        self.alpss_config_mode = QRadioButton("Use Config File")
        self.alpss_config_mode_group.addButton(self.alpss_manual_mode)
        self.alpss_config_mode_group.addButton(self.alpss_config_mode)
        self.alpss_manual_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.alpss_manual_mode)
        mode_layout.addWidget(self.alpss_config_mode)
        config_layout.addLayout(mode_layout, 0, 1, 1, 3)
        
        # Config file path
        config_layout.addWidget(QLabel("Config File Path:"), 1, 0)
        self.alpss_config_path = QLineEdit()
        self.alpss_config_path.setPlaceholderText("Select a config file to load ALPSS parameters")
        self.alpss_config_path.setEnabled(False)
        config_layout.addWidget(self.alpss_config_path, 1, 1, 1, 2)
        
        self.alpss_config_browse_btn = QPushButton("Browse")
        self.alpss_config_browse_btn.clicked.connect(self.browse_alpss_config)
        self.alpss_config_browse_btn.setEnabled(False)
        config_layout.addWidget(self.alpss_config_browse_btn, 1, 3)
        
        # Load and Save buttons
        self.alpss_config_load_btn = QPushButton("Load Config")
        self.alpss_config_load_btn.clicked.connect(self.load_alpss_config)
        self.alpss_config_load_btn.setEnabled(False)
        config_layout.addWidget(self.alpss_config_load_btn, 2, 0)
        
        self.alpss_config_save_btn = QPushButton("Save Current Settings to Config")
        self.alpss_config_save_btn.clicked.connect(self.save_alpss_config)
        config_layout.addWidget(self.alpss_config_save_btn, 2, 1, 1, 3)
        
        # Connect mode change
        self.alpss_config_mode.toggled.connect(self.on_alpss_config_mode_changed)
        
        layout.addWidget(config_group)
        
        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QGridLayout(basic_group)
        basic_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        basic_layout.addWidget(QLabel("Save Data:"), 0, 0)
        self.save_data = QComboBox()
        self.save_data.addItems(["yes", "no"])
        self.save_data.setCurrentText("yes")
        basic_layout.addWidget(self.save_data, 0, 1)
        
        basic_layout.addWidget(QLabel("Display Plots:"), 0, 2)
        self.display_plots = QComboBox()
        self.display_plots.addItems(["yes", "no"])
        self.display_plots.setCurrentText("no")
        basic_layout.addWidget(self.display_plots, 0, 3)
        
        # Row 1
        basic_layout.addWidget(QLabel("Save ALPSS Plots:"), 1, 0)
        self.save_all_plots = QComboBox()
        self.save_all_plots.addItems(["no", "subfolder", "main_dir"])
        self.save_all_plots.setCurrentText("no")
        self.save_all_plots.setToolTip("'no': Only save CSV data files (unless individual plots are selected below). 'subfolder': Save plots in individual subfolders. 'main_dir': Save plots in main output directory.")
        basic_layout.addWidget(self.save_all_plots, 1, 1)
        
        # Row 2
        basic_layout.addWidget(QLabel("Spall Calculation:"), 2, 0)
        self.spall_calculation = QComboBox()
        self.spall_calculation.addItems(["yes", "no"])
        self.spall_calculation.setCurrentText("yes")
        basic_layout.addWidget(self.spall_calculation, 2, 1)
        
        basic_layout.addWidget(QLabel("Header Lines:"), 2, 2)
        self.header_lines = QSpinBox()
        self.header_lines.setRange(0, 100)
        self.header_lines.setValue(5)
        basic_layout.addWidget(self.header_lines, 2, 3)
        
        # Row 3
        basic_layout.addWidget(QLabel("Start Time User:"), 3, 0)
        self.start_time_user = QLineEdit("none")
        basic_layout.addWidget(self.start_time_user, 3, 1)
        
        basic_layout.addWidget(QLabel("Start Time Correction (s):"), 3, 2)
        self.start_time_correction = ScientificSpinBox()
        self.start_time_correction.setRange(-1e-3, 1e-3)
        self.start_time_correction.setValue(0e-9)
        basic_layout.addWidget(self.start_time_correction, 3, 3)
        
        layout.addWidget(basic_group)
        
        # Time parameters
        time_group = QGroupBox("Time Parameters")
        time_layout = QGridLayout(time_group)
        time_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        time_layout.addWidget(QLabel("Time to Skip (s):"), 0, 0)
        self.time_to_skip = ScientificSpinBox()
        self.time_to_skip.setRange(0, 1e-3)
        self.time_to_skip.setValue(0e-6)
        time_layout.addWidget(self.time_to_skip, 0, 1)
        
        time_layout.addWidget(QLabel("Time to Take (s):"), 0, 2)
        self.time_to_take = ScientificSpinBox()
        self.time_to_take.setRange(1e-9, 1e-3)
        self.time_to_take.setValue(10e-6)
        time_layout.addWidget(self.time_to_take, 0, 3)
        
        # Row 1
        time_layout.addWidget(QLabel("t_before (s):"), 1, 0)
        self.t_before = ScientificSpinBox()
        self.t_before.setRange(1e-12, 1e-6)
        self.t_before.setValue(10e-9)
        time_layout.addWidget(self.t_before, 1, 1)
        
        time_layout.addWidget(QLabel("t_after (s):"), 1, 2)
        self.t_after = ScientificSpinBox()
        self.t_after.setRange(1e-12, 1e-6)
        self.t_after.setValue(60e-9)
        time_layout.addWidget(self.t_after, 1, 3)

        # Row 2
        time_layout.addWidget(QLabel("IQ Threshold Factor:"), 2, 0)
        self.iq_threshold_factor = QDoubleSpinBox()
        self.iq_threshold_factor.setRange(0.0, 2.0)
        self.iq_threshold_factor.setDecimals(3)
        self.iq_threshold_factor.setSingleStep(0.05)
        self.iq_threshold_factor.setValue(0.4)
        self.iq_threshold_factor.setToolTip("Fraction of initial IQ amplitude used to detect start time (default 0.4)")
        time_layout.addWidget(self.iq_threshold_factor, 2, 1)
        
        layout.addWidget(time_group)
        
        # Frequency parameters
        freq_group = QGroupBox("Frequency Parameters")
        freq_layout = QGridLayout(freq_group)
        freq_layout.setSpacing(10)  # Increase spacing between elements
        
        freq_layout.addWidget(QLabel("Freq Min (Hz):"), 0, 0)
        self.freq_min = ScientificSpinBox()
        self.freq_min.setRange(1e6, 10e9)
        self.freq_min.setValue(1e9)
        freq_layout.addWidget(self.freq_min, 0, 1)
        
        freq_layout.addWidget(QLabel("Freq Max (Hz):"), 0, 2)
        self.freq_max = ScientificSpinBox()
        self.freq_max.setRange(1e6, 10e9)
        self.freq_max.setValue(3.5e9)
        freq_layout.addWidget(self.freq_max, 0, 3)
        
        layout.addWidget(freq_group)
        
        # Smoothing parameters
        smooth_group = QGroupBox("Smoothing Parameters")
        smooth_layout = QGridLayout(smooth_group)
        smooth_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        smooth_layout.addWidget(QLabel("Smoothing Window:"), 0, 0)
        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(1, 10000)
        self.smoothing_window.setValue(601)
        smooth_layout.addWidget(self.smoothing_window, 0, 1)
        
        smooth_layout.addWidget(QLabel("Smoothing Width:"), 0, 2)
        self.smoothing_wid = QDoubleSpinBox()
        self.smoothing_wid.setRange(0.1, 100)
        self.smoothing_wid.setValue(3)
        self.smoothing_wid.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_wid, 0, 3)
        
        # Row 1
        smooth_layout.addWidget(QLabel("Smoothing Amp:"), 1, 0)
        self.smoothing_amp = QDoubleSpinBox()
        self.smoothing_amp.setRange(0.1, 10)
        self.smoothing_amp.setValue(1)
        self.smoothing_amp.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_amp, 1, 1)
        
        smooth_layout.addWidget(QLabel("Smoothing Sigma:"), 1, 2)
        self.smoothing_sigma = QDoubleSpinBox()
        self.smoothing_sigma.setRange(0.1, 10)
        self.smoothing_sigma.setValue(1)
        self.smoothing_sigma.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_sigma, 1, 3)
        
        # Row 2
        smooth_layout.addWidget(QLabel("Smoothing Mu:"), 2, 0)
        self.smoothing_mu = QDoubleSpinBox()
        self.smoothing_mu.setRange(-10, 10)
        self.smoothing_mu.setValue(0)
        self.smoothing_mu.setDecimals(1)
        smooth_layout.addWidget(self.smoothing_mu, 2, 1)
        
        layout.addWidget(smooth_group)
        
        # Peak detection parameters
        peak_group = QGroupBox("Peak Detection Parameters")
        peak_layout = QGridLayout(peak_group)
        peak_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        peak_layout.addWidget(QLabel("PB Neighbors:"), 0, 0)
        self.pb_neighbors = QSpinBox()
        self.pb_neighbors.setRange(1, 1000)  # Minimum value is 1, not 0
        self.pb_neighbors.setValue(400)
        self.pb_neighbors.setToolTip("Number of neighbors to compare when searching for pullback local minimum. Must be >= 1 (scipy requirement).")
        peak_layout.addWidget(self.pb_neighbors, 0, 1)
        
        peak_layout.addWidget(QLabel("PB Index Correction:"), 0, 2)
        self.pb_idx_correction = QSpinBox()
        self.pb_idx_correction.setRange(-100, 100)
        self.pb_idx_correction.setValue(0)
        peak_layout.addWidget(self.pb_idx_correction, 0, 3)
        
        # Row 1
        peak_layout.addWidget(QLabel("RC Neighbors:"), 1, 0)
        self.rc_neighbors = QSpinBox()
        self.rc_neighbors.setRange(1, 1000)  # Minimum value is 1, not 0
        self.rc_neighbors.setValue(400)
        self.rc_neighbors.setToolTip("Number of neighbors to compare when searching for recompression local maximum. Must be >= 1 (scipy requirement).")
        peak_layout.addWidget(self.rc_neighbors, 1, 1)
        
        peak_layout.addWidget(QLabel("RC Index Correction:"), 1, 2)
        self.rc_idx_correction = QSpinBox()
        self.rc_idx_correction.setRange(-100, 100)
        self.rc_idx_correction.setValue(0)
        peak_layout.addWidget(self.rc_idx_correction, 1, 3)
        
        layout.addWidget(peak_group)
        
        # STFT parameters
        stft_group = QGroupBox("STFT Parameters")
        stft_layout = QGridLayout(stft_group)
        stft_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        stft_layout.addWidget(QLabel("Sample Rate (Hz):"), 0, 0)
        self.sample_rate = ScientificSpinBox()
        self.sample_rate.setRange(1e6, 1e12)
        self.sample_rate.setValue(128e9)
        stft_layout.addWidget(self.sample_rate, 0, 1)
        
        stft_layout.addWidget(QLabel("Nperseg:"), 0, 2)
        self.nperseg = QSpinBox()
        self.nperseg.setRange(64, 4096)
        self.nperseg.setValue(512)
        stft_layout.addWidget(self.nperseg, 0, 3)
        
        # Row 1
        stft_layout.addWidget(QLabel("Noverlap:"), 1, 0)
        self.noverlap = QSpinBox()
        self.noverlap.setRange(0, 4096)
        self.noverlap.setValue(400)
        stft_layout.addWidget(self.noverlap, 1, 1)
        
        stft_layout.addWidget(QLabel("NFFT:"), 1, 2)
        self.nfft = QSpinBox()
        self.nfft.setRange(64, 8192)
        self.nfft.setValue(5120)
        stft_layout.addWidget(self.nfft, 1, 3)
        
        # Row 2
        stft_layout.addWidget(QLabel("Window:"), 2, 0)
        self.window = QComboBox()
        self.window.addItems(["hann", "hamming", "blackman", "bartlett"])
        self.window.setCurrentText("hann")
        stft_layout.addWidget(self.window, 2, 1)
        
        stft_layout.addWidget(QLabel("Carrier Band Time (s):"), 2, 2)
        self.carrier_band_time = ScientificSpinBox()
        self.carrier_band_time.setRange(1e-12, 1e-6)
        self.carrier_band_time.setValue(250e-9)
        stft_layout.addWidget(self.carrier_band_time, 2, 3)
        
        layout.addWidget(stft_group)
        
        # Blur parameters
        blur_group = QGroupBox("Blur Parameters")
        blur_layout = QGridLayout(blur_group)
        blur_layout.setSpacing(10)  # Increase spacing between elements
        
        blur_layout.addWidget(QLabel("Blur Kernel X:"), 0, 0)
        self.blur_kernel_x = QSpinBox()
        self.blur_kernel_x.setRange(1, 20)
        self.blur_kernel_x.setValue(5)
        blur_layout.addWidget(self.blur_kernel_x, 0, 1)
        
        blur_layout.addWidget(QLabel("Blur Kernel Y:"), 0, 2)
        self.blur_kernel_y = QSpinBox()
        self.blur_kernel_y.setRange(1, 20)
        self.blur_kernel_y.setValue(5)
        blur_layout.addWidget(self.blur_kernel_y, 0, 3)
        
        # Row 1
        blur_layout.addWidget(QLabel("Blur Sigma X:"), 1, 0)
        self.blur_sigx = QDoubleSpinBox()
        self.blur_sigx.setRange(0, 10)
        self.blur_sigx.setValue(0)
        self.blur_sigx.setDecimals(1)
        blur_layout.addWidget(self.blur_sigx, 1, 1)
        
        blur_layout.addWidget(QLabel("Blur Sigma Y:"), 1, 2)
        self.blur_sigy = QDoubleSpinBox()
        self.blur_sigy.setRange(0, 10)
        self.blur_sigy.setValue(0)
        self.blur_sigy.setDecimals(1)
        blur_layout.addWidget(self.blur_sigy, 1, 3)
        
        layout.addWidget(blur_group)
        
        # Filter parameters
        filter_group = QGroupBox("Filter Parameters")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0 - Add notch filter toggle
        filter_layout.addWidget(QLabel("Use Gaussian Notch Filter:"), 0, 0)
        self.use_notch_filter = QCheckBox("Enable carrier frequency removal")
        self.use_notch_filter.setChecked(True)
        self.use_notch_filter.setToolTip("Remove carrier frequency using Gaussian notch filter. Disable if signal is weak or carrier/signal frequencies are close.")
        filter_layout.addWidget(self.use_notch_filter, 0, 1)
        
        filter_layout.addWidget(QLabel("Order:"), 0, 2)
        self.order = QSpinBox()
        self.order.setRange(1, 20)
        self.order.setValue(6)
        filter_layout.addWidget(self.order, 0, 3)
        
        # Row 1
        filter_layout.addWidget(QLabel("Width:"), 1, 0)
        self.wid = ScientificSpinBox()
        self.wid.setRange(1e3, 1e10)
        self.wid.setValue(15e4)
        filter_layout.addWidget(self.wid, 1, 1)
        
        filter_layout.addWidget(QLabel("Uncertainty Multiplier:"), 1, 2)
        self.uncert_mult = QDoubleSpinBox()
        self.uncert_mult.setRange(0.1, 100)
        self.uncert_mult.setValue(10)
        self.uncert_mult.setDecimals(1)
        filter_layout.addWidget(self.uncert_mult, 1, 3)
        
        # Row 2
        filter_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "plasma", "inferno", "magma", "jet"])
        self.cmap.setCurrentText("viridis")
        filter_layout.addWidget(self.cmap, 2, 1)
        
        layout.addWidget(filter_group)
        
        # Material parameters
        material_group = QGroupBox("Material Parameters")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        material_layout.addWidget(QLabel("Bulk Wavespeed (m/s):"), 0, 0)
        self.C0 = QDoubleSpinBox()
        self.C0.setRange(1000, 10000)
        self.C0.setValue(3950)
        self.C0.setDecimals(0)
        material_layout.addWidget(self.C0, 0, 1)
        
        # Row 1
        material_layout.addWidget(QLabel("Density (kg/m³):"), 1, 0)
        self.density = QDoubleSpinBox()
        self.density.setRange(100, 20000)
        self.density.setValue(8960)
        self.density.setDecimals(0)
        material_layout.addWidget(self.density, 1, 1)
        
        layout.addWidget(material_group)
        
        # PDV parameters
        pdv_group = QGroupBox("PDV Parameters")
        pdv_layout = QGridLayout(pdv_group)
        pdv_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        pdv_layout.addWidget(QLabel("Target Wavelength:"), 0, 0)
        self.lam = ScientificSpinBox()
        self.lam.setRange(1e-12, 1e-3)  # Wider range to allow more wavelengths
        self.lam.setValue(1550.016e-9)
        self.lam.setDecimals(15)  # Allow high precision for significant figures
        self.lam.setSingleStep(1e-12)  # Allow fine control with arrow keys
        self.lam.setSuffix(" m")  # Add units suffix
        pdv_layout.addWidget(self.lam, 0, 1)
        
        pdv_layout.addWidget(QLabel("Angle of Incidence (deg):"), 0, 2)
        self.theta = QDoubleSpinBox()
        self.theta.setRange(-90, 90)
        self.theta.setValue(0)
        self.theta.setDecimals(1)
        pdv_layout.addWidget(self.theta, 0, 3)
        
        layout.addWidget(pdv_group)
        
        # Uncertainty parameters
        uncert_group = QGroupBox("Uncertainty Parameters")
        uncert_layout = QGridLayout(uncert_group)
        uncert_layout.setSpacing(10)  # Increase spacing between elements
        
        # Row 0
        uncert_layout.addWidget(QLabel("Delta Density (kg/m³):"), 0, 0)
        self.delta_rho = QDoubleSpinBox()
        self.delta_rho.setRange(0, 1000)
        self.delta_rho.setValue(9)
        self.delta_rho.setDecimals(0)
        uncert_layout.addWidget(self.delta_rho, 0, 1)
        
        uncert_layout.addWidget(QLabel("Delta C0 (m/s):"), 0, 2)
        self.delta_C0 = QDoubleSpinBox()
        self.delta_C0.setRange(0, 1000)
        self.delta_C0.setValue(23)
        self.delta_C0.setDecimals(0)
        uncert_layout.addWidget(self.delta_C0, 0, 3)
        
        # Row 1
        uncert_layout.addWidget(QLabel("Delta Wavelength (m):"), 1, 0)
        self.delta_lam = QDoubleSpinBox()
        self.delta_lam.setRange(0, 1e-15)
        self.delta_lam.setValue(8e-18)
        self.delta_lam.setDecimals(18)
        uncert_layout.addWidget(self.delta_lam, 1, 1)
        
        uncert_layout.addWidget(QLabel("Delta Theta (deg):"), 1, 2)
        self.delta_theta = QDoubleSpinBox()
        self.delta_theta.setRange(0, 90)
        self.delta_theta.setValue(5)
        self.delta_theta.setDecimals(1)
        uncert_layout.addWidget(self.delta_theta, 1, 3)
        
        layout.addWidget(uncert_group)
        
        # Plot parameters
        plot_group = QGroupBox("Plot Parameters")
        plot_layout = QGridLayout(plot_group)
        plot_layout.setSpacing(10)  # Increase spacing between elements
        
        plot_layout.addWidget(QLabel("Figure Width:"), 0, 0)
        self.plot_width = QSpinBox()
        self.plot_width.setRange(5, 100)
        self.plot_width.setValue(30)
        plot_layout.addWidget(self.plot_width, 0, 1)
        
        plot_layout.addWidget(QLabel("Figure Height:"), 0, 2)
        self.plot_height = QSpinBox()
        self.plot_height.setRange(5, 100)
        self.plot_height.setValue(10)
        plot_layout.addWidget(self.plot_height, 0, 3)
        
        # Row 1
        plot_layout.addWidget(QLabel("DPI:"), 1, 0)
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(50, 600)
        self.plot_dpi.setValue(300)
        plot_layout.addWidget(self.plot_dpi, 1, 1)
        
        layout.addWidget(plot_group)
        
        # Image selection group
        image_group = QGroupBox("ALPSS Output Images")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(10)
        
        # Description
        desc_label = QLabel("Only the combined ALPSS summary plot and optional IQ start-time diagnostic are available in the current fast-plot workflow.")
        desc_label.setWordWrap(True)
        image_layout.addWidget(desc_label)
        
        self.save_combined_plot = QCheckBox("Combined ALPSS Summary Plot")
        self.save_combined_plot.setChecked(True)
        self.save_combined_plot.setToolTip("Generate the 5-panel summary plot (velocity, noise, ROI spectrograms, IQ analysis).")
        image_layout.addWidget(self.save_combined_plot)
        
        self.save_iq_start_time_plot = QCheckBox("IQ Start Time Detection Plot")
        self.save_iq_start_time_plot.setChecked(False)
        self.save_iq_start_time_plot.setToolTip("Generate IQ analysis start time detection plot with threshold overlays.")
        image_layout.addWidget(self.save_iq_start_time_plot)
        
        # Select all/none buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_images = QPushButton("Select All")
        self.select_all_images.clicked.connect(self.select_all_alpss_images)
        select_buttons_layout.addWidget(self.select_all_images)
        
        self.deselect_all_images = QPushButton("Deselect All")
        self.deselect_all_images.clicked.connect(self.deselect_all_alpss_images)
        select_buttons_layout.addWidget(self.deselect_all_images)
        
        image_layout.addLayout(select_buttons_layout)
        
        layout.addWidget(image_group)
        
        # Output file selection group
        output_files_group = QGroupBox("ALPSS Output Files")
        output_files_layout = QVBoxLayout(output_files_group)
        output_files_layout.setSpacing(10)
        
        # Description
        output_desc_label = QLabel("Select which ALPSS output files to save:")
        output_desc_label.setWordWrap(True)
        output_files_layout.addWidget(output_desc_label)
        
        # Output file checkboxes
        self.save_velocity_csv = QCheckBox("Velocity CSV (*--velocity.csv)")
        self.save_velocity_csv.setChecked(True)
        self.save_velocity_csv.setToolTip("Save raw velocity data")
        output_files_layout.addWidget(self.save_velocity_csv)
        
        self.save_velocity_smooth_csv = QCheckBox("Smoothed Velocity CSV (*--velocity--smooth.csv)")
        self.save_velocity_smooth_csv.setChecked(True)
        self.save_velocity_smooth_csv.setToolTip("Save smoothed velocity data")
        output_files_layout.addWidget(self.save_velocity_smooth_csv)
        
        self.save_velocity_uncert_csv = QCheckBox("Velocity with Uncertainty (*--vel--uncert.csv)")
        self.save_velocity_uncert_csv.setChecked(True)
        self.save_velocity_uncert_csv.setToolTip("Save velocity data with uncertainty")
        output_files_layout.addWidget(self.save_velocity_uncert_csv)
        
        self.save_velocity_smooth_uncert_csv = QCheckBox("Smoothed Velocity with Uncertainty (*--vel-smooth-with-uncert.csv)")
        self.save_velocity_smooth_uncert_csv.setChecked(True)
        self.save_velocity_smooth_uncert_csv.setToolTip("Save smoothed velocity with uncertainty (required for SPADE)")
        output_files_layout.addWidget(self.save_velocity_smooth_uncert_csv)
        
        self.save_results_csv = QCheckBox("Results CSV (*--results.csv)")
        self.save_results_csv.setChecked(True)
        self.save_results_csv.setToolTip("Save analysis results with uncertainties")
        output_files_layout.addWidget(self.save_results_csv)
        
        self.save_noise_csv = QCheckBox("Noise Fraction CSV (*--noise--frac.csv)")
        self.save_noise_csv.setChecked(True)
        self.save_noise_csv.setToolTip("Save noise fraction data")
        output_files_layout.addWidget(self.save_noise_csv)
        
        # Smart selection for combined mode
        self.smart_selection_checkbox = QCheckBox("Smart Selection for Combined Mode")
        self.smart_selection_checkbox.setChecked(True)
        self.smart_selection_checkbox.setToolTip("When ALPSS+SPADE mode is selected, automatically save files needed for SPADE analysis + enhanced filtering")
        output_files_layout.addWidget(self.smart_selection_checkbox)
        
        # Select all/none buttons for output files
        output_select_buttons_layout = QHBoxLayout()
        self.select_all_output_files = QPushButton("Select All")
        self.select_all_output_files.clicked.connect(self.select_all_output_files_func)
        output_select_buttons_layout.addWidget(self.select_all_output_files)
        
        self.deselect_all_output_files = QPushButton("Deselect All")
        self.deselect_all_output_files.clicked.connect(self.deselect_all_output_files_func)
        output_select_buttons_layout.addWidget(self.deselect_all_output_files)
        
        self.smart_select_output_files = QPushButton("Smart Select")
        self.smart_select_output_files.clicked.connect(self.smart_select_output_files_func)
        self.smart_select_output_files.setToolTip("Select files needed for SPADE analysis + enhanced filtering")
        output_select_buttons_layout.addWidget(self.smart_select_output_files)
        
        output_files_layout.addLayout(output_select_buttons_layout)
        
        layout.addWidget(output_files_group)
        
        scroll.setWidget(scroll_widget)
        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "ALPSS Parameters")
        
    def select_all_alpss_images(self):
        """Select all ALPSS output images"""
        self.save_combined_plot.setChecked(True)
        self.save_iq_start_time_plot.setChecked(True)
        
    def deselect_all_alpss_images(self):
        """Deselect all ALPSS output images"""
        self.save_combined_plot.setChecked(False)
        self.save_iq_start_time_plot.setChecked(False)
        
    def select_all_output_files_func(self):
        """Select all ALPSS output files"""
        self.save_velocity_csv.setChecked(True)
        self.save_velocity_smooth_csv.setChecked(True)
        self.save_velocity_uncert_csv.setChecked(True)
        self.save_velocity_smooth_uncert_csv.setChecked(True)
        self.save_results_csv.setChecked(True)
        self.save_noise_csv.setChecked(True)
        
    def deselect_all_output_files_func(self):
        """Deselect all ALPSS output files"""
        self.save_velocity_csv.setChecked(False)
        self.save_velocity_smooth_csv.setChecked(False)
        self.save_velocity_uncert_csv.setChecked(False)
        self.save_velocity_smooth_uncert_csv.setChecked(False)
        self.save_results_csv.setChecked(False)
        self.save_noise_csv.setChecked(False)
        
    def smart_select_output_files_func(self):
        """Smart select files needed for SPADE analysis + enhanced filtering"""
        # For SPADE analysis, we need the smoothed velocity with uncertainty file + noise file
        self.save_velocity_csv.setChecked(False)
        self.save_velocity_smooth_csv.setChecked(False)
        self.save_velocity_uncert_csv.setChecked(False)
        self.save_velocity_smooth_uncert_csv.setChecked(True)  # Main file SPADE needs
        self.save_results_csv.setChecked(False)
        self.save_noise_csv.setChecked(True)  # Also save noise file for enhanced filtering
        
    def create_spade_params_tab(self):
        """Create SPADE parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
        # Config File Management
        config_group = QGroupBox("Configuration File Management")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        
        # Mode selection
        config_layout.addWidget(QLabel("Parameter Entry Mode:"), 0, 0)
        self.spade_config_mode_group = QButtonGroup()
        self.spade_manual_mode = QRadioButton("Manual Entry (use GUI controls)")
        self.spade_config_mode = QRadioButton("Use Config File")
        self.spade_config_mode_group.addButton(self.spade_manual_mode)
        self.spade_config_mode_group.addButton(self.spade_config_mode)
        self.spade_manual_mode.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.spade_manual_mode)
        mode_layout.addWidget(self.spade_config_mode)
        config_layout.addLayout(mode_layout, 0, 1, 1, 3)
        
        # Config file path
        config_layout.addWidget(QLabel("Config File Path:"), 1, 0)
        self.spade_config_path = QLineEdit()
        self.spade_config_path.setPlaceholderText("Select a config file to load SPADE parameters")
        self.spade_config_path.setEnabled(False)
        config_layout.addWidget(self.spade_config_path, 1, 1, 1, 2)
        
        self.spade_config_browse_btn = QPushButton("Browse")
        self.spade_config_browse_btn.clicked.connect(self.browse_spade_config)
        self.spade_config_browse_btn.setEnabled(False)
        config_layout.addWidget(self.spade_config_browse_btn, 1, 3)
        
        # Load and Save buttons
        self.spade_config_load_btn = QPushButton("Load Config")
        self.spade_config_load_btn.clicked.connect(self.load_spade_config)
        self.spade_config_load_btn.setEnabled(False)
        config_layout.addWidget(self.spade_config_load_btn, 2, 0)
        
        self.spade_config_save_btn = QPushButton("Save Current Settings to Config")
        self.spade_config_save_btn.clicked.connect(self.save_spade_config)
        config_layout.addWidget(self.spade_config_save_btn, 2, 1, 1, 3)
        
        # Connect mode change
        self.spade_config_mode.toggled.connect(self.on_spade_config_mode_changed)
        
        layout.addWidget(config_group)
        
        # Experiment type selection
        experiment_group = QGroupBox("Experiment Type")
        experiment_layout = QVBoxLayout(experiment_group)
        experiment_layout.setSpacing(10)
        
        # Checkboxes for experiment type (can select both)
        self.experiment_velocity_shots = QCheckBox("Velocity Shots")
        self.experiment_velocity_shots.setChecked(True)
        experiment_layout.addWidget(self.experiment_velocity_shots)
        
        self.experiment_spall_analysis = QCheckBox("Spall Analysis")
        self.experiment_spall_analysis.setChecked(False)
        experiment_layout.addWidget(self.experiment_spall_analysis)

        self.experiment_hel_detection = QCheckBox("HEL Detection")
        self.experiment_hel_detection.setChecked(False)
        self.experiment_hel_detection.toggled.connect(self.on_hel_detection_toggled)
        experiment_layout.addWidget(self.experiment_hel_detection)
        
        # Description text for experiment types
        experiment_desc = QPlainTextEdit()
        experiment_desc.setMaximumHeight(120)
        experiment_desc.setReadOnly(True)
        experiment_desc.setPlainText(
            "Velocity Shots: Generate velocity shot summary with impact velocity calculations and combined velocity plot.\n"
            "Spall Analysis: Generate spall summary with spall strength and strain rate analysis.\n"
            "Note: You can select both options to run both analyses."
        )
        experiment_layout.addWidget(experiment_desc)
        
        layout.addWidget(experiment_group)
        
        # Material properties
        material_group = QGroupBox("Material Properties")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(10)  # Increase spacing between elements
        
        # Density
        material_layout.addWidget(QLabel("Default Density (kg/m³):"), 0, 0)
        self.spade_density = QDoubleSpinBox()
        self.spade_density.setRange(1000, 25000)
        self.spade_density.setDecimals(2)
        self.spade_density.setValue(8960)  # Default: Copper
        self.spade_density.setToolTip("Fallback value if material not found in database")
        material_layout.addWidget(self.spade_density, 0, 1)
        
        # Acoustic velocity
        material_layout.addWidget(QLabel("Default Acoustic Velocity (m/s):"), 0, 2)
        self.spade_acoustic_velocity = QDoubleSpinBox()
        self.spade_acoustic_velocity.setRange(1000, 10000)
        self.spade_acoustic_velocity.setDecimals(2)
        self.spade_acoustic_velocity.setValue(3950)  # Default: Copper
        self.spade_acoustic_velocity.setToolTip("Fallback value if material not found in database")
        material_layout.addWidget(self.spade_acoustic_velocity, 0, 3)
        
        # Add help text explaining material properties priority
        material_help = QPlainTextEdit()
        material_help.setMaximumHeight(80)
        material_help.setReadOnly(True)
        material_help.setPlainText(
            "📌 Material Properties Priority:\n"
            "1. Parameter file explicit columns (Density_kg_m3, Bulk_Wave_Speed_m_s) - HIGHEST\n"
            "2. Material Database lookup via 'Sample material' column - OVERRIDES GUI values\n"
            "3. GUI values above - FALLBACK only when material not found\n"
            "💡 Tip: Leave defaults as-is. System auto-detects from parameter file!"
        )
        material_layout.addWidget(material_help, 1, 0, 1, 4)
        
        layout.addWidget(material_group)
        
        # Analysis model
        model_group = QGroupBox("Analysis Model")
        model_layout = QGridLayout(model_group)
        model_layout.setSpacing(10)  # Increase spacing between elements
        
        model_layout.addWidget(QLabel("Analysis Model:"), 0, 0)
        self.analysis_model = QComboBox()
        self.analysis_model.addItems(["hybrid_5_segment", "max_min"])
        self.analysis_model.setCurrentText("hybrid_5_segment")
        model_layout.addWidget(self.analysis_model, 0, 1)
        
        layout.addWidget(model_group)

        # HEL Detection parameters
        self.hel_group = QGroupBox("HEL Detection Parameters")
        hel_layout = QGridLayout(self.hel_group)
        hel_layout.setSpacing(10)

        hel_layout.addWidget(QLabel("HEL Start Time (ns):"), 0, 0)
        self.hel_start_time_ns = QDoubleSpinBox()
        self.hel_start_time_ns.setRange(-1000.0, 1000.0)
        self.hel_start_time_ns.setDecimals(2)
        self.hel_start_time_ns.setValue(0.0)
        self.hel_start_time_ns.setToolTip("Relative to t=0 (after alignment). Start HEL analysis window.")
        hel_layout.addWidget(self.hel_start_time_ns, 0, 1)

        hel_layout.addWidget(QLabel("HEL End Time (ns):"), 0, 2)
        self.hel_end_time_ns = QDoubleSpinBox()
        self.hel_end_time_ns.setRange(-1000.0, 5000.0)
        self.hel_end_time_ns.setDecimals(2)
        self.hel_end_time_ns.setValue(12.0)
        self.hel_end_time_ns.setToolTip("Relative to t=0 (after alignment). End HEL analysis window.")
        hel_layout.addWidget(self.hel_end_time_ns, 0, 3)

        hel_layout.addWidget(QLabel("Angle Threshold (deg):"), 1, 0)
        self.hel_angle_threshold_deg = QDoubleSpinBox()
        self.hel_angle_threshold_deg.setRange(1.0, 89.0)
        self.hel_angle_threshold_deg.setDecimals(1)
        self.hel_angle_threshold_deg.setValue(45.0)
        hel_layout.addWidget(self.hel_angle_threshold_deg, 1, 1)

        hel_layout.addWidget(QLabel("Min HEL Velocity (m/s):"), 1, 2)
        self.minimum_hel_velocity = QDoubleSpinBox()
        self.minimum_hel_velocity.setRange(0.0, 2000.0)
        self.minimum_hel_velocity.setDecimals(2)
        self.minimum_hel_velocity.setValue(15.0)
        self.minimum_hel_velocity.setToolTip("Reject HEL detections below this free-surface velocity.")
        hel_layout.addWidget(self.minimum_hel_velocity, 1, 3)

        hel_layout.addWidget(QLabel("Min Consecutive Points:"), 2, 0)
        self.hel_detection_min_points = QSpinBox()
        self.hel_detection_min_points.setRange(1, 10000)
        self.hel_detection_min_points.setValue(50)
        self.hel_detection_min_points.setToolTip("Minimum number of consecutive low-slope points required to accept HEL plateau.")
        hel_layout.addWidget(self.hel_detection_min_points, 2, 1)

        # Initially hidden unless HEL Detection is enabled
        self.hel_group.setVisible(self.experiment_hel_detection.isChecked())
        layout.addWidget(self.hel_group)
        
        # Signal length
        signal_group = QGroupBox("Signal Length")
        signal_layout = QGridLayout(signal_group)
        signal_layout.setSpacing(10)  # Increase spacing between elements
        
        signal_layout.addWidget(QLabel("Signal Length:"), 0, 0)
        self.signal_length_combo = QComboBox()
        self.signal_length_combo.addItems(["Full Signal (None)", "Custom..."])
        self.signal_length_combo.currentIndexChanged.connect(self.toggle_signal_length_spin)
        signal_layout.addWidget(self.signal_length_combo, 0, 1)
        
        signal_layout.addWidget(QLabel("Custom Length (ns):"), 1, 0)
        self.signal_length_spin = QDoubleSpinBox()
        self.signal_length_spin.setRange(0, 10000)
        self.signal_length_spin.setValue(20.0)
        self.signal_length_spin.setSuffix(" ns")
        self.signal_length_spin.setEnabled(False)
        signal_layout.addWidget(self.signal_length_spin, 1, 1)
        
        layout.addWidget(signal_group)
        
        # Filtering parameters
        filter_group = QGroupBox("Peak Detection Parameters")
        filter_layout = QGridLayout(filter_group)
        filter_layout.setSpacing(10)  # Increase spacing between elements
        
        filter_layout.addWidget(QLabel("Prominence Factor:"), 0, 0)
        self.prominence_factor = QDoubleSpinBox()
        self.prominence_factor.setRange(0, 1)
        self.prominence_factor.setSingleStep(0.01)
        self.prominence_factor.setValue(0.01)
        self.prominence_factor.setDecimals(3)
        self.prominence_factor.setSuffix(" (fraction)")
        filter_layout.addWidget(self.prominence_factor, 0, 1)
        
        filter_layout.addWidget(QLabel("Peak Distance (ns):"), 0, 2)
        self.peak_distance_ns = QDoubleSpinBox()
        self.peak_distance_ns.setRange(0, 1000)
        self.peak_distance_ns.setValue(5.0)
        self.peak_distance_ns.setSuffix(" ns")
        filter_layout.addWidget(self.peak_distance_ns, 0, 3)
        
        filter_layout.addWidget(QLabel("Smooth Window:"), 1, 0)
        self.spade_smooth_window = QSpinBox()
        self.spade_smooth_window.setRange(3, 1001)
        self.spade_smooth_window.setValue(101)
        self.spade_smooth_window.setSingleStep(2)
        filter_layout.addWidget(self.spade_smooth_window, 1, 1)
        
        filter_layout.addWidget(QLabel("Polyorder:"), 1, 2)
        self.polyorder = QSpinBox()
        self.polyorder.setRange(1, 5)
        self.polyorder.setValue(1)
        filter_layout.addWidget(self.polyorder, 1, 3)
        
        layout.addWidget(filter_group)

        # Spall detection window parameters
        spall_group = QGroupBox("Spall Window & Thresholds")
        spall_layout = QGridLayout(spall_group)
        spall_layout.setSpacing(10)

        spall_layout.addWidget(QLabel("Spall Start Time (ns):"), 0, 0)
        self.spall_start_time_ns = QDoubleSpinBox()
        self.spall_start_time_ns.setRange(-1000.0, 10000.0)
        self.spall_start_time_ns.setDecimals(2)
        self.spall_start_time_ns.setValue(10.0)
        self.spall_start_time_ns.setToolTip("Lower bound of the velocity window (relative to t=0) for spall fitting.")
        spall_layout.addWidget(self.spall_start_time_ns, 0, 1)

        spall_layout.addWidget(QLabel("Spall End Time (ns):"), 0, 2)
        self.spall_end_time_ns = QDoubleSpinBox()
        self.spall_end_time_ns.setRange(-1000.0, 10000.0)
        self.spall_end_time_ns.setDecimals(2)
        self.spall_end_time_ns.setValue(100.0)
        self.spall_end_time_ns.setToolTip("Upper bound of the velocity window (relative to t=0) for spall fitting.")
        spall_layout.addWidget(self.spall_end_time_ns, 0, 3)

        spall_layout.addWidget(QLabel("Threshold Velocity (m/s):"), 1, 0)
        self.threshold_velocity_ms = QDoubleSpinBox()
        self.threshold_velocity_ms.setRange(0.0, 1000.0)
        self.threshold_velocity_ms.setDecimals(2)
        self.threshold_velocity_ms.setValue(10.0)
        self.threshold_velocity_ms.setToolTip("Minimum velocity required before searching for spall peaks (also used for alignment warnings).")
        spall_layout.addWidget(self.threshold_velocity_ms, 1, 1)

        layout.addWidget(spall_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        output_layout.setSpacing(10)  # Increase spacing between elements
        
        self.plot_individual = QCheckBox("Generate Individual Plots")
        self.plot_individual.setChecked(True)
        self.plot_individual.setToolTip("Generate individual plots for each file:\n"
                                       "• Spall analysis: spall detection plots saved in 'spall_plots' subfolder\n"
                                       "• HEL detection: HEL window plots with peak/valley markers saved in 'HEL_plots' subfolder\n"
                                       "Plots organized in separate subfolders within SPADE_analysis for easy review")
        output_layout.addWidget(self.plot_individual, 0, 0)
        
        self.save_summary = QCheckBox("Save Summary Table")
        self.save_summary.setChecked(True)
        output_layout.addWidget(self.save_summary, 0, 1)
        
        self.show_plots = QCheckBox("Show Plots (if possible)")
        self.show_plots.setChecked(False)
        output_layout.addWidget(self.show_plots, 0, 2)
        
        layout.addWidget(output_group)
        
        # Plot axis limits
        axis_group = QGroupBox("Combined Plot Axis Limits")
        axis_layout = QGridLayout(axis_group)
        axis_layout.setSpacing(10)  # Increase spacing between elements
        
        # X-axis limits for main plot
        axis_layout.addWidget(QLabel("Main Plot X-Limits (ns):"), 0, 0)
        x_limits_layout = QHBoxLayout()
        self.x_min_main = QDoubleSpinBox()
        self.x_min_main.setRange(-1000, 1000)
        self.x_min_main.setValue(0)
        self.x_min_main.setSuffix(" ns")
        x_limits_layout.addWidget(self.x_min_main)
        
        x_limits_layout.addWidget(QLabel("to"))
        
        self.x_max_main = QDoubleSpinBox()
        self.x_max_main.setRange(-1000, 1000)
        self.x_max_main.setValue(100)
        self.x_max_main.setSuffix(" ns")
        x_limits_layout.addWidget(self.x_max_main)
        axis_layout.addLayout(x_limits_layout, 0, 1)
        
        # Y-axis limits for main plot
        axis_layout.addWidget(QLabel("Main Plot Y-Limits (m/s):"), 1, 0)
        y_limits_layout = QHBoxLayout()
        self.y_min_main = QDoubleSpinBox()
        self.y_min_main.setRange(-1000, 1000)
        self.y_min_main.setValue(0)
        self.y_min_main.setSuffix(" m/s")
        y_limits_layout.addWidget(self.y_min_main)
        
        y_limits_layout.addWidget(QLabel("to"))
        
        self.y_max_main = QDoubleSpinBox()
        self.y_max_main.setRange(0, 2000)
        self.y_max_main.setValue(700)
        self.y_max_main.setSuffix(" m/s")
        y_limits_layout.addWidget(self.y_max_main)
        axis_layout.addLayout(y_limits_layout, 1, 1)
        
        # X-axis limits for zoomed plot
        axis_layout.addWidget(QLabel("Zoomed Plot X-Limits (ns):"), 2, 0)
        x_zoom_layout = QHBoxLayout()
        self.x_min_zoom = QDoubleSpinBox()
        self.x_min_zoom.setRange(-1000, 1000)
        self.x_min_zoom.setValue(0)
        self.x_min_zoom.setSuffix(" ns")
        x_zoom_layout.addWidget(self.x_min_zoom)
        
        x_zoom_layout.addWidget(QLabel("to"))
        
        self.x_max_zoom = QDoubleSpinBox()
        self.x_max_zoom.setRange(-1000, 1000)
        self.x_max_zoom.setValue(20)
        self.x_max_zoom.setSuffix(" ns")
        x_zoom_layout.addWidget(self.x_max_zoom)
        axis_layout.addLayout(x_zoom_layout, 2, 1)
        
        # Y-axis limits for zoomed plot
        axis_layout.addWidget(QLabel("Zoomed Plot Y-Limits (m/s):"), 3, 0)
        y_zoom_layout = QHBoxLayout()
        self.y_min_zoom = QDoubleSpinBox()
        self.y_min_zoom.setRange(-1000, 1000)
        self.y_min_zoom.setValue(0)
        self.y_min_zoom.setSuffix(" m/s")
        y_zoom_layout.addWidget(self.y_min_zoom)
        
        y_zoom_layout.addWidget(QLabel("to"))
        
        self.y_max_zoom = QDoubleSpinBox()
        self.y_max_zoom.setRange(0, 2000)
        self.y_max_zoom.setValue(700)
        self.y_max_zoom.setSuffix(" m/s")
        y_zoom_layout.addWidget(self.y_max_zoom)
        axis_layout.addLayout(y_zoom_layout, 3, 1)
        
        # Auto-calculate checkbox
        self.auto_calculate_limits = QCheckBox("Auto-calculate limits from data")
        self.auto_calculate_limits.setChecked(True)
        self.auto_calculate_limits.setToolTip("Automatically calculate axis limits from the data. Uncheck to use custom limits above.")
        axis_layout.addWidget(self.auto_calculate_limits, 4, 0, 1, 2)
        
        layout.addWidget(axis_group)

        # Combined Velocity Plot controls
        combined_group = QGroupBox("Combined Velocity Plot (Aligned)")
        combined_layout = QGridLayout(combined_group)
        combined_layout.setSpacing(10)

        self.generate_all_velocity_plot = QCheckBox("Generate combined aligned velocity plot")
        self.generate_all_velocity_plot.setChecked(True)
        self.generate_all_velocity_plot.setToolTip("Generate combined plot of all velocity traces aligned at t=0 (threshold), with noise fraction > 1 filtered, and uncertainty threshold applied.")
        combined_layout.addWidget(self.generate_all_velocity_plot, 0, 0, 1, 2)

        # Alignment threshold
        combined_layout.addWidget(QLabel("Alignment threshold:"), 1, 0)
        self.align_velocity_threshold = QDoubleSpinBox()
        self.align_velocity_threshold.setRange(0.0, 1000.0)
        self.align_velocity_threshold.setDecimals(2)
        self.align_velocity_threshold.setValue(30.0)
        self.align_velocity_threshold.setSuffix(" m/s")
        self.align_velocity_threshold.setToolTip("Set the velocity threshold for alignment (t=0 at first time ≥ threshold).")
        combined_layout.addWidget(self.align_velocity_threshold, 1, 1)

        # Uncertainty threshold
        combined_layout.addWidget(QLabel("Uncertainty threshold:"), 2, 0)
        self.uncertainty_threshold = QDoubleSpinBox()
        self.uncertainty_threshold.setRange(0.0, 5000.0)
        self.uncertainty_threshold.setDecimals(2)
        self.uncertainty_threshold.setValue(50.0)
        self.uncertainty_threshold.setSuffix(" m/s")
        self.uncertainty_threshold.setToolTip("Remove points with uncertainty > this threshold from the combined plot.")
        combined_layout.addWidget(self.uncertainty_threshold, 2, 1)

        # Include uncertainty bands
        self.include_uncert_bands = QCheckBox("Include uncertainty bands (±)")
        self.include_uncert_bands.setChecked(True)
        self.include_uncert_bands.setToolTip("Shade ± uncertainty around each trace after filtering and alignment.")
        combined_layout.addWidget(self.include_uncert_bands, 3, 0, 1, 2)

        # Uncertainty band alpha
        combined_layout.addWidget(QLabel("Uncertainty band alpha:"), 4, 0)
        self.uncert_alpha = QDoubleSpinBox()
        self.uncert_alpha.setRange(0.0, 1.0)
        self.uncert_alpha.setDecimals(2)
        self.uncert_alpha.setSingleStep(0.05)
        self.uncert_alpha.setValue(0.2)
        combined_layout.addWidget(self.uncert_alpha, 4, 1)

        # Zoom window (ns)
        combined_layout.addWidget(QLabel("Zoom window (ns):"), 5, 0)
        self.zoom_window_ns = QSpinBox()
        self.zoom_window_ns.setRange(0, 100000)
        self.zoom_window_ns.setValue(1000)
        self.zoom_window_ns.setToolTip("Length of time window shown in the bottom (zoom) plot, starting at t=0.")
        combined_layout.addWidget(self.zoom_window_ns, 5, 1)

        layout.addWidget(combined_group)
        
        # SPADE input mode
        spade_input_group = QGroupBox("SPADE Input Mode")
        spade_input_layout = QVBoxLayout(spade_input_group)
        
        self.spade_auto_radio = QCheckBox("Automatic: Use ALPSS outputs")
        self.spade_auto_radio.setChecked(True)
        self.spade_auto_radio.toggled.connect(self.on_spade_input_mode_changed)
        spade_input_layout.addWidget(self.spade_auto_radio)
        
        self.spade_manual_radio = QCheckBox("Manual: Select SPADE input files")
        self.spade_manual_radio.toggled.connect(self.on_spade_input_mode_changed)
        spade_input_layout.addWidget(self.spade_manual_radio)
        
        # Manual input selection
        manual_input_layout = QHBoxLayout()
        self.spade_input_path = QLineEdit()
        self.spade_input_path.setPlaceholderText("Select SPADE input files or directory...")
        self.spade_input_path.setEnabled(False)
        manual_input_layout.addWidget(self.spade_input_path)
        
        self.spade_input_btn = QPushButton("Browse")
        self.spade_input_btn.clicked.connect(self.select_spade_input)
        self.spade_input_btn.setEnabled(False)
        manual_input_layout.addWidget(self.spade_input_btn)
        
        spade_input_layout.addLayout(manual_input_layout)
        
        # File pattern for manual mode
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("File Pattern:"))
        self.spade_file_pattern = QLineEdit("*--vel-smooth-with-uncert.csv")
        self.spade_file_pattern.setEnabled(False)
        pattern_layout.addWidget(self.spade_file_pattern)
        spade_input_layout.addLayout(pattern_layout)
        
        layout.addWidget(spade_input_group)
        
        layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "SPADE Parameters")
        
    def create_documentation_tab(self):
        """Create documentation tab with ALPSS parameter key"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Create scroll area for documentation
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        doc_layout = QVBoxLayout(scroll_widget)
        doc_layout.setSpacing(15)  # Increase spacing between groups
        
        # ALPSS Parameter Key
        alpss_doc_group = QGroupBox("ALPSS Parameter Key")
        alpss_doc_layout = QVBoxLayout(alpss_doc_group)
        
        alpss_doc_text = QPlainTextEdit()
        alpss_doc_text.setReadOnly(True)
        alpss_doc_text.setMaximumHeight(400)
        alpss_doc_text.setPlainText("""
ALPSS Parameter Key:

filename:                   str; filename for the data to run
save_data:                  str; 'yes' or 'no' to save output data
save_all_plots:             str; 'no', 'subfolder', or 'main_dir' to control plot saving location (CSV data files are always saved when save_data='yes')
start_time_user:            str or float; if 'none' the program will attempt to find the
                                             signal start time automatically. if float then
                                             the program will use that as the signal start time
header_lines:               int; number of header lines to skip in the data file
time_to_skip:               float; the amount of time to skip in the full data file before beginning to read in data
time_to_take:               float; the amount of time to take in the data file after skipping time_to_skip
t_before:                   float; amount of time before the signal start time to include in the velocity calculation
t_after:                    float; amount of time after the signal start time to include in the velocity calculation
start_time_correction:      float; amount of time to adjust the signal start time by
freq_min:                   float; minimum frequency for the region of interest
freq_max:                   float; maximum frequency for the region of interest
smoothing_window:           int; number of points to use for the smoothing window. must be an odd number
smoothing_wid:              float; half the width of the normal distribution used
                                   to calculate the smoothing weights (recommend 3)
smoothing_amp:              float; amplitude of the normal distribution used to calculate
                                   the smoothing weights (recommend 1)
smoothing_sigma:            float; standard deviation of the normal distribution used
                                   to calculate the smoothing weights (recommend 1)
smoothing_mu:               float; mean of the normal distribution used to calculate
                                   the smoothing weights (recommend 0)
pb_neighbors:               int; number of neighbors to compare to when searching
                                     for the pullback local minimum (must be >= 1)
pb_idx_correction:          int; number of local minima to adjust by if the program grabs the wrong one
rc_neighbors:               int; number of neighbors to compare to when searching
                                     for the recompression local maximum (must be >= 1)
rc_idx_correction:          int; number of local maxima to adjust by if the program grabs the wrong one
sample_rate:                float; sample rate of the oscilloscope used in the experiment
nperseg:                    int; number of points to use per segment of the stft
noverlap:                   int; number of points to overlap per segment of the stft
nfft:                       int; number of points to zero pad per segment of the stft
window:                     str or tuple or array_like; window function to use for the stft (recommend 'hann')
blur_kernel:                tuple; kernel size for gaussian blur smoothing (recommend (5, 5))
blur_sigx:                  float; standard deviation of the gaussian blur kernel in the x direction (recommend 0)
blur_sigy:                  float; standard deviation of the gaussian blur kernel in the y direction (recommend 0)
carrier_band_time:          float; length of time from the beginning of the imported data window to average
                                   the frequency of the top of the carrier band in the thresholded spectrogram
cmap:                       str; colormap for the spectrograms (recommend 'viridis')
uncert_mult:                float; factor to multiply the velocity uncertainty by when plotting - allows for easier
                                   visulaization when uncertainties are small
use_notch_filter:           bool; whether to use the gaussian notch filter to remove the carrier band (recommend True for strong signals, False for weak signals)
order:                      int; order for the gaussian notch filter used to remove the carrier band (recommend 6)
wid:                        float; width of the gaussian notch filter used to remove the carrier band (recommend 1e8)
lam:                        float; wavelength of the target laser
C0:                         float; bulk wavespeed of the sample
density:                    float; density of the sample
delta_rho:                  float; uncertainty in density of the sample
delta_C0:                   float; uncertainty in the bulk wavespeed of the sample
delta_lam:                  float; uncertainty in the wavelength of the target laser
theta:                      float; angle of incidence of the PDV probe
delta_theta:                float; uncertainty in the angle of incidence of the PDV probe
exp_data_dir:               str; directory from which to read the experimental data file
out_files_dir:              str; directory to save output data to
display_plots:              str; 'yes' to display the final plots and 'no' to not display them. if save_data='yes'
                                     and and display_plots='no' the plots will be saved but not displayed
spall_calculation:          str; 'yes' to run the calculations for the spall analysis and 'no' to extract the velocity
                                  without doing the spall analysis
plot_figsize:               tuple; figure size for the final plots
plot_dpi:                   float; dpi for the final plots
        """)
        alpss_doc_layout.addWidget(alpss_doc_text)
        doc_layout.addWidget(alpss_doc_group)
        
        # SPADE Documentation
        spade_doc_group = QGroupBox("SPADE Analysis Information")
        spade_doc_layout = QVBoxLayout(spade_doc_group)
        
        spade_doc_text = QPlainTextEdit()
        spade_doc_text.setReadOnly(True)
        spade_doc_text.setMaximumHeight(300)
        spade_doc_text.setPlainText("""
SPADE (Spall Analysis Toolkit) Parameters:

Material Properties:
- density: Material density in kg/m³
- acoustic_velocity: Acoustic velocity in m/s

Analysis Model:
- hybrid_5_segment: Advanced 5-segment analysis model
- max_min: Simple maximum/minimum analysis

Signal Processing:
- signal_length_ns: Custom signal length in nanoseconds (None for full signal)
- smooth_window: Smoothing window size (odd number)
- polyorder: Polynomial order for Savitzky-Golay filter

Peak Detection:
- prominence_factor: Minimum peak prominence as fraction of signal
- peak_distance_ns: Minimum distance between peaks in nanoseconds

Output Options:
- plot_individual: Generate individual analysis plots
  * For spall analysis: individual spall detection plots
  * For HEL detection: individual HEL detection plots (shows full trace + zoomed HEL window with peak/valley markers)
- save_summary_table: Save summary CSV with results
- show_plots: Display plots during analysis (if possible)

Input Requirements:
- Velocity files must be in CSV format with 'Time' and 'Velocity' columns
- Time should be in nanoseconds
- Velocity should be in m/s
        """)
        spade_doc_layout.addWidget(spade_doc_text)
        doc_layout.addWidget(spade_doc_group)
        
        # Usage Instructions
        usage_group = QGroupBox("Usage Instructions")
        usage_layout = QVBoxLayout(usage_group)
        
        usage_text = QPlainTextEdit()
        usage_text.setReadOnly(True)
        usage_text.setMaximumHeight(200)
        usage_text.setPlainText("""
How to Use This GUI:

1. File Selection Tab:
   - Choose Manual Select (pick files individually) or Multiple Files (directory)
   - Select input files or directory
   - Set output directory

2. Analysis Mode Tab:
   - ALPSS Only: Process raw data to velocity traces
   - SPADE Only: Analyze existing velocity files
   - Combined: Full pipeline from raw data to spall analysis

3. ALPSS Parameters Tab:
   - Configure all ALPSS processing parameters
   - Use recommended values for most parameters
   - Adjust based on your experimental setup
   - Gaussian Notch Filter: Enable to remove carrier frequency (recommended for strong signals)
     Disable if signal is weak or carrier/signal frequencies are close together
   - PB/RC Neighbors: Must be ≥ 1 (scipy requirement for peak detection)

4. SPADE Parameters Tab:
   - Set material properties (density, acoustic velocity)
   - Choose analysis model and parameters
   - Configure output options
   - Smooth Window: Only used when SPADE performs its own smoothing (not in combined mode)

5. Control Tab:
   - Run analysis with current settings
   - Monitor progress
   - View output directory

Gaussian Notch Filter Guidelines:
- ENABLE when: Strong carrier signal masks Doppler-shifted signal, clear frequency separation
- DISABLE when: Weak signal relative to noise, carrier and signal frequencies are close
- Effects: Removes carrier frequency but may introduce ringing or phase distortion
- Default: Enabled (True) for backward compatibility

Parameter Constraints:
- PB Neighbors and RC Neighbors: Must be ≥ 1 (required by scipy's peak detection functions)
- Smooth Window: Only applies when SPADE performs smoothing (automatically skipped in combined mode)

Output Files:
- ALPSS outputs: CSV files with velocity data, PNG plots, results with uncertainties
- SPADE outputs: Analysis plots, summary CSV in SPADE_analysis subfolder
- Enhanced outputs: Complete results combining ALPSS and SPADE with all uncertainties
- Key outputs include:
  * spall_summary.csv: Basic SPADE results
  * enhanced_spall_summary.csv: Complete results with ALPSS data and uncertainties
  * spall_vs_strain_rate.png: Spall strength vs strain rate plot
  * spall_vs_shock_stress.png: Spall strength vs shock stress plot
  * all_smoothed_velocity_traces.png: Combined velocity traces
  * Individual ALPSS files: *--results.csv (with uncertainties), *--velocity.csv, etc.
        """)
        usage_layout.addWidget(usage_text)
        doc_layout.addWidget(usage_group)
        
        doc_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "Documentation")
        
    def create_control_tab(self):
        """Create control and progress tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins to tab layout
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.stop_btn)
        
        self.open_output_btn = QPushButton("Open Output Directory")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.open_output_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.open_output_btn)
        
        layout.addLayout(control_layout)
        
        # Progress bars
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # ALPSS Progress
        alpss_progress_layout = QHBoxLayout()
        alpss_progress_layout.addWidget(QLabel("ALPSS:"))
        self.alpss_progress_bar = QProgressBar()
        self.alpss_progress_bar.setVisible(False)
        alpss_progress_layout.addWidget(self.alpss_progress_bar)
        progress_layout.addLayout(alpss_progress_layout)
        
        # SPADE Progress
        spade_progress_layout = QHBoxLayout()
        spade_progress_layout.addWidget(QLabel("SPADE:"))
        self.spade_progress_bar = QProgressBar()
        self.spade_progress_bar.setVisible(False)
        spade_progress_layout.addWidget(self.spade_progress_bar)
        progress_layout.addLayout(spade_progress_layout)
        
        layout.addWidget(progress_group)
        
        # Progress text (QPlainTextEdit scales better for large logs)
        self.progress_text = QPlainTextEdit()
        self.progress_text.setMaximumHeight(300)
        self.progress_text.setPlaceholderText("Analysis progress will appear here...")
        self.progress_text.setReadOnly(True)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_text)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "Control & Progress")
        
    def on_file_mode_changed(self):
        """Handle file mode radio button changes"""
        if self.single_file_radio.isChecked():
            self.single_file_path.setEnabled(True)
            self.single_file_btn.setEnabled(True)
            self.multi_file_path.setEnabled(False)
            self.multi_file_btn.setEnabled(False)
            self.file_pattern.setEnabled(False)
        else:
            self.single_file_path.setEnabled(False)
            self.single_file_btn.setEnabled(False)
            self.multi_file_path.setEnabled(True)
            self.multi_file_btn.setEnabled(True)
            self.file_pattern.setEnabled(True)
            
        self.update_file_list()
        
    def on_analysis_mode_changed(self):
        """Handle analysis mode radio button changes"""
        # Update UI based on selected mode
        self.update_ui_for_analysis_mode()
        
    def update_ui_for_analysis_mode(self):
        """Update UI elements based on selected analysis mode"""
        if self.mode_alpss_only.isChecked():
            # ALPSS only: Enable ALPSS tab, disable SPADE tab
            self.tab_widget.setTabEnabled(2, True)  # ALPSS params tab
            self.tab_widget.setTabEnabled(3, False)  # SPADE params tab
            # Force SPADE input mode to manual for SPADE-only mode
            self.spade_auto_radio.setChecked(False)
            self.spade_manual_radio.setChecked(True)
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
            
        elif self.mode_spade_only.isChecked():
            # SPADE only: Disable ALPSS tab, enable SPADE tab
            self.tab_widget.setTabEnabled(2, False)  # ALPSS params tab
            self.tab_widget.setTabEnabled(3, True)   # SPADE params tab
            # Force SPADE input mode to manual
            self.spade_auto_radio.setChecked(False)
            self.spade_manual_radio.setChecked(True)
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
            
        else:  # Both modes
            # Both: Enable both tabs
            self.tab_widget.setTabEnabled(2, True)   # ALPSS params tab
            self.tab_widget.setTabEnabled(3, True)   # SPADE params tab
            # Allow SPADE input mode selection
            self.spade_auto_radio.setEnabled(True)
            self.spade_manual_radio.setEnabled(True)
            
    def on_spade_input_mode_changed(self):
        """Handle SPADE input mode radio button changes"""
        if self.spade_auto_radio.isChecked():
            self.spade_input_path.setEnabled(False)
            self.spade_input_btn.setEnabled(False)
            self.spade_file_pattern.setEnabled(False)
        else:
            self.spade_input_path.setEnabled(True)
            self.spade_input_btn.setEnabled(True)
            self.spade_file_pattern.setEnabled(True)
    
    def on_hel_detection_toggled(self, checked):
        """Show/Hide HEL parameters group based on checkbox."""
        try:
            self.hel_group.setVisible(bool(checked))
        except Exception:
            pass
    
    # ========== Config File Management Functions ==========
    
    def on_alpss_config_mode_changed(self, checked):
        """Enable/disable config file controls when ALPSS config mode changes"""
        if checked:  # Config file mode selected
            self.alpss_config_path.setEnabled(True)
            self.alpss_config_browse_btn.setEnabled(True)
            self.alpss_config_load_btn.setEnabled(True)
        else:  # Manual mode selected
            self.alpss_config_path.setEnabled(False)
            self.alpss_config_browse_btn.setEnabled(False)
            self.alpss_config_load_btn.setEnabled(False)
    
    def on_spade_config_mode_changed(self, checked):
        """Enable/disable config file controls when SPADE config mode changes"""
        if checked:  # Config file mode selected
            self.spade_config_path.setEnabled(True)
            self.spade_config_browse_btn.setEnabled(True)
            self.spade_config_load_btn.setEnabled(True)
        else:  # Manual mode selected
            self.spade_config_path.setEnabled(False)
            self.spade_config_browse_btn.setEnabled(False)
            self.spade_config_load_btn.setEnabled(False)
    
    def browse_alpss_config(self):
        """Browse for ALPSS config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ALPSS Config File", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.alpss_config_path.setText(file_path)
    
    def browse_spade_config(self):
        """Browse for SPADE config file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SPADE Config File", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.spade_config_path.setText(file_path)
    
    def load_alpss_config(self):
        """Load ALPSS parameters from config file"""
        config_path = self.alpss_config_path.text()
        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, "Error", "Please select a valid config file")
            return
        
        success, config_dict, message = load_config_from_file(config_path)
        if not success:
            QMessageBox.critical(self, "Error", message)
            return
        
        # Apply loaded config to GUI controls
        try:
            self.apply_alpss_config(config_dict)
            QMessageBox.information(self, "Success", f"{message}\n\nALPSS parameters loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying config: {str(e)}")
    
    def load_spade_config(self):
        """Load SPADE parameters from config file"""
        config_path = self.spade_config_path.text()
        if not config_path or not os.path.exists(config_path):
            QMessageBox.warning(self, "Error", "Please select a valid config file")
            return
        
        success, config_dict, message = load_config_from_file(config_path)
        if not success:
            QMessageBox.critical(self, "Error", message)
            return
        
        # Apply loaded config to GUI controls
        try:
            self.apply_spade_config(config_dict)
            QMessageBox.information(self, "Success", f"{message}\n\nSPADE parameters loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying config: {str(e)}")
    
    def save_alpss_config(self):
        """Save current ALPSS parameters to config file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ALPSS Config File", "alpss_config.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        # Get current parameters
        alpss_params = self.get_alpss_params()
        
        success, message = save_config_to_file(alpss_params, file_path)
        if success:
            QMessageBox.information(self, "Success", message)
            self.alpss_config_path.setText(file_path)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def save_spade_config(self):
        """Save current SPADE parameters to config file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save SPADE Config File", "spade_config.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return
        
        # Get current parameters
        spade_params = self.get_spade_params()
        
        success, message = save_config_to_file(spade_params, file_path)
        if success:
            QMessageBox.information(self, "Success", message)
            self.spade_config_path.setText(file_path)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def apply_alpss_config(self, config_dict):
        """Apply loaded ALPSS config to GUI controls"""
        # Basic parameters
        if 'save_data' in config_dict:
            self.save_data.setCurrentText(config_dict['save_data'])
        if 'display_plots' in config_dict:
            self.display_plots.setCurrentText(config_dict['display_plots'])
        if 'save_all_plots' in config_dict:
            self.save_all_plots.setCurrentText(config_dict['save_all_plots'])
        if 'spall_calculation' in config_dict:
            self.spall_calculation.setCurrentText(config_dict['spall_calculation'])
        if 'header_lines' in config_dict:
            self.header_lines.setValue(config_dict['header_lines'])
        if 'start_time_user' in config_dict:
            self.start_time_user.setText(config_dict['start_time_user'])
        if 'start_time_correction' in config_dict:
            self.start_time_correction.setValue(config_dict['start_time_correction'])
        
        # Time parameters
        if 'time_to_skip' in config_dict:
            self.time_to_skip.setValue(config_dict['time_to_skip'])
        if 'time_to_take' in config_dict:
            self.time_to_take.setValue(config_dict['time_to_take'])
        if 't_before' in config_dict:
            self.t_before.setValue(config_dict['t_before'])
        if 't_after' in config_dict:
            self.t_after.setValue(config_dict['t_after'])
        if 'iq_threshold_factor' in config_dict:
            self.iq_threshold_factor.setValue(config_dict['iq_threshold_factor'])
        
        # Frequency parameters
        if 'freq_min' in config_dict:
            self.freq_min.setValue(config_dict['freq_min'])
        if 'freq_max' in config_dict:
            self.freq_max.setValue(config_dict['freq_max'])
        
        # Smoothing parameters
        if 'smoothing_window' in config_dict:
            self.smoothing_window.setValue(config_dict['smoothing_window'])
        if 'smoothing_wid' in config_dict:
            self.smoothing_wid.setValue(config_dict['smoothing_wid'])
        if 'smoothing_amp' in config_dict:
            self.smoothing_amp.setValue(config_dict['smoothing_amp'])
        if 'smoothing_sigma' in config_dict:
            self.smoothing_sigma.setValue(config_dict['smoothing_sigma'])
        if 'smoothing_mu' in config_dict:
            self.smoothing_mu.setValue(config_dict['smoothing_mu'])
        
        # Peak detection parameters
        if 'pb_neighbors' in config_dict:
            self.pb_neighbors.setValue(config_dict['pb_neighbors'])
        if 'pb_idx_correction' in config_dict:
            self.pb_idx_correction.setValue(config_dict['pb_idx_correction'])
        if 'rc_neighbors' in config_dict:
            self.rc_neighbors.setValue(config_dict['rc_neighbors'])
        if 'rc_idx_correction' in config_dict:
            self.rc_idx_correction.setValue(config_dict['rc_idx_correction'])
        
        # STFT parameters
        if 'sample_rate' in config_dict:
            self.sample_rate.setValue(config_dict['sample_rate'])
        if 'nperseg' in config_dict:
            self.nperseg.setValue(config_dict['nperseg'])
        if 'noverlap' in config_dict:
            self.noverlap.setValue(config_dict['noverlap'])
        if 'nfft' in config_dict:
            self.nfft.setValue(config_dict['nfft'])
        if 'window' in config_dict:
            self.window.setCurrentText(config_dict['window'])
        if 'carrier_band_time' in config_dict:
            self.carrier_band_time.setValue(config_dict['carrier_band_time'])
        
        # Blur parameters
        if 'blur_kernel_x' in config_dict:
            self.blur_kernel_x.setValue(config_dict['blur_kernel_x'])
        if 'blur_kernel_y' in config_dict:
            self.blur_kernel_y.setValue(config_dict['blur_kernel_y'])
        if 'blur_sigx' in config_dict:
            self.blur_sigx.setValue(config_dict['blur_sigx'])
        if 'blur_sigy' in config_dict:
            self.blur_sigy.setValue(config_dict['blur_sigy'])
        
        # Signal processing
        if 'use_notch_filter' in config_dict:
            self.use_notch_filter.setChecked(config_dict['use_notch_filter'])
        if 'order' in config_dict:
            self.order.setValue(config_dict['order'])
        if 'wid' in config_dict:
            self.wid.setValue(config_dict['wid'])
        if 'uncert_mult' in config_dict:
            self.uncert_mult.setValue(config_dict['uncert_mult'])
        if 'cmap' in config_dict:
            self.cmap.setCurrentText(config_dict['cmap'])
        
        # PDV wavelength
        if 'lam' in config_dict:
            self.lam.setValue(config_dict['lam'])
        if 'theta' in config_dict:
            self.theta.setValue(config_dict['theta'])
        
        # Uncertainty deltas
        if 'delta_rho' in config_dict:
            self.delta_rho.setValue(config_dict['delta_rho'])
        if 'delta_C0' in config_dict:
            self.delta_C0.setValue(config_dict['delta_C0'])
        if 'delta_lam' in config_dict:
            self.delta_lam.setValue(config_dict['delta_lam'])
        if 'delta_theta' in config_dict:
            self.delta_theta.setValue(config_dict['delta_theta'])
        
        # Material properties
        if 'C0' in config_dict:
            self.C0.setValue(config_dict['C0'])
        if 'density' in config_dict:
            self.density.setValue(config_dict['density'])
        
        # Output file selection checkboxes
        if 'save_velocity_csv' in config_dict:
            self.save_velocity_csv.setChecked(config_dict['save_velocity_csv'])
        if 'save_velocity_smooth_csv' in config_dict:
            self.save_velocity_smooth_csv.setChecked(config_dict['save_velocity_smooth_csv'])
        if 'save_velocity_uncert_csv' in config_dict:
            self.save_velocity_uncert_csv.setChecked(config_dict['save_velocity_uncert_csv'])
        if 'save_velocity_smooth_uncert_csv' in config_dict:
            self.save_velocity_smooth_uncert_csv.setChecked(config_dict['save_velocity_smooth_uncert_csv'])
        if 'save_results_csv' in config_dict:
            self.save_results_csv.setChecked(config_dict['save_results_csv'])
        if 'save_noise_csv' in config_dict:
            self.save_noise_csv.setChecked(config_dict['save_noise_csv'])
    
    def apply_spade_config(self, config_dict):
        """Apply loaded SPADE config to GUI controls"""
        # Experiment type
        if 'experiment_velocity_shots' in config_dict:
            self.experiment_velocity_shots.setChecked(config_dict['experiment_velocity_shots'])
        if 'experiment_spall_analysis' in config_dict:
            self.experiment_spall_analysis.setChecked(config_dict['experiment_spall_analysis'])
        if 'experiment_hel_detection' in config_dict:
            self.experiment_hel_detection.setChecked(config_dict['experiment_hel_detection'])
        
        # Material properties
        if 'density' in config_dict:
            self.spade_density.setValue(config_dict['density'])
        if 'acoustic_velocity' in config_dict:
            self.spade_acoustic_velocity.setValue(config_dict['acoustic_velocity'])
        
        # Velocity shots parameters
        if 'impact_velocity_window_start' in config_dict:
            self.impact_velocity_window_start.setValue(config_dict['impact_velocity_window_start'])
        if 'impact_velocity_window_end' in config_dict:
            self.impact_velocity_window_end.setValue(config_dict['impact_velocity_window_end'])
        if 'align_velocity_threshold_ms' in config_dict:
            self.align_velocity_threshold.setValue(config_dict['align_velocity_threshold_ms'])
        
        # Spall analysis parameters
        if 'smoothing_method' in config_dict:
            self.smoothing_method.setCurrentText(config_dict['smoothing_method'])
        if 'smoothing_window_length' in config_dict:
            self.smoothing_window_length.setValue(config_dict['smoothing_window_length'])
        if 'derivative_smoothing_window_length' in config_dict:
            self.derivative_smoothing_window_length.setValue(config_dict['derivative_smoothing_window_length'])
        if 'pullback_threshold_fraction' in config_dict:
            self.pullback_threshold_fraction.setValue(config_dict['pullback_threshold_fraction'])
        if 'min_pullback_velocity_ms' in config_dict:
            self.min_pullback_velocity_ms.setValue(config_dict['min_pullback_velocity_ms'])
        if 'uncertainty_threshold_ms' in config_dict:
            self.uncertainty_threshold.setValue(config_dict['uncertainty_threshold_ms'])
        if 'include_uncert_bands' in config_dict:
            self.include_uncert_bands.setChecked(config_dict['include_uncert_bands'])
        
        # Combined plot parameters
        if 'auto_calculate_limits' in config_dict:
            self.auto_calculate_limits.setChecked(config_dict['auto_calculate_limits'])
        if 'x_min_main' in config_dict:
            self.x_min_main.setValue(config_dict['x_min_main'])
        if 'x_max_main' in config_dict:
            self.x_max_main.setValue(config_dict['x_max_main'])
        if 'y_min_main' in config_dict:
            self.y_min_main.setValue(config_dict['y_min_main'])
        if 'y_max_main' in config_dict:
            self.y_max_main.setValue(config_dict['y_max_main'])
        if 'x_min_zoom' in config_dict:
            self.x_min_zoom.setValue(config_dict['x_min_zoom'])
        if 'x_max_zoom' in config_dict:
            self.x_max_zoom.setValue(config_dict['x_max_zoom'])
        if 'y_min_zoom' in config_dict:
            self.y_min_zoom.setValue(config_dict['y_min_zoom'])
        if 'y_max_zoom' in config_dict:
            self.y_max_zoom.setValue(config_dict['y_max_zoom'])
        
        # HEL parameters
        if 'hel_time_window_start_ns' in config_dict:
            self.hel_time_window_start_ns.setValue(config_dict['hel_time_window_start_ns'])
        if 'hel_time_window_end_ns' in config_dict:
            self.hel_time_window_end_ns.setValue(config_dict['hel_time_window_end_ns'])
        if 'hel_derivative_threshold' in config_dict:
            self.hel_derivative_threshold.setValue(config_dict['hel_derivative_threshold'])
        if 'hel_smoothing_window' in config_dict:
            self.hel_smoothing_window.setValue(config_dict['hel_smoothing_window'])
        if 'minimum_HEL_velocity_expected' in config_dict:
            self.minimum_hel_velocity.setValue(config_dict['minimum_HEL_velocity_expected'])
        if 'hel_detection_min_points' in config_dict:
            self.hel_detection_min_points.setValue(int(config_dict['hel_detection_min_points']))

        # Spall window parameters
        if 'spall_start_time_ns' in config_dict:
            self.spall_start_time_ns.setValue(config_dict['spall_start_time_ns'])
        if 'spall_end_time_ns' in config_dict:
            self.spall_end_time_ns.setValue(config_dict['spall_end_time_ns'])
        if 'threshold_velocity_ms' in config_dict:
            self.threshold_velocity_ms.setValue(config_dict['threshold_velocity_ms'])
    
    # ========== End Config File Management ==========
            
    def toggle_signal_length_spin(self):
        """Toggle signal length spin box based on combo selection"""
        if self.signal_length_combo.currentText() == "Custom...":
            self.signal_length_spin.setEnabled(True)
        else:
            self.signal_length_spin.setEnabled(False)
    
    # Experiment type change handler removed - now allows both options to be selected
            
    def select_single_file(self):
        """Select one or more input files (manual selection)"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input File(s)", "", 
            "Data Files (*.csv *.txt);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if file_paths:
            self.single_file_path.setText(";".join(file_paths))
            self.update_file_list()
            
    def select_multi_file_dir(self):
        """Select directory for multiple files"""
        try:
            print("Opening directory dialog...")
            
            # Try with different options for macOS compatibility
            try:
                dir_path = QFileDialog.getExistingDirectory(
                    self, 
                    "Select Input Directory",
                    os.getcwd(),  # Start from current working directory
                    QFileDialog.ShowDirsOnly
                )
            except Exception as dialog_error:
                print(f"First dialog attempt failed: {dialog_error}")
                # Fallback to simpler dialog
                dir_path = QFileDialog.getExistingDirectory(
                    self, 
                    "Select Input Directory"
                )
            
            print(f"Directory dialog result: {dir_path}")
            if dir_path:
                self.multi_file_path.setText(dir_path)
                print(f"Set directory path: {dir_path}")
                self.update_file_list()
            else:
                print("No directory selected")
        except Exception as e:
            print(f"Error in select_multi_file_dir: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to select directory: {e}")
            
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)
            
    def select_param_folder(self):
        """Select parameter folder containing Excel files"""
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Parameter Folder", 
            ""
        )
        if folder_path:
            self.param_folder_path.setText(folder_path)
            self.load_param_folder_info()
            
    def load_param_folder_info(self):
        """Load and display information from all Excel/CSV files in the parameter folder"""
        folder_path = self.param_folder_path.text()
        if not folder_path or not os.path.exists(folder_path):
            self.param_file_info.setText("No parameter folder selected")
            return
            
        # Find all Excel/CSV files in the folder
        excel_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                excel_files.append(os.path.join(folder_path, file))
                
        if not excel_files:
            self.param_file_info.setText(f"No parameter files (.xlsx/.xls/.csv) found in {os.path.basename(folder_path)}")
            return
            
        total_experiments = 0
        total_pdv_files = 0
        all_materials = set()
        all_pdv_files = []
        
        info_text = f"Parameter Folder: {os.path.basename(folder_path)}\n"
        # Summarize by type
        num_xlsx = sum(1 for f in excel_files if f.lower().endswith(('.xlsx', '.xls')))
        num_csv = sum(1 for f in excel_files if f.lower().endswith('.csv'))
        info_text += f"Parameter Files Found: {len(excel_files)} (Excel: {num_xlsx}, CSV: {num_csv})\n"
        
        for file_path in excel_files:
            try:
                # Try to read the file depending on extension
                if file_path.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e:
                        info_text += f"\n{os.path.basename(file_path)}: Error reading CSV - {str(e)}"
                        continue
                else:
                    try:
                        import openpyxl
                    except ImportError:
                        self.param_file_info.setText("Error: openpyxl not installed. Please install with: pip install openpyxl")
                        return
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as e:
                        info_text += f"\n{os.path.basename(file_path)}: Error reading Excel - {str(e)}"
                        continue
                    
                # Display basic information for this file
                file_experiments = len(df)
                total_experiments += file_experiments
                info_text += f"\n{os.path.basename(file_path)}: {file_experiments} experiments"
                
                # Check for required columns
                if 'PDV_FileName' in df.columns:
                    pdv_files = df['PDV_FileName'].dropna().astype(str).tolist()
                    total_pdv_files += len(pdv_files)
                    all_pdv_files.extend(pdv_files)
                    info_text += f", {len(pdv_files)} PDV files"
                else:
                    info_text += ", no PDV_FileName column"
                    
                # Check for sample material column
                
                # Look for sample material column with various possible names
                sample_material_col = None
                for col in df.columns:
                    col_lower = col.lower()
                    # Check for various possible column names including spaces, underscores, and different formats
                    if any(name in col_lower for name in [
                        'sample_material', 'samplematerial', 'sample material', 'sample-material',
                        'material', 'sample', 'sample_mat', 'sample mat', 'sample-mat'
                    ]):
                        sample_material_col = col
                        break
                
                if sample_material_col:
                    materials = df[sample_material_col].dropna().unique()
                    all_materials.update(materials)
                else:
                    pass
                    
            except Exception as e:
                info_text += f"\n{os.path.basename(file_path)}: Error - {str(e)}"
                
        # Add summary information
        info_text += f"\n\nSummary: {total_experiments} total experiments, {total_pdv_files} PDV files"
        if all_materials:
            info_text += f"\nSample Materials: {', '.join(sorted(all_materials))}"
        if all_pdv_files:
            # Ensure all PDV files are strings and handle any remaining non-string values
            pdv_files_display = []
            for pdv_file in all_pdv_files[:5]:
                try:
                    pdv_files_display.append(str(pdv_file))
                except:
                    pdv_files_display.append("Unknown")
            info_text += f"\nSample PDV Files: {', '.join(pdv_files_display)}"
            if len(all_pdv_files) > 5:
                info_text += f" ... and {len(all_pdv_files) - 5} more"
                
        self.param_file_info.setText(info_text)
            
    def select_spade_input(self):
        """Select SPADE input files or directory"""
        # Provide clear options: Individual Files or Directory
        dialog = QMessageBox(self)
        dialog.setWindowTitle("SPADE Input Selection")
        dialog.setText("Choose how to select SPADE input:")
        btn_files = dialog.addButton("Individual Files", QMessageBox.AcceptRole)
        btn_dir = dialog.addButton("Directory", QMessageBox.AcceptRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked == btn_files:
            # Select one or multiple individual files
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Velocity File(s)", "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file_paths:
                self.spade_input_path.setText(";".join(file_paths))
        elif clicked == btn_dir:
            # Select a directory
            dir_path = QFileDialog.getExistingDirectory(self, "Select Velocity Files Directory")
            if dir_path:
                self.spade_input_path.setText(dir_path)
            
    def update_file_list(self):
        """Update the file list display"""
        try:
            print("Updating file list...")
            self.file_list.clear()
            
            if self.single_file_radio.isChecked():
                file_text = self.single_file_path.text()
                if file_text:
                    paths = [p for p in file_text.split(";") if p]
                    valid_paths = [p for p in paths if os.path.exists(p)]
                    if valid_paths:
                        self.file_list.appendPlainText(f"Manual Select: {len(valid_paths)} file(s)")
                        for p in valid_paths:
                            self.file_list.appendPlainText(f"  • {os.path.basename(p)}")
            else:
                dir_path = self.multi_file_path.text()
                pattern = self.file_pattern.text()
                print(f"Multi-file mode - Directory: {dir_path}, Pattern: {pattern}")
                
                if dir_path and os.path.exists(dir_path):
                    try:
                        print(f"Scanning directory: {dir_path}")
                        files = glob.glob(os.path.join(dir_path, pattern))
                        print(f"Found {len(files)} files matching pattern")
                        if files:
                            self.file_list.appendPlainText(f"Found {len(files)} files in {dir_path}:")
                            for file_path in sorted(files):
                                self.file_list.appendPlainText(f"  • {os.path.basename(file_path)}")
                        else:
                            self.file_list.appendPlainText(f"No files found matching pattern '{pattern}' in {dir_path}")
                    except Exception as e:
                        print(f"Error scanning directory: {e}")
                        self.file_list.appendPlainText(f"Error scanning directory: {e}")
                else:
                    print(f"Invalid directory: {dir_path}")
                    self.file_list.appendPlainText("No valid directory selected")
            print("File list update complete")
        except Exception as e:
            print(f"Error in update_file_list: {e}")
            import traceback
            traceback.print_exc()
            self.file_list.appendPlainText(f"Error updating file list: {e}")
                    
    def get_input_files(self):
        """Get list of input files based on current selection"""
        if self.single_file_radio.isChecked():
            file_text = self.single_file_path.text()
            if file_text:
                paths = [p for p in file_text.split(";") if p]
                return [p for p in paths if os.path.exists(p)]
        else:
            dir_path = self.multi_file_path.text()
            pattern = self.file_pattern.text()
            
            if dir_path and os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, pattern))
                return sorted(files)
        
        return []
        
    def get_param_file_data(self):
        """Get parameter file data from all Excel files in the selected parameter folder"""
        folder_path = self.param_folder_path.text()
        if not folder_path or not os.path.exists(folder_path):
            return None
            
        # Find all Excel files in the folder
        excel_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                excel_files.append(os.path.join(folder_path, file))
                
        if not excel_files:
            return None
            
        # Combine data from all parameter files
        combined_param_data = {}
        
        for param_file_path in excel_files:
            if not os.path.exists(param_file_path):
                continue
                
            try:
                # Try to read the file (Excel or CSV)
                try:
                    if param_file_path.lower().endswith('.csv'):
                        df = pd.read_csv(param_file_path)
                    else:
                        try:
                            import openpyxl
                        except ImportError:
                            print(f"Error: openpyxl not installed for {param_file_path}")
                            continue
                        df = pd.read_excel(param_file_path)
                except Exception as e:
                    print(f"Error reading file {param_file_path}: {str(e)}")
                    continue
                    
                # Create a mapping from PDV_FileName to experiment data
                param_data = {}
                
                # Handle different possible column names for PDV file name
                pdv_col = None
                # First pass: exact-ish known variants, ignoring spaces/underscores/dashes
                import re
                normalized_columns = {col: re.sub(r"[^a-z0-9]", "", col.lower()) for col in df.columns}
                known_variants = [
                    'pdvfilename', 'pdvfile', 'pdv_file', 'pdv_file_name', 'pdv file name',
                    'dvfilename', 'dv_file', 'dvfile', 'filename', 'file_name', 'file name'
                ]
                normalized_variants = {re.sub(r"[^a-z0-9]", "", v): v for v in known_variants}
                for col, norm in normalized_columns.items():
                    if norm in normalized_variants:
                        pdv_col = col
                        break
                # Second pass: heuristic containing tokens 'pdv' and ('file' or 'name')
                if pdv_col is None:
                    for col in df.columns:
                        col_lower = col.lower()
                        if ('pdv' in col_lower or 'dv' in col_lower) and ('file' in col_lower or 'name' in col_lower):
                            pdv_col = col
                            break
                # Final fallback: a standalone 'filename' or 'file name' column
                if pdv_col is None:
                    for col in df.columns:
                        if col.strip().lower() in ['filename', 'file name', 'file_name']:
                            pdv_col = col
                            break
                        
                if pdv_col is None:
                    continue
                    
                # Create mapping for each experiment
                for idx, row in df.iterrows():
                    pdv_file = row[pdv_col]
                    if pd.isna(pdv_file) or pdv_file == 0:
                        continue
                        
                    # Convert PDV file name to string to ensure consistency
                    pdv_file_str = str(pdv_file).strip()
                    
                    # Clean the filename for better matching
                    # Remove common extensions and clean up the name
                    clean_pdv_file = pdv_file_str
                    for ext in ['.csv', '.txt', '.dat', '.xlsx', '.xls']:
                        if clean_pdv_file.lower().endswith(ext):
                            clean_pdv_file = clean_pdv_file[:-len(ext)]
                    
                    # Extract ALL columns from the row (except the PDV filename column itself)
                    exp_info = {}
                    for col in df.columns:
                        if col != pdv_col:  # Skip the PDV filename column
                            value = row.get(col)
                            if not pd.isna(value):  # Only include non-NaN values
                                exp_info[col] = value
                    
                    # Store both original and cleaned versions for better matching
                    combined_param_data[pdv_file_str] = exp_info
                    if clean_pdv_file != pdv_file_str:
                        combined_param_data[clean_pdv_file] = exp_info
                    
                    # Also store with common variations for better matching
                    # Remove date patterns if present
                    date_cleaned = re.sub(r'\d{4}[-_]\d{2}[-_]\d{2}', '', clean_pdv_file)
                    if date_cleaned != clean_pdv_file:
                        combined_param_data[date_cleaned] = exp_info
                
            except Exception as e:
                print(f"Error loading parameter file {param_file_path}: {e}")
                continue
                
        return combined_param_data if combined_param_data else None
        
    def get_alpss_params(self):
        """Get ALPSS parameters from GUI"""
        # Determine save_all_plots strictly from dropdown (authoritative)
        dropdown_value = self.save_all_plots.currentText()
        save_plots_value = 'yes' if dropdown_value in ['subfolder', 'main_dir'] else 'no'
        plots_enabled = (save_plots_value == 'yes')
        
        return {
            'filename': 'example_file.csv',  # Will be updated per file in thread
            'save_data': self.save_data.currentText(),
            'save_all_plots': dropdown_value,  # Save actual dropdown value for config files
            'save_all_plots_enabled': save_plots_value,  # Converted yes/no for ALPSS
            'save_plots_in_subfolder': dropdown_value == 'subfolder',
            'start_time_user': self.start_time_user.text(),
            'header_lines': self.header_lines.value(),
            'time_to_skip': self.time_to_skip.value(),
            'time_to_take': self.time_to_take.value(),
            't_before': self.t_before.value(),
            't_after': self.t_after.value(),
            'start_time_correction': self.start_time_correction.value(),
            'iq_threshold_factor': self.iq_threshold_factor.value(),
            'freq_min': self.freq_min.value(),
            'freq_max': self.freq_max.value(),
            'smoothing_window': self.smoothing_window.value(),
            'smoothing_wid': self.smoothing_wid.value(),
            'smoothing_amp': self.smoothing_amp.value(),
            'smoothing_sigma': self.smoothing_sigma.value(),
            'smoothing_mu': self.smoothing_mu.value(),
            'pb_neighbors': self.pb_neighbors.value(),
            'pb_idx_correction': self.pb_idx_correction.value(),
            'rc_neighbors': self.rc_neighbors.value(),
            'rc_idx_correction': self.rc_idx_correction.value(),
            'sample_rate': self.sample_rate.value(),
            'nperseg': self.nperseg.value(),
            'noverlap': self.noverlap.value(),
            'nfft': self.nfft.value(),
            'window': self.window.currentText(),
            'blur_kernel': (self.blur_kernel_x.value(), self.blur_kernel_y.value()),
            'blur_sigx': self.blur_sigx.value(),
            'blur_sigy': self.blur_sigy.value(),
            'carrier_band_time': self.carrier_band_time.value(),
            'cmap': self.cmap.currentText(),
            'uncert_mult': self.uncert_mult.value(),
            'use_notch_filter': self.use_notch_filter.isChecked(),
            'order': self.order.value(),
            'wid': self.wid.value(),
            'lam': self.lam.value(),
            'C0': self.C0.value(),
            'density': self.density.value(),
            'delta_rho': self.delta_rho.value(),
            'delta_C0': self.delta_C0.value(),
            'delta_lam': self.delta_lam.value(),
            'theta': self.theta.value(),
            'delta_theta': self.delta_theta.value(),
            'exp_data_dir': '',  # Will be updated per file in thread
            'out_files_dir': '',  # Will be updated per file in thread
            'display_plots': self.display_plots.currentText(),
            'spall_calculation': self.spall_calculation.currentText(),
            'plot_figsize': (self.plot_width.value(), self.plot_height.value()),
            'plot_dpi': self.plot_dpi.value(),
            # Image selection parameters (globally gated by Save ALPSS Plots dropdown)
            'save_combined_plot': self.save_combined_plot.isChecked() and plots_enabled,
            'save_iq_start_time_plot': self.save_iq_start_time_plot.isChecked() and plots_enabled,
            # Output file selection parameters
            'save_velocity_csv': self.save_velocity_csv.isChecked(),
            'save_velocity_smooth_csv': self.save_velocity_smooth_csv.isChecked(),
            'save_velocity_uncert_csv': self.save_velocity_uncert_csv.isChecked(),
            'save_velocity_smooth_uncert_csv': self.save_velocity_smooth_uncert_csv.isChecked(),
            'save_results_csv': self.save_results_csv.isChecked(),
            'save_noise_csv': self.save_noise_csv.isChecked(),
            'smart_selection_enabled': self.smart_selection_checkbox.isChecked(),
        }
        
    def get_spade_params(self):
        """Get SPADE parameters from GUI"""
        # Get signal length
        if self.signal_length_combo.currentText() == "Full Signal (None)":
            signal_length_ns = None
        else:
            signal_length_ns = self.signal_length_spin.value()
        
        # Get experiment types (can be both)
        velocity_shots_enabled = self.experiment_velocity_shots.isChecked()
        spall_analysis_enabled = self.experiment_spall_analysis.isChecked()
        
        # Default to velocity_shots if neither is selected
        if not velocity_shots_enabled and not spall_analysis_enabled:
            velocity_shots_enabled = True
            
        return {
            'density': self.spade_density.value(),
            'acoustic_velocity': self.spade_acoustic_velocity.value(),
            'analysis_model': self.analysis_model.currentText(),
            'signal_length_ns': signal_length_ns,
            'prominence_factor': self.prominence_factor.value(),
            'peak_distance_ns': self.peak_distance_ns.value(),
            'smooth_window': self.spade_smooth_window.value(),
            'polyorder': self.polyorder.value(),
            'plot_individual': self.plot_individual.isChecked(),
            'save_summary_table': self.save_summary.isChecked(),
            'show_plots': self.show_plots.isChecked(),
            # Experiment types
            'velocity_shots_enabled': velocity_shots_enabled,
            'spall_analysis_enabled': spall_analysis_enabled,
            'hel_detection_enabled': self.experiment_hel_detection.isChecked(),
            # HEL parameters
            'hel_start_time_ns': self.hel_start_time_ns.value(),
            'hel_end_time_ns': self.hel_end_time_ns.value(),
            'hel_angle_threshold_deg': self.hel_angle_threshold_deg.value(),
            'minimum_HEL_velocity_expected': self.minimum_hel_velocity.value(),
            'hel_detection_min_points': self.hel_detection_min_points.value(),
            # Spall window parameters
            'spall_start_time_ns': self.spall_start_time_ns.value(),
            'spall_end_time_ns': self.spall_end_time_ns.value(),
            'threshold_velocity_ms': self.threshold_velocity_ms.value(),
            # Axis limits for combined plots
            'auto_calculate_limits': self.auto_calculate_limits.isChecked(),
            'x_min_main': self.x_min_main.value(),
            'x_max_main': self.x_max_main.value(),
            'y_min_main': self.y_min_main.value(),
            'y_max_main': self.y_max_main.value(),
            'x_min_zoom': self.x_min_zoom.value(),
            'x_max_zoom': self.x_max_zoom.value(),
            'y_min_zoom': self.y_min_zoom.value(),
            'y_max_zoom': self.y_max_zoom.value(),
            # Combined velocity plot options
            'generate_all_velocity_plot': self.generate_all_velocity_plot.isChecked(),
            'uncertainty_threshold_ms': self.uncertainty_threshold.value(),
            'align_velocity_threshold_ms': self.align_velocity_threshold.value(),
            'include_uncert_bands': self.include_uncert_bands.isChecked(),
            'uncert_alpha': self.uncert_alpha.value(),
            'zoom_window_ns': self.zoom_window_ns.value(),
        }

    def _apply_runtime_spade_overrides(self, spade_params):
        """Ensure critical SPADE settings respect the current GUI selections."""
        overrides = {
            'analysis_model': self.analysis_model.currentText(),
            'spall_start_time_ns': self.spall_start_time_ns.value(),
            'spall_end_time_ns': self.spall_end_time_ns.value(),
            'threshold_velocity_ms': self.threshold_velocity_ms.value(),
            'hel_start_time_ns': self.hel_start_time_ns.value(),
            'hel_end_time_ns': self.hel_end_time_ns.value(),
        }
        spade_params.update(overrides)
        if hasattr(self, 'progress_text') and self.progress_text:
            try:
                self.progress_text.appendPlainText(
                    "✓ Applied GUI overrides to SPADE parameters: "
                    + ", ".join(f"{k}={v}" for k, v in overrides.items())
                )
            except Exception:
                pass
        return spade_params
        
    def run_analysis(self):
        """Run the analysis"""
        # Get output directory
        output_dir = self.output_path.text()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return
            
        # Get parameters - from config file or GUI
        # ALPSS Parameters
        if self.alpss_config_mode.isChecked():
            # Use config file
            config_path = self.alpss_config_path.text()
            if not config_path or not os.path.exists(config_path):
                QMessageBox.warning(self, "Error", "Config file mode selected but no valid ALPSS config file specified.\nPlease select a config file or switch to manual mode.")
                return
            success, alpss_params, message = load_config_from_file(config_path)
            if not success:
                QMessageBox.critical(self, "Error", f"Failed to load ALPSS config:\n{message}")
                return
            self.progress_text.appendPlainText(f"✓ Using ALPSS parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            alpss_params = self.get_alpss_params()
            self.progress_text.appendPlainText("✓ Using ALPSS parameters from GUI")
        
        # SPADE Parameters
        if self.spade_config_mode.isChecked():
            # Use config file
            config_path = self.spade_config_path.text()
            if not config_path or not os.path.exists(config_path):
                QMessageBox.warning(self, "Error", "Config file mode selected but no valid SPADE config file specified.\nPlease select a config file or switch to manual mode.")
                return
            success, spade_params, message = load_config_from_file(config_path)
            if not success:
                QMessageBox.critical(self, "Error", f"Failed to load SPADE config:\n{message}")
                return
            self.progress_text.appendPlainText(f"✓ Using SPADE parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            spade_params = self.get_spade_params()
            self.progress_text.appendPlainText("✓ Using SPADE parameters from GUI")
        
        spade_params = self._apply_runtime_spade_overrides(spade_params)
        self.spade_params = spade_params.copy()
        
        # Get parameter file data if available
        param_data = self.get_param_file_data()
        if param_data:
            # Count parameter files (Excel/CSV) in the parameter folder
            folder_path = self.param_folder_path.text()
            total_count = 0
            num_xlsx = 0
            num_csv = 0
            if folder_path and os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.xlsx', '.xls', '.csv')):
                        total_count += 1
                        if file.lower().endswith('.csv'):
                            num_csv += 1
                        else:
                            num_xlsx += 1
            self.progress_text.appendPlainText(f"Loaded {total_count} parameter files (Excel: {num_xlsx}, CSV: {num_csv}) with {len(param_data)} total experiments")
        else:
            self.progress_text.appendPlainText("No parameter files provided - using default file names")
        
        # Determine analysis mode
        if self.mode_alpss_only.isChecked():
            # ALPSS only mode
            input_files = self.get_input_files()
            if not input_files:
                QMessageBox.warning(self, "No Input Files", "Please select input files for ALPSS analysis.")
                return
                
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(True)
            self.alpss_progress_bar.setRange(0, len(input_files))
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(False)
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start ALPSS-only analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, input_files, output_dir, param_data,
                spade_auto_mode=False, spade_input_files=None, analysis_mode="alpss_only",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
            
        elif self.mode_spade_only.isChecked():
            # SPADE only mode
            spade_input_path = self.spade_input_path.text()
            if not spade_input_path:
                QMessageBox.warning(self, "No SPADE Input", "Please select SPADE input files or directory.")
                return
                
            # Get SPADE input files
            spade_input_files = None
            if ";" in spade_input_path:
                # Multiple individual files
                spade_input_files = spade_input_path.split(";")
            elif os.path.isdir(spade_input_path):
                # Directory with pattern
                pattern = self.spade_file_pattern.text()
                spade_input_files = glob.glob(os.path.join(spade_input_path, pattern))
            elif os.path.isfile(spade_input_path):
                # Single file
                spade_input_files = [spade_input_path]
            else:
                QMessageBox.warning(self, "Invalid SPADE Input", "Please select valid SPADE input files or directory.")
                return
                
            if not spade_input_files:
                QMessageBox.warning(self, "No SPADE Files", "No files found matching the specified pattern.")
                return
                
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(False)
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(True)
            self.spade_progress_bar.setRange(0, len(spade_input_files))
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start SPADE-only analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, [], output_dir, param_data,
                spade_auto_mode=False, spade_input_files=spade_input_files, analysis_mode="spade_only",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
            
        else:
            # Combined ALPSS + SPADE mode
            input_files = self.get_input_files()
            if not input_files:
                QMessageBox.warning(self, "No Input Files", "Please select input files for ALPSS analysis.")
                return
                
            # Get SPADE input mode and files
            spade_auto_mode = self.spade_auto_radio.isChecked()
            spade_input_files = None
            
            if not spade_auto_mode:
                spade_input_path = self.spade_input_path.text()
                if ";" in spade_input_path:
                    # Multiple individual files
                    spade_input_files = spade_input_path.split(";")
                elif os.path.isdir(spade_input_path):
                    # Directory with pattern
                    pattern = self.spade_file_pattern.text()
                    spade_input_files = glob.glob(os.path.join(spade_input_path, pattern))
                elif os.path.isfile(spade_input_path):
                    # Single file
                    spade_input_files = [spade_input_path]
                else:
                    QMessageBox.warning(self, "Invalid SPADE Input", "Please select valid SPADE input files or directory.")
                    return
            
            # Update UI
            self.run_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.alpss_progress_bar.setVisible(True)
            self.alpss_progress_bar.setRange(0, len(input_files))
            self.alpss_progress_bar.setValue(0)
            self.spade_progress_bar.setVisible(False)
            self.spade_progress_bar.setValue(0)
            self.progress_text.clear()
            
            # Start combined analysis thread
            self.analysis_thread = AnalysisThread(
                alpss_params, spade_params, input_files, output_dir, param_data,
                spade_auto_mode=spade_auto_mode, spade_input_files=spade_input_files, analysis_mode="both",
                material_properties={}  # GUI doesn't use config material_properties, falls back to database
            )
            self.analysis_thread.progress_signal.connect(self.update_progress)
            self.analysis_thread.finished_signal.connect(self.analysis_finished)
            self.analysis_thread.start()
        
    def stop_analysis(self):
        """Stop the analysis"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.analysis_thread.wait()
            self.progress_text.appendPlainText("Analysis stopped by user.")
            
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.spade_progress_bar.setVisible(False)
        
    def update_progress(self, message):
        """Update progress display without overloading the GUI"""
        # Append to UI with newline
        self.progress_text.appendPlainText(message)
        # Truncate to last N lines to avoid massive widget growth
        max_lines = 2000
        current_text = self.progress_text.toPlainText()
        if current_text.count("\n") > max_lines + 100:
            lines = current_text.splitlines()
            trimmed = "\n".join(lines[-max_lines:])
            self.progress_text.setPlainText(trimmed)
            self.progress_text.moveCursor(self.progress_text.textCursor().End)
        
        # Also write to a rolling log file
        try:
            log_dir = os.path.join(self.output_path.text() or os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"helix_analysis_{datetime.now().strftime('%Y%m%d')}.log")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        except Exception:
            pass
        
        def _safe_int_from_token(token):
            if not token:
                return None
            token_digits = ''.join(ch for ch in token if ch.isdigit())
            return int(token_digits) if token_digits.isdigit() else None
        
        # Update the correct progress bar
        if "ALPSS" in message and "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.alpss_progress_bar.maximum() or 100
                    self.alpss_progress_bar.setMaximum(total)
                self.alpss_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        elif "SPADE" in message and "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.spade_progress_bar.maximum() or 100
                    self.spade_progress_bar.setMaximum(total)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        elif "Processing file" in message:
            try:
                parts = message.split("Processing file ")[1].split("/")
                if len(parts) >= 2:
                    current = int(parts[0])
                    # Extract total - handle cases like "21" or "21: Starting..."
                    total_str = parts[1].split()[0] if parts[1].split() else parts[1]
                    total_val = _safe_int_from_token(total_str)
                    total = total_val if total_val is not None else self.spade_progress_bar.maximum() or 100
                    self.spade_progress_bar.setMaximum(total)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except (ValueError, IndexError, AttributeError):
                pass
        # Handle completion messages - set progress to 100%
        elif "All analysis completed successfully" in message or "Analysis completed successfully" in message:
            if self.alpss_progress_bar.isVisible():
                max_val = self.alpss_progress_bar.maximum()
                if max_val > 0:
                    self.alpss_progress_bar.setValue(max_val)
            if self.spade_progress_bar.isVisible():
                max_val = self.spade_progress_bar.maximum()
                if max_val > 0:
                    self.spade_progress_bar.setValue(max_val)
            QApplication.processEvents()  # Force immediate update
                
    def analysis_finished(self, success, message):
        """Handle analysis completion"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Set progress bars to 100% when analysis completes
        if self.alpss_progress_bar.isVisible():
            max_val = self.alpss_progress_bar.maximum()
            if max_val > 0:
                self.alpss_progress_bar.setValue(max_val)
        if self.spade_progress_bar.isVisible():
            max_val = self.spade_progress_bar.maximum()
            if max_val > 0:
                self.spade_progress_bar.setValue(max_val)
        
        self.spade_progress_bar.setVisible(False)
        
        if success:
            self.progress_text.appendPlainText("Analysis completed successfully!")
            QMessageBox.information(self, "Success", "Analysis completed successfully!")
        else:
            self.progress_text.appendPlainText(f"Analysis failed: {message}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {message}")
            
    def open_output_directory(self):
        """Open the output directory in file explorer"""
        output_dir = self.output_path.text()
        if output_dir and os.path.exists(output_dir):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", output_dir])
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["explorer", output_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", output_dir])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "Output directory does not exist.")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    # Set application font with larger size and better readability (increased by 25%)
    # Use system-appropriate font to avoid Qt warnings
    import platform
    if platform.system() == "Darwin":  # macOS
        font = QFont("Helvetica", 14)  # macOS system font (always available)
    elif platform.system() == "Windows":
        font = QFont("Segoe UI", 14)  # Windows system font
    else:  # Linux and others
        font = QFont("Arial", 14)  # Universal fallback
    font.setWeight(QFont.Normal)
    app.setFont(font)
    
    # Set application style sheet for modern look (increased font sizes by 25%)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QTabWidget::pane {
            border: 1px solid #c0c0c0;
            background-color: white;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-weight: bold;
            font-size: 16px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #0078d4;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 15px;
            border: 2px solid #c0c0c0;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #2c2c2c;
        }
        QLabel {
            font-size: 14px;
            color: #2c2c2c;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 6px;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #0078d4;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #c0c0c0;
            color: #666666;
        }
        QCheckBox {
            font-size: 14px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QTextEdit {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
        }
        QScrollArea {
            border: none;
        }
        QProgressBar {
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            text-align: center;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }
    """)
    
    window = HELIXAnalysisToolbox()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
