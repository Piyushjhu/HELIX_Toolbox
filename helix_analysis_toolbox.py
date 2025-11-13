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
from SPADE.spall_analysis_release.spall_analysis import plot_combined_mean_traces, plot_spall_vs_strain_rate, plot_spall_vs_shock_stress
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
     analysis_mode="both"):
        super().__init__()
        self.alpss_params = alpss_params
        self.spade_params = spade_params
        self.input_files = input_files
        self.output_dir = output_dir
        self.param_data = param_data  # Parameter file data mapping
        self.spade_auto_mode = spade_auto_mode
        self.spade_input_files = spade_input_files
        self.analysis_mode = analysis_mode  # "alpss_only", "spade_only", or "both"

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
                    alpss_params['save_velocity_plot'] = self.alpss_params.get(
                        'save_velocity_plot', True)
                    alpss_params['save_stft_plot'] = self.alpss_params.get(
                        'save_stft_plot', True)
                    alpss_params['save_filtered_plot'] = self.alpss_params.get(
                        'save_filtered_plot', True)
                    alpss_params['save_phase_plot'] = self.alpss_params.get(
                        'save_phase_plot', True)
                    alpss_params['save_amplitude_plot'] = self.alpss_params.get(
                        'save_amplitude_plot', True)
                    alpss_params['save_peak_detection_plot'] = self.alpss_params.get(
                        'save_peak_detection_plot', True)
                    alpss_params['save_uncertainty_plot'] = self.alpss_params.get(
                        'save_uncertainty_plot', True)
                    
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

                            process_velocity_files(
                                input_folder=self.output_dir,
                                # Use ALPSS smoothed data with uncertainty
                                file_pattern="*--vel-smooth-with-uncert.csv",
                                output_folder=spade_output_dir,
                                summary_table_name=os.path.join(
                                    spade_output_dir, "spall_summary.csv"),
                                plot_individual=False,
                                **{k: v for k, v in spade_params_with_skip.items() if k != 'plot_individual'}
                            )

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

                        process_velocity_files(
                            input_folder=input_dir,
                            file_pattern=file_pattern,
                            output_folder=spade_output_dir,
                            summary_table_name=os.path.join(
                                spade_output_dir, "spall_summary.csv"),
                            plot_individual=False,
                            files_list=self.spade_input_files,
                            **{k: v for k, v in spade_params_with_skip.items() if k != 'plot_individual'}
                        )

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
            self.finished_signal.emit(True, "Analysis completed successfully")
        except Exception as e:
            self.progress_signal.emit(f"Error during analysis: {str(e)}")
            self.finished_signal.emit(False, f"Analysis failed: {str(e)}")

    def generate_velocity_shots_summary(self, spade_output_dir):
        """Generate velocity shots summary CSV with impact velocity calculations and combined velocity plot"""
        self.progress_signal.emit("Generating velocity shots summary...")
        
        # In SPADE-only mode, use the provided spade_input_files
        # In combined/automatic mode, use files from output_dir
        if self.analysis_mode == "spade_only" and self.spade_input_files:
            velocity_files = [f for f in self.spade_input_files if os.path.exists(f)]
            self.progress_signal.emit(f"SPADE-only mode: Using {len(velocity_files)} provided input files")
        else:
            # Find all velocity files with uncertainty data (which include noise information)
            velocity_files = glob.glob(
        os.path.join(
            self.output_dir,
             '*--vel-smooth-with-uncert.csv'))
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
                else:
                    # Align time data to t=0 at velocity threshold
                    t0 = time_data[t0_idx]
                    time_aligned = time_data - t0
                    self.progress_signal.emit(
                        f"Aligned trace: t=0 at {t0:.2f} ns when velocity reached {velocity_threshold} m/s")

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
                base_name = os.path.splitext(
    os.path.basename(file_path))[0].replace(
        '--vel-smooth-with-uncert', '')

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
                
                if self.spade_params.get('hel_detection_enabled', False):
                    try:
                        hel_start = self.spade_params.get('hel_start_time_ns', 0.0)
                        hel_end = self.spade_params.get('hel_end_time_ns', 12.0)
                        
                        # Crop to HEL analysis window (relative to aligned time)
                        hel_mask = (time_aligned >= hel_start) & (time_aligned <= hel_end)
                        if np.sum(hel_mask) > 10:  # Need at least 10 points
                            hel_time = time_aligned[hel_mask]
                            hel_velocity = velocity_filtered[hel_mask]
                            hel_uncertainty_data = uncertainty_data[hel_mask]
                            
                            # Remove NaN values
                            valid_hel = ~np.isnan(hel_velocity)
                            if np.sum(valid_hel) > 5:
                                hel_time_clean = hel_time[valid_hel]
                                hel_velocity_clean = hel_velocity[valid_hel]
                                hel_unc_clean = hel_uncertainty_data[valid_hel]
                                
                                # Find peaks and valleys in HEL window
                                from scipy.signal import find_peaks
                                peaks, _ = find_peaks(hel_velocity_clean, prominence=np.std(hel_velocity_clean)*0.1)
                                valleys, _ = find_peaks(-hel_velocity_clean, prominence=np.std(hel_velocity_clean)*0.1)
                                
                                if len(peaks) > 0 and len(valleys) > 0:
                                    # Calculate elastic response (difference between peaks and valleys)
                                    first_peak_vel = hel_velocity_clean[peaks[0]] if peaks[0] < len(hel_velocity_clean) else np.nan
                                    first_valley_vel = hel_velocity_clean[valleys[0]] if valleys[0] < len(hel_velocity_clean) else np.nan
                                    
                                    if np.isfinite(first_peak_vel) and np.isfinite(first_valley_vel):
                                        free_surface_velocity = first_peak_vel
                                        
                                        # HEL strength = 0.5 * density * c * (v_peak - v_valley)
                                        # Get material properties from database based on 'Sample material' column
                                        sample_material = param_info.get('Sample material', 'Unknown')
                                        mat_props = get_material_properties(sample_material)
                                        
                                        # Use material-specific properties or parameter file overrides
                                        density = param_info.get('Density_kg_m3', mat_props['density'])
                                        acoustic_velocity = param_info.get('Bulk_Wave_Speed_m_s', mat_props['bulk_wave_speed'])
                                        
                                        # Log which material properties are being used
                                        if mat_props['material_found']:
                                            self.progress_signal.emit(f"Using {mat_props['material_name']} properties: ρ={density:.0f} kg/m³, c={acoustic_velocity:.0f} m/s")
                                        
                                        pullback_velocity = abs(first_peak_vel - first_valley_vel)
                                        if pullback_velocity > 0:
                                            hel_strength = 0.5 * density * acoustic_velocity * pullback_velocity / 1e9  # Convert Pa to GPa
                                            hel_ok = True
                                            
                                            # Estimate uncertainty
                                            u_max = np.abs(hel_unc_clean[peaks[0]]) if peaks[0] < len(hel_unc_clean) else 0
                                            u_min = np.abs(hel_unc_clean[valleys[0]]) if valleys[0] < len(hel_unc_clean) else 0
                                            pullback_unc = np.sqrt(u_max**2 + u_min**2)
                                            hel_uncertainty = 0.5 * density * acoustic_velocity * pullback_unc / 1e9  # GPa
                                            
                                            self.progress_signal.emit(f"HEL detected: {hel_strength:.3f} GPa for {base_name}")
                                            
                                            # Generate individual HEL detection plot if plot_individual is enabled
                                            if self.spade_params.get('plot_individual', False):
                                                try:
                                                    self._plot_individual_hel_detection(
                                                        base_name, time_aligned, velocity_filtered, 
                                                        hel_start, hel_end, hel_time_clean, hel_velocity_clean,
                                                        peaks, valleys, first_peak_vel, first_valley_vel,
                                                        hel_strength, hel_uncertainty, sample_material,
                                                        spade_output_dir
                                                    )
                                                except Exception as plot_error:
                                                    self.progress_signal.emit(f"Warning: Could not create HEL plot for {base_name}: {str(plot_error)[:50]}")
                                    else:
                                        self.progress_signal.emit(f"HEL: Invalid peak/valley velocities for {base_name}")
                                else:
                                    self.progress_signal.emit(f"HEL: Insufficient peaks/valleys in {base_name}")
                        else:
                            self.progress_signal.emit(f"HEL: Insufficient data points in window for {base_name}")
                    except Exception as hel_error:
                        self.progress_signal.emit(f"HEL detection error for {base_name}: {str(hel_error)[:50]}")
                
                shot_data = {
                    'file_name': base_name,
                    'mean_velocity_300_400ns_ms': mean_velocity_300_400,
                    'time_window_used': time_window_used,
                    'uncertainty_avg_ms': np.nanmean(uncertainty_data),
                    't0_ns': t0 if t0_idx is not None else np.nan,
                    'velocity_threshold_ms': velocity_threshold,
                    'hel_strength_gpa': hel_strength,
                    'hel_uncertainty_gpa': hel_uncertainty,
                    'free_surface_velocity_ms': free_surface_velocity,
                    'hel_ok': hel_ok,
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
                    if key not in ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms', 't0_ns', 'velocity_threshold_ms']:
                        all_param_columns.add(key)

            # Add missing parameter columns with NaN values to each row
            for shot_data in velocity_shots_data:
                for param_col in all_param_columns:
                    if param_col not in shot_data:
                        shot_data[param_col] = np.nan

            velocity_shots_df = pd.DataFrame(velocity_shots_data)

            # Reorder columns: standard columns first, then all parameter columns (sorted)
            standard_cols = ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 'uncertainty_avg_ms', 't0_ns', 'velocity_threshold_ms']
            param_cols = sorted([c for c in all_param_columns if c not in standard_cols])
            final_cols = standard_cols + param_cols
            # Include any unexpected columns at the end to avoid dropping
            remaining_cols = [c for c in velocity_shots_df.columns if c not in final_cols]
            velocity_shots_df = velocity_shots_df[final_cols + remaining_cols]
            
            velocity_shots_path = os.path.join(
    spade_output_dir, 'velocity_shots_summary.csv')
            velocity_shots_df.to_csv(velocity_shots_path, index=False)
            self.progress_signal.emit(
                f"Generated velocity shots summary with {len(velocity_shots_data)} shots")
            self.progress_signal.emit(f"Saved to: {velocity_shots_path}")
            self.progress_signal.emit(f"Parameter columns included: {param_cols}")

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
                        self.generate_all_velocity_traces_plot(input_path, spade_output_dir, uncertainty_threshold)
            except Exception as e:
                self.progress_signal.emit(f"Warning: Failed to create comprehensive aligned velocity plot: {e}")
            
            # Create parameter mapping report for debugging
            self.create_parameter_mapping_report(velocity_shots_data, spade_output_dir)
            
            # Generate HEL vs Laser Energy plot if HEL detection was enabled
            if self.spade_params.get('hel_detection_enabled', False):
                self.generate_hel_vs_laser_energy_plot(spade_output_dir)
        else:
            self.progress_signal.emit("No velocity shots data generated")

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
                                                                      't0_ns', 'velocity_threshold_ms']]
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
                return
            
            df = pd.read_csv(velocity_shots_path)
            
            # Check if HEL and laser energy columns exist
            if 'hel_strength_gpa' not in df.columns:
                self.progress_signal.emit("⚠ HEL strength not found in velocity shots - skipping HEL vs Laser Energy plot")
                return
            
            # Look for laser energy column (various possible names)
            laser_energy_col = None
            possible_names = ['Laser energy (J)', 'Laser_energy_J', 'laser_energy', 'Laser Energy', 
                            'Energy (J)', 'Energy_J', 'energy', 'Laser Power', 'laser_power']
            for col_name in df.columns:
                col_normalized = col_name.lower().replace('_', ' ').replace('-', ' ')
                for possible in possible_names:
                    if possible.lower().replace('_', ' ').replace('-', ' ') in col_normalized:
                        laser_energy_col = col_name
                        break
                if laser_energy_col:
                    break
            
            if laser_energy_col is None:
                self.progress_signal.emit("⚠ Laser energy column not found in parameter file - skipping HEL vs Laser Energy plot")
                return
            
            # Filter data: only rows with valid HEL and laser energy
            valid_data = df[(df['hel_strength_gpa'].notna()) & (df[laser_energy_col].notna())]
            
            if len(valid_data) == 0:
                self.progress_signal.emit("⚠ No valid HEL + Laser Energy data points - skipping plot")
                return
            
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
            
            # Set labels and title
            ax.set_xlabel('Laser Energy (J)', fontsize=14, fontweight='bold')
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

    def _plot_individual_hel_detection(self, base_name, time_aligned, velocity_filtered,
                                       hel_start, hel_end, hel_time_clean, hel_velocity_clean,
                                       peaks, valleys, first_peak_vel, first_valley_vel,
                                       hel_strength, hel_uncertainty, sample_material,
                                       spade_output_dir):
        """Generate individual HEL detection plot showing detection results"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
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
        
        # Bottom subplot: Zoomed HEL window with peak/valley detection
        ax2.plot(hel_time_clean, hel_velocity_clean, 'b-', linewidth=2, label='Velocity in HEL window')
        
        # Mark detected peaks and valleys
        if len(peaks) > 0 and peaks[0] < len(hel_time_clean):
            peak_time = hel_time_clean[peaks[0]]
            ax2.plot(peak_time, first_peak_vel, 'ro', markersize=12, 
                    label=f'Peak (Elastic Limit): {first_peak_vel:.1f} m/s', zorder=5)
            ax2.axhline(first_peak_vel, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        if len(valleys) > 0 and valleys[0] < len(hel_time_clean):
            valley_time = hel_time_clean[valleys[0]]
            ax2.plot(valley_time, first_valley_vel, 'gs', markersize=12,
                    label=f'Valley (Pullback): {first_valley_vel:.1f} m/s', zorder=5)
            ax2.axhline(first_valley_vel, color='green', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add arrow showing pullback
        if len(peaks) > 0 and len(valleys) > 0 and peaks[0] < len(hel_time_clean) and valleys[0] < len(hel_time_clean):
            mid_time = (hel_time_clean[peaks[0]] + hel_time_clean[valleys[0]]) / 2
            ax2.annotate('', xy=(mid_time, first_valley_vel), xytext=(mid_time, first_peak_vel),
                        arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
            pullback = abs(first_peak_vel - first_valley_vel)
            ax2.text(mid_time, (first_peak_vel + first_valley_vel) / 2, 
                    f'  ΔV = {pullback:.1f} m/s', fontsize=10, color='purple',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Time (ns)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title(f'HEL Window Detail', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Add text box with HEL results
        result_text = f'HEL Strength: {hel_strength:.3f} ± {hel_uncertainty:.3f} GPa'
        ax2.text(0.02, 0.98, result_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot in SPADE_analysis folder
        plot_filename = f'{base_name}--hel_detection.png'
        plot_path = os.path.join(spade_output_dir, plot_filename)
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

    def generate_all_velocity_traces_plot(self, input_path, spade_output_dir, uncertainty_threshold):
        """Generate an all-traces plot aligned at 30 m/s using ALPSS output files in input_path.
        Applies noise fraction filtering (>1) and removes points with uncertainty > threshold.
        Saves PNG to both main output_dir and SPADE output dir."""
        try:
            import glob
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt

            pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if os.path.getsize(f) > 0]
            if not files:
                self.progress_signal.emit("No '*--vel-smooth-with-uncert.csv' files found for all-traces plot")
                return

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, max(1, len(files))))

            traces_plotted = 0
            for i, file_path in enumerate(sorted(files)):
                try:
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
                        continue

                    # Align at first >= threshold
                    align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
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
                        ax2.plot(time_clean[mask_1000], velocity_clean[mask_1000], color=color, alpha=0.7, linewidth=1)
                        if self.spade_params.get('include_uncert_bands', True) and uncert_clean is not None and len(uncert_clean) == len(velocity_clean):
                            alpha = float(self.spade_params.get('uncert_alpha', 0.2))
                            ax2.fill_between(time_clean[mask_1000],
                                             (velocity_clean - uncert_clean)[mask_1000],
                                             (velocity_clean + uncert_clean)[mask_1000],
                                             color=color, alpha=alpha)

                    traces_plotted += 1
                except Exception:
                    continue

            align_threshold = self.spade_params.get('align_velocity_threshold_ms', 30.0)
            ax1.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax1.set_ylabel('Velocity (m/s)', fontsize=12)
            ax1.set_title(f'All Velocity Traces (Aligned) - {traces_plotted} traces', fontsize=14)
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12)
            ax2.set_ylabel('Velocity (m/s)', fontsize=12)
            ax2.grid(True, alpha=0.3)

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
                fig.savefig(out_main, dpi=300, bbox_inches='tight')
            except Exception:
                pass
            try:
                fig.savefig(out_spade, dpi=300, bbox_inches='tight')
            except Exception:
                pass
            plt.close(fig)

            self.progress_signal.emit(f"Saved aligned all-traces velocity plot to: {out_spade}")
        except Exception as e:
            self.progress_signal.emit(f"Error generating all-traces velocity plot: {e}")

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
            if not filename:
                continue

            # Get file base name for parameter matching
            base_name = os.path.splitext(filename)[0]

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
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Box plot of spall strength by material
                if 'Spall Strength (GPa)' in enhanced_spall_df.columns:
                    materials = enhanced_spall_df[color_col].unique()
                    spall_data = [enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Spall Strength (GPa)'].dropna() 
                                 for mat in materials]
                    
                    bp1 = ax1.boxplot(spall_data, labels=materials, patch_artist=True)
                    for patch, color in zip(bp1['boxes'], [color_map.get(mat, 'gray') for mat in materials]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax1.set_ylabel('Spall Strength (GPa)', fontsize=14)
                    ax1.set_title('Spall Strength by Material', fontsize=16)
                    ax1.grid(True, alpha=0.3)
                
                # Box plot of strain rate by material
                if 'Strain Rate (s^-1)' in enhanced_spall_df.columns:
                    strain_data = [enhanced_spall_df[enhanced_spall_df[color_col] == mat]['Strain Rate (s^-1)'].dropna() 
                                 for mat in materials]
                    
                    bp2 = ax2.boxplot(strain_data, labels=materials, patch_artist=True)
                    for patch, color in zip(bp2['boxes'], [color_map.get(mat, 'gray') for mat in materials]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax2.set_ylabel('Strain Rate (s⁻¹)', fontsize=14)
                    ax2.set_title('Strain Rate by Material', fontsize=16)
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(spade_output_dir, 'material_comparison_spall.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.progress_signal.emit(f"Generated material comparison plots")
            
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
                merged['Mean Velocity (m/s)'] = merged[velocity_cols].mean(axis=1)
                merged['Std Dev Velocity (m/s)'] = merged[velocity_cols].std(axis=1)

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
                    
                    # Get material-specific properties from database
                    mat_props = get_material_properties(sample_material, default_density, default_acoustic_velocity)
                    density = mat_props['density']
                    acoustic_velocity = mat_props['bulk_wave_speed']
                    
                    # Add material information to enhanced row
                    enhanced_row['Material'] = sample_material
                    enhanced_row['Density_kg_m3'] = density
                    enhanced_row['Acoustic_Velocity_m_s'] = acoustic_velocity
                    enhanced_row['Material_Found_In_Database'] = mat_props['material_found']
                    
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
                        
                        if group_value == "Unknown":
                            # Debug: show first few mismatches
                            if i < 5:
                                self.progress.emit(f"  No match for: {base_name}")
                            elif i == 5:
                                self.progress.emit(f"  ... (more non-matching files)")
                    
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
                self._generate_material_subplots(files, param_data, spade_out, current_params, group_colors, align_threshold)
            
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
                    # Try exact match first
                    if base_name in param_data:
                        material = param_data[base_name].get('Sample material', 'Unknown')
                    else:
                        # Try date-shot pattern matching (YYYYMMDD--NNNNN)
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
            if hasattr(self, 'save_velocity_plot') and 'save_velocity_plot' in params:
                if hasattr(self.save_velocity_plot, 'setChecked'):
                    self.save_velocity_plot.setChecked(params['save_velocity_plot'])
            if hasattr(self, 'save_iq_amplitude') and 'save_iq_amplitude' in params:
                if hasattr(self.save_iq_amplitude, 'setChecked'):
                    self.save_iq_amplitude.setChecked(params['save_iq_amplitude'])
            if hasattr(self, 'save_error_plot') and 'save_error_plot' in params:
                if hasattr(self.save_error_plot, 'setChecked'):
                    self.save_error_plot.setChecked(params['save_error_plot'])
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
        desc_label = QLabel("Select which ALPSS output images to generate:")
        desc_label.setWordWrap(True)
        image_layout.addWidget(desc_label)
        
        # Image checkboxes
        self.save_velocity_plot = QCheckBox("Velocity vs Time Plot")
        self.save_velocity_plot.setChecked(True)
        self.save_velocity_plot.setToolTip("Generate velocity vs time plot with uncertainty bands")
        image_layout.addWidget(self.save_velocity_plot)
        
        self.save_stft_plot = QCheckBox("STFT Spectrogram")
        self.save_stft_plot.setChecked(True)
        self.save_stft_plot.setToolTip("Generate Short-Time Fourier Transform spectrogram")
        image_layout.addWidget(self.save_stft_plot)
        
        self.save_filtered_plot = QCheckBox("Filtered Signal Plot")
        self.save_filtered_plot.setChecked(True)
        self.save_filtered_plot.setToolTip("Generate plot showing original vs filtered signal")
        image_layout.addWidget(self.save_filtered_plot)
        
        self.save_phase_plot = QCheckBox("Phase Plot")
        self.save_phase_plot.setChecked(True)
        self.save_phase_plot.setToolTip("Generate phase vs time plot")
        image_layout.addWidget(self.save_phase_plot)
        
        self.save_amplitude_plot = QCheckBox("Amplitude Plot")
        self.save_amplitude_plot.setChecked(True)
        self.save_amplitude_plot.setToolTip("Generate amplitude vs time plot")
        image_layout.addWidget(self.save_amplitude_plot)
        
        self.save_iq_start_time_plot = QCheckBox("IQ Start Time Detection Plot")
        # Default off so selecting IQ Analysis does not implicitly save this diagnostic plot
        self.save_iq_start_time_plot.setChecked(False)
        self.save_iq_start_time_plot.setToolTip("Generate IQ analysis start time detection plot with red line marking detected start time")
        image_layout.addWidget(self.save_iq_start_time_plot)
        
        self.save_peak_detection_plot = QCheckBox("Peak Detection Plot")
        self.save_peak_detection_plot.setChecked(True)
        self.save_peak_detection_plot.setToolTip("Generate plot showing detected peaks and pullback")
        image_layout.addWidget(self.save_peak_detection_plot)
        
        self.save_uncertainty_plot = QCheckBox("Uncertainty Analysis Plot")
        self.save_uncertainty_plot.setChecked(True)
        self.save_uncertainty_plot.setToolTip("Generate uncertainty analysis plots")
        image_layout.addWidget(self.save_uncertainty_plot)
        
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
        self.save_velocity_plot.setChecked(True)
        self.save_stft_plot.setChecked(True)
        self.save_filtered_plot.setChecked(True)
        self.save_phase_plot.setChecked(True)
        self.save_amplitude_plot.setChecked(True)
        self.save_peak_detection_plot.setChecked(True)
        self.save_uncertainty_plot.setChecked(True)
        
    def deselect_all_alpss_images(self):
        """Deselect all ALPSS output images"""
        self.save_velocity_plot.setChecked(False)
        self.save_stft_plot.setChecked(False)
        self.save_filtered_plot.setChecked(False)
        self.save_phase_plot.setChecked(False)
        self.save_amplitude_plot.setChecked(False)
        self.save_peak_detection_plot.setChecked(False)
        self.save_uncertainty_plot.setChecked(False)
        
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
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout(output_group)
        output_layout.setSpacing(10)  # Increase spacing between elements
        
        self.plot_individual = QCheckBox("Generate Individual Plots")
        self.plot_individual.setChecked(True)
        self.plot_individual.setToolTip("Generate individual plots for each file:\n"
                                       "• Spall analysis: spall detection plots\n"
                                       "• HEL detection: HEL window plots with peak/valley markers\n"
                                       "Plots saved in SPADE_analysis folder")
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
            'save_all_plots': save_plots_value,
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
            'save_velocity_plot': self.save_velocity_plot.isChecked() and plots_enabled,
            'save_stft_plot': self.save_stft_plot.isChecked() and plots_enabled,
            'save_filtered_plot': self.save_filtered_plot.isChecked() and plots_enabled,
            'save_phase_plot': self.save_phase_plot.isChecked() and plots_enabled,
            'save_amplitude_plot': self.save_amplitude_plot.isChecked() and plots_enabled,
            'save_iq_start_time_plot': self.save_iq_start_time_plot.isChecked() and plots_enabled,
            'save_peak_detection_plot': self.save_peak_detection_plot.isChecked() and plots_enabled,
            'save_uncertainty_plot': self.save_uncertainty_plot.isChecked() and plots_enabled,
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
            self.log_message(f"✓ Using ALPSS parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            alpss_params = self.get_alpss_params()
            self.log_message("✓ Using ALPSS parameters from GUI")
        
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
            self.log_message(f"✓ Using SPADE parameters from config file: {config_path}")
        else:
            # Use GUI parameters
            spade_params = self.get_spade_params()
            self.log_message("✓ Using SPADE parameters from GUI")
        
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
                spade_auto_mode=False, spade_input_files=None, analysis_mode="alpss_only"
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
                spade_auto_mode=False, spade_input_files=spade_input_files, analysis_mode="spade_only"
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
                spade_auto_mode=spade_auto_mode, spade_input_files=spade_input_files, analysis_mode="both"
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
        
        # Update the correct progress bar
        if "ALPSS" in message and "Processing file" in message:
            try:
                current, total = message.split("Processing file ")[1].split("/")
                current = int(current)
                self.alpss_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except:
                pass
        elif "SPADE" in message and "Processing file" in message:
            try:
                current, total = message.split("Processing file ")[1].split("/")
                current = int(current)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except:
                pass
        elif "Processing file" in message:
            try:
                current, total = message.split("Processing file ")[1].split("/")
                current = int(current)
                self.spade_progress_bar.setValue(current)
                QApplication.processEvents()  # Force immediate update
            except:
                pass
                
    def analysis_finished(self, success, message):
        """Handle analysis completion"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
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