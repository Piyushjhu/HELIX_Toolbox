#!/usr/bin/env python3
"""
ALPSS + SPADE Combined GUI
A comprehensive GUI for running ALPSS analysis followed by SPADE spall analysis
"""
# %%
import sys
import os
import glob
import subprocess
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QLineEdit, QPushButton, QTextEdit, QProgressBar,
                             QFileDialog, QCheckBox, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QGroupBox, QScrollArea, QMessageBox,
                             QSplitter, QFrame, QStyleFactory, QTabBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QValidator
from SPADE.spall_analysis_release.spall_analysis import plot_combined_mean_traces, plot_spall_vs_strain_rate, plot_spall_vs_shock_stress

class ScientificSpinBox(QDoubleSpinBox):
    """Custom spin box that accepts scientific notation input"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(15)  # Allow high precision
        
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
            # Handle scientific notation (e.g., 1e9, 1.5e-6)
            # Also handle common variations like 1E9, 1.5E-6
            text = text.strip().replace('E', 'e').replace('E+', 'e+').replace('E-', 'e-')
            return float(text)
        except ValueError:
            return 0.0
            
    def validate(self, text, pos):
        """Validate scientific notation input"""
        try:
            if text.strip() == "":
                return (QValidator.Acceptable, text, pos)
            # Allow partial input during typing
            if text.endswith('e') or text.endswith('E'):
                return (QValidator.Intermediate, text, pos)
            if text.endswith('e+') or text.endswith('E+') or text.endswith('e-') or text.endswith('E-'):
                return (QValidator.Intermediate, text, pos)
            # Try to parse the value
            text_clean = text.strip().replace('E', 'e').replace('E+', 'e+').replace('E-', 'e-')
            float(text_clean)
            return (QValidator.Acceptable, text, pos)
        except ValueError:
            return (QValidator.Invalid, text, pos)

class AnalysisThread(QThread):
    """Thread for running ALPSS and SPADE analysis"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, alpss_params, spade_params, input_files, output_dir, spade_auto_mode=True, spade_input_files=None, analysis_mode="both"):
        super().__init__()
        self.alpss_params = alpss_params
        self.spade_params = spade_params
        self.input_files = input_files
        self.output_dir = output_dir
        self.spade_auto_mode = spade_auto_mode
        self.spade_input_files = spade_input_files
        self.analysis_mode = analysis_mode  # "alpss_only", "spade_only", or "both"
        
    def run(self):
        try:
            # Import ALPSS and SPADE modules
            sys.path.append('ALPSS')
            sys.path.append('SPADE/spall_analysis_release')
            
            from alpss_main import alpss_main
            from spall_analysis import process_velocity_files
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Process ALPSS files if provided and not SPADE-only mode
            if self.analysis_mode != "spade_only" and self.input_files:
                for i, input_file in enumerate(self.input_files):
                    self.progress_signal.emit(f"ALPSS Processing file {i+1}/{len(self.input_files)}: {os.path.basename(input_file)}")
                    
                    # Run ALPSS
                    self.progress_signal.emit("Running ALPSS analysis...")
                    
                    # Update ALPSS parameters with current file
                    alpss_params = self.alpss_params.copy()
                    alpss_params['filename'] = os.path.basename(input_file)
                    alpss_params['exp_data_dir'] = os.path.dirname(input_file)
                    alpss_params['out_files_dir'] = self.output_dir
                    
                    alpss_main(**alpss_params)
                    
                    self.progress_signal.emit(f"Completed ALPSS analysis for {os.path.basename(input_file)}")
            
            # Run SPADE analysis if not ALPSS-only mode
            if self.analysis_mode != "alpss_only":
                if self.spade_auto_mode:
                    # Automatic mode: use ALPSS output
                    if self.input_files:
                        self.progress_signal.emit("Running SPADE analysis on ALPSS outputs...")
                        
                        # Find all velocity files generated by ALPSS
                        vel_files = []
                        for input_file in self.input_files:
                            base_name = os.path.splitext(os.path.basename(input_file))[0]
                            # Use the velocity with uncertainty file (contains smoothed velocity + uncertainty)
                            vel_file = os.path.join(self.output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
                            if os.path.exists(vel_file):
                                vel_files.append(vel_file)
                        
                        if vel_files:
                            # Create SPADE output subdirectory
                            spade_output_dir = os.path.join(self.output_dir, "SPADE_analysis")
                            os.makedirs(spade_output_dir, exist_ok=True)
                            
                            # Debug: Check paths
                            self.progress_signal.emit(f"Debug: output_dir = {self.output_dir}")
                            self.progress_signal.emit(f"Debug: spade_output_dir = {spade_output_dir}")
                            self.progress_signal.emit(f"Debug: Found {len(vel_files)} velocity files")
                            
                            # Validate paths
                            if not os.path.exists(self.output_dir):
                                raise ValueError(f"Output directory does not exist: {self.output_dir}")
                            if not os.path.exists(spade_output_dir):
                                raise ValueError(f"SPADE output directory could not be created: {spade_output_dir}")
                            
                            # Run SPADE with progress updates
                            self.progress_signal.emit(f"SPADE Processing file 1/{len(vel_files)}: Starting SPADE analysis...")
                            
                            # Add skip_smoothing parameter to avoid double smoothing
                            spade_params_with_skip = self.spade_params.copy()
                            spade_params_with_skip['skip_smoothing'] = True  # Skip SPADE smoothing since ALPSS already smoothed
                            
                            # Remove smooth_window and polyorder when skipping smoothing to avoid confusion
                            if spade_params_with_skip.get('skip_smoothing', False):
                                spade_params_with_skip.pop('smooth_window', None)
                                spade_params_with_skip.pop('polyorder', None)
                            
                            process_velocity_files(
                                input_folder=self.output_dir,
                                file_pattern="*--vel-smooth-with-uncert.csv",  # Use ALPSS smoothed data with uncertainty
                                output_folder=spade_output_dir,
                                summary_table_name=os.path.join(spade_output_dir, "spall_summary.csv"),
                                plot_individual=self.spade_params.get('plot_individual', True),
                                **{k: v for k, v in spade_params_with_skip.items() if k != 'plot_individual'}
                            )
                            
                            # Update progress after completion
                            for i in range(len(vel_files)):
                                self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(vel_files)}: Completed")
                            
                            self.progress_signal.emit(f"Completed SPADE analysis for {len(vel_files)} files")
                        else:
                            self.progress_signal.emit("Warning: No velocity files found for SPADE analysis")
                    else:
                        self.progress_signal.emit("No ALPSS files to process for automatic SPADE mode")
                else:
                    # Manual mode: use provided SPADE input files
                    if self.spade_input_files:
                        self.progress_signal.emit(f"Running SPADE analysis on {len(self.spade_input_files)} manual input files...")
                        
                        # Create SPADE output subdirectory
                        spade_output_dir = os.path.join(self.output_dir, "SPADE_analysis")
                        os.makedirs(spade_output_dir, exist_ok=True)
                        
                        # Run SPADE - for manual mode, we need to create a temporary directory with the files
                        # or use a different approach since SPADE expects input_folder and file_pattern
                        if len(self.spade_input_files) == 1:
                            # Single file - use its directory as input_folder
                            input_dir = os.path.dirname(self.spade_input_files[0])
                            file_pattern = os.path.basename(self.spade_input_files[0])
                        else:
                            # Multiple files - use the first file's directory and a pattern that matches all
                            input_dir = os.path.dirname(self.spade_input_files[0])
                            file_pattern = "*--vel-smooth-with-uncert.csv"
                        
                        # Start SPADE processing
                        self.progress_signal.emit(f"SPADE Processing file 1/{len(self.spade_input_files)}: Starting SPADE analysis...")
                        
                        # Add skip_smoothing parameter to avoid double smoothing
                        spade_params_with_skip = self.spade_params.copy()
                        spade_params_with_skip['skip_smoothing'] = True  # Skip SPADE smoothing since ALPSS already smoothed
                        
                        # Remove smooth_window and polyorder when skipping smoothing to avoid confusion
                        if spade_params_with_skip.get('skip_smoothing', False):
                            spade_params_with_skip.pop('smooth_window', None)
                            spade_params_with_skip.pop('polyorder', None)
                        
                        process_velocity_files(
                            input_folder=input_dir,
                            file_pattern=file_pattern,
                            output_folder=spade_output_dir,
                            summary_table_name=os.path.join(spade_output_dir, "spall_summary.csv"),
                            plot_individual=self.spade_params.get('plot_individual', True),
                            **{k: v for k, v in spade_params_with_skip.items() if k != 'plot_individual'}
                        )
                        
                        # Update progress after completion
                        for i in range(len(self.spade_input_files)):
                            self.progress_signal.emit(f"SPADE Processing file {i+1}/{len(self.spade_input_files)}: Completed")
                        
                        self.progress_signal.emit(f"Completed SPADE analysis for {len(self.spade_input_files)} files")
                    else:
                        self.progress_signal.emit("No SPADE input files provided")
            self.progress_signal.emit("All analysis completed successfully!")
            self.finished_signal.emit(True, "Analysis completed successfully")
            
            # After SPADE analysis, generate mean velocity file and combined plots
            try:
                output_dir = self.output_dir
                spade_output_dir = os.path.join(output_dir, "SPADE_analysis")
                os.makedirs(spade_output_dir, exist_ok=True)

                # 1. Find all ALPSS velocity files (raw, not smooth)
                velocity_files = glob.glob(os.path.join(output_dir, '*--velocity.csv'))
                if velocity_files:
                    # 2. Read and align all velocity files by time
                    dfs = []
                    for f in velocity_files:
                        df = pd.read_csv(f)
                        if 'Time' in df.columns and 'Velocity' in df.columns:
                            dfs.append(df[['Time', 'Velocity']].rename(columns={'Velocity': os.path.basename(f)}))
                    if dfs:
                        # Merge on Time
                        merged = dfs[0]
                        for d in dfs[1:]:
                            merged = pd.merge(merged, d, on='Time', how='outer')
                        merged = merged.sort_values('Time').reset_index(drop=True)
                        # Compute mean and std dev
                        velocity_cols = [col for col in merged.columns if col != 'Time']
                        merged['Mean Velocity (m/s)'] = merged[velocity_cols].mean(axis=1)
                        merged['Std Dev Velocity (m/s)'] = merged[velocity_cols].std(axis=1)
                        
                        # Create a properly named file that matches SPADE's expected pattern
                        # Use a generic name that will work with the plotting function
                        mean_vel_file = os.path.join(spade_output_dir, 'combined_mean_raw_velocity.csv')
                        merged[['Time', 'Mean Velocity (m/s)', 'Std Dev Velocity (m/s)']].to_csv(mean_vel_file, index=False)

                        # 3. Plot combined mean velocity - create a simple plot since SPADE's function expects specific naming
                        try:
                            fig, ax = plt.subplots(figsize=(12, 8))
                            time = merged['Time']
                            mean_velocity = merged['Mean Velocity (m/s)']
                            velocity_std = merged['Std Dev Velocity (m/s)']
                            
                            # Plot mean line
                            ax.plot(time, mean_velocity, 'b-', linewidth=2, label='Mean Velocity')
                            
                            # Plot shaded uncertainty if available
                            if not velocity_std.isna().all():
                                ax.fill_between(time, mean_velocity - velocity_std, mean_velocity + velocity_std, 
                                              alpha=0.3, color='blue', label='±1σ')
                            
                            ax.set_xlabel('Time (ns)', fontsize=14)
                            ax.set_ylabel('Mean Free Surface Velocity (m/s)', fontsize=14)
                            ax.set_title('Combined Mean Velocity Traces', fontsize=16)
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim(0, 700)
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(spade_output_dir, 'combined_mean_velocity.png'), dpi=300)
                            plt.close(fig)
                            self.progress_signal.emit("Generated combined_mean_velocity.png")
                        except Exception as e:
                            msg = f"[WARNING] Failed to generate combined_mean_velocity.png: {e}"
                            print(msg)
                            self.progress_signal.emit(msg)

                # --- NEW: Plot all smoothed velocity traces with uncertainty, aligned to t=0 at first significant rise ---
                smoothed_files = glob.glob(os.path.join(output_dir, '*--velocity--smooth.csv'))
                
                # Create figure with two subplots: main plot and zoomed region
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
                
                # Store all velocity data to determine adaptive y-axis limits
                all_velocities = []
                all_times = []
                
                for i, file in enumerate(sorted(smoothed_files)):
                    try:
                        # Try reading with and without header
                        try:
                            df = pd.read_csv(file)
                            if df.shape[1] < 2:
                                raise ValueError('File has less than 2 columns')
                        except Exception:
                            df = pd.read_csv(file, header=None)
                        # Use first two columns for time and velocity
                        time = df.iloc[:, 0].values
                        velocity = df.iloc[:, 1].values
                        print(f"[DEBUG] Plotting {file}: using columns {df.columns[0]}, {df.columns[1]}")
                        if np.nanmax(time) < 1.0:
                            time = time * 1e9
                        threshold = 0.1 * np.nanmax(velocity)
                        above_thresh = np.where(velocity > threshold)[0]
                        if len(above_thresh) > 0:
                            t0_idx = above_thresh[0]
                        else:
                            t0_idx = 0
                        t0 = time[t0_idx]
                        time_shifted = time - t0
                        label = os.path.basename(file).replace('--velocity--smooth.csv','')
                        if i == 0:
                            print(f"[DEBUG] First 10 time_shifted: {time_shifted[:10]}")
                            print(f"[DEBUG] First 10 velocity: {velocity[:10]}")
                        
                        # Store velocity data for adaptive limits (ignore noise/uncertainty)
                        all_velocities.extend(velocity)
                        all_times.extend(time_shifted)
                        
                        # Plot uncertainty shading
                        uncert_file = file.replace('--velocity--smooth.csv', '--vel--uncert.csv')
                        if os.path.exists(uncert_file):
                            try:
                                df_unc = pd.read_csv(uncert_file)
                                uncert = df_unc.iloc[:, -1].values
                                if len(uncert) == len(velocity):
                                    ax1.fill_between(time_shifted, velocity-uncert, velocity+uncert, alpha=0.2)
                                    ax2.fill_between(time_shifted, velocity-uncert, velocity+uncert, alpha=0.2)
                            except Exception as e:
                                print(f"[WARNING] Could not read uncertainty for {file}: {e}")
                        
                        # Plot velocity traces on both subplots
                        ax1.plot(time_shifted, velocity, label=label, marker='.', linestyle='-', markersize=2)
                        ax2.plot(time_shifted, velocity, label=label, marker='.', linestyle='-', markersize=2)
                        
                    except Exception as e:
                        print(f"[WARNING] Could not plot {file}: {e}")
                
                # Calculate adaptive y-axis limits based on velocity data (ignoring noise/uncertainty)
                if all_velocities:
                    max_velocity = np.nanmax(all_velocities)
                    min_velocity = np.nanmin(all_velocities)
                    velocity_range = max_velocity - min_velocity
                    y_margin = velocity_range * 0.1  # 10% margin
                    y_min = max(0, min_velocity - y_margin)
                    y_max = max_velocity + y_margin
                else:
                    y_min, y_max = 0, 700  # fallback values
                
                # Configure main plot (ax1)
                ax1.set_xlabel('Time (ns)', fontsize=12)
                ax1.set_ylabel('Velocity (m/s)', fontsize=12)
                ax1.set_ylim(y_min, y_max)
                ax1.set_title('All Smoothed Free Surface Velocity Traces (Aligned)', fontsize=14, fontweight='bold')
                ax1.legend(fontsize='small', loc='best')
                ax1.grid(True, linestyle='--', alpha=0.5)
                ax1.tick_params(axis='both', which='major', labelsize=10)
                ax1.tick_params(axis='both', which='minor', labelsize=8)
                ax1.minorticks_on()
                
                # Add bounding box to main plot
                for spine in ax1.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
                
                # Configure zoomed subplot (ax2) - 0 to 20 ns region
                ax2.set_xlabel('Time (ns)', fontsize=12)
                ax2.set_ylabel('Velocity (m/s)', fontsize=12)
                ax2.set_xlim(0, 20)
                ax2.set_ylim(y_min, y_max)
                ax2.set_title('Zoomed Region: 0-20 ns', fontsize=12, fontweight='bold')
                ax2.legend(fontsize='small', loc='best')
                ax2.grid(True, linestyle='--', alpha=0.5)
                ax2.tick_params(axis='both', which='major', labelsize=10)
                ax2.tick_params(axis='both', which='minor', labelsize=8)
                ax2.minorticks_on()
                
                # Add bounding box to zoomed subplot
                for spine in ax2.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_color('black')
                
                # Adjust layout and save
                fig.tight_layout()
                out_path = os.path.join(spade_output_dir, 'all_smoothed_velocity_traces.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                # --- END NEW PLOT ---

                # 4. Spall Strength vs. Strain Rate and Shock Stress
                summary_csv = os.path.join(spade_output_dir, 'spall_summary.csv')
                if os.path.exists(summary_csv):
                    summary_df = pd.read_csv(summary_csv)
                    # Get density and acoustic velocity from GUI/spade_params
                    density = self.spade_params.get('density', 8960)
                    acoustic_velocity = self.spade_params.get('acoustic_velocity', 3950)
                    
                    # Enhance SPADE summary with ALPSS results and additional calculations
                    enhanced_summary = []
                    
                    for idx, row in summary_df.iterrows():
                        enhanced_row = row.copy()
                        filename = row.get('Filename', '')
                        
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
                        
                        # Calculate shock stress uncertainty if we have velocity uncertainty
                        if 'ALPSS_Velocity_at_Max_Compression_ms' in enhanced_row and not pd.isna(enhanced_row['ALPSS_Velocity_at_Max_Compression_ms']):
                            # Estimate velocity uncertainty as 1% of velocity (typical for PDV)
                            vel_uncertainty = enhanced_row['ALPSS_Velocity_at_Max_Compression_ms'] * 0.01
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
                    
                    # Log available outputs
                    self.progress_signal.emit("Available outputs:")
                    self.progress_signal.emit("  - spall_summary.csv: Basic SPADE results")
                    self.progress_signal.emit("  - enhanced_spall_summary.csv: Complete results with ALPSS data and uncertainties")
                    self.progress_signal.emit("  - spall_vs_strain_rate.png: Spall strength vs strain rate plot")
                    self.progress_signal.emit("  - spall_vs_shock_stress.png: Spall strength vs shock stress plot")
                    self.progress_signal.emit("  - all_smoothed_velocity_traces.png: Combined velocity traces")
                    self.progress_signal.emit("  - Individual ALPSS files: *--results.csv, *--velocity.csv, etc.")
                    self.progress_signal.emit("  - Individual SPADE analysis plots (if enabled)")
                    self.progress_signal.emit("  - ALPSS velocity files: 4 columns (Time, Velocity, Uncertainty, Velocity+Uncertainty)")
                    self.progress_signal.emit("  - SPADE uses ALPSS uncertainty data for error bars and analysis")
                    
                    # Spall Strength vs. Strain Rate
                    try:
                        plot_spall_vs_strain_rate(
                            df=summary_df,
                            output_filename=os.path.join(spade_output_dir, 'spall_vs_strain_rate.png'),
                            literature_data_file=os.path.join('SPADE', 'spall_analysis_release', 'data', 'combined_lit_table.csv'),
                            spall_unc_col='Spall Strength Uncertainty (GPa)'
                        )
                    except Exception as e:
                        msg = f"[WARNING] Failed to generate spall_vs_strain_rate.png: {e}"
                        print(msg)
                        self.progress_signal.emit(msg)
                    try:
                        plot_spall_vs_shock_stress(
                            df=summary_df,
                            output_filename=os.path.join(spade_output_dir, 'spall_vs_shock_stress.png'),
                            literature_data_file=os.path.join('SPADE', 'spall_analysis_release', 'data', 'combined_lit_table_only_poly.csv'),
                            spall_unc_col='Spall Strength Uncertainty (GPa)'
                        )
                    except Exception as e:
                        msg = f"[WARNING] Failed to generate spall_vs_shock_stress.png: {e}"
                        print(msg)
                        self.progress_signal.emit(msg)
                # Check for missing plots and warn
                for plot_name in [
                    'combined_mean_velocity.png',
                    'spall_vs_strain_rate.png',
                    'spall_vs_shock_stress.png',
                    'all_smoothed_velocity_traces.png']:
                    plot_path = os.path.join(spade_output_dir, plot_name)
                    if not os.path.exists(plot_path):
                        msg = f"[WARNING] Expected plot missing: {plot_name}"
                        print(msg)
                        self.progress_signal.emit(msg)
            except Exception as e:
                print(f"[WARNING] Post-processing for SPADE plots failed: {e}")
                self.progress_signal.emit(f"[WARNING] Post-processing for SPADE plots failed: {e}")
            
        except Exception as e:
            self.progress_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False, str(e))

class ALPSSSPADEGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = 'light'
        self.init_ui()
        
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
        
        # Increase tab size for better text fit
        self.tab_widget.setStyleSheet("""
            QTabBar::tab {
                min-width: 200px;
                min-height: 40px;
                font-size: 13px;
                padding: 10px 20px;
            }
        """)
        
        # Create tabs
        self.create_file_selection_tab()
        self.create_analysis_mode_tab()
        self.create_alpss_params_tab()
        self.create_spade_params_tab()
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
                    font-size: 13px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: #181a1b;
                    border-bottom: 2px solid #0078d4;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 12px;
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
                    font-size: 11px;
                    color: #f0f0f0;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 11px;
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
                    background-color: #444;
                    color: #888;
                }
                QCheckBox {
                    font-size: 11px;
                    color: #f0f0f0;
                    spacing: 8px;
                    background: transparent;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QTextEdit {
                    border: 1px solid #444;
                    border-radius: 4px;
                    background-color: #232629;
                    color: #f0f0f0;
                    font-size: 11px;
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
                    font-size: 13px;
                    padding: 10px 20px;
                }
                QTabBar::tab:selected {
                    background-color: white;
                    border-bottom: 2px solid #0078d4;
                    color: #2c2c2c;
                }
                QGroupBox {
                    font-weight: bold;
                    font-size: 12px;
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
                    font-size: 11px;
                    color: #2c2c2c;
                    background: transparent;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 6px;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 11px;
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
                    font-size: 11px;
                    color: #2c2c2c;
                    spacing: 8px;
                    background: transparent;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                }
                QTextEdit {
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    background-color: white;
                    color: #2c2c2c;
                    font-size: 11px;
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
        
        # Single file selection
        single_file_layout = QHBoxLayout()
        self.single_file_radio = QCheckBox("Single File")
        self.single_file_radio.setChecked(True)
        self.single_file_radio.toggled.connect(self.on_file_mode_changed)
        single_file_layout.addWidget(self.single_file_radio)
        
        self.single_file_path = QLineEdit()
        self.single_file_path.setPlaceholderText("Select input file...")
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
        
        # File list display
        self.file_list = QTextEdit()
        self.file_list.setMaximumHeight(200)
        self.file_list.setPlaceholderText("Selected files will appear here...")
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
        
        # Radio buttons for different modes
        self.mode_alpss_only = QCheckBox("ALPSS Only")
        self.mode_alpss_only.setChecked(False)
        self.mode_alpss_only.toggled.connect(self.on_analysis_mode_changed)
        mode_layout.addWidget(self.mode_alpss_only)
        
        self.mode_spade_only = QCheckBox("SPADE Only")
        self.mode_spade_only.setChecked(False)
        self.mode_spade_only.toggled.connect(self.on_analysis_mode_changed)
        mode_layout.addWidget(self.mode_spade_only)
        
        self.mode_both = QCheckBox("ALPSS + SPADE (Combined)")
        self.mode_both.setChecked(True)
        self.mode_both.toggled.connect(self.on_analysis_mode_changed)
        mode_layout.addWidget(self.mode_both)
        
        # Description text
        desc_text = QTextEdit()
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
        
    def create_alpss_params_tab(self):
        """Create ALPSS parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
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
        basic_layout.addWidget(QLabel("Save ALPSS Combined Plot:"), 1, 0)
        self.save_all_plots = QComboBox()
        self.save_all_plots.addItems(["yes", "no"])
        self.save_all_plots.setCurrentText("no")
        self.save_all_plots.setToolTip("If 'yes', saves the original comprehensive ALPSS combined plot with all subplots. CSV data files are always saved when 'Save data' is 'yes'.")
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
        
        scroll.setWidget(scroll_widget)
        layout = QVBoxLayout(tab)
        layout.addWidget(scroll)
        self.tab_widget.addTab(tab, "ALPSS Parameters")
        
    def create_spade_params_tab(self):
        """Create SPADE parameters tab"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setContentsMargins(10, 10, 10, 10)  # Add margins to scroll area
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(15)  # Increase spacing between groups
        
        # Material properties
        material_group = QGroupBox("Material Properties")
        material_layout = QGridLayout(material_group)
        material_layout.setSpacing(10)  # Increase spacing between elements
        
        # Density
        material_layout.addWidget(QLabel("Density (kg/m³):"), 0, 0)
        self.spade_density = QDoubleSpinBox()
        self.spade_density.setRange(1000, 25000)
        self.spade_density.setDecimals(2)
        self.spade_density.setValue(8960)  # Default: Copper
        material_layout.addWidget(self.spade_density, 0, 1)
        
        # Acoustic velocity
        material_layout.addWidget(QLabel("Acoustic Velocity (m/s):"), 0, 2)
        self.spade_acoustic_velocity = QDoubleSpinBox()
        self.spade_acoustic_velocity.setRange(1000, 10000)
        self.spade_acoustic_velocity.setDecimals(2)
        self.spade_acoustic_velocity.setValue(3950)  # Default: Copper
        material_layout.addWidget(self.spade_acoustic_velocity, 0, 3)
        
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
        output_layout.addWidget(self.plot_individual, 0, 0)
        
        self.save_summary = QCheckBox("Save Summary Table")
        self.save_summary.setChecked(True)
        output_layout.addWidget(self.save_summary, 0, 1)
        
        self.show_plots = QCheckBox("Show Plots (if possible)")
        self.show_plots.setChecked(False)
        output_layout.addWidget(self.show_plots, 0, 2)
        
        layout.addWidget(output_group)
        
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
        
        alpss_doc_text = QTextEdit()
        alpss_doc_text.setReadOnly(True)
        alpss_doc_text.setMaximumHeight(400)
        alpss_doc_text.setPlainText("""
ALPSS Parameter Key:

filename:                   str; filename for the data to run
save_data:                  str; 'yes' or 'no' to save output data
        save_all_plots:             str; 'yes' or 'no' to save the original comprehensive ALPSS combined plot with all subplots (CSV data files are always saved when save_data='yes')
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
        
        spade_doc_text = QTextEdit()
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
        
        usage_text = QTextEdit()
        usage_text.setReadOnly(True)
        usage_text.setMaximumHeight(200)
        usage_text.setPlainText("""
How to Use This GUI:

1. File Selection Tab:
   - Choose single file or multiple files mode
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
        
        # Progress text
        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(300)
        self.progress_text.setPlaceholderText("Analysis progress will appear here...")
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
        # Ensure only one mode is selected
        if self.mode_alpss_only.isChecked():
            self.mode_spade_only.setChecked(False)
            self.mode_both.setChecked(False)
        elif self.mode_spade_only.isChecked():
            self.mode_alpss_only.setChecked(False)
            self.mode_both.setChecked(False)
        elif self.mode_both.isChecked():
            self.mode_alpss_only.setChecked(False)
            self.mode_spade_only.setChecked(False)
        else:
            # If none are checked, default to both
            self.mode_both.setChecked(True)
            
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
            
    def toggle_signal_length_spin(self):
        """Toggle signal length spin box based on combo selection"""
        if self.signal_length_combo.currentText() == "Custom...":
            self.signal_length_spin.setEnabled(True)
        else:
            self.signal_length_spin.setEnabled(False)
            
    def select_single_file(self):
        """Select single input file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.single_file_path.setText(file_path)
            self.update_file_list()
            
    def select_multi_file_dir(self):
        """Select directory for multiple files"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir_path:
            self.multi_file_path.setText(dir_path)
            self.update_file_list()
            
    def select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)
            
    def select_spade_input(self):
        """Select SPADE input files or directory"""
        # Ask user if they want to select files or directory
        reply = QMessageBox.question(
            self, "SPADE Input Selection",
            "Do you want to select individual files or a directory?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Select individual files
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Velocity Files", "",
                "CSV Files (*.csv);;All Files (*)"
            )
            if file_paths:
                self.spade_input_path.setText(";".join(file_paths))
        else:
            # Select directory
            dir_path = QFileDialog.getExistingDirectory(self, "Select Velocity Files Directory")
            if dir_path:
                self.spade_input_path.setText(dir_path)
            
    def update_file_list(self):
        """Update the file list display"""
        self.file_list.clear()
        
        if self.single_file_radio.isChecked():
            file_path = self.single_file_path.text()
            if file_path and os.path.exists(file_path):
                self.file_list.append(f"Single file: {os.path.basename(file_path)}")
        else:
            dir_path = self.multi_file_path.text()
            pattern = self.file_pattern.text()
            
            if dir_path and os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, pattern))
                if files:
                    self.file_list.append(f"Found {len(files)} files in {dir_path}:")
                    for file_path in sorted(files):
                        self.file_list.append(f"  • {os.path.basename(file_path)}")
                else:
                    self.file_list.append(f"No files found matching pattern '{pattern}' in {dir_path}")
                    
    def get_input_files(self):
        """Get list of input files based on current selection"""
        if self.single_file_radio.isChecked():
            file_path = self.single_file_path.text()
            if file_path and os.path.exists(file_path):
                return [file_path]
        else:
            dir_path = self.multi_file_path.text()
            pattern = self.file_pattern.text()
            
            if dir_path and os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, pattern))
                return sorted(files)
        
        return []
        
    def get_alpss_params(self):
        """Get ALPSS parameters from GUI"""
        return {
            'filename': 'example_file.csv',  # Will be updated per file in thread
            'save_data': self.save_data.currentText(),
            'save_all_plots': self.save_all_plots.currentText(),
            'start_time_user': self.start_time_user.text(),
            'header_lines': self.header_lines.value(),
            'time_to_skip': self.time_to_skip.value(),
            'time_to_take': self.time_to_take.value(),
            't_before': self.t_before.value(),
            't_after': self.t_after.value(),
            'start_time_correction': self.start_time_correction.value(),
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
        }
        
    def get_spade_params(self):
        """Get SPADE parameters from GUI"""
        # Get signal length
        if self.signal_length_combo.currentText() == "Full Signal (None)":
            signal_length_ns = None
        else:
            signal_length_ns = self.signal_length_spin.value()
            
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
        }
        
    def run_analysis(self):
        """Run the analysis"""
        # Get output directory
        output_dir = self.output_path.text()
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return
            
        # Get parameters
        alpss_params = self.get_alpss_params()
        spade_params = self.get_spade_params()
        
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
                alpss_params, spade_params, input_files, output_dir,
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
                alpss_params, spade_params, [], output_dir,
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
                alpss_params, spade_params, input_files, output_dir,
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
            self.progress_text.append("Analysis stopped by user.")
            
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.spade_progress_bar.setVisible(False)
        
    def update_progress(self, message):
        """Update progress display"""
        self.progress_text.append(message)
        self.progress_text.ensureCursorVisible()
        
        # Force GUI update
        QApplication.processEvents()
        
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
            self.progress_text.append("Analysis completed successfully!")
            QMessageBox.information(self, "Success", "Analysis completed successfully!")
        else:
            self.progress_text.append(f"Analysis failed: {message}")
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
    
    # Set application font with larger size and better readability
    font = QFont("Segoe UI", 11)  # Changed from Arial 9 to Segoe UI 11
    font.setWeight(QFont.Normal)
    app.setFont(font)
    
    # Set application style sheet for modern look
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
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #0078d4;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 12px;
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
            font-size: 11px;
            color: #2c2c2c;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 6px;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            background-color: white;
            font-size: 11px;
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
            font-size: 11px;
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
            font-size: 11px;
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
    
    window = ALPSSSPADEGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
# %%
