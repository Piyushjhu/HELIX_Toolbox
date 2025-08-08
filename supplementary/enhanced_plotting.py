#!/usr/bin/env python3
"""
Enhanced Plotting Module for ALPSS-SPADE Analysis
=================================================

This standalone module provides enhanced plotting capabilities for velocity analysis data.
It can be run independently of the main SPADE workflow.

Features:
- 6 different figure types with customizable options
- Noise filtering and trace alignment
- Material and waveplate angle color coding
- Spread analysis and statistical plots
- CSV data export for further analysis

Usage:
    python enhanced_plotting.py --input_dir /path/to/velocity/files --output_dir /path/to/output --param_file /path/to/parameters.xlsx
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any


class EnhancedPlotting:
    """Enhanced plotting class for velocity analysis data"""
    
    def __init__(self, input_dir: str, output_dir: str, param_file: Optional[str] = None, 
                 plot_options: Optional[Dict[str, bool]] = None):
        """
        Initialize the enhanced plotting module
        
        Args:
            input_dir: Directory containing velocity files (*--velocity--smooth.csv)
            output_dir: Directory to save plots and data
            param_file: Optional parameter file (Excel format)
            plot_options: Dictionary of plot options (Figure 1-6)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.param_file = param_file
        self.param_data = self._load_parameter_data() if param_file else {}
        
        # Default plot options (all enabled)
        self.plot_options = plot_options or {
            'plot_individual_legends': True,    # Figure 1
            'plot_color_meaning': True,         # Figure 2
            'plot_spread_analysis': True,       # Figure 3
            'plot_velocity_vs_angle': True,     # Figure 4
            'plot_shot_time_vs_material': True, # Figure 5
            'plot_pdv_power_vs_material': True  # Figure 6
        }
        
        # Color palettes
        self.material_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.waveplate_palette = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
            '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43'
        ]
        
        # Data structures
        self.material_colors = {}
        self.waveplate_colors = {}
        self.material_counter = 0
        self.waveplate_counter = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_parameter_data(self) -> Dict[str, Dict[str, Any]]:
        """Load parameter data from Excel file"""
        try:
            if not os.path.exists(self.param_file):
                print(f"Warning: Parameter file not found: {self.param_file}")
                return {}
            
            # Read Excel file
            df = pd.read_excel(self.param_file)
            
            # Convert to dictionary format
            param_data = {}
            for _, row in df.iterrows():
                # Use exp_id as key, or first column if exp_id doesn't exist
                key = row.get('exp_id', row.iloc[0])
                param_data[str(key)] = row.to_dict()
            
            print(f"Loaded parameter data for {len(param_data)} experiments")
            return param_data
            
        except Exception as e:
            print(f"Error loading parameter data: {e}")
            return {}
    
    def _find_velocity_files(self) -> List[str]:
        """Find all velocity files in input directory"""
        pattern = os.path.join(self.input_dir, "*--velocity--smooth.csv")
        files = glob.glob(pattern)
        print(f"Found {len(files)} velocity files")
        return sorted(files)
    
    def _load_velocity_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and process velocity data from file
        
        Returns:
            time_data: Time array (ns)
            velocity: Velocity array (m/s)
            velocity_filtered: Filtered velocity array (noise removed)
        """
        try:
            # Try reading with and without header
            try:
                df = pd.read_csv(file_path)
                if df.shape[1] < 2:
                    raise ValueError('File has less than 2 columns')
            except Exception:
                df = pd.read_csv(file_path, header=None)
            
            # Use first two columns for time and velocity
            time_data = df.iloc[:, 0].values
            velocity = df.iloc[:, 1].values
            
            # Convert time to ns if needed
            if np.nanmax(time_data) < 1.0:
                time_data = time_data * 1e9
            
            # Align time to t=0 when velocity reaches threshold
            threshold = 0.1 * np.nanmax(velocity)
            above_thresh = np.where(velocity > threshold)[0]
            if len(above_thresh) > 0:
                t0_idx = above_thresh[0]
            else:
                t0_idx = 0
            t0 = time_data[t0_idx]
            time_shifted = time_data - t0
            
            # Load noise fraction data and filter velocity
            velocity_filtered = velocity.copy()
            base_name = os.path.basename(file_path).replace('--velocity--smooth.csv', '')
            noise_file = file_path.replace('--velocity--smooth.csv', '--noise--frac.csv')
            
            if os.path.exists(noise_file):
                try:
                    df_noise = pd.read_csv(noise_file)
                    if df_noise.shape[1] >= 1:
                        noise_fraction = df_noise.iloc[:, -1].values
                        if len(noise_fraction) == len(velocity):
                            # Filter out data points where noise fraction > 1
                            high_noise_mask = noise_fraction > 1.0
                            velocity_filtered[high_noise_mask] = np.nan
                            print(f"Filtered {np.sum(high_noise_mask)} high-noise points from {base_name}")
                except Exception as e:
                    print(f"Warning: Could not read noise fraction for {file_path}: {e}")
            
            return time_shifted, velocity, velocity_filtered
            
        except Exception as e:
            print(f"Error loading velocity data from {file_path}: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _get_material_and_angle(self, base_name: str) -> Tuple[str, str]:
        """Get material and waveplate angle from parameter data"""
        material = 'Unknown'
        waveplate_angle = 'Unknown'
        
        if base_name in self.param_data:
            exp_info = self.param_data[base_name]
            material = exp_info.get('sample_material', 'Unknown')
            waveplate_angle = exp_info.get('waveplate_angle', 'Unknown')
        
        return material, waveplate_angle
    
    def _assign_colors(self, material: str, waveplate_angle: str):
        """Assign colors for materials and waveplate angles"""
        if material not in self.material_colors:
            self.material_colors[material] = self.material_palette[self.material_counter % len(self.material_palette)]
            self.material_counter += 1
        
        if waveplate_angle not in self.waveplate_colors:
            self.waveplate_colors[waveplate_angle] = self.waveplate_palette[self.waveplate_counter % len(self.waveplate_palette)]
            self.waveplate_counter += 1
    
    def create_figure_1(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 1: Individual file legends"""
        if not self.plot_options.get('plot_individual_legends', True):
            return None
            
        print("Creating Figure 1: Individual file legends...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), height_ratios=[1, 1, 0.8])
        
        # Plot data for each subplot
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            label_material = f"{base_name} ({material})"
            label_waveplate = f"{base_name} ({waveplate_angle}°)"
            
            # Material-based subplot
            ax1.plot(time_data, velocity_filtered, label=label_material, marker='.', linestyle='-',
                    markersize=2, color=self.material_colors[material])
            
            # Waveplate angle-based subplot
            ax2.plot(time_data, velocity_filtered, label=label_waveplate, marker='.', linestyle='-',
                    markersize=2, color=self.waveplate_colors[waveplate_angle])
            
            # Zoomed subplot (material colors)
            ax3.plot(time_data, velocity_filtered, label=label_material, marker='.', linestyle='-',
                    markersize=2, color=self.material_colors[material])
        
        # Configure subplots
        for ax, title in [(ax1, 'Velocity Traces by Material'), 
                         (ax2, 'Velocity Traces by Waveplate Angle'),
                         (ax3, 'Zoomed Region: 0-20 ns (Material Colors)')]:
            ax.set_xlabel('Time (ns)', fontsize=20)
            ax.set_ylabel('Velocity (m/s)', fontsize=20)
            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.legend(fontsize=16, loc='best', ncol=2)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.minorticks_on()
            
            # Set zoom limits for third subplot
            if ax == ax3:
                ax.set_xlim(0, 20)
            
            # Add bounding box
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
        
        return fig
    
    def create_figure_2(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 2: Color meaning legends only"""
        if not self.plot_options.get('plot_color_meaning', True):
            return None
            
        print("Creating Figure 2: Color meaning legends...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), height_ratios=[1, 1, 0.8])
        
        # Plot data for each subplot
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            # Material-based subplot
            ax1.plot(time_data, velocity_filtered, marker='.', linestyle='-',
                    markersize=2, color=self.material_colors[material])
            
            # Waveplate angle-based subplot
            ax2.plot(time_data, velocity_filtered, marker='.', linestyle='-',
                    markersize=2, color=self.waveplate_colors[waveplate_angle])
            
            # Zoomed subplot (material colors)
            ax3.plot(time_data, velocity_filtered, marker='.', linestyle='-',
                    markersize=2, color=self.material_colors[material])
        
        # Configure subplots with color legends
        for ax, title, colors_dict in [(ax1, 'Velocity Traces by Material', self.material_colors),
                                      (ax2, 'Velocity Traces by Waveplate Angle', self.waveplate_colors),
                                      (ax3, 'Zoomed Region: 0-20 ns (Material Colors)', self.material_colors)]:
            ax.set_xlabel('Time (ns)', fontsize=20)
            ax.set_ylabel('Velocity (m/s)', fontsize=20)
            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.minorticks_on()
            
            # Set zoom limits for third subplot
            if ax == ax3:
                ax.set_xlim(0, 20)
            
            # Add color legend
            legend_elements = [plt.Line2D([0], [0], color=color, label=name) 
                             for name, color in colors_dict.items()]
            ax.legend(handles=legend_elements, fontsize=16, loc='best')
            
            # Add bounding box
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
        
        return fig
    
    def create_figure_3(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 3: Spread analysis"""
        if not self.plot_options.get('plot_spread_analysis', True):
            return None
            
        print("Creating Figure 3: Spread analysis...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[1, 1])
        
        # Group data by material and waveplate angle
        material_data = {}
        waveplate_data = {}
        
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            if material not in material_data:
                material_data[material] = []
            material_data[material].append((time_data, velocity_filtered, base_name))
            
            if waveplate_angle not in waveplate_data:
                waveplate_data[waveplate_angle] = []
            waveplate_data[waveplate_angle].append((time_data, velocity_filtered, base_name))
        
        # Create spread plots
        for ax, data_dict, title, colors_dict in [(ax1, material_data, 'Velocity Traces by Material (Spread Analysis)', self.material_colors),
                                                  (ax2, waveplate_data, 'Velocity Traces by Waveplate Angle (Spread Analysis)', self.waveplate_colors)]:
            ax.set_xlabel('Time (ns)', fontsize=20)
            ax.set_ylabel('Velocity (m/s)', fontsize=20)
            ax.set_title(title, fontsize=20, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.minorticks_on()
            
            # Add bounding box
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_color('black')
            
            # Plot spread data
            for name, traces in data_dict.items():
                if len(traces) > 0:
                    color = colors_dict[name]
                    # Plot individual traces with low alpha
                    for time_data, velocity_data, file_name in traces:
                        ax.plot(time_data, velocity_data, color=color, alpha=0.3, linewidth=0.5)
                    
                    # Calculate and plot min/max bounds
                    all_times = []
                    all_velocities = []
                    for time_data, velocity_data, _ in traces:
                        all_times.extend(time_data)
                        all_velocities.extend(velocity_data)
                    
                    if all_times and all_velocities:
                        time_array = np.array(all_times)
                        velocity_array = np.array(all_velocities)
                        
                        # Group by time bins to calculate statistics
                        time_bins = np.linspace(min(time_array), max(time_array), 100)
                        min_velocities = []
                        max_velocities = []
                        mean_velocities = []
                        
                        for i in range(len(time_bins) - 1):
                            mask = (time_array >= time_bins[i]) & (time_array < time_bins[i + 1])
                            if np.any(mask):
                                velocities_in_bin = velocity_array[mask]
                                velocities_in_bin = velocities_in_bin[~np.isnan(velocities_in_bin)]
                                if len(velocities_in_bin) > 0:
                                    min_velocities.append(np.min(velocities_in_bin))
                                    max_velocities.append(np.max(velocities_in_bin))
                                    mean_velocities.append(np.mean(velocities_in_bin))
                                else:
                                    min_velocities.append(np.nan)
                                    max_velocities.append(np.nan)
                                    mean_velocities.append(np.nan)
                            else:
                                min_velocities.append(np.nan)
                                max_velocities.append(np.nan)
                                mean_velocities.append(np.nan)
                        
                        # Plot min/max bounds and mean
                        time_centers = (time_bins[:-1] + time_bins[1:]) / 2
                        ax.fill_between(time_centers, min_velocities, max_velocities, 
                                     alpha=0.4, color=color, label=f'{name} (n={len(traces)})')
                        ax.plot(time_centers, mean_velocities, color=color, linewidth=2, alpha=0.8)
            
            ax.legend(fontsize=16, loc='best')
        
        return fig
    
    def create_figure_4(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 4: Maximum velocity vs waveplate angle"""
        if not self.plot_options.get('plot_velocity_vs_angle', True):
            return None
            
        print("Creating Figure 4: Maximum velocity vs waveplate angle...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Collect scatter data
        scatter_data = {}
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            # Calculate maximum velocity (average between 300-400ns)
            time_ns = time_data.copy()
            if np.nanmax(time_ns) < 1000:  # If time is in ns already
                pass
            else:
                time_ns = time_ns / 1e9  # Convert to ns
            
            # Find data points between 300-400ns
            mask_300_400 = (time_ns >= 300) & (time_ns <= 400)
            if np.any(mask_300_400):
                velocities_300_400 = velocity_filtered[mask_300_400]
                velocities_300_400 = velocities_300_400[~np.isnan(velocities_300_400)]
                if len(velocities_300_400) > 0:
                    max_velocity = np.mean(velocities_300_400)
                    
                    # Convert waveplate angle to numeric for plotting
                    try:
                        waveplate_angle_numeric = float(str(waveplate_angle).replace('°', '').replace('Unknown', '0'))
                    except:
                        waveplate_angle_numeric = 0
                    
                    # Collect data for scatter plot
                    if material not in scatter_data:
                        scatter_data[material] = []
                    scatter_data[material].append((waveplate_angle_numeric, max_velocity, base_name))
        
        # Plot scatter data
        for material, data_points in scatter_data.items():
            if len(data_points) > 0:
                color = self.material_colors[material]
                waveplate_angles = [point[0] for point in data_points]
                max_velocities = [point[1] for point in data_points]
                
                # Plot scatter points
                ax.scatter(waveplate_angles, max_velocities, color=color, s=100, alpha=0.7, 
                           label=f'{material} (n={len(data_points)})')
        
        # Configure scatter plot
        ax.set_xlabel('Wave Plate Angle (degrees)', fontsize=20)
        ax.set_ylabel('Maximum Velocity (m/s)', fontsize=20)
        ax.set_title('Maximum Velocity vs Wave Plate Angle by Material', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, loc='best', title='Flyer Material', title_fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.minorticks_on()
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')
        
        return fig
    
    def create_figure_5(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 5: Shot time vs material"""
        if not self.plot_options.get('plot_shot_time_vs_material', True):
            return None
            
        print("Creating Figure 5: Shot time vs material...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Collect shot time data
        shot_time_data = {}
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            # Get shot time from parameter data
            shot_time_s = None
            if base_name in self.param_data:
                exp_info = self.param_data[base_name]
                shot_time_from_param = exp_info.get('shot_time', None)
                if shot_time_from_param is not None and shot_time_from_param != 'Unknown':
                    try:
                        shot_time_s = float(shot_time_from_param)
                    except (ValueError, TypeError):
                        pass
            
            if shot_time_s is not None:
                if material not in shot_time_data:
                    shot_time_data[material] = []
                shot_time_data[material].append((shot_time_s, base_name))
        
        # Create box plot
        material_order = ['Al', 'Ti', 'Cu']
        box_data = []
        box_labels = []
        box_colors = []
        
        for material in material_order:
            if material in shot_time_data and len(shot_time_data[material]) > 0:
                shot_times = [point[0] for point in shot_time_data[material]]
                box_data.append(shot_times)
                box_labels.append(f'{material} (n={len(shot_times)})')
                box_colors.append(self.material_colors[material])
        
        if box_data:
            # Calculate y-axis limits
            all_times = [time for data in box_data for time in data]
            if all_times:
                y_min = max(0, min(all_times) * 0.9)
                y_max = max(all_times) * 1.1
                if y_max - y_min < 0.001:
                    y_max = y_min + 0.001
                
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                              showfliers=True, flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'red'},
                              widths=0.6)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Color the medians
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                
                ax.set_ylim(y_min, y_max)
        
        # Configure box plot
        ax.set_xlabel('Material', fontsize=16)
        ax.set_ylabel('Shot Time (s)', fontsize=16)
        ax.set_title('Shot Time vs Material (Box Plot with Outliers)', fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.minorticks_on()
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('black')
        
        return fig
    
    def create_figure_6(self, all_data: List[Tuple]) -> plt.Figure:
        """Create Figure 6: PDV power vs material"""
        if not self.plot_options.get('plot_pdv_power_vs_material', True):
            return None
            
        print("Creating Figure 6: PDV power vs material...")
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Collect PDV power data
        pdv_power_data = {}
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            # Calculate PDV return power (average power in the signal)
            if len(velocity_filtered) > 0:
                velocity_power = np.abs(velocity_filtered)
                mean_power = np.mean(velocity_power)
                
                if mean_power > 1e-10:
                    pdv_power_dbm = 10 * np.log10(mean_power)
                    
                    if material not in pdv_power_data:
                        pdv_power_data[material] = []
                    pdv_power_data[material].append((pdv_power_dbm, base_name))
        
        # Plot PDV power data
        material_order = ['Al', 'Ti', 'Cu']
        material_positions = {material: i for i, material in enumerate(material_order)}
        
        for material, data_points in pdv_power_data.items():
            if len(data_points) > 0 and material in material_positions:
                color = self.material_colors[material]
                pdv_powers = [point[0] for point in data_points]
                file_names = [point[1] for point in data_points]
                
                # Use material position for x-axis
                x_pos = material_positions[material]
                x_positions = [x_pos] * len(pdv_powers)
                
                # Plot scatter points
                ax.scatter(x_positions, pdv_powers, color=color, s=100, alpha=0.7, 
                           label=f'{material} (n={len(data_points)})')
                
                # Add file name annotations for some points
                if len(data_points) <= 10:
                    for i, (pdv_power, file_name) in enumerate(data_points):
                        ax.annotate(file_name, (x_pos, pdv_power), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, alpha=0.8)
        
        # Configure PDV power plot
        ax.set_xlabel('Material', fontsize=20)
        ax.set_ylabel('PDV Return Power (dBm)', fontsize=20)
        ax.set_title('PDV Return Power vs Material', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, loc='best', title='Flyer Material', title_fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.minorticks_on()
        
        # Set x-axis ticks to material names
        ax.set_xticks(range(len(material_order)))
        ax.set_xticklabels(material_order)
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_color('black')
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str):
        """Save figure to output directory"""
        if fig is None:
            return
            
        fig.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filename}")
    
    def export_csv_data(self, all_data: List[Tuple]):
        """Export analysis data to CSV"""
        print("Exporting analysis data to CSV...")
        
        # Collect all data for CSV export
        csv_data = []
        for time_data, velocity_filtered, base_name, material, waveplate_angle in all_data:
            # Calculate maximum velocity (average between 300-400ns)
            time_ns = time_data.copy()
            if np.nanmax(time_ns) < 1000:
                pass
            else:
                time_ns = time_ns / 1e9
            
            mask_300_400 = (time_ns >= 300) & (time_ns <= 400)
            max_velocity = np.nan
            if np.any(mask_300_400):
                velocities_300_400 = velocity_filtered[mask_300_400]
                velocities_300_400 = velocities_300_400[~np.isnan(velocities_300_400)]
                if len(velocities_300_400) > 0:
                    max_velocity = np.mean(velocities_300_400)
            
            # Get additional data from parameter file
            shot_time_s = None
            pdv_power_dbm = None
            laser_energy_mj = None
            
            if base_name in self.param_data:
                exp_info = self.param_data[base_name]
                
                # Get shot time
                shot_time_from_param = exp_info.get('shot_time', None)
                if shot_time_from_param is not None and shot_time_from_param != 'Unknown':
                    try:
                        shot_time_s = float(shot_time_from_param)
                    except (ValueError, TypeError):
                        pass
                
                # Get laser energy
                for key, value in exp_info.items():
                    if any(term in key.lower() for term in ['laser', 'energy', 'pulse']) and value != 'Unknown':
                        try:
                            laser_energy_mj = float(value)
                            break
                        except (ValueError, TypeError):
                            continue
            
            # Calculate PDV power
            if len(velocity_filtered) > 0:
                velocity_power = np.abs(velocity_filtered)
                mean_power = np.mean(velocity_power)
                if mean_power > 1e-10:
                    pdv_power_dbm = 10 * np.log10(mean_power)
            
            # Convert waveplate angle to numeric
            try:
                waveplate_angle_numeric = float(str(waveplate_angle).replace('°', '').replace('Unknown', '0'))
            except:
                waveplate_angle_numeric = 0
            
            csv_data.append({
                'file_name': base_name,
                'material': material,
                'waveplate_angle_degrees': waveplate_angle_numeric,
                'max_velocity_ms': max_velocity,
                'shot_time_s': shot_time_s,
                'pdv_return_power_dbm': pdv_power_dbm,
                'laser_energy_mj': laser_energy_mj
            })
        
        # Save to CSV file
        if csv_data:
            csv_df = pd.DataFrame(csv_data)
            csv_path = os.path.join(self.output_dir, 'analysis_data.csv')
            csv_df.to_csv(csv_path, index=False)
            print(f"Saved {len(csv_data)} data points to: analysis_data.csv")
            print(f"Columns: {list(csv_df.columns)}")
            
            # Show data summary
            for material in csv_df['material'].unique():
                material_data = csv_df[csv_df['material'] == material]
                print(f"{material}: {len(material_data)} samples")
        else:
            print("No data available for CSV export")
    
    def run_enhanced_plotting(self):
        """Run the complete enhanced plotting workflow"""
        print("Starting enhanced plotting workflow...")
        
        # Find velocity files
        velocity_files = self._find_velocity_files()
        if not velocity_files:
            print("No velocity files found!")
            return
        
        # Load and process all data
        all_data = []
        for file_path in velocity_files:
            time_data, velocity, velocity_filtered = self._load_velocity_data(file_path)
            if len(time_data) > 0:
                base_name = os.path.basename(file_path).replace('--velocity--smooth.csv', '')
                material, waveplate_angle = self._get_material_and_angle(base_name)
                self._assign_colors(material, waveplate_angle)
                all_data.append((time_data, velocity_filtered, base_name, material, waveplate_angle))
        
        if not all_data:
            print("No valid data found!")
            return
        
        print(f"Processing {len(all_data)} files...")
        
        # Create and save figures
        figures = [
            (self.create_figure_1(all_data), 'all_smoothed_velocity_traces_with_legends.png'),
            (self.create_figure_2(all_data), 'all_smoothed_velocity_traces_color_meaning.png'),
            (self.create_figure_3(all_data), 'all_smoothed_velocity_traces_spread.png'),
            (self.create_figure_4(all_data), 'max_velocity_vs_waveplate_angle.png'),
            (self.create_figure_5(all_data), 'shot_time_vs_material.png'),
            (self.create_figure_6(all_data), 'pdv_power_vs_material.png')
        ]
        
        for fig, filename in figures:
            self.save_figure(fig, filename)
        
        # Export CSV data
        self.export_csv_data(all_data)
        
        print("Enhanced plotting completed successfully!")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description='Enhanced Plotting for ALPSS-SPADE Analysis')
    parser.add_argument('--input_dir', required=True, help='Directory containing velocity files')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots and data')
    parser.add_argument('--param_file', help='Optional parameter file (Excel format)')
    parser.add_argument('--plot_options', nargs='+', help='Plot options to enable (Figure1 Figure2 Figure3 Figure4 Figure5 Figure6)')
    
    args = parser.parse_args()
    
    # Parse plot options
    plot_options = {}
    if args.plot_options:
        plot_options = {
            'plot_individual_legends': 'Figure1' in args.plot_options,
            'plot_color_meaning': 'Figure2' in args.plot_options,
            'plot_spread_analysis': 'Figure3' in args.plot_options,
            'plot_velocity_vs_angle': 'Figure4' in args.plot_options,
            'plot_shot_time_vs_material': 'Figure5' in args.plot_options,
            'plot_pdv_power_vs_material': 'Figure6' in args.plot_options
        }
    
    # Create and run enhanced plotting
    plotting = EnhancedPlotting(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        param_file=args.param_file,
        plot_options=plot_options
    )
    
    plotting.run_enhanced_plotting()


if __name__ == "__main__":
    main() 