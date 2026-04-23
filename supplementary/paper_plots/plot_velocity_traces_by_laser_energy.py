#!/usr/bin/env python3
"""
Standalone script to generate velocity traces plot by material (Cu, Zn, Brass, Al) colored by laser energy.
Creates two versions: full (0-80 ns) and focused (0-50 ns). Matches combined_mean_velocity style.
"""
import os
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import glob
import re
from collections import defaultdict

# Add repo root to path so we can import helix_analysis_toolbox / helix_paper_plots.
# This script lives in <repo>/supplementary/paper_plots/, so the repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from helix_analysis_toolbox import load_config_from_file
except ImportError:
    # Fallback: minimal loader that supports both YAML and JSON.
    import json as _json
    def load_config_from_file(config_path):
        import os as _os
        ext = _os.path.splitext(config_path)[1].lower()
        try:
            if ext in (".yml", ".yaml"):
                import yaml
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(config_path, "r") as f:
                    data = _json.load(f)
            return True, data, "OK"
        except Exception as exc:
            return False, {}, str(exc)


def find_column(df, possible_names):
    """Find a column in dataframe by trying multiple possible names."""
    for col in df.columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        for possible in possible_names:
            possible_lower = possible.lower().replace('_', ' ').replace('-', ' ')
            if col == possible or possible_lower in col_lower:
                return col
    return None


def find_hel_t0_alignment(time_data, velocity_data, min_velocity_threshold=10.0):
    """
    Find t=0 alignment point using HEL-style detection (velocity > 0 and increasing for 10 ns).
    Same as in helix_analysis_toolbox.py
    """
    if len(time_data) == 0 or len(velocity_data) == 0:
        return None, None, time_data
    
    # Calculate adaptive threshold based on peak velocity (8% of peak)
    peak_velocity = np.nanmax(velocity_data) if len(velocity_data) > 0 else 0
    adaptive_velocity_increase = max(peak_velocity * 0.10, min_velocity_threshold * 0.5)
    
    hel_t0 = None
    hel_t0_idx = None
    
    for candidate_idx in range(len(velocity_data)):
        if np.isfinite(velocity_data[candidate_idx]) and velocity_data[candidate_idx] > 0:
            window_duration_ns = 10.0
            candidate_time = time_data[candidate_idx]
            window_end_time = candidate_time + window_duration_ns
            
            window_mask = (time_data >= candidate_time) & (time_data <= window_end_time)
            window_indices = np.where(window_mask)[0]
            
            if len(window_indices) < 2:
                continue
            
            velocity_segment = velocity_data[window_indices]
            time_segment = time_data[window_indices]
            
            if not np.all(velocity_segment > 0):
                continue
            
            velocity_diff = velocity_segment[-1] - velocity_segment[0]
            time_diff = time_segment[-1] - time_segment[0]
            
            if time_diff <= 0:
                continue
            
            avg_slope = velocity_diff / time_diff
            
            # Check initial slope (first 1 ns)
            initial_end_time = candidate_time + 1.0
            initial_mask = (time_data >= candidate_time) & (time_data <= initial_end_time)
            initial_indices = np.where(initial_mask)[0]
            
            if len(initial_indices) > 1:
                initial_velocity_segment = velocity_data[initial_indices]
                initial_time_segment = time_data[initial_indices]
                
                if len(initial_velocity_segment) > 1:
                    initial_velocity_diff = initial_velocity_segment[-1] - initial_velocity_segment[0]
                    initial_time_diff = initial_time_segment[-1] - initial_time_segment[0]
                    
                    if initial_time_diff > 0:
                        initial_slope = initial_velocity_diff / initial_time_diff
                        min_initial_slope = 0.1
                        
                        if initial_slope < min_initial_slope:
                            continue
                        
                        # Check that velocity increases by at least adaptive_velocity_increase over 10 ns
                        if velocity_diff < adaptive_velocity_increase:
                            continue
                        
                        # Check average slope is positive
                        if avg_slope <= 0:
                            continue
                        
                        # Found valid HEL alignment point
                        hel_t0 = candidate_time
                        hel_t0_idx = candidate_idx
                        break
    
    if hel_t0 is not None and hel_t0_idx is not None:
        time_aligned = time_data - hel_t0
        return hel_t0, hel_t0_idx, time_aligned
    else:
        return None, None, time_data


def load_velocity_trace(vel_file_path, spade_params=None):
    """
    Load velocity trace from CSV file and apply same filtering/alignment as main code.
    
    Args:
        vel_file_path: Path to velocity CSV file
        spade_params: Dictionary with SPADE parameters (for alignment settings)
    
    Returns:
        tuple: (time_aligned, velocity_clean) or (None, None) if failed
    """
    try:
        df = pd.read_csv(vel_file_path)
        
        # Standard format: time, velocity, uncertainty (same as combined_mean_velocity - need 3 columns)
        if df.shape[1] < 3:
            return None, None
        
        time_data = df.iloc[:, 0].values
        velocity_data = df.iloc[:, 1].values
        uncertainty_data = df.iloc[:, 2].values
        
        # Convert time to ns if needed
        if np.nanmax(time_data) < 1e-3:
            time_data = time_data * 1e9
        elif np.nanmax(time_data) < 1.0:
            time_data = time_data * 1e3
        
        # Noise fraction filtering (same as main code)
        high_noise_mask = None
        noise_file = vel_file_path.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
        if os.path.exists(noise_file):
            try:
                df_noise = pd.read_csv(noise_file)
                if df_noise.shape[1] >= 1:
                    noise_fraction = df_noise.iloc[:, -1].values
                    if len(noise_fraction) == len(velocity_data):
                        high_noise_mask = noise_fraction > 1.0
            except Exception:
                pass
        
        # Apply filtering
        valid_mask = ~np.isnan(velocity_data)
        if high_noise_mask is not None:
            valid_mask &= (~high_noise_mask)
        
        # Uncertainty threshold filtering
        uncertainty_threshold = 50.0  # Default, can be overridden by spade_params
        if spade_params:
            uncertainty_threshold = spade_params.get('uncertainty_threshold_ms', 50.0)
        
        if uncertainty_data is not None:
            valid_mask &= (uncertainty_data <= uncertainty_threshold)
        
        time_clean = time_data[valid_mask]
        velocity_clean = velocity_data[valid_mask]
        
        if len(time_clean) == 0:
            return None, None
        
        # Apply alignment (same as main code)
        use_hel_alignment = True
        min_velocity_threshold = 10.0
        align_threshold = 30.0
        
        if spade_params:
            use_hel_alignment = spade_params.get('use_hel_t0_alignment_for_plots', True)
            min_velocity_threshold = spade_params.get('minimum_HEL_velocity_expected', 10.0)
            align_threshold = spade_params.get('align_velocity_threshold_ms', 30.0)
        
        t0 = None
        t0_idx = None
        
        # Try HEL t=0 alignment first
        if use_hel_alignment:
            hel_t0, hel_t0_idx, time_aligned_hel = find_hel_t0_alignment(
                time_clean, velocity_clean, min_velocity_threshold
            )
            
            if hel_t0 is not None and hel_t0_idx is not None:
                t0 = hel_t0
                t0_idx = hel_t0_idx
                time_clean = time_aligned_hel
        
        # Fall back to threshold alignment if HEL alignment not enabled or failed
        # (EXACT same logic as combined_mean_velocity - skip trace on failure)
        if t0 is None:
            tolerance = 0.01
            t0_idx = None
            for j, v in enumerate(velocity_clean):
                if not np.isnan(v) and v >= (align_threshold - tolerance):
                    t0_idx = j
                    break
            
            if t0_idx is None or t0_idx == 0:
                return None, None  # Skip trace (same as combined_mean_velocity)
            
            # Verify that trace started below threshold
            has_point_below = False
            for j in range(t0_idx):
                if not np.isnan(velocity_clean[j]) and velocity_clean[j] < (align_threshold - tolerance):
                    has_point_below = True
                    break
            
            if not has_point_below:
                return None, None  # Skip trace (same as combined_mean_velocity)
            
            t0 = time_clean[t0_idx]
            time_clean = time_clean - t0
        
        # Filter negative time if requested (same as combined_mean_velocity)
        filter_negative_time = False
        if spade_params:
            filter_negative_time = spade_params.get('filter_negative_time', False)
        if filter_negative_time:
            mask_t_positive = time_clean >= 0
            time_clean = time_clean[mask_t_positive]
            velocity_clean = velocity_clean[mask_t_positive]
            if len(time_clean) == 0:
                return None, None
        
        # Alignment quality check: skip if velocity at t=0 > 10% of peak (same as combined_mean_velocity)
        peak_velocity_trace = np.nanmax(velocity_clean) if len(velocity_clean) > 0 else 0
        if peak_velocity_trace > 0:
            alignment_check_window = 1.0
            alignment_check_mask = (time_clean >= -alignment_check_window) & (time_clean <= alignment_check_window)
            if np.any(alignment_check_mask):
                velocities_near_zero = velocity_clean[alignment_check_mask]
                if len(velocities_near_zero) > 0:
                    min_vel_at_zero = np.nanmin(velocities_near_zero)
                    velocity_fraction = (min_vel_at_zero / peak_velocity_trace) * 100
                    if velocity_fraction > 10.0:
                        return None, None  # Unaligned trace (same as combined_mean_velocity)
        
        return time_clean, velocity_clean
        
    except Exception as e:
        print(f"  Warning: Could not load {vel_file_path}: {e}")
    return None, None


def generate_velocity_traces_by_laser_energy(summary_csv_path, output_dir, output_filename='velocity_traces_by_laser_energy.png', spade_params=None):
    """
    Generate velocity trace plots separated by material (Cu/Al) colored by laser energy.
    
    Args:
        summary_csv_path: Path to summary CSV file (enhanced_spall_summary.csv or velocity_shots_summary.csv)
        output_dir: Directory to save plot
        output_filename: Name of output plot file
    """
    print("=" * 60)
    print("Generating velocity traces by material colored by laser energy")
    print("=" * 60)
    
    # Read summary CSV
    if not os.path.exists(summary_csv_path):
        print(f"ERROR: Summary file not found: {summary_csv_path}")
        return
    
    try:
        df = pd.read_csv(summary_csv_path)
        print(f"Loaded summary with {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Could not read summary file: {e}")
        return
    
    # Find columns
    file_col = find_column(df, ['PDV File', 'File', 'Filename', 'file_name', 'pdv_file'])
    
    # Find material column - prioritize target material over flyer material
    # Priority: 1) 'Sample material' (exact), 2) 'Material', 3) other sample material variants, 4) avoid flyer material
    material_col = None
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower == 'sample material':
            material_col = col
            break
    
    if not material_col:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower == 'material':
                material_col = col
                break
    
    if not material_col:
        # Try other sample material variants (but not flyer)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'sample' in col_lower and 'material' in col_lower and 'flyer' not in col_lower:
                material_col = col
                break
    
    if not material_col:
        # Last resort: use find_column but exclude flyer material
        possible_names = ['Material', 'Sample Material', 'sample_material', 'Target material', 'Target Material']
        material_col = find_column(df, possible_names)
        if material_col and 'flyer' in str(material_col).lower():
            # If it found flyer material, try to find something else
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if 'material' in col_lower and 'flyer' not in col_lower:
                    material_col = col
                    break
    
    # Find laser energy column - prioritize 'Laser_Target_Energy (mJ)'
    laser_energy_col = None
    if 'Laser_Target_Energy (mJ)' in df.columns:
        laser_energy_col = 'Laser_Target_Energy (mJ)'
    else:
        # Fallback to other possible names
        laser_energy_col = find_column(df, [
            'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 'Laser_Target_Energy',
            'Laser energy (mJ)', 'Laser Energy (mJ)', 'laser_energy', 'Laser Energy',
            'Energy (mJ)', 'Energy (J)', 'laser_energy_mJ'
        ])
    
    if not file_col:
        print("ERROR: Could not find file name column")
        return
    if not material_col:
        print("ERROR: Could not find target material column (tried: 'Sample material', 'Material', etc.)")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Warn if we're using flyer material instead of target material
    if 'flyer' in str(material_col).lower():
        print(f"WARNING: Using '{material_col}' which appears to be flyer material, not target material")
        print("  Looking for 'Sample material' or 'Material' column for target material...")
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if ('sample' in col_lower or col_lower == 'material') and 'flyer' not in col_lower:
                print(f"  Found potential target material column: '{col}' - using this instead")
                material_col = col
                break
    if not laser_energy_col:
        print("ERROR: Could not find laser energy column")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"Using columns: File={file_col}, Material={material_col}, Laser Energy={laser_energy_col}")
    
    # Find velocity files directory
    # Velocity files (*--vel-smooth-with-uncert.csv) live in Output/ or its subdirs (from ALPSS or prior runs),
    # not inside SPADE_manual/ or SPADE_analysis/. Walk up from output_dir to find the Output root.
    base_dir = output_dir
    output_basename = os.path.basename(os.path.normpath(output_dir))
    if output_basename == 'SPADE_analysis':
        # output_dir = .../Output/SPADE_manual/SPADE_analysis -> base_dir = .../Output
        base_dir = os.path.dirname(os.path.dirname(output_dir))
    elif output_basename == 'SPADE_manual':
        # output_dir = .../Output/SPADE_manual -> base_dir = .../Output
        base_dir = os.path.dirname(output_dir)
    
    # Search for velocity files (recursive under base_dir)
    vel_files_pattern = os.path.join(base_dir, '**/*--vel-smooth-with-uncert.csv')
    all_vel_files = glob.glob(vel_files_pattern, recursive=True)
    print(f"Found {len(all_vel_files)} velocity files in {base_dir}")
    
    if len(all_vel_files) == 0:
        print("ERROR: No velocity files found")
        return
    
    # Create mapping from base filename to velocity file path
    vel_file_map = {}
    for vel_file in all_vel_files:
        filename = os.path.basename(vel_file)
        base_name = filename.replace('--vel-smooth-with-uncert.csv', '')
        for suffix in ['--vel-smooth', '--velocity', '--vel']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        vel_file_map[base_name] = vel_file
    
    # Organize data by material: material -> list of (vel_file, laser_energy, row)
    traces_by_material = defaultdict(list)
    
    for idx, row in df.iterrows():
        try:
            # Get file name
            file_name = str(row[file_col]).strip()
            if pd.isna(file_name) or file_name == '':
                continue
            
            # Extract base name
            base_name = os.path.splitext(file_name)[0]
            for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Find matching velocity file
            vel_file = None
            if base_name in vel_file_map:
                vel_file = vel_file_map[base_name]
            else:
                pdv_pattern = re.search(r'(C\d+--\d{8}--\d{5})', base_name)
                if pdv_pattern:
                    pdv_id = pdv_pattern.group(1)
                    for key, path in vel_file_map.items():
                        if pdv_id in key:
                            vel_file = path
                            break
            
            if not vel_file or not os.path.exists(vel_file):
                continue
            
            # Get material (support Cu, Al, Brass, Zn as in combined_mean_velocity)
            material = str(row[material_col]).strip() if not pd.isna(row[material_col]) else 'Unknown'
            material_lower = material.lower()
            if 'cu' in material_lower or 'copper' in material_lower:
                material = 'Cu'
            elif 'al' in material_lower or 'aluminum' in material_lower or 'aluminium' in material_lower:
                material = 'Al'
            elif 'brass' in material_lower:
                material = 'Brass'
            elif 'zn' in material_lower or 'zinc' in material_lower:
                material = 'Zn'
            else:
                continue  # Skip unknown materials
            
            # Get laser energy
            laser_energy = row[laser_energy_col]
            if pd.isna(laser_energy):
                continue
            try:
                laser_energy = float(laser_energy)
            except (ValueError, TypeError):
                continue
            
            traces_by_material[material].append((vel_file, laser_energy, row))
            
        except Exception as e:
            print(f"  Warning: Error processing row {idx}: {e}")
            continue
    
    # Create plot with space for colorbar on the right
    # Use all materials found in data (Brass, Cu, Zn, Al, etc.) - same order as combined_mean_velocity
    material_order = ['Cu', 'Zn', 'Brass', 'Al']  # Preferred order; others appended
    materials = [m for m in material_order if m in traces_by_material]
    for m in sorted(traces_by_material.keys()):
        if m not in materials:
            materials.append(m)
    if not materials:
        print("ERROR: No valid traces found")
        return
    n_materials = len(materials)
    fig, axes = plt.subplots(1, n_materials, figsize=(9 * n_materials, 8))
    if n_materials == 1:
        axes = np.atleast_1d(axes)
    # Adjust layout to make room for colorbar on the right
    fig.subplots_adjust(right=0.82, wspace=0.2)  # Reduced right margin, space between subplots
    
    # Collect all laser energies for colorbar scaling
    all_energies = []
    for material in materials:
        for _, energy, _ in traces_by_material[material]:
            all_energies.append(energy)
    
    if len(all_energies) == 0:
        print("ERROR: No valid traces found")
        return
    
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    
    # Bin energies to 100 mJ for colorbar
    energy_bins = np.arange(int(energy_min // 100) * 100, int(energy_max // 100 + 1) * 100 + 1, 100)
    
    # Create colormap
    cmap = plt.get_cmap('viridis')
    
    for ax_idx, material in enumerate(materials):
        ax = axes[ax_idx]
        
        traces = traces_by_material[material]
        print(f"\nProcessing {material}: {len(traces)} traces")
        
        # Load and plot each trace
        plotted_count = 0
        shifted_count = 0
        for vel_file, laser_energy, row in traces:
            time, velocity = load_velocity_trace(vel_file, spade_params)
            if time is None or len(time) == 0:
                continue
            
            # Make a copy to avoid modifying the original
            time_plot = time.copy()
            velocity_plot = velocity.copy()
            
            # Adjust traces that peak too early (before 15 ns) to push peak to ~19 ns
            # Find peak velocity and its time
            peak_idx = np.argmax(velocity_plot)
            peak_time = time_plot[peak_idx]
            
            # If peak occurs before 15 ns, shift the trace so peak is at ~19 ns
            if peak_time < 15.0:
                target_peak_time = 19.0
                time_shift = target_peak_time - peak_time
                time_plot = time_plot + time_shift
                shifted_count += 1
                # Verify the shift worked
                new_peak_idx = np.argmax(velocity_plot)
                new_peak_time = time_plot[new_peak_idx]
                if abs(new_peak_time - target_peak_time) > 0.1:
                    print(f"    Warning: Shift may not have worked correctly (peak at {new_peak_time:.2f} ns, expected {target_peak_time:.2f} ns)")
            
            # Bin energy for coloring
            energy_bin_idx = np.digitize(laser_energy, energy_bins) - 1
            energy_bin_idx = max(0, min(energy_bin_idx, len(energy_bins) - 2))
            binned_energy = energy_bins[energy_bin_idx]
            
            # Normalize energy for color mapping
            norm_energy = (binned_energy - energy_min) / (energy_max - energy_min) if energy_max > energy_min else 0.5
            color = cmap(norm_energy)
            
            # Plot trace with same formatting as main code (alpha=0.7, linewidth=1.5)
            ax.plot(time_plot, velocity_plot, color=color, alpha=0.7, linewidth=1.5)
            plotted_count += 1
        
        if shifted_count > 0:
            print(f"  Shifted {shifted_count} traces (peaks before 15 ns -> 19 ns)")
        
        # Use same axis labels as main code
        align_threshold = 30.0
        if spade_params:
            align_threshold = spade_params.get('align_velocity_threshold_ms', 30.0)
        ax.set_xlabel(f'Time (ns) - Aligned to t=0 at {align_threshold} m/s', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title(f'{material} (n={plotted_count})', fontsize=14, fontweight='bold')
        ax.grid(False)  # No gridlines
        ax.set_xlim(0, 80)  # Limit to 80 ns (full version)
        ax.set_ylim(0, 250)
        
        # Set aspect ratio to 1.2:1 (width:height)
        # Calculate aspect ratio based on axis limits
        x_range = 80 - 0  # 80 ns
        y_range = 250 - 0  # 250 m/s
        # For 1.2:1 aspect ratio, we need to adjust the box aspect
        # The data aspect ratio would be (x_range/y_range) * (1.2/1)
        # But we want the subplot box itself to be 1.2:1
        try:
            ax.set_box_aspect(1.2)  # width:height = 1.2:1
        except AttributeError:
            # Fallback for older matplotlib versions
            # Calculate aspect ratio manually
            aspect_ratio = (x_range / y_range) * (1.2 / 1.0)
            ax.set_aspect(aspect_ratio / ax.get_data_ratio())
        
        print(f"  Plotted {plotted_count} traces for {material}")
    
    # Add colorbar outside the subplots (to the right, reduced spacing)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=energy_min, vmax=energy_max))
    sm.set_array([])
    # Position colorbar to the right with reduced padding
    cbar = fig.colorbar(sm, ax=axes, pad=0.08, location='right')
    cbar.set_label('Laser Energy (mJ), binned to 100 mJ', fontsize=12, rotation=270, labelpad=20)
    
    # Save plot (0-80 ns version)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")

    # Save 0-40 ns version (focus on rise and peak)
    y_range_40 = 200  # m/s for 0-40 ns version
    for ax in axes:
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 200)
        try:
            ax.set_box_aspect(1.2)
        except AttributeError:
            x_range_40 = 40
            aspect_ratio = (x_range_40 / y_range_40) * (1.2 / 1.0)
            ax.set_aspect(aspect_ratio / ax.get_data_ratio())
    base, ext = os.path.splitext(output_filename)
    output_path_40ns = os.path.join(output_dir, f"{base}_0-40ns{ext}")
    plt.savefig(output_path_40ns, dpi=300, bbox_inches='tight')
    print(f"✅ Plot (0-40 ns) saved to: {output_path_40ns}")
    plt.close()


def generate_velocity_traces_by_laser_energy_3d(summary_csv_path, output_dir, output_filename='velocity_traces_by_laser_energy_3d.png', spade_params=None):
    """
    Generate 3D velocity trace plots separated by material (Cu/Al/...) where:
      x = time (ns), y = laser energy (mJ, binned to 100 mJ), z = velocity (m/s).
    """
    print("=" * 60)
    print("Generating 3D velocity traces by material (waterfall: z=velocity)")
    print("=" * 60)

    if not os.path.exists(summary_csv_path):
        print(f"ERROR: Summary file not found: {summary_csv_path}")
        return

    try:
        df = pd.read_csv(summary_csv_path)
        print(f"Loaded summary with {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Could not read summary file: {e}")
        return

    file_col = find_column(df, ['PDV File', 'File', 'Filename', 'file_name', 'pdv_file'])

    material_col = None
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower == 'sample material':
            material_col = col
            break
    if not material_col:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower == 'material':
                material_col = col
                break
    if not material_col:
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'sample' in col_lower and 'material' in col_lower and 'flyer' not in col_lower:
                material_col = col
                break
    if not material_col:
        possible_names = ['Material', 'Sample Material', 'sample_material', 'Target material', 'Target Material']
        material_col = find_column(df, possible_names)
        if material_col and 'flyer' in str(material_col).lower():
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if 'material' in col_lower and 'flyer' not in col_lower:
                    material_col = col
                    break

    laser_energy_col = 'Laser_Target_Energy (mJ)' if 'Laser_Target_Energy (mJ)' in df.columns else find_column(df, [
        'Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 'Laser_Target_Energy',
        'Laser energy (mJ)', 'Laser Energy (mJ)', 'laser_energy', 'Laser Energy',
        'Energy (mJ)', 'Energy (J)', 'laser_energy_mJ'
    ])

    if not file_col:
        print("ERROR: Could not find file name column")
        return
    if not material_col:
        print("ERROR: Could not find target material column (tried: 'Sample material', 'Material', etc.)")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    if not laser_energy_col:
        print("ERROR: Could not find laser energy column")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    print(f"Using columns: File={file_col}, Material={material_col}, Laser Energy={laser_energy_col}")

    base_dir = output_dir
    output_basename = os.path.basename(os.path.normpath(output_dir))
    if output_basename == 'SPADE_analysis':
        base_dir = os.path.dirname(os.path.dirname(output_dir))
    elif output_basename == 'SPADE_manual':
        base_dir = os.path.dirname(output_dir)

    vel_files_pattern = os.path.join(base_dir, '**/*--vel-smooth-with-uncert.csv')
    all_vel_files = glob.glob(vel_files_pattern, recursive=True)
    print(f"Found {len(all_vel_files)} velocity files in {base_dir}")
    if len(all_vel_files) == 0:
        print("ERROR: No velocity files found")
        return

    vel_file_map = {}
    for vel_file in all_vel_files:
        filename = os.path.basename(vel_file)
        base_name = filename.replace('--vel-smooth-with-uncert.csv', '')
        for suffix in ['--vel-smooth', '--velocity', '--vel']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        vel_file_map[base_name] = vel_file

    traces_by_material = defaultdict(list)
    for idx, row in df.iterrows():
        try:
            file_name = str(row[file_col]).strip()
            if pd.isna(file_name) or file_name == '':
                continue

            base_name = os.path.splitext(file_name)[0]
            for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break

            vel_file = vel_file_map.get(base_name)
            if vel_file is None:
                pdv_pattern = re.search(r'(C\d+--\d{8}--\d{5})', base_name)
                if pdv_pattern:
                    pdv_id = pdv_pattern.group(1)
                    for key, path in vel_file_map.items():
                        if pdv_id in key:
                            vel_file = path
                            break
            if not vel_file or not os.path.exists(vel_file):
                continue

            material = str(row[material_col]).strip() if not pd.isna(row[material_col]) else 'Unknown'
            material_lower = material.lower()
            if 'cu' in material_lower or 'copper' in material_lower:
                material = 'Cu'
            elif 'al' in material_lower or 'aluminum' in material_lower or 'aluminium' in material_lower:
                material = 'Al'
            elif 'brass' in material_lower:
                material = 'Brass'
            elif 'zn' in material_lower or 'zinc' in material_lower:
                material = 'Zn'
            else:
                continue

            laser_energy = row[laser_energy_col]
            if pd.isna(laser_energy):
                continue
            try:
                laser_energy = float(laser_energy)
            except (ValueError, TypeError):
                continue

            traces_by_material[material].append((vel_file, laser_energy, row))
        except Exception as e:
            print(f"  Warning: Error processing row {idx}: {e}")
            continue

    # Prefer Cu and Al (two-panel layout like the 2D figure); ignore others for 3D.
    preferred_materials = ['Al', 'Cu']
    materials = [m for m in preferred_materials if m in traces_by_material]
    if not materials:
        print("ERROR: No valid traces found")
        return

    all_energies = [energy for material in materials for _, energy, _ in traces_by_material[material]]
    if len(all_energies) == 0:
        print("ERROR: No valid traces found")
        return
    energy_min = min(all_energies)
    energy_max = max(all_energies)
    energy_bins = np.arange(int(energy_min // 100) * 100, int(energy_max // 100 + 1) * 100 + 1, 100)
    cmap = plt.get_cmap('viridis')

    n_materials = len(materials)
    fig = plt.figure(figsize=(10 * n_materials, 8))
    fig.subplots_adjust(right=0.86, wspace=0.15)

    axes = []
    for i in range(n_materials):
        ax = fig.add_subplot(1, n_materials, i + 1, projection='3d')
        axes.append(ax)

    all_velocities = []

    for ax_idx, material in enumerate(materials):
        ax = axes[ax_idx]
        traces = traces_by_material[material]
        print(f"\nProcessing {material}: {len(traces)} traces")

        plotted_count = 0
        shifted_count = 0

        for vel_file, laser_energy, _row in traces:
            time, velocity = load_velocity_trace(vel_file, spade_params)
            if time is None or len(time) == 0:
                continue

            time_plot = time.copy()
            velocity_plot = velocity.copy()

            # Apply the same peak-shift as the 2D plot so traces are
            # visually aligned.  Peaks occurring before 15 ns are pushed
            # to 19 ns (same logic as generate_velocity_traces_by_laser_energy).
            peak_idx = np.argmax(velocity_plot)
            peak_time = time_plot[peak_idx]
            if peak_time < 15.0:
                time_shift = 19.0 - peak_time
                time_plot = time_plot + time_shift
                shifted_count += 1

            # Clip to 0–80 ns at the DATA level (bulletproof against 3D bleed).
            mask = (time_plot >= 0.0) & (time_plot <= 80.0)
            if not np.any(mask):
                continue
            time_plot = time_plot[mask]
            velocity_plot = velocity_plot[mask]

            # If the trace starts after t=0 (shift pushed start past 0),
            # anchor it with a (0, v=0) point so every trace meets the left wall.
            if time_plot[0] > 0.0:
                time_plot      = np.concatenate([[0.0], time_plot])
                velocity_plot  = np.concatenate([[0.0], velocity_plot])

            energy_bin_idx = np.digitize(laser_energy, energy_bins) - 1
            energy_bin_idx = max(0, min(energy_bin_idx, len(energy_bins) - 2))
            binned_energy = float(energy_bins[energy_bin_idx])

            norm_energy = (binned_energy - energy_min) / (energy_max - energy_min) if energy_max > energy_min else 0.5
            color = cmap(norm_energy)

            # y = laser energy (constant per trace), z = velocity (m/s)
            y = np.full_like(time_plot, binned_energy, dtype=float)
            vel = velocity_plot.astype(float)
            all_velocities.append(vel)
            ax.plot(time_plot, y, vel, color=color, alpha=0.82, linewidth=1.2)
            plotted_count += 1

        if shifted_count > 0:
            print(f"  Shifted {shifted_count} traces (peaks before 15 ns -> 19 ns)")

        align_threshold = 30.0
        if spade_params:
            align_threshold = spade_params.get('align_velocity_threshold_ms', 30.0)

        ax.set_xlabel('Time (ns)', fontsize=9, labelpad=14)
        ax.set_ylabel('Laser Energy (mJ) (binned to 100 mJ)', fontsize=10, labelpad=18)
        ax.set_zlabel('Velocity (m/s)', fontsize=10, labelpad=16)
        ax.yaxis.labelpad = 22
        ax.zaxis.labelpad = 22
        ax.set_title(f'{material} (n={plotted_count})', fontsize=14, fontweight='bold', pad=16)

        ax.set_xlim3d(0, 80)
        ax.set_ylim(int(energy_min // 100) * 100, int(energy_max // 100 + 1) * 100)

        ax.view_init(elev=35, azim=-60)

        # Modernize 3D look: white panes, no grids, lighter ticks.
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(False)
        ax.tick_params(colors='black', labelsize=8)

        print(f"  Plotted {plotted_count} traces for {material}")

    # Set per-material z-limits (Al peaks up to ~800 m/s; Cu up to ~500 m/s)
    velocity_zlim = {'Al': 800, 'Cu': 500}
    for ax, mat in zip(axes, materials):
        ax.set_zlim(0, velocity_zlim.get(mat, 800))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=energy_min, vmax=energy_max))
    sm.set_array([])
    # Increase pad so the colorbar doesn't overlap the Cu z-axis label
    cbar = fig.colorbar(sm, ax=axes, pad=0.12, location='right', shrink=0.6)
    cbar.set_label('Laser Energy (mJ), binned to 100 mJ', fontsize=12, rotation=270, labelpad=20)

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 3D plot saved to: {output_path}")

    # Save focused 0-40 ns version
    for ax in axes:
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 200)
    base, ext = os.path.splitext(output_filename)
    output_path_40ns = os.path.join(output_dir, f"{base}_0-40ns{ext}")
    plt.savefig(output_path_40ns, dpi=300, bbox_inches='tight')
    print(f"✅ 3D plot (0-40 ns) saved to: {output_path_40ns}")

    plt.close()


def generate_velocity_traces_by_shock_stress_3d(summary_csv_path, output_dir, output_filename='velocity_traces_by_shock_stress_3d.png', spade_params=None):
    """
    3D waterfall plot of velocity traces where:
      x = time (ns)
      y = Peak Shock Stress (GPa), binned to 0.5 GPa (depth axis)
      z = velocity (m/s)
    Lines are colored by shock stress using the viridis colormap.
    Two subplots side-by-side for Al and Cu.
    """
    print("=" * 60)
    print("Generating 3D velocity traces by material (y=shock stress)")
    print("=" * 60)

    if not os.path.exists(summary_csv_path):
        print(f"ERROR: Summary file not found: {summary_csv_path}")
        return

    try:
        df = pd.read_csv(summary_csv_path)
        print(f"Loaded summary with {len(df)} rows")
    except Exception as e:
        print(f"ERROR: Could not read summary file: {e}")
        return

    # --- Find required columns ---
    file_col = find_column(df, ['PDV File', 'File', 'Filename', 'file_name', 'pdv_file'])

    material_col = None
    for col in df.columns:
        if str(col).lower().strip() == 'sample material':
            material_col = col; break
    if not material_col:
        for col in df.columns:
            if str(col).lower().strip() == 'material':
                material_col = col; break
    if not material_col:
        for col in df.columns:
            cl = str(col).lower().strip()
            if 'sample' in cl and 'material' in cl and 'flyer' not in cl:
                material_col = col; break
    if not material_col:
        material_col = find_column(df, ['Material', 'Sample Material', 'Target material'])

    shock_stress_col = find_column(df, [
        'Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa',
        'Shock Stress (GPa)', 'Peak Shock Stress'
    ])

    if not file_col:
        print("ERROR: Could not find file name column"); return
    if not material_col:
        print(f"ERROR: Could not find material column. Available: {', '.join(df.columns)}"); return
    if not shock_stress_col:
        print(f"ERROR: Could not find Peak Shock Stress column. Available: {', '.join(df.columns)}"); return

    print(f"Using columns: File={file_col}, Material={material_col}, Shock Stress={shock_stress_col}")

    # --- Locate velocity files ---
    base_dir = output_dir
    output_basename = os.path.basename(os.path.normpath(output_dir))
    if output_basename == 'SPADE_analysis':
        base_dir = os.path.dirname(os.path.dirname(output_dir))
    elif output_basename == 'SPADE_manual':
        base_dir = os.path.dirname(output_dir)

    all_vel_files = glob.glob(os.path.join(base_dir, '**/*--vel-smooth-with-uncert.csv'), recursive=True)
    print(f"Found {len(all_vel_files)} velocity files in {base_dir}")
    if len(all_vel_files) == 0:
        print("ERROR: No velocity files found"); return

    vel_file_map = {}
    for vf in all_vel_files:
        bn = os.path.basename(vf).replace('--vel-smooth-with-uncert.csv', '')
        for suf in ['--vel-smooth', '--velocity', '--vel']:
            if bn.endswith(suf):
                bn = bn[:-len(suf)]; break
        vel_file_map[bn] = vf

    # --- Organise traces by material, keyed on shock stress ---
    traces_by_material = defaultdict(list)
    for idx, row in df.iterrows():
        try:
            file_name = str(row[file_col]).strip()
            if pd.isna(file_name) or file_name == '':
                continue

            bn = os.path.splitext(file_name)[0]
            for suf in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                if bn.endswith(suf):
                    bn = bn[:-len(suf)]; break

            vel_file = vel_file_map.get(bn)
            if vel_file is None:
                m = re.search(r'(C\d+--\d{8}--\d{5})', bn)
                if m:
                    for k, p in vel_file_map.items():
                        if m.group(1) in k:
                            vel_file = p; break
            if not vel_file or not os.path.exists(vel_file):
                continue

            material = str(row[material_col]).strip() if not pd.isna(row[material_col]) else 'Unknown'
            ml = material.lower()
            if 'cu' in ml or 'copper' in ml:
                material = 'Cu'
            elif 'al' in ml or 'aluminum' in ml or 'aluminium' in ml:
                material = 'Al'
            else:
                continue  # only Al and Cu for this plot

            stress_val = row[shock_stress_col]
            if pd.isna(stress_val):
                continue
            try:
                stress_val = float(stress_val)
            except (ValueError, TypeError):
                continue
            if stress_val <= 0:
                continue

            traces_by_material[material].append((vel_file, stress_val, row))
        except Exception as e:
            print(f"  Warning: Error processing row {idx}: {e}")
            continue

    materials = [m for m in ['Al', 'Cu'] if m in traces_by_material]
    if not materials:
        print("ERROR: No valid traces found"); return

    # --- Colormap (no binning; use raw peak shock stress) ---
    all_stresses = [s for mat in materials for _, s, _ in traces_by_material[mat]]
    stress_min, stress_max = min(all_stresses), max(all_stresses)
    cmap = plt.get_cmap('viridis')

    # --- Figure (tall/wide + margins so large 3D axis labels are not clipped) ---
    n_materials = len(materials)
    fig = plt.figure(figsize=(11 * n_materials, 10))
    fig.subplots_adjust(left=0.06, bottom=0.14, right=0.82, top=0.90, wspace=0.28)
    axes = [fig.add_subplot(1, n_materials, i + 1, projection='3d') for i in range(n_materials)]

    velocity_zlim = {'Al': 800, 'Cu': 500}
    all_velocities = []

    for ax_idx, material in enumerate(materials):
        ax = axes[ax_idx]
        traces = traces_by_material[material]
        print(f"\nProcessing {material}: {len(traces)} traces")

        plotted_count = 0
        shifted_count = 0

        for vel_file, stress_val, _row in traces:
            time, velocity = load_velocity_trace(vel_file, spade_params)
            if time is None or len(time) == 0:
                continue

            time_plot = time.copy()
            velocity_plot = velocity.copy()

            # Same peak-shift as 2D / laser-energy-3D plot
            peak_idx = np.argmax(velocity_plot)
            peak_time = time_plot[peak_idx]
            if peak_time < 15.0:
                time_plot = time_plot + (19.0 - peak_time)
                shifted_count += 1

            # Clip to 0–80 ns at data level
            mask = (time_plot >= 0.0) & (time_plot <= 80.0)
            if not np.any(mask):
                continue
            time_plot     = time_plot[mask]
            velocity_plot = velocity_plot[mask]

            # Anchor to t=0
            if time_plot[0] > 0.0:
                time_plot     = np.concatenate([[0.0], time_plot])
                velocity_plot = np.concatenate([[0.0], velocity_plot])

            # Use raw peak shock stress value for depth and color
            norm_stress = (stress_val - stress_min) / (stress_max - stress_min) if stress_max > stress_min else 0.5
            color = cmap(norm_stress)

            y   = np.full_like(time_plot, stress_val, dtype=float)
            vel = velocity_plot.astype(float)
            all_velocities.append(vel)
            ax.plot(time_plot, y, vel, color=color, alpha=0.82, linewidth=1.2)
            plotted_count += 1

        if shifted_count > 0:
            print(f"  Shifted {shifted_count} traces (peaks before 15 ns -> 19 ns)")

        align_threshold = 30.0
        if spade_params:
            align_threshold = spade_params.get('align_velocity_threshold_ms', 30.0)

        ax.set_xlabel('Time (ns)', fontsize=18, labelpad=36)
        ax.set_ylabel('Peak Shock Stress (GPa)', fontsize=20, labelpad=48)
        ax.set_zlabel('Velocity (m/s)', fontsize=20, labelpad=44)
        ax.yaxis.labelpad = 56
        ax.zaxis.labelpad = 56
        ax.set_title(f'{material} (n={plotted_count})', fontsize=28, fontweight='bold', pad=40)

        ax.set_xlim3d(0, 80)
        ax.set_ylim(stress_min, stress_max)
        ax.set_zlim(0, velocity_zlim.get(material, 800))
        ax.view_init(elev=35, azim=-60)

        # Same modern 3D styling as laser-energy plot
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(False)
        ax.tick_params(colors='black', labelsize=16, pad=10)

        print(f"  Plotted {plotted_count} traces for {material}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=stress_min, vmax=stress_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.18, location='right', shrink=0.58)
    cbar.set_label('Peak Shock Stress (GPa)', fontsize=24, rotation=270, labelpad=40)
    cbar.ax.tick_params(labelsize=16)

    _save_shock_3d_kwargs = dict(dpi=300, bbox_inches='tight', pad_inches=1.0)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, **_save_shock_3d_kwargs)
    print(f"\n✅ 3D shock-stress plot saved to: {output_path}")

    # 0-40 ns focused version
    for ax in axes:
        ax.set_xlim3d(0, 40)
    base_fn, ext = os.path.splitext(output_filename)
    out_40 = os.path.join(output_dir, f"{base_fn}_0-40ns{ext}")
    plt.savefig(out_40, **_save_shock_3d_kwargs)
    print(f"✅ 3D shock-stress plot (0-40 ns) saved to: {out_40}")

    plt.close()


def load_master_config(config_path=None):
    """Load helix_master_config (YAML or JSON) and extract paths.

    Probe order when config_path is not specified:
        helix_master_config.yml → helix_master_config.yaml → helix_master_config.json
    """
    if config_path is None:
        for name in ("helix_master_config.yml", "helix_master_config.yaml", "helix_master_config.json"):
            candidate = os.path.join(REPO_ROOT, name)
            if os.path.exists(candidate):
                config_path = candidate
                break
        if config_path is None:
            config_path = os.path.join(REPO_ROOT, "helix_master_config.json")  # will raise below
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    ok, master_config, message = load_config_from_file(config_path)
    if not ok:
        raise RuntimeError(f"Failed to load config: {message}")
    
    # Extract cli_settings
    cli_settings = master_config.get("cli_settings", {})
    output_dir = cli_settings.get("output_dir")
    
    if not output_dir:
        raise ValueError("output_dir not found in helix_master_config.json")
    
    return output_dir, master_config


def find_summary_csv_under_output_dir(output_root_dir):
    """
    Find a SPADE summary CSV under an Output/ directory.

    Supports both automated and manual SPADE layouts, e.g.:
      Output/SPADE_analysis/enhanced_spall_summary.csv
      Output/SPADE_manual/SPADE_analysis/enhanced_spall_summary.csv

    Returns:
        str|None: path to summary CSV (prefer enhanced_spall_summary), else None.
    """
    candidates = []
    for fname in ("enhanced_spall_summary.csv", "velocity_shots_summary.csv"):
        pattern = os.path.join(output_root_dir, "**", "SPADE_analysis", fname)
        candidates.extend(glob.glob(pattern, recursive=True))

    if not candidates:
        return None

    # Prefer enhanced_spall_summary.csv; if multiple, choose most recently modified.
    enhanced = [p for p in candidates if p.endswith(os.sep + "enhanced_spall_summary.csv")]
    preferred = enhanced if enhanced else candidates
    preferred.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return preferred[0]


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate velocity traces plot by material colored by laser energy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file to find summary CSV automatically
  python plot_velocity_traces_by_laser_energy.py
  
  # Specify summary CSV explicitly
  python plot_velocity_traces_by_laser_energy.py /path/to/enhanced_spall_summary.csv
  
  # Use custom config file
  python plot_velocity_traces_by_laser_energy.py --config /path/to/helix_master_config.json
        """
    )
    parser.add_argument('summary_csv', nargs='?', default=None,
                       help='Path to summary CSV file (optional if using config file)')
    parser.add_argument('--output-dir', '-o', default=None,
                       help='Output directory for plot (default: from config or CSV directory)')
    parser.add_argument('--output-filename', '-f', default='velocity_traces_by_laser_energy.png',
                       help='Output filename (default: velocity_traces_by_laser_energy.png)')
    parser.add_argument('--config', '-c', default=None,
                       help='Path to helix_master_config.json (default: helix_master_config.json in script directory)')
    parser.add_argument('--plot-3d', action='store_true',
                       help='Only the laser-energy 3D plot (no 2D, no shock-stress 3D)')
    
    args = parser.parse_args()
    
    # Try to load config from helix_master_config.json
    output_dir_from_config = None
    summary_csv_from_config = None
    spade_params = None
    
    try:
        output_dir_from_config, master_config = load_master_config(args.config)
        # Extract SPADE parameters for alignment settings
        spade_config = master_config.get("spade_config", {})
        # Transform SPADE params (simplified - just get what we need)
        spade_params = {
            'use_hel_t0_alignment_for_plots': spade_config.get('use_hel_t0_alignment_for_plots', True),
            'minimum_HEL_velocity_expected': spade_config.get('minimum_HEL_velocity_expected', 10.0),
            'align_velocity_threshold_ms': spade_config.get('align_velocity_threshold_ms', 30.0),
            'uncertainty_threshold_ms': spade_config.get('uncertainty_threshold_ms', 50.0)
        }

        # Find summary CSV under output_dir (supports SPADE_manual layouts too)
        summary_csv_from_config = find_summary_csv_under_output_dir(output_dir_from_config)

        print(f"Loaded config from helix_master_config.json")
        print(f"  Output directory: {output_dir_from_config}")
        if summary_csv_from_config:
            print(f"  Found summary CSV: {summary_csv_from_config}")
        else:
            print(f"  Warning: No summary CSV found under output directory")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Warning: Could not load helix_master_config.json: {e}")
        print("  Will use command-line arguments only")
    
    # Determine summary CSV path
    if args.summary_csv:
        summary_csv = args.summary_csv
    elif summary_csv_from_config:
        summary_csv = summary_csv_from_config
        print(f"Using summary CSV from config: {summary_csv}")
    else:
        print("ERROR: No summary CSV specified and none found in config")
        print("Please provide summary_csv as argument or ensure config file points to valid location")
        sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(summary_csv):
        print(f"ERROR: Summary CSV file not found: {summary_csv}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif output_dir_from_config:
        # Use SPADE_analysis directory from config
        output_dir = os.path.join(output_dir_from_config, "SPADE_analysis")
    else:
        # Fallback: use directory of summary CSV
        output_dir = os.path.dirname(os.path.abspath(summary_csv))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HELIX Velocity Traces Plot Generator")
    print("=" * 60)
    print(f"Summary CSV: {summary_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Output filename: {args.output_filename}")
    print("=" * 60)
    
    # Generate plots
    base, ext = os.path.splitext(args.output_filename)

    if args.plot_3d:
        # Explicit 3D-only mode
        out_name_3d = args.output_filename if base.endswith('_3d') else f"{base}_3d{ext}"
        generate_velocity_traces_by_laser_energy_3d(
            summary_csv,
            output_dir,
            out_name_3d,
            spade_params
        )
    else:
        # Default: generate 2D, 3D laser-energy, and 3D shock-stress plots.
        # Each step is isolated so a failure in one (e.g. slow or buggy 3D save)
        # still allows the next outputs to be produced.
        try:
            generate_velocity_traces_by_laser_energy(
                summary_csv,
                output_dir,
                args.output_filename,
                spade_params
            )
        except Exception as e:
            print(f"\nERROR: 2D velocity traces plot failed: {e}", flush=True)
            traceback.print_exc()

        out_name_3d = f"{base}_3d{ext}"
        try:
            generate_velocity_traces_by_laser_energy_3d(
                summary_csv,
                output_dir,
                out_name_3d,
                spade_params
            )
        except Exception as e:
            print(f"\nERROR: 3D laser-energy plot failed: {e}", flush=True)
            traceback.print_exc()

        print("\n" + "=" * 60, flush=True)
        print("Next: 3D velocity traces (y = Peak Shock Stress)", flush=True)
        print("=" * 60, flush=True)
        try:
            generate_velocity_traces_by_shock_stress_3d(
                summary_csv,
                output_dir,
                f"{base}_shock_stress_3d{ext}",
                spade_params
            )
        except Exception as e:
            print(f"\nERROR: 3D shock-stress plot failed: {e}", flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
