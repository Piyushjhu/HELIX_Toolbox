#!/usr/bin/env python3
"""
Standalone script to generate HELIX paper plots from summary CSV files
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import json
from collections import defaultdict
from helix_paper_plots import (
    generate_spall_vs_strain_rate_plot,
    generate_spall_vs_strain_rate_by_material_subplots,
    generate_all_plots_from_summary_files,
    find_column_name,
)

# Add helix_analysis_toolbox to path for config loading.
# This script lives in <repo>/supplementary/paper_plots/, so the repo root is two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from helix_analysis_toolbox import load_config_from_file
except ImportError:
    # Fallback: minimal loader that supports both YAML and JSON.
    def load_config_from_file(config_path):
        ext = os.path.splitext(config_path)[1].lower()
        try:
            if ext in (".yml", ".yaml"):
                import yaml
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(config_path, "r") as f:
                    data = json.load(f)
            return True, data, "OK"
        except Exception as exc:
            return False, {}, str(exc)


def generate_failure_detection_table(enhanced_spall_summary_path, output_dir, progress_callback=None):
    """
    Generate a table of failure detection binned by reason from enhanced_spall_summary.
    Outputs failure_detection_summary.csv with columns: Reason, Count, Percentage.
    """
    def log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(f"  {message}")

    log("Generating failure detection summary table...")

    if not enhanced_spall_summary_path or not os.path.exists(enhanced_spall_summary_path):
        log("⚠ Enhanced spall summary not found - skipping failure detection table")
        return

    try:
        df = pd.read_csv(enhanced_spall_summary_path)
    except Exception as e:
        log(f"ERROR: Could not read enhanced spall summary: {e}")
        return

    # Find status column (Processing_Status or DNS_Classification)
    status_col = None
    for col in ['Processing_Status', 'Processing Status', 'DNS_Classification', 'DNS Classification']:
        if col in df.columns:
            status_col = col
            break

    if status_col is None:
        log("⚠ No Processing_Status or DNS_Classification column found - skipping failure detection table")
        return

    try:
        # Count by reason
        counts = df[status_col].value_counts(dropna=False)
        total = len(df)
        if total == 0:
            log("⚠ Enhanced spall summary is empty - skipping failure detection table")
            return

        # Build table: Reason, Count, Percentage
        table_data = []
        for reason, count in counts.items():
            reason_str = str(reason).strip() if pd.notna(reason) else "(blank)"
            pct = 100.0 * count / total
            table_data.append({'Reason': reason_str, 'Count': count, 'Percentage (%)': round(pct, 2)})

        # Sort by count descending
        table_df = pd.DataFrame(table_data)
        table_df = table_df.sort_values('Count', ascending=False).reset_index(drop=True)

        # Save to CSV
        out_path = os.path.join(output_dir, 'failure_detection_summary.csv')
        table_df.to_csv(out_path, index=False)
        log(f"✅ Saved failure detection table: {out_path}")

        # Also print summary to console
        log(f"   Total traces: {total}")
        for _, row in table_df.iterrows():
            log(f"   - {row['Reason']}: {row['Count']} ({row['Percentage (%)']}%)")

    except Exception as e:
        log(f"ERROR generating failure detection table: {e}")
        import traceback
        traceback.print_exc()


def _normalize_material_al_cu(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if 'cu' in s or 'copper' in s:
        return 'Cu'
    if 'al' in s or 'alumin' in s:
        return 'Al'
    return None


def generate_spall_svr_surface_3d(enhanced_spall_summary_path, output_dir, progress_callback=None):
    """
    3D plot: x = strain rate, y = peak shock stress, z = spall strength.
    Scatter Al and Cu valid spall points; fit a single RBF-SVR surface z = f(ln strain rate, stress)
    on pooled data (features scaled), then draw the surface on a regular grid.
    Output: spall_svr_surface_3d.png in output_dir.
    """
    def log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(f"  {message}")

    log("Generating 3D SVR surface (spall vs strain rate vs shock stress, Al & Cu)...")

    if not enhanced_spall_summary_path or not os.path.exists(enhanced_spall_summary_path):
        log("⚠ Enhanced spall summary not found - skipping SVR surface plot")
        return

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
    except ImportError as e:
        log(f"⚠ scikit-learn required for SVR surface: {e}")
        return

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    try:
        df = pd.read_csv(enhanced_spall_summary_path)
    except Exception as e:
        log(f"ERROR: Could not read summary CSV: {e}")
        return

    spall_col = find_column_name(
        df,
        ['Spall Strength (GPa)', 'Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall Strength'],
        progress_callback,
    )
    strain_col = find_column_name(
        df,
        ['Strain Rate (s^-1)', 'Spall_StrainRate_s^-1', 'Strain_Rate_s^-1', 'Strain Rate'],
        progress_callback,
    )
    stress_col = find_column_name(
        df,
        ['Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa', 'Peak_Shock_Stress_GPa_Final',
         'Shock Stress (GPa)', 'Shock_Stress_GPa', 'Peak Shock Stress'],
        progress_callback,
    )
    mat_col = find_column_name(
        df,
        ['Material', 'material', 'Sample material', 'Sample_Material'],
        progress_callback,
    )

    if not all([spall_col, strain_col, stress_col]):
        log("⚠ Missing spall / strain rate / shock stress columns - skipping SVR surface plot")
        return
    if mat_col is None:
        log("⚠ No material column - skipping SVR surface plot")
        return

    work = df[[spall_col, strain_col, stress_col, mat_col]].copy()
    for c in (spall_col, strain_col, stress_col):
        work[c] = pd.to_numeric(work[c], errors='coerce')
    work = work.dropna(subset=[spall_col, strain_col, stress_col])
    work = work[
        (work[spall_col] > 0) & (work[strain_col] > 0) & (work[stress_col] > 0)
    ].copy()
    work['Mat'] = work[mat_col].apply(_normalize_material_al_cu)
    work = work[work['Mat'].isin(['Al', 'Cu'])]

    if len(work) < 10:
        log(f"⚠ Too few Al/Cu points with valid spall+strain+stress (n={len(work)}); need ≥10 - skipping")
        return

    # Features: log10(strain rate) + shock stress (stabilizes SVR across decades of strain rate)
    eps_sr = 1e-30
    X = np.column_stack([
        np.log10(work[strain_col].astype(float).values + eps_sr),
        work[stress_col].astype(float).values,
    ])
    y = work[spall_col].astype(float).values

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100.0, epsilon=0.05, gamma='scale')),
    ])
    model.fit(X, y)

    sr_min = float(work[strain_col].min())
    sr_max = float(work[strain_col].max())
    st_min = float(work[stress_col].min())
    st_max = float(work[stress_col].max())
    n_grid = 55
    sr_grid = np.linspace(sr_min, sr_max, n_grid)
    st_grid = np.linspace(st_min, st_max, n_grid)
    SR, ST = np.meshgrid(sr_grid, st_grid)
    X_grid = np.column_stack([
        np.log10(SR.ravel() + eps_sr),
        ST.ravel(),
    ])
    Z_pred = model.predict(X_grid).reshape(SR.shape)
    Z_pred = np.clip(Z_pred, 0.0, None)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        SR, ST, Z_pred,
        cmap='viridis',
        alpha=0.38,
        linewidth=0,
        antialiased=True,
        edgecolor='none',
    )

    colors = {'Al': '#d62728', 'Cu': '#1f77b4'}
    for mat in ['Al', 'Cu']:
        part = work[work['Mat'] == mat]
        if len(part) == 0:
            continue
        ax.scatter(
            part[strain_col].values,
            part[stress_col].values,
            part[spall_col].values,
            c=colors[mat],
            label=f'{mat} (n={len(part)})',
            s=72,
            depthshade=True,
            edgecolors='black',
            linewidths=0.35,
            alpha=0.95,
        )

    ax.set_xlabel('Strain rate (s$^{-1}$)', fontsize=12, labelpad=12)
    ax.set_ylabel('Peak shock stress (GPa)', fontsize=12, labelpad=12)
    ax.set_zlabel('Spall strength (GPa)', fontsize=12, labelpad=12)
    ax.set_title(
        'Spall strength: SVR surface vs strain rate and shock stress\n'
        r'(RBF SVR on $\log_{10}$(strain rate), stress); Al & Cu points',
        fontsize=12,
        pad=14,
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=22, azim=-58)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('white')
        axis.pane.set_alpha(0.0)
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label('SVR predicted spall (GPa)', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'spall_svr_surface_3d.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.35)
    plt.close()
    log(f"✅ Saved SVR surface plot: {out_path}")


def generate_velocity_traces_by_waveplate_angle(summary_csv_path, output_dir, progress_callback=None):
    """
    Generate velocity trace plots separated by material (Cu/Al) with subplots for each waveplate angle.
    
    Args:
        summary_csv_path: Path to summary CSV file (enhanced_spall_summary.csv or velocity_shots_summary.csv)
        output_dir: Directory to save plots
        progress_callback: Optional callback function for progress messages
    """
    def log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(f"  {message}")
    
    log("Generating velocity traces by material and waveplate angle...")
    
    # Read summary CSV
    if not os.path.exists(summary_csv_path):
        log(f"ERROR: Summary file not found: {summary_csv_path}")
        return
    
    try:
        df = pd.read_csv(summary_csv_path)
        log(f"Loaded summary with {len(df)} rows")
    except Exception as e:
        log(f"ERROR: Could not read summary file: {e}")
        return
    
    # Find columns for file name, material, and waveplate angle
    file_col = None
    material_col = None
    waveplate_col = None
    
    # Try to find file name column
    for col in df.columns:
        col_lower = col.lower()
        if 'pdv' in col_lower and 'file' in col_lower:
            file_col = col
            break
        elif 'file' in col_lower and 'name' in col_lower:
            file_col = col
            break
    
    # Try to find material column
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['material', 'sample material', 'sample_material']:
            material_col = col
            break
    
    # Try to find waveplate angle column
    for col in df.columns:
        col_lower = col.lower()
        if 'waveplate' in col_lower and 'angle' in col_lower:
            waveplate_col = col
            break
        elif 'angle' in col_lower and 'degree' in col_lower:
            waveplate_col = col
            break
    
    # Try to find flyer thickness column
    flyer_thickness_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'flyer' in col_lower and 'thickness' in col_lower:
            flyer_thickness_col = col
            break
    
    if not file_col:
        log("WARNING: Could not find file name column in summary CSV")
        return
    
    if not material_col:
        log("WARNING: Could not find material column in summary CSV")
        return
    
    if not waveplate_col:
        log("WARNING: Could not find waveplate angle column in summary CSV")
        return
    
    if not flyer_thickness_col:
        log("WARNING: Could not find flyer thickness column in summary CSV")
        return
    
    log(f"Using columns: File={file_col}, Material={material_col}, Waveplate={waveplate_col}, Thickness={flyer_thickness_col}")
    
    # Find velocity files directory (parent of SPADE_analysis or Output directory)
    base_dir = os.path.dirname(output_dir)
    if 'SPADE_analysis' in output_dir:
        base_dir = os.path.dirname(output_dir)  # Go up one level from SPADE_analysis
    elif 'Output' in output_dir:
        base_dir = output_dir  # Already in Output directory
    
    # Search for velocity files
    vel_files_pattern = os.path.join(base_dir, '**/*--vel-smooth-with-uncert.csv')
    all_vel_files = glob.glob(vel_files_pattern, recursive=True)
    log(f"Found {len(all_vel_files)} velocity files in {base_dir}")
    
    if len(all_vel_files) == 0:
        log("ERROR: No velocity files found")
        return
    
    # Create a mapping from base filename to velocity file path
    vel_file_map = {}
    for vel_file in all_vel_files:
        filename = os.path.basename(vel_file)
        # Extract base name (remove --vel-smooth-with-uncert.csv)
        base_name = filename.replace('--vel-smooth-with-uncert.csv', '')
        # Also try other suffixes
        for suffix in ['--vel-smooth', '--velocity', '--vel']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        vel_file_map[base_name] = vel_file
    
    # Organize data by material, waveplate angle, and flyer thickness
    # material -> angle -> thickness -> list of (vel_file, row_data)
    traces_by_material_angle_thickness = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for idx, row in df.iterrows():
        try:
            # Get file name from summary
            file_name = str(row[file_col]).strip()
            if pd.isna(file_name) or file_name == '':
                continue
            
            # Extract base name from file name
            base_name = os.path.splitext(file_name)[0]
            # Remove common suffixes
            for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Find matching velocity file
            vel_file = None
            # Try exact match first
            if base_name in vel_file_map:
                vel_file = vel_file_map[base_name]
            else:
                # Try partial match (PDV filename pattern)
                pdv_pattern = re.search(r'(C\d+--\d{8}--\d{5})', base_name)
                if pdv_pattern:
                    pdv_id = pdv_pattern.group(1)
                    for key, path in vel_file_map.items():
                        if pdv_id in key:
                            vel_file = path
                            break
            
            if not vel_file or not os.path.exists(vel_file):
                continue
            
            # Get material
            material = str(row[material_col]).strip() if not pd.isna(row[material_col]) else 'Unknown'
            # Normalize material names
            material_lower = material.lower()
            if 'cu' in material_lower or 'copper' in material_lower:
                material = 'Cu'
            elif 'al' in material_lower or 'aluminum' in material_lower or 'aluminium' in material_lower:
                material = 'Al'
            else:
                continue  # Skip unknown materials
            
            # Get waveplate angle
            waveplate_angle = row[waveplate_col]
            if pd.isna(waveplate_angle):
                continue
            try:
                waveplate_angle = float(waveplate_angle)
            except (ValueError, TypeError):
                continue
            
            # Get flyer thickness
            flyer_thickness = row[flyer_thickness_col]
            if pd.isna(flyer_thickness):
                continue
            try:
                flyer_thickness_val = float(flyer_thickness)
                # Categorize into 50 um and 100 um (with tolerance)
                if abs(flyer_thickness_val - 50) < abs(flyer_thickness_val - 100):
                    thickness_category = '50 um'
                else:
                    thickness_category = '100 um'
            except (ValueError, TypeError):
                continue
            
            traces_by_material_angle_thickness[material][waveplate_angle][thickness_category].append((vel_file, row))
            
        except Exception as e:
            log(f"  Warning: Error processing row {idx}: {e}")
            continue
    
    # Color mapping for flyer thickness
    thickness_colors = {
        '50 um': 'blue',
        '100 um': 'red'
    }
    
    # Generate plots for each material
    for material in ['Cu', 'Al']:
        if material not in traces_by_material_angle_thickness:
            log(f"No data found for {material}, skipping...")
            continue
        
        angles = sorted(traces_by_material_angle_thickness[material].keys())
        if len(angles) == 0:
            log(f"No waveplate angles found for {material}, skipping...")
            continue
        
        log(f"Generating plot for {material} with {len(angles)} waveplate angles: {angles}")
        
        # Create subplots - one per waveplate angle
        n_angles = len(angles)
        n_cols = min(3, n_angles)  # Max 3 columns
        n_rows = (n_angles + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_angles == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Alignment threshold (30 m/s)
        align_threshold = 30.0
        tolerance = 0.01
        
        total_traces = 0
        
        for idx, angle in enumerate(angles):
            ax = axes[idx]
            
            traces_plotted = 0
            thickness_counts = {'50 um': 0, '100 um': 0}
            
            # Plot traces grouped by thickness
            for thickness_category in ['50 um', '100 um']:
                if thickness_category not in traces_by_material_angle_thickness[material][angle]:
                    continue
                
                traces = traces_by_material_angle_thickness[material][angle][thickness_category]
                color = thickness_colors[thickness_category]
                
                for vel_file, _ in traces:
                    try:
                        # Read velocity file
                        vel_df = pd.read_csv(vel_file)
                        if vel_df.shape[1] < 2:
                            continue
                        
                        # Get time and velocity data
                        time_data = vel_df.iloc[:, 0].values
                        velocity_data = vel_df.iloc[:, 1].values
                        uncertainty_data = vel_df.iloc[:, 2].values if vel_df.shape[1] >= 3 else None
                        
                        # Convert time to ns if needed
                        if np.nanmax(time_data) < 1e-3:
                            time_data = time_data * 1e9  # Convert from seconds to ns
                        elif np.nanmax(time_data) < 1.0:
                            time_data = time_data * 1e3  # Convert from microseconds to ns
                        
                        # Noise fraction filtering
                        noise_file = vel_file.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
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
                        
                        # Build valid mask with noise suppression
                        valid_mask = ~np.isnan(velocity_data)
                        if high_noise_mask is not None:
                            valid_mask &= (~high_noise_mask)
                        
                        # Uncertainty threshold filtering (50 m/s default)
                        uncertainty_threshold = 50.0
                        if uncertainty_data is not None:
                            valid_mask &= (uncertainty_data <= uncertainty_threshold)
                        
                        time_clean = time_data[valid_mask]
                        velocity_clean = velocity_data[valid_mask]
                        
                        if len(time_clean) == 0 or len(velocity_clean) == 0:
                            continue
                        
                        # Apply threshold alignment
                        t0_idx = None
                        for j, v in enumerate(velocity_clean):
                            if not np.isnan(v) and v >= (align_threshold - tolerance):
                                t0_idx = j
                                break
                        
                        if t0_idx is not None:
                            t0 = time_clean[t0_idx]
                            time_clean = time_clean - t0
                        
                        # Filter to x-axis range (-20 to 100 ns)
                        time_mask = (time_clean >= -20) & (time_clean <= 100)
                        time_plot = time_clean[time_mask]
                        velocity_plot = velocity_clean[time_mask]
                        
                        if len(time_plot) == 0 or len(velocity_plot) == 0:
                            continue
                        
                        # Plot trace with thickness-specific color
                        ax.plot(time_plot, velocity_plot, alpha=0.6, linewidth=0.8, color=color, label=thickness_category if thickness_counts[thickness_category] == 0 else '')
                        traces_plotted += 1
                        thickness_counts[thickness_category] += 1
                        total_traces += 1
                    
                    except Exception as e:
                        log(f"  Warning: Could not plot {os.path.basename(vel_file)}: {e}")
                        continue
            
            # Format subplot
            ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
            title_parts = [f'{material} - {angle}°']
            if thickness_counts['50 um'] > 0:
                title_parts.append(f'50um: n={thickness_counts["50 um"]}')
            if thickness_counts['100 um'] > 0:
                title_parts.append(f'100um: n={thickness_counts["100 um"]}')
            ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
            ax.grid(False)
            ax.set_xlim(-20, 100)
            
            # Add legend for thickness colors (outside plot area)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.9)
        
        # Hide unused subplots
        for idx in range(n_angles, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'velocity_traces_{material}_by_waveplate_angle.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        log(f"✅ Generated {plot_filename} ({total_traces} traces total)")
    
    log("Completed velocity traces by waveplate angle plots")


def generate_velocity_traces_by_waveplate_angle_hel(summary_csv_path, output_dir, progress_callback=None):
    """
    Generate velocity trace plots focused on HEL detection (±20 ns from t=0).
    Separated by material (Cu/Al) with subplots for each waveplate angle, color-coded by flyer thickness.
    
    Args:
        summary_csv_path: Path to summary CSV file (enhanced_spall_summary.csv or velocity_shots_summary.csv)
        output_dir: Directory to save plots
        progress_callback: Optional callback function for progress messages
    """
    def log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(f"  {message}")
    
    log("Generating velocity traces by material and waveplate angle (HEL detection focus, ±20 ns)...")
    
    # Read summary CSV
    if not os.path.exists(summary_csv_path):
        log(f"ERROR: Summary file not found: {summary_csv_path}")
        return
    
    try:
        df = pd.read_csv(summary_csv_path)
        log(f"Loaded summary with {len(df)} rows")
    except Exception as e:
        log(f"ERROR: Could not read summary file: {e}")
        return
    
    # Find columns for file name, material, waveplate angle, and flyer thickness
    file_col = None
    material_col = None
    waveplate_col = None
    flyer_thickness_col = None
    
    # Try to find file name column
    for col in df.columns:
        col_lower = col.lower()
        if 'pdv' in col_lower and 'file' in col_lower:
            file_col = col
            break
        elif 'file' in col_lower and 'name' in col_lower:
            file_col = col
            break
    
    # Try to find material column
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['material', 'sample material', 'sample_material']:
            material_col = col
            break
    
    # Try to find waveplate angle column
    for col in df.columns:
        col_lower = col.lower()
        if 'waveplate' in col_lower and 'angle' in col_lower:
            waveplate_col = col
            break
        elif 'angle' in col_lower and 'degree' in col_lower:
            waveplate_col = col
            break
    
    # Try to find flyer thickness column
    for col in df.columns:
        col_lower = col.lower()
        if 'flyer' in col_lower and 'thickness' in col_lower:
            flyer_thickness_col = col
            break
    
    if not file_col:
        log("WARNING: Could not find file name column in summary CSV")
        return
    
    if not material_col:
        log("WARNING: Could not find material column in summary CSV")
        return
    
    if not waveplate_col:
        log("WARNING: Could not find waveplate angle column in summary CSV")
        return
    
    if not flyer_thickness_col:
        log("WARNING: Could not find flyer thickness column in summary CSV")
        return
    
    log(f"Using columns: File={file_col}, Material={material_col}, Waveplate={waveplate_col}, Thickness={flyer_thickness_col}")
    
    # Find velocity files directory (parent of SPADE_analysis or Output directory)
    base_dir = os.path.dirname(output_dir)
    if 'SPADE_analysis' in output_dir:
        base_dir = os.path.dirname(output_dir)  # Go up one level from SPADE_analysis
    elif 'Output' in output_dir:
        base_dir = output_dir  # Already in Output directory
    
    # Search for velocity files
    vel_files_pattern = os.path.join(base_dir, '**/*--vel-smooth-with-uncert.csv')
    all_vel_files = glob.glob(vel_files_pattern, recursive=True)
    log(f"Found {len(all_vel_files)} velocity files in {base_dir}")
    
    if len(all_vel_files) == 0:
        log("ERROR: No velocity files found")
        return
    
    # Create a mapping from base filename to velocity file path
    vel_file_map = {}
    for vel_file in all_vel_files:
        filename = os.path.basename(vel_file)
        # Extract base name (remove --vel-smooth-with-uncert.csv)
        base_name = filename.replace('--vel-smooth-with-uncert.csv', '')
        # Also try other suffixes
        for suffix in ['--vel-smooth', '--velocity', '--vel']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        vel_file_map[base_name] = vel_file
    
    # Organize data by material, waveplate angle, and flyer thickness
    # material -> angle -> thickness -> list of (vel_file, row_data)
    traces_by_material_angle_thickness = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for idx, row in df.iterrows():
        try:
            # Get file name from summary
            file_name = str(row[file_col]).strip()
            if pd.isna(file_name) or file_name == '':
                continue
            
            # Extract base name from file name
            base_name = os.path.splitext(file_name)[0]
            # Remove common suffixes
            for suffix in ['--vel-smooth-with-uncert', '--vel-smooth', '--velocity', '--vel']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
                    break
            
            # Find matching velocity file
            vel_file = None
            # Try exact match first
            if base_name in vel_file_map:
                vel_file = vel_file_map[base_name]
            else:
                # Try partial match (PDV filename pattern)
                pdv_pattern = re.search(r'(C\d+--\d{8}--\d{5})', base_name)
                if pdv_pattern:
                    pdv_id = pdv_pattern.group(1)
                    for key, path in vel_file_map.items():
                        if pdv_id in key:
                            vel_file = path
                            break
            
            if not vel_file or not os.path.exists(vel_file):
                continue
            
            # Get material
            material = str(row[material_col]).strip() if not pd.isna(row[material_col]) else 'Unknown'
            # Normalize material names
            material_lower = material.lower()
            if 'cu' in material_lower or 'copper' in material_lower:
                material = 'Cu'
            elif 'al' in material_lower or 'aluminum' in material_lower or 'aluminium' in material_lower:
                material = 'Al'
            else:
                continue  # Skip unknown materials
            
            # Get waveplate angle
            waveplate_angle = row[waveplate_col]
            if pd.isna(waveplate_angle):
                continue
            try:
                waveplate_angle = float(waveplate_angle)
            except (ValueError, TypeError):
                continue
            
            # Get flyer thickness
            flyer_thickness = row[flyer_thickness_col]
            if pd.isna(flyer_thickness):
                continue
            try:
                flyer_thickness_val = float(flyer_thickness)
                # Categorize into 50 um and 100 um (with tolerance)
                if abs(flyer_thickness_val - 50) < abs(flyer_thickness_val - 100):
                    thickness_category = '50 um'
                else:
                    thickness_category = '100 um'
            except (ValueError, TypeError):
                continue
            
            traces_by_material_angle_thickness[material][waveplate_angle][thickness_category].append((vel_file, row))
            
        except Exception as e:
            log(f"  Warning: Error processing row {idx}: {e}")
            continue
    
    # Color mapping for flyer thickness
    thickness_colors = {
        '50 um': 'blue',
        '100 um': 'red'
    }
    
    # Generate plots for each material
    for material in ['Cu', 'Al']:
        if material not in traces_by_material_angle_thickness:
            log(f"No data found for {material}, skipping...")
            continue
        
        angles = sorted(traces_by_material_angle_thickness[material].keys())
        if len(angles) == 0:
            log(f"No waveplate angles found for {material}, skipping...")
            continue
        
        log(f"Generating HEL-focused plot for {material} with {len(angles)} waveplate angles: {angles}")
        
        # Create subplots - one per waveplate angle
        n_angles = len(angles)
        n_cols = min(3, n_angles)  # Max 3 columns
        n_rows = (n_angles + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_angles == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Alignment threshold (10 m/s for HEL detection)
        align_threshold = 10.0
        tolerance = 0.01
        
        total_traces = 0
        
        for idx, angle in enumerate(angles):
            ax = axes[idx]
            
            traces_plotted = 0
            thickness_counts = {'50 um': 0, '100 um': 0}
            
            # Plot traces grouped by thickness
            for thickness_category in ['50 um', '100 um']:
                if thickness_category not in traces_by_material_angle_thickness[material][angle]:
                    continue
                
                traces = traces_by_material_angle_thickness[material][angle][thickness_category]
                color = thickness_colors[thickness_category]
                
                for vel_file, _ in traces:
                    try:
                        # Read velocity file
                        vel_df = pd.read_csv(vel_file)
                        if vel_df.shape[1] < 2:
                            continue
                        
                        # Get time and velocity data
                        time_data = vel_df.iloc[:, 0].values
                        velocity_data = vel_df.iloc[:, 1].values
                        uncertainty_data = vel_df.iloc[:, 2].values if vel_df.shape[1] >= 3 else None
                        
                        # Convert time to ns if needed
                        if np.nanmax(time_data) < 1e-3:
                            time_data = time_data * 1e9  # Convert from seconds to ns
                        elif np.nanmax(time_data) < 1.0:
                            time_data = time_data * 1e3  # Convert from microseconds to ns
                        
                        # Noise fraction filtering
                        noise_file = vel_file.replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
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
                        
                        # Build valid mask with noise suppression
                        valid_mask = ~np.isnan(velocity_data)
                        if high_noise_mask is not None:
                            valid_mask &= (~high_noise_mask)
                        
                        # Uncertainty threshold filtering (50 m/s default)
                        uncertainty_threshold = 50.0
                        if uncertainty_data is not None:
                            valid_mask &= (uncertainty_data <= uncertainty_threshold)
                        
                        time_clean = time_data[valid_mask]
                        velocity_clean = velocity_data[valid_mask]
                        
                        if len(time_clean) == 0 or len(velocity_clean) == 0:
                            continue
                        
                        # Apply threshold alignment
                        t0_idx = None
                        for j, v in enumerate(velocity_clean):
                            if not np.isnan(v) and v >= (align_threshold - tolerance):
                                t0_idx = j
                                break
                        
                        if t0_idx is not None:
                            t0 = time_clean[t0_idx]
                            time_clean = time_clean - t0
                        
                        # Filter to HEL detection range: -20 to +20 ns
                        time_mask = (time_clean >= -20) & (time_clean <= 20)
                        time_plot = time_clean[time_mask]
                        velocity_plot = velocity_clean[time_mask]
                        
                        if len(time_plot) == 0 or len(velocity_plot) == 0:
                            continue
                        
                        # Plot trace with thickness-specific color
                        ax.plot(time_plot, velocity_plot, alpha=0.6, linewidth=0.8, color=color, label=thickness_category if thickness_counts[thickness_category] == 0 else '')
                        traces_plotted += 1
                        thickness_counts[thickness_category] += 1
                        total_traces += 1
                    
                    except Exception as e:
                        log(f"  Warning: Could not plot {os.path.basename(vel_file)}: {e}")
                        continue
            
            # Format subplot
            ax.set_xlabel(f'Time (ns) - aligned to t=0 at {align_threshold} m/s', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
            title_parts = [f'{material} - {angle}° (HEL)']
            if thickness_counts['50 um'] > 0:
                title_parts.append(f'50um: n={thickness_counts["50 um"]}')
            if thickness_counts['100 um'] > 0:
                title_parts.append(f'100um: n={thickness_counts["100 um"]}')
            ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')
            ax.grid(False)
            ax.set_xlim(-20, 20)  # HEL detection range: ±20 ns
            ax.set_ylim(0, 50)  # Y-axis limit for HEL detection focus
            
            # Add legend for thickness colors (outside plot area)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right', fontsize=10, framealpha=0.9)
        
        # Hide unused subplots
        for idx in range(n_angles, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'velocity_traces_{material}_by_waveplate_angle_HEL.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        log(f"✅ Generated {plot_filename} ({total_traces} traces total)")
    
    log("Completed velocity traces by waveplate angle plots (HEL detection focus)")


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


def main():
    # Try to load config from helix_master_config.json
    config_path = None
    if len(sys.argv) > 2:
        config_path = sys.argv[2]  # Optional: config path as second argument
    
    output_dir_from_config = None
    try:
        output_dir_from_config, master_config = load_master_config(config_path)
        # Construct path to summary CSV in SPADE_analysis subdirectory
        spade_analysis_dir = os.path.join(output_dir_from_config, "SPADE_analysis")
        default_input = os.path.join(spade_analysis_dir, "enhanced_spall_summary.csv")
        print(f"Loaded config from helix_master_config.json")
        print(f"  Output directory: {output_dir_from_config}")
        print(f"  SPADE analysis directory: {spade_analysis_dir}")
        print(f"  Default input: {default_input}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Warning: Could not load helix_master_config.json: {e}")
        print("  Falling back to command line argument or hardcoded default")
        # Fallback to hardcoded path
        default_input = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Stieff_Scope/PDV_data/20251218_Cu_Al_500micron/Output/SPADE_analysis/enhanced_spall_summary.csv"
    
    # Get input path from command line argument or use default
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = default_input
    
    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)
    
    # Determine output directory
    # Priority: 1) SPADE_analysis from config, 2) directory of input CSV
    if output_dir_from_config:
        # Use SPADE_analysis directory from config
        output_dir = os.path.join(output_dir_from_config, "SPADE_analysis")
    else:
        # Fallback: use directory of input CSV
        output_dir = os.path.dirname(input_csv)
    
    print("=" * 60)
    print("HELIX Paper Plots Generator")
    print("=" * 60)
    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Progress callback function
    def progress_callback(message):
        print(f"  {message}")
    
    # Generate all plots from available summary files
    print("\n" + "=" * 60)
    print("Generating all available plots...")
    print("=" * 60)
    
    # Check what summary files are available
    enhanced_spall_summary = input_csv  # The provided file
    velocity_shots_summary = os.path.join(output_dir, 'velocity_shots_summary.csv')
    
    print(f"\nChecking for summary files:")
    print(f"  Enhanced Spall Summary: {enhanced_spall_summary}")
    print(f"    Exists: {os.path.exists(enhanced_spall_summary)}")
    print(f"  Velocity Shots Summary: {velocity_shots_summary}")
    print(f"    Exists: {os.path.exists(velocity_shots_summary)}")
    
    # Generate all plots
    try:
        generate_all_plots_from_summary_files(
            enhanced_spall_summary_path=enhanced_spall_summary if os.path.exists(enhanced_spall_summary) else None,
            velocity_shots_summary_path=velocity_shots_summary if os.path.exists(velocity_shots_summary) else None,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        print("\n✅ All plots generated successfully!")
    except Exception as e:
        print(f"\n❌ ERROR generating plots: {e}")
        import traceback
        traceback.print_exc()

    # 3D SVR surface: spall vs strain rate vs shock stress (Al & Cu)
    print("\n" + "=" * 60)
    print("Generating 3D SVR spall surface (strain rate × shock stress)...")
    print("=" * 60)
    if os.path.exists(enhanced_spall_summary):
        try:
            generate_spall_svr_surface_3d(
                enhanced_spall_summary_path=enhanced_spall_summary,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
        except Exception as e:
            print(f"\n❌ ERROR generating SVR surface plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Enhanced spall summary not found - skipping SVR surface plot")

    # Generate failure detection summary table
    print("\n" + "=" * 60)
    print("Generating failure detection summary table...")
    print("=" * 60)
    if os.path.exists(enhanced_spall_summary):
        try:
            generate_failure_detection_table(
                enhanced_spall_summary_path=enhanced_spall_summary,
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            print("\n✅ Failure detection table generated successfully!")
        except Exception as e:
            print(f"\n❌ ERROR generating failure detection table: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Enhanced spall summary not found - skipping failure detection table")
    
    # Generate velocity traces by material and waveplate angle
    print("\n" + "=" * 60)
    print("Generating velocity traces by material and waveplate angle...")
    print("=" * 60)
    
    # Try both summary files for velocity traces
    summary_files_to_try = []
    if os.path.exists(enhanced_spall_summary):
        summary_files_to_try.append(enhanced_spall_summary)
    if os.path.exists(velocity_shots_summary):
        summary_files_to_try.append(velocity_shots_summary)
    
    if summary_files_to_try:
        try:
            # Use the first available summary file
            generate_velocity_traces_by_waveplate_angle(
                summary_csv_path=summary_files_to_try[0],
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            print("\n✅ Velocity traces by waveplate angle generated successfully!")
        except Exception as e:
            print(f"\n❌ ERROR generating velocity traces by waveplate angle: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No summary files available for velocity traces by waveplate angle")
    
    # Generate velocity traces by material and waveplate angle (HEL detection focus)
    print("\n" + "=" * 60)
    print("Generating velocity traces by material and waveplate angle (HEL detection, ±20 ns)...")
    print("=" * 60)
    
    if summary_files_to_try:
        try:
            # Use the first available summary file
            generate_velocity_traces_by_waveplate_angle_hel(
                summary_csv_path=summary_files_to_try[0],
                output_dir=output_dir,
                progress_callback=progress_callback
            )
            print("\n✅ Velocity traces by waveplate angle (HEL focus) generated successfully!")
        except Exception as e:
            print(f"\n❌ ERROR generating velocity traces by waveplate angle (HEL focus): {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  No summary files available for velocity traces by waveplate angle (HEL focus)")
    
    print("\n" + "=" * 60)
    print("Plot generation complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

