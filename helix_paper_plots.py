#!/usr/bin/env python3
"""
HELIX Paper Custom Plots
Custom plotting functions for HELIX paper figures
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def apply_data_to_viz_poster_style():
    """
    Apply a clean, minimal, publication-friendly matplotlib style inspired by the
    "Data to Viz" poster aesthetic (white background, no grid, consistent fonts).
    """
    plt.rcParams.update({
        # Typography
        'font.family': 'DejaVu Sans',
        # +50% vs previous defaults
        'font.size': 27,
        'axes.titlesize': 30,
        'axes.labelsize': 27,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'legend.title_fontsize': 24,

        # Figure / axes
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#222222',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid (disabled for clean look)
        'axes.grid': False,
        'axes.axisbelow': True,
        'grid.color': '#e6e6e6',
        'grid.linewidth': 1.0,
        'grid.alpha': 1.0,

        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#222222',
        'ytick.color': '#222222',

        # Lines/markers defaults
        'lines.linewidth': 2.0,
        'lines.markersize': 8,

        # Savefig
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
    })


# Apply style once on import so all plots are consistent.
apply_data_to_viz_poster_style()


def filter_3sigma_outliers(data_df, x_col, y_col, progress_callback=None):
    """
    Filter outliers using 3-sigma rule on both x and y axes.
    Returns filtered dataframe and list of removed filenames (if available).
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Input data
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data
    progress_callback : callable, optional
        Callback function for progress messages
    
    Returns:
    --------
    filtered_df : pd.DataFrame
        Data with outliers removed
    outliers_removed : list
        List of filenames or indices of removed outliers
    """
    if len(data_df) == 0:
        return data_df, []
    
    # Calculate mean and std for both axes
    x_data = pd.to_numeric(data_df[x_col], errors='coerce')
    y_data = pd.to_numeric(data_df[y_col], errors='coerce')
    
    x_mean = x_data.mean()
    x_std = x_data.std()
    y_mean = y_data.mean()
    y_std = y_data.std()
    
    # Identify outliers (beyond 3 sigma on either axis)
    x_outliers = np.abs(x_data - x_mean) > 3 * x_std
    y_outliers = np.abs(y_data - y_mean) > 3 * y_std
    outlier_mask = x_outliers | y_outliers
    
    # Get list of removed entries
    outliers_removed = []
    if 'Filename' in data_df.columns:
        outliers_removed = data_df[outlier_mask]['Filename'].tolist()
    else:
        outliers_removed = data_df[outlier_mask].index.tolist()
    
    # Filter data
    filtered_df = data_df[~outlier_mask].copy()
    
    if progress_callback and len(outliers_removed) > 0:
        progress_callback(f"   3-sigma filter: Removed {len(outliers_removed)} outlier(s) from {len(data_df)} data points")
        if len(outliers_removed) <= 10:
            for outlier in outliers_removed:
                progress_callback(f"     Removed: {outlier}")
    
    return filtered_df, outliers_removed


def get_material_color_mapping(materials):
    """
    Generate consistent color mapping for materials across all plots.
    Uses predefined colors for common materials, then colormap for others.
    Ensures same material always gets same color.
    """
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
        'SS': '#8c564b',      # Stainless Steel
        'Stainless Steel': '#8c564b',
        'Unknown': '#7f7f7f'  # Gray
    }
    
    # Create mapping
    color_map = {}
    cmap = plt.get_cmap('tab10' if len(materials) <= 10 else 'tab20')
    
    for i, material in enumerate(materials):
        # Check predefined colors first
        if material in predefined_colors:
            color_map[material] = predefined_colors[material]
        else:
            # Use colormap for other materials
            color_map[material] = cmap(i / max(len(materials), 1))
    
    return color_map


def get_material_marker_mapping(materials):
    """
    Generate consistent marker mapping for materials across all plots.
    Ensures same material always gets same marker shape.
    """
    # Predefined markers for common materials (consistent across all plots)
    predefined_markers = {
        'Cu': 'o',           # Circle
        'Copper': 'o',
        'Zn': 's',           # Square
        'Zinc': 's',
        'Brass': '^',        # Triangle up
        'Al': 'D',           # Diamond
        'Aluminum': 'D',
        'Ti': 'v',           # Triangle down
        'Titanium': 'v',
        'Steel': '<',        # Triangle left
        'SS': '<',
        'Stainless Steel': '<',
        'Unknown': 'x'       # X
    }
    
    # Available markers
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
    
    # Create mapping
    marker_map = {}
    marker_idx = 0
    
    for material in materials:
        # Check predefined markers first
        if material in predefined_markers:
            marker_map[material] = predefined_markers[material]
        else:
            # Use next available marker
            marker_map[material] = available_markers[marker_idx % len(available_markers)]
            marker_idx += 1
    
    return marker_map


def apply_consistent_plot_formatting(ax, xlabel, ylabel, title=None, fontsize=27):
    """
    Apply consistent formatting to plot axes.
    - Clean, minimal look (grid handled by global rcParams)
    """
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    ax.tick_params(labelsize=max(10, fontsize - 2))


def find_column_name(df, possible_names, progress_callback=None):
    """
    Find a column name in DataFrame by trying multiple possible names.
    Returns the first matching column name, or None if not found.
    Excludes ALPSS columns (columns starting with 'ALPSS_') to prioritize SPADE data.
    """
    # Filter out ALPSS columns - we only want SPADE data
    non_alpss_columns = [col for col in df.columns if not col.startswith('ALPSS_')]
    
    for name in possible_names:
        if name in non_alpss_columns:
            return name
    
    # Try case-insensitive matching (excluding ALPSS columns)
    df_cols_lower = {col.lower(): col for col in non_alpss_columns}
    for name in possible_names:
        name_lower = name.lower()
        if name_lower in df_cols_lower:
            return df_cols_lower[name_lower]
    
    # Try partial matching (contains) - excluding ALPSS columns
    for name in possible_names:
        name_lower = name.lower().replace(' ', '').replace('_', '').replace('-', '')
        for col in non_alpss_columns:
            col_lower = col.lower().replace(' ', '').replace('_', '').replace('-', '')
            if name_lower in col_lower or col_lower in name_lower:
                if progress_callback:
                    progress_callback(f"   Found column '{col}' for '{name}'")
                return col
    
    if progress_callback:
        progress_callback(f"   WARNING: Could not find column matching: {possible_names}")
    return None


def generate_spall_vs_strain_rate_plot(summary_df, spade_output_dir, progress_callback=None):
    """Generate Spall Strength vs Strain Rate plot matching HEL plot format"""
    if progress_callback:
        progress_callback("Generating Spall Strength vs Strain Rate plot...")
    
    try:
        # Find column names (handle different naming conventions)
        spall_strength_col = find_column_name(
            summary_df, 
            ['Spall Strength (GPa)', 'Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall Strength'],
            progress_callback
        )
        strain_rate_col = find_column_name(
            summary_df,
            ['Strain Rate (s^-1)', 'Spall_StrainRate_s^-1', 'Strain_Rate_s^-1', 'Strain Rate'],
            progress_callback
        )
        
        if spall_strength_col is None or strain_rate_col is None:
            if progress_callback:
                progress_callback("⚠ Required columns not found - skipping plot")
                progress_callback(f"   Available columns: {list(summary_df.columns)[:10]}...")
            return
        
        if progress_callback:
            progress_callback(f"   Using columns: '{spall_strength_col}' and '{strain_rate_col}'")
        
        # Filter data: only rows with valid Spall Strength and Strain Rate
        valid_data = summary_df[
            (summary_df[spall_strength_col].notna()) & 
            (summary_df[strain_rate_col].notna())
        ].copy()
        
        # Remove rows with non-positive values
        valid_data = valid_data[
            (pd.to_numeric(valid_data[spall_strength_col], errors='coerce') > 0) &
            (pd.to_numeric(valid_data[strain_rate_col], errors='coerce') > 0)
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid Spall Strength vs Strain Rate data - skipping plot")
            return
        
        # Apply 3-sigma outlier filter to remove extreme points
        if len(valid_data) > 3:
            valid_data, outliers = filter_3sigma_outliers(
                valid_data, 
                strain_rate_col, 
                spall_strength_col,
                progress_callback=progress_callback
            )
            
            if len(valid_data) == 0:
                if progress_callback:
                    progress_callback("⚠ No valid data after 3-sigma filtering - skipping plot")
                return
        
        # Get material column
        material_col = None
        for col_name in valid_data.columns:
            if col_name.lower() == 'material':
                material_col = col_name
                break
        
        if material_col is None:
            valid_data['Material'] = 'Unknown'
            material_col = 'Material'
        
        # Ensure numeric
        valid_data[spall_strength_col] = pd.to_numeric(valid_data[spall_strength_col], errors='coerce')
        valid_data[strain_rate_col] = pd.to_numeric(valid_data[strain_rate_col], errors='coerce')
        
        # Find uncertainty column names
        spall_unc_col = find_column_name(
            valid_data,
            ['Spall Strength Uncertainty (GPa)', 'Spall_Strength_Unc_GPa', 'Spall_Strength_Uncertainty_GPa', 
             'Spall_Strength_Unc_GPa_Final', 'Spall Strength Uncertainty'],
            progress_callback
        )
        
        strain_unc_col = find_column_name(
            valid_data,
            ['Strain Rate Uncertainty (s^-1)', 'Strain_Rate_Uncertainty_s^-1', 'Strain_Rate_Unc_s^-1',
             'StrainRate_Unc_s^-1', 'Spall_StrainRate_UNCERTAINITY', 'Strain_Rate_Uncertainty_s1_Final',
             'Strain_Rate_Uncertainty_s1', 'Strain Rate Uncertainty'],
            progress_callback
        )
        
        # Convert uncertainty to numeric (replace strings like "DNS" with NaN)
        if spall_unc_col:
            valid_data[spall_unc_col] = pd.to_numeric(valid_data[spall_unc_col], errors='coerce')
        
        if strain_unc_col:
            valid_data[strain_unc_col] = pd.to_numeric(valid_data[strain_unc_col], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        valid_data = valid_data[
            (valid_data[spall_strength_col].notna()) & 
            (valid_data[strain_rate_col].notna())
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data after numeric conversion - skipping plot")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique materials and assign colors and markers
        materials = valid_data[material_col].unique()
        colors = get_material_color_mapping(materials)
        markers = get_material_marker_mapping(materials)
        
        # Plot data grouped by material
        legend_handles = []
        legend_labels = []
        
        for material in materials:
            material_data = valid_data[valid_data[material_col] == material]
            
            if len(material_data) == 0:
                continue
            
            marker = markers[material]
            color = colors[material]
            n_points = len(material_data)
            
            # Get uncertainty values for both x and y axes
            # Initialize with NaN arrays - matplotlib will skip error bars for NaN values
            yerr = np.full(len(material_data), np.nan)
            xerr = np.full(len(material_data), np.nan)
            
            # Y-error bars: Spall Strength Uncertainty
            if spall_unc_col and spall_unc_col in material_data.columns:
                yerr_series = pd.to_numeric(material_data[spall_unc_col], errors='coerce')
                # Replace NaN values in yerr array with actual uncertainty values where available
                valid_mask = yerr_series.notna() & (yerr_series > 0)
                yerr[valid_mask] = yerr_series[valid_mask].values
            
            # X-error bars: Strain Rate Uncertainty
            if strain_unc_col and strain_unc_col in material_data.columns:
                xerr_series = pd.to_numeric(material_data[strain_unc_col], errors='coerce')
                # Replace NaN values in xerr array with actual uncertainty values where available
                valid_mask = xerr_series.notna() & (xerr_series > 0)
                xerr[valid_mask] = xerr_series[valid_mask].values
            
            # Always use errorbar for consistency - matplotlib handles NaN values gracefully
            errorbar_handle = ax.errorbar(
                material_data[strain_rate_col],
                material_data[spall_strength_col],
                xerr=xerr,  # X-error bars for strain rate uncertainty
                yerr=yerr,  # Y-error bars for spall strength uncertainty
                fmt=marker,
                color=color,
                markersize=10,
                linewidth=0,
                elinewidth=2.0,  # Thicker error bars for better visibility
                capsize=5,  # Larger caps for better visibility
                capthick=2.0,  # Thicker caps
                alpha=0.7,
                label=f"{material} (n={n_points})"
            )
            legend_handles.append(errorbar_handle[0])
            
            legend_labels.append(f"{material} (n={n_points})")
        
        # Set labels and formatting
        apply_consistent_plot_formatting(ax, 'Strain Rate (s^-1)', 'Spall Strength (GPa)', 
                                         'Spall Strength vs Strain Rate by Material')
        ax.set_xscale('log')  # Use log scale for strain rate
        y_max_data = pd.to_numeric(valid_data[spall_strength_col], errors='coerce').max()
        ax.set_ylim(0, max(3.5, y_max_data * 1.15))
        ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=20)
        
        # Tight layout and save
        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'spall_vs_strain_rate.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if progress_callback:
            progress_callback(f"✅ Generated Spall Strength vs Strain Rate plot: {plot_path}")
            progress_callback(f"   Plotted {len(valid_data)} data points from {len(materials)} material(s)")
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Spall Strength vs Strain Rate plot: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_spall_vs_shock_stress_plot(summary_df, spade_output_dir, progress_callback=None):
    """Generate Spall Strength vs Shock Stress plot with error bars"""
    if progress_callback:
        progress_callback("Generating Spall Strength vs Shock Stress plot...")
    
    try:
        # Find column names
        spall_strength_col = find_column_name(
            summary_df,
            ['Spall Strength (GPa)', 'Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall Strength'],
            progress_callback
        )
        
        shock_stress_col = find_column_name(
            summary_df,
            ['Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa', 'Peak_Shock_Stress_GPa_Final',
             'Shock Stress (GPa)', 'Shock_Stress_GPa', 'Peak Shock Stress'],
            progress_callback
        )
        
        if spall_strength_col is None or shock_stress_col is None:
            if progress_callback:
                progress_callback("⚠ Required columns not found - skipping plot")
            return
        
        # Filter data
        valid_data = summary_df[
            (summary_df[spall_strength_col].notna()) &
            (summary_df[shock_stress_col].notna())
        ].copy()
        
        valid_data = valid_data[
            (pd.to_numeric(valid_data[spall_strength_col], errors='coerce') > 0) &
            (pd.to_numeric(valid_data[shock_stress_col], errors='coerce') > 0)
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data - skipping plot")
            return

        # Apply 3-sigma outlier filter (consistent with strain-rate plots)
        if len(valid_data) > 3:
            valid_data, outliers = filter_3sigma_outliers(
                valid_data,
                shock_stress_col,
                spall_strength_col,
                progress_callback=progress_callback
            )
            if len(valid_data) == 0:
                if progress_callback:
                    progress_callback("⚠ No valid data after 3-sigma filtering - skipping plot")
                return

        # Get material column
        material_col = find_column_name(valid_data, ['Material', 'material', 'Sample material'], progress_callback)
        if material_col is None:
            valid_data['Material'] = 'Unknown'
            material_col = 'Material'
        
        # Ensure numeric
        valid_data[spall_strength_col] = pd.to_numeric(valid_data[spall_strength_col], errors='coerce')
        valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
        
        # Find uncertainty columns
        spall_unc_col = find_column_name(
            valid_data,
            ['Spall Strength Uncertainty (GPa)', 'Spall_Strength_Unc_GPa', 'Spall_Strength_Uncertainty_GPa'],
            progress_callback
        )
        
        shock_unc_col = find_column_name(
            valid_data,
            ['Peak Shock Stress Uncertainty (GPa)', 'Peak_Shock_Stress_Uncertainty_GPa'],
            progress_callback
        )
        
        if spall_unc_col:
            valid_data[spall_unc_col] = pd.to_numeric(valid_data[spall_unc_col], errors='coerce')
        if shock_unc_col:
            valid_data[shock_unc_col] = pd.to_numeric(valid_data[shock_unc_col], errors='coerce')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        materials = valid_data[material_col].unique()
        colors = get_material_color_mapping(materials)
        markers = get_material_marker_mapping(materials)
        legend_handles = []
        legend_labels = []
        
        for material in materials:
            material_data = valid_data[valid_data[material_col] == material]
            if len(material_data) == 0:
                continue
            
            marker = markers[material]
            color = colors[material]
            n_points = len(material_data)
            
            # Error bars
            yerr = np.full(len(material_data), np.nan)
            xerr = np.full(len(material_data), np.nan)
            
            if spall_unc_col and spall_unc_col in material_data.columns:
                yerr_series = pd.to_numeric(material_data[spall_unc_col], errors='coerce')
                valid_mask = yerr_series.notna() & (yerr_series > 0)
                yerr[valid_mask] = yerr_series[valid_mask].values
            
            if shock_unc_col and shock_unc_col in material_data.columns:
                xerr_series = pd.to_numeric(material_data[shock_unc_col], errors='coerce')
                valid_mask = xerr_series.notna() & (xerr_series > 0)
                xerr[valid_mask] = xerr_series[valid_mask].values
            
            errorbar_handle = ax.errorbar(
                material_data[shock_stress_col],
                material_data[spall_strength_col],
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                color=color,
                markersize=10,
                linewidth=0,
                elinewidth=2.0,
                capsize=5,
                capthick=2.0,
                alpha=0.7,
                label=f"{material} (n={n_points})"
            )
            legend_handles.append(errorbar_handle[0])
            legend_labels.append(f"{material} (n={n_points})")
        
        # Set labels and formatting
        apply_consistent_plot_formatting(ax, 'Peak Shock Stress (GPa)', 'Spall Strength (GPa)',
                                         'Spall Strength vs Shock Stress by Material')
        y_max_data = pd.to_numeric(valid_data[spall_strength_col], errors='coerce').max()
        ax.set_ylim(0, max(3.5, y_max_data * 1.15))
        ax.legend(legend_handles, legend_labels, title='Material', loc='best', fontsize=20)
        
        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'spall_vs_shock_stress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if progress_callback:
            progress_callback(f"✅ Generated Spall Strength vs Shock Stress plot: {plot_path}")
    
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Spall Strength vs Shock Stress plot: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_spall_vs_shock_stress_by_material_subplots(summary_df, spade_output_dir, progress_callback=None):
    """
    Generate Spall Strength vs Shock Stress plot with two subplots side by side:
    - Left: Copper (Cu)
    - Right: Aluminum (Al)
    Data points are color-coded by flyer thickness (50 um / 100 um)
    """
    if progress_callback:
        progress_callback("Generating Spall Strength vs Shock Stress by Material (Cu/Al subplots)...")
    
    try:
        # Find column names
        spall_strength_col = find_column_name(
            summary_df,
            ['Spall Strength (GPa)', 'Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall Strength'],
            progress_callback
        )
        
        shock_stress_col = find_column_name(
            summary_df,
            ['Peak Shock Stress (GPa)', 'Peak_Shock_Stress_GPa', 'Peak_Shock_Stress_GPa_Final',
             'Shock Stress (GPa)', 'Shock_Stress_GPa', 'Peak Shock Stress'],
            progress_callback
        )
        
        if spall_strength_col is None or shock_stress_col is None:
            if progress_callback:
                progress_callback("⚠ Required columns not found - skipping plot")
            return
        
        # Filter data
        valid_data = summary_df[
            (summary_df[spall_strength_col].notna()) &
            (summary_df[shock_stress_col].notna())
        ].copy()
        
        valid_data = valid_data[
            (pd.to_numeric(valid_data[spall_strength_col], errors='coerce') > 0) &
            (pd.to_numeric(valid_data[shock_stress_col], errors='coerce') > 0)
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data - skipping plot")
            return

        # Apply 3-sigma outlier filter (consistent with strain-rate plots)
        if len(valid_data) > 3:
            valid_data, outliers = filter_3sigma_outliers(
                valid_data,
                shock_stress_col,
                spall_strength_col,
                progress_callback=progress_callback
            )
            if len(valid_data) == 0:
                if progress_callback:
                    progress_callback("⚠ No valid data after 3-sigma filtering - skipping plot")
                return

        # Get material column
        material_col = find_column_name(valid_data, ['Material', 'material', 'Sample material'], progress_callback)
        if material_col is None:
            if progress_callback:
                progress_callback("⚠ Material column not found - skipping plot")
            return
        
        # Filter for Cu and Al only
        valid_data = valid_data[
            (valid_data[material_col].str.upper().isin(['CU', 'COPPER', 'AL', 'ALUMINUM', 'ALUMINIUM']))
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No Cu or Al data found - skipping plot")
            return
        
        # Normalize material names
        valid_data['Material_Normalized'] = valid_data[material_col].str.upper()
        valid_data.loc[valid_data['Material_Normalized'].isin(['CU', 'COPPER']), 'Material_Normalized'] = 'Cu'
        valid_data.loc[valid_data['Material_Normalized'].isin(['AL', 'ALUMINUM', 'ALUMINIUM']), 'Material_Normalized'] = 'Al'
        
        # Get flyer thickness column
        flyer_thickness_col = None
        for col_name in valid_data.columns:
            if 'flyer' in col_name.lower() and 'thickness' in col_name.lower():
                flyer_thickness_col = col_name
                break
        
        if flyer_thickness_col is None:
            if progress_callback:
                progress_callback("⚠ Flyer thickness column not found - skipping plot")
            return
        
        # Convert flyer thickness to numeric and categorize
        valid_data[flyer_thickness_col] = pd.to_numeric(valid_data[flyer_thickness_col], errors='coerce')
        # Categorize into 50 um and 100 um (with some tolerance)
        valid_data['Flyer_Thickness_Category'] = valid_data[flyer_thickness_col].apply(
            lambda x: '50 um' if pd.notna(x) and abs(x - 50) < abs(x - 100) else '100 um' if pd.notna(x) else None
        )
        
        # Ensure numeric
        valid_data[spall_strength_col] = pd.to_numeric(valid_data[spall_strength_col], errors='coerce')
        valid_data[shock_stress_col] = pd.to_numeric(valid_data[shock_stress_col], errors='coerce')
        
        # Find uncertainty columns
        spall_unc_col = find_column_name(
            valid_data,
            ['Spall Strength Uncertainty (GPa)', 'Spall_Strength_Unc_GPa', 'Spall_Strength_Uncertainty_GPa'],
            progress_callback
        )
        
        shock_unc_col = find_column_name(
            valid_data,
            ['Peak Shock Stress Uncertainty (GPa)', 'Peak_Shock_Stress_Uncertainty_GPa'],
            progress_callback
        )
        
        if spall_unc_col:
            valid_data[spall_unc_col] = pd.to_numeric(valid_data[spall_unc_col], errors='coerce')
        if shock_unc_col:
            valid_data[shock_unc_col] = pd.to_numeric(valid_data[shock_unc_col], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        valid_data = valid_data[
            (valid_data[spall_strength_col].notna()) &
            (valid_data[shock_stress_col].notna()) &
            (valid_data['Flyer_Thickness_Category'].notna())
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data after filtering - skipping plot")
            return
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Color mapping for flyer thickness
        thickness_colors = {
            '50 um': '#2ca02c',   # Green
            '100 um': '#ff7f0e'   # Orange
        }
        
        # Plot Cu data (left subplot)
        cu_data = valid_data[valid_data['Material_Normalized'] == 'Cu']
        if len(cu_data) > 0:
            for thickness in ['50 um', '100 um']:
                thickness_data = cu_data[cu_data['Flyer_Thickness_Category'] == thickness]
                if len(thickness_data) > 0:
                    # Get uncertainty values
                    yerr = np.full(len(thickness_data), np.nan)
                    xerr = np.full(len(thickness_data), np.nan)
                    
                    if spall_unc_col and spall_unc_col in thickness_data.columns:
                        yerr_series = pd.to_numeric(thickness_data[spall_unc_col], errors='coerce')
                        valid_mask = yerr_series.notna() & (yerr_series > 0)
                        yerr[valid_mask] = yerr_series[valid_mask].values
                    
                    if shock_unc_col and shock_unc_col in thickness_data.columns:
                        xerr_series = pd.to_numeric(thickness_data[shock_unc_col], errors='coerce')
                        valid_mask = xerr_series.notna() & (xerr_series > 0)
                        xerr[valid_mask] = xerr_series[valid_mask].values
                    
                    ax1.errorbar(
                        thickness_data[shock_stress_col],
                        thickness_data[spall_strength_col],
                        xerr=xerr,
                        yerr=yerr,
                        fmt='o',
                        color=thickness_colors[thickness],
                        markersize=10,
                        linewidth=0,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.7,
                        label=f"{thickness} (n={len(thickness_data)})"
                    )
        
        # Plot Al data (right subplot)
        al_data = valid_data[valid_data['Material_Normalized'] == 'Al']
        if len(al_data) > 0:
            for thickness in ['50 um', '100 um']:
                thickness_data = al_data[al_data['Flyer_Thickness_Category'] == thickness]
                if len(thickness_data) > 0:
                    # Get uncertainty values
                    yerr = np.full(len(thickness_data), np.nan)
                    xerr = np.full(len(thickness_data), np.nan)
                    
                    if spall_unc_col and spall_unc_col in thickness_data.columns:
                        yerr_series = pd.to_numeric(thickness_data[spall_unc_col], errors='coerce')
                        valid_mask = yerr_series.notna() & (yerr_series > 0)
                        yerr[valid_mask] = yerr_series[valid_mask].values
                    
                    if shock_unc_col and shock_unc_col in thickness_data.columns:
                        xerr_series = pd.to_numeric(thickness_data[shock_unc_col], errors='coerce')
                        valid_mask = xerr_series.notna() & (xerr_series > 0)
                        xerr[valid_mask] = xerr_series[valid_mask].values
                    
                    ax2.errorbar(
                        thickness_data[shock_stress_col],
                        thickness_data[spall_strength_col],
                        xerr=xerr,
                        yerr=yerr,
                        fmt='o',
                        color=thickness_colors[thickness],
                        markersize=10,
                        linewidth=0,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.7,
                        label=f"{thickness} (n={len(thickness_data)})"
                    )
        
        # Configure left subplot (Cu)
        apply_consistent_plot_formatting(ax1, 'Peak Shock Stress (GPa)', 'Spall Strength (GPa)', 'Copper (Cu)')
        ax1.legend(title='Flyer Thickness', loc='best')
        
        # Configure right subplot (Al)
        apply_consistent_plot_formatting(ax2, 'Peak Shock Stress (GPa)', 'Spall Strength (GPa)', 'Aluminum (Al)')
        ax2.legend(title='Flyer Thickness', loc='best')
        
        # Set y-axis to fit data, with 3.5 GPa as a minimum ceiling
        y_max_data = pd.to_numeric(valid_data[spall_strength_col], errors='coerce').max()
        y_top = max(3.5, y_max_data * 1.15)
        ax1.set_ylim(0, y_top)
        ax2.set_ylim(0, y_top)

        # Set same x-axis limits for both subplots
        x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)
        
        # Tight layout and save
        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'spall_vs_shock_stress_by_material_subplots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if progress_callback:
            progress_callback(f"✅ Generated Spall Strength vs Shock Stress by Material subplots: {plot_path}")
            progress_callback(f"   Cu: {len(cu_data)} points, Al: {len(al_data)} points")
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Spall Strength vs Shock Stress by Material subplots: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_spall_vs_strain_rate_by_material_subplots(summary_df, spade_output_dir, progress_callback=None):
    """
    Generate Spall Strength vs Strain Rate plot with two subplots side by side:
    - Left: Copper (Cu)
    - Right: Aluminum (Al)
    Data points are color-coded by flyer thickness (50 um / 100 um)
    """
    if progress_callback:
        progress_callback("Generating Spall Strength vs Strain Rate by Material (Cu/Al subplots)...")
    
    try:
        # Find column names (handle different naming conventions)
        spall_strength_col = find_column_name(
            summary_df, 
            ['Spall Strength (GPa)', 'Spall_Strength_GPa', 'Spall_Strength_GPa_Final', 'Spall Strength'],
            progress_callback
        )
        strain_rate_col = find_column_name(
            summary_df,
            ['Strain Rate (s^-1)', 'Spall_StrainRate_s^-1', 'Strain_Rate_s^-1', 'Strain Rate'],
            progress_callback
        )
        
        if spall_strength_col is None or strain_rate_col is None:
            if progress_callback:
                progress_callback("⚠ Required columns not found - skipping plot")
                progress_callback(f"   Available columns: {list(summary_df.columns)[:10]}...")
            return
        
        if progress_callback:
            progress_callback(f"   Using columns: '{spall_strength_col}' and '{strain_rate_col}'")
        
        # Filter data: only rows with valid Spall Strength and Strain Rate
        valid_data = summary_df[
            (summary_df[spall_strength_col].notna()) & 
            (summary_df[strain_rate_col].notna())
        ].copy()
        
        # Remove rows with non-positive values
        valid_data = valid_data[
            (pd.to_numeric(valid_data[spall_strength_col], errors='coerce') > 0) &
            (pd.to_numeric(valid_data[strain_rate_col], errors='coerce') > 0)
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid Spall Strength vs Strain Rate data - skipping plot")
            return
        
        # Apply 3-sigma outlier filter to remove extreme points
        if len(valid_data) > 3:
            valid_data, outliers = filter_3sigma_outliers(
                valid_data, 
                strain_rate_col, 
                spall_strength_col,
                progress_callback=progress_callback
            )
            
            if len(valid_data) == 0:
                if progress_callback:
                    progress_callback("⚠ No valid data after 3-sigma filtering - skipping plot")
                return
        
        # Get material column
        material_col = find_column_name(
            valid_data,
            ['Material', 'material', 'Sample material', 'Sample_Material'],
            progress_callback
        )
        
        if material_col is None:
            if progress_callback:
                progress_callback("⚠ Material column not found - skipping plot")
            return
        
        # Filter for Cu and Al only
        valid_data = valid_data[
            (valid_data[material_col].str.upper().isin(['CU', 'COPPER', 'AL', 'ALUMINUM', 'ALUMINIUM']))
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No Cu or Al data found - skipping plot")
            return
        
        # Normalize material names
        valid_data['Material_Normalized'] = valid_data[material_col].str.upper()
        valid_data.loc[valid_data['Material_Normalized'].isin(['CU', 'COPPER']), 'Material_Normalized'] = 'Cu'
        valid_data.loc[valid_data['Material_Normalized'].isin(['AL', 'ALUMINUM', 'ALUMINIUM']), 'Material_Normalized'] = 'Al'
        
        # Get flyer thickness column
        flyer_thickness_col = None
        for col_name in valid_data.columns:
            if 'flyer' in col_name.lower() and 'thickness' in col_name.lower():
                flyer_thickness_col = col_name
                break
        
        if flyer_thickness_col is None:
            if progress_callback:
                progress_callback("⚠ Flyer thickness column not found - skipping plot")
            return
        
        # Convert flyer thickness to numeric and categorize
        valid_data[flyer_thickness_col] = pd.to_numeric(valid_data[flyer_thickness_col], errors='coerce')
        # Categorize into 50 um and 100 um (with some tolerance)
        valid_data['Flyer_Thickness_Category'] = valid_data[flyer_thickness_col].apply(
            lambda x: '50 um' if pd.notna(x) and abs(x - 50) < abs(x - 100) else '100 um' if pd.notna(x) else None
        )
        
        # Ensure numeric (use the found column names)
        valid_data[spall_strength_col] = pd.to_numeric(valid_data[spall_strength_col], errors='coerce')
        valid_data[strain_rate_col] = pd.to_numeric(valid_data[strain_rate_col], errors='coerce')
        
        # Find uncertainty column names
        spall_unc_col = find_column_name(
            valid_data,
            ['Spall Strength Uncertainty (GPa)', 'Spall_Strength_Unc_GPa', 'Spall_Strength_Uncertainty_GPa', 
             'Spall_Strength_Unc_GPa_Final', 'Spall Strength Uncertainty'],
            progress_callback
        )
        
        strain_unc_col = find_column_name(
            valid_data,
            ['Strain Rate Uncertainty (s^-1)', 'Strain_Rate_Uncertainty_s^-1', 'Strain_Rate_Unc_s^-1',
             'StrainRate_Unc_s^-1', 'Spall_StrainRate_UNCERTAINITY', 'Strain_Rate_Uncertainty_s1_Final',
             'Strain_Rate_Uncertainty_s1', 'Strain Rate Uncertainty'],
            progress_callback
        )
        
        # Convert uncertainty to numeric
        if spall_unc_col:
            valid_data[spall_unc_col] = pd.to_numeric(valid_data[spall_unc_col], errors='coerce')
        
        if strain_unc_col:
            valid_data[strain_unc_col] = pd.to_numeric(valid_data[strain_unc_col], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        valid_data = valid_data[
            (valid_data[spall_strength_col].notna()) & 
            (valid_data[strain_rate_col].notna()) &
            (valid_data['Flyer_Thickness_Category'].notna())
        ].copy()
        
        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data after filtering - skipping plot")
            return
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Color mapping for flyer thickness
        thickness_colors = {
            '50 um': '#2ca02c',   # Green
            '100 um': '#ff7f0e'   # Orange
        }
        
        # Plot Cu data (left subplot)
        cu_data = valid_data[valid_data['Material_Normalized'] == 'Cu']
        if len(cu_data) > 0:
            for thickness in ['50 um', '100 um']:
                thickness_data = cu_data[cu_data['Flyer_Thickness_Category'] == thickness]
                if len(thickness_data) > 0:
                    # Get uncertainty values
                    yerr = np.full(len(thickness_data), np.nan)
                    xerr = np.full(len(thickness_data), np.nan)
                    
                    if spall_unc_col and spall_unc_col in thickness_data.columns:
                        yerr_series = pd.to_numeric(thickness_data[spall_unc_col], errors='coerce')
                        valid_mask = yerr_series.notna() & (yerr_series > 0)
                        yerr[valid_mask] = yerr_series[valid_mask].values
                    
                    if strain_unc_col and strain_unc_col in thickness_data.columns:
                        xerr_series = pd.to_numeric(thickness_data[strain_unc_col], errors='coerce')
                        valid_mask = xerr_series.notna() & (xerr_series > 0)
                        xerr[valid_mask] = xerr_series[valid_mask].values
                    
                    ax1.errorbar(
                        thickness_data[strain_rate_col],
                        thickness_data[spall_strength_col],
                        xerr=xerr,
                        yerr=yerr,
                        fmt='o',
                        color=thickness_colors[thickness],
                        markersize=10,
                        linewidth=0,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.7,
                        label=f"{thickness} (n={len(thickness_data)})"
                    )
        
        # Plot Al data (right subplot)
        al_data = valid_data[valid_data['Material_Normalized'] == 'Al']
        if len(al_data) > 0:
            for thickness in ['50 um', '100 um']:
                thickness_data = al_data[al_data['Flyer_Thickness_Category'] == thickness]
                if len(thickness_data) > 0:
                    # Get uncertainty values
                    yerr = np.full(len(thickness_data), np.nan)
                    xerr = np.full(len(thickness_data), np.nan)
                    
                    if spall_unc_col and spall_unc_col in thickness_data.columns:
                        yerr_series = pd.to_numeric(thickness_data[spall_unc_col], errors='coerce')
                        valid_mask = yerr_series.notna() & (yerr_series > 0)
                        yerr[valid_mask] = yerr_series[valid_mask].values
                    
                    if strain_unc_col and strain_unc_col in thickness_data.columns:
                        xerr_series = pd.to_numeric(thickness_data[strain_unc_col], errors='coerce')
                        valid_mask = xerr_series.notna() & (xerr_series > 0)
                        xerr[valid_mask] = xerr_series[valid_mask].values
                    
                    ax2.errorbar(
                        thickness_data[strain_rate_col],
                        thickness_data[spall_strength_col],
                        xerr=xerr,
                        yerr=yerr,
                        fmt='o',
                        color=thickness_colors[thickness],
                        markersize=10,
                        linewidth=0,
                        elinewidth=2.0,
                        capsize=5,
                        capthick=2.0,
                        alpha=0.7,
                        label=f"{thickness} (n={len(thickness_data)})"
                    )
        
        # Configure left subplot (Cu)
        apply_consistent_plot_formatting(ax1, 'Strain Rate (s^-1)', 'Spall Strength (GPa)', 'Copper (Cu)')
        ax1.set_xscale('log')
        ax1.legend(title='Flyer Thickness', loc='best')

        # Configure right subplot (Al)
        apply_consistent_plot_formatting(ax2, 'Strain Rate (s^-1)', 'Spall Strength (GPa)', 'Aluminum (Al)')
        ax2.set_xscale('log')
        ax2.legend(title='Flyer Thickness', loc='best')

        # Set shared y-axis ceiling based on actual data (minimum 3.5 GPa)
        y_max_data = pd.to_numeric(valid_data[spall_strength_col], errors='coerce').max()
        y_top = max(3.5, y_max_data * 1.15)
        ax1.set_ylim(0, y_top)
        ax2.set_ylim(0, y_top)
        
        # Tight layout and save
        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'spall_vs_strain_rate_by_material_subplots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if progress_callback:
            progress_callback(f"✅ Generated Spall Strength vs Strain Rate by Material subplots: {plot_path}")
            progress_callback(f"   Cu: {len(cu_data)} points, Al: {len(al_data)} points")
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Spall Strength vs Strain Rate by Material subplots: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_peak_velocity_vs_time_plot(summary_df, spade_output_dir, progress_callback=None):
    """
    Generate Peak Velocity vs Time plot with time relative to first shot.
    X-axis: Relative time (first shot = 0)
    Y-axis: Peak velocity (m/s)
    Points colored by laser energy bin (300 mJ intervals).
    Dashed trendlines show piecewise-linear mean velocity vs time for each bin.
    """
    if progress_callback:
        progress_callback("Generating Peak Velocity vs Time (relative) plot...")

    try:
        # Find timestamp column
        timestamp_col = find_column_name(
            summary_df,
            ['Timestamp', 'timestamp', 'Shot_Time', 'Shot Time', 'Exp_Time', 'Exp_Time (seconds)',
             'Time (seconds)', 'time', 'Shot_time'],
            progress_callback
        )

        # Find peak velocity column
        peak_vel_col = find_column_name(
            summary_df,
            ['Plateau Mean Velocity (m/s)', 'Plateau_Mean_Velocity_ms', 'First_Maxima_m_s',
             'Peak Velocity (m/s)', 'Peak_Velocity', 'Peak velocity'],
            progress_callback
        )

        # Find laser energy column
        laser_energy_col = find_column_name(
            summary_df,
            ['Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 'Laser_Ref_Energy (mJ)',
             'Laser energy (mJ)', 'Laser Energy (mJ)', 'laser_energy', 'Energy (mJ)'],
            progress_callback
        )

        if timestamp_col is None:
            if progress_callback:
                progress_callback("⚠ Timestamp column not found - skipping Peak Velocity vs Time plot")
            return

        if peak_vel_col is None:
            if progress_callback:
                progress_callback("⚠ Peak velocity column not found - skipping Peak Velocity vs Time plot")
            return

        if laser_energy_col is None:
            if progress_callback:
                progress_callback("⚠ Laser energy column not found - skipping Peak Velocity vs Time plot")
            return

        # Filter valid rows
        valid_data = summary_df[
            (summary_df[timestamp_col].notna()) &
            (summary_df[peak_vel_col].notna()) &
            (summary_df[laser_energy_col].notna())
        ].copy()

        # Convert peak velocity to numeric
        valid_data[peak_vel_col] = pd.to_numeric(valid_data[peak_vel_col], errors='coerce')
        valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
        valid_data = valid_data[valid_data[peak_vel_col].notna() & valid_data[laser_energy_col].notna()]

        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data for Peak Velocity vs Time plot")
            return

        # Parse timestamps: try datetime first, else numeric (seconds)
        timestamps = pd.to_datetime(valid_data[timestamp_col], errors='coerce')
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            # Fallback: treat as numeric (seconds since epoch or relative)
            ts_numeric = pd.to_numeric(valid_data[timestamp_col], errors='coerce')
            valid_mask = ts_numeric.notna()
            if not valid_mask.any():
                if progress_callback:
                    progress_callback("⚠ Could not parse timestamps - skipping plot")
                return
            valid_data = valid_data[valid_mask].copy()
            # Use numeric values directly as relative time (assume already in seconds)
            valid_data['time_rel_seconds'] = ts_numeric[valid_mask].values
            t_min_val = valid_data['time_rel_seconds'].min()
            valid_data['time_rel_seconds'] = valid_data['time_rel_seconds'] - t_min_val
        else:
            valid_data = valid_data[valid_mask].copy()
            timestamps = timestamps[valid_mask]
            t_min = timestamps.min()
            valid_data['time_rel_seconds'] = (timestamps - t_min).dt.total_seconds()

        # Bin energies in 300 mJ intervals
        bin_size = 300
        e_max = valid_data[laser_energy_col].max()
        e_min = valid_data[laser_energy_col].min()
        bin_start = max(0, int(np.floor(e_min / bin_size) * bin_size))
        bin_end = int(np.ceil(e_max / bin_size) * bin_size) + 1
        energy_bin_edges = np.arange(bin_start, bin_end, bin_size)

        valid_data['energy_bin'] = pd.cut(
            valid_data[laser_energy_col],
            bins=energy_bin_edges,
            include_lowest=True,
            right=True
        )
        # Create readable bin labels (e.g., "300-600 mJ")
        valid_data['energy_bin_label'] = valid_data['energy_bin'].apply(
            lambda x: f'{int(x.left)}-{int(x.right)} mJ' if pd.notna(x) else None
        )

        energy_bins = sorted(valid_data['energy_bin_label'].dropna().unique(),
                            key=lambda s: int(s.split('-')[0]) if s else 0)
        n_bins = len(energy_bins)
        cmap = plt.get_cmap('tab10' if n_bins <= 10 else 'tab20')
        bin_colors = {b: cmap(i / max(n_bins, 1)) for i, b in enumerate(energy_bins)}

        # Choose sensible time unit for x-axis
        max_rel_time = valid_data['time_rel_seconds'].max()
        if max_rel_time >= 3600:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds'] / 3600
            time_label = 'Relative Time (hours)'
        elif max_rel_time >= 60:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds'] / 60
            time_label = 'Relative Time (minutes)'
        else:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds']
            time_label = 'Relative Time (seconds)'

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        n_time_bins = min(10, max(3, len(valid_data) // 5))

        for bin_label in energy_bins:
            subset = valid_data[valid_data['energy_bin_label'] == bin_label].copy()
            ax.scatter(
                subset['time_rel_display'],
                subset[peak_vel_col],
                c=[bin_colors[bin_label]],
                label=f'{bin_label} (n={len(subset)})',
                s=80,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

            # Piecewise linear trendline: bin time, compute mean velocity per bin, connect
            if len(subset) >= 2:
                time_edges = np.linspace(
                    subset['time_rel_display'].min(),
                    subset['time_rel_display'].max(),
                    n_time_bins + 1
                )
                subset['time_bin'] = pd.cut(
                    subset['time_rel_display'],
                    bins=time_edges,
                    include_lowest=True,
                    right=True
                )
                trend = subset.groupby('time_bin', observed=True).agg(
                    t_mean=('time_rel_display', 'mean'),
                    v_mean=(peak_vel_col, 'mean')
                ).reset_index()
                trend = trend.dropna()
                if len(trend) >= 2:
                    trend = trend.sort_values('t_mean')
                    ax.plot(
                        trend['t_mean'],
                        trend['v_mean'],
                        linestyle='--',
                        color=bin_colors[bin_label],
                        linewidth=2.5,
                        zorder=1
                    )

        ax.set_xlabel(time_label)
        ax.set_ylabel('Peak Velocity (m/s)')
        ax.set_title('Peak Velocity vs Time (First Shot = t=0)')
        ax.legend(title='Laser Energy (300 mJ bins)', loc='best')

        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'peak_velocity_vs_time.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if progress_callback:
            progress_callback(f"✅ Generated Peak Velocity vs Time plot: {plot_path}")

    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Peak Velocity vs Time plot: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_laser_energy_stability_table(summary_df, spade_output_dir, progress_callback=None):
    """
    Generate a table of shot-to-shot laser energy stability (1σ as % of mean) for each 300 mJ energy bin.
    Output: laser_energy_stability.csv with Energy Bin, Mean (mJ), Std (mJ), Stability ±X.X% (1σ), Count.
    """
    if progress_callback:
        progress_callback("Generating laser energy stability table...")

    try:
        laser_energy_col = find_column_name(
            summary_df,
            ['Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 'Laser_Ref_Energy (mJ)',
             'Laser energy (mJ)', 'Laser Energy (mJ)', 'laser_energy', 'Energy (mJ)'],
            progress_callback
        )

        if laser_energy_col is None:
            if progress_callback:
                progress_callback("⚠ Laser energy column not found - skipping stability table")
            return

        valid_data = summary_df[summary_df[laser_energy_col].notna()].copy()
        valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
        valid_data = valid_data[valid_data[laser_energy_col].notna()]

        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid laser energy data - skipping stability table")
            return

        # Bin energies in 300 mJ intervals
        bin_size = 300
        e_max = valid_data[laser_energy_col].max()
        e_min = valid_data[laser_energy_col].min()
        bin_start = max(0, int(np.floor(e_min / bin_size) * bin_size))
        bin_end = int(np.ceil(e_max / bin_size) * bin_size) + 1
        energy_bin_edges = np.arange(bin_start, bin_end, bin_size)

        valid_data['energy_bin'] = pd.cut(
            valid_data[laser_energy_col],
            bins=energy_bin_edges,
            include_lowest=True,
            right=True
        )
        valid_data['energy_bin_label'] = valid_data['energy_bin'].apply(
            lambda x: f'{int(x.left)}-{int(x.right)} mJ' if pd.notna(x) else None
        )

        energy_bins = sorted(valid_data['energy_bin_label'].dropna().unique(),
                            key=lambda s: int(s.split('-')[0]) if s else 0)

        table_data = []
        for bin_label in energy_bins:
            subset = valid_data[valid_data['energy_bin_label'] == bin_label][laser_energy_col]
            n = len(subset)
            mean_e = subset.mean()
            std_e = subset.std()

            if n >= 2 and mean_e > 0:
                if pd.notna(std_e) and std_e > 0:
                    stability_pct = 100.0 * std_e / mean_e
                    stability_str = f"±{stability_pct:.1f}%"
                else:
                    stability_pct = 0.0
                    stability_str = "±0.0%"  # All shots same energy
            else:
                stability_pct = np.nan
                stability_str = "N/A" if n < 2 else "±0.0%"

            table_data.append({
                'Energy Bin': bin_label,
                'Mean (mJ)': round(mean_e, 2),
                'Std (mJ)': round(std_e, 2) if pd.notna(std_e) else np.nan,
                'Stability ±% (1σ)': stability_str,
                'Stability_pct': round(stability_pct, 2) if pd.notna(stability_pct) else np.nan,
                'Count': n
            })

        table_df = pd.DataFrame(table_data)
        out_path = os.path.join(spade_output_dir, 'laser_energy_stability.csv')
        table_df.to_csv(out_path, index=False)
        if progress_callback:
            progress_callback(f"✅ Saved laser energy stability table: {out_path}")
            for row in table_data:
                progress_callback(f"   {row['Energy Bin']}: {row['Stability ±% (1σ)']} (1σ), n={row['Count']}")

    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating laser energy stability table: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_laser_energy_vs_time_plot(summary_df, spade_output_dir, progress_callback=None):
    """
    Generate Laser Energy vs Time plot with time relative to first shot.
    X-axis: Relative time (first shot = 0)
    Y-axis: Laser energy (mJ)
    Points colored by laser energy bin (300 mJ intervals).
    Dashed trendlines show piecewise-linear mean energy vs time for each bin.
    """
    if progress_callback:
        progress_callback("Generating Laser Energy vs Time (relative) plot...")

    try:
        # Find timestamp column
        timestamp_col = find_column_name(
            summary_df,
            ['Timestamp', 'timestamp', 'Shot_Time', 'Shot Time', 'Exp_Time', 'Exp_Time (seconds)',
             'Time (seconds)', 'time', 'Shot_time'],
            progress_callback
        )

        # Find laser energy column
        laser_energy_col = find_column_name(
            summary_df,
            ['Laser_Target_Energy (mJ)', 'Laser Target Energy (mJ)', 'Laser_Ref_Energy (mJ)',
             'Laser energy (mJ)', 'Laser Energy (mJ)', 'laser_energy', 'Energy (mJ)'],
            progress_callback
        )

        if timestamp_col is None:
            if progress_callback:
                progress_callback("⚠ Timestamp column not found - skipping Laser Energy vs Time plot")
            return

        if laser_energy_col is None:
            if progress_callback:
                progress_callback("⚠ Laser energy column not found - skipping Laser Energy vs Time plot")
            return

        # Filter valid rows
        valid_data = summary_df[
            (summary_df[timestamp_col].notna()) &
            (summary_df[laser_energy_col].notna())
        ].copy()

        # Convert laser energy to numeric
        valid_data[laser_energy_col] = pd.to_numeric(valid_data[laser_energy_col], errors='coerce')
        valid_data = valid_data[valid_data[laser_energy_col].notna()]

        if len(valid_data) == 0:
            if progress_callback:
                progress_callback("⚠ No valid data for Laser Energy vs Time plot")
            return

        # Parse timestamps: try datetime first, else numeric (seconds)
        timestamps = pd.to_datetime(valid_data[timestamp_col], errors='coerce')
        valid_mask = timestamps.notna()
        if not valid_mask.any():
            # Fallback: treat as numeric (seconds since epoch or relative)
            ts_numeric = pd.to_numeric(valid_data[timestamp_col], errors='coerce')
            valid_mask = ts_numeric.notna()
            if not valid_mask.any():
                if progress_callback:
                    progress_callback("⚠ Could not parse timestamps - skipping plot")
                return
            valid_data = valid_data[valid_mask].copy()
            valid_data['time_rel_seconds'] = ts_numeric[valid_mask].values
            t_min_val = valid_data['time_rel_seconds'].min()
            valid_data['time_rel_seconds'] = valid_data['time_rel_seconds'] - t_min_val
        else:
            valid_data = valid_data[valid_mask].copy()
            timestamps = timestamps[valid_mask]
            t_min = timestamps.min()
            valid_data['time_rel_seconds'] = (timestamps - t_min).dt.total_seconds()

        # Bin energies in 300 mJ intervals
        bin_size = 300
        e_max = valid_data[laser_energy_col].max()
        e_min = valid_data[laser_energy_col].min()
        bin_start = max(0, int(np.floor(e_min / bin_size) * bin_size))
        bin_end = int(np.ceil(e_max / bin_size) * bin_size) + 1
        energy_bin_edges = np.arange(bin_start, bin_end, bin_size)

        valid_data['energy_bin'] = pd.cut(
            valid_data[laser_energy_col],
            bins=energy_bin_edges,
            include_lowest=True,
            right=True
        )
        valid_data['energy_bin_label'] = valid_data['energy_bin'].apply(
            lambda x: f'{int(x.left)}-{int(x.right)} mJ' if pd.notna(x) else None
        )

        energy_bins = sorted(valid_data['energy_bin_label'].dropna().unique(),
                            key=lambda s: int(s.split('-')[0]) if s else 0)
        n_bins = len(energy_bins)
        cmap = plt.get_cmap('tab10' if n_bins <= 10 else 'tab20')
        bin_colors = {b: cmap(i / max(n_bins, 1)) for i, b in enumerate(energy_bins)}

        # Choose sensible time unit for x-axis
        max_rel_time = valid_data['time_rel_seconds'].max()
        if max_rel_time >= 3600:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds'] / 3600
            time_label = 'Relative Time (hours)'
        elif max_rel_time >= 60:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds'] / 60
            time_label = 'Relative Time (minutes)'
        else:
            valid_data['time_rel_display'] = valid_data['time_rel_seconds']
            time_label = 'Relative Time (seconds)'

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        n_time_bins = min(10, max(3, len(valid_data) // 5))

        for bin_label in energy_bins:
            subset = valid_data[valid_data['energy_bin_label'] == bin_label].copy()
            ax.scatter(
                subset['time_rel_display'],
                subset[laser_energy_col],
                c=[bin_colors[bin_label]],
                label=f'{bin_label} (n={len(subset)})',
                s=80,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )

            # Piecewise linear trendline: bin time, compute mean energy per bin, connect
            if len(subset) >= 2:
                time_edges = np.linspace(
                    subset['time_rel_display'].min(),
                    subset['time_rel_display'].max(),
                    n_time_bins + 1
                )
                subset['time_bin'] = pd.cut(
                    subset['time_rel_display'],
                    bins=time_edges,
                    include_lowest=True,
                    right=True
                )
                trend = subset.groupby('time_bin', observed=True).agg(
                    t_mean=('time_rel_display', 'mean'),
                    e_mean=(laser_energy_col, 'mean')
                ).reset_index()
                trend = trend.dropna()
                if len(trend) >= 2:
                    trend = trend.sort_values('t_mean')
                    ax.plot(
                        trend['t_mean'],
                        trend['e_mean'],
                        linestyle='--',
                        color=bin_colors[bin_label],
                        linewidth=2.5,
                        zorder=1
                    )

        ax.set_xlabel(time_label)
        ax.set_ylabel('Laser Energy (mJ)')
        ax.set_title('Laser Energy vs Time (First Shot = t=0)')
        ax.legend(title='Laser Energy (300 mJ bins)', loc='best')

        plt.tight_layout()
        plot_path = os.path.join(spade_output_dir, 'laser_energy_vs_time.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if progress_callback:
            progress_callback(f"✅ Generated Laser Energy vs Time plot: {plot_path}")

    except Exception as e:
        if progress_callback:
            progress_callback(f"Error generating Laser Energy vs Time plot: {str(e)}")
            import traceback
            progress_callback(f"Traceback: {traceback.format_exc()}")


def generate_all_plots_from_summary_files(enhanced_spall_summary_path=None, velocity_shots_summary_path=None, 
                                         output_dir=None, progress_callback=None):
    """
    Generate all available plots from summary CSV files.
    
    Args:
        enhanced_spall_summary_path: Path to enhanced_spall_summary.csv (for spall plots)
        velocity_shots_summary_path: Path to velocity_shots_summary.csv (for HEL/shock stress plots)
        output_dir: Directory to save plots
        progress_callback: Optional callback function for progress messages
    """
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Generating all plots from summary files...")
        progress_callback("=" * 60)
    
    plot_count = 0
    error_count = 0
    
    # Load enhanced spall summary if available
    enhanced_spall_df = None
    if enhanced_spall_summary_path and os.path.exists(enhanced_spall_summary_path):
        try:
            enhanced_spall_df = pd.read_csv(enhanced_spall_summary_path)
            if progress_callback:
                progress_callback(f"✅ Loaded enhanced_spall_summary.csv: {len(enhanced_spall_df)} rows")
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠ Failed to load enhanced_spall_summary.csv: {e}")
    
    # Load velocity shots summary if available
    velocity_shots_df = None
    if velocity_shots_summary_path and os.path.exists(velocity_shots_summary_path):
        try:
            velocity_shots_df = pd.read_csv(velocity_shots_summary_path)
            if progress_callback:
                progress_callback(f"✅ Loaded velocity_shots_summary.csv: {len(velocity_shots_df)} rows")
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠ Failed to load velocity_shots_summary.csv: {e}")
    
    # Generate spall plots from enhanced_spall_summary
    if enhanced_spall_df is not None:
        if progress_callback:
            progress_callback("\n--- Generating Spall Analysis Plots ---")
        
        # Plot 1: Spall vs Strain Rate
        try:
            generate_spall_vs_strain_rate_plot(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")
        
        # Plot 2: Spall vs Strain Rate by Material (Cu/Al subplots)
        try:
            generate_spall_vs_strain_rate_by_material_subplots(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")
        
        # Plot 3: Spall vs Shock Stress
        try:
            generate_spall_vs_shock_stress_plot(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")
        
        # Plot 4: Spall vs Shock Stress by Material (Cu/Al subplots with flyer thickness)
        try:
            generate_spall_vs_shock_stress_by_material_subplots(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")

        # Plot 5: Peak Velocity vs Time (relative), colored by laser energy
        try:
            generate_peak_velocity_vs_time_plot(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")

        # Plot 6: Laser Energy vs Time (relative), colored by 300 mJ energy bins
        try:
            generate_laser_energy_vs_time_plot(enhanced_spall_df, output_dir, progress_callback)
            plot_count += 1
        except Exception as e:
            error_count += 1
            if progress_callback:
                progress_callback(f"❌ Error: {e}")

        # Table: Shot-to-shot laser energy stability (±% 1σ) per energy bin
        try:
            generate_laser_energy_stability_table(enhanced_spall_df, output_dir, progress_callback)
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠ Laser energy stability table: {e}")

    # Generate velocity/shock stress plots from velocity_shots_summary
    if velocity_shots_df is not None:
        if progress_callback:
            progress_callback("\n--- Generating Velocity/Shock Stress Plots ---")
        
        # Import plotting functions that read from velocity_shots_summary
        # These will be added as standalone functions
        # For now, we'll generate what we can from the data
        
        if progress_callback:
            progress_callback("Note: Additional plots from velocity_shots_summary.csv can be added here")
    
    if progress_callback:
        progress_callback("\n" + "=" * 60)
        progress_callback(f"Plot generation complete!")
        progress_callback(f"  Successfully generated: {plot_count} plots")
        if error_count > 0:
            progress_callback(f"  Errors encountered: {error_count} plots")
        progress_callback(f"  Output directory: {output_dir}")
        progress_callback("=" * 60)
