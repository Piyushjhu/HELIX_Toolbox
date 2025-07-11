# spall_analysis/utils.py
"""
Utility functions and constants for the spall_analysis package.
"""
import os
import pandas as pd
import numpy as np
import glob
import logging

# Default material properties
DENSITY = None  # kg/m^3
ACOUSTIC_VELOCITY = None # m/s

# Reference values for Copper
DENSITY_COPPER = 8960.0 # kg/m^3
ACOUSTIC_VELOCITY_COPPER = 3940.0 # m/s

# Material type mappings
MATERIAL_MAPPING = {
    "4um": "Poly",
    "100nm": "Nano",
    "SC100": "SC [100]",
    "SC110": "SC [110]",
    "SC111": "SC [111]",
}

# Energy to velocity mappings
ENERGY_VELOCITY_MAPPING = {
    "600mJ": "408 m/s",
    "800mJ": "550 m/s",
    "1000mJ": "600 m/s",
    "1200mJ": "730 m/s",
    "1350mJ": "650 m/s",
    "1500mJ": "830 m/s",
    "1600mJ": "850 m/s",
    "1700mJ": "870 m/s"
}

# Color mappings for plotting
COLOR_MAPPING = {
    "Nano": "r",
    "Poly": "c",
    "SC [100]": "m",
    "SC [110]": "g",
    "SC [111]": "b",
    "Default": "gray",
    "Kanel": "#1f77b4",
    "Moshe": "#ff7f0e",
    "Chen": "#2ca02c",
    "Wilkerson": "#9467bd",
    "Priyadarshan": "#8c564b",
    "Arad et al. (poly)": "#e377c2",
    "Escobedo et al. (poly)": "#7f7f7f",
    "Fortov et al.": "#bcbd22",
    "G. Kanel et al. (poly)": "#17becf",
    "Minich et al. (single Crystal)": "#aec7e8",
    "Mukherjee et al. (poly)": "#ffbb78",
    "Ogoronikov et al. (poly)": "#98df8a",
    "Paisley et al. (poly)": "#ff9896",
    "Peralta et al. (poly)": "#c5b0d5",
    "T.Chen et al. (poly)": "#c49c94",
    "Turney et al. (Single Crystal)": "#f7b6d2",
    "Yong-Gang et al. (nano)": "#dbdb8d",
    "Poly (4um)": "#2ca02c",
    "Nano (100nm)": "#d62728",
    "Single Crystal (Est. dG=1mm)": "#9467bd",
}

# Marker mappings for plotting
MARKER_MAPPING = {
    "Nano": "s",
    "Poly": "o",
    "SC [100]": "D",
    "SC [110]": "P",
    "SC [111]": "X",
    "Default": "x",
    "Kanel": "v",
    "Moshe": "D",
    "Chen": "p",
    "Wilkerson": "h",
    "Priyadarshan": "*",
    "Arad et al. (poly)": "d",
    "Escobedo et al. (poly)": "<",
    "Fortov et al.": ">",
    "G. Kanel et al. (poly)": "1",
    "Minich et al. (single Crystal)": "2",
    "Mukherjee et al. (poly)": "3",
    "Ogoronikov et al. (poly)": "4",
    "Paisley et al. (poly)": "8",
    "Peralta et al. (poly)": "P",
    "T.Chen et al. (poly)": "X",
    "Turney et al. (Single Crystal)": "H",
    "Yong-Gang et al. (nano)": "+",
}

def find_data_files(input_dir, file_pattern):
    """Find files matching a pattern in a directory."""
    search_path = os.path.join(input_dir, file_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        logging.warning(f"No files found matching '{search_path}'")
    else:
        logging.info(f"Found {len(files)} files matching '{search_path}'")
    return files

def extract_legend_info(filename, material_map=MATERIAL_MAPPING, energy_map=ENERGY_VELOCITY_MAPPING):
    """
    Extract material type and velocity label from filename.
    
    Returns:
        tuple: (material_type, velocity_label, energy_key)
    """
    if material_map is None: material_map = {}
    if energy_map is None: energy_map = {}

    base = os.path.basename(filename).replace('.csv', '')
    parts = base.split('_')

    material_type = None
    velocity_label = None
    energy_key = None

    # Find material type
    for prefix, mat_type in material_map.items():
        if base.startswith(prefix):
            material_type = mat_type
            break

    # Find energy/velocity
    for energy, vel_label in energy_map.items():
        if energy in parts:
            velocity_label = vel_label
            energy_key = energy
            break
        for part in parts:
            if energy in part:
                velocity_label = vel_label
                energy_key = energy
                break
        if velocity_label is not None:
            break

    if material_type is None:
        logging.warning(f"Could not parse material type from '{base}'")
    if velocity_label is None:
        logging.warning(f"Could not parse energy/velocity label from '{base}'")

    return material_type, velocity_label, energy_key

def calculate_shock_stress(velocity, density, acoustic_velocity):
    """Calculate Hugoniot shock stress (GPa)."""
    if pd.isna(velocity) or density <= 0 or acoustic_velocity <= 0:
        return np.nan
    return 0.5 * density * acoustic_velocity * velocity * 1e-9

def add_shock_stress_column(df, velocity_col, density, acoustic_velocity, new_col_name='Shock Stress (GPa)'):
    """Add shock stress column to DataFrame."""
    if velocity_col not in df.columns:
        logging.error(f"Velocity column '{velocity_col}' not found in DataFrame.")
        return df
    df[new_col_name] = df[velocity_col].apply(
        lambda v: calculate_shock_stress(v, density, acoustic_velocity)
    )
    return df

# --- Add other utility functions from previous versions if needed ---

