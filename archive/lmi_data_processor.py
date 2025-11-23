"""
LMI Data Processor
Processes CSV/XLSX files from LMI experiments and generates analysis plots.

File naming convention: LMI_YYYYMMDD_IGSN
Example: LMI_20251023_JHAMAB00019-06
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class LMIDataProcessor:
    """Process LMI experiment files and generate analysis plots."""
    
    def __init__(self, input_folder: str, output_folder: str):
        """
        Initialize the processor.
        
        Parameters:
        -----------
        input_folder : str
            Path to folder containing CSV/XLSX files
        output_folder : str
            Path to folder where output files will be saved
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Storage for processed data
        self.file_data = []
        self.sample_data = defaultdict(list)
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Parse filename to extract experiment type, date, and IGSN.
        
        Expected format: LMI_YYYYMMDD_IGSN
        Example: LMI_20251023_JHAMAB00019-06
        
        Parameters:
        -----------
        filename : str
            Name of the file (without extension)
            
        Returns:
        --------
        dict : Dictionary with 'experiment_type', 'date', 'igsn'
               Returns None if filename doesn't match expected pattern
        """
        # Remove extension if present
        filename = os.path.splitext(filename)[0]
        
        # Pattern: TYPE_YYYYMMDD_IGSN
        pattern = r'^([A-Z]+)_(\d{8})_(.+)$'
        match = re.match(pattern, filename)
        
        if match:
            experiment_type = match.group(1)
            date = match.group(2)
            igsn = match.group(3)
            
            return {
                'experiment_type': experiment_type,
                'date': date,
                'igsn': igsn,
                'filename': filename
            }
        
        return None
    
    def find_wave_plate_angle_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column containing wave plate angle data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to search
            
        Returns:
        --------
        str : Name of the wave plate angle column, or None if not found
        """
        # Common column name variations for wave plate angle
        possible_names = [
            'wave plate angle', 'wave_plate_angle', 'waveplate angle', 'waveplate_angle',
            'wave plate', 'wave_plate', 'waveplate',
            'angle', 'plate angle', 'plate_angle',
            'wp angle', 'wp_angle', 'wpa'
        ]
        
        # Check for exact matches (case-insensitive)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for name in possible_names:
                if col_lower == name or col_lower.replace(' ', '_') == name:
                    return col
        
        # Check for partial matches
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(term in col_lower for term in ['wave', 'plate', 'angle']) and 'angle' in col_lower:
                return col
        
        return None
    
    def find_pdv_filename_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column containing PDV_FileName data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to search
            
        Returns:
        --------
        str : Name of the PDV_FileName column, or None if not found
        """
        # Normalize column names (remove spaces, underscores, dashes, case-insensitive)
        normalized_columns = {
            col: re.sub(r"[^a-z0-9]", "", str(col).lower().strip()) 
            for col in df.columns
        }
        
        # Known variants of PDV filename column
        known_variants = [
            'pdvfilename', 'pdvfile', 'pdv_file', 'pdv_file_name', 'pdv file name',
            'dvfilename', 'dv_file', 'dvfile', 'filename', 'file_name', 'file name'
        ]
        normalized_variants = {re.sub(r"[^a-z0-9]", "", v): v for v in known_variants}
        
        # First pass: exact matches
        for col, norm in normalized_columns.items():
            if norm in normalized_variants:
                return col
        
        # Second pass: heuristic - contains 'pdv' and ('file' or 'name')
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if ('pdv' in col_lower or 'dv' in col_lower) and ('file' in col_lower or 'name' in col_lower):
                return col
        
        # Final fallback: standalone 'filename' or 'file name' column
        for col in df.columns:
            if str(col).strip().lower() in ['filename', 'file name', 'file_name']:
                return col
        
        return None
    
    def find_laser_target_energy_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column containing Laser_Target_Energy data.
        Prioritizes exact match for 'Laser_Target_Energy (mJ)'.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to search
            
        Returns:
        --------
        str : Name of the Laser_Target_Energy column, or None if not found
        """
        # First priority: exact match for "Laser_Target_Energy (mJ)" (case-insensitive)
        for col in df.columns:
            col_str = str(col).strip()
            if col_str.lower() == 'laser_target_energy (mj)':
                return col
        
        # Second priority: check for columns containing "Laser_Target_Energy" (with or without units)
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            # Match "Laser_Target_Energy" with optional units like "(mJ)", "(MJ)", etc.
            if re.match(r'laser[_\s]?target[_\s]?energy', col_lower):
                return col
        
        # Third priority: check for 'laser' and 'target' and 'energy' together
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'laser' in col_lower and 'target' in col_lower and 'energy' in col_lower:
                return col
        
        # Fourth priority: check for 'laser' and 'energy' together
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'laser' in col_lower and 'energy' in col_lower:
                return col
        
        # Fifth priority: check for 'target' and 'energy' together
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'target' in col_lower and 'energy' in col_lower:
                return col
        
        return None
    
    def find_sample_material_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column containing sample material data (Cu, Zn, Brass, etc.).
        Prioritizes 'Sample material' over 'Flyer_material'.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to search
            
        Returns:
        --------
        str : Name of the sample material column, or None if not found
        """
        # First priority: Look for 'Sample material' (with space) - exact match
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower == 'sample material':
                return col
        
        # Second priority: Look for columns with both 'sample' and 'material'
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'sample' in col_lower and 'material' in col_lower:
                return col
        
        # Third priority: Normalize and check for sample material variants
        normalized_columns = {
            col: re.sub(r"[^a-z0-9]", "", str(col).lower().strip()) 
            for col in df.columns
        }
        
        # Known variants of sample material column (normalized) - prioritize sample over flyer
        sample_variants = ['samplematerial', 'sample_material']
        for variant in sample_variants:
            for col, norm in normalized_columns.items():
                if norm == variant:
                    return col
        
        # Fourth priority: Just 'material' (without sample/flyer prefix)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower == 'material':
                return col
        
        # Last resort: Flyer_material (only if nothing else found)
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'flyer' in col_lower and 'material' in col_lower:
                return col
        
        return None
    
    def load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a CSV or XLSX file.
        
        Parameters:
        -----------
        file_path : Path
            Path to the file
            
        Returns:
        --------
        pd.DataFrame : Loaded data, or None if loading fails
        """
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return None
            
            return df
        except Exception as e:
            print(f"Error loading file {file_path.name}: {str(e)}")
            return None
    
    def process_files(self) -> Dict[str, List]:
        """
        Process all CSV/XLSX files in the input folder.
        
        Returns:
        --------
        dict : Dictionary with processed data organized by sample material
        """
        print(f"Scanning folder: {self.input_folder}")
        
        # Find all CSV and XLSX files
        csv_files = list(self.input_folder.glob('*.csv'))
        xlsx_files = list(self.input_folder.glob('*.xlsx'))
        xls_files = list(self.input_folder.glob('*.xls'))
        
        all_files = csv_files + xlsx_files + xls_files
        
        if not all_files:
            print(f"No CSV/XLSX files found in {self.input_folder}")
            return {}
        
        print(f"Found {len(all_files)} files to process")
        
        # Process each file
        for file_path in all_files:
            filename = file_path.stem  # filename without extension
            
            # Parse filename
            file_info = self.parse_filename(filename)
            if not file_info:
                print(f"Skipping {filename}: doesn't match expected naming pattern")
                continue
            
            # Load file
            df = self.load_file(file_path)
            if df is None or df.empty:
                print(f"Skipping {filename}: could not load or file is empty")
                continue
            
            # Find wave plate angle column
            angle_col = self.find_wave_plate_angle_column(df)
            if angle_col is None:
                print(f"Warning: Could not find wave plate angle column in {filename}")
                print(f"  Available columns: {list(df.columns)}")
                # Store file info anyway for reference
                file_info['wave_plate_angles'] = []
                file_info['dataframe'] = df
                self.file_data.append(file_info)
                continue
            
            # Find PDV_FileName column
            pdv_col = self.find_pdv_filename_column(df)
            if pdv_col is None:
                print(f"Warning: Could not find PDV_FileName column in {filename}")
                print(f"  Available columns: {list(df.columns)}")
                # Store file info anyway for reference
                file_info['wave_plate_angles'] = []
                file_info['dataframe'] = df
                self.file_data.append(file_info)
                continue
            
            # Find sample material column
            material_col = self.find_sample_material_column(df)
            if material_col is None:
                print(f"Warning: Could not find sample material column in {filename}")
                print(f"  Available columns: {list(df.columns)}")
                print(f"  Will use 'Unknown' as material")
                # Continue anyway, we'll use 'Unknown' as material
                material_col = None
            else:
                print(f"Found sample material column: '{material_col}' in {filename}")
                # Debug: Show some sample material values (filter out empty/blank values)
                sample_materials_raw = df[material_col].dropna().unique()[:10]
                sample_materials = []
                for mat in sample_materials_raw:
                    mat_str = str(mat).strip()
                    if (mat_str and 
                        mat_str.lower() not in ['nan', 'none', '', '[]'] and 
                        mat_str != '[]' and 
                        len(mat_str) > 0):
                        sample_materials.append(mat_str)
                if sample_materials:
                    print(f"  Sample material values found: {sample_materials[:5]}")
                else:
                    print(f"  WARNING: No valid material values found in '{material_col}' column")
                    print(f"  Raw values (first 5): {[str(x) for x in sample_materials_raw[:5]]}")
            
            # Find Laser_Target_Energy column
            energy_col = self.find_laser_target_energy_column(df)
            if energy_col is None:
                print(f"Warning: Could not find Laser_Target_Energy column in {filename}")
                print(f"  Available columns: {list(df.columns)}")
                energy_col = None
            else:
                print(f"Found Laser_Target_Energy column: '{energy_col}' in {filename}")
                # Debug: Show some sample energy values
                if energy_col in df.columns:
                    try:
                        sample_energies = df[energy_col].dropna().unique()[:5]
                        valid_energies = []
                        for e in sample_energies:
                            try:
                                valid_energies.append(float(e))
                            except (ValueError, TypeError):
                                pass
                        if valid_energies:
                            print(f"  Sample energy values: {valid_energies}")
                    except Exception:
                        pass
            
            # Create mapping: (angle, material) -> count of rows with non-blank PDV_FileName
            # Also create mapping: (energy, material) -> count of rows with non-blank PDV_FileName
            # Each row with a non-blank PDV_FileName value = 1 experiment
            # Group by both angle AND material from 'Sample material' column
            angle_material_experiment_count = defaultdict(int)  # (angle, material) -> count
            angle_material_pdv_list = defaultdict(list)  # (angle, material) -> [pdv_filenames]
            energy_material_experiment_count = defaultdict(int)  # (energy, material) -> count
            energy_material_pdv_list = defaultdict(list)  # (energy, material) -> [pdv_filenames]
            
            for _, row in df.iterrows():
                angle_val = row[angle_col]
                pdv_val = row[pdv_col]
                material_val = row[material_col] if material_col else None
                energy_val = row[energy_col] if energy_col else None
                
                # Skip rows with missing wave plate angle
                if pd.isna(angle_val):
                    continue
                
                # Get material from this row's 'Sample material' column
                if material_col and not pd.isna(material_val):
                    material_str = str(material_val).strip()
                    # Check for empty values, including string representation of empty lists
                    if (material_str.lower() in ['nan', 'none', '', '[]', 'nan', 'none'] or 
                        material_str == '[]' or 
                        len(material_str) == 0):
                        material_str = 'Unknown'
                else:
                    material_str = 'Unknown'
                
                try:
                    angle_val = float(angle_val)
                    
                    # Get energy value if available
                    energy_val_float = None
                    if energy_col and not pd.isna(energy_val):
                        try:
                            energy_val_float = float(energy_val)
                        except (ValueError, TypeError):
                            pass
                    
                    # Check if PDV_FileName is non-blank (this row = 1 experiment)
                    pdv_val_str = str(pdv_val).strip() if not pd.isna(pdv_val) else ''
                    is_blank = (pd.isna(pdv_val) or 
                               pdv_val_str == '' or 
                               pdv_val_str.lower() in ['nan', 'none', ''])
                    
                    if not is_blank:
                        # This row has a non-blank PDV_FileName = 1 experiment
                        # Group by (angle, material) tuple
                        key_angle = (angle_val, material_str)
                        angle_material_experiment_count[key_angle] += 1
                        angle_material_pdv_list[key_angle].append(pdv_val_str)
                        
                        # Also group by (energy, material) if energy is available
                        if energy_val_float is not None:
                            key_energy = (energy_val_float, material_str)
                            energy_material_experiment_count[key_energy] += 1
                            energy_material_pdv_list[key_energy].append(pdv_val_str)
                except (ValueError, TypeError):
                    continue
            
            # Store data organized by material
            # Convert (angle, material) -> count to material -> {angle: count}
            material_angle_data = defaultdict(lambda: {'counts': defaultdict(int), 'pdv_lists': defaultdict(list)})
            
            for (angle, material), count in angle_material_experiment_count.items():
                material_angle_data[material]['counts'][angle] = count
                material_angle_data[material]['pdv_lists'][angle] = angle_material_pdv_list[(angle, material)]
            
            # Store data organized by material for energy
            # Convert (energy, material) -> count to material -> {energy: count}
            material_energy_data = defaultdict(lambda: {'counts': defaultdict(int), 'pdv_lists': defaultdict(list)})
            
            for (energy, material), count in energy_material_experiment_count.items():
                material_energy_data[material]['counts'][energy] = count
                material_energy_data[material]['pdv_lists'][energy] = energy_material_pdv_list[(energy, material)]
            
            # Store file info with material-specific data
            file_info['material_angle_data'] = {
                material: {
                    'angle_experiment_count': dict(data['counts']),
                    'angle_pdv_list': {angle: list(pdv_list) for angle, pdv_list in data['pdv_lists'].items()}
                }
                for material, data in material_angle_data.items()
            }
            file_info['material_energy_data'] = {
                material: {
                    'energy_experiment_count': dict(data['counts']),
                    'energy_pdv_list': {energy: list(pdv_list) for energy, pdv_list in data['pdv_lists'].items()}
                }
                for material, data in material_energy_data.items()
            }
            file_info['angle_column'] = angle_col
            file_info['energy_column'] = energy_col
            file_info['pdv_column'] = pdv_col
            file_info['material_column'] = material_col
            file_info['dataframe'] = df
            
            # If no material data was found (no experiments), still store the file info
            if not material_angle_data:
                # Try to get materials from the dataframe directly for summary purposes
                if material_col:
                    unique_materials = df[material_col].dropna().unique()
                    unique_materials = [str(m).strip() for m in unique_materials 
                                      if str(m).strip().lower() not in ['nan', 'none', '']]
                    if unique_materials:
                        # Store with the first material found (even if no experiments)
                        file_info['material_angle_data'] = {
                            mat: {'angle_experiment_count': {}, 'angle_pdv_list': {}}
                            for mat in unique_materials
                        }
                        file_info['material_energy_data'] = {
                            mat: {'energy_experiment_count': {}, 'energy_pdv_list': {}}
                            for mat in unique_materials
                        }
                    else:
                        file_info['material_angle_data'] = {'Unknown': {'angle_experiment_count': {}, 'angle_pdv_list': {}}}
                        file_info['material_energy_data'] = {'Unknown': {'energy_experiment_count': {}, 'energy_pdv_list': {}}}
                else:
                    file_info['material_angle_data'] = {'Unknown': {'angle_experiment_count': {}, 'angle_pdv_list': {}}}
                    file_info['material_energy_data'] = {'Unknown': {'energy_experiment_count': {}, 'energy_pdv_list': {}}}
            
            # Store by each material found in this file
            for material in file_info['material_angle_data'].keys():
                # Create a copy of file_info for each material
                material_file_info = file_info.copy()
                material_file_info['sample_material'] = material
                self.sample_data[material].append(material_file_info)
            
            self.file_data.append(file_info)
            
            # Print summary for each material in this file
            # Use the stored material_angle_data (which includes materials even if no experiments)
            stored_material_data = file_info.get('material_angle_data', {})
            for material, data in stored_material_data.items():
                angle_experiment_count = data.get('angle_experiment_count', {})
                angle_counts = angle_experiment_count if angle_experiment_count else {}
                total_experiments = sum(angle_counts.values()) if angle_counts else 0
                print(f"Processed {filename} - Material '{material}': {len(angle_counts)} unique wave plate angles, {total_experiments} total experiments (rows with non-blank PDV_FileName)")
                
                # Debug: Show breakdown by angle for this material
                if len(angle_counts) <= 5 and len(angle_counts) > 0:  # Only show if not too many angles and has data
                    for angle in sorted(angle_counts.keys()):
                        count = angle_counts[angle]
                        pdv_list = data.get('angle_pdv_list', {}).get(angle, [])
                        pdv_examples = sorted(pdv_list)[:5]
                        print(f"  Angle {angle:.2f}°: {count} experiments -> {', '.join(pdv_examples)}{'...' if count > 5 else ''}")
        
        return dict(self.sample_data)
    
    def plot_wave_plate_angle_distribution(self, save_path: Optional[str] = None):
        """
        Generate a bar plot showing wave plate angle vs number of experiments.
        X-axis: wave plate angle
        Y-axis: number of experiments
        One bar for each material at each wave plate angle.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, saves to output_folder
        """
        if not self.sample_data:
            print("No data to plot. Run process_files() first.")
            return
        
        # Create a single figure for bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get unique materials and assign colors
        materials = sorted(self.sample_data.keys())
        # Use a predefined list of distinct colors for materials
        # If more materials than colors, cycle through them
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = {material: base_colors[i % len(base_colors)] 
                    for i, material in enumerate(materials)}
        
        # Collect data: angle -> {material: count}
        angle_material_counts = defaultdict(lambda: defaultdict(int))
        all_angles = set()
        
        # Process each material
        for sample_material, file_list in sorted(self.sample_data.items()):
            # Collect data: wave plate angle -> number of rows with non-blank PDV_FileName
            # Sum counts across all files for the same sample material
            angle_experiment_count_total = defaultdict(int)
            angle_pdv_examples = defaultdict(list)  # Store some examples for debugging
            
            for file_info in file_list:
                # Get data for this specific material from the file
                material_data = file_info.get('material_angle_data', {}).get(sample_material, {})
                angle_experiment_count = material_data.get('angle_experiment_count', {})
                angle_pdv_list = material_data.get('angle_pdv_list', {})
                
                if not angle_experiment_count:
                    continue
                
                # Sum experiment counts for each angle
                for angle, count in angle_experiment_count.items():
                    angle_experiment_count_total[angle] += count
                    all_angles.add(angle)
                    
                    # Store some PDV_FileName examples for this angle
                    if angle in angle_pdv_list:
                        angle_pdv_examples[angle].extend(angle_pdv_list[angle][:3])  # Store up to 3 examples per file
            
            # Store counts for this material
            for angle, count in angle_experiment_count_total.items():
                angle_material_counts[angle][sample_material] = count
            
            if not angle_experiment_count_total:
                continue
            
            # Debug: Print calculation details
            print(f"\n{'='*70}")
            print(f"Sample Material: {sample_material}")
            print(f"{'='*70}")
            print(f"Wave Plate Angle -> Number of Experiments (rows with non-blank PDV_FileName)")
            print(f"Note: Each row with a non-blank PDV_FileName = 1 experiment")
            print(f"{'-'*70}")
            for angle in sorted(angle_experiment_count_total.keys()):
                count = angle_experiment_count_total[angle]
                pdv_examples = sorted(set(angle_pdv_examples[angle]))[:10]  # Show up to 10 unique examples
                print(f"  Angle {angle:.2f}°: {count} experiments")
                if len(pdv_examples) > 0:
                    if len(pdv_examples) <= 10:
                        print(f"    Examples: {', '.join(pdv_examples)}")
                    else:
                        print(f"    Examples: {', '.join(pdv_examples[:10])} ... (and {len(set(angle_pdv_examples[angle]))-10} more)")
            print(f"{'='*70}\n")
        
        if not angle_material_counts:
            ax.text(0.5, 0.5, 'No data to plot',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Wave Plate Angle vs Number of Experiments', fontsize=14, fontweight='bold')
        else:
            # Prepare data for grouped bar chart
            sorted_angles = sorted(all_angles)
            n_angles = len(sorted_angles)
            n_materials = len(materials)
            
            # Calculate bar width and positions
            # Group bars by angle, with one bar per material
            bar_width = 0.8 / n_materials if n_materials > 0 else 0.8
            x_positions = np.arange(n_angles)
            
            # Find maximum value for y-axis (10% more than highest)
            max_value = 0
            for angle in sorted_angles:
                for material in materials:
                    count = angle_material_counts[angle].get(material, 0)
                    max_value = max(max_value, count)
            
            y_max = max_value * 1.1 if max_value > 0 else 1.1
            ax.set_ylim(0, y_max)
            
            # Plot bars for each material
            for i, material in enumerate(materials):
                # Calculate x positions for this material's bars
                offset = (i - (n_materials - 1) / 2) * bar_width
                x_pos = x_positions + offset
                
                # Get counts for this material at each angle
                counts = [angle_material_counts[angle].get(material, 0) for angle in sorted_angles]
                
                # Plot bars for this material
                ax.bar(x_pos, counts,
                      width=bar_width,
                      label=material,
                      color=color_map[material],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.5)
            
            # Set x-axis labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'{angle:.1f}°' for angle in sorted_angles], rotation=45, ha='right')
            
            # Set labels and title
            ax.set_xlabel('Wave Plate Angle', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Experiments', fontsize=12, fontweight='bold')
            ax.set_title('Wave Plate Angle vs Number of Experiments\n(Grouped by Material)', 
                        fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Add legend
            ax.legend(loc='best', framealpha=0.9, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_folder / 'wave_plate_angle_distribution.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to: {save_path}")
        
        plt.close()
    
    def plot_laser_target_energy_distribution(self, save_path: Optional[str] = None):
        """
        Generate a bar plot showing Laser_Target_Energy vs number of experiments.
        X-axis: Laser_Target_Energy (mJ) grouped into 50 mJ bins
        Y-axis: number of experiments
        One bar for each material at each energy bin.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, saves to output_folder
        """
        if not self.sample_data:
            print("No data to plot. Run process_files() first.")
            return
        
        # Create a single figure for bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get unique materials and assign colors
        materials = sorted(self.sample_data.keys())
        # Use a predefined list of distinct colors for materials
        # If more materials than colors, cycle through them
        base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = {material: base_colors[i % len(base_colors)] 
                    for i, material in enumerate(materials)}
        
        # First, collect all energy values to determine the minimum for binning
        all_energy_values = []
        for sample_material, file_list in sorted(self.sample_data.items()):
            for file_info in file_list:
                material_data = file_info.get('material_energy_data', {}).get(sample_material, {})
                energy_experiment_count = material_data.get('energy_experiment_count', {})
                # Collect energy values, filtering out any None or invalid values
                for energy in energy_experiment_count.keys():
                    if energy is not None and isinstance(energy, (int, float)) and energy >= 0:
                        all_energy_values.append(float(energy))
        
        # Determine minimum energy to start binning from
        if not all_energy_values:
            ax.text(0.5, 0.5, 'No data to plot',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Laser_Target_Energy vs Number of Experiments', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_path is None:
                save_path = self.output_folder / 'laser_target_energy_distribution.png'
            else:
                save_path = Path(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Bar plot saved to: {save_path}")
            plt.close()
            return
        
        min_energy = min(all_energy_values)
        max_energy = max(all_energy_values)
        print(f"\nDEBUG: Collected {len(all_energy_values)} energy values")
        print(f"DEBUG: Sample energy values: {sorted(set(all_energy_values))[:10]}")
        print(f"Energy range in data: {min_energy:.2f} - {max_energy:.2f} mJ")
        print(f"Binning will start from minimum: {min_energy:.2f} mJ\n")
        
        # Helper function to determine energy bin (50 mJ bins starting from minimum)
        def get_energy_bin(energy):
            """Return the bin label for an energy value based on 50 mJ bins from minimum"""
            # Calculate which bin this energy falls into (starting from min_energy)
            bin_index = int((energy - min_energy) // 50)
            bin_start = min_energy + (bin_index * 50)
            bin_end = bin_start + 50
            return (bin_start, bin_end)
        
        # Collect data: energy_bin -> {material: count}
        energy_bin_material_counts = defaultdict(lambda: defaultdict(int))
        all_energy_bins = set()
        
        # Process each material
        for sample_material, file_list in sorted(self.sample_data.items()):
            # Collect data: Laser_Target_Energy -> number of rows with non-blank PDV_FileName
            # Sum counts across all files for the same sample material
            energy_experiment_count_total = defaultdict(int)
            energy_pdv_examples = defaultdict(list)  # Store some examples for debugging
            
            for file_info in file_list:
                # Get data for this specific material from the file
                material_data = file_info.get('material_energy_data', {}).get(sample_material, {})
                energy_experiment_count = material_data.get('energy_experiment_count', {})
                energy_pdv_list = material_data.get('energy_pdv_list', {})
                
                if not energy_experiment_count:
                    continue
                
                # Sum experiment counts for each energy
                for energy, count in energy_experiment_count.items():
                    energy_experiment_count_total[energy] += count
                    
                    # Store some PDV_FileName examples for this energy
                    if energy in energy_pdv_list:
                        energy_pdv_examples[energy].extend(energy_pdv_list[energy][:3])  # Store up to 3 examples per file
            
            # Group energies into bins and aggregate counts
            for energy, count in energy_experiment_count_total.items():
                energy_bin = get_energy_bin(energy)
                all_energy_bins.add(energy_bin)
                energy_bin_material_counts[energy_bin][sample_material] += count
            
            if not energy_experiment_count_total:
                continue
            
            # Debug: Print calculation details (grouped by bins)
            print(f"\n{'='*70}")
            print(f"Sample Material: {sample_material}")
            print(f"{'='*70}")
            print(f"Laser_Target_Energy (50 mJ bins) -> Number of Experiments (rows with non-blank PDV_FileName)")
            print(f"Note: Each row with a non-blank PDV_FileName = 1 experiment")
            print(f"{'-'*70}")
            
            # Group by bins for display
            bin_totals = defaultdict(int)
            for energy in sorted(energy_experiment_count_total.keys()):
                energy_bin = get_energy_bin(energy)
                bin_totals[energy_bin] += energy_experiment_count_total[energy]
            
            for energy_bin in sorted(bin_totals.keys()):
                count = bin_totals[energy_bin]
                bin_start, bin_end = energy_bin
                print(f"  Energy {bin_start}-{bin_end} mJ: {count} experiments")
            print(f"{'='*70}\n")
        
        if not energy_bin_material_counts:
            ax.text(0.5, 0.5, 'No data to plot',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Laser_Target_Energy vs Number of Experiments', fontsize=14, fontweight='bold')
        else:
            # Prepare data for grouped bar chart
            sorted_energy_bins = sorted(all_energy_bins)
            n_bins = len(sorted_energy_bins)
            n_materials = len(materials)
            
            # Calculate bar width and positions
            # Group bars by energy bin, with one bar per material
            bar_width = 0.8 / n_materials if n_materials > 0 else 0.8
            x_positions = np.arange(n_bins)
            
            # Find maximum value for y-axis (10% more than highest)
            max_value = 0
            for energy_bin in sorted_energy_bins:
                for material in materials:
                    count = energy_bin_material_counts[energy_bin].get(material, 0)
                    max_value = max(max_value, count)
            
            y_max = max_value * 1.1 if max_value > 0 else 1.1
            ax.set_ylim(0, y_max)
            
            # Plot bars for each material
            for i, material in enumerate(materials):
                # Calculate x positions for this material's bars
                offset = (i - (n_materials - 1) / 2) * bar_width
                x_pos = x_positions + offset
                
                # Get counts for this material at each energy bin
                counts = [energy_bin_material_counts[energy_bin].get(material, 0) for energy_bin in sorted_energy_bins]
                
                # Plot bars for this material
                ax.bar(x_pos, counts,
                      width=bar_width,
                      label=material,
                      color=color_map[material],
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.5)
            
            # Set x-axis labels (show bin ranges)
            ax.set_xticks(x_positions)
            bin_labels = [f'{bin_start:.1f}-{bin_end:.1f}' for bin_start, bin_end in sorted_energy_bins]
            print(f"\nEnergy bins created: {bin_labels}\n")
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')
            
            # Set labels and title
            ax.set_xlabel('Laser_Target_Energy (mJ) - 50 mJ bins', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Experiments', fontsize=12, fontweight='bold')
            ax.set_title('Laser_Target_Energy vs Number of Experiments\n(Grouped by Material, 50 mJ bins)', 
                        fontsize=14, fontweight='bold')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Add legend
            ax.legend(loc='best', framealpha=0.9, fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_folder / 'laser_target_energy_distribution.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar plot saved to: {save_path}")
        plt.close()
    
    def save_summary(self, save_path: Optional[str] = None):
        """
        Save a summary of processed files to CSV.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the summary CSV. If None, saves to output_folder
        """
        if not self.file_data:
            print("No data to save. Run process_files() first.")
            return
        
        summary_data = []
        for file_info in self.file_data:
            # Get all materials in this file
            material_angle_data = file_info.get('material_angle_data', {})
            
            if material_angle_data:
                # Create one row per material in the file
                for material, data in material_angle_data.items():
                    angle_experiment_count = data.get('angle_experiment_count', {})
                    num_angles = len(angle_experiment_count)
                    total_experiments = sum(angle_experiment_count.values())
                    angles_str = ', '.join([str(a) for a in sorted(angle_experiment_count.keys())])
                    
                    summary_data.append({
                        'Filename': file_info['filename'],
                        'Experiment_Type': file_info['experiment_type'],
                        'Date': file_info['date'],
                        'IGSN': file_info['igsn'],
                        'Sample_Material': material,
                        'Num_Wave_Plate_Angles': num_angles,
                        'Total_Experiments': total_experiments,
                        'Wave_Plate_Angles': angles_str
                    })
            else:
                # No material data found
                summary_data.append({
                    'Filename': file_info['filename'],
                    'Experiment_Type': file_info['experiment_type'],
                    'Date': file_info['date'],
                    'IGSN': file_info['igsn'],
                    'Sample_Material': 'Unknown',
                    'Num_Wave_Plate_Angles': 0,
                    'Total_Experiments': 0,
                    'Wave_Plate_Angles': 'N/A'
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        if save_path is None:
            save_path = self.output_folder / 'file_summary.csv'
        else:
            save_path = Path(save_path)
        
        df_summary.to_csv(save_path, index=False)
        print(f"Summary saved to: {save_path}")


def main():
    """
    Main function - Update these paths as needed.
    """
    # ============================================================================
    # USER-DEFINED PATHS - UPDATE THESE AS NEEDED
    # ============================================================================
    INPUT_FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/meta_data"  # Update this path
    OUTPUT_FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/meta_data_output"  # Update this path
    # ============================================================================
    
    # Check if paths are still default
    if INPUT_FOLDER == "/path/to/your/input/folder":
        print("=" * 70)
        print("ERROR: Please update INPUT_FOLDER and OUTPUT_FOLDER in the script!")
        print("=" * 70)
        print("\nExample:")
        print('  INPUT_FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/HELIX_Toolbox_v_2/input_data/LMI_files"')
        print('  OUTPUT_FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/HELIX_Toolbox_v_2/output/LMI_analysis"')
        print("=" * 70)
        return
    
    # Initialize processor
    processor = LMIDataProcessor(INPUT_FOLDER, OUTPUT_FOLDER)
    
    # Process files
    print("\n" + "=" * 70)
    print("Processing LMI experiment files...")
    print("=" * 70 + "\n")
    sample_data = processor.process_files()
    
    if not sample_data:
        print("\nNo valid files were processed. Please check:")
        print("  1. Input folder path is correct")
        print("  2. Files follow naming convention: LMI_YYYYMMDD_IGSN")
        print("  3. Files contain wave plate angle data")
        return
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating wave plate angle distribution plot...")
    print("=" * 70 + "\n")
    processor.plot_wave_plate_angle_distribution()
    
    print("\n" + "=" * 70)
    print("Generating Laser_Target_Energy distribution plot...")
    print("=" * 70 + "\n")
    processor.plot_laser_target_energy_distribution()
    
    # Save summary
    print("\n" + "=" * 70)
    print("Saving summary...")
    print("=" * 70 + "\n")
    processor.save_summary()
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print(f"Processed {len(processor.file_data)} files")
    print(f"Found {len(sample_data)} unique sample materials")
    print(f"Output saved to: {processor.output_folder}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

