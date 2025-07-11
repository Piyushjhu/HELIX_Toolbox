# Spall Analysis Toolkit (SPADE)

**Author:** Piyush Wanchoo  
**Institution:** Johns Hopkins University  
**Year:** 2025  
**GitHub:** [https://github.com/Piyushjhu/SPADE](https://github.com/Piyushjhu/SPADE)  
**Version:** 0.1.0

A Python package for processing and analyzing data from spallation experiments, particularly those involving velocity interferometry (e.g., VISAR, PDV). It provides tools to calculate spall strength, strain rate, and other relevant parameters from velocity-time traces, compare results with literature data, and visualize the findings.

## Features

* **Velocity Trace Processing:** Calculates key spall parameters (peak velocity, pullback velocity, spall strength, strain rate) from raw velocity-time data using dynamic feature detection and linear fits.
* **Data Visualization:** Generates various plots:
    * Velocity trace comparisons (with optional error bands).
    * Spall Strength vs. Strain Rate (log or linear scale, with literature comparison).
    * Spall Strength vs. Shock Stress (with literature comparison).
    * Model comparison plots (e.g., Wilkerson model).
    * Elastic Net regression surface plots.
* **Modeling:** Includes implementations for:
    * Wilkerson spall model (requires solving implicit equation).
    * Elastic Net regression for correlating spall strength with shock stress and strain rate.
* **Literature Data Integration:** Functions to load and incorporate literature data into plots.
* **Utilities:** Helper functions for file handling, unit conversions, constants, and data manipulation.
* **GUI Application:** Standalone executable with graphical user interface for easy data processing.

## Installation

### Option 1: Install as Python Package

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Piyushjhu/SPADE.git
    cd spall_analysis_package
    ```

2. **Install using pip:**
    ```bash
    # Create and activate a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    # Install the package and its dependencies
    pip install .
    ```

### Option 2: Build Standalone Executable

For users who prefer a standalone application without Python installation:

#### Prerequisites
- Python 3.7+ installed
- pip3 available

#### Build Process

1. **Navigate to the release directory:**
    ```bash
    cd spall_analysis_release
    ```

2. **Test dependencies:**
    ```bash
    python3 test_dependencies.py
    ```

3. **Build the executable:**
    ```bash
    # Option A: Standard build
    ./build_mac.sh
    
    # Option B: Comprehensive build (recommended)
    ./build_mac_comprehensive.sh
    ```

4. **Run the executable:**
    ```bash
    ./executables/SPADE_mac_comprehensive
    ```

#### Troubleshooting Build Issues

If you encounter the "no module named pandas" error or other dependency issues:

1. **Check the troubleshooting guide:** See `TROUBLESHOOTING.md` for detailed solutions
2. **Test dependencies first:** Always run `python3 test_dependencies.py` before building
3. **Use the comprehensive build script:** `./build_mac_comprehensive.sh` includes all necessary hidden imports
4. **Check console output:** Remove `--windowed` flag to see error messages during execution

Common solutions:
- Install missing dependencies: `pip3 install -r requirements.txt`
- Use the comprehensive build script for better dependency handling
- Build in a clean virtual environment for consistent results

## Dependencies

* `pandas`
* `numpy`
* `matplotlib`
* `scipy`
* `scikit-learn`
* `plotly`
* `PyQt5`
* `imageio` (Optional, for GIF generation)

## Package Structure

* `spall_analysis/`: Main package source code.
    * `data_processing.py`: Core velocity trace analysis functions.
    * `plotting.py`: Plotting functions.
    * `models.py`: Material model implementations.
    * `literature.py`: Literature data handling.
    * `utils.py`: Constants, mappings, helper functions.
* `examples/`: Example scripts demonstrating usage.
    * `spall_analysis_gui.py`: GUI application for easy data processing.
* `data/`: Contains example literature data CSV files.
* `executables/`: Standalone executables.
    * `SPADE_mac_comprehensive`: Latest Mac executable with all dependencies.
* `build_mac.sh`: Standard build script for Mac executables.
* `build_mac_comprehensive.sh`: Comprehensive build script with all dependencies.
* `test_dependencies.py`: Script to verify all dependencies are available.
* `setup.py`: Package build script.
* `requirements.txt`: List of dependencies.
* `README.md`: Main documentation.
* `TROUBLESHOOTING.md`: Comprehensive troubleshooting guide.
* `SOLUTION_SUMMARY.md`: Summary of common solutions.
* `PACKAGE_STRUCTURE.md`: Package structure documentation.

## Usage

```python
import spall_analysis as sa
import os

# Define input/output paths
base_dir = 'path/to/your/experiment/data' # CHANGE THIS
raw_velocity_dir = os.path.join(base_dir, 'raw_csv')
results_dir = os.path.join(base_dir, 'analysis_output')
lit_data_dir = os.path.join(base_dir, 'literature') # Or path to package's data dir

os.makedirs(results_dir, exist_ok=True)

# --- Example 1: Process Raw Velocity Traces ---

print("Processing velocity traces...")
# Assumes CSVs in 'raw_velocity_dir' have Time (ns) in col 0, Velocity (m/s) in col 1
# Adjust file_pattern as needed
summary_df = sa.process_velocity_files(
    input_folder=raw_velocity_dir,
    file_pattern='*.csv', # Adjust pattern if needed
    output_folder=os.path.join(results_dir, 'individual_trace_analysis'),
    plot_individual=True, # Generate plot for each trace
    density=sa.DENSITY_COPPER, # Use default copper density or specify another
    acoustic_velocity=sa.ACOUSTIC_VELOCITY_COPPER, # Default C0
    smooth_window=5 # Smoothing window for feature finding
)

print("\nGenerated Summary Table:")
print(summary_df.head())

# --- Example 2: Plot Spall Strength vs. Strain Rate ---

print("\nGenerating Spall vs Strain Rate plot...")
# Use the generated summary table or specific result files
# Here, we assume the summary table was saved and reload it, or use files directly
# For simplicity, let's use a pattern matching output files if generated separately
# Or better, use the summary_df directly if needed columns are present
# Assuming summary_df has 'Spall Strength (GPa)', 'Strain Rate (s^-1)', etc.

# Path to the summary table generated above
summary_table_path = os.path.join(results_dir, f"spall_summary_table_{os.path.basename(raw_velocity_dir)}.csv")
exp_files_pattern = os.path.join(results_dir,"individual_trace_analysis/*_table_*.csv") # Pattern for individual result files if needed


# Path to literature data (using the example one provided with the package)
# Find the package data directory (this is a bit complex, might need pkg_resources or importlib.resources)
# Simpler: Assume literature file is accessible at a known path
literature_file = os.path.join(lit_data_dir, 'combined_lit_table.csv') # CHANGE PATH if needed

sa.plot_spall_vs_strain_rate(
    # experimental_data_files=[summary_table_path], # Pass list with summary table path
    experimental_data_files=exp_files_pattern, # Or use pattern for individual files
    output_filename=os.path.join(results_dir, 'spall_vs_strain_rate.png'),
    literature_data_file=literature_file,
    x_col='Strain Rate (s^-1)',
    y_col='Spall Strength (GPa)',
    y_unc_col=None, # Specify column name if uncertainty data exists, e.g., 'Spall Strength Uncertainty (GPa)'
    log_scale=True,
    xlim=(1e4, 1e8), # Example limits
    ylim=(0, 7),    # Example limits
    # material_map=sa.MATERIAL_MAPPING, # Use defaults or provide custom map
    # energy_map=sa.ENERGY_VELOCITY_MAPPING, # Use defaults or provide custom map
    filter_high_error_perc=100 # Optional: filter points with >100% relative error in Y
)

# --- Example 3: Plot Spall Strength vs. Shock Stress ---

print("\nGenerating Spall vs Shock Stress plot...")
# Needs experimental data with Peak Velocity (e.g., U_fs_max(m/s)) and Spall Strength
# Can use the same summary table or individual files if they contain the velocity

literature_poly_file = os.path.join(lit_data_dir, 'combined_lit_table_only_poly.csv') # CHANGE PATH if needed

sa.plot_spall_vs_shock_stress(
    experimental_data_files=exp_files_pattern, # Pattern for individual files
    output_filename=os.path.join(results_dir, 'spall_vs_shock_stress.png'),
    literature_data_file=literature_poly_file,
    exp_vel_col='U_fs_max(m/s)', # Column containing peak velocity in exp files
    exp_spall_col='Spall Strength (GPa)', # Column containing spall strength in exp files
    x_col='Shock Stress (GPa)',
    y_col='Spall Strength (GPa)',
    y_unc_col=None, # Specify if uncertainty data exists
    log_scale=True,
    xlim=(1, 100), # Example limits
    ylim=(0, 7),   # Example limits
    filter_high_error_perc=100
)

# --- Example 4: Elastic Net Regression Analysis ---

print("\nRunning Elastic Net regression...")
# This creates a 3D surface plot and optional GIF animation
sa.run_elastic_net_analysis(
    experimental_data_files=exp_files_pattern,
    output_dir=os.path.join(results_dir, 'elastic_net_analysis'),
    x_col='Shock Stress (GPa)',
    y_col='Strain Rate (s^-1)',
    z_col='Spall Strength (GPa)',
    create_gif=True, # Set to False if you don't want GIF
    gif_duration=5 # Duration in seconds
)

print("\nAnalysis complete! Check the results directory for outputs.")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to [https://github.com/Piyushjhu/SPADE](https://github.com/Piyushjhu/SPADE).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spall_analysis_toolkit,
  title={Spall Analysis Toolkit (SPADE)},
  author={Piyush Wanchoo},
  year={2025},
  url={https://github.com/Piyushjhu/SPADE}
}
```

## Acknowledgments

* This package was developed for analyzing spallation experiments using velocity interferometry techniques.
* Special thanks to the research community for providing literature data and validation 
* Jake Diamond and the original ALPSS package literature
* K.T. Ramesh overall project guidence and resources
* Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-23-2-0062. 
* Cursor, Visual Studio Code
* The code heavily leveraged LLM based tools for restructuring and package compilation of the original SPADE package
* Developed at Johns Hopkins University, Department of Mechanical Engineering.