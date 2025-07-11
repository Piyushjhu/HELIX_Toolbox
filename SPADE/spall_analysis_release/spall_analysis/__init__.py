# spall_analysis/__init__.py
"""
Spall Analysis Toolkit (SPADE)

A Python package for processing experimental data from spallation or shock physics 
experiments to analyze material failure dynamics.

Author: Piyush Wanchoo
Institution: Johns Hopkins University
Year: 2025
GitHub: https://github.com/Piyushjhu/SPADE

This toolkit provides comprehensive tools for:
- Velocity trace processing and analysis
- Spall strength and strain rate calculations
- Literature data comparison and visualization
- Material model implementations
- Interactive GUI for data processing
"""

__version__ = "0.1.0"
__author__ = "Piyush Wanchoo"
__email__ = "pwanchoo@jhu.edu"
__institution__ = "Johns Hopkins University"
__year__ = "2025"
__github__ = "https://github.com/Piyushjhu/SPADE"

from .utils import (
    DENSITY,
    ACOUSTIC_VELOCITY,
    MATERIAL_MAPPING,
    ENERGY_VELOCITY_MAPPING,
    COLOR_MAPPING,
    MARKER_MAPPING,
    find_data_files,
    extract_legend_info,
    calculate_shock_stress,
    add_shock_stress_column
)

from .data_processing import (
    process_velocity_files,
    calculate_spall_parameters
)

from .plotting import (
    plot_velocity_comparison,
    plot_spall_vs_strain_rate,
    plot_spall_vs_shock_stress,
    plot_wilkerson_comparison,
    plot_spall_vs_strain_rate_multi_wilkerson,
    plot_combined_mean_traces,
    plot_combined_mean_models,
    plot_model_vs_feature,
    plot_model_3d_surface,
    plot_combined_material_surfaces,
    plot_interactive_spall_vs_strain_rate,
    plot_elastic_net_results,
    plot_spall_vs_strain_rate_multi_wilkerson_historic,
    plot_spall_vs_grain_size_with_wilkerson_model,
    plot_spall_vs_grain_size_with_wilkerson_model_extended,
    plot_spall_vs_grain_size_with_revised_wilkerson_model_2025,
)

from .models import (
    calculate_wilkerson_spall_complex,
    train_elastic_net,
    prepare_feature_matrix,
    train_and_plot_models_per_material
)

from .literature import load_literature_data
