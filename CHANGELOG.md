# Changelog

All notable changes to HELIX Toolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **IGSN → Material Mapping**: New `igsn_material_map` section in the master config files maps IGSNs (or parent-IGSN prefixes like `JHAMAL00016`) to material names (longest key wins, case-insensitive), and the resolved material is then looked up in `material_properties` as usual. The map now **takes priority** over a trace's parameter-file `Sample material` column, so a known sample's IGSN always wins over a stale or missing parameter-file value; the parameter-file column is used only as a fallback when the IGSN has no map entry. Material resolution across the pipeline (spall analysis, HEL detection, Data_Summary enhancement, combined velocity plots, and `batch_summary_plot.py`) is consolidated into a shared `resolve_sample_material()` / `refresh_material_column()`, which also treats `"[]"` (an empty-list artifact left by some parameter-file exports) as an invalid material value alongside `nan`/`none`/empty. The map is recorded in the per-run `Run_Config.json` snapshot
- **`pullback_smoothing_ns`** (`spade_config`): the smoothing window used before searching for the post-plateau spall pullback (P3) is now sized in real time (ns) rather than a fixed sample count, so it scales correctly across oscilloscope sample rates from 2 GS/s to 128+ GS/s instead of silently becoming a no-op at high sample rates
- Added `pyyaml` to `ALPSS/requirements.txt` (required for YAML config support); README troubleshooting updated accordingly

### Fixed
- **ALPSS**: rows with unparseable/NaN values (e.g. a truncated final oscilloscope sample) are now dropped before FFT-based smoothing in IQ start-time detection. Previously a single bad row would NaN out the entire smoothed amplitude trace via FFT convolution, crashing detection for the whole file

### Removed
- Removed unused `alpss_config` keys `use_robust_iq_detection`, `iq_smoothing_window_ns`, `iq_skip_start_ns`, `iq_persistence_ns` from the default/master config files — the code path they controlled had already been removed; only `iq_threshold_factor` remains configurable for IQ onset detection
- Removed the vestigial `analysis_model` key from `spade_config` in the default/master config files — calculations have used the hybrid approach unconditionally since v2.1.0, and the key was already a no-op

## [2.1.0] - 2026-07-16

### Added
- **Batch Processing Mode**: `helix_cli_runner.py` can now walk a parent directory of per-shot subfolders (`batch_mode`, `subfolder_pattern` in `cli_settings`) and run the full analysis pipeline on each one in sequence
- **Batch Summary Plotting**: New `batch_summary_plot.py` aggregates `enhanced_spall_summary.csv` and `velocity_shots_summary.csv` across all subfolders of a batch run into a combined Spall/HEL strength-vs-strain-rate figure, color-coded by material
- **ALPSS Noise-Fraction Filter**: New `noise_filter_enabled`/`noise_filter_threshold` options in `alpss_config` replace high-noise velocity samples with linear interpolation before plotting/saving
- New example batch config `helix_master_config_batch_process.json`
- New standalone plotting script `supplementary/paper_plots/plot_energy_waveplate_and_velocity_violin.py`

### Changed
- **Spall Pullback (P3) Detection Rewritten**: the 5-segment/hybrid spall algorithm now finds the *first* local minimum after the plateau (prominence-based peak detection on the inverted, smoothed post-plateau signal), instead of taking the global minimum of the post-plateau trace. Secondary reverberations can produce a deeper but later minimum that isn't the true spall pullback — the previous global-minimum approach could lock onto those. There is no longer a global-minimum fallback: if no valley clears the prominence threshold (1% of plateau mean velocity, floor 2 m/s), the trace is classified DNS immediately rather than substituting a spurious P3
- **Spall Strength Uncertainty** is now propagated from the pullback velocity uncertainty (`0.5 * density * acoustic_velocity * pullback_velocity_uncertainty`) instead of being hardcoded to `0.0`
- **Consolidated Summary Filename**: spall/HEL results now save to an IGSN-prefixed `<parent-folder>-Data_Summary.csv` (via new `_get_summary_filename()`) instead of the fixed `enhanced_spall_summary.csv` name, falling back to `Data_Summary.csv` when the parent folder name is generic
- **CLI Parameter Folder Matching**: `_load_parameter_folder` now accepts an `experiment_id` to disambiguate parameter lookups in batch mode
- Updated `supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py` for Ti datasets and energy-bin statistics, with improved standalone config fallback and a faster execution mode
- Windows GUI analysis output now avoids Unicode-only characters that failed under `cp1252` encoding

### Removed
- Removed `SPALL_DETECTION_ALGORITHM.md` and `SPALL_DETECTION_ALGORITHM_5SEGMENT_ONLY.md` (superseded by the consolidated README documentation)

## [2.0.0] - 2025-11-22

### Added
- **Command-Line Interface (CLI)**: New `helix_cli_runner.py` for batch processing without GUI
- **Master Configuration File**: Single `helix_master_config.json` for all settings
- **Post-Processing Mode**: Generate plots from existing analysis results without rerunning full analysis
- **HEL (Hugoniot Elastic Limit) Detection**: Gradient-based method with configurable parameters
  - Elastic shock strain rate calculation
  - Minimum HEL velocity constraint
  - Configurable consecutive points threshold
  - Individual HEL detection plots with strain rate slope marking
- **MAD (Median Absolute Deviation) Filter**: Statistical outlier removal for peak velocities
  - Applied per material and laser energy bracket
  - Asymmetric MAD calculation
  - Configurable threshold
- **Enhanced Plotting**: Multiple new analysis plots
  - Shock stress vs laser energy (by material)
  - Shock stress vs waveplate angle (by material)
  - Shock stress vs peak velocity (by material)
  - HEL vs peak velocity (by material)
  - HEL vs HEL strain rate (by material)
  - Row/column peak velocity heatmap
  - Row/column pair scatter plots
  - Peak velocity pattern analysis
  - Laser energy vs waveplate angle
- **Material Properties Enhancement**: Added `C_L` (longitudinal wave velocity) to material properties
- **IQ Detection Filtering**: Filter traces with poor IQ detection start times
- **Trace Alignment Improvements**: Enhanced threshold crossing detection and alignment
- **Diagnostic Tools**: `diagnose_high_laser_energy.py` for identifying data anomalies

### Changed
- **Shock Stress Calculation**: Updated to use Hugoniot EOS (`U = c + S*u_p`, `σ = ρ * U * u_p`)
- **Particle Velocity**: Corrected to use `u_p = u_fs / 2` in shock stress calculations
- **HEL Detection**: Removed peak-valley fallback method, now uses gradient-based method only
- **Configuration Structure**: Consolidated into master config file format
- **Plot Styling**: Improved line widths, legend formatting, and color consistency across plots

### Fixed
- Fixed trace alignment to require crossing from below threshold
- Fixed IQ detection failure filtering
- Fixed spall analysis running when disabled in config
- Fixed post-processing mode error handling
- Fixed heatmap plot aspect ratio and white space issues
- Fixed material color consistency across all plots

### Improved
- Enhanced error handling and user feedback
- Improved trace counting and rejection reporting
- Better summary output at end of CLI runs
- Optimized plot generation and layout
- Streamlined documentation (consolidated into single README)

### Technical Details
- Added `hel_detection_min_points` config parameter (default: 3)
- Added `minimum_HEL_velocity_expected` config parameter (default: 15.0 m/s)
- Added `mad_filter_enabled` and `mad_filter_threshold` config parameters
- Added `skip_unknown_material_traces` config parameter
- Post-processing config with selective plot generation
- Improved material property retrieval with fallback options

---

## [1.0.0] - 2025-01-XX

### Added
- Initial release of HELIX Toolbox
- Comprehensive GUI for single point PDV data analysis
- Integration of ALPSS and SPADE packages
- Three analysis modes: ALPSS Only, SPADE Only, and Combined
- Optional Gaussian notch filter for carrier frequency removal
- Complete uncertainty propagation throughout analysis pipeline
- Batch processing capabilities for multiple files
- Real-time progress monitoring
- Dark/light theme support
- Comprehensive parameter configuration options
- Rich output generation including plots and summary tables

### Features
- **ALPSS Integration**: Raw PDV signal processing to velocity traces
- **SPADE Integration**: Spall strength and strain rate analysis
- **Gaussian Notch Filter**: Optional carrier frequency removal with user control
- **Uncertainty Analysis**: Complete error propagation from velocity to spall strength
- **Peak Detection**: Automated feature detection with configurable parameters
- **Material Properties**: Support for various materials with customizable properties
- **Output Formats**: CSV data files, PNG plots, and enhanced summary tables

### Technical Details
- Built with PyQt5 for cross-platform compatibility
- Scientific notation support for high-precision parameters
- Parameter validation and constraint enforcement
- Modular architecture for easy maintenance and extension
- Comprehensive error handling and user feedback

### Credits
- **ALPSS**: Original package by Jake Diamond (@Jake-Diamond-9)
- **SPADE**: Spall analysis toolkit by Piyush Wanchoo (@Piyushjhu)
- **HELIX Toolbox**: Integration and GUI by Piyush Wanchoo (@Piyushjhu)

---

## Version History

### Version 2.0.0
- Major feature release with CLI, HEL detection, MAD filtering, and enhanced plotting
- Comprehensive post-processing capabilities
- Improved configuration management

### Version 1.0.0
- Initial release with full ALPSS and SPADE integration
- Complete GUI with all analysis modes
- Comprehensive documentation and user guides
