# Archive Folder

This folder contains outdated or non-essential Python scripts that have been archived from the main HELIX Toolbox package. These files are kept for reference but are **not required** for the core HELIX Toolbox functionality.

## Why These Files Are Archived

The main HELIX Toolbox package has been streamlined to include only essential files:
- `helix_analysis_toolbox.py` - Main GUI application
- `helix_cli_runner.py` - CLI runner for batch processing
- `material_properties.py` - Material properties database
- `ALPSS/alpss_main.py` - Core ALPSS functionality
- `setup.py` - Package setup/installation

All functionality from the archived files has been integrated into the main toolbox or replaced by newer implementations.

---

## Archived Files Description

### Root Directory Scripts

#### `lmi_data_processor.py`
**Purpose**: LMI (Laser Material Interaction) specific data processor  
**Description**: Processes CSV/XLSX files from LMI experiments and generates analysis plots. Designed for files with naming convention `LMI_YYYYMMDD_IGSN` (e.g., `LMI_20251023_JHAMAB00019-06`).  
**Why Archived**: Not part of core HELIX Toolbox functionality - specific to LMI experiments only.  
**Status**: Outdated - functionality not integrated into main toolbox.

#### `run_alpss_spade.py`
**Purpose**: Old combined ALPSS-SPADE CLI pipeline  
**Description**: Command-line script that runs ALPSS on input files and automatically runs SPADE on ALPSS outputs. Provides a single pipeline for processing PDV data through both analysis stages.  
**Why Archived**: Replaced by `helix_cli_runner.py` which provides more features, better configuration management, and post-processing capabilities.  
**Status**: Superseded - use `helix_cli_runner.py` instead.

#### `create_release.py`
**Purpose**: GitHub release creation script  
**Description**: Development tool that creates GitHub releases with versioning and documentation. Automates the release process including version tagging, release notes generation, and GitHub API interactions.  
**Why Archived**: Development/maintenance script, not needed for end users.  
**Status**: Development tool - only needed for package maintainers.

#### `release.py`
**Purpose**: Release preparation and tagging script  
**Description**: Helper script for preparing and tagging releases for GitHub. Checks git status, updates version numbers, and creates release tags.  
**Why Archived**: Development/maintenance script, not needed for end users.  
**Status**: Development tool - only needed for package maintainers.

#### `update_materials.py`
**Purpose**: Material property update utility  
**Description**: Script to update material information in CSV parameter files. Extracts material type from filenames using JHAMAB codes (19=Cu, 20=Zn, 21=Brass) and updates the "Sample material" column in metadata CSV files.  
**Why Archived**: Contains hardcoded paths and is specific to a particular dataset structure. Functionality should be handled through the main toolbox's parameter file integration.  
**Status**: Outdated - hardcoded paths make it non-portable.

#### `requirements_current.txt`
**Purpose**: Snapshot of exact package versions  
**Description**: Contains exact pinned versions (e.g., `numpy==2.0.2`) of all dependencies that were installed at a specific point in time.  
**Why Archived**: Not used by the codebase. The main `requirements.txt` uses version ranges (e.g., `numpy>=1.19.0`) which is more flexible and standard practice.  
**Status**: Superseded - use `requirements.txt` instead.

---

### ALPSS Directory Scripts

#### `alpss_run.py`
**Purpose**: ALPSS runner script  
**Description**: Standalone script to run ALPSS analysis on PDV data files. Provides command-line interface for ALPSS processing.  
**Why Archived**: ALPSS functionality is now fully integrated into `helix_analysis_toolbox.py` and `helix_cli_runner.py`. No need for separate runner script.  
**Status**: Superseded - functionality integrated into main toolbox.

#### `alpss_auto_run.py`
**Purpose**: ALPSS auto-runner script  
**Description**: Automated runner for ALPSS that processes files with minimal user input. Designed for batch processing scenarios.  
**Why Archived**: ALPSS functionality is now fully integrated into `helix_analysis_toolbox.py` and `helix_cli_runner.py` with better batch processing capabilities.  
**Status**: Superseded - functionality integrated into main toolbox.

---

### Supplementary Directory Scripts

#### `alpss_fix_script.py`
**Purpose**: ALPSS fix script for performance and broadcasting issues  
**Description**: Contains fixes for identified performance and array broadcasting issues in ALPSS. Includes functions for safe array trimming and handling array length mismatches.  
**Why Archived**: These fixes have been incorporated into the main ALPSS codebase (`ALPSS/alpss_main.py`). The script is no longer needed as a separate fix.  
**Status**: Outdated - fixes integrated into main code.

#### `create_velocity_summary.py`
**Purpose**: Velocity summary creation tool  
**Description**: Supplementary tool that creates velocity summaries from processed data. Aggregates velocity data from multiple files and calculates statistics.  
**Why Archived**: Velocity summary functionality is now built into the main SPADE analysis workflow. The `velocity_shots_summary.csv` generated by SPADE provides comprehensive summaries.  
**Status**: Superseded - functionality integrated into main toolbox.

#### `enhanced_plotting.py`
**Purpose**: Enhanced plotting module for velocity analysis  
**Description**: Standalone plotting module with 6 different figure types, noise filtering, trace alignment, material/waveplate angle color coding, spread analysis, and statistical plots. Can generate CSV exports for further analysis.  
**Why Archived**: Enhanced plotting capabilities are now integrated into the main SPADE analysis workflow. The main toolbox generates comprehensive plots automatically.  
**Status**: Superseded - functionality integrated into main toolbox.

#### `velocity_summary_post_processor.py`
**Purpose**: Post-processor for velocity shots summary  
**Description**: Post-processes velocity shots summary data to create box plots (shot time vs material, PDV Return Power vs material), handles different material column names, saves high-resolution plots, and generates statistics summaries.  
**Why Archived**: Post-processing capabilities are now built into the main toolbox through the post-processing mode in `helix_cli_runner.py`. The main toolbox generates comprehensive plots and statistics automatically.  
**Status**: Superseded - functionality integrated into main toolbox.

---

## Migration Guide

If you were using any of these archived scripts, here's how to achieve the same functionality with the current HELIX Toolbox:

### For `run_alpss_spade.py` users:
**Use**: `helix_cli_runner.py` with master config file
```bash
python helix_cli_runner.py --config helix_master_config.json
```

### For `alpss_run.py` or `alpss_auto_run.py` users:
**Use**: `helix_cli_runner.py` with `--analysis-mode alpss_only`
```bash
python helix_cli_runner.py --config helix_master_config.json --analysis-mode alpss_only
```

### For `enhanced_plotting.py` users:
**Use**: Post-processing mode in `helix_cli_runner.py`
```bash
# Enable post-processing in helix_master_config.json
python helix_cli_runner.py --config helix_master_config.json
```

### For `velocity_summary_post_processor.py` users:
**Use**: Post-processing mode with selective plot generation
```bash
# Configure plots in helix_master_config.json post_processing_config section
python helix_cli_runner.py --config helix_master_config.json
```

### For `create_velocity_summary.py` users:
**Use**: SPADE analysis automatically generates `velocity_shots_summary.csv`
```bash
python helix_cli_runner.py --config helix_master_config.json --analysis-mode spade_only
```

---

## Notes

- **These files are kept for reference only** - they are not maintained or updated
- **Do not use these files** - use the main HELIX Toolbox instead
- **If you need functionality from these files**, check if it's available in the main toolbox first
- **For questions or issues**, refer to the main README.md in the root directory

---

## File Summary Table

| File | Type | Status | Replacement |
|------|------|--------|-------------|
| `lmi_data_processor.py` | LMI-specific | Outdated | N/A (not core functionality) |
| `run_alpss_spade.py` | CLI pipeline | Superseded | `helix_cli_runner.py` |
| `create_release.py` | Dev tool | Dev only | N/A (maintainer tool) |
| `release.py` | Dev tool | Dev only | N/A (maintainer tool) |
| `update_materials.py` | Utility | Outdated | Parameter file integration |
| `requirements_current.txt` | Dependencies | Superseded | `requirements.txt` |
| `alpss_run.py` | ALPSS runner | Superseded | Integrated into main toolbox |
| `alpss_auto_run.py` | ALPSS auto-runner | Superseded | Integrated into main toolbox |
| `alpss_fix_script.py` | Fix script | Outdated | Fixes integrated into code |
| `create_velocity_summary.py` | Summary tool | Superseded | SPADE analysis |
| `enhanced_plotting.py` | Plotting tool | Superseded | Post-processing mode |
| `velocity_summary_post_processor.py` | Post-processor | Superseded | Post-processing mode |

---

**Last Updated**: November 22, 2025  
**HELIX Toolbox Version**: 2.0.0
