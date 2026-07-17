# Supplementary Files

This folder contains files that are **not required** to run the core HELIX Toolbox
(either the GUI `helix_analysis_toolbox.py` or the CLI `helix_cli_runner.py`).
They are kept here because they are still useful for post-analysis, testing, or
reference, but they are intentionally separated from the main runtime so the
essential entry points are easy to spot at the repo root.

## Layout

```
supplementary/
├── paper_plots/        # Standalone scripts for publication figures (post-analysis)
│   ├── generate_paper_plots_standalone.py
│   ├── plot_velocity_traces_by_laser_energy.py
│   └── run_plot_all_traces.sh
├── tests/              # Ad-hoc test / comparison scripts
│   ├── test_combined_velocity_plot.py
│   └── test_iq_detection_comparison.py
└── references/         # Academic / LaTeX derivations
    └── SPALL_STRENGTH_CALCULATION.tex
```

## Running the paper-plot scripts from this folder

The two standalone scripts (`generate_paper_plots_standalone.py` and
`plot_velocity_traces_by_laser_energy.py`) automatically add the repo root
(`<repo>/`) to `sys.path` so they can still import `helix_paper_plots` and
`helix_analysis_toolbox` from their new location. You can run them from
anywhere:

```bash
# From the repo root
python supplementary/paper_plots/generate_paper_plots_standalone.py
python supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py --config ./helix_master_config.json
```

They will look for `helix_master_config.json` at the repo root by default.

## What is considered "essential" (kept at the repo root)

| File / Directory | Role |
|---|---|
| `helix_analysis_toolbox.py` | GUI entry point + shared library |
| `helix_cli_runner.py` | Command-line batch runner |
| `helix_paper_plots.py` | **Imported at runtime** by `helix_analysis_toolbox.py` |
| `material_properties.py` | Imported at runtime by `helix_analysis_toolbox.py` |
| `ALPSS/` | Core signal-processing package |
| `SPADE/` | Core spall-analysis package |
| `helix_master_config.json` | Default master config |
| `alpss_config_default.json`, `spade_config_default.json` | Default per-tool configs |
| `requirements.txt`, `setup.py`, `run_helix_toolbox.bat` | Install / launch |
| `README.md`, `LICENSE`, `CHANGELOG.md` | Project metadata |
| `HEL_DETECTION_ALGORITHM.md` | Algorithm reference linked from README (spall detection is documented in the main README) |
