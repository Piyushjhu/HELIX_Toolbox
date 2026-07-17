---
name: verify
description: Build/launch/drive recipe for verifying HELIX Toolbox changes end-to-end via the CLI runner
---

# Verifying HELIX Toolbox changes

## Runtime

Use the project venv directly (plain `python3` has no pandas; `source activate` in a subshell doesn't stick):

```bash
helix_toolbox_env/bin/python3 ...
```

Set `QT_QPA_PLATFORM=offscreen` — the toolbox imports PyQt5 even in CLI mode.

## Sample data

A real Ti alloy PDV oscilloscope trace ships in the repo (128 GS/s, 6 µs, 22 header lines):

```
input_data/C1_files/JHAMAL00016-004_2026-04-23_21-52-09_shot20_ch1.csv
```

## Drive an end-to-end run (~5-7 s)

1. Copy the sample file into a scratch dir, e.g. `<scratch>/JHAMAL00016-004/PDV/`.
   Naming the parent folder like an IGSN matters: the summary/config filenames
   are prefixed from `basename(dirname(output_dir))`.
2. Load `helix_master_config.json`, override `cli_settings`:
   `input_dir`, `output_dir` (e.g. `<scratch>/JHAMAL00016-004/Output`),
   `param_folder: null`, `analysis_mode: "both"`, `spade_mode: "auto"`,
   and set `spade_config.skip_unknown_material_traces: false` (no param folder).
3. Run:

```bash
QT_QPA_PLATFORM=offscreen helix_toolbox_env/bin/python3 helix_cli_runner.py --config <scratch>/config.json
```

Exit 0 and `✅ Analysis completed successfully.` on success.

## Where outputs land

Final consolidated outputs go to `<output_dir>/SPADE_analysis/`:
`<prefix>-Data_Summary.csv`, `<prefix>-Run_Config.json`,
`velocity_shots_summary.csv`, plots. ALPSS per-trace CSVs go to
`<output_dir>` itself.

## Gotchas

- With no param folder all traces are material "Unknown"; several combined
  plots warn and skip — expected, not a failure.
- The `<prefix>-Data_Summary.csv` write (and the run-config save next to it)
  happens twice per run — two code paths both save; second overwrites.
- Generic parent folder names (`Output`, `output`, `Results`, `results`)
  drop the prefix → plain `Data_Summary.csv` / `Run_Config.json`.
