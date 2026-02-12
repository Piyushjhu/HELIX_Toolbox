"""HELIX Toolbox CLI — replaces helix_cli_runner.py with no Qt dependency."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")

from helix.config.loader import load_config, load_master_config
from helix.config.transform import transform_alpss_params, transform_spade_params
from helix.pipeline.params import load_parameter_folder, resolve_file_list
from helix.pipeline.runner import run_pipeline

logger = logging.getLogger("helix")


def _setup_logging():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    root = logging.getLogger("helix")
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HELIX Toolbox CLI — PDV velocity analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to master config JSON.")
    parser.add_argument("--alpss-config", help="Path to ALPSS JSON config.")
    parser.add_argument("--spade-config", help="Path to SPADE JSON config.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument("--input-files", nargs="+", help="Explicit PDV input files.")
    parser.add_argument("--input-dir", help="Directory of PDV input files.")
    parser.add_argument("--input-pattern", default="*.csv", help="Glob pattern (default: *.csv).")
    parser.add_argument("--param-folder", help="Experiment metadata folder.")
    parser.add_argument(
        "--analysis-mode", choices=["both", "alpss_only", "spade_only"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--spade-mode", choices=["auto", "manual"],
        help="Auto (use ALPSS output) or manual (provide SPADE input files).",
    )
    parser.add_argument("--spade-input-files", nargs="+", help="Manual SPADE input files.")
    parser.add_argument("--spade-input-dir", help="Manual SPADE input directory.")
    parser.add_argument("--spade-input-pattern", help="Manual SPADE glob pattern.")
    return parser.parse_args()


def main():
    _setup_logging()
    args = _parse_args()

    # ---- Load config ----
    if args.config:
        master = load_master_config(os.path.abspath(args.config))
        cli_settings = master.get("cli_settings", {})
        alpss_params = transform_alpss_params(master["alpss_config"])
        spade_params = transform_spade_params(master["spade_config"])
        material_properties = master.get("material_properties", {})
    elif args.alpss_config and args.spade_config:
        alpss_params = transform_alpss_params(load_config(os.path.abspath(args.alpss_config)))
        spade_params = transform_spade_params(load_config(os.path.abspath(args.spade_config)))
        cli_settings = {}
        material_properties = {}
    else:
        print("Error: provide --config or both --alpss-config and --spade-config", file=sys.stderr)
        sys.exit(1)

    # ---- Resolve CLI overrides ----
    def _or(a, b):
        return a if a is not None else b

    input_dir = _or(args.input_dir, cli_settings.get("input_dir"))
    input_files_arg = _or(args.input_files, cli_settings.get("input_files"))
    input_pattern = args.input_pattern or cli_settings.get("input_pattern", "*.csv")
    output_dir = _or(args.output_dir, cli_settings.get("output_dir"))
    param_folder = _or(args.param_folder, cli_settings.get("param_folder"))
    analysis_mode = _or(args.analysis_mode, cli_settings.get("analysis_mode", "both"))
    spade_mode = _or(args.spade_mode, cli_settings.get("spade_mode", "auto"))

    if not output_dir:
        print("Error: --output-dir is required", file=sys.stderr)
        sys.exit(1)
    output_dir = os.path.abspath(output_dir)

    # ---- Resolve input files ----
    input_files = []
    if analysis_mode != "spade_only":
        input_files = resolve_file_list(input_files_arg, input_dir, input_pattern)
        if not input_files:
            print("Error: no input files found for ALPSS.", file=sys.stderr)
            sys.exit(1)
        print(f"Resolved {len(input_files)} input file(s)")

    spade_input_files = None
    if spade_mode == "manual":
        spade_input_files = resolve_file_list(
            args.spade_input_files,
            _or(args.spade_input_dir, cli_settings.get("spade_input_dir")),
            args.spade_input_pattern or cli_settings.get("spade_input_pattern", "*--vel-smooth-with-uncert.csv"),
        )

    param_data = load_parameter_folder(param_folder) if param_folder else None

    # ---- Run ----
    def on_progress(msg):
        print(msg, flush=True)

    result = run_pipeline(
        input_files=input_files,
        output_dir=output_dir,
        alpss_params=alpss_params,
        spade_params=spade_params,
        analysis_mode=analysis_mode,
        param_data=param_data,
        material_properties=material_properties,
        spade_input_files=spade_input_files,
        progress_callback=on_progress,
    )

    n_ok = len(result.get("successful_files", []))
    n_fail = len(result.get("failed_files", []))
    if n_fail:
        print(f"\n{n_ok} succeeded, {n_fail} failed.")
        sys.exit(1)
    else:
        print(f"\nAll {n_ok} files processed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
