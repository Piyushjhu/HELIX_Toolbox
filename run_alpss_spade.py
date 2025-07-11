#!/usr/bin/env python3
"""
Combined ALPSS-SPADE CLI Pipeline
- Runs ALPSS on input file(s) to generate outputs
- Runs SPADE on ALPSS output(s) automatically
- All outputs go to the specified output directory
"""
import os
import sys
import glob
import argparse
import shutil
from ALPSS.alpss_main import alpss_main
from SPADE.spall_analysis_release.spall_analysis import process_velocity_files

# ---- CLI Argument Parsing ----
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ALPSS and SPADE in a single pipeline. Provide input file(s) or directory, output directory, and options for both packages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input/output
    parser.add_argument('-i', '--input', required=True, help="Input file or directory (for batch mode)")
    parser.add_argument('-o', '--output', required=True, help="Output directory for all results")
    parser.add_argument('--batch', action='store_true', help="If set, process all files in the input directory")

    # ALPSS options (add more as needed)
    alpss_group = parser.add_argument_group('ALPSS options')
    alpss_group.add_argument('--save_all_plots', type=str, default='no', choices=['yes', 'no'], 
                            help='Save all plots in individual subfolders for each file')
    alpss_group.add_argument('--header_lines', type=int, default=5)
    alpss_group.add_argument('--time_to_skip', type=float, default=5e-6)
    alpss_group.add_argument('--time_to_take', type=float, default=2e-6)
    alpss_group.add_argument('--t_before', type=float, default=10e-9)
    alpss_group.add_argument('--t_after', type=float, default=60e-9)
    alpss_group.add_argument('--start_time_correction', type=float, default=0e-9)
    alpss_group.add_argument('--freq_min', type=float, default=1e9)
    alpss_group.add_argument('--freq_max', type=float, default=3.5e9)
    alpss_group.add_argument('--smoothing_window', type=int, default=601)
    alpss_group.add_argument('--smoothing_wid', type=float, default=3)
    alpss_group.add_argument('--smoothing_amp', type=float, default=1)
    alpss_group.add_argument('--smoothing_sigma', type=float, default=1)
    alpss_group.add_argument('--smoothing_mu', type=float, default=0)
    alpss_group.add_argument('--pb_neighbors', type=int, default=400)
    alpss_group.add_argument('--pb_idx_correction', type=int, default=0)
    alpss_group.add_argument('--rc_neighbors', type=int, default=400)
    alpss_group.add_argument('--rc_idx_correction', type=int, default=0)
    alpss_group.add_argument('--sample_rate', type=float, default=128e9)
    alpss_group.add_argument('--nperseg', type=int, default=512)
    alpss_group.add_argument('--noverlap', type=int, default=400)
    alpss_group.add_argument('--nfft', type=int, default=5120)
    alpss_group.add_argument('--window', type=str, default='hann')
    alpss_group.add_argument('--blur_kernel', type=str, default='(5,5)')
    alpss_group.add_argument('--blur_sigx', type=float, default=0)
    alpss_group.add_argument('--blur_sigy', type=float, default=0)
    alpss_group.add_argument('--carrier_band_time', type=float, default=250e-9)
    alpss_group.add_argument('--cmap', type=str, default='viridis')
    alpss_group.add_argument('--uncert_mult', type=float, default=10)
    alpss_group.add_argument('--order', type=int, default=6)
    alpss_group.add_argument('--wid', type=float, default=15e7)
    alpss_group.add_argument('--lam', type=float, default=1550.016e-9)
    alpss_group.add_argument('--C0', type=float, default=3950)
    alpss_group.add_argument('--density', type=float, default=8960)
    alpss_group.add_argument('--delta_rho', type=float, default=9)
    alpss_group.add_argument('--delta_C0', type=float, default=23)
    alpss_group.add_argument('--delta_lam', type=float, default=8e-18)
    alpss_group.add_argument('--theta', type=float, default=0)
    alpss_group.add_argument('--delta_theta', type=float, default=5)
    alpss_group.add_argument('--display_plots', type=str, default='no')
    alpss_group.add_argument('--spall_calculation', type=str, default='yes')
    alpss_group.add_argument('--plot_figsize', type=str, default='(30,10)')
    alpss_group.add_argument('--plot_dpi', type=int, default=300)
    alpss_group.add_argument('--exp_data_dir', type=str, default=None, help="Directory for experimental data (default: input directory)")
    alpss_group.add_argument('--start_time_user', type=str, default='none')

    # SPADE options (add more as needed)
    spade_group = parser.add_argument_group('SPADE options')
    spade_group.add_argument('--spade_density', type=float, default=8960)
    spade_group.add_argument('--spade_acoustic_velocity', type=float, default=3950)
    spade_group.add_argument('--spade_analysis_model', type=str, default='hybrid_5_segment')
    spade_group.add_argument('--spade_signal_length_ns', type=float, default=None)
    spade_group.add_argument('--spade_prominence_factor', type=float, default=1.0)
    spade_group.add_argument('--spade_peak_distance_ns', type=float, default=2.0)
    spade_group.add_argument('--spade_plot_individual', action='store_true')
    spade_group.add_argument('--spade_smooth_window', type=int, default=101)
    spade_group.add_argument('--spade_polyorder', type=int, default=1)
    return parser.parse_args()

# ---- Main Pipeline ----
def main():
    args = parse_args()
    input_path = args.input
    output_dir = args.output
    batch_mode = args.batch
    os.makedirs(output_dir, exist_ok=True)

    # Prepare ALPSS input files
    if batch_mode:
        if not os.path.isdir(input_path):
            print(f"Error: {input_path} is not a directory.")
            sys.exit(1)
        # Accept .csv and .txt files
        input_files = sorted(glob.glob(os.path.join(input_path, '*.csv')) + glob.glob(os.path.join(input_path, '*.txt')))
        if not input_files:
            print(f"No .csv or .txt files found in {input_path}.")
            sys.exit(1)
        exp_data_dir = input_path
    else:
        if not os.path.isfile(input_path):
            print(f"Error: {input_path} is not a file.")
            sys.exit(1)
        input_files = [input_path]
        exp_data_dir = os.path.dirname(input_path)

    # Run ALPSS for each file
    alpss_outputs = []
    for file_path in input_files:
        filename = os.path.basename(file_path)
        print(f"\n[ALPSS] Processing: {filename}")
        
        # Ensure output directory exists before ALPSS runs
        os.makedirs(output_dir, exist_ok=True)
        
        alpss_kwargs = dict(
            filename=filename,
            save_data="yes",
            save_all_plots=args.save_all_plots,
            start_time_user=args.start_time_user,
            header_lines=args.header_lines,
            time_to_skip=args.time_to_skip,
            time_to_take=args.time_to_take,
            t_before=args.t_before,
            t_after=args.t_after,
            start_time_correction=args.start_time_correction,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            smoothing_window=args.smoothing_window,
            smoothing_wid=args.smoothing_wid,
            smoothing_amp=args.smoothing_amp,
            smoothing_sigma=args.smoothing_sigma,
            smoothing_mu=args.smoothing_mu,
            pb_neighbors=args.pb_neighbors,
            pb_idx_correction=args.pb_idx_correction,
            rc_neighbors=args.rc_neighbors,
            rc_idx_correction=args.rc_idx_correction,
            sample_rate=args.sample_rate,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            nfft=args.nfft,
            window=args.window,
            blur_kernel=eval(args.blur_kernel),
            blur_sigx=args.blur_sigx,
            blur_sigy=args.blur_sigy,
            carrier_band_time=args.carrier_band_time,
            cmap=args.cmap,
            uncert_mult=args.uncert_mult,
            order=args.order,
            wid=args.wid,
            lam=args.lam,
            C0=args.C0,
            density=args.density,
            delta_rho=args.delta_rho,
            delta_C0=args.delta_C0,
            delta_lam=args.delta_lam,
            theta=args.theta,
            delta_theta=args.delta_theta,
            exp_data_dir=exp_data_dir,
            out_files_dir=output_dir,
            display_plots=args.display_plots,
            spall_calculation=args.spall_calculation,
            plot_figsize=eval(args.plot_figsize),
            plot_dpi=args.plot_dpi,
        )
        # Copy file to output dir if needed
        if exp_data_dir != output_dir:
            shutil.copy(file_path, os.path.join(output_dir, filename))
        
        # Use absolute paths to avoid directory issues
        alpss_kwargs['exp_data_dir'] = os.path.abspath(exp_data_dir)
        alpss_kwargs['out_files_dir'] = os.path.abspath(output_dir)
        
        try:
            alpss_main(**alpss_kwargs)
        except Exception as e:
            print(f"[ALPSS] Error processing {filename}: {e}")
        # Track output for SPADE
        out_base = os.path.splitext(filename)[0]
        vel_smooth_uncert = os.path.join(output_dir, f"{out_base}--vel-smooth-with-uncert.csv")
        if os.path.exists(vel_smooth_uncert):
            alpss_outputs.append(vel_smooth_uncert)
        else:
            print(f"[ALPSS] Warning: {vel_smooth_uncert} not found.")

    # ---- Run SPADE on ALPSS outputs ----
    print(f"\n[SPADE] Processing {len(alpss_outputs)} ALPSS output file(s)...")
    # Place SPADE outputs in a subfolder for clarity
    spade_output_dir = os.path.join(output_dir, "SPADE_results")
    os.makedirs(spade_output_dir, exist_ok=True)
    # Copy ALPSS outputs to SPADE input folder
    for f in alpss_outputs:
        shutil.copy(f, spade_output_dir)
    # Run SPADE batch processing
    spade_kwargs = dict(
        input_folder=spade_output_dir,
        file_pattern='*.csv',
        output_folder=spade_output_dir,
        save_summary_table=True,
        summary_table_name=os.path.join(spade_output_dir, "spall_summary.csv"),
        density=args.spade_density,
        acoustic_velocity=args.spade_acoustic_velocity,
        analysis_model=args.spade_analysis_model,
        signal_length_ns=args.spade_signal_length_ns,
        prominence_factor=args.spade_prominence_factor,
        peak_distance_ns=args.spade_peak_distance_ns,
        plot_individual=args.spade_plot_individual,
        smooth_window=args.spade_smooth_window,
        polyorder=args.spade_polyorder,
    )
    try:
        process_velocity_files(**spade_kwargs)
        print(f"[SPADE] Results saved in {spade_output_dir}")
    except Exception as e:
        print(f"[SPADE] Error: {e}")

    print("\nPipeline complete. All results are in:", output_dir)

if __name__ == "__main__":
    main() 