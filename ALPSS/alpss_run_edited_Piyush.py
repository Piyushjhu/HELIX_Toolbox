# %%
from alpss_main import *
import os
import glob

def process_all_files(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files in the input directory using glob
    input_files = glob.glob(os.path.join(input_dir, '*.txt'))

    for file_path in input_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        
        alpss_main(
            filename=filename,
            save_data="yes",
            save_all_plots="yes",
            start_time_user="none",
            header_lines=5,
            time_to_skip=2e-6,
            time_to_take=2e-6,
            t_before=20e-9,
            t_after=150e-9,
            start_time_correction=0e-9,
            freq_min=1.5e9,
            freq_max=3.5e9,
            smoothing_window=601,
            smoothing_wid=3,
            smoothing_amp=1,
            smoothing_sigma=1,
            smoothing_mu=0,
            pb_neighbors=400,
            pb_idx_correction=0,
            rc_neighbors=400,
            rc_idx_correction=0,
            sample_rate=128e9,
            nperseg=512,
            noverlap=435,
            nfft=5120,
            window="hann",
            blur_kernel=(5, 5),
            blur_sigx=0,
            blur_sigy=0,
            carrier_band_time=250e-9,
            cmap="viridis",
            uncert_mult=10,
            order=6,
            wid=15e7,
            lam=1550.016e-9,
            C0=3950,
            density=8960,
            delta_rho=9,
            delta_C0=23,
            delta_lam=8e-18,
            theta=0,
            delta_theta=5,
            exp_data_dir=input_dir,
            out_files_dir=output_dir,
            display_plots="yes",
            spall_calculation="yes",
            plot_figsize=(30, 10),
            plot_dpi=300,
        )

# Define input and output directories
input_directory = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Stieff_Scope/PDV_data/2025_04_25_spall"
output_directory = "/Users/piyushwanchoo/Library/CloudStorage/OneDrive-JohnsHopkins/Stieff_Scope/PDV_data/2025_04_25_spall/Output"

# Run the processing function
process_all_files(input_directory, output_directory)

# %%
