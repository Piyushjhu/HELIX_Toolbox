"""
ALPSS
Jake Diamond (2024)
Johns Hopkins University
Hopkins Extreme Materials Institute (HEMI)
Please report any bugs or comments to jdiamo15@jhu.edu


Key for input variables:
filename:                   str; filename for the data to run
save_data:                  str; 'yes' or 'no' to save output data
start_time_user:            str or float; if 'none' the program will attempt to find the
                                             signal start time automatically. if float then
                                             the program will use that as the signal start time
header_lines:               int; number of header lines to skip in the data file
time_to_skip:               float; the amount of time to skip in the full data file before beginning to read in data
time_to_take:               float; the amount of time to take in the data file after skipping time_to_skip
t_before:                   float; amount of time before the signal start time to include in the velocity calculation
t_after:                    float; amount of time after the signal start time to include in the velocity calculation
start_time_correction:      float; amount of time to adjust the signal start time by
freq_min:                   float; minimum frequency for the region of interest
freq_max:                   float; maximum frequency for the region of interest
smoothing_window:           int; number of points to use for the smoothing window. must be an odd number
smoothing_wid:              float; half the width of the normal distribution used
                                   to calculate the smoothing weights (recommend 3)
smoothing_amp:              float; amplitude of the normal distribution used to calculate
                                   the smoothing weights (recommend 1)
smoothing_sigma:            float; standard deviation of the normal distribution used
                                   to calculate the smoothing weights (recommend 1)
smoothing_mu:               float; mean of the normal distribution used to calculate
                                   the smoothing weights (recommend 0)
pb_neighbors:               int; number of neighbors to compare to when searching
                                     for the pullback local minimum
pb_idx_correction:          int; number of local minima to adjust by if the program grabs the wrong one
rc_neighbors:               int; number of neighbors to compare to when searching
                                     for the recompression local maximum
rc_idx_correction:          int; number of local maxima to adjust by if the program grabs the wrong one
sample_rate:                float; sample rate of the oscilloscope used in the experiment
nperseg:                    int; number of points to use per segment of the stft
noverlap:                   int; number of points to overlap per segment of the stft
nfft:                       int; number of points to zero pad per segment of the stft
window:                     str or tuple or array_like; window function to use for the stft (recommend 'hann')
blur_kernel:                tuple; kernel size for gaussian blur smoothing (recommend (5, 5))
blur_sigx:                  float; standard deviation of the gaussian blur kernel in the x direction (recommend 0)
blur_sigy:                  float; standard deviation of the gaussian blur kernel in the y direction (recommend 0)
carrier_band_time:          float; length of time from the beginning of the imported data window to average
                                   the frequency of the top of the carrier band in the thresholded spectrogram
cmap:                       str; colormap for the spectrograms (recommend 'viridis')
uncert_mult:                float; factor to multiply the velocity uncertainty by when plotting - allows for easier
                                   visulaization when uncertainties are small
order:                      int; order for the gaussian notch filter used to remove the carrier band (recommend 6)
wid:                        float; width of the gaussian notch filter used to remove the carrier band (recommend 1e8)
lam:                        float; wavelength of the target laser
C0:                         float; bulk wavespeed of the sample
density:                    float; density of the sample
delta_rho:                  float; uncertainty in density of the sample
delta_C0:                   float; uncertainty in the bulk wavespeed of the sample
delta_lam:                  float; uncertainty in the wavelength of the target laser
theta:                      float; angle of incidence of the PDV probe
delta_theta:                float; uncertainty in the angle of incidence of the PDV probe
exp_data_dir:               str; directory from which to read the experimental data file
out_files_dir:              str; directory to save output data to
display_plots:              str; 'yes' to display the final plots and 'no' to not display them. if save_data='yes'
                                     and and display_plots='no' the plots will be saved but not displayed
spall_calculation:          str; 'yes' to run the calculations for the spall analysis and 'no' to extract the velocity
                                  without doing the spall analysis
plot_figsize:               tuple; figure size for the final plots
plot_dpi:                   float; dpi for the final plots
"""


#%%
from alpss_main import *
import os


alpss_main(
    filename="example_file.csv",
    save_data="yes",
    save_all_plots="no",
    start_time_user="none",
    header_lines=5,
    time_to_skip=0e-6,
    time_to_take=10e-6,
    t_before=10e-9,
    t_after=60e-9,
    start_time_correction=0e-9,
    freq_min=1e9,
    freq_max=3.5e9,
    smoothing_window=601,
    smoothing_wid=3,
    smoothing_amp=1,
    smoothing_sigma=1,
    smoothing_mu=0,
    pb_neighbors=00,
    pb_idx_correction=0,
    rc_neighbors=400,
    rc_idx_correction=0,
    sample_rate=128e9,
    nperseg=512,
    noverlap=400,
    nfft=5120,
    window="hann",
    blur_kernel=(5, 5),
    blur_sigx=0,
    blur_sigy=0,
    carrier_band_time=250e-9,
    cmap="viridis",
    uncert_mult=10,
    order=6,
    wid=15e4,
    lam=1550.016e-9,
    C0=3950,
    density=8960,
    delta_rho=9,
    delta_C0=23,
    delta_lam=8e-18,
    theta=0,
    delta_theta=5,
    exp_data_dir="/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/ALPSS_SPADE_combo/ALPSS/input_data",
    out_files_dir="/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/ALPSS_SPADE_combo/ALPSS/output_data",
    display_plots="no",
    spall_calculation="yes",
    plot_figsize=(30, 10),
    plot_dpi=300,
)

# Print summary of outputs
print("\n" + "="*60)
print("ALPSS PROCESSING COMPLETE")
print("="*60)
print(f"Input file: {os.path.join('/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/ALPSS_SPADE_combo/ALPSS/input_data', 'example_file.csv')}")
print(f"Output directory: /Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/ALPSS_SPADE_combo/ALPSS/output_data")
print("\nGenerated output files:")
print("-" * 40)

# List all files in output directory
output_dir = "/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/ALPSS_SPADE_combo/ALPSS/output_data"
if os.path.exists(output_dir):
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') or f.endswith('.png')]
    for i, file in enumerate(sorted(output_files), 1):
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"{i:2d}. {file:<35} ({file_size:,} bytes)")
else:
    print("No output directory found!")

print("\n" + "="*60)
  # %%
