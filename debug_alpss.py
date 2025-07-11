#!/usr/bin/env python3
"""
Debug script to run ALPSS and see exactly where it's failing
"""
import sys
import traceback
import os

# Add current directory to path
sys.path.append('.')

try:
    print("=== Starting ALPSS Debug Run ===")
    print("Current working directory:", os.getcwd())
    
    from ALPSS.alpss_main import alpss_main
    
    print("ALPSS imported successfully")
    
    # Test parameters
    params = {
        'filename': 'example_file.csv',
        'save_data': 'yes',
        'start_time_user': 'none',
        'header_lines': 5,
        'time_to_skip': 5e-6,
        'time_to_take': 2e-6,
        't_before': 10e-9,
        't_after': 60e-9,
        'start_time_correction': 0e-9,
        'freq_min': 1e9,
        'freq_max': 3.5e9,
        'smoothing_window': 601,
        'smoothing_wid': 3,
        'smoothing_amp': 1,
        'smoothing_sigma': 1,
        'smoothing_mu': 0,
        'pb_neighbors': 400,
        'pb_idx_correction': 0,
        'rc_neighbors': 400,
        'rc_idx_correction': 0,
        'sample_rate': 128e9,
        'nperseg': 512,
        'noverlap': 400,
        'nfft': 5120,
        'window': 'hann',
        'blur_kernel': (5, 5),
        'blur_sigx': 0,
        'blur_sigy': 0,
        'carrier_band_time': 250e-9,
        'cmap': 'viridis',
        'uncert_mult': 10,
        'order': 6,
        'wid': 15e7,
        'lam': 1550.016e-9,
        'C0': 3950,
        'density': 8960,
        'delta_rho': 9,
        'delta_C0': 23,
        'delta_lam': 8e-18,
        'theta': 0,
        'delta_theta': 5,
        'exp_data_dir': 'ALPSS/input_data',
        'out_files_dir': 'ALPSS/output_data',
        'display_plots': 'no',
        'spall_calculation': 'yes',
        'plot_figsize': (30, 10),
        'plot_dpi': 300
    }
    
    print("Parameters set, calling alpss_main...")
    print("save_data =", params['save_data'])
    
    # Run ALPSS
    alpss_main(**params)
    
    print("=== ALPSS completed successfully ===")
    
    # Check what files were created
    print("\n=== Checking output files ===")
    if os.path.exists('ALPSS/output_data'):
        files = os.listdir('ALPSS/output_data')
        csv_files = [f for f in files if f.endswith('.csv')]
        png_files = [f for f in files if f.endswith('.png')]
        
        print(f"CSV files found: {len(csv_files)}")
        for f in csv_files:
            print(f"  - {f}")
            
        print(f"PNG files found: {len(png_files)}")
        for f in png_files:
            print(f"  - {f}")
            
        # Check specifically for the vel-smooth-with-uncert file
        vel_smooth_files = [f for f in files if 'vel-smooth-with-uncert' in f]
        if vel_smooth_files:
            print(f"✓ Found vel-smooth-with-uncert file: {vel_smooth_files[0]}")
        else:
            print("✗ vel-smooth-with-uncert file NOT found")
    else:
        print("Output directory does not exist!")

except Exception as e:
    print(f"\n=== ERROR OCCURRED ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc() 