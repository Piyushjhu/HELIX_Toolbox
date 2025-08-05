#!/usr/bin/env python3
"""
ALPSS Output and SPADE Input Debugging Script
This script helps identify issues with ALPSS output generation and SPADE input file creation.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

def check_alpss_outputs(output_dir):
    """Check what ALPSS outputs are being generated"""
    print("=== ALPSS Output Analysis ===")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(output_dir):
        print("❌ Output directory does not exist!")
        return False
    
    # Check for all possible ALPSS output files
    expected_files = [
        '*--velocity--smooth.csv',
        '*--vel-smooth-with-uncert.csv', 
        '*--results.csv',
        '*--inputs.csv',
        '*--voltage.csv',
        '*--plots.png',
        '*--velocity_with_uncertainty.png',
        '*--iq_analysis.png',
        '*--velocity_comparison.png',
        '*--noise_analysis.png',
        '*--imported_spectrogram.png',
        '*--roi_spectrogram.png',
        '*--filtered_roi_spectrogram.png',
        '*--thresholded_spectrogram.png',
        '*--velocity_spectrogram_overlay.png',
        '*--voltage_data.png',
        '*--voltage_roi.png'
    ]
    
    found_files = []
    missing_files = []
    
    for pattern in expected_files:
        files = glob.glob(os.path.join(output_dir, pattern))
        if files:
            found_files.extend(files)
            print(f"✅ Found {len(files)} files matching {pattern}")
            for f in files:
                size = os.path.getsize(f)
                print(f"   - {os.path.basename(f)} ({size} bytes)")
        else:
            missing_files.append(pattern)
            print(f"❌ No files found matching {pattern}")
    
    print(f"\n📊 Summary:")
    print(f"   Found files: {len(found_files)}")
    print(f"   Missing patterns: {len(missing_files)}")
    
    return len(found_files) > 0

def check_spade_inputs(output_dir):
    """Check what SPADE input files are available"""
    print("\n=== SPADE Input Analysis ===")
    
    # Look for SPADE input files
    spade_input_patterns = [
        '*--vel-smooth-with-uncert.csv',  # Primary SPADE input
        '*--velocity--smooth.csv',        # Alternative SPADE input
    ]
    
    spade_input_files = []
    for pattern in spade_input_patterns:
        files = glob.glob(os.path.join(output_dir, pattern))
        spade_input_files.extend(files)
    
    print(f"Found {len(spade_input_files)} potential SPADE input files:")
    for f in spade_input_files:
        size = os.path.getsize(f)
        print(f"   - {os.path.basename(f)} ({size} bytes)")
    
    # Check SPADE output directory
    spade_output_dir = os.path.join(output_dir, "SPADE_analysis")
    if os.path.exists(spade_output_dir):
        spade_outputs = glob.glob(os.path.join(spade_output_dir, "*"))
        print(f"\nSPADE output directory exists with {len(spade_outputs)} files")
        for f in spade_outputs:
            if os.path.isfile(f):
                size = os.path.getsize(f)
                print(f"   - {os.path.basename(f)} ({size} bytes)")
    else:
        print(f"\n❌ SPADE output directory does not exist: {spade_output_dir}")
    
    return len(spade_input_files) > 0

def analyze_failed_files(output_dir):
    """Analyze which files failed to generate SPADE inputs"""
    print("\n=== Failed Files Analysis ===")
    
    # Get all CSV files in output directory
    all_csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    successful_files = []
    failed_files = []
    
    for csv_file in all_csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        # Check if this file has corresponding SPADE input
        spade_input = os.path.join(output_dir, f"{base_name}--vel-smooth-with-uncert.csv")
        
        if os.path.exists(spade_input):
            successful_files.append(base_name)
        else:
            failed_files.append(base_name)
    
    print(f"Successful files (have SPADE input): {len(successful_files)}")
    for f in successful_files:
        print(f"   ✅ {f}")
    
    print(f"\nFailed files (missing SPADE input): {len(failed_files)}")
    for f in failed_files:
        print(f"   ❌ {f}")
    
    return failed_files

def check_file_contents(output_dir):
    """Check the contents of key files to ensure they're valid"""
    print("\n=== File Content Analysis ===")
    
    # Check velocity files
    velocity_files = glob.glob(os.path.join(output_dir, "*--velocity--smooth.csv"))
    for vf in velocity_files:
        try:
            df = pd.read_csv(vf)
            print(f"✅ {os.path.basename(vf)}: {df.shape[0]} rows, {df.shape[1]} columns")
            if df.shape[0] > 0:
                print(f"   Time range: {df.iloc[0, 0]:.2e} to {df.iloc[-1, 0]:.2e}")
                print(f"   Velocity range: {df.iloc[:, 1].min():.2f} to {df.iloc[:, 1].max():.2f}")
        except Exception as e:
            print(f"❌ {os.path.basename(vf)}: Error reading file - {e}")
    
    # Check results files
    results_files = glob.glob(os.path.join(output_dir, "*--results.csv"))
    for rf in results_files:
        try:
            df = pd.read_csv(rf)
            print(f"✅ {os.path.basename(rf)}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"❌ {os.path.basename(rf)}: Error reading file - {e}")

def main():
    """Main debugging function"""
    print("🔍 ALPSS Output and SPADE Input Debugging Tool")
    print("=" * 50)
    
    # Check common output directories
    output_dirs = [
        "output",
        "ALPSS/output_data", 
        "test_output",
        "ALPSS/test_output"
    ]
    
    found_outputs = False
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"\n📁 Found output directory: {output_dir}")
            found_outputs = True
            
            # Check ALPSS outputs
            alpss_ok = check_alpss_outputs(output_dir)
            
            # Check SPADE inputs
            spade_ok = check_spade_inputs(output_dir)
            
            # Analyze failed files
            failed_files = analyze_failed_files(output_dir)
            
            # Check file contents
            check_file_contents(output_dir)
            
            print(f"\n📋 Summary for {output_dir}:")
            print(f"   ALPSS outputs: {'✅' if alpss_ok else '❌'}")
            print(f"   SPADE inputs: {'✅' if spade_ok else '❌'}")
            print(f"   Failed files: {len(failed_files)}")
    
    if not found_outputs:
        print("❌ No output directories found!")
        print("Common locations checked:")
        for d in output_dirs:
            print(f"   - {d}")
    
    print("\n💡 Recommendations:")
    print("1. If ALPSS outputs are missing, check the GUI parameter settings")
    print("2. If SPADE inputs are missing, ensure ALPSS completed successfully")
    print("3. Check that all required parameters are set in the GUI")
    print("4. Verify that input files are in the correct format")

if __name__ == "__main__":
    main() 