#!/usr/bin/env python3
"""
Diagnostic script to understand why velocity summary CSV has missing data
"""
import os
import sys
import pandas as pd
import numpy as np
import glob

def check_velocity_files():
    """Check what velocity files exist and their structure"""
    print("=== VELOCITY FILE DIAGNOSTIC ===")
    
    # Find all velocity files
    velocity_files = glob.glob('./**/*--vel-smooth-with-uncert.csv', recursive=True)
    print(f"Found {len(velocity_files)} velocity files:")
    
    for file_path in velocity_files:
        file_size = os.path.getsize(file_path)
        print(f"  {file_path} ({file_size} bytes)")
        
        if file_size > 0:
            try:
                # Try to read the file
                df = pd.read_csv(file_path, header=None)
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {df.shape[1]}")
                
                if df.shape[1] >= 3:
                    # Check time range
                    time_data = df.iloc[:, 0].values
                    print(f"    Time range: {np.min(time_data):.2e} to {np.max(time_data):.2e}")
                    
                    # Check velocity range
                    velocity_data = df.iloc[:, 1].values
                    print(f"    Velocity range: {np.min(velocity_data):.2f} to {np.max(velocity_data):.2f} m/s")
                    
                    # Check if time is in seconds or nanoseconds
                    if np.max(time_data) < 1e-6:
                        print(f"    Time appears to be in seconds (max: {np.max(time_data):.2e}s)")
                    elif np.max(time_data) < 1000:
                        print(f"    Time appears to be in nanoseconds (max: {np.max(time_data):.2f}ns)")
                    else:
                        print(f"    Time appears to be in microseconds (max: {np.max(time_data):.2f}μs)")
                        
                else:
                    print(f"    WARNING: Insufficient columns ({df.shape[1]})")
                    
            except Exception as e:
                print(f"    ERROR reading file: {e}")
        else:
            print(f"    WARNING: Empty file")
        print()

def check_input_files():
    """Check what input files exist"""
    print("=== INPUT FILE DIAGNOSTIC ===")
    
    # Check C1 files
    c1_files = glob.glob('./input_data/C1_files/*.csv')
    print(f"Found {len(c1_files)} C1 files:")
    
    for file_path in c1_files:
        file_size = os.path.getsize(file_path)
        print(f"  {file_path} ({file_size} bytes)")
        
        if file_size > 0:
            try:
                # Try to read the file
                df = pd.read_csv(file_path, header=None)
                print(f"    Shape: {df.shape}")
                if df.shape[1] > 0:
                    print(f"    First few values: {df.iloc[:5, 0].values}")
            except Exception as e:
                print(f"    ERROR reading file: {e}")
        else:
            print(f"    WARNING: Empty file")
        print()

def test_velocity_calculation():
    """Test the velocity calculation logic"""
    print("=== VELOCITY CALCULATION TEST ===")
    
    # Find a non-empty velocity file
    velocity_files = glob.glob('./**/*--vel-smooth-with-uncert.csv', recursive=True)
    
    for file_path in velocity_files:
        if os.path.getsize(file_path) > 0:
            print(f"Testing velocity calculation on: {file_path}")
            
            try:
                # Read velocity data
                df = pd.read_csv(file_path, header=None)
                if df.shape[1] < 3:
                    print(f"  ERROR: Insufficient columns ({df.shape[1]})")
                    continue
                
                time_data = df.iloc[:, 0].values
                velocity_data = df.iloc[:, 1].values
                uncertainty_data = df.iloc[:, 2].values
                
                print(f"  Original data shape: {df.shape}")
                print(f"  Time range: {np.min(time_data):.2e} to {np.max(time_data):.2e}")
                print(f"  Velocity range: {np.min(velocity_data):.2f} to {np.max(velocity_data):.2f} m/s")
                
                # Convert time to ns if needed
                if np.nanmax(time_data) < 1.0:
                    time_data = time_data * 1e9
                    print(f"  Converted time to nanoseconds")
                
                # Test velocity threshold detection
                velocity_threshold = 30.0  # m/s
                t0_idx = None
                
                # Find first point where velocity exceeds threshold
                for i, vel in enumerate(velocity_data):
                    if not np.isnan(vel) and vel >= velocity_threshold:
                        t0_idx = i
                        break
                
                if t0_idx is None:
                    print(f"  WARNING: Could not find velocity threshold {velocity_threshold} m/s")
                    print(f"  Maximum velocity: {np.max(velocity_data):.2f} m/s")
                    time_aligned = time_data
                else:
                    t0 = time_data[t0_idx]
                    time_aligned = time_data - t0
                    print(f"  Found t0 at index {t0_idx}: {t0:.2f} ns")
                
                # Test time window calculations
                mask_300_400 = (time_aligned >= 300) & (time_aligned <= 400)
                velocities_300_400 = velocity_data[mask_300_400]
                velocities_300_400 = velocities_300_400[~np.isnan(velocities_300_400)]
                
                print(f"  Data points in 300-400ns window: {len(velocities_300_400)}")
                
                if len(velocities_300_400) > 0:
                    mean_velocity = np.mean(velocities_300_400)
                    print(f"  Mean velocity (300-400ns): {mean_velocity:.2f} m/s")
                else:
                    print(f"  WARNING: No data in 300-400ns window")
                    
                    # Try fallback windows
                    mask_200_300 = (time_aligned >= 200) & (time_aligned <= 300)
                    velocities_200_300 = velocity_data[mask_200_300]
                    velocities_200_300 = velocities_200_300[~np.isnan(velocities_200_300)]
                    
                    if len(velocities_200_300) > 0:
                        mean_velocity = np.mean(velocities_200_300)
                        print(f"  Mean velocity (200-300ns fallback): {mean_velocity:.2f} m/s")
                    else:
                        mask_400_500 = (time_aligned >= 400) & (time_aligned <= 500)
                        velocities_400_500 = velocity_data[mask_400_500]
                        velocities_400_500 = velocities_400_500[~np.isnan(velocities_400_500)]
                        
                        if len(velocities_400_500) > 0:
                            mean_velocity = np.mean(velocities_400_500)
                            print(f"  Mean velocity (400-500ns fallback): {mean_velocity:.2f} m/s")
                        else:
                            print(f"  ERROR: No data in any time window")
                
            except Exception as e:
                print(f"  ERROR: {e}")
            
            print()
            break  # Only test the first valid file

def check_output_directory():
    """Check what's in the output directory"""
    print("=== OUTPUT DIRECTORY DIAGNOSTIC ===")
    
    output_dirs = ['./output', './ALPSS/output_data']
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"Contents of {output_dir}:")
            files = os.listdir(output_dir)
            for file in files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")
            print()
        else:
            print(f"Directory {output_dir} does not exist")

if __name__ == "__main__":
    print("VELOCITY SUMMARY DIAGNOSTIC")
    print("=" * 50)
    
    check_velocity_files()
    check_input_files()
    test_velocity_calculation()
    check_output_directory()
    
    print("Diagnostic complete!") 