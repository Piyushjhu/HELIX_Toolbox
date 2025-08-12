#!/usr/bin/env python3
"""
Script to plot all velocity smooth data files in a single plot
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_input_path():
    """Get the input path from user or use default"""
    print("=== INPUT PATH CONFIGURATION ===")
    
    # Check if path is provided as command line argument
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        print(f"Using command line argument path: {input_path}")
    else:
        # Ask user for input path
        default_path = "./input_data"
        print(f"Enter the path to your velocity data files (or press Enter for default: {default_path}):")
        user_input = input().strip()
        
        if user_input:
            input_path = user_input
        else:
            input_path = default_path
            print(f"Using default path: {input_path}")
    
    # Validate the path exists
    if not os.path.exists(input_path):
        print(f"⚠️  Warning: Path '{input_path}' does not exist!")
        print("The script will search in the current directory and subdirectories instead.")
        input_path = "."
    
    return input_path

def find_velocity_files(input_path="."):
    """Find all velocity files in the specified path"""
    print("=== SEARCHING FOR VELOCITY FILES ===")
    print(f"Searching in: {os.path.abspath(input_path)}")
    
    # Search patterns for different velocity file types
    search_patterns = [
        os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv'),
        # os.path.join(input_path, '**/*--velocity--smooth.csv'), 
        # os.path.join(input_path, '**/*--velocity.csv'),
        # os.path.join(input_path, '**/*--vel--uncert.csv'),
        # # Also search in subdirectories if input_path is not current directory
        # os.path.join(input_path, '*--vel-smooth-with-uncert.csv'),
        # os.path.join(input_path, '*--velocity--smooth.csv'), 
        # os.path.join(input_path, '*--velocity.csv'),
        # os.path.join(input_path, '*--vel--uncert.csv')
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # Remove duplicates and filter empty files
    unique_files = list(set(all_files))
    valid_files = []
    
    print(f"Found {len(unique_files)} total velocity files:")
    for file_path in unique_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  {file_path} ({file_size} bytes)")
            
            if file_size > 0:
                valid_files.append(file_path)
            else:
                print(f"    ⚠️  Empty file - skipping")
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    print(f"\nValid files (non-empty): {len(valid_files)}")
    return valid_files

def read_velocity_file(file_path):
    """Read a velocity file and return time and velocity data"""
    try:
        # Try reading with headers first
        df = pd.read_csv(file_path)
        
        # Check if it has headers
        if 'Time' in df.columns and 'Velocity' in df.columns:
            time_data = df['Time'].values
            velocity_data = df['Velocity'].values
            has_headers = True
        elif 'time' in df.columns and 'velocity' in df.columns:
            time_data = df['time'].values
            velocity_data = df['velocity'].values
            has_headers = True
        else:
            # No headers, assume first column is time, second is velocity
            time_data = df.iloc[:, 0].values
            velocity_data = df.iloc[:, 1].values
            has_headers = False
        
        # Check if we have uncertainty data
        uncertainty_data = None
        if df.shape[1] >= 3:
            if has_headers and 'Uncertainty' in df.columns:
                uncertainty_data = df['Uncertainty'].values
            elif has_headers and 'uncertainty' in df.columns:
                uncertainty_data = df['uncertainty'].values
            else:
                uncertainty_data = df.iloc[:, 2].values
        
        return {
            'time': time_data,
            'velocity': velocity_data,
            'uncertainty': uncertainty_data,
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'shape': df.shape,
            'has_headers': has_headers
        }
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_velocity_data(data_list):
    """Analyze the velocity data to understand time scales and ranges"""
    print("\n=== VELOCITY DATA ANALYSIS ===")
    
    for data in data_list:
        if data is None:
            continue
            
        time_data = data['time']
        velocity_data = data['velocity']
        filename = data['filename']
        
        print(f"\nFile: {filename}")
        print(f"  Shape: {data['shape']}")
        print(f"  Time range: {np.min(time_data):.2e} to {np.max(time_data):.2e}")
        print(f"  Velocity range: {np.min(velocity_data):.2f} to {np.max(velocity_data):.2f} m/s")
        
        # Determine time units and convert to nanoseconds for analysis
        max_time = np.max(time_data)
        print(f"  Sample time values: {time_data[:5]}")
        
        if max_time < 1e-3:
            print(f"  Time appears to be in seconds (max: {max_time:.2e}s)")
            time_data_ns = time_data * 1e9
        elif max_time < 1e0:
            print(f"  Time appears to be in microseconds (max: {max_time:.2f}μs)")
            time_data_ns = time_data * 1e3
        elif max_time < 1e6:
            print(f"  Time appears to be in nanoseconds (max: {max_time:.2f}ns)")
            time_data_ns = time_data
        else:
            print(f"  Time appears to be in larger units (max: {max_time:.2f})")
            time_data_ns = time_data / 1e3
        
        print(f"  Time range in nanoseconds: {np.min(time_data_ns):.2f} to {np.max(time_data_ns):.2f} ns")
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(velocity_data))
        print(f"  NaN values in velocity: {nan_count}/{len(velocity_data)}")
        
        # Check for zero or very small velocities
        small_vel_count = np.sum(np.abs(velocity_data) < 1.0)
        print(f"  Velocities < 1 m/s: {small_vel_count}/{len(velocity_data)}")

def get_uncertainty_threshold():
    """Get uncertainty threshold from user"""
    print("\n=== UNCERTAINTY FILTERING ===")
    print("Enter uncertainty threshold (m/s) to filter out high-uncertainty data points.")
    print("Data points with uncertainty > threshold will be removed.")
    print("Press Enter for default threshold of 50 m/s, or enter a custom value:")
    
    try:
        user_input = input().strip()
        if user_input:
            threshold = float(user_input)
            print(f"Using custom uncertainty threshold: {threshold} m/s")
        else:
            threshold = 50.0
            print(f"Using default uncertainty threshold: {threshold} m/s")
        return threshold
    except ValueError:
        print("Invalid input. Using default threshold of 50 m/s")
        return 50.0

def plot_all_velocity_data(data_list):
    """Plot all velocity data in a single figure with zoom capabilities and uncertainty filtering"""
    print("\n=== CREATING VELOCITY PLOT ===")
    
    if not data_list:
        print("No valid velocity data to plot")
        return
    
    # Get uncertainty threshold from user
    uncertainty_threshold = get_uncertainty_threshold()
    
    # Create figure with interactive backend
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Color palette for different files
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(data_list)))
    
    # Track legend entries
    legend_entries = []
    
    # Statistics for filtering
    total_points_before = 0
    total_points_after = 0
    
    # Counter for traces
    trace_counter = 0
    trace_counter_1000ns = 0
    
    # Track trace names
    all_trace_names = []
    traces_in_1000ns = []
    
    # Set time unit to nanoseconds (all data will be converted to ns)
    time_unit = "ns"
    
    for i, data in enumerate(data_list):
        if data is None:
            continue
            
        time_data = data['time']
        velocity_data = data['velocity']
        uncertainty_data = data['uncertainty']
        filename = data['filename']
        
        # Convert time to nanoseconds consistently
        max_time = np.max(time_data)
        min_time = np.min(time_data)
        
        print(f"  ⏱️  {filename}: Original time range: {min_time:.2e} to {max_time:.2e}")
        print(f"  ⏱️  {filename}: Sample time values: {time_data[:5]}")
        
        # Improved time unit detection for seconds input
        if max_time < 1e-3:
            # Time is in seconds, convert to nanoseconds
            time_data = time_data * 1e9
            print(f"  ⏱️  {filename}: Converting time from seconds to nanoseconds")
            print(f"  ⏱️  {filename}: Converted time range: {np.min(time_data):.2f} to {np.max(time_data):.2f} ns")
        elif max_time < 1e0:
            # Time is in microseconds, convert to nanoseconds
            time_data = time_data * 1e3
            print(f"  ⏱️  {filename}: Converting time from microseconds to nanoseconds")
            print(f"  ⏱️  {filename}: Converted time range: {np.min(time_data):.2f} to {np.max(time_data):.2f} ns")
        elif max_time < 1e6:
            # Time is already in nanoseconds
            print(f"  ⏱️  {filename}: Time already in nanoseconds")
            print(f"  ⏱️  {filename}: Time range: {np.min(time_data):.2f} to {np.max(time_data):.2f} ns")
        else:
            # Time is in larger units, convert to nanoseconds
            time_data = time_data / 1e3
            print(f"  ⏱️  {filename}: Converting time to nanoseconds")
            print(f"  ⏱️  {filename}: Converted time range: {np.min(time_data):.2f} to {np.max(time_data):.2f} ns")
        
        # Load optional noise fraction file and build filtering mask (noise > 1 removed)
        noise_fraction = None
        high_noise_mask = None
        try:
            # Expect matching naming: *--vel-smooth-with-uncert.csv -> *--noise--frac.csv
            noise_file = data['filepath'].replace('--vel-smooth-with-uncert.csv', '--noise--frac.csv')
            if os.path.exists(noise_file):
                df_noise = pd.read_csv(noise_file)
                if df_noise.shape[1] >= 1:
                    noise_fraction = df_noise.iloc[:, -1].values
                    if len(noise_fraction) == len(velocity_data):
                        high_noise_mask = noise_fraction > 1.0
                        print(f"  🔇 {filename}: Removing {np.sum(high_noise_mask)} points with noise fraction > 1.0")
                    else:
                        print(f"  ⚠️  {filename}: Noise fraction length mismatch (noise={len(noise_fraction)}, vel={len(velocity_data)})")
                else:
                    print(f"  ⚠️  {filename}: Noise fraction file has insufficient columns: {os.path.basename(noise_file)}")
            else:
                print(f"  ℹ️  {filename}: No noise fraction file found, skipping noise-based filtering")
        except Exception as e:
            print(f"  ⚠️  {filename}: Could not read noise fraction file: {e}")

        # Remove NaNs and apply noise-based filtering
        valid_mask = ~np.isnan(velocity_data)
        if high_noise_mask is not None:
            valid_mask = valid_mask & (~high_noise_mask)

        time_clean = time_data[valid_mask]
        velocity_clean = velocity_data[valid_mask]
        uncertainty_clean = None

        if uncertainty_data is not None:
            uncertainty_clean = uncertainty_data[valid_mask]
        
        # Apply uncertainty filtering
        if uncertainty_clean is not None and len(uncertainty_clean) > 0:
            # Create mask for low uncertainty data
            low_uncertainty_mask = uncertainty_clean <= uncertainty_threshold
            
            # Apply the filter
            time_filtered = time_clean[low_uncertainty_mask]
            velocity_filtered = velocity_clean[low_uncertainty_mask]
            uncertainty_filtered = uncertainty_clean[low_uncertainty_mask]
            
            # Update statistics
            points_before = len(time_clean)
            points_after = len(time_filtered)
            total_points_before += points_before
            total_points_after += points_after
            
            print(f"  📊 {filename}: {points_before} → {points_after} points (filtered {points_before - points_after} high-uncertainty points)")
            
            # Use filtered data for plotting
            time_clean = time_filtered
            velocity_clean = velocity_filtered
            uncertainty_clean = uncertainty_filtered
        else:
            print(f"  📊 {filename}: No uncertainty data available, using all {len(time_clean)} points")
            total_points_before += len(time_clean)
            total_points_after += len(time_clean)
        
        if len(time_clean) == 0:
            print(f"  ⚠️  No valid data for {filename} after filtering")
            continue

        # Align traces: set t=0 at first time velocity reaches 30 m/s
        velocity_threshold = 30.0
        t0_idx = None
        for j, vel in enumerate(velocity_clean):
            if not np.isnan(vel) and vel >= velocity_threshold:
                t0_idx = j
                break
        if t0_idx is not None:
            t0 = time_clean[t0_idx]
            time_clean = time_clean - t0
            print(f"  Ⓜ️  {filename}: Aligned t=0 at {t0:.2f} ns (first ≥ {velocity_threshold} m/s)")
        else:
            print(f"  ⚠️  {filename}: Could not find velocity ≥ {velocity_threshold} m/s for alignment; using raw time")
        
        # Plot velocity trace
        color = colors[i]
        line, = ax1.plot(time_clean, velocity_clean, color=color, alpha=0.7, linewidth=1, label=filename)
        legend_entries.append(line)
        trace_counter += 1
        all_trace_names.append(filename)
        
        # Plot first 1000ns data
        mask_1000ns = time_clean <= 1000
        if np.any(mask_1000ns):
            time_1000ns = time_clean[mask_1000ns]
            velocity_1000ns = velocity_clean[mask_1000ns]
            ax2.plot(time_1000ns, velocity_1000ns, color=color, alpha=0.7, linewidth=1, label=filename)
            trace_counter_1000ns += 1
            traces_in_1000ns.append(filename)
        
        # Plot uncertainty if available
        if uncertainty_clean is not None and len(uncertainty_clean) > 0:
            if not np.all(np.isnan(uncertainty_clean)):
                ax1.fill_between(time_clean, 
                               velocity_clean - uncertainty_clean,
                               velocity_clean + uncertainty_clean,
                               color=color, alpha=0.2)
        

        
        print(f"  ✓ Plotted {filename}: {len(time_clean)} points, velocity range: {np.min(velocity_clean):.1f}-{np.max(velocity_clean):.1f} m/s")
    
    # Print filtering summary
    if total_points_before > 0:
        filtered_percentage = ((total_points_before - total_points_after) / total_points_before) * 100
        print(f"\n📈 Filtering Summary:")
        print(f"   Total points before filtering: {total_points_before:,}")
        print(f"   Total points after filtering: {total_points_after:,}")
        print(f"   Points removed: {total_points_before - total_points_after:,} ({filtered_percentage:.1f}%)")
        print(f"   Uncertainty threshold: {uncertainty_threshold} m/s")
        print(f"   Total traces plotted: {trace_counter}")
        print(f"   Traces in first 1000ns: {trace_counter_1000ns}")
        
        # Find missing traces
        missing_traces = [name for name in all_trace_names if name not in traces_in_1000ns]
        if missing_traces:
            print(f"\n📋 Missing traces (no data in first 1000ns):")
            print(f"   Count: {len(missing_traces)} traces")
            print("   Files:")
            for i, trace_name in enumerate(missing_traces, 1):
                print(f"     {i:2d}. {trace_name}")
        else:
            print(f"\n✅ All {trace_counter} traces have data in the first 1000ns")
    
    # Customize velocity vs time plot
    ax1.set_xlabel(f'Time ({time_unit}) - aligned to t=0 at 30 m/s', fontsize=12)
    ax1.set_ylabel('Velocity (m/s)', fontsize=12)
    ax1.set_title(f'All Velocity Traces (Zoomable) - {trace_counter} traces', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Customize first 1000ns plot
    ax2.set_xlabel(f'Time ({time_unit}) - aligned to t=0 at 30 m/s', fontsize=12)
    ax2.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2.set_title(f'First 1000ns Velocity Traces - {trace_counter_1000ns} traces', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlim(0, 1000)  # Set x-axis limit to 1000ns
    
    # Enable zoom and pan functionality
    fig.canvas.manager.set_window_title('Velocity Data Plot - Interactive')
    
    # Add zoom instructions
    fig.text(0.02, 0.02, 'Zoom: Mouse wheel or zoom tool\nPan: Click and drag\nReset: Double-click', 
             fontsize=8, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'all_velocity_traces.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {plot_filename}")
    
    # Show interactive plot
    plt.show(block=True)  # This will keep the plot open and interactive
    
    print("\n📊 Interactive plot displayed!")
    print("   - Use mouse wheel to zoom in/out")
    print("   - Click and drag to pan around")
    print("   - Use the toolbar buttons for additional controls")
    print("   - Double-click to reset the view")

def create_velocity_summary(data_list):
    """Create a summary of all velocity data"""
    print("\n=== CREATING VELOCITY SUMMARY ===")
    
    summary_data = []
    
    for data in data_list:
        if data is None:
            continue
            
        time_data = data['time']
        velocity_data = data['velocity']
        filename = data['filename']
        
        # Convert time to nanoseconds consistently
        max_time = np.max(time_data)
        if max_time < 1e-3:
            # Time is in seconds, convert to nanoseconds
            time_data = time_data * 1e9
        elif max_time < 1e0:
            # Time is in microseconds, convert to nanoseconds
            time_data = time_data * 1e3
        elif max_time < 1e6:
            # Time is already in nanoseconds
            pass
        else:
            # Time is in larger units, convert to nanoseconds
            time_data = time_data / 1e3
        
        # Calculate statistics
        valid_velocities = velocity_data[~np.isnan(velocity_data)]
        
        if len(valid_velocities) > 0:
            mean_velocity = np.mean(valid_velocities)
            std_velocity = np.std(valid_velocities)
            max_velocity = np.max(valid_velocities)
            min_velocity = np.min(valid_velocities)
            time_range = np.max(time_data) - np.min(time_data)
            
            summary_data.append({
                'filename': filename,
                'mean_velocity_ms': mean_velocity,
                'std_velocity_ms': std_velocity,
                'max_velocity_ms': max_velocity,
                'min_velocity_ms': min_velocity,
                'time_range_ns': time_range,
                'data_points': len(valid_velocities),
                'nan_count': np.sum(np.isnan(velocity_data))
            })
            
            print(f"  ✓ {filename}: mean={mean_velocity:.1f}±{std_velocity:.1f} m/s, range={min_velocity:.1f}-{max_velocity:.1f} m/s")
        else:
            print(f"  ⚠️  {filename}: No valid velocity data")
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_filename = 'velocity_data_summary.csv'
        summary_df.to_csv(summary_filename, index=False)
        print(f"\n✓ Saved summary to: {summary_filename}")
        
        # Display summary table
        print("\nVelocity Data Summary:")
        print(summary_df.to_string(index=False))
    
    return summary_data

def main():
    """Main function"""
    print("VELOCITY DATA PLOTTER")
    print("=" * 50)
    
    # Get input path
    input_path = get_input_path()
    
    # Find all velocity files
    velocity_files = find_velocity_files(input_path)
    
    if not velocity_files:
        print("No valid velocity files found!")
        return
    
    # Read all velocity data
    print(f"\n=== READING VELOCITY DATA ===")
    data_list = []
    for file_path in velocity_files:
        data = read_velocity_file(file_path)
        data_list.append(data)
    
    # Analyze the data
    analyze_velocity_data(data_list)
    
    # Create summary
    summary_data = create_velocity_summary(data_list)
    
    # Plot all data
    plot_all_velocity_data(data_list)
    
    print("\n" + "=" * 50)
    print("✓ VELOCITY DATA ANALYSIS COMPLETE")
    print(f"  - Found {len(velocity_files)} valid velocity files")
    print(f"  - Created velocity summary CSV")
    print(f"  - Generated combined velocity plot")

if __name__ == "__main__":
    main() 