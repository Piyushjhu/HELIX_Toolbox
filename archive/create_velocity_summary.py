#!/usr/bin/env python3
"""
Script to create a proper velocity summary using the correct velocity files
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_best_velocity_files():
    """Find the best velocity files for analysis"""
    print("=== FINDING BEST VELOCITY FILES ===")
    
    # Priority order for velocity files
    file_patterns = [
        './**/*--vel-smooth-with-uncert.csv',  # Best: smoothed with uncertainty
        './**/*--velocity--smooth.csv',        # Good: smoothed velocity
        './**/*--velocity.csv'                 # Raw: velocity data
    ]
    
    found_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if os.path.getsize(file_path) > 0:
                found_files.append(file_path)
    
    # Remove duplicates and filter by quality
    unique_files = list(set(found_files))
    quality_files = []
    
    print(f"Found {len(unique_files)} velocity files:")
    for file_path in unique_files:
        try:
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] >= 2:
                velocity_data = df.iloc[:, 1].values
                mean_vel = np.mean(velocity_data[~np.isnan(velocity_data)])
                
                # Only include files with reasonable velocity ranges (>10 m/s)
                if mean_vel > 10:
                    quality_files.append(file_path)
                    print(f"  ✓ {os.path.basename(file_path)}: mean={mean_vel:.1f} m/s")
                else:
                    print(f"  ⚠️  {os.path.basename(file_path)}: mean={mean_vel:.1f} m/s (too low)")
        except Exception as e:
            print(f"  ✗ {os.path.basename(file_path)}: Error reading file")
    
    print(f"\nQuality files (mean velocity > 10 m/s): {len(quality_files)}")
    return quality_files

def calculate_velocity_statistics(file_path):
    """Calculate velocity statistics for a single file"""
    try:
        # Read velocity data
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            return None
        
        time_data = df.iloc[:, 0].values
        velocity_data = df.iloc[:, 1].values
        
        # Convert time to nanoseconds if needed
        if np.max(time_data) < 1e-6:
            time_data = time_data * 1e9
        
        # Remove NaN values
        valid_mask = ~np.isnan(velocity_data)
        time_clean = time_data[valid_mask]
        velocity_clean = velocity_data[valid_mask]
        
        if len(velocity_clean) == 0:
            return None
        
        # Calculate basic statistics
        mean_velocity = np.mean(velocity_clean)
        std_velocity = np.std(velocity_clean)
        max_velocity = np.max(velocity_clean)
        min_velocity = np.min(velocity_clean)
        
        # Calculate time-aligned statistics
        time_range = np.max(time_clean) - np.min(time_clean)
        
        # Find velocity threshold crossing (30 m/s)
        velocity_threshold = 30.0
        t0_idx = None
        for i, vel in enumerate(velocity_clean):
            if vel >= velocity_threshold:
                t0_idx = i
                break
        
        if t0_idx is not None:
            t0 = time_clean[t0_idx]
            time_aligned = time_clean - t0
        else:
            t0 = np.nan
            time_aligned = time_clean
        
        # Use adaptive time window for mean calculation
        time_span = np.max(time_aligned) - np.min(time_aligned)
        
        if time_span > 1000:  # Long time range
            mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
            window_start = mid_time - 50
            window_end = mid_time + 50
            time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
        elif time_span > 100:  # Medium time range
            mid_time = (np.min(time_aligned) + np.max(time_aligned)) / 2
            window_start = mid_time - 50
            window_end = mid_time + 50
            time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (adaptive)"
        else:  # Short time range
            window_start = np.min(time_aligned)
            window_end = np.max(time_aligned)
            time_window_used = f"{window_start:.0f}-{window_end:.0f}ns (full range)"
        
        # Calculate mean in the selected window
        mask_window = (time_aligned >= window_start) & (time_aligned <= window_end)
        velocities_window = velocity_clean[mask_window]
        
        if len(velocities_window) > 0:
            mean_velocity_window = np.mean(velocities_window)
        else:
            # Fallback to all data
            mean_velocity_window = mean_velocity
            time_window_used = f"All data ({len(velocity_clean)} points)"
        
        # Get file base name
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        return {
            'file_name': base_name,
            'mean_velocity_300_400ns_ms': mean_velocity_window,
            'time_window_used': time_window_used,
            'mean_velocity_all_ms': mean_velocity,
            'std_velocity_ms': std_velocity,
            'max_velocity_ms': max_velocity,
            'min_velocity_ms': min_velocity,
            'time_range_ns': time_span,
            'data_points': len(velocity_clean),
            't0_ns': t0,
            'velocity_threshold_ms': velocity_threshold
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_velocity_summary():
    """Create a comprehensive velocity summary"""
    print("=== CREATING VELOCITY SUMMARY ===")
    
    # Find quality velocity files
    velocity_files = find_best_velocity_files()
    
    if not velocity_files:
        print("No quality velocity files found!")
        return
    
    # Calculate statistics for each file
    summary_data = []
    for file_path in velocity_files:
        stats = calculate_velocity_statistics(file_path)
        if stats:
            summary_data.append(stats)
            print(f"  ✓ {stats['file_name']}: {stats['mean_velocity_300_400ns_ms']:.1f} m/s ({stats['time_window_used']})")
    
    if not summary_data:
        print("No valid velocity statistics calculated!")
        return
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Reorder columns
    standard_cols = ['file_name', 'mean_velocity_300_400ns_ms', 'time_window_used', 
                    'mean_velocity_all_ms', 'std_velocity_ms', 'max_velocity_ms', 
                    'min_velocity_ms', 'time_range_ns', 'data_points', 't0_ns', 
                    'velocity_threshold_ms']
    
    # Ensure all columns exist
    for col in standard_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
    
    # Reorder columns
    summary_df = summary_df[standard_cols]
    
    # Save summary
    summary_filename = 'velocity_summary_final.csv'
    summary_df.to_csv(summary_filename, index=False)
    
    print(f"\n✓ Created velocity summary with {len(summary_data)} entries")
    print(f"  Saved to: {summary_filename}")
    
    # Display summary
    print("\nVelocity Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def plot_velocity_summary(summary_df):
    """Create plots of the velocity summary"""
    print("\n=== CREATING VELOCITY PLOTS ===")
    
    if summary_df.empty:
        print("No data to plot!")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean velocity distribution
    mean_velocities = summary_df['mean_velocity_300_400ns_ms'].values
    ax1.hist(mean_velocities, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Mean Velocity (m/s)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Mean Velocities', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity range vs mean
    velocity_ranges = summary_df['max_velocity_ms'] - summary_df['min_velocity_ms']
    ax2.scatter(mean_velocities, velocity_ranges, alpha=0.7, color='red')
    ax2.set_xlabel('Mean Velocity (m/s)', fontsize=12)
    ax2.set_ylabel('Velocity Range (m/s)', fontsize=12)
    ax2.set_title('Velocity Range vs Mean Velocity', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time range vs mean velocity
    time_ranges = summary_df['time_range_ns'].values
    ax3.scatter(time_ranges, mean_velocities, alpha=0.7, color='green')
    ax3.set_xlabel('Time Range (ns)', fontsize=12)
    ax3.set_ylabel('Mean Velocity (m/s)', fontsize=12)
    ax3.set_title('Time Range vs Mean Velocity', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data points vs mean velocity
    data_points = summary_df['data_points'].values
    ax4.scatter(data_points, mean_velocities, alpha=0.7, color='purple')
    ax4.set_xlabel('Number of Data Points', fontsize=12)
    ax4.set_ylabel('Mean Velocity (m/s)', fontsize=12)
    ax4.set_title('Data Points vs Mean Velocity', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'velocity_summary_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved velocity summary plots to: {plot_filename}")
    
    plt.show()

def main():
    """Main function"""
    print("VELOCITY SUMMARY CREATOR")
    print("=" * 50)
    
    # Create velocity summary
    summary_df = create_velocity_summary()
    
    if summary_df is not None and not summary_df.empty:
        # Create plots
        plot_velocity_summary(summary_df)
        
        print("\n" + "=" * 50)
        print("✓ VELOCITY SUMMARY CREATION COMPLETE")
        print(f"  - Processed {len(summary_df)} velocity files")
        print(f"  - Created velocity summary CSV")
        print(f"  - Generated summary plots")
        
        # Check for missing values
        missing_velocity = summary_df['mean_velocity_300_400ns_ms'].isna().sum()
        if missing_velocity == 0:
            print(f"  - ✅ No missing velocity values")
        else:
            print(f"  - ⚠️  {missing_velocity} missing velocity values")
    else:
        print("❌ Failed to create velocity summary")

if __name__ == "__main__":
    main() 