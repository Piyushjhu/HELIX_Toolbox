#!/usr/bin/env python3
"""
Automation Paper Velocity Analysis
=================================

This script reads the violin_plot_data.csv file created by the ALPSS-SPADE GUI
and generates violin and box plots for velocity analysis.

Plots generated:
1. Maximum velocity vs waveplate angle for each material
2. Maximum velocity vs laser energy for each material

Author: Piyush Wanchoo
Date: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Try to import seaborn, but provide fallback if not available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    print("Seaborn imported successfully")
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, using matplotlib fallback")

def load_violin_plot_data(csv_path):
    """
    Load the violin plot data from CSV file.
    
    Args:
        csv_path (str): Path to the violin_plot_data.csv file
        
    Returns:
        pd.DataFrame: Loaded data with proper data types
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Print data summary
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Materials: {df['material'].unique()}")
    print(f"Waveplate angles: {sorted(df['waveplate_angle_degrees'].dropna().unique())}")
    print(f"Laser energy range: {df['laser_energy_mj'].dropna().min():.2f} - {df['laser_energy_mj'].dropna().max():.2f} mJ")
    print(f"Max velocity range: {df['max_velocity_ms'].dropna().min():.2f} - {df['max_velocity_ms'].dropna().max():.2f} m/s")
    
    return df

def create_violin_plots(df, output_dir):
    """
    Create violin plots for maximum velocity analysis.
    
    Args:
        df (pd.DataFrame): Loaded data
        output_dir (str): Output directory for plots
    """
    # Set up the plotting style
    plt.style.use('default')
    if SEABORN_AVAILABLE:
        sns.set_palette("Set2")
    else:
        # Use matplotlib default colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Maximum velocity vs waveplate angle for each material
    print("\nCreating violin plots for max velocity vs waveplate angle...")
    
    # Filter data with valid waveplate angles and max velocity
    angle_data = df.dropna(subset=['waveplate_angle_degrees', 'max_velocity_ms'])
    
    if len(angle_data) > 0:
        # Create violin plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if SEABORN_AVAILABLE:
            # Create violin plot with seaborn
            sns.violinplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                          hue='material', palette="Set2", cut=0, inner=None, split=False, 
                          linewidth=1, dodge=False, saturation=0.5, ax=ax)
            
            # Add swarm plot overlay
            sns.swarmplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                         hue='material', palette="Set2", alpha=0.5, size=4, dodge=False,
                         edgecolor='k', linewidth=1, ax=ax)
        else:
            # Fallback: create scatter plot with different markers for materials
            materials = angle_data['material'].unique()
            for i, material in enumerate(materials):
                material_data = angle_data[angle_data['material'] == material]
                ax.scatter(material_data['waveplate_angle_degrees'], material_data['max_velocity_ms'], 
                          label=material, alpha=0.7, s=50, color=colors[i % len(colors)])
        
        # Handle legend
        if SEABORN_AVAILABLE:
            # Remove duplicate legend caused by swarm
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:3], labels[:3], title="Material", fontsize=14, title_fontsize=16)
        else:
            # Simple legend for scatter plot
            ax.legend(title="Material", fontsize=14, title_fontsize=16)
        
        # Configure plot
        ax.set_xlabel("Wave Plate Angle (degrees)", fontsize=16)
        ax.set_ylabel("Maximum Velocity (m/s)", fontsize=16)
        ax.set_title("Maximum Velocity by Waveplate Angle and Material", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'max_velocity_vs_waveplate_angle_violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved violin plot: {output_path}")
        
        # Create box plot version
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if SEABORN_AVAILABLE:
            # Create box plot
            sns.boxplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                       hue='material', palette="Set2", ax=ax)
            
            # Add swarm plot overlay
            sns.swarmplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                         hue='material', palette="Set2", alpha=0.5, size=4, dodge=False,
                         edgecolor='k', linewidth=1, ax=ax)
            
            # Remove duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:3], labels[:3], title="Material", fontsize=14, title_fontsize=16)
        else:
            # Fallback: create scatter plot with different markers for materials
            materials = angle_data['material'].unique()
            for i, material in enumerate(materials):
                material_data = angle_data[angle_data['material'] == material]
                ax.scatter(material_data['waveplate_angle_degrees'], material_data['max_velocity_ms'], 
                          label=material, alpha=0.7, s=50, color=colors[i % len(colors)], marker='s')
            
            # Simple legend for scatter plot
            ax.legend(title="Material", fontsize=14, title_fontsize=16)
        
        # Configure plot
        ax.set_xlabel("Wave Plate Angle (degrees)", fontsize=16)
        ax.set_ylabel("Maximum Velocity (m/s)", fontsize=16)
        ax.set_title("Maximum Velocity by Waveplate Angle and Material (Box Plot)", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'max_velocity_vs_waveplate_angle_box.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved box plot: {output_path}")
    else:
        print("No valid data for waveplate angle analysis")
    
    # 2. Maximum velocity vs laser energy for each material
    print("\nCreating violin plots for max velocity vs laser energy...")
    
    # Filter data with valid laser energy and max velocity
    energy_data = df.dropna(subset=['laser_energy_mj', 'max_velocity_ms'])
    
    if len(energy_data) > 0:
        # Create violin plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create violin plot with seaborn
        sns.violinplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                      hue='material', palette="Set2", cut=0, inner=None, split=False, 
                      linewidth=1, dodge=False, saturation=0.5, ax=ax)
        
        # Add swarm plot overlay
        sns.swarmplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                     hue='material', palette="Set2", alpha=0.5, size=4, dodge=False,
                     edgecolor='k', linewidth=1, ax=ax)
        
        # Remove duplicate legend caused by swarm
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="Material", fontsize=14, title_fontsize=16)
        
        # Configure plot
        ax.set_xlabel("Laser Energy (mJ)", fontsize=16)
        ax.set_ylabel("Maximum Velocity (m/s)", fontsize=16)
        ax.set_title("Maximum Velocity by Laser Energy and Material", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'max_velocity_vs_laser_energy_violin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved violin plot: {output_path}")
        
        # Create box plot version
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create box plot
        sns.boxplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                   hue='material', palette="Set2", ax=ax)
        
        # Add swarm plot overlay
        sns.swarmplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                     hue='material', palette="Set2", alpha=0.5, size=4, dodge=False,
                     edgecolor='k', linewidth=1, ax=ax)
        
        # Remove duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], title="Material", fontsize=14, title_fontsize=16)
        
        # Configure plot
        ax.set_xlabel("Laser Energy (mJ)", fontsize=16)
        ax.set_ylabel("Maximum Velocity (m/s)", fontsize=16)
        ax.set_title("Maximum Velocity by Laser Energy and Material (Box Plot)", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add bounding box
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'max_velocity_vs_laser_energy_box.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved box plot: {output_path}")
    else:
        print("No valid data for laser energy analysis")

def create_material_specific_plots(df, output_dir):
    """
    Create individual plots for each material.
    
    Args:
        df (pd.DataFrame): Loaded data
        output_dir (str): Output directory for plots
    """
    print("\nCreating material-specific plots...")
    
    materials = df['material'].unique()
    
    for material in materials:
        material_data = df[df['material'] == material]
        
        if len(material_data) == 0:
            continue
            
        print(f"\nProcessing {material} ({len(material_data)} samples)...")
        
        # Waveplate angle analysis for this material
        angle_data = material_data.dropna(subset=['waveplate_angle_degrees', 'max_velocity_ms'])
        
        if len(angle_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Violin plot
            sns.violinplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                          palette="Set2", cut=0, inner=None, ax=ax1)
            sns.swarmplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                         palette="Set2", alpha=0.5, size=4, edgecolor='k', linewidth=1, ax=ax1)
            
            ax1.set_xlabel("Wave Plate Angle (degrees)", fontsize=14)
            ax1.set_ylabel("Maximum Velocity (m/s)", fontsize=14)
            ax1.set_title(f"{material} - Max Velocity vs Waveplate Angle (Violin)", fontsize=16, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Box plot
            sns.boxplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                       palette="Set2", ax=ax2)
            sns.swarmplot(data=angle_data, x='waveplate_angle_degrees', y='max_velocity_ms', 
                         palette="Set2", alpha=0.5, size=4, edgecolor='k', linewidth=1, ax=ax2)
            
            ax2.set_xlabel("Wave Plate Angle (degrees)", fontsize=14)
            ax2.set_ylabel("Maximum Velocity (m/s)", fontsize=14)
            ax2.set_title(f"{material} - Max Velocity vs Waveplate Angle (Box)", fontsize=16, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f'{material}_max_velocity_vs_waveplate_angle.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved {material} waveplate angle plots")
        
        # Laser energy analysis for this material
        energy_data = material_data.dropna(subset=['laser_energy_mj', 'max_velocity_ms'])
        
        if len(energy_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Violin plot
            sns.violinplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                          palette="Set2", cut=0, inner=None, ax=ax1)
            sns.swarmplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                         palette="Set2", alpha=0.5, size=4, edgecolor='k', linewidth=1, ax=ax1)
            
            ax1.set_xlabel("Laser Energy (mJ)", fontsize=14)
            ax1.set_ylabel("Maximum Velocity (m/s)", fontsize=14)
            ax1.set_title(f"{material} - Max Velocity vs Laser Energy (Violin)", fontsize=16, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.5)
            
            # Box plot
            sns.boxplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                       palette="Set2", ax=ax2)
            sns.swarmplot(data=energy_data, x='laser_energy_mj', y='max_velocity_ms', 
                         palette="Set2", alpha=0.5, size=4, edgecolor='k', linewidth=1, ax=ax2)
            
            ax2.set_xlabel("Laser Energy (mJ)", fontsize=14)
            ax2.set_ylabel("Maximum Velocity (m/s)", fontsize=14)
            ax2.set_title(f"{material} - Max Velocity vs Laser Energy (Box)", fontsize=16, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(output_dir, f'{material}_max_velocity_vs_laser_energy.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved {material} laser energy plots")

def main():
    """
    Main function to run the velocity analysis.
    """
    print("Automation Paper Velocity Analysis")
    print("==================================")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python automation_paper_velocity_analysis.py <csv_file_path> [output_directory]")
        print("Example: python automation_paper_velocity_analysis.py output/violin_plot_data.csv output/velocity_analysis")
        sys.exit(1)
    
    # Get input file path
    csv_path = sys.argv[1]
    
    # Get output directory (optional)
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        # Default to same directory as CSV file
        csv_dir = os.path.dirname(csv_path)
        output_dir = os.path.join(csv_dir, 'velocity_analysis')
    
    try:
        # Load data
        print(f"Loading data from: {csv_path}")
        df = load_violin_plot_data(csv_path)
        
        # Create plots
        print(f"\nCreating plots in: {output_dir}")
        create_violin_plots(df, output_dir)
        create_material_specific_plots(df, output_dir)
        
        print(f"\nAnalysis complete! Plots saved in: {output_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.pdf')):
                print(f"  - {file}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 