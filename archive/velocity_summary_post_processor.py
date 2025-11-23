#!/usr/bin/env python3
"""
Velocity Summary Post-Processor
==============================

This script takes the velocity shots summary CSV file as input and creates
box plots of shot time vs material and PDV Return Power vs material.

Usage:
    python velocity_summary_post_processor.py --input velocity_shots_summary.csv --output output_directory

Features:
- Reads velocity shots summary CSV
- Creates box plots of shot time vs material
- Creates box plots of PDV Return Power vs material
- Handles different material column names
- Saves plots in high resolution
- Generates statistics summary
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class VelocitySummaryPostProcessor:
    """Post-processor for velocity shots summary data"""
    
    def __init__(self, input_file, output_dir):
        """
        Initialize the post-processor
        
        Args:
            input_file: Path to velocity shots summary CSV file
            output_dir: Directory to save output plots
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get input file directory for saving plots there too
        self.input_dir = Path(input_file).parent
        
        # Load data
        self.df = self._load_data()
        
    def _load_data(self):
        """Load and validate the velocity summary data"""
        try:
            df = pd.read_csv(self.input_file)
            print(f"✅ Loaded {len(df)} rows from {self.input_file}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error loading {self.input_file}: {e}")
            sys.exit(1)
    
    def _find_material_column(self):
        """Find the material column in the dataframe"""
        material_columns = [
            'Flyer_material', 'Sample material', 'sample_material', 
            'material', 'Material', 'Flyer Material'
        ]
        
        for col in material_columns:
            if col in self.df.columns:
                print(f"✅ Found material column: {col}")
                return col
        
        print("❌ No material column found. Available columns:")
        for col in self.df.columns:
            print(f"  - {col}")
        return None
    
    def _find_shot_time_column(self):
        """Find the shot time column in the dataframe"""
        shot_time_columns = [
            'Shot_Time (seconds)', 'Shot_Time', 'shot_time', 
            'Shot Time', 'shot time', 'Time (seconds)'
        ]
        
        for col in shot_time_columns:
            if col in self.df.columns:
                print(f"✅ Found shot time column: {col}")
                return col
        
        print("❌ No shot time column found. Available columns:")
        for col in self.df.columns:
            print(f"  - {col}")
        return None
    
    def _find_pdv_return_power_column(self):
        """Find the PDV Return Power column in the dataframe"""
        pdv_power_columns = [
            'PDV_Return_Power (dBm)', 'PDV_Return_Power', 'pdv_return_power',
            'PDV Return Power (dBm)', 'PDV Return Power', 'pdv return power',
            'Return_Power (dBm)', 'Return_Power', 'return_power',
            'PDV_Power (dBm)', 'PDV_Power', 'pdv_power'
        ]
        
        for col in pdv_power_columns:
            if col in self.df.columns:
                print(f"✅ Found PDV Return Power column: {col}")
                return col
        
        print("❌ No PDV Return Power column found. Available columns:")
        for col in self.df.columns:
            print(f"  - {col}")
        return None
    
    def create_shot_time_vs_material_boxplot(self):
        """Create box plot of shot time vs material"""
        print("\n📊 Creating shot time vs material box plot...")
        
        # Find required columns
        material_col = self._find_material_column()
        shot_time_col = self._find_shot_time_column()
        
        if not material_col or not shot_time_col:
            print("❌ Missing required columns for box plot")
            return
        
        # Clean data
        df_clean = self.df[[material_col, shot_time_col]].dropna()
        
        if len(df_clean) == 0:
            print("❌ No valid data for box plot")
            return
        
        # Get unique materials
        materials = df_clean[material_col].unique()
        print(f"📋 Materials found: {materials}")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        box_data = [df_clean[df_clean[material_col] == mat][shot_time_col].values 
                   for mat in materials]
        
        # Create box plot with custom styling
        bp = plt.boxplot(box_data, labels=materials, patch_artist=True)
        
        # Color the boxes and make them thicker
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for patch, color in zip(bp['boxes'], colors[:len(materials)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(5)  # Increase box border thickness
        
        # Make median lines thicker
        for median in bp['medians']:
            median.set_linewidth(5)
        
        # Make whiskers thicker
        for whisker in bp['whiskers']:
            whisker.set_linewidth(4)
        
        # Make caps thicker
        for cap in bp['caps']:
            cap.set_linewidth(4)
        
        # Style the plot with larger fonts
        plt.title('Shot Time vs Material', fontsize=24, pad=20)
        plt.xlabel('Material', fontsize=30)
        plt.ylabel('Shot Time (seconds)', fontsize=30)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Increase tick label font sizes
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        
        # Rotate x-axis labels if needed
        if len(materials) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Add statistics
        stats_text = []
        for i, mat in enumerate(materials):
            data = box_data[i]
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                stats_text.append(f"{mat}: {mean_val:.3f}±{std_val:.3f}s (n={len(data)})")
        
        # Add statistics as text box with larger font in top right
        stats_str = "\n".join(stats_text)
        plt.figtext(0.98, 0.98, f"Statistics:\n{stats_str}", 
                   fontsize=24, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   ha='right', va='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot in both output directory and input file directory (PNG and PDF)
        output_file_png = self.output_dir / "shot_time_vs_material_boxplot.png"
        input_dir_file_png = self.input_dir / "shot_time_vs_material_boxplot.png"
        output_file_pdf = self.output_dir / "shot_time_vs_material_boxplot.pdf"
        input_dir_file_pdf = self.input_dir / "shot_time_vs_material_boxplot.pdf"
        
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(input_dir_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, bbox_inches='tight', format='pdf')
        plt.savefig(input_dir_file_pdf, bbox_inches='tight', format='pdf')
        
        print(f"✅ Saved box plot to: {output_file_png}")
        print(f"✅ Saved box plot to: {input_dir_file_png}")
        print(f"✅ Saved box plot to: {output_file_pdf}")
        print(f"✅ Saved box plot to: {input_dir_file_pdf}")
        
        # Show plot
        plt.show()
        
        return output_file_png
    
    def create_pdv_return_power_vs_material_boxplot(self):
        """Create box plot of PDV Return Power vs material"""
        print("\n📊 Creating PDV Return Power vs material box plot...")
        
        # Find required columns
        material_col = self._find_material_column()
        pdv_power_col = self._find_pdv_return_power_column()
        
        if not material_col or not pdv_power_col:
            print("❌ Missing required columns for PDV Return Power box plot")
            return
        
        # Clean data
        df_clean = self.df[[material_col, pdv_power_col]].dropna()
        
        if len(df_clean) == 0:
            print("❌ No valid data for PDV Return Power box plot")
            return
        
        # Get unique materials
        materials = df_clean[material_col].unique()
        print(f"📋 Materials found: {materials}")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        box_data = [df_clean[df_clean[material_col] == mat][pdv_power_col].values 
                   for mat in materials]
        
        # Create box plot with custom styling
        bp = plt.boxplot(box_data, labels=materials, patch_artist=True)
        
        # Color the boxes and make them thicker
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for patch, color in zip(bp['boxes'], colors[:len(materials)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(5)  # Increase box border thickness
        
        # Make median lines thicker
        for median in bp['medians']:
            median.set_linewidth(5)
        
        # Make whiskers thicker
        for whisker in bp['whiskers']:
            whisker.set_linewidth(4)
        
        # Make caps thicker
        for cap in bp['caps']:
            cap.set_linewidth(4)
        
        # Style the plot with larger fonts
        plt.title('PDV Return Power vs Material', fontsize=24, pad=20)
        plt.xlabel('Material', fontsize=30)
        plt.ylabel('PDV Return Power (dBm)', fontsize=30)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Increase tick label font sizes
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        
        # Rotate x-axis labels if needed
        if len(materials) > 4:
            plt.xticks(rotation=45, ha='right')
        
        # Add statistics
        stats_text = []
        for i, mat in enumerate(materials):
            data = box_data[i]
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                stats_text.append(f"{mat}: {mean_val:.2f}±{std_val:.2f} dBm (n={len(data)})")
        
        # Add statistics as text box with larger font in top right
        stats_str = "\n".join(stats_text)
        plt.figtext(0.98, 0.98, f"Statistics:\n{stats_str}", 
                   fontsize=24, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                   ha='right', va='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot in both output directory and input file directory (PNG and PDF)
        output_file_png = self.output_dir / "pdv_return_power_vs_material_boxplot.png"
        input_dir_file_png = self.input_dir / "pdv_return_power_vs_material_boxplot.png"
        output_file_pdf = self.output_dir / "pdv_return_power_vs_material_boxplot.pdf"
        input_dir_file_pdf = self.input_dir / "pdv_return_power_vs_material_boxplot.pdf"
        
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(input_dir_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, bbox_inches='tight', format='pdf')
        plt.savefig(input_dir_file_pdf, bbox_inches='tight', format='pdf')
        
        print(f"✅ Saved PDV Return Power box plot to: {output_file_png}")
        print(f"✅ Saved PDV Return Power box plot to: {input_dir_file_png}")
        print(f"✅ Saved PDV Return Power box plot to: {output_file_pdf}")
        print(f"✅ Saved PDV Return Power box plot to: {input_dir_file_pdf}")
        
        # Show plot
        plt.show()
        
        return output_file_png
    
    def create_enhanced_boxplot(self):
        """Create enhanced box plot with additional statistics"""
        print("\n📊 Creating enhanced shot time vs material box plot...")
        
        # Find required columns
        material_col = self._find_material_column()
        shot_time_col = self._find_shot_time_column()
        
        if not material_col or not shot_time_col:
            print("❌ Missing required columns for enhanced box plot")
            return
        
        # Clean data
        df_clean = self.df[[material_col, shot_time_col]].dropna()
        
        if len(df_clean) == 0:
            print("❌ No valid data for enhanced box plot")
            return
        
        # Create figure with single plot
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 10))
        
        # Get unique materials
        materials = df_clean[material_col].unique()
        
        # Create box plot
        box_data = [df_clean[df_clean[material_col] == mat][shot_time_col].values 
                   for mat in materials]
        
        # Create box plot with custom styling
        bp = ax1.boxplot(box_data, labels=materials, patch_artist=True)
        
        # Color the boxes and make them thicker
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for patch, color in zip(bp['boxes'], colors[:len(materials)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(5)  # Increase box border thickness
        
        # Make median lines thicker
        for median in bp['medians']:
            median.set_linewidth(5)
        
        # Make whiskers thicker
        for whisker in bp['whiskers']:
            whisker.set_linewidth(4)
        
        # Make caps thicker
        for cap in bp['caps']:
            cap.set_linewidth(4)
        
        # Style the main plot with larger fonts
        ax1.set_title('Shot Time vs Material', fontsize=30, pad=20)
        ax1.set_ylabel('Shot Time (seconds)', fontsize=30)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Increase tick label font sizes
        ax1.tick_params(axis='both', labelsize=30)
        
        # Rotate x-axis labels if needed
        if len(materials) > 4:
            ax1.tick_params(axis='x', rotation=45)
        
        # Create statistics for CSV file
        stats_data = []
        for i, mat in enumerate(materials):
            data = box_data[i]
            if len(data) > 0:
                stats_data.append({
                    'Material': mat,
                    'Count': len(data),
                    'Mean (s)': f"{np.mean(data):.2f}",
                    'Std (s)': f"{np.std(data):.2f}",
                    'Min (s)': f"{np.min(data):.2f}",
                    'Max (s)': f"{np.max(data):.2f}",
                    'Median (s)': f"{np.median(data):.2f}"
                })
        
        # Save statistics to CSV file
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            # Save statistics in both output directory and input file directory
            output_file = self.output_dir / "enhanced_plot_statistics.csv"
            input_dir_file = self.input_dir / "enhanced_plot_statistics.csv"
            
            stats_df.to_csv(output_file, index=False)
            stats_df.to_csv(input_dir_file, index=False)
            
            print(f"✅ Saved enhanced plot statistics to: {output_file}")
            print(f"✅ Saved enhanced plot statistics to: {input_dir_file}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot in both output directory and input file directory (PNG and PDF)
        output_file_png = self.output_dir / "shot_time_vs_material_enhanced.png"
        input_dir_file_png = self.input_dir / "shot_time_vs_material_enhanced.png"
        output_file_pdf = self.output_dir / "shot_time_vs_material_enhanced.pdf"
        input_dir_file_pdf = self.input_dir / "shot_time_vs_material_enhanced.pdf"
        
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(input_dir_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, bbox_inches='tight', format='pdf')
        plt.savefig(input_dir_file_pdf, bbox_inches='tight', format='pdf')
        
        print(f"✅ Saved enhanced box plot to: {output_file_png}")
        print(f"✅ Saved enhanced box plot to: {input_dir_file_png}")
        print(f"✅ Saved enhanced box plot to: {output_file_pdf}")
        print(f"✅ Saved enhanced box plot to: {input_dir_file_pdf}")
        
        # Show plot
        plt.show()
        
        return output_file_png
    
    def create_statistics_summary(self):
        """Create a statistics summary CSV file"""
        print("\n📊 Creating statistics summary...")
        
        # Find required columns
        material_col = self._find_material_column()
        shot_time_col = self._find_shot_time_column()
        
        if not material_col or not shot_time_col:
            print("❌ Missing required columns for statistics summary")
            return
        
        # Clean data
        df_clean = self.df[[material_col, shot_time_col]].dropna()
        
        if len(df_clean) == 0:
            print("❌ No valid data for statistics summary")
            return
        
        # Calculate statistics by material
        stats_list = []
        for material in df_clean[material_col].unique():
            data = df_clean[df_clean[material_col] == material][shot_time_col].values
            
            if len(data) > 0:
                stats = {
                    'Material': material,
                    'Count': len(data),
                    'Mean (s)': np.mean(data),
                    'Std (s)': np.std(data),
                    'Min (s)': np.min(data),
                    'Max (s)': np.max(data),
                    'Median (s)': np.median(data),
                    'Q1 (s)': np.percentile(data, 25),
                    'Q3 (s)': np.percentile(data, 75),
                    'IQR (s)': np.percentile(data, 75) - np.percentile(data, 25)
                }
                stats_list.append(stats)
        
        # Create summary dataframe
        stats_df = pd.DataFrame(stats_list)
        
        # Save statistics in both output directory and input file directory
        output_file = self.output_dir / "shot_time_statistics_summary.csv"
        input_dir_file = self.input_dir / "shot_time_statistics_summary.csv"
        
        stats_df.to_csv(output_file, index=False)
        stats_df.to_csv(input_dir_file, index=False)
        
        print(f"✅ Saved statistics summary to: {output_file}")
        print(f"✅ Saved statistics summary to: {input_dir_file}")
        
        # Print summary
        print("\n📋 Statistics Summary:")
        print(stats_df.to_string(index=False))
        
        return output_file
    
    def run_analysis(self):
        """Run the complete post-processing analysis"""
        print("🚀 Starting Velocity Summary Post-Processing...")
        print(f"📁 Input file: {self.input_file}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Create all plots and summaries
        results = {}
        
        # Basic box plot
        try:
            results['boxplot'] = self.create_shot_time_vs_material_boxplot()
        except Exception as e:
            print(f"❌ Error creating basic box plot: {e}")
        
        # PDV Return Power box plot
        try:
            results['pdv_power_boxplot'] = self.create_pdv_return_power_vs_material_boxplot()
        except Exception as e:
            print(f"❌ Error creating PDV Return Power box plot: {e}")
        
        # Enhanced box plot
        try:
            results['enhanced_boxplot'] = self.create_enhanced_boxplot()
        except Exception as e:
            print(f"❌ Error creating enhanced box plot: {e}")
        
        # Statistics summary
        try:
            results['statistics'] = self.create_statistics_summary()
        except Exception as e:
            print(f"❌ Error creating statistics summary: {e}")
        
        print("\n✅ Post-processing complete!")
        print("📊 Generated files:")
        for key, file_path in results.items():
            if file_path:
                print(f"  - {key}: {file_path}")
        
        return results


def main():
    """Main function with interactive input"""
    print("🚀 Velocity Summary Post-Processor")
    print("=" * 50)
    
    # Ask for input file
    while True:
        input_file = input("\n📁 Enter the full path to your velocity shots summary CSV file: ").strip()
        
        # Remove quotes if user added them
        input_file = input_file.strip('"\'')
        
        if not input_file:
            print("❌ Please enter a file path.")
            continue
            
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            print("💡 Please check the path and try again.")
            continue
            
        if not input_file.lower().endswith('.csv'):
            print("⚠️  Warning: File doesn't have .csv extension. Continue anyway? (y/n): ", end='')
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                continue
        
        print(f"✅ Found file: {input_file}")
        break
    
    # Ask for output directory
    while True:
        output_dir = input("\n📁 Enter the output directory path (or press Enter for 'plots'): ").strip()
        
        # Remove quotes if user added them
        output_dir = output_dir.strip('"\'')
        
        if not output_dir:
            output_dir = 'plots'
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Output directory: {output_dir}")
            break
        except Exception as e:
            print(f"❌ Error creating output directory: {e}")
            print("💡 Please try a different path.")
            continue
    
    # Confirm and run
    print(f"\n📋 Summary:")
    print(f"   Input file: {input_file}")
    print(f"   Output directory: {output_dir}")
    
    confirm = input("\n🚀 Start processing? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        print("\n" + "=" * 50)
        processor = VelocitySummaryPostProcessor(input_file, output_dir)
        processor.run_analysis()
    else:
        print("❌ Processing cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main() 