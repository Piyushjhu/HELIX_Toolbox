#!/usr/bin/env python3
"""
Analysis Monitor - Tracks analysis failures and provides diagnostics
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import json

class AnalysisMonitor:
    def __init__(self):
        self.failed_files = []
        self.successful_files = []
        self.error_log = []
        
    def scan_output_directories(self):
        """Scan all output directories for analysis results"""
        print("=== Analysis Monitor ===")
        print(f"Scanning at: {datetime.now()}")
        
        # Check common output directories
        output_dirs = [
            "./output",
            "./ALPSS/output_data", 
            "./test_output",
            "./ALPSS/test_output"
        ]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                print(f"\n📁 Checking: {output_dir}")
                self.analyze_output_directory(output_dir)
            else:
                print(f"❌ Directory not found: {output_dir}")
    
    def analyze_output_directory(self, output_dir):
        """Analyze a specific output directory"""
        # Check for ALPSS outputs
        alpss_outputs = self.check_alpss_outputs(output_dir)
        
        # Check for SPADE inputs
        spade_inputs = self.check_spade_inputs(output_dir)
        
        # Check for SPADE outputs
        spade_outputs = self.check_spade_outputs(output_dir)
        
        # Generate summary
        self.generate_summary(output_dir, alpss_outputs, spade_inputs, spade_outputs)
    
    def check_alpss_outputs(self, output_dir):
        """Check what ALPSS outputs are present"""
        print(f"  🔍 Checking ALPSS outputs...")
        
        expected_files = [
            '*--velocity.csv',
            '*--velocity--smooth.csv', 
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
        missing_patterns = []
        
        for pattern in expected_files:
            files = glob.glob(os.path.join(output_dir, pattern))
            if files:
                found_files.extend(files)
                print(f"    ✅ {pattern}: {len(files)} files")
            else:
                missing_patterns.append(pattern)
                print(f"    ❌ {pattern}: No files found")
        
        return {
            'found_files': found_files,
            'missing_patterns': missing_patterns,
            'total_found': len(found_files),
            'total_missing': len(missing_patterns)
        }
    
    def check_spade_inputs(self, output_dir):
        """Check what SPADE input files are available"""
        print(f"  🔍 Checking SPADE inputs...")
        
        spade_input_patterns = [
            '*--vel-smooth-with-uncert.csv',  # Primary SPADE input
            '*--velocity--smooth.csv',        # Alternative SPADE input
        ]
        
        spade_input_files = []
        for pattern in spade_input_patterns:
            files = glob.glob(os.path.join(output_dir, pattern))
            spade_input_files.extend(files)
            print(f"    {'✅' if files else '❌'} {pattern}: {len(files)} files")
        
        return {
            'input_files': spade_input_files,
            'total_inputs': len(spade_input_files)
        }
    
    def check_spade_outputs(self, output_dir):
        """Check what SPADE outputs are present"""
        print(f"  🔍 Checking SPADE outputs...")
        
        spade_output_dir = os.path.join(output_dir, "SPADE_analysis")
        if not os.path.exists(spade_output_dir):
            print(f"    ❌ SPADE output directory not found: {spade_output_dir}")
            return {'output_files': [], 'total_outputs': 0}
        
        # Check for SPADE output files
        spade_output_patterns = [
            '*.png',
            # PDF files removed 
            '*.csv',
            'summary_table.csv',
            'spall_analysis_summary.csv',
            'velocity_shots_summary.csv'
        ]
        
        spade_output_files = []
        for pattern in spade_output_patterns:
            files = glob.glob(os.path.join(spade_output_dir, pattern))
            spade_output_files.extend(files)
            print(f"    {'✅' if files else '❌'} {pattern}: {len(files)} files")
        
        return {
            'output_files': spade_output_files,
            'total_outputs': len(spade_output_files)
        }
    
    def generate_summary(self, output_dir, alpss_outputs, spade_inputs, spade_outputs):
        """Generate a summary of the analysis status"""
        print(f"\n📊 Summary for {output_dir}:")
        print(f"   ALPSS outputs: {alpss_outputs['total_found']} found, {alpss_outputs['total_missing']} missing")
        print(f"   SPADE inputs: {spade_inputs['total_inputs']} available")
        print(f"   SPADE outputs: {spade_outputs['total_outputs']} generated")
        
        # Identify potential issues
        issues = []
        if alpss_outputs['total_found'] == 0:
            issues.append("No ALPSS outputs found")
        if spade_inputs['total_inputs'] == 0:
            issues.append("No SPADE input files available")
        if spade_outputs['total_outputs'] == 0:
            issues.append("No SPADE outputs generated")
        
        if issues:
            print(f"   ⚠️  Issues detected:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ Analysis appears successful")
    
    def analyze_failed_files(self, output_dir):
        """Analyze which files failed to generate expected outputs"""
        print(f"\n🔍 Analyzing failed files in {output_dir}...")
        
        # Get all CSV files that might be input files
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
        
        print(f"   Successful files (have SPADE input): {len(successful_files)}")
        for f in successful_files:
            print(f"      ✅ {f}")
        
        print(f"   Failed files (missing SPADE input): {len(failed_files)}")
        for f in failed_files:
            print(f"      ❌ {f}")
        
        return failed_files
    
    def check_file_contents(self, output_dir):
        """Check the contents of key files to ensure they're valid"""
        print(f"\n🔍 Checking file contents in {output_dir}...")
        
        # Check velocity files
        velocity_files = glob.glob(os.path.join(output_dir, "*--velocity--smooth.csv"))
        for vf in velocity_files:
            try:
                df = pd.read_csv(vf)
                print(f"   ✅ {os.path.basename(vf)}: {df.shape[0]} rows, {df.shape[1]} columns")
                if df.shape[0] > 0:
                    print(f"      Time range: {df.iloc[0, 0]:.2e} to {df.iloc[-1, 0]:.2e}")
                    print(f"      Velocity range: {df.iloc[:, 1].min():.2f} to {df.iloc[:, 1].max():.2f}")
            except Exception as e:
                print(f"   ❌ {os.path.basename(vf)}: Error reading file - {e}")
        
        # Check results files
        results_files = glob.glob(os.path.join(output_dir, "*--results.csv"))
        for rf in results_files:
            try:
                df = pd.read_csv(rf)
                print(f"   ✅ {os.path.basename(rf)}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"   ❌ {os.path.basename(rf)}: Error reading file - {e}")
    
    def run_full_analysis(self):
        """Run complete analysis monitoring"""
        print("🚀 Starting Analysis Monitor...")
        print("=" * 50)
        
        # Scan all output directories
        self.scan_output_directories()
        
        # Analyze failed files in each directory
        output_dirs = ["./output", "./ALPSS/output_data", "./test_output", "./ALPSS/test_output"]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                self.analyze_failed_files(output_dir)
                self.check_file_contents(output_dir)
        
        print("\n💡 Recommendations:")
        print("1. If ALPSS outputs are missing, check the GUI parameter settings")
        print("2. If SPADE inputs are missing, ensure ALPSS completed successfully")
        print("3. Check that all required parameters are set in the GUI")
        print("4. Verify that input files are in the correct format")
        print("5. Check the console output for error messages during analysis")

def main():
    monitor = AnalysisMonitor()
    monitor.run_full_analysis()

if __name__ == "__main__":
    main() 