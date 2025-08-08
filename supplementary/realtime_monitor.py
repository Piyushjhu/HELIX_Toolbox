#!/usr/bin/env python3
"""
Real-time Analysis Monitor - Tracks analysis progress and failures in real-time
"""

import os
import glob
import time
import subprocess
import psutil
from datetime import datetime
import pandas as pd

class RealtimeMonitor:
    def __init__(self):
        self.previous_files = set()
        self.gui_process = None
        self.monitoring = True
        
    def find_gui_process(self):
        """Find the running GUI process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'helix_analysis_toolbox.py' in ' '.join(proc.info['cmdline'] or []):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_all_analysis_files(self):
        """Get all analysis files from output directories"""
        files = set()
        output_dirs = ["./output", "./ALPSS/output_data"]
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                # Get all CSV, PNG, and PDF files
                for ext in ['*.csv', '*.png']:
                    files.update(glob.glob(os.path.join(output_dir, ext)))
                    # Also check SPADE subdirectories
                    spade_dir = os.path.join(output_dir, "SPADE_analysis")
                    if os.path.exists(spade_dir):
                        files.update(glob.glob(os.path.join(spade_dir, ext)))
        
        return files
    
    def check_gui_status(self):
        """Check if GUI is running and its status"""
        proc = self.find_gui_process()
        if proc:
            try:
                # Get process info
                cpu_percent = proc.cpu_percent()
                memory_info = proc.memory_info()
                status = proc.status()
                
                print(f"🖥️  GUI Status: {status}")
                print(f"   CPU: {cpu_percent:.1f}%")
                print(f"   Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
                print(f"   PID: {proc.pid}")
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print("❌ GUI process not accessible")
                return False
        else:
            print("❌ GUI process not found")
            return False
    
    def monitor_file_changes(self):
        """Monitor for new files being created"""
        current_files = self.get_all_analysis_files()
        new_files = current_files - self.previous_files
        
        if new_files:
            print(f"\n🆕 New files detected ({len(new_files)}):")
            for file in sorted(new_files):
                try:
                    size = os.path.getsize(file)
                    mtime = datetime.fromtimestamp(os.path.getmtime(file))
                    print(f"   📄 {os.path.basename(file)} ({size} bytes, {mtime.strftime('%H:%M:%S')})")
                except OSError:
                    print(f"   📄 {os.path.basename(file)} (error getting info)")
        
        self.previous_files = current_files
        return len(new_files) > 0
    
    def check_analysis_progress(self):
        """Check the current state of analysis"""
        print(f"\n📊 Analysis Status Check - {datetime.now().strftime('%H:%M:%S')}")
        
        # Check output directories
        output_dirs = ["./output", "./ALPSS/output_data"]
        total_files = 0
        
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                csv_files = len(glob.glob(os.path.join(output_dir, "*.csv")))
                png_files = len(glob.glob(os.path.join(output_dir, "*.png")))
                # PDF files removed
                
                spade_dir = os.path.join(output_dir, "SPADE_analysis")
                spade_files = 0
                if os.path.exists(spade_dir):
                    spade_files = len(glob.glob(os.path.join(spade_dir, "*")))
                
                total_files += csv_files + png_files + spade_files
                
                print(f"   📁 {output_dir}: {csv_files} CSV, {png_files} PNG, {spade_files} SPADE")
        
        print(f"   📈 Total analysis files: {total_files}")
        return total_files
    
    def check_for_errors(self):
        """Check for common error patterns"""
        error_indicators = []
        
        # Check for empty or corrupted files
        output_dirs = ["./output", "./ALPSS/output_data"]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                for file in glob.glob(os.path.join(output_dir, "*.csv")):
                    try:
                        size = os.path.getsize(file)
                        if size == 0:
                            error_indicators.append(f"Empty file: {os.path.basename(file)}")
                        elif size < 100:  # Very small file might be corrupted
                            error_indicators.append(f"Very small file: {os.path.basename(file)} ({size} bytes)")
                    except OSError:
                        error_indicators.append(f"Cannot access file: {os.path.basename(file)}")
        
        if error_indicators:
            print(f"\n⚠️  Potential issues detected:")
            for error in error_indicators:
                print(f"   - {error}")
        
        return error_indicators
    
    def run_monitoring(self, interval=5):
        """Run continuous monitoring"""
        print("🚀 Starting Real-time Analysis Monitor...")
        print("=" * 50)
        
        # Initial file count
        initial_count = self.check_analysis_progress()
        self.previous_files = self.get_all_analysis_files()
        
        print(f"\n📈 Starting with {initial_count} analysis files")
        print(f"⏱️  Monitoring every {interval} seconds...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.monitoring:
                # Check GUI status
                gui_running = self.check_gui_status()
                
                # Monitor file changes
                new_files_detected = self.monitor_file_changes()
                
                # Check analysis progress
                current_count = self.check_analysis_progress()
                
                # Check for errors
                errors = self.check_for_errors()
                
                if new_files_detected:
                    print(f"🎉 Analysis progress detected!")
                
                if errors:
                    print(f"⚠️  {len(errors)} potential issues found")
                
                # Wait for next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
        
        print(f"\n📊 Final Analysis Summary:")
        final_count = self.check_analysis_progress()
        print(f"   Files at start: {initial_count}")
        print(f"   Files at end: {final_count}")
        print(f"   Net change: {final_count - initial_count}")

def main():
    monitor = RealtimeMonitor()
    monitor.run_monitoring(interval=10)  # Check every 10 seconds

if __name__ == "__main__":
    main() 