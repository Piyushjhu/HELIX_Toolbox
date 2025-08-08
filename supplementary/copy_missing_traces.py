#!/usr/bin/env python3
"""
Script to copy files with the same base name as the missing traces (43 files)
"""
import os
import sys
import glob
import shutil
from pathlib import Path

def get_user_paths():
    """Get input and output paths from user"""
    print("=== MISSING TRACES FILE COPY UTILITY ===")
    
    # Get input path
    print("Enter the path to your data files (or press Enter for default: ./input_data):")
    user_input = input().strip()
    if user_input:
        input_path = user_input
    else:
        input_path = "./input_data"
    
    # Validate input path
    if not os.path.exists(input_path):
        print(f"❌ Error: Path '{input_path}' does not exist!")
        return None, None
    
    # Get output path
    print("Enter the output folder path (or press Enter for default: ./missing_traces_files):")
    user_input = input().strip()
    if user_input:
        output_path = user_input
    else:
        output_path = "./missing_traces_files"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return input_path, output_path

def get_missing_traces_list():
    """Return the list of missing traces base names"""
    missing_traces = [
        "C1--20250728--00017",
        "C1--20250725--00118",
        "C1--20250725--00093",
        "C1--20250725--00069",
        "C1--20250725--00058",
        "C1--20250725--00098",
        "C1--20250728--00007",
        "C1--20250728--00013",
        "C1--20250725--00107",
        "C1--20250725--00105",
        "C1--20250728--00003",
        "C1--20250728--00006",
        "C1--20250725--00115",
        "C1--20250728--00005",
        "C1--20250728--00021",
        "C1--20250725--00029",
        "C1--20250728--00020",
        "C1--20250728--00022",
        "C1--20250725--00111",
        "C1--20250725--00096",
        "C1--20250725--00049",
        "C1--20250725--00114",
        "C1--20250725--00110",
        "C1--20250725--00102",
        "C1--20250725--00085",
        "C1--20250725--00104",
        "C1--20250728--00009",
        "C1--20250725--00119",
        "C1--20250725--00108",
        "C1--20250725--00113",
        "C1--20250725--00117",
        "C1--20250728--00014",
        "C1--20250728--00001",
        "C1--20250728--00023",
        "C1--20250725--00120",
        "C1--20250728--00019",
        "C1--20250725--00097",
        "C1--20250728--00002",
        "C1--20250725--00080",
        "C1--20250728--00018",
        "C1--20250725--00084",
        "C1--20250725--00121",
        "C1--20250728--00004"
    ]
    return missing_traces

def find_related_files(input_path, base_name):
    """Find all files with the same base name"""
    # More comprehensive search patterns
    search_patterns = [
        os.path.join(input_path, '**/*' + base_name + '*'),
        os.path.join(input_path, base_name + '*'),
        os.path.join(input_path, '**/*' + base_name + '.*'),
        os.path.join(input_path, '**/*' + base_name + '--*'),
        os.path.join(input_path, base_name + '--*'),
        os.path.join(input_path, '**/*' + base_name + '.csv'),
        os.path.join(input_path, '**/*' + base_name + '.txt'),
        os.path.join(input_path, '**/*' + base_name + '.dat')
    ]
    
    related_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        related_files.extend(files)
    
    # Remove duplicates
    related_files = list(set(related_files))
    
    # Debug: Show what we're searching for
    print(f"    🔍 Searching for patterns containing: {base_name}")
    
    return related_files

def copy_missing_traces_files(input_path, output_path, missing_traces):
    """Copy all files related to missing traces to output folder"""
    print(f"\n=== COPYING MISSING TRACES FILES ===")
    print(f"Output folder: {os.path.abspath(output_path)}")
    print(f"Targeting {len(missing_traces)} missing traces")
    
    total_copied = 0
    total_skipped = 0
    traces_with_files = 0
    
    for i, base_name in enumerate(missing_traces, 1):
        print(f"\n📁 [{i:2d}/{len(missing_traces)}] Base name: {base_name}")
        
        # Find all related files
        related_files = find_related_files(input_path, base_name)
        
        if not related_files:
            print(f"  ⚠️  No related files found for {base_name}")
            continue
        
        traces_with_files += 1
        print(f"  Found {len(related_files)} related files:")
        
        for file_path in related_files:
            # Get the filename for the output
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_path, filename)
            
            # Check if file already exists in output
            if os.path.exists(output_file_path):
                print(f"    ⚠️  Skipped {filename} (already exists in output)")
                total_skipped += 1
                continue
            
            try:
                # Copy the file
                shutil.copy2(file_path, output_file_path)
                file_size = os.path.getsize(file_path)
                print(f"    ✅ Copied {filename} ({file_size} bytes)")
                total_copied += 1
            except Exception as e:
                print(f"    ❌ Error copying {filename}: {e}")
                total_skipped += 1
    
    return total_copied, total_skipped, traces_with_files

def main():
    """Main function"""
    print("MISSING TRACES FILE COPY UTILITY")
    print("=" * 50)
    
    # Get user paths
    input_path, output_path = get_user_paths()
    if input_path is None:
        return
    
    # Get the list of missing traces
    missing_traces = get_missing_traces_list()
    
    print(f"\n📋 Targeting {len(missing_traces)} missing traces:")
    for i, trace_name in enumerate(missing_traces, 1):
        print(f"  {i:2d}. {trace_name}")
    
    # Debug: List some files in the input directory
    print(f"\n🔍 DEBUG: Listing some files in {input_path}:")
    all_files = glob.glob(os.path.join(input_path, '**/*'), recursive=True)
    sample_files = all_files[:10]  # Show first 10 files
    for file_path in sample_files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            print(f"  📄 {filename}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more files")
    
    # Copy related files
    total_copied, total_skipped, traces_with_files = copy_missing_traces_files(input_path, output_path, missing_traces)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 COPY SUMMARY")
    print(f"  Input path: {os.path.abspath(input_path)}")
    print(f"  Output path: {os.path.abspath(output_path)}")
    print(f"  Missing traces targeted: {len(missing_traces)}")
    print(f"  Traces with related files: {traces_with_files}")
    print(f"  Files copied: {total_copied}")
    print(f"  Files skipped: {total_skipped}")
    print(f"  Total processed: {total_copied + total_skipped}")
    
    if total_copied > 0:
        print(f"\n✅ Successfully copied {total_copied} files to {output_path}")
    else:
        print(f"\n⚠️  No files were copied")
    
    if traces_with_files < len(missing_traces):
        print(f"\n⚠️  Only {traces_with_files}/{len(missing_traces)} traces had related files")

if __name__ == "__main__":
    main() 