#!/usr/bin/env python3
"""
Script to copy files with the same base name as velocity files to a user-defined output folder
"""
import os
import sys
import glob
import shutil
from pathlib import Path

def get_user_paths():
    """Get input and output paths from user"""
    print("=== FILE COPY UTILITY ===")
    
    # Get input path
    print("Enter the path to your velocity data files (or press Enter for default: ./input_data):")
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
    print("Enter the output folder path (or press Enter for default: ./copied_files):")
    user_input = input().strip()
    if user_input:
        output_path = user_input
    else:
        output_path = "./copied_files"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    return input_path, output_path

def find_velocity_files(input_path):
    """Find all velocity files in the specified path"""
    print(f"\n=== SEARCHING FOR VELOCITY FILES ===")
    print(f"Searching in: {os.path.abspath(input_path)}")
    
    # Search pattern for velocity files
    search_pattern = os.path.join(input_path, '**/*--vel-smooth-with-uncert.csv')
    
    velocity_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(velocity_files)} velocity files:")
    for file_path in velocity_files:
        file_size = os.path.getsize(file_path)
        print(f"  {file_path} ({file_size} bytes)")
    
    return velocity_files

def extract_base_name(velocity_file_path):
    """Extract the base name from a velocity file path"""
    filename = os.path.basename(velocity_file_path)
    
    # Remove the --vel-smooth-with-uncert.csv suffix
    if filename.endswith('--vel-smooth-with-uncert.csv'):
        base_name = filename[:-len('--vel-smooth-with-uncert.csv')]
        return base_name
    
    # If the pattern doesn't match, try other common patterns
    if '--vel' in filename:
        base_name = filename.split('--vel')[0]
        return base_name
    
    # If no pattern matches, return the filename without extension
    return os.path.splitext(filename)[0]

def find_related_files(input_path, base_name):
    """Find all files with the same base name"""
    # Search for files with the same base name
    search_patterns = [
        os.path.join(input_path, '**/*' + base_name + '*'),
        os.path.join(input_path, base_name + '*')
    ]
    
    related_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        related_files.extend(files)
    
    # Remove duplicates
    related_files = list(set(related_files))
    
    return related_files

def copy_related_files(input_path, output_path, velocity_files):
    """Copy all files related to velocity files to output folder"""
    print(f"\n=== COPYING RELATED FILES ===")
    print(f"Output folder: {os.path.abspath(output_path)}")
    
    total_copied = 0
    total_skipped = 0
    
    for velocity_file in velocity_files:
        base_name = extract_base_name(velocity_file)
        print(f"\n📁 Base name: {base_name}")
        
        # Find all related files
        related_files = find_related_files(input_path, base_name)
        
        if not related_files:
            print(f"  ⚠️  No related files found for {base_name}")
            continue
        
        print(f"  Found {len(related_files)} related files:")
        
        for file_path in related_files:
            # Skip if it's the same as the velocity file (already processed)
            if file_path == velocity_file:
                continue
            
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
    
    return total_copied, total_skipped

def main():
    """Main function"""
    print("RELATED FILE COPY UTILITY")
    print("=" * 50)
    
    # Get user paths
    input_path, output_path = get_user_paths()
    if input_path is None:
        return
    
    # Find velocity files
    velocity_files = find_velocity_files(input_path)
    
    if not velocity_files:
        print("❌ No velocity files found!")
        return
    
    # Copy related files
    total_copied, total_skipped = copy_related_files(input_path, output_path, velocity_files)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 COPY SUMMARY")
    print(f"  Input path: {os.path.abspath(input_path)}")
    print(f"  Output path: {os.path.abspath(output_path)}")
    print(f"  Velocity files found: {len(velocity_files)}")
    print(f"  Files copied: {total_copied}")
    print(f"  Files skipped: {total_skipped}")
    print(f"  Total processed: {total_copied + total_skipped}")
    
    if total_copied > 0:
        print(f"\n✅ Successfully copied {total_copied} files to {output_path}")
    else:
        print(f"\n⚠️  No files were copied")

if __name__ == "__main__":
    main() 