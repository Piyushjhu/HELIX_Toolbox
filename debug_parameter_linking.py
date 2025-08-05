#!/usr/bin/env python3
"""
Debug script to help identify parameter linking issues
"""

import os
import pandas as pd
import glob

def debug_parameter_linking(param_folder_path, csv_files_pattern):
    """Debug parameter linking between Excel files and CSV files"""
    
    print("=== PARAMETER LINKING DEBUG ===")
    
    # 1. Check parameter folder
    print(f"\n1. Parameter folder: {param_folder_path}")
    if not os.path.exists(param_folder_path):
        print("❌ Parameter folder does not exist!")
        return
    
    # 2. Find Excel files
    excel_files = []
    for file in os.listdir(param_folder_path):
        if file.lower().endswith(('.xlsx', '.xls')):
            excel_files.append(os.path.join(param_folder_path, file))
    
    print(f"Found {len(excel_files)} Excel files:")
    for file in excel_files:
        print(f"  - {os.path.basename(file)}")
    
    if not excel_files:
        print("❌ No Excel files found in parameter folder!")
        return
    
    # 3. Analyze each Excel file
    combined_param_data = {}
    
    for param_file_path in excel_files:
        print(f"\n2. Analyzing Excel file: {os.path.basename(param_file_path)}")
        
        try:
            df = pd.read_excel(param_file_path)
            print(f"   Columns found: {list(df.columns)}")
            print(f"   Number of rows: {len(df)}")
            
            # Look for PDV filename column
            pdv_col = None
            for col in df.columns:
                col_lower = col.lower()
                if any(name in col_lower for name in ['pdv_filename', 'pdvfilename', 'dv_filename', 'dvfilename', 'pdv_file', 'pdvfile']):
                    pdv_col = col
                    break
            
            if pdv_col is None:
                print("   ❌ No PDV filename column found!")
                print("   Looking for columns containing: pdv_filename, pdvfilename, dv_filename, dvfilename, pdv_file, pdvfile")
                continue
            
            print(f"   ✅ Found PDV filename column: '{pdv_col}'")
            
            # Show unique values in PDV filename column
            unique_pdv_files = df[pdv_col].dropna().unique()
            print(f"   Unique PDV filenames ({len(unique_pdv_files)}):")
            for i, pdv_file in enumerate(unique_pdv_files[:10]):  # Show first 10
                print(f"     {i+1}. {pdv_file}")
            if len(unique_pdv_files) > 10:
                print(f"     ... and {len(unique_pdv_files) - 10} more")
            
            # Create mapping
            param_data = {}
            for idx, row in df.iterrows():
                pdv_file = row[pdv_col]
                if pd.isna(pdv_file) or pdv_file == 0:
                    continue
                
                pdv_file_str = str(pdv_file)
                exp_info = {}
                for col in df.columns:
                    if col != pdv_col:
                        value = row.get(col)
                        if not pd.isna(value):
                            exp_info[col] = value
                
                param_data[pdv_file_str] = exp_info
                combined_param_data[pdv_file_str] = exp_info
            
            print(f"   ✅ Created {len(param_data)} parameter mappings")
            
        except Exception as e:
            print(f"   ❌ Error reading Excel file: {e}")
            continue
    
    # 4. Find CSV files
    print(f"\n3. CSV files pattern: {csv_files_pattern}")
    csv_files = glob.glob(csv_files_pattern)
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) > 10:
        print("First 10 CSV files:")
        for i, csv_file in enumerate(csv_files[:10]):
            print(f"  {i+1}. {os.path.basename(csv_file)}")
        print(f"  ... and {len(csv_files) - 10} more")
    else:
        for i, csv_file in enumerate(csv_files):
            print(f"  {i+1}. {os.path.basename(csv_file)}")
    
    # 5. Test matching
    print(f"\n4. Testing parameter matching:")
    matched_count = 0
    unmatched_count = 0
    
    for csv_file in csv_files[:20]:  # Test first 20 files
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        if base_name in combined_param_data:
            exp_info = combined_param_data[base_name]
            # Handle different possible column names
            exp_id = exp_info.get('exp_id', exp_info.get('Exp_ID', 'Unknown'))
            sample_material = exp_info.get('sample_material', exp_info.get('Flyer_material', 'Unknown'))
            print(f"  ✅ {base_name} -> {exp_id} - {sample_material}")
            matched_count += 1
        else:
            print(f"  ❌ {base_name} -> No match found")
            unmatched_count += 1
    
    if len(csv_files) > 20:
        print(f"  ... tested first 20 files, {len(csv_files) - 20} remaining")
    
    # 6. Summary
    print(f"\n5. SUMMARY:")
    print(f"   Total parameter mappings: {len(combined_param_data)}")
    print(f"   Matched CSV files: {matched_count}")
    print(f"   Unmatched CSV files: {unmatched_count}")
    
    if unmatched_count > 0:
        print(f"\n6. TROUBLESHOOTING:")
        print("   The issue is likely one of these:")
        print("   a) CSV filenames don't match PDV filenames in Excel")
        print("   b) Excel file doesn't have the expected column names")
        print("   c) PDV filename column has different format than CSV names")
        print("\n   To fix this:")
        print("   1. Check your Excel file column names")
        print("   2. Ensure PDV filenames in Excel match your CSV filenames")
        print("   3. Verify the parameter folder path is correct")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python debug_parameter_linking.py <param_folder_path> <csv_files_pattern>")
        print("Example: python debug_parameter_linking.py ./parameter_files './input_data/*.csv'")
        sys.exit(1)
    
    param_folder_path = sys.argv[1]
    csv_files_pattern = sys.argv[2]
    
    debug_parameter_linking(param_folder_path, csv_files_pattern) 