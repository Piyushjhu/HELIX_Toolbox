# %%
import os

# Configuration
FOLDER = "/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/pdv_files"  # Change this to your actual folder path
PREFIX = "C1--20251023--"
START, END = 1, 924

def find_missing_csvs():
    """Find missing CSV files in the specified range."""
    missing = []
    
    for i in range(START, END + 1):
        filename = f"{PREFIX}{i:05d}.csv"
        filepath = os.path.join(FOLDER, filename)
        
        if not os.path.exists(filepath):
            missing.append(filename)
    
    print(f"Missing ({len(missing)} files):")
    for filename in missing:
        print(filename)
    
    return missing

if __name__ == "__main__":
    missing_files = find_missing_csvs()
    print(f"\nTotal missing files: {len(missing_files)}")

# %%
