#!/usr/bin/env python3
"""
GitHub Release Script for HELIX Toolbox
Creates a proper release with versioning and documentation
"""

import os
import subprocess
import json
import datetime
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.stdout.strip()

def get_version_info():
    """Get version information from setup.py"""
    try:
        with open('setup.py', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'version=' in line:
                    version = line.split('version=')[1].split(',')[0].strip().strip('"\'')
                    return version
    except:
        pass
    return "1.0.0"

def create_release_notes():
    """Create comprehensive release notes"""
    version = get_version_info()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    notes = f"""# HELIX Toolbox v{version} Release

## Release Date: {today}

### 🎉 Major Features

#### Combined ALPSS + SPADE Analysis Pipeline
- **Unified GUI**: Modern PyQt5 interface with dark/light themes
- **Three Analysis Modes**: ALPSS-only, SPADE-only, or combined pipeline
- **Batch Processing**: Support for single files or multiple files with pattern matching
- **Real-time Progress**: Dual progress bars for ALPSS and SPADE processing

#### Enhanced ALPSS Features
- **Optional Gaussian Notch Filter**: Toggle carrier frequency removal
- **Improved Uncertainty Handling**: 4-column velocity output with uncertainty
- **Scientific Notation Support**: High-precision wavelength input
- **Enhanced Plotting**: All uncertainty plots show multiplier information
- **Parameter Validation**: Proper constraints for peak detection parameters

#### Enhanced SPADE Features
- **Uncertainty Propagation**: Full uncertainty calculations for all outputs
- **Peak Shock Stress**: Calculation with uncertainty from velocity data
- **Enhanced Summary**: Complete results combining ALPSS and SPADE data
- **Skip Smoothing Option**: Avoid double smoothing when using ALPSS outputs
- **Multiple Analysis Models**: Support for hybrid_5_segment and max_min models

#### Advanced Outputs
- **Combined Velocity Traces**: Aligned and uncertainty-shaded plots
- **Spall Strength Analysis**: vs. strain rate and shock stress plots
- **Enhanced Summary CSV**: Complete results with all uncertainties
- **Individual Analysis Plots**: Detailed analysis for each file
- **Literature Comparison**: Built-in literature data for comparison

### 🔧 Technical Improvements

#### Code Quality
- **Modular Architecture**: Clean separation between ALPSS and SPADE
- **Error Handling**: Comprehensive error handling and user feedback
- **Documentation**: Extensive inline documentation and user guides
- **Testing**: Automated test suite for all major functions

#### User Experience
- **Modern GUI**: Large fonts, better spacing, responsive design
- **Input Validation**: Real-time validation with helpful error messages
- **Progress Tracking**: Detailed progress updates during analysis
- **Output Organization**: Structured output directories with clear naming

#### Performance
- **Optimized Processing**: Efficient batch processing capabilities
- **Memory Management**: Proper cleanup and resource management
- **Parallel Processing**: Threaded analysis to prevent GUI freezing

### 📁 File Structure

```
HELIX_Toolbox/
├── ALPSS/                    # ALPSS analysis package
│   ├── alpss_main.py        # Main ALPSS processing
│   ├── alpss_auto_run.py    # Automated ALPSS runner
│   └── requirements.txt     # ALPSS dependencies
├── SPADE/                    # SPADE analysis package
│   └── spall_analysis_release/
│       ├── spall_analysis/  # SPADE analysis modules
│       └── requirements.txt # SPADE dependencies
├── alpss_spade_gui.py       # Main GUI application
├── setup.py                 # Package installation
├── requirements.txt         # Main dependencies
├── README.md               # Comprehensive documentation
└── docs/                   # User guides and documentation
```

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
cd HELIX_Toolbox

# Install dependencies
pip install -r requirements.txt

# Run the GUI
python alpss_spade_gui.py
```

### 📋 System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for installation

### 🎯 Key Features Summary

1. **Unified Interface**: Single GUI for both ALPSS and SPADE analysis
2. **Flexible Input**: Support for single files, multiple files, or directories
3. **Advanced Processing**: Optional notch filtering, uncertainty propagation
4. **Comprehensive Outputs**: Multiple plot types, summary tables, enhanced results
5. **User-Friendly**: Modern interface with themes, validation, and progress tracking
6. **Well-Documented**: Extensive documentation and user guides

### 🔬 Scientific Applications

- **Spallation Experiments**: Complete analysis from raw PDV data to spall strength
- **Material Characterization**: Velocity and stress analysis for material properties
- **Research Workflows**: Streamlined processing for high-throughput experiments
- **Data Validation**: Uncertainty quantification and literature comparison

### 👥 Credits

- **Author**: Piyush Wanchoo
- **ALPSS Credits**: Jake Diamond (original ALPSS development)
- **SPADE Credits**: SPADE development team
- **GUI Development**: Enhanced PyQt5 interface with modern design

### 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/Piyushjhu/HELIX_Toolbox/issues
- **Documentation**: See README.md and docs/ directory
- **Examples**: Check the examples/ directory for usage examples

---

**HELIX Toolbox v{version}** - A comprehensive solution for spallation experiment analysis combining ALPSS and SPADE capabilities in a modern, user-friendly interface.
"""

    return notes

def main():
    """Main release creation process"""
    print("🚀 Creating HELIX Toolbox Release")
    print("=" * 50)
    
    # Get current version
    version = get_version_info()
    print(f"Version: {version}")
    
    # Check if we're on main branch
    current_branch = run_command("git branch --show-current")
    if current_branch != "main":
        print(f"Warning: Not on main branch (currently on {current_branch})")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Check for uncommitted changes
    status = run_command("git status --porcelain")
    if status:
        print("Warning: There are uncommitted changes:")
        print(status)
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Create release notes
    print("\n📝 Creating release notes...")
    release_notes = create_release_notes()
    
    # Save release notes
    release_file = f"RELEASE_v{version}.md"
    with open(release_file, 'w') as f:
        f.write(release_notes)
    print(f"Release notes saved to: {release_file}")
    
    # Create and push tag
    print(f"\n🏷️  Creating tag v{version}...")
    run_command(f'git tag -a v{version} -m "HELIX Toolbox v{version} Release"')
    run_command(f'git push origin v{version}')
    
    # Create release assets
    print("\n📦 Creating release assets...")
    
    # Create source distribution
    run_command("python setup.py sdist")
    
    # Create wheel distribution
    run_command("python setup.py bdist_wheel")
    
    print("\n✅ Release created successfully!")
    print(f"Version: v{version}")
    print(f"Tag: v{version}")
    print(f"Release notes: {release_file}")
    print("\nNext steps:")
    print("1. Go to https://github.com/Piyushjhu/HELIX_Toolbox/releases")
    print("2. Click 'Edit' on the draft release")
    print("3. Copy the content from RELEASE_v{version}.md")
    print("4. Upload the distribution files from dist/ directory")
    print("5. Publish the release")

if __name__ == "__main__":
    main() 