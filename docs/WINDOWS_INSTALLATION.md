# HELIX Toolbox - Windows Installation Guide

## Overview
HELIX Toolbox is fully compatible with Windows and provides a unified GUI for ALPSS and SPADE analysis. This guide covers installation and usage on Windows systems.

## System Requirements

### Minimum Requirements
- **Windows**: Windows 10 or later (64-bit)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Display**: 1024x768 minimum resolution

### Recommended Requirements
- **Windows**: Windows 11 (64-bit)
- **Python**: 3.9 or higher
- **RAM**: 16GB
- **Storage**: 2GB free space
- **Display**: 1920x1080 or higher

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Open Command Prompt or PowerShell**:
   ```cmd
   # Check Python version
   python --version
   
   # Upgrade pip
   python -m pip install --upgrade pip
   ```

3. **Install HELIX Toolbox**:
   ```cmd
   # Install from GitHub (when available)
   pip install git+https://github.com/Piyushjhu/HELIX_Toolbox.git
   
   # Or install from local directory
   cd path\to\HELIX_Toolbox
   pip install -e .
   ```

### Method 2: Manual Installation

1. **Clone the repository**:
   ```cmd
   git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
   cd HELIX_Toolbox
   ```

2. **Create a virtual environment** (recommended):
   ```cmd
   python -m venv helix_env
   helix_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the GUI**:
   ```cmd
   python alpss_spade_gui.py
   ```

## Dependencies

HELIX Toolbox automatically installs these dependencies:

### Core Dependencies
- **PyQt5** (≥5.15.0) - GUI framework
- **numpy** (≥1.19.0) - Numerical computing
- **scipy** (≥1.7.0) - Scientific computing
- **pandas** (≥1.3.0) - Data manipulation
- **matplotlib** (≥3.3.0) - Plotting
- **scikit-learn** (≥1.0.0) - Machine learning

### Optional Dependencies
- **seaborn** (≥0.11.0) - Enhanced plotting (optional)

## Windows-Specific Features

### File Explorer Integration
- **"Open Output Directory"** button opens Windows Explorer
- **File dialogs** use native Windows file picker
- **Path handling** automatically uses Windows path separators

### GUI Appearance
- **Native Windows styling** with Segoe UI font
- **Dark/Light theme** support
- **High DPI** support for modern displays
- **Responsive design** that adapts to window size

### Performance Optimizations
- **Multi-threaded processing** prevents GUI freezing
- **Memory management** optimized for Windows
- **Progress tracking** with real-time updates

## Usage on Windows

### Starting the Application
```cmd
# From Command Prompt
python alpss_spade_gui.py

# From PowerShell
python .\alpss_spade_gui.py

# Create desktop shortcut (optional)
# Right-click desktop → New → Shortcut
# Target: "C:\Path\To\Python\python.exe" "C:\Path\To\HELIX_Toolbox\alpss_spade_gui.py"
```

### File Selection
- **Single file**: Use file dialog to select individual CSV files
- **Multiple files**: Select directory containing multiple files
- **File patterns**: Use wildcards like `*.csv` or `*_data.csv`

### Output Management
- **Default output**: `C:\Users\YourUsername\ALPSS_SPADE_output`
- **Custom output**: Select any directory using file dialog
- **Automatic organization**: Results saved in structured folders

## Troubleshooting

### Common Issues

#### 1. Python Not Found
```cmd
# Check if Python is in PATH
python --version

# If not found, add Python to PATH manually
# Or reinstall Python with "Add to PATH" checked
```

#### 2. PyQt5 Installation Issues
```cmd
# Try installing PyQt5 separately
pip install PyQt5

# If that fails, try PySide2 as alternative
pip install PySide2
```

#### 3. Permission Errors
```cmd
# Run Command Prompt as Administrator
# Or change output directory to user folder
```

#### 4. Memory Issues
- Close other applications
- Reduce batch size (process fewer files at once)
- Increase virtual memory in Windows settings

#### 5. Display Issues
- Update graphics drivers
- Try different DPI settings
- Use Windows compatibility mode if needed

### Performance Tips

1. **Use SSD storage** for faster file I/O
2. **Close unnecessary applications** during analysis
3. **Process files in smaller batches** for large datasets
4. **Use virtual environment** to avoid conflicts

### Getting Help

1. **Check the logs** in the GUI progress window
2. **Verify file formats** (CSV files should be properly formatted)
3. **Test with example files** first
4. **Report issues** on GitHub with:
   - Windows version
   - Python version
   - Error messages
   - Sample data (if possible)

## Advanced Configuration

### Environment Variables
```cmd
# Set matplotlib backend (if needed)
set MPLBACKEND=Qt5Agg

# Set PyQt5 platform (if needed)
set QT_QPA_PLATFORM=windows
```

### Custom Installation
```cmd
# Install with specific versions
pip install PyQt5==5.15.9
pip install numpy==1.24.3
pip install scipy==1.10.1

# Install development version
pip install -e .[dev]
```

## Integration with Windows

### File Associations
- Associate `.csv` files with HELIX Toolbox (optional)
- Create batch files for common operations
- Use Windows Task Scheduler for automated processing

### Windows Subsystem for Linux (WSL)
- HELIX Toolbox works in WSL2
- GUI requires X11 forwarding
- Performance may be slower than native Windows

## Support

For Windows-specific issues:
1. Check this guide first
2. Search existing GitHub issues
3. Create new issue with Windows details
4. Include system information and error logs

---

**HELIX Toolbox** - Cross-platform spallation analysis made easy on Windows! 🪟 