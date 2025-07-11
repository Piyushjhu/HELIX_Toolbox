#!/bin/bash
# Build Mac executable for SPADE
# Author: Piyush Wanchoo
# Institution: Johns Hopkins University
# Year: 2025
# GitHub: https://github.com/Piyushjhu/SPADE
# Make sure you have Python and PyInstaller installed

echo "Building Mac executable for SPADE..."

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip3 install pyinstaller
fi

# Install required dependencies if not already installed
echo "Installing required dependencies..."
pip3 install -r requirements.txt

# Build the executable with all necessary hidden imports
pyinstaller --onefile --windowed --name=SPADE \
    --add-data=spall_analysis:spall_analysis \
    --add-data=data:data \
    --hidden-import=spall_analysis \
    --hidden-import=spall_analysis.data_processing \
    --hidden-import=spall_analysis.plotting \
    --hidden-import=spall_analysis.models \
    --hidden-import=spall_analysis.literature \
    --hidden-import=spall_analysis.utils \
    --hidden-import=pandas \
    --hidden-import=numpy \
    --hidden-import=scipy \
    --hidden-import=scipy.interpolate \
    --hidden-import=scipy.signal \
    --hidden-import=matplotlib \
    --hidden-import=matplotlib.pyplot \
    --hidden-import=matplotlib.backends.backend_qt5agg \
    --hidden-import=sklearn \
    --hidden-import=sklearn.linear_model \
    --hidden-import=sklearn.preprocessing \
    --hidden-import=plotly \
    --hidden-import=plotly.graph_objects \
    --hidden-import=plotly.express \
    --hidden-import=PyQt5 \
    --hidden-import=PyQt5.QtWidgets \
    --hidden-import=PyQt5.QtCore \
    --hidden-import=PyQt5.QtGui \
    --hidden-import=threading \
    --hidden-import=os \
    --hidden-import=glob \
    --hidden-import=warnings \
    --hidden-import=traceback \
    --hidden-import=logging \
    --collect-all=pandas \
    --collect-all=numpy \
    --collect-all=scipy \
    --collect-all=matplotlib \
    --collect-all=sklearn \
    --collect-all=plotly \
    --collect-all=PyQt5 \
    examples/spall_analysis_gui.py

# Move to executables directory
if [ -f "dist/SPADE" ]; then
    mkdir -p executables
    mv dist/SPADE executables/SPADE_mac_new
    echo "Mac executable created: executables/SPADE_mac_new"
else
    echo "Error: SPADE not found in dist/ directory"
fi

# Clean up build artifacts
rm -rf build/
rm -rf dist/
rm -f SPADE.spec

echo "Build complete!" 