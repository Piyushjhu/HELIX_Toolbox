#!/bin/bash
# Comprehensive build script for Mac executable with better dependency handling
# Author: Piyush Wanchoo
# Institution: Johns Hopkins University
# Year: 2025
# GitHub: https://github.com/Piyushjhu/SPADE

echo "Building Mac executable for SPADE (Comprehensive Build)..."

# Check if PyInstaller is installed
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip3 install pyinstaller
fi

# Install required dependencies
echo "Installing required dependencies..."
pip3 install -r requirements.txt

# Create a comprehensive spec file
cat > SPADE_comprehensive.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['examples/spall_analysis_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('spall_analysis', 'spall_analysis'),
        ('data', 'data'),
    ],
    hiddenimports=[
        'spall_analysis',
        'spall_analysis.data_processing',
        'spall_analysis.plotting',
        'spall_analysis.models',
        'spall_analysis.literature',
        'spall_analysis.utils',
        'pandas',
        'numpy',
        'scipy',
        'scipy.interpolate',
        'scipy.signal',
        'scipy.optimize',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_agg',
        'sklearn',
        'sklearn.linear_model',
        'sklearn.preprocessing',
        'sklearn.utils',
        'plotly',
        'plotly.graph_objects',
        'plotly.express',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'threading',
        'os',
        'glob',
        'warnings',
        'traceback',
        'logging',
        'json',
        'csv',
        'pathlib',
        'shutil',
        'tempfile',
        'subprocess',
        'sys',
        'time',
        'datetime',
        'copy',
        'itertools',
        'functools',
        'collections',
        'math',
        'statistics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SPADE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF

# Build using the spec file
echo "Building with comprehensive spec file..."
pyinstaller SPADE_comprehensive.spec

# Move to executables directory
if [ -f "dist/SPADE" ]; then
    mkdir -p executables
    mv dist/SPADE executables/SPADE_mac_comprehensive
    echo "Mac executable created: executables/SPADE_mac_comprehensive"
else
    echo "Error: SPADE not found in dist/ directory"
fi

# Clean up build artifacts
rm -rf build/
rm -rf dist/
rm -f SPADE_comprehensive.spec

echo "Comprehensive build complete!" 