# Troubleshooting Guide for SPADE

**Author:** Piyush Wanchoo  
**Institution:** Johns Hopkins University  
**Year:** 2025  
**GitHub:** [https://github.com/Piyushjhu/SPADE](https://github.com/Piyushjhu/SPADE)

## Common Issues and Solutions

### 1. "No module named pandas" Error

**Problem**: When running the Mac executable, you get an error like:
```
ModuleNotFoundError: No module named 'pandas'
```

**Causes**:
- PyInstaller didn't include all required dependencies in the executable
- Missing hidden imports in the build configuration
- Dependencies not installed in the build environment

**Solutions**:

#### Option A: Use the Updated Build Script
```bash
cd spall_analysis_release
./build_mac.sh
```

#### Option B: Use the Comprehensive Build Script (Recommended)
```bash
cd spall_analysis_release
./build_mac_comprehensive.sh
```

#### Option C: Manual Build with All Dependencies
```bash
cd spall_analysis_release

# Install dependencies
pip3 install -r requirements.txt
pip3 install pyinstaller

# Build with comprehensive hidden imports
pyinstaller --onefile --windowed --name=SPADE \
    --add-data=spall_analysis:spall_analysis \
    --add-data=data:data \
    --hidden-import=pandas \
    --hidden-import=numpy \
    --hidden-import=scipy \
    --hidden-import=matplotlib \
    --hidden-import=sklearn \
    --hidden-import=plotly \
    --hidden-import=PyQt5 \
    --hidden-import=spall_analysis \
    --collect-all=pandas \
    --collect-all=numpy \
    --collect-all=scipy \
    --collect-all=matplotlib \
    --collect-all=sklearn \
    --collect-all=plotly \
    --collect-all=PyQt5 \
    examples/spall_analysis_gui.py
```

### 2. Test Dependencies Before Building

Before building, run the dependency test script:
```bash
cd spall_analysis_release
python3 test_dependencies.py
```

This will verify that all required modules can be imported successfully.

### 3. Environment Issues

**Problem**: Build works but executable fails on different machines

**Solutions**:
- Build on a clean virtual environment
- Use the same Python version across environments
- Consider using Docker for consistent builds

### 4. PyQt5 Issues

**Problem**: GUI doesn't start or crashes

**Solutions**:
- Ensure PyQt5 is properly installed: `pip3 install PyQt5`
- On macOS, you might need: `pip3 install PyQt5-sip`
- Try building with `--windowed` flag removed for console output

### 5. File Size Issues

**Problem**: Executable is very large (>500MB)

**Solutions**:
- Use `--onefile` instead of `--onedir` for single file
- Exclude unnecessary modules with `--exclude-module`
- Consider using UPX compression (already enabled in build scripts)

### 6. Permission Issues

**Problem**: "Permission denied" when running executable

**Solutions**:
```bash
chmod +x executables/SPADE_mac_new
```

### 7. Missing Data Files

**Problem**: Executable can't find data files

**Solutions**:
- Ensure `--add-data=data:data` is included in PyInstaller command
- Check that data files exist in the source directory
- Verify file paths in the code are relative

## Build Process Checklist

1. **Prerequisites**:
   - Python 3.7+ installed
   - pip3 available
   - Working directory is `spall_analysis_release`

2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   pip3 install pyinstaller
   ```

3. **Test Dependencies**:
   ```bash
   python3 test_dependencies.py
   ```

4. **Build Executable**:
   ```bash
   ./build_mac_comprehensive.sh
   ```

5. **Test Executable**:
   ```bash
   ./executables/SPADE_mac_comprehensive
   ```

## Debugging Tips

### Enable Console Output
Remove `--windowed` flag to see error messages:
```bash
pyinstaller --onefile --name=SPADE [other options] examples/spall_analysis_gui.py
```

### Check Executable Contents
```bash
# List contents of executable (macOS)
otool -L executables/SPADE_mac_new

# Check for missing dependencies
ldd executables/SPADE_mac_new  # Linux equivalent
```

### Verbose PyInstaller Output
```bash
pyinstaller --onefile --windowed --name=SPADE --log-level=DEBUG [other options] examples/spall_analysis_gui.py
```

## Getting Help

If you continue to have issues:

1. Run the dependency test script and share the output
2. Check the console output when running the executable
3. Verify your Python version: `python3 --version`
4. Check installed packages: `pip3 list`
5. Try building in a fresh virtual environment

## Alternative Solutions

If PyInstaller continues to cause issues, consider:

1. **Using cx_Freeze**: Alternative Python-to-executable tool
2. **Using py2app**: macOS-specific packaging tool
3. **Distributing as Python package**: Install via pip instead of executable
4. **Using Docker**: Containerized solution for consistent environments 