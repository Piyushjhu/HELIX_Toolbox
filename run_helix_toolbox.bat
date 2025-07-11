@echo off
REM HELIX Toolbox Launcher for Windows
REM This batch file launches the HELIX Toolbox GUI

echo.
echo ========================================
echo    HELIX Toolbox - Windows Launcher
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "alpss_spade_gui.py" (
    echo ERROR: alpss_spade_gui.py not found in current directory
    echo Please run this batch file from the HELIX_Toolbox directory
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist "helix_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call helix_env\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
    echo Consider creating a virtual environment for better isolation.
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PyQt5, numpy, scipy, pandas, matplotlib, sklearn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies may be missing.
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

echo.
echo Starting HELIX Toolbox...
echo.

REM Launch the GUI
python alpss_spade_gui.py

REM Check if the GUI exited with an error
if errorlevel 1 (
    echo.
    echo ERROR: HELIX Toolbox encountered an error
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo HELIX Toolbox has been closed.
pause 