# Installation Guide

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.7 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB free space

### Python Dependencies
- PyQt5 (GUI framework)
- NumPy (numerical computing)
- SciPy (scientific computing)
- Pandas (data manipulation)
- Matplotlib (plotting)

## Installation Methods

### Method 1: Direct Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
   cd HELIX_Toolbox
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv helix_env
   
   # On Windows
   helix_env\Scripts\activate
   
   # On macOS/Linux
   source helix_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the GUI**
   ```bash
   python alpss_spade_gui.py
   ```

### Method 2: Using pip (Development)

1. **Install from GitHub**
   ```bash
   pip install git+https://github.com/Piyushjhu/HELIX_Toolbox.git
   ```

2. **Run the GUI**
   ```bash
   helix-toolbox
   ```

### Method 3: Development Installation

1. **Clone and install in development mode**
   ```bash
   git clone https://github.com/Piyushjhu/HELIX_Toolbox.git
   cd HELIX_Toolbox
   pip install -e .
   ```

## Troubleshooting

### Common Issues

#### PyQt5 Installation Problems

**Windows:**
```bash
pip install PyQt5
```

**macOS:**
```bash
brew install pyqt5
pip install PyQt5
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-pyqt5
pip install PyQt5
```

#### Matplotlib Backend Issues

If you encounter matplotlib backend errors:

```python
import matplotlib
matplotlib.use('Qt5Agg')
```

#### Permission Errors

**Windows:**
Run Command Prompt as Administrator

**macOS/Linux:**
```bash
sudo pip install -r requirements.txt
```

### Verification

To verify the installation:

1. **Run the test script**
   ```bash
   python -c "import alpss_spade_gui; print('HELIX Toolbox imported successfully!')"
   ```

2. **Check GUI launch**
   ```bash
   python alpss_spade_gui.py
   ```

## Updating

To update to the latest version:

```bash
cd HELIX_Toolbox
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove HELIX Toolbox:

```bash
pip uninstall helix-toolbox
```

Or if installed in development mode:

```bash
pip uninstall -e .
``` 