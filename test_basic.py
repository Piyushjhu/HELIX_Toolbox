#!/usr/bin/env python3
"""
Basic tests for HELIX Toolbox
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import numpy as np
        print("OK NumPy imported successfully")
    except ImportError as e:
        print(f"FAIL NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("OK Pandas imported successfully")
    except ImportError as e:
        print(f"FAIL Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("OK Matplotlib imported successfully")
    except ImportError as e:
        print(f"FAIL Matplotlib import failed: {e}")
        return False
    
    try:
        from PyQt5.QtWidgets import QApplication
        print("OK PyQt5 imported successfully")
    except ImportError as e:
        print(f"FAIL PyQt5 import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that required files exist"""
    required_files = [
        'helix_analysis_toolbox.py',
        'requirements.txt',
        'README.md',
        'setup.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"OK {file} exists")
        else:
            print(f"FAIL {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_supplementary_structure():
    """Test that supplementary directory structure is correct"""
    if os.path.exists('supplementary'):
        print("OK supplementary directory exists")
        
        # Check for README
        if os.path.exists('supplementary/README.md'):
            print("OK supplementary/README.md exists")
        else:
            print("FAIL supplementary/README.md missing")
            return False
        
        # Check for key files
        key_files = [
            'supplementary/analysis_monitor.py',
            'supplementary/realtime_monitor.py',
            'supplementary/enhanced_plotting.py'
        ]
        
        for file in key_files:
            if os.path.exists(file):
                print(f"OK {file} exists")
            else:
                print(f"FAIL {file} missing")
                return False
        
        return True
    else:
        print("FAIL supplementary directory missing")
        return False

def main():
    """Run all tests"""
    print("Running HELIX Toolbox basic tests...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("File Structure Tests", test_file_structure),
        ("Supplementary Structure Tests", test_supplementary_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"OK {test_name} PASSED")
                passed += 1
            else:
                print(f"FAIL {test_name} FAILED")
        except Exception as e:
            print(f"FAIL {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 