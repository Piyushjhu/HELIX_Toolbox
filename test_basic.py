#!/usr/bin/env python3
"""
Basic test script for HELIX Toolbox
Tests core functionality without GUI dependencies
"""

import sys
import os
import numpy as np
import pandas as pd

def test_imports():
    """Test all basic imports"""
    print("Testing basic imports...")
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ scipy imported successfully")
    except ImportError as e:
        print(f"❌ scipy import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ sklearn imported successfully")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
        return False
    
    return True

def test_alpss_imports():
    """Test ALPSS imports"""
    print("\nTesting ALPSS imports...")
    
    try:
        from ALPSS.alpss_main import alpss_main
        print("✅ ALPSS main module imported successfully")
    except ImportError as e:
        print(f"❌ ALPSS import failed: {e}")
        return False
    
    return True

def test_spade_imports():
    """Test SPADE imports"""
    print("\nTesting SPADE imports...")
    
    try:
        from SPADE.spall_analysis_release.spall_analysis import process_velocity_files
        print("✅ SPADE main module imported successfully")
    except ImportError as e:
        print(f"❌ SPADE import failed: {e}")
        return False
    
    return True

def test_gui_imports():
    """Test GUI imports (if possible)"""
    print("\nTesting GUI imports...")
    
    try:
        import alpss_spade_gui
        print("✅ GUI module imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  GUI import failed (expected in headless environment): {e}")
        return True  # Not a failure in headless environment
    
    except Exception as e:
        print(f"❌ GUI import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    # Test numpy operations
    try:
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        assert mean_val == 3.0
        print("✅ numpy basic operations work")
    except Exception as e:
        print(f"❌ numpy operations failed: {e}")
        return False
    
    # Test pandas operations
    try:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        print("✅ pandas basic operations work")
    except Exception as e:
        print(f"❌ pandas operations failed: {e}")
        return False
    
    # Test scipy operations
    try:
        from scipy import signal
        # Create a simple signal
        t = np.linspace(0, 1, 1000)
        signal_data = np.sin(2 * np.pi * 10 * t)
        # Apply a simple filter
        filtered = signal.savgol_filter(signal_data, 51, 3)
        assert len(filtered) == len(signal_data)
        print("✅ scipy signal processing works")
    except Exception as e:
        print(f"❌ scipy operations failed: {e}")
        return False
    
    return True

def test_sklearn_functionality():
    """Test sklearn functionality"""
    print("\nTesting sklearn functionality...")
    
    try:
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.linear_model import ElasticNetCV
        
        # Test MinMaxScaler
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        assert scaled_data.shape == data.shape
        print("✅ sklearn preprocessing works")
        
        # Test ElasticNetCV
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = ElasticNetCV()
        model.fit(X, y)
        print("✅ sklearn linear models work")
        
    except Exception as e:
        print(f"❌ sklearn operations failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("HELIX Toolbox Basic Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_alpss_imports,
        test_spade_imports,
        test_gui_imports,
        test_basic_functionality,
        test_sklearn_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HELIX Toolbox is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 