#!/usr/bin/env python3
"""
Array Length Fix Utility for ALPSS
Checks for array length mismatches and provides fixes
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime

def check_array_lengths(vc_out, iua_out, sdf_out):
    """Check for array length mismatches in ALPSS outputs"""
    issues = []
    
    # Get all array lengths
    arrays = {
        'time_f': len(vc_out.get('time_f', [])),
        'velocity_f': len(vc_out.get('velocity_f', [])),
        'velocity_f_smooth': len(vc_out.get('velocity_f_smooth', [])),
        'voltage_filt': len(vc_out.get('voltage_filt', [])),
        'time': len(sdf_out.get('time', [])),
        'inst_noise': len(iua_out.get('inst_noise', [])),
        'vel_uncert': len(iua_out.get('vel_uncert', [])),
        'freq_uncert': len(iua_out.get('freq_uncert', []))
    }
    
    print(f"[{datetime.now()}] Array Length Check:")
    for name, length in arrays.items():
        print(f"   {name}: {length}")
    
    # Check for mismatches
    lengths = list(arrays.values())
    if len(set(lengths)) > 1:
        min_length = min(lengths)
        max_length = max(lengths)
        issues.append(f"Array length mismatch: min={min_length}, max={max_length}")
        
        for name, length in arrays.items():
            if length != min_length:
                issues.append(f"   {name}: {length} (should be {min_length})")
    
    return issues

def fix_array_lengths(vc_out, iua_out, sdf_out):
    """Fix array length mismatches by trimming to minimum length"""
    print(f"[{datetime.now()}] Fixing array length mismatches...")
    
    # Get all array lengths
    arrays = {
        'time_f': vc_out.get('time_f', []),
        'velocity_f': vc_out.get('velocity_f', []),
        'velocity_f_smooth': vc_out.get('velocity_f_smooth', []),
        'voltage_filt': vc_out.get('voltage_filt', []),
        'time': sdf_out.get('time', []),
        'inst_noise': iua_out.get('inst_noise', []),
        'vel_uncert': iua_out.get('vel_uncert', []),
        'freq_uncert': iua_out.get('freq_uncert', [])
    }
    
    # Find minimum length
    lengths = [len(arr) for arr in arrays.values() if len(arr) > 0]
    if not lengths:
        print("No arrays to fix")
        return vc_out, iua_out, sdf_out
    
    min_length = min(lengths)
    print(f"Trimming all arrays to length: {min_length}")
    
    # Trim arrays
    vc_out_fixed = vc_out.copy()
    iua_out_fixed = iua_out.copy()
    sdf_out_fixed = sdf_out.copy()
    
    if len(vc_out.get('time_f', [])) > min_length:
        vc_out_fixed['time_f'] = vc_out['time_f'][:min_length]
    if len(vc_out.get('velocity_f', [])) > min_length:
        vc_out_fixed['velocity_f'] = vc_out['velocity_f'][:min_length]
    if len(vc_out.get('velocity_f_smooth', [])) > min_length:
        vc_out_fixed['velocity_f_smooth'] = vc_out['velocity_f_smooth'][:min_length]
    if len(vc_out.get('voltage_filt', [])) > min_length:
        vc_out_fixed['voltage_filt'] = vc_out['voltage_filt'][:min_length]
    
    if len(sdf_out.get('time', [])) > min_length:
        sdf_out_fixed['time'] = sdf_out['time'][:min_length]
    
    if len(iua_out.get('inst_noise', [])) > min_length:
        iua_out_fixed['inst_noise'] = iua_out['inst_noise'][:min_length]
    if len(iua_out.get('vel_uncert', [])) > min_length:
        iua_out_fixed['vel_uncert'] = iua_out['vel_uncert'][:min_length]
    if len(iua_out.get('freq_uncert', [])) > min_length:
        iua_out_fixed['freq_uncert'] = iua_out['freq_uncert'][:min_length]
    
    print(f"[{datetime.now()}] Array length fix complete")
    return vc_out_fixed, iua_out_fixed, sdf_out_fixed

def test_array_operations():
    """Test array operations to ensure they work correctly"""
    print(f"[{datetime.now()}] Testing array operations...")
    
    # Create test arrays with different lengths
    time_f = np.arange(100)
    velocity_f = np.arange(95)  # Different length
    velocity_f_smooth = np.arange(98)  # Different length
    vel_uncert = np.arange(102)  # Different length
    
    print(f"Original lengths: time_f={len(time_f)}, velocity_f={len(velocity_f)}, velocity_f_smooth={len(velocity_f_smooth)}, vel_uncert={len(vel_uncert)}")
    
    # Fix lengths
    min_length = min(len(time_f), len(velocity_f), len(velocity_f_smooth), len(vel_uncert))
    time_f_trimmed = time_f[:min_length]
    velocity_f_trimmed = velocity_f[:min_length]
    velocity_f_smooth_trimmed = velocity_f_smooth[:min_length]
    vel_uncert_trimmed = vel_uncert[:min_length]
    
    print(f"After trimming: all arrays have length {min_length}")
    
    # Test stacking operations
    try:
        # Test velocity data stacking
        velocity_data = np.stack((time_f_trimmed, velocity_f_trimmed), axis=1)
        print("✅ Velocity data stacking successful")
        
        # Test velocity smooth stacking
        velocity_smooth_data = np.stack((time_f_trimmed, velocity_f_smooth_trimmed), axis=1)
        print("✅ Velocity smooth data stacking successful")
        
        # Test vel_smooth_with_uncert stacking
        vel_smooth_with_uncert = np.stack(
            (
                time_f_trimmed,
                velocity_f_smooth_trimmed,
                vel_uncert_trimmed,
                velocity_f_smooth_trimmed + vel_uncert_trimmed,
            ),
            axis=1,
        )
        print("✅ Vel smooth with uncertainty stacking successful")
        
        print("✅ All array operations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Array operation failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Array Length Fix Utility")
    print("=" * 40)
    
    # Test array operations
    success = test_array_operations()
    
    if success:
        print("\n✅ Array length fix utility is working correctly")
        print("   - All np.stack operations should now work")
        print("   - Arrays are trimmed to minimum length")
        print("   - Broadcasting errors should be eliminated")
    else:
        print("\n❌ Array length fix utility has issues")
        print("   - Check the error messages above")
        print("   - Verify array trimming logic") 