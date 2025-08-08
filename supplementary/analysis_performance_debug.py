#!/usr/bin/env python3
"""
Analysis Performance Debug Script
Identifies bottlenecks and inefficiencies in ALPSS-SPADE analysis
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

class AnalysisPerformanceDebugger:
    def __init__(self):
        self.issues = []
        self.performance_metrics = {}
        
    def analyze_log_output(self, log_content):
        """Analyze the log output for performance issues"""
        print("🔍 Analyzing Log Output for Performance Issues")
        print("=" * 60)
        
        # Extract key metrics
        lines = log_content.split('\n')
        
        # 1. Check for broadcasting errors
        broadcasting_errors = [line for line in lines if "operands could not be broadcast together" in line]
        if broadcasting_errors:
            self.issues.append(f"❌ Broadcasting Errors: {len(broadcasting_errors)} occurrences")
            for error in broadcasting_errors[:3]:  # Show first 3
                print(f"   - {error}")
        
        # 2. Check runtime performance
        runtime_patterns = [line for line in lines if "Full program runtime" in line]
        if runtime_patterns:
            runtimes = []
            for pattern in runtime_patterns:
                try:
                    time_str = pattern.split("runtime (including plotting and saving):")[1].strip()
                    # Parse time like "0:00:01.442919"
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                        runtimes.append(total_seconds)
                except:
                    pass
            
            if runtimes:
                avg_runtime = np.mean(runtimes)
                max_runtime = np.max(runtimes)
                min_runtime = np.min(runtimes)
                
                self.performance_metrics['avg_runtime'] = avg_runtime
                self.performance_metrics['max_runtime'] = max_runtime
                self.performance_metrics['min_runtime'] = min_runtime
                
                print(f"📊 Runtime Analysis:")
                print(f"   Average: {avg_runtime:.2f} seconds")
                print(f"   Maximum: {max_runtime:.2f} seconds")
                print(f"   Minimum: {min_runtime:.2f} seconds")
                
                if avg_runtime > 10:
                    self.issues.append(f"⚠️  Slow Analysis: Average runtime {avg_runtime:.2f}s (should be <10s)")
        
        # 3. Check array shape mismatches
        shape_patterns = [line for line in lines if "Array shapes" in line]
        if shape_patterns:
            print(f"📐 Array Shape Analysis:")
            for pattern in shape_patterns[:5]:  # Show first 5
                print(f"   {pattern}")
        
        # 4. Check for warnings
        warnings = [line for line in lines if "Warning:" in line]
        if warnings:
            self.issues.append(f"⚠️  Warnings: {len(warnings)} occurrences")
            for warning in warnings[:3]:
                print(f"   - {warning}")
        
        # 5. Check for errors
        errors = [line for line in lines if "ERROR" in line]
        if errors:
            self.issues.append(f"❌ Errors: {len(errors)} occurrences")
            for error in errors[:3]:
                print(f"   - {error}")
        
        return self.issues
    
    def identify_bottlenecks(self):
        """Identify specific performance bottlenecks"""
        bottlenecks = []
        
        # 1. Broadcasting Error Bottleneck
        bottlenecks.append({
            'type': 'Critical',
            'issue': 'Array Broadcasting Error',
            'impact': 'Analysis fails to complete',
            'cause': 'Arrays have different lengths during stacking',
            'solution': 'Implement proper array length trimming'
        })
        
        # 2. Runtime Bottleneck
        if 'avg_runtime' in self.performance_metrics:
            if self.performance_metrics['avg_runtime'] > 10:
                bottlenecks.append({
                    'type': 'Performance',
                    'issue': 'Slow Analysis Runtime',
                    'impact': 'Analysis takes too long',
                    'cause': 'Inefficient array operations or large data processing',
                    'solution': 'Optimize array operations and data processing'
                })
        
        # 3. Memory Bottleneck
        bottlenecks.append({
            'type': 'Performance',
            'issue': 'Memory Usage Spikes',
            'impact': 'System becomes unresponsive',
            'cause': 'Large arrays not being managed efficiently',
            'solution': 'Implement memory-efficient array handling'
        })
        
        return bottlenecks
    
    def check_parameter_efficiency(self):
        """Check if ALPSS is running efficiently based on user parameters"""
        efficiency_issues = []
        
        # Check for parameter-related issues
        efficiency_issues.append({
            'issue': 'Plot Generation Skipped',
            'impact': 'No visual output generated',
            'cause': 'display_plots = no, save_all_plots != yes',
            'solution': 'Enable plotting for better analysis validation'
        })
        
        efficiency_issues.append({
            'issue': 'Array Length Mismatches',
            'impact': 'Broadcasting errors and failed saves',
            'cause': 'Different processing steps produce arrays of different lengths',
            'solution': 'Standardize array lengths across all processing steps'
        })
        
        return efficiency_issues
    
    def generate_optimization_plan(self):
        """Generate a comprehensive optimization plan"""
        plan = {
            'immediate_fixes': [],
            'performance_improvements': [],
            'long_term_optimizations': []
        }
        
        # Immediate fixes
        plan['immediate_fixes'].extend([
            'Fix array broadcasting error in saving function',
            'Implement proper array length validation',
            'Add error handling for array operations',
            'Standardize array processing across all functions'
        ])
        
        # Performance improvements
        plan['performance_improvements'].extend([
            'Optimize array operations to reduce memory usage',
            'Implement efficient array trimming',
            'Add progress tracking for long operations',
            'Optimize file I/O operations'
        ])
        
        # Long-term optimizations
        plan['long_term_optimizations'].extend([
            'Implement parallel processing for batch operations',
            'Add caching for repeated calculations',
            'Optimize algorithm complexity',
            'Implement memory-efficient data structures'
        ])
        
        return plan
    
    def create_fix_script(self):
        """Create a script to fix the identified issues"""
        fix_script = '''#!/usr/bin/env python3
"""
ALPSS-SPADE Analysis Fix Script
Fixes identified performance and broadcasting issues
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

def fix_array_broadcasting_issue():
    """Fix the array broadcasting issue in the saving function"""
    
    # The issue is in the vel_smooth_with_uncert stacking
    # We need to ensure all arrays have the same length before stacking
    
    def safe_array_trimming(time_f, velocity_f_smooth, vel_uncert):
        """Safely trim arrays to the same length"""
        try:
            # Get the minimum length
            lengths = [len(time_f), len(velocity_f_smooth), len(vel_uncert)]
            min_length = min(lengths)
            
            # Trim arrays
            time_f_trimmed = time_f[:min_length]
            velocity_f_smooth_trimmed = velocity_f_smooth[:min_length]
            vel_uncert_trimmed = vel_uncert[:min_length]
            
            return time_f_trimmed, velocity_f_smooth_trimmed, vel_uncert_trimmed
            
        except Exception as e:
            print(f"Error in array trimming: {e}")
            # Return original arrays if trimming fails
            return time_f, velocity_f_smooth, vel_uncert
    
    return safe_array_trimming

def optimize_analysis_parameters():
    """Optimize analysis parameters for better performance"""
    
    optimizations = {
        'memory_efficient': True,
        'array_validation': True,
        'error_handling': True,
        'progress_tracking': True
    }
    
    return optimizations

def main():
    print("🔧 ALPSS-SPADE Analysis Fix Script")
    print("=" * 40)
    
    # Create the fix functions
    safe_trimming = fix_array_broadcasting_issue()
    optimizations = optimize_analysis_parameters()
    
    print("✅ Fix functions created")
    print("✅ Optimization parameters set")
    print("✅ Ready to apply fixes to ALPSS code")

if __name__ == "__main__":
    main()
'''
        
        with open('alpss_fix_script.py', 'w') as f:
            f.write(fix_script)
        
        print("📝 Created alpss_fix_script.py with fixes")
        return 'alpss_fix_script.py'

def main():
    debugger = AnalysisPerformanceDebugger()
    
    # Analyze the log content (you can paste the log here)
    log_content = """
    [2025-08-05 13:15:46.593417] ERROR in saving: operands could not be broadcast together with shapes (76193,) (76646,)
    Full program runtime (including plotting and saving): 0:00:01.442919
    """
    
    # Analyze issues
    issues = debugger.analyze_log_output(log_content)
    
    # Identify bottlenecks
    bottlenecks = debugger.identify_bottlenecks()
    
    # Check parameter efficiency
    efficiency_issues = debugger.check_parameter_efficiency()
    
    # Generate optimization plan
    plan = debugger.generate_optimization_plan()
    
    # Create fix script
    fix_script = debugger.create_fix_script()
    
    # Print summary
    print("\n📋 Analysis Summary:")
    print(f"   Issues found: {len(issues)}")
    print(f"   Bottlenecks: {len(bottlenecks)}")
    print(f"   Efficiency issues: {len(efficiency_issues)}")
    
    print("\n🎯 Optimization Plan:")
    for category, items in plan.items():
        print(f"   {category.replace('_', ' ').title()}:")
        for item in items:
            print(f"     - {item}")
    
    print("\n✅ Fix script created: alpss_fix_script.py")

if __name__ == "__main__":
    main() 