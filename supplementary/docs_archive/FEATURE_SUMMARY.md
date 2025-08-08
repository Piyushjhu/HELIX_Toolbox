# ALPSS-SPADE GUI Feature Enhancements

## Overview
This document summarizes the new features implemented in the ALPSS-SPADE GUI to improve user control over output generation and provide performance monitoring.

## New Features

### 1. ALPSS Output Image Selection

#### **Feature Description**
Users can now selectively choose which ALPSS output images to generate, allowing for faster processing and reduced disk usage when only specific plots are needed.

#### **Implementation Details**
- **Location**: ALPSS Parameters tab in the GUI
- **UI Components**: 
  - 7 checkboxes for different plot types
  - "Select All" and "Deselect All" buttons
  - Tooltips explaining each plot type

#### **Available Plot Types**
1. **Velocity vs Time Plot** - Velocity trace with uncertainty bands
2. **STFT Spectrogram** - Short-Time Fourier Transform spectrograms
3. **Filtered Signal Plot** - Original vs filtered signal comparison
4. **Phase Plot** - Phase vs time plots
5. **Amplitude Plot** - Amplitude vs time plots
6. **Peak Detection Plot** - Detected peaks and pullback visualization
7. **Uncertainty Analysis Plot** - Uncertainty analysis plots

#### **Technical Implementation**
- **GUI**: Added image selection group in `create_alpss_params_tab()`
- **Methods**: `select_all_alpss_images()` and `deselect_all_alpss_images()`
- **Parameter Collection**: Updated `get_alpss_params()` to include image selection parameters
- **ALPSS Integration**: Modified `simple_plotting()` function in `alpss_main.py` to respect selection parameters

#### **Usage**
1. Navigate to the "ALPSS Parameters" tab
2. Scroll to the "ALPSS Output Images" section
3. Check/uncheck desired plot types
4. Use "Select All" or "Deselect All" buttons for quick selection
5. Run analysis - only selected plots will be generated

### 2. Performance Monitoring

#### **Feature Description**
The analysis now tracks and reports timing information for each processing step, helping users understand performance characteristics and identify bottlenecks.

#### **Implementation Details**
- **Timing Granularity**: Per-file timing for ALPSS and overall timing for SPADE
- **Progress Updates**: Real-time timing information in progress messages
- **Summary Reports**: Average time per file and total processing time

#### **Timing Information Provided**
1. **Per-File ALPSS Timing**: Time taken for each individual file
2. **ALPSS Summary**: Total time and average time per file for ALPSS processing
3. **SPADE Timing**: Total time for SPADE analysis
4. **Overall Timing**: Total processing time for the entire analysis

#### **Technical Implementation**
- **Import**: Added `time` module import
- **Analysis Thread**: Added timing variables and calculations in `AnalysisThread.run()`
- **Progress Messages**: Enhanced progress updates with timing information
- **Start Time**: Initialize timing at the beginning of analysis

#### **Example Output**
```
ALPSS Processing file 1/3: example_file.csv
Completed ALPSS analysis for example_file.csv in 2.45 seconds
ALPSS Processing file 2/3: test_file.csv
Completed ALPSS analysis for test_file.csv in 2.31 seconds
ALPSS Processing file 3/3: data_file.csv
Completed ALPSS analysis for data_file.csv in 2.67 seconds
ALPSS Analysis Summary: 3 files processed in 7.43 seconds (avg: 2.48s per file)
Completed SPADE analysis for 3 files in 1.23 seconds
Total processing time: 8.66 seconds
```

## Benefits

### **Image Selection Benefits**
1. **Faster Processing**: Skip unnecessary plot generation
2. **Reduced Disk Usage**: Only save required plots
3. **Customized Output**: Generate only plots relevant to analysis
4. **Batch Processing**: Different selection for different file types

### **Performance Monitoring Benefits**
1. **Performance Analysis**: Identify slow files or processing steps
2. **Resource Planning**: Estimate processing time for large datasets
3. **Optimization**: Identify bottlenecks for improvement
4. **User Feedback**: Provide transparency about processing progress

## Testing

### **Image Selection Testing**
- ✅ Select All functionality
- ✅ Deselect All functionality  
- ✅ Individual checkbox selection
- ✅ Parameter collection and passing
- ✅ Integration with ALPSS plotting

### **Performance Monitoring Testing**
- ✅ Timing initialization
- ✅ Per-file timing calculation
- ✅ Progress message updates
- ✅ Summary reporting

## Future Enhancements

### **Potential Improvements**
1. **Advanced Image Selection**: Save/load image selection presets
2. **Performance Profiling**: Detailed breakdown of processing steps
3. **Memory Monitoring**: Track memory usage during processing
4. **Progress Visualization**: Real-time progress bars with timing
5. **Batch Optimization**: Parallel processing for multiple files

## Technical Notes

### **File Modifications**
- `alpss_spade_gui.py`: Added UI components and methods
- `ALPSS/alpss_main.py`: Modified plotting function for conditional generation
- `test_image_selection.py`: Test script for validation

### **Dependencies**
- No additional dependencies required
- Uses existing PyQt5 and matplotlib components
- Backward compatible with existing ALPSS functionality

### **Error Handling**
- Graceful handling of missing plot data
- Default to generating all plots if parameters not specified
- Maintains existing error handling in ALPSS processing 