#!/bin/bash
# Script to plot velocity traces for all materials (Zn, Cu, Brass)

PYTHON=/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/HELIX_Toolbox_v_2/helix_toolbox_env/bin/python
SCRIPT=/Users/piyushwanchoo/Documents/Post_Doc/DATA_ANALYSIS/HELIX_Toolbox_v_2/supplementary/paper_plots/plot_velocity_traces_by_laser_energy.py
SUMMARY=/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/Output_new/SPADE_analysis/enhanced_spall_summary.csv
VELOCITY_DIR=/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/Output_new
BASE_OUTPUT=/Users/piyushwanchoo/Documents/Post_Doc/1000_RUN_SHOTS/Output_new
BATCH_SIZE=10

echo "========================================="
echo "Plotting Velocity Traces for All Materials"
echo "========================================="
echo ""

# Plot Zn traces
echo ">>> Processing Zn traces..."
$PYTHON $SCRIPT \
    --summary $SUMMARY \
    --velocity-dir $VELOCITY_DIR \
    --output-dir ${BASE_OUTPUT}/Zn_velocity_plots \
    --material Zn \
    --batch-size $BATCH_SIZE

echo ""
echo "----------------------------------------"
echo ""

# Plot Cu traces
echo ">>> Processing Cu traces..."
$PYTHON $SCRIPT \
    --summary $SUMMARY \
    --velocity-dir $VELOCITY_DIR \
    --output-dir ${BASE_OUTPUT}/Cu_velocity_plots \
    --material Cu \
    --batch-size $BATCH_SIZE

echo ""
echo "----------------------------------------"
echo ""

# Plot Brass traces
echo ">>> Processing Brass traces..."
$PYTHON $SCRIPT \
    --summary $SUMMARY \
    --velocity-dir $VELOCITY_DIR \
    --output-dir ${BASE_OUTPUT}/Brass_velocity_plots \
    --material Brass \
    --batch-size $BATCH_SIZE

echo ""
echo "========================================="
echo "All materials processed!"
echo "========================================="
echo "Output directories:"
echo "  - ${BASE_OUTPUT}/Zn_velocity_plots/"
echo "  - ${BASE_OUTPUT}/Cu_velocity_plots/"
echo "  - ${BASE_OUTPUT}/Brass_velocity_plots/"
echo "========================================="
