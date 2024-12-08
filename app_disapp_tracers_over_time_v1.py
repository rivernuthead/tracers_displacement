#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:44:30 2024

@author: erri
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import time
start_time = time.time()
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNCTIONS
# =============================================================================
# Function to count rows not full of NaN for each time step
def count_non_nan_rows(stack):
    non_nan_counts = []
    for t in range(stack.shape[0]):
        rows_not_nan = np.any(~np.isnan(stack[t, :, :]), axis=1)
        non_nan_counts.append(np.sum(rows_not_nan))
    return np.array(non_nan_counts)

# =============================================================================
# SCRIPT PARAMETERS
# =============================================================================
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
             'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
             'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q05_1r1']


# =============================================================================
# FOLDERS SETUP
# =============================================================================


for run_name in run_names:
    print(run_name, ' is running...')
    
    
    # FOLDERS SETUP
    output_dir = os.path.join(os.getcwd(), 'output_data')
    path_out = os.path.join(os.getcwd(), 'output_data', 'output_images', run_name)
    plot_dir = os.path.join(os.getcwd(), 'output_data', 'number_tracers_analtysis')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    
    # LOAD DATA
    tracers_stack        = np.load(os.path.join(path_out,'alltracers_'+ run_name +'.npy'))
    tracers_app_stack    = np.load(os.path.join(path_out, 'tracers_appeared_'+ run_name +'.npy'))
    tracers_disapp_stack = np.load(os.path.join(path_out, 'tracers_disappeared_'+ run_name +'.npy'))
    
    # COUNT NUMER OF TRACERS
    non_nan_tracers = count_non_nan_rows(tracers_stack)
    non_nan_tracers_app = count_non_nan_rows(tracers_app_stack)
    non_nan_tracers_disapp = count_non_nan_rows(tracers_disapp_stack)
    
    
    # PLOTTING
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    time_steps = np.arange(tracers_stack.shape[0])*4/60 # Time in minutes
    
    # Primary Y-axis
    ax1.plot(time_steps[:-1], non_nan_tracers[:-1], label="Tracers Stack", marker="o", linestyle="--", linewidth=2, color='blue')
    ax1.set_xlabel("Time [min]", fontsize=14)
    ax1.set_ylabel("Number of detecred tracers", fontsize=14, color='blue')
    ax1.tick_params(axis='y', labelsize=12, color='blue')
    ax1.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    # Secondary Y-axis
    ax2 = ax1.twinx()
    ax2.plot(time_steps[1:-1], non_nan_tracers_app[1:], label="Tracers App", marker="s", linestyle="-", linewidth=1, color="green")
    ax2.plot(time_steps[1:-1], non_nan_tracers_disapp[1:], label="Tracers Disapp", marker="d", linestyle="-", linewidth=1, color='red')
    ax2.set_ylabel("Number of appeared and disappeared tracers", fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    
    # Title and legend
    fig.suptitle("Detected, appeared, and disappeared tracers - Run: " + run_name, fontsize=16, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.025), fontsize=12, ncol=3)
    plt.text(0.5, -0.2, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, run_name + '_detected_appeared_disappeared_over_time.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
