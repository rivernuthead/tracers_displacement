#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:31:26 2024

@author: erri
"""

#%%
# =============================================================================
# SCOUR, NOT DETECTED, AND FILL POSITION OF TRACERS OVER TIME
# =============================================================================
# IMPORT LIBRARIES
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


start_time = time.time()


# SCRIPT PARAMETERS
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
              'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
              'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# run_names = ['q07_1r11']



# SURVEY PIXEL DIMENSION
px_x, px_y = 50, 5 # [mm]

# FOLDERS SETUP
# w_dir = '/Volumes/T7_Shield/PhD/repos/tracers_displacement/'
w_dir = os.getcwd()
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')

run_parameters = pd.read_csv(os.path.join(w_dir, 'input_data', 'tracers_DEM_DoD_combination_v2.csv'))

for run_name in run_names:
    
    path_in_DoD = os.path.join(input_dir,'DoDs','DoD_'+run_name[0:5])
    
    start_time = time.time()
    
    # Initialize arrays to store the counts
    positive_counts_over_time = []
    negative_counts_over_time = []
    zero_counts_over_time = []
    
    print(run_name, '...is running.')
    # =============================================================================
    # IMPORT FILES
    # =============================================================================
    # tracers_reduced_path = os.path.join(w_dir, 'output_data', 'output_images', run_name, 'tracers_reduced_'+run_name+'.npy')
    # tracers_reduced_raw = np.load(tracers_reduced_path)
    
    tracers_path = os.path.join(w_dir, 'output_data', 'output_images', run_name,'alltracers_'+ run_name +'.npy')
    tracers_raw = np.load(tracers_path)
    tracers_raw = tracers_raw[:, :, 1:]
    tracers_reduced_raw = np.copy(tracers_raw)
    
    
    for t in range(tracers_reduced_raw.shape[0]):
        # Sort rows based on the values in the first column (column index 0) for each time slice
        tracers_reduced_raw[t] = tracers_reduced_raw[t][np.argsort(tracers_reduced_raw[t][:, 0])]

    # REMOVE TRACERS THAT COME FROM UPSTREAM AND HAVE NEGATIVE X VALUE
    for t in range(tracers_reduced_raw.shape[0]):
        negative_indices = np.where(tracers_reduced_raw[t, :, 0] < 0)[0]
        tracers_reduced_raw[t, negative_indices, :] = np.nan

    # # REMOVE EMPTY MATRICES
    tracers_reduced = tracers_reduced_raw
    
    selected_row = run_parameters.loc[run_parameters['RUN'] == run_name]
    if not selected_row.empty:
        
        DEM_name = int(selected_row['DEM'].values[0]) # DEM name
        
        DoD_name = selected_row['DoD timespan1'].values[0] # DoD name timespan 1
        DoD_name_tspan2_v1 = selected_row['DoD timespan2'].values[0] # DoD name timespan 2 - combination 1
        DoD_name_tspan2_v2 = selected_row['DoD timespan2.1'].values[0] # DoD name timespan 2 - combination 2
        DoD_name_tspan1_pre = selected_row['DoD timespan1 pre'].values[0]
        DoD_name_tspan1_post = selected_row['DoD timespan1 post'].values[0]
        
        feed_x = selected_row['feed-x'].values[0] # Feeding x-coordinate
        
        frame_position = selected_row['frame_position 1  [m]'].values[0] # Frame position in meters
        frame_position += 0.44 # # Frame position with respect to the CRS of the DEM (The beginning of the DEM is at -0.44 meters with respect to the laser survey CRS)
        
        frame_position2 = selected_row['frame_position 2  [m]'].values[0] # Frame position 2 in meters
        
    DoD_path = os.path.join(path_in_DoD, 'DoD_' + DoD_name+'_filt_ult.txt')
    DoD = np.loadtxt(DoD_path)
    
    DoD_tspan2_1_path = os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan2_v1+'_filt_ult.txt')
    DoD_tspan2_1 = np.loadtxt(DoD_tspan2_1_path)
    
    DoD_tspan2_2_path = os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan2_v2+'_filt_ult.txt')
    DoD_tspan2_2 = np.loadtxt(DoD_tspan2_2_path)
    
    # LOAD THE PRE AND POST DoD -----------------------------------------------
    # CHECK PRE
    if not DoD_name_tspan1_pre == 'False':
        DoD_tspan1_pre = np.loadtxt(os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan1_pre+'_filt_ult.txt'))
    
    # CHECK POST
    if not DoD_name_tspan1_post == 'False':
        DoD_tspan1_post = np.loadtxt(os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan1_post+'_filt_ult.txt'))
    
    # ADD THE VALUES OF THE DOD
    t, r, c = tracers_reduced.shape
    tracers_reduced = np.concatenate((tracers_reduced, np.full((t, r, 2), np.nan)), axis=2)
    # Update the new column in the stack for each t and row
    for i in range(t):
        for j in range(r):
            
            if tracers_reduced[i,j,3]==0:
                # print('time: ', i, '  row: ', j)
                pure_filt_check=[]
                weak_dep_check=[]
                    
                # CHECK PRE
                if not DoD_name_tspan1_pre == 'False':
                    DoD_tspan2_1_value_pre = DoD_tspan2_1[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    value_tspan2_pre = (DoD_tspan2_1_value_pre>0)*1
                    if value_tspan2_pre ==1:

                        DoD_tspan1_value_pre = DoD_tspan1_pre[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                        
                        # print(DoD_tspan1_value_pre, '   ',(DoD_tspan1_value_pre>0)*1)
                        # print(DoD_tspan2_1_value_pre, '   ', (DoD_tspan2_1_value_pre>0)*1)
                        
                        value_tspan1_pre = (DoD_tspan1_value_pre>0)*1

                        
                        if value_tspan1_pre == 0: # Check pure filtering effect
                            # print('Pure filt')
                            pure_filt_check.append(True)
                        
                        if value_tspan1_pre == 1: # Check weak deposition
                            # print('Weak dep')
                            weak_dep_check.append(True)
                        
                # CHECK POST
                if not DoD_name_tspan1_post == 'False':
                    DoD_tspan2_2_value_post = DoD_tspan2_2[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    value_tspan2_post = (DoD_tspan2_2_value_post>0)*1
                    if value_tspan2_post ==1:

                        DoD_tspan1_value_post = DoD_tspan1_post[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                        
                        value_tspan1_post = (DoD_tspan1_value_post>0)*1

                        
                        if value_tspan1_post == 0: # Check pure filtering effect
                            pure_filt_check.append(True)
                        
                        if value_tspan1_post == 1: # Check weak deposition
                            weak_dep_check.append(True)
                
                        
                if any(pure_filt_check):
                    tracers_reduced[i,j,4]=1
                    
                if any(weak_dep_check):
                    tracers_reduced[i,j,5]=1
                
                
                
                
                
                
                
                
    # =============================================================================
    # COUNT POSITIVE, NEGATIVE, ZERO VALUES, PURE FILTERING, WEAK DEPOSIT, AND COMBINED FILL
    # =============================================================================
    # Initialize lists to store counts
    positive_counts_over_time = []
    negative_counts_over_time = []
    zero_counts_over_time = []
    pure_filtering_counts_over_time = []  # For "Pure filtering"
    weak_deposit_counts_over_time = []   # For "Weak deposit"
    combined_fill_counts_over_time = []  # For "Fill + Pure filtering + Weak deposit"
    
    for t in range(tracers_reduced.shape[0]-1):
        # Extract the slice for the current time step
        values = tracers_reduced[t, :, 3]  # Column index 3
        pure_filtering_values = tracers_reduced[t, :, 4]  # Column index 4
        weak_deposit_values = tracers_reduced[t, :, 5]    # Column index 5
    
        # Trim np.nan values
        values = values[~np.isnan(values)]
        pure_filtering_values = pure_filtering_values[~np.isnan(pure_filtering_values)]
        weak_deposit_values = weak_deposit_values[~np.isnan(weak_deposit_values)]
    
        # Count positive, negative, and zero values
        positive_count = np.sum(values > 0)
        negative_count = np.sum(values < 0)
        zero_count = np.sum(values == 0)
    
        # Count pure filtering (column index 4) and weak deposit (column index 5) values
        pure_filtering_count = np.sum(pure_filtering_values == 1)
        weak_deposit_count = np.sum(weak_deposit_values == 1)
    
        # Compute the combined fill count
        combined_fill_count = positive_count + pure_filtering_count + weak_deposit_count
    
        # Append counts to their respective lists
        positive_counts_over_time.append(positive_count)
        negative_counts_over_time.append(negative_count)
        zero_counts_over_time.append(zero_count)
        pure_filtering_counts_over_time.append(pure_filtering_count)
        weak_deposit_counts_over_time.append(weak_deposit_count)
        combined_fill_counts_over_time.append(combined_fill_count)
    
    # =============================================================================
    # PLOT THE TRENDS
    # =============================================================================
    time_steps = np.arange(len(positive_counts_over_time)) * 4 / 60  # Convert time steps to minutes
    
    plt.figure(figsize=(14, 7))
    
    # Plot the existing curves
    plt.plot(time_steps, positive_counts_over_time, label='Fill', color='blue', linewidth=2)
    plt.plot(time_steps, zero_counts_over_time, label='Not-detected', color='gray', linestyle='-', linewidth=2)
    plt.plot(time_steps, negative_counts_over_time, label='Scour', color='red', linewidth=2)
    
    # Plot the new curves
    plt.plot(time_steps, pure_filtering_counts_over_time, label='Pure filtering', color='green', linestyle='--', linewidth=2)
    plt.plot(time_steps, weak_deposit_counts_over_time, label='Weak deposit', color='purple', linestyle='-.', linewidth=2)
    
    # Plot the combined fill line
    plt.plot(time_steps, combined_fill_counts_over_time, label='Fill + Pure filtering + Weak deposit', color='blue', linestyle='--', linewidth=3)
    
    # Add labels, title, and legend
    plt.title('Trend of fill, not-detected, scour, pure filtering, weak deposit, and combined fill over time\n' + run_name, fontsize=16)
    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Number of Values', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save and show the plot
    plt.tight_layout()
    plt.text(0.5, -0.15, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.savefig(os.path.join(output_dir, run_name + '_temporal_trends.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
