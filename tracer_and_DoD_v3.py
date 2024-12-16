#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:08:37 2024

@author: erri
"""

# IMPORT LIBRARIES
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

start_time = time.time()


# SCRIPT PARAMETERS
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q07_1r10']

# FOLDERS SETUP
w_dir = '/Volumes/T7_Shield/PhD/repos/tracers_displacement/'
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')

# =============================================================================
# SET FOLDERS
# =============================================================================

# Dictionary to track the last row index where 1 was inserted for each point
last_inserted_index = {}


for run_name in run_names:
    start_time = time.time()
    print(run_name, '...is running.')
    # =============================================================================
    # IMPORT FILES
    # =============================================================================
    tracers_reduced_path = os.path.join(output_dir, 'output_images', run_name, 'tracers_reduced_'+run_name+'.npy')
    tracers_reduced_raw = np.load(tracers_reduced_path)
    
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run_name)
    all_tracers_path = os.path.join(path_out,'alltracers_'+ run_name +'.npy')
    all_tracers = np.load(all_tracers_path)
    
    
    for t in range(tracers_reduced_raw.shape[0]):
        # Sort rows based on the values in the first column (column index 0) for each time slice
        tracers_reduced_raw[t] = tracers_reduced_raw[t][np.argsort(tracers_reduced_raw[t][:, 0])]

    # REMOVE TRACERS THAT COME FROM UPSTREAM AND HAVE NEGATIVE X VALUE
    for t in range(tracers_reduced_raw.shape[0]):
        negative_indices = np.where(tracers_reduced_raw[t, :, 0] < 0)[0]
        tracers_reduced_raw[t, negative_indices, :] = np.nan

    # REMOVE EMPTY MATRICES
    full_nan_mask = np.all(np.isnan(tracers_reduced_raw), axis=(1, 2))
    tracers_reduced = tracers_reduced_raw[~full_nan_mask]

    # =============================================================================
    # HISTOGRAM PLOTTING
    # =============================================================================
    for t in range(tracers_reduced.shape[0]):
        values = tracers_reduced[t, :, 3]  # Extract the slice for the current time step
        values = values[~np.isnan(values)]  # Trim np.nan values

        # Separate values into negative, zero, and positive categories
        negative_values = values[values < 0]
        zero_values = values[values == 0]
        positive_values = values[values > 0]

        # Compute histogram for each category
        counts, bin_edges = np.histogram(values, bins=41, range=(-20, 20))
        mid_bin_edges = bin_edges[:-1] + np.diff(bin_edges) / 2

        total_count = np.sum(counts)
        normalized_counts = counts / total_count if total_count > 0 else counts

        # Masks for bin categories
        negative_mask = mid_bin_edges < 0
        positive_mask = mid_bin_edges > 0
        zero_mask = (mid_bin_edges >= 0) & (mid_bin_edges <= 0)

        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.bar(mid_bin_edges[negative_mask], normalized_counts[negative_mask],
                width=np.diff(bin_edges)[0], color='red', edgecolor='black', alpha=0.7, align='center', label='Scour')
        plt.bar(mid_bin_edges[positive_mask], normalized_counts[positive_mask],
                width=np.diff(bin_edges)[0], color='blue', edgecolor='black', alpha=0.7, align='center', label='Fill')
        plt.bar(mid_bin_edges[zero_mask], normalized_counts[zero_mask],
                width=np.diff(bin_edges)[0], color='gray', edgecolor='black', alpha=0.7, align='center', label='Not-detected')

        # Set y-axis limit
        plt.ylim(0, max(normalized_counts) + 0.05)

        # Add legend and adjust its position
        plt.legend(loc='upper left', fontsize=12)

        # Titles and labels
        plt.title(f'Normalized Histogram of Values at Time t = {np.round(t * 4 / 60, decimals=2)} mins  - ' + run_name)
        plt.xlabel("Value")
        plt.ylabel("Normalized Frequency")
        plt.grid(True)
        plt.text(0.5, -0.15, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.show()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    
    
#%%
# =============================================================================
# SCOUR, NOT DETECTED, AND FILL POSITION OF TRACERS OVER TIME
# =============================================================================
# IMPORT LIBRARIES
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


start_time = time.time()


# SCRIPT PARAMETERS
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
              'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
              'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q07_1r11']



# SURVEY PIXEL DIMENSION
px_x, px_y = 50, 5 # [mm]

# FOLDERS SETUP
w_dir = '/Volumes/T7_Shield/PhD/repos/tracers_displacement/'
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
            pure_filt_check=[]
            weak_dep_check=[]
            # Check if the row is not full of np.nan
            if not np.all(np.isnan(tracers_reduced[i, j, :4])):  # Only check the original 4 columns              
                DoD_value = DoD[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                
                # CHECK PRE
                if not DoD_name_tspan1_pre == 'False':

                    DoD_tspan2_1_value_pre = DoD_tspan2_1[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    # DoD_tspan1_pre = np.loadtxt(os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan1_pre+'_filt_ult.txt'))
                    DoD_tspan1_value_pre = DoD_tspan1_pre[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    
                    print((DoD_value>0)*1)
                    print((DoD_tspan1_value_pre>0)*1)
                    print((DoD_tspan2_1_value_pre>0)*1)
                    
                    tuple_value = ((DoD_value>0)*1,(DoD_tspan1_value_pre>0)*1)
                    value = (DoD_tspan2_1_value_pre>0)*1
                    
                    if tuple_value == (0, 0) and value == 1: # Check if the tuple is (0, 0) and the single value is 1
                        pure_filt_check.append(True)
                    
                    if tuple_value in [(0, 1), (1, 0)] and value == 1: # Check if the tuple is (0, 1) or (1, 0) and the single value is 1
                        weak_dep_check.append(True)
                        
                # CHECK POST
                if not DoD_name_tspan1_post == 'False':

                    DoD_tspan2_2_value_post = DoD_tspan2_2[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    # DoD_tspan1_post = np.loadtxt(os.path.join(path_in_DoD, 'DoD_' + DoD_name_tspan1_post+'_filt_ult.txt'))
                    DoD_tspan1_value_post = DoD_tspan1_post[(tracers_reduced[i,j,1]/(px_y)).astype(int),(tracers_reduced[i,j,0]/(px_x)).astype(int)]
                    
                    
                    tuple_value = ((DoD_value>0)*1,(DoD_tspan1_value_post>0)*1)
                    value = (DoD_tspan2_2_value_post>0)*1
                    
                    if tuple_value == (0, 0) and value == 1: # Check if the tuple is (0, 0) and the single value is 1
                        pure_filt_check.append(True)
                    
                    if tuple_value in [(0, 1), (1, 0)] and value == 1: # Check if the tuple is (0, 1) or (1, 0) and the single value is 1
                        weak_dep_check.append(True)
                
                        
            if any(pure_filt_check):
                tracers_reduced[i,j,4]==1
                
            if any(weak_dep_check):
                tracers_reduced[i,j,5]==1
                
                
                
                
                
                
                
                
    # =============================================================================
    # COUNT POSITIVE, NEGATIVE, AND ZERO VALUES OVER TIME
    # =============================================================================
    for t in range(tracers_reduced.shape[0]-1):
        values = tracers_reduced[t, :, 3]  # Extract the slice for the current time step
        values = values[~np.isnan(values)]  # Trim np.nan values

        # Count positive, negative, and zero values
        positive_count = np.sum(values > 0)
        negative_count = np.sum(values < 0)
        zero_count = np.sum(values == 0)

        positive_counts_over_time.append(positive_count)
        negative_counts_over_time.append(negative_count)
        zero_counts_over_time.append(zero_count)

    # =============================================================================
    # PLOT THE TRENDS
    # =============================================================================
    time_steps = np.arange(len(positive_counts_over_time)) * 4 / 60  # Convert time steps to minutes
    
    plt.figure(figsize=(14, 7))
    plt.plot(time_steps, positive_counts_over_time, label='Fill ', color='blue', linewidth=2)
    plt.plot(time_steps, zero_counts_over_time, label='Not-detected', color='gray', linestyle='-', linewidth=2)
    plt.plot(time_steps, negative_counts_over_time, label='Scour', color='red', linewidth=2)
    
    # Add labels, title, and legend
    plt.title('Trend of fill, not-detected, and scour tracer position over time\n' + run_name, fontsize=16)
    plt.xlabel('Time (minutes)', fontsize=14)
    plt.ylabel('Number of Values', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save and show the plot
    plt.tight_layout()
    plt.text(0.5, -0.15, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.savefig(os.path.join(output_dir, run_name + '_fill_not_detected_scour_temporal_trends.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

#%%
# =============================================================================
# SINGLE CHART
# =============================================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
             'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
             'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# run_names = ['q07_1r10']

w_dir = '/Volumes/T7_Shield/PhD/repos/tracers_displacement/'
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data', 'tracers_topo_overlapping')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# IMPORT SCRIPT PARAMETERS
run_parameters = pd.read_csv(os.path.join(w_dir, 'input_data', 'tracers_DEM_DoD_combination_v2.csv'))

for run_name in run_names:
    
    # Find the row where RUN equals run_name
    selected_row = run_parameters.loc[run_parameters['RUN'] == run_name]
    
    if not selected_row.empty:
        # Extract variables for each column in the selected row
        
        DEM_name = int(selected_row['DEM'].values[0])
        
        DoD_name = selected_row['DoD timespan1'].values[0]
        # DoD_name = selected_row['DoD timespan2'].values[0]
        # DoD_name = selected_row['DoD timespan2.1'].values[0]
        
        x_coord_offset = selected_row['x_coord_offset'].values[0]
        
        feed_x = selected_row['feed-x'].values[0]
    
    # TRACERS PATH
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run_name)
    all_tracers_path = os.path.join(path_out, 'alltracers_' + run_name + '.npy')
    all_tracers = np.load(all_tracers_path)
    

    # DEFINE A SPECIFIC TIME
    all_tracers = all_tracers[200, :, :]
    
    
    # TOPOGRAPHIC DATA PATH
    path_in_DEM = os.path.join(input_dir, 'surveys', run_name[0:5])
    path_in_DoD = os.path.join(input_dir, 'DoDs', 'DoD_' + run_name[0:5])
    DEM_path = os.path.join(path_in_DEM, 'matrix_bed_norm_' + run_name[0:3] + '_1s' + str(DEM_name) + '.txt')
    DEM = np.loadtxt(DEM_path, skiprows=8)
    if 'q10' in run_name:
        DEM=DEM[:,:236]
    DEM = np.where(DEM == -999, np.nan, DEM)
    DoD_path = os.path.join(path_in_DoD, 'DoD_'+DoD_name+'_filt_ult.txt')
    DoD = np.loadtxt(DoD_path)
    
    # SCALE DEM AND DoD
    DEM = np.repeat(DEM, 10, axis=1)
    DoD = np.repeat(DoD, 10, axis=1)
    
    # EXTRACT X AND Y COORDINATES
    x_coord = all_tracers[:, 1]  # First column
    y_coord = all_tracers[:, 2]  # Second column
    
    # DEFINE AXIS LIMITS
    x_min, x_max = 0, DoD.shape[1] * 5  # Convert pixel count to mm
    y_min, y_max = 0, DoD.shape[0] * 5  # Convert pixel count to mm

    # Define vmin, vmax, and colormap
    DoD_vmin, DoD_vmax = -10, 10  # Adjust these values as needed
    DoD_colormap = plt.colormaps['RdBu']
    DEM_vmin, DEM_vmax = -10, 10  # Adjust these values as needed
    DEM_colormap = plt.colormaps['BrBG_r']

    # PLOT --------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))  # One row, two columns

    plt.suptitle(f"Run: {run_name}", fontsize=16, fontweight='bold')

    x_coord_low_lim = feed_x -100
    x_coord_upp_lim = np.nanmax(x_coord)+600
    if abs(x_coord_upp_lim-x_coord_low_lim)<2200:
        x_coord_upp_lim = x_coord_low_lim + 2200
    
    # PLOT TRACERS OVER DEM
    ax1 = axes[0]
    im1 = ax1.imshow(DEM, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=DEM_colormap, alpha=0.8, vmin=DEM_vmin, vmax=DEM_vmax)
    ax1.plot(x_coord, y_coord, 'o', color='magenta', markersize=1, label='Tracers')
    ax1.axvline(x=feed_x, color='red', linestyle='--', linewidth=1.5, label=f'Feed x-coord = {feed_x}')  # Add vertical line
    ax1.set_title('Tracers over DEM' + str(DEM_name), fontsize=14)
    ax1.set_xlabel('X Coordinate (mm)', fontsize=10)
    ax1.set_ylabel('Y Coordinate (mm)', fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2, aspect=4, shrink=0.2)
    cbar1.set_label('Elevation (mm)', fontsize=10)
    ax1.grid(True)
    ax1.axis('scaled')
    # ax1.set_xlim(x_coord_low_lim, x_coord_upp_lim)
    ax1.set_ylim(0, 800)

    # PLOT TRACERS OVER DoD
    ax2 = axes[1]
    im2 = ax2.imshow(DoD, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=DoD_colormap, alpha=0.8, vmin=DoD_vmin, vmax=DoD_vmax)
    ax2.plot(x_coord, y_coord, 'o', color='lime', markersize=1, label='Tracers')
    ax2.axvline(x=feed_x, color='red', linestyle='--', linewidth=1.5, label=f'Feed x-coord = {feed_x}')  # Add vertical line
    ax2.set_title('Tracers over DoD'+DoD_name, fontsize=14)
    ax2.set_xlabel('X Coordinate (mm)', fontsize=10)
    ax2.set_ylabel('Y Coordinate (mm)', fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2, aspect=4, shrink=0.2)
    cbar2.set_label('Elevation Change (mm)', fontsize=10)
    ax2.grid(True)
    ax2.axis('scaled')
    # ax2.set_xlim(x_coord_low_lim, x_coord_upp_lim)
    ax2.set_ylim(0, 800)
    
    plt.text(0.5, -1, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, run_name + '_tracers_DEM_DoD_split.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


#%%
# =============================================================================
# VIDEO TIMELAPSE
# =============================================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2  # Required for video creation with OpenCV

# Initialize directories
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')

output_dir = os.path.join(w_dir, 'output_data', 'tracers_topo_overlapping')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


run_names = ['q05_1r1']

for run_name in run_names:
    temp_image_dir = "temp_frames"
    os.makedirs(temp_image_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{run_name}_animation.mp4")
    
    # TRACERS PATH
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run_name)
    all_tracers_path = os.path.join(path_out, 'alltracers_' + run_name + '.npy')
    all_tracers = np.load(all_tracers_path)
    
    # REDUCE THE DATASET FOR TESTING
    # all_tracers = all_tracers[100:120,:,:]

    # TOPOGRAPHIC DATA PATH
    path_in_DEM = os.path.join(input_dir, 'surveys', run_name[0:5])
    path_in_DoD = os.path.join(input_dir, 'DoDs', 'DoD_' + run_name[0:5])
    DEM = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_' + run_name[0:3] + '_1s0.txt'), skiprows=8)
    DEM = np.where(DEM == -999, np.nan, DEM)
    DoD = np.loadtxt(os.path.join(path_in_DoD, 'DoD_1-0_filt_ult.txt'))

    # Scale DEM and DoD
    DEM = np.repeat(DEM, 10, axis=1)
    DoD = np.repeat(DoD, 10, axis=1)

    # Loop through time steps
    for t in range(all_tracers.shape[0]):
        tracers = all_tracers[t, :, :]

        # Extract x and y coordinates
        x_coord = tracers[:, 1]  # First column
        y_coord = tracers[:, 2]  # Second column

        # Define axis limits
        x_min, x_max = 0, DoD.shape[1] * 5  # Convert pixel count to mm
        y_min, y_max = 0, DoD.shape[0] * 5  # Convert pixel count to mm

        # Define vmin, vmax, and colormap
        DoD_vmin, DoD_vmax = -10, 10
        DoD_colormap = plt.colormaps['RdBu_r']
        DEM_vmin, DEM_vmax = -20, 20
        DEM_colormap = plt.colormaps['BrBG_r']

        # Create a figure with two columns
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        # Plot tracers over DEM
        ax1 = axes[0]
        im1 = ax1.imshow(DEM, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=DEM_colormap, alpha=0.8, vmin=DEM_vmin, vmax=DEM_vmax)
        ax1.plot(x_coord, y_coord, 'o', color='magenta', markersize=1, label='Tracers')
        ax1.set_title(f'Tracers over DEM (Time {t*4/60})', fontsize=14)
        cbar1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.2, aspect=4, shrink=0.2)
        cbar1.set_label('Elevation (mm)', fontsize=10)
        ax1.axis('scaled')
        ax1.set_xlim(np.nanmin(all_tracers[:,:, 1])*0.9, np.nanmax(all_tracers[:,:, 1])*1.1)
        ax1.set_ylim(0, 800)
        ax1.grid(True)

        # Plot tracers over DoD
        ax2 = axes[1]
        im2 = ax2.imshow(DoD, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=DoD_colormap, alpha=0.8, vmin=DoD_vmin, vmax=DoD_vmax)
        ax2.plot(x_coord, y_coord, 'o', color='lime', markersize=1, label='Tracers')
        ax2.set_title(f'Tracers over DoD (Time {t})', fontsize=14)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.2, aspect=4, shrink=0.2)
        cbar2.set_label('Elevation Change (mm)', fontsize=10)
        ax2.axis('scaled')
        ax2.set_xlim(np.nanmin(all_tracers[:,:, 1])*0.9, np.nanmax(all_tracers[:,:, 1])*1.1)
        ax2.set_ylim(0, 800)
        ax2.grid(True)

        # Save frame
        frame_path = os.path.join(temp_image_dir, f"frame_{t:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=300)
        plt.close(fig)

# Create a video from the frames
frame_files = sorted([os.path.join(temp_image_dir, f) for f in os.listdir(temp_image_dir) if f.endswith('.png') and f.startswith('f')])
frame = cv2.imread(frame_files[0])
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))  # 10 fps

for frame_file in frame_files:
    video_writer.write(cv2.imread(frame_file))

video_writer.release()
print(f"Video saved at {output_video_path}")

# Clean up temporary images (optional)
for file in frame_files:
    os.remove(file)
os.rmdir(temp_image_dir)
