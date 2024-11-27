#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:08:37 2024

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
# SCRIPT PARAMETERS
# =============================================================================
# run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q05_1r1']

# =============================================================================
# FOLDERS SETUP
# =============================================================================
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')

# =============================================================================
# FUNCTIONS
# =============================================================================
def set_first_nan_to_one(matrix):
    # Iterate over each row in the matrix
    for i in range(matrix.shape[0]):
        row = matrix[i]
        
        # Check if the row is not full of NaNs
        if not np.all(np.isnan(row)):
            # Find the index of the first np.nan value
            nan_index = np.where(np.isnan(row))[0]
            if nan_index.size > 0:  # Check if there is at least one np.nan
                first_nan_index = nan_index[0]
                row[first_nan_index] = 1  # Set the first np.nan to 1

    return matrix



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
    # tracers_reduced_raw = tracers_reduced_raw[50:55,:,:]
    
    for t in range(tracers_reduced_raw.shape[0]):
        # Sort rows based on the values in the first column (column index 0) for each time slice
        tracers_reduced_raw[t] = tracers_reduced_raw[t][np.argsort(tracers_reduced_raw[t][:, 0])]

    
    # tracers_reduced_raw = tracers_reduced_raw[24:,:,:] # Reduce the time intervall (test purpose)
    
    # REMOVE TRACERS THAT COME FROM UPSTREAM AND HAVE NAGATIVE X VALUE
    # Iterate through each time step t
    for t in range(tracers_reduced_raw.shape[0]):
        # Find indices where the first element of the slice is negative
        negative_indices = np.where(tracers_reduced_raw[t, :, 0] < 0)[0]
        
        # Set the entire slice tracers_reduced[t,i,:] to np.nan for each index i where the first element is negative
        tracers_reduced_raw[t, negative_indices, :] = np.nan
        
    
    # REMOVE EMPTY MATRICES ---------------------------------------------------
    # Identify which matrices along the t-axis are full of np.nan
    ''' This matrices are present in the early instants of the run where no
    moved tracers are detected and where tracers move, rest and then went out
    of the UV frame. This produces matrices that are full of np.nan even later
    than matrices where tracers are present'''
    
    full_nan_mask = np.all(np.isnan(tracers_reduced_raw), axis=(1, 2))
    
    # Trim those matrices
    tracers_reduced = tracers_reduced_raw[~full_nan_mask]
    
    # =============================================================================
    #
    # =============================================================================
    
    for t in range(tracers_reduced.shape[0]):
        values = tracers_reduced[t, :, 3]  # Extract the slice for the current time step
        values = values[~np.isnan(values) & (values != 0)] # Trim np.nan and 0 values
        
        # Compute histogram using np.histogram
        counts, bin_edges = np.histogram(values, bins=40, range=(-20, 20))
        
        # Plot using plt.bar
        plt.figure()
        plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='blue', edgecolor='black', alpha=0.7, align='edge')
        plt.title('Histogram of values at time t= ' + str(np.round(t*4/60, decimals=2)) + ' mins')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()















    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")