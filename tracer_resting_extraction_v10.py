#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:08:37 2024

@author: erri
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SCRIPT PARAMETERS
# =============================================================================
run_names = ['q05_1r1']

# =============================================================================
# FOLDERS SETUP
# =============================================================================
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')

# NEIGHBOURHOOD ANALYSIS PARAMETERS
# delta_x = 10
# delta_y = 10

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
    

    # Create a report with the following structure:
    '''
    t0 x-coordiante, t0 y-coordinate, t0 elevation, t0 delta eta, tn x-coordinate, tn y-coordinate, tn elevation, tn delta eta, 1, 1, 1, 1, 0, 0, 0, np.nan, np.nan
    Where:
        t0 is the information the first time a certain tracer is found
        tn is the last available information
        1 means that in that position the pixel is active
        0 means that in that position the pixel is inactive.
    '''
    
    
    tracers_matrix = np.hstack((tracers_reduced[0,:,:], np.full((tracers_reduced.shape[1], tracers_reduced.shape[0]+2), np.nan)))
    tracers_matrix[:,4:6] = tracers_matrix[:,:2]
    tracers_matrix = set_first_nan_to_one(tracers_matrix)
    
    dx, dy = 5, 5
    first_iteration = True

    for t in range(1, tracers_reduced.shape[0]):
        dataset = tracers_reduced[t,:,:]
        
        # Define the helper function to check if a point is within dx, dy range
        def is_within_range(coord, coord_list, dx, dy):
            return np.any((np.abs(coord_list - coord) <= [dx, dy]).all(axis=1))
        
        # Assuming matrix1 and matrix2 are your input matrices, and dx, dy are the specified ranges
        coords1 = tracers_matrix[:, :2]  # Extract coordinates (x, y) from the first matrix
        coords2 = dataset[:, :2]  # Extract coordinates (x, y) from the second matrix
        
        # Initialize empty lists to store indices or points for each category
        present_in_first_only = []
        present_in_both = []
        present_in_second_only = []
        
        # Loop through each point in matrix1 and check if it's in matrix2
        for i, point1 in enumerate(coords1):
            if is_within_range(point1, coords2, dx, dy):     # PRESENCE
                # present_in_both.append(i)
                
                # Check each row in tracers_matrix to find a matching point
                for tracer_row in tracers_matrix:
                    # Select comparison coordinates based on iteration
                    tracer_x, tracer_y = (tracer_row[0], tracer_row[1]) if first_iteration else (tracer_row[4], tracer_row[5])
                    
                    # If the tracer point matches the analyzed point within dx, dy range
                    if abs(tracer_x - point1[0]) <= dx and abs(tracer_y - point1[1]) <= dy:
                        # Find the first np.nan in the active times column and replace it with 1
                        active_times = tracer_row[6:]  # Assuming active_times are in the last columns
                        nan_index = np.where(np.isnan(active_times))[0]
                        if nan_index.size > 0:  # Check if there's at least one np.nan
                            active_times[nan_index[0]] = 1  # Set the first np.nan to 1
        
                        # Update x_L and y_L in tracers_matrix with the current coordinates of point1
                        tracer_row[4], tracer_row[5] = point1[0], point1[1]
                        break  # Stop searching once the matching tracer is updated
                
                
                
                
            else:                                            # DISAPPEARENCE
                # present_in_first_only.append(i)
                
                # Check each row in tracers_matrix to find a matching point for (x, y)
                for tracer_row in tracers_matrix:
                    tracer_x, tracer_y = tracer_row[0], tracer_row[1]  # Use x and y for range checking
        
                    # If the tracer point matches the analyzed point within dx, dy range
                    if abs(tracer_x - point1[0]) <= dx and abs(tracer_y - point1[1]) <= dy:
                        # Find the first np.nan in the active times column and replace it with 0
                        active_times = tracer_row[6:]  # Assuming active_times are in the last columns
                        nan_index = np.where(np.isnan(active_times))[0]
                        if nan_index.size > 0:  # Check if there's at least one np.nan
                            active_times[nan_index[0]] = 0  # Set the first np.nan to 0
                        break  # Stop searching once the matching tracer is updated
                
        
        
        
        # Loop through each point in matrix2 and check if it's not in matrix1
        for j, point2 in enumerate(coords2):
            if not is_within_range(point2, coords1, dx, dy): # APPEARENCE
                # present_in_second_only.append(j)
                # Find the first empty row in tracers_matrix (all elements are np.nan)
                empty_row_index = np.where(np.all(np.isnan(tracers_matrix), axis=1))[0]
                
                if np.isnan(point2).any():
                    break
                elif empty_row_index.size > 0:  # Check if there is an empty row available
                    # empty_row_index[0] is the row index of the first full np.nan aavailable row
                
                    ## TEST
                    tracers_matrix[empty_row_index[0],:4] = dataset[j, :4]
                    tracers_matrix[empty_row_index[0],4] = point2[0]
                    tracers_matrix[empty_row_index[0],5] = point2[1]
                    tracers_matrix[empty_row_index[0],6] = 1
                    ## TEST
                
                
                    # empty_row = tracers_matrix[empty_row_index[0]]  # Select the first empty row
        
                    # # Copy the first 4 columns from matrix2's point to the empty row in tracers_matrix
                    # empty_row[:4] = dataset[j, :4]
        
                    # # Set x_L and y_L in tracers_matrix to the current coordinates of point2
                    # if not np.isnan(coords2.any()):
                    #     empty_row[4], empty_row[5], empty_row[6] = point2[0], point2[1], 1 #TODO check this



        first_iteration = True

if not os.path.exists(os.path.join(output_dir, 'tracer_matrix')):
        os.makedirs(os.path.join(output_dir, 'tracer_matrix'))
        
np.save(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix.npy'), tracers_matrix)

