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
delta_x = 30
delta_y = 30

# =============================================================================
# FUNCTIONS
# =============================================================================
# Function to check neighborhood
def is_in_neighborhood(point1, point2, delta_x, delta_y):
    return (abs(point1[0] - point2[0]) <= delta_x) and (abs(point1[1] - point2[1]) <= delta_y)

def find_first_nan_row(matrix):
    for i, row in enumerate(matrix):
        if np.all(np.isnan(row)):  # Check if all elements in the row are NaN
            return i
    # return 'Error: no fully NaN row is found'  # Return -1 if no fully NaN row is found
    raise ValueError("No fully NaN row is found")
    
def insert_one_in_first_nan(arr):
    """
    Inserts 1 in the first np.nan entry available in the given numpy array.
    
    Parameters:
    arr (numpy.ndarray): The input numpy array.
    
    Returns:
    numpy.ndarray: The modified array with 1 inserted in the first np.nan entry.
    """
    # Check if the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Find the first occurrence of np.nan
    nan_index = np.where(np.isnan(arr))[0]
    
    # If there is at least one np.nan, replace the first occurrence with 1
    if nan_index.size > 0:
        arr[nan_index[0]] = 1
    
    return arr

def find_row_with_entries(matrix, first_entry, second_entry):
    """
    Find the index of the row where the first two entries match the specified values.

    Parameters:
    matrix (np.ndarray): The input NumPy matrix.
    first_entry (float): The value of the first entry to match.
    second_entry (float): The value of the second entry to match.

    Returns:
    int: The index of the row where the first two entries match the specified values, or -1 if no match is found.
    """
    # Iterate through each row
    for i, row in enumerate(matrix):
        # Check if the first two entries match the specified values
        if row[0] == first_entry and row[1] == second_entry:
            return i
    raise ValueError("No matching row is found")
    # return 'Error: no matching row is found'  # Return -1 if no matching row is found

def sort_rows_by_first_column(arr):
    """
    Sorts the rows of a given numpy array based on the values of the first column.
    
    Parameters:
    arr (numpy.ndarray): The input numpy array.
    
    Returns:
    numpy.ndarray: The sorted array.
    """
    # Check if the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Check if the array has at least one row and one column
    if arr.ndim < 2 or arr.shape[1] < 1:
        raise ValueError("Input array must have at least one row and one column")
    
    # Sort the array by the first column
    sorted_arr = arr[arr[:, 0].argsort()]
    
    return sorted_arr

def append_out_of_range_points(resting_report, dataset1, dataset2, dx, dy):
    """
    Appends to resting_report, rows from dataset2 that are out of range (in terms of distace)
    if compared with dataset1. In other words this function add new tracers
    that are considere as "new".

    Parameters:
    resting_report (numpy.ndarray): The resting report array.
    dataset1 (numpy.ndarray): The first dataset to compare.
    dataset2 (numpy.ndarray): The second dataset with points to append.
    dx (float): The maximum distance in x direction.
    dy (float): The maximum distance in y direction.

    Returns:
    numpy.ndarray: Updated resting report with out-of-range points appended.
    """
    # Ensure both datasets are NumPy arrays
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)
    
    # Loop through each point in dataset2
    for point2 in dataset2:
        out_of_range = True
        for point1 in dataset1:
            if is_in_neighborhood(point1[:2], point2[:2], dx, dy):
                out_of_range = False
                break
        if out_of_range:
            first_nan_row = find_first_nan_row(resting_report)
            if first_nan_row == -1:
                # Expand resting_report by adding more empty rows (full of NaNs)
                expanded_rows = np.full((10, resting_report.shape[1]), np.nan)
                resting_report = np.vstack([resting_report, expanded_rows])
                first_nan_row = find_first_nan_row(resting_report)
            # Ensure point2 fits the shape of resting_report
            if point2.shape[0] > resting_report.shape[1]:
                point2 = point2[:resting_report.shape[1]]  # Trim if necessary
            else:
                point2 = np.pad(point2, (0, resting_report.shape[1] - point2.shape[0]), constant_values=np.nan)
            resting_report[first_nan_row] = point2
    
    return resting_report

def trim_nan_rows(matrix):
    """
    Removes rows that are fully NaN from the matrix.

    Parameters:
    matrix (numpy.ndarray): The input numpy array.

    Returns:
    numpy.ndarray: The matrix with fully NaN rows removed.
    """
    return matrix[~np.isnan(matrix).all(axis=1)]

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
    
    # tracers_reduced_raw = tracers_reduced_raw[24:,:,:] # Reduce the time intervall (test purpose)
    
    # REMOVE EMPTY MATRICES ---------------------------------------------------
    # Identify which matrices along the t-axis are full of np.nan
    ''' This matrices are present in the early instants of the run where no
    moved tracers are detected and where tracers move, rest and then went out
    of the UV frame. This produces matrices that are full og np.nan even later
    than matrices where tracers are present'''
    
    full_nan_mask = np.all(np.isnan(tracers_reduced_raw), axis=(1, 2))
    
    # Trim those matrices
    tracers_reduced = tracers_reduced_raw[~full_nan_mask]
    
    # =========================================================================
    # INITIALIZE THE ARRAYS
    # =========================================================================
    
    # CREATE THE RESTING REPORT MATRIX ----------------------------------------
    resting_report = np.zeros((2000,100))*np.nan

    # INITIALIZE RESTING REPORT WITH THE POINTS IN THE FIRST ITERATION---------
    # Extract all the points at the first iteration trimming empty rows
    first_iter = tracers_reduced[0,:,:]
    first_iter = first_iter[:find_first_nan_row(first_iter),:]
    
    # FILL RESTING_REPORT WITH THE DATA AT THE FIRST INTERACTION---------------
    resting_report[:first_iter.shape[0],:4] = first_iter
    
    # =========================================================================
    # LOOP OVER TIME
    # =========================================================================
    
    for i in range(tracers_reduced.shape[0] - 1):
        
        # DEFINE TWO CONSECUTIVE DATASETS
        dataset1 = tracers_reduced[i,:find_first_nan_row(tracers_reduced[0,:,:]),:]
        dataset2 = tracers_reduced[i+1,:find_first_nan_row(tracers_reduced[1,:,:]),:]
        
        # SORT DATASET ENTRY BY THE X-COORDINATE
        dataset1 = sort_rows_by_first_column(dataset1)
        dataset2 = sort_rows_by_first_column(dataset2)
        
        # CHECK POINT-BY-POINT DISTANCE ---------------------------------------
        #TODO check if dataset1 is empty, if so skip 2 iteration
        
        for point1 in dataset1[:,:2]:
            for point2 in dataset2[:,:2]:
                
                if is_in_neighborhood(point1, point2, delta_x, delta_y):
  
                    # Find the row index in resting_report
                    row_index = find_row_with_entries(resting_report, point1[0], point1[1])
                    # print('row_index: ', row_index)
                    
                    # Update the position
                    resting_report[row_index,:2] = point2
                    
                    # Generate a unique key for the point to track its last insertion row
                    point_key = (point1[0], point1[1])
                    # Rounding each number in the tuple to 4 decimal places
                    point_key = tuple(round(num, 4) for num in point_key)
                    
                    # Check if the point_key exists in the dictionary, if so, increment the index
                    if point_key in last_inserted_index:
                        print('Point in the dict')
                        row_index += last_inserted_index[point_key] #TODO Check this. Should it be row_index += 1?
                        last_inserted_index[point_key] += 1
                        
                        resting_report[row_index,:] = insert_one_in_first_nan(resting_report[row_index,:])
                    
                    else:
                        index = find_first_nan_row(resting_report)
                        resting_report[index,:] = insert_one_in_first_nan(resting_report[row_index,:])
                    
                    # print('row_index after dict: ', row_index)
                    
                    # Insert 1 at the appropriate index in the resting report
                    # resting_report[row_index,:] = insert_one_in_first_nan(resting_report[row_index,:])
                    
                    # print(resting_report[row_index,:])
                    break
    
        # Append out-of-range points from dataset2 to resting_report
        resting_report = append_out_of_range_points(resting_report, dataset1, dataset2, delta_x, delta_y)
        
    # Trim fully NaN rows from resting_report
    resting_report = trim_nan_rows(resting_report)
    
    # =========================================================================
    # COLLAPSE RESTING TIME 
    # =========================================================================
    
    # Process each row in the matrix
    # Extract the first four columns
    first_four_columns = resting_report[:, :4]
    
    # Calculate the sum of ones in each row, ignoring np.nan
    sum_of_ones = np.nansum(resting_report[:, 4:] == 1, axis=1, keepdims=True)
    
    # Concatenate the first four columns with the sum of ones
    resting_report_collapsed = np.concatenate((first_four_columns, sum_of_ones), axis=1)



    #%% RESTING TIME ANALYSIS
    # =============================================================================
    # 
    # =============================================================================
        
