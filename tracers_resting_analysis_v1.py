#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:32:19 2024

@author: erri
"""


# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# FUNCTION
# =============================================================================
def fill_consecutive_zeros(tracers_matrix, N):
    filled_matrix = np.copy(tracers_matrix)  # Create a copy to avoid modifying the original matrix
    
    # Iterate over each row in the matrix
    for i, row in enumerate(filled_matrix):
        start = None  # Placeholder for the start of a sequence of zeros
        count_zeros = 0  # Counter for consecutive zeros
        
        # Iterate over the elements in the row, ignoring trailing NaNs
        for j in range(7, len(row)):  # Start from the eighth column
            if np.isnan(row[j]):
                break  # Stop if we reach NaNs at the end of the row
            
            if row[j] == 1:
                # If we hit a 1, check if we had a short enough sequence of zeros to fill
                if start is not None and count_zeros <= N:
                    row[start:j] = 1  # Fill the identified sequence with 1s
                # Reset the start and zero count
                start = None
                count_zeros = 0
            elif row[j] == 0:
                # If we hit a zero, start counting a sequence if not already started
                if start is None:
                    start = j
                count_zeros += 1
                
    return filled_matrix


def report_consecutive_zeros_lengths(tracers_matrix_filled):
    # Initialize a list to store the lengths of all consecutive zero periods across all rows
    all_zeros_lengths = []
    
    for row in tracers_matrix_filled:
        count_zeros = 0
        
        # Iterate over the row starting from the 8th column
        for j in range(7, len(row)):
            if np.isnan(row[j]):
                break  # Stop if we reach NaNs at the end of the row
            
            if row[j] == 0:
                count_zeros += 1  # Increment count if it's a 0
            elif row[j] == 1:
                if count_zeros > 0:
                    all_zeros_lengths.append(count_zeros)  # Append the count to the list
                count_zeros = 0  # Reset count after encountering a 1
        
        # Append the final count if the row ended with a 0 sequence
        if count_zeros > 0:
            all_zeros_lengths.append(count_zeros)
    
    return np.array(all_zeros_lengths)


def count_consecutive_ones(tracers_matrix_filled):
    # Initialize a list to store the new rows with only first seven columns and consecutive ones counts
    new_matrix = []
    
    for row in tracers_matrix_filled:
        # Get the first seven columns
        new_row = list(row[:7])
        
        # List to collect lengths of each consecutive ones period
        ones_lengths = []
        count_ones = 0
        
        # Iterate over the row starting from the 8th column
        for j in range(7, len(row)):
            if np.isnan(row[j]):
                break  # Stop if we reach NaNs at the end of the row
            
            if row[j] == 1:
                count_ones += 1  # Increment count if it's a 1
            elif row[j] == 0:
                if count_ones > 0:
                    ones_lengths.append(count_ones)  # Append the count to ones_lengths
                count_ones = 0  # Reset count after encountering a 0
        
        # Append the final count if the row ended with a 1 sequence
        if count_ones > 0:
            ones_lengths.append(count_ones)
        
        # Add the lengths of ones periods to the new row
        new_row.extend(ones_lengths)
        
        # Append the new row to the new matrix
        new_matrix.append(new_row)
    
    # Convert the new_matrix list to a numpy array
    max_len = max(len(row) for row in new_matrix)  # Find max row length for padding
    new_matrix_padded = np.array([np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in new_matrix])
    
    return new_matrix_padded

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

for run_name in run_names:
    # =============================================================================
    # TEMPORAL ANALYSIS
    # =============================================================================
    '''
    The aim of this section is analyse the activity period of each tracer in the
    tracers_matrix and fill inactivity period due to errors of the
    tracers_extractio.py script.
    '''
    
    
    tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix.npy'))
    
    tracers_matrix_filled = fill_consecutive_zeros(tracers_matrix, 1)
    
    
    tracer_matrix_rest_periods = count_consecutive_ones(tracers_matrix_filled)
    
    
    zeros_lengths_vector = report_consecutive_zeros_lengths(tracers_matrix_filled)
    
    
    # Define bins from 1 to the maximum value in the vector
    max_value = int(np.max(zeros_lengths_vector))
    bins = np.arange(1, max_value + 2)  # +2 to include max value in the last bin
    
    # Compute the histogram (frequency distribution)
    counts, bin_edges = np.histogram(zeros_lengths_vector, bins=bins)
    
    # Plot the histogram
    plt.figure(figsize=(16, 6))
    plt.hist(zeros_lengths_vector, bins=bin_edges, edgecolor='black', alpha=0.7, align='left')
    plt.xlabel("Length of Consecutive Zeros")
    plt.ylabel("Frequency")
    plt.title("Distribution of Consecutive Zero Lengths")
    
    # Set x-ticks centered with respect to each bar
    plt.xticks(bins[:])  # Center x-ticks by shifting by 0.5
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
