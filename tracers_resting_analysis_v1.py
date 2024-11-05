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
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r1']
# FOLDERS SETUP
# =============================================================================
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')
plot_dir = os.path.join(output_dir, 'resting_time')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



for run_name in run_names:
    # Load the tracer matrix and process it
    tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix_buffer3.npy'))
    tracers_matrix_filled = fill_consecutive_zeros(tracers_matrix, 0)
    zeros_lengths_vector = report_consecutive_zeros_lengths(tracers_matrix_filled)

    # Define bins and compute frequencies
    max_value = int(np.max(zeros_lengths_vector))
    bins = np.arange(1, max_value + 2)  # Bins from 1 to max_value inclusive
    counts, _ = np.histogram(zeros_lengths_vector, bins=bins)
    
    # Normalize the counts to get probabilities
    counts_normalized = counts / counts.sum()  # Sum of all bars will be 1

    # Set up the figure
    plt.figure(figsize=(20, 8))
    
    # Plot with plt.bar using normalized counts
    plt.bar(bins[:-1], counts_normalized, color='skyblue', edgecolor='darkblue', alpha=0.75, width=0.8)

    # Add titles and labels
    plt.title(f"Normalized Distribution of Consecutive Zero Lengths - {run_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Length of Consecutive Zeros", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    # Set y-axis limit
    plt.ylim(0, 0.3)

    # Set custom x-ticks for better readability
    plt.xticks(np.arange(0, max_value + 5, 5), fontsize=12)

    # Improve layout with gridlines and add a light grey background
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.gca().set_facecolor('whitesmoke')
    
    # Add a footer with file info
    plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", 
             transform=plt.gca().transAxes, ha='center', va='center', fontsize=12, color='grey')
    
    # Save and display
    plt.savefig(os.path.join(plot_dir, f'normalized_consec_zero_periods_length_{run_name}.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
    
    

    test = report_consecutive_zeros_lengths(tracers_matrix_filled[3:4,:]) 