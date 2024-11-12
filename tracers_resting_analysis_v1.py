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


# PLOT MODES
inactivity_periods_analysis = False
activity_periods_analysis   = True

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
        for j in range(6, len(row)):  # Start from the eighth column
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
        for j in range(6, len(row)):
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
        new_row = list(row[:6])
        
        # List to collect lengths of each consecutive ones period
        ones_lengths = []
        count_ones = 0
        
        # Iterate over the row starting from the 6th column
        for j in range(6, len(row)):
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
    new_matrix_padded = np.pad(new_matrix_padded, ((0, 0), (0, 1)), constant_values=np.nan)
    
    return new_matrix_padded

# =============================================================================
# SCRIPT PARAMETERS
# run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
run_names = ['q05_1r3']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
run_names = ['q05_1r3', 'q05_1r3copy']


# FOLDERS SETUP
# =============================================================================
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')
plot_dir = os.path.join(output_dir, 'resting_time')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)


if inactivity_periods_analysis:
    print('\n Inactivity periods analysis.')
    for run_name in run_names:
        
        print(run_name, ' is running...')
        
        print
        # LOAD DATA
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix.npy'))
        
        # =============================================================================
        # INVESTIGATE INACTIVE PERIODS TO CHECK WHETHER THE SCRIPT FAILS TO DETEC TRACERS OR NOT
        # =============================================================================
        
        # Perform temporal analysis - fill inactive periods <= N long
        tracers_matrix_filled = fill_consecutive_zeros(tracers_matrix, 0)
        zeros_lengths_vector = report_consecutive_zeros_lengths(tracers_matrix_filled)
    
        # Define bins and compute frequencies
        max_value = int(np.max(zeros_lengths_vector))
        bins = np.arange(1, max_value + 2)
        counts, _ = np.histogram(zeros_lengths_vector, bins=bins)
        
        # Normalize counts to get probabilities
        counts_normalized = counts / counts.sum()
        
        # Compute cumulative sum of the normalized counts
        cumulative_counts = np.cumsum(counts_normalized)
        
        # Set up the figure
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        # Plot histogram with plt.bar on the left y-axis
        ax1.bar(bins[:-1], counts_normalized, color='skyblue', edgecolor='darkblue', alpha=0.75, width=0.8)
        ax1.set_ylim(0, 0.4)  # Set limits for left y-axis
        ax1.set_xlabel("Length of Consecutive Zeros", fontsize=14)
        ax1.set_ylabel("Probability (Histogram)", fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a secondary y-axis for the cumulative sum
        ax2 = ax1.twinx()
        ax2.plot(bins[:-1], cumulative_counts, color='orange', linestyle='-', marker='o', linewidth=2, label="Cumulative Sum")
        ax2.set_ylim(0, 1)  # Set limits for right y-axis
        ax2.set_ylabel("Cumulative Probability", fontsize=14, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Add titles, labels, and legend
        plt.title(f"Normalized Distribution of Consecutive Zero Lengths - {run_name}", fontsize=16, fontweight='bold')
        
        # Custom x-ticks for better readability
        ax1.set_xticks(np.arange(0, max_value + 5, 5))
        
        # Add grid and background color to the left y-axis
        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        ax1.set_facecolor('whitesmoke')
        
        # Add footer with file info
        plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", 
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='grey')
        
        # Save and display
        plt.savefig(os.path.join(plot_dir, f'normalized_consec_zero_periods_length_{run_name}.pdf'), dpi=800, bbox_inches='tight')
        plt.show()
        
        
    # =============================================================================


# =============================================================================
# INVESTIGATE ACTIVE PERIODS AFTER THE TEMPORAL ANALYSIS
# =============================================================================
if activity_periods_analysis:
    print('\nEffects of temporal filtering process')
    '''
    The aim of this section is first of all to check the effectiveness of the
    data analysis.
    '''
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    for run_name in run_names:
        print(run_name, ' is running...')
        # LOAD DATA
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix.npy'))
     
        N_array     = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160]
        # N_array     = [0, 1]
        ratio_array = []

        
        for n in N_array:
            # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
            tracers_matrix_filled = fill_consecutive_zeros(tracers_matrix, n)
            
            # COUNT CONSECUTIVE ACTIVE PERIODS
            activity_periods = count_consecutive_ones(tracers_matrix_filled)
            
            # TRIM EMPTY ROWS
            activity_periods = activity_periods[~np.isnan(activity_periods).all(axis=1)]
            all_active_position = activity_periods.shape[0]
            
            # TRIM ALL THE TRACERS THAT ARE ACTIVE MORE THAN ONCE
            activity_periods_trim = activity_periods[np.isnan(activity_periods[:, 7])]
            once_active_position = activity_periods_trim.shape[0]
        
            ratio_array.append(once_active_position/all_active_position)

        plt.plot(N_array, ratio_array, marker='o', linestyle='-', label=run_name)
    plt.xlabel('Filling analysis window dimension')
    plt.ylabel('Actived once positions / All positions')
    plt.title('Effects of temporal filtering - Run ' +  run_name[:5])
    plt.grid(True)
    plt.ylim(0.2, 1)
    plt.legend()
    plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=6)
    plt.savefig(os.path.join(plot_dir, 'effect_of_temporal_filtering.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
    
    
