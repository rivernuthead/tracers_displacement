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
inactivity_periods_analysis = True
inactivity_periods_analysis_single_charts = True
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
        for j in range(0, len(row)):  # Start from the eighth column
            if np.isnan(row[j]) and start is None:
                break  # Stop if we reach NaNs at the end of the row
            
            if row[j] == 1:
                # If we hit a 1, check if we had a short enough sequence of zeros to fill
                if start is not None and count_zeros <= N:
                    row[start:j] = 1  # Fill the identified sequence with 1s
                # Reset the start and zero count
                start = None
                count_zeros = 0
            if np.isnan(row[j]):
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



def calculate_consecutive_periods(arr, t, y):
    """
    Calculate consecutive periods of 1s, 0s, and -1s in a 3D array and store 
    the lengths in separate stacks.

    Parameters:
    arr (ndarray): 3D array of shape (t, y, x) containing the time series.
    t (int): Number of time steps.
    y (int): Height dimension.

    Returns:
    tuple: Three 3D arrays storing lengths of consecutive 1s, 0s, and -1s.
    """
    ones_stack = np.zeros((y,t), dtype=int)
    zeros_stack = np.zeros((y,t), dtype=int)
    neg_ones_stack = np.zeros((y,t), dtype=int)

    for i in range(y):
        series = arr[i,:]
        count = 1
        idx = 0
        for k in range(1, t):
            if series[k] == series[k - 1]:
                count += 1
            else:
                if series[k - 1] == 1:
                    ones_stack[i, idx] = count
                elif series[k - 1] == 0:
                    zeros_stack[i, idx] = count
                elif series[k - 1] == -1:
                    neg_ones_stack[i, idx] = count
                count = 1
                idx += 1

        # Assign the last counted period
        if series[-1] == 1:
            ones_stack[i, idx] = count
        elif series[-1] == 0:
            zeros_stack[i, idx] = count
        elif series[-1] == -1:
            neg_ones_stack[i, idx] = count
            
    
    ones_stack = ones_stack[:, ::2]
    zeros_stack = zeros_stack[:, 1::2]
    # neg_ones_stack = neg_ones_stack[???]
    
    ones_stack = np.where(ones_stack==0, np.nan, ones_stack)
    zeros_stack = np.where(zeros_stack==0, np.nan, zeros_stack)
    neg_ones_stack = np.where(neg_ones_stack==0, np.nan, neg_ones_stack)
    
    return ones_stack, zeros_stack, neg_ones_stack



# =============================================================================
# SCRIPT PARAMETERS
# run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r3']
run_names = ['q05_1r3copy']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']

# PROXIMITY ANALYSIS
dx, dy = 3, 3

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
    
    # for n in [0, 1, 2, 5, 10, 15, 20]:
    for n in [0]:
        inactivity_periods = []
        activity_periods = []
        
        for run_name in run_names:
            
            print(run_name, ' is running...')
            

            # LOAD DATA
            tracer_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
            
            # tracer_matrix = np.delete(tracer_matrix, 6, axis=1)
            # tracer_matrix[:,6] = np.where(tracer_matrix[:,6]==0, 1,tracer_matrix[:,6])
            
            tracer_positions = tracer_matrix[:,:6]
            
            tracer_periods = tracer_matrix[:,6:]
            
            
            # =============================================================================
            # INVESTIGATE INACTIVE PERIODS TO CHECK WHETHER THE SCRIPT FAILS TO DETEC TRACERS OR NOT
            # =============================================================================
            
            # Perform temporal analysis - fill inactive periods <= N long
            tracers_matrix_filled = fill_consecutive_zeros(tracer_periods, n)
            
            ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, tracers_matrix_filled.shape[1], tracers_matrix_filled.shape[0])
            
            ones_stack_matrix = np.hstack((tracer_positions,ones_stack))
            zeros_stack_matrix = np.hstack((tracer_positions,zeros_stack))
            
            
            
            zeros_lengths_vector = np.copy(zeros_stack)
            inactivity_periods.append(zeros_stack.flatten())
            
            ones_lengths_vector = np.copy(ones_stack)
            activity_periods.append(ones_stack.flatten())
            
            if inactivity_periods_analysis_single_charts:
                # Define bins and compute frequencies
                max_value = int(np.nanmax(zeros_lengths_vector))
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
                plt.title(f"Normalized Distribution of Inactivity Periods Lengths - {run_name}\nFill length: "+str(n)+' - Proximity analysis: '+str(dx)+'x'+str(dy), fontsize=16, fontweight='bold')
                
                # Custom x-ticks for better readability
                ax1.set_xticks(np.arange(0, max_value + 5, 5))
                
                # Add grid and background color to the left y-axis
                ax1.grid(axis='y', linestyle='--', alpha=0.6)
                ax1.set_facecolor('whitesmoke')
                
                # Add footer with file info
                plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", 
                         transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='grey')
                
                # Save and display
                plt.savefig(os.path.join(plot_dir, f'normalized_consec_zero_periods_length_{run_name}_FillLength'+str(n)+ '.pdf'), dpi=800, bbox_inches='tight')
                plt.show()
            
            
            
        # =============================================================================
        # INACTIVITY PERIODS LENGTH ANALYSIS
        # =============================================================================
        # Define bins and compute frequencies
        inactivity_periods = np.concatenate(inactivity_periods)
        max_value = int(np.nanmax(inactivity_periods))
        bins = np.arange(1, max_value + 2)
        counts, _ = np.histogram(inactivity_periods, bins=bins)
        
        # Normalize counts to get probabilities
        counts_normalized = counts / counts.sum()
        
        # Compute cumulative sum of the normalized counts
        cumulative_counts = np.cumsum(counts_normalized)
        
        # Set up the figure
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        # Plot histogram with plt.bar on the left y-axis
        ax1.bar(bins[:-1], counts_normalized, color='skyblue', edgecolor='darkblue', alpha=0.75, width=0.8)
        ax1.set_ylim(0, 0.4)  # Set limits for left y-axis
        ax1.set_xlabel("Length of Inactivity periods", fontsize=14)
        ax1.set_ylabel("Probability (Histogram)", fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a secondary y-axis for the cumulative sum
        ax2 = ax1.twinx()
        ax2.plot(bins[:-1], cumulative_counts, color='orange', linestyle='-', marker='o', linewidth=2, label="Cumulative Sum")
        ax2.set_ylim(0, 1)  # Set limits for right y-axis
        ax2.set_ylabel("Cumulative Probability", fontsize=14, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Add titles, labels, and legend
        plt.title("Normalized Distribution of Inactivity Periods Lengths for all the runs\nFill length: "+str(n)+' - Proximity analysis: '+str(dx)+'x'+str(dy), fontsize=16, fontweight='bold')
        
        # Custom x-ticks for better readability
        ax1.set_xticks(np.arange(0, max_value + 5, 5))
        
        # Add grid and background color to the left y-axis
        ax2.grid(axis='y', linestyle='--', alpha=1)
        ax2.set_facecolor('whitesmoke')
        
        # Add footer with file info
        plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", 
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='grey')
        
        # Save and display
        plt.savefig(os.path.join(plot_dir, 'normalized_inactivity_periods_length.pdf'), dpi=800, bbox_inches='tight')
        plt.show()
            
        # =============================================================================

        
        # =============================================================================
        # ACTIVITY PERIODS LENGTH ANALYSIS
        # =============================================================================
        # Define bins and compute frequencies
        activity_periods = np.concatenate(activity_periods)
        max_value = int(np.nanmax(activity_periods))
        bins = np.arange(1, max_value + 2)
        counts, _ = np.histogram(activity_periods, bins=bins)
        
        # Normalize counts to get probabilities
        counts_normalized = counts / counts.sum()
        
        # Compute cumulative sum of the normalized counts
        cumulative_counts = np.cumsum(counts_normalized)
        
        # Set up the figure
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        # Plot histogram with plt.bar on the left y-axis
        ax1.bar(bins[:-1], counts_normalized, color='skyblue', edgecolor='darkblue', alpha=0.75, width=0.8)
        ax1.set_ylim(0, 0.4)  # Set limits for left y-axis
        ax1.set_xlabel("Length of Activity Periods", fontsize=14)
        ax1.set_ylabel("Probability (Histogram)", fontsize=14, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Create a secondary y-axis for the cumulative sum
        ax2 = ax1.twinx()
        ax2.plot(bins[:-1], cumulative_counts, color='orange', linestyle='-', marker='o', linewidth=2, label="Cumulative Sum")
        ax2.set_ylim(0, 1)  # Set limits for right y-axis
        ax2.set_ylabel("Cumulative Probability", fontsize=14, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Add titles, labels, and legend
        plt.title("Normalized Distribution of Activity Periods Lengths for all the runs\nFill length: "+str(n)+' - Proximity analysis: '+str(dx)+'x'+str(dy), fontsize=16, fontweight='bold')
        
        # Custom x-ticks for better readability
        ax1.set_xticks(np.arange(0, max_value + 5, 5))
        
        # Add grid and background color to the left y-axis
        ax2.grid(axis='y', linestyle='--', alpha=1)
        ax2.set_facecolor('whitesmoke')
        
        # Add footer with file info
        plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", 
                 transform=ax1.transAxes, ha='center', va='center', fontsize=12, color='grey')
        
        # Save and display
        plt.savefig(os.path.join(plot_dir, 'normalized_activity_periods_length.pdf'), dpi=800, bbox_inches='tight')
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
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
        periods_matrix = tracers_matrix[:,6:]
        
        N_array     = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 120, 140, 160]
        # N_array     = [0, 1]
        ratio_array = []

        
        for n in N_array:
            
            # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
            tracers_matrix_filled = fill_consecutive_zeros(periods_matrix, n)
            
            # COUNT CONSECUTIVE ACTIVE PERIODS
            ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, periods_matrix.shape[1], periods_matrix.shape[0]-1)
            
            activity_periods = np.copy(ones_stack)
            
            # TRIM EMPTY ROWS
            activity_periods = activity_periods[~np.isnan(activity_periods).all(axis=1)]
            all_active_position = activity_periods.shape[0]
            
            # TRIM ALL THE TRACERS THAT ARE ACTIVE MORE THAN ONCE
            activity_periods_trim = activity_periods[np.isnan(activity_periods[:, 1])]
            once_active_position = activity_periods_trim.shape[0]
        
            ratio_array.append(once_active_position/all_active_position)

        plt.plot(N_array, ratio_array, marker='o', linestyle='-', label=run_name)
    plt.xlabel('Filling analysis window dimension')
    plt.ylabel('Actived once positions / All positions')
    plt.title('Effects of temporal filtering - Run ' +  run_name[:5] + '\nProximity analysis: '+str(dx)+'x'+str(dy))
    plt.grid(True)
    plt.ylim(0.2, 1)
    plt.legend()
    plt.text(0.5, -0.12, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=6)
    plt.savefig(os.path.join(plot_dir, 'effect_of_temporal_filtering.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
    
    
