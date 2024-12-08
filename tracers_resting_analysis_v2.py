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
inactivity_periods_analysis_single_charts = False
tempora_filtering_effects = False
activity_periods_analysis   = False
single_periods_analysis = False
activity_periods__DoD_analysis = True

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
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# run_names = ['q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r3']
# run_names = ['q05_1r3copy']


# PROXIMITY ANALYSIS
dx, dy = 5, 5

# FOLDERS SETUP
# =============================================================================
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')
plot_dir = os.path.join(output_dir, 'resting_time')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)



#%%
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
if tempora_filtering_effects:
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
    plt.text(0.5, -0.20, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.savefig(os.path.join(plot_dir, 'effect_of_temporal_filtering.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
    


# =============================================================================
# ACTIVITY PERIODS ANALYSIS
# =============================================================================
if activity_periods_analysis:
    for run_name in run_names:
        print(run_name, ' is running...')
        # LOAD DATA
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
        periods_matrix = tracers_matrix[:, 6:]
        tracer_positions = tracers_matrix[:, :6]
        
        # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
        tracers_matrix_filled = fill_consecutive_zeros(periods_matrix, 4)
        
        # COUNT CONSECUTIVE ACTIVE PERIODS
        ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, periods_matrix.shape[1], periods_matrix.shape[0]-1)
        
        # REMOVE ONES AS THEY ARE CONSIDERED AS MOVING PARTCLES
        ones_stack = np.where(ones_stack==1,np.nan,ones_stack)
        
        activity_periods = np.hstack((tracer_positions[:-1, :], ones_stack))
        activity_periods = activity_periods[~np.all(np.isnan(activity_periods), axis=1)]
        
        # Total number of rows for normalization
        total_rows = activity_periods.shape[0]
        
        # Create datasets based on column 3 values
        datasets = {
            'Fill': activity_periods[activity_periods[:, 3] > 0],
            'Not detectable changes': activity_periods[activity_periods[:, 3] == 0],
            'Scour': activity_periods[activity_periods[:, 3] < 0]
        }
        
        # Define bins for left and right histograms
        bins_left = np.linspace(1, 10, 11)  # Values <= 10
        bins_right = np.linspace(10,int(np.nanmax(ones_stack)), int(np.nanmax(ones_stack))+1)  # Values > 10
        
        # Plot frequency distributions of columns from index 6 onward
        fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex='col')
        labels = ['Fill', 'Not detectable changes', 'Scour']
        colors = ['lightblue', 'lightgray', 'lightcoral']
        
        for i, key in enumerate(['Fill', 'Not detectable changes', 'Scour']):
            data = datasets[key][:, 6:]  # Select columns from index 6 onward
            data = data[~np.isnan(data)]  # Remove np.nan
            
            # Data:
            data_left = data[data <= 10]
            frequency_left, _ = np.histogram(data_left.flatten(), bins=bins_left, density=False)
            
            data_right = data[data > 10]
            frequency_right, _ = np.histogram(data_right.flatten(), bins=bins_right, density=False)

            # Normalize the data
            norm_sum = np.nansum(frequency_left) + np.nansum(frequency_right)
            normalized_frequency_left = frequency_left / norm_sum
            normalized_frequency_right = frequency_right / norm_sum
            
            # Left: Values <= 10
            axes[i, 0].bar(bins_left[:-1], normalized_frequency_left, width=np.diff(bins_left), align='edge', alpha=1, color=colors[i])
            axes[i, 0].set_title(f'{labels[i]} (<= 10)')
            axes[i, 0].set_ylabel('Normalized Frequency')

            # Right: Values > 10
            axes[i, 1].bar(bins_right[:-1], normalized_frequency_right, width=np.diff(bins_right), align='edge', alpha=1, color=colors[i])
            axes[i, 1].set_title(f'{labels[i]} (> 10)')

        fig.suptitle(f'Frequency Distribution Analysis - {run_name}', fontsize=16)
        plt.xlabel('Values')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.text(0.5, -0.20, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        plt.savefig(os.path.join(plot_dir, run_name + '_activity_periods_length_divided.pdf'), dpi=800, bbox_inches='tight')
        plt.show()

#%%
# =============================================================================
# SINGLE CHART DISTRIBUTION  
# =============================================================================
if activity_periods_analysis:
    
    run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
    run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
    run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

    
    # Initialize lists to store mean and median for each run_name
    mean_values = []
    median_values = []
    
    for run_name in run_names:
        print(run_name, ' is running...')
        # LOAD DATA
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
        periods_matrix = tracers_matrix[:, 6:]
        tracer_positions = tracers_matrix[:, :6]
        
        # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
        tracers_matrix_filled = fill_consecutive_zeros(periods_matrix, 4)
        
        # COUNT CONSECUTIVE ACTIVE PERIODS
        ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, periods_matrix.shape[1], periods_matrix.shape[0]-1)
        
        # REMOVE ONES AS THEY ARE CONSIDERED AS MOVING PARTICLES
        ones_stack = np.where(ones_stack == 1, np.nan, ones_stack)
        
        activity_periods = np.hstack((tracer_positions[:-1, :], ones_stack))
        activity_periods = activity_periods[~np.all(np.isnan(activity_periods), axis=1)]
        
        # Combine all data from columns 6 onward
        all_data = activity_periods[:, 6:].flatten()
        all_data = all_data[~np.isnan(all_data)]  # Remove NaN values
        
        # Compute the mean and median
        data_mean = np.nanmean(all_data)
        data_median = np.nanmedian(all_data)
        
        # Store mean and median
        mean_values.append(data_mean)
        median_values.append(data_median)
        
        # Define bins
        bins = np.linspace(1, int(np.nanmax(all_data)) + 1, int(np.nanmax(all_data)) + 1)
        
        # Calculate frequency distribution
        frequency, bins_edges = np.histogram(all_data, bins=bins, density=False)
        
        # Normalize the frequencies
        normalized_frequency = frequency / np.nansum(frequency)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.bar(bins[:-1], normalized_frequency, width=np.diff(bins), align='edge', color='lightblue', edgecolor='black', alpha=0.7, label='Frequency Distribution')
        
        # Add vertical lines for mean and median
        plt.axvline(data_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {data_mean:.2f}')
        plt.axvline(data_median, color='green', linestyle='-', linewidth=2, label=f'Median: {data_median:.2f}')
        
        # Labels and legend
        plt.title(f'Activity Periods Length Distribution - {run_name}', fontsize=16)
        plt.xlabel('Activity Period Length', fontsize=14)
        plt.ylabel('Normalized Frequency', fontsize=14)
        plt.legend(fontsize=12)
        plt.text(0.5, -0.20, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
        
        # Save and show the plot
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, run_name[:5] + '_activity_periods_length_combined.pdf'), dpi=800, bbox_inches='tight')
        plt.show()
    
    
    # Convert lists to numpy arrays for further processing if needed
    mean_values = np.array(mean_values)
    median_values = np.array(median_values)
    report_values = np.stack((mean_values, median_values))
    header = f"Generated by: {os.path.basename(__file__)}" + '\n' + run_name[:5] + '\n Trend of mean and median of tracers resting time'
    np.savetxt(os.path.join(plot_dir, run_name[:5] + '_mean_median_trends.txt'), report_values, header=header)
    # Plot the trends
    plt.figure(figsize=(12, 8))
    plt.plot(run_names, mean_values, marker='o', linestyle='-', color='blue', label='Mean', linewidth=2, markersize=8)
    plt.plot(run_names, median_values, marker='o', linestyle='--', color='green', label='Median', linewidth=2, markersize=8)
    
    # Add labels, title, and legend
    plt.title('Trend of mean and median of activity period lengths', fontsize=16)
    plt.xlabel('Run Names', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(alpha=0.4)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.text(0.5, -0.25, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir,run_name[:5] + '_mean_median_trends.pdf'), dpi=800, bbox_inches='tight')
    plt.show()
    
#%%
# =============================================================================
# RESTING TIME AND DoD
# =============================================================================
if activity_periods__DoD_analysis:
    
    run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
                 'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
                 'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']
    
    # run_names = ['q05_1r1']

    
    # Initialize lists to store mean and median for each run_name
    mean_values = []
    median_values = []
    
    for run_name in run_names:
        print(run_name, ' is running...')
        # LOAD DATA
        tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
        periods_matrix = tracers_matrix[:, 6:]
        tracer_positions = tracers_matrix[:, :6]
        
        # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
        tracers_matrix_filled = fill_consecutive_zeros(periods_matrix, 4)
        
        # COUNT CONSECUTIVE ACTIVE PERIODS
        ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, periods_matrix.shape[1], periods_matrix.shape[0]-1)
        
        # REMOVE ONES AS THEY ARE CONSIDERED AS MOVING PARTICLES
        ones_stack = np.where(ones_stack == 1, np.nan, ones_stack)
        
        # ADD POSTION AND TOPOGRAPHIC INFORMATION TO THE RESTING TIME LENGTH COUNT
        activity_periods = np.hstack((tracer_positions[:-1, :], ones_stack))
        activity_periods = activity_periods[~np.all(np.isnan(activity_periods), axis=1)]
        
        # Extract resting times and category indices

        # Extract resting times and category indices
        resting_times = activity_periods[:, 6:].flatten()  # Flatten resting time values
        categories = activity_periods[:, 3].repeat(activity_periods.shape[1] - 6)  # Repeat categories for each column >= 6
        
        # Remove NaN values
        valid_mask = ~np.isnan(resting_times)
        resting_times = resting_times[valid_mask]
        categories = categories[valid_mask]
        
        # Filter data based on categories
        scour_values = resting_times[categories < 0]
        no_change_values = resting_times[categories == 0]
        fill_values = resting_times[categories > 0]
        
        # Calculate histograms and normalize
        bins = np.linspace(0, np.nanmax(resting_times), 30)  # Define uniform bins
        hist_scour, _ = np.histogram(scour_values, bins=bins)
        hist_no_change, _ = np.histogram(no_change_values, bins=bins)
        hist_fill, _ = np.histogram(fill_values, bins=bins)
        
        # Normalize histograms so their sum equals 1
        total_sum = hist_scour.sum() + hist_no_change.sum() + hist_fill.sum()
        hist_scour = hist_scour / total_sum
        hist_no_change = hist_no_change / total_sum
        hist_fill = hist_fill / total_sum
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        axes[0].bar(bins[:-1], hist_scour, width=np.diff(bins), align="edge", color="red", alpha=0.7)
        axes[0].set_title("Scour")
        axes[0].set_xlabel("Resting Time")
        axes[0].set_ylabel("Normalized Frequency")
        
        axes[1].bar(bins[:-1], hist_no_change, width=np.diff(bins), align="edge", color="grey", alpha=0.7)
        axes[1].set_title("No Change")
        axes[1].set_xlabel("Resting Time")
        
        axes[2].bar(bins[:-1], hist_fill, width=np.diff(bins), align="edge", color="blue", alpha=0.7)
        axes[2].set_title("Fill")
        axes[2].set_xlabel("Resting Time")
        
        # Add suptitle
        fig.suptitle("Normalized Frequency Distribution of Resting Times - " + run_name, fontsize=12, weight="bold")
        plt.text(0.5, -0.25, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=6)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plot_dir, run_name + '_activity_periods_length_and_DoD.pdf'),dpi=800, format="pdf")
        plt.show()


#%%
# =============================================================================
# MEAN AND MEDIAN OVER DISCHARGE
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# File paths for the data
file_paths = [os.path.join(plot_dir, 'q05_1_mean_median_trends.txt'),
              os.path.join(plot_dir, 'q07_1_mean_median_trends.txt'),
              os.path.join(plot_dir, 'q10_1_mean_median_trends.txt')]
discharge_labels = ["0.5 l/s", "0.7 l/s", "1.0 l/s"]

# Initialize lists to store mean and median values for each discharge
means = []
medians = []

# Load the data
for file_path in file_paths:
    data = np.loadtxt(file_path)
    means.append(data[0])  # First row: mean values
    medians.append(data[1])  # Second row: median values

# Create the subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plot the boxplot for means
axes[0].boxplot(means, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
axes[0].set_title("Boxplot of Mean Resting Time Values", fontsize=16)
axes[0].set_ylabel("Value", fontsize=14)
axes[0].set_xticks(range(1, len(discharge_labels) + 1))
axes[0].set_xticklabels([f"Discharge {label}" for label in discharge_labels], rotation=45, fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot the boxplot for medians
axes[1].boxplot(medians, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='red', linewidth=2))
axes[1].set_title("Boxplot of Median Resting Time Values", fontsize=16)
axes[1].set_xticks(range(1, len(discharge_labels) + 1))
axes[1].set_xticklabels([f"Discharge {label}" for label in discharge_labels], rotation=45, fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and save
fig.suptitle("Boxplots of Mean and Median Resting Time Values by Discharge", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.text(0.0, -0.28, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
plt.savefig(os.path.join(plot_dir,"REPORT_boxplot_mean_median_side_by_side.pdf"), dpi=800, bbox_inches='tight')
plt.show()

#%%
# =============================================================================
# PERCENTAGE OF ONES
# =============================================================================
if single_periods_analysis:
    run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
    # run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
    # run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']
    
    
    percentages_of_ones = []  # To store the percentage of 1's for each run_name
    
    if activity_periods_analysis:
        for run_name in run_names:
            print(run_name, ' is running...')
            # LOAD DATA
            tracers_matrix = np.load(os.path.join(output_dir, 'tracer_matrix', run_name + '_tracer_matrix'+str(dx)+'x'+str(dy)+'.npy'))
            periods_matrix = tracers_matrix[:, 6:]
            tracer_positions = tracers_matrix[:, :6]
            
            # PERFORM TEMPORAL ANAYSIS - FILL INACTIVE PERIODS <=N LONG
            tracers_matrix_filled = fill_consecutive_zeros(periods_matrix, 4)
            
            # COUNT CONSECUTIVE ACTIVE PERIODS
            ones_stack, zeros_stack, neg_ones_stack = calculate_consecutive_periods(tracers_matrix_filled, periods_matrix.shape[1], periods_matrix.shape[0]-1)
            
            # Calculate percentage of 1's in `ones_stack`
            total_elements = np.size(ones_stack[~np.isnan(ones_stack)])  # Total elements
            num_ones = np.sum(ones_stack == 1)   # Number of ones
            percentage_ones = (num_ones / total_elements) * 100  # Percentage of ones
            percentages_of_ones.append(percentage_ones)  # Store the percentage
            
            # REMOVE ONES AS THEY ARE CONSIDERED AS MOVING PARTICLES
            ones_stack = np.where(ones_stack == 1, np.nan, ones_stack)
            
            activity_periods = np.hstack((tracer_positions[:-1, :], ones_stack))
            activity_periods = activity_periods[~np.all(np.isnan(activity_periods), axis=1)]
            
            # Combine all data from columns 6 onward
            all_data = activity_periods[:, 6:].flatten()
            all_data = all_data[~np.isnan(all_data)]  # Remove NaN values
    
    # Convert percentages to a NumPy array for further processing if needed
    percentages_of_ones = np.array(percentages_of_ones)
    
    # Plot the percentages for each run_name
    plt.figure(figsize=(10, 6))
    plt.bar(run_names, percentages_of_ones, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Number of one-long activity period over the total number of activity periods', fontsize=14)
    plt.xlabel('Run Names', fontsize=14)
    plt.ylabel('Percentage of 1\'s (%)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.text(0.5, -0.30, f"Generated by: {os.path.basename(__file__)}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=12)
    plt.savefig(os.path.join(plot_dir, run_name[:5] + '_percentage_of_ones_per_run.pdf'), dpi=800, bbox_inches='tight')
    plt.show()