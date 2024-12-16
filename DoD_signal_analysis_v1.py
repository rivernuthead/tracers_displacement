#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:19:31 2024

@author: erri
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

start_time = time.time()

# SET RUN NAME
# run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
              'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
              'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# run_names = ['q07_1r10']


# SET DIRECTORIES -------------------------------------------------------------
w_dir = '/Volumes/T7_Shield/PhD/repos/tracers_displacement/'
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data', 'DoD_analysis')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# IMPORT RUN PARAMETERS
run_parameters = pd.read_csv(os.path.join(w_dir, 'input_data', 'tracers_DEM_DoD_combination_v2.csv'))

# SURVEY PIXEL DIMENSION
px_x = 50  # [mm]
px_y = 5  # [mm]

# =============================================================================
# LOOP OVER run_names
# =============================================================================
for run_name in run_names:
    print(run_name, ' is running...')

    path_in = os.path.join(input_dir, 'cropped_images', run_name)
    path_in_DEM = os.path.join(input_dir, 'surveys', run_name[0:5])
    path_in_DoD = os.path.join(input_dir, 'DoDs', 'DoD_' + run_name[0:5])

    # EXTRACT RUN PARAMETERS --------------------------------------------------
    selected_row = run_parameters.loc[run_parameters['RUN'] == run_name]
    if not selected_row.empty:

        DEM_name = int(selected_row['DEM'].values[0])  # DEM name

        DoD_name = selected_row['DoD timespan1'].values[0]  # DoD name timespan 1

        feed_x = selected_row['feed-x'].values[0]  # Feeding x-coordinate

        frame_position = selected_row['frame_position 1  [m]'].values[0]  # Frame position in meters
        frame_position += 0.44  # Adjust frame position

        frame_position2 = selected_row['frame_position 2  [m]'].values[0]  # Frame position 2 in meters

    # TOPOGRAPHIC DATA IMPORT
    array_mask = np.loadtxt(os.path.join(input_dir, 'array_mask.txt'))
    array_mask = np.where(array_mask != -999, 1, np.nan)

    DEM_path = os.path.join(path_in_DEM, 'matrix_bed_norm_' + run_name[0:3] + '_1s' + str(DEM_name) + '.txt')
    DEM = np.loadtxt(DEM_path, skiprows=8)
    if 'q10_1r' in run_name:
        DEM = DEM[:, :236]
        array_mask = np.loadtxt(os.path.join(input_dir, 'array_mask_reduced.txt'))
        array_mask = np.where(array_mask != -999, 1, np.nan)
    DEM = np.where(DEM == -999, np.nan, DEM)
    DEM = DEM * array_mask

    DoD_path = os.path.join(path_in_DoD, 'DoD_' + DoD_name + '_filt_ult.txt')
    DoD = np.loadtxt(DoD_path)

    # =============================================================================
    # REPEAT COLUMNS OF DoD
    # =============================================================================
    DoD_repeated = np.repeat(DoD, 10, axis=1)

    # =============================================================================
    # FILL ONLY
    # =============================================================================
    DoD_fill = np.copy(DoD_repeated)
    DoD_fill = DoD_fill * (DoD_repeated > 0)
    DoD_fill_array = np.nansum(DoD_fill, axis=0)

    DoD_fill_bool_array = np.nansum(DoD_fill > 0, axis=0)

    # =============================================================================
    # SCOUR AND FILL
    # =============================================================================
    DoD_array = np.nansum(DoD_fill, axis=0)

    DoD_bool = np.where(DoD_repeated > 0, 1, DoD_repeated)
    DoD_bool = np.where(DoD_bool < 0, -1, DoD_bool)
    DoD_bool_array = np.nansum(DoD_bool, axis=0)

    # =============================================================================
    # APPLY SMOOTHING
    # =============================================================================
    window_length = 201  # Smoothing window length (must be odd)
    polyorder = 3       # Polynomial order for Savitzky-Golay filter

    DoD_fill_array_smooth = savgol_filter(DoD_fill_array, window_length, polyorder, mode='interp')
    DoD_fill_bool_array_smooth = savgol_filter(DoD_fill_bool_array, window_length, polyorder, mode='interp')
    DoD_array_smooth = savgol_filter(DoD_array, window_length, polyorder, mode='interp')
    DoD_bool_array_smooth = savgol_filter(DoD_bool_array, window_length, polyorder, mode='interp')

    # =============================================================================
    # IDENTIFY PEAKS
    # =============================================================================
    prominence_threshold = 10  # Adjust this value based on your data

    peaks_fill, properties_fill = find_peaks(DoD_fill_array_smooth, prominence=prominence_threshold)
    peaks_fill_bool, properties_fill_bool = find_peaks(DoD_fill_bool_array_smooth, prominence=prominence_threshold)
    peaks_array, properties_array = find_peaks(DoD_array_smooth, prominence=prominence_threshold)
    peaks_bool, properties_bool = find_peaks(DoD_bool_array_smooth, prominence=prominence_threshold)
    
    # Save pronounced peaks to report
    report_data = {
        'DoD_fill_array': [(p, DoD_fill_array_smooth[p]) for p in peaks_fill],
        'DoD_fill_bool_array': [(p, DoD_fill_bool_array_smooth[p]) for p in peaks_fill_bool],
        'DoD_array': [(p, DoD_array_smooth[p]) for p in peaks_array],
        'DoD_bool_array': [(p, DoD_bool_array_smooth[p]) for p in peaks_bool]
    }
    
    # Save peaks to report
    report_rows = []
    for key, peaks in report_data.items():
        for peak in peaks:
            report_rows.append({
                'Array Name': key,
                'Peak Index': peak[0],
                'Peak Value': peak[1]
            })
    
    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(output_dir, f'peaks_report_{run_name}.csv')
    report_df.to_csv(report_path, index=False)


    # =============================================================================
    # PLOTTING
    # =============================================================================
    fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    
    # Adjust ticks
    x_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700]
    x_labels = ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5', '13.0', '13.5']
    y_ticks = [12, 72, 132]
    y_labels = ['0', '0.3', '0.6']
    
    # Plot DoD matrix
    axs[0].imshow(DoD_repeated, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axs[0].set_title('DoD Matrix')
    axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[0].set_xticks(ticks=x_ticks, labels=x_labels)
    axs[0].set_xlabel('Longitudinal coordinate [m]')
    
    # Plot DoD_fill_array
    axs[1].plot(DoD_fill_array, label='DoD Fill Array', color='blue', alpha=0.7)
    axs[1].plot(DoD_fill_array_smooth, label='Smoothed Trend', color='orange', linestyle='--')
    axs[1].vlines(peaks_fill, ymin=DoD_fill_array_smooth.min(), ymax=DoD_fill_array_smooth.max(), colors='red', linestyles='dotted', label='Peaks')
    axs[1].set_ylabel('Sum [units]')
    axs[1].legend(loc='upper left')

    # Plot DoD_fill_bool_array
    axs[2].plot(DoD_fill_bool_array, label='DoD Fill Bool Array', color='green', alpha=0.7)
    axs[2].plot(DoD_fill_bool_array_smooth, label='Smoothed Trend', color='orange', linestyle='--')
    axs[2].vlines(peaks_fill_bool, ymin=DoD_fill_bool_array_smooth.min(), ymax=DoD_fill_bool_array_smooth.max(), colors='red', linestyles='dotted', label='Peaks')
    axs[2].set_ylabel('Count')
    axs[2].legend(loc='upper left')

    # Plot DoD_array
    axs[3].plot(DoD_array, label='DoD Array', color='red', alpha=0.7)
    axs[3].plot(DoD_array_smooth, label='Smoothed Trend', color='orange', linestyle='--')
    axs[3].vlines(peaks_array, ymin=DoD_array_smooth.min(), ymax=DoD_array_smooth.max(), colors='red', linestyles='dotted', label='Peaks')
    axs[3].set_ylabel('Sum [units]')
    axs[3].legend(loc='upper left')

    # Plot DoD_bool_array
    axs[4].plot(DoD_bool_array, label='DoD Bool Array', color='purple', alpha=0.7)
    axs[4].plot(DoD_bool_array_smooth, label='Smoothed Trend', color='orange', linestyle='--')
    axs[4].vlines(peaks_bool, ymin=DoD_bool_array_smooth.min(), ymax=DoD_bool_array_smooth.max(), colors='red', linestyles='dotted', label='Peaks')
    axs[4].set_ylabel('Count')
    axs[4].legend(loc='upper left')
    axs[4].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[4].set_xticks(ticks=x_ticks, labels=x_labels)
    axs[4].set_xlabel('Longitudinal coordinate [m]')

    
    plt.tight_layout()
    plt.show()

    

# =============================================================================
# MAW STABILITY ANALYSIS
# =============================================================================


    MAW = np.nansum(abs(DoD_bool), axis=0)
    
    # Compute cumulative average
    cumulative_avg = np.cumsum(MAW) / np.arange(1, len(MAW) + 1)
    
    # Plot the input array (MAW) and the cumulative average
    plt.figure(figsize=(8, 5))
    plt.plot(MAW, label='Input Array (MAW)', marker='o')
    plt.plot(cumulative_avg, label='Cumulative Average', marker='x', linestyle='--')
    plt.title('Input Array and Cumulative Average - ' + run_name)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # Initialize parameters and results
    window_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])*120  # Different non-overlapping portion sizes
    avg_results = []  # To store the average of averages for each window size
    stdev_results = []  # To store the standard deviation of averages for each window size
    
    # Process for each window size
    for window_size in window_sizes:
        # Split the array into non-overlapping portions of the current window size
        reshaped_array = MAW[: len(MAW) // window_size * window_size].reshape(-1, window_size)
        
        print(reshaped_array.shape)
        
        # Compute the average of each portion
        portion_averages = reshaped_array.mean(axis=1)
        
        # Compute the overall average and standard deviation of these averages
        avg_results.append(portion_averages.mean())
        stdev_results.append(portion_averages.std())
    
    # Compute the overall average and standard deviation of these averages
    avg_results.append(MAW.mean())
    stdev_results.append(0)
    window_sizes = np.append(window_sizes,len(MAW))
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(window_sizes, avg_results, yerr=stdev_results, fmt='-o', capsize=5, label='Mean ± Stdev')
    plt.title('Average and Standard Deviation for Different Window Sizes - ' + run_name)
    plt.xlabel('Window Size (Columns)')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    

    
    