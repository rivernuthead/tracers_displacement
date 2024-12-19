<<<<<<< HEAD
'''


'''

# IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SET DIRECTORIES
w_dir = os.getcwd()
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data', 'Morphological_method')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# IMPORT SCRIPT PARAMETERS
# run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
#               'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
#               'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q07_1r10']

run_parameters = pd.read_csv(os.path.join(w_dir, 'input_data', 'tracers_DEM_DoD_combination_v2.csv'))

px_x, px_y = 50, 5 # Survey cell dimension [mm]

for run_name in run_names:
    print(run_name, ' is running...')

    # Define run-specific folders
    path_in = os.path.join(input_dir, 'cropped_images', run_name)
    path_in_DEM = os.path.join(input_dir, 'surveys', run_name[0:5])
    path_in_DoD = os.path.join(input_dir, 'DoDs', 'DoD_' + run_name[0:5])

    # EXTRACT RUN PARAMETERS
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

    DEM_scaled = np.repeat(DEM, 10, axis=1)
    DoD_scaled = np.repeat(DoD, 10, axis=1)

    # MORPHOLOGICAL APPROACH

    mean_values = np.nanmean(DoD_scaled, axis=0)

    # Shifted curve: Subtract the minimum value from the mean_values to make the minimum zero
    shifted_mean_values = mean_values - np.min(mean_values)

    # PLOT
    plt.plot(mean_values, linestyle='-', color='b', label='Original Mean Values')
    # plt.plot(shifted_mean_values, linestyle='--', color='r', label='Shifted Mean Values')
    plt.title('Profile of Mean Values Along Axis 1')
    plt.xlabel('Row Index')
    plt.ylabel('Mean Value')
    plt.grid(True)
    plt.legend()
=======
'''


'''

# IMPORT LIBRARIES
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# SET DIRECTORIES
w_dir = os.getcwd()
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data', 'Morphological_method')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# IMPORT SCRIPT PARAMETERS
# run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
#               'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
#               'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q07_1r10']

run_parameters = pd.read_csv(os.path.join(w_dir, 'input_data', 'tracers_DEM_DoD_combination_v2.csv'))

px_x, px_y = 50, 5 # Survey cell dimension [mm]

for run_name in run_names:
    print(run_name, ' is running...')

    # Define run-specific folders
    path_in = os.path.join(input_dir, 'cropped_images', run_name)
    path_in_DEM = os.path.join(input_dir, 'surveys', run_name[0:5])
    path_in_DoD = os.path.join(input_dir, 'DoDs', 'DoD_' + run_name[0:5])

    # EXTRACT RUN PARAMETERS
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

    DEM_scaled = np.repeat(DEM, 10, axis=1)
    DoD_scaled = np.repeat(DoD, 10, axis=1)

    # MORPHOLOGICAL APPROACH

    mean_values = np.nanmean(DoD_scaled, axis=0)

    # Shifted curve: Subtract the minimum value from the mean_values to make the minimum zero
    shifted_mean_values = mean_values - np.min(mean_values)

    # PLOT
    plt.plot(mean_values, linestyle='-', color='b', label='Original Mean Values')
    # plt.plot(shifted_mean_values, linestyle='--', color='r', label='Shifted Mean Values')
    plt.title('Profile of Mean Values Along Axis 1')
    plt.xlabel('Row Index')
    plt.ylabel('Mean Value')
    plt.grid(True)
    plt.legend()
>>>>>>> 60c5b80393d30e118e171a656b92d12979b8b689
    plt.show()