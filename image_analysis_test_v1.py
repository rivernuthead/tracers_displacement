# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:58:53 2022

@author: Erri
"""

# import necessary packages
import time
start_time = time.time()
import os
from PIL import Image
import numpy as np
import imageio
import re
from scipy.ndimage import label, find_objects, convolve
import imageio.core.util

# =============================================================================
# FUNCTIONS
# =============================================================================

def clean_matrix(matrix, compact_threshold, min_size):
    """
    Removes oblong-shaped and small components from a binary matrix.

    Parameters:
    - matrix (np.ndarray): Binary matrix (2D) with values 0 and 1.
    - compact_threshold (float): Maximum aspect ratio to consider a component as compact.
    - min_size (int): Minimum size (number of 1s) for a component to be retained.

    Returns:
    - np.ndarray: Binary matrix with oblong and small components removed.
    """
    # Label connected components
    labeled_matrix, num_features = label(matrix)
    
    # Loop through each labeled component and apply filters
    for i in range(1, num_features + 1):
        slice_x, slice_y = find_objects(labeled_matrix == i)[0]
        component = labeled_matrix[slice_x, slice_y] == i
        
        # Calculate dimensions of the bounding box and aspect ratio
        height, width = component.shape
        aspect_ratio = max(height, width) / min(height, width) if min(height, width) > 0 else float('inf')
        
        # Check size of the component
        component_size = component.sum()
        
        # Remove component if it's oblong or too small
        if aspect_ratio > compact_threshold or component_size < min_size:
            labeled_matrix[slice_x, slice_y][component] = 0

    # Return cleaned binary matrix
    return (labeled_matrix > 0).astype(int)


def fill_small_holes(matrix, max_hole_size=10, surround_threshold=7):
    """
    Fills small holes (connected 0-regions) in a binary matrix, and fills 0s surrounded by a majority of 1s.

    Parameters:
    - matrix (np.ndarray): Binary matrix (2D) with values 0 and 1.
    - max_hole_size (int): Maximum size (number of 0s) for a hole to be filled.
    - surround_threshold (int): Minimum count of surrounding 1s to fill a 0.

    Returns:
    - np.ndarray: Binary matrix with small holes and isolated 0s filled.
    """
    # Step 1: Fill small holes
    inverted_matrix = (matrix == 0).astype(int)
    holes_labeled, num_holes = label(inverted_matrix)
    
    for i in range(1, num_holes + 1):
        slice_x, slice_y = find_objects(holes_labeled == i)[0]
        hole = holes_labeled[slice_x, slice_y] == i
        hole_size = hole.sum()
        
        # Fill the hole if it is smaller than max_hole_size
        if hole_size <= max_hole_size:
            matrix[slice_x, slice_y][hole] = 1

    # Step 2: Fill isolated 0s surrounded by a majority of 1s
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0  # Exclude the center

    # Convolve to count surrounding 1s for each cell
    surrounding_counts = convolve(matrix, kernel, mode='constant', cval=0)

    # Fill 0s that are surrounded by a certain threshold of 1s
    matrix[(matrix == 0) & (surrounding_counts >= surround_threshold)] = 1

    return matrix
# =============================================================================


'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# SET RUN NAME
# run_names = ['q05_1r7']
# run_names = ['q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q05_1r3copy']


# Script parameters:
run_mode = 1
# Set working directory
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')
tracer_extraction_folder_path = os.path.join(output_dir, 'tracer_extraction')
if not os.path.exists(tracer_extraction_folder_path):
    os.mkdir(tracer_extraction_folder_path)

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def silence_imageio_warning(*args, **kwargs):
    pass

#imageio.core.util._precision_warn = silence_imageio_warning

# Set parameters
gmr_thr = 25   # G - R threshold 
dt = 4  #time between photos
thr = 36 #area threshold of a single tracer
ntracmax = 1000
new_ntracmax = 2000
area_threshold = 6
areaperimeter_threshold = 0.5
mask_buffer = 12
tdiff = 4

# INITIALIZE ARRAYS
feeding_position = []


# =============================================================================
# # LOOP OVER run_names
# =============================================================================
for run_name in run_names:
    print(run_name, ' is running...')
    
    tracers_extraction_imgs = []
    
    t = 0
    i = 0
    path_in = os.path.join(input_dir, 'cropped_images', run_name)
    
    tracer_extraction_path = os.path.join(tracer_extraction_folder_path, run_name)
    if not os.path.exists(tracer_extraction_path):
        os.mkdir(tracer_extraction_path)
        
    # Create outputs script directory
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run_name)
    if not os.path.exists(path_out):
        os.mkdir(path_out) 
        
    run_param = np.loadtxt(os.path.join(input_dir, 'run_param_'+run_name[0:3]+'.txt'), skiprows=1, delimiter=',')
    
    # =============================================================================
    # POSITION PARAMETERS
    # =============================================================================

    # List input directory files and filter for files that end with .jpg
    files_tot = sorted([f for f in os.listdir(path_in) if f.endswith('.jpg') and not f.startswith('.')])
    files = files_tot
    
#%%
    for file in sorted(files,key = numericalSort):
        # print('Time: ', t, 's')
        path = os.path.join(path_in, file) # Build path
        
        img = Image.open(path) # Open image
        img_array = np.asarray(img)    # Convert image in numpy array
        
        # IMPORT AND APPLY IMAGE MASK
        img_mask = Image.open(os.path.join(input_dir, 'img_masks', run_name + '_img_domain_mask.tif'))
        img_mask_array = np.asarray(img_mask)

          
        # EXTRACT RGB BANDS AND CONVERT AS INT64:
        band_red = img_array[:,:,0]*img_mask_array
        band_red = band_red.astype(np.int64)
        band_green = img_array[:,:,1]*img_mask_array
        band_green = band_green.astype(np.int64)
        band_blue = img_array[:,:,2]*img_mask_array
        band_blue = band_blue.astype(np.int64)
        
        # EXTRACT THE FLUORESCENT TRACERS
        # img_extract = ((band_green-band_red)>-1)*((band_red+band_green+band_blue)>350)*((band_red+band_green+band_blue)<700) # modified 2024/10/20
        # img_extract = ((band_green-band_red)>-20)*((band_red+band_green+band_blue)>350)*((band_red+band_green+band_blue)<700) # modified 2024/10/20
        img_extract = ((band_green-band_red)>-10)*((band_red+band_green+band_blue)>350)*((band_red+band_green+band_blue)<700)*(band_green>100)*(band_red<190) # modified 2024/11/11
        img_gmr_filt = np.where(img_extract==1, 1, np.nan)
        

        tracers_extraction_imgs.append(img_gmr_filt)
        
        # RESCALE IMAGE WITH VALUES FROM 0 TO 255
        img_gmr_filt = np.where(np.logical_not(np.isnan(img_gmr_filt)),255,0)
        img_gmr_filt = img_gmr_filt.astype(np.uint8)
        imageio.imwrite(os.path.join(tracer_extraction_path, str(t) + 's_gmr.png'), img_gmr_filt)
        
        
        i +=1
        t += dt
    
    # STACK ALL THE TRACER EXTRACTION MAPS
    tracers_extraction_imgs_stack = np.stack(tracers_extraction_imgs)
    np.save(os.path.join(tracer_extraction_path, 'tracer_extraction_images_stack.npy'), tracers_extraction_imgs_stack)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")


#%%
for run_name in run_names:
    
    tracer_extraction_path = os.path.join(tracer_extraction_folder_path, run_name)
    
    tracers_extraction_imgs_stack = np.load(os.path.join(tracer_extraction_path, 'tracer_extraction_images_stack.npy'))
    tracers_extraction_imgs_envelope = np.nansum(tracers_extraction_imgs_stack, axis=0)
    
    # FILT
    tracers_extraction_imgs_envelope = 1*(tracers_extraction_imgs_envelope>2)
    np.save(os.path.join(tracer_extraction_path, 'tracer_extraction_image_envelope.npy'), tracers_extraction_imgs_envelope)

#%%
for run_name in run_names:
    tracer_extraction_path = os.path.join(tracer_extraction_folder_path, run_name)
    tracers_extraction_imgs_envelope = np.load(os.path.join(tracer_extraction_path, 'tracer_extraction_image_envelope.npy'))
    
    # REMOVE SMALL AND OBLONG OBJECTS
    compact_threshold=5
    min_size=2
    mask_filt = clean_matrix(tracers_extraction_imgs_envelope, compact_threshold, min_size)
    
    mask_filt = np.where(tracers_extraction_imgs_envelope==0, np.nan, tracers_extraction_imgs_envelope)
    # RESCALE IMAGE WITH VALUES FROM 0 TO 255
    mask_filt = np.where(mask_filt>0, 255,0)
    mask_filt = mask_filt.astype(np.uint8)
    imageio.imwrite(os.path.join(tracer_extraction_path, 'gmr_mask.png'), mask_filt)
    
#%%
tracer_ultimate_extraction_folder_path = os.path.join(tracer_extraction_folder_path, 'ultimate_extraction')
if not os.path.exists(tracer_ultimate_extraction_folder_path):
    os.mkdir(tracer_ultimate_extraction_folder_path)
    
    
for run_name in run_names:
    tracer_extraction_ultimate_path = os.path.join(tracer_ultimate_extraction_folder_path, run_name)
    if not os.path.exists(tracer_extraction_ultimate_path):
        os.mkdir(tracer_extraction_ultimate_path)
    
    path_in = os.path.join(input_dir, 'cropped_images', run_name)
    tracer_extraction_path = os.path.join(tracer_extraction_folder_path, run_name)
    
    # IMPORT FILTERING MASK
    mask_filt = Image.open(os.path.join(tracer_extraction_path, 'gmr_mask.png')) # Open image
    mask_filt_array = np.asarray(mask_filt)    # Convert image in numpy array
    
    
    t = 0
    i = 0
    files = files_tot[:-1]
    for file in sorted(files,key = numericalSort):
        
        # IMPORT THE FIRST STEP BAND EXTRACTION
        image_rmg = Image.open(os.path.join(tracer_extraction_path, str(t) + 's_gmr.png'))
        
        # IMPORT AND APPLY IMAGE MASK
        img_mask = Image.open(os.path.join(input_dir, 'img_masks', run_name + '_img_domain_mask.tif'))
        img_mask_array = np.asarray(img_mask)
        
        # PERFORM THE SECOND STEP BAND EXTRACTION
        img_path = os.path.join(path_in, file) # Build path
        img = Image.open(img_path) # Open image
        img_array = np.asarray(img)    # Convert image in numpy array
        
        # BAND EXTRACTION
        band_red = img_array[:,:,0]*img_mask_array
        band_red = band_red.astype(np.int64)
        band_green = img_array[:,:,1]*img_mask_array
        band_green = band_green.astype(np.int64)
        band_blue = img_array[:,:,2]*img_mask_array
        band_blue = band_blue.astype(np.int64)
        
        img_extract = ((band_green-band_red)>-24)*((band_red+band_green+band_blue)>350)*((band_red+band_green+band_blue)<700)*mask_filt_array # modified 2024/11/11
        
        # OVERLAP OLD EXTRACTIO AND NEW EXTRACTION
        tracers_extraction_img = img_extract+image_rmg
        
        tracers_extraction_img = fill_small_holes(tracers_extraction_img, max_hole_size=3, surround_threshold=6)
        # tracers_extraction_img = fill_small_holes(tracers_extraction_img, max_hole_size=3, surround_threshold=5)
        tracers_extraction_img = np.where(tracers_extraction_img>0,1,np.nan)
        
        
        # RESCALE IMAGE WITH VALUES FROM 0 TO 255
        tracers_extraction_img_print = np.where(np.logical_not(np.isnan(tracers_extraction_img)),255,0)
        tracers_extraction_img_print = tracers_extraction_img_print.astype(np.uint8)
        imageio.imwrite(os.path.join(tracer_extraction_ultimate_path, str(t) + 's_gmr_ult.png'), tracers_extraction_img_print)
        
        
        i +=1
        t += dt
    
    
    