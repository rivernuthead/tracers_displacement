#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:26:48 2024

@author: erri
"""

import numpy as np
from scipy.ndimage import label, find_objects, convolve

def clean_matrix(matrix, compact_threshold=1.5, min_size=5):
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

# Example usage
matrix = np.array([
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1],
])

# # First, clean the matrix to remove oblong and small objects
# cleaned_matrix = clean_matrix(matrix, compact_threshold=1.5, min_size=5)
# Then, fill small holes in the cleaned matrix
final_matrix = fill_small_holes(matrix, max_hole_size=3, surround_threshold=5)
print(final_matrix)
