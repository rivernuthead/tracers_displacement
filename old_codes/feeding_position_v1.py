#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 08:57:26 2024

@author: erri
"""

import time

import os
from PIL import Image
import numpy as np
import imageio
import pandas as pd
import geopandas as gpd
import shutil
from osgeo import gdal,ogr
import re
import imageio.core.util


w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')


run_names = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
             'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
             'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# INITIALIZE ARRAYS
feeding_position = []

area_threshold = 6


for run_name in run_names:
    print(run_name, ' is running...')
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run_name)

    run_param = np.loadtxt(os.path.join(input_dir, 'run_param_'+run_name[0:3]+'.txt'), skiprows=1, delimiter=',')
    frame_position = run_param[int(run_name[6])-1,1] # Extract frame position from run parameters file
    frame_position += 0.44 # Frame position in DEM CRS
    scale_to_mm = 0.0016*frame_position + 0.4827 # Factor scaling from px to mm
    x_center = frame_position*1000 + 1100 # Frame centroids x-coordinates in DEM CRS
    x_0 = x_center - 4288/2*scale_to_mm # Upstream edges x-coordinate in DEM CRS
    
    # Set .shp file where to save the feeding geometry
    feeding_poly = gpd.read_file(os.path.join(path_out, "0s_tracers.shp"))
    feeding_poly.drop(feeding_poly.index[feeding_poly.area == max(feeding_poly.area)], inplace = True) # Feeding contains a polygon that is theentire photo domain. Here this polygon is removed.
    feeding_poly.drop(feeding_poly.index[feeding_poly.area < area_threshold], inplace = True)
    feeding_poly.to_file(os.path.join(path_out, 'feeding_tracers.shp'))
    
    feeding = gpd.read_file(os.path.join(path_out, 'feeding_tracers.shp'))
    feeding['Centroid'] = feeding.centroid
    feeding = feeding.set_geometry('Centroid')
    feeding = feeding.drop(columns=['geometry'])
    feeding.to_file(os.path.join(path_out, 'feeding_tracers_centroids.shp'))
    feeding['x'] = feeding['Centroid'].x
    
    print('Feeding point x_coordinate (Photo CRS): ', min(feeding.x))
    # feeding_position.append(min(feeding.x))
    
    feeding['x'] = feeding['x'].mul(scale_to_mm)
    feeding['x'] = feeding['x'].add(x_0)
    
    # TODO The feeding x-coordinate could be the weighted mean with respect to the feeding_poly areas
    # feeding['x_weighted'] = feeding['x'] 
    
    x_start = min(feeding.x) # Feeding x-coordinate referred to the DEM CRS (first column of the DEM) as the minimum of the x-coordinate of each detected feeding area.
    
    print(run_name, ' x start: ', x_start)
    feeding_position.append(x_start)
    
    
# SAVE REPORT
np.savetxt(os.path.join(output_dir, 'feeding_position.txt'), feeding_position, header= 'Feeding coordinate imn mm in the laser CRS\n' + str(run_names))