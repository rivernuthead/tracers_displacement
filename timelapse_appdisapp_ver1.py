#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:31:24 2022

@author: Marco
"""
# Import libraries
import os
import cv2
import glob
import shutil
import re
import numpy as np


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Script parameters:
run_mode = 2
# Set working directory
run = 'q05_1r2'
w_dir = os.getcwd() # Set Python script location as w_dir

# List all available runs, depending on run_mode
runs =[]
if run_mode==1:
    runs = [run] # Comment to perform batch process over folder
elif run_mode==2:
    for f in sorted(os.listdir(os.path.join(w_dir, 'appdisapp'))):
        path = os.path.join(os.path.join(w_dir, 'appdisapp'), f)
        if os.path.isdir(path) and not(f.startswith('_')):
            runs = np.append(runs, f)
else:
    pass

for run in runs:
    path = os.path.join(w_dir, 'appdisapp', run)
    path_out = os.path.join(w_dir, 'timelapse_appdisapp', run)
    if os.path.exists(path_out):
       shutil.rmtree(path_out, ignore_errors=False, onerror=None)
       os.mkdir(path_out) 
    else: 
       os.mkdir(path_out)

    img_array = []
    for filename in sorted(glob.glob(path + '\\*.png'),key = numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    
    out = cv2.VideoWriter(path_out+'\\timelapse.avi',cv2.VideoWriter_fourcc(*('M','J','P','G') ), 2, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()