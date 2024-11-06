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
run = 'q05_1r1'


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Set the working directory
w_dir = os.getcwd()  # Get directory of current Python script
path_in = os.path.join(w_dir, 'cropped_images', run)
 # Create outputs script directory
path_out = os.path.join(w_dir, 'timelapse', run)
if os.path.exists(path_out):
   shutil.rmtree(path_out, ignore_errors=False, onerror=None)
   os.mkdir(path_out) 
else: 
   os.mkdir(path_out)

img_array = []
for filename in sorted(glob.glob(path_in + '\\*.jpg'),key = numericalSort):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(path_out+'\\timelapse.avi',cv2.VideoWriter_fourcc(*('M','J','P','G') ), 3, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()