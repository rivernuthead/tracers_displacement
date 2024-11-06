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
import re



numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

w_dir = os.getcwd() # Set Python script location as w_dir

path = os.path.join(w_dir, 'distribuzioni_q05_1r1')


img_array = []
for filename in sorted(glob.glob(path + '\\*.png'),key = numericalSort):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter(path+'\\timelapse.avi',cv2.VideoWriter_fourcc(*('M','J','P','G') ), 5, size)
     
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()