# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:13:38 2022

@author: Marco
"""

# import necessary packages
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# Script parameters:
run_mode = 1
ntracmax= 3000
# Set working directory
run = 'q05_1r6'
w_dir = os.getcwd() # Set Python script location as w_dir

# List all available runs, depending on run_mode
runs =[]
if run_mode==1:
    runs = [run] # Comment to perform batch process over folder
elif run_mode==2:
    for f in sorted(os.listdir(os.path.join(w_dir, 'output_images'))):
        path = os.path.join(os.path.join(w_dir, 'output_images'), f)
        if os.path.isdir(path) and not(f.startswith('_')):
            runs = np.append(runs, f)
else:
    pass

###############################################################################
# LOOP OVER RUNS
###############################################################################
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)

    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers_appeared = np.load(path_in_tracers + '\\tracers_appeared_'+ run +'.npy',allow_pickle=True)
    tracers_disappeared = np.load(path_in_tracers + '\\tracers_disappeared_'+ run +'.npy',allow_pickle=True)
    
    run_param = np.loadtxt(os.path.join(w_dir, 'run_param_'+run[0:3]+'.txt'), skiprows=1, delimiter=',')
     
    new_tracers = np.zeros((len(tracers),ntracmax,4))
    new_tracers_appeared = np.zeros((len(tracers_appeared),ntracmax,4))
    new_tracers_disappeared = np.zeros((len(tracers_disappeared),ntracmax,4))
    
    
    frame_position = run_param[int(run[6:])-1,1]
    frame_position += 0.44
    scale_to_mm = 0.0016*frame_position + 0.4827
    
    x_center = frame_position*1000 + 1100
    x_0 = x_center - 4288/2*scale_to_mm
    
    for i in range(len(tracers)):
        page = np.empty((0,4))
        for j in range(1000-1):
            ntrac = tracers[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([tracers[i,j,1]-np.nanmin(tracers[:,:,1]),tracers[i,j,2],tracers[i,j,3],tracers[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers[i,:,:] = page
    
    for i in range(len(tracers_appeared)):
        page = np.empty((0,4))
        for j in range(1000-1):
            ntrac = tracers_appeared[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([tracers_appeared[i,j,1]-np.nanmin(tracers[:,:,1]),tracers_appeared[i,j,2],tracers_appeared[i,j,3],tracers_appeared[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers_appeared[i,:,:] = page
     
    for i in range(len(tracers_disappeared)):
        page = np.empty((0,4))
        for j in range(1000-1):
            ntrac = tracers_disappeared[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([tracers_disappeared[i,j,1]-np.nanmin(tracers[:,:,1]),tracers_disappeared[i,j,2],tracers_disappeared[i,j,3],tracers_disappeared[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers_disappeared[i,:,:] = page    
    
    #np.save(path_in_tracers + '\\tracers_reduced_'+ run +'.npy',new_tracers)
    np.save(path_in_tracers + '\\tracers_appeared_reduced_'+ run +'.npy',new_tracers_appeared)
    np.save(path_in_tracers + '\\tracers_disappeared_reduced_'+ run +'.npy',new_tracers_disappeared)
    
    print('########################')
    print(run,' completed') 
    print('########################') 
    