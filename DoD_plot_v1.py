# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:13:38 2022

@author: Marco
"""

# import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from matplotlib.pyplot import cm

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

w_dir = os.getcwd() # Set Python script location as w_dir



# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]
vmin = -10
vmax = 10

path_in_DEM = os.path.join(w_dir, 'input_data','surveys','q10_1')
path_in_DoD_q05 = os.path.join(w_dir, 'input_data','DoDs','DoD_q05_1')
path_in_DoD_q07 = os.path.join(w_dir, 'input_data','DoDs','DoD_q07_1')
path_in_DoD_q10 = os.path.join(w_dir, 'input_data','DoDs','DoD_q10_1')

report_modes = pd.read_csv(os.path.join(w_dir, 'report_all_modes.txt'))
report_modes['media_g_DoD'] = (report_modes.media+report_modes.deltax)/px_x
report_modes['moda_1_g_DoD'] = (report_modes.moda_1+report_modes.deltax)/px_x
report_modes['moda_2_g_DoD'] = (report_modes.moda_2+report_modes.deltax)/px_x
report_modes['moda_3_g_DoD'] = (report_modes.moda_3+report_modes.deltax)/px_x
report_modes['L_DoD'] = (report_modes.L)/px_x
report_modes['x_0_DoD'] = (report_modes.x_0)/px_x


report_modes_q05 = report_modes[report_modes.portata == 5]
report_modes_q05_1 = report_modes_q05[report_modes_q05.prova <= 6]
report_modes_q05_1 = report_modes_q05_1.reset_index()
report_modes_q05_2 = report_modes_q05[report_modes_q05.prova > 6]
report_modes_q05_2 = report_modes_q05_2.reset_index()
report_modes_q07 = report_modes[report_modes.portata == 7]
report_modes_q07_1 = report_modes_q07[report_modes_q07.prova <= 6]
report_modes_q07_1 = report_modes_q07_1.reset_index()
report_modes_q07_2 = report_modes_q07[report_modes_q07.prova > 6]
report_modes_q07_2 = report_modes_q07_2.reset_index()
report_modes_q10 = report_modes[report_modes.portata == 10]
report_modes_q10_1 = report_modes_q10[report_modes_q10.prova <= 6]
report_modes_q10_1 = report_modes_q10_1.reset_index()
report_modes_q10_2 = report_modes_q10[report_modes_q10.prova > 6]
report_modes_q10_2 = report_modes_q10_2.reset_index()
     
DoD_q05_1 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_2-0_filt_fill.txt'))
DoD_q05_2 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-1_filt_fill.txt'))
DoD_q05_3 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-0_filt_fill.txt'))
        
x_ticks = [0, 10, 20, 30, 40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230]
x_labels = ['0', '0.5', '1.0','1.5','2.0','2.5','3.0', '3.5', '4.0','4.5','5.0','5.5', '6.0', '6.5','7.0','7.5','8.0', '8.5', '9.0','9.5','10.0','10.5', '11.0', '11.5'] 
y_ticks = [12, 72, 132]
y_labels = ['0','0.3','0.6'] 

fig, axs = plt.subplots(2,sharex=True,figsize = (16.5,11.7), tight_layout=True)   
im = axs[0].imshow(DoD_q05_1, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[0].set_title('1-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[0].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[1].imshow(DoD_q05_1, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[1].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[1].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[2].imshow(DoD_q05_2, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[2].set_title('2-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[2].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[3].imshow(DoD_q05_2, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[3].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[3].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[4].imshow(DoD_q05_3, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[4].set_title('3-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[4].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[5].imshow(DoD_q05_3, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[5].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[5].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[3], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[4], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[5], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
axs[5].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[3].set_yticks(ticks=y_ticks, labels=y_labels)
axs[4].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_xlabel('Length [m]',fontsize = 'large')
axs[0].set_ylabel('Width [m]',fontsize = 'large')
axs[1].set_ylabel('Width [m]',fontsize = 'large')
axs[2].set_ylabel('Width [m]',fontsize = 'large')
axs[3].set_ylabel('Width [m]',fontsize = 'large')
axs[4].set_ylabel('Width [m]',fontsize = 'large')
axs[5].set_ylabel('Width [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q05_DoD.png'), dpi=300)



    
fig, axs = plt.subplots(6,sharex=True,figsize = (16.5,11.7), tight_layout=True)   
im = axs[0].imshow(DoD_q05_1, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[0].set_title('1-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[0].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[1].imshow(DoD_q05_1, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[1].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[1].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[2].imshow(DoD_q05_2, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[2].set_title('2-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[2].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[3].imshow(DoD_q05_2, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[3].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[3].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[4].imshow(DoD_q05_3, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
axs[4].set_title('3-0 q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[4].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[5].imshow(DoD_q05_3, cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
runs =[]
for f in range(len(report_modes_q05_1)):
    runs = np.append(runs, 'q05_1r'+str(int(report_modes_q05_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[5].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[5].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[3], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[4], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[5], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
axs[5].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[3].set_yticks(ticks=y_ticks, labels=y_labels)
axs[4].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_xlabel('Length [m]',fontsize = 'large')
axs[0].set_ylabel('Width [m]',fontsize = 'large')
axs[1].set_ylabel('Width [m]',fontsize = 'large')
axs[2].set_ylabel('Width [m]',fontsize = 'large')
axs[3].set_ylabel('Width [m]',fontsize = 'large')
axs[4].set_ylabel('Width [m]',fontsize = 'large')
axs[5].set_ylabel('Width [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q05_DoD.png'), dpi=300)

DoD_q07_1 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_1-0_filt_fill.txt'))
DoD_q07_2 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_2-0_filt_fill.txt'))
DoD_q07_3 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_2-1_filt_fill.txt'))
        
fig, axs = plt.subplots(6,sharex=True,figsize = (16.5,11.7), tight_layout=True)   
im = axs[0].imshow(DoD_q07_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[0].set_title('1-0 q07',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[0].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[1].imshow(DoD_q07_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[1].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[1].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[2].imshow(DoD_q07_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[2].set_title('2-0 q07',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[2].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[3].imshow(DoD_q07_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[3].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[3].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[4].imshow(DoD_q07_3, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[4].set_title('3-0 q07',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[4].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[5].imshow(DoD_q07_3, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q07_1)):
    runs = np.append(runs, 'q07_1r'+str(int(report_modes_q07_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[5].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q07_1)):    
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[5].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[3], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[4], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[5], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
axs[5].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[3].set_yticks(ticks=y_ticks, labels=y_labels)
axs[4].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_xlabel('Length [m]',fontsize = 'large')
axs[0].set_ylabel('Width [m]',fontsize = 'large')
axs[1].set_ylabel('Width [m]',fontsize = 'large')
axs[2].set_ylabel('Width [m]',fontsize = 'large')
axs[3].set_ylabel('Width [m]',fontsize = 'large')
axs[4].set_ylabel('Width [m]',fontsize = 'large')
axs[5].set_ylabel('Width [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q07_DoD.png'), dpi=300)

DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_1-0_filt_fill.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-1_filt_fill.txt'))
DoD_q10_3 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-0_filt_fill.txt'))
        
fig, axs = plt.subplots(6,sharex=True,figsize = (16.5,11.7), tight_layout=True)   
im = axs[0].imshow(DoD_q10_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[0].set_title('1-0 q10',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[0].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[1].imshow(DoD_q10_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[1].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[1].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[2].imshow(DoD_q10_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[2].set_title('2-0 q10',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[2].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[3].imshow(DoD_q10_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[3].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[3].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[4].imshow(DoD_q10_3, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[4].set_title('3-0 q10',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[4].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
im = axs[5].imshow(DoD_q10_3, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))  
runs =[]
for f in range(len(report_modes_q10_1)):
    runs = np.append(runs, 'q10_1r'+str(int(report_modes_q10_1.prova[f])))        
for run in runs:
    path_in_tracers = os.path.join(w_dir, 'output_images', run)
    tracers = np.load(path_in_tracers + '\\alltracers_'+ run +'.npy',allow_pickle=True)
    tracers[:,:,1] = (tracers[:,:,1])/px_x
    tracers[:,:,2] = (tracers[:,:,2])/px_y
    axs[5].plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
for i in range(len(report_modes_q10_1)):    
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[5].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3)) 
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[3], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[4], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
cbaxes = inset_axes(axs[5], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,15])
axs[5].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[3].set_yticks(ticks=y_ticks, labels=y_labels)
axs[4].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_yticks(ticks=y_ticks, labels=y_labels)
axs[5].set_xlabel('Length [m]',fontsize = 'large')
axs[0].set_ylabel('Width [m]',fontsize = 'large')
axs[1].set_ylabel('Width [m]',fontsize = 'large')
axs[2].set_ylabel('Width [m]',fontsize = 'large')
axs[3].set_ylabel('Width [m]',fontsize = 'large')
axs[4].set_ylabel('Width [m]',fontsize = 'large')
axs[5].set_ylabel('Width [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q10_DoD.png'), dpi=300)


DEM0 = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_q10_1s0.txt'),skiprows=8) 
DEM1 = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_q10_1s1.txt'),skiprows=8) 
array_mask = np.loadtxt(os.path.join(w_dir, 'array_mask.txt'))
array_mask = np.where(array_mask != -999,1,np.nan)
DEM0 = DEM0*array_mask
DEM1 = DEM1*array_mask
fig, axs = plt.subplots(3,sharex=True,figsize = (16.5,5), tight_layout=True)   
im = axs[0].imshow(DEM0, cmap='BrBG_r',  vmin=-15, vmax=15, aspect='0.1')
axs[0].set_title('DEM 0 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,0,15])
im = axs[1].imshow(DEM1, cmap='BrBG_r',  vmin=-15, vmax=15, aspect='0.1')
axs[1].set_title('DEM 1 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')  
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-15,0,15])
im = axs[2].imshow(DoD_q10_1, cmap='RdBu',  vmin=-10, vmax=10, aspect='0.1')
axs[2].set_title('DoD 1-0 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-10,0,10])
axs[2].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_xlabel('Coordinata longitudinale x [m]',fontsize = 'large')
axs[0].set_ylabel('Larghezza B [m]',fontsize = 'large')
axs[1].set_ylabel('Larghezza B [m]',fontsize = 'large')
axs[2].set_ylabel('Larghezza B [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q10_DoD_test.png'), dpi=300,bbox_inches='tight')


DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_1-0_filt_fill.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-1_filt_fill.txt'))
DoD_q10_3 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-0_filt_fill.txt'))
  
fig, axs = plt.subplots(3,sharex=True,figsize = (16.5,5), tight_layout=True)   
im = axs[0].imshow(DoD_q10_1, cmap='RdBu',  vmin=-10, vmax=10, aspect='0.1')
axs[0].set_title('DoD 1-0 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')
cbaxes = inset_axes(axs[0], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-10,0,10])
im = axs[1].imshow(DoD_q10_2, cmap='RdBu',  vmin=-10, vmax=10, aspect='0.1')
axs[1].set_title('DoD 2-1 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')  
cbaxes = inset_axes(axs[1], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-10,0,10])
im = axs[2].imshow(DoD_q10_3, cmap='RdBu',  vmin=-10, vmax=10, aspect='0.1')
axs[2].set_title('DoD 2-0 [1 l/s]',fontsize='x-large',loc='center',fontweight='bold')
axs[0].add_patch(mpatches.Rectangle((150, 3),14, 140,edgecolor = 'black',lw=3,fill = False))  
axs[1].add_patch(mpatches.Rectangle((150, 3),14, 140,edgecolor = 'black',lw=3,fill = False))  
axs[2].add_patch(mpatches.Rectangle((150, 3),14, 140,edgecolor = 'black',lw=3,fill = False))  
cbaxes = inset_axes(axs[2], width="2%", height="80%") 
plt.colorbar(im, cax=cbaxes, ticks=[-10,0,10])
axs[2].set_xticks(ticks=x_ticks, labels=x_labels)
axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_yticks(ticks=y_ticks, labels=y_labels)
axs[2].set_xlabel('Coordinata longitudinale x [m]',fontsize = 'large')
axs[0].set_ylabel('Larghezza B [m]',fontsize = 'large')
axs[1].set_ylabel('Larghezza B [m]',fontsize = 'large')
axs[2].set_ylabel('Larghezza B [m]',fontsize = 'large')
plt.savefig(os.path.join(w_dir, 'q10_DoD_test2.png'), dpi=300,bbox_inches='tight')