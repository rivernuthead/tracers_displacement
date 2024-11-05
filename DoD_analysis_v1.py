# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:13:38 2022

@author: Marco
"""

# import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import matplotlib.patches as mpatches
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from scipy.ndimage import uniform_filter1d
# from matplotlib.pyplot import cm

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

w_dir = os.getcwd() # Set Python script location as w_dir
morphodir = os.path.join(w_dir,'output_data', 'morpho_overlap')
if not os.path.exists(morphodir):
    os.mkdir(morphodir)

def get_bins_from_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    bins_from_patches = np.unique([ax.patches[0].get_x()]+[bar.get_x()+bar.get_width() for bar in ax.patches])
    return bins_from_patches

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]
vmin = -10
vmax = 10
  
path_in_DoD_q05 = os.path.join(w_dir, 'input_data','DoDs','DoD_q05_1')
path_in_DoD_q07 = os.path.join(w_dir, 'input_data','DoDs','DoD_q07_1')
path_in_DoD_q10 = os.path.join(w_dir, 'input_data','DoDs','DoD_q10_1')
    
DoD_q05_1 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_2-0_filt_ult.txt'))
DoD_q05_2 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-1_filt_ult.txt'))
DoD_q05_3 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_6-4_filt_ult.txt'))
DoD_q05_4 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_7-5_filt_ult.txt'))

DoD_q07_1 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_2-0_filt_ult.txt'))
DoD_q07_2 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_3-1_filt_ult.txt'))
DoD_q07_3 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_6-4_filt_ult.txt'))
DoD_q07_4 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_7-5_filt_ult.txt'))

DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_1-0_filt_ult.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-1_filt_ult.txt'))
DoD_q10_3 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_3-2_filt_ult.txt'))
DoD_q10_4 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_4-3_filt_ult.txt'))
DoD_q10_5 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_5-4_filt_ult.txt'))
DoD_q10_6 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_6-5_filt_ult.txt'))
  
        
x_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
x_labels = ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5', '13.0', '13.5']
y_ticks = [12, 72, 132]
y_labels = ['0', '0.3', '0.6']

DoDs = [DoD_q05_1, DoD_q05_2, DoD_q05_3, DoD_q05_4]  # Assuming DoD_q05_x arrays are defined elsewhere

multiarray = np.zeros((278, 4))
V_array = np.zeros((4, 1))

for j in range(1, 5):
    fig, axs = plt.subplots(2, sharex=True, figsize=(22, 3.5), tight_layout=True)
    fig.subplots_adjust(hspace=0)
    
    # Plot the DoD data
    im = axs[0].imshow(DoDs[j-1], cmap='RdBu', vmin=vmin, vmax=vmax, aspect='0.1')
    
    # Calculate volume eroded
    DoD_er = np.where(DoDs[j-1] > 0, 0, DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er) == True, 0, DoD_er)
    V_e = -np.sum(DoD_er) * 50 * 5
    V_array[j-1] = V_e

    # Process deposition data and plot cleaned profile
    dep_array = np.empty((278, 1))
    DoD_dep = np.where(DoDs[j-1] < 0, 0, DoDs[j-1])
    DoD_dep = np.where(np.isnan(DoD_dep) == True, 0, DoD_dep)
    for i in range(278):
        dep_array[i] = np.count_nonzero(DoD_dep[:, i])
    mean = np.mean(dep_array)
    dep_array = dep_array - mean
    dep_array = np.where(dep_array < 0, 0, dep_array)
    cleaned_dep = uniform_filter1d(dep_array[:, 0], size=10)
    axs[1].plot(cleaned_dep)

    # Colorbar setup
    cbaxes = inset_axes(axs[0], width="2%", height="80%", loc='center right')
    plt.colorbar(im, cax=cbaxes, ticks=[vmin, 0, vmax], label='mm')

    # Set ticks and labels
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    axs[0].set_ylabel('Larghezza B [m]', fontsize='large')
    axs[0].set_title('DoD q05_' + str(j), fontsize='x-large', loc='center', fontweight='bold')
    axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xlabel('Coordinata longitudinale x [m]', fontsize='large')

    # Save the figure
    plt.savefig(os.path.join(morphodir, 'morph_DoD_05_' + str(j) + '.png'), dpi=300)
    plt.show()
    
    # Store cleaned deposition data
    multiarray[:, j-1] = cleaned_dep
    
np.savetxt(os.path.join(morphodir,  'report_morpho_q05.txt'), multiarray, fmt='%.2f', delimiter=',', header = 'DoD1,DoD2,DoD3,DoD4')    
V_e_q05 = V_array


multiarray = np.zeros((278,4))
V_array = np.zeros((4,1))
DoDs = [DoD_q07_1,DoD_q07_2,DoD_q07_3,DoD_q07_4]
for j in range(1,5):
    fig, axs = plt.subplots(2,sharex=True,figsize = (22,3.5), tight_layout=True)   
    fig.subplots_adjust(hspace=0)
    
    # Plot the DoD data
    im = axs[0].imshow(DoDs[j-1], cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
    
    # Calculate volume eroded
    DoD_er = np.where(DoDs[j-1]>0,0,DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er)==True,0,DoD_er)
    V_e = -np.sum(DoD_er)*50*5
    V_array[j-1] = V_e
    
    # Process deposition data and plot cleaned profile
    dep_array = np.empty((278,1))
    DoD_dep = np.where(DoDs[j-1]<0,0,DoDs[j-1])
    DoD_dep = np.where(np.isnan(DoD_dep)==True,0,DoD_dep)
    for i in range(278):    
        dep_array[i] = np.count_nonzero(DoD_dep[:,i])
    mean = np.mean(dep_array)
    dep_array = dep_array-mean
    dep_array = np.where(dep_array<0,0,dep_array)
    cleaned_dep = uniform_filter1d(dep_array[:,0],size = 10)
    axs[1].plot(cleaned_dep)
    
    # Colorbar setup
    cbaxes = inset_axes(axs[0], width="2%", height="80%", loc='center right') 
    plt.colorbar(im, cax=cbaxes, ticks=[vmin,0,vmax],label = 'mm')
    
    # Set ticks and labels
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    axs[0].set_ylabel('Larghezza B [m]',fontsize = 'large')
    axs[0].set_title('DoD q07_'+str(j),fontsize='x-large',loc='center',fontweight='bold')
    axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xlabel('Coordinata longitudinale x [m]',fontsize = 'large')
    
    # Save the figure
    plt.savefig(os.path.join(morphodir, 'morph_DoD_07_'+str(j)+'.png'), dpi=300)
    plt.show()
    
    # Store cleaned deposition data
    multiarray[:,j-1]=cleaned_dep
    
np.savetxt(os.path.join(morphodir,  'report_morpho_q07.txt'), multiarray, fmt='%.2f', delimiter=',', header = 'DoD1,DoD2,DoD3,DoD4')    
V_e_q07 = V_array

x_ticks = [0, 10, 20, 30, 40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230]
x_labels = ['0', '0.5', '1.0','1.5','2.0','2.5','3.0', '3.5', '4.0','4.5','5.0','5.5', '6.0', '6.5','7.0','7.5','8.0', '8.5', '9.0','9.5','10.0','10.5', '11.0', '11.5'] 

multiarray = np.zeros((236,6)) 
V_array = np.zeros((6,1))  
DoDs = [DoD_q10_1,DoD_q10_2,DoD_q10_3,DoD_q10_4,DoD_q10_5,DoD_q10_6]
for j in range(1,7):
    fig, axs = plt.subplots(2,sharex=True,figsize = (22,3.5), tight_layout=True)   
    fig.subplots_adjust(hspace=0)
    
    # Plot the DoD data
    im = axs[0].imshow(DoDs[j-1], cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
    
    # Calculate volume eroded
    DoD_er = np.where(DoDs[j-1]>0,0,DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er)==True,0,DoD_er)
    V_e = -np.sum(DoD_er)*50*5
    V_array[j-1] = V_e
    
    # Process deposition data and plot cleaned profile
    dep_array = np.empty((236,1))
    DoD_dep = np.where(DoDs[j-1]<0,0,DoDs[j-1])
    DoD_dep = np.where(np.isnan(DoD_dep)==True,0,DoD_dep)
    for i in range(236):    
        dep_array[i] = np.count_nonzero(DoD_dep[:,i])
    mean = np.mean(dep_array)
    dep_array = dep_array-mean
    dep_array = np.where(dep_array<0,0,dep_array)
    cleaned_dep = uniform_filter1d(dep_array[:,0],size = 10)
    axs[1].plot(cleaned_dep)
    
    # Colorbar setup
    cbaxes = inset_axes(axs[0], width="2%", height="80%", loc='center right') 
    plt.colorbar(im, cax=cbaxes, ticks=[vmin,0,vmax])
    
    # Set ticks and labels
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    # axs[1].set_yticks(ticks=[], ylabel="")
    axs[0].set_ylabel('Larghezza [m]',fontsize = 'large')
    axs[0].set_title('DoD q10_'+str(j),fontsize='x-large',loc='center',fontweight='bold')
    axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xlabel('Coordinata longitudinale [m]',fontsize = 'large')
    
    # Save the figure
    plt.savefig(os.path.join(morphodir, 'morph_DoD_10_'+str(j)+'.png'), dpi=300)
    plt.show()    
    multiarray[:,j-1]=cleaned_dep
    
np.savetxt(os.path.join(morphodir,  'report_morpho_q10.txt'), multiarray, fmt='%.2f', delimiter=',', header = 'DoD1,DoD2,DoD3,DoD4.DoD5,DoD6')    
V_e_q10 = V_array

# x_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270]
# x_labels = ['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5', '13.0', '13.5']
# y_ticks = [12, 72, 132]
# y_labels = ['0', '0.3', '0.6']

# path_in_DEM = os.path.join(w_dir, 'input_data', 'surveys', 'q05_1')
# DEM0 = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_q05_1s0.txt'), skiprows=8)
# DEM1 = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_q05_1s2.txt'), skiprows=8)
# array_mask = np.loadtxt(os.path.join(w_dir, 'input_data', 'array_mask.txt'))
# array_mask = np.where(array_mask != -999, 1, np.nan)
# DEM0 = DEM0 * array_mask
# DEM1 = DEM1 * array_mask

# fig, axs = plt.subplots(2, sharex=True, figsize=(16, 6))
# fig.subplots_adjust(hspace=0)

# # First subplot
# im = axs[0].imshow(DEM0, cmap='BrBG_r', vmin=-15, vmax=15, aspect='auto')
# cbaxes = inset_axes(axs[0], width="2%", height="90%", loc='center right')
# plt.colorbar(im, cax=cbaxes, ticks=[-15, 0, 15], label='[mm]')
# fig.canvas.draw()  # Initialize renderer

# # Second subplot
# im = axs[1].imshow(DoD_q05_1, cmap='RdBu', vmin=-10, vmax=10, aspect='auto')
# axs[1].set_title('Scavi e depositi', fontsize='xx-large', loc='center', fontweight='bold')
# cbaxes = inset_axes(axs[1], width="2%", height="90%", loc='center right')
# plt.colorbar(im, cax=cbaxes, ticks=[-10, 0, 10])
# fig.canvas.draw()  # Initialize renderer

# # Set ticks and labels
# axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
# axs[0].set_yticks(ticks=y_ticks, labels=y_labels)
# axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
# axs[0].set_title('Rilievo topografico', fontsize='xx-large', loc='center', fontweight='bold')
# axs[1].set_xlim(60, 160)
# axs[0].set_ylabel('Larghezza B [m]', fontsize='x-large')
# axs[1].set_ylabel('Larghezza B [m]', fontsize='x-large')
# axs[1].set_xlabel('Coordinata longitudinale x [m]', fontsize='x-large')

# plt.savefig(os.path.join(w_dir, 'q05_DoD_test.pdf'), dpi=300)
# plt.show()

# np.savetxt(os.path.join(morphodir, 'report_morpho_q05.txt'), multiarray, fmt='%.2f', delimiter=',', header='DoD1,DoD2,DoD3,DoD4')
