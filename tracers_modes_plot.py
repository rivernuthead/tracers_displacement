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

# Script parameters:
w_dir = os.getcwd() # Set Python script location as w_dir

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

path_in_DoD_q05 = os.path.join(w_dir,'DoDs','DoD_q05_1')
path_in_DoD_q07 = os.path.join(w_dir,'DoDs','DoD_q07_1')
path_in_DoD_q10 = os.path.join(w_dir,'DoDs','DoD_q10_1')

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
     
DoD_q05_1 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-0_filt_fill.txt'))
DoD_q05_2 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_7-4_filt_fill.txt'))
DoD_q07_1 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_3-0_filt_fill.txt'))
DoD_q07_2 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_7-4_filt_fill.txt'))
DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_3-0_filt_fill.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_6-3_filt_fill.txt'))
 
    
    
x_ticks = [0, 10, 20, 30, 40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230]
x_labels = ['0', '0.5', '1.0','1.5','2.0','2.5','3.0', '3.5', '4.0','4.5','5.0','5.5', '6.0', '6.5','7.0','7.5','8.0', '8.5', '9.0','9.5','10.0','10.5', '11.0', '11.5'] 
y_ticks = [12, 72, 132]
y_labels = ['0','0.3','0.6'] 
    
fig, axs = plt.subplots(6,sharex=True,figsize = (16.5,11.7), tight_layout=True)   
im = axs[0].imshow(DoD_q05_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[0].set_title('Mode q05',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q05))))  
for i in range(len(report_modes_q05_1)):    
    c = next(color)
    line = report_modes_q05_1[report_modes_q05_1.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[0].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        axs[0].axvline(x=moda1, color=c, linestyle='-')
        axs[0].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        axs[0].axvline(x=moda2, color=c, linestyle='-')
        axs[0].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        axs[0].axvline(x=moda3, color=c, linestyle='-')
        axs[0].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
im = axs[1].imshow(DoD_q05_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[1].set_title('Mode q05',fontsize='x-large',loc='center',fontweight='bold')
for i in range(len(report_modes_q05_2)):
    c = next(color)
    line = report_modes_q05_2[report_modes_q05_2.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[1].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        axs[1].axvline(x=moda1, color=c, linestyle='-')
        axs[1].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        axs[1].axvline(x=moda2, color=c, linestyle='-')
        axs[1].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        axs[1].axvline(x=moda3, color=c, linestyle='-')
        axs[1].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
im = axs[2].imshow(DoD_q07_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[2].set_title('Mode q07',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q07))))
for i in range(len(report_modes_q07_1)):
    c = next(color)
    line = report_modes_q07_1[report_modes_q07_1.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[2].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        axs[2].axvline(x=moda1, color=c, linestyle='-')
        axs[2].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        axs[2].axvline(x=moda2, color=c, linestyle='-')
        axs[2].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        axs[2].axvline(x=moda3, color=c, linestyle='-')
        axs[2].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
im = axs[3].imshow(DoD_q07_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[3].set_title('Mode q07',fontsize='x-large',loc='center',fontweight='bold')
for i in range(len(report_modes_q07_2)):
    c = next(color)
    line = report_modes_q07_2[report_modes_q07_2.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[3].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        axs[3].axvline(x=moda1, color=c, linestyle='-')
        axs[3].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        axs[3].axvline(x=moda2, color=c, linestyle='-')
        axs[3].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        axs[3].axvline(x=moda3, color=c, linestyle='-')
        axs[3].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
im = axs[4].imshow(DoD_q10_1, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[4].set_title('Mode q10',fontsize='x-large',loc='center',fontweight='bold')
color = iter(cm.rainbow(np.linspace(0, 1, len(report_modes_q10))))
for i in range(len(report_modes_q10_1)):
    c = next(color)
    line = report_modes_q10_1[report_modes_q10_1.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[4].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        axs[4].axvline(x=moda1, color=c, linestyle='-')
        axs[4].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        axs[4].axvline(x=moda2, color=c, linestyle='-')
        axs[4].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        axs[4].axvline(x=moda3, color=c, linestyle='-')
        axs[4].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')    
im = axs[5].imshow(DoD_q10_2, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
axs[5].set_title('Mode q10',fontsize='x-large',loc='center',fontweight='bold')
for i in range(len(report_modes_q10_2)):
    c = next(color)
    line = report_modes_q10_2[report_modes_q10_2.index == i]
    moda1 = line.iloc[0]['moda_1_g_DoD']
    moda2 = line.iloc[0]['moda_2_g_DoD']
    moda3 = line.iloc[0]['moda_3_g_DoD']
    origin = line.iloc[0]['x_0_DoD']
    width = line.iloc[0]['L_DoD']
    axs[5].add_patch(Rectangle((origin-1, 3), width, 140, edgecolor = c,fill=False,lw=3))  
    if moda1 != 0:
        line1=axs[5].axvline(x=moda1, color=c, linestyle='-')
        axs[5].text(moda1-1,50,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda2 != 0:
        line2=axs[5].axvline(x=moda2, color=c, linestyle='-')
        axs[5].text(moda2-1,80,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
    if moda3 != 0:
        line3=axs[5].axvline(x=moda3, color=c, linestyle='-')
        axs[5].text(moda3-1,110,str(int(line.iloc[0]['prova'])),fontweight='bold',fontsize = 'large')
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
# axs[0].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.9, 1))
# axs[1].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.9, 1))
# axs[2].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.97, 1))
# axs[3].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.9, 1))
# axs[4].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.97, 1))
# axs[5].legend([line1, line2, line3], ['moda 1', 'moda 2', 'moda 3'],loc='best', bbox_to_anchor=(0.97, 1))
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
plt.savefig(os.path.join(w_dir, 'all_modes_overlap_DoD.png'), dpi=300)

