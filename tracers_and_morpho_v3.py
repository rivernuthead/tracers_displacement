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
import seaborn as sns
import geopandas as gpd
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# from matplotlib.pyplot import cm


w_dir = os.getcwd() # Set Python script location as w_dir
morphodir = os.path.join(w_dir, 'output_data', 'morpho_overlap')

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
  
path_in_DoD_q05 = os.path.join(w_dir,'input_data','DoDs','DoD_q05_1')
path_in_DoD_q07 = os.path.join(w_dir,'input_data','DoDs','DoD_q07_1')
path_in_DoD_q10 = os.path.join(w_dir,'input_data','DoDs','DoD_q10_1')
    
DoD_q05_1 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_2-0_filt_fill.txt'))
DoD_q05_2 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-1_filt_fill.txt'))
DoD_q05_3 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_6-4_filt_fill.txt'))
DoD_q05_4 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_7-5_filt_fill.txt'))

DoD_q07_1 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_2-0_filt_fill.txt'))
DoD_q07_2 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_3-1_filt_fill.txt'))
DoD_q07_3 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_6-4_filt_fill.txt'))
DoD_q07_4 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_7-5_filt_fill.txt'))

DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_1-0_filt_fill.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-1_filt_fill.txt'))
DoD_q10_3 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_3-2_filt_fill.txt'))
DoD_q10_4 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_4-3_filt_fill.txt'))
DoD_q10_5 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_5-4_filt_fill.txt'))
DoD_q10_6 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_6-5_filt_fill.txt'))
  
        
x_ticks = [0, 10, 20, 30, 40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270]
x_labels = ['0', '0.5', '1.0','1.5','2.0','2.5','3.0', '3.5', '4.0','4.5','5.0','5.5', '6.0', '6.5','7.0','7.5','8.0', '8.5', '9.0','9.5','10.0','10.5', '11.0', '11.5','12.0','12.5','13.0','13.5'] 
y_ticks = [12, 72, 132]
y_labels = ['0','0.3','0.6'] 


# =============================================================================
# 
# =============================================================================
DoDs = [DoD_q05_1,DoD_q05_2,DoD_q05_3,DoD_q05_4]
k = 1
for j in range(1,5):
    fig, axs = plt.subplots(2,sharex=True,figsize = (11,3.5), tight_layout=True)     
    fig.subplots_adjust(hspace=0)
    axs[0].set_title('0.5 l/s-'+str(j),fontsize='x-large',loc='center',fontweight='bold')
    im = axs[1].imshow(DoDs[j-1], cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
    runs =[]
    borders=np.zeros((3,2))
    i = 0
    all_df_tracers = pd.DataFrame()
    for f in range(k,k+3):
        runs = np.append(runs, 'q05_1r'+str(f))        
    for run in runs:
        path_in_tracers = os.path.join(w_dir, 'output_data', 'output_images', run)
        run_param = np.loadtxt(os.path.join(w_dir,'input_data', 'run_param_'+run[0:3]+'.txt'), skiprows=1, delimiter=',')
        frame_position = run_param[int(run[6:])-1,1]
        frame_position += 0.44
        scale_to_mm = 0.0016*frame_position + 0.4827
        x_center = frame_position*1000 + 1100
        x_0 = x_center - 4288/2*scale_to_mm   
        feeding = gpd.read_file(os.path.join(path_in_tracers, "0s_tracers.shp"))
        feeding.drop(feeding.index[feeding.area == max(feeding.area)], inplace = True)
        feeding.drop(feeding.index[feeding.area < 4], inplace = True)
        feeding['Centroid'] = feeding.centroid
        feeding = feeding.set_geometry('Centroid')
        feeding = feeding.drop(columns=['geometry'])  
        feeding['x'] = feeding['Centroid'].x
        feeding['x'] = feeding['x'].mul(scale_to_mm)
        feeding['x']= feeding['x'].add(x_0)
        x_start = min(feeding.x)/px_x
        frame_reposition = run_param[int(run[6:])-1,5]
        frame_reposition += 0.44
        scale_to_mm_rep = 0.0016*frame_reposition + 0.4827
        x_center_rep = frame_reposition*1000 + 1100
        x_fin = x_center_rep - 4288/2*scale_to_mm_rep
        L = 4288*scale_to_mm_rep
        x_fin += L
        x_0 = x_0/px_x
        x_fin = x_fin/px_x
        borders[i,0] = x_0
        borders[i,1] = x_fin
        axs[1].axvline(x=x_start, color='black', linestyle='-')    
        
        tracers = np.load(os.path.join(path_in_tracers, 'alltracers_'+ run +'.npy'),allow_pickle=True)
        
        tracers[:,:,1] = (tracers[:,:,1])/px_x
        tracers[:,:,2] = (tracers[:,:,2])/px_y
        axs[1].plot(tracers[-1,:,1],tracers[-1,:,2],'.',color = 'g',markersize=2)
        dict_tracers = {'ntracc':tracers[-1,:,0],'x':tracers[-1,:,1],'y':tracers[-1,:,2],'z':tracers[-1,:,3],'z_DoD':tracers[-1,:,4]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        hue = np.array(df_tracers.z_DoD)
        hue = np.where(hue>0,1,hue)
        hue = np.where(hue<0,-1,hue)
        color_dict = dict({'Scour': 'red','Fill': 'dodgerblue','Not detected':'orange'})
        df_tracers['Legend'] = hue
        df_tracers['Legend'] = df_tracers['Legend'].replace([1], 'Deposition')
        df_tracers['Legend'] = df_tracers['Legend'].replace([-1], 'Scour')
        df_tracers['Legend'] = df_tracers['Legend'].replace([0], 'Not detected')
        all_df_tracers = pd.concat([all_df_tracers, df_tracers]) 
        i += 1
    k += 3
    all_df_tracers.reset_index(inplace = True)
    all_df_tracers.drop(columns=['index'],inplace = True)
    axs[1].add_patch(mpatches.Rectangle((0, 3),min(borders[:,0]), 140,facecolor = 'grey',lw=3,alpha = 0.7) )
    axs[1].add_patch(mpatches.Rectangle((max(borders[:,1]), 3),270 - max(borders[:,1]), 140,facecolor = 'grey',lw=3,alpha = 0.7))  
    if j == 3:
        axs[1].add_patch(mpatches.Rectangle((borders[0,1], 3),borders[2,0]-borders[0,1], 140,facecolor = 'grey',lw=3,alpha = 0.7))         
    sns.histplot(x=all_df_tracers.x, binwidth=2,binrange=(0,278),weights=all_df_tracers.ntracc,stat = 'percent',ax=axs[0],hue = all_df_tracers.Legend,palette = color_dict,multiple = "stack",hue_order =['Scour','Fill','Not detected'])
    sns.despine(ax=axs[0])
    
    cbar_ax = fig.add_axes([1, 0.20, 0.02, 0.2])  # Adjust the position and size as needed
    plt.colorbar(im, cax=cbar_ax, ticks=[vmin, 0, vmax], label='mm')
 
    
    # axs[0].set_yticks([])
    # axs[0].set_ylabel("")
    axs[0].set_xlim(0,278)
    axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    axs[1].set_xlabel('Longitudinal coordinate [m]',fontsize = 'large')
    axs[1].set_ylabel('Width [m]',fontsize = 'large')
    axs[0].set_ylabel('Number of tracers [%]',fontsize = 'large')
    plt.savefig(os.path.join(morphodir, 'all_tracers_q05_overlap_DoD_'+str(j)+'.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
# =============================================================================
# 
# =============================================================================
DoDs = [DoD_q07_1,DoD_q07_2,DoD_q07_3,DoD_q07_4]
k = 1
for j in range(1,5):
    fig, axs = plt.subplots(2,sharex=True,figsize = (11,3.5), tight_layout=True)    
    fig.subplots_adjust(hspace=0)
    axs[0].set_title('0.7 l/s-'+str(j),fontsize='x-large',loc='center',fontweight='bold')
    im = axs[1].imshow(DoDs[j-1], cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
    runs =[]
    borders=np.zeros((3,2))
    i = 0
    all_df_tracers = pd.DataFrame()
    for f in range(k,k+3):
        runs = np.append(runs, 'q07_1r'+str(f))        
    for run in runs:
        path_in_tracers = os.path.join(w_dir,'output_data', 'output_images', run)
        run_param = np.loadtxt(os.path.join(w_dir, 'input_data', 'run_param_'+run[0:3]+'.txt'), skiprows=1, delimiter=',')
        frame_position = run_param[int(run[6:])-1,1]
        frame_position += 0.44
        scale_to_mm = 0.0016*frame_position + 0.4827
        x_center = frame_position*1000 + 1100
        x_0 = x_center - 4288/2*scale_to_mm   
        feeding = gpd.read_file(os.path.join(path_in_tracers, "0s_tracers.shp"))
        feeding.drop(feeding.index[feeding.area == max(feeding.area)], inplace = True)
        feeding.drop(feeding.index[feeding.area < 4], inplace = True)
        feeding['Centroid'] = feeding.centroid
        feeding = feeding.set_geometry('Centroid')
        feeding = feeding.drop(columns=['geometry'])  
        feeding['x'] = feeding['Centroid'].x
        feeding['x'] = feeding['x'].mul(scale_to_mm)
        feeding['x']= feeding['x'].add(x_0)
        x_start = min(feeding.x)/px_x
        frame_reposition = run_param[int(run[6:])-1,5]
        frame_reposition += 0.44
        scale_to_mm_rep = 0.0016*frame_reposition + 0.4827
        x_center_rep = frame_reposition*1000 + 1100
        x_fin = x_center_rep - 4288/2*scale_to_mm_rep
        L = 4288*scale_to_mm_rep
        x_fin += L
        x_0 = x_0/px_x
        x_fin = x_fin/px_x
        borders[i,0] = x_0
        borders[i,1] = x_fin
        axs[1].axvline(x=x_start, color='black', linestyle='-')       
        tracers = np.load(os.path.join(path_in_tracers, 'alltracers_'+ run +'.npy'),allow_pickle=True)
        tracers[:,:,1] = (tracers[:,:,1])/px_x
        tracers[:,:,2] = (tracers[:,:,2])/px_y
        axs[1].plot(tracers[-1,:,1],tracers[-1,:,2],'.',color = 'g',markersize=2)
        dict_tracers = {'ntracc':tracers[-1,:,0],'x':tracers[-1,:,1],'y':tracers[-1,:,2],'z':tracers[-1,:,3],'z_DoD':tracers[-1,:,4]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        hue = np.array(df_tracers.z_DoD)
        hue = np.where(hue>0,1,hue)
        hue = np.where(hue<0,-1,hue)
        ccolor_dict = dict({'Scour': 'red','Fill': 'dodgerblue','Not detected':'orange'})
        df_tracers['Legend'] = hue
        df_tracers['Legend'] = df_tracers['Legend'].replace([1], 'Fill')
        df_tracers['Legend'] = df_tracers['Legend'].replace([-1], 'Scour')
        df_tracers['Legend'] = df_tracers['Legend'].replace([0], 'Not detected')
        all_df_tracers = pd.concat([all_df_tracers, df_tracers]) 
        i += 1
    k += 3
    all_df_tracers.reset_index(inplace = True)
    all_df_tracers.drop(columns=['index'],inplace = True)
    axs[1].add_patch(mpatches.Rectangle((0, 3),min(borders[:,0]), 140,facecolor = 'grey',lw=3,alpha = 0.7) )
    axs[1].add_patch(mpatches.Rectangle((max(borders[:,1]), 3),270 - max(borders[:,1]), 140,facecolor = 'grey',lw=3,alpha = 0.7))      
    if j == 2:
         axs[1].add_patch(mpatches.Rectangle((borders[0,1], 3),borders[1,0]-borders[0,1], 140,facecolor = 'grey',lw=3,alpha = 0.7))         
    sns.histplot(x=all_df_tracers.x, binwidth=2,binrange=(0,278),weights=all_df_tracers.ntracc,stat = 'percent',ax=axs[0],hue = all_df_tracers.Legend,palette = color_dict,multiple = "stack",hue_order =['Scour','Fill','Not detected'])
    sns.despine(ax=axs[0])
    cbar_ax = fig.add_axes([1, 0.20, 0.02, 0.2])  # Adjust the position and size as needed
    plt.colorbar(im, cax=cbar_ax, ticks=[vmin, 0, vmax], label='mm')
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    # axs[0].set_yticks([])
    # axs[0].set_ylabel("")  #TODO
    axs[0].set_xlim(0,278)
    axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xlabel('Longitudinal coordinate [m]',fontsize = 'large')
    axs[1].set_ylabel('Width [m]',fontsize = 'large')
    axs[0].set_ylabel('Number of tracers [%]',fontsize = 'large')
    plt.savefig(os.path.join(morphodir, 'all_tracers_q07_overlap_DoD_'+str(j)+'.png'), dpi=300,bbox_inches='tight')
    plt.show()


# =============================================================================
# 
# =============================================================================
x_ticks = [0, 10, 20, 30, 40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230]
x_labels = ['0', '0.5', '1.0','1.5','2.0','2.5','3.0', '3.5', '4.0','4.5','5.0','5.5', '6.0', '6.5','7.0','7.5','8.0', '8.5', '9.0','9.5','10.0','10.5', '11.0', '11.5'] 

DoDs = [DoD_q10_1,DoD_q10_2,DoD_q10_3,DoD_q10_4,DoD_q10_5,DoD_q10_6]
k = 1
for j in range(1,7):
    fig, axs = plt.subplots(2,sharex=True,figsize = (11,3.5), tight_layout=True)   
    fig.subplots_adjust(hspace=0)
    axs[0].set_title('1.0 l/s-'+str(j),fontsize='x-large',loc='center',fontweight='bold')
    im = axs[1].imshow(DoDs[j-1], cmap='RdBu',  vmin=vmin, vmax=vmax, aspect='0.1')
    runs =[]
    borders=np.zeros((2,2))
    i = 0
    all_df_tracers = pd.DataFrame()
    for f in range(k,k+2):
        runs = np.append(runs, 'q10_1r'+str(f))        
    for run in runs:
        path_in_tracers = os.path.join(w_dir, 'output_data', 'output_images', run)
        run_param = np.loadtxt(os.path.join(w_dir, 'input_data', 'run_param_'+run[0:3]+'.txt'), skiprows=1, delimiter=',')
        frame_position = run_param[int(run[6:])-1,1]
        frame_position += 0.44
        scale_to_mm = 0.0016*frame_position + 0.4827
        x_center = frame_position*1000 + 1100
        x_0 = x_center - 4288/2*scale_to_mm   
        feeding = gpd.read_file(os.path.join(w_dir, 'output_data', 'output_images', run, "0s_tracers.shp"))
        feeding.drop(feeding.index[feeding.area == max(feeding.area)], inplace = True)
        feeding.drop(feeding.index[feeding.area < 4], inplace = True)
        feeding['Centroid'] = feeding.centroid
        feeding = feeding.set_geometry('Centroid')
        feeding = feeding.drop(columns=['geometry'])  
        feeding['x'] = feeding['Centroid'].x
        feeding['x'] = feeding['x'].mul(scale_to_mm)
        feeding['x']= feeding['x'].add(x_0)
        x_start = min(feeding.x)/px_x
        frame_reposition = run_param[int(run[6:])-1,5]
        frame_reposition += 0.44
        scale_to_mm_rep = 0.0016*frame_reposition + 0.4827
        x_center_rep = frame_reposition*1000 + 1100
        x_fin = x_center_rep - 4288/2*scale_to_mm_rep
        L = 4288*scale_to_mm_rep
        x_fin += L
        x_0 = x_0/px_x
        x_fin = x_fin/px_x
        borders[i,0] = x_0
        borders[i,1] = x_fin
        axs[1].axvline(x=x_start, color='black', linestyle='-')       
        tracers = np.load(os.path.join(path_in_tracers, 'alltracers_'+ run +'.npy'),allow_pickle=True)
        tracers[:,:,1] = (tracers[:,:,1])/px_x
        tracers[:,:,2] = (tracers[:,:,2])/px_y
        axs[1].plot(tracers[-1,:,1],tracers[-1,:,2],'.',color = 'g',markersize=2)
        dict_tracers = {'ntracc':tracers[-1,:,0],'x':tracers[-1,:,1],'y':tracers[-1,:,2],'z':tracers[-1,:,3],'z_DoD':tracers[-1,:,4]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        hue = np.array(df_tracers.z_DoD)
        hue = np.where(hue>0,1,hue)
        hue = np.where(hue<0,-1,hue)
        color_dict = dict({'Scour': 'red','Fill': 'dodgerblue','Not detected':'orange'})
        df_tracers['Legend'] = hue
        df_tracers['Legend'] = df_tracers['Legend'].replace([1], 'Fill')
        df_tracers['Legend'] = df_tracers['Legend'].replace([-1], 'Scour')
        df_tracers['Legend'] = df_tracers['Legend'].replace([0], 'Not detected')
        all_df_tracers = pd.concat([all_df_tracers, df_tracers]) 
        i += 1
    k += 2
    all_df_tracers.reset_index(inplace = True)
    all_df_tracers.drop(columns=['index'],inplace = True)
    axs[1].add_patch(mpatches.Rectangle((0, 3),min(borders[:,0]), 140,facecolor = 'grey',lw=3,alpha = 0.7) )
    axs[1].add_patch(mpatches.Rectangle((max(borders[:,1]), 3),270 - max(borders[:,1]), 140,facecolor = 'grey',lw=3,alpha = 0.7))  
    if j == 2 or j ==3:
         axs[1].add_patch(mpatches.Rectangle((borders[0,1], 3),borders[1,0]-borders[0,1], 140,facecolor = 'grey',lw=3,alpha = 0.7))             
    sns.histplot(x=all_df_tracers.x, binwidth=2,binrange=(0,236),weights=all_df_tracers.ntracc,stat = 'percent',ax=axs[0],hue = all_df_tracers.Legend,palette = color_dict,multiple = "stack",hue_order =['Scour','Fill','Not detected'])
    sns.despine(ax=axs[0])
    cbar_ax = fig.add_axes([1, 0.20, 0.02, 0.2])  # Adjust the position and size as needed
    plt.colorbar(im, cax=cbar_ax, ticks=[vmin, 0, vmax], label='mm')
    axs[1].set_xticks(ticks=x_ticks, labels=x_labels)
    # axs[0].set_yticks([])
    # axs[0].set_ylabel("")
    axs[0].set_xlim(0,236)
    axs[1].set_yticks(ticks=y_ticks, labels=y_labels)
    axs[1].set_xlabel('Longitudinal coordinate [m]',fontsize = 'large')
    axs[1].set_ylabel('Width [m]',fontsize = 'large')
    axs[0].set_ylabel('Number of tracers [%]',fontsize = 'large')
    plt.savefig(os.path.join(morphodir, 'all_tracers_q10_overlap_DoD_'+str(j)+'.png'), dpi=300,bbox_inches='tight')
    plt.show()