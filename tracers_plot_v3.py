# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:13:38 2022

@author: Marco
"""

# import necessary packages
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
# import shutil
import matplotlib.pyplot as plt
import seaborn as sns
# import scipy
# import statistics
from sklearn.mixture import GaussianMixture
import pandas as pd

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



n_bin = 22
# Set working directory
w_dir = os.getcwd() # Set Python script location as w_dir

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]


# List all available runs, depending on run_mode
# run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

run_names = ['q05_1r2']

###############################################################################
# LOOP OVER RUNS
###############################################################################
for run_name in run_names:
    path_in_tracers = os.path.join(w_dir, 'output_data','output_images', run_name)
    path_in_DEM = os.path.join(w_dir, 'input_data', 'surveys',run_name[0:5])
    path_in_DoD = os.path.join(w_dir, 'input_data', 'DoDs','DoD_'+run_name[0:5])
    # Create outputs script directory)
    if not os.path.exists(os.path.join(w_dir, 'output_data', 'analysis_plot')):
        os.mkdir(os.path.join(w_dir, 'output_data', 'analysis_plot'))
    if not os.path.exists(os.path.join(w_dir, 'output_data', 'appdisapp')):
        os.mkdir(os.path.join(w_dir, 'output_data', 'appdisapp'))
    plot_dir = os.path.join(w_dir, 'output_data', 'analysis_plot', run_name)
    plot_dir_appdisapp = os.path.join(w_dir, 'output_data', 'appdisapp', run_name)
   
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir) 
 
    if not os.path.exists(plot_dir_appdisapp):
        os.mkdir(plot_dir_appdisapp) 

        
    run_param = np.loadtxt(os.path.join(w_dir, 'input_data', 'run_param_'+run_name[0:3]+'.txt'), skiprows=1, delimiter=',')
        
    tracers = np.load(path_in_tracers + '/alltracers_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared = np.load(path_in_tracers + '/tracers_appeared_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared = np.load(path_in_tracers + '/tracers_disappeared_'+ run_name +'.npy',allow_pickle=True)
    tracers_reduced = np.load(path_in_tracers + '/tracers_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared_reduced = np.load(path_in_tracers + '/tracers_appeared_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_reduced = np.load(path_in_tracers + '/tracers_disappeared_reduced_'+ run_name +'.npy',allow_pickle=True) 
    tracers_appeared_stopped = np.load(path_in_tracers + '/tracers_appeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_stopped = np.load(path_in_tracers + '/tracers_disappeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True) 
    
    
    nDEM = int(run_param[int(run_name[6:])-1,2])
    nDoD1 = int(run_param[int(run_name[6:])-1,3])
    nDoD2 = int(run_param[int(run_name[6:])-1,4])
    start = int(run_param[int(run_name[6:])-1,6])
    yrami = int(run_param[int(run_name[6:])-1,7])
    nmode1 = int(run_param[int(run_name[6:])-1,8])
    nmode2 = int(run_param[int(run_name[6:])-1,9])
    end = int(run_param[int(run_name[6:])-1,11])
    
    DEM = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_'+run_name[0:3]+'_1s'+ str(nDEM) +'.txt'),skiprows=8)
    DoD = np.loadtxt(os.path.join(path_in_DoD, 'DoD_'+ str(nDoD1) + '-'+ str(nDoD2) + '_filt_fill.txt'))

    frame_position = run_param[int(run_name[6:])-1,1]
    frame_position += 0.44
    scale_to_mm = 0.0016*frame_position + 0.4827
    
    x_center = frame_position*1000 + 1100
    x_0 = x_center - 4288/2*scale_to_mm
    
    y_center = 51 + 622.5*scale_to_mm #51 è la distanza in mm dallo scan laser alla sponda interna
    y_0 = y_center - (2190-750)/2*scale_to_mm
    
    L = 4288*scale_to_mm # photo length in meters [m]
    
    i = 0
    
    for i in range(start,end,15):
        dict_tracers = {'x':tracers_reduced[i,:,0],'y':tracers_reduced[i,:,1],'z':tracers_reduced[i,:,2],'z_DoD':tracers_reduced[i,:,3]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        df_tracers = df_tracers.drop(df_tracers[df_tracers.x<100].index)
        df_tracers = df_tracers.reset_index(drop = True)
        if yrami != 0:
            df_tracers_ramo1 = df_tracers[df_tracers.y<yrami]
            df_tracers_ramo2 = df_tracers[df_tracers.y>yrami]
            if len(df_tracers_ramo1)>10 and nmode1 == 2:
                gmm = GaussianMixture(n_components=nmode1, tol = 1e-1, warm_start=(True), init_params='kmeans',covariance_type='tied')
                gmm.fit(df_tracers_ramo1.x.values.reshape(-1, 1))
                df_tracers_ramo1['target_class'] = gmm.predict(df_tracers_ramo1.x.values.reshape(-1, 1))
                moda1 = df_tracers_ramo1[df_tracers_ramo1.target_class==0].median().x
                moda2 = df_tracers_ramo1[df_tracers_ramo1.target_class==1].median().x
                if moda2 > moda1:
                    df_tracers_ramo1['ramo1_dist1'] = df_tracers_ramo1[df_tracers_ramo1.target_class==0].x
                    df_tracers_ramo1['ramo1_dist2'] = df_tracers_ramo1[df_tracers_ramo1.target_class==1].x
                else:
                    df_tracers_ramo1['ramo1_dist1'] = df_tracers_ramo1[df_tracers_ramo1.target_class==1].x
                    df_tracers_ramo1['ramo1_dist2'] = df_tracers_ramo1[df_tracers_ramo1.target_class==0].x 
            if len(df_tracers_ramo2)>10 and nmode2 == 2:
                gmm = GaussianMixture(n_components=nmode2, tol = 1e-1, warm_start=(False), init_params='kmeans',covariance_type='tied')
                gmm.fit(df_tracers_ramo2.x.values.reshape(-1, 1))
                df_tracers_ramo2['target_class'] = gmm.predict(df_tracers_ramo2.x.values.reshape(-1, 1))
                moda1 = df_tracers_ramo2[df_tracers_ramo2.target_class==0].median().x
                moda2 = df_tracers_ramo2[df_tracers_ramo2.target_class==1].median().x
                if moda2 > moda1:
                    df_tracers_ramo2['ramo2_dist1'] = df_tracers_ramo2[df_tracers_ramo2.target_class==0].x
                    df_tracers_ramo2['ramo2_dist2'] = df_tracers_ramo2[df_tracers_ramo2.target_class==1].x
                else:
                    df_tracers_ramo2['ramo2_dist1'] = df_tracers_ramo2[df_tracers_ramo2.target_class==1].x
                    df_tracers_ramo2['ramo2_dist2'] = df_tracers_ramo2[df_tracers_ramo2.target_class==0].x 
            f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            f.suptitle('Longitudinal distribution at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
            sns.histplot(data=df_tracers.x, ax=ax[0], binwidth=100,binrange=(0,2200),stat = 'density')
            ax[0].set_title('Before GMM', fontsize=16)
            ax[0].set(xlabel='Distance [mm]')
            if nmode1 == 1:
                sns.histplot(data=(df_tracers_ramo1.x), ax=ax[1], binwidth=100,binrange=(0,2200),stat = 'density')
            if len(df_tracers_ramo1)>15 and nmode1 == 2:
                sns.histplot(data=(df_tracers_ramo1.ramo1_dist1,df_tracers_ramo1.ramo1_dist2), ax=ax[1], binwidth=100,binrange=(0,2200),stat = 'density')
            ax[1].set_title('After GMM ramo 1', fontsize=16)
            ax[1].set(xlabel='Distance [mm]')
            if len(df_tracers_ramo2)>15 and nmode2 == 2:
                sns.histplot(data=(df_tracers_ramo2.ramo2_dist1,df_tracers_ramo2.ramo2_dist2), ax=ax[2], binwidth=100,binrange=(0,2200),stat = 'density')    
            if nmode2 == 1:
                sns.histplot(data=(df_tracers_ramo2.x), ax=ax[2], binwidth=100,binrange=(0,2200),stat = 'density')
            ax[2].set_title('After GMM ramo 2', fontsize=16)
            ax[2].set(xlabel='Distance [mm]')
            plt.savefig(os.path.join(plot_dir, run_name +' '+ str(float("{:.2f}".format(i*4/60))) +'min_gmm.png'), dpi=300)
            plt.show()
            
        if len(df_tracers)>15 and nmode1 > 1 and yrami == 0:
            gmm = GaussianMixture(n_components=nmode1, tol = 1e-1, warm_start=(False), init_params='kmeans',covariance_type='full')
            gmm.fit(df_tracers.x.values.reshape(-1, 1))
            df_tracers['target_class'] = gmm.predict(df_tracers.x.values.reshape(-1, 1))
            if nmode1 == 2:
                moda1 = df_tracers[df_tracers.target_class==0].median().x
                moda2 = df_tracers[df_tracers.target_class==1].median().x
                if moda2 > moda1:
                    df_tracers['dist1'] = df_tracers[df_tracers.target_class==0].x
                    df_tracers['dist2'] = df_tracers[df_tracers.target_class==1].x
                else:
                    df_tracers['dist1'] = df_tracers[df_tracers.target_class==1].x
                    df_tracers['dist2'] = df_tracers[df_tracers.target_class==0].x 
            if nmode1 == 3:
                dist3_df = pd.DataFrame()
                moda1 = df_tracers[df_tracers.target_class==0].median().x
                moda2 = df_tracers[df_tracers.target_class==1].median().x
                moda3 = df_tracers[df_tracers.target_class==2].median().x
                if moda3 > moda1 and moda3 > moda1:
                    df_tracers['dist3'] = df_tracers[df_tracers.target_class==2].x
                    if moda2 > moda1:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==0].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==1].x
                    else:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==1].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==0].x
                elif moda2 > moda1 and moda2 > moda3:
                    df_tracers['dist3'] = df_tracers[df_tracers.target_class==1].x
                    if moda3 > moda1:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==0].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==2].x
                    else:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==2].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==0].x
                else:
                    df_tracers['dist3'] = df_tracers[df_tracers.target_class==0].x
                    if moda3 > moda2:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==1].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==2].x
                    else:
                        df_tracers['dist1'] = df_tracers[df_tracers.target_class==2].x
                        df_tracers['dist2'] = df_tracers[df_tracers.target_class==1].x        
            f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            f.suptitle('Longitudinal distribution at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
            sns.histplot(data=df_tracers.x, ax=ax[0], binwidth=100,binrange=(0,2200),stat = 'density')
            ax[0].set_title('Before GMM', fontsize=16)
            ax[0].set(xlabel='Distance [mm]')
            if nmode1 == 1:
                sns.histplot(data=(df_tracers.x), ax=ax[1], binwidth=100,binrange=(0,2200),stat = 'density')
            if nmode1 == 2:
                sns.histplot(data=(df_tracers.dist1,df_tracers.dist2),ax=ax[1], binwidth=100, binrange=(0,2200),stat = 'density', common_norm = True)
            if nmode1 == 3:
                sns.histplot(data=(df_tracers.dist1,df_tracers.dist2,df_tracers.dist3),ax=ax[1], binwidth=50, binrange=(0,2200),stat = 'density', common_norm = True)
            ax[1].set_title('After GMM', fontsize=16)
            ax[1].set(xlabel='Distance [mm]')
            plt.savefig(os.path.join(plot_dir, run_name +' '+ str(float("{:.2f}".format(i*4/60))) +'_tiedcv_min_gmm.png'), dpi=300)
            plt.show() 
            
                
        sns.histplot(data=df_tracers.z, binwidth=3,binrange=(-15,15),stat = 'density')
        plt.xlabel('Elevation [mm]')
        plt.title('Elevation at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
        plt.savefig(os.path.join(plot_dir, run_name +' '+ str(float("{:.2f}".format(i*4/60))) +'min_z_distribution.png'), dpi=300)
        plt.show()
        
        sns.histplot(data=df_tracers.z_DoD, binwidth=3,binrange=(-15,15),stat = 'density')
        plt.xlabel('Scour and deposition [mm]')
        plt.title('Scour and deposition distribution at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
        plt.savefig(os.path.join(plot_dir, run_name +' '+ str(float("{:.2f}".format(i*4/60))) +'min_zDoD_distribution.png'), dpi=300)
        plt.show()
    
    all_df_tracers = pd.DataFrame()
    all_df_tracers_app = pd.DataFrame()
    all_df_tracers_app_stop = pd.DataFrame()
    all_df_tracers_dis = pd.DataFrame()
    all_df_tracers_dis_stop = pd.DataFrame()
    for i in range(1,len(tracers_appeared_reduced),1):
        dict_tracers = {'x':tracers_reduced[i,:,0],'y':tracers_reduced[i,:,1],'z':tracers_reduced[i,:,2],'z_DoD':tracers_reduced[i,:,3]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        df_tracers = df_tracers.drop(df_tracers[df_tracers.x<100].index)
        df_tracers = df_tracers.reset_index(drop = True)
        hue = np.array(df_tracers.z_DoD)
        hue = np.where(hue>0,1,hue)
        hue = np.where(hue<0,-1,hue)
        color_dict = dict({'erosion': 'red','deposition': 'dodgerblue','no_detection':'orange'})
        df_tracers['hue'] = hue
        df_tracers['hue'] = df_tracers['hue'].replace([1], 'deposition')
        df_tracers['hue'] = df_tracers['hue'].replace([-1], 'erosion')
        df_tracers['hue'] = df_tracers['hue'].replace([0], 'no_detection')
        all_df_tracers = pd.concat([all_df_tracers, df_tracers]) 
        dict_tracers_app_stop = {'x_app':tracers_appeared_stopped[i,:,0],'z_DoD_app':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        df_tracers_app_stop = df_tracers_app_stop.drop(df_tracers_app_stop[df_tracers_app_stop.x_app<100].index)
        df_tracers_app_stop = df_tracers_app_stop.reset_index(drop = True)
        all_df_tracers_app_stop = pd.concat([all_df_tracers_app_stop, df_tracers_app_stop])
        dict_tracers_dis_stop = {'x_disapp':tracers_disappeared_stopped[i,:,0],'z_DoD_dis':tracers_disappeared_stopped[i,:,3]}
        df_tracers_dis_stop = pd.DataFrame(dict_tracers_dis_stop)
        df_tracers_dis_stop = df_tracers_dis_stop.dropna()
        df_tracers_dis_stop = df_tracers_dis_stop.drop(df_tracers_dis_stop[df_tracers_dis_stop.x_disapp<100].index)
        df_tracers_dis_stop = df_tracers_dis_stop.reset_index(drop = True)
        all_df_tracers_dis_stop = pd.concat([all_df_tracers_dis_stop, df_tracers_dis_stop])
        # if len(df_tracers) > 5:
        #     f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6),sharey=True)
        #     sns.histplot(data=(df_tracers_app_stop.x_app,df_tracers_dis_stop.x_disapp,df_tracers.x),ax=ax[0], binwidth=100,binrange=(0,2200),stat = 'percent',fill = True,common_norm = False,discrete = False,kde=True,element="step")
        #     sns.histplot(x=df_tracers.x,ax=ax[1], binwidth=100,binrange=(0,2200),stat = 'percent',hue = df_tracers.hue,fill = True,discrete = False,palette = color_dict,multiple = 'stack')
        #     ax[0].text(0,57,'Total number of tracers: '+str(len(df_tracers)),fontweight='bold',fontsize = 'large',color = 'green')
        #     ax[0].text(0,53,'Number of appeared tracers: '+str(len(df_tracers_app_stop)),fontweight='bold',fontsize = 'large',color='cornflowerblue')
        #     ax[0].text(0,49,'Number of disappeared tracers: '+str(len(df_tracers_dis_stop)),fontweight='bold',fontsize = 'large',color='sandybrown')
        #     ax[0].set_ylim([0, 60])
        #     ax[0].set_xlabel('Distance [mm]')
        #     ax[1].set_xlabel('Distance [mm]')
        #     if i < start:
        #         f.suptitle('Appeared tracers at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16,color = 'r')
        #     else:
        #         f.suptitle('Appeared tracers at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
        #     plt.savefig(os.path.join(plot_dir_appdisapp, run_name +' '+ str(float("{:.2f}".format(i*4))) +'density_s_xappdisapp_distribution.png'), dpi=300)
        #     plt.show()
            





#%%
for run_name in run_names:
    path_in_tracers = os.path.join(w_dir, 'output_data','output_images', run_name)
    path_in_DEM = os.path.join(w_dir, 'input_data', 'surveys',run_name[0:5])
    path_in_DoD = os.path.join(w_dir, 'input_data', 'DoDs','DoD_'+run_name[0:5])
    # Create outputs script directory)
    if not os.path.exists(os.path.join(w_dir, 'output_data', 'analysis_plot')):
        os.mkdir(os.path.join(w_dir, 'output_data', 'analysis_plot'))
    if not os.path.exists(os.path.join(w_dir, 'output_data', 'appdisapp')):
        os.mkdir(os.path.join(w_dir, 'output_data', 'appdisapp'))
    plot_dir = os.path.join(w_dir, 'output_data', 'analysis_plot', run_name)
    plot_dir_appdisapp = os.path.join(w_dir, 'output_data', 'appdisapp', run_name)
   
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir) 
 
    if not os.path.exists(plot_dir_appdisapp):
        os.mkdir(plot_dir_appdisapp) 

        
    run_param = np.loadtxt(os.path.join(w_dir, 'input_data', 'run_param_'+run_name[0:3]+'.txt'), skiprows=1, delimiter=',')
        
    tracers = np.load(path_in_tracers + '/alltracers_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared = np.load(path_in_tracers + '/tracers_appeared_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared = np.load(path_in_tracers + '/tracers_disappeared_'+ run_name +'.npy',allow_pickle=True)
    tracers_reduced = np.load(path_in_tracers + '/tracers_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared_reduced = np.load(path_in_tracers + '/tracers_appeared_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_reduced = np.load(path_in_tracers + '/tracers_disappeared_reduced_'+ run_name +'.npy',allow_pickle=True) 
    tracers_appeared_stopped = np.load(path_in_tracers + '/tracers_appeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_stopped = np.load(path_in_tracers + '/tracers_disappeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True) 
    
    
    nDEM = int(run_param[int(run_name[6:])-1,2])
    nDoD1 = int(run_param[int(run_name[6:])-1,3])
    nDoD2 = int(run_param[int(run_name[6:])-1,4])
    start = int(run_param[int(run_name[6:])-1,6])
    yrami = int(run_param[int(run_name[6:])-1,7])
    nmode1 = int(run_param[int(run_name[6:])-1,8])
    nmode2 = int(run_param[int(run_name[6:])-1,9])
    end = int(run_param[int(run_name[6:])-1,11])
    
    DEM = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_'+run_name[0:3]+'_1s'+ str(nDEM) +'.txt'),skiprows=8)
    DoD = np.loadtxt(os.path.join(path_in_DoD, 'DoD_'+ str(nDoD1) + '-'+ str(nDoD2) + '_filt_fill.txt'))

    frame_position = run_param[int(run_name[6:])-1,1]
    frame_position += 0.44
    scale_to_mm = 0.0016*frame_position + 0.4827
    
    x_center = frame_position*1000 + 1100
    x_0 = x_center - 4288/2*scale_to_mm
    
    y_center = 51 + 622.5*scale_to_mm #51 è la distanza in mm dallo scan laser alla sponda interna
    y_0 = y_center - (2190-750)/2*scale_to_mm
    
    L = 4288*scale_to_mm # photo length in meters [m]
    
    i = 0
    
    for i in range(start,len(tracers_reduced)-1,1):
        dict_tracers_app = {'x_app':tracers_appeared_reduced[i,:,0],'z_DoD_app':tracers_appeared_reduced[i,:,3]}
        df_tracers_app = pd.DataFrame(dict_tracers_app)
        df_tracers_app = df_tracers_app.dropna()
        all_df_tracers_app = pd.concat([all_df_tracers_app, df_tracers_app])
        dict_tracers_app_stop = {'x_app_stop':tracers_appeared_stopped[i,:,0],'z_DoD_app_stop':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        all_df_tracers_app_stop = pd.concat([all_df_tracers_app_stop, df_tracers_app_stop])
        dict_tracers_dis = {'x_disapp':tracers_disappeared_reduced[i,:,0],'z_DoD_disapp':tracers_disappeared_reduced[i,:,3]}
        df_tracers_dis = pd.DataFrame(dict_tracers_dis)
        df_tracers_dis = df_tracers_dis.dropna()
        all_df_tracers_dis = pd.concat([all_df_tracers_dis, df_tracers_dis])
        dict_tracers_dis_stop = {'x_disapp_stop':tracers_disappeared_stopped[i,:,0],'z_DoD_dis_stop':tracers_disappeared_stopped[i,:,3]}
        df_tracers_dis_stop = pd.DataFrame(dict_tracers_dis_stop)
        df_tracers_dis_stop = df_tracers_dis_stop.dropna()
        all_df_tracers_dis_stop = pd.concat([all_df_tracers_dis_stop, df_tracers_dis_stop])
        if i == end-2:
            df_tracers_end = df_tracers
        # if len(df_tracers_app_stop)>0 and len(df_tracers_dis_stop)>0:
        #     f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6),sharey=True)
        #     sns.histplot(data=(df_tracers_app.x_app,df_tracers_app_stop.x_app_stop),ax=ax[0], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False)
        #     sns.histplot(data=(df_tracers_dis.x_disapp,df_tracers_dis_stop.x_disapp_stop),ax=ax[1], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False)
        #     ax[0].set_xlabel('Length x [mm]')
        #     ax[1].set_xlabel('Length x [mm]')
        #     if i < start:
        #         f.suptitle('Appeared tracers at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16,color = 'r')
        #     else:
        #         f.suptitle('Appeared tracers at '+str(float("{:.2f}".format(i*4/60))) +' minutes', fontsize=16)
        #     plt.show()
    
            
    all_df_tracers_app = all_df_tracers_app.reset_index()
    all_df_tracers_app_stop = all_df_tracers_app_stop.reset_index()
    all_df_tracers_dis = all_df_tracers_dis.reset_index()
    all_df_tracers_dis_stop = all_df_tracers_dis_stop.reset_index()
    all_df_tracers = all_df_tracers.reset_index()
    
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6),sharey=True)
    sns.histplot(data=(all_df_tracers_app.x_app,all_df_tracers_app_stop.x_app_stop),ax=ax[0], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False)
    sns.histplot(data=(all_df_tracers_dis.x_disapp,all_df_tracers_dis_stop.x_disapp_stop),ax=ax[1], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False)
    ax[0].set_title('Envelope of appeared tracers', fontsize=16)
    ax[1].set_title('Envelope of disappeared tracer', fontsize=16)
    ax[0].set_xlabel('Distance [mm]')
    ax[1].set_xlabel('Distance [mm]')
    plt.savefig(os.path.join(plot_dir, 'stopped_vs_nonstopped.png'), dpi=300) 
    plt.show()
    
    all_df_tracers_app_stop = pd.DataFrame()
    for i in range(start,end-1,1):
        dict_tracers_app_stop = {'x_app_stop':tracers_appeared_stopped[i,:,0],'z_DoD_app_stop':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        all_df_tracers_app_stop = pd.concat([all_df_tracers_app_stop, df_tracers_app_stop])
    
    
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 6),sharey=True)
    f.suptitle('Envelope of appeared tracers vs the total distribution', fontsize=16,color = 'k')
    sns.histplot(x=all_df_tracers_app_stop.x_app_stop,ax=ax[0], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False,stat ="percent")
    sns.histplot(x=df_tracers_end.x,ax=ax[1], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False,stat ="percent")
    sns.histplot(x=df_tracers.x,ax=ax[2], binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False,stat ="percent")
    ax[0].set_title('Envelope of appeared tracers', fontsize=16)
    ax[1].set_title('Distribution of the end frame', fontsize=16)
    ax[2].set_title('Distribution of the last frame', fontsize=16)
    ax[0].set_xlabel('Distance [mm]')
    ax[1].set_xlabel('Distance [mm]')
    ax[2].set_xlabel('Distance [mm]')
    plt.savefig(os.path.join(plot_dir, 'Inviluppo.png'), dpi=300)        
    plt.show()

    envelope_tracers_app = pd.DataFrame()
    for i in range(start,len(tracers_appeared_reduced)-1,1):
        dict_tracers_app_stop = {'x_app_stop':tracers_appeared_stopped[i,:,0],'z_DoD_app_stop':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        envelope_tracers_app = pd.concat([envelope_tracers_app, df_tracers_app_stop])
        if len(df_tracers_app_stop)>0 and i%15 == 0:
            sns.histplot(x=(all_df_tracers_app_stop.x_app_stop), binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False,stat ="percent")
            sns.histplot(x=(envelope_tracers_app.x_app_stop), binwidth=100,binrange=(0,2200),fill = True,common_norm = False,element = "step",discrete = False,stat ="percent")
            plt.xlabel('Distance [mm]')
            plt.title('Envelope of appeared tracers at '+str(float("{:.2f}".format(i*4/60))) +' minutes vs the total envelope', fontsize=16)
            plt.savefig(os.path.join(plot_dir, run_name +' '+ str(float("{:.2f}".format(i*4/60))) +'envelope of appeared vs totale envelope.png'), dpi=300)
            plt.show()
            envelope_tracers_app = pd.DataFrame()
                    
        
    tracers[:,:,1] = (tracers[:,:,1]-x_0)/px_x+5
    tracers[:,:,2] = (tracers[:,:,2]-y_0)/px_y
    tracers_appeared[:,:,1] = tracers_appeared[:,:,1]/px_x 
    tracers_appeared[:,:,2] = tracers_appeared[:,:,2]/px_y
    
    tracers_disappeared[:,:,1] = tracers_disappeared[:,:,1]/px_x
    tracers_disappeared[:,:,2] = tracers_disappeared[:,:,2]/px_y
    
    array_mask = np.loadtxt(os.path.join(w_dir, 'input_data', 'array_mask.txt'))
    array_mask = np.where(array_mask != -999,1,np.nan)
    if run_name == 'q10_1r8' or run_name == 'q10_1r9':
        array_mask = np.loadtxt(os.path.join(w_dir, 'input_data', 'array_mask_reduced.txt'))
        array_mask = np.where(array_mask != -999,1,np.nan)
    DEM = DEM*array_mask
    
    initial = int(x_0/px_x)-5
    
    final = initial + int(L/px_x)+5
    
    DEM = DEM[:,initial:final]
    DoD = DoD[:,initial:final]
    
    x_ticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    x_labels = ['0','0.25', '0.5', '0.75', '1.0', '1.25','1.5','1.75','2.0','2.25'] 
    y_ticks = [12, 72, 132]
    y_labels = ['0','0.3','0.6'] 
    
    
    fig, ax = plt.subplots(dpi=200, tight_layout=True)
    plt.plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
    im = ax.imshow(DoD, cmap='RdBu',  vmin=-15, vmax=15, aspect='0.1')
    plt.colorbar(im, shrink = 0.2, aspect = 3)
    plt.title('Tracers overlapped with DoD at ' + str(float("{:.2f}".format((len(tracers)-1)*4/60)))+ ' minutes', fontweight='bold')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.xlabel('Length [m]')
    plt.ylabel('Width [m]')
    plt.savefig(os.path.join(plot_dir, run_name +'_overlap_DoD.png'), dpi=1600)
    plt.savefig(os.path.join(plot_dir, run_name +'_overlap_DoD.pdf'))
    plt.show()
    
    
    fig, ax = plt.subplots(dpi=200, tight_layout=True)
    plt.plot(tracers[-2,:,1],tracers[-2,:,2],'.',color = 'g')
    im = ax.imshow(DEM, cmap='BrBG_r',  vmin=-15, vmax=15, aspect='0.1')
    plt.colorbar(im, shrink = 0.2, aspect = 3)
    plt.title('Tracers overlapped with DEM at ' + str(float("{:.2f}".format((len(tracers)-1)*4/60)))+ ' minutes', fontweight='bold')
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.yticks(ticks=y_ticks, labels=y_labels)
    plt.xlabel('Length [m]')
    plt.ylabel('Width [m]')
    plt.savefig(os.path.join(plot_dir, run_name +'_overlap_DEM.png'), dpi=1600)
    plt.savefig(os.path.join(plot_dir, run_name +'_overlap_DEM.pdf'))
    plt.show()
      