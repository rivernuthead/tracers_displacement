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
# import seaborn as sns
# import scipy
from scipy.ndimage import uniform_filter1d
import pandas as pd
from sklearn.mixture import GaussianMixture

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

w_dir = os.getcwd() # Set Python script location as w_dir

# Script parameters:
clean_size = 5


# run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# run_names = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# run_names = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

'''
TO PRINT THE report_distances.txt FILE RUN ALL THE run_name TOGETHER
'''
run_names = ['q05_1r1', 'q05_1r2','q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12',
             'q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12',
             'q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']



report_distances = np.empty((0,12)) # distance between modes all distribution from start to end
###############################################################################
# LOOP OVER RUNS
###############################################################################
for run_name in run_names:
    print(run_name, ' is running...')
    
    path_in_tracers = os.path.join(w_dir, 'output_data', 'output_images', run_name)
    
    if not os.path.exists(os.path.join(w_dir,'output_data', 'distances')):
        os.mkdir(os.path.join(w_dir,'output_data', 'distances'))
        
    plot_dir = os.path.join(w_dir,'output_data', 'distances', 'plot')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        
    report_dir = os.path.join(w_dir,'output_data', 'distances', 'report')
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)
        
    # Create outputs script directory
    if not os.path.exists(os.path.join(w_dir, 'output_data', 'output_tracers_analysis')):
        os.mkdir(os.path.join(w_dir, 'output_data', 'output_tracers_analysis'))
                 
                 
    path_out = os.path.join(w_dir, 'output_data', 'output_tracers_analysis', run_name[0:5])
    if not os.path.exists(path_out):
        os.mkdir(path_out)
     
    run_param = np.loadtxt(os.path.join(w_dir, 'input_data', 'run_param_'+run_name[0:3]+'.txt'), skiprows=1, delimiter=',')
    
    tracers = np.load(path_in_tracers + '/alltracers_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared = np.load(path_in_tracers + '/tracers_appeared_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared = np.load(path_in_tracers + '/tracers_disappeared_'+ run_name +'.npy',allow_pickle=True)
    
    tracers_reduced = np.load(path_in_tracers + '/tracers_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_appeared_reduced = np.load(path_in_tracers + '/tracers_appeared_reduced_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_reduced = np.load(path_in_tracers + '/tracers_disappeared_reduced_'+ run_name +'.npy',allow_pickle=True)
    
    tracers_appeared_stopped = np.load(path_in_tracers + '/tracers_appeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True)
    tracers_disappeared_stopped = np.load(path_in_tracers + '/tracers_disappeared_reduced_stopped_'+ run_name +'.npy',allow_pickle=True) 
    
    
    start = int(run_param[int(run_name[6:])-1,6])
    yrami = int(run_param[int(run_name[6:])-1,7])
    nmode1 = int(run_param[int(run_name[6:])-1,8])
    nmode2 = int(run_param[int(run_name[6:])-1,9])
    skip = int(run_param[int(run_name[6:])-1,10])
    end = int(run_param[int(run_name[6:])-1,11])
    
    time = []
    for i in range(0,len(tracers_reduced),1):    
        time = np.append(time,i*4/60)
    
    

    if skip == 1 and yrami == 0:
        continue
    
    report_tracers = np.zeros((len(tracers_reduced),8))
    report_modes = np.zeros((end-start,14))
    
    for i in range(start,len(tracers_reduced),1):
        dict_tracers = {'x':tracers_reduced[i,:,0],'y':tracers_reduced[i,:,1],'z':tracers_reduced[i,:,2],'z_DoD':tracers_reduced[i,:,3]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        #df_tracers = df_tracers.drop(df_tracers[df_tracers.x<100].index)
        df_tracers = df_tracers.reset_index(drop = True)
        
        report_tracers[i,0] = len(df_tracers)
        report_tracers[i,2] = df_tracers.mean().x
        report_tracers[i,4] = len(df_tracers[df_tracers.z_DoD<0])/len(df_tracers)
        report_tracers[i,6] = len(df_tracers[df_tracers.z_DoD>0])/len(df_tracers)
        
        if yrami != 0:
            df_tracers_ramo1 = df_tracers[df_tracers.y<yrami]
            report_tracers[i,0] = len(df_tracers_ramo1)
            report_tracers[i,2] = df_tracers_ramo1.mean().x
            report_tracers[i,4] = len(df_tracers_ramo1[df_tracers_ramo1.z_DoD<0])/len(df_tracers_ramo1)
            report_tracers[i,6] = len(df_tracers_ramo1[df_tracers_ramo1.z_DoD>0])/len(df_tracers_ramo1)
            df_tracers_ramo2 = df_tracers[df_tracers.y>yrami]
            report_tracers[i,1] = len(df_tracers_ramo2)
            report_tracers[i,3] = df_tracers_ramo2.mean().x
            report_tracers[i,5] = len(df_tracers_ramo2[df_tracers_ramo2.z_DoD<0])/len(df_tracers_ramo2)
            report_tracers[i,7] = len(df_tracers_ramo2[df_tracers_ramo2.z_DoD>0])/len(df_tracers_ramo2)
            
                    
    report_tracers[:,0] = np.where(np.isnan(report_tracers[:,0])==True,0,report_tracers[:,0])
    report_tracers[:,1] = np.where(np.isnan(report_tracers[:,1])==True,0,report_tracers[:,1])
    report_tracers[:,2] = np.where(np.isnan(report_tracers[:,2])==True,0,report_tracers[:,2])
    report_tracers[:,3] = np.where(np.isnan(report_tracers[:,3])==True,0,report_tracers[:,3])
    report_tracers[:,4] = np.where(np.isnan(report_tracers[:,4])==True,0,report_tracers[:,4])
    report_tracers[:,5] = np.where(np.isnan(report_tracers[:,5])==True,0,report_tracers[:,5])
    report_tracers[:,6] = np.where(np.isnan(report_tracers[:,6])==True,0,report_tracers[:,6])
    report_tracers[:,7] = np.where(np.isnan(report_tracers[:,7])==True,0,report_tracers[:,7])
    
    cleaned_ntracc = uniform_filter1d(report_tracers[:,0],size = clean_size)
    cleaned_ntracc_ramo2 = uniform_filter1d(report_tracers[:,1],size = clean_size)
    
    cleaned_mean = uniform_filter1d(report_tracers[:,2],size = clean_size)
    cleaned_mean_ramo2 = uniform_filter1d(report_tracers[:,3],size = clean_size)
    
    cleaned_erosione = uniform_filter1d(report_tracers[:,4],size = clean_size)
    cleaned_deposito = uniform_filter1d(report_tracers[:,6],size = clean_size)
    cleaned_erosione_ramo2 = uniform_filter1d(report_tracers[:,5],size = clean_size)
    cleaned_deposito_ramo2 = uniform_filter1d(report_tracers[:,7],size = clean_size)
    
    
    
    np.savetxt(os.path.join(path_out,  run_name +'_report_tracers.txt'), report_tracers, fmt='%.2f', delimiter=',', header = 'ntracc, ntracc ramo2, x media [mm], x media ramo2 [mm], ntracc erosione [%], ntracc erosione ramo 2 [%], n tracc deposito [%], n tracc deposito ramo 2 [%]')    
    
    
    ###################################################################################
    # Distance method 1 # distance between modes all distribution from start to end
    ###################################################################################
    for i in range(start,end,1):
        dict_tracers = {'x':tracers_reduced[i,:,0],'y':tracers_reduced[i,:,1],'z':tracers_reduced[i,:,2],'z_DoD':tracers_reduced[i,:,3]}
        df_tracers = pd.DataFrame(dict_tracers)
        df_tracers = df_tracers.dropna()
        #df_tracers = df_tracers.drop(df_tracers[df_tracers.x<100].index)
        df_tracers = df_tracers.reset_index(drop = True)
        if yrami == 0:
            if len(df_tracers) >= 20 and nmode1 == 1:
                report_modes[i-start,10] = df_tracers.mean().x
            if len(df_tracers) >= 20 and nmode1 > 1:
                gmm = GaussianMixture(n_components=nmode1, tol = 1e-1, warm_start=(True), init_params='kmeans',covariance_type='tied')
                gmm.fit(df_tracers.x.values.reshape(-1, 1))
                df_tracers['target_class'] = gmm.predict(df_tracers.x.values.reshape(-1, 1))
                if nmode1 == 2:
                    moda1 = df_tracers[df_tracers.target_class==0].median().x
                    moda2 = df_tracers[df_tracers.target_class==1].median().x
                    mode = [moda1,moda2]
                    ntracc = [len(df_tracers[df_tracers.target_class==0]),len(df_tracers[df_tracers.target_class==1])]
                    dict_mode = {'mode':mode,'ntracc':ntracc}
                    modedf = pd.DataFrame(dict_mode)
                    modedf = modedf.sort_values('mode')
                    report_modes[i-start,0] = modedf.iloc[0]['mode']
                    report_modes[i-start,1] = modedf.iloc[1]['mode']
                    report_modes[i-start,3] = modedf.iloc[0]['ntracc']/modedf.sum().ntracc
                    report_modes[i-start,4] = modedf.iloc[1]['ntracc']/modedf.sum().ntracc
                    #report_modes[i-start,10] = report_modes[i-start,0]*report_modes[i-start,3]+report_modes[i-start,1]*report_modes[i-start,4] 
                    report_modes[i-start,10] = df_tracers.mean().x
                    report_modes[i-start,12] = abs(report_modes[i-start,1]-report_modes[i-start,0])
                elif nmode1 == 3:
                    moda1 = df_tracers[df_tracers.target_class==0].median().x
                    moda2 = df_tracers[df_tracers.target_class==1].median().x
                    moda3 = df_tracers[df_tracers.target_class==2].median().x
                    ntracc = [len(df_tracers[df_tracers.target_class==0]),len(df_tracers[df_tracers.target_class==1]),len(df_tracers[df_tracers.target_class==2])]
                    mode = [moda1,moda2,moda3]
                    dict_mode = {'mode':mode,'ntracc':ntracc}
                    modedf = pd.DataFrame(dict_mode)
                    modedf = modedf.sort_values('mode')
                    report_modes[i-start,0] = modedf.iloc[0]['mode']
                    report_modes[i-start,1] = modedf.iloc[1]['mode']
                    report_modes[i-start,2] = modedf.iloc[2]['mode']
                    report_modes[i-start,3] = modedf.iloc[0]['ntracc']/modedf.sum().ntracc
                    report_modes[i-start,4] = modedf.iloc[1]['ntracc']/modedf.sum().ntracc
                    report_modes[i-start,5] = modedf.iloc[2]['ntracc']/modedf.sum().ntracc
                    #report_modes[i-start,10] = report_modes[i-start,0]*report_modes[i-start,3]+report_modes[i-start,1]*report_modes[i-start,4]+report_modes[i-start,2]*report_modes[i-start,5]
                    report_modes[i-start,10] = df_tracers.mean().x
        if yrami != 0:
            df_tracers_ramo1 = df_tracers[df_tracers.y<yrami]
            df_tracers_ramo2 = df_tracers[df_tracers.y>yrami]  
            if len(df_tracers_ramo1) >= 20 and nmode1 == 1:
                report_modes[i-start,10] = df_tracers_ramo1.mean().x
            if len(df_tracers_ramo2) >= 20 and nmode2 == 1:
                report_modes[i-start,11] = df_tracers_ramo2.mean().x
            if len(df_tracers_ramo1)>20 and nmode1 == 2 and skip != 1:
                gmm = GaussianMixture(n_components=nmode1, tol = 1e-1, warm_start=(False), init_params='kmeans',covariance_type='tied')
                gmm.fit(df_tracers_ramo1.x.values.reshape(-1, 1))
                df_tracers_ramo1['target_class'] = gmm.predict(df_tracers_ramo1.x.values.reshape(-1, 1))
                moda1 = df_tracers_ramo1[df_tracers_ramo1.target_class==0].median().x
                moda2 = df_tracers_ramo1[df_tracers_ramo1.target_class==1].median().x
                mode = [moda1,moda2]
                ntracc = [len(df_tracers_ramo1[df_tracers_ramo1.target_class==0]),len(df_tracers_ramo1[df_tracers_ramo1.target_class==1])]
                dict_mode = {'mode':mode,'ntracc':ntracc}
                modedf = pd.DataFrame(dict_mode)
                modedf = modedf.sort_values('mode')
                report_modes[i-start,0] = modedf.iloc[0]['mode']
                report_modes[i-start,1] = modedf.iloc[1]['mode']
                report_modes[i-start,3] = modedf.iloc[0]['ntracc']/modedf.sum().ntracc
                report_modes[i-start,4] = modedf.iloc[1]['ntracc']/modedf.sum().ntracc
                #report_modes[i-start,10] = report_modes[i-start,0]*report_modes[i-start,3]+report_modes[i-start,1]*report_modes[i-start,4]
                report_modes[i-start,10] = df_tracers_ramo1.mean().x
                report_modes[i-start,12] = abs(report_modes[i-start,1]-report_modes[i-start,0])    
            if len(df_tracers_ramo2)>20 and nmode2 == 2 and skip != 2:
                gmm = GaussianMixture(n_components=nmode2, tol = 1e-1, warm_start=(False), init_params='kmeans',covariance_type='tied')
                gmm.fit(df_tracers_ramo2.x.values.reshape(-1, 1))
                df_tracers_ramo2['target_class'] = gmm.predict(df_tracers_ramo2.x.values.reshape(-1, 1))
                moda1 = df_tracers_ramo2[df_tracers_ramo2.target_class==0].median().x
                moda2 = df_tracers_ramo2[df_tracers_ramo2.target_class==1].median().x
                mode = [moda1,moda2]
                ntracc = [len(df_tracers_ramo2[df_tracers_ramo2.target_class==0]),len(df_tracers_ramo2[df_tracers_ramo2.target_class==1])]
                dict_mode = {'mode':mode,'ntracc':ntracc}
                modedf = pd.DataFrame(dict_mode)
                modedf = modedf.sort_values('mode')
                report_modes[i-start,6] = modedf.iloc[0]['mode']
                report_modes[i-start,7] = modedf.iloc[1]['mode']
                report_modes[i-start,8] = modedf.iloc[0]['ntracc']/modedf.sum().ntracc
                report_modes[i-start,9] = modedf.iloc[1]['ntracc']/modedf.sum().ntracc
                #report_modes[i-start,11] = report_modes[i-start,6]*report_modes[i-start,8]+report_modes[i-start,7]*report_modes[i-start,9] 
                report_modes[i-start,11] = df_tracers_ramo2.mean().x
                report_modes[i-start,13] = abs(report_modes[i-start,7]-report_modes[i-start,6])
    
    report_modes[:,0] = np.where(np.isnan(report_modes[:,0])==True,0,report_modes[:,0])
    report_modes[:,1] = np.where(np.isnan(report_modes[:,1])==True,0,report_modes[:,1])
    report_modes[:,2] = np.where(np.isnan(report_modes[:,2])==True,0,report_modes[:,2])
    cleaned_moda1 = uniform_filter1d(report_modes[:,0],size = clean_size)
    cleaned_moda2 = uniform_filter1d(report_modes[:,1],size = clean_size)
    cleaned_moda3 = uniform_filter1d(report_modes[:,2],size = clean_size)
    
    report_modes[:,3] = np.where(np.isnan(report_modes[:,3])==True,0,report_modes[:,3])
    report_modes[:,4] = np.where(np.isnan(report_modes[:,4])==True,0,report_modes[:,4])
    report_modes[:,5] = np.where(np.isnan(report_modes[:,5])==True,0,report_modes[:,5])
    
    cleaned_ntracc1 = uniform_filter1d(report_modes[:,3],size = clean_size)
    cleaned_ntracc2 = uniform_filter1d(report_modes[:,4],size = clean_size)
    cleaned_ntracc3 = uniform_filter1d(report_modes[:,5],size = clean_size)
    
    report_modes[:,6] = np.where(np.isnan(report_modes[:,6])==True,0,report_modes[:,6])
    report_modes[:,7] = np.where(np.isnan(report_modes[:,7])==True,0,report_modes[:,7])
    
    cleaned_moda1_ramo2 = uniform_filter1d(report_modes[:,6],size = clean_size)
    cleaned_moda2_ramo2 = uniform_filter1d(report_modes[:,7],size = clean_size)
    
    report_modes[:,8] = np.where(np.isnan(report_modes[:,8])==True,0,report_modes[:,8])
    report_modes[:,9] = np.where(np.isnan(report_modes[:,9])==True,0,report_modes[:,9])
    
    cleaned_ntracc1_ramo2 = uniform_filter1d(report_modes[:,8],size = clean_size)
    cleaned_ntracc2_ramo2 = uniform_filter1d(report_modes[:,9],size = clean_size)
    
    report_modes[:,10] = np.where(np.isnan(report_modes[:,10])==True,0,report_modes[:,10])
    report_modes[:,11] = np.where(np.isnan(report_modes[:,11])==True,0,report_modes[:,11])
    report_modes[:,12] = np.where(np.isnan(report_modes[:,12])==True,0,report_modes[:,12])
    report_modes[:,13] = np.where(np.isnan(report_modes[:,13])==True,0,report_modes[:,13])
    
    cleaned_distanza = uniform_filter1d(report_modes[:,10],size = clean_size)
    cleaned_distanza_ramo2 = uniform_filter1d(report_modes[:,11],size = clean_size)
    cleaned_distanza_two_modes = uniform_filter1d(report_modes[:,12],size = clean_size)
    cleaned_distanza_two_modes_ramo2 = uniform_filter1d(report_modes[:,13],size = clean_size)
    
    np.savetxt(os.path.join(path_out,  run_name +'_report_modes.txt'), report_modes, fmt='%.2f', delimiter=',', header = 'moda1 [mm], moda2 [mm], moda 3 [mm], ntracc1 [%],ntracc2 [%],ntracc3 [%],moda1 ramo2 [mm], moda2 ramo2 [mm], ntracc1 ramo2 [%],ntracc2 [%],distanza [mm], distanza ramo 2 [mm],distanza bimodali [mm],distanza bimodali ramo2 [mm]')    
    distance1 = cleaned_distanza[-1]
    velocity1 = distance1/((end-start)*4)
    distance1ramo2 = cleaned_distanza_ramo2[-1]
    velocity1ramo2 = distance1ramo2/((end-start)*4)
    distance_two_modes = np.mean(cleaned_distanza_two_modes) 
    distance_two_modes_ramo2 = np.mean(cleaned_distanza_two_modes_ramo2) 
    
    #####PLOT######
    media = np.median(cleaned_mean[start:end])
    moda1 = np.median(cleaned_moda1)
    moda2 = np.median(cleaned_moda2)
    moda3 = np.median(cleaned_moda3)
    
    if yrami != 0:
       media_ramo2 = np.median(cleaned_mean_ramo2[start:end])
       moda1_ramo2 = np.median(cleaned_moda1_ramo2)
       moda2_ramo2 = np.median(cleaned_moda2_ramo2)
    
    fig, axs = plt.subplots(5,sharex=True,figsize = (16.5,11.7), tight_layout=True)
    fig.suptitle('Traccianti metodo 1 '+run_name)
    axs[0].set_title('Mode',fontsize='large',loc='left')
    axs[0].axhline(media,color = 'k',ls='--')
    axs[0].plot(time[start:end],cleaned_mean[start:end],'k',label = 'Media')
    if nmode1 > 1:
        axs[0].axhline(moda1,color = 'orange',ls='--')
        axs[0].plot(time[start:end],cleaned_moda1,color = 'orange', label ='Moda dist 1')
        axs[0].axhline(moda2,color = 'red',ls='--')
        axs[0].plot(time[start:end],cleaned_moda2,color = 'red', label ='Moda dist 2')
    if nmode1 == 3:
        axs[0].axhline(moda3,color='green',ls='--')
        axs[0].plot(time[start:end],cleaned_moda3,color = 'green', label ='Moda dist 3')
    axs[0].grid()
    axs[0].set_ylim(0)
    axs[0].legend(loc = 'upper left')
    axs[1].set_title('Numero traccianti totali',fontsize='large',loc='left')
    axs[1].plot(time[start:end],cleaned_ntracc[start:end], color = 'blue')
    axs[1].grid()
    axs[1].set_ylim(0)
    axs[2].set_title('Numero traccianti',fontsize='large',loc='left')
    if nmode1 > 1:
        axs[2].plot(time[start:end],cleaned_ntracc1, color = 'orange', label ='dist 1')
        axs[2].plot(time[start:end],cleaned_ntracc2, color = 'red', label ='dist 2')
        axs[2].legend(loc = 'upper left')
    if nmode1 == 3:
        axs[2].plot(time[start:end],cleaned_ntracc3, color = 'green', label ='dist 3')
    axs[2].grid()
    axs[2].set_ylim(0)
    axs[3].set_title('Traccianti in zone di erosione e in zone di deposito',fontsize='large',loc='left') 
    axs[3].plot(time[start:end],cleaned_erosione[start:end], color = 'red', label ='N° traccianti in zona di erosione')
    axs[3].plot(time[start:end],cleaned_deposito[start:end], color = 'blue', label ='N° traccianti in zona di deposito')
    axs[3].legend(loc = 'upper left')
    axs[3].grid()
    axs[3].set_ylim(0)
    axs[4].set_title('Velocità',fontsize='large',loc='left')
    axs[4].plot(time[start:end],cleaned_distanza/(time[start:end]*60), color = 'purple', label ='Velocità')
    axs[4].set_xlabel('time')
    axs[4].legend(loc = 'upper left')
    axs[4].grid()
    axs[4].set_ylim(0)
    fig.tight_layout()
    
    plt.savefig(os.path.join(path_out, run_name +'_tracers_in_time_method1.png'), dpi=300, bbox_inches='tight')
    
    
    if yrami != 0:
        fig, axs = plt.subplots(5,sharex=True,figsize = (16.5,11.7), tight_layout=True)
        fig.suptitle('Traccianti ramo2 metodo 2 '+run_name)
        axs[0].set_title('Mode',fontsize='large',loc='left')
        axs[0].axhline(media_ramo2,color ='k',ls='--')
        axs[0].plot(time[start:end],cleaned_mean_ramo2[start:end],'k',label = 'Media')        
        if nmode2 > 1:
            axs[0].axhline(moda1_ramo2,color ='orange',ls='--')
            axs[0].plot(time[start:end],cleaned_moda1_ramo2,color = 'orange', label ='Moda dist 1')            
            axs[0].axhline(moda2_ramo2,color='red',ls='--')
            axs[0].plot(time[start:end],cleaned_moda2_ramo2,color = 'red', label ='Moda dist 2')            
        axs[0].grid()
        axs[0].legend(loc = 'upper left')
        axs[0].set_ylim(0)
        axs[1].set_title('Numero traccianti totali',fontsize='large',loc='left')
        axs[1].plot(time[start:end],cleaned_ntracc_ramo2[start:end], color = 'blue')
        axs[1].grid()
        axs[1].set_ylim(0)
        axs[2].set_title('Numero traccianti',fontsize='large',loc='left')
        if nmode2 > 1:
            axs[2].plot(time[start:end],cleaned_ntracc1_ramo2, color = 'orange', label ='dist 1')
            axs[2].plot(time[start:end],cleaned_ntracc2_ramo2, color = 'red', label ='dist 2')
            axs[2].legend(loc = 'upper left')
        axs[2].grid()
        axs[2].set_ylim(0)
        axs[3].set_title('Traccianti in zone di erosione e in zone di deposito',fontsize='large',loc='left')
        axs[3].plot(time[start:end],cleaned_erosione_ramo2[start:end], color = 'red', label ='N° traccianti in zona di erosione')
        axs[3].plot(time[start:end],cleaned_deposito_ramo2[start:end], color = 'blue', label ='N° traccianti in zona di deposito')
        axs[3].legend(loc = 'upper left')
        axs[3].grid()
        axs[3].set_ylim(0)
        axs[4].set_title('Velocità',fontsize='large',loc='left')
        axs[4].plot(time[start:end],cleaned_distanza_ramo2/(time[start:end]*60), color = 'purple', label ='Velocità')
        axs[4].set_xlabel('time')
        axs[4].legend(loc = 'upper left')
        axs[4].grid()
        axs[4].set_ylim(0)
        fig.tight_layout()

        
        plt.savefig(os.path.join(path_out, run_name +'_tracers_in_time_ramo2_method1.png'), dpi=300, bbox_inches='tight')
        
        
    ###################################################################################
    # Distance method 2 # distance between the mean of appeared and disappeared tracers with a 15 frame envelope for all the frames
    ################################################################################### 
    envelope_tracers_app = pd.DataFrame()
    envelope_tracers_dis = pd.DataFrame()
    envelope_tracers_app_ramo2 = pd.DataFrame()
    envelope_tracers_dis_ramo2 = pd.DataFrame()
    distance_array = []
    distance_ramo2_array = []
    mean_appeared = []
    mean_disappeared = []
    mean_appeared_ramo2 = []
    mean_disappeared_ramo2 = []
    for i in range(start,end,1):
        dict_tracers_app_stop = {'x':tracers_appeared_stopped[i,:,0],'y':tracers_appeared_stopped[i,:,1],'z_DoD':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        #df_tracers_app_stop = df_tracers_app_stop.drop(df_tracers_app_stop[df_tracers_app_stop.x<100].index)
        df_tracers_app_stop = df_tracers_app_stop.reset_index(drop = True)
        dict_tracers_dis_stop = {'x':tracers_disappeared_stopped[i,:,0],'y':tracers_disappeared_stopped[i,:,1],'z_DoD':tracers_disappeared_stopped[i,:,3]}
        df_tracers_dis_stop = pd.DataFrame(dict_tracers_dis_stop)
        df_tracers_dis_stop = df_tracers_dis_stop.dropna()
        #df_tracers_dis_stop = df_tracers_dis_stop.drop(df_tracers_dis_stop[df_tracers_dis_stop.x<100].index)
        df_tracers_dis_stop = df_tracers_dis_stop.reset_index(drop = True)
        if yrami != 0:
            df_tracers_app_stop_ramo2 = df_tracers_app_stop[df_tracers_app_stop.y>yrami]
            envelope_tracers_app_ramo2 = pd.concat([envelope_tracers_app_ramo2, df_tracers_app_stop_ramo2])
            df_tracers_app_stop = df_tracers_app_stop[df_tracers_app_stop.y<yrami]
            envelope_tracers_app = pd.concat([envelope_tracers_app, df_tracers_app_stop])
            df_tracers_dis_stop_ramo2 = df_tracers_dis_stop[df_tracers_dis_stop.y>yrami]
            envelope_tracers_dis_ramo2 = pd.concat([envelope_tracers_dis_ramo2, df_tracers_dis_stop_ramo2]) 
            df_tracers_dis_stop = df_tracers_dis_stop[df_tracers_dis_stop.y<yrami]
            envelope_tracers_dis = pd.concat([envelope_tracers_dis, df_tracers_dis_stop])
            if (i-start)%15 == 0 and i!= start:
                mean_appeared_ramo2 = np.append(mean_appeared_ramo2,envelope_tracers_app_ramo2.mean().x)
                mean_disappeared_ramo2 = np.append(mean_disappeared_ramo2,envelope_tracers_dis_ramo2.mean().x)
                if envelope_tracers_app_ramo2.mean().x>envelope_tracers_dis_ramo2.mean().x:
                    distance_ramo2_array = np.append(distance_ramo2_array,envelope_tracers_app_ramo2.mean().x-envelope_tracers_dis_ramo2.mean().x)
                else:
                    distance_ramo2_array = np.append(distance_ramo2_array,0)
                envelope_tracers_app_ramo2 = pd.DataFrame() 
                envelope_tracers_dis_ramo2 = pd.DataFrame() 
                mean_appeared = np.append(mean_appeared,envelope_tracers_app.mean().x)
                mean_disappeared = np.append(mean_disappeared,envelope_tracers_dis.mean().x)
                if envelope_tracers_app.mean().x>envelope_tracers_dis.mean().x:
                    distance_array = np.append(distance_array,envelope_tracers_app.mean().x-envelope_tracers_dis.mean().x)
                else:
                    distance_array = np.append(distance_array,0)
                envelope_tracers_app = pd.DataFrame() 
                envelope_tracers_dis = pd.DataFrame()
        else:
            envelope_tracers_app = pd.concat([envelope_tracers_app, df_tracers_app_stop])
            envelope_tracers_dis = pd.concat([envelope_tracers_dis, df_tracers_dis_stop])
            if (i-start)%15 == 0 and i!= start:
                mean_appeared = np.append(mean_appeared,envelope_tracers_app.mean().x)
                mean_disappeared = np.append(mean_disappeared,envelope_tracers_dis.mean().x)
                if envelope_tracers_app.mean().x>envelope_tracers_dis.mean().x:
                    distance_array = np.append(distance_array,envelope_tracers_app.mean().x-envelope_tracers_dis.mean().x)
                else:
                    distance_array = np.append(distance_array,0)
                envelope_tracers_app = pd.DataFrame() 
                envelope_tracers_dis = pd.DataFrame()
               
    distance2 = np.mean(distance_array)
    velocity2 = distance2/(4)
    distance2ramo2 =  np.mean(distance_ramo2_array)  
    velocity2ramo2 = distance2ramo2/((end-start)*4)
    
    #####PLOT######
    time = []
    for i in range(start+15,end,15):    
        time = np.append(time,i*4/60)
    
   
    fig, axs = plt.subplots(2,sharex=True,figsize = (16.5,12), tight_layout=True)
    fig.suptitle('Traccianti metodo 2 '+run_name)
    axs[0].set_title('Mean Appeared and Disappeared - ' + run_name,fontsize='large',loc='left')
    axs[0].plot(time,mean_appeared,'b',label = 'Appeared')
    axs[0].plot(time,mean_disappeared,'orange',label = 'Disappeared')
    axs[0].grid()
    axs[0].legend(loc = 'upper left')   
    # axs[0].set_ylim(0,1000)
    axs[1].set_title('Instantaneous velocity - ' + run_name,fontsize='large',loc='left')
    axs[1].axhline(velocity2,color = 'k',ls='--')
    axs[1].plot(time,distance_array/4,'k',label = 'Velocity')
    axs[1].grid()
    axs[1].legend(loc = 'upper left')    
    axs[1].set_xlabel('time')
    axs[1].set_ylim(0)
    fig.tight_layout()
    
    plt.savefig(os.path.join(path_out, run_name +'_tracers_in_time_method2.png'), dpi=300,bbox_inches='tight')
    
    
    if yrami != 0:
        fig, axs = plt.subplots(2,sharex=True,figsize = (16.5,12), tight_layout=True)
        axs[0].set_title('Median Appeared and Disappeared - ' + run_name,fontsize='large',loc='left')
        axs[0].plot(time,mean_appeared_ramo2,'b',label = 'Appeared')
        axs[0].plot(time,mean_disappeared_ramo2,'orange',label = 'Disappeared')
        axs[0].grid()
        axs[0].legend(loc = 'upper left')   
        # axs[0].set_ylim(0,1000)
        axs[1].set_title('Instantaneous velocity - ' + run_name,fontsize='large',loc='left')
        axs[1].axhline(velocity2,color = 'k',ls='--')
        axs[1].plot(time,distance_ramo2_array/4,'k',label = 'Velocity')
        axs[1].grid()
        axs[1].legend(loc = 'upper left')    
        axs[1].set_xlabel('time')
        axs[1].set_ylim(0)
        fig.tight_layout()
        
        plt.savefig(os.path.join(path_out, run_name +'_tracers_in_time_method2_ramo2.png'), dpi=300, bbox_inches='tight')
        
    ###################################################################################
    #Distance method 3 # distance between the mean of the envelope of appeared tracers from start to end
    ###################################################################################
    all_df_tracers_app_stop = pd.DataFrame()
    all_df_tracers_app_stop_ramo2 = pd.DataFrame()
    for i in range(start,end,1):
        dict_tracers_app_stop = {'x':tracers_appeared_stopped[i,:,0],'y':tracers_appeared_stopped[i,:,1],'z_DoD':tracers_appeared_stopped[i,:,3]}
        df_tracers_app_stop = pd.DataFrame(dict_tracers_app_stop)
        df_tracers_app_stop = df_tracers_app_stop.dropna()
        #df_tracers_app_stop = df_tracers_app_stop.drop(df_tracers_app_stop[df_tracers_app_stop.x<100].index)
        df_tracers_app_stop = df_tracers_app_stop.reset_index(drop = True)
        if yrami != 0:
            df_tracers_app_stop_ramo2 = df_tracers_app_stop[df_tracers_app_stop.y>yrami]
            all_df_tracers_app_stop_ramo2 = pd.concat([all_df_tracers_app_stop_ramo2, df_tracers_app_stop_ramo2])
            df_tracers_app_stop = df_tracers_app_stop[df_tracers_app_stop.y<yrami]
            all_df_tracers_app_stop = pd.concat([all_df_tracers_app_stop, df_tracers_app_stop])
        else:
            all_df_tracers_app_stop = pd.concat([all_df_tracers_app_stop, df_tracers_app_stop])        
    distance3 = all_df_tracers_app_stop.mean().x
    velocity3 = distance3/((end-start)*4)
    if yrami != 0:
        distance3ramo2 = all_df_tracers_app_stop_ramo2.mean().x
        velocity3ramo2 = distance3ramo2/((end-start)*4)
     
    ############################################################
    #Exporting all the distances
    ############################################################
    finaldata = np.zeros((12))
    finaldata[0] = int(run_name[1:3])/10    
    finaldata[1] = int(run_name[6:8])
    finaldata[2] = distance1
    finaldata[3] = distance2
    finaldata[4] = distance3
    finaldata[5] = velocity1/1000
    finaldata[6] = velocity2/1000
    finaldata[7] = velocity3/1000
    finaldata[8] = distance_two_modes
    finaldata[9] = np.mean(cleaned_deposito[start:end])
    finaldata[10] = np.mean(cleaned_erosione[start:end])
    finaldata[11] = end/len(tracers_reduced)
    if skip != 1:
        report_distances = np.vstack((report_distances,finaldata))  
 
    if yrami != 0 and skip != 2:
       finaldata = np.zeros((12))
       finaldata[0] = int(run_name[1:3])/10    
       finaldata[1] = int(run_name[6:8])
       finaldata[2] = distance1ramo2
       finaldata[3] = distance2ramo2
       finaldata[4] = distance3ramo2
       finaldata[5] = velocity1ramo2/1000
       finaldata[6] = velocity2ramo2/1000
       finaldata[7] = velocity3ramo2/1000
       finaldata[8] = distance_two_modes_ramo2
       finaldata[9] = np.mean(cleaned_deposito_ramo2[start:end])
       finaldata[10] = np.mean(cleaned_erosione_ramo2[start:end])
       finaldata[11] = end/len(tracers_reduced)
       report_distances = np.vstack((report_distances,finaldata))  
       
    # print(str(run_name)+' finished')

    
np.savetxt(os.path.join(report_dir, 'report_distances.txt'), report_distances, comments='', fmt='%.4f', delimiter=',', header = 'portata,prova,distanza1,distanza2,distanza3,velocita1,velocita2,velocita3,distanzatraduemode,percdeposito,percer,percrun')    