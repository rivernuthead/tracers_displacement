# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:13:38 2022

@author: Marco
"""

# import necessary packages
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# Script parameters:
w_dir = os.getcwd() # Set Python script location as w_dir
plot_dir = os.path.join(w_dir, 'distances_plot')

report_distances = pd.read_csv(os.path.join(w_dir, 'report_distances.txt'))
report_DoDdistances = pd.read_csv(os.path.join(w_dir, 'DoDdistanze.txt'))

report_distances['percdeposito']=report_distances['percdeposito']*100
report_distances['velocita1']=report_distances['velocita1']*1000
report_distances['velocita2']=report_distances['velocita2']*1000
report_distances['velocita3']=report_distances['velocita3']*1000

path_in_DoD_q05 = os.path.join(w_dir,'DoDs','DoD_q05_1')
path_in_DoD_q07 = os.path.join(w_dir,'DoDs','DoD_q07_1')
path_in_DoD_q10 = os.path.join(w_dir,'DoDs','DoD_q10_1')
    
DoD_q05_1 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_1-0_filt_fill.txt'))
DoD_q05_2 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_2-1_filt_fill.txt'))
DoD_q05_3 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_3-2_filt_fill.txt'))
DoD_q05_4 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_5-4_filt_fill.txt'))
DoD_q05_5 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_6-5_filt_fill.txt'))
DoD_q05_6 = np.loadtxt(os.path.join(path_in_DoD_q05, 'DoD_7-6_filt_fill.txt'))

DoD_q07_1 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_1-0_filt_fill.txt'))
DoD_q07_2 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_2-1_filt_fill.txt'))
DoD_q07_3 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_3-2_filt_fill.txt'))
DoD_q07_4 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_5-4_filt_fill.txt'))
DoD_q07_5 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_6-5_filt_fill.txt'))
DoD_q07_6 = np.loadtxt(os.path.join(path_in_DoD_q07, 'DoD_7-6_filt_fill.txt'))

DoD_q10_1 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_1-0_filt_fill.txt'))
DoD_q10_2 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_2-1_filt_fill.txt'))
DoD_q10_3 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_3-2_filt_fill.txt'))
DoD_q10_4 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_4-3_filt_fill.txt'))
DoD_q10_5 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_5-4_filt_fill.txt'))
DoD_q10_6 = np.loadtxt(os.path.join(path_in_DoD_q10, 'DoD_6-5_filt_fill.txt'))
  

DoDs = [DoD_q05_1,DoD_q05_2,DoD_q05_3,DoD_q05_4,DoD_q05_5,DoD_q05_6]

V_q05_array = np.zeros((6,1))
for j in range(1,7):
    DoD_er = np.where(DoDs[j-1]>0,0,DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er)==True,0,DoD_er)
    V_e = -np.sum(DoD_er)*50*5
    V_q05_array[j-1] = V_e
V_e_q05 = np.mean(V_q05_array)

V_q07_array = np.zeros((6,1))
DoDs = [DoD_q07_1,DoD_q07_2,DoD_q07_3,DoD_q07_4,DoD_q07_5,DoD_q07_6]
for j in range(1,7):
    DoD_er = np.where(DoDs[j-1]>0,0,DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er)==True,0,DoD_er)
    V_e = -np.sum(DoD_er)*50*5
    V_q07_array [j-1] = V_e
V_e_q07 = np.mean(V_q07_array )

V_q10_array = np.zeros((6,1))  
DoDs = [DoD_q10_1,DoD_q10_2,DoD_q10_3,DoD_q10_4,DoD_q10_5,DoD_q10_6]
for j in range(1,7):
    DoD_er = np.where(DoDs[j-1]>0,0,DoDs[j-1])
    DoD_er = np.where(np.isnan(DoD_er)==True,0,DoD_er)
    V_e = -np.sum(DoD_er)*50*5
    V_q10_array[j-1] = V_e
V_e_q10 = np.mean(V_q10_array)

report_distances_q05 = report_distances[report_distances.portata == 0.5]
report_distances_q05_1 = pd.concat((report_distances_q05[report_distances_q05.prova == 1],report_distances_q05[report_distances_q05.prova == 2]))
report_distances_q05_2 = pd.concat((report_distances_q05[report_distances_q05.prova == 3],report_distances_q05[report_distances_q05.prova == 4]))
report_distances_q05_3 = pd.concat((report_distances_q05[report_distances_q05.prova == 5],report_distances_q05[report_distances_q05.prova == 6]))
report_distances_q05_4 = pd.concat((report_distances_q05[report_distances_q05.prova == 7],report_distances_q05[report_distances_q05.prova == 8]))
report_distances_q05_5 = pd.concat((report_distances_q05[report_distances_q05.prova == 9],report_distances_q05[report_distances_q05.prova == 10]))
report_distances_q05_6 = pd.concat((report_distances_q05[report_distances_q05.prova == 11],report_distances_q05[report_distances_q05.prova == 12]))

report_distances_q05_1['portatas_dist']=report_distances_q05_1['velocita1']*0.63*V_q05_array[0]/2*(2.65/1000)/13900
report_distances_q05_1['portatas_comp']=report_distances_q05_1['velocita3']*0.63*V_q05_array[0]/2*(2.65/1000)/13900
report_distances_q05_2['portatas_dist']=report_distances_q05_2['velocita1']*0.63*V_q05_array[1]/2*(2.65/1000)/13900
report_distances_q05_2['portatas_comp']=report_distances_q05_2['velocita3']*0.63*V_q05_array[1]/2*(2.65/1000)/13900
report_distances_q05_3['portatas_dist']=report_distances_q05_3['velocita1']*0.63*V_q05_array[2]/2*(2.65/1000)/13900
report_distances_q05_3['portatas_comp']=report_distances_q05_3['velocita3']*0.63*V_q05_array[2]/2*(2.65/1000)/13900
report_distances_q05_4['portatas_dist']=report_distances_q05_4['velocita1']*0.63*V_q05_array[3]/2*(2.65/1000)/13900
report_distances_q05_4['portatas_comp']=report_distances_q05_4['velocita3']*0.63*V_q05_array[3]/2*(2.65/1000)/13900
report_distances_q05_5['portatas_dist']=report_distances_q05_5['velocita1']*0.63*V_q05_array[4]/2*(2.65/1000)/13900
report_distances_q05_6['portatas_comp']=report_distances_q05_5['velocita3']*0.63*V_q05_array[4]/2*(2.65/1000)/13900
report_distances_q05_6['portatas_dist']=report_distances_q05_6['velocita1']*0.63*V_q05_array[5]/2*(2.65/1000)/13900
report_distances_q05_6['portatas_comp']=report_distances_q05_6['velocita3']*0.63*V_q05_array[5]/2*(2.65/1000)/13900

report_distances_q05 = pd.concat((report_distances_q05_1,report_distances_q05_2,report_distances_q05_3,report_distances_q05_4,report_distances_q05_5,report_distances_q05_6))
report_distances_q05['distanza1'] = report_distances_q05['distanza1']/306
report_distances_q05['distanza3'] = report_distances_q05['distanza3']/306

report_distances_q07 = report_distances[report_distances.portata == 0.7]
report_distances_q07_1 = pd.concat((report_distances_q07[report_distances_q07.prova == 1],report_distances_q07[report_distances_q07.prova == 2]))
report_distances_q07_2 = pd.concat((report_distances_q07[report_distances_q07.prova == 3],report_distances_q07[report_distances_q07.prova == 4]))
report_distances_q07_3 = pd.concat((report_distances_q07[report_distances_q07.prova == 5],report_distances_q07[report_distances_q07.prova == 6]))
report_distances_q07_4 = pd.concat((report_distances_q07[report_distances_q07.prova == 7],report_distances_q07[report_distances_q07.prova == 8]))
report_distances_q07_5 = pd.concat((report_distances_q07[report_distances_q07.prova == 9],report_distances_q07[report_distances_q07.prova == 10]))
report_distances_q07_6 = pd.concat((report_distances_q07[report_distances_q07.prova == 11],report_distances_q07[report_distances_q07.prova == 12]))

report_distances_q07_1['portatas_dist']=report_distances_q07_1['velocita1']*0.63*V_q07_array[0]/2*(2.65/1000)/13900
report_distances_q07_1['portatas_comp']=report_distances_q07_1['velocita3']*0.63*V_q07_array[0]/2*(2.65/1000)/13900
report_distances_q07_2['portatas_dist']=report_distances_q07_2['velocita1']*0.63*V_q07_array[1]/2*(2.65/1000)/13900
report_distances_q07_2['portatas_comp']=report_distances_q07_2['velocita3']*0.63*V_q07_array[1]/2*(2.65/1000)/13900
report_distances_q07_3['portatas_dist']=report_distances_q07_3['velocita1']*0.63*V_q07_array[2]/2*(2.65/1000)/13900
report_distances_q07_3['portatas_comp']=report_distances_q07_3['velocita3']*0.63*V_q07_array[2]/2*(2.65/1000)/13900
report_distances_q07_4['portatas_dist']=report_distances_q07_4['velocita1']*0.63*V_q07_array[3]/2*(2.65/1000)/13900
report_distances_q07_4['portatas_comp']=report_distances_q07_4['velocita3']*0.63*V_q07_array[3]/2*(2.65/1000)/13900
report_distances_q07_5['portatas_dist']=report_distances_q07_5['velocita1']*0.63*V_q07_array[4]/2*(2.65/1000)/13900
report_distances_q07_6['portatas_comp']=report_distances_q07_5['velocita3']*0.63*V_q07_array[4]/2*(2.65/1000)/13900
report_distances_q07_6['portatas_dist']=report_distances_q07_6['velocita1']*0.63*V_q07_array[5]/2*(2.65/1000)/13900
report_distances_q07_6['portatas_comp']=report_distances_q07_6['velocita3']*0.63*V_q07_array[5]/2*(2.65/1000)/13900

report_distances_q07 = pd.concat((report_distances_q07_1,report_distances_q07_2,report_distances_q07_3,report_distances_q07_4,report_distances_q07_5,report_distances_q07_6))
report_distances_q07['distanza1'] = report_distances_q07['distanza1']/396
report_distances_q07['distanza3'] = report_distances_q07['distanza3']/396


report_distances_q10 = report_distances[report_distances.portata == 1.0]
report_distances_q10_1 = pd.concat((report_distances_q10[report_distances_q10.prova == 1],report_distances_q10[report_distances_q10.prova == 2]))
report_distances_q10_2 = pd.concat((report_distances_q10[report_distances_q10.prova == 3],report_distances_q10[report_distances_q10.prova == 4]))
report_distances_q10_3 = pd.concat((report_distances_q10[report_distances_q10.prova == 5],report_distances_q10[report_distances_q10.prova == 6]))
report_distances_q10_4 = pd.concat((report_distances_q10[report_distances_q10.prova == 7],report_distances_q10[report_distances_q10.prova == 8]))
report_distances_q10_5 = pd.concat((report_distances_q10[report_distances_q10.prova == 9],report_distances_q10[report_distances_q10.prova == 10]))
report_distances_q10_6 = pd.concat((report_distances_q10[report_distances_q10.prova == 11],report_distances_q10[report_distances_q10.prova == 12]))

report_distances_q10_1['portatas_dist']=report_distances_q10_1['velocita1']*0.63*V_q10_array[0]/2*(2.65/1000)/11800
report_distances_q10_1['portatas_comp']=report_distances_q10_1['velocita3']*0.63*V_q10_array[0]/2*(2.65/1000)/11800
report_distances_q10_2['portatas_dist']=report_distances_q10_2['velocita1']*0.63*V_q10_array[1]/2*(2.65/1000)/11800
report_distances_q10_2['portatas_comp']=report_distances_q10_2['velocita3']*0.63*V_q10_array[1]/2*(2.65/1000)/11800
report_distances_q10_3['portatas_dist']=report_distances_q10_3['velocita1']*0.63*V_q10_array[2]/2*(2.65/1000)/11800
report_distances_q10_3['portatas_comp']=report_distances_q10_3['velocita3']*0.63*V_q10_array[2]/2*(2.65/1000)/11800
report_distances_q10_4['portatas_dist']=report_distances_q10_4['velocita1']*0.63*V_q10_array[3]/2*(2.65/1000)/11800
report_distances_q10_4['portatas_comp']=report_distances_q10_4['velocita3']*0.63*V_q10_array[3]/2*(2.65/1000)/11800
report_distances_q10_5['portatas_dist']=report_distances_q10_5['velocita1']*0.63*V_q10_array[4]/2*(2.65/1000)/11800
report_distances_q10_6['portatas_comp']=report_distances_q10_5['velocita3']*0.63*V_q10_array[4]/2*(2.65/1000)/11800
report_distances_q10_6['portatas_dist']=report_distances_q10_6['velocita1']*0.63*V_q10_array[5]/2*(2.65/1000)/11800
report_distances_q10_6['portatas_comp']=report_distances_q10_6['velocita3']*0.63*V_q10_array[5]/2*(2.65/1000)/11800

report_distances_q10 = pd.concat((report_distances_q10_1,report_distances_q10_2,report_distances_q10_3,report_distances_q10_4,report_distances_q10_5,report_distances_q10_6))
report_distances_q10['distanza1'] = report_distances_q10['distanza1']/498
report_distances_q10['distanza3'] = report_distances_q10['distanza3']/498

report_port = pd.concat((report_distances_q05,report_distances_q07,report_distances_q10))
report_port['portatas_comp'] = report_port['portatas_comp']*report_port['percrun']
report_port['portatas_dist'] = report_port['portatas_dist']*report_port['percrun']


sns.set_style("whitegrid")
sns.set_context('talk')
fig, ax = plt.subplots()
sns.boxplot(data = report_port, x="portata", y="portatas_dist",saturation=0.9,medianprops={"color": "firebrick"},palette='Set1_r')
sns.despine(left = True, bottom = True)
# plt.axhline(0.78,0.7,0.97,color='blue',ls='--',lw = 2)
# ax.add_patch(mpatches.Rectangle((1.59, 0.78-0.3),0.82, 0.6,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue' ))  
# plt.axhline(0.33,0.37,0.635,color='blue',ls='--',lw = 2)
# ax.add_patch(mpatches.Rectangle((0.59, 0.33-0.18),0.82, 0.36,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
# plt.axhline(0.2,0.035,0.3,color='blue',ls='--',lw = 2)
# ax.add_patch(mpatches.Rectangle((-0.41, 0.2-0.12),0.82, 0.24,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
plt.ylim(0,1.2)
plt.ylabel("Portata solida [g/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'portatas_dist.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.boxplot(data = report_port, x="portata", y="portatas_comp",saturation=0.9,medianprops={"color": "firebrick"},palette='Set1_r')
sns.despine(left = True, bottom = True)
plt.axhline(0.78,0.7,0.97,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((1.59, 0.78-0.3),0.82, 0.6,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue' ))  
plt.axhline(0.33,0.37,0.635,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((0.59, 0.33-0.18),0.82, 0.36,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
plt.axhline(0.2,0.035,0.3,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((-0.41, 0.2-0.12),0.82, 0.24,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
plt.ylim(0,1.2)
plt.ylabel("Portata solida [g/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'portatas_comp.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_port, x="portata", y="distanza1",saturation=0.9,medianprops={"color": "firebrick"},palette='pastel')
sns.despine(left = True, bottom = True)
plt.ylabel("Distanza/Larghezza bagnata [-]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'metodo1_beta.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_port, x="portata", y="distanza3",saturation=0.9,medianprops={"color": "firebrick"},palette='pastel')
sns.despine(left = True, bottom = True)
plt.ylabel("Distanza/Larghezza bagnata [-]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'metodo3_beta.png'), dpi=300,bbox_inches='tight')
plt.show()

report_DoDdistances = pd.read_csv(os.path.join(w_dir, 'DoDdistanze.txt'))
report_DoDdistances_q05 = report_DoDdistances[report_DoDdistances.portata == 0.5]
report_DoDdistances_q07 = report_DoDdistances[report_DoDdistances.portata == 0.7]
report_DoDdistances_q10 = report_DoDdistances[report_DoDdistances.portata == 1.0]

report_DoDdistances_q05['portatas']=report_DoDdistances_q05['distanza']/1800*0.63*V_e_q05*(2.65/1000)/13900
report_DoDdistances_q07['portatas']=report_DoDdistances_q07['distanza']/1800*0.63*V_e_q07*(2.65/1000)/13900
report_DoDdistances_q10['portatas']=report_DoDdistances_q10['distanza']/1800*0.63*V_e_q10*(2.65/1000)/11800
report_DoDport = pd.concat((report_DoDdistances_q05,report_DoDdistances_q07,report_DoDdistances_q10))

sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.boxplot(data = report_DoDport, x="portata", y="portatas",saturation=0.9,medianprops={"color": "firebrick"},palette='Set1_r')
sns.despine(left = True, bottom = True)
plt.axhline(0.78,0.7,0.97,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((1.59, 0.78-0.3),0.82, 0.6,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue' ))  
plt.axhline(0.33,0.37,0.635,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((0.59, 0.33-0.18),0.82, 0.36,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
plt.axhline(0.2,0.035,0.3,color='blue',ls='--',lw = 2)
ax.add_patch(mpatches.Rectangle((-0.41, 0.2-0.12),0.82, 0.24,facecolor = 'lightskyblue',lw=3,alpha = 0.3,hatch = '/',edgecolor ='darkblue'))  
plt.ylim(0,1.2)
plt.ylabel("Portata solida [g/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'portatas_DoD.png'), dpi=300,bbox_inches='tight')
plt.show()