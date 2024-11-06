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
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# Script parameters:
w_dir = os.getcwd() # Set Python script location as w_dir

if not os.path.exists(os.path.join(w_dir,'output_data', 'distances')):
    os.mkdir(os.path.join(w_dir,'output_data', 'distances'))
    
plot_dir = os.path.join(w_dir,'output_data', 'distances', 'plot')
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
    
report_dir = os.path.join(w_dir,'output_data', 'distances', 'report')
if not os.path.exists(report_dir):
    os.mkdir(report_dir)


report_distances = pd.read_csv(os.path.join(report_dir, 'report_distances.txt'))
report_DoDdistances = pd.read_csv(os.path.join(report_dir, 'DoDdistanze.txt'))

report_distances['percdeposito']=report_distances['percdeposito']*100
report_distances['percer']=report_distances['percer']*100
report_distances['velocita1']=report_distances['velocita1']*1000
report_distances['velocita2']=report_distances['velocita2']*1000
report_distances['velocita3']=report_distances['velocita3']*1000
#%%
sns.set_context("talk")
sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="distanza1",saturation=0.9,medianprops={"color": "firebrick"},palette='pastel')
sns.despine(left = True, bottom = True)
plt.ylim(0,900)
plt.ylabel("Distanza [mm]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'metodo1.png'), dpi=300,bbox_inches='tight')
plt.show()
#%%
sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="distanza2",saturation=0.9,medianprops={"color": "firebrick"},palette='pastel')
sns.despine(left = True, bottom = True)
plt.ylabel("Distanza [mm]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'metodo2.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="distanza3",saturation=0.9,medianprops={"color": "firebrick"},palette='pastel')
sns.despine(left = True, bottom = True)
plt.ylim(0,900)
plt.ylabel("Distanza [mm]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'metodo3.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="velocita1",saturation=0.9,medianprops={"color": "firebrick"},palette='Set2_r')
sns.despine(left = True, bottom = True)
plt.ylabel("Velocità [mm/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'velocita1.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="velocita2",saturation=0.9,medianprops={"color": "firebrick"},palette='Set2_r')
sns.despine(left = True, bottom = True)
plt.ylabel("Velocità istantanea [mm/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'velocita2.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="velocita3",saturation=0.9,medianprops={"color": "firebrick"},palette='Set2_r')
sns.despine(left = True, bottom = True)
plt.ylabel("Velocità [mm/s]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'velocita3.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="percdeposito",saturation=0.9,medianprops={"color": "firebrick"},palette='Set3_r')
sns.despine(left = True, bottom = True)
plt.ylabel("Traccianti in zona di deposito [%]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'deposito.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="percer",saturation=0.9,medianprops={"color": "firebrick"},palette='Set3_r')
sns.despine(left = True, bottom = True)
plt.ylabel("Traccianti in zona di deposito [%]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'erosione.png'), dpi=300,bbox_inches='tight')
plt.show()

er = report_distances[['portata','percer']]
er['morpho'] = er.percer
er['Legenda'] = 'erosione'
dep = report_distances[['portata','percdeposito']]
dep['morpho'] = dep.percdeposito
dep['Legenda'] = 'deposito'
report_er_dep = pd.concat((er,dep))

sns.set_style("whitegrid")
color_dict = dict({'erosione': 'red','deposito': 'dodgerblue'})
sns.boxplot(data = report_er_dep, x="portata", y="morpho",saturation=0.7,hue = 'Legenda',medianprops={"color": "green"}, palette = color_dict,hue_order =['erosione','deposito'])
sns.despine(left = True, bottom = True)
plt.ylabel("Traccianti [%]")
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.legend().set_title('')
plt.savefig(os.path.join(plot_dir, 'morpho.png'), dpi=300,bbox_inches='tight')
plt.show()

report_distances = report_distances.drop(report_distances[report_distances.distanzatraduemode == 0].index)
dist2mode = report_distances[['portata','distanzatraduemode']]
dist2mode['dist'] = dist2mode.distanzatraduemode
dist2mode['Legenda'] = 'prove'
distDoD = report_DoDdistances[['portata','distanza']]
distDoD['dist'] = distDoD.distanza
distDoD['Legenda'] = 'rilievo'
report = pd.concat((dist2mode,distDoD))

sns.set_style("whitegrid")
sns.boxplot(data = report_distances, x="portata", y="distanzatraduemode",saturation=0.9,medianprops={"color": "firebrick"},palette='Set3')
sns.despine(left = True, bottom = True)
plt.title('Distanza tra le mode')
plt.ylabel("Distanza [mm]")
plt.ylim(0,2200)
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'distanza_tra_due_mode.png'), dpi=300,bbox_inches='tight')
plt.show()

sns.set_style("whitegrid")
sns.boxplot(data = report_DoDdistances, x="portata", y="distanza",saturation=0.9,medianprops={"color": "firebrick"},palette='Set3')
sns.despine(left = True, bottom = True)
plt.title('Distanza tra le aree di deposito')
plt.ylabel("Distanza [mm]")
plt.ylim(0,2200)
plt.xlabel("Portata [l/s]")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'distanza_DoD.png'), dpi=300,bbox_inches='tight')
plt.show()

# sns.set_style("whitegrid")
# sns.boxplot(data = report, x="portata", y="dist",saturation=0.9,medianprops={"color": "firebrick"},palette='Set3',hue = 'Legenda')
# sns.despine(left = True, bottom = True)
# plt.ylabel("Distanza [mm]")
# plt.ylim(0,2200)
# plt.xlabel("Portata [l/s]")
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, 'distanza_DoD.png'), dpi=300,bbox_inches='tight')
# plt.show()



