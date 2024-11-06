# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:58:53 2022

@author: Marco
"""

# import necessary packages
import os
from PIL import Image
import numpy as np
import imageio
import geopandas as gpd
import shutil
from osgeo import gdal,ogr
import re

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# Script parameters:
run_mode = 1

# Set working directory
run = 'q10_1r1'
w_dir = os.getcwd() # Set Python script location as w_dir

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Set parameters
gmr_thr = 25   # G - R threshold 
dt = 4  #time between photos
thr = 36 #area threshold of a single tracer
ntracmax = 1000
area_threshold = 4
mask_buffer = 4

# List all available runs, depending on run_mode
runs =[]
if run_mode==1:
    runs = [run] # Comment to perform batch process over folder
elif run_mode==2:
    for f in sorted(os.listdir(os.path.join(w_dir, 'area'))):
        path = os.path.join(os.path.join(w_dir, 'area'), f)
        if os.path.isdir(path) and not(f.startswith('_')):
            runs = np.append(runs, f)
else:
    pass

qset = runs[0]
run_param = np.loadtxt(os.path.join(w_dir, 'run_param_'+qset[0:3]+'.txt'), skiprows=1, delimiter=',')
    
###############################################################################
# LOOP OVER RUNS
###############################################################################
for run in runs:
    time = 0
    i = 0
    path_in = os.path.join(w_dir, 'area', run)
    path_in_DEM = os.path.join(w_dir,'surveys',run[0:5])
    path_in_DoD = os.path.join(w_dir,'DoDs','DoD_'+run[0:5])
    
    # Create outputs script directory
    path_out = os.path.join(w_dir, 'area', run+'output')
    if os.path.exists(path_out):
        shutil.rmtree(path_out, ignore_errors=False, onerror=None)
        os.mkdir(path_out) 
    else: 
        os.mkdir(path_out)  
    
        
    
    # List input directory files
    files_tot = sorted(os.listdir(path_in))
    files = files_tot
    all_tracers = np.zeros((len(files),ntracmax,1))
    all_tracers_appeared = np.zeros((len(files)-1,ntracmax,1))
    all_tracers_disappeared = np.zeros((len(files)-1,ntracmax,1))
    all_tracers_areas = []
    all_tracers_appeared_areas = []
    all_tracers_disappeared_areas = []
    img_old = np.zeros((1440,4288)) 
    
    
    for file in sorted(files,key = numericalSort):
        path = os.path.join(path_in, file) # Build path
        
        img = Image.open(path) # Open image
        img_array = np.asarray(img)    # Convert image in numpy array
          
        # Extract RGB bands and convert as int64:
        band_red = img_array[:,:,0]    
        band_red = band_red.astype(np.int64)
        band_green = img_array[:,:,1]
        band_green = band_green.astype(np.int64)
        band_blue = img_array[:,:,2]
        band_blue = band_blue.astype(np.int64)
        
        #Calculate G - R and extract when G - R > threshold
        img_gmr = band_green - band_red
        img_gmr_filt = np.where(img_gmr < gmr_thr, np.nan, img_gmr)
        img_gmr_filt = np.where(np.logical_not(np.isnan(img_gmr_filt)),1,0)
         
        if time != 0:
            diff = img_gmr_filt - img_old
            img_diff_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_diff.png'), diff)
            tracers_appeared = np.where(diff==1,1,0)
            img_ta_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_ta.png'), tracers_appeared)
            tracers_disappeared = np.where(diff==-1,1,0)
            img_td_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_td.png'), tracers_disappeared)
        
        img_gmr_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_gmr.png'), img_gmr_filt)
        img_old = img_gmr_filt    
        
        # get raster data source N.B. you need to have the file on your pc 
        open_image = gdal.Open(path_out + "\\" + str(time)+ "s_gmr.png")
        input_band = open_image.GetRasterBand(1)
        if time != 0:
            open_image_ta = gdal.Open(path_out + "\\" + str(time)+ "s_ta.png")
            input_band_ta = open_image_ta.GetRasterBand(1)
            open_image_td = gdal.Open(path_out + "\\" + str(time)+ "s_td.png")
            input_band_td = open_image_td.GetRasterBand(1)
            
        # create output data source
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            
        # create output file name
        output_shapefile = shp_driver.CreateDataSource(path_out + "\\" + str(time)+ "s_tracers.shp" )
        new_shapefile = output_shapefile.CreateLayer(path_out, srs = None )
        
        if time != 0:
            output_shapefile_ta = shp_driver.CreateDataSource(path_out + "\\" + str(time)+ "s_tracers_ta.shp" )
            new_shapefile_ta = output_shapefile_ta.CreateLayer(path_out, srs = None )    
            output_shapefile_td = shp_driver.CreateDataSource(path_out + "\\" + str(time)+ "s_tracers_td.shp" )
            new_shapefile_td = output_shapefile_td.CreateLayer(path_out, srs = None )    
        
        # transform the raster photo in a shp file
        gdal.Polygonize(input_band, None, new_shapefile, -1, [], callback=None)
        new_shapefile.SyncToDisk()
        
        if time != 0:
            gdal.Polygonize(input_band_ta, None, new_shapefile_ta, -1, [], callback=None)
            new_shapefile_ta.SyncToDisk()
            gdal.Polygonize(input_band_td, None, new_shapefile_td, -1, [], callback=None)
            new_shapefile_td.SyncToDisk()
        #output_shapefile.Destroy()
            

        # Open the shp file with geopandas
        tracers = gpd.read_file(path_out + "\\" + str(time)+ "s_tracers.shp")
        # calculate the area of polygons
        tracers['Area'] = tracers.area
        # drop the polygons where there aren't tracers
        tracers.drop(tracers.index[tracers.area == max(tracers.area)], inplace = True)
        tracers.drop(tracers.index[tracers.area < area_threshold], inplace = True)
        if time == 0:
            mask = tracers
            mask['geometry'] = mask.geometry.buffer(mask_buffer)
        else:
            tracers = tracers.overlay(mask, how = 'difference')
        maskdis = tracers
        maskdis['geometry'] = maskdis.geometry.buffer(mask_buffer)
        tracers = tracers.drop(columns=['geometry'])    
        tracers = tracers.drop(columns=['FID']) 
        nptracers = tracers.to_numpy()
        all_tracers_areas=np.append(all_tracers_areas,nptracers)
        while nptracers.shape != (ntracmax,1):
            newrow = np.array([np.nan])
            nptracers = np.vstack((nptracers,newrow))
        # add the dataframe to the matrix of all photos
        all_tracers[i,:,:] = nptracers  

        
        
        if time != 0:
            maskapp = gpd.read_file(path_out + "\\" + str(time-4)+ "s_tracers.shp")
            maskapp.drop(maskapp.index[maskapp.area == max(maskapp.area)], inplace = True)
            maskapp.drop(maskapp.index[maskapp.area < area_threshold], inplace = True)
            maskapp['geometry'] = maskapp.geometry.buffer(mask_buffer)

            tracers_appeared = gpd.read_file(path_out + "\\" + str(time)+ "s_tracers_ta.shp")
            tracers_appeared['Area'] = tracers_appeared.area
            # drop the polygons where there aren't tracers_appeared
            tracers_appeared.drop(tracers_appeared.index[tracers_appeared.area == max(tracers_appeared.area)], inplace = True)
            tracers_appeared.drop(tracers_appeared.index[tracers_appeared.area < area_threshold], inplace = True)     
            tracers_appeared = tracers_appeared.overlay(maskapp, how = 'difference')
            tracers_appeared = tracers_appeared.drop(columns=['geometry'])
            tracers_appeared = tracers_appeared.drop(columns=['FID'])
            nptracers_ta = tracers_appeared.to_numpy()
            all_tracers_appeared_areas=np.append(all_tracers_appeared_areas,nptracers_ta)
            while nptracers_ta.shape != (ntracmax,1):
                newrow = np.array([np.nan])
                nptracers_ta = np.vstack((nptracers_ta,newrow))
            # add the dataframe to the matrix of all photos
            all_tracers_appeared[i-1,:,:] = nptracers_ta

            tracers_disappeared = gpd.read_file(path_out + "\\" + str(time)+ "s_tracers_td.shp")
            tracers_disappeared['Area'] = tracers_disappeared.area
            # drop the polygons where there aren't tracers_disappeared
            tracers_disappeared.drop(tracers_disappeared.index[tracers_disappeared.area == max(tracers_disappeared.area)], inplace = True)
            tracers_disappeared.drop(tracers_disappeared.index[tracers_disappeared.area < area_threshold], inplace = True)
            tracers_disappeared = tracers_disappeared.overlay(maskdis, how = 'difference')
            tracers_disappeared = tracers_disappeared.drop(columns=['geometry'])
            tracers_disappeared = tracers_disappeared.drop(columns=['FID'])
            nptracers_td = tracers_disappeared.to_numpy()
            all_tracers_disappeared_areas=np.append(all_tracers_disappeared_areas,nptracers_td)
            while nptracers_td.shape != (ntracmax,1):
                newrow = np.array([np.nan])
                nptracers_td = np.vstack((nptracers_td,newrow))
            # add the dataframe to the matrix of all photos
            all_tracers_disappeared[i-1,:,:] = nptracers_td
            

        i +=1
        time += dt
        
    np.save(path_out + '\\alltracers'+ run +'.npy',all_tracers)
    np.save(path_out + '\\tracers_appeared'+ run +'.npy',all_tracers_appeared)
    np.save(path_out + '\\tracers_disappeared'+ run +'.npy',all_tracers_disappeared)
    np.save(path_out + '\\alltracers_area'+ run +'.npy',all_tracers_areas)
    np.save(path_out + '\\tracers_appeared_area'+ run +'.npy',all_tracers_appeared_areas)
    np.save(path_out + '\\tracers_disappeared_area'+ run +'.npy',all_tracers_disappeared_areas)
    print('########################')
    print(run,' completed') 
    print('########################')        