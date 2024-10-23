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
import pandas as pd
import geopandas as gpd
import shutil
from osgeo import gdal,ogr
import re
# from skimage import img_as_ubyte
import imageio.core.util
from skimage import morphology

'''
Run mode:
    run_mode == 1 : single run
    run_mode == 2 : batch process
'''

# SET RUN NAME
# runs = ['q05_1r7']
runs = ['q05_1r1']
# runs = ['q05_1r1', 'q05_1r2', 'q05_1r3', 'q05_1r4', 'q05_1r5', 'q05_1r6', 'q05_1r7', 'q05_1r8', 'q05_1r9', 'q05_1r10', 'q05_1r11', 'q05_1r12']
# runs = ['q07_1r1', 'q07_1r2', 'q07_1r3', 'q07_1r4', 'q07_1r5', 'q07_1r6', 'q07_1r7', 'q07_1r8', 'q07_1r9', 'q07_1r10', 'q07_1r11', 'q07_1r12']
# runs = ['q10_1r1', 'q10_1r2', 'q10_1r3', 'q10_1r4', 'q10_1r5', 'q10_1r6', 'q10_1r7', 'q10_1r8', 'q10_1r9', 'q10_1r10', 'q10_1r11', 'q10_1r12']

# runs = ['q05_1r5']


# Script parameters:
run_mode = 1
# Set working directory
# w_dir = os.path.join(os.getcwd(), 'tracers_displacement')
w_dir = os.path.join(os.getcwd())
input_dir = os.path.join(w_dir, 'input_data')
output_dir = os.path.join(w_dir, 'output_data')

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def silence_imageio_warning(*args, **kwargs):
    pass

#imageio.core.util._precision_warn = silence_imageio_warning

# Set parameters
gmr_thr = 25   # G - R threshold 
dt = 4  #time between photos
thr = 36 #area threshold of a single tracer
ntracmax = 1000
new_ntracmax = 2000
area_threshold = 5
areaperimeter_threshold = 0.5
mask_buffer = 6
tdiff = 4



# =============================================================================
# # LOOP OVER RUNS
# =============================================================================
for run in runs:
    print(run, ' is running...')
    
    time = 0
    i = 0
    path_in = os.path.join(input_dir, 'cropped_images', run)
    path_in_DEM = os.path.join(input_dir,'surveys',run[0:5])
    path_in_DoD = os.path.join(input_dir,'DoDs','DoD_'+run[0:5])
    
    
    
    # Create outputs script directory
    path_out = os.path.join(w_dir, 'output_data', 'output_images', run)
    if os.path.exists(path_out):
        shutil.rmtree(path_out, ignore_errors=False, onerror=None)
        os.mkdir(path_out) 
    else: 
        os.mkdir(path_out)  
        
    run_param = np.loadtxt(os.path.join(input_dir, 'run_param_'+run[0:3]+'.txt'), skiprows=1, delimiter=',')
    
    # =============================================================================
    # POSITION PARAMETERS
    # =============================================================================

    # List input directory files and filter for files that end with .jpg
    files_tot = sorted([f for f in os.listdir(path_in) if f.endswith('.jpg') and not f.startswith('.')])
    files = files_tot
    
    all_tracers = np.zeros((len(files),ntracmax,5))
    all_tracers_appeared = np.zeros((len(files)-1,ntracmax,5))
    all_tracers_disappeared = np.zeros((len(files)-1,ntracmax,5))
    img_old = np.zeros((1440,4288)) 
    
    frame_position = run_param[int(run[6:])-1,1]
    frame_position += 0.44
    scale_to_mm = 0.0016*frame_position + 0.4827
    
    x_center = frame_position*1000 + 1100
    x_0 = x_center - 4288/2*scale_to_mm
    
    y_center = 51 + 622.5*scale_to_mm #51 è la distanza in mm dallo scan laser alla sponda interna
    y_0 = y_center - (2190-750)/2*scale_to_mm
    
    L = 4288*scale_to_mm # photo length in meters [m]

    # Survey pixel dimension
    px_x = 50 # [mm]
    px_y = 5 # [mm]
    # =============================================================================

    
    nDEM = int(run_param[int(run[6:])-1,2])
    nDoD1 = int(run_param[int(run[6:])-1,3])
    nDoD2 = int(run_param[int(run[6:])-1,4])
        
    DEM = np.loadtxt(os.path.join(path_in_DEM, 'matrix_bed_norm_'+run[0:3]+'_1s'+ str(nDEM) +'.txt'),skiprows=8)
    DEM = np.where(DEM==-999, np.nan, DEM)
    DoD = np.loadtxt(os.path.join(path_in_DoD, 'DoD_'+ str(nDoD1) + '-'+ str(nDoD2) + '_filt_fill.txt'))


    array_mask = np.loadtxt(os.path.join(input_dir, 'array_mask.txt'))
    array_mask = np.where(array_mask != -999,1,np.nan)
    if run == 'q10_1r8' or run == 'q10_1r9':
        array_mask = np.loadtxt(os.path.join(w_dir, 'array_mask_reduced.txt'))
        array_mask = np.where(array_mask != -999,1,np.nan)
    
    
    DEM = DEM*array_mask
    
    for file in sorted(files,key = numericalSort):
        print('Time: ', time, 's')
        path = os.path.join(path_in, file) # Build path
        
        img = Image.open(path) # Open image
        img_array = np.asarray(img)    # Convert image in numpy array
          
        # EXTRACT RGB BANDS AND CONVERT AS INT64:
        band_red = img_array[:,:,0]    
        band_red = band_red.astype(np.int64)
        band_green = img_array[:,:,1]
        band_green = band_green.astype(np.int64)
        band_blue = img_array[:,:,2]
        band_blue = band_blue.astype(np.int64)
        
        # EXTRACT THE FLUORESCENT TRACERS - modified 2024/9/12
        img_extract = ((band_green-band_red)>-1)*((band_red+band_green+band_blue)>350)*((band_red+band_green+band_blue)<700)
        img_gmr_filt = np.where(img_extract==1, 1, np.nan)
        
        # FILL SMALL HOLES
        # Create a boolean mask where True represents 1's and False represents np.nan
        mask = ~np.isnan(img_gmr_filt)
        # Remove small holes (False regions, i.e., np.nan) surrounded by True (1's)
        filled_mask = morphology.remove_small_holes(mask, area_threshold=10)
        # Replace np.nan where the holes have been filled with 1
        img_gmr_filt[filled_mask] = 1
        
        # RESCALE IMAGE WITH VALUES FROM 0 TO 255
        img_gmr_filt = np.where(np.logical_not(np.isnan(img_gmr_filt)),255,0)
        img_gmr_filt = img_gmr_filt.astype(np.uint8)
        imageio.imwrite(os.path.join(path_out, str(time) + 's_gmr.png'), img_gmr_filt)

         
        # Make the difference between previous and current image
        # Get the difference between this two and then get appeared and disappeared fluorescent area
        if time >= tdiff and file != sorted(files,key = numericalSort)[-1]:
            
            # COLLECT CONSECUTIVE IMAGES
            img_old = Image.open(os.path.join(path_out, str(time-tdiff) + 's_gmr.png')) # Open image
            img_old_array = np.asarray(img_old)    # Convert image in numpy array
            
            # COMPUTE THE DIFFERENCE
            img_gmr_filt_bool = np.where(img_gmr_filt==255, 1, 0)
            img_old_bool = np.where(img_old_array==255, 1, 0)
            
            diff = img_gmr_filt_bool - img_old_bool
            # print('max(diff): ', np.max(diff), '   min(diff): ', np.min(diff))
            
            
            # img_diff_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_diff.png'), diff)
            tracers_appeared = np.where(diff==1,255,0)
            tracers_appeared = tracers_appeared.astype(np.uint8)
            img_ta_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_ta.png'), tracers_appeared)
            tracers_disappeared = np.where(diff==-1,255,0)
            tracers_disappeared = tracers_disappeared.astype(np.uint8)
            img_td_print = imageio.imwrite(os.path.join(path_out,str(time) + 's_td.png'), tracers_disappeared)
        
          
        # FROM RASTER TO VECTOR
        # get raster data source N.B. you need to have the file on your pc 
        open_image = gdal.Open(os.path.join(path_out, str(time)+ "s_gmr.png"))
        input_band = open_image.GetRasterBand(1)
        if time >= tdiff and file != sorted(files,key = numericalSort)[-1]:
            open_image_ta = gdal.Open(os.path.join(path_out, str(time)+ "s_ta.png"))
            input_band_ta = open_image_ta.GetRasterBand(1)
            open_image_td = gdal.Open(os.path.join(path_out, str(time)+ "s_td.png"))
            input_band_td = open_image_td.GetRasterBand(1)
            
        # create output data source
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            
        # create output file name
        output_shapefile = shp_driver.CreateDataSource(os.path.join(path_out, str(time) + "s_tracers.shp" ))
        new_shapefile = output_shapefile.CreateLayer(path_out, srs = None )
        
        if time >= tdiff and file != sorted(files,key = numericalSort)[-1]:
            output_shapefile_ta = shp_driver.CreateDataSource(os.path.join(path_out, str(time)+ "s_tracers_ta.shp" ))
            new_shapefile_ta = output_shapefile_ta.CreateLayer(path_out, srs = None )    
            output_shapefile_td = shp_driver.CreateDataSource(os.path.join(path_out, str(time)+ "s_tracers_td.shp" ))
            new_shapefile_td = output_shapefile_td.CreateLayer(path_out, srs = None )    
        
        # transform the raster photo in a shp file
        gdal.Polygonize(input_band, None, new_shapefile, -1, [], callback=None)
        new_shapefile.SyncToDisk()
        
        if time >= tdiff and file != sorted(files,key = numericalSort)[-1]:
            gdal.Polygonize(input_band_ta, None, new_shapefile_ta, -1, [], callback=None)
            new_shapefile_ta.SyncToDisk()
            gdal.Polygonize(input_band_td, None, new_shapefile_td, -1, [], callback=None)
            new_shapefile_td.SyncToDisk()
        #output_shapefile.Destroy()
            
        # =============================================================================
        # GEOPANDAS        
        # =============================================================================
        # Open the shp file with geopandas
        tracers = gpd.read_file(os.path.join(path_out, str(time)+ "s_tracers.shp"))
        
        # Make a copy of tracers
        tracers_copy = tracers.copy()
        
        # Add new columns to the copy: Area, Perimeter, and Area/Perimeter ratio
        tracers_copy['Area'] = tracers_copy.area
        tracers_copy['Perimeter'] = tracers_copy.length  # In GeoPandas, 'length' gives the perimeter of polygons
        tracers_copy['Area_Perimeter_Ratio'] = tracers_copy['Area'] / tracers_copy['Perimeter']

        
        # =============================================================================
        # CALCULATE THE AREA OF POLYGONS
        # drop the polygons where there aren't tracers
        tracers.drop(tracers.index[tracers.area == max(tracers.area)], inplace = True) # 
        tracers.drop(tracers.index[tracers.area < area_threshold], inplace = True) # Area thresholding
        tracers.drop(tracers.index[tracers.area/tracers.length < areaperimeter_threshold], inplace = True) # Area/perimeter thresholding
        mask = gpd.read_file(os.path.join(path_out, "0s_tracers.shp")) # Trim tracers in the initial position
        # drop the polygons where there aren't tracers
        mask.drop(mask.index[mask.area == max(mask.area)], inplace = True)
        mask['geometry'] = mask.geometry.buffer(mask_buffer)
        if file != sorted(files,key = numericalSort)[-1] and len(tracers) != 0:
            tracers = tracers.overlay(mask, how = 'difference')
        tracers['Area'] = tracers.area
        maskdis =  gpd.read_file(os.path.join(path_out, str(time)+ "s_tracers.shp"))
        maskdis.drop(maskdis.index[maskdis.area == max(maskdis.area)], inplace = True)
        maskdis['geometry'] = maskdis.geometry.buffer(mask_buffer)
        
        # =============================================================================
        # CALCULATE THE NUMBER OF TRACERS IN EVERY POLYGON
        tracers['N_tracc'] = tracers['Area'].floordiv(thr)
        tracers['N_tracc'] = tracers['N_tracc'].replace(0,1)
        
        # =============================================================================
        # CALCULATE THE CENTROIDS
        tracers['Centroid'] = tracers.centroid
        # set the centroids as geometry of the Geodataframe
        tracers = tracers.set_geometry('Centroid')
        # drop the polygon column
        tracers = tracers.drop(columns=['geometry'])
        
        # =============================================================================
        # EXPORTING THE NEW SHP FILE OF THE CEONTROIDS
        os.chdir(path_out)
        tracers.to_file(str(time) + "s_centroids.shp")
        # create a dataframe of n tracc, x, y
        if file == sorted(files,key = numericalSort)[-1]:
            frame_position = run_param[int(run[6:])-1,5]
            frame_position += 0.44
            scale_to_mm = 0.0016*frame_position + 0.4827
            x_center = frame_position*1000 + 1100
            x_0 = x_center - 4288/2*scale_to_mm
            #y_center = 590 + 684*scale_to_mm
            y_center = 51 + 622.5*scale_to_mm #51 è la distanza in mm dallo scan laser alla sponda interna
            y_0 = y_center - (2190-750)/2*scale_to_mm
            L = 4288*scale_to_mm # photo length in meters [m]
        tracers['x']    = tracers['Centroid'].x
        tracers['x']    = tracers['x'].mul(scale_to_mm)
        tracers['x']    = tracers['x'].add(x_0)
        tracers['y']    = tracers['Centroid'].y
        tracers['y']    = tracers['y'].mul(scale_to_mm)
        tracers['y']    = tracers['y'].add(y_0)
        tracers['z']    = DEM[(tracers['y'].div(px_y)).astype(int),tracers['x'].div(px_x).astype(int)]
        tracers['zDoD'] = DoD[tracers['y'].div(px_y).astype(int),(tracers['x'].div(px_x)).astype(int)]
        tracers         = tracers.drop(columns=['FID'])
        tracers         = tracers.drop(columns=['Area'])
        tracers         = tracers.drop(columns=['Centroid']) 
        tracers         = tracers.round(1)
        if file == sorted(files,key = numericalSort)[-2]:
            nptracers_last = tracers.to_numpy()
        nptracers = tracers.to_numpy()
        if file == sorted(files,key = numericalSort)[-1]:
            nptracers = np.vstack((nptracers_last,nptracers)) 
        while nptracers.shape != (ntracmax,5):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
            nptracers = np.vstack((nptracers,newrow))
        # add the dataframe to the matrix of all photos
        all_tracers[i,:,:] = nptracers  
        
        
        if time >= tdiff and file != sorted(files, key=numericalSort)[-1]:
    
            # =============================================================================
            # APPEARED TRACERS      
            # =============================================================================
            maskapp = gpd.read_file(os.path.join(path_out, str(time-tdiff)+ "s_tracers.shp"))
            maskapp.drop(maskapp.index[maskapp.area == max(maskapp.area)], inplace=True)
            maskapp['geometry'] = maskapp.geometry.buffer(mask_buffer)
        
            tracers_appeared = gpd.read_file(os.path.join(path_out, str(time)+ "s_tracers_ta.shp"))
            tracers_appeared['Area'] = tracers_appeared.area
            tracers_appeared.drop(tracers_appeared.index[tracers_appeared.area == max(tracers_appeared.area)], inplace=True)
            tracers_appeared.drop(tracers_appeared.index[tracers_appeared.area < area_threshold], inplace=True)  # Area thresholding
            tracers_appeared.drop(tracers_appeared.index[tracers_appeared.area / tracers_appeared.length > areaperimeter_threshold], inplace=True)  # Area/perimeter thresholding
        
            if not tracers_appeared.empty and not maskapp.empty:
                # =============================================================================
                # COMPUTE THE NUMBER OF APPEARED TRACERS
                tracers_appeared = tracers_appeared.overlay(maskapp, how='difference')
                tracers_appeared['N_tracc'] = tracers_appeared['Area'].floordiv(thr)
                tracers_appeared['N_tracc'] = tracers_appeared['N_tracc'].replace(0, 1)
        
                # =============================================================================
                # CALCULATE THE CENTROIDS
                tracers_appeared['Centroid'] = tracers_appeared.centroid
                tracers_appeared = tracers_appeared.set_geometry('Centroid')
                tracers_appeared = tracers_appeared.drop(columns=['geometry'])
                
                # =============================================================================
                # EXPORTING THE NEW SHP FILE OF THE CENTROIDS
                tracers_appeared.to_file(str(time) + "s_centroids_ta.shp")
        
                # Create a dataframe of n tracc, x, y
                tracers_appeared['x'] = tracers_appeared['Centroid'].x * scale_to_mm + x_0
                tracers_appeared['y'] = tracers_appeared['Centroid'].y * scale_to_mm + y_0
                tracers_appeared['z'] = DEM[tracers_appeared['y'].div(px_y).astype(int), tracers_appeared['x'].div(px_x).astype(int)]
                tracers_appeared['zDoD'] = DoD[tracers_appeared['y'].div(px_y).astype(int), tracers_appeared['x'].div(px_x).astype(int)]
                tracers_appeared = tracers_appeared.drop(columns=['FID', 'Area', 'Centroid'])
                tracers_appeared = tracers_appeared.round(1)
                nptracers_ta = tracers_appeared.to_numpy()
        
                while nptracers_ta.shape != (ntracmax, 5):
                    newrow = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    nptracers_ta = np.vstack((nptracers_ta, newrow))
        
                all_tracers_appeared[i, :, :] = nptracers_ta
        
            else:
                print(f"No tracers appeared at time {time} or maskapp is empty.")
                all_tracers_appeared[i, :, :] = np.full((ntracmax, 5), np.nan)
        
            # =============================================================================
            # DISAPPEARED TRACERS        
            # =============================================================================
            tracers_disappeared = gpd.read_file(os.path.join(path_out, str(time)+ "s_tracers_td.shp"))
            tracers_disappeared['Area'] = tracers_disappeared.area
            tracers_disappeared.drop(tracers_disappeared.index[tracers_disappeared.area == max(tracers_disappeared.area)], inplace=True)
            tracers_disappeared.drop(tracers_disappeared.index[tracers_disappeared.area < area_threshold], inplace=True)
            tracers_disappeared.drop(tracers_disappeared.index[tracers_disappeared.area / tracers_disappeared.length > areaperimeter_threshold], inplace=True)
        
            if not tracers_disappeared.empty and not maskdis.empty:
                tracers_disappeared = tracers_disappeared.overlay(maskdis, how='difference')
                tracers_disappeared['N_tracc'] = tracers_disappeared['Area'].floordiv(thr)
                tracers_disappeared['N_tracc'] = tracers_disappeared['N_tracc'].replace(0, 1)
        
                # Calculate the centroids
                tracers_disappeared['Centroid'] = tracers_disappeared.centroid
                tracers_disappeared = tracers_disappeared.set_geometry('Centroid')
                tracers_disappeared = tracers_disappeared.drop(columns=['geometry'])
        
                # Exporting the new shapefile of the centroids
                tracers_disappeared.to_file(str(time) + "s_centroids_td.shp")
        
                # Create a dataframe of n tracc, x, y
                tracers_disappeared['x'] = tracers_disappeared['Centroid'].x * scale_to_mm + x_0
                tracers_disappeared['y'] = tracers_disappeared['Centroid'].y * scale_to_mm + y_0
                tracers_disappeared['z'] = DEM[tracers_disappeared['y'].div(px_y).astype(int), tracers_disappeared['x'].div(px_x).astype(int)]
                tracers_disappeared['zDoD'] = DoD[tracers_disappeared['y'].div(px_y).astype(int), tracers_disappeared['x'].div(px_x).astype(int)]
                tracers_disappeared = tracers_disappeared.drop(columns=['FID', 'Area', 'Centroid'])
                tracers_disappeared = tracers_disappeared.round(1)
                nptracers_td = tracers_disappeared.to_numpy()
        
                while nptracers_td.shape != (ntracmax, 5):
                    newrow = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                    nptracers_td = np.vstack((nptracers_td, newrow))
        
                all_tracers_disappeared[i, :, :] = nptracers_td
        
            else:
                print(f"No tracers disappeared at time {time} or maskdis is empty.")
                all_tracers_disappeared[i, :, :] = np.full((ntracmax, 5), np.nan)

            

        i +=1
        time += dt
        
    np.save(os.path.join(path_out,'alltracers_'+ run +'.npy'),all_tracers)
    np.save(os.path.join(path_out, 'tracers_appeared_'+ run +'.npy'),all_tracers_appeared)
    np.save(os.path.join(path_out, 'tracers_disappeared_'+ run +'.npy'),all_tracers_disappeared)
    
    print('########################')
    print(run,' extraction completed') 
    print('########################')  
    
    
    # =============================================================================
    # CONVERT NPY STACK IN TRACER MOVEMENT     
    # =============================================================================
    # At each tracer spatial coordinate is substracted the coordinate of the
    # centroid of the feeding position to have te actual traveled distance
    
    frame_position = run_param[int(run[6:])-1,1]
    
    frame_position += 0.44
    scale_to_mm = 0.0016*frame_position + 0.4827
    x_center = frame_position*1000 + 1100
    x_0 = x_center - 4288/2*scale_to_mm

    feeding = gpd.read_file(os.path.join(path_out, "0s_tracers.shp"))
    feeding.drop(feeding.index[feeding.area == max(feeding.area)], inplace = True)
    feeding.drop(feeding.index[feeding.area < area_threshold], inplace = True)
    feeding['Centroid'] = feeding.centroid
    feeding = feeding.set_geometry('Centroid')
    feeding = feeding.drop(columns=['geometry'])  
    feeding['x'] = feeding['Centroid'].x
    feeding['x'] = feeding['x'].mul(scale_to_mm)
    feeding['x']= feeding['x'].add(x_0)
    
    x_start = min(feeding.x)
            
    new_tracers = np.zeros((len(all_tracers),new_ntracmax,4))
    new_tracers_appeared = np.zeros((len(all_tracers_appeared),new_ntracmax,4))
    new_tracers_disappeared = np.zeros((len(all_tracers_disappeared),new_ntracmax,4))
    
    for i in range(len(all_tracers)):
        page = np.empty((0,4))
        for j in range(ntracmax-1):
            ntrac = all_tracers[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([all_tracers[i,j,1]-x_start,all_tracers[i,j,2],all_tracers[i,j,3],all_tracers[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (new_ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers[i,:,:] = page
    
    for i in range(len(all_tracers_appeared)):
        page = np.empty((0,4))
        for j in range(ntracmax-1):
            ntrac = all_tracers_appeared[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([all_tracers_appeared[i,j,1]-x_start,all_tracers_appeared[i,j,2],all_tracers_appeared[i,j,3],all_tracers_appeared[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (new_ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers_appeared[i,:,:] = page
    
     
    for i in range(len(all_tracers_disappeared)):
        page = np.empty((0,4))
        for j in range(ntracmax-1):
            ntrac = all_tracers_disappeared[i,j,0]
            if np.isnan(ntrac) == True:
                break
            line = np.array([all_tracers_disappeared[i,j,1]-x_start,all_tracers_disappeared[i,j,2],all_tracers_disappeared[i,j,3],all_tracers_disappeared[i,j,4]])
            for trac in range(int(ntrac)):
                page = np.vstack((page,line))
        while page.shape != (new_ntracmax,4):
            newrow = np.array([np.nan, np.nan, np.nan, np.nan])
            page = np.vstack((page,newrow))
        new_tracers_disappeared[i,:,:] = page    
         
    np.save(os.path.join(path_out, 'tracers_reduced_'+ run +'.npy'),new_tracers)
    np.save(os.path.join(path_out, 'tracers_appeared_reduced_'+ run +'.npy'),new_tracers_appeared)
    np.save(os.path.join(path_out, 'tracers_disappeared_reduced_'+ run +'.npy'),new_tracers_disappeared)
    
    
    print('########################')
    print(run,' reduction completed') 
    print('########################')    

    
    # =============================================================================
    # COMPUTE THE NUMBER OF TRACERS THAT STOPPED
    # =============================================================================
    tracers_appeared_stopped = np.zeros((len(all_tracers_appeared),new_ntracmax,4))
    tracers_disappeared_stopped = np.zeros((len(all_tracers_disappeared),new_ntracmax,4))
    
    for i in range(len(all_tracers_appeared)-1):
        dict_tracers_appeared = {'x':new_tracers_appeared[i,:,0],'y':new_tracers_appeared[i,:,1],'z':new_tracers_appeared[i,:,2],'z_DoD':new_tracers_appeared[i,:,3]}
        df_tracers_app = pd.DataFrame(dict_tracers_appeared)
        df_tracers_app = df_tracers_app.dropna() 
        dict_tracers_disappeared = {'x':new_tracers_disappeared[i+1,:,0],'y':new_tracers_disappeared[i+1,:,1],'z':new_tracers_disappeared[i+1,:,2],'z_DoD':new_tracers_disappeared[i+1,:,3]}
        df_tracers_dis = pd.DataFrame(dict_tracers_disappeared)
        df_tracers_dis = df_tracers_dis.dropna() 
        int_df = pd.merge(df_tracers_dis, df_tracers_app, how ='inner')
        
        df_tracers_app = df_tracers_app[~df_tracers_app['x'].isin(int_df['x'])]
        tracers_app = df_tracers_app.to_numpy()
        while tracers_app.shape != (new_ntracmax,4):
              newrow = np.array([np.nan, np.nan, np.nan, np.nan])
              tracers_app = np.vstack((tracers_app,newrow))
        tracers_appeared_stopped[i,:,:] = tracers_app 
        
        df_tracers_dis = df_tracers_dis[~df_tracers_dis['x'].isin(int_df['x'])]
        tracers_dis = df_tracers_dis.to_numpy()
        while tracers_dis.shape != (new_ntracmax,4):
              newrow = np.array([np.nan, np.nan, np.nan, np.nan])
              tracers_dis = np.vstack((tracers_dis,newrow))
        tracers_disappeared_stopped[i+1,:,:] = tracers_dis 
    tracers_appeared_stopped[-1,:,:] = new_tracers_appeared[-1,:,:] 
    tracers_disappeared_stopped[0,:,:] = new_tracers_disappeared[0,:,:] 
    
    np.save(os.path.join(path_out, 'tracers_appeared_reduced_stopped_'+ run +'.npy'),tracers_appeared_stopped)
    np.save(os.path.join(path_out, 'tracers_disappeared_reduced_stopped_'+ run +'.npy'),tracers_disappeared_stopped)
    
    
    print('########################')
    print(run,' completed') 
    print('########################') 

    print('\n\n\n')