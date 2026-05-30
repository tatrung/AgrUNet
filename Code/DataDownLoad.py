#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:26:57 2025

@author: tahoangtrung
"""
#GEE PART TO DOWNLOAD IMAGES
import ee
import geemap

import os

import numpy as np
import geopandas as gpd
import json

import rasterio
#from rasterio.merge import merge

from datetime import datetime

import shutil

from multiprocessing import Pool, cpu_count
from functools import partial 

from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

ee.Authenticate()
ee.Initialize(project='ee-tahoangtrung')


def create_polygon(lon_min, lat_min, lon_max, lat_max):
    """Creates an Earth Engine polygon from given bounds."""
    return ee.Geometry.Polygon([
        [[lon_min, lat_min], [lon_min, lat_max], [lon_max, lat_max], [lon_max, lat_min], [lon_min, lat_min]]
    ])

def get_column_label(col_index):
    """
    Convert a zero-indexed column number to a two-letter label.
    For example:
      0 -> "AA"
      1 -> "AB"
      ...
      25 -> "AZ"
      26 -> "BA"
      27 -> "BB"
    """
    first_letter = chr(65 + (col_index // 26))
    second_letter = chr(65 + (col_index % 26))
    return first_letter + second_letter

def generate_grid(lon_start, lon_end, lat_start, lat_end, interval):
    """
    Generates a grid of polygons over the specified geographic region,
    assigning each cell a name such as "AA1", "AA2", "BA1", etc.
    
    Parameters:
      lon_start, lon_end : float
          The starting and ending longitudes.
      lat_start, lat_end : float
          The starting and ending latitudes.
      interval : float
          The cell size (assumed to be the same for both dimensions).
          
    Returns:
      ee.FeatureCollection: A FeatureCollection of grid cells, 
                            each with a property 'cell_name'.
    """
    import ee  # Ensure Earth Engine is initialized
    grid = []
    col_index = 0  # Column counter for naming
    
    lon = lon_start
    while lon < lon_end:
        row_index = 0  # Row counter for naming
        lat = lat_start
        while lat < lat_end:
            # Create a polygon for the grid cell
            cell = create_polygon(lon, lat, lon + interval, lat + interval)
            # Generate cell name: column label + row number (row_index + 1)
            cell_name = get_column_label(col_index) + str(row_index + 1)
            #print(cell_name)
            # Append the feature with the cell name property
            grid.append(ee.Feature(cell, {'cell_name': cell_name}))
            row_index += 1
            lat += interval
        col_index += 1
        lon += interval
    grid_fc = ee.FeatureCollection(grid)
    
    # If you want to export the grid as a shapefile:
    task = ee.batch.Export.table.toDrive(
           collection=grid_fc,
           description='Grid_Export',
           fileFormat='SHP'
      )
    task.start()
    
    return ee.FeatureCollection(grid)

def get_cell_corners_latlon(shapefile_path, cell_name_field, cell_name_list):
    """
    Get top-left and bottom-right lat/lon for a list of cell names from a shapefile.

    Parameters:
        shapefile_path (str): Path to the .shp file.
        cell_name_field (str): Name of the field/column with cell names (e.g., 'cell_id').
        cell_name_list (list): List of cell names (e.g., ['AD1', 'AE2', ...]).

    Returns:
        dict: Mapping from cell name to {
            'top_left': (lat, lon),
            'bottom_right': (lat, lon)
        }
    """
    gdf = gpd.read_file(shapefile_path)

    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")  # Ensure it's in lat/lon (EPSG:4326)

    results = {}

    for name in cell_name_list:
        match = gdf[gdf[cell_name_field] == name]
        if match.empty:
            print(f"Warning: Cell name '{name}' not found.")
            continue

        polygon = match.geometry.values[0]
        minx, miny, maxx, maxy = polygon.bounds

        # Round coordinates
        #top_left = (round(maxy, 1), round(minx, 1))      # (lat, lon)
        #bottom_right = (round(miny, 1), round(maxx, 1))  # (lat, lon)
        
        top_left = (maxy, minx)      # (lat, lon)
        bottom_right = (miny, maxx)  # (lat, lon)

        results[name] = {
            'top_left': top_left,
            'bottom_right': bottom_right
        }

    return results

def refined_lee(img, band_name, geometry):
    """
    Applies Refined Lee filter to a specific band in an image.
    """
    band = img.select(band_name)
    kernel = ee.Kernel.square(radius=3)
    mean = band.reduceNeighborhood(ee.Reducer.mean(), kernel)
    variance = band.reduceNeighborhood(ee.Reducer.variance(), kernel)

    # Estimate noise variance from region
    sample_var = ee.Number(band.reduceRegion(
        reducer=ee.Reducer.variance(),
        geometry=geometry,
        scale=10,
        maxPixels=1e9
    ).get(band_name))

    b = variance.divide(variance.add(sample_var))
    filtered = mean.add(b.multiply(band.subtract(mean)))
    return filtered.rename(band_name + '_denoised')

def download_sentinel1_vv_vh(
    cell_name,
    topleft_lat,
    topleft_lon,
    bottom_right_lat,
    bottom_right_lon,
    start_date, end_date,
    out_dir,
    coverage_threshold, target_months
):
    """
    Downloads the highest-coverage Sentinel-1 VV and VH image per month.

    Args:
        top_left_lat (float): Top-left latitude of region.
        top_left_lon (float): Top-left longitude of region.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        out_dir (str): Output directory for GeoTIFFs.
        coverage_threshold (float): Minimum coverage ratio (0–1) for accepted images.
        target_months (list[int]): List of allowed months (e.g. [1, 4, 6, 9]).
    """
    region = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])
    region_area = region.area().getInfo()

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .sort('system:time_start'))

    os.makedirs(out_dir, exist_ok=True)

    image_list = s1.toList(s1.size())
    num_images = image_list.size().getInfo()
    print(f"🛰️ Found {num_images} Sentinel-1 images in range.")

    best_images = {}

    for i in range(num_images):
        try:
            image = ee.Image(image_list.get(i))
            date_str = image.date().format('YYYY-MM-dd').getInfo()
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt.month not in target_months:
                continue

            month_key = f"{dt.year}-{dt.month:02d}"

            footprint = image.geometry()
            intersection = region.intersection(footprint, ee.ErrorMargin(1))
            intersect_area = intersection.area().getInfo()
            coverage_ratio = intersect_area / region_area

            if coverage_ratio < coverage_threshold:
                continue

            if (month_key not in best_images) or (coverage_ratio > best_images[month_key]['coverage']):
                best_images[month_key] = {
                    'image': image,
                    'date_str': date_str,
                    'coverage': coverage_ratio
                }

        except Exception as e:
            print(f"⚠️ Skipping image {i} due to error: {e}")

    print(f"\n✅ Selected {len(best_images)} best images for months: {sorted(best_images.keys())}\n")

    for month_key, info in sorted(best_images.items()):
        image = info['image']
        date_str = info['date_str']

        for band_name in ['VV', 'VH']:
            band_image = image.select(band_name)
            file_name = f"Sentinel1_{band_name}_{cell_name}_{date_str}.tif"
            file_path = os.path.join(out_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"⏩ Skipping {file_name}, already exists.")
                continue

            try:
                print(f"⬇️ Downloading {band_name} for {month_key}: {file_name}")
                geemap.download_ee_image(
                    image=band_image,
                    filename=file_path,
                    scale=10,
                    region=region,
                    crs='EPSG:4326'
                )
            except Exception as e:
                print(f"❌ Error downloading {file_name}: {e}")

def download_sentinel1_vv_vh_mean(
    cell_name,
    topleft_lat,
    topleft_lon,
    bottom_right_lat,
    bottom_right_lon,
    start_date, end_date,
    out_dir,
    coverage_threshold,  # ignored but kept for compatibility
    target_months
):
    """
    Downloads the mean Sentinel-1 VV and VH image per target month.

    Args:
        top_left_lat (float): Top-left latitude of region.
        top_left_lon (float): Top-left longitude of region.
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        out_dir (str): Output directory for GeoTIFFs.
        coverage_threshold (float): Unused here (kept for compatibility).
        target_months (list[int]): List of allowed months (e.g. [1, 4, 6, 9]).
    """
    region = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])

    os.makedirs(out_dir, exist_ok=True)

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .map(lambda img: img.set('year', ee.Date(img.get('system:time_start')).get('year'))
                            .set('month', ee.Date(img.get('system:time_start')).get('month'))))

    # Get list of year-months that match target months
    dates = s1.aggregate_array('system:time_start').getInfo()
    month_keys = set()
    for millis in dates:
        dt = datetime.utcfromtimestamp(millis / 1000)
        if dt.month in target_months:
            month_keys.add(f"{dt.year}-{dt.month:02d}")
    month_keys = sorted(month_keys)

    print(f"🛰️ Processing monthly mean composites for months: {month_keys}\n")

    for key in month_keys:
        year, month = map(int, key.split('-'))

        # Filter to this month's images
        monthly_collection = s1.filter(ee.Filter.calendarRange(year, year, 'year')) \
                               .filter(ee.Filter.calendarRange(month, month, 'month'))

        if monthly_collection.size().getInfo() == 0:
            print(f"⚠️ No images for {key}, skipping.")
            continue

        # Compute mean image
        mean_image = monthly_collection.mean().clip(region)

        # Get the date string as the first image date (just for naming)
        date_str = f"{year}-{month:02d}-01"

        for band_name in ['VV', 'VH']:
            band_image = mean_image.select(band_name)
            file_name = f"Sentinel1_{band_name}_{cell_name}_{date_str}.tif"
            file_path = os.path.join(out_dir, file_name)

            if os.path.exists(file_path):
                print(f"⏩ Skipping {file_name}, already exists.")
                continue

            try:
                print(f"⬇️ Downloading {band_name} mean image for {key}: {file_name}")
                geemap.download_ee_image(
                    image=band_image,
                    filename=file_path,
                    scale=10,
                    region=region,
                    crs='EPSG:4326'
                )
            except Exception as e:
                print(f"❌ Error downloading {file_name}: {e}")

def download_sentinel1_vv_vh_ReadSHPFileDirect(
    shapefile_path,
    cell_name_field,
    cell_name,
    start_date, end_date,
    out_dir,
    coverage_threshold,
    target_months
):
    """
    Downloads the highest-coverage Sentinel-1 VV and VH image per month using a polygon region from shapefile.
    Use shape as downdload region, No need other function to read 4 corners of the cell

    Args:
        shapefile_path (str): Path to input shapefile.
        cell_name_field (str): Field name in shapefile to match a cell (e.g., 'cell_id').
        cell_name (str): Name of the cell to extract polygon.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        out_dir (str): Output directory for GeoTIFFs.
        coverage_threshold (float): Minimum coverage ratio (0–1).
        target_months (list[int]): Allowed months (e.g., [1, 4, 6, 9]).
    """
    # Load and extract region from shapefile
    gdf = gpd.read_file(shapefile_path)
    polygon = gdf[gdf[cell_name_field] == cell_name]
    if polygon.empty:
        raise ValueError(f"❌ Cell name '{cell_name}' not found in {shapefile_path}")
    geom_json = json.loads(polygon.to_json())["features"][0]["geometry"]
    region = ee.Geometry(geom_json)
    region_area = region.area().getInfo()

    # Filter Sentinel-1 collection
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .sort('system:time_start'))

    os.makedirs(out_dir, exist_ok=True)
    image_list = s1.toList(s1.size())
    num_images = image_list.size().getInfo()
    print(f"🛰️ Found {num_images} Sentinel-1 images in range.")

    best_images = {}

    for i in range(num_images):
        try:
            image = ee.Image(image_list.get(i))
            date_str = image.date().format('YYYY-MM-dd').getInfo()
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt.month not in target_months:
                continue

            month_key = f"{dt.year}-{dt.month:02d}"

            footprint = image.geometry()
            intersection = region.intersection(footprint, ee.ErrorMargin(1))
            intersect_area = intersection.area().getInfo()
            coverage_ratio = intersect_area / region_area

            if coverage_ratio < coverage_threshold:
                continue

            if (month_key not in best_images) or (coverage_ratio > best_images[month_key]['coverage']):
                best_images[month_key] = {
                    'image': image,
                    'date_str': date_str,
                    'coverage': coverage_ratio
                }

        except Exception as e:
            print(f"⚠️ Skipping image {i} due to error: {e}")

    print(f"\n✅ Selected {len(best_images)} best images for months: {sorted(best_images.keys())}\n")

    for month_key, info in sorted(best_images.items()):
        image = info['image']
        date_str = info['date_str']

        for band_name in ['VV', 'VH']:
            band_image = image.select(band_name).clip(region)
            file_name = f"Sentinel1_{band_name}_{cell_name}_{date_str}.tif"
            file_path = os.path.join(out_dir, file_name)

            if os.path.exists(file_path):
                print(f"⏩ Skipping {file_name}, already exists.")
                continue

            try:
                print(f"⬇️ Downloading {band_name} for {month_key}: {file_name}")
                geemap.download_ee_image(
                    image=band_image,
                    filename=file_path,
                    scale=10,
                    region=region,
                    crs='EPSG:4326'
                )
            except Exception as e:
                print(f"❌ Error downloading {file_name}: {e}")

def apply_speckle_filter(image):
    """Applies mean-based speckle noise filter."""
    kernel = ee.Kernel.square(1)
    mean = image.reduceNeighborhood(ee.Reducer.mean(), kernel)
    variance = image.reduceNeighborhood(ee.Reducer.variance(), kernel)
    sample = variance.divide(mean.multiply(mean))
    weight = sample.divide(sample.add(1))
    filtered = mean.add(weight.multiply(image.subtract(mean)))
    return filtered.copyProperties(image, image.propertyNames())

def download_dem_from_gee(cell_name,topleft_lat,topleft_lon,bottom_right_lat,bottom_right_lon, folder):
    """
    Export DEM, slope, and aspect separately from Google Earth Engine to Google Drive.

    Args:
        top_left_lat (float): Top-left latitude.
        top_left_lon (float): Top-left longitude.
        interval_deg (float): Size of bounding box in degrees (both lat and lon).
        folder (str): Local folder where the DEM bands will be saved.
    """
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)
 
    # Define bounding box
    region = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])

    # Load DEM
    dem_image = ee.ImageCollection("COPERNICUS/DEM/GLO30").mosaic().select('DEM').clip(region)
    dem = dem_image.unmask(dem_image.focal_mean(radius=1, units='pixels')).rename('DEM')
    
    dem_SRTM = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region).rename('DEM')
    slope = ee.Terrain.slope(dem_SRTM).rename('Slope')
    


   # Calculate terrain gradient in x and y direction
   # terrain_grad = dem.gradient()
   # dx = terrain_grad.select('x')  # ∂z/∂x
   # dy = terrain_grad.select('y')  # ∂z/∂y
   
    #terrain = ee.Terrain.products(dem)
    #slope = terrain.select('slope') # Slope in degrees
    #slope = ee.Terrain.slope(dem)

    # Calculate slope in degrees: arctangent of the gradient magnitude
    #slope = (dx.pow(2).add(dy.pow(2)).sqrt()).atan().multiply(180.0 / 3.141592653589793)

    # Download each band separately
    filename=os.path.join(folder, f"{cell_name}_DEM.tif")
    
    if os.path.exists(filename):
        print(f"⏩ Skipping {filename}, already exists.")  
    try:
        geemap.download_ee_image(
            image=dem.rename('DEM'),
            filename=filename,
            scale=10,
            region=region,
            crs='EPSG:4326'
        )
    
        geemap.download_ee_image(
            image=slope.rename('Slope'),
            filename=os.path.join(folder, f"{cell_name}_Slope.tif"),
            scale=10,
            region=region,
            crs='EPSG:4326'
        )
    
    except:
        pass
        print("Error while downloading " + cell_name)
    print('finish')

def download_alphaearth_from_gee(cell_name, topleft_lat, topleft_lon, bottom_right_lat, bottom_right_lon, folder, year):
    """
    Download AlphaEarth embedding image from Google Earth Engine to local folder.

    Args:
        cell_name (str): Name prefix for output file.
        topleft_lat (float): Top-left latitude.
        topleft_lon (float): Top-left longitude.
        bottom_right_lat (float): Bottom-right latitude.
        bottom_right_lon (float): Bottom-right longitude.
        folder (str): Local folder where the image will be saved.
        year (int): Year of AlphaEarth embedding (2017–2024).
    """
    # Ensure output directory exists
    os.makedirs(folder, exist_ok=True)

    # Define bounding box
    region = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])

    # Load AlphaEarth embedding (64 bands: A00–A63)
    embeddings = ( ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") 
        .filter(ee.Filter.calendarRange(int(year), int(year), "year")) 
        .filterBounds(region)
     )   
    
    embedding_image = embeddings.mosaic().clip(region)

    # Define filename
    filename = os.path.join(folder, f"{cell_name}_AlphaEarth_{year}.tif")

    if os.path.exists(filename):
        print(f"⏩ Skipping {filename}, already exists.")
        return

    try:
        geemap.download_ee_image(
            image=embedding_image,
            filename=filename,
            scale=10,  # native resolution
            region=region,
            crs="EPSG:4326"
        )
        print(f"✅ Finished downloading {filename}")
    except Exception as e:
        print(f"❌ Error while downloading {cell_name}: {e}")
        
def mask_s2_clouds_prob(image):
    # Join with cloud probability image
    cloud_prob = ee.Image(image.get('cloud_mask')).select('probability')
    is_not_cloud = cloud_prob.lt(40)  # 40% threshold
    return image.updateMask(is_not_cloud).copyProperties(image, ["system:time_start"])

def mask_s2_sr(image):
    """
    Masks clouds and cloud shadows from Sentinel-2 SR image using the SCL band.
    Keeps vegetation, bare soil, water, unclassified.
    """
    scl = image.select('SCL')
    valid = (scl.eq(4)
             .Or(scl.eq(5))
             .Or(scl.eq(6))
             .Or(scl.eq(7)))# 4=veg, 5=barren, 6=water, 7=unclassified

    # Apply mask and scale reflectance to [0,1]
    return image.updateMask(valid).copyProperties(image, ['system:time_start'])

def download_sentinel2_with_indices(
    cell_name,
    topleft_lat,
    topleft_lon,
    bottom_right_lat,
    bottom_right_lon,
    start_date,
    end_date,
    output_dir,
    cloud_threshold=20,
    max_cloud_threshold=60
):
    roi = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])
    
    os.makedirs(output_dir, exist_ok=True)

    start = ee.Date(start_date)
    end = ee.Date(end_date)
    start_prev = start.advance(-1, 'year')
    end_prev = end.advance(-1, 'year')
    

    def get_clean_median_with_prob(start_date, end_date, cloud_threshold):
        # Get Sentinel-2 SR and Cloud Probability collections
    
        s2_sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterDate(start_date, end_date)
                 .filterBounds(roi)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)))
    
        s2_clouds = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                     .filterDate(start_date, end_date)
                     .filterBounds(roi))
    
        # Join by system:index
        joined = ee.Join.saveFirst('cloud_mask').apply(
            primary=s2_sr,
            secondary=s2_clouds,
            condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
        )
    
        # Apply cloud probability masking and create median
        return (ee.ImageCollection(joined)
                .map(mask_s2_clouds_prob)
                .select(["B2", "B3", "B4", "B8", "B11"])
                .median()
                .clip(roi))

    #cloud_threshold = initial_cloud_threshold
    image_found = False
    while cloud_threshold <= max_cloud_threshold:
        print(f"Trying with cloud threshold: {cloud_threshold}%")
        s2_test = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(roi)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold)))
        if s2_test.size().getInfo() > 0:
            image_found = True
            break
        cloud_threshold += 10

    if not image_found:
        raise ValueError("No Sentinel-2 images found within cloud coverage limits.")

    s2_current = get_clean_median_with_prob(start, end, cloud_threshold)
    s2_previous = get_clean_median_with_prob(start_prev, end_prev, cloud_threshold)

    # Create cloud mask from a raw image just for fusion logic
    s2_raw = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterDate(start_date, end_date)
              .filterBounds(roi)
              .select("QA60")
              .median()
              .clip(roi))
    cloud_mask = s2_raw.gt(0)

    # Replace cloudy pixels using previous year's image
    fused = s2_current.where(cloud_mask, s2_previous)

    # Band selection
    blue = fused.select('B2')
    green = fused.select('B3')
    red = fused.select('B4')
    nir = fused.select('B8')
    swir = fused.select('B11')

    # Indices
    indices = {
        "NDVI": nir.subtract(red).divide(nir.add(red)).rename('NDVI'),
        "NDWI": green.subtract(nir).divide(green.add(nir)).rename('NDWI'),
        "UrbanIndex": swir.subtract(nir).divide(swir.add(nir)).rename('UrbanIndex'),
        "NDPI": green.subtract(swir).divide(green.add(swir)).rename('NDPI'),
    }

    # GLCM Entropy from band 8
    entropy_input = nir.multiply(255).divide(10000).toUint8().rename('nir')
    
    glcm = entropy_input.glcmTexture(size=3) 
    
    #print(glcm.bandNames().getInfo())  # Optional: for debugging
    
    glcm_entropy = glcm.select('nir_ent')
    glcm_correlation = glcm.select('nir_corr')
    
    indices["GLCM_Entropy"] = glcm_entropy
    indices["GLCM_Correlation"] = glcm_correlation

    # Combine all outputs
    bands = {
        "B2": blue,
        "B3": green,
        "B4": red,
        "B8": nir,
        "B11": swir
    }
    all_layers = {**bands, **indices}
   

    # Timestamp
    timestamp = s2_test.sort("system:time_start").first().date().format("YYYY-MM").getInfo()

    # Download
    for name, image in all_layers.items():
        file_name = f"{cell_name}_{name}_{timestamp}.tif"
        full_path = os.path.join(output_dir, file_name)
        
        if os.path.exists(full_path):
                print(f"⏩ Skipping {file_name}, already exists.")
                continue
        
        try:
            print(f"Downloading {name} to {full_path} ...")
            geemap.download_ee_image(
                image=image,
                filename=full_path,
                scale=10,
                region=roi,
                crs='EPSG:4326'
            )
        except Exception:
            pass
            print ("Error while downloading Sentinel2 " + name) 
    print("All downloads complete.")

def download_sentinel2_median_with_indices(
    cell_name,
    topleft_lat,
    topleft_lon,
    bottom_right_lat,
    bottom_right_lon,
    start_date,
    end_date,
    output_dir,
    cloud_threshold=20
):
    """
    Download Sentinel-2 bands and spectral indices using MODE composite.
    
    Args:
        cell_name (str): Tile name or identifier for output filenames.
        topleft_lat (float): Latitude of top-left corner of ROI.
        topleft_lon (float): Longitude of top-left corner of ROI.
        bottom_right_lat (float): Latitude of bottom-right corner of ROI.
        bottom_right_lon (float): Longitude of bottom-right corner of ROI.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        output_dir (str): Directory to save downloaded files.
        cloud_threshold (int): Max CLOUDY_PIXEL_PERCENTAGE to filter collection.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    roi = ee.Geometry.Rectangle([
        topleft_lon,
        bottom_right_lat,
        bottom_right_lon,
        topleft_lat
    ])

    # Filter Sentinel-2 collection and take mode
    s2_mode = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(start_date, end_date)
               .filterBounds(roi)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
               .map(mask_s2_sr)
               .select(["B2", "B3", "B4", "B8", "B11"])
               .median()
               .clip(roi))

    # Bands
    blue = s2_mode.select('B2')
    green = s2_mode.select('B3')
    red = s2_mode.select('B4')
    nir = s2_mode.select('B8')
    swir = s2_mode.select('B11')

    # Indices
    indices = {
        "NDVI": nir.subtract(red).divide(nir.add(red)).rename('NDVI'),
        "NDWI": green.subtract(nir).divide(green.add(nir)).rename('NDWI'),
        "UrbanIndex": swir.subtract(nir).divide(swir.add(nir)).rename('UrbanIndex'),
        "NDPI": green.subtract(swir).divide(green.add(swir)).rename('NDPI'),
    }

    # GLCM Entropy/Correlation from band 8 (NIR)
    entropy_input = nir.multiply(255).divide(10000).toUint8().rename('nir')
    glcm = entropy_input.glcmTexture(size=3)
    glcm_entropy = glcm.select('nir_ent')
    glcm_correlation = glcm.select('nir_corr')
    
    indices["GLCM_Entropy"] = glcm_entropy
    indices["GLCM_Correlation"] = glcm_correlation

    # Combine all outputs
    bands = {
        "B2": blue,
        "B3": green,
        "B4": red,
        "B8": nir,
        "B11": swir
    }
    all_layers = {**bands, **indices}

    # Timestamp from first available image
    s2_first = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
                .sort("system:time_start")
                .first())
    
    if s2_first is None:
        raise ValueError("No Sentinel-2 images found for the given period/ROI.")

    timestamp = s2_first.date().format("YYYY-MM").getInfo()

    # Download each band/index
    for name, image in all_layers.items():
        file_name = f"{cell_name}_{name}_{timestamp}.tif"
        full_path = os.path.join(output_dir, file_name)

        if os.path.exists(full_path):
            print(f"⏩ Skipping {file_name}, already exists.")
            continue

        try:
            print(f"⬇️ Downloading {name} to {full_path} ...")
            geemap.download_ee_image(
                image=image,
                filename=full_path,
                scale=10,
                region=roi,
                crs="EPSG:4326"
            )
        except Exception as e:
            print(f"⚠️ Error while downloading {name}: {e}")

    print("✅ All downloads complete.")

def merge_tiffs_from_folders(folder_list, output_path):
    all_tiff_paths = []

    # Collect TIFF paths
    for folder in folder_list:
        for root, dirs, files in os.walk(folder):
            for file in sorted(files):  # optional: ensure consistent order
                if file.lower().endswith((".tif", ".tiff")):
                    all_tiff_paths.append(os.path.join(root, file))

    if not all_tiff_paths:
        raise ValueError("No TIFF files found in the given folders.")

    # Print the order
    print("Stacking TIFF files in the following order (each will be a band):")
    for i, path in enumerate(all_tiff_paths, 1):
        print(f"{i}: {path}")

    # Read the first file to get metadata
    with rasterio.open(all_tiff_paths[0]) as src0:
        meta = src0.meta
        width = src0.width
        height = src0.height
        #dtype = src0.dtypes[0]
        crs = src0.crs
        transform = src0.transform

    # Create output metadata
    meta.update({
        "count": len(all_tiff_paths),  # one band per input file
        "driver": "GTiff",
        "height": height,
        "width": width,
        #"dtype": dtype,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": 0 
    })

    # Write stacked output
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        for idx, path in enumerate(all_tiff_paths):
            with rasterio.open(path) as src:
                data = src.read(1)
                nodata_val = src.nodata
    
                if nodata_val is not None and nodata_val != 0:
                    data = np.where(data == nodata_val, 0, data)  # Replace with output nodata
    
                dst.write(data.astype('float32'), idx + 1)


    print(f"\n✅ Stacked {len(all_tiff_paths)} TIFFs into: {output_path}")
  
def normalize_images_threshold(input_dir, output_dir, norm_value, indices_keywords=('nd', 'index','corr'), clip_indices=True):
    """
    Normalize and clip Sentinel-2 raster images to [0, 1] range.

    Parameters:
    - input_dir: root folder with original .tif files
    - output_dir: target folder to save normalized files
    - norm_value: fixed value to divide each pixel by (e.g. 10000)
    - indices_keywords: substrings to identify files to skip normalization (but still clip if clip_skipped=True)
    - clip_indices: if True, clip indices files to [0, 1] instead of copying raw
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise ValueError("No .tif files found.")

    for path in tqdm(all_files, desc=f"Normalizing and clipping (/ {norm_value})"):
        filename = os.path.basename(path).lower()
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(path) as src:
            img = src.read().astype('float32')
            profile = src.profile
            profile.update(dtype='float32')
            
            if "entropy" in filename.lower():
                try:
                    img = np.clip(img/8.0, 0, 1)
                    print(f"[Entropy-NORM] {filename}")
                except Exception as e:
                    print(f"[ERROR entropy] {filename}: {e}")

            elif any(k in filename for k in indices_keywords):
                if clip_indices:
                    #img = np.clip(img, 0, 1)
                    img = np.clip((img + 1) * 0.5, 0, 1)
                    print(f"[CLIP-ONLY] {filename}")
                else:
                    shutil.copy2(path, out_path)
                    print(f"[SKIP/COPY] {filename}")
                    continue
            else:
                img = img / norm_value
                img = np.clip(img, 0, 1)
                print(f"[NORM] {filename}")

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(img)

    print("\n✅ Normalization and clipping complete.")
      
def normalize_threshold(input_dir, output_dir, norm_value, keyword=None):
    """
    Normalize and clip raster images to [0, 1] range.

    Parameters:
    - input_dir: root folder with original .tif files
    - output_dir: target folder to save normalized files
    - norm_value: fixed value to divide each pixel by (e.g. 10000)
    - keyword: only normalize files containing this substring in their filename;
               other files will be ignored
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise ValueError("No .tif files found.")

    selected_files = []
    if keyword:
        keyword = keyword.lower()
        selected_files = [f for f in all_files if keyword in os.path.basename(f).lower()]
    else:
        selected_files = all_files

    if not selected_files:
        print(f"⚠️ No files found containing keyword '{keyword}'. Nothing to normalize.")
        return

    for path in tqdm(selected_files, desc=f"Normalizing files containing '{keyword}' (/ {norm_value})"):
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(path) as src:
            img = src.read().astype('float32')
            profile = src.profile
            profile.update(dtype='float32')
            
            img = img / norm_value
            img = img.clip(0, 1)  # ensure values stay within [0, 1]

            #print(f"[NORM] {os.path.basename(path)}")

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(img)
  
    print("\n✅ Normalization complete.")
    
def normalize_tile_folder_fullpath(tile_folder_path, output_dir, grid_shapefile, tile_field="tile_id"):
    """
    Normalize all Sentinel-1 images in a single tile folder using its neighbors.
    
    Parameters
    ----------
    tile_folder_path : str
        Full path to the tile folder, e.g., ".../2024/AL10"
    output_dir : str
        Directory to save normalized images.
    grid_shapefile : str
        Path to grid shapefile with tile geometries.
    tile_field : str
        Column in shapefile that contains tile names (matching folder names).
    """

    if not os.path.isdir(tile_folder_path):
        print(f"Tile folder not found: {tile_folder_path}")
        return

    # Extract tile name and parent directory automatically
    tile_folder_path = os.path.abspath(tile_folder_path)
    parent_dir = os.path.dirname(tile_folder_path)
    tile_name = os.path.basename(tile_folder_path).upper()

    print(f"Processing tile: {tile_name}")
    print(f"Parent directory: {parent_dir}")

    # Load grid shapefile
    grid = gpd.read_file(grid_shapefile)
    grid[tile_field] = grid[tile_field].astype(str).str.upper()
    grid = grid.set_index(tile_field)

    if tile_name not in grid.index:
        print(f"Tile {tile_name} not found in grid shapefile.")
        return

    tile_geom = grid.loc[tile_name].geometry

    # Find neighbors (touching/intersecting polygons)
    neighbors = grid[grid.geometry.touches(tile_geom) | grid.geometry.intersects(tile_geom)]
    neighbors = neighbors.drop(tile_name, errors="ignore")
    neighbor_names = [n for n in neighbors.index if os.path.isdir(os.path.join(parent_dir, n))]

    print(f"[{tile_name}] Found {len(neighbor_names)} neighbors: {neighbor_names}")

    # Collect min/max values for VV and VH across tile + neighbors
    stats = {"VV": [], "VH": []}
    search_folders = [tile_folder_path] + [os.path.join(parent_dir, n) for n in neighbor_names]

    for folder in search_folders:
        for f in os.listdir(folder):
            if not f.lower().endswith(".tif"):
                continue
            fname_lower = f.lower()
            band = "VV" if "vv" in fname_lower or "hh" in fname_lower else "VH" if "vh" in fname_lower or "hv" in fname_lower else None
            if band is None:
                continue
            with rasterio.open(os.path.join(folder, f)) as src:
                arr = src.read().astype("float32")
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    stats[band].append((arr.min(), arr.max()))

    # Compute global min/max per band
    band_ranges = {}
    for band in ["VV", "VH"]:
        if stats[band]:
            band_ranges[band] = (min(v[0] for v in stats[band]), max(v[1] for v in stats[band]))
        else:
            band_ranges[band] = (-60, 10) if band=="VV" else (-25, 0)

    print(f"VV range: {band_ranges['VV']}, VH range: {band_ranges['VH']}")

    # Normalize all .tif files in current tile folder
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(tile_folder_path):
        if not f.lower().endswith(".tif"):
            continue
        in_fp = os.path.join(tile_folder_path, f)
        out_fp = os.path.join(output_dir, f)

        fname_lower = f.lower()
        band = "VV" if "vv" in fname_lower or "hh" in fname_lower else "VH" if "vh" in fname_lower or "hv" in fname_lower else None
        if band is None:
            continue

        with rasterio.open(in_fp) as src:
            arr = src.read().astype("float32")
            vmin, vmax = band_ranges[band]
            arr_norm = (arr - vmin) / (vmax - vmin)
            arr_norm = np.clip(arr_norm, 0, 1).astype("float32")

            profile = src.profile
            profile.update(dtype="float32")

            with rasterio.open(out_fp, "w", **profile) as dst:
                dst.write(arr_norm)

    print(f"✅ Normalized all images in {tile_name} saved to {output_dir}")

def normalize_images_sen1(input_dir, output_dir):
    """
    Normalize Sentinel-1 raster images (.tif)

    Parameters:
    - input_dir: root folder with original .tif files
    - output_dir: target folder to save normalized files
    - norm_value: fixed value to divide each pixel by
    - skip_keywords: tuple of substrings to skip normalization (copied instead)
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                all_files.append(os.path.join(root, f))

    if not all_files:
        raise ValueError("No .tif files found.")

    for path in tqdm(all_files):
        filename = os.path.basename(path).lower()
        rel_path = os.path.relpath(path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with rasterio.open(path) as src:
            img = src.read().astype('float32')
            
            # Normalize from -60 to 10 dB → [0,1]
            img_norm = (img - (-60)) / (10 - (-60)) 
            img_norm = np.clip(img_norm, 0, 1).astype('float32')
            
            profile = src.profile
            profile.update(dtype='float32')

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(img_norm)

        print(f"[NORM] {rel_path}")

    print("\n✅ Normalization complete.")


def main():
    ee.Initialize()
    
    """ Grid creatation parameters"""
    lon_start = 102.0
    lon_end = 109.5
    lat_start = 8.5
    lat_end = 23.4
    interval = 0.5
   
    #grid = generate_grid(lon_start, lon_end, lat_start, lat_end, interval)
    #print("Total grid cells:", grid.size().getInfo())
       
    """ Image download"""  
    
    start_date_AlphaEarth = "2020"
    
    #Recommend month for the Vietnam Mekong Delta
    #start_date_Sen2 = "2024-01-01"
    #end_date_Sen2 = "2024-04-28"
    
    #Recommend month for the Vietnam Northern and Central
    start_date_Sen2 = "2019-11-01"
    end_date_Sen2 = "2020-04-28"
    
    
    
    start_date_Sen1 = "2020-01-01"
    end_date_Sen1 = "2020-12-31"
    coverage_threshold = 0.8
    target_months = [1,3,5,7,9,11]
    
    output_path = "/media/tahoangtrung/data/SatelliteData/"
    
    #PARAMETERS 
    Grid_shapefile_path ="/media/tahoangtrung/data/SpatialData/Grid0_5x_0_5/Grid_Export.shp"
    #Grid_shapefile_path ="/media/tahoangtrung/data/SpatialData/Grid0_1/Grid_Export.shp"
    
    #------For download test image use Grid shape 0.1 degree x 0.1 degree-----
    #Grid_name = []
    #'BN10','BN12','BT28','BQ21','CJ42','BW21','BJ23','CM33','AS139','CN55','BX33','BS16','CH25','CN56','BG3','AS139','BT130','BM127','BS20','CI31'
   
    #------For download classification image use Grid shape 0.5 degree x 0.5 degree-----   
    
    Grid_name = ['AJ7'
                 
    #                              'AF30','AG30','AH30',
    #     'AA29','AB29','AC29','AD29','AE29','AF29','AG29','AH29','AI29','AJ29',
    #     'AA28','AB28','AC28','AD28','AE28','AF28','AG28','AH28','AI28','AJ28',
    #            'AB27','AC27','AD27','AE27','AF27','AG27','AH27','AI27','AJ27','AK27','AL27','AM27',
    #            'AB26','AC26','AD26','AE26','AF26','AG26','AH26','AI26','AJ26','AK26','AL26','AM26',
    #                   'AC25','AD25','AE25','AF25','AG25','AH25','AI25','AJ25','AK25',
    #                                        'AE24','AF24','AG24','AH24','AI24','AJ24',  
    #                                        'AE23','AF23','AG23','AH23','AI23',
    #                                        'AE22','AF22','AG22','AH22',
    #                                               'AF21','AG21','AH21',
    #                                                      'AG20','AH20','AI20',
    #                                                             'AH19','AI19','AJ19',
    #                                                                    'AI18','AJ18','AK18', 
    #                                                                           'AJ17','AK17','AL17',
    #                                                                           'AJ16','AK16','AL16','AM16',
    #                                                                                  'AK15','AL15','AM15','AN15',
    #                                                                                        'AL14','AM14','AN14','AO14',
    #                                                                                         'AL13','AM13','AN13','AO13',
    #                                                                                  'AK12','AL12','AM12','AN12','AO12',
    #                                                                                  'AK11','AL11','AM11','AN11','AO11',
    #                                                                                         'AL10','AM10','AN10','AO10',
    #                                                                                         'AL9','AM9','AN9','AO9',
    #                                                                            'AJ8','AK8','AL8','AM8','AN8','AO8',
    #                                                                 'AH7','AI7','AJ7','AK7','AL7','AM7','AN7','AO7',
    #                                                                 'AH6','AI6','AJ6','AK6','AL6','AM6','AN6','AO6',
    #                                                    'AF5','AG5','AH5','AI5','AJ5','AK5','AL5','AM5','AN5','AO5',
    #                                         'AD4','AE4','AF4','AG4','AH4','AI4','AJ4','AK4',
    #                                                    'AF3','AG3','AH3','AI3','AJ3',
    #                                                    'AF2','AG2','AH2','AI2',
    #                                                    'AF1','AG1'
                                                      ]
    
  
   
    
    #AD27 Loi Sentinel1-thang 5, thang 7 - thay bang thang 3, thang 9 - da sua
    #AD28 Loi Sentinel1 - thang 5, thang 7 - thay bang thang 3, thang 9 - da sua
    #AC27 Loi Sentinel1-thang 5, thang 7 - thay bang thang 3, thang 9 - da sua
    #AC28 Loi Sentinel1-thang 5, thang 7, thang 11 - thay bang thang 3, thang 3, thang 9 - da sua
    #AD29 Loi Sentinel1-thang 5, thang 7, thang 11 - thay bang thang 3, thang 3, thang 9 - da sua
    #AC29 Loi Sentinel1-thang 5, thang 7, thang 11 - thay bang thang 3, thang 3, thang 9 - da sua
    
    
    
    #Get lat, lon of top left and bottom right of a cell
    pos = get_cell_corners_latlon(Grid_shapefile_path,"cell_name",Grid_name)
    
    for name, coords in pos.items():
        
        
        top_left_lat = coords['top_left'][0]
        top_left_lon = coords['top_left'][1]
        bottom_right_lat = coords['bottom_right'][0]
        bottom_right_lon = coords['bottom_right'][1]
        
        
        
        print("Downloading Sentinel 2 images, grid " + name + "...")
        output_folder = output_path + "00_Sentinel2/2020/" + name + "/" 
        
        # Use Mean to combine image, firs use this code, if still error, try the sencond option
        #download_sentinel2_with_indices(name,top_left_lat, top_left_lon, bottom_right_lat,  bottom_right_lon,start_date_Sen2, end_date_Sen2, output_folder, cloud_threshold=20, max_cloud_threshold=60)
        
        # Use Median to combine image, use when too much error on the downloaded images
        download_sentinel2_median_with_indices(name,top_left_lat, top_left_lon, bottom_right_lat,  bottom_right_lon,start_date_Sen2, end_date_Sen2, output_folder, cloud_threshold=40)   
        print("------")
        
                
        """
        print("Downloading Sentinel 1 images, grid " + name + "...")
        output_folder = output_path + "00_Sentinel1/2024/TrainingImage/" + name + "/"
        #download_sentinel1_vv_vh(name,top_left_lat, top_left_lon, bottom_right_lat,  bottom_right_lon,start_date_Sen1, end_date_Sen1,output_folder, coverage_threshold,target_months)
        download_sentinel1_vv_vh_mean( name, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon,start_date_Sen1, end_date_Sen1,output_folder,coverage_threshold,target_months)
        print("------")
        
        
        print("Downloading DEM, grid " + name + "...")
        output_folder = output_path + "00_DEM/2024/TrainingImage/" + name + "/"
        download_dem_from_gee(name,top_left_lat, top_left_lon, bottom_right_lat,  bottom_right_lon, output_folder)
        print("------")
        
        
        
        
        #print("Downloading Alpha Earth, grid " + name + "...")
        #output_folder = output_path + "00_AlphaEarth/2024"
        #download_alphaearth_from_gee(name, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, output_folder, start_date_AlphaEarth)
        
        
        
        #Normalize images       
        try:
           input_sen1= output_path + "00_Sentinel1/2024/TrainingImage/" + name
           norm_sen1= output_path + "00_Sentinel1_norm/2024/TrainingImage/" + name 
        
           # Use fix coeefficient to normalize
           normalize_images_sen1(input_sen1, norm_sen1)
        #  Normalize by statisic coefficient by considering neighbour tiles.
        #  normalize_tile_folder_fullpath(input_sen1, norm_sen1, Grid_shapefile_path, tile_field="cell_name") #Normalize by neighbor value, not fixed value like the above function
        except:
            print("error while normalized Sentinel 1, grid " + name )
        
        try:    
            input_DEM= output_path + "00_DEM/2024/TrainingImage/" + name
            norm_DEM= output_path + "00_DEM_norm/2024/TrainingImage/" + name
          
            normalize_threshold(input_DEM, norm_DEM, 3148, keyword=('DEM'))
       
            normalize_threshold(input_DEM, norm_DEM, 90, keyword=('Slope'))

        except:
            print("error while normalized DEM, grid " + name ) 
      
       """
        try:    
           input_sen2= output_path + "00_Sentinel2/2020/" + name
           norm_sen2= output_path + "00_Sentinel2_norm/2020/" + name
           normalize_images_threshold(input_sen2, norm_sen2, 10000, indices_keywords=('nd', 'index','corr'))
        except:
           print("error while normalized Sentinel 2, grid " + name )
    
        
    #Merge image
    
    for name, coords in pos.items():
        
        folder_Sen1 = output_path + "00_Sentinel1_norm/2020/" + name
        folder_Sen2 = output_path + "00_Sentinel2_norm/2020/" + name
        folder_DEM = output_path + "00_DEM_norm/2020/" + name
        

        folder_list = [folder_Sen1, folder_Sen2, folder_DEM]
       
     
        output_image =  "/media/tahoangtrung/workSSD/" + "S12_" + name + ".tif"
        #output_image =  "/media/tahoangtrung/workSSD/" + "AE_" + name + ".tif"
        
        try:
            print ("Merging " + name + "...")
            merge_tiffs_from_folders(folder_list, output_image)
        except:
            print ("error while merging " + name)
       
         
    
      
if __name__ == "__main__":
    main()
    print('finish')