#!/usr/bin/env python3
"""
Generate test data for agreement_maker tests.
Creates two rasters with stripe patterns and different nodata values to generate different agreement values.
"""
import numpy as np
import os
import json
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Configuration
candidate_file = "candidate_raster.tif"
benchmark_file = "benchmark_raster.tif"
clip_geom_file = "clip_square.json"
clip_gpkg_file = "clip_square.gpkg"

# Create 512x512 arrays
height, width = 512, 512

# Different nodata values to test handling
candidate_nodata = 255
benchmark_nodata = 254

# Define spatial reference system (using Albers Equal Area Conic - EPSG:5070)
crs = CRS.from_epsg(5070)

# Define bounds for EPSG:5070 (centered over CONUS)
# Using meters as units, starting around central US coordinates
x_min = -1000000.0  # Western extent in meters
y_max = 1500000.0  # Northern extent in meters
pixel_size = 30.0  # 30 meter pixels
x_max = x_min + (width * pixel_size)
y_min = y_max - (height * pixel_size)

# Create candidate raster with vertical stripes
# Pattern: columns 0-127=dry(0), 128-255=wet(1), 256-383=dry(0), 384-511=wet(1)
candidate_data = np.zeros((height, width), dtype=np.uint8)
candidate_data[:, 128:256] = 1  # Wet stripe
candidate_data[:, 384:512] = 1  # Wet stripe

# Create benchmark raster with horizontal stripes
# Pattern: rows 0-127=dry(0), 128-255=wet(1), 256-383=dry(0), 384-511=wet(1)
benchmark_data = np.zeros((height, width), dtype=np.uint8)
benchmark_data[128:256, :] = 1  # Wet stripe
benchmark_data[384:512, :] = 1  # Wet stripe

# Add different nodata areas in corners with different nodata values
candidate_data[0:32, 0:32] = candidate_nodata
candidate_data[480:512, 480:512] = candidate_nodata
benchmark_data[0:32, 480:512] = benchmark_nodata
benchmark_data[480:512, 0:32] = benchmark_nodata


def create_raster(filename, data, bounds, crs, nodata_val):
    """Create a GeoTIFF raster file using rasterio."""
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_val,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(data, 1)


# Create candidate and benchmark rasters with different nodata values
bounds = (x_min, y_min, x_max, y_max)
create_raster(candidate_file, candidate_data, bounds, crs, candidate_nodata)
create_raster(benchmark_file, benchmark_data, bounds, crs, benchmark_nodata)

# Create a small clip geometry in upper-left corner (dry-dry area) 
# This avoids all intersection areas where True Positives occur
# Intersection areas are at: rows 128-255 & cols 128-255, rows 128-255 & cols 384-511,
# rows 384-511 & cols 128-255, rows 384-511 & cols 384-511
clip_x = x_min + (60 * pixel_size)  # Well before first wet stripe at col 128
clip_y = y_max - (60 * pixel_size)  # Well before first wet stripe at row 128
square_size = 50 * pixel_size  # Small 50x50 pixel area = 1500m square

# Create polygon geometry in upper-left dry area (only clips TN, preserves all TP areas)
clip_polygon = Polygon(
    [
        (clip_x - square_size / 2, clip_y - square_size / 2),
        (clip_x + square_size / 2, clip_y - square_size / 2),
        (clip_x + square_size / 2, clip_y + square_size / 2),
        (clip_x - square_size / 2, clip_y + square_size / 2),
        (clip_x - square_size / 2, clip_y - square_size / 2),
    ]
)

# Create GeoDataFrame and save as geopackage
gdf = gpd.GeoDataFrame([{"id": 1, "geometry": clip_polygon}], crs="EPSG:5070")
gdf.to_file(clip_gpkg_file, driver="GPKG")

# Create clip geometry list JSON file for the agreement_maker interface
# Use absolute path to ensure it works from any working directory
clip_geoms_list = [os.path.abspath(clip_gpkg_file)]
with open(clip_geom_file, "w") as f:
    json.dump(clip_geoms_list, f)

print("Test data created successfully:")
print(f"- Candidate raster: {candidate_file} (nodata={candidate_nodata})")
print(f"- Benchmark raster: {benchmark_file} (nodata={benchmark_nodata})")
print(f"- Clip geometry: {clip_gpkg_file}")
print(f"- Clip geometry list: {clip_geom_file}")
print("\nExpected agreement patterns:")
print("- True Negatives (0): dry-dry intersections")
print("- False Negatives (1): candidate dry, benchmark wet")
print("- False Positives (2): candidate wet, benchmark dry")
print("- True Positives (3): wet-wet intersections")
print("- Masked (4): areas excluded by clip geometry")
print("- NoData (10): corner areas with nodata values")
