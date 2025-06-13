#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import sys
from typing import Tuple

import fsspec
import geopandas as gpd
import gval
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from dask import delayed
from dask.distributed import Client, LocalCluster
from fsspec.core import url_to_fs
from rasterio.env import Env
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import LZWProfile

from utils.logging import setup_logger

# GLOBAL GDAL CONFIGURATION (via rasterio)
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["GDAL_TIFF_DIRECT_IO"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "TRUE"
os.environ["CPL_LOG_ERRORS"] = "ON"

# GLOBAL DASK CONFIGURATION
DASK_CLUST_MAX_MEM = os.getenv("DASK_CLUST_MAX_MEM")


JOB_ID = "agreement_maker"


def open_file(path: str, mode: str = "rb"):
    """
    Open a local or remote file (s3://, gcs://, http://, etc.) via fsspec.
    Returns a file-like object.
    """
    fs, fs_path = url_to_fs(path)
    return fs.open(fs_path, mode)


def to_vsi(path: str) -> str:
    """
    Convert a standard file path or S3 path to a GDAL VSI path. This allows us to feed in either a standard filesystem path or an S3 object path to GDAL and have the script be able to work with either without the user having to think about it.
    """
    if path.lower().startswith("s3://"):
        return "/vsis3/" + path[5:]
    return path


def setup_dask_cluster(log: logging.Logger) -> Tuple[Client, LocalCluster]:
    """Set up a local Dask cluster and return the client and cluster."""
    log.info("Starting Dask local cluster")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        memory_limit=DASK_CLUST_MAX_MEM,
    )
    client = Client(cluster)
    log.info(f"Dask dashboard link: {client.dashboard_link}")
    return client, cluster


def load_rasters(candidate_path: str, benchmark_path: str, log: logging.Logger) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load candidate and benchmark rasters without remapping."""
    log.info(f"Loading candidate raster: {candidate_path}")
    candidate = rxr.open_rasterio(
        candidate_path,
        mask_and_scale=True,
        chunks={"x": 2048, "y": 2048},
        lock=False,
    )
    log.info(f"Loading benchmark raster: {benchmark_path}")
    benchmark = rxr.open_rasterio(
        benchmark_path,
        mask_and_scale=True,
        chunks={"x": 2048, "y": 2048},
        lock=False,
    )

    # Handle nodata values by converting them to a consistent nodata value (10)
    log.info("Processing nodata values")
    candidate.data = xr.where(candidate == candidate.rio.nodata, 10, candidate)
    candidate = candidate.rio.write_nodata(10)
    benchmark.data = xr.where(benchmark == benchmark.rio.nodata, 10, benchmark)
    benchmark = benchmark.rio.write_nodata(10)

    return candidate, benchmark


def process_clip_geometries(clip_geoms_path: str, log: logging.Logger) -> dict:
    """Process clip geometries from JSON file and return mask dictionary."""
    if not clip_geoms_path:
        return {}

    log.info(f"Loading clip geometries from {clip_geoms_path}")
    with open_file(clip_geoms_path, "rt") as f:
        clip_geoms = json.load(f)

    # Convert to mask dictionary format similar to the reference function
    mask_dict = {}
    for i, geom_path in enumerate(clip_geoms):
        mask_dict[f"clip_layer_{i}"] = {
            "path": geom_path,
            "operation": "exclude",  # Default to exclude operation
            "buffer": None,
        }

    return mask_dict


def apply_exclusion_masks(candidate: xr.DataArray, mask_dict: dict, log: logging.Logger) -> gpd.GeoDataFrame:
    """Apply exclusion masks and return combined mask geometry."""
    all_masks_df = None

    for poly_layer in mask_dict:
        operation = mask_dict[poly_layer]["operation"]

        if operation == "exclude":
            poly_path = mask_dict[poly_layer]["path"]
            buffer_val = 0 if mask_dict[poly_layer]["buffer"] is None else mask_dict[poly_layer]["buffer"]

            log.info(f"Processing exclusion mask: {poly_layer}")

            # Read mask bounds with candidate boundary box
            poly_all = gpd.read_file(poly_path, bbox=candidate.rio.bounds())

            # Make sure features are present in bounding box area before projecting
            if poly_all.empty:
                log.warning(f"No features found in bounding box for {poly_layer}")
                del poly_all
                gc.collect()
                continue

            # Project layer to reference crs
            poly_all_proj = poly_all.to_crs(candidate.rio.crs)

            # Buffer if buffer val exists
            if buffer_val != 0:
                poly_all_proj = poly_all_proj.buffer(buffer_val)

            if all_masks_df is not None:
                all_masks_df = pd.concat([all_masks_df, poly_all_proj])
            else:
                all_masks_df = poly_all_proj

            del poly_all, poly_all_proj
            gc.collect()

    return all_masks_df


def compute_agreement_map(
    candidate: xr.DataArray,
    benchmark: xr.DataArray,
    metrics_path: str,
    clip_geoms_path: str,
    log: logging.Logger,
) -> xr.DataArray:
    """Compute the agreement map between candidate and benchmark rasters using pairing dictionary."""

    # Define pairing dictionary for binary rasters (0=dry, 1=wet, 10=nodata)
    # Agreement map encoding: 0=TN, 1=FN, 2=FP, 3=TP, 4=Masked, 10=NoData
    pairing_dictionary = {
        (0, 0): 0,  # True Negative: both dry
        (0, 1): 1,  # False Negative: candidate dry, benchmark wet
        (0, 10): 10,  # NoData
        (1, 0): 2,  # False Positive: candidate wet, benchmark dry
        (1, 1): 3,  # True Positive: both wet
        (1, 10): 10,  # NoData
        (4, 0): 4,  # Masked
        (4, 1): 4,  # Masked
        (4, 10): 10,  # NoData
        (10, 0): 10,  # NoData
        (10, 1): 10,  # NoData
        (10, 10): 10,  # NoData
    }

    # Process clip geometries if provided
    mask_dict = process_clip_geometries(clip_geoms_path, log)

    # Apply exclusion masks
    all_masks_df = apply_exclusion_masks(candidate, mask_dict, log) if mask_dict else None

    log.info("Homogenizing rasters")
    c_aligned, b_aligned = candidate.gval.homogenize(benchmark_map=benchmark, target_map="candidate")

    # Clean up original rasters
    del candidate, benchmark
    gc.collect()

    log.info("Computing agreement map using pairing dictionary")
    agreement_map = c_aligned.gval.compute_agreement_map(
        benchmark_map=b_aligned,
        comparison_function="pairing_dict",
        pairing_dict=pairing_dictionary,
    )

    # Convert nan values to 10 (nodata) to match our encoding
    agreement_map = agreement_map.where(~np.isnan(agreement_map), 10)

    # Set pairing dictionary as attribute for later use by gval functions
    agreement_map.attrs["pairing_dictionary"] = pairing_dictionary

    # Clean up aligned rasters
    del c_aligned, b_aligned
    gc.collect()

    # Keep original agreement map for reference
    agreement_map_og = agreement_map.copy()

    # Store original nodata mask before changing nodata value
    original_nodata_mask = agreement_map == 10

    # Set nodata to 4 for clipping (clipped areas will become 4)
    agreement_map.rio.write_nodata(4, inplace=True)

    # Apply masking if exclusion masks are present
    if all_masks_df is not None:
        log.info("Applying exclusion masks to agreement map")
        # Clip the agreement map (clipped areas become nodata value 4)
        agreement_map = agreement_map.rio.clip(all_masks_df["geometry"], invert=True)
        # Restore original nodata areas (corners, etc.) to value 10 using coordinate selection
        # This ensures original nodata remains 10, while clipped areas stay 4
        agreement_map.data = xr.where(
            agreement_map_og.sel({"x": agreement_map.coords["x"], "y": agreement_map.coords["y"]}) == 10,
            10,
            agreement_map,
        )

    # Preserve the pairing dictionary attribute
    if hasattr(agreement_map_og, "attrs") and "pairing_dictionary" in agreement_map_og.attrs:
        agreement_map.attrs["pairing_dictionary"] = agreement_map_og.attrs["pairing_dictionary"]

    log.info("Computing crosstab table for metrics")
    crosstab_table = agreement_map.gval.compute_crosstab()

    # Only compute and write metrics if metrics_path is provided
    if metrics_path:
        log.info("Computing metrics table and writing")
        metrics_table = crosstab_table.gval.compute_categorical_metrics(
            positive_categories=[1], negative_categories=[0], metrics="all"
        )

        # Write metrics table using fsspec for S3 compatibility
        with open_file(metrics_path, "wt") as f:
            metrics_table.to_csv(f)

        # Clean up metrics table
        del metrics_table

    # Clean up
    del crosstab_table
    gc.collect()

    return agreement_map


def write_agreement_map(
    agreement_map: xr.DataArray,
    outpath: str,
    client: Client,
    block_size: int,
    log: logging.Logger,
) -> None:
    log.info(f"Writing agreement map to {outpath}")

    # Create temporary output path for initial write
    temp_outpath = outpath + "_temp.tif"

    # use rasterio to write agreement map (better for large rasters)
    tasks = []
    output_profile = {
        "driver": "GTiff",
        "height": agreement_map.rio.height,
        "width": agreement_map.rio.width,
        "count": 1,
        "dtype": agreement_map.dtype,
        "crs": agreement_map.rio.crs,
        "transform": agreement_map.rio.transform(),
        "compress": "LZW",
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
        "nodata": 10,  # Updated nodata value
    }

    # Write data block by block
    with rasterio.open(temp_outpath, "w", **output_profile) as dst:
        for ij, window in dst.block_windows(1):
            i, j = ij

            @delayed
            def write_window(win, ii, jj):
                block = agreement_map.isel(
                    x=slice(win.col_off, win.col_off + win.width),
                    y=slice(win.row_off, win.row_off + win.height),
                ).compute()  # only compute this block!

                arr2d = np.squeeze(block.values).astype("uint8")
                with rasterio.open(temp_outpath, "r+") as d:
                    d.write(arr2d, 1, window=win)
                return True

            tasks.append(write_window(window, i, j))

    client.compute(tasks, sync=True)

    # Convert to COG using rio-cogeo
    log.info("Converting to Cloud Optimized GeoTIFF")

    # Configure COG profile
    cog_profile = LZWProfile().data.copy()
    cog_profile.update(
        {
            "BLOCKSIZE": 512,
            "OVERVIEW_RESAMPLING": "nearest",
        }
    )

    # Convert to COG
    cog_translate(temp_outpath, outpath, cog_profile, overview_level=4, overview_resampling="nearest", quiet=True)

    os.remove(temp_outpath)
    log.info("Agreement map write completed")


def main():
    log = setup_logger(JOB_ID)

    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Compare two raster datasets.")
    p.add_argument(
        "--fim_type",
        required=True,
        choices=["depth", "extent"],
        help="Specifies whether agreement is based on spatial 'extent' overlap (binary) or potentially 'depth' values. Influences output raster format.",
    )
    p.add_argument(
        "--candidate_path",
        required=True,
        help="Path to candidate raster (local or S3)",
    )
    p.add_argument(
        "--benchmark_path",
        required=True,
        help="Path to benchmark raster (local or S3)",
    )
    p.add_argument(
        "--output_path",
        required=True,
        help="Path for output agreement map (local or S3)",
    )
    p.add_argument(
        "--metrics_path",
        required=False,
        help="Optional path for output metrics table (local or S3)",
    )
    p.add_argument(
        "--clip_geoms",
        required=False,
        help="Optional path/URI to a JSON file containing paths to geopackage or geojson vector masks used to exclude or include areas in the final agreement raster.",
    )
    p.add_argument(
        "--block_size",
        required=False,
        default="4096",
        help="Block size for writing agreement raster. Default is 4096.",
    )

    args = p.parse_args()

    # Validate block size
    try:
        block_size = int(args.block_size)
        os.environ["GDAL_TIFF_OVR_BLOCKSIZE"] = str(block_size)
    except ValueError:
        log.error(f"Invalid block_size: {args.block_size}. Must be an integer.")
        sys.exit(1)

    # Set up Dask cluster
    client, cluster = setup_dask_cluster(log)

    try:
        # Load and preprocess rasters
        candidate, benchmark = load_rasters(args.candidate_path, args.benchmark_path, log)

        # Compute agreement map
        agreement_map = compute_agreement_map(candidate, benchmark, args.metrics_path, args.clip_geoms, log)

        # Write agreement map to GeoTIFF
        with rasterio.Env():
            # Set nodata to 10 for writing (following reference implementation)
            agreement_map_write = agreement_map.rio.write_nodata(10, encoded=True)
            write_agreement_map(agreement_map_write, args.output_path, client, block_size, log)

        success_outputs = {"output_path": args.output_path}
        if args.metrics_path:
            success_outputs["metrics_path"] = args.metrics_path
        log.success(success_outputs)

    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        sys.exit(1)

    finally:
        log.info("Shutting down Dask client and cluster")
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
