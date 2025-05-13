#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import tempfile
import shutil
from typing import Tuple
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import gval
from dask.distributed import Client, LocalCluster
from dask import delayed
from pythonjsonlogger import jsonlogger
from osgeo import gdal

# -----------------------------------------------------------------------------
# GLOBAL GDAL CONFIGURATION
# -----------------------------------------------------------------------------
gdal.SetConfigOption("GDAL_CACHEMAX", os.getenv("GDAL_CACHEMAX"))
gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
gdal.SetConfigOption("GDAL_TIFF_DIRECT_IO", "YES")
gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "256")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")
gdal.UseExceptions()
gdal.SetConfigOption("CPL_LOG_ERRORS", "ON")

# -----------------------------------------------------------------------------
# GLOBAL DASK CONFIGURATION
# -----------------------------------------------------------------------------
DASK_CLUST_MAX_MEM = os.getenv("DASK_CLUST_MAX_MEM")

def setup_logger(name="make_agreement") -> logging.Logger:
    """Return a JSON-formatter logger with timestamp+level fields."""
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            json_ensure_ascii=False,
        )
    )
    log.addHandler(handler)
    log.propagate = False
    return log


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


def load_rasters(
    candidate_path: str, benchmark_path: str, log: logging.Logger
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load and preprocess candidate and benchmark rasters."""
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

    # Remap values
    log.info("Remapping raster values")
    candidate_remapped = xr.where(candidate > 0, 2, 1)
    benchmark_remapped = xr.where(benchmark > 0, 2, 0)

    # Restore spatial attributes
    candidate_remapped = candidate_remapped.rio.write_crs(candidate.rio.crs)
    candidate_remapped.rio.write_transform(candidate.rio.transform(), inplace=True)
    candidate_remapped.rio.write_nodata(255, inplace=True)
    benchmark_remapped = benchmark_remapped.rio.write_nodata(255)

    return candidate_remapped, benchmark_remapped


def compute_agreement_map(
    candidate: xr.DataArray, benchmark: xr.DataArray, crosstab_path: str, metrics_table: str, log: logging.Logger
) -> xr.DataArray:
    """Compute the agreement map between candidate and benchmark rasters."""
    log.info("Homogenizing rasters")
    candidate_homog, benchmark_homog = candidate.gval.homogenize(benchmark_map=benchmark)

    log.info("Computing agreement map")
    agreement_map = candidate_homog.gval.compute_agreement_map(
        benchmark_map=benchmark_homog
    )
    print(agreement_map)

    log.info("Computing crosstab_table and writing")
    crosstab_table = agreement_map.gval.compute_crosstab()
    crosstab_table.to_csv(crosstab_path)

    return agreement_map


def write_agreement_map(
    agreement_map: xr.DataArray,
    outpath: str,
    client: Client,
    log: logging.Logger,
) -> None:
    """Write the agreement map to a tiled GeoTIFF using block-wise processing."""
    log.info(f"Writing agreement map to {outpath}")
    output_profile = {
        "driver": "GTiff",
        "height": agreement_map.rio.height,
        "width": agreement_map.rio.width,
        "count": 1,
        "dtype": np.uint8,
        "crs": agreement_map.rio.crs,
        "transform": agreement_map.rio.transform(),
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 8192,
        "blockysize": 8192,
        "nodata": 255,
    }

    tasks = []
    with rasterio.open(outpath, "w", **output_profile) as dst:
        for ij, window in dst.block_windows(1):
            i, j = ij

            @delayed
            def write_window(win, ii, jj):
                block = agreement_map.isel(
                    x=slice(win.col_off, win.col_off + win.width),
                    y=slice(win.row_off, win.row_off + win.height),
                ).compute()
                arr2d = np.squeeze(block.values).astype(np.uint8)
                with rasterio.open(outpath, "r+") as d:
                    d.write(arr2d, 1, window=win)
                return True

            tasks.append(write_window(window, i, j))

    log.info("Submitting write tasks to Dask")
    client.compute(tasks, sync=True)
    log.info(f"Agreement map written to {outpath}")


def main():
    log = setup_logger()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compare two raster datasets.")
    parser.add_argument(
        "--candidate_path",
        required=True,
        help="Path to candidate raster (local or S3)",
    )
    parser.add_argument(
        "--benchmark_path",
        required=True,
        help="Path to benchmark raster (local or S3)",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path for output agreement map (local or S3)",
    )
    parser.add_argument(
        "--crosstab_path",
        required=True,
        help="Path for output crosstab table (local or S3)",
    )
    parser.add_argument(
        "--metrics_path",
        required=True,
        help="Path for output metrics table (local or S3)",
    )

    args = parser.parse_args()
    print(args)

    # Set up Dask cluster
    client, cluster = setup_dask_cluster(log)

    try:
        # Load and preprocess rasters
        candidate, benchmark = load_rasters(
            args.candidate_path, args.benchmark_path, log
        )

        # Compute agreement map
        agreement_map = compute_agreement_map(candidate, benchmark, args.crosstab_path, args.metrics_path, log)

        # Write agreement map to GeoTIFF
        with rasterio.Env(
            GDAL_CACHEMAX=1024,
            GDAL_NUM_THREADS=1,
            GDAL_TIFF_DIRECT_IO="YES",
            GDAL_TIFF_OVR_BLOCKSIZE=256,
            GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
        ):
            write_agreement_map(agreement_map, args.output_path, client, log)

    finally:
        log.info("Shutting down Dask client and cluster")
        client.close()
        cluster.close()
        log.info("Dask shutdown complete")


if __name__ == "__main__":
    main()

# Benchmark Raster
# s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_ble_huc_11090202_cog.tif

# Candidate Raster (HAND-derived)
# s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_hand_huc_11090202_cog.tif


# - Sample Inputs Local - #

# Benchmark Raster
# /efs/fim-data/hand_fim/temp/autoeval/formatted_ble_huc_11090202_cog.tif

# Candidate Raster (HAND-derived)
#/efs/fim-data/hand_fim/temp/autoeval/formatted_hand_huc_11090202_cog.tif


# python make_agreement.py --candidate_path s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_hand_huc_11090202_cog.tif --benchmark_path s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_ble_huc_11090202_cog.tif --output /test/mock_data/huc_11090202_agreement_brad.tif --crosstab_path /test/mock_data/crosstab1.csv --metrics_path /test/mock_data/metrics1.csv
