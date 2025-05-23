#!/usr/bin/env python3
import os
import sys
import argparse
import logging
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
    block_size: int,
    log: logging.Logger,
) -> None:
    log.info(f"Writing agreement map to {outpath}")

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
        "nodata": -9999,
    }

    # Write data block by block
    with rasterio.open(outpath, "w", **output_profile) as dst:
        for ij, window in dst.block_windows(1):
            i, j = ij

            @delayed
            def write_window(win, ii, jj):
                block = agreement_map.isel(
                    x=slice(win.col_off, win.col_off + win.width),
                    y=slice(win.row_off, win.row_off + win.height),
                ).compute()  # only compute this block!

                arr2d = np.squeeze(block.values).astype("uint8")
                with rasterio.open(outpath, "r+") as d:
                    d.write(arr2d, 1, window=win)
                return True

            tasks.append(write_window(window, i, j))

    client.compute(tasks, sync=True)


def main():
    log = setup_logger()

    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Compare two raster datasets.")
    p.add_argument(
        "--fim_type",
        required=True,
        choices=["depth", "extent"],
        help="Specifies whether agreement is based on spatial 'extent' overlap (binary) or potentially 'depth' values. Influences output raster format."
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
        "--crosstab_path",
        required=True,
        help="Path for output crosstab table (local or S3)",
    )
    p.add_argument(
        "--metrics_path",
        required=True,
        help="Path for output metrics table (local or S3)",
    )
    p.add_argument(
        "--clip_geoms",
        required=False,
        help="Optional path/URI to a JSON file containing paths to geopackage or geojson vector masks used to exclude or include areas in the final agreement raster."
    )
    p.add_argument(
        "--block_size",
        required=False,
        default="4096",
        help="Block size for writing agreement raster. Default is 4096.",
    )

    args = p.parse_args()

    # Validate and set GDAL block size
    try:
        block_size = int(args.block_size)
        gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", str(block_size))
    except ValueError:
        log.error(f"Invalid block_size: {args.block_size}. Must be an integer.")
        sys.exit(1)

    # Set up Dask cluster
    client, cluster = setup_dask_cluster(log)

    try:
        # Load and preprocess rasters
        candidate, benchmark = load_rasters(
            args.candidate_path, args.benchmark_path, log
        )

        # Compute agreement map
        agreement_map = compute_agreement_map(
            candidate, benchmark, args.crosstab_path, args.metrics_path, log
        )

        # Prepare GDAL_CACHEMAX as an integer
        gdal_cachemax = os.getenv("GDAL_CACHEMAX")
        try:
            gdal_cachemax = int(gdal_cachemax) if gdal_cachemax is not None else 1024  # Default to 512 MB
        except ValueError:
            log.warning(f"Invalid GDAL_CACHEMAX: {gdal_cachemax}. Using default of 1024 MB.")
            gdal_cachemax = 1024

        # Write agreement map to GeoTIFF
        with rasterio.Env():
            write_agreement_map(
                agreement_map, args.output_path, client, block_size, log
            )

    except Exception as e:
        log.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

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


# python make_agreement.py --fim_type extent --candidate_path s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_hand_huc_11090202_cog.tif --benchmark_path s3://fimc-data/autoeval/test_data/agreement/inputs/huc_11090202/formatted_ble_huc_11090202_cog.tif --output /test/mock_data/huc_11090202_agreement_brad.tif --crosstab_path /test/mock_data/crosstab1.csv --metrics_path /test/mock_data/metrics1.csv
