#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import rioxarray as rxr
import xarray as xr
import pandas as pd
import numpy as np
import gval
from pythonjsonlogger import jsonlogger
from osgeo import gdal
import dask
from dask.distributed import LocalCluster, Client

# -----------------------------------------------------------------------------
# GLOBAL GDAL CONFIGURATION
# -----------------------------------------------------------------------------
gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
gdal.SetConfigOption("GDAL_NUM_THREADS", "1")
gdal.UseExceptions()


def setup_logger(name="raster_metrics") -> logging.Logger:
    """Return a JSON-formatter logger."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S.%fZ", rename_fields={"asctime": "timestamp", "levelname": "level"}
        )
    )
    log.addHandler(handler)
    log.propagate = False
    return log


def load_agreement_map(agreement_path: str, chunk_size: int, log: logging.Logger) -> xr.DataArray:
    """Load a single agreement map raster with chunking."""
    log.info(f"Loading agreement map: {agreement_path} with chunk size {chunk_size}")
    
    # Open the dataset with chunking enabled
    ds = rxr.open_rasterio(
        agreement_path,
        mask_and_scale=True,
        chunks={'x': chunk_size, 'y': chunk_size},
        cache=False,  # Disable caching to reduce memory usage
        lock=False    # Disable file locking which can help with performance
    )
    
    # Get dataset information
    shape = ds.shape
    log.info(f"Raster dimensions: {shape}")
    log.info(f"Data type: {ds.dtype}")
    
    return ds


def write_outputs(crosstab_table: pd.DataFrame, metric_table: pd.DataFrame, crosstab_path: str, metrics_path: str, log: logging.Logger) -> None:
    """Write tables to CSV."""
    log.info(f"Writing cross-tabulation table to {crosstab_path}")
    crosstab_table.to_csv(crosstab_path, index=True)
    log.info(f"Writing metrics table to {metrics_path}")
    metric_table.to_csv(metrics_path, index=True)


def main():
    log = setup_logger()
    parser = argparse.ArgumentParser(description="Compute metrics from a single agreement map.")
    parser.add_argument("--agreement_path", required=True, help="Path to agreement map raster (local or S3)")
    parser.add_argument("--crosstab_path", required=True, help="Output path for cross-tabulation CSV (local or S3)")
    parser.add_argument("--metrics_path", required=True, help="Output path for metrics CSV (local or S3)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for processing large rasters")
    args = parser.parse_args()

    client = None
    cluster = None
    
    try:
        # --- Dask Configuration ---
        # Start a local Dask cluster
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            memory_limit="7GiB",
        )
        client = Client(cluster)
        log.info(f"Dask dashboard link: {client.dashboard_link}")
        
        # Define your GDAL tuning parameters
        gdal_opts = dict(
            GDAL_CACHEMAX=1024,
            GDAL_NUM_THREADS=1,
            GDAL_TIFF_DIRECT_IO="YES",
            GDAL_TIFF_OVR_BLOCKSIZE=256,
            GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
        )
        
        # Apply GDAL configuration options
        for key, value in gdal_opts.items():
            gdal.SetConfigOption(key, str(value))
            log.info(f"Set GDAL option: {key}={value}")

        # Set dask memory settings
        dask.config.set({"array.slicing.split-large-chunks": True})
        
        # Load agreement map with chunking
        agreement_map = load_agreement_map(args.agreement_path, args.chunk_size, log)

        classes = [1, 2, 3, 4, 5]
        pairing_dictionary = {(x, y): int(f'{x}{y}') for x, y in product(*([classes]*2))}

        # compute cross-tabulation table
        crosstab_table = agreement_map.gval.compute_crosstab()
        print(crosstab_table)
        crosstab_table.to_csv("cross_tab_table.csv")

        # compute metrics
        metric_table_select = crosstab_table.gval.compute_categorical_metrics(
            negative_categories=[0, 1], positive_categories=[2]
        )


        """
        # Compute cross-tabulation table - this is the simplest approach
        log.info("Computing cross-tabulation table")
        with dask.config.set({"array.slicing.split_large_chunks": True}):
            crosstab_table = agreement_map.gval.compute_crosstab()
        
        log.info("Computing categorical metrics")
        metric_table = crosstab_table.gval.compute_categorical_metrics(
            negative_categories=[0, 1], 
            positive_categories=[2]
        )
        
        # Write outputs
        write_outputs(crosstab_table, metric_table, args.crosstab_path, args.metrics_path, log)
        """
        log.info("Processing completed successfully")
        
    except Exception as e:
        log.error(f"Error processing raster: {str(e)}")
        
        # More detailed error information
        import traceback
        error_details = traceback.format_exc()
        log.error(f"Detailed error: {error_details}")
        
        raise
    
    finally:
        # Clean up resources
        if client is not None:
            log.info("Closing Dask client")
            client.close()
        
        if cluster is not None:
            log.info("Shutting down Dask cluster")
            cluster.close()


if __name__ == "__main__":
    main()


    # Sample usage
    # python calculate_metrics.py --agreement_path /test/mock_data/huc_11090202_agreement_brad.tif --crosstab_path /test/mock_data/crosstab1.csv --metrics_path /test/mock_data/metris1.csv