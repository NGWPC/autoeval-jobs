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
            fmt=fmt, 
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ", 
            rename_fields={"asctime": "timestamp", "levelname": "level"}
        )
    )
    log.addHandler(handler)
    log.propagate = False
    return log


def calculate_metrics(crosstab_path: str, log: logging.Logger) -> pd.DataFrame:
    """Read cross-tabulation CSV and compute categorical metrics."""
    log.info("Reading crosstab CSV into DataFrame")
    crosstab_table = pd.read_csv(crosstab_path, index_col=0)

    log.info("Computing categorical metrics")
    metrics_table_select = crosstab_table.gval.compute_categorical_metrics(
                negative_categories=[0, 1], positive_categories=[2]
            )

    return metrics_table_select


def write_outputs(metrics_table: pd.DataFrame, metrics_path: str, log: logging.Logger) -> None:
    """Write metrics table to CSV."""
    log.info(f"Writing metrics table to {metrics_path}")
    metrics_table.to_csv(metrics_path, index=True)


def main():
    log = setup_logger()
    parser = argparse.ArgumentParser(description="Compute metrics from a single agreement map.")
    parser.add_argument("--crosstab_path", required=True, help="Input path for cross-tabulation CSV (local or S3)")
    parser.add_argument("--metrics_path", required=True, help="Output path for metrics CSV (local or S3)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for processing large rasters")
    args = parser.parse_args()

    client = None
    cluster = None

    try:
        # --- Dask Configuration ---
        log.info("Setting up Dask local cluster")
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            memory_limit="7GiB",
        )
        client = Client(cluster)
        log.info(f"Dask dashboard link: {client.dashboard_link}")

        # GDAL tuning parameters
        gdal_opts = dict(
            GDAL_CACHEMAX=1024,
            GDAL_NUM_THREADS=1,
            GDAL_TIFF_DIRECT_IO="YES",
            GDAL_TIFF_OVR_BLOCKSIZE=256,
            GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
        )
        for key, value in gdal_opts.items():
            gdal.SetConfigOption(key, str(value))
            log.info(f"Set GDAL option: {key}={value}")

        # Dask config
        dask.config.set({"array.slicing.split-large-chunks": True})

        # Compute metrics
        metrics_table = calculate_metrics(args.crosstab_path, log)

        # Write output
        write_outputs(metrics_table, args.metrics_path, log)

        log.info("Processing completed successfully")

    except Exception as e:
        log.error(f"Error processing: {e}", exc_info=True)

    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()

if __name__ == "__main__":
    main()
