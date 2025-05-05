#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from typing import Tuple
import rioxarray as rxr
import xarray as xr
import pandas as pd
import gval
from pythonjsonlogger import jsonlogger
from osgeo import gdal

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


def load_agreement_map(agreement_path: str, log: logging.Logger) -> xr.DataArray:
    """Load a single agreement map raster."""
    log.info(f"Loading agreement map: {agreement_path}")
    return rxr.open_rasterio(agreement_path, mask_and_scale=True)


def compute_crosstab_and_metrics(agreement_map: xr.DataArray, log: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cross-tabulation table and metrics for a single raster."""
    log.info("Computing cross-tabulation table")
    crosstab_table = agreement_map.gval.compute_crosstab()
    log.info("Computing categorical metrics")
    metric_table = crosstab_table.gval.compute_categorical_metrics(negative_categories=[0, 1], positive_categories=[2])
    return crosstab_table, metric_table


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
    args = parser.parse_args()

    agreement_map = load_agreement_map(args.agreement_path, log)
    crosstab_table, metric_table = compute_crosstab_and_metrics(agreement_map, log)
    write_outputs(crosstab_table, metric_table, args.crosstab_path, args.metrics_path, log)

if __name__ == "__main__":
    main()