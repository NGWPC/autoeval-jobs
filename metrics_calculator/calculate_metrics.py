#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import rioxarray as rxr
import xarray as xr
import gval
import fsspec
from fsspec.core import url_to_fs
from utils.logging import setup_logger

# GDAL CONFIGURATION FOR OPTIMAL PERFORMANCE (via environment variables)
os.environ["AWS_REGION"] = os.getenv("AWS_REGION", "us-east-1")
os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["CPL_LOG_ERRORS"] = "ON"


JOB_ID = "metrics_calculator"


def open_file(path: str, mode: str = "rb"):
    """
    Open a local or remote file (s3://, gcs://, http://, etc.) via fsspec.
    Returns a file-like object.
    """
    fs, fs_path = url_to_fs(path)
    return fs.open(fs_path, mode)


def calculate_metrics(agreement_map_path: str, log: logging.Logger) -> pd.DataFrame:
    """Load agreement map and compute categorical metrics."""
    log.info(f"Loading agreement map from {agreement_map_path}")
    agreement_map = rxr.open_rasterio(
        agreement_map_path,
        mask_and_scale=True,
        chunks={"x": 2048, "y": 2048},
        lock=False,
    )

    # Define pairing dictionary for agreement map interpretation
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

    log.info("Computing crosstab table from agreement map")
    # Set pairing dictionary as attribute if needed by gval
    if hasattr(agreement_map, 'attrs'):
        agreement_map.attrs['pairing_dictionary'] = pairing_dictionary
    crosstab_table = agreement_map.gval.compute_crosstab()

    log.info("Computing categorical metrics")
    # After computing crosstab from agreement map, use standard binary classification encoding
    # positive_categories=[1] for positive class, negative_categories=[0] for negative class
    metrics_table = crosstab_table.gval.compute_categorical_metrics(
        positive_categories=[1], 
        negative_categories=[0], 
        metrics="all"
    )

    return metrics_table


def write_outputs(metrics_table: pd.DataFrame, metrics_path: str, log: logging.Logger) -> None:
    """Write metrics table to CSV using fsspec for S3 compatibility."""
    log.info(f"Writing metrics table to {metrics_path}")
    with open_file(metrics_path, 'wt') as f:
        metrics_table.to_csv(f, index=True)


def main():
    log = setup_logger(JOB_ID)
    parser = argparse.ArgumentParser(description="Compute metrics from a single agreement map.")
    parser.add_argument("--agreement_map_path", required=True, help="Input path for agreement map raster (local or S3)")
    parser.add_argument("--metrics_path", required=True, help="Output path for metrics CSV (local or S3)")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for processing large rasters")
    args = parser.parse_args()

    try:
        # Compute metrics
        metrics_table = calculate_metrics(args.agreement_map_path, log)

        # Write output
        write_outputs(metrics_table, args.metrics_path, log)

        log.success({"metrics_path": args.metrics_path})

    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

    # example usage local
    # python calculate_metrics.py --agreement_map_path /test/mock_data/agreement_map.tif --metrics_path /test/mock_data/metrics.csv