#!/usr/bin/env python3
import argparse
import logging
import os
import sys

import fsspec
import gval
import pandas as pd
import rioxarray as rxr
import xarray as xr
from fsspec.core import url_to_fs

from utils.logging import setup_logger
from utils.pairing import AGREEMENT_PAIRING_DICT


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
        chunks={"x": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")), "y": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048"))},
        lock=False,
    )

    # Use shared pairing dictionary from utils
    pairing_dictionary = AGREEMENT_PAIRING_DICT

    log.info("Computing crosstab table from agreement map")
    # Set pairing dictionary as attribute if needed by gval
    if hasattr(agreement_map, "attrs"):
        agreement_map.attrs["pairing_dictionary"] = pairing_dictionary
    crosstab_table = agreement_map.gval.compute_crosstab()

    log.info("Computing categorical metrics")
    # After computing crosstab from agreement map, use standard binary classification encoding
    # positive_categories=[1] for positive class, negative_categories=[0] for negative class
    metrics_table = crosstab_table.gval.compute_categorical_metrics(
        positive_categories=[1], negative_categories=[0], metrics="all"
    )

    return metrics_table


def write_outputs(metrics_table: pd.DataFrame, metrics_path: str, log: logging.Logger) -> None:
    """Write metrics table to CSV using fsspec for S3 compatibility."""
    log.info(f"Writing metrics table to {metrics_path}")
    with open_file(metrics_path, "wt") as f:
        metrics_table.to_csv(f, index=True)


def main():
    log = setup_logger(JOB_ID)
    parser = argparse.ArgumentParser(description="Compute metrics from a single agreement map.")
    parser.add_argument("--agreement_map_path", required=True, help="Input path for agreement map raster (local or S3)")
    parser.add_argument("--metrics_path", required=True, help="Output path for metrics CSV (local or S3)")
    parser.add_argument("--chunk_size", type=int, default=int(os.getenv("DEFAULT_METRICS_CHUNK_SIZE", "1024")), help="Chunk size for processing large rasters")
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
