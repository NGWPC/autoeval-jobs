#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import pandas as pd
import gval
from pythonjsonlogger import jsonlogger


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

    try:
        # Compute metrics
        metrics_table = calculate_metrics(args.crosstab_path, log)

        # Write output
        write_outputs(metrics_table, args.metrics_path, log)

        log.info("Processing completed successfully")

    except Exception as e:
        log.error(f"Error processing: {e}", exc_info=True)


if __name__ == "__main__":
    main()

    # example usage local
    # python calculate_metrics.py --crosstab_path /test/mock_data/crosstab1.csv --metrics_path /test/mock_data/metrics2.csv