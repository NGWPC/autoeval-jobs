#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import sys
import tempfile
from typing import Any, Dict

import boto3
import fsspec
import numpy as np
import pandas as pd
import rasterio
from fsspec.core import url_to_fs
from rasterio.windows import Window

from utils.logging import setup_logger


def open_file(path: str, mode: str = "rb"):
    """
    Open a local or remote file (s3://, gcs://, http://, etc.) via fsspec.
    Returns a file-like object.
    """
    fs, fs_path = url_to_fs(path)
    return fs.open(fs_path, mode)


JOB_ID = "hand_inundator"


def inundate(
    catchment_parquet: str,
    forecast_path: str,
    output_path: str,
    fim_type: str,
    log: logging.Logger,
) -> str:
    """
    Generate an inundation map from:
      1) A catchment Parquet (with csv_path, rem_raster_path, catchment_raster_path)
      2) A hydrotable CSV (individual stage/discharge rows per HydroID)
      3) An NWM forecast CSV (feature_id, discharge)
      4) Two rasters: REM & catchment ID

    Writes a uint8 flooded/dry mask to `output_path` (local or s3://).
    """

    log.info(f"Loading catchment data from {catchment_parquet}")
    with open_file(catchment_parquet, "rb") as f:
        catchment_df = pd.read_parquet(f)

    # Extract CSV path and raster paths from the parquet
    csv_path = catchment_df["csv_path"].iloc[0]
    rem_path = catchment_df["rem_raster_path"].iloc[0]
    cat_path = catchment_df["catchment_raster_path"].iloc[0]

    log.info(f"Loading hydrotable data from {csv_path}")
    with open_file(csv_path, "rt") as f:
        hydrotable_df = pd.read_csv(f)

    log.info(f"Loading forecast data from {forecast_path}")
    with open_file(forecast_path, "rt") as f:
        forecast = pd.read_csv(
            f,
            usecols=[0, 1],
            header=0,
            names=["feature_id", "discharge"],
            dtype={"feature_id": np.int32, "discharge": np.float32},
        )

    # Build HydroID → stage lookup
    log.info("Building stage lookup from hydrotable and forecast data")
    hydrotable_df["LakeID"] = hydrotable_df["LakeID"].astype(int)
    # now filter out lakes
    lake_filter_value = int(os.getenv("LAKE_ID_FILTER_VALUE", "-999"))
    hydro_df = hydrotable_df[hydrotable_df["LakeID"] == lake_filter_value]
    if hydro_df.empty:
        raise ValueError("No catchments with negative LakeID -999 found in hydrotable CSV")

    # Get unique HydroID and feature_id combinations for merging
    hydro_features = hydro_df[["HydroID", "feature_id"]].drop_duplicates()
    merged = hydro_features.merge(forecast, left_on="feature_id", right_on="feature_id", how="inner")
    if merged.empty:
        raise ValueError("No matching forecast data for catchment features")

    log.info(f"Found {len(merged)} matching catchments with forecast data")

    # Sort and group the rating curves
    curves = {
        hid: (grp["discharge_cms"].values, grp["stage"].values)
        for hid, grp in hydro_df.sort_values("stage").groupby("HydroID")
    }

    # interpolation
    hids = merged["HydroID"].to_numpy()
    discharges = merged["discharge"].to_numpy()

    stages = np.array([np.interp(q, *curves[hid]) for hid, q in zip(hids, discharges)], dtype=np.float32)

    # Build the final stage map
    stage_map = dict(zip(hids.tolist(), stages.tolist()))

    # rem_path and cat_path already extracted from parquet above

    log.info("Starting inundation mapping")
    # create a unique temp file to avoid file collisions when running locally
    tmp_fd, tmp_tif = tempfile.mkstemp(suffix=".tif", prefix="temp_inundation_", dir="/tmp")
    os.close(tmp_fd)  # Close the file descriptor as rasterio will open it

    with rasterio.Env():
        with rasterio.open(rem_path) as rem, rasterio.open(cat_path) as cat:
            profile = rem.profile.copy()

            # Set profile based on fim_type
            if fim_type == "depth":
                profile.update(
                    dtype="float32",
                    count=1,
                    nodata=float(os.getenv("DEPTH_NODATA_VALUE", "-9999")),
                    compress=os.getenv("INUNDATION_COMPRESS_TYPE", "lzw"),
                    tiled=True,
                    blockxsize=int(os.getenv("INUNDATION_BLOCK_SIZE", "256")),
                    blockysize=int(os.getenv("INUNDATION_BLOCK_SIZE", "256")),
                )
                output_dtype = np.float32
            else:  # extent
                profile.update(
                    dtype="uint8",
                    count=1,
                    nodata=int(os.getenv("INUNDATION_NODATA_VALUE", "255")),
                    compress=os.getenv("INUNDATION_COMPRESS_TYPE", "lzw"),
                    tiled=True,
                    blockxsize=int(os.getenv("INUNDATION_BLOCK_SIZE", "256")),
                    blockysize=int(os.getenv("INUNDATION_BLOCK_SIZE", "256")),
                )
                output_dtype = np.uint8

            with rasterio.open(tmp_tif, "w", **profile) as dst:
                for _, window in rem.block_windows(1):
                    rem_win = rem.read(1, window=window, out_dtype=np.float32, masked=True)
                    cat_win = cat.read(1, window=window, out_dtype=np.int32)

                    result = np.zeros(rem_win.shape, dtype=output_dtype)

                    # rem_win is a MaskedArray when read with masked=True
                    valid = ~rem_win.mask
                    if valid.any():
                        for uid in np.unique(cat_win[valid]):
                            if uid in stage_map:
                                stg = stage_map[uid]
                                # Find inundated pixels for this catchment
                                mask = (cat_win == uid) & valid & (rem_win.data <= stg)

                                if fim_type == "depth":
                                    # Calculate depth: stage - REM
                                    result[mask] = stg - rem_win.data[mask]
                                else:
                                    # Binary extent: 1 where inundated
                                    result[mask] = 1

                    # Set nodata values where REM has nodata
                    result[rem_win.mask] = profile["nodata"]
                    dst.write(result, 1, window=window)

    # Move or upload result
    if output_path.startswith("s3://"):
        path_no_scheme = output_path[len("s3://") :]
        bucket, key = path_no_scheme.split("/", 1)
        boto3.client("s3").upload_file(tmp_tif, bucket, key)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(tmp_tif, output_path)

    return output_path


def main():
    log = setup_logger(JOB_ID)
    parser = argparse.ArgumentParser(
        description="Inundation mapping from NWM forecasts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catchment_data_path",
        required=True,
        help="Path to catchment Parquet (local or s3://)",
    )
    parser.add_argument("--forecast_path", required=True, help="Path to forecast CSV (local or s3://)")
    parser.add_argument("--fim_output_path", required=True, help="Where to write .tif (local or s3://)")
    parser.add_argument(
        "--fim_type",
        choices=["extent", "depth"],
        default="extent",
        help="Output type: extent (binary) or depth (float values)",
    )

    args = parser.parse_args()

    try:
        out = inundate(
            catchment_parquet=args.catchment_data_path,
            forecast_path=args.forecast_path,
            output_path=args.fim_output_path,
            fim_type=args.fim_type,
            log=log,
        )
        log.success({"output_path": out})
    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
