#!/usr/bin/env python3
import os
import shutil
import argparse
import pdb
from typing import Any, Dict

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import boto3
import fsspec
from fsspec.core import url_to_fs


def open_file(path: str, mode: str = "rb"):
    """
    Open a local or remote file (s3://, gcs://, http://, etc.) via fsspec.
    Returns a file-like object.
    """
    fs, fs_path = url_to_fs(path)
    return fs.open(fs_path, mode)


def inundate(
    catchment_parquet: str,
    forecast_path: str,
    output_path: str,
    geo_mem_cache: int = 512,  # in MB
) -> str:
    """
    Generate an inundation map from:
      1) A catchment HAND‐table Parquet (with list‐columns: stage & discharge_cms)
      2) An NWM forecast CSV (feature_id, discharge)
      3) Two rasters: REM & catchment ID

    Writes a uint8 flooded/dry mask to `output_path` (local or s3://).
    """
    # Configure AWS creds so fsspec/s3fs & rasterio pick them up:
    session = boto3.Session(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    creds = session.get_credentials()
    if creds:
        os.environ.update(
            {
                "AWS_ACCESS_KEY_ID": creds.access_key,
                "AWS_SECRET_ACCESS_KEY": creds.secret_key,
                **({"AWS_SESSION_TOKEN": creds.token} if creds.token else {}),
            }
        )

    # Load catchment table from Parquet (via fsspec)
    with open_file(catchment_parquet, "rb") as f:
        catchment_df = pd.read_parquet(f)

    # Read forecast CSV (only two columns) using fsspec
    with open_file(forecast_path, "rt") as f:
        forecast = pd.read_csv(
            f,
            usecols=[0, 1],
            header=0,
            names=["feature_id", "discharge"],
            dtype={"feature_id": np.int32, "discharge": np.float32},
        )

    # Build HydroID → stage lookup
    # ensure lake_id is integer
    catchment_df["lake_id"] = catchment_df["lake_id"].astype(int)
    # now filter out lakes (lake_id == -999)
    hydro_df = catchment_df[catchment_df["lake_id"] == -999]
    if hydro_df.empty:
        raise ValueError("No catchments with negative lake_id -999 found in Parquet")

    merged = hydro_df.merge(
        forecast, left_on="nwm_feature_id", right_on="feature_id", how="inner"
    )
    if merged.empty:
        raise ValueError("No matching forecast data for catchment features")

    # Interpolate each feature’s forecast discharge onto its lists
    stage_map: Dict[int, float] = {
        int(row["HydroID"]): float(
            np.interp(
                row["discharge"],  # scalar from forecast
                row["discharge_cms"],  # list from Parquet
                row["stage"],  # list from Parquet
            )
        )
        for _, row in merged.iterrows()
    }

    rem_path = catchment_df["rem_raster_path"].iat[0]
    cat_path = catchment_df["catchment_raster_path"].iat[0]

    # Prepare temporary output
    tmp_tif = "/tmp/temp_inundation.tif"
    config_options = {
        "GDAL_CACHEMAX": geo_mem_cache,
        "VSI_CACHE_SIZE": 1024 * 1024 * min(256, geo_mem_cache),
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
    }

    # Raster processing with rasterio.Env
    with rasterio.Env(**config_options):
        with rasterio.open(rem_path) as rem, rasterio.open(cat_path) as cat:
            profile = rem.profile.copy()
            profile.update(
                dtype="uint8",
                count=1,
                nodata=255,
                compress="lzw",
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )

            with rasterio.open(tmp_tif, "w", **profile) as dst:
                for _, window in rem.block_windows(1):
                    rem_win = rem.read(
                        1, window=window, out_dtype=np.float32, masked=True
                    )
                    cat_win = cat.read(1, window=window, out_dtype=np.int32)

                    inund = np.zeros(rem_win.shape, dtype=np.uint8)
                    valid = ~rem_win.masked
                    if valid.any():
                        for uid in np.unique(cat_win[valid]):
                            if uid in stage_map:
                                stg = stage_map[uid]
                                mask = (cat_win == uid) & valid & (rem_win <= stg)
                                inund[mask] = 1

                    # stamp nodata value into any place the REM has nodata
                    inund[rem_win.mask] = profile["nodata"]
                    dst.write(inund, 1, window=window)

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
    parser = argparse.ArgumentParser(
        description="Inundation mapping from NWM forecasts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catchment-data",
        required=True,
        help="Path to catchment Parquet (local or s3://)",
    )
    parser.add_argument(
        "--forecast-path", required=True, help="Path to forecast CSV (local or s3://)"
    )
    parser.add_argument(
        "--output-path", required=True, help="Where to write .tif (local or s3://)"
    )
    parser.add_argument(
        "--geo-mem-cache", type=int, default=512, help="GDAL cache in MB"
    )

    args = parser.parse_args()

    try:
        out = inundate(
            catchment_parquet=args.catchment_data,
            forecast_path=args.forecast_path,
            output_path=args.output_path,
            geo_mem_cache=args.geo_mem_cache,
        )
        print(f"Successfully wrote: {out}")
    except Exception as e:
        import sys

        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
