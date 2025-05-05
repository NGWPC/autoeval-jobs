#!/usr/bin/env python3
import os
import shutil
import json
import argparse
from typing import Union, Dict, Any

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
    catchment_data: Union[str, Dict[str, Any]],
    forecast_path: str,
    output_path: str,
    geo_mem_cache: int = 512,  # in MB
) -> str:
    """
    Generate inundation map from NWM forecasts and HAND data using fsspec.
    """
    # 1) Configure AWS creds so fsspec/s3fs & rasterio pick them up:
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

    # 2) Load catchment JSON
    if isinstance(catchment_data, str):
        with open_file(catchment_data, "rt") as f:
            catchment = json.load(f)
    else:
        catchment = catchment_data

    # 3) Read forecast CSV (only two columns) using fsspec
    with open_file(forecast_path, "rt") as f:
        forecast = pd.read_csv(
            f,
            usecols=[0, 1],
            header=0,
            names=["feature_id", "discharge"],
            dtype={"feature_id": np.int32, "discharge": np.float32},
        )

    # 4) Build hydro-stage lookup
    hydro_df = (
        pd.DataFrame(catchment["hydrotable_entries"])
        .T.reset_index(names="HydroID")
        .query("lake_id == -999")
        .explode(["stage", "discharge_cms"])
    )
    hydro_df = hydro_df.astype(
        {
            "HydroID": np.int32,
            "stage": np.float32,
            "discharge_cms": np.float32,
            "nwm_feature_id": np.int32,
        }
    )
    merged = hydro_df.merge(
        forecast, left_on="nwm_feature_id", right_on="feature_id", how="inner"
    )
    if merged.empty:
        raise ValueError("No matching forecast data for catchment features")

    merged.set_index("HydroID", inplace=True)
    stage_map = (
        merged.groupby(level=0, group_keys=False)
        .apply(lambda g: np.interp(g.discharge.iloc[0], g.discharge_cms, g.stage))
        .to_dict()
    )

    # 5) Prepare temporary output
    tmp_tif = "/tmp/temp_inundation.tif"
    config_options = {
        "GDAL_CACHEMAX": geo_mem_cache,
        "VSI_CACHE_SIZE": 1024 * 1024 * min(256, geo_mem_cache),
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
    }

    # 6) Raster processing with rasterio.Env
    with rasterio.Env(**config_options):
        rem_path = catchment["raster_pair"]["rem_raster_path"]
        cat_path = catchment["raster_pair"]["catchment_raster_path"]

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
                    rem_win = rem.read(1, window=window, out_dtype=np.float32)
                    cat_win = cat.read(1, window=window, out_dtype=np.int32)

                    inund = np.zeros(rem_win.shape, dtype=np.uint8)
                    valid = rem_win >= 0
                    if valid.any():
                        for uid in np.unique(cat_win[valid]):
                            if uid in stage_map:
                                stg = stage_map[uid]
                                mask = (cat_win == uid) & valid & (rem_win <= stg)
                                inund[mask] = 1

                    dst.write(inund, 1, window=window)

    # 7) Move or upload result
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
        help="Path to catchment JSON (local or s3://)",
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
            catchment_data=args.catchment_data,
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
