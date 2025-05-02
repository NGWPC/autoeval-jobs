#!/usr/bin/env python3
import os
import shutil
import pdb
import json
import argparse
from smart_open import open
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import boto3
from typing import Optional, Union, Dict, Any


def inundate(
    catchment_data: Union[str, Dict[str, Any]],
    forecast_path: str,
    output_path: str,
    geo_mem_cache: int = 512,  # Control GDAL cache size in MB
) -> str:
    """
    Generate inundation map from NWM forecasts and HAND data.

    Parameters
    ----------
    catchment_data : str or dict
        Path to catchment JSON file or the loaded catchment data dictionary.
    forecast_path : str
        Path to forecast CSV file.
    output_path : str
        Output path for inundation raster.
    geo_mem_cache : int, optional
        GDAL cache size in megabytes, by default 256 MB.
        Controls memory usage during raster processing.

    Returns
    -------
    str
        Path to the output inundation raster.

    Raises
    ------
    ValueError
        If no matching forecast data for catchment features.
    """
    # Configure AWS credentials for raster access
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

    # Load data efficiently
    if isinstance(catchment_data, str):
        with open(catchment_data) as f:
            catchment = json.load(f)
    else:
        catchment = catchment_data

    # Use optimized CSV reading with just the columns we need
    with open(forecast_path, "r") as f:
        forecast = pd.read_csv(
            f,
            usecols=[0, 1],  # Only read the first two columns
            header=0,
            names=["feature_id", "discharge"],
            dtype={
                "feature_id": np.int32,
                "discharge": np.float32,
            },  # Specify exact dtypes
        )

    # Create stage mapping with proper group handling
    hydro_df = pd.DataFrame(catchment["hydrotable_entries"]).T.reset_index(
        names="HydroID"
    )
    hydro_df = hydro_df[hydro_df.lake_id == -999].explode(["stage", "discharge_cms"])

    # Optimize data types for memory efficiency
    hydro_df = hydro_df.astype(
        {
            "HydroID": np.int32,
            "stage": np.float32,
            "discharge_cms": np.float32,
            "nwm_feature_id": np.int32,
        }
    )

    # Create merged DataFrame
    merged = hydro_df.merge(
        forecast, left_on="nwm_feature_id", right_on="feature_id", how="inner"
    )

    if merged.empty:
        raise ValueError("No matching forecast data for catchment features")

    # More efficient stage interpolation
    merged.set_index("HydroID", inplace=True)

    stage_map = (
        merged.groupby(level=0, group_keys=False)
        .apply(lambda g: np.interp(g.discharge.iloc[0], g.discharge_cms, g.stage))
        .to_dict()
    )

    # Create temporary output path
    temp_output = "/tmp/temp_inundation.tif"

    # Enhanced GDAL environment settings for better performance
    config_options = {
        "GDAL_CACHEMAX": geo_mem_cache,
        "VSI_CACHE_SIZE": 1024
        * 1024
        * min(256, geo_mem_cache),  # VSI cache in bytes (smaller than GDAL cache)
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",  # Performance boost for S3/cloud storage
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",  # Limit allowed extensions for security
    }

    # Set up rasterio environment with optimized settings
    with rasterio.Env(**config_options):
        # Windowed raster processing using rasterio's built-in blocking
        rem_path = catchment["raster_pair"]["rem_raster_path"]
        cat_path = catchment["raster_pair"]["catchment_raster_path"]

        with rasterio.open(rem_path) as rem, rasterio.open(cat_path) as cat:
            profile = rem.profile
            profile.update(
                dtype="uint8",
                count=1,
                nodata=255,
                compress="lzw",
                tiled=True,  # Ensure output is tiled for efficient access
                blockxsize=256,  # Reasonable block size for most systems
                blockysize=256,
            )

            with rasterio.open(temp_output, "w", **profile) as dst:
                windows = list(rem.block_windows(1))

                # Process each window
                for _, window in windows:
                    # Read data with exact dtypes to avoid unnecessary conversions
                    rem_win = rem.read(1, window=window, out_dtype=np.float32)
                    cat_win = cat.read(1, window=window, out_dtype=np.int32)

                    # Create output array directly with required dtype
                    inundation = np.zeros(rem_win.shape, dtype=np.uint8)

                    # Process only where rem_win is valid to save computation
                    valid_mask = rem_win >= 0

                    if np.any(valid_mask):
                        # Use a approach that avoids vectorize which is slow
                        unique_ids = np.unique(cat_win[valid_mask])
                        for uid in unique_ids:
                            if uid in stage_map:
                                stage_value = stage_map[uid]
                                # Apply inundation logic for this specific catchment ID
                                id_mask = (cat_win == uid) & valid_mask
                                # Create a temporary mask of the same shape as id_mask
                                temp_mask = np.zeros_like(id_mask, dtype=bool)
                                # Only set True values where the condition is met for the current ID
                                temp_mask[id_mask] = rem_win[id_mask] <= stage_value
                                # Update inundation based on the combined mask
                                inundation[temp_mask] = 1
                    # Write the results
                    dst.write(inundation, 1, window=window)

    # Handle output path
    if output_path.startswith("s3://"):
        s3_path = output_path[5:]
        bucket, _, key = s3_path.partition("/")

        s3_client = boto3.client("s3")

        s3_client.upload_file(
            temp_output,
            bucket,
            key,
            ExtraArgs={"StorageClass": "STANDARD"},  # Can be adjusted based on needs
        )
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(temp_output, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Inundation mapping from NWM forecasts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catchment-data", required=True, help="Path to catchment JSON"
    )
    parser.add_argument("--forecast-path", required=True, help="Path to forecast CSV")
    parser.add_argument(
        "--output-path", required=True, help="Output path for inundation raster"
    )
    parser.add_argument(
        "--geo-mem-cache", type=int, default=512, help="GDAL cache size in megabytes"
    )
    args = parser.parse_args()
    try:
        output_raster = inundate(
            catchment_data=args.catchment_data,
            forecast_path=args.forecast_path,
            output_path=args.output_path,
            geo_mem_cache=args.geo_mem_cache,
        )
        print(f"Successfully created inundation raster: {output_raster}")
    except Exception as e:
        import sys

        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
