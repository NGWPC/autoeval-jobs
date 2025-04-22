import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.mask import mask
import numpy as np
from typing import Union, Optional, Literal
import os
import sys
import fiona
import json
import argparse
from pathlib import Path
import boto3
import gc


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def get_raster_info(raster_paths: list[str], config_options: dict):
    """
    Get metadata from rasters without keeping them all open at once.
    """
    # Get info from first raster for basic metadata
    with rasterio.Env(**config_options):
        with rasterio.open(raster_paths[0]) as src:
            profile = src.profile.copy()
            crs = src.crs
            res = src.res
            first_bounds = src.bounds

    # Get bounds from all rasters without keeping them open simultaneously
    bounds_list = [first_bounds]
    with rasterio.Env(**config_options):
        for path in raster_paths[1:]:  # Skip the first one we already processed
            with rasterio.open(path) as src:
                # Verify CRS and resolution match
                if src.crs != crs:
                    raise ValueError(f"Raster {path} has different CRS than others")
                if src.res != res:
                    raise ValueError(
                        f"Raster {path} has different resolution than others"
                    )
                bounds_list.append(src.bounds)

    # Calculate overall bounds
    left = min(bound.left for bound in bounds_list)
    bottom = min(bound.bottom for bound in bounds_list)
    right = max(bound.right for bound in bounds_list)
    top = max(bound.top for bound in bounds_list)

    # Calculate output dimensions
    width = int((right - left) / res[0] + 0.5)
    height = int((top - bottom) / res[1] + 0.5)

    # Create output transform
    transform = rasterio.transform.from_bounds(left, bottom, right, top, width, height)

    return profile, crs, res, width, height, transform, bounds_list


def process_window_for_source(
    src, window, window_bounds, nodata, fim_type, profile, out_data=None
):
    """
    Process a window for a single source raster.
    Returns modified output data and a flag indicating if valid data was found.
    """
    # Initialize output if not provided
    if out_data is None:
        out_data = np.full(
            (window.height, window.width), nodata, dtype=profile["dtype"]
        )

    has_valid_data = False

    # Calculate the window in the source raster's coordinate system
    src_window = src.window(*window_bounds)

    # Check if the source overlaps this window
    if src_window.width <= 0 or src_window.height <= 0:
        return out_data, has_valid_data

    try:
        # Ensure the window is valid for reading
        src_window = src_window.round_offsets().round_lengths()
        data = src.read(
            1,
            window=src_window,
            boundless=True,
            fill_value=src.nodata,
        )

        if data is None or data.size == 0:
            return out_data, has_valid_data

        # Create mask of valid data (not nodata)
        src_nodata = src.nodata if src.nodata is not None else None
        valid_mask = (
            np.ones_like(data, dtype=bool) if src_nodata is None else data != src_nodata
        )

        if not np.any(valid_mask):
            return out_data, has_valid_data

        if fim_type == "extent":
            # For extent type, convert all non-zero values to 1
            # But only for valid (not nodata) cells
            data_valid = np.zeros_like(data, dtype=np.uint8)
            np.place(data_valid, valid_mask & (data != 0), 1)
            data = data_valid
            del data_valid

        # Update the output array - only where valid data exists
        has_valid_data = True

        # Create a mask for where out_data is nodata
        out_nodata_mask = out_data == nodata

        # Where both are valid, take max. Where only temp is valid, take temp.
        # Where only out_data is valid, keep out_data
        mask_both_valid = valid_mask & ~out_nodata_mask
        mask_only_src_valid = valid_mask & out_nodata_mask

        # Only process where needed
        if np.any(mask_both_valid):
            # Operate directly on out_data where possible
            np.maximum(out_data, data, out=out_data, where=mask_both_valid)

        if np.any(mask_only_src_valid):
            out_data[mask_only_src_valid] = data[mask_only_src_valid]

        # Clean up
        del valid_mask, out_nodata_mask, mask_both_valid, mask_only_src_valid

    except Exception as e:
        print(f"Warning: Error reading window from source: {e}")

    # Clean up
    del data
    gc.collect()

    return out_data, has_valid_data


def mosaic_rasters(
    raster_paths: list[str],
    output_path: str,
    clip_geometry: Optional[Union[str, dict]] = None,
    fim_type: Literal["depth", "extent"] = "depth",
    geo_mem_cache: int = 256,  # Control GDAL cache size in MB
    block_size: int = 256,  # Block size for processing
) -> str:
    """Raster mosaicking using rasterio with optimized memory settings.

    Parameters
    ----------
    raster_paths : list[str]
        List of paths to rasters to be mosaicked.
    output_path : str
        Path to save the mosaicked output. Can be local or S3 path.
    clip_geometry : str or dict, optional
        Vector file path or GeoJSON-like geometry to clip the output raster.
    fim_type : str, optional
        Type of FIM output, either "depth" or "extent".
        For depth: uses float32 dtype and -9999 as nodata.
        For extent: uses uint8 dtype and 255 as nodata, converts all nonzero values to 1.
    geo_mem_cache : int, optional
        GDAL cache size in megabytes, by default 256 MB.
        Controls memory usage during raster processing.
    block_size : int, optional
        Size of blocks for processing, smaller values use less memory.

    Returns
    -------
    str
        Path to the output raster.

    Raises
    ------
    ValueError
        If input parameters are invalid or no rasters provided.
    RuntimeError
        If unable to open input files or process data.
    """
    if fim_type not in ["depth", "extent"]:
        raise ValueError("fim_type must be either 'depth' or 'extent'")

    if not raster_paths:
        raise ValueError("No rasters provided for mosaicking.")

    # Set raster properties based on fim_type
    if fim_type == "depth":
        nodata = -9999
        dtype = "float32"
    else:  # extent
        nodata = 255
        dtype = "uint8"

    # Convert S3 path format if needed
    vsis3_output_path = output_path
    if is_s3_path(output_path):
        # Convert s3:// to /vsis3/ format for GDAL
        bucket_and_key = output_path[5:]  # Remove 's3://'
        vsis3_output_path = f"/vsis3/{bucket_and_key}"

    # Set up AWS credentials in environment variables
    session = boto3.Session(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    creds = session.get_credentials()
    if creds:
        # Update environment variables with credentials
        os.environ.update(
            {
                "AWS_ACCESS_KEY_ID": creds.access_key,
                "AWS_SECRET_ACCESS_KEY": creds.secret_key,
                **({"AWS_SESSION_TOKEN": creds.token} if creds.token else {}),
            }
        )

    # Enhanced GDAL environment settings for better S3 writing support with memory optimization
    config_options = {
        "GDAL_CACHEMAX": geo_mem_cache,
        "VSI_CACHE_SIZE": 1024 * 1024 * min(128, geo_mem_cache),  # Smaller VSI cache
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
        "AWS_VIRTUAL_HOSTING": "TRUE",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "25000000",  # Reduced cache size (25MB)
        # Critical settings for S3 write operations
        "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": "YES",
        "VSI_TEMP_DIR": "/tmp",
        "AWS_REQUEST_PAYER": "requester",  # If bucket is requester pays
        "AWS_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    }

    # Only create directories for local paths, not for S3
    if not is_s3_path(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Get metadata without keeping all rasters open
        profile, crs, res, width, height, transform, bounds_list = get_raster_info(
            raster_paths, config_options
        )

        # Update output profile
        profile.update(
            {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "dtype": dtype,
                "nodata": nodata,
                "tiled": True,
                "blockxsize": block_size,  # Smaller block size for less memory usage
                "blockysize": block_size,
                "compress": "lzw",
                "predictor": 2,
            }
        )

        print(
            f"Writing output to: {output_path}"
            + (" (using VSIS3)" if is_s3_path(output_path) else "")
        )

        # Create output raster - use vsis3 path for S3 writing
        with rasterio.Env(**config_options):
            with rasterio.open(vsis3_output_path, "w", **profile) as dst:
                # Get a list of all blocks for processing
                windows = list(dst.block_windows(1))

                # Process each window
                for idx, (_, window) in enumerate(windows):
                    if idx % 10 == 0:  # Status update every 10 windows
                        print(f"Processing window {idx+1} of {len(windows)}")

                    # Create a transform for this particular window
                    window_transform = dst.window_transform(window)

                    # Initialize output block with nodata
                    out_data = np.full(
                        (window.height, window.width), nodata, dtype=profile["dtype"]
                    )

                    # Compute the window bounds in coordinate space
                    window_bounds = rasterio.transform.array_bounds(
                        window.height, window.width, window_transform
                    )

                    # Track if we've found any valid data for this window
                    has_valid_data = False

                    # Process each source one at a time to reduce memory usage
                    for path in raster_paths:
                        with rasterio.open(path) as src:
                            # Process this source for the current window
                            out_data, src_has_valid = process_window_for_source(
                                src,
                                window,
                                window_bounds,
                                nodata,
                                fim_type,
                                profile,
                                out_data,
                            )

                            if src_has_valid:
                                has_valid_data = True

                    # For extent type, ensure we only have 0, 1, or nodata in the output
                    if fim_type == "extent" and has_valid_data:
                        # Operate directly on out_data
                        valid_mask = out_data != nodata
                        np.place(out_data, valid_mask & (out_data > 0), 1)
                        del valid_mask

                    # Write block to output
                    dst.write(out_data, window=window, indexes=1)

                    # Clean up
                    del out_data, window_transform, window_bounds
                    gc.collect()

                # Force GC after all windows
                gc.collect()

            # Handle clipping in a separate step to reduce memory pressure
            if clip_geometry is not None:
                print("Applying clip geometry...")

                # Read clip geometry only once
                if isinstance(clip_geometry, str):
                    with fiona.open(clip_geometry, "r") as clip_file:
                        geoms = [feature["geometry"] for feature in clip_file]
                else:
                    geoms = (
                        [clip_geometry]
                        if isinstance(clip_geometry, dict)
                        else clip_geometry
                    )

                # Process in blocks for clipping to reduce memory usage
                with rasterio.open(vsis3_output_path, "r+") as src:
                    for _, window in src.block_windows(1):
                        # Read the block
                        data = src.read(1, window=window)

                        # Get the window transform
                        window_transform = src.window_transform(window)

                        # Create a mask for this window
                        masked, _ = mask(
                            rasterio.io.MemoryFile().open(
                                driver="GTiff",
                                height=window.height,
                                width=window.width,
                                count=1,
                                dtype=src.dtypes[0],
                                nodata=nodata,
                                transform=window_transform,
                            ),
                            geoms,
                            crop=False,
                            nodata=nodata,
                            all_touched=True,
                        )

                        # Apply the mask to the data
                        data = masked[0]

                        # Write back
                        src.write(data, indexes=1, window=window)

                        # Clean up
                        del data, masked, window_transform
                        gc.collect()

    except Exception as e:
        print(f"Error in mosaic operation: {e}")
        raise

    # Final cleanup
    gc.collect()

    return output_path


if __name__ == "__main__":
    """
    Entry point to the mosaic_rasters function that mosaics a list of FIM extents or depths together.

    Expected JSON format for raster paths:
    [
        "/path/to/raster1.tif",
        "/path/to/raster2.tif",
        ...
    ]

    The JSON can be provided either:
    1. As a JSON string directly to --raster_paths
    2. As a path to a JSON file containing the list of paths

    Example JSON file contents:
    [
        "/data/fim/raster_001.tif",
        "/data/fim/raster_002.tif",
        "/data/fim/raster_003.tif"
    ]

    Supports S3 paths in the format:
    s3://bucket-name/path/to/file.tif
    """

    parser = argparse.ArgumentParser(
        description="Mosaic multiple rasters provided as a JSON list, with optional clipping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--raster_paths",
        required=True,
        type=str,
        help="Required: JSON string representation of a list of input raster paths OR path to a JSON file containing such a list",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save the mosaicked output (can be local or S3 path)",
    )

    parser.add_argument(
        "--clip-geometry",
        type=str,
        help="Optional path to vector file for clipping the output",
    )

    parser.add_argument(
        "--fim-type",
        choices=["depth", "extent"],
        default="depth",
        help="Type of FIM output (affects data type and nodata value)",
    )

    parser.add_argument(
        "--geo-mem-cache",
        type=int,
        default=256,
        help="GDAL cache size in megabytes",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size for processing (smaller = less memory usage)",
    )

    args = parser.parse_args()

    # Check if raster_paths is a path to a JSON file
    if os.path.isfile(args.raster_paths):
        try:
            with open(args.raster_paths, "r") as f:
                loaded_paths = json.load(f)
            print(
                f"Loaded {len(loaded_paths)} raster paths from file: {args.raster_paths}"
            )
        except json.JSONDecodeError:
            print(
                f"Error: Invalid JSON in file {args.raster_paths}",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"Error reading file {args.raster_paths}: {str(e)}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Treat as a JSON string
        try:
            loaded_paths = json.loads(args.raster_paths)
            print(f"Parsed {len(loaded_paths)} raster paths from JSON string")
        except json.JSONDecodeError:
            print(
                f"Error: Invalid JSON provided to --raster_paths: {args.raster_paths}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Validate the loaded paths
    if isinstance(loaded_paths, list):
        # Ensure all elements are strings
        raster_paths_list = [str(p) for p in loaded_paths]
    else:
        print(
            f"Error: JSON must contain a list of raster paths. Found: {type(loaded_paths).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not raster_paths_list:
        print(
            "Error: The list of raster paths is empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        output_raster = mosaic_rasters(
            raster_paths=raster_paths_list,
            output_path=args.output_path,
            clip_geometry=args.clip_geometry if args.clip_geometry else None,
            fim_type=args.fim_type,
            geo_mem_cache=args.geo_mem_cache,
            block_size=args.block_size,
        )
        print(f"Successfully created mosaic: {output_raster}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
