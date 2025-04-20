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


def mosaic_rasters(
    raster_paths: list[str],
    output_path: str,
    clip_geometry: Optional[Union[str, dict]] = None,
    fim_type: Literal["depth", "extent"] = "depth",
    geo_mem_cache: int = 256,  # Control GDAL cache size in MB
) -> str:
    """Raster mosaicking using rasterio with optimized memory settings.

    Parameters
    ----------
    raster_paths : list[str]
        List of paths to rasters to be mosaicked.
    output_path : str
        Path where to save the mosaicked output.
    clip_geometry : str or dict, optional
        Vector file path or GeoJSON-like geometry to clip the output raster.
    fim_type : str, optional
        Type of FIM output, either "depth" or "extent".
        For depth: uses float32 dtype and -9999 as nodata.
        For extent: uses uint8 dtype and 255 as nodata, converts all nonzero values to 1.
    geo_mem_cache : int, optional
        GDAL cache size in megabytes, by default 256 MB.
        Controls memory usage during raster processing.

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

    # Enhanced GDAL environment settings for better performance
    config_options = {
        "GDAL_CACHEMAX": geo_mem_cache,
        "VSI_CACHE_SIZE": 1024 * 1024 * min(256, geo_mem_cache),  # VSI cache in bytes
        "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",  # Performance boost for S3/cloud storage
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",  # Limit allowed extensions
    }

    # Open all rasters and get their metadata
    src_files = []
    try:
        with rasterio.Env(**config_options):
            for path in raster_paths:
                src = rasterio.open(path)
                src_files.append(src)

            # Check that all rasters have the same CRS
            crs = src_files[0].crs
            if not all(src.crs == crs for src in src_files):
                raise ValueError("All rasters must have the same CRS")

            # Get bounds of the mosaic
            bounds = [src.bounds for src in src_files]
            left = min(bound.left for bound in bounds)
            bottom = min(bound.bottom for bound in bounds)
            right = max(bound.right for bound in bounds)
            top = max(bound.top for bound in bounds)

            # Get resolution (assuming all rasters have same resolution)
            res = src_files[0].res
            if not all(src.res == res for src in src_files):
                raise ValueError("All rasters must have the same resolution")

            # Calculate output dimensions
            width = int((right - left) / res[0] + 0.5)
            height = int((top - bottom) / res[1] + 0.5)

            # Create output profile
            profile = src_files[0].profile.copy()
            profile.update(
                {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "transform": rasterio.transform.from_bounds(
                        left, bottom, right, top, width, height
                    ),
                    "dtype": dtype,
                    "nodata": nodata,
                    "tiled": True,
                    "blockxsize": 256,  # Standard block size for efficient processing
                    "blockysize": 256,
                    "compress": "lzw",
                    "predictor": 2,
                }
            )

            # Create output raster
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as dst:
                # Get a list of all blocks for processing
                windows = list(dst.block_windows(1))

                # Process each window
                for idx, (_, window) in enumerate(windows):
                    # Create a transform for this particular window
                    window_transform = dst.window_transform(window)

                    # Initialize output block with nodata
                    out_data = np.full(
                        (window.height, window.width), nodata, dtype=profile["dtype"]
                    )

                    # Track if we've found any valid data for this window
                    has_valid_data = False

                    # Compute the window bounds in coordinate space
                    window_bounds = rasterio.transform.array_bounds(
                        window.height, window.width, window_transform
                    )

                    # For each source raster, check if it overlaps with this window
                    for src in src_files:
                        # Calculate the window in the source raster's coordinate system
                        src_window = src.window(*window_bounds)

                        # Check if the source overlaps this window
                        if src_window.width <= 0 or src_window.height <= 0:
                            continue

                        # Read data from the source raster
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
                                continue

                            # Create mask of valid data (not nodata)
                            src_nodata = src.nodata if src.nodata is not None else None
                            valid_mask = (
                                np.ones_like(data, dtype=bool)
                                if src_nodata is None
                                else data != src_nodata
                            )

                            if fim_type == "extent":
                                # For extent type, convert all non-zero values to 1
                                # But only for valid (not nodata) cells
                                data_temp = np.zeros_like(data, dtype=np.uint8)
                                # Set 1 where valid and non-zero
                                data_temp[valid_mask & (data != 0)] = 1
                                data = data_temp

                            # Update the output array - only where valid data exists
                            if np.any(valid_mask):
                                has_valid_data = True
                                # Only keep maximum values where mask is True
                                temp = np.full_like(data, nodata)
                                temp[valid_mask] = data[valid_mask]

                                # Create a mask for where out_data is nodata
                                out_nodata_mask = out_data == nodata

                                # Where both are valid, take max. Where only temp is valid, take temp.
                                # Where only out_data is valid, keep out_data
                                mask_both_valid = valid_mask & ~out_nodata_mask
                                mask_only_temp_valid = valid_mask & out_nodata_mask

                                if np.any(mask_both_valid):
                                    out_data[mask_both_valid] = np.maximum(
                                        out_data[mask_both_valid], temp[mask_both_valid]
                                    )
                                if np.any(mask_only_temp_valid):
                                    out_data[mask_only_temp_valid] = temp[
                                        mask_only_temp_valid
                                    ]

                        except Exception as e:
                            print(f"Warning: Error reading window from source: {e}")
                            continue

                    # For extent type, ensure we only have 0, 1, or nodata in the output
                    if fim_type == "extent" and has_valid_data:
                        valid_mask = out_data != nodata
                        out_data[valid_mask & (out_data > 0)] = 1

                    # Write block to output
                    dst.write(out_data, window=window, indexes=1)

            # Apply clipping if geometry provided
            if clip_geometry is not None:
                with rasterio.open(output_path, "r+") as src:
                    if isinstance(clip_geometry, str):
                        with fiona.open(clip_geometry, "r") as clip_file:
                            geoms = [feature["geometry"] for feature in clip_file]
                    else:
                        geoms = (
                            [clip_geometry]
                            if isinstance(clip_geometry, dict)
                            else clip_geometry
                        )

                    out_data, out_transform = mask(
                        src, geoms, crop=False, nodata=nodata
                    )
                    src.write(out_data[0], indexes=1)

    finally:
        # Close all source files
        for src in src_files:
            src.close()

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
        type=Path,
        help="Path to save the mosaicked output",
    )

    parser.add_argument(
        "--clip-geometry",
        type=Path,
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
            output_path=str(args.output_path),
            clip_geometry=str(args.clip_geometry) if args.clip_geometry else None,
            fim_type=args.fim_type,
            geo_mem_cache=args.geo_mem_cache,
        )
        print(f"Successfully created mosaic: {output_raster}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
