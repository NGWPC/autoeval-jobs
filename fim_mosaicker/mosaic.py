#!/usr/bin/env python3
import numpy
from osgeo import gdal, gdal_array, osr, ogr  # Need ogr for clip geometry handling
from osgeo_utils.auxiliary import extent_util
from osgeo_utils.auxiliary.base import PathLikeOrStr
from osgeo_utils.auxiliary.extent_util import (
    Extent,
    GeoTransform,
)
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.util import GetOutputDriverFor, open_ds
import argparse
import os
import sys
from pathlib import Path
import logging
from pythonjsonlogger import jsonlogger  # Import directly
import datetime
import tempfile
from typing import (
    Union,
    Optional,
    Literal,
    Tuple,
    List,
    Sequence,
)
import fiona
import shutil
import json

# --- osgeo.gdal Setup ---
try:
    # Ensure exceptions are enabled for osgeo.gdal functions used directly
    gdal.UseExceptions()
    gdal.SetConfigOption("CPL_LOG_ERRORS", "ON")
except AttributeError:
    print("Warning: Could not configure GDAL exceptions/logging.", file=sys.stderr)

# --- Logging Setup ---
JOB_ID = "fim_mosaicker"
logger = logging.getLogger("fim_mosaicker")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

handler = logging.StreamHandler(sys.stderr)
formatter = jsonlogger.JsonFormatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
    rename_fields={"asctime": "timestamp", "levelname": "level"},
    json_ensure_ascii=False,
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
# --- End Logging Setup ---

# Default NoData values (similar to gdal_calc)
DEFAULT_NDV_LOOKUP = {
    gdal.GDT_Byte: 255,
    gdal.GDT_Int8: -128,
    gdal.GDT_UInt16: 65535,
    gdal.GDT_Int16: -32768,
    gdal.GDT_UInt32: 4294967295,
    gdal.GDT_Int32: -2147483648,
    gdal.GDT_UInt64: None,
    gdal.GDT_Int64: None,  # Choose appropriate if needed
    gdal.GDT_Float32: -3.4028235e38,
    gdal.GDT_Float64: -1.7976931348623157e308,
    # Add complex types if necessary
}


def _get_lowest_resolution_info(
    datasets: List[gdal.Dataset],
) -> Tuple[gdal.Dataset, tuple, str]:
    """Finds the dataset with the lowest resolution (largest pixel area) and its info."""
    lowest_res_area = -1.0
    target_ds_for_info = None  # Keep track of the ds we are getting the info FROM
    target_res = None  # Tuple (pixel_width, pixel_height) - positive values
    target_crs_wkt = None  # WKT string

    for ds in datasets:
        gt = ds.GetGeoTransform()
        if gt is None:
            continue  # Skip datasets without geotransform

        pixel_width = abs(gt[1])
        pixel_height = abs(gt[5])
        if gt[2] != 0 or gt[4] != 0:
            logger.warning(
                f"Dataset {ds.GetDescription()} has rotation/shear ({gt[2]}, {gt[4]}). Resolution calculation assumes orthogonal grid."
            )
        if pixel_width == 0 or pixel_height == 0:
            logger.warning(
                f"Dataset {ds.GetDescription()} has zero pixel dimension in geotransform. Cannot determine resolution."
            )
            continue

        # Calculate pixel area
        pixel_area = pixel_width * pixel_height

        if pixel_area > lowest_res_area:
            lowest_res_area = pixel_area
            target_ds_for_info = ds
            target_res = (pixel_width, pixel_height)
            target_crs_wkt = ds.GetProjection()

    if target_ds_for_info is None:
        raise ValueError(
            "Could not determine lowest resolution raster (e.g., no valid geotransforms found)."
        )

    logger.info(
        f"Reference raster for resolution/CRS: {target_ds_for_info.GetDescription()} (Res: {target_res})"
    )
    return target_ds_for_info, target_res, target_crs_wkt


def mosaic_rasters_gdalcalc_style(
    raster_paths: List[PathLikeOrStr],
    mosaic_output_path: PathLikeOrStr,
    clip_geometry_path: Optional[PathLikeOrStr] = None,
    fim_type: Literal["depth", "extent"] = "depth",
) -> str:
    """
    Mosaics rasters using gdal_calc-inspired techniques for alignment and memory efficiency.

    - Calculates union extent and determines target grid based on lowest input resolution.
    - Aligns inputs on-the-fly using VRTs if necessary.
    - Applies a pixel-wise NAN-MAX policy.
    - Processes data in windows/blocks to limit memory usage.
    - Always overwrites existing output file.
    - Always outputs a Cloud Optimized GeoTIFF (COG).

    Parameters:
        raster_paths: List of input raster file paths (local or cloud).
        mosaic_output_path: Path for the output mosaicked COG GeoTIFF.
        clip_geometry_path: Optional path to vector file for clipping.
        fim_type: 'depth' (float32) or 'extent' (uint8, 0/1).

    Returns:
        Path to the output raster.

    Raises:
        ValueError: Invalid inputs.
        RuntimeError: Processing errors (GDAL, file IO, etc.).
        ImportError: If essential GDAL components are missing.
    """
    temp_vrt_dir = None  # Initialize outside try block for cleanup
    temp_vrt_files = []  # Keep track of temporary VRTs to clean up
    input_datasets_info = []  # Store info about each input
    aligned_datasets = []  # Store datasets to read from (original or VRT handle)
    in_datasets_raw = []  # Temp list for initial open phases
    out_ds = None  # Initialize output dataset handle
    out_band = None  # Initialize output band handle
    out_ds_overview = None  # Initialize handle for overview building
    clip_ds = None  # Initialize handle for potential clipping dataset
    mem_ds = None  # Initialize handle for clipping memory layer

    # --- Overwrite Logic ---
    if os.path.exists(mosaic_output_path):
        logger.info(f"Output file {mosaic_output_path} exists. Overwriting.")
        try:
            os.remove(mosaic_output_path)
            logger.debug(f"Successfully removed existing file: {mosaic_output_path}")
        except Exception as e:  # Catch PermissionError, OSError etc.
            logger.warning(
                f"Could not delete existing file {mosaic_output_path}: {e}. Creation might fail."
            )
    # --- End Overwrite Logic ---

    if not raster_paths:
        logger.error("No input raster paths provided.")
        raise ValueError("No rasters provided for mosaicking.")

    gdal_cache_mb = os.environ.get("GDAL_CACHEMAX", "Default")
    logger.info(
        f"Using GDAL_CACHEMAX: {gdal_cache_mb} MB (applied during VRT/raster reads)"
    )

    # --- Determine Output Format(Fixed), Data Type, Nodata, COG Options ---
    out_format = "GTiff"  # Always output GeoTIFF
    if fim_type == "depth":
        out_datatype_gdal = gdal.GDT_Float32
        out_nodata = DEFAULT_NDV_LOOKUP.get(
            out_datatype_gdal, -3.4028235e38
        )  # Use .get for safety
        predictor = 3  # PREDICTOR=3 (Floating point prediction)
        resampling_gdal = gdal.GRIORA_Bilinear  # Resampling for alignment VRTs
        overview_resampling = "AVERAGE"  # Resampling for overviews
    elif fim_type == "extent":
        out_datatype_gdal = gdal.GDT_Byte
        out_nodata = DEFAULT_NDV_LOOKUP.get(
            out_datatype_gdal, 255
        )  # Use .get for safety
        predictor = 2  # PREDICTOR=2 (Horizontal differencing)
        resampling_gdal = gdal.GRIORA_NearestNeighbour  # Resampling for alignment VRTs
        overview_resampling = "NEAREST"  # Resampling for overviews
    else:
        raise ValueError("fim_type must be either 'depth' or 'extent'")
    logger.info(
        f"Output Type: {gdal.GetDataTypeName(out_datatype_gdal)}, NoData: {out_nodata}, Align Resampling: {resampling_gdal}"
    )

    cog_options = [
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "COMPRESS=LZW",
        f"PREDICTOR={predictor}",
        "COPY_SRC_OVERVIEWS=NO",
        "BIGTIFF=IF_SAFER",
    ]
    logger.info(f"Using COG creation options: {cog_options}")
    # --------------------------------

    try:
        # --- 1. Open Inputs and Get Metadata ---
        logger.info(f"Opening {len(raster_paths)} input rasters...")
        for i, path in enumerate(raster_paths):
            # Convert Path objects if they exist, otherwise use string directly
            path_str = str(path) if isinstance(path, Path) else path
            ds = open_ds(path_str, access_mode=gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR)
            if ds is None:
                raise RuntimeError(f"Failed to open input raster: {path_str}")

            band = ds.GetRasterBand(1)  # Assume single band inputs
            nodata = band.GetNoDataValue()
            gt = ds.GetGeoTransform(can_return_null=True)
            proj = ds.GetProjection()
            dims = (ds.RasterXSize, ds.RasterYSize)
            if gt is None or dims[0] == 0 or dims[1] == 0:
                logger.warning(
                    f"Skipping {path_str} due to missing GeoTransform or zero dimensions."
                )
                ds = None  # Close dataset
                continue  # Skip this problematic input

            logger.debug(
                f"Opened: {path_str} (Res: {gt[1]:.4g}, {gt[5]:.4g} | NDV: {nodata})"
            )
            in_datasets_raw.append(ds)
            input_datasets_info.append(
                {
                    "path": path_str,  # Store string path
                    "ds": ds,
                    "nodata": nodata,
                    "gt": gt,
                    "proj_wkt": proj,
                    "dims": dims,
                    "needs_vrt": False,
                    "vrt_path": None,
                }
            )
        if not in_datasets_raw:
            raise ValueError("No valid input rasters could be opened.")

        # --- 2. Determine Target Resolution, CRS, and Union Extent Rectangle ---
        target_ref_ds, target_res_tuple, target_crs_wkt = _get_lowest_resolution_info(
            in_datasets_raw
        )

        # Collect valid GTs and Dims for extent calculation
        input_gts: Sequence[GeoTransform] = [
            info["gt"] for info in input_datasets_info
        ]  # Already checked for None GTs
        input_dims = [info["dims"] for info in input_datasets_info]

        logger.info("Calculating union extent rectangle...")
        try:
            # Use extent_util to get the union bounding box (GeoRectangle)
            # We only need the GeoRectangle from this call's result tuple [_, _, rect]
            _, _, union_extent_rect = extent_util.calc_geotransform_and_dimensions(
                input_gts,
                input_dims,
                input_extent=Extent.UNION,  # Correct keyword based on function signature
            )
            if union_extent_rect is None or not isinstance(
                union_extent_rect, GeoRectangle
            ):
                msg = f"Extent calculation returned invalid type: {type(union_extent_rect)}"
                logger.error(msg)
                raise ValueError(msg)
            logger.info(
                f"Calculated Union Extent (MinX, MinY, MaxX, MaxY): ({union_extent_rect.min_x}, {union_extent_rect.min_y}, {union_extent_rect.max_x}, {union_extent_rect.max_y})"
            )
        except Exception as e:
            logger.error(
                f"Failed to calculate union extent using extent_util: {e}",
                exc_info=True,
            )
            raise RuntimeError("Could not calculate valid union extent.") from e

        # --- 3. Manually Calculate Final Output Grid (GT and Dims) ---
        logger.info("Calculating final output grid dimensions and transform...")
        try:
            union_left, union_bottom, union_right, union_top = (
                union_extent_rect.min_x,
                union_extent_rect.min_y,
                union_extent_rect.max_x,
                union_extent_rect.max_y,
            )

            # Ensure positive resolution values
            res_x = abs(target_res_tuple[0])
            res_y = abs(target_res_tuple[1])
            if res_x == 0 or res_y == 0:
                raise ValueError("Target resolution cannot be zero.")

            # Calculate dimensions using ceiling to ensure full coverage
            target_dims_x = int(numpy.ceil((union_right - union_left) / res_x))
            target_dims_y = int(numpy.ceil((union_top - union_bottom) / res_y))
            target_dims = (
                max(1, target_dims_x),
                max(1, target_dims_y),
            )  # Final dimensions tuple

            # Define the final output geotransform (north-up, no rotation/skew)
            target_gt: GeoTransform = (union_left, res_x, 0.0, union_top, 0.0, -res_y)

            logger.info(
                f"Final Output Grid: Dims={target_dims}, GT={tuple(f'{x:.4g}' for x in target_gt)}"
            )
        except Exception as e:
            logger.error(
                f"Failed to manually calculate final grid parameters: {e}",
                exc_info=True,
            )
            raise RuntimeError("Could not calculate final grid parameters.") from e

        # --- 4. Create Alignment VRTs if needed ---
        logger.info("Checking for and creating alignment VRTs as needed...")
        temp_vrt_dir = tempfile.mkdtemp(prefix="mosaic_vrt_")
        logger.debug(f"Using temporary directory for VRTs: {temp_vrt_dir}")

        for i, info in enumerate(input_datasets_info):
            ds = info["ds"]
            current_gt = info["gt"]
            current_dims = info["dims"]
            current_proj = info["proj_wkt"]

            # Check if alignment is needed (compare against final calculated grid)
            # Use approx comparison for GT, strict for dims/proj
            gt_differ = not numpy.allclose(current_gt, target_gt, atol=1e-6, rtol=1e-6)
            dims_differ = current_dims != target_dims
            proj_differ = current_proj != target_crs_wkt

            # needs_alignment = gt_differ or dims_differ or proj_differ
            # Compare source grid against the *manually calculated* target grid
            needs_alignment = gt_differ or dims_differ or proj_differ
            if not needs_alignment:
                logger.debug(f"Grids match (manual check) for: {info['path']}")

            if needs_alignment:
                logger.info(
                    f"Creating alignment VRT for: {info['path']} (Resampling: {resampling_gdal})"
                )
                info["needs_vrt"] = True
                # Define the output VRT path
                # Handle potential non-string paths (though unlikely after input handling)
                base_name = os.path.splitext(os.path.basename(str(info["path"])))[0]
                vrt_path_out = os.path.join(temp_vrt_dir, f"{base_name}_aligned.vrt")

                # Prepare gdal.Warp options for creating an aligned VRT
                warp_options = gdal.WarpOptions(
                    format="VRT",  # Output format is VRT
                    outputBounds=(
                        union_left,
                        union_bottom,
                        union_right,
                        union_top,
                    ),  # Target extent minx, miny, maxx, maxy
                    width=target_dims[0],  # Target width in pixels
                    height=target_dims[1],  # Target height in pixels
                    dstSRS=target_crs_wkt,  # Target CRS WKT
                    resampleAlg=resampling_gdal,  # Resampling algorithm (numeric code)
                    # Optional: Set nodata value for the VRT if needed (source or target?)
                    # srcNodata=info['nodata'] if info['nodata'] is not None else None, # Handle potential None nodata
                    # dstNodata=out_nodata,            # Often useful to set target nodata
                    multithread=True,  # Enable multithreading if possible
                    warpOptions=["NUM_THREADS=ALL_CPUS"],  # Use available cores
                    # targetAlignedPixels=True,      # Consider adding this
                )

                # Execute gdal.Warp to create the VRT file
                logger.debug(f"Creating VRT using gdal.Warp: {vrt_path_out}")
                vrt_ds_warp = gdal.Warp(vrt_path_out, ds, options=warp_options)

                if vrt_ds_warp is None:
                    # Ensure VRT wasn't partially created or left open
                    vrt_ds_warp = None
                    if os.path.exists(vrt_path_out):
                        logger.warning(
                            f"gdal.Warp seemed to fail but VRT exists: {vrt_path_out}. Attempting cleanup/re-raise."
                        )
                        # Potentially try to remove vrt_path_out here
                    raise RuntimeError(
                        f"Failed to create alignment VRT using gdal.Warp for {info['path']}"
                    )

                # MUST close the dataset returned by gdal.Warp to flush VRT XML,
                # then re-open it for reading later.
                vrt_ds_warp = None

                # Re-open the created VRT for reading in the processing loop
                vrt_ds = gdal.Open(vrt_path_out, gdal.GA_ReadOnly)
                if vrt_ds is None:
                    raise RuntimeError(f"Failed to re-open created VRT: {vrt_path_out}")

                logger.debug(f"Created VRT: {vrt_path_out}")
                info["vrt_path"] = vrt_path_out
                temp_vrt_files.append(vrt_path_out)
                aligned_datasets.append(vrt_ds)  # Use the VRT dataset handle
                info["ds"] = None  # Close original dataset handle
            else:
                logger.info(f"No alignment VRT needed for: {info['path']}")
                info["needs_vrt"] = False
                aligned_datasets.append(ds)  # Use original dataset handle
                # Keep info['ds'] pointing to the dataset, will be closed in finally

        # Explicitly close raw datasets that are no longer needed
        in_datasets_raw = []

        # --- 5. Create Output COG Raster ---
        driver = gdal.GetDriverByName(out_format)
        logger.info(f"Creating output COG file: {mosaic_output_path}")
        out_ds = driver.Create(
            str(mosaic_output_path),
            target_dims[0],
            target_dims[1],  # Use final calculated dims
            1,  # Single output band
            out_datatype_gdal,
            options=cog_options,  # Use specific COG options
        )
        if out_ds is None:
            raise RuntimeError(f"Failed to create output file: {mosaic_output_path}")

        out_ds.SetGeoTransform(target_gt)  # Use final calculated GT
        if target_crs_wkt:
            out_ds.SetProjection(target_crs_wkt)
        out_band = out_ds.GetRasterBand(1)
        if out_nodata is not None:
            out_band.SetNoDataValue(out_nodata)

        # --- 6. Windowed Processing (Core Loop) ---
        block_size_x, block_size_y = (
            out_band.GetBlockSize()
        )  # Get blocksize from COG options
        logger.info(
            f"Starting windowed processing with block size: {block_size_x}x{block_size_y}"
        )
        nx_blocks = (target_dims[0] + block_size_x - 1) // block_size_x
        ny_blocks = (target_dims[1] + block_size_y - 1) // block_size_y
        total_blocks = nx_blocks * ny_blocks
        progress_count = 0

        for y_block in range(ny_blocks):
            for x_block in range(nx_blocks):
                progress_count += 1
                if progress_count % 100 == 0 or progress_count == total_blocks:
                    logger.debug(  # Change progress to debug to avoid excessive logs
                        f"Processing block {progress_count}/{total_blocks} ({x_block+1}/{nx_blocks}, {y_block+1}/{ny_blocks})"
                    )

                x_off = x_block * block_size_x
                y_off = y_block * block_size_y
                x_valid = min(block_size_x, target_dims[0] - x_off)
                y_valid = min(block_size_y, target_dims[1] - y_off)
                if x_valid <= 0 or y_valid <= 0:
                    continue

                # Use float64 accumulator, initialize with NaN for fmax logic
                out_block_accum = numpy.full(
                    (y_valid, x_valid), numpy.nan, dtype=numpy.float64
                )
                wrote_valid_data_mask = numpy.zeros((y_valid, x_valid), dtype=bool)

                for i, read_ds in enumerate(aligned_datasets):
                    if read_ds is None:
                        continue
                    try:
                        # Reading from VRT will trigger on-the-fly warp/resample
                        in_block_data = gdal_array.BandReadAsArray(
                            read_ds.GetRasterBand(1),
                            xoff=x_off,
                            yoff=y_off,
                            win_xsize=x_valid,
                            win_ysize=y_valid,
                        )
                    except Exception as read_error:
                        # Log path of failed source using input_datasets_info
                        src_path = input_datasets_info[i]["path"]
                        logger.warning(
                            f"Failed read: Block({x_off},{y_off}) Src {i} ({src_path}): {read_error}. Skip source for this block."
                        )
                        continue
                    if in_block_data is None:
                        continue  # Should not happen if read succeeded

                    # Mask based on original source nodata
                    source_nodata = input_datasets_info[i]["nodata"]
                    valid_mask = numpy.ones(in_block_data.shape, dtype=bool)
                    if source_nodata is not None:
                        # Handle NaN nodata correctly
                        if numpy.isnan(source_nodata):
                            valid_mask = ~numpy.isnan(in_block_data)
                        else:
                            valid_mask = (in_block_data != source_nodata) & (
                                ~numpy.isnan(in_block_data)
                            )

                    if not numpy.any(valid_mask):
                        continue  # Skip if block has no valid data

                    # Convert to float64, set nodata=NaN for calculation
                    work_data = in_block_data.astype(numpy.float64, copy=True)
                    work_data[~valid_mask] = numpy.nan

                    if fim_type == "extent":  # Apply 0/1 logic for extent type
                        work_data[valid_mask & (work_data != 0)] = 1.0
                        work_data[valid_mask & (work_data == 0)] = 0.0

                    # NAN-MAX: fmax ignores NaN unless both inputs are NaN
                    out_block_accum = numpy.fmax(out_block_accum, work_data)
                    # Track where any valid data (from any source) was placed
                    wrote_valid_data_mask |= valid_mask

                # Finalize output block
                out_dtype_np = gdal_array.GDALTypeCodeToNumericTypeCode(
                    out_datatype_gdal
                )
                out_block_final = numpy.full(
                    (y_valid, x_valid), out_nodata, dtype=out_dtype_np
                )

                # Identify valid data in accumulator (not NaN) AND where we wrote something
                valid_accum_mask = wrote_valid_data_mask & ~numpy.isnan(out_block_accum)
                # Cast valid accumulator values to output type and place into final block
                # Handle potential NaN -> int casting issues for extent type
                if fim_type == "extent":
                    # Ensure accumulator values are finite before casting (should be 0 or 1)
                    valid_accum_mask &= numpy.isfinite(out_block_accum)
                # Place valid data
                out_block_final[valid_accum_mask] = out_block_accum[
                    valid_accum_mask
                ].astype(out_dtype_np)

                # Write the final block
                gdal_array.BandWriteArray(
                    out_band, out_block_final, xoff=x_off, yoff=y_off
                )

        # --- 7. Build Overviews (Essential for COG performance) ---
        logger.info("Building overviews for COG...")
        out_band = None  # Dereference band
        out_ds.FlushCache()  # Flush before closing for overview generation
        out_ds = None  # Close dataset

        local_cog_options_dict = {}
        for opt in cog_options:  # Use the list defined within this function
            try:
                key, value = opt.split("=", 1)
                # Attempt to convert known numeric options to int
                if key.upper() in [
                    "BLOCKXSIZE",
                    "BLOCKYSIZE",
                    "PREDICTOR",
                ]:  # Add others if needed
                    local_cog_options_dict[key.upper()] = int(value)
                else:
                    local_cog_options_dict[key.upper()] = (
                        value  # Keep others as strings (e.g., 'YES', 'LZW')
                    )
            except ValueError:  # Catch case where no '=' exists or int fails
                local_cog_options_dict[key.upper()] = None  # or handle appropriately
            except Exception as e_dict:
                logger.warning(f"Could not parse COG option '{opt}': {e_dict}")

        # Reopen dataset in Update mode to build overviews
        out_ds_overview = gdal.Open(str(mosaic_output_path), gdal.GA_Update)
        if out_ds_overview is None:
            logger.warning(
                f"Failed to reopen {mosaic_output_path} to build overviews. Output may not be a fully optimized COG."
            )
        else:
            overview_levels = []
            min_dim = min(target_dims)
            level = 2
            # Heuristic: build levels until smallest dimension is < ~blocksize
            blockxsize = (
                local_cog_options_dict.get("BLOCKXSIZE", 512) or 512
            )  # Default if None
            blockysize = (
                local_cog_options_dict.get("BLOCKYSIZE", 512) or 512
            )  # Default if None
            while (min_dim / level) >= max(blockxsize, blockysize):
                overview_levels.append(level)
                level *= 2
            # Ensure at least one level if possible
            if not overview_levels and min_dim > max(blockxsize, blockysize):
                overview_levels.append(2)

            if not overview_levels:
                logger.info("Output dimensions too small for standard overviews.")
            else:
                logger.info(
                    f"Generating overview levels: {overview_levels} using {overview_resampling} resampling."
                )
                try:
                    # Use default GDAL progress callback (prints to stdout/stderr)
                    ret = out_ds_overview.BuildOverviews(
                        overview_resampling, overview_levels, gdal.TermProgress
                    )
                    if ret != 0:
                        logger.warning(f"BuildOverviews returned error code: {ret}")
                except Exception as ovr_err:
                    logger.warning(
                        f"Error building overviews: {ovr_err}", exc_info=True
                    )

            out_ds_overview.FlushCache()  # Flush after overviews
            out_ds_overview = None  # Close dataset handle

        # --- 8. Clipping (Optional - using gdal.Warp) ---
        if clip_geometry_path is not None:
            logger.info(
                f"Clipping output COG using geometry from: {clip_geometry_path}"
            )
            temp_clip_output = None  # Define before try block for cleanup
            try:
                # Temporary file for clipped output to avoid overwriting potentially corrupt file
                temp_clip_output = str(mosaic_output_path) + "_clipped_temp.tif"

                # Read clip geometry WKT (Ensure OGR is available)
                if ogr is None:
                    raise ImportError("osgeo.ogr is required for clipping.")

                # Ensure target_crs_wkt is valid before creating target SR
                target_sr = None
                if target_crs_wkt:
                    target_sr = osr.SpatialReference()
                    if target_sr.ImportFromWkt(target_crs_wkt) != ogr.OGRERR_NONE:
                        logger.warning(
                            f"Failed to import target CRS WKT for clipping reprojection. CRS: {target_crs_wkt}"
                        )
                        target_sr = None  # Treat as invalid

                all_ogr_geoms = []
                with fiona.open(clip_geometry_path, "r") as clip_src:
                    clip_crs_wkt = clip_src.crs_wkt
                    clip_crs = None
                    if clip_crs_wkt:
                        clip_crs = osr.SpatialReference()
                        if clip_crs.ImportFromWkt(clip_crs_wkt) != ogr.OGRERR_NONE:
                            logger.warning(
                                f"Failed to import clip source CRS: {clip_crs_wkt}. Assuming no CRS."
                            )
                            clip_crs = None

                    coord_trans = None
                    if target_sr and clip_crs and not clip_crs.IsSame(target_sr):
                        logger.info(
                            "Clip geometry CRS differs from target CRS. Reprojecting clip geometry."
                        )
                        coord_trans = osr.CoordinateTransformation(clip_crs, target_sr)

                    for feature in clip_src:
                        try:
                            geom_data = feature.get("geometry")
                            # Check if geometry is None or not a dictionary-like structure
                            if not geom_data or not hasattr(geom_data, "get"):
                                logger.warning(
                                    f"Skipping feature {feature.get('id', 'N/A')} due to missing or invalid geometry structure."
                                )
                                continue
                            # Convert geometry dict to JSON string for OGR
                            geom_json_str = json.dumps(geom_data)

                            ogr_geom = ogr.CreateGeometryFromJson(geom_json_str)
                            if ogr_geom is None or ogr_geom.IsEmpty():
                                logger.warning(
                                    f"Skipping invalid or empty geometry in feature {feature.get('id', 'N/A')}"
                                )
                                continue
                            if coord_trans:  # Reproject if needed
                                ogr_err = ogr_geom.Transform(coord_trans)
                                if ogr_err != ogr.OGRERR_NONE:
                                    logger.warning(
                                        f"Geometry reprojection failed for feature {feature.get('id', 'N/A')}. Error code: {ogr_err}"
                                    )
                                    continue  # Skip geometry that failed reprojection

                            all_ogr_geoms.append(ogr_geom)
                        except (TypeError, json.JSONDecodeError) as json_err:
                            logger.warning(
                                f"Skipping feature {feature.get('id', 'N/A')} due to geometry serialization error: {json_err}"
                            )
                            continue
                        except (
                            Exception
                        ) as feat_err:  # Catch other unexpected errors per feature
                            logger.warning(
                                f"Skipping feature {feature.get('id', 'N/A')} due to error processing geometry: {feat_err}"
                            )
                            continue

                if not all_ogr_geoms:
                    raise ValueError(
                        "No valid geometries found or reprojected in clip file."
                    )

                # Create a unified geometry (e.g., MultiPolygon or GeometryCollection) for Warp cutline
                # Simpler approach: Create a temporary in-memory layer
                mem_driver = ogr.GetDriverByName("Memory")
                mem_ds = mem_driver.CreateDataSource("clip_mem")
                # Use target SR for the layer if known
                # Determine dominant geometry type or use a flexible type
                # For simplicity, start with MultiPolygon, but might need GeometryCollection for mixes
                geom_type = ogr.wkbMultiPolygon  # Default assumption

                # A quick check if non-polygons exist (more robust check might be needed)
                if any(
                    g.GetGeometryType() not in [ogr.wkbPolygon, ogr.wkbMultiPolygon]
                    for g in all_ogr_geoms
                ):
                    logger.warning(
                        "Mixed geometry types found in clip features. Using wkbGeometryCollection for cutline layer."
                    )
                    geom_type = ogr.wkbGeometryCollection

                mem_layer = mem_ds.CreateLayer(
                    "clip", srs=target_sr, geom_type=geom_type
                )

                for ogr_geom in all_ogr_geoms:
                    feat = ogr.Feature(mem_layer.GetLayerDefn())
                    # Try converting points/lines to polygons? Or handle different cutline geometry types?
                    # For simplicity, assume input is polygon-like or Warp handle conversion
                    feat.SetGeometry(ogr_geom)
                    ret_code = mem_layer.CreateFeature(feat)
                    if ret_code != ogr.OGRERR_NONE:
                        logger.warning(
                            f"Failed to add geometry to in-memory layer (OGR Error: {ret_code})."
                        )

                    feat = None  # Dereference

                # Use the temporary layer as the cutline source
                # Calculate available memory for warp (heuristic)
                try:
                    gdal_cache_bytes = (
                        int(os.environ.get("GDAL_CACHEMAX", 1024)) * 1024 * 1024
                    )
                except ValueError:
                    gdal_cache_bytes = (
                        1024 * 1024 * 1024
                    )  # Default to 1GB if env var is invalid
                warp_memory_limit = max(
                    256 * 1024 * 1024, int(gdal_cache_bytes * 0.5)
                )  # Use 50% or min 256MB

                warp_options = gdal.WarpOptions(
                    format="GTiff",  # Output is GTiff
                    creationOptions=cog_options,  # Reapply COG options
                    cutlineDSName="clip_mem",  # Use layer name in memory DS
                    cutlineLayer="clip",
                    # cutlineWhere=None, # Optional SQL filter on clip features
                    cropToCutline=False,  # Keep original extent, mask values outside
                    dstNodata=out_nodata,
                    # targetAlignedPixels=True, # Can sometimes cause issues with cutlines, test if needed
                    multithread=True,
                    warpOptions=["NUM_THREADS=ALL_CPUS"],
                    warpMemoryLimit=warp_memory_limit,
                    # Ensure correct CRS handling if needed (should be aligned now)
                    # srcSRS=target_crs_wkt if target_crs_wkt else None, # Not needed if source/target align
                    # dstSRS=target_crs_wkt if target_crs_wkt else None
                    resampleAlg=resampling_gdal,  # Use same resampling as alignment for consistency when applying mask
                )
                logger.info("Running gdal.Warp for clipping...")
                clip_ds = gdal.Warp(
                    temp_clip_output, str(mosaic_output_path), options=warp_options
                )
                if clip_ds is None:
                    raise RuntimeError("gdal.Warp clipping failed.")

                clip_ds = None  # Close warped dataset before replacing
                mem_ds = None  # Close memory datasource

                # Replace original with clipped version safely
                try:
                    shutil.move(temp_clip_output, str(mosaic_output_path))
                    logger.info("Clipping completed successfully.")
                except Exception as move_err:
                    logger.error(
                        f"Failed to replace original with clipped file: {move_err}"
                    )
                    # Attempt cleanup of temp file maybe?
                    if os.path.exists(temp_clip_output):
                        os.remove(temp_clip_output)
                    raise RuntimeError(
                        "Failed to move clipped file into place."
                    ) from move_err

            except ImportError as e:
                logger.error(
                    f"Clipping requires missing library: {e}. Skipping clipping.",
                    exc_info=True,
                )
                # Decide if this is fatal or just a warning
                # raise RuntimeError("Clipping library missing.") from e # Make it fatal
            except Exception as e:
                logger.error(f"Clipping failed: {e}", exc_info=True)
                # Clean up temp file if it exists and warp failed
                if temp_clip_output and os.path.exists(temp_clip_output):
                    try:
                        os.remove(temp_clip_output)
                    except OSError as rm_err:
                        logger.warning(
                            f"Could not remove temporary clip file {temp_clip_output}: {rm_err}"
                        )
                # Decide if clipping failure is fatal or just a warning
                # logger.warning("Clipping process failed, continuing with unclipped mosaic.")
                raise RuntimeError("Clipping process failed.") from e
            finally:
                # Ensure memory datasource is cleaned up even if warp fails after its creation
                clip_ds = None
                mem_ds = None  # Explicitly dereference

    finally:
        # --- Ensure all datasets are properly closed ---
        logger.debug("Cleaning up datasets and temporary files...")
        # Explicitly dereference GDAL Python objects to help GC and release file handles
        out_band = None
        if out_ds is not None:
            out_ds = None
        if out_ds_overview is not None:
            out_ds_overview = None
        if clip_ds is not None:
            clip_ds = None
        if mem_ds is not None:  # Cleanup clip memory source
            mem_ds = None

        # Close all datasets used for reading (originals or VRTs)
        if isinstance(aligned_datasets, list):
            for i in range(len(aligned_datasets)):
                if aligned_datasets[i] is not None:
                    aligned_datasets[i] = None
            aligned_datasets.clear()

        # Close any remaining handles from input_datasets_info (should be None or released)
        if isinstance(input_datasets_info, list):
            for info in input_datasets_info:
                if isinstance(info, dict) and info.get("ds") is not None:
                    info["ds"] = None

        # --- Clean up temporary VRT files and directory ---
        if isinstance(temp_vrt_files, list):
            for vrt_file in temp_vrt_files:
                for ext in ["", ".aux.xml"]:  # Remove VRT and common sidecar file
                    try:
                        fpath = str(vrt_file) + ext
                        if os.path.exists(fpath):
                            os.remove(fpath)
                    except OSError as e:
                        logger.warning(
                            f"Could not remove temporary VRT file {fpath}: {e}"
                        )
        # Remove temp directory if it was created and exists
        if temp_vrt_dir and os.path.exists(temp_vrt_dir):
            try:
                shutil.rmtree(temp_vrt_dir)
                logger.debug(f"Removed temporary directory {temp_vrt_dir}")
            except Exception as e_shutil:
                logger.warning(
                    f"Could not remove temporary VRT directory {temp_vrt_dir}: {e_shutil}"
                )

    logger.info(
        f"Mosaic COG generation complete. Output saved to: {mosaic_output_path}"
    )
    return str(mosaic_output_path)


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mosaic multiple rasters into a COG using gdal_calc-style alignment and NAN-MAX policy. Always overwrites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raster-paths",
        # nargs="+", # REMOVED
        # type=Path, # REMOVED
        type=str,  # CHANGED
        required=True,
        # help="Paths (local or cloud URI) to input rasters.", # OLD HELP
        help="Required: JSON string representation of a list of input raster paths OR path to a JSON file containing such a list (local or S3 URIs).",  # NEW HELP
    )
    parser.add_argument(
        "--mosaic-output-path",
        type=Path,  # Keep Path here for potential validation/conversion
        required=True,
        help="Path (local or cloud URI) for the output COG GeoTIFF.",
    )
    parser.add_argument(
        "--clip-geometry-path",
        type=Path,  # Keep Path here for potential validation/conversion
        default=None,
        help="Optional path (local or cloud URI) to vector file for clipping.",
    )
    parser.add_argument(
        "--fim-type",
        choices=["depth", "extent"],
        default="depth",
        help="FIM type affects output dtype, nodata and COG predictor.",
    )
    args = parser.parse_args()

    # --- Parse --raster-paths argument (JSON string or file path) ---
    raster_paths_input = args.raster_paths
    loaded_paths = None

    if os.path.isfile(raster_paths_input):
        logger.info(f"Attempting to load raster paths from file: {raster_paths_input}")
        try:
            with open(raster_paths_input, "r") as f:
                loaded_paths = json.load(f)
            logger.info(f"Loaded {len(loaded_paths)} raster paths from file.")
        except json.JSONDecodeError as e:
            logger.critical(
                f"Invalid JSON in file {raster_paths_input}: {e}", exc_info=True
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"Error reading file {raster_paths_input}: {e}", exc_info=True
            )
            sys.exit(1)
    else:
        logger.info("Attempting to load raster paths from direct JSON string argument.")
        try:
            loaded_paths = json.loads(raster_paths_input)
            logger.info(f"Parsed {len(loaded_paths)} raster paths from JSON string.")
        except json.JSONDecodeError as e:
            logger.critical(
                f"Invalid JSON string provided to --raster-paths: {e} (String: '{raster_paths_input[:100]}{'...' if len(raster_paths_input)>100 else ''}')",
                exc_info=False,
            )  # Log error, don't include potentially huge string in exc_info
            sys.exit(1)

    # Validate the loaded paths
    if isinstance(loaded_paths, list):
        # Ensure all elements are strings (or convert if needed, e.g. from Path if JSON contained them)
        raster_paths_str = [str(p) for p in loaded_paths]
        logger.debug(f"Validated raster paths list: {raster_paths_str}")
    else:
        logger.critical(
            f"Error: JSON must contain a LIST of raster paths. Found type: {type(loaded_paths).__name__}"
        )
        sys.exit(1)

    if not raster_paths_str:
        logger.critical("Error: The list of raster paths is empty after parsing.")
        sys.exit(1)
    # --- End --raster-paths parsing ---

    # Convert other Path arguments to strings for GDAL/subprocess compatibility if needed
    # mosaic_output_path_str = str(args.mosaic_output_path)
    # clip_geometry_path_str = str(args.clip_geometry_path) if args.clip_geometry_path else None
    # The function itself now handles PathLikeOrStr, so direct passing might be okay,
    # but converting here ensures strings are passed if any library strictly needs them.
    mosaic_output_path_str = str(args.mosaic_output_path)
    clip_geometry_path_str = (
        str(args.clip_geometry_path) if args.clip_geometry_path else None
    )

    logger.info(f"Starting {JOB_ID} job (gdal_calc style COG output)")
    logger.info(
        f"Input Raster Paths: {len(raster_paths_str)} files (from { 'file' if os.path.isfile(raster_paths_input) else 'string' })"
    )
    logger.info(f"Output Mosaic Path: {mosaic_output_path_str}")
    logger.info(
        f"Clip Geometry Path: {clip_geometry_path_str if clip_geometry_path_str else 'None'}"
    )
    logger.info(f"FIM Type: {args.fim_type}")

    exit_code = 0
    try:
        output_raster = mosaic_rasters_gdalcalc_style(
            raster_paths=raster_paths_str,  # Pass the validated list of strings
            mosaic_output_path=mosaic_output_path_str,
            clip_geometry_path=clip_geometry_path_str,
            fim_type=args.fim_type,
        )
        success_payload = {"mosaic_output_path": output_raster}
        # Use logger.info with extra for structured success logging
        logger.info(
            "Job finished successfully.",
            extra={"level": "SUCCESS", "payload": success_payload},
        )

    except (ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Job failed due to error: {e}", exc_info=True)
        # Use logger.critical with extra for structured error logging
        logger.critical(
            f"{JOB_ID} job run failed: {e}",
            extra={"level": "ERROR", "error_type": type(e).__name__},
        )
        exit_code = 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        logger.critical(
            f"{JOB_ID} job run failed: An unexpected error occurred - {e}",
            extra={"level": "ERROR", "error_type": type(e).__name__},
        )
        exit_code = 1

    # No need for separate "finished successfully" log if using the structured log above
    # if exit_code == 0:
    #    logger.info(f"{JOB_ID} finished successfully.")
    sys.exit(exit_code)
