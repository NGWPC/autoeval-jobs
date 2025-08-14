#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import warnings
from typing import Tuple

import dask
import fsspec
import geopandas as gpd
import gval
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from gval_optimizations import apply_gval_optimizations
from dask import delayed
from dask.distributed import Client, LocalCluster
from fsspec.core import url_to_fs
from rasterio import features
from rasterio.enums import Resampling
from rasterio.env import Env
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import transform
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import LZWProfile

from utils.logging import setup_logger
from utils.pairing import AGREEMENT_PAIRING_DICT

# GLOBAL DASK CONFIGURATION
DASK_CLUST_MAX_MEM = os.getenv("DASK_CLUST_MAX_MEM")


JOB_ID = "agreement_maker"


def open_file(path: str, mode: str = "rb"):
    """
    Open a local or remote file (s3://, gcs://, http://, etc.) via fsspec.
    Returns a file-like object.
    """
    fs, fs_path = url_to_fs(path)
    return fs.open(fs_path, mode)


def cross_walk_gval_fim(
    metric_df: pd.DataFrame, cell_area: int, masked_count: int
) -> dict:
    """
    Crosswalks metrics made from GVAL to standard FIM names and conventions

    Parameters
    ----------
    metric_df: pd.DataFrame
        Dataframe for getting
    cell_area: int
        Area in meters of squared resolution
    masked_count: int
        How many pixels are masked

    Returns
    -------
    dict
        Dictionary of statistical metrics
    """

    # Remove band entry
    metric_df = metric_df.iloc[:, 1:]

    # Dictionary to crosswalk column names
    crosswalk = {
        "tn": "true_negatives_count",
        "fn": "false_negatives_count",
        "fp": "false_positives_count",
        "tp": "true_positives_count",
        "accuracy": "ACC",
        "balanced_accuracy": "Bal_ACC",
        "critical_success_index": "CSI",
        "equitable_threat_score": "EQUITABLE_THREAT_SCORE",
        "f_score": "F1_SCORE",
        "false_discovery_rate": "FAR",
        "false_negative_rate": "PND",
        "false_omission_rate": "FALSE_OMISSION_RATE",
        "false_positive_rate": "FALSE_POSITIVE_RATE",
        "fowlkes_mallows_index": "FOWLKES_MALLOW_INDEX",
        "matthews_correlation_coefficient": "MCC",
        "negative_likelihood_ratio": "NEGATIVE_LIKELIHOOD_RATIO",
        "negative_predictive_value": "NPV",
        "overall_bias": "BIAS",
        "positive_likelihood_ratio": "POSITIVE_LIKELIHOOD_RATIO",
        "positive_predictive_value": "PPV",
        "prevalence": "PREVALENCE",
        "prevalence_threshold": "PREVALENCE_THRESHOLD",
        "true_negative_rate": "TNR",
        "true_positive_rate": "TPR",
    }

    metric_df.columns = [crosswalk[x] for x in metric_df.columns]

    # Build
    tn, fn, tp, fp = (
        metric_df["true_negatives_count"].values[0],
        metric_df["false_negatives_count"].values[0],
        metric_df["true_positives_count"].values[0],
        metric_df["false_positives_count"].values[0],
    )
    total_population = tn + fn + tp + fp
    metric_df["contingency_tot_count"] = total_population

    metric_df["TP_perc"] = (
        (tp / total_population) * 100 if total_population > 0 else "NA"
    )
    metric_df["FP_perc"] = (
        (fp / total_population) * 100 if total_population > 0 else "NA"
    )
    metric_df["TN_perc"] = (
        (tn / total_population) * 100 if total_population > 0 else "NA"
    )
    metric_df["FN_perc"] = (
        (fn / total_population) * 100 if total_population > 0 else "NA"
    )

    predPositive = tp + fp
    predNegative = tn + fn
    obsPositive = tp + fn
    obsNegative = tn + fp

    metric_df["cell_area_m2"] = cell_area
    sq_km_converter = 1000000

    # This checks if a cell_area has been provided, thus making areal calculations possible.
    metric_df["TP_area_km2"] = (
        (tp * cell_area) / sq_km_converter if cell_area is not None else None
    )
    metric_df["FP_area_km2"] = (
        (fp * cell_area) / sq_km_converter if cell_area is not None else None
    )
    metric_df["TN_area_km2"] = (
        (tn * cell_area) / sq_km_converter if cell_area is not None else None
    )
    metric_df["FN_area_km2"] = (
        (fn * cell_area) / sq_km_converter if cell_area is not None else None
    )
    metric_df["contingency_tot_area_km2"] = (
        (total_population * cell_area) / sq_km_converter
        if cell_area is not None
        else None
    )

    metric_df["predPositive_area_km2"] = (
        (predPositive * cell_area) / sq_km_converter
        if cell_area is not None
        else None
    )
    metric_df["predNegative_area_km2"] = (
        (predNegative * cell_area) / sq_km_converter
        if cell_area is not None
        else None
    )
    metric_df["obsPositive_area_km2"] = (
        (obsPositive * cell_area) / sq_km_converter
        if cell_area is not None
        else None
    )
    metric_df["obsNegative_area_km2"] = (
        (obsNegative * cell_area) / sq_km_converter
        if cell_area is not None
        else None
    )
    metric_df["positiveDiff_area_km2"] = (
        (
            metric_df["predPositive_area_km2"]
            - metric_df["obsPositive_area_km2"]
        )[0]
        if cell_area is not None
        else None
    )

    total_pop_and_mask_pop = (
        total_population + masked_count if masked_count > 0 else None
    )
    metric_df["masked_count"] = masked_count if masked_count > 0 else 0
    metric_df["masked_perc"] = (
        (masked_count / total_pop_and_mask_pop) * 100 if masked_count > 0 else 0
    )
    metric_df["masked_area_km2"] = (
        (masked_count * cell_area) / sq_km_converter if masked_count > 0 else 0
    )
    metric_df["predPositive_perc"] = (
        (predPositive / total_population) * 100
        if total_population > 0
        else "NA"
    )
    metric_df["predNegative_perc"] = (
        (predNegative / total_population) * 100
        if total_population > 0
        else "NA"
    )
    metric_df["obsPositive_perc"] = (
        (obsPositive / total_population) * 100 if total_population > 0 else "NA"
    )
    metric_df["obsNegative_perc"] = (
        (obsNegative / total_population) * 100 if total_population > 0 else "NA"
    )
    metric_df["positiveDiff_perc"] = (
        metric_df["predPositive_perc"].values[0]
        - metric_df["obsPositive_perc"].values[0]
        if total_population > 0
        else "NA"
    )

    return {x: y for x, y in zip(metric_df.columns, metric_df.values[0])}


def setup_dask_cluster(log: logging.Logger) -> Tuple[Client, LocalCluster]:
    """Set up a local Dask cluster and return the client and cluster."""
    log.info("Starting Dask local cluster")

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,  # Keep single thread to avoid memory pressure
        memory_limit=DASK_CLUST_MAX_MEM,
        processes=False,  # Use threads instead of processes to reduce memory overhead
        silence_logs=False,
    )

    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,  # GC more aggressively
            "distributed.worker.memory.spill": 0.75,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": 0.9,
            "distributed.comm.compression": "lz4",  # Faster compression
            "distributed.scheduler.allowed-failures": 5,
            "distributed.client.heartbeat": "10s",
        }
    )

    client = Client(cluster)

    log.info(f"Dask dashboard link: {client.dashboard_link}")
    return client, cluster


def load_rasters(
    candidate_path: str, benchmark_path: str, log: logging.Logger
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load candidate and benchmark rasters without remapping."""
    log.info(f"Loading candidate raster: {candidate_path}")
    candidate = rxr.open_rasterio(
        candidate_path,
        mask_and_scale=True,
        chunks={
            "x": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")),
            "y": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")),
        },
        lock=False,
    )
    log.info(f"Loading benchmark raster: {benchmark_path}")
    benchmark = rxr.open_rasterio(
        benchmark_path,
        mask_and_scale=True,
        chunks={
            "x": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")),
            "y": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")),
        },
        lock=False,
    )

    # Validate that rasters have data
    if candidate.size == 0:
        log.error("Candidate raster is empty")
        sys.exit(1)
    if benchmark.size == 0:
        log.error("Benchmark raster is empty")
        sys.exit(1)

    # Handle nodata values by converting them to a consistent nodata value (255)
    log.info("Processing nodata values")
    nodata_value = int(os.getenv("EXTENT_NODATA_VALUE", "255"))
    candidate.data = xr.where(
        candidate == candidate.rio.nodata, nodata_value, candidate
    )
    candidate = candidate.rio.write_nodata(nodata_value)
    benchmark.data = xr.where(
        benchmark == benchmark.rio.nodata, nodata_value, benchmark
    )
    benchmark = benchmark.rio.write_nodata(nodata_value)

    return candidate, benchmark


def create_exclusion_masks(
    candidate: xr.DataArray, mask_dict: dict, log: logging.Logger
) -> gpd.GeoDataFrame:
    """Apply exclusion masks and return combined mask geometry."""
    all_masks_df = None

    for poly_layer in mask_dict:
        operation = mask_dict[poly_layer]["operation"]

        if operation == "exclude":
            poly_path = mask_dict[poly_layer]["path"]
            buffer_val = (
                0
                if mask_dict[poly_layer]["buffer"] is None
                else mask_dict[poly_layer]["buffer"]
            )

            log.info(f"Processing exclusion mask: {poly_layer}")

            with fsspec.open(poly_path, "rb") as f:
                poly_all = gpd.read_file(f, bbox=candidate.rio.bounds())

            # Make sure features are present in bounding box area before projecting
            if poly_all.empty:
                log.warning(
                    f"No features found in bounding box for {poly_layer}"
                )
                del poly_all
                gc.collect()
                continue

            # Project layer to reference crs
            poly_all_proj = poly_all.to_crs(candidate.rio.crs)

            # Buffer if buffer val exists
            if buffer_val != 0:
                poly_all_proj = poly_all_proj.buffer(buffer_val)

            if all_masks_df is not None:
                all_masks_df = pd.concat([all_masks_df, poly_all_proj])
            else:
                all_masks_df = poly_all_proj

            del poly_all, poly_all_proj
            gc.collect()

    return all_masks_df


def compute_agreement_map(
    candidate: xr.DataArray,
    benchmark: xr.DataArray,
    metrics_path: str,
    mask_dict: dict,
    log: logging.Logger,
) -> xr.DataArray:
    """Compute the agreement map between candidate and benchmark rasters using pairing dictionary."""

    # Use shared pairing dictionary from utils
    pairing_dictionary = AGREEMENT_PAIRING_DICT

    # Store reference transform and CRS for final validation
    reference_transform = candidate.rio.transform()
    reference_crs = candidate.rio.crs

    # Apply exclusion masks
    all_masks_df = (
        create_exclusion_masks(candidate, mask_dict, log) if mask_dict else None
    )

    log.info("Homogenizing rasters")
    c_aligned, b_aligned = candidate.gval.homogenize(
        benchmark_map=benchmark,
        target_map="candidate",
        resampling=Resampling.nearest,  # Use nearest categorical flood extent data. Bilinear for depth. TODO make this configurable as part of improvements to handle depth rasters
    )

    # Validate that homogenization produced valid results
    if c_aligned.size == 0 or b_aligned.size == 0:
        log.error("Homogenization failed - no overlapping area between rasters")
        sys.exit(1)

    # Check if rasters have any valid (non-nodata) data after homogenization
    valid_candidate = (c_aligned != 255).sum().compute()
    valid_benchmark = (b_aligned != 255).sum().compute()

    if valid_candidate == 0:
        log.error("Candidate raster has no valid data after homogenization")
        sys.exit(1)
    if valid_benchmark == 0:
        log.error("Benchmark raster has no valid data after homogenization")
        sys.exit(1)

    # Clean up original rasters
    del candidate, benchmark
    gc.collect()

    log.info("Computing agreement map using pairing dictionary")
    agreement_map = c_aligned.gval.compute_agreement_map(
        benchmark_map=b_aligned,
        comparison_function="pairing_dict",
        pairing_dict=pairing_dictionary,
    )

    # Validate that agreement map was successfully created
    if agreement_map is None or agreement_map.size == 0:
        log.error("Failed to compute agreement map")
        sys.exit(1)

    # Cast any NaNs reintroduced by gval.compute_agreement_map to nodata
    agreement_map = agreement_map.where(~np.isnan(agreement_map), 255)

    # Set pairing dictionary as attribute for later use by gval functions
    agreement_map.attrs["pairing_dictionary"] = pairing_dictionary

    # Apply masking if exclusion masks are present
    if all_masks_df is not None:
        log.info("Applying exclusion masks to agreement map")

        # Get the transform and shape from the agreement map
        transform = agreement_map.rio.transform()
        height = agreement_map.rio.height
        width = agreement_map.rio.width

        # Rasterize the exclusion geometries to create a mask
        # Areas inside geometries will have value 1, outside will be 0
        mask_raster = features.rasterize(
            shapes=[(geom, 1) for geom in all_masks_df.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        # Update agreement map: set excluded areas (mask==1) to value 4
        # but preserve original nodata values (255)
        agreement_map.data = xr.where(
            (mask_raster == 1) & (agreement_map != 255),
            4,  # Set to masked value
            agreement_map,  # Keep original value
        )

    # Clean up aligned rasters
    del c_aligned, b_aligned
    gc.collect()

    # Final validation: check if agreement map has any valid data
    valid_agreement_data = (
        ((agreement_map != 255) & (agreement_map != 4)).sum().compute()
    )
    if valid_agreement_data == 0:
        log.error(
            "Agreement map contains no valid data - all pixels are either nodata or masked"
        )
        sys.exit(1)

    # Apply gval optimizations for memory-efficient crosstab computation
    apply_gval_optimizations()

    log.info(
        "Computing crosstab table for metrics using monkey patched compute_crosstab"
    )
    crosstab_table = agreement_map.gval.compute_crosstab()

    # Only compute and write metrics if metrics_path is provided
    if metrics_path:
        log.info("Computing metrics table and writing")
        metrics_table = crosstab_table.gval.compute_categorical_metrics(
            positive_categories=[1], negative_categories=[0], metrics="all"
        )

        # Calculate cell area from agreement map transform
        transform = agreement_map.rio.transform()
        cell_area = abs(transform[0]) * abs(
            transform[4]
        )  # pixel width * pixel height
        log.info(f"Calculated cell area: {cell_area} square meters")

        # Count masked pixels (value 4)
        masked_count = int((agreement_map == 4).sum().compute())
        log.info(f"Masked pixel count: {masked_count}")

        # Apply cross_walk_gval_fim to enrich metrics
        log.info("Applying cross_walk_gval_fim to enrich metrics")
        enriched_metrics_dict = cross_walk_gval_fim(
            metrics_table, cell_area, masked_count
        )

        # Convert enriched metrics dictionary back to DataFrame
        enriched_metrics_df = pd.DataFrame([enriched_metrics_dict])

        # Write enriched metrics table using fsspec for S3 compatibility
        with open_file(metrics_path, "wt") as f:
            enriched_metrics_df.to_csv(f, index=False)

        # Clean up metrics table
        del metrics_table

    # Clean up
    del crosstab_table
    gc.collect()

    # Final transform validation - ensure agreement map has proper georeference information
    if (
        not hasattr(agreement_map.rio, "transform")
        or agreement_map.rio.transform() is None
        or not hasattr(agreement_map.rio, "crs")
        or agreement_map.rio.crs is None
    ):
        log.info("Restoring georeference information to agreement map")
        agreement_map = agreement_map.rio.write_transform(reference_transform)
        agreement_map = agreement_map.rio.write_crs(reference_crs)

    return agreement_map


def write_agreement_map(
    agreement_map: xr.DataArray,
    outpath: str,
    client: Client,
    block_size: int,
    log: logging.Logger,
) -> None:
    log.info(f"Writing agreement map to {outpath}")

    # Create temporary output paths for local work
    temp_fd, temp_tiff_path = tempfile.mkstemp(suffix="_temp.tif")
    os.close(temp_fd)

    temp_fd, temp_cog_path = tempfile.mkstemp(suffix="_cog.tif")
    os.close(temp_fd)

    try:
        # Persist the agreement map to workers to avoid large graph serialization
        log.info("Persisting agreement map to workers")
        agreement_map = agreement_map.persist()
        
        # use rasterio to write agreement map (better for large rasters)
        tasks = []

        # Store transform outside delayed function
        base_transform = agreement_map.rio.transform()
        base_crs = agreement_map.rio.crs

        output_profile = {
            "driver": "GTiff",
            "height": agreement_map.rio.height,
            "width": agreement_map.rio.width,
            "count": 1,
            "dtype": agreement_map.dtype,
            "crs": base_crs,
            "transform": base_transform,
            "compress": "LZW",
            "tiled": True,
            "blockxsize": block_size,
            "blockysize": block_size,
            "nodata": 255,  # Updated nodata value
        }

        # Write data block by block using batch processing to reduce graph size
        with rasterio.open(temp_tiff_path, "w", **output_profile) as dst:
            windows = list(dst.block_windows(1))
            batch_size = min(16, len(windows))  # Process in smaller batches

            for batch_start in range(0, len(windows), batch_size):
                batch_end = min(batch_start + batch_size, len(windows))
                batch_tasks = []

                for idx in range(batch_start, batch_end):
                    ij, window = windows[idx]
                    i, j = ij

                    # Pre-compute block to avoid large graph serialization
                    # With persist() called earlier, this should be fast
                    block = agreement_map.isel(
                        x=slice(window.col_off, window.col_off + window.width),
                        y=slice(window.row_off, window.row_off + window.height),
                    ).compute()

                    @delayed
                    def write_window_batch(
                        computed_block, win, ii, jj, base_tf, base_crs_val
                    ):
                        try:
                            # Validate block has transform
                            if (
                                not hasattr(computed_block.rio, "transform")
                                or computed_block.rio.transform() is None
                            ):
                                # Calculate window-specific transform
                                block_transform = transform(win, base_tf)
                                computed_block = (
                                    computed_block.rio.write_transform(
                                        block_transform
                                    )
                                )
                                computed_block = computed_block.rio.write_crs(
                                    base_crs_val
                                )
                                log.info(
                                    f"Restored transform for block ({ii}, {jj})"
                                )

                            # Safely extract 2D array from block, handling various dimension scenarios
                            if (
                                computed_block.ndim == 3
                                and computed_block.shape[0] == 1
                            ):
                                # 3D array with single band - extract the first band
                                arr2d = computed_block.values[0, :, :].astype(
                                    "uint8"
                                )
                            elif computed_block.ndim == 2:
                                # Already 2D - use as is
                                arr2d = computed_block.values.astype("uint8")
                            else:
                                # Fallback for unexpected dimensions
                                arr2d = computed_block.values.squeeze().astype(
                                    "uint8"
                                )
                                # Ensure we have a 2D array
                                if arr2d.ndim != 2:
                                    # Force reshape to match window dimensions
                                    arr2d = arr2d.reshape(
                                        (win.height, win.width)
                                    )

                            # Return the array and window for writing
                            return (ii, jj, True, arr2d, win)
                        except Exception as e:
                            log.error(
                                f"Block ({ii}, {jj}) write failed: {str(e)}"
                            )
                            return (ii, jj, False, None, None)

                    batch_tasks.append(
                        write_window_batch(
                            block, window, i, j, base_transform, base_crs
                        )
                    )

                # Process batch and clear memory
                log.info(
                    f"Processing batch {batch_start // batch_size + 1}/{(len(windows) + batch_size - 1) // batch_size}"
                )
                batch_results = client.compute(batch_tasks, sync=True)

                # Write the results to the file and check for failures
                failed_in_batch = []
                for r in batch_results:
                    ii, jj, success, arr2d, win = r
                    if success and arr2d is not None and win is not None:
                        # Write the array to the file
                        dst.write(arr2d, 1, window=win)
                    elif not success:
                        failed_in_batch.append((ii, jj, "Processing failed"))
                
                if failed_in_batch:
                    raise RuntimeError(
                        f"Failed to write {len(failed_in_batch)} blocks in batch: {failed_in_batch}"
                    )

                # Clear batch tasks to free memory
                del batch_tasks, batch_results

        log.info("Agreement map written successfully")

        # Convert to COG locally
        log.info("Converting to Cloud Optimized GeoTIFF")

        # Configure COG profile
        cog_profile = LZWProfile().data.copy()
        # Set blocksize via profile (cog_translate calculates tilesize from blockxsize/blockysize)
        blocksize = int(os.getenv("COG_BLOCKSIZE", "512"))
        cog_profile.update(
            {
                "blockxsize": blocksize,
                "blockysize": blocksize,
            }
        )

        # Writing cog to a tempfile and then uploading to fsspec in one go to avoid needing to use delete object privilages (nomad clients won't have that)
        cog_translate(
            temp_tiff_path,
            temp_cog_path,
            cog_profile,
            overview_level=int(os.getenv("COG_OVERVIEW_LEVEL", "4")),
            overview_resampling="nearest",
            quiet=True,
        )

        # Upload to S3 if output path is S3, otherwise just move the file
        if outpath.startswith("s3://"):
            log.info(f"Uploading COG to S3: {outpath}")
            fs, fs_path = url_to_fs(outpath)
            fs.put_file(temp_cog_path, fs_path)
        else:
            # For local output, just move the file
            shutil.move(temp_cog_path, outpath)

        log.info("Agreement map write completed")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_tiff_path):
            os.remove(temp_tiff_path)
        if os.path.exists(temp_cog_path):
            os.remove(temp_cog_path)


def main():
    log = setup_logger(JOB_ID)

    # Configure warnings to treat NotGeoreferencedWarning as an error
    warnings.filterwarnings("error", category=NotGeoreferencedWarning)

    # Parse command-line arguments
    p = argparse.ArgumentParser(description="Compare two raster datasets.")
    p.add_argument(
        "--fim_type",
        required=True,
        choices=["depth", "extent"],
        help="Specifies whether agreement is based on spatial 'extent' overlap (binary) or potentially 'depth' values. Influences output raster format.",
    )
    p.add_argument(
        "--candidate_path",
        required=True,
        help="Path to candidate raster (local or S3)",
    )
    p.add_argument(
        "--benchmark_path",
        required=True,
        help="Path to benchmark raster (local or S3)",
    )
    p.add_argument(
        "--output_path",
        required=True,
        help="Path for output agreement map (local or S3)",
    )
    p.add_argument(
        "--metrics_path",
        required=False,
        help="Optional path for output metrics table (local or S3)",
    )
    p.add_argument(
        "--mask_dict",
        required=False,
        help="Optional path/URI to a JSON file containing mask dictionary with geometry paths, operations, and buffer settings.",
    )
    p.add_argument(
        "--block_size",
        required=False,
        default=os.getenv("DEFAULT_WRITE_BLOCK_SIZE", "4096"),
        help="Block size for writing agreement raster. Default is 4096.",
    )

    args = p.parse_args()

    # Set up Dask cluster
    client, cluster = setup_dask_cluster(log)

    try:
        # Load and preprocess rasters
        candidate, benchmark = load_rasters(
            args.candidate_path, args.benchmark_path, log
        )

        # Load mask dictionary if provided
        mask_dict = {}
        if args.mask_dict:
            log.info(f"Loading mask dictionary from {args.mask_dict}")
            with open_file(args.mask_dict, "rt") as f:
                mask_dict = json.load(f)

        # Compute agreement map
        agreement_map = compute_agreement_map(
            candidate, benchmark, args.metrics_path, mask_dict, log
        )

        # Write agreement map to GeoTIFF
        with rasterio.Env(
            CHECK_WITH_INVERT_PROJ=True,
            GTIFF_FORCE_RGBA=False,  # Prevent unwanted band expansion
        ):
            # Store transform and CRS before write_nodata (which may strip them)
            stored_transform = agreement_map.rio.transform()
            stored_crs = agreement_map.rio.crs

            # Set nodata to 255 for writing (following reference implementation)
            agreement_map_write = agreement_map.rio.write_nodata(
                255, encoded=True
            )

            # Restore transform and CRS if lost during write_nodata
            if (
                not hasattr(agreement_map_write.rio, "transform")
                or agreement_map_write.rio.transform() is None
            ):
                log.info("Restoring transform after write_nodata operation")
                agreement_map_write = agreement_map_write.rio.write_transform(
                    stored_transform
                )
                agreement_map_write = agreement_map_write.rio.write_crs(
                    stored_crs
                )

            write_agreement_map(
                agreement_map_write,
                args.output_path,
                client,
                int(args.block_size),
                log,
            )

        success_outputs = {"output_path": args.output_path}
        if args.metrics_path:
            success_outputs["metrics_path"] = args.metrics_path
        log.success(success_outputs)

    except NotGeoreferencedWarning as e:
        log.error(f"Raster file is not georeferenced: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        sys.exit(1)

    finally:
        log.info("Shutting down Dask client and cluster")
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
