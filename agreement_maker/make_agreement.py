#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Tuple

import fsspec
import geopandas as gpd
import gval
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import xarray as xr
from dask import delayed
from dask.distributed import Client, LocalCluster
from fsspec.core import url_to_fs
from rasterio import features
from rasterio.env import Env
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


def cross_walk_gval_fim(metric_df: pd.DataFrame, cell_area: int, masked_count: int) -> dict:
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

    metric_df["TP_perc"] = (tp / total_population) * 100 if total_population > 0 else "NA"
    metric_df["FP_perc"] = (fp / total_population) * 100 if total_population > 0 else "NA"
    metric_df["TN_perc"] = (tn / total_population) * 100 if total_population > 0 else "NA"
    metric_df["FN_perc"] = (fn / total_population) * 100 if total_population > 0 else "NA"

    predPositive = tp + fp
    predNegative = tn + fn
    obsPositive = tp + fn
    obsNegative = tn + fp

    metric_df["cell_area_m2"] = cell_area
    sq_km_converter = 1000000

    # This checks if a cell_area has been provided, thus making areal calculations possible.
    metric_df["TP_area_km2"] = (tp * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["FP_area_km2"] = (fp * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["TN_area_km2"] = (tn * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["FN_area_km2"] = (fn * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["contingency_tot_area_km2"] = (
        (total_population * cell_area) / sq_km_converter if cell_area is not None else None
    )

    metric_df["predPositive_area_km2"] = (predPositive * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["predNegative_area_km2"] = (predNegative * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["obsPositive_area_km2"] = (obsPositive * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["obsNegative_area_km2"] = (obsNegative * cell_area) / sq_km_converter if cell_area is not None else None
    metric_df["positiveDiff_area_km2"] = (
        (metric_df["predPositive_area_km2"] - metric_df["obsPositive_area_km2"])[0] if cell_area is not None else None
    )

    total_pop_and_mask_pop = total_population + masked_count if masked_count > 0 else None
    metric_df["masked_count"] = masked_count if masked_count > 0 else 0
    metric_df["masked_perc"] = (masked_count / total_pop_and_mask_pop) * 100 if masked_count > 0 else 0
    metric_df["masked_area_km2"] = (masked_count * cell_area) / sq_km_converter if masked_count > 0 else 0
    metric_df["predPositive_perc"] = (predPositive / total_population) * 100 if total_population > 0 else "NA"
    metric_df["predNegative_perc"] = (predNegative / total_population) * 100 if total_population > 0 else "NA"
    metric_df["obsPositive_perc"] = (obsPositive / total_population) * 100 if total_population > 0 else "NA"
    metric_df["obsNegative_perc"] = (obsNegative / total_population) * 100 if total_population > 0 else "NA"
    metric_df["positiveDiff_perc"] = (
        metric_df["predPositive_perc"].values[0] - metric_df["obsPositive_perc"].values[0]
        if total_population > 0
        else "NA"
    )

    return {x: y for x, y in zip(metric_df.columns, metric_df.values[0])}


def setup_dask_cluster(log: logging.Logger) -> Tuple[Client, LocalCluster]:
    """Set up a local Dask cluster and return the client and cluster."""
    log.info("Starting Dask local cluster")
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        memory_limit=DASK_CLUST_MAX_MEM,
    )
    client = Client(cluster)
    log.info(f"Dask dashboard link: {client.dashboard_link}")
    return client, cluster


def load_rasters(candidate_path: str, benchmark_path: str, log: logging.Logger) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load candidate and benchmark rasters without remapping."""
    log.info(f"Loading candidate raster: {candidate_path}")
    candidate = rxr.open_rasterio(
        candidate_path,
        mask_and_scale=True,
        chunks={"x": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")), "y": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048"))},
        lock=False,
    )
    log.info(f"Loading benchmark raster: {benchmark_path}")
    benchmark = rxr.open_rasterio(
        benchmark_path,
        mask_and_scale=True,
        chunks={"x": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048")), "y": int(os.getenv("RASTERIO_CHUNK_SIZE", "2048"))},
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
    candidate.data = xr.where(candidate == candidate.rio.nodata, nodata_value, candidate)
    candidate = candidate.rio.write_nodata(nodata_value)
    benchmark.data = xr.where(benchmark == benchmark.rio.nodata, nodata_value, benchmark)
    benchmark = benchmark.rio.write_nodata(nodata_value)

    return candidate, benchmark


def create_exclusion_masks(candidate: xr.DataArray, mask_dict: dict, log: logging.Logger) -> gpd.GeoDataFrame:
    """Apply exclusion masks and return combined mask geometry."""
    all_masks_df = None

    for poly_layer in mask_dict:
        operation = mask_dict[poly_layer]["operation"]

        if operation == "exclude":
            poly_path = mask_dict[poly_layer]["path"]
            buffer_val = 0 if mask_dict[poly_layer]["buffer"] is None else mask_dict[poly_layer]["buffer"]

            log.info(f"Processing exclusion mask: {poly_layer}")

            with fsspec.open(poly_path, "rb") as f:
                poly_all = gpd.read_file(f, bbox=candidate.rio.bounds())

            # Make sure features are present in bounding box area before projecting
            if poly_all.empty:
                log.warning(f"No features found in bounding box for {poly_layer}")
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

    # Apply exclusion masks
    all_masks_df = create_exclusion_masks(candidate, mask_dict, log) if mask_dict else None

    log.info("Homogenizing rasters")
    c_aligned, b_aligned = candidate.gval.homogenize(benchmark_map=benchmark, target_map="candidate")

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

    # Clean up aligned rasters
    del c_aligned, b_aligned
    gc.collect()

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

    # Final validation: check if agreement map has any valid data
    valid_agreement_data = ((agreement_map != 255) & (agreement_map != 4)).sum().compute()
    if valid_agreement_data == 0:
        log.error("Agreement map contains no valid data - all pixels are either nodata or masked")
        sys.exit(1)

    log.info("Computing crosstab table for metrics")
    crosstab_table = agreement_map.gval.compute_crosstab()

    # Only compute and write metrics if metrics_path is provided
    if metrics_path:
        log.info("Computing metrics table and writing")
        metrics_table = crosstab_table.gval.compute_categorical_metrics(
            positive_categories=[1], negative_categories=[0], metrics="all"
        )

        # Calculate cell area from agreement map transform
        transform = agreement_map.rio.transform()
        cell_area = abs(transform[0]) * abs(transform[4])  # pixel width * pixel height
        log.info(f"Calculated cell area: {cell_area} square meters")

        # Count masked pixels (value 4)
        masked_count = int((agreement_map == 4).sum().compute())
        log.info(f"Masked pixel count: {masked_count}")

        # Apply cross_walk_gval_fim to enrich metrics
        log.info("Applying cross_walk_gval_fim to enrich metrics")
        enriched_metrics_dict = cross_walk_gval_fim(metrics_table, cell_area, masked_count)

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
        # use rasterio to write agreement map (better for large rasters)
        tasks = []
        output_profile = {
            "driver": "GTiff",
            "height": agreement_map.rio.height,
            "width": agreement_map.rio.width,
            "count": 1,
            "dtype": agreement_map.dtype,
            "crs": agreement_map.rio.crs,
            "transform": agreement_map.rio.transform(),
            "compress": "LZW",
            "tiled": True,
            "blockxsize": block_size,
            "blockysize": block_size,
            "nodata": 255,  # Updated nodata value
        }

        # Write data block by block
        with rasterio.open(temp_tiff_path, "w", **output_profile) as dst:
            for ij, window in dst.block_windows(1):
                i, j = ij

                @delayed
                def write_window(win, ii, jj):
                    try:
                        block = agreement_map.isel(
                            x=slice(win.col_off, win.col_off + win.width),
                            y=slice(win.row_off, win.row_off + win.height),
                        ).compute()  # only compute this block!

                        arr2d = np.squeeze(block.values).astype("uint8")
                        with rasterio.open(temp_tiff_path, "r+") as d:
                            d.write(arr2d, 1, window=win)
                        return (ii, jj, True, None)
                    except Exception as e:
                        return (ii, jj, False, str(e))

                tasks.append(write_window(window, i, j))

        # Compute all tasks and check results
        results = client.compute(tasks, sync=True)

        # Check for any failed writes
        failed_writes = [(r[0], r[1], r[3]) for r in results if not r[2]]
        if failed_writes:
            error_msg = f"Failed to write {len(failed_writes)} blocks: "
            error_msg += "; ".join([f"Block ({i},{j}): {err}" for i, j, err in failed_writes[:5]])
            if len(failed_writes) > 5:
                error_msg += f" ... and {len(failed_writes) - 5} more"
            raise RuntimeError(error_msg)

        # Convert to COG locally
        log.info("Converting to Cloud Optimized GeoTIFF")

        # Configure COG profile
        cog_profile = LZWProfile().data.copy()
        cog_profile.update(
            {
                "BLOCKSIZE": int(os.getenv("COG_BLOCKSIZE", "512")),
                "OVERVIEW_RESAMPLING": "nearest",
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
        candidate, benchmark = load_rasters(args.candidate_path, args.benchmark_path, log)

        # Load mask dictionary if provided
        mask_dict = {}
        if args.mask_dict:
            log.info(f"Loading mask dictionary from {args.mask_dict}")
            with open_file(args.mask_dict, "rt") as f:
                mask_dict = json.load(f)

        # Compute agreement map
        agreement_map = compute_agreement_map(candidate, benchmark, args.metrics_path, mask_dict, log)

        # Write agreement map to GeoTIFF
        with rasterio.Env():
            # Set nodata to 255 for writing (following reference implementation)
            agreement_map_write = agreement_map.rio.write_nodata(255, encoded=True)
            write_agreement_map(agreement_map_write, args.output_path, client, int(args.block_size), log)

        success_outputs = {"output_path": args.output_path}
        if args.metrics_path:
            success_outputs["metrics_path"] = args.metrics_path
        log.success(success_outputs)

    except Exception as e:
        log.error(f"{JOB_ID} run failed: {e}")
        sys.exit(1)

    finally:
        log.info("Shutting down Dask client and cluster")
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
