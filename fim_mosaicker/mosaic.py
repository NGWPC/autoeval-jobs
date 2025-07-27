#!/usr/bin/env python3
import argparse
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import fsspec
import numpy as np
from osgeo import gdal, gdal_array
from osgeo_utils.auxiliary import extent_util
from osgeo_utils.auxiliary.extent_util import Extent, GeoTransform
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.util import open_ds

from utils.logging import setup_logger

# Enable GDAL exceptions
gdal.UseExceptions()


def to_vsi(path: str) -> str:
    """
    Convert a standard file path or S3 path to a GDAL VSI path. This allows us to feed in either a standard filesystem path or an S3 object path to GDAL and have the script be able to work with either without the user having to think about it.
    """
    if path.lower().startswith("s3://"):
        return "/vsis3/" + path[5:]
    return path


JOB_ID = "fim_mosaicker"


@dataclass
class RasterInfo:
    path: str
    ds: gdal.Dataset
    nodata: float
    gt: GeoTransform
    proj: str
    dims: Tuple[int, int]


def load_rasters(paths: List[str], log: logging.Logger) -> List[RasterInfo]:
    """
    Groups raster info into RasterInfo dataclasses. Ensures later processing functions operate
    on only valid rasters.
    """
    recs: List[RasterInfo] = []
    for p in paths:
        vst = to_vsi(p)
        ds = open_ds(vst, access_mode=gdal.OF_RASTER | gdal.OF_VERBOSE_ERROR)
        if not ds:
            log.warning(f"Could not open {p}")
            continue
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform(can_return_null=True)
        dims = (ds.RasterXSize, ds.RasterYSize)
        if gt is None or 0 in dims:
            log.warning(f"Skipping {p}, missing GT or zero size")
            ds = None
            continue
        recs.append(
            RasterInfo(
                path=p,
                ds=ds,
                nodata=band.GetNoDataValue(),
                gt=gt,
                proj=ds.GetProjection(),
                dims=dims,
            )
        )
    if not recs:
        raise ValueError("No valid rasters found")
    return recs


def pick_target_grid(srcs: List[RasterInfo], log: logging.Logger) -> Tuple[GeoTransform, Tuple[int, int], str]:
    """
    Finds the lowest resolution raster and picks that as the resolution to project to.
    Determines the bounds of the final mosaic. Also, uses the projection of the lowest
    resolution raster as the reference projection for the output mosaic.
    """
    # pick lowest resolution (largest abs(pixel area))
    ref = max(srcs, key=lambda r: abs(r.gt[1]) * abs(r.gt[5]))
    log.info(f"Reference: {ref.path} (@ res {ref.gt[1]}, {ref.gt[5]})")

    gts = [r.gt for r in srcs]
    dims_list = [r.dims for r in srcs]
    _, _, rect = extent_util.calc_geotransform_and_dimensions(gts, dims_list, input_extent=Extent.UNION)
    if not isinstance(rect, GeoRectangle):
        raise RuntimeError("Invalid union extent")

    rx, ry = abs(ref.gt[1]), abs(ref.gt[5])
    tx = int(np.ceil((rect.max_x - rect.min_x) / rx))
    ty = int(np.ceil((rect.max_y - rect.min_y) / ry))
    gt = (rect.min_x, rx, 0.0, rect.max_y, 0.0, -ry)
    log.info(f"Target grid: {tx}×{ty}, GT={gt}")
    return gt, (tx, ty), ref.proj


def build_vrts(
    srcs: List[RasterInfo],
    gt: GeoTransform,
    dims: Tuple[int, int],
    crs_wkt: str,
    log: logging.Logger,
) -> Tuple[List[gdal.Dataset], str]:
    """
    Warps rasters to the projection, resolution, and alignment returned by the
    "pick_target_grid" function. The returned warped rasters are VRT's stored in
    stored in a temporary directory that is cleaned up after the script exits.
    This approach is an efficient way to compare multiple input rasters that may
    have heterogenous projections, alignments, resolutions, etc.
    """
    tmpdir = tempfile.mkdtemp(prefix="vrt_")
    aligned = []
    for r in srcs:
        same = np.allclose(r.gt, gt, atol=1e-6) and r.dims == dims and r.proj == crs_wkt
        if same:
            aligned.append(r.ds)
        else:
            log.info(f"Warp into VRT: {r.path}")
            vrt_name = f"{Path(r.path).stem}_aligned.vrt"
            vrt_path = os.path.join(tmpdir, vrt_name)
            gdal.Warp(
                vrt_path,
                to_vsi(r.path),
                options=gdal.WarpOptions(
                    format="VRT",
                    outputBounds=(
                        gt[0],
                        gt[3] + dims[1] * gt[5],
                        gt[0] + dims[0] * gt[1],
                        gt[3],
                    ),
                    width=dims[0],
                    height=dims[1],
                    dstSRS=crs_wkt,
                    resampleAlg=gdal.GRIORA_Bilinear,
                    multithread=True,
                ),
            )
            ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            aligned.append(ds)
    return aligned, tmpdir


def mosaic_blocks(
    aligned: List[gdal.Dataset],
    outpath: str,
    gt: GeoTransform,
    dims: Tuple[int, int],
    crs_wkt: str,
    dtype: int,
    nodata,
    log: logging.Logger,
) -> gdal.Dataset:
    """
    Do a block‐wise NaN‐max merge of aligned rasters into a final mosaic in  COG format.
    Reuses four fixed buffers at the driver’s block size to keep memory constant to do
    the merging. Numpy is used to perform the max operation by casting the blocks to
    Numpy arrays to keep things performant. This approach supports mosaicing many, large
    rasters as long as the input rasters are tiled.
    """
    # Create with GTiff driver first, then convert to COG
    gtiff_drv = gdal.GetDriverByName("GTiff")
    temp_gtiff = outpath.replace(".tif", "_temp.tif")
    ds = gtiff_drv.Create(
        temp_gtiff,
        dims[0],
        dims[1],
        1,
        dtype,
        options=[
            "TILED=YES",
            f"BLOCKXSIZE={os.getenv('MOSAIC_BLOCK_SIZE', '512')}",
            f"BLOCKYSIZE={os.getenv('MOSAIC_BLOCK_SIZE', '512')}",
            f"COMPRESS={os.getenv('MOSAIC_COMPRESS_TYPE', 'LZW')}",
            f"PREDICTOR={os.getenv('MOSAIC_PREDICTOR', '2')}",
            "BIGTIFF=IF_SAFER",
        ],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(crs_wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)

    # block‐size
    bx, by = band.GetBlockSize()
    total_blocks = ((dims[0] + bx - 1) // bx) * ((dims[1] + by - 1) // by)
    block_count = 0

    # pre-alloc fixed buffers at max shape (by, bx)
    read_buf = np.empty((by, bx), dtype=np.float64)
    acc_buf = np.empty((by, bx), dtype=np.float64)
    mask_buf = np.empty((by, bx), dtype=bool)
    if dtype == gdal.GDT_Byte:
        write_buf = np.empty((by, bx), dtype=np.uint8)
    else:
        write_buf = np.empty((by, bx), dtype=np.float32)

    for y in range(0, dims[1], by):
        for x in range(0, dims[0], bx):
            h = min(by, dims[1] - y)
            w = min(bx, dims[0] - x)

            # slice out just the h×w region
            acc = acc_buf[:h, :w]
            acc.fill(np.nan)
            mask_any = mask_buf[:h, :w]
            mask_any.fill(False)

            # read & fmax each source
            for src_ds in aligned:
                src_band = src_ds.GetRasterBand(1)
                arr = gdal_array.BandReadAsArray(
                    src_band,
                    xoff=x,
                    yoff=y,
                    win_xsize=w,
                    win_ysize=h,
                    buf_obj=read_buf[:h, :w],
                )
                ndv = src_band.GetNoDataValue()
                if ndv is not None and not np.isnan(ndv):
                    valid = arr != ndv
                else:
                    valid = ~np.isnan(arr)
                arr[~valid] = np.nan
                # in‐place fmax
                np.fmax(acc, arr, out=acc)
                mask_any |= valid

            # prepare output tile
            out = write_buf[:h, :w]
            out.fill(nodata)
            ok = mask_any & ~np.isnan(acc)
            out[ok] = acc[ok].astype(out.dtype)

            # write it
            gdal_array.BandWriteArray(band, out, xoff=x, yoff=y)

            block_count += 1
            if block_count % 100 == 0:
                log.debug(f"Block {block_count}/{total_blocks}")

    ds.FlushCache()
    ds = None  # Close the GTiff

    # Convert GTiff to COG
    log.info(f"Converting GTiff to COG: {outpath}")
    cog_drv = gdal.GetDriverByName("COG")
    src_ds = gdal.Open(temp_gtiff, gdal.GA_ReadOnly)
    cog_ds = cog_drv.CreateCopy(outpath, src_ds, strict=0, options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
    src_ds = None

    # Clean up temp file
    os.remove(temp_gtiff)

    return cog_ds


def clip_output(src: str, clip_path: str, nodata, log: logging.Logger):
    """
    Apply a clip geometry to the mosaic. Functionally this step will often
    be necessary since we are producing metrics using two mosaicked rasters that
    need to be clipped to have the same ROI so that a pixelwise agreement can be computed.
    """
    tmp = src + "_clipped.tif"

    with fsspec.open(clip_path, "rb") as f:
        # Create temporary file for GDAL to use
        temp_clip = tempfile.NamedTemporaryFile(suffix=Path(clip_path).suffix, delete=False).name
        with open(temp_clip, "wb") as local_file:
            shutil.copyfileobj(f, local_file)

    try:
        # Get layer name from temporary file
        clip_ds = gdal.OpenEx(temp_clip, gdal.OF_VECTOR)
        if not clip_ds:
            raise ValueError(f"Could not open clip geometry file: {clip_path}")

        layer_name = clip_ds.GetLayer(0).GetName()
        clip_ds = None
        log.info(f"Using layer '{layer_name}' for clipping")

        gdal.Warp(
            tmp,
            src,
            options=gdal.WarpOptions(
                format="COG",
                cutlineDSName=temp_clip,
                cutlineLayer=layer_name,
                cropToCutline=True,
                dstNodata=nodata,
                multithread=True,
                creationOptions=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
            ),
        )

        shutil.move(tmp, src)
    finally:
        os.remove(temp_clip)


def main():
    """
    Mosaic and clip raster files into a single COG (Cloud Optimized GeoTIFF).

    This job supports both local files and S3 sources or destinations. The logging format follows that outlined in job_conventions.md

    Arguments:
        --raster_paths           Required. One or more input raster file paths, or a single directory containing rasters.
        --mosaic_output_path     Required. Output path for the mosaic (local file path or S3 URI).
        --clip_geometry_path     Optional. Path to a vector file (e.g., GeoJSON, Shapefile) used to clip the output mosaic.
        --fim_type               Optional. Either 'extent' (default, produces a byte raster with nodata=255) or 
                                 'depth' (produces a float32 raster with nodata=-9999).

    Example usage:

        python mosaic_script.py \\
            --raster_paths s3://mybucket/tiles/*.tif \\
            --mosaic_output_path s3://mybucket/output/mosaic_cog.tif \\
            --clip_geometry_path ./aoi.geojson \\
            --fim_type depth
    """
    log = setup_logger(JOB_ID)
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raster_paths",
        required=True,
        nargs="+",
        help="Directory of rasters or a space-separated list of paths. List can be a single string or space seperated tokens if calling script directly from bash.",
    )
    p.add_argument("--mosaic_output_path", required=True)
    p.add_argument("--clip_geometry_path", default=None)
    p.add_argument(
        "--fim_type",
        choices=["depth", "extent"],
        default="extent",
        help="‘extent’→ byte, 255 nodata; ‘depth’ -> float32, -9999 no data",
    )
    args = p.parse_args()
    try:
        # load raster paths
        input_rp = args.raster_paths
        if len(input_rp) == 1 and os.path.isdir(input_rp[0]):
            raster_extensions = {".tif", ".tiff", ".vrt", ".img", ".hdf", ".nc", ".netcdf"}
            all_files = list(Path(input_rp[0]).iterdir())
            paths = [str(p) for p in all_files if p.is_file() and p.suffix.lower() in raster_extensions]
            log.info(f"Found {len(paths)} raster files (out of {len(all_files)} total) in directory {input_rp}")
        else:
            # either multiple shell-split tokens or one quoted string
            joined = " ".join(input_rp)
            # split on whitespace to get list of paths
            paths = joined.split()
            log.info(f"Using {len(paths)} provided raster paths")

        rasters = load_rasters([str(p) for p in paths], log)
        gt, dims, crs = pick_target_grid(rasters, log)

        aligned_ds, tmpdir = build_vrts(rasters, gt, dims, crs, log)

        # choose output type
        if args.fim_type == "extent":
            dtype, nodata = gdal.GDT_Byte, int(os.getenv("EXTENT_NODATA_VALUE", "255"))
        else:
            dtype, nodata = gdal.GDT_Float32, float(os.getenv("DEPTH_NODATA_VALUE", "-9999"))

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name

        # do the merge
        out_ds = mosaic_blocks(aligned_ds, tmp_out, gt, dims, crs, dtype, nodata, log)
        out_ds.FlushCache()
        out_ds = None
        gc.collect()

        if args.clip_geometry_path:
            clip_output(tmp_out, args.clip_geometry_path, nodata, log)
            gc.collect()

        # push via fsspec
        log.info(f"Pushing temp COG → {args.mosaic_output_path}")
        with open(tmp_out, "rb") as ifp, fsspec.open(args.mosaic_output_path, "wb") as ofp:
            shutil.copyfileobj(ifp, ofp)
        os.remove(tmp_out)

        # close all warped VRTs (so GDAL cache can free memory)
        for i in range(len(aligned_ds)):
            aligned_ds[i] = None
        aligned_ds.clear()
        shutil.rmtree(tmpdir, ignore_errors=True)

        # also close original rasters
        for r in rasters:
            r.ds = None

        log.success({"mosaic_output_path": args.mosaic_output_path})

    except Exception as e:
        # Last record on failure must be ERROR
        log.error(f"{JOB_ID} run failed: {type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
