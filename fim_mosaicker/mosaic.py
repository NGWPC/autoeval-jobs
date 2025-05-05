#!/usr/bin/env python3
import pdb
import os
import sys
import argparse
import gc
import tempfile
import shutil
import json
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import fsspec
from osgeo import gdal, gdal_array
from osgeo_utils.auxiliary import extent_util
from osgeo_utils.auxiliary.base import PathLikeOrStr
from osgeo_utils.auxiliary.extent_util import Extent, GeoTransform
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.util import open_ds
from pythonjsonlogger import jsonlogger

# GDAL / AWS S3 CONFIGURATION
# 1) Pick up credentials from ENV or IAM role
gdal.SetConfigOption("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
gdal.SetConfigOption("AWS_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN"))
gdal.SetConfigOption("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
gdal.SetConfigOption("CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE", "YES")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
# Enable GDAL exceptions + error logging
gdal.UseExceptions()
gdal.SetConfigOption("CPL_LOG_ERRORS", "ON")


def to_vsi(path: str) -> str:
    """
    Turn an s3://bucket/key URI into GDAL's VSI path /vsis3/bucket/key.
    Leaves any other path unchanged.
    """
    if path.lower().startswith("s3://"):
        bucket_key = path[5:]
        return f"/vsis3/{bucket_key}"
    return path


def setup_logger(name="fim_mosaicker") -> logging.Logger:
    """Return a JSON‐formatter logger with timestamp+level fields."""
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            json_ensure_ascii=False,
        )
    )
    log.addHandler(handler)
    log.propagate = False
    return log


@dataclass
class RasterInfo:
    """Metadata for one input raster."""

    path: str
    ds: gdal.Dataset
    nodata: float
    gt: GeoTransform
    proj: str
    dims: Tuple[int, int]


def load_rasters(paths: List[str], log: logging.Logger) -> List[RasterInfo]:
    """
    Open rasters (local or S3), skip invalid ones, collect GT/proj/dims.
    """
    records: List[RasterInfo] = []
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
        records.append(
            RasterInfo(
                path=p,
                ds=ds,
                nodata=band.GetNoDataValue(),
                gt=gt,
                proj=ds.GetProjection(),
                dims=dims,
            )
        )
    if not records:
        raise ValueError("No valid rasters found")
    return records


def pick_target_grid(
    srcs: List[RasterInfo], log: logging.Logger
) -> Tuple[GeoTransform, Tuple[int, int], str]:
    """
    Choose the lowest‐resolution raster as reference (largest pixel area),
    compute the union extent, and return GT, dims, and CRS WKT.
    """
    # Pick raster with largest pixel area
    ref = max(srcs, key=lambda r: abs(r.gt[1]) * abs(r.gt[5]))
    log.info(f"Reference: {ref.path} (@ res {ref.gt[1]}, {ref.gt[5]})")

    # Build union extent rectangle
    gts = [r.gt for r in srcs]
    dims_list = [r.dims for r in srcs]
    _, _, rect = extent_util.calc_geotransform_and_dimensions(
        gts, dims_list, input_extent=Extent.UNION
    )
    if not isinstance(rect, GeoRectangle):
        raise RuntimeError("Invalid union extent")

    # Compute final grid dims aligned to ref resolution
    rx, ry = abs(ref.gt[1]), abs(ref.gt[5])
    tx = int(np.ceil((rect.max_x - rect.min_x) / rx))
    ty = int(np.ceil((rect.max_y - rect.min_y) / ry))

    gt: GeoTransform = (rect.min_x, rx, 0.0, rect.max_y, 0.0, -ry)
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
    For any source whose grid/proj differs, create an aligned VRT
    at the target GT/dims.  Returns list of GDAL datasets + temp dir.
    """
    tmpdir = tempfile.mkdtemp(prefix="vrt_")
    aligned: List[gdal.Dataset] = []
    for r in srcs:
        same_grid = (
            np.allclose(r.gt, gt, atol=1e-6) and r.dims == dims and r.proj == crs_wkt
        )
        if same_grid:
            aligned.append(r.ds)
        else:
            vrt_fn = Path(r.path).stem + "_aligned.vrt"
            vrt_path = os.path.join(tmpdir, vrt_fn)
            log.info(f"Warp→VRT: {r.path}")
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
            aligned.append(gdal.Open(vrt_path, gdal.GA_ReadOnly))
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
    Windowed NAN‐MAX mosaic into a tiled COG; returns the GDAL Dataset.
    """
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        to_vsi(outpath),
        dims[0],
        dims[1],
        1,
        dtype,
        options=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=LZW",
            "PREDICTOR=2",
            "BIGTIFF=IF_SAFER",
        ],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(crs_wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)

    bx, by = band.GetBlockSize()
    total_blocks = ((dims[0] + bx - 1) // bx) * ((dims[1] + by - 1) // by)
    block_count = 0

    for y in range(0, dims[1], by):
        for x in range(0, dims[0], bx):
            block_count += 1
            h = min(by, dims[1] - y)
            w = min(bx, dims[0] - x)
            acc = np.full((h, w), np.nan, np.float64)
            mask_any = np.zeros((h, w), bool)

            for src_ds in aligned:
                arr = gdal_array.BandReadAsArray(
                    src_ds.GetRasterBand(1), xoff=x, yoff=y, win_xsize=w, win_ysize=h
                ).astype(np.float64)
                nd = src_ds.GetRasterBand(1).GetNoDataValue()
                valid = (~np.isnan(arr)) if np.isnan(nd) else (arr != nd)
                if not valid.any():
                    continue
                arr[~valid] = np.nan
                acc = np.fmax(acc, arr)
                mask_any |= valid

            # prepare output block
            out_arr = np.full(
                (h, w), nodata, np.uint8 if dtype == gdal.GDT_Byte else np.float32
            )
            ok = mask_any & ~np.isnan(acc)
            out_arr[ok] = acc[ok].astype(out_arr.dtype)
            gdal_array.BandWriteArray(band, out_arr, xoff=x, yoff=y)

            if block_count % 100 == 0:
                log.debug(f"Block {block_count}/{total_blocks}")

    ds.FlushCache()
    return ds


def build_overviews(ds: gdal.Dataset, log: logging.Logger):
    """Build overviews on the COG for improved performance."""
    ds = gdal.Open(ds.GetDescription(), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    size = min(ds.RasterXSize, ds.RasterYSize)
    bx, by = band.GetBlockSize()

    levels = []
    lvl = 2
    while (size / lvl) >= max(bx, by):
        levels.append(lvl)
        lvl *= 2

    if levels:
        log.info(f"Building overviews: {levels}")
        ds.BuildOverviews("NEAREST", levels)
    ds = None


def clip_output(src: str, clip_path: str, nodata, log: logging.Logger):
    """Apply an optional cutline mask via gdal.Warp."""
    tmp = src + "_clipped.tif"
    gdal.Warp(
        tmp,
        src,
        options=gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=to_vsi(clip_path),
            cutlineLayer="",
            cropToCutline=False,
            dstNodata=nodata,
            multithread=True,
            creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
        ),
    )
    shutil.move(tmp, src)


def main():
    log = setup_logger()

    p = argparse.ArgumentParser()
    p.add_argument(
        "--raster_paths",
        required=True,
        help="JSON list of rasters or path to JSON file",
    )
    p.add_argument("--mosaic_output_path", required=True)
    p.add_argument("--clip_geometry_path", default=None)
    p.add_argument(
        "--fim_type",
        choices=["depth", "extent"],
        default="extent",
        help="‘extent’→byte, 255 nodata; ‘depth’→float32",
    )
    args = p.parse_args()

    # Load input paths (either JSON text or JSON file)
    txt = args.raster_paths
    if os.path.isfile(txt):
        paths = json.load(open(txt))
    else:
        paths = json.loads(txt)

    rasters = load_rasters([str(p) for p in paths], log)
    gt, dims, crs = pick_target_grid(rasters, log)

    aligned_ds, tmpdir = build_vrts(rasters, gt, dims, crs, log)

    # Pick output dtype/nodata
    if args.fim_type == "extent":
        dtype = gdal.GDT_Byte
        nodata = 255
    else:
        dtype = gdal.GDT_Float32
        nodata = -3.4028235e38

    # Create a local temp file for the mosaic
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name

    # Mosaic into that temp COG
    out_ds = mosaic_blocks(aligned_ds, tmp_out, gt, dims, crs, dtype, nodata, log)
    build_overviews(out_ds, log)
    out_ds.FlushCache()
    del out_ds
    gc.collect()

    # Optional clipping to vector geometry
    if args.clip_geometry_path:
        clip_output(tmp_out, args.clip_geometry_path, nodata, log)
        gc.collect()

    # Push the temp COG to the final URI via fsspec (doing it this way to avoid delete object errors with IAM permissions on Test Nomad)
    final = args.mosaic_output_path
    log.info(f"Pushing temp COG → {final}")
    with open(tmp_out, "rb") as ifp, fsspec.open(final, "wb") as ofp:
        shutil.copyfileobj(ifp, ofp)
    os.remove(tmp_out)

    # Cleanup
    for r in rasters:
        r.ds = None
    for ds in aligned_ds:
        ds = None
    shutil.rmtree(tmpdir, ignore_errors=True)

    log.info("Mosaic COG complete → %s", args.mosaic_output_path)


if __name__ == "__main__":
    main()
