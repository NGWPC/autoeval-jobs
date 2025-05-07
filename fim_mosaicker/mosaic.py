#!/usr/bin/env python3
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
from osgeo_utils.auxiliary.extent_util import Extent, GeoTransform
from osgeo_utils.auxiliary.rectangle import GeoRectangle
from osgeo_utils.auxiliary.util import open_ds
from pythonjsonlogger import jsonlogger

# GDAL / AWS S3 CONFIGURATION
gdal.SetConfigOption("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
gdal.SetConfigOption("AWS_SESSION_TOKEN", os.getenv("AWS_SESSION_TOKEN"))
gdal.SetConfigOption("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
gdal.SetConfigOption("CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE", "YES")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
gdal.UseExceptions()
gdal.SetConfigOption("CPL_LOG_ERRORS", "ON")


def to_vsi(path: str) -> str:
    if path.lower().startswith("s3://"):
        return "/vsis3/" + path[5:]
    return path


def setup_logger(name="fim_mosaicker") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    h = logging.StreamHandler(sys.stderr)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    h.setFormatter(
        jsonlogger.JsonFormatter(
            fmt=fmt,
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            json_ensure_ascii=False,
        )
    )
    log.addHandler(h)
    log.propagate = False
    return log


@dataclass
class RasterInfo:
    path: str
    ds: gdal.Dataset
    nodata: float
    gt: GeoTransform
    proj: str
    dims: Tuple[int, int]


def load_rasters(paths: List[str], log: logging.Logger) -> List[RasterInfo]:
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


def pick_target_grid(
    srcs: List[RasterInfo], log: logging.Logger
) -> Tuple[GeoTransform, Tuple[int, int], str]:
    # pick lowest resolution (largest abs(pixel area))
    ref = max(srcs, key=lambda r: abs(r.gt[1]) * abs(r.gt[5]))
    log.info(f"Reference: {ref.path} (@ res {ref.gt[1]}, {ref.gt[5]})")

    gts = [r.gt for r in srcs]
    dims_list = [r.dims for r in srcs]
    _, _, rect = extent_util.calc_geotransform_and_dimensions(
        gts, dims_list, input_extent=Extent.UNION
    )
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
    Warp any mis-aligned sources into per-tile VRTs at the target grid.
    We’ll keep them alive in-memory only long enough to read their blocks,
    then close them explicitly.
    """
    tmpdir = tempfile.mkdtemp(prefix="vrt_")
    aligned = []
    for r in srcs:
        same = np.allclose(r.gt, gt, atol=1e-6) and r.dims == dims and r.proj == crs_wkt
        if same:
            aligned.append(r.ds)
        else:
            log.info(f"Warp→VRT: {r.path}")
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
    Do a block‐wise NaN‐max merge into a COG.  Reuses four fixed buffers
    at the driver’s block size to keep memory constant.
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
    return ds


def build_overviews(ds: gdal.Dataset, log: logging.Logger):
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
        help="‘extent’→ byte, 255 nodata; ‘depth’ → float32, -9999 no data",
    )
    args = p.parse_args()

    if os.path.isfile(args.raster_paths):
        paths = json.load(open(args.raster_paths))
    else:
        paths = json.loads(args.raster_paths)

    rasters = load_rasters([str(p) for p in paths], log)
    gt, dims, crs = pick_target_grid(rasters, log)

    aligned_ds, tmpdir = build_vrts(rasters, gt, dims, crs, log)

    # choose output type
    if args.fim_type == "extent":
        dtype, nodata = gdal.GDT_Byte, 255
    else:
        dtype, nodata = gdal.GDT_Float32, -9999

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name

    # do the merge
    out_ds = mosaic_blocks(aligned_ds, tmp_out, gt, dims, crs, dtype, nodata, log)
    build_overviews(out_ds, log)
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

    log.info("Mosaic COG complete → %s", args.mosaic_output_path)


if __name__ == "__main__":
    main()
