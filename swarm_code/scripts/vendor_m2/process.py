#!/usr/bin/env python3
"""End-to-end transect processor: LAS/LAZ -> 3 m DTM + product stack + tiles.

    python3 process.py INPUT.las|.laz -o OUTDIR [--cell 3] [--block 2000000]
                       [--no-tiles] [--relief 10,15,30] [--max-points N]

Writes into OUTDIR:
    rasters.npz            dtm, chm, hillshade, svf, relief_<m>, hole, canopy
    summary.json           grid, EPSG/origin, counts, timings, empty-DTM %
    tiles/*.png            1.2 km review tiles (hillshade + relief + canopy)
    transect_row.json      the transects.csv-shaped record for this file

The file never loads more than ``--block`` points plus one raster stack, so a
1 GB LAZ and a 20 MB LAS go through the same code path.  If the file is LAZ
and no decoder is available, the run fails *loudly* and writes a status file
instead of producing an empty DTM that a detector would happily scan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import lasio
import raster
import products


def _reader_for(path, header):
    """Pick the LAS or LAZ point iterator.

    Uncompressed LAS is read with lasio (numpy memmap, no third-party code).
    LAZ needs a LASzip decoder: the VM's system python has none, so we use the
    coordinator-provisioned ``repo/.venv`` (laspy + lazrs).  If neither is
    available the run fails loudly rather than emitting an empty DTM.
    """
    is_compressed = (lasio.laszip_vlr(header) is not None) or lasio.is_laz(path)
    if not is_compressed:
        return lasio.iter_xyz, "las"
    try:
        import lazreader_laspy as lz
        if not lz.have_laspy():
            raise ImportError("laspy not importable in this interpreter")
        lz.check_backend()
        return lz.iter_xyz, "laz(lazrs)"
    except Exception as e:
        raise RuntimeError(
            "LAZ input but no working LAZ decoder in %s (%s).  Run with "
            "repo/.venv/bin/python, or convert the file to LAS first "
            "(pdal translate / laszip) and re-run." % (sys.executable, e))


def process(path, outdir, cell=3.0, block=2_000_000, reliefs=(10, 15, 30),
            tiles=True, max_points=None, upscale=2, epsg_override=None,
            windows_m=(40.0, 20.0, 10.0), thresh_m=0.7, svf_max_dist=90.0,
            max_cells=40_000_000, quiet=False, tile_m=1200.0,
            canopy_thresh=5.0, crop_data=False, crop_margin=90.0,
            dtm_source="hybrid", write_diagnostics=True):
    t_start = time.time()
    timings = {}
    os.makedirs(outdir, exist_ok=True)
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    row = dict(transect=os.path.basename(path), source=path,
               status="failed", error=None, cell_m=cell,
               download_time="", process_time_s=None, empty_dtm_pct=None,
               n_points=None, n_features=0)

    def fail(msg):
        row["status"] = "failed"; row["error"] = msg
        with open(os.path.join(outdir, "transect_row.json"), "w") as f:
            json.dump(row, f, indent=1)
        raise RuntimeError(msg)

    # 1. header ---------------------------------------------------------
    t0 = time.time()
    try:
        h = lasio.read_header(path)
    except Exception as e:
        fail("cannot read header: %r" % (e,))
    npts = h["n_points"]
    if npts == 0:
        fail("header reports 0 points")
    try:
        minx, maxx = sorted((h["minx"], h["maxx"]))
        miny, maxy = sorted((h["miny"], h["maxy"]))
        if not (np.isfinite(minx) and np.isfinite(maxx) and maxx > minx):
            raise ValueError("bad bounds")
        if maxx - minx < 1e-6 or maxy - miny < 1e-6:
            raise ValueError("degenerate bounds")
    except Exception:
        log("header bounds unusable; falling back to a sampling pass")
        minx = miny = np.inf; maxx = maxy = -np.inf
        for x, y, z, blk in lasio.iter_xyz(path, block=block, header=h):
            minx = min(minx, x.min()); maxx = max(maxx, x.max())
            miny = min(miny, y.min()); maxy = max(maxy, y.max())
        if not np.isfinite(minx):
            fail("no points to bound the grid")
    g = raster.Grid.from_bounds(minx, maxx, miny, maxy, cell, pad=2)
    if g.nx * g.ny > max_cells:
        fail("grid %dx%d = %.1fM cells at %.1f m exceeds --max-cells %.1fM; "
             "re-run with a larger --cell or a split along the transect"
             % (g.nx, g.ny, g.nx * g.ny / 1e6, cell, max_cells / 1e6))
    timings["header_s"] = round(time.time() - t0, 2)

    # 2. accumulate -----------------------------------------------------
    t0 = time.time()
    try:
        reader, fmt = _reader_for(path, h)
    except Exception as e:
        fail(str(e))
    log("input %s (%s) %d points, grid %dx%d @ %.1f m" % (path, fmt, npts, g.nx, g.ny, cell))
    last = [time.time()]
    acc = raster.accumulate(path, g, block=block, reader=reader, max_points=max_points)
    timings["accumulate_s"] = round(time.time() - t0, 2)
    timings["points_read"] = acc.n_points_seen
    timings["points_per_s"] = round(acc.n_points_seen / max(1e-9, timings["accumulate_s"]))
    log("accumulated %d points in %.1f s (%.2f M pts/s)"
        % (acc.n_points_seen, timings["accumulate_s"], timings["points_per_s"] / 1e6))
    if acc.n_points_seen == 0:
        fail("no points decoded")
    if crop_data:
        # A transect bbox can be 90% empty (EBA swaths).  Cropping to the data
        # footprint makes rasters, SVF and review tiles honest and cheap; the
        # origin/EPSG sidecar keeps every cell georeferenced.
        cm = int(np.ceil(crop_margin / cell))
        r0, r1, c0, c1 = acc.data_bbox(cm)
        before = (g.nx, g.ny)
        acc = acc.crop(r0, r1, c0, c1)
        g = acc.grid
        if not quiet:
            print("cropped grid %dx%d -> %dx%d cells (margin %.0f m)"
                  % (before[0], before[1], g.nx, g.ny, crop_margin))

    # 3. ground pick ----------------------------------------------------
    t0 = time.time()
    gp = raster.ground_pick(acc.zmin_grid(), acc.count_grid(), cell=cell,
                            windows_m=windows_m, thresh_m=thresh_m,
                            zlow2=acc.zlow2_grid(), zq10=acc.zq10_grid(),
                            source=dtm_source)
    timings["ground_pick_s"] = round(time.time() - t0, 2)
    dtm = gp["dtm"]
    chm = np.clip(acc.zmax_grid().astype(np.float64) - dtm, 0, None)
    chm = np.where(np.isnan(dtm), np.nan, chm)
    canopy_top = np.where(np.isnan(chm), np.nan, np.minimum(chm, 60.0))
    # A cell whose canopy is tall has an interpolated, not measured, ground
    # surface: detectors must not read relief there.  rel_detect excludes those
    # cells plus every unfilled gap.
    canopy_mask = np.nan_to_num(chm) > float(canopy_thresh)
    try:
        from scipy import ndimage as _ndi
        forest_mask = _ndi.binary_dilation(canopy_mask, iterations=1)
    except Exception:
        forest_mask = canopy_mask
    detect_mask = np.isfinite(dtm) & ~gp["big_hole"] & ~forest_mask

    # 4. products -------------------------------------------------------
    t0 = time.time()
    hs = products.hillshade(dtm, cell)
    relief = {}
    for s in reliefs:
        relief[str(int(s))] = products.gauss_relief(dtm, cell, float(s))
    # Pre-clamp relief: mounds/embanked roads up to ~3 m high survive here but are
    # removed from `dtm` by the anti-canopy clamp.  Mound work uses these layers
    # together with detect_mask.
    pre = gp.get("pre_clamp")
    relief_pre = {}
    if pre is not None:
        for s in reliefs:
            relief_pre[str(int(s))] = products.gauss_relief(pre.astype(np.float64), cell, float(s))
    timings["hillshade_relief_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    svf, svf_info = products.sky_view_factor(dtm, cell, max_dist_m=svf_max_dist)
    timings["svf_s"] = round(time.time() - t0, 2)

    empty_pct = float(100 * np.isnan(dtm).mean())
    extra = dict(
        source=os.path.basename(path), format=fmt, epsg=h.get("epsg") or epsg_override,
        las_version="%d.%d" % (h["version_major"], h["version_minor"]),
        point_format=int(h["point_format"]), n_points_header=int(npts),
        accumulate=acc.summary(),
        ground_pick=dict(windows_m=list(windows_m), thresh_m=thresh_m,
                         removed_pct=float(100 * gp["removed"].mean()),
                         outlier_pct=float(100 * gp.get("outlier", np.zeros(1, bool)).mean()),
                         n_stray_replaced=int(gp.get("n_stray_replaced", 0)),
                         n_lone_low_replaced=int(gp.get("n_lone_low", 0)),
                         n_outlier_cells=int(gp.get("outlier", np.zeros(1, bool)).sum()),
                         cells_using_second_lowest=int((acc.count >= 3).sum()),
                         data_hole_pct=float(100 * gp["data_hole"].mean()),
                         big_hole_pct=float(100 * gp["big_hole"].mean()),
                         unfilled_pct=float(100 * gp["unfilled"].mean())),
        svf=svf_info, timings=timings,
        canopy=dict(cover_gt5m_pct=float(100 * np.nanmean(np.nan_to_num(chm) > 5)),
                    mean_m=float(np.nanmean(chm)) if np.isfinite(np.nanmean(chm)) else None,
                    threshold_m=float(canopy_thresh),
                    masked_pct=float(100 * forest_mask.mean())),
        detect=dict(mask_pct=float(100 * detect_mask.mean()),
                    usable_cells=int(detect_mask.sum())),
        tile_m=tile_m, upscale=upscale,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        pipeline="m2/process.py v1",
    )
    if write_diagnostics:
        base = np.float32(np.nanmin(dtm)) if np.isfinite(dtm).any() else np.float32(0)
        np.savez_compressed(
            os.path.join(outdir, "diagnostics.npz"),
            zmin=(acc.zmin_grid() - base).astype(np.float32),
            zq10=(acc.zq10_grid() - base).astype(np.float32),
            zmax=(acc.zmax_grid() - base).astype(np.float32),
            count=acc.count_grid().astype(np.uint16),
            base_z=base)
        d = (acc.zq10_grid() - acc.zmin_grid()).astype(np.float64)
        d = d[np.isfinite(d)]
        extra["per_cell"] = dict(
            q10_minus_min_median_m=float(np.median(d)) if d.size else None,
            q10_minus_min_p90_m=float(np.percentile(d, 90)) if d.size else None,
            q10_minus_min_p99_m=float(np.percentile(d, 99)) if d.size else None,
            dtm_source=dtm_source,
            diagnostics="diagnostics.npz (zmin,zq10,zmax relative to base_z; count)")
    t0 = time.time()
    files = products.write_products(outdir, g, dtm, chm, hs, relief, svf,
                                    gp["data_hole"], canopy_top,
                                    summary_extra=extra, tiles=tiles, upscale=upscale,
                                    removed_mask=gp["removed"], reliefs_pre=relief_pre)
    timings["write_s"] = round(time.time() - t0, 2)
    np.save(os.path.join(outdir, "masks.npy"),
            dict(data_hole=gp["data_hole"], big_hole=gp["big_hole"],
                 removed=gp["removed"], filled=gp["filled"],
                 canopy_mask=canopy_mask, forest_mask=forest_mask,
                 detect_mask=detect_mask, outlier=gp.get("outlier"),
                 pre_clamp=gp.get("pre_clamp")), allow_pickle=True)

    row.update(status="done", process_time_s=round(time.time() - t_start, 2),
               empty_dtm_pct=round(empty_pct, 2), n_points=int(acc.n_points_seen),
               epsg=extra["epsg"], grid=g.to_dict(), outputs=files,
               timings=timings)
    from raster import Grid
    row.pop("grid", None)
    row["grid"] = g.to_dict()
    with open(os.path.join(outdir, "transect_row.json"), "w") as f:
        json.dump(row, f, indent=1)
    log("done %s: empty DTM %.1f%%, %.1f s total, %d tiles"
        % (os.path.basename(path), empty_pct, time.time() - t_start,
           extra["n_tiles"] if "n_tiles" in extra else len(files.get("tile_index", []))))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--outdir", required=True)
    p.add_argument("--cell", type=float, default=3.0)
    p.add_argument("--block", type=int, default=2_000_000)
    p.add_argument("--relief", default="10,15,30")
    p.add_argument("--no-tiles", action="store_true")
    p.add_argument("--max-points", type=int, default=None)
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--tile-m", type=float, default=1200.0)
    p.add_argument("--epsg", type=int, default=None)
    p.add_argument("--max-cells", type=float, default=40e6)
    p.add_argument("--crop-data", action="store_true",
                   help="crop the grid to the data footprint (+--crop-margin)")
    p.add_argument("--crop-margin", type=float, default=90.0)
    p.add_argument("--dtm-source", choices=("hybrid", "q10", "min"), default="hybrid",
                   help="hybrid (default: min, q10 where the min is a lone low excursion), q10, or min")
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    reliefs = tuple(float(v) for v in a.relief.split(",") if v.strip())
    rows = []
    for i, inp in enumerate(a.inputs):
        out = os.path.join(a.outdir, os.path.splitext(os.path.basename(inp))[0]) \
            if len(a.inputs) > 1 else a.outdir
        try:
            rows.append(process(inp, out, cell=a.cell, block=a.block, reliefs=reliefs,
                                tiles=not a.no_tiles, max_points=a.max_points,
                                upscale=a.upscale, epsg_override=a.epsg,
                                max_cells=int(a.max_cells), quiet=a.quiet,
                                tile_m=a.tile_m, crop_data=a.crop_data,
                                crop_margin=a.crop_margin, dtm_source=a.dtm_source,
                                write_diagnostics=not a.no_diagnostics))
        except Exception as e:
            print("FAILED %s: %s" % (inp, e), file=sys.stderr)
            rows.append(dict(transect=os.path.basename(inp), status="failed", error=str(e)))
    with open(os.path.join(a.outdir, "transects_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ok = sum(1 for r in rows if r.get("status") == "done")
    print("%d/%d transects done" % (ok, len(rows)))
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
