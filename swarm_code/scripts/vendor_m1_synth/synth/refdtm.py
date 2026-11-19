"""Reference 3 m DTM/products for the synthetic benchmark (m1).

This is NOT the production processor: m2 owns that.  It exists so the
generator and the detection metrics can be validated against a known,
deliberately simple implementation of the method the previous run settled on:

  1. stream all returns (ignore the class field), take the LOWEST return per
     3 m cell;
  2. remove cells more than 0.7 m above a grey-opened, smoothed surface at
     window sizes 40, 20, 10 m in turn (canopy contamination);
  3. fill only holes up to 5 cells wide;
  4. multi-scale Gaussian local relief at 10 / 15 / 30 m, blanked within
     15 m of any data hole (empty-data halos fake rings);
  5. sky-view factor (8 azimuths, octave distances) - approximation;
  6. CHM = cell max z - DTM.

Output .npz carries x0,y0,x1,y1,grid_m,epsg so nothing needs GDAL/rasterio.
Memory is O(cells), not O(points), so a 12.5 km transect is ~5 MB of arrays.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def scan_laz(path, grid=3.0, block=4_000_000, verbose=False):
    """Stream a LAS/LAZ file and reduce to per-cell min/max/count."""
    import laspy
    mnx = mny = np.inf
    mxx = mxy = -np.inf
    mnz, mxz = np.inf, -np.inf
    n = 0
    t = time.time()
    with laspy.open(path, laz_backend=laspy.LazBackend.LazrsParallel) as f:
        h = f.header
        crs = h.parse_crs()
        epsg = crs.to_epsg() if crs is not None else None
        for pts in f.chunk_iterator(block):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float64)
            n += x.size
            mnx = min(mnx, x.min()); mxx = max(mxx, x.max())
            mny = min(mny, y.min()); mxy = max(mxy, y.max())
            mnz = min(mnz, z.min()); mxz = max(mxz, z.max())
            yield x, y, z
    scan_laz.info = dict(n_points=n, bounds=[mnx, mny, mnz, mxx, mxy, mxz],
                         epsg=epsg,
                         offsets=[float(v) for v in h.offsets],
                         scales=[float(v) for v in h.scales],
                         seconds=time.time() - t)


def build_dtm(path, grid=3.0, thresh=0.7, fill_cells=5, relief_win=(10, 15, 30),
              hole_blank_m=15.0, svf=True, block=4_000_000, verbose=True,
              bbox=None, method="median", vstruct_min_extent=2.5,
              vstruct_max_blob_cells=4):
    """Build 3 m products from one LAS/LAZ file.

    method: how contaminated (too-high) cells are rejected:
      'literal' - grey-opening then gaussian smoothing at 40/20/10 m, reject
                  cells > thresh above that surface (the recipe as written in
                  the project brief);
      'median'  - median filter at 40/20/10 m, same rejection (edge preserving:
                  a narrow ditch does not depress the reference surface);
      'min'     - minimum filter (erosion) at 10/20/40 m;
      'vstruct' - median filter, but a too-high cell is only rejected when the
                  returns in that cell also span >= vstruct_min_extent metres
                  (real canopy structure); a 0.5-3 m mound has no structure
                  above it and therefore survives.  This is the method that
                  keeps mound villages; see notes/;
      'none'    - keep every lowest-return cell (control).
    The choice is recorded in the output; see notes/ for the comparison.
    """
    t0 = time.time()
    timings = {}
    # ---- pass 1: reduce points to cells (single streaming pass) -----------
    gen = scan_laz(path, grid=grid, block=block)
    # first chunk fixes the grid extent; we then keep growing arrays if needed
    if bbox is None:
        import laspy
        with laspy.open(path, laz_backend=laspy.LazBackend.Lazrs) as f:
            b = f.header.mins, f.header.maxs
            bbox = (float(b[0][0]), float(b[0][1]), float(b[1][0]), float(b[1][1]))
    x0, y0, x1, y1 = bbox

def products_from_minz(minz, maxz, cnt, grid=3.0, thresh=0.7,
                       fill_cells=5, relief_win=(10, 15, 30),
                       hole_blank_m=15.0, svf=True, method='vstruct',
                       vstruct_min_extent=2.5, vstruct_max_blob_cells=4,
                       low_thresh=6.0, low_window_cells=7):
    """Turn a per-cell min/max/count raster into the product set."""
    t = time.time()
    timings = {}
    z0 = np.asarray(minz, dtype=np.float32).copy()
    z0[cnt == 0] = np.nan
    maxz = np.asarray(maxz, dtype=np.float32)
    dem = z0.astype(np.float64)
    valid = np.isfinite(dem)
    removed = np.zeros_like(valid)
    removed_low = np.zeros_like(valid)
    # ---- low outliers (multipath / below-ground returns) ------------------
    # A min-per-cell DTM inherits every low outlier as a pit.  Threshold is
    # 6 m: deeper than the 4 m maximum documented ditch depth, so genuine
    # ditches survive, while the isolated 6-60 m pits do not.
    t = time.time()
    if method != "none" and low_thresh and valid.any():
        for _ in range(2):
            idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                                return_indices=True)
            wf = dem[tuple(idx)]
            ref_low = ndimage.median_filter(wf, size=low_window_cells,
                                            mode="nearest")
            low = valid & (dem < ref_low - low_thresh)
            if not low.any():
                break
            removed_low |= low
            valid &= ~low
            dem[low] = np.nan
    timings["low_outlier_s"] = time.time() - t
    # ---- ground picking: lowest return per cell ---------------------------
    t = time.time()
    if method != "none" and valid.any():
        for w_m in (40.0, 20.0, 10.0):
            k = max(3, int(round(w_m / grid)) | 1)
            if not valid.any():
                break
            idx = ndimage.distance_transform_edt(~valid, return_distances=False,
                                                return_indices=True)
            wf = dem[tuple(idx)]
            if method == "literal":
                op = ndimage.grey_opening(wf, size=(k, k), mode="nearest")
                ref = ndimage.gaussian_filter(op, sigma=max(1.0, k / 4.0),
                                              mode="nearest")
            elif method == "median":
                ref = ndimage.median_filter(wf, size=k, mode="nearest")
            elif method == "min":
                ref = ndimage.minimum_filter(wf, size=k, mode="nearest")
            elif method == "vstruct":
                # median surface + only reject cells that ALSO show vertical
                # structure above them (canopy), so 0.5-3 m mounds survive
                ref = ndimage.median_filter(wf, size=k, mode="nearest")
            else:
                raise ValueError("unknown method %r" % method)
            hi = valid & (dem > ref + thresh)
            if method == "vstruct":
                # reject a too-high cell only if it is canopy: either the
                # returns in the cell span >= 2.5 m (vertical structure), or
                # the high patch is <= 4 cells (nothing cultural is that
                # small; smallest mounds are ~8 m across = ~7 cells of 3 m)
                extent = np.where(np.isfinite(maxz - z0), maxz - z0, 0.0)
                lab, nl = ndimage.label(hi)
                if nl:
                    area = ndimage.sum(np.ones_like(dem, dtype=np.float32), lab,
                                       index=np.arange(1, nl + 1))
                    tiny = np.isin(lab, np.nonzero(area <= vstruct_max_blob_cells)[0] + 1)
                else:
                    tiny = np.zeros_like(hi)
                bad = hi & ((extent > vstruct_min_extent) | tiny)
            else:
                bad = hi
            removed |= bad
            valid &= ~bad
            dem[bad] = np.nan
    timings["ground_filter_s"] = time.time() - t

    # ---- hole filling: only small holes ----------------------------------
    t = time.time()
    no_data = (~np.isfinite(z0)) | removed
    filled = np.zeros_like(no_data)
    lab, nlab = ndimage.label(no_data)
    if nlab:
        dist = ndimage.distance_transform_edt(no_data)
        maxd = ndimage.maximum(dist, lab, index=np.arange(1, nlab + 1))
        area = ndimage.sum(np.ones_like(dist), lab, index=np.arange(1, nlab + 1))
        ok_ids = np.nonzero((maxd <= 0.5 * fill_cells + 0.5)
                            & (area <= fill_cells ** 2))[0] + 1
        if ok_ids.size:
            small = np.isin(lab, ok_ids)
            idx = ndimage.distance_transform_edt(no_data, return_distances=False,
                                                 return_indices=True)
            nearest = np.where(valid, dem, np.nan)[tuple(idx)]
            dem[small] = nearest[small]
            filled = small
    hole = no_data & ~filled        # remaining holes (unfilled)
    good = np.isfinite(dem)
    timings["hole_fill_s"] = time.time() - t

    # ---- relief models ----------------------------------------------------
    t = time.time()
    if good.any():
        idx = ndimage.distance_transform_edt(~good, return_distances=False,
                                             return_indices=True)
        dem_all = dem[tuple(idx)]
    else:
        dem_all = np.zeros_like(dem)
    relief = {}
    for w in relief_win:
        sig = max(1.0, (w / grid) / 2.0)
        base = ndimage.gaussian_filter(dem_all, sigma=sig, mode="nearest")
        r = dem - base
        r[~good] = np.nan
        relief["relief%d" % int(w)] = r.astype(np.float32)
    # Blank relief near *unfilled data holes* only (15 m): empty-data halos
    # fake rings.  Do NOT use the vegetation-removed cells here: under partial
    # canopy they are isolated single cells, and a 15 m buffer around each one
    # erases the whole pasture interior (measured: 70% of a 60 m box).
    d_hole = ndimage.distance_transform_edt(~hole) * grid
    near_hole = d_hole <= hole_blank_m
    for k in relief:
        relief[k][near_hole] = np.nan
    timings["relief_s"] = time.time() - t

    # ---- CHM --------------------------------------------------------------
    chm = np.where(cnt > 0, maxz - z0, np.nan)
    chm[np.isfinite(chm) & (chm < 0)] = 0.0

    # ---- sky view factor (approximation) ----------------------------------
    t = time.time()
    svf = sky_view_factor(dem_all, grid) if svf else np.full_like(dem_all, np.nan)
    svf[~good] = np.nan
    timings["svf_s"] = time.time() - t

    out = dict(dem=dem.astype(np.float32), z0=z0.astype(np.float32),
               count=cnt, removed=removed, filled=filled, hole=hole,
               no_data=no_data, near_hole=near_hole, removed_low=removed_low,
               valid=good, chm=chm.astype(np.float32), svf=svf.astype(np.float32),
               method=np.array(method), grid_m=float(grid))
    out.update(relief)
    stats = dict(empty_pct=100.0 * float((~np.isfinite(z0)).mean()),
                 removed_pct=100.0 * float(removed.mean()),
                 filled_pct=100.0 * float(filled.mean()),
                 hole_pct=100.0 * float(hole.mean()), method=method,
                 removed_low_pct=100.0 * float(removed_low.mean()),
                 valid_pct=100.0 * float(good.mean()),
                 z_range=[float(np.nanmin(dem)) if good.any() else None,
                          float(np.nanmax(dem)) if good.any() else None],
                 timings=timings)
    return out, stats


def build_dtm(path, grid=3.0, thresh=0.7, fill_cells=5, relief_win=(10, 15, 30),
              hole_blank_m=15.0, svf=True, block=4_000_000, verbose=True,
              bbox=None, method="vstruct", vstruct_min_extent=2.5,
              vstruct_max_blob_cells=4, low_thresh=6.0, low_window_cells=7):
    """Build 3 m products from one LAS/LAZ file (streaming, bounded memory).

    method: how contaminated (too-high) cells are rejected:
      'vstruct' - (default) median reference surface at 40/20/10 m; a cell is
                  rejected only when it is > thresh above it AND (its returns
                  span >= 2.5 m vertically OR the high patch is <= 4 cells).
                  Keeps 0.5-3 m mounds and causeways, removes canopy;
      'median'  - median filter at 40/20/10 m, reject > thresh above;
      'literal' - grey-opening + smoothing at 40/20/10 m, reject > thresh
                  (the recipe as written in the brief - measured to erase
                  40 m ring controls in the relief model, see notes/);
      'min'     - minimum filter (erosion) at 40/20/10 m;
      'none'    - keep every lowest-return cell (no canopy rejection).
    """
    t0 = time.time()
    if bbox is None:
        import laspy
        with laspy.open(path, laz_backend=laspy.LazBackend.Lazrs) as f:
            b = f.header.mins, f.header.maxs
            bbox = (float(b[0][0]), float(b[0][1]), float(b[1][0]), float(b[1][1]))
    x0, y0, x1, y1 = bbox
    x0 = float(np.floor(x0 / grid) * grid)      # snap the grid to whole cells
    y0 = float(np.floor(y0 / grid) * grid)
    nx = int(np.ceil((x1 - x0) / grid))
    ny = int(np.ceil((y1 - y0) / grid))
    minz = np.full((ny, nx), np.inf, dtype=np.float32)
    maxz = np.full((ny, nx), -np.inf, dtype=np.float32)
    cnt = np.zeros((ny, nx), dtype=np.int32)
    n_pts = 0
    gen = scan_laz(path, grid=grid, block=block)
    for x, y, z in gen:
        ix = np.clip(((x - x0) / grid).astype(np.int64), 0, nx - 1)
        iy = np.clip(((y - y0) / grid).astype(np.int64), 0, ny - 1)
        flat = iy * nx + ix
        np.minimum.at(minz.reshape(-1), flat, z.astype(np.float32))
        np.maximum.at(maxz.reshape(-1), flat, z.astype(np.float32))
        np.add.at(cnt.reshape(-1), flat, 1)
        n_pts += x.size
    info = dict(scan_laz.info)
    out, stats = products_from_minz(minz, maxz, cnt, grid=grid, thresh=thresh,
                                    fill_cells=fill_cells, relief_win=relief_win,
                                    hole_blank_m=hole_blank_m, svf=svf,
                                    method=method,
                                    vstruct_min_extent=vstruct_min_extent,
                                    vstruct_max_blob_cells=vstruct_max_blob_cells,
                                    low_thresh=low_thresh,
                                    low_window_cells=low_window_cells)
    out.update(x0=float(x0), y0=float(y0), x1=float(x0 + nx * grid),
               y1=float(y0 + ny * grid), grid_m=float(grid),
               epsg=info.get("epsg"), thresh=float(thresh),
               fill_cells=int(fill_cells), hole_blank_m=float(hole_blank_m),
               n_points=int(n_pts))
    stats.update(points=int(n_pts), cells=int(nx * ny), total_s=time.time() - t0,
                 header=info)
    if verbose:
        print("%-28s pts %9d cells %7d empty %5.2f%% removed %5.2f%% "
              "filled %4.2f%% hole %5.2f%% %5.1fs [%s]"
              % (os.path.basename(path), stats["points"], nx * ny,
                 stats["empty_pct"], stats["removed_pct"], stats["filled_pct"],
                 stats["hole_pct"], stats["total_s"], method), flush=True)
    return out, stats


def minz_after(z0, removed):     # small helper kept for clarity
    z = z0.copy()
    z[removed] = np.nan
    return z


def sky_view_factor(dem, grid, azimuths=8, max_cells=128.0):
    """Approximate SVF (1 - mean sin(horizon angle)) over `azimuths` directions.

    The horizon in each direction is the max slope over octave distances
    1,2,4,...,128 cells.  This is a cheap approximation of the Zaksek et al.
    algorithm: it is used for review images only, not for detection.
    """
    z = np.asarray(dem, dtype=np.float32)
    steps = [1, 2, 4, 8, 16, 32, 64, 128]
    acc = np.zeros_like(z)
    n_used = 0
    for k in range(azimuths):
        a = 2.0 * np.pi * k / azimuths
        dx, dy = int(round(np.cos(a))), int(round(np.sin(a)))
        if dx == 0 and dy == 0:
            continue
        tan_h = np.zeros_like(z)
        for s in steps:
            sh = np.roll(np.roll(z, -dy * s, axis=0), -dx * s, axis=1)
            dist = s * grid * np.hypot(dx, dy)
            np.maximum(tan_h, (sh - z) / dist, out=tan_h)
        acc += tan_h / np.sqrt(1.0 + tan_h * tan_h)   # sin(atan(tan_h))
        n_used += 1
    return (1.0 - acc / max(1, n_used)).astype(np.float32)


def save(out, stats, npz_path, png_path=None, features=None, holes=None):
    np.savez_compressed(npz_path, **out)
    with open(npz_path.replace("_dtm.npz", "_dtm.json").replace(".npz", ".json"), "w") as f:
        json.dump(stats, f, indent=1, default=str)
    if png_path:
        from viz import quicklook
        quicklook(png_path, np.asarray(out["dem"]), out["x0"], out["y0"],
                  out["grid_m"], title=os.path.basename(png_path),
                  features=features or [], holes=holes or [],
                  hpng=png_path.replace(".png", "_dem.png"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("laz", nargs="+", help="one or more LAS/LAZ files, or a glob")
    ap.add_argument("--out", default=None, help="output dir (default: beside input)")
    ap.add_argument("--grid", type=float, default=3.0)
    ap.add_argument("--no-svf", action="store_true")
    ap.add_argument("--method", default="vstruct",
                    choices=["literal", "median", "min", "vstruct", "none"])
    ap.add_argument("--vstruct-extent", type=float, default=2.5)
    ap.add_argument("--vstruct-max-blob", type=int, default=4)
    ap.add_argument("--low-thresh", type=float, default=6.0,
                    help="reject cells this far below the local median (0=off)")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--truth", action="store_true",
                    help="overlay truth features from <scene>_truth.json")
    a = ap.parse_args(argv)
    paths = []
    for p in a.laz:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    for p in paths:
        outdir = a.out or os.path.dirname(os.path.abspath(p))
        os.makedirs(outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(p))[0]
        out, stats = build_dtm(p, grid=a.grid, svf=not a.no_svf, method=a.method,
                               vstruct_min_extent=a.vstruct_extent,
                               vstruct_max_blob_cells=a.vstruct_max_blob,
                               low_thresh=a.low_thresh)
        npz = os.path.join(outdir, base + "_dtm.npz")
        feats, holes = None, None
        tj = p.replace(".laz", "_truth.json")
        if a.truth and os.path.exists(tj):
            tr = json.load(open(tj))
            feats, holes = tr.get("features"), tr.get("holes")
        save(out, stats, npz, (os.path.join(outdir, base + "_dtm.png")
                               if a.png else None), feats, holes)
        print("   -> %s" % npz, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
