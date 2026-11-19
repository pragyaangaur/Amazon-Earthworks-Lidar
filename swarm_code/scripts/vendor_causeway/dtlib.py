"""dtlib.py -- dependency-light DTM utilities for the EBA earthwork screen.

Only numpy / scipy / Pillow / numba are used (no laspy, rasterio, GDAL, pyproj
in this VM).  Grid convention (documented once, used everywhere):

  * z is a 2-D float32 array, shape (nrows, ncols), row 0 is the NORTH edge.
  * cell (i, j) CENTRE has world coords
        x = x0 + (j + 0.5) * dx          (easting, m, in `epsg`)
        y = y0 - (i + 0.5) * dy          (northing, m, in `epsg`)
  * meta dict keys: epsg, x0, y0, dx, dy, nodata, source, note
  * invalid/empty cells are NaN in z.

Author: swarm smwduocxj member m3 (DETECTOR role).
"""
import heapq
import json
import os

import numpy as np
from scipy import ndimage as ndi

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                               # noqa: D103
        def deco(f):
            return f
        return deco if (a and callable(a[0])) is False else a[0]

DEFAULT_META = dict(epsg=0, x0=0.0, y0=0.0, dx=3.0, dy=3.0, nodata=-9999.0,
                    source="unknown", note="")


# --------------------------------------------------------------------------
# grid bookkeeping
# --------------------------------------------------------------------------
def make_meta(dx=3.0, dy=None, x0=0.0, y0=0.0, epsg=0, **kw):
    m = dict(DEFAULT_META)
    m.update(dx=float(dx), dy=float(dx if dy is None else dy),
             x0=float(x0), y0=float(y0), epsg=int(epsg))
    m.update(kw)
    return m


def cell_centres(shape, meta):
    """Return (X, Y) world-coordinate meshgrids of cell centres."""
    n, m = shape
    dx, dy = meta["dx"], meta["dy"]
    xs = meta["x0"] + (np.arange(m) + 0.5) * dx
    ys = meta["y0"] - (np.arange(n) + 0.5) * dy
    return np.meshgrid(xs, ys)


def field_meta(shape, meta, **kw):
    f = dict(meta)
    f.update(shape=[int(shape[0]), int(shape[1])])
    f.update(kw)
    return f


def save_grid(path, z, meta):
    """Save float32 grid + JSON sidecar.  `path` ends in .npy (sidecar .npy.json)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, np.asarray(z, dtype=np.float32))
    with open(path + ".json", "w") as fh:
        json.dump(field_meta(np.shape(z), meta), fh, indent=1, sort_keys=True)
    return path + ".json"


def load_grid(path):
    z = np.load(path).astype(np.float32)
    with open(path + ".json") as fh:
        meta = json.load(fh)
    return z, meta


def save_geotiff(path, z, meta, nodata=None):
    """Best-effort float32 GeoTIFF (Pillow, tiled deflate).

    Writes ModelPixelScale (33550) and ModelTiepoint (33922) so the file has a
    real geotransform; EPSG is recorded in GeoKeyDirectory (34735) when known.
    The npy+json pair is the authoritative interchange format -- GeoTIFF is for
    human/QGIS review only.
    """
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    nodata = meta.get("nodata", DEFAULT_META["nodata"]) if nodata is None else nodata
    a = np.array(z, dtype=np.float32)
    a[~np.isfinite(a)] = nodata
    im = Image.fromarray(a)
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (float(meta["dx"]), float(meta["dy"]), 0.0)
    # tiepoint: raster (0,0,0) -> world (x0, y0, 0)
    ifd[33922] = (0.0, 0.0, 0.0, float(meta["x0"]), float(meta["y0"]), 0.0)
    keys = [1, 1, 0, 3,             # version, rev, minor, nkeys
            1024, 0, 1, 2,          # GTModelType = geographic (2)
            1025, 0, 1, 1]          # GTRasterType = PixelIsArea (1)
    epsg = int(meta.get("epsg") or 0)
    if epsg > 32767:                # projected
        keys[7] = 1
        keys += [3072, 0, 1, epsg]
    elif epsg > 0:
        keys += [2048, 0, 1, epsg]
    ifd[34735] = tuple(keys)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path, compression="tiff_deflate", tiled=True, tile=(256, 256),
            tiffinfo=ifd)
    return path


# --------------------------------------------------------------------------
# masks, smoothing
# --------------------------------------------------------------------------
def valid_mask(z, meta=None):
    z = np.asarray(z, dtype=np.float32)
    nod = DEFAULT_META["nodata"] if meta is None else meta.get("nodata", DEFAULT_META["nodata"])
    return np.isfinite(z) & (z > -1e4) & (np.abs(z - nod) > 1e-6)


def fill_holes_small(mask, max_cells=5, structure=None):
    """Fill invalid holes whose area is <= max_cells (in 8-connectivity)."""
    inv = ~mask
    lab, n = ndi.label(inv, structure=structure or np.ones((3, 3), bool))
    if n == 0:
        return mask.copy()
    sizes = ndi.sum(inv, lab, np.arange(1, n + 1))
    small = np.concatenate(([False], sizes <= max_cells))
    return mask | small[lab]


def mask_gauss(z, w, sigma, fill=0.0):
    """Normalised-convolution Gaussian smoothing that respects the valid mask."""
    wf = w.astype(np.float32)
    a = np.where(w, np.asarray(z, dtype=np.float32) - fill, 0.0)
    num = ndi.gaussian_filter(a, sigma, mode="nearest")
    den = ndi.gaussian_filter(wf, sigma, mode="nearest")
    out = np.where(den > 1e-4, num / np.maximum(den, 1e-6), np.nan)
    return out + fill


def local_relief(z, w, sigma):
    """z - mask_gauss(z, sigma); local relief relative to a broad surface."""
    base = mask_gauss(z, w, sigma)
    return np.asarray(z, dtype=np.float32) - base


def robust_sigma(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 16:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


# --------------------------------------------------------------------------
# hydrology: priority-flood fill, D8 directions, flow accumulation
# --------------------------------------------------------------------------
def fill_depressions(z, w, eps=1e-4):
    """Barnes et al. priority-flood depression fill on the valid mask."""
    z = np.asarray(z, np.float64)
    n, m = z.shape
    filled = z.copy()
    closed = ~w
    heap = []
    # seed: valid cells on the array border
    for i in range(n):
        for j in (0, m - 1):
            if w[i, j]:
                heapq.heappush(heap, (z[i, j], i, j)); closed[i, j] = True
    for j in range(m):
        for i in (0, n - 1):
            if w[i, j] and not closed[i, j]:
                heapq.heappush(heap, (z[i, j], i, j)); closed[i, j] = True
    # seed: valid cells adjacent to a hole (data edge acts as an outlet)
    hole = ~w
    if hole.any():
        hole_nb = ndi.binary_dilation(hole, np.ones((3, 3), bool)) & w & ~closed
        ii, jj = np.nonzero(hole_nb)
        for i, j in zip(ii.tolist(), jj.tolist()):
            heapq.heappush(heap, (z[i, j], i, j)); closed[i, j] = True
    nbrs = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    while heap:
        hcur, i, j = heapq.heappop(heap)
        for di, dj in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < m and not closed[ni, nj] and w[ni, nj]:
                v = max(z[ni, nj], hcur + eps)
                filled[ni, nj] = v
                closed[ni, nj] = True
                heapq.heappush(heap, (v, ni, nj))
    filled[~w] = np.nan
    return filled


D8_OFF = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def d8_downstream(filled, w, dx, dy):
    """Index of the steepest-downslope neighbour (-1 if sink/invalid), flattened."""
    n, m = filled.shape
    f = np.where(w, filled, np.inf)
    best = np.zeros((n, m), np.float32)
    kbest = np.full((n, m), -1, np.int8)
    for k, (di, dj) in enumerate(D8_OFF):
        dist = float(np.hypot(di * dy, dj * dx))
        sh = np.full((n, m), np.inf, np.float32)
        # sh[i, j] must hold f[i+di, j+dj]
        ti = slice(max(0, -di), n - max(0, di))
        tj = slice(max(0, -dj), m - max(0, dj))
        si = slice(max(0, di), n - max(0, -di))
        sj = slice(max(0, dj), m - max(0, -dj))
        sh[ti, tj] = f[si, sj]
        ok = np.isfinite(sh) & np.isfinite(f)
        slope = np.where(ok, (f - sh) / dist, -np.inf)
        upd = w & (slope > best + 1e-9)
        best[upd] = slope[upd]
        kbest[upd] = k
    down = np.full(n * m, -1, np.int64)
    ii, jj = np.nonzero(w & (kbest >= 0))
    for k in range(8):
        sel = kbest[ii, jj] == k
        if not sel.any():
            continue
        di, dj = D8_OFF[k]
        i2 = ii[sel] + di
        j2 = jj[sel] + dj
        ok2 = (i2 >= 0) & (i2 < n) & (j2 >= 0) & (j2 < m)
        down[ii[sel][ok2] * m + jj[sel][ok2]] = i2[ok2] * m + j2[ok2]
    return down


@njit(cache=True)
def _accumulate(order, down, acc):
    for t in range(order.size):
        i = order[t]
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
    return acc


def flow_accumulation(filled, w, dx, dy, opened_holes=False):
    """D8 flow accumulation in cells.  Filled surface => no cycles."""
    n, m = filled.shape
    down = d8_downstream(filled, w, dx, dy)
    zf = np.where(w, filled, np.inf).ravel()
    idx = np.nonzero(w.ravel())[0]
    order = idx[np.argsort(-zf[idx], kind="stable")]
    acc = np.ones(n * m, np.float64)
    acc = _accumulate(order.astype(np.int64), down, acc)
    acc = acc.reshape(n, m)
    acc[~w] = np.nan
    return acc, down


# --------------------------------------------------------------------------
# curvatures / ridge measure
# --------------------------------------------------------------------------
def hessian_eigs(zs, w, sigma, dx, dy):
    """Eigenvalues of the (mask-smoothed) Hessian at scale sigma (cells)."""
    Hxx = ndi.gaussian_filter(zs, sigma, order=(0, 2), mode="nearest") / dx ** 2
    Hyy = ndi.gaussian_filter(zs, sigma, order=(2, 0), mode="nearest") / dy ** 2
    Hxy = ndi.gaussian_filter(zs, sigma, order=(1, 1), mode="nearest") / (dx * dy)
    tr = Hxx + Hyy
    det = Hxx * Hyy - Hxy ** 2
    disc = np.maximum(tr ** 2 - 4 * det, 0.0)
    sq = np.sqrt(disc)
    l1 = 0.5 * (tr + sq)
    l2 = 0.5 * (tr - sq)
    lo = np.minimum(l1, l2)
    hi = np.maximum(l1, l2)
    lo[~w] = np.nan
    hi[~w] = np.nan
    return lo, hi


def hessian_dir(zs, w, sigma, dx, dy):
    """Angle (radians, world x-y) of the *along-ridge* eigenvector of the Hessian
    (eigenvector of the eigenvalue of largest magnitude, i.e. the direction of
    least curvature).  NaN where invalid."""
    Hxx = ndi.gaussian_filter(zs, sigma, order=(0, 2), mode="nearest") / dx ** 2
    Hyy = ndi.gaussian_filter(zs, sigma, order=(2, 0), mode="nearest") / dy ** 2
    Hxy = ndi.gaussian_filter(zs, sigma, order=(1, 1), mode="nearest") / (dx * dy)
    tr = Hxx + Hyy
    det = Hxx * Hyy - Hxy ** 2
    disc = np.maximum(tr ** 2 - 4 * det, 0.0)
    sq = np.sqrt(disc)
    lo = 0.5 * (tr - sq)
    # eigenvector for lo (the across-ridge, most-curved one); ridge runs
    # perpendicular to it.  Use (vx, vy) = (lo - Hyy, Hxy) reflected in dy.
    vx = lo - Hyy
    vy = Hxy
    ang_across = np.arctan2(-vy, vx)          # world y decreases with row
    ang = ang_across + np.pi / 2
    ang = np.where(np.isfinite(ang), ang, np.nan)
    ang[~w] = np.nan
    return ang


def ridge_measure(z, w, sigma=2.0, dx=3.0, dy=3.0, fill=0.0):
    """Positive on ridge crests: -lambda_min with |lambda_max| small.

    Returns (measure, planarity) where planarity in [0,1] is 1 - |hi|/|lo|.
    """
    zs = mask_gauss(z, w, sigma, fill=fill)
    lo, hi = hessian_eigs(zs, w, sigma, dx, dy)
    meas = -lo
    denom = np.maximum(np.abs(lo), 1e-9)
    plan = 1.0 - np.abs(hi) / denom
    plan = np.clip(plan, 0.0, 1.0)
    plan[~w] = np.nan
    return meas, plan


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def bilinear(z, xs, ys, meta):
    """Sample z at world coords (arrays), NaN outside."""
    z = np.asarray(z, np.float32)
    n, m = z.shape
    fj = (np.asarray(xs, np.float64) - meta["x0"]) / meta["dx"] - 0.5
    fi = (meta["y0"] - np.asarray(ys, np.float64)) / meta["dy"] - 0.5
    i0 = np.floor(fi)
    j0 = np.floor(fj)
    ti = fi - i0
    tj = fj - j0
    i0 = i0.astype(np.int64)
    j0 = j0.astype(np.int64)
    out = np.full(np.broadcast(fi, fj).shape, np.nan, np.float64)
    ok = (i0 >= 0) & (i0 < n - 1) & (j0 >= 0) & (j0 < m - 1)
    if ok.any():
        z00 = z[i0[ok], j0[ok]]
        z01 = z[i0[ok], j0[ok] + 1]
        z10 = z[i0[ok] + 1, j0[ok]]
        z11 = z[i0[ok] + 1, j0[ok] + 1]
        a = ti[ok]; b = tj[ok]
        out[ok] = (z00 * (1 - a) * (1 - b) + z01 * (1 - a) * b +
                   z10 * a * (1 - b) + z11 * a * b)
    return out


def world_to_index(xs, ys, meta):
    fi = np.floor((meta["y0"] - np.asarray(ys, np.float64)) / meta["dy"]).astype(np.int64)
    fj = np.floor((np.asarray(xs, np.float64) - meta["x0"]) / meta["dx"]).astype(np.int64)
    return fi, fj


def index_to_world(i, j, meta):
    return (meta["x0"] + (np.asarray(j) + 0.5) * meta["dx"],
            meta["y0"] - (np.asarray(i) + 0.5) * meta["dy"])
