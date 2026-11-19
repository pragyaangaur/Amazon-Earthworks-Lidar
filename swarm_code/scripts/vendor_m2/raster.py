"""Memory-bounded streaming rasterisation + ground picking (m2 processor core).

Design constraints that shaped this file
----------------------------------------
* An EBA transect LAZ can hold 1e8-1e9 returns.  Nothing here ever holds the
  point cloud: ``accumulate`` walks the file in blocks and keeps only per-cell
  aggregates (lowest return, highest return, count, return-number stats).
* Cells are 3 m because the nominal EBA density is ~2 returns/m^2 (9 m^2 per
  cell -> ~18 returns, enough for a lowest-return estimate).
* LAS class 2 is contaminated by canopy in EBA data, so classes are ignored
  entirely: ground evidence is the geometric lowest return per cell.
* Ground picking follows the proven recipe: smooth the lowest-return raster,
  then a cascade of grey-scale openings at 40 m, 20 m, 10 m; any cell standing
  more than 0.7 m above the opened surface is pulled down to that surface.
  Opening is anti-extensive, so narrow ditches (3-15 m) survive while canopy
  clumps, buildings and noise spikes are removed.
* Holes are data gaps (no return in the cell).  Only holes of <= 5 connected
  cells are interpolated; larger gaps stay NaN and are reported, because a
  filled 40 m gap invents terrain exactly where a detector would look.
"""
from __future__ import annotations

import numpy as np
import lasio

try:                        # scipy is present in this VM, but stay defensive
    from scipy import ndimage as ndi
except Exception:           # pragma: no cover
    ndi = None


# ------------------------------------------------------------------ grid ----

class Grid:
    """North-up cell grid in projected metres (EPSG carried separately)."""

    def __init__(self, x0, y0, nx, ny, cell):
        self.x0 = float(x0); self.y0 = float(y0)
        self.nx = int(nx); self.ny = int(ny); self.cell = float(cell)

    @classmethod
    def from_bounds(cls, minx, maxx, miny, maxy, cell, pad=1):
        x0 = np.floor(minx / cell) * cell
        y0 = np.floor(miny / cell) * cell
        nx = int(np.ceil((maxx - x0) / cell)) + pad
        ny = int(np.ceil((maxy - y0) / cell)) + pad
        return cls(x0, y0, nx, ny, cell)

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def extent(self):
        return (self.x0, self.x0 + self.nx * self.cell,
                self.y0, self.y0 + self.ny * self.cell)

    def centers_x(self):
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    def centers_y(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    def index(self, x, y):
        ix = np.floor((x - self.x0) / self.cell).astype(np.int64)
        iy = np.floor((y - self.y0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy, ix, ok

    def flat(self, x, y):
        iy, ix, ok = self.index(x, y)
        return iy * self.nx + ix, ok

    def to_dict(self):
        return dict(x0=self.x0, y0=self.y0, nx=self.nx, ny=self.ny,
                    cell=self.cell, extent=list(self.extent))

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["y0"], d["nx"], d["ny"], d["cell"])


class Accum:
    """Per-cell aggregates for one grid: lowest return, highest return, counts."""

    def __init__(self, grid):
        self.grid = grid
        n = grid.nx * grid.ny
        self.zmin = np.full(n, np.nan, np.float32)
        self.zlow2 = np.full(n, np.nan, np.float32)   # 2nd-lowest return
        self.zq10 = np.full(n, np.nan, np.float32)    # ~10th percentile return
        self.zmax = np.full(n, np.nan, np.float32)
        self.count = np.zeros(n, np.uint32)
        self.n_first = np.zeros(n, np.uint32)   # return number 1
        self.n_last = np.zeros(n, np.uint32)    # return number == n returns
        self.n_points_seen = 0
        self.n_outside = 0

    # -- merging ---------------------------------------------------------
    def add_block(self, x, y, z, ret=None, nret=None):
        g = self.grid
        flat, ok = g.flat(x, y)
        self.n_outside += int((~ok).sum())
        if not ok.any():
            return
        flat = flat[ok]; zz = z[ok]
        # sort by (cell, z): the first two entries of a group are its two
        # lowest returns, which is what makes the DTM robust to a single stray
        # return per cell (EBA files contain a few returns at -300 m).
        order = np.lexsort((zz, flat))
        f = flat[order]; zs = zz[order]
        start = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
        uid = f[start]
        cnt = np.diff(np.r_[start, f.size]).astype(np.uint32)
        mn = zs[start]
        mx = np.maximum.reduceat(zs, start)
        second = zs[np.minimum(start + 1, zs.size - 1)]
        # q10: the k-th smallest return of the cell, k = floor((n-1)/10).  The
        # group is already sorted by z, so this is exact within the block; when a
        # cell spans blocks the minimum of the per-block q10 is kept, which is
        # slightly low-biased (i.e. conservative) and never hides an outlier.
        k10 = start + ((cnt - 1) // 10).astype(np.int64)
        q10 = zs[np.minimum(k10, zs.size - 1)]

        cur = self.zmin[uid]
        better = np.isnan(cur) | (mn < cur)
        self.zmin[uid[better]] = mn[better]
        cur = self.zq10[uid]
        bq = np.isnan(cur) | (q10 < cur)
        self.zq10[uid[bq]] = q10[bq]
        has2 = cnt >= 2
        cur = self.zlow2[uid[has2]]
        b2 = np.isnan(cur) | (second[has2] < cur)
        u2 = uid[has2]
        self.zlow2[u2[b2]] = second[has2][b2]
        cur = self.zmax[uid]
        self.zmax[uid[better]] = mx[better]
        self.count[uid] += cnt
        self.n_points_seen += int(ok.sum())
        if ret is not None:
            r = np.asarray(ret)[ok][order]
            first = (r == 1)
            self.n_first[uid] += np.bincount(np.searchsorted(uid, f[first]),
                                             minlength=uid.size).astype(np.uint32)
            if nret is not None:
                nr = np.asarray(nret)[ok][order]
                last = (r == nr)
                self.n_last[uid] += np.bincount(np.searchsorted(uid, f[last]),
                                                minlength=uid.size).astype(np.uint32)

    # -- views -----------------------------------------------------------
    def zmin_grid(self):
        return self.zmin.reshape(self.grid.shape)

    def zlow2_grid(self):
        return self.zlow2.reshape(self.grid.shape)

    def zq10_grid(self):
        return self.zq10.reshape(self.grid.shape)

    def zmax_grid(self):
        return self.zmax.reshape(self.grid.shape)

    def count_grid(self):
        return self.count.reshape(self.grid.shape)

    def hole_mask(self):
        return (self.count.reshape(self.grid.shape) == 0)

    def crop(self, row0, row1, col0, col1):
        """Return a cropped Accum (same aggregates, smaller grid)."""
        g = self.grid
        new = Grid(g.x0 + col0 * g.cell, g.y0 + row0 * g.cell,
                   col1 - col0, row1 - row0, g.cell)
        a = Accum(new)
        sl = (slice(row0, row1), slice(col0, col1))
        for name in ("zmin", "zlow2", "zq10", "zmax", "count", "n_first", "n_last"):
            setattr(a, name, getattr(self, name).reshape(g.shape)[sl].copy().ravel())
        a.n_points_seen = self.n_points_seen
        a.n_outside = self.n_outside
        return a

    def data_bbox(self, margin_cells=0):
        """Tight (row0,row1,col0,col1) around cells that hold returns."""
        c = self.count.reshape(self.grid.shape) > 0
        if not c.any():
            return (0, self.grid.ny, 0, self.grid.nx)
        rows = np.flatnonzero(c.any(1)); cols = np.flatnonzero(c.any(0))
        m = int(margin_cells)
        return (max(0, rows[0] - m), min(self.grid.ny, rows[-1] + 1 + m),
                max(0, cols[0] - m), min(self.grid.nx, cols[-1] + 1 + m))

    def summary(self):
        n = self.count.size
        cf = self.count > 0
        return dict(n_cells_total=int(n), n_cells_filled=int(cf.sum()),
                    empty_cell_pct=float(100.0 * (1 - cf.mean())),
                    n_points_in_grid=int(self.n_points_seen),
                    n_points_outside_grid=int(self.n_outside),
                    mean_returns_per_cell=float(self.count[cf].mean()) if cf.any() else 0.0,
                    median_returns_per_cell=float(np.median(self.count[cf])) if cf.any() else 0.0,
                    cells_using_second_lowest=int(((self.count >= 3)).sum()))


def accumulate(path, grid, block=2_000_000, max_points=None, reader=None,
               progress=None):
    """One streaming pass over a LAS/LAZ file filling an ``Accum``.

    ``reader`` lets a LAZ decoder be injected with the same iterator interface
    as ``lasio.iter_xyz`` (used by lazbite.decode_iter).
    """
    reader = reader or lasio.iter_xyz
    acc = Accum(grid)
    for x, y, z, blk in reader(path, block=block):
        acc.add_block(x, y, z,
                      ret=_return_number(blk), nret=_n_returns(blk))
        if progress is not None:
            progress(int(x.size))
        if max_points is not None and acc.n_points_seen >= max_points:
            break
    return acc


def _return_number(blk):
    if blk.dtype.names and "flags" in blk.dtype.names:
        return (blk["flags"] & 0b111).astype(np.uint8)
    return None


def _n_returns(blk):
    if blk.dtype.names and "flags" in blk.dtype.names:
        return ((blk["flags"] >> 3) & 0b111).astype(np.uint8)
    return None


# ------------------------------------------------------------- filtering ----

def _fill_nearest(a):
    """Nearest-neighbour fill of NaNs (used only to build the opening surface)."""
    if not np.isnan(a).any():
        return a
    if ndi is None:
        return np.where(np.isnan(a), np.nanmax(a), a)
    idx = ndi.distance_transform_edt(np.isnan(a), return_distances=False,
                                    return_indices=True)
    return a[tuple(idx)]


def grey_open(surf, size_cells):
    """Grey-scale opening with a square window (min then max filter)."""
    size_cells = int(max(1, size_cells))
    if size_cells % 2 == 0:
        size_cells += 1
    filled = _fill_nearest(surf.astype(np.float64))
    er = ndi.minimum_filter(filled, size=size_cells, mode="nearest")
    op = ndi.maximum_filter(er, size=size_cells, mode="nearest")
    return op


def ground_pick(zmin, count, cell=3.0, windows_m=(40.0, 20.0, 10.0),
                thresh_m=0.7, smooth_cells=0, light_smooth_sigma=None,
                zlow2=None, stray_gap_m=4.0, outlier_below_m=30.0,
                outlier_window_m=300.0, zq10=None, source="hybrid"):
    """DTM from the lowest-return raster.  Returns a dict of rasters/masks.

    Returns
    -------
    dtm            float32 (NaN in unfilled holes)
    removed        bool, cells pulled down by the opening cascade
    data_hole      bool, cells with no return at all
    big_hole       bool, holes larger than 5 cells left unfilled
    filled         bool, hole cells that were interpolated
    """
    # Surface choice.  Three candidates, and the trade-off is real (m5 skeptic
    # findings f3/f8):
    #   * min  : keeps the deepest return of a narrow ditch, so it preserves
    #            feature depth, but one return below ground poisons the cell;
    #   * q10  : robust to that, but in a cell that straddles a 4 m ditch the
    #            deepest returns are a minority, so q10 loses ~20% depth;
    #   * hybrid (default): use min, but replace it by q10 where the min is a
    #            lone excursion (> stray_gap_m) below a cluster that is itself at
    #            regional ground level.  Keeps ditch depth, rejects lone lows, and
    #            does not promote canopy cells (their q10 sits far above ground).
    if source not in ("q10", "min", "hybrid"):
        raise ValueError("unknown ground surface source %r" % (source,))
    z = np.array(zmin, dtype=np.float64, copy=True)
    zq = None if zq10 is None else np.asarray(zq10, np.float64)
    # diagnostics m5 asked for: cells whose minimum sits far below their own
    # 10th-percentile return (outlier-fragile cells), and how many of those were
    # rewritten by the hybrid rule below.
    n_stray = 0
    if zq is not None:
        n_stray = int(np.sum(np.isfinite(zq) & np.isfinite(z) & ((zq - z) > 0.3)))
    if source != "min" and zq is not None:
        if source == "q10":
            use = np.isfinite(zq)
            z[use] = zq[use]
        else:
            ref0 = np.full(z.shape, np.nan)
            if ndi is not None and np.isfinite(z).any():
                step = max(1, int(round(60.0 / cell)))
                sub = _fill_nearest(z)[::step, ::step]
                k0 = int(round(300.0 / (cell * step))) | 1
                ref0s = ndi.median_filter(sub, size=k0, mode="nearest")
                ref0 = np.repeat(np.repeat(ref0s, step, axis=0), step, axis=1)[:z.shape[0], :z.shape[1]]
                if ref0.shape != z.shape:
                    ref0 = np.pad(ref0, [(0, z.shape[0] - ref0.shape[0]),
                                         (0, z.shape[1] - ref0.shape[1])], mode="edge")
            lone_low = (np.isfinite(zq) & np.isfinite(z) &
                        ((zq - z) > stray_gap_m) & np.isfinite(ref0) &
                        (np.abs(zq - ref0) < 5.0))
            z[lone_low] = zq[lone_low]
            n_lone_low = int(lone_low.sum())
            globals().setdefault("_LAST_LONE_LOW", 0)
    data_hole = (count == 0) | np.isnan(z)
    z[data_hole] = np.nan

    # Outliers far below the regional surface cannot be terrain: EBA transect
    # 0072 carries isolated returns down to -337 m in ~180 m terrain, and a
    # lowest-return raster turns each into a 300 m pit.  The reference is a
    # median over a ~300 m window (computed on a 60 m subsample for speed), so
    # a 4 m ditch or a 150 m closed-canopy patch never triggers it; a masked
    # cell becomes a hole and is filled only if the hole is <=5 cells.
    outlier = np.zeros(z.shape, bool)
    if outlier_below_m and ndi is not None and np.isfinite(z).any():
        step = max(1, int(round(60.0 / cell)))
        sub = _fill_nearest(z)[::step, ::step]
        k = int(round(outlier_window_m / (cell * step))) | 1
        ref_sub = ndi.median_filter(sub, size=k, mode="nearest")
        ref = np.repeat(np.repeat(ref_sub, step, axis=0), step, axis=1)[:z.shape[0], :z.shape[1]]
        if ref.shape != z.shape:
            pad = [(0, z.shape[i] - ref.shape[i]) for i in range(2)]
            ref = np.pad(ref, pad, mode="edge")
        outlier = np.isfinite(z) & (z < ref - outlier_below_m)
        z[outlier] = np.nan
    if smooth_cells and ndi is not None:
        # Optional median pre-clean.  Default OFF: the 300 m outlier guard
        # already removes strays, while a 3x3 median cuts the peak of an 8 m
        # mound by ~0.5 m (measured), which is a target feature class here.
        z = _nanaware_median(z, smooth_cells)
    removed = np.zeros(z.shape, bool)
    surf = z.copy()
    # Surface before the opening clamp: positive features (mounds, embanked
    # roads) taller than thresh_m look exactly like canopy to the clamp, so mound
    # work must use this layer together with the canopy mask.
    pre = surf.copy()
    for w in windows_m:
        k = int(round(w / cell))
        k = k if k % 2 == 1 else k + 1
        op = grey_open(surf, k)
        # A grey opening fills any depression NARROWER than the window, and the
        # fill spreads about (window - width) into it.  For an enclosure of
        # about the window diameter (a 40 m ring ditch with a 40 m window) the
        # filled surface is the ditch bottom across the whole interior, so a
        # plain "surf - op > 0.7 m" test rewrites the interior and erases the
        # ditch-versus-interior contrast (measured: interior 1.16 m too low on a
        # clean synthetic control, no outliers involved).
        # Guard: a cell is only a spike if it also stands above the *median* of
        # its neighbourhood, which a level interior does not and a canopy/building
        # cell does.  This keeps the opening's job (remove positive protrusions)
        # without letting it excavate enclosures.
        med = _nanaware_median(surf, max(1, k // 2))
        spike = ((surf - op) > thresh_m) & ((surf - med) > thresh_m)
        # never raise: only clamp cells that stand above the surface
        surf = np.where(spike & ~np.isnan(surf), op, surf)
        removed |= spike
    if light_smooth_sigma:
        surf = _nanaware_gauss(surf, light_smooth_sigma / cell)

    filled, big_hole, fill_mask = fill_small_holes(surf, data_hole | removed | outlier,
                                                   max_cells=5)
    pre_filled, _, _ = fill_small_holes(pre, data_hole | outlier, max_cells=5)
    return dict(dtm=filled.astype(np.float32), pre_clamp=pre_filled.astype(np.float32),
                removed=removed, source=source,
                n_lone_low=int(n_lone_low) if "n_lone_low" in dir() else 0,
                data_hole=data_hole, big_hole=big_hole, outlier=outlier,
                n_stray_replaced=n_stray,
                filled=fill_mask,
                unfilled=np.isnan(filled))


def _nanaware_median(a, k):
    if ndi is None or k <= 0:
        return a
    size = 2 * k + 1
    med = ndi.median_filter(_fill_nearest(a), size=size, mode="nearest")
    out = np.where(np.isnan(a), np.nan, med)
    return out


def _nanaware_gauss(a, sigma):
    if ndi is None or sigma <= 0:
        return a
    num = ndi.gaussian_filter(np.where(np.isnan(a), 0.0, a), sigma, mode="nearest")
    den = ndi.gaussian_filter((~np.isnan(a)).astype(np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def fill_small_holes(surf, hole, max_cells=5):
    """Fill only connected holes with <= max_cells cells; report the rest."""
    nan = np.isnan(surf)
    empty = np.zeros(surf.shape, bool)
    if ndi is None or not nan.any():
        return surf, empty, empty
    lab, n = ndi.label(nan)
    if n == 0:
        return surf, empty, empty
    sizes = np.bincount(lab.ravel())
    small = sizes <= max_cells
    small[0] = False
    fill_mask = small[lab]
    big_hole = nan & ~fill_mask
    out = surf.copy()
    if fill_mask.any():
        idx = ndi.distance_transform_edt(nan, return_distances=False,
                                        return_indices=True)
        neigh = surf[tuple(idx)]
        out[fill_mask] = neigh[fill_mask]
    return out, big_hole, fill_mask
