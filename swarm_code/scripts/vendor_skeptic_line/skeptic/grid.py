"""Grid container + NaN-aware sampling for the m6 skeptic harness.

Interface contract (see SPEC_interfaces.md, section A):

    np.savez_compressed(path,
        z=z,               # 2D float array, NaN = no data
        res=3.0,           # cell size in metres
        x0=..., y0=...,    # map coordinate of the outer corner of cell (0,0)
        epsg=5880,         # optional
        y_ascending=False, # optional; default False = north-up (GeoTIFF style)
        chm=chm,           # optional, same shape as z
    )

North-up convention (y_ascending=False): row index i increases southwards, so the
centre of cell (i, j) is

    x = x0 + (j + 0.5) * res
    y = y0 - (i + 0.5) * res

Everything downstream uses map coordinates in metres, so the EPSG only has to be
consistent within one transect (any projected metric CRS works; EBA L3A uses
EPSG:5880).

No network. No file is written outside the caller's chosen output directory.
"""
from __future__ import annotations

import math

import numpy as np


class Grid:
    """A regular north-up (or y-ascending) raster with an explicit origin."""

    __slots__ = ("z", "res", "x0", "y0", "y_ascending", "epsg", "name", "_yi")

    def __init__(self, z, res, x0, y0, y_ascending=False, epsg=None, name=""):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"z must be 2D, got shape {z.shape}")
        res = float(res)
        if not res > 0:
            raise ValueError("res must be positive")
        self.z = z
        self.res = res
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.y_ascending = bool(y_ascending)
        self.epsg = epsg
        self.name = name
        self._yi = 1.0 if self.y_ascending else -1.0

    # ---------------------------------------------------------------- basics
    @property
    def ny(self):
        return self.z.shape[0]

    @property
    def nx(self):
        return self.z.shape[1]

    @property
    def shape(self):
        return self.z.shape

    @property
    def x_max(self):
        return self.x0 + self.nx * self.res

    @property
    def y_min(self):
        return self.y0 if self.y_ascending else self.y0 - self.ny * self.res

    @property
    def y_max(self):
        return self.y0 + self.ny * self.res if self.y_ascending else self.y0

    def cell_center(self, i, j):
        return (self.x0 + (np.asarray(j) + 0.5) * self.res,
                self.y0 + self._yi * (np.asarray(i) + 0.5) * self.res)

    def frac_index(self, x, y):
        """Continuous (row, col) index of map point(s); (0,0) = centre of cell 0,0."""
        fi = (np.asarray(y, dtype=float) - self.y0) * self._yi / self.res - 0.5
        fj = (np.asarray(x, dtype=float) - self.x0) / self.res - 0.5
        return fi, fj

    def contains(self, x, y, margin_m=0.0):
        m = float(margin_m)
        return ((x >= self.x0 + m) & (x <= self.x_max - m) &
                (y >= self.y_min + m) & (y <= self.y_max - m))

    def border_distance(self, x, y):
        """Distance in metres from a map point to the nearest grid border."""
        return np.minimum(np.minimum(x - self.x0, self.x_max - x),
                          np.minimum(y - self.y_min, self.y_max - y))

    # -------------------------------------------------------------- sampling
    def sample(self, x, y):
        """NaN-aware bilinear sample. NaN where all 4 neighbours are NaN."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fi, fj = self.frac_index(x, y)
        i0 = np.floor(fi).astype(np.int64)
        j0 = np.floor(fj).astype(np.int64)
        ti = fi - i0
        tj = fj - j0
        ny, nx = self.shape
        out = np.full(np.broadcast(x, y).shape, np.nan, dtype=float)
        acc = np.zeros_like(out)
        wsum = np.zeros_like(out)
        for di, dj, w in ((0, 0, (1 - ti) * (1 - tj)),
                          (0, 1, (1 - ti) * tj),
                          (1, 0, ti * (1 - tj)),
                          (1, 1, ti * tj)):
            ii = i0 + di
            jj = j0 + dj
            inside = (ii >= 0) & (ii < ny) & (jj >= 0) & (jj < nx)
            ic = np.clip(ii, 0, ny - 1)
            jc = np.clip(jj, 0, nx - 1)
            v = self.z[ic, jc]
            finite = np.isfinite(v) & inside
            acc = acc + np.where(finite, w * np.where(finite, v, 0.0), 0.0)
            wsum = wsum + np.where(finite, w, 0.0)
        good = wsum > 1e-9
        out[good] = acc[good] / wsum[good]
        return out

    def window(self, cx, cy, half_w_m):
        """Slices of a square window centred on (cx, cy), clipped to the grid."""
        fi = float(np.asarray(self.frac_index(cx, cy)[0]))
        fj = float(np.asarray(self.frac_index(cx, cy)[1]))
        h = half_w_m / self.res
        i0 = max(0, int(math.floor(fi - h)))
        i1 = min(self.ny, int(math.ceil(fi + h)) + 1)
        j0 = max(0, int(math.floor(fj - h)))
        j1 = min(self.nx, int(math.ceil(fj + h)) + 1)
        if i0 >= i1 or j0 >= j1:
            return None
        return slice(i0, i1), slice(j0, j1)

    def subgrid(self, cx, cy, half_w_m, name=None):
        """Copy of a square window as a new Grid (clipped at the source border)."""
        sl = self.window(cx, cy, half_w_m)
        if sl is None:
            return None
        si, sj = sl
        xs, ys = self.cell_center(si.start, sj.start)
        return Grid(self.z[si, sj].copy(), self.res,
                    xs - 0.5 * self.res, ys - self._yi * 0.5 * self.res,
                    y_ascending=self.y_ascending, epsg=self.epsg,
                    name=name or self.name)

    # ------------------------------------------------------------- nodata etc
    @property
    def nodata(self):
        return ~np.isfinite(self.z)

    def origin_dict(self):
        return {"res": self.res, "x0": self.x0, "y0": self.y0,
                "y_ascending": self.y_ascending, "epsg": self.epsg,
                "shape": list(self.shape), "name": self.name}

    # ---------------------------------------------------------------- output
    def save_npz(self, path, compress=True):
        d = dict(z=self.z, res=self.res, x0=self.x0, y0=self.y0,
                 y_ascending=self.y_ascending, name=self.name)
        if self.epsg is not None:
            d["epsg"] = self.epsg
        (np.savez_compressed if compress else np.savez)(path, **d)

    def __repr__(self):
        return (f"Grid({self.name!r}, shape={self.shape}, res={self.res} m, "
                f"x0={self.x0:.1f}, y0={self.y0:.1f}, "
                f"y_asc={'up' if self.y_ascending else 'down'}, "
                f"epsg={self.epsg}, nodata={float(self.nodata.mean()):.1%})")


def _as_float_scalar(v, default=None):
    try:
        return float(np.asarray(v).ravel()[0])
    except Exception:
        return default


def load_grid(path, z_key=None, name=None, res=None, x0=None, y0=None) -> Grid:
    """Load a Grid from .npz (preferred) or .npy plus explicit metadata.

    Tolerant about key spelling so a processor written by another member can still
    be consumed.  z keys tried: z, dtm, elev, height, dem.
    Origin keys tried: (x0, y0), origin=[x0,y0,...], extent=(xmin,ymin,xmax,ymax).
    """
    path = str(path)
    if path.endswith(".npy"):
        if res is None:
            raise ValueError("loading .npy needs res= (no metadata in file)")
        return Grid(np.load(path, allow_pickle=False), res,
                    x0 or 0.0, y0 or 0.0, epsg=None,
                    name=name or path.split("/")[-1])
    if not path.endswith(".npz"):
        raise ValueError(f"unsupported grid file {path}; convert to .npz with z/res/x0/y0")
    raw_meta = {}
    with np.load(path, allow_pickle=False) as f:
        keys = set(f.files)
        zk = z_key
        if zk is None:
            for cand in ("z", "dtm", "elev", "height", "dem"):
                if cand in keys:
                    zk = cand
                    break
        if zk is None:
            raise KeyError(f"{path}: no elevation array among z/dtm/elev/height/dem "
                           f"(keys={sorted(keys)})")
        z = np.asarray(f[zk], dtype=np.float64)
        raw_meta = {k: np.asarray(f[k]).ravel()[0]
                    for k in keys if k != zk and np.asarray(f[k]).size == 1}
        meta = {}
        for k, v in raw_meta.items():
            try:
                meta[k] = float(v)
            except (TypeError, ValueError):
                pass
    if "x0" in meta and "y0" in meta:
        gx0, gy0 = meta["x0"], meta["y0"]
        y_asc = bool(meta.get("y_ascending", 0.0))
        if "row_order" in raw_meta:
            y_asc = str(raw_meta["row_order"]).lower().startswith("south")
    elif "origin" in meta:
        raise ValueError(f"{path}: origin stored as >=2 values; pass x0=,y0= explicitly")
    elif "extent" in meta:
        raise ValueError(f"{path}: extent-only origin; pass x0=,y0= explicitly")
    else:
        if x0 is None or y0 is None:
            raise KeyError(f"{path}: no origin in file (need x0 and y0, or pass them); "
                           f"have {sorted(meta)}")
        gx0, gy0, y_asc = float(x0), float(y0), False
    r = meta.get("res", meta.get("cellsize", meta.get("cell", res)))
    if r is None:
        raise KeyError(f"{path}: no cell size (need res in file or res= argument)")
    ep = meta.get("epsg")
    return Grid(z, r, gx0, gy0, y_ascending=y_asc, epsg=(int(ep) if ep else None),
                name=name or str(meta.get("name", path.split("/")[-1])))


def load_chm(path, like: Grid, key=None) -> Grid:
    """Load a canopy height raster and check it lines up with `like`."""
    with np.load(str(path), allow_pickle=False) as f:
        k = key
        if k is None:
            for cand in ("chm", "canopy", "chm_m"):
                if cand in f.files:
                    k = cand
                    break
        if k is None:
            raise KeyError(f"{path}: no canopy array among chm/canopy/chm_m")
        z = np.asarray(f[k], dtype=np.float64)
    if z.shape != like.shape:
        raise ValueError(f"chm shape {z.shape} != dtm shape {like.shape}")
    return Grid(z, like.res, like.x0, like.y0, y_ascending=like.y_ascending,
                epsg=like.epsg, name=like.name + "/chm")
