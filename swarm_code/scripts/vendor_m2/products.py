"""Raster products from a DTM/CHM grid: hillshade, multiscale relief, SVF,
hole masks, review tiles, JSON summaries (m2 processor).

All functions take plain numpy arrays in a north-up grid with ``cell`` metres
per pixel, and are safe with NaN (data holes) because a hole must never be
rendered as a feature.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

try:
    from scipy import ndimage as ndi
except Exception:                       # pragma: no cover
    ndi = None

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco
    prange = range


# ------------------------------------------------------------- primitives ----

def hillshade(dem, cell, azimuth=315.0, altitude=45.0, z_factor=1.5):
    """Horn hillshade in [0, 1]; NaN where the DEM is NaN."""
    z = np.asarray(dem, np.float64)
    nan = np.isnan(z)
    zf = np.where(nan, 0.0, z)
    if ndi is not None:
        zf = ndi.gaussian_filter(zf, 0.0)  # no-op, keeps dtype/shape contract
    # Horn gradients (3x3 weighted)
    a = np.zeros_like(zf); b = np.zeros_like(zf); c = np.zeros_like(zf)
    d = np.zeros_like(zf); e = np.zeros_like(zf); f_ = np.zeros_like(zf)
    g = np.zeros_like(zf); h = np.zeros_like(zf); i = np.zeros_like(zf)
    a[1:-1, 1:-1] = zf[:-2, :-2]; b[1:-1, 1:-1] = zf[:-2, 1:-1]; c[1:-1, 1:-1] = zf[:-2, 2:]
    d[1:-1, 1:-1] = zf[1:-1, :-2]; e[1:-1, 1:-1] = zf[1:-1, 1:-1]; f_[1:-1, 1:-1] = zf[1:-1, 2:]
    g[1:-1, 1:-1] = zf[2:, :-2];  h[1:-1, 1:-1] = zf[2:, 1:-1];  i[1:-1, 1:-1] = zf[2:, 2:]
    dzdx = ((c + 2 * f_ + i) - (a + 2 * d + g)) / (8.0 * cell)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * cell)
    slope = np.arctan(z_factor * np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az = np.deg2rad(azimuth); alt = np.deg2rad(altitude)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0.0, 1.0)
    hs[nan] = np.nan
    return hs


def gauss_relief(dem, cell, sigma_m):
    """High-pass: dem - gaussian(dem, sigma_m).  NaN-sane."""
    z = np.asarray(dem, np.float64)
    if ndi is None:
        return z - np.nanmean(z)
    num = ndi.gaussian_filter(np.where(np.isnan(z), 0.0, z), sigma_m / cell, mode="nearest")
    den = ndi.gaussian_filter((~np.isnan(z)).astype(np.float64), sigma_m / cell, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(den > 1e-4, num / den, np.nan)
    out = z - smooth
    out[np.isnan(z)] = np.nan
    return out


def detrend_window(dem, cell, window_m):
    """dem - grey opening at window (removes regional trend at that scale)."""
    from raster import grey_open
    z = np.asarray(dem, np.float64)
    k = int(round(window_m / cell))
    op = grey_open(z, k)
    out = z - op
    out[np.isnan(z)] = np.nan
    return out


@njit(cache=True, parallel=True)
def _svf_kernel(z, cell, n_dir, max_cells):
    ny, nx = z.shape
    out = np.full((ny, nx), np.nan)
    for j in prange(ny):
        for i in range(nx):
            z0 = z[j, i]
            if np.isnan(z0):
                continue
            acc = 0.0
            for k in range(n_dir):
                ang = 2.0 * np.pi * k / n_dir
                cth = np.cos(ang); sth = np.sin(ang)
                hmax = 0.0
                for d in range(1, max_cells + 1):
                    ii = int(round(i + cth * d)); jj = int(round(j + sth * d))
                    if ii < 0 or ii >= nx or jj < 0 or jj >= ny:
                        break
                    zz = z[jj, ii]
                    if np.isnan(zz):
                        continue
                    sl = (zz - z0) / (d * cell)
                    if sl > hmax:
                        hmax = sl
                acc += 1.0 - hmax / np.sqrt(1.0 + hmax * hmax)
            out[j, i] = acc / n_dir
    return out


def sky_view_factor(dem, cell, n_dir=16, max_dist_m=90.0):
    z = np.asarray(dem, np.float32)
    nc = int(round(max_dist_m / cell))
    t0 = time.time()
    out = _svf_kernel(z, float(cell), int(n_dir), int(nc))
    return out.astype(np.float32), dict(numba=HAVE_NUMBA, seconds=round(time.time() - t0, 2),
                                       n_dir=int(n_dir), max_dist_m=float(max_dist_m))


def norm_uint8(a, lo, hi):
    with np.errstate(invalid="ignore"):
        u = (np.asarray(a, np.float64) - lo) / (hi - lo)
    return (np.clip(u, 0, 1) * 255).astype(np.uint8)


def _colormap(name, n=256):
    import matplotlib
    cm = matplotlib.colormaps[name]
    return (cm(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)


def review_tile_png(dtm, relief, hs, canopy, y0, y1, x0, x1, cell, path,
                    relief_clim=(-1.5, 1.5), upscale=2, title=None):
    """Write one RGB review tile: hillshade + 15 m relief colour + canopy shade.

    The colour overlay is the interpretive layer: ditches read blue/cyan, banks
    and mounds read red/orange.  Areas without DTM are painted mid-grey so an
    empty-data halo can never be confused with flat terrain.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = (slice(y0, y1), slice(x0, x1))
    hs_ = hs[sl]; hh = dtm[sl]; rf = relief[sl]; cp = canopy[sl]
    nan = np.isnan(hh)
    base = np.where(np.isnan(hs_), 0.45, hs_)
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    cm = _colormap("RdYlBu_r") if False else _colormap("seismic")
    lo, hi = relief_clim
    with np.errstate(invalid="ignore"):
        t = np.clip((rf - lo) / (hi - lo), 0, 1)
    idx = np.where(np.isnan(t), 0, (t * 255)).astype(np.int32)
    over = cm[idx].astype(np.float64) / 255.0
    alpha = np.where(np.isnan(t), 0.0, 0.55)
    rgb = rgb * (1 - alpha[:, :, None]) + over * alpha[:, :, None]
    # brighten where canopy is tall: keeps forest obvious in the same image
    cnorm = np.clip(np.nan_to_num(cp) / 20.0, 0, 1)
    rgb = rgb * (1 - 0.25 * cnorm[:, :, None])
    rgb[nan] = rgb[nan] * 0.35 + 0.25          # dark grey for no-data
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    fig = plt.figure(figsize=(img.shape[1] / 100.0, img.shape[0] / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.text(0.01, 0.98, title, transform=ax.transAxes, va="top",
                fontsize=7, color="black",
                bbox=dict(facecolor="white", alpha=0.6, pad=2, edgecolor="none"))
    fig.savefig(path, dpi=100, pil_kwargs=dict(optimize=True))
    plt.close(fig)
    return path


def tile_index(grid, tile_m=1200.0, margin_m=60.0):
    """Yield (name, row0, row1, col0, col1, extent) review tiles."""
    g = grid
    step = int(round(tile_m / g.cell))
    m = int(round(margin_m / g.cell))
    for j0 in range(0, g.ny, step):
        for i0 in range(0, g.nx, step):
            r0 = max(0, j0 - m); r1 = min(g.ny, j0 + step + m)
            c0 = max(0, i0 - m); c1 = min(g.nx, i0 + step + m)
            ext = (g.x0 + c0 * g.cell, g.x0 + c1 * g.cell,
                   g.y0 + r0 * g.cell, g.y0 + r1 * g.cell)
            name = "tileY%05d_X%05d" % (int(round(ext[2])), int(round(ext[0])))
            yield name, r0, r1, c0, c1, ext


def write_products(outdir, grid, dtm, chm, hs, reliefs, svf, hole_mask,
                   canopy_top, summary_extra=None, tiles=True, upscale=2,
                   prefix="", min_tile_fill=0.02, removed_mask=None,
                   reliefs_pre=None):
    """Save numpy raster stack (npz), tiles and a JSON summary."""
    import raster as _raster
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    files = {}
    npz = os.path.join(outdir, (prefix + "rasters.npz"))
    np.savez_compressed(npz, dtm=dtm, chm=chm, hillshade=hs.astype(np.float32),
                        svf=svf.astype(np.float32), hole=hole_mask,
                        canopy=canopy_top.astype(np.float32),
                        **{"relief_%s" % k: v.astype(np.float32) for k, v in reliefs.items()},
                        **{"relief_pre_%s" % k: v.astype(np.float32)
                           for k, v in (reliefs_pre or {}).items()})
    files["rasters"] = npz
    if tiles:
        tdir = os.path.join(outdir, (prefix + "tiles"))
        os.makedirs(tdir, exist_ok=True)
        idx = []
        relief_key = "15" if "15" in reliefs else list(reliefs)[0]
        n_skipped = 0
        for name, r0, r1, c0, c1, ext in tile_index(grid):
            fill = float(np.mean(np.isfinite(dtm[r0:r1, c0:c1])))
            if fill < min_tile_fill:
                n_skipped += 1
                continue
            p = os.path.join(tdir, name + ".png")
            review_tile_png(dtm, reliefs[relief_key], hs, canopy_top,
                            r0, r1, c0, c1, grid.cell, p,
                            upscale=upscale,
                            title="%s r=%s m  ext=%.0f,%.0f" % (name, relief_key, ext[0], ext[2]))
            entry = dict(name=name, png=p, extent=list(ext),
                         data_pct=round(100 * fill, 1))
            if removed_mask is not None:
                rm = removed_mask[r0:r1, c0:c1]
                entry["clamp_rewritten_pct"] = round(100 * float(rm.mean()), 2)
            idx.append(entry)
        files["tiles_dir"] = tdir
        files["tile_index"] = idx
        files["tiles_skipped_empty"] = n_skipped
    summary = dict(
        grid=grid.to_dict(),
        products=dict(dtm=dict(min=float(np.nanmin(dtm)), max=float(np.nanmax(dtm)),
                              nan_pct=float(100 * np.isnan(dtm).mean())),
                      chm=dict(mean=float(np.nanmean(chm)),
                               p95=float(np.nanpercentile(chm, 95)),
                               max=float(np.nanmax(chm))),
                      relief_sigma_m=sorted(float(k) for k in reliefs),
                      hillshade=dict(azimuth=315.0, altitude=45.0)),
        empty_dtm_pct=float(100 * np.isnan(dtm).mean()),
        n_tiles=len(files.get("tile_index", [])),
        write_seconds=round(time.time() - t0, 2),
    )
    if summary_extra:
        summary.update(summary_extra)
    with open(os.path.join(outdir, (prefix + "summary.json")), "w") as f:
        json.dump(summary, f, indent=1)
    files["summary"] = os.path.join(outdir, (prefix + "summary.json"))
    return files
