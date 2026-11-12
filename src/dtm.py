"""Build a bare-ground model from one EBA lidar transect and render views for inspection.

Usage: python -I src/dtm.py data/raw/NP_T-0414.laz outputs/T-0414

Outputs a 1 m DTM as a .npy file plus PNG tiles of a hillshade and a local relief model. The local relief model is the DTM minus a smoothed copy of itself, which removes the natural slope and leaves small features such as ditches and mounds.
"""
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage

RES = 3.0          # grid size in metres; about 2 points per m2 leaves few ground hits per 1 m cell under forest
LRM_SIGMA = 15.0   # smoothing scale in metres; ditches are 5 to 15 m wide
TILE = 400         # tile edge in pixels for review images


def read_ground(path):
    # The class 2 points in the EBA files include canopy returns up to 40 m above the ground
    # under forest, so every point is kept and ground_filter() does the work instead.
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    print(f"{len(z)} points, crs={las.header.parse_crs()}")
    return x, y, z


def ground_filter(dtm, passes=(40, 20, 10), tol=0.7):
    # Iterative surface fit: cells that sit more than tol metres above a smoothed copy of the
    # surface are treated as vegetation and removed, from coarse to fine scale.
    dtm = dtm.copy()
    for size in passes:
        k = max(3, int(size / RES))
        filled = fill_gaps(dtm, 10 ** 6)
        base = ndimage.grey_opening(filled, size=k)
        base = ndimage.uniform_filter(base, k)
        dtm[(dtm - base) > tol] = np.nan
    return dtm


def grid_min(x, y, z):
    x0, y0 = x.min(), y.max()
    ix = ((x - x0) / RES).astype(int)
    iy = ((y0 - y) / RES).astype(int)
    dtm = np.full((iy.max() + 1, ix.max() + 1), np.inf)
    np.minimum.at(dtm, (iy, ix), z)
    dtm[np.isinf(dtm)] = np.nan
    return dtm, (x0, y0)


def fill_gaps(dtm, max_gap=10):
    # Nearest neighbour fill for small holes under dense canopy. Large holes stay empty.
    mask = np.isnan(dtm)
    dist, (iy, ix) = ndimage.distance_transform_edt(mask, return_indices=True)
    out = dtm[iy, ix]
    out[dist > max_gap] = np.nan
    return out


def remove_spikes(dtm, k=5, tol=1.0):
    # Drop cells far above their neighbourhood median, which are usually missed vegetation.
    med = ndimage.median_filter(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), size=k)
    bad = (dtm - med) > tol
    dtm = dtm.copy()
    dtm[bad] = np.nan
    return dtm


def lrm(dtm):
    filled = np.nan_to_num(dtm, nan=np.nanmedian(dtm))
    return dtm - ndimage.gaussian_filter(filled, LRM_SIGMA / RES)


def hillshade(dtm, az=315, alt=35):
    gy, gx = np.gradient(np.nan_to_num(dtm, nan=np.nanmedian(dtm)), RES)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    return np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)


def save_tiles(dtm, origin, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rel, hs = lrm(dtm), hillshade(dtm)
    # Holes in the ground data (ponds, buildings, dense canopy) give ring-shaped halos in the
    # relief image that look like small enclosures, so the relief is blanked near every hole.
    near_hole = ndimage.distance_transform_edt(~np.isnan(dtm)) * RES < 15
    rel[near_hole] = np.nan
    h, w = dtm.shape
    for r in range(0, h, TILE):
        for c in range(0, w, TILE):
            sl = (slice(r, r + TILE), slice(c, c + TILE))
            if np.isnan(dtm[sl]).mean() > 0.8:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(hs[sl], cmap="gray")
            ax[1].imshow(rel[sl], cmap="RdBu_r", vmin=-0.8, vmax=0.8)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"tile r{r} c{c} (1 px = {RES} m)")
            fig.tight_layout()
            fig.savefig(outdir / f"tile_r{r:05d}_c{c:05d}.png", dpi=80)
            plt.close(fig)


def main(src, out):
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    x, y, z = read_ground(src)
    dtm, origin = grid_min(x, y, z)
    dtm = fill_gaps(ground_filter(dtm), 5)
    np.save(outdir / "dtm.npy", dtm)
    np.save(outdir / "origin.npy", np.array(origin))
    print(f"DTM {dtm.shape}, {np.isnan(dtm).mean():.1%} empty")
    save_tiles(dtm, origin, outdir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
