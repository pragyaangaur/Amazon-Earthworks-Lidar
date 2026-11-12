"""Build a bare-ground model from one EBA lidar transect and render views for inspection.

Usage: python -I src/dtm.py data/raw/NP_T-0414.laz outputs/T-0414

Outputs a 1 m DTM as a .npy file plus PNG tiles of a hillshade and a local relief model. The local relief model is the DTM minus a smoothed copy of itself, which removes the natural slope and leaves small features such as ditches and mounds.
"""
import sys
from pathlib import Path

import laspy
import numpy as np
from scipy import ndimage

RES = 1.0          # grid size in metres
LRM_SIGMA = 15.0   # smoothing scale in metres; ditches are 5 to 15 m wide
TILE = 800         # tile edge in pixels for review images


def read_ground(path):
    las = laspy.read(path)
    cls = np.asarray(las.classification)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    ground = cls == 2
    if ground.sum() < 1000:
        # Unclassified data: keep everything and let the grid minimum act as a crude ground filter.
        ground = np.ones_like(cls, dtype=bool)
    print(f"{len(cls)} points, {ground.sum()} ground, crs={las.header.parse_crs()}")
    return x[ground], y[ground], z[ground]


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
    h, w = dtm.shape
    for r in range(0, h, TILE):
        for c in range(0, w, TILE):
            sl = (slice(r, r + TILE), slice(c, c + TILE))
            if np.isnan(dtm[sl]).mean() > 0.8:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(hs[sl], cmap="gray")
            ax[1].imshow(rel[sl], cmap="RdBu_r", vmin=-0.6, vmax=0.6)
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
    dtm = fill_gaps(remove_spikes(fill_gaps(dtm, 3)))
    np.save(outdir / "dtm.npy", dtm)
    np.save(outdir / "origin.npy", np.array(origin))
    print(f"DTM {dtm.shape}, {np.isnan(dtm).mean():.1%} empty")
    save_tiles(dtm, origin, outdir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
