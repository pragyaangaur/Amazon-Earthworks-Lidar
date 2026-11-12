"""Render one region of a transect at full detail with canopy height, to judge a feature by eye.

Usage: python -I src/inspect_region.py <transect.laz> <outdir> <row0> <col0> <size> <name>

Rows and columns are in DTM pixels. The PNG goes into outdir, which should be under outputs/private.
"""
import sys
from pathlib import Path

import laspy
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dtm import RES, hillshade, lrm


def canopy(laz, origin, dtm):
    las = laspy.read(laz)
    x0, y0 = origin
    ix = ((np.asarray(las.x) - x0) / RES).astype(int)
    iy = ((y0 - np.asarray(las.y)) / RES).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < dtm.shape[1]) & (iy < dtm.shape[0])
    top = np.full(dtm.shape, -np.inf)
    np.maximum.at(top, (iy[ok], ix[ok]), np.asarray(las.z)[ok])
    return top - dtm


def main(laz, outdir, r0, c0, n, name):
    out = Path(outdir)
    dtm = np.load(out / "dtm.npy")
    origin = np.load(out / "origin.npy")
    chm = canopy(laz, origin, dtm)
    sl = (slice(r0, r0 + n), slice(c0, c0 + n))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(hillshade(dtm)[sl], cmap="gray"); ax[0].set_title("hillshade")
    ax[1].imshow(lrm(dtm)[sl], cmap="RdBu_r", vmin=-0.8, vmax=0.8); ax[1].set_title("local relief (m)")
    im = ax[2].imshow(chm[sl], cmap="Greens", vmin=0, vmax=35); ax[2].set_title("canopy height (m)")
    fig.colorbar(im, ax=ax[2], fraction=0.04)
    fig.suptitle(f"{name}: rows {r0}+{n}, cols {c0}+{n}, {RES} m per px")
    fig.tight_layout()
    fig.savefig(out / f"inspect_{name}.png", dpi=80)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), int(a[4]), int(a[5]), a[6])
