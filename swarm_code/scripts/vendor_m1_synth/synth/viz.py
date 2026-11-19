"""Small plotting / hillshade helpers (matplotlib only, no GDAL)."""
from __future__ import annotations

import numpy as np


def hillshade(dem, res=3.0, az=315.0, alt=45.0):
    """Standard hillshade; NaN cells stay NaN."""
    z = np.asarray(dem, dtype=float)
    gy, gx = np.gradient(np.where(np.isfinite(z), z, np.nanmedian(z)), res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    hs = np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)
    hs = np.clip(hs, 0.0, 1.0)
    hs[~np.isfinite(z)] = np.nan
    return hs


def quicklook(png, dem, x0, y0, res, title="", features=(), holes=(),
              vmin=None, vmax=None, figsize=(9, 9), dpi=130, hpng=None):
    """Hillshade quicklook with optional truth overlays (metres, scene coords)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    hs = hillshade(dem, res)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ext = [x0, x0 + dem.shape[1] * res, y0, y0 + dem.shape[0] * res]
    ax.imshow(hs, cmap="gray", extent=ext, origin="lower", vmin=0, vmax=1)
    if vmin is not None or vmax is not None:
        ax.imshow(np.where(np.isfinite(dem), dem, np.nan), cmap="terrain",
                  alpha=0.35, extent=ext, origin="lower", vmin=vmin, vmax=vmax)
    for f in features:
        col = {"target": "#ff3b30", "negative": "#007aff",
               "modern": "#34c759", "artefact": "#ff9500"}.get(
                   f.get("role"), "#ffffff")
        if f["type"] in ("ring_ditch", "square_ditch"):
            ax.add_patch(Circle((f["cx"], f["cy"]), f["r"], fill=False,
                                color=col, lw=1.2, ls="--"))
            ax.annotate(f["id"], (f["cx"], f["cy"]), color=col, fontsize=7)
        elif f["type"] == "mound_ring":
            ax.add_patch(Circle((f["cx"], f["cy"]), f["R"], fill=False,
                                color=col, lw=1.2, ls="--"))
            ax.annotate(f["id"], (f["cx"], f["cy"]), color=col, fontsize=7)
        elif "p0" in f:
            ax.plot([f["p0"][0], f["p1"][0]], [f["p0"][1], f["p1"][1]],
                    color=col, lw=1.0, ls="-")
            ax.annotate(f["id"], f["p0"], color=col, fontsize=7)
        elif f["type"] == "hill":
            ax.add_patch(Circle((f["cx"], f["cy"]), f["r"], fill=False,
                                color=col, lw=0.8, ls=":"))
        elif f["type"] in ("oxbow", "pond", "hole_halo"):
            ax.add_patch(Circle((f["cx"], f["cy"]), f["r"], fill=False,
                                color=col, lw=1.0, ls=":"))
            ax.annotate(f["id"], (f["cx"], f["cy"]), color=col, fontsize=7)
    for h in holes:
        if h.get("type") == "circle":
            ax.add_patch(Circle((h["cx"], h["cy"]), h["r"], fill=True,
                                color="#00ffff", alpha=0.18))
        elif h.get("type") == "rect":
            bx0, by0, bx1, by1 = h["bbox"]
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                   color="#00ffff", alpha=0.18))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m, EPSG:32719 easting)")
    ax.set_ylabel("y (m, northing)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    if hpng:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(dem, cmap="gray", extent=ext, origin="lower")
        ax.set_title("DEM (m)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(hpng)
        plt.close(fig)
