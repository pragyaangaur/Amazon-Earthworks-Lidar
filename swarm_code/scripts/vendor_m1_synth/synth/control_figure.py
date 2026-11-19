"""Crop a product around a centre and draw hillshade / relief / canopy panels."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz import hillshade    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--half", type=float, default=200.0)
    ap.add_argument("--png", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--radii", default="20,50")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    d = np.load(a.dtm, allow_pickle=True)
    g = float(d["grid_m"]); x0 = float(d["x0"]); y0 = float(d["y0"])
    i0 = max(0, int((a.x - a.half - x0) / g)); i1 = int((a.x + a.half - x0) / g) + 1
    j0 = max(0, int((a.y - a.half - y0) / g)); j1 = int((a.y + a.half - y0) / g) + 1
    cx0 = x0 + i0 * g; cy0 = y0 + j0 * g
    panels = [("dem", "DTM", "gray"), ("relief15", "relief 15 m", "RdBu_r"),
              ("chm", "canopy height", "viridis")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=140)
    for ax, (k, lab, cm) in zip(axes, panels):
        if k not in d:
            ax.text(0.5, 0.5, "no %s" % k, ha="center")
            continue
        a2 = np.asarray(d[k], dtype=float)[j0:j1, i0:i1]
        ext = [cx0, cx0 + a2.shape[1] * g, cy0, cy0 + a2.shape[0] * g]
        if k == "dem":
            im = ax.imshow(hillshade(a2, g), cmap=cm, extent=ext, origin="lower",
                           vmin=0, vmax=1)
        else:
            if k == "relief15":
                v = np.nanpercentile(np.abs(a2), 98) or 1.0
                im = ax.imshow(a2, cmap=cm, extent=ext, origin="lower", vmin=-v, vmax=v)
            else:
                im = ax.imshow(a2, cmap=cm, extent=ext, origin="lower", vmin=0,
                               vmax=max(5.0, np.nanpercentile(a2, 98)))
            fig.colorbar(im, ax=ax, shrink=0.75)
        for rad in [float(v) for v in a.radii.split(",") if v.strip()]:
            ax.add_patch(Circle((a.x, a.y), rad, fill=False, color="r", lw=1.0, ls="--"))
        ax.plot(a.x, a.y, "r+", ms=10, mew=1.5)
        ax.set_title(lab, fontsize=9)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=6)
    fig.suptitle(a.title or os.path.basename(a.dtm), fontsize=9)
    fig.tight_layout()
    fig.savefig(a.png)
    print("wrote", a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
