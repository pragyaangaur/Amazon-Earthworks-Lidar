"""Make the README figures from the ground models on disk.

Usage: python -I docs/make_figures.py

It reads the local 3 m DTMs in outputs/private/ and the swarm DTMs in swarm/dtm/, which are not in this repository. The figures it writes to docs/figures/ have no axes, no coordinates and no feature positions, so they are safe to publish.
"""
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
SURFACE, INK, INK2, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def hillshade(d, res=3.0, az=315, alt=35):
    f = np.nan_to_num(d, nan=np.nanmedian(d))
    gy, gx = np.gradient(f, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, b = np.radians(az), np.radians(alt)
    hs = np.sin(b) * np.cos(slope) + np.cos(b) * np.sin(slope) * np.cos(a - aspect)
    hs[np.isnan(d)] = np.nan
    return hs


def relief(d, sigma_m=15.0, res=3.0):
    f = np.nan_to_num(d, nan=np.nanmedian(d))
    r = d - ndimage.gaussian_filter(f, sigma_m / res)
    r[ndimage.distance_transform_edt(~np.isnan(d)) * res < 15] = np.nan
    return r


def local_dtm(t):
    return np.load(ROOT / "outputs" / "private" / f"T-{t}" / "dtm.npy")


def swarm_dtm(t):
    z = np.load(ROOT / "swarm" / "dtm" / f"{t}_dtm3m.npz")
    d = z["dtm"].astype(float)
    return d[::-1] if str(z["row_order"]) == "south-up" else d


def panel(ax, img, title, cmap="gray", vmin=None, vmax=None):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(SURFACE)
    ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", color=INK)


def hero():
    known = local_dtm("0414")[1446 - 45:1446 + 45, 1930 - 45:1930 + 45]
    d = swarm_dtm("0470")
    # The control crop is given in south-up rows, so convert it to north-up rows.
    h = d.shape[0]
    r0, r1 = h - 1816, h - 1612
    forest = d[r0 - 20:r1 + 20, 305:547]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
    panel(ax[0], hillshade(known), "Open pasture: a recorded geoglyph,\na ditched circle about 98 m across")
    panel(ax[1], hillshade(forest), "Under 20 to 35 m of forest: two reported\nearthworks lie in this view, and nothing shows")
    for a, n in zip(ax, ["270 m", "720 m"]):
        a.text(0.02, 0.03, f"view about {n} wide", transform=a.transAxes, fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02, wspace=0.06)
    fig.savefig(OUT / "hero.png", dpi=110)


def funnel():
    stages = [("Lidar strips with a ground model", 899), ("Features flagged by the detectors", 3537),
              ("Possible, with an open explanation", 42), ("Strong candidates", 0), ("New sites", 0)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y = np.arange(len(stages))[::-1]
    for yi, (name, v) in zip(y, stages):
        if v > 0:
            ax.barh(yi, v, left=1, color=BLUE, height=0.55)
        ax.text(0.85, yi, name, va="center", ha="right", fontsize=10, color=INK)
        ax.text(max(v, 1) * 1.15 if v else 1.05, yi, f"{v:,}", va="center", fontsize=10, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1, 12000)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_yticks([]); ax.set_xticks([]); ax.minorticks_off()
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("From 899 strips to zero discoveries (bar length on a log scale)", fontsize=11, loc="left")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.88, bottom=0.04)
    fig.savefig(OUT / "funnel.png", dpi=110)


def lookalikes():
    items = [
        ("0413", (349, 749, 3700, 4100), "A meandering river"),
        ("0415", (1600, 2000, 3450, 3850), "Edges of holes in the data"),
        ("0414", (3750, 4050, 100, 400), "Farm ponds and buildings"),
        ("0414", (2520, 2820, 930, 1230), "A straight stream valley"),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.8))
    for a, (t, (r0, r1, c0, c1), name) in zip(ax, items):
        d = local_dtm(t)[r0:r1, c0:c1]
        f = np.nan_to_num(d, nan=np.nanmedian(d))
        r = d - ndimage.gaussian_filter(f, 5)
        panel(a, r, name, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    fig.suptitle("Things that fooled the first look (local relief, red is high and blue is low)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "lookalikes.png", dpi=100)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hero()
    funnel()
    lookalikes()
    print("figures written to", OUT)
