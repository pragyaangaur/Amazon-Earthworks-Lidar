"""Check a published control site (or any centre) against a 3 m product (m1).

Answers three questions with one estimator, on real or synthetic products:

  1. is there a ditch at radius R around this point?  (dip, azimuthal
     consistency, bank, interior-vs-exterior flatness -- a pond or pit has a
     LOW interior, a ring ditch does not)
  2. what is the same statistic worth elsewhere in this transect?  (null
     distribution over an open-canopy scan grid at the same radii; a dip is
     only interesting relative to that null)
  3. does the answer change between filter recipes?  (run it on two products)

No network, no coordinates leave the workspace; output is local JSON + text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import ring_metrics     # noqa: E402


def open_mask(d, chm_max=3.0, hole_dist_m=20.0, grid=None):
    g = float(d["grid_m"])
    chm = np.asarray(d["chm"], dtype=float)
    hole = np.asarray(d["hole"]) | ~np.isfinite(np.asarray(d["dem"], float))
    dist = ndimage.distance_transform_edt(~hole) * g
    return (np.isfinite(chm) & (chm < chm_max) & (dist > hole_dist_m)
            & np.isfinite(np.asarray(d["dem"], float)))


def null_scan(dem, g, x0, y0, radii, mask, stride=25.0, halfw=None, nth=36,
              min_valid=0.9):
    ny, nx = dem.shape
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return dict(error="no open-canopy cells")
    xs = x0 + (cols + 0.5) * g
    ys = y0 + (rows + 0.5) * g
    # thin to a grid
    keep = ((xs - x0) % stride < g) & ((ys - y0) % stride < g)
    xs, ys = xs[keep], ys[keep]
    out = {}
    for r in radii:
        hw = halfw or max(4.0, 0.15 * r)
        vals = []
        for cx, cy in zip(xs, ys):
            m = ring_metrics(dem, g, x0, y0, cx, cy, r, halfw=hw, nth=nth)
            if m.get("dip_m") is not None and (m.get("valid_frac") or 0) >= min_valid:
                vals.append((m["dip_m"], m.get("dip_az25", np.nan),
                             m.get("interior_m", np.nan), m.get("bank_m", np.nan)))
        if not vals:
            continue
        a = np.asarray(vals, float)
        out["%.0f" % r] = dict(
            n=int(a.shape[0]),
            dip_p01=float(np.nanpercentile(a[:, 0], 1)),
            dip_p05=float(np.nanpercentile(a[:, 0], 5)),
            dip_p50=float(np.nanpercentile(a[:, 0], 50)),
            dip_p95=float(np.nanpercentile(a[:, 0], 95)),
            dip_max=float(np.nanmax(a[:, 0])),
            az25_p95=float(np.nanpercentile(a[:, 1], 95)),
            az25_max=float(np.nanmax(a[:, 1])))
    return out


def check_centre(d, cx, cy, radii, nth=72):
    g = float(d["grid_m"])
    x0, y0 = float(d["x0"]), float(d["y0"])
    dem = np.asarray(d["dem"], dtype=float)
    out = {}
    for r in radii:
        hw = max(4.0, 0.15 * r) if r >= 30 else max(3.0, 0.12 * r)
        out["%.0f" % r] = ring_metrics(dem, g, x0, y0, cx, cy, r, halfw=hw, nth=nth)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm", required=True, nargs="+")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--radii", default="12,15,20,25,30,40,50,60,80,100")
    ap.add_argument("--stride", type=float, default=25.0)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--null-radii", default="20,30,45,60")
    ap.add_argument("--chm-max", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    radii = [float(v) for v in a.radii.split(",")]
    nradii = [float(v) for v in a.null_radii.split(",")]
    res = dict(centre=[a.x, a.y], products={})
    for p in a.dtm:
        d = np.load(p, allow_pickle=True)
        entry = dict(product=os.path.basename(p),
                     method=str(d["method"]) if "method" in d else "?",
                     epsg=int(d["epsg"]) if "epsg" in d else None,
                     centre=check_centre(d, a.x, a.y, radii))
        if a.null:
            m = open_mask(d, chm_max=a.chm_max)
            entry["null"] = null_scan(np.asarray(d["dem"], float),
                                       float(d["grid_m"]), float(d["x0"]),
                                       float(d["y0"]), nradii, m, stride=a.stride)
            entry["open_cells"] = int(m.sum())
        res["products"][entry["method"]] = entry
        print("\n== %s  [%s]  EPSG %s" % (p, entry["method"], entry["epsg"]))
        print("   r(m) |   dip | az_frac | az25 | interior | exterior | bank | valid")
        for k, m in entry["centre"].items():
            print("  %5s | %5s | %7s | %4s | %8s | %8s | %4s | %5s" % (
                k,
                "-" if m.get("dip_m") is None else "%.2f" % m["dip_m"],
                "-" if m.get("az_frac_dip") is None else "%.2f" % m["az_frac_dip"],
                "-" if m.get("dip_az25") is None else "%.2f" % m["dip_az25"],
                "-" if m.get("interior_m") is None else "%.2f" % m["interior_m"],
                "-" if m.get("exterior_m") is None else "%.2f" % m["exterior_m"],
                "-" if m.get("bank_m") is None else "%.2f" % m["bank_m"],
                "%.2f" % (m.get("valid_frac") or 0)))
        if "null" in entry:
            nf = entry["null"]
            print("   null (open canopy, %d cells):" % entry["open_cells"])
            for k, v in nf.items():
                print("     r=%-4s n=%-4d dip p01 %6.2f p05 %6.2f p50 %5.2f "
                      "p95 %5.2f max %6.2f | az25 p95 %5.2f max %5.2f" % (
                          k, v["n"], v["dip_p01"], v["dip_p05"], v["dip_p50"],
                          v["dip_p95"], v["dip_max"], v["az25_p95"], v["az25_max"]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
