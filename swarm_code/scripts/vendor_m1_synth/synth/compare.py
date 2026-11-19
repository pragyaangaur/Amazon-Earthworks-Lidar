"""Compare a produced DTM against the noiseless synthetic truth DTM (m1).

Reports, per scene and method:
  * coverage: fraction of cells with a value;
  * vertical error of valid cells against the true ground surface: bias,
    RMSE, MAD, p95|err|, fraction within +-0.3 m - overall and split by
    canopy class (open < 2 m, sparse 2-10 m, closed > 10 m);
  * how much of the *in-forest* ground is recovered at all.

With no real EBA data in the workspace this is the only honest measure of
processor accuracy available, and it is a property of the synthetic scenes
(known noise, known canopy), not of Acre.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def compare(dtm_npz, truth_npz, scene=None):
    d = np.load(dtm_npz, allow_pickle=True)
    t = np.load(truth_npz, allow_pickle=True)
    dem = np.asarray(d["dem"], dtype=np.float64)
    g = float(d["grid_m"])
    # resample the truth rasters onto the product grid (origins differ by <1 m
    # and the product grid is snapped to whole cells)
    from scipy import ndimage
    x0p, y0p = float(d["x0"]), float(d["y0"])
    x0t, y0t, gt = float(t["x0"]), float(t["y0"]), float(t["grid_m"])
    ny, nx = dem.shape
    rows = (y0p + (np.arange(ny) + 0.5) * g - y0t) / gt - 0.5
    cols = (x0p + (np.arange(nx) + 0.5) * g - x0t) / gt - 0.5
    R, C = np.meshgrid(rows, cols, indexing="ij")
    true = ndimage.map_coordinates(np.asarray(t["dem_true"], float), [R, C],
                                   order=1, mode="constant", cval=np.nan)
    chm = ndimage.map_coordinates(np.asarray(t["chm_mean"], float), [R, C],
                                  order=1, mode="constant", cval=0.0)
    pulses = ndimage.map_coordinates(np.asarray(t["pulses"], float), [R, C],
                                     order=0, mode="constant", cval=0.0)
    ok = np.isfinite(dem) & np.isfinite(true)
    err = np.where(ok, dem - true, np.nan)
    out = dict(scene=scene or os.path.basename(dtm_npz),
               method=str(d["method"]) if "method" in d else "?",
               grid_m=float(d["grid_m"]),
               cells=int(dem.size),
               coverage_pct=100.0 * ok.mean(),
               empty_truth_pct=100.0 * (pulses == 0).mean())
    def stats(mask, tag):
        m = mask & ok & np.isfinite(err)
        if m.sum() < 20:
            out[tag + "_n"] = int(m.sum())
            return
        e = err[m]
        out.update({tag + "_n": int(m.sum()),
                    tag + "_bias_m": round(float(np.mean(e)), 3),
                    tag + "_rmse_m": round(float(np.sqrt(np.mean(e ** 2))), 3),
                    tag + "_mad_m": round(float(np.median(np.abs(e - np.median(e)))), 3),
                    tag + "_p95abs_m": round(float(np.percentile(np.abs(e), 95)), 3),
                    tag + "_frac_within_0p3": round(float(np.mean(np.abs(e) <= 0.3)), 3)})
    stats(np.ones_like(ok), "all")
    stats(chm < 2.0, "open")
    stats((chm >= 2.0) & (chm < 10.0), "sparse")
    stats(chm >= 10.0, "closed")
    # how much of the closed-canopy *ground* is present at all
    cl = chm >= 10.0
    out["closed_coverage_pct"] = round(100.0 * (ok & cl).sum() / max(1, cl.sum()), 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm", nargs="+", required=True, help="*_dtm.npz")
    ap.add_argument("--truth", nargs="+", required=True, help="*_truth.npz")
    ap.add_argument("--out", default=None, help="csv path")
    a = ap.parse_args(argv)
    dtms = []
    for p in a.dtm:
        dtms.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    truths = {}
    for p in a.truth:
        for f in (sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]):
            truths[os.path.basename(f).replace("_truth.npz", "")] = f
    rows = []
    for npz in dtms:
        base = os.path.basename(npz).replace("_dtm.npz", "")
        tn = truths.get(base)
        if tn is None:
            print("skip %s (no truth npz)" % npz)
            continue
        r = compare(npz, tn, scene=base)
        r["dir"] = os.path.basename(os.path.dirname(npz))
        rows.append(r)
        print("%-40s cov %5.1f%% all_rmse %6.3f open_rmse %6.3f closed_rmse %s "
              "closed_cov %5.1f%%" % (
                  base + " [" + r.get("method", "?") + "]",
                  r.get("coverage_pct", float("nan")),
                  r.get("all_rmse_m", float("nan")),
                  r.get("open_rmse_m", float("nan")),
                  r.get("closed_rmse_m", "n/a"),
                  r.get("closed_coverage_pct", float("nan"))))
    if a.out and rows:
        keys = sorted({k for r in rows for k in r})
        front = ["scene", "method", "dir"]
        keys = front + [k for k in keys if k not in front]
        with open(a.out, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
