"""Build the same products from the *noiseless truth DTM* (benchmark ceiling).

This isolates losses that come from the product chain (grid, filters, relief,
hole masking) from losses that come from the point cloud (noise, canopy
occlusion, gaps).  Nothing here mimics real data; it is the upper bound of
what the pipeline could deliver on this synthetic scene.

Usage: python synth/ceiling.py out/s1_ring40_open_truth.npz --out out/s1_ceiling
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refdtm import products_from_minz, save, sky_view_factor   # noqa: E402


def build(truth_npz, method="vstruct"):
    t = np.load(truth_npz, allow_pickle=True)
    dem = np.asarray(t["dem_true"], dtype=np.float32)
    pulses = np.asarray(t["pulses"])
    cnt = (pulses > 0).astype(np.int32)
    minz = np.where(np.isfinite(dem), dem, np.inf)
    maxz = np.where(np.isfinite(dem), dem, -np.inf)
    out, stats = products_from_minz(minz, maxz, cnt, grid=float(t["grid_m"]),
                                    method=method)
    out["method"] = np.array("ceiling")
    out.update(x0=float(t["x0"]), y0=float(t["y0"]), x1=float(t["x1"]),
               y1=float(t["y1"]), grid_m=float(t["grid_m"]),
               epsg=int(t["epsg"]))
    stats.update(source=os.path.basename(truth_npz), kind="ceiling-from-truth")
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("truth", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="vstruct")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args(argv)
    paths = []
    for p in a.truth:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    os.makedirs(a.out, exist_ok=True)
    for p in paths:
        out, stats = build(p, method=a.method)
        base = os.path.basename(p).replace("_truth.npz", "")
        npz = os.path.join(a.out, base + "_dtm.npz")
        tj = p.replace("_truth.npz", "_truth.json")
        feats, holes = None, None
        if os.path.exists(tj):
            tr = json.load(open(tj))
            feats, holes = tr.get("features"), tr.get("holes")
        save(out, stats, npz, (os.path.join(a.out, base + "_dtm.png")
                               if a.png else None), feats, holes)
        print("ceiling %-24s removed %.2f%% hole %.2f%% valid %.1f%% -> %s"
              % (base, stats["removed_pct"], stats["hole_pct"],
                 stats["valid_pct"], npz), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
