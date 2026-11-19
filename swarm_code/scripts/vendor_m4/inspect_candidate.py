"""inspect_candidate.py -- print numeric cross-sections through a candidate.

This is the check that decides borderline cases without a screen: sample the DTM
outward from the candidate centre along N/E/S/W (and any extra azimuths) and print
the levels.  A ditch shows a narrow trough that returns to the interior level on
every azimuth; a terrace/scarp steps down and stays down; a pond/basin is low at
the centre.

Usage:
  python3 det/inspect_candidate.py --npz ../m2/out/0414_v3/rasters.npz \
      --summary ../m2/out/0414_v3/summary.json --det out/real/0414_v3/detections.json \
      --id R0414-001 --step 3 --rmax 90
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess
import run_real


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--masks", default="")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--det", required=True, help="detections.json from run_real.py")
    ap.add_argument("--id", required=True)
    ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--rmax", type=float, default=90.0)
    ap.add_argument("--az", default="E,N,W,S")
    args = ap.parse_args(argv)
    prod = run_real.load_products(args.npz, args.masks, args.summary, "inspect")
    dets = json.load(open(args.det))
    det = next(d for d in dets if d.get("id") == args.id)
    row, col = det["row"], det["col"]
    cell = prod["cell"]
    dtm = prod["dtm"]
    hp = preprocess.local_relief(np.where(np.isfinite(dtm), dtm, np.nan),
                                 np.isfinite(dtm), 15.0, cell)
    print("candidate %s: R=%.1f m dip=%.2f m width=%s shape=%s flags=%s"
          % (args.id, det["radius_m"], det["dip_m"], det.get("width_m"),
             det.get("shape"), det.get("flags")))
    print("cell row/col %.1f %.1f (only local cells printed; no map coordinates)"
          % (row, col))
    t = np.arange(0.0, args.rmax + 0.5 * args.step, args.step)
    for name in [a.strip() for a in args.az.split(",") if a.strip()]:
        ang = {"E": 0.0, "N": 90.0, "W": 180.0, "S": -90.0}[name] if name in "ENWS" \
            else float(name)
        a = np.radians(ang)
        yy = row - t * np.sin(a) / cell
        xx = col + t * np.cos(a) / cell
        f = np.vstack([yy, xx])
        z = ndi.map_coordinates(np.where(np.isfinite(dtm), dtm, -999.0), f, order=1,
                                mode="constant", cval=-999.0)
        z = np.where(z < -900, np.nan, z)
        l = ndi.map_coordinates(np.where(np.isfinite(hp), hp, -999.0), f, order=1,
                                mode="constant", cval=-999.0)
        l = np.where(l < -900, np.nan, l)
        print("\nazimuth %-3s  r = %s m" % (name, t.astype(int).tolist()))
        print("   dz (rel. centre): %s" % np.round(z - z[0], 2).tolist())
        print("   LRM15           : %s" % np.round(l, 2).tolist())


if __name__ == "__main__":
    main()
