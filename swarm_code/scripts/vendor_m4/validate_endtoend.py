"""validate_endtoend.py -- detector validation on a processed synthetic transect.

Uses another member's point-cloud -> DTM product whose ground truth is known
(m2's synthetic EBA control: 40 m ring ditch 6 m wide 1.5 m deep with a low bank
in open pasture, straight sunken/embanked roads, a 6-mound ring, a meander river
and a pond).  This is the end-to-end test: points -> DTM -> detector, including
the processor's ground filter, which the digital spike-in cannot test.

Usage:
  python3 det/validate_endtoend.py \
     --npz ../m2/out/synth/rasters.npz --masks ../m2/out/synth/masks.npy \
     --summary ../m2/out/synth/summary.json --truth ../m2/synth_out/synth_truth.json \
     --out out/real/m2synth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose
import linedet
import mounddet
import preprocess
import ringdet
import run_real
import scoring


def map_xy(prod, x, y):
    """(x, y) -> (row, col) for the processor grid (rows increase north)."""
    cell = prod["cell"]
    if prod.get("row_order", "south-up") == "south-up":
        row = (y - prod["y0"]) / cell - 0.5
    else:
        row = (prod["y0"] - y) / cell - 0.5
    col = (x - prod["x0"]) / cell - 0.5
    return row, col


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../m2/out/synth/rasters.npz")
    ap.add_argument("--masks", default="../m2/out/synth/masks.npy")
    ap.add_argument("--summary", default="../m2/out/synth/summary.json")
    ap.add_argument("--truth", default="../m2/synth_out/synth_truth.json")
    ap.add_argument("--out", default="out/real/m2synth")
    ap.add_argument("--transect", default="m2synth")
    ap.add_argument("--evidence", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    prod = run_real.load_products(args.npz, args.masks, args.summary, args.transect)
    truth = json.load(open(args.truth))
    prep = run_real.build_prep(prod)
    cell = prod["cell"]
    area_km2 = float(prep["infer"].sum()) * cell ** 2 / 1e6

    dets = ringdet.detect_rings(prep, dict(min_dip_m=0.30))
    mounds = mounddet.detect_mounds(prep)
    lines = linedet.detect_lines(prep)

    L = ["# End-to-end detector validation on the processed synthetic control", "",
         "Input: `%s` (+ masks) from the processor, %dx%d cells at %.0f m, EPSG %s, "
         "inferable area %.3f km2. Truth: `%s`."
         % (os.path.basename(args.npz), prod["dtm"].shape[0], prod["dtm"].shape[1],
            cell, prod["epsg"], area_km2, os.path.basename(args.truth)), "",
         "This tests the whole chain (point cloud -> ground filter -> DTM -> detector). "
         "A miss here is a chain failure, not necessarily a detector failure.", ""]
    rows = []
    ring_truth = [f for f in truth["features"] if f["type"] == "ring_ditch"]
    L.append("## Ring ditches")
    L.append("")
    L.append("| truth | x,y | R truth | found | R meas | dip meas | retention | width | SNR | cov | rel.std | screened | reason |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ring_truth:
        tr, tc = map_xy(prod, t["x"], t["y"])
        best, bd = None, 1e9
        for d in dets:
            dist = np.hypot(d["col"] * cell - (t["x"] - prod["x0"]),
                            d["row"] * cell - ((t["y"] - prod["y0"])))
            if dist < 20.0 and 0.6 * t["radius_m"] <= d["radius_m"] <= 1.5 * t["radius_m"] \
                    and dist < bd:
                best, bd = d, dist
        ok, reason = scoring.screened(best) if best else (False, "missed")
        rows.append(dict(truth=t["id"], found=best is not None, dist_m=bd,
                         R_meas=(best["radius_m"] if best else None),
                         dip=(best["dip_m"] if best else None),
                         screened=ok, reason=reason))
        L.append("| %s | %.0f,%.0f | %.0f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (t["id"], t["x"], t["y"], t["radius_m"], "YES" if best else "no",
                    _f(best and best["radius_m"], 1), _f(best and best["dip_m"], 2),
                    _f((best["dip_m"] / t["depth_m"]) if best else None, 2),
                    _f(best and best["width_m"], 0), _f(best and best.get("snr"), 1),
                    _f(best and best["cov"], 2), _f(best and best["rel_std"], 2),
                    "PASS" if ok else "reject", reason))
    L.append("")
    L.append("All ring detections in this scene (%d):" % len(dets))
    L.append("")
    L.append("| row | col | R (m) | dip (m) | width (m) | SNR | cov | shape | screened | reason |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in dets[:25]:
        ok, reason = scoring.screened(d)
        L.append("| %.0f | %.0f | %.0f | %.2f | %s | %.1f | %.2f | %s | %s | %s |"
                 % (d["row"], d["col"], d["radius_m"], d["dip_m"], _f(d.get("width_m"), 0),
                    d.get("snr", np.nan), d["cov"], d["shape"],
                    "PASS" if ok else "reject", reason))
    L.append("")
    L.append("## Mounds and mound ring")
    L.append("")
    mt = truth.get("mound_ring", {})
    L.append("Truth: %d mounds of 0.5-3 m on a %.0f m ring around a %.0f m plaza at %.0f,%.0f; "
             "plus one isolated 2 m mound.  Note the processor's own validation reports that the "
             "anti-canopy clamp removes most of the >=1.5 m mounds by design, so a partial "
             "recovery here is a DTM limitation, not a detector miss."
             % (mt.get("n_mounds", 0), mt.get("mound_ring_radius_m", 0),
                mt.get("plaza_radius_m", 0), mt.get("x", 0), mt.get("y", 0)))
    L.append("")
    L.append("Detector found %d individual mounds and %d clusters (>=5)." %
             (mounds["n_mounds"], len(mounds["clusters"])))
    L.append("")
    if mounds["clusters"]:
        L.append("| n mounds | ring R (m) | rms/R | coverage | plaza clear | h median (m) | flags |")
        L.append("|---|---|---|---|---|---|---|")
        for c in mounds["clusters"][:10]:
            L.append("| %d | %.0f | %.2f | %.2f | %s | %.2f | %s |"
                     % (c["n_mounds"], c["ring_radius_m"], c["ring_rms_frac"],
                        c["angular_coverage"], c["plaza_clear"], c["height_median_m"],
                        ", ".join(c["flags"])))
    L.append("")
    L.append("## Linear features (EXPERIMENTAL)")
    L.append("")
    L.append("Truth: two sunken roads (4 m, 0.7 m) and one embanked road (3 m, +0.5 m), "
             "each 300-740 m long. Detector status: NOT VALIDATED; it returned %d lines here."
             % len(lines))
    for q in lines[:6]:
        ok, reason = linedet.line_flags_for_screening(q)
        L.append("")
        L.append("* %s at (%.0f,%.0f) az %.0f deg, L %.0f m, w %.0f m, dip %.2f m, "
                 "screened=%s (%s)" % (q["kind"], q["row"], q["col"], q["azimuth_deg"],
                                       q["length_m"], q["width_m"], q["dip_m"], ok, reason))
    L.append("")
    with open(os.path.join(args.out, "report_endtoend.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(args.out, "endtoend.json"), "w") as f:
        json.dump(dict(rings=rows, n_ring_dets=len(dets), mounds=mounds["n_mounds"],
                       clusters=[{k: v for k, v in c.items() if k != "mounds"}
                                 for c in mounds["clusters"]],
                       n_lines=len(lines), inferable_km2=area_km2), f, indent=1, default=str)
    if args.evidence:
        for k, d in enumerate(dets):
            ok, reason = scoring.screened(d)
            if ok:
                diagnose.write_evidence(os.path.join(args.out, "evidence"), prep, d,
                                        "%s-R%02d" % (args.transect, k + 1))
    print("\n".join(L))


def _f(v, n=2):
    try:
        if v is None:
            return "-"
        v = float(v)
        return "-" if not np.isfinite(v) else ("%%.%df" % n) % v
    except Exception:
        return "-"


if __name__ == "__main__":
    main()
