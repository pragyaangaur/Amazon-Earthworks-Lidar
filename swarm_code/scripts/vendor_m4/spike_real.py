"""spike_real.py -- digital spike-in test of the ring detector on the REAL 0414 DTM.

Why this is the decisive test now: transect 0414 is the project's positive
control, but the published geoglyph coordinate shows no ring in the DTM (both
processors agree, independently).  A pipeline cannot be trusted until it is shown
to recover a known ditch *from this real DTM*.  So we plant parametric ring
ditches (and one square, one U) into copies of the real DTM at real open-pasture
locations, run the full detector, and report recovery and depth retention.

This is a *digital* spike-in: it tests the detector and the terrain/clutter
competition, NOT the point-cloud sampling chain (that is m2's spike_test.py, which
plants rings in the raw returns).  Both are needed; neither alone proves the
physical detectability of a real ditch.

Outputs go under out/real/<transect>_spike/ only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose
import preprocess
import ringdet
import scoring
import synth
import run_real


def pick_frac_map(prep, radius_cells):
    """Fraction of inferable cells in a disc of radius_cells around every cell."""
    from scipy import ndimage as ndi
    infer = prep["infer"].astype(np.float32)
    k = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), np.float32)
    yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    k[np.hypot(yy, xx) <= radius_cells] = 1.0
    num = ndi.convolve(infer, k, mode="constant", cval=0.0)
    den = k.sum()
    return num / den


def pick_site(prep, radius_m, cell, min_frac=0.98, seed=0, avoid=(), edge_cells=0):
    """Pick an open-pasture centre whose disc of radius_m (plus margin) is valid."""
    frac = pick_frac_map(prep, int(np.ceil((radius_m + 20.0) / cell)))
    infer = prep["infer"]
    nx, ny = infer.shape
    rng = np.random.default_rng(seed)
    ok = frac >= min_frac
    ok[: int(2 * radius_m / cell) + 2, :] = False
    ok[-(int(2 * radius_m / cell) + 2):, :] = False
    ok[:, : int(2 * radius_m / cell) + 2] = False
    ok[:, -(int(2 * radius_m / cell) + 2):] = False
    ys, xs = np.nonzero(ok)
    if len(ys) == 0:
        return None
    order = rng.permutation(len(ys))
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any(np.hypot(y - a[0], x - a[1]) * cell < 400.0 for a in avoid):
            continue
        return (y, x)
    return None


def build_window(prod, prep, row, col, half, cell=3.0):
    nx, ny = prod["dtm"].shape
    r0, r1 = max(0, row - half), min(nx, row + half)
    c0, c1 = max(0, col - half), min(ny, col + half)
    dtm = np.array(prod["dtm"][r0:r1, c0:c1], dtype=np.float32)
    chm = (np.array(prod["canopy"][r0:r1, c0:c1], dtype=np.float32)
           if prod.get("canopy") is not None else None)
    det = (prod.get("detect_mask")[r0:r1, c0:c1] if prod.get("detect_mask") is not None else None)
    return dtm, chm, det, (r0, c0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transect", default="0414")
    ap.add_argument("--npz", default="../m2/out/0414_v3/rasters.npz")
    ap.add_argument("--masks", default="../m2/out/0414_v3/masks.npy")
    ap.add_argument("--summary", default="../m2/out/0414_v3/summary.json")
    ap.add_argument("--out", default="out/real/0414_spike")
    ap.add_argument("--n-sites", type=int, default=6)
    ap.add_argument("--half", type=int, default=280)
    ap.add_argument("--dipmin", type=float, default=0.30)
    ap.add_argument("--no-evidence", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    prod = run_real.load_products(args.npz, args.masks, args.summary, args.transect)
    prep = run_real.build_prep(prod)
    cell = float(prod["cell"])
    # avoid the processor's published-control pit (known ~50 m depression)
    avoid = []
    if prod.get("summary"):
        pass
    kinds = [("ring", 20.0, 6.0, 1.2), ("ring", 25.0, 8.0, 1.0), ("ring", 35.0, 9.0, 0.8),
             ("ring", 60.0, 10.0, 1.5), ("square", 30.0, 8.0, 1.2), ("U", 40.0, 7.0, 1.0)]
    kinds = kinds[: args.n_sites]
    sites = []
    for j, (kind, R, w, dep) in enumerate(kinds):
        s = pick_site(prep, R, cell, seed=11 + j, avoid=sites)
        sites.append(s if s is not None else None)
    print("sites:", sites, flush=True)
    results = []
    for k, ((row, col), (kind, R, w, dep)) in enumerate(zip(sites, kinds)):
        if row is None:
            results.append(dict(site=k, kind=kind, R_m=R, depth_m=dep, found=False,
                                screened=False, reason="no valid open-pasture site",
                                n_dets=0))
            continue
        dtm, chm, det_mask, (r0, c0) = build_window(prod, prep, row, col, args.half, cell)
        yy, xx = np.mgrid[0:dtm.shape[0], 0:dtm.shape[1]]
        X = (xx - (col - c0)) * cell
        Y = (yy - (row - r0)) * cell
        g = np.zeros_like(dtm, dtype=np.float64)
        if kind == "ring":
            synth.add_ring(g, X, Y, 0.0, 0.0, R, w, dep, bank_h=0.30)
        elif kind == "square":
            synth.add_rect(g, X, Y, 0.0, 0.0, R, R, w, dep, bank_h=0.30)
        else:
            synth.add_ring(g, X, Y, 0.0, 0.0, R, w, dep, bank_h=0.30, phi0=20.0, phi1=300.0)
        spiked = np.where(np.isfinite(dtm), dtm + g.astype(np.float32), np.nan).astype(np.float32)
        sp = preprocess.prepare(spiked, np.isfinite(spiked), cell=cell, canopy=chm)
        if det_mask is not None and det_mask.shape == spiked.shape:
            sp["infer"] = sp["infer"] & det_mask
        sp["hillshade_dtm"] = preprocess.hillshade(spiked, cell, np.isfinite(spiked))
        t0 = __import__("time").time()
        dets = ringdet.detect_rings(sp, dict(min_dip_m=args.dipmin, input_mode="lrm15"))
        dt = __import__("time").time() - t0
        # is there a detection at the planted feature?
        best, bd = None, 1e9
        for d in dets:
            dist = np.hypot(d["row"] - (row - r0), d["col"] - (col - c0)) * cell
            if dist < max(20.0, 0.35 * R) and (0.6 * R <= d["radius_m"] <= 1.5 * R):
                if dist < bd:
                    best, bd = d, dist
        ok = best is not None
        scr = scoring.screened(best) if ok else (False, "missed")
        rec = dict(site=k, kind=kind, R_m=R, width_m=w, depth_m=dep,
                   found=bool(ok), dist_m=float(bd) if ok else None,
                   detected_r_m=float(best["radius_m"]) if ok else None,
                   detected_dip_m=float(best["dip_m"]) if ok else None,
                   retention=(float(best["dip_m"] / dep) if ok else None),
                   width_meas_m=float(best["width_m"]) if ok else None,
                   snr=float(best.get("snr", np.nan)) if ok else None,
                   cov=float(best["cov"]) if ok else None,
                   rel_std=float(best["rel_std"]) if ok else None,
                   shape=best.get("shape") if ok else None,
                   interior_m=float(best.get("interior_m", np.nan)) if ok else None,
                   screened=bool(scr[0]), reason=scr[1],
                   n_dets=int(len(dets)), detect_s=round(dt, 2),
                   row=int(row), col=int(col))
        results.append(rec)
        print("%-6s R=%4.0f d=%.1f -> found=%s r=%.0f dip=%.2f ret=%.2f scr=%s %s (%d dets, %.1fs)"
              % (kind, R, dep, ok, rec["detected_r_m"] or -1, rec["detected_dip_m"] or -1,
                 rec["retention"] or -1, scr[0], scr[1], len(dets), dt), flush=True)
        if ok and scr[0] and not args.no_evidence:
            diagnose.write_evidence(os.path.join(args.out, "evidence"), sp, best,
                                    "S%s-%02d-%s" % (args.transect, k + 1, kind))
    with open(os.path.join(args.out, "spike_results.json"), "w") as f:
        json.dump(dict(transect=args.transect, sites=sites, results=results,
                       note="digital spike-in into the real DTM: tests detector + terrain "
                            "competition, not the point-cloud sampling chain"), f, indent=1)
    n = len(results)
    ok = sum(1 for r in results if r["found"])
    scr = sum(1 for r in results if r["screened"])
    ret = [r["retention"] for r in results if r["retention"]]
    L = ["# Ring detector spike-in test on the real %s DTM" % args.transect, "",
         "Rings/squares/U planted into copies of the real processor DTM at real open-pasture "
         "locations (digital spike-in). It tests the detector and terrain competition; the "
         "point-cloud sampling chain is tested separately by the processor.", "",
         "Recovered: **%d/%d** planted features; screened as `Possible`: **%d/%d**; "
         "median depth retention %.2f." % (ok, n, scr, n,
                                           float(np.median(ret)) if ret else float("nan")), "",
         "| site | kind | R true (m) | depth true (m) | found | R meas (m) | dip meas (m) | retention | width meas (m) | SNR | cov | rel.std | interior (m) | screened | reason | other dets |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append("| %d | %s | %.0f | %.1f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |"
                 % (r["site"] + 1, r["kind"], r["R_m"], r["depth_m"],
                    "YES" if r["found"] else "no", _f(r["detected_r_m"], 0),
                    _f(r["detected_dip_m"], 2), _f(r["retention"], 2),
                    _f(r["width_meas_m"], 0), _f(r["snr"], 1), _f(r["cov"], 2),
                    _f(r["rel_std"], 2), _f(r["interior_m"], 2),
                    "PASS" if r["screened"] else "reject", r["reason"], r["n_dets"]))
    with open(os.path.join(args.out, "report_spike.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def _f(v, n=2):
    try:
        v = float(v)
        return "-" if not np.isfinite(v) else ("%%.%df" % n) % v
    except Exception:
        return "-"


if __name__ == "__main__":
    main()
