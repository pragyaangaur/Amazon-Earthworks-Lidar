"""run_real.py -- run the detectors on a real processed transect (read-only on the
processor's products) and write detector-local outputs.

Designed for m2's products of transect 0414:
  .ultramath/smwduocxj/m2/out/0414/{rasters.npz,masks.npy,summary.json}

Outputs (all in the detector's own directory, never next to the source):
  out/real/<transect>/detections.json      every detection, all diagnostics
  out/real/<transect>/detections.csv       candidates.csv-shaped rows (private)
  out/real/<transect>/null_scan.json       null distribution of the dip statistic
  out/real/<transect>/report.md            what was found, FP rate, limitations
  out/real/<transect>/evidence/*.png       panels for screened candidates only

The processor's DTM is used as-is: this script never writes to or modifies the
source directory.  Positions are written only to the private CSV/JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose
import linedet
import mounddet
import preprocess
import ringdet
import scoring


def load_products(npz_path, masks_path, summary_path, transect, row_order="south-up"):
    z = np.load(npz_path, allow_pickle=False)
    dtm = np.asarray(z["dtm"], dtype=np.float32)
    chm = np.asarray(z["chm"], dtype=np.float32) if "chm" in z.files else None
    hole = np.asarray(z["hole"]).astype(bool) if "hole" in z.files else None
    summ = json.load(open(summary_path)) if summary_path and os.path.exists(summary_path) else {}
    g = summ.get("grid", {})
    out = dict(dtm=dtm, canopy=chm, hole=hole, cell=float(g.get("cell", 3.0)),
               x0=float(g.get("x0", 0.0)), y0=float(g.get("y0", 0.0)),
               epsg=str(summ.get("epsg", "")), summary=summ, transect=transect,
               row_order=row_order)
    if masks_path and os.path.exists(masks_path):
        m = np.load(masks_path, allow_pickle=True)
        if isinstance(m, np.ndarray) and m.dtype == object and m.shape == ():
            m = m.item()                      # processor saved a pickled dict
        if isinstance(m, dict):
            out["masks_keys"] = sorted(m.keys())
            get = m.get
        else:
            out["masks_keys"] = list(m.files)
            get = lambda k: (m[k] if k in m.files else None)
        if get("detect_mask") is not None:
            out["detect_mask"] = np.asarray(get("detect_mask")).astype(bool)
        if get("canopy_mask") is not None:
            out["canopy_mask"] = np.asarray(get("canopy_mask")).astype(bool)
        if get("pre_clamp") is not None:
            out["pre_clamp"] = np.asarray(get("pre_clamp"), dtype=np.float32)
    return out


def build_prep(prod, cfg=None):
    dtm = prod["dtm"]
    valid = np.isfinite(dtm)
    prep = preprocess.prepare(dtm, valid, cell=prod["cell"], canopy=prod.get("canopy"),
                              cfg=cfg)
    det_mask = prod.get("detect_mask")
    if det_mask is not None:
        if det_mask.shape == dtm.shape:
            prep["infer"] = prep["infer"] & det_mask
            prep["product_detect_mask"] = det_mask
        else:
            print("WARNING: detect_mask shape %s != dtm %s; ignoring"
                  % (det_mask.shape, dtm.shape))
    prep["x0"], prep["y0"] = prod["x0"], prod["y0"]
    prep["epsg"] = prod["epsg"]
    prep["hillshade_dtm"] = preprocess.hillshade(dtm, prod["cell"], valid)
    return prep


def null_scan(prep, radii_m=(15, 20, 25, 30, 40, 50, 60, 80, 100), of_m=12.0,
              stride=4, min_frac=0.9, chunk=1024):
    """Distribution of the radial-dip statistic at fixed radii over all centres.

    This is the direct analogue of the processor's spike-test null, computed by an
    independent implementation, and is the number a promoter must beat.
    """
    cell = float(prep["cell"])
    hz, wgt = ringdet._fields(prep, dict(input_mode="lrm15"))
    centers, _ = ringdet.candidate_centers(prep, stride)
    of = int(round(of_m / cell))
    res = {}
    for r_m in radii_m:
        r = r_m / cell
        ladder = np.array([max(1.0, r - of), r, r + of])
        vals = []
        for s0 in range(0, len(centers), chunk):
            ch = centers[s0:s0 + chunk]
            cube, cw = ringdet.sample_polar(hz, wgt, ch, ladder, 48)
            fin = cw > 0.5
            M = np.nanmean(np.where(fin, cube, np.nan), axis=1)      # (nc,3)
            frac = fin.mean(axis=1)
            ok = (frac[:, 0] > min_frac) & (frac[:, 1] > min_frac) & (frac[:, 2] > min_frac)
            dip = 0.5 * (M[:, 0] + M[:, 2]) - M[:, 1]
            vals.append(dip[ok])
        v = np.concatenate(vals) if vals else np.array([])
        v = v[np.isfinite(v)]
        if v.size < 20:
            res["%gm" % r_m] = dict(n=int(v.size))
            continue
        pc = np.percentile(v, [0.1, 1, 5, 25, 50, 75, 95, 99])
        res["%gm" % r_m] = dict(n=int(v.size), p0_1=float(pc[0]), p1=float(pc[1]),
                                p5=float(pc[2]), p25=float(pc[3]), p50=float(pc[4]),
                                p75=float(pc[5]), p95=float(pc[6]), p99=float(pc[7]),
                                frac_below_0_5=float(np.mean(v < -0.5)),
                                frac_below_1_0=float(np.mean(v < -1.0)),
                                frac_below_1_5=float(np.mean(v < -1.5)))
    return dict(radii=res, n_centers=int(len(centers)), stride_cells=step_stride(stride),
                of_m=of_m, input_mode="lrm15")


def step_stride(s):
    return int(s)


def det_to_row(det, ident, transect, prod, certainty, reason):
    """Map cell -> native EPSG coordinates.

    IMPORTANT: the processor's grid (m2/raster.py) stores y increasing with row
    index (`y = y0 + (row + 0.5) * cell`, y0 = SOUTH edge).  That is the opposite
    of the usual north-up GeoTIFF convention, so the row order is a parameter
    (`prod["row_order"]`) and is written into every output.  Getting this wrong
    would silently mirror every candidate's northing.
    """
    cell = float(prod["cell"])
    x = prod["x0"] + (det["col"] + 0.5) * cell
    if prod.get("row_order", "south-up") == "south-up":
        y = prod["y0"] + (det["row"] + 0.5) * cell
    else:
        y = prod["y0"] - (det["row"] + 0.5) * cell
    return dict(id=ident, state="", transect=transect, EPSG=prod["epsg"], x=round(x, 1),
                y=round(y, 1), lon="", lat="", type=det.get("shape", "ring"),
                size_m=round(det.get("diameter_m", np.nan), 1),
                relief_m=round(det.get("dip_m", np.nan), 2),
                canopy_m=(round(det["ctx_canopy_ann"], 1)
                          if np.isfinite(det.get("ctx_canopy_ann", np.nan)) else ""),
                nearest_known_db="", nearest_known_m="", literature_checked="no",
                certainty=certainty, reason=reason)


CSV_COLS = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type", "size_m",
            "relief_m", "canopy_m", "nearest_known_db", "nearest_known_m",
            "literature_checked", "certainty", "reason"]


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write(",".join(CSV_COLS) + "\n")
        for r in rows:
            vals = []
            for c in CSV_COLS:
                v = r.get(c, "")
                if isinstance(v, str) and ("," in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                vals.append(str(v))
            f.write(",".join(vals) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transect", default="0414")
    ap.add_argument("--npz", default="../m2/out/0414_v3/rasters.npz")
    ap.add_argument("--masks", default="../m2/out/0414_v3/masks.npy")
    ap.add_argument("--summary", default="../m2/out/0414_v3/summary.json")
    ap.add_argument("--out", default="out/real/0414")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--dipmin", type=float, default=0.30)
    ap.add_argument("--max-dets", type=int, default=200)
    ap.add_argument("--evidence-top", type=int, default=12)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--row-order", default="south-up", choices=["south-up", "north-up"])
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    prod = load_products(args.npz, args.masks, args.summary, args.transect,
                         row_order=args.row_order)
    dtm = prod["dtm"]
    print("loaded %s: dtm %s, valid %.3f, epsg %s, x0/y0 %.0f/%.0f"
          % (args.transect, dtm.shape, float(np.isfinite(dtm).mean()), prod["epsg"],
             prod["x0"], prod["y0"]), flush=True)
    prep = build_prep(prod)
    usable = int(prep["infer"].sum())
    area_km2 = usable * prod["cell"] ** 2 / 1e6
    print("inferable cells %d = %.3f km2" % (usable, area_km2), flush=True)

    params = dict(min_dip_m=args.dipmin, stride_cells=args.stride, max_dets=args.max_dets,
                  input_mode="lrm15")
    t1 = time.time()
    dets = ringdet.detect_rings(prep, params, verbose=True)
    t2 = time.time()
    print("ring detector: %d raw detections in %.1fs" % (len(dets), t2 - t1), flush=True)

    # Mound clusters use the pre-clamp surface when the processor provides it:
    # the anti-canopy clamp intentionally removes some positive 0.5-3 m mounds.
    # Ring detection remains on the clamped DTM. This adapter was added by the
    # coordinator after checking m2's synthetic mound-recovery test.
    mound_prep = prep
    if prod.get("pre_clamp") is not None and prod["pre_clamp"].shape == dtm.shape:
        mound_product = dict(prod)
        mound_product["dtm"] = prod["pre_clamp"]
        mound_prep = build_prep(mound_product)
        print("mound detector: using pre_clamp surface", flush=True)
    mound = mounddet.detect_mounds(mound_prep)
    mound_rows = []
    for k, c in enumerate(mound["clusters"]):
        ident = "M%s-%03d" % (args.transect, k + 1)
        ok_m = len([f for f in c["flags"] if f in (
            "not_circular", "incomplete_ring", "plaza_not_clear", "ring_radius_out_of_range",
            "mounds_too_broad")]) == 0
        rec = {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
               for kk, vv in c.items() if kk != "mounds"}
        rec["mounds"] = c["mounds"]
        rec["id"] = ident
        rec["certainty"] = "Possible" if ok_m else "Rejected"
        rec["screen_reason"] = "clean" if ok_m else ";".join(c["flags"])
        rec["cell_rowcol"] = [c["row"], c["col"]]
        mound_rows.append(rec)

    rows, kept = [], []
    screened_rows = []
    for k, d in enumerate(dets):
        ident = "R%s-%03d" % (args.transect, k + 1)
        ok, reason = scoring.screened(d)
        certainty = "Possible" if ok else "Rejected"
        rows.append(det_to_row(d, ident, args.transect, prod, certainty, reason))
        rec = diagnose.profile_json(d)
        rec["id"] = ident
        rec["certainty"] = certainty
        rec["screen_reason"] = reason
        rec["cell_rowcol"] = [d["row"], d["col"]]
        kept.append(rec)
        if ok:
            screened_rows.append((ident, d, reason))

    null = None if args.no_null else null_scan(prep, stride=args.stride)
    np.save(os.path.join(args.out, "detections.npy"), np.array(kept, dtype=object),
            allow_pickle=True)
    with open(os.path.join(args.out, "detections.json"), "w") as f:
        json.dump(kept, f, indent=1, default=str)
    write_csv(os.path.join(args.out, "detections.csv"), rows)
    write_csv(os.path.join(args.out, "candidates.csv"),
              [det_to_row(d, i, args.transect, prod, "Possible", r)
               for (i, d, r) in screened_rows])
    if null:
        with open(os.path.join(args.out, "null_scan.json"), "w") as f:
            json.dump(null, f, indent=1, default=str)
    with open(os.path.join(args.out, "mound_clusters.json"), "w") as f:
        json.dump(mound_rows, f, indent=1, default=str)
    with open(os.path.join(args.out, "mounds_all.json"), "w") as f:
        json.dump(mound["mounds"], f, indent=1, default=str)
    ev_dir = os.path.join(args.out, "evidence")
    if not args.no_evidence:
        for (ident, d, reason) in sorted(screened_rows,
                                         key=lambda t: -t[1].get("snr", 0))[: args.evidence_top]:
            diagnose.write_evidence(ev_dir, prep, d, ident)
        print("wrote %d evidence panels to %s"
              % (min(len(screened_rows), args.evidence_top), ev_dir), flush=True)

    # ---- report -----------------------------------------------------------
    L = ["# Detector run on transect %s (real DTM)" % args.transect, "",
         "Source DTM: `%s` (processor product, read-only). EPSG %s, %g m cells, "
         "origin %.0f/%.0f, row order `%s` (the processor stores rows south-to-north: "
         "y = y0 + (row + 0.5) * cell)." % (os.path.basename(args.npz), prod["epsg"],
                                             prod["cell"], prod["x0"], prod["y0"],
                                             args.row_order),
         "", "Inferable area: **%.3f km2** (%d cells). Ring detector: %.1f s, "
         "**%d raw detections**, **%d screened (Possible)**, %d rejected."
         % (area_km2, usable, t2 - t1, len(dets), len(screened_rows),
            len(dets) - len(screened_rows)), ""]
    if screened_rows:
        L.append("| id | R (m) | D (m) | dip (m) | width (m) | SNR | cov | canopy (m) | reason |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for (ident, d, reason) in screened_rows[:40]:
            L.append("| %s | %.0f | %.0f | %.2f | %s | %.1f | %.2f | %s | %s |"
                     % (ident, d["radius_m"], d["diameter_m"], d["dip_m"],
                        _f(d.get("width_m"), 0), d.get("snr", np.nan), d["cov"],
                        _f(d.get("ctx_canopy_ann"), 1), reason))
    L.append("")
    L.append("## Mound clusters (>= 5 bumps of 0.55-4 m, ring fit)")
    L.append("")
    nm = len(mound["clusters"])
    nm_ok = len([r for r in mound_rows if r["certainty"] == "Possible"])
    L.append("Individual mounds found: %d; clusters with >= 5 mounds: %d; clusters that pass "
             "the ring test (circular, complete, plaza clear, radius in range): **%d**."
             % (mound["n_mounds"], nm, nm_ok))
    L.append("")
    if mound_rows:
        L.append("| id | n mounds | ring R (m) | rms/R | angular coverage | plaza clear | median mound height (m) | certainty | reason |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in mound_rows[:30]:
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |"
                     % (r["id"], r["n_mounds"], _f(r["ring_radius_m"], 0),
                        _f(r["ring_rms_frac"], 2), _f(r["angular_coverage"], 2),
                        r["plaza_clear"], _f(r["height_median_m"], 2), r["certainty"],
                        r["screen_reason"]))
    L.append("")
    L.append("## Linear features (EXPERIMENTAL, not used for candidates)")
    L.append("")
    L.append("`linedet.py` is not validated (see its docstring): at the operating point that "
             "gives zero false lines in featureless scenes it misses a planted 300 m road, and at "
             "a looser point it invents 1 km lines in null terrain. It is therefore **not run** "
             "in this screening; long linear features in this transect remain unscreened and are "
             "listed as a gap, not as a negative result.")
    L.append("")
    L.append("## Null distribution of the dip statistic in this transect")
    L.append("")
    L.append("Independent implementation of the processor's spike-test null: dip at "
             "fixed radius over every inferable centre on a 12 m grid, LRM-15 input, "
             "flanks at +-12 m. A candidate must beat this, not just beat zero.")
    L.append("")
    if null:
        L.append("| radius | n | p0.1 | p1 | p5 | p50 | % < -0.5 m | % < -1.0 m | % < -1.5 m |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for k, v in null["radii"].items():
            if v.get("n", 0) < 20:
                continue
            L.append("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |"
                     % (k, v["n"], v["p0_1"], v["p1"], v["p5"], v["p50"],
                        100 * v["frac_below_0_5"], 100 * v["frac_below_1_0"],
                        100 * v["frac_below_1_5"]))
    L.append("")
    L.append("## Honest limitations of this run")
    L.append("")
    L.append("* A raw detection is not a candidate. Only `Possible` rows are offered for "
             "skeptic review; `Strong`/`Known` need the coordinator's record and "
             "literature checks.")
    L.append("* The DTM is a lower bound on ditch depth (DTM relief attenuates a narrow "
             "ditch; the processor measured 8-16% attenuation on spike-ins).")
    L.append("* Nothing under canopy is claimed: the processor's `detect_mask` is applied, "
             "so forest cells are excluded rather than guessed at.")
    L.append("* An empty result is a result: it says no *screenable* ring was found in the "
             "inferable area, at the stated thresholds.")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print("total %.1fs; wrote %s" % (time.time() - t0, args.out))


def _f(v, n=1):
    try:
        v = float(v)
        return "-" if not np.isfinite(v) else ("%%.%df" % n) % v
    except Exception:
        return "-"


if __name__ == "__main__":
    main()
