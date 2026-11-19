#!/usr/bin/env python3
"""Score the processor against the synthetic control with known ground truth.

Every check is numerically defined and stored in validation.json, so a skeptic
can rerun it.  The synthetic scene contains an occluding canopy, a large
closed-canopy patch (the honest hard case), swath gaps and one of each target
feature type.

Usage:  python3 validate_synth.py synth_out [--out=out/synth]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import lasio
import raster
import synth as synth_mod


def robust_sigma(a):
    a = np.asarray(a)[np.isfinite(a)]
    if a.size < 10:
        return float("nan")
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def radial_profile(dtm, grid, cx, cy, r_max, dr=1.0):
    X, Y = np.meshgrid(grid.centers_x(), grid.centers_y())
    R = np.hypot(X - cx, Y - cy)
    edges = np.arange(0, r_max + dr, dr)
    med = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        m = (R >= edges[i]) & (R < edges[i + 1]) & np.isfinite(dtm)
        if m.sum() >= 3:
            med[i] = float(np.median(dtm[m]))
    return edges, med


def true_surface(X, Y, truth):
    """Full synthetic ground surface: regional trend + every carved feature."""
    x0, y0 = truth["utm"]
    z = synth_mod._ground(X, Y, x0, y0)
    for f in truth["features"]:
        if f["type"] == "ring_ditch":
            z = z + synth_mod._ring_ditch(X, Y, f["x"], f["y"], f["radius_m"],
                                          f["width_m"] / 2, f["depth_m"],
                                          f.get("bank_m", 0.0))
        elif f["type"] == "sunken_road":
            z = z + synth_mod._trench_line(X, Y, f["p0"], f["p1"],
                                           f["width_m"] / 2, f["depth_m"])
        elif f["type"] == "embanked_road":
            z = z + synth_mod._trench_line(X, Y, f["p0"], f["p1"],
                                           f["width_m"] / 2, -f["height_m"])
    for n in truth["negatives"]:
        if n["type"] == "meandering_river":
            z = z + synth_mod._meander(X, Y, n["y_axis"], n["amp_m"], n["wl_m"],
                                       n["width_m"] / 2, n["depth_m"])
        elif n["type"] == "cattle_pond":
            rr = np.hypot(X - n["x"], Y - n["y"])
            z = z + np.where(rr < n["radius_m"],
                             -n["depth_m"] * (1 - (rr / n["radius_m"]) ** 2), 0.0)
    return z


def main(argv):
    sdir = argv[1] if len(argv) > 1 else "synth_out"
    out = "out/synth"
    for a in argv[2:]:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
    truth = json.load(open(os.path.join(sdir, "synth_truth.json")))
    R = np.load(os.path.join(out, "rasters.npz"))
    dtm, chm, hs, svf = R["dtm"], R["chm"], R["hillshade"], R["svf"]
    masks = np.load(os.path.join(out, "masks.npy"), allow_pickle=True).item()
    summ = json.load(open(os.path.join(out, "summary.json")))
    grid = raster.Grid.from_dict(summ["grid"])
    feats = {f["id"]: f for f in truth["features"]}
    X, Y = np.meshgrid(grid.centers_x(), grid.centers_y())
    zg = true_surface(X, Y, truth)
    err = dtm - zg
    fin = np.isfinite(dtm)
    results = {}

    def rec(name, value, ok, detail=""):
        results[name] = dict(value=value, pass_=bool(ok), detail=detail)
        print("%-18s %-9s %-9s %s" % (name, "PASS" if ok else "FAIL",
                                      _fmt(value), detail))

    # 1. ring amplitude ------------------------------------------------
    ring = feats["S-RING-001"]
    edges, med = radial_profile(dtm, grid, ring["x"], ring["y"], 45.0)
    r = 0.5 * (edges[:-1] + edges[1:])
    refm = (r >= 26) & (r <= 34)
    ref = np.nanmedian(med[refm])
    ditch = np.nanmedian(med[(r >= 18) & (r <= 22)])
    dip = float(ditch - ref)
    # noise from CELL values (profile medians are over ~900 cells, too smooth)
    rr = np.hypot(X - ring["x"], Y - ring["y"])
    noise = robust_sigma(err[(rr > 26) & (rr < 34) & fin])
    snr = float(abs(dip) / noise) if noise > 0 else float("inf")
    rec("ring_amplitude", round(dip, 2), dip < -0.9,
        "ditch %.2f m vs outboard %.2f m; truth -1.5 m paraboloid, mean -1.0 m"
        % (ditch, ref))
    rec("ring_snr", round(snr, 1), snr > 5.0,
        "cell-level robust noise in reference annulus %.3f m" % noise)

    # 2. roads ---------------------------------------------------------
    road = feats["S-ROAD-002"]
    d_axis = _axis_distance(X, Y, road["p0"], road["p1"])
    sel = (X > truth["utm"][0] + 200) & (X < truth["utm"][0] + 700)
    onv = np.nanmedian(dtm[(d_axis < 1.2) & sel & fin])
    offv = np.nanmedian(dtm[(d_axis > 8) & (d_axis < 16) & sel & fin])
    rdip = float(onv - offv)
    rec("road_amplitude", round(rdip, 2), rdip < -0.4,
        "on-axis %.2f m vs 8-16 m off-axis %.2f m; truth -0.7 m" % (onv, offv))

    # 3. canopy: emergent blobs must be removed, big forest is the limit --
    h = lasio.read_header(os.path.join(sdir, "synth.las"))
    acc = raster.accumulate(os.path.join(sdir, "synth.las"), grid, block=500000)
    naive = acc.zmin_grid().astype(np.float64) - zg
    can = np.zeros(dtm.shape, bool)
    for b in truth["canopy_blobs"]:
        can |= np.hypot(X - b["x"], Y - b["y"]) < 0.6 * b["radius_m"]
    can &= fin
    bf = truth["big_forest"]
    forest = np.hypot(X - bf["x"], Y - bf["y"]) < 0.6 * bf["radius_m"]
    forest &= fin
    can_bias = float(np.median(err[can])) if can.any() else float("nan")
    naive_bias = float(np.median(naive[can])) if can.any() else float("nan")
    forest_bias = float(np.median(err[forest])) if forest.any() else float("nan")
    rec("canopy_blob_bias", round(can_bias, 2),
        abs(can_bias) < 0.6 and abs(naive_bias) > 1.0,
        "DTM error under emergent blobs %.2f m vs naive lowest-return %.2f m"
        % (can_bias, naive_bias))
    results["forest_bias_limitation"] = dict(
        value=round(forest_bias, 2),
        detail="DTM error under the 150 m closed-canopy patch. Read it carefully: the "
               "opening cascade penetrates about the sum of its windows (~70 m) inwards "
               "from the canopy edge, so the surface under a patch narrower than that is "
               "extrapolated from the surrounding ground, not measured. It is close to "
               "truth here (gentle terrain) but that is not evidence of accuracy in real "
               "forest; such cells are excluded from detect_mask instead of trusted.",
        mask_pct=float(100 * masks["forest_mask"].mean()))
    print("%-18s %-9s %-9s %s" % ("forest_bias", "INFO", _fmt(forest_bias),
                                  results["forest_bias_limitation"]["detail"]))

    # 3b. enclosure preservation: the grey-opening clamp must not excavate the
    #     interior of a ring ditch.  Regression for the defect where a 40 m
    #     window pulled a 40 m enclosure interior down to ditch level.
    inter = rr < 0.7 * ring["radius_m"]
    ditchm = np.abs(rr - ring["radius_m"]) < f_ditch_half_width(ring)
    outer = (rr > 26) & (rr < 34)
    d_ie = float(np.nanmedian(dtm[inter]) - np.nanmedian(dtm[outer]))
    t_ie = float(np.nanmedian(zg[inter]) - np.nanmedian(zg[outer]))
    rec("enclosure_interior", round(d_ie - t_ie, 2), abs(d_ie - t_ie) < 0.3,
        "DTM interior-minus-exterior %.2f m vs truth %.2f m (clamp must not excavate "
        "the enclosure); clamp-rewritten: interior %.0f%%, ditch %.0f%%"
        % (d_ie, t_ie, 100 * np.nanmean(masks["removed"][inter]),
           100 * np.nanmean(masks["removed"][ditchm])))

    # 3c. mounds: positive features of the same size as canopy clumps.  The
    #     anti-canopy clamp removes them from `dtm` (by design), so mound work
    #     needs the pre-clamp surface; this check pins both behaviours.
    mounds = [f for f in truth["features"] if f["type"] == "mound"]
    if mounds:
        relpre = R["relief_pre_10"] if "relief_pre_10" in R else None
        got, kept = [], []
        for m in mounds:
            rm = np.hypot(X - m["x"], Y - m["y"])
            inner = rm < 0.5 * m["radius_m"]
            outer = (rm > 2.2 * m["radius_m"]) & (rm < 3.5 * m["radius_m"])
            rel = lambda a: float(np.nanmedian(a[inner]) - np.nanmedian(a[outer]))
            got.append((m["height_m"], rel(dtm)))
            if relpre is not None:
                kept.append((m["height_m"], rel(relpre)))
        tall = [(h, v) for h, v in got if h >= 1.5]
        tall_kept = [(h, v) for h, v in kept if h >= 1.5]
        ok_m = (not tall_kept) or all(v >= 0.5 * h for h, v in tall_kept)
        rec("mound_recovery", {("h=%.1f" % h): round(v, 2) for h, v in got},
            ok_m,
            "planted vs recovered mound height: dtm keeps %.0f-%.0f%% (>=1.5 m mounds are "
            "removed by the anti-canopy clamp, by design); relief_pre_10 keeps %s"
            % (100 * min(v / h for h, v in got), 100 * max(v / h for h, v in got),
               "%.0f-%.0f%%" % (100 * min(v / h for h, v in tall_kept),
                                100 * max(v / h for h, v in tall_kept)) if tall_kept else "n/a"))

    # 4. open-ground accuracy ------------------------------------------
    openm = fin & (np.nan_to_num(chm) < 1.0)
    rmse = float(np.sqrt(np.mean(err[openm] ** 2)))
    p95 = float(np.percentile(np.abs(err[openm]), 95))
    rec("open_ground_rmse", round(rmse, 3), rmse < 0.25,
        "open cells (CHM<1 m, n=%d): RMSE %.3f m, p95 |err| %.3f m"
        % (openm.sum(), rmse, p95))

    # 5. holes ---------------------------------------------------------
    x0 = truth["utm"][0]
    strip = (X > x0 + 305) & (X < x0 + 335)
    gap_nan = float(np.mean(~np.isfinite(dtm[strip])))
    small_ok = _test_fill_small_holes()
    rec("hole_handling", round(gap_nan, 3), gap_nan > 0.9 and small_ok,
        "swath gap left NaN %.0f%%; <=5-cell fill unit test %s"
        % (100 * gap_nan, "ok" if small_ok else "BROKEN"))

    # 6. masks ---------------------------------------------------------
    unexplained = int((~fin & ~masks["data_hole"] & ~masks["big_hole"]).sum())
    bad_detect = int((masks["detect_mask"] & ~fin).sum())
    rec("mask_consistency", dict(unexplained_nan=unexplained, detect_off_dtm=bad_detect),
        unexplained == 0 and bad_detect == 0,
        "every DTM NaN is labelled; detect_mask only on finite DTM")

    # 7. products ------------------------------------------------------
    bad_hs = int((~np.isfinite(hs) & fin).sum())
    bad_svf = int((~np.isfinite(svf) & fin).sum())
    neg_chm = int((np.nan_to_num(chm) < -1e-6).sum())
    rec("product_sanity", dict(hillshade=bad_hs, svf=bad_svf, chm_negative=neg_chm),
        bad_hs == 0 and bad_svf == 0 and neg_chm == 0,
        "hillshade/SVF defined on every DTM cell; CHM >= 0")

    # 8. scale separation ---------------------------------------------
    rel15, rel30 = R["relief_15"], R["relief_30"]
    def dipat(a):
        return float(np.nanmedian(a[(rr > 18) & (rr < 22)]) -
                     np.nanmedian(a[(rr > 26) & (rr < 34)]))
    d15, d30 = dipat(rel15), dipat(rel30)
    rec("scale_separation", dict(relief15=round(d15, 2), relief30=round(d30, 2)),
        d15 < -0.5 and d30 < -0.3, "ring dip survives the relief products")

    npass = sum(1 for v in results.values() if v.get("pass_"))
    ncheck = sum(1 for v in results.values() if "pass_" in v)
    summary = dict(n_checks=ncheck, n_pass=npass, all_pass=npass == ncheck,
                   results=results, timings=summ.get("timings"),
                   empty_dtm_pct=summ.get("empty_dtm_pct"),
                   synth=dict(n_points=truth["n_points"], seed=truth["seed"]))
    with open(os.path.join(out, "validation.json"), "w") as f:
        json.dump(summary, f, indent=1)
    _plot(out, r, med, ref, dtm, rel15, grid, ring)
    print("\n%d/%d checks pass (validated on %d synthetic points)"
          % (npass, ncheck, truth["n_points"]))
    return 0 if npass == ncheck else 1


def f_ditch_half_width(ring):
    return ring.get("width_m", 6.0) / 2.0


def _axis_distance(X, Y, p0, p1):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / L2, 0, 1)
    return np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1]))


def _test_fill_small_holes():
    a = np.arange(400, dtype=float).reshape(20, 20)
    a[5, 5] = np.nan                       # isolated -> filled
    a[8, 8:10] = np.nan                    # 2 cells -> filled
    a[12:16, 12:17] = np.nan               # 20 cells -> must stay NaN
    out, big, fm = raster.fill_small_holes(a, np.isnan(a), max_cells=5)
    return (np.isfinite(out[5, 5]) and np.isfinite(out[8, 8]) and np.isfinite(out[8, 9])
            and np.isnan(out[12:16, 12:17]).all() and big[12, 12] and fm[5, 5])


def _fmt(v):
    if isinstance(v, float):
        return "%.3f" % v
    return str(v)


def _plot(out, r, med, ref, dtm, rel15, grid, ring):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].plot(r, med, "k.-", ms=3)
        ax[0].axhline(ref, color="b", lw=0.8, ls="--")
        ax[0].set_xlabel("radius from ring centre (m)")
        ax[0].set_ylabel("median DTM (m)")
        ax[0].set_title("40 m ring control: radial profile")
        ax[0].grid(alpha=0.3)
        i0 = int((ring["x"] - grid.x0) / grid.cell)
        j0 = int((ring["y"] - grid.y0) / grid.cell)
        sl = (slice(max(0, j0 - 45), j0 + 45), slice(max(0, i0 - 45), i0 + 45))
        v = dtm[sl]
        lo, hi = np.nanpercentile(v, [2, 98])
        ax[1].imshow(v, cmap="gray", vmin=lo, vmax=hi)
        ax[1].set_title("DTM crop 270 m"); ax[1].set_xticks([]); ax[1].set_yticks([])
        w = rel15[sl]
        lo2, hi2 = np.nanpercentile(w, [2, 98])
        ax[2].imshow(w, cmap="seismic", vmin=lo2, vmax=hi2)
        ax[2].set_title("15 m relief crop"); ax[2].set_xticks([]); ax[2].set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out, "validation.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
