"""Acceptance metrics for synthetic EBA lidar controls (m1).

For a *known* synthetic feature and a *given* product (DTM or relief model):

  * dip depth on the feature's own band (median over azimuth or along-line),
    after removing a local quadratic regional trend fitted on the surround;
  * valid fraction on that band (holed / masked cells are the honest failure
    mode, so they are reported, not hidden);
  * azimuthal consistency: on what fraction of azimuths is the dip >= 0.25 m;
  * bank height just outside the ditch;
  * contrast-to-noise: dip / robust sigma of the surround;
  * for roads: cross-profile dip and the fraction of 20 along-line sections
    that still show it;
  * for mound rings: how many of the k mounds are still >= 0.4 m high;
  * a matched-ring *noise floor*: the same statistic evaluated on a centre /
    radius grid where the scene has no target (natural negatives), giving the
    p99 / max that a detector threshold has to beat.

The verdicts are benchmark words ('clear', 'partial', 'absent', 'masked').
They say nothing about real EBA data and are not certainty grades.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIP_FRACTION = 0.25      # an azimuth/section "shows the dip" above this (m)


# --------------------------------------------------------------- sampling ---
def polar(dem, g, x0, y0, cx, cy, rmax, dr=1.0, nth=72):
    """Polar sampling about (cx, cy): returns r, P[nth, nr], X, Y."""
    r = np.arange(0.0, rmax + 0.5 * dr, dr)
    th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    R, TH = np.meshgrid(r, th, indexing="xy")
    X = cx + R * np.cos(TH)
    Y = cy + R * np.sin(TH)
    col = (X - x0) / g - 0.5
    row = (Y - y0) / g - 0.5
    P = ndimage.map_coordinates(dem, [row, col], order=1, mode="constant",
                                cval=np.nan)
    return r, P, X, Y


def grid_box(dem, g, x0, y0, cx, cy, half, step=1.0):
    """Cartesian sampling on a box +-half around (cx, cy)."""
    n = int(np.floor(2 * half / step)) + 1
    ax = cx + (np.arange(n) - n // 2) * step
    ay = cy + (np.arange(n) - n // 2) * step
    X, Y = np.meshgrid(ax, ay)
    col = (X - x0) / g - 0.5
    row = (Y - y0) / g - 0.5
    P = ndimage.map_coordinates(dem, [row, col], order=1, mode="constant",
                                cval=np.nan)
    return X, Y, P


def quad_resid(X, Y, Z, sel, valid, clip=3.0, iters=3):
    """Residual of a sigma-clipped least-squares quadratic surface.

    Clipping matters: real lidar DTMs contain isolated 10-60 m pits from
    below-ground/multipath returns, and an unclipped fit through them bends
    the regional surface enough to change a 1 m ditch measurement by 0.4 m
    (measured on s5 before this was added).
    """
    base = sel & valid & np.isfinite(Z)
    out = np.full(Z.shape, np.nan)
    if base.sum() < 12:
        return out
    m = base.copy()
    coef = None
    for _ in range(max(1, iters)):
        if m.sum() < 12:
            break
        mx, my = X[m].mean(), Y[m].mean()
        dx, dy = X[m] - mx, Y[m] - my
        A = np.column_stack([np.ones(m.sum()), dx, dy, dx * dx, dx * dy, dy * dy])
        coef, *_ = np.linalg.lstsq(A, Z[m], rcond=None)
        r_all = _surface(X, Y, mx, my, coef)
        res = Z - r_all
        sc = res[m]
        sig = 1.4826 * np.median(np.abs(sc - np.median(sc)))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_m = base & (np.abs(res - np.median(sc)) < clip * sig)
        if new_m.sum() == m.sum() or new_m.sum() < 12:
            m = new_m if new_m.sum() >= 12 else m
            break
        m = new_m
    if coef is None:
        return out
    mx, my = X[m].mean(), Y[m].mean()
    return Z - _surface(X, Y, mx, my, coef)


def _surface(X, Y, mx, my, coef):
    dx, dy = X - mx, Y - my
    return (coef[0] + coef[1] * dx + coef[2] * dy + coef[3] * dx * dx
            + coef[4] * dx * dy + coef[5] * dy * dy)


def _stats(B, band2, surr2, bank2, parts=None):
    band_vals = B[band2]
    band_vals = band_vals[np.isfinite(band_vals)]
    surr_vals = B[surr2]
    surr_vals = surr_vals[np.isfinite(surr_vals)]
    valid_frac = float(np.isfinite(B[band2]).sum() / max(1, int(band2.sum())))
    if band_vals.size < 15 or surr_vals.size < 15:
        return dict(dip_m=None, valid_frac=valid_frac, n_band=int(band_vals.size),
                    n_surr=int(surr_vals.size), err="too few valid samples")
    dip = float(-np.median(band_vals))
    sigma = float(1.4826 * np.median(np.abs(surr_vals - np.median(surr_vals))))
    rows = np.unique(np.nonzero(band2)[0])
    per = []
    for i in rows:
        v = B[i][band2[i]]
        v = v[np.isfinite(v)]
        per.append(float(-np.median(v)) if v.size >= 3 else np.nan)
    per = np.asarray(per)
    az_frac = float(np.nanmean(per > DIP_FRACTION)) if np.isfinite(per).any() else 0.0
    # robust "present on most azimuths" statistics: a true ring dips on every
    # azimuth, scroll-bar texture / a channel crossing dips on only some.
    pf = per[np.isfinite(per)]
    az25 = float(np.percentile(pf, 25)) if pf.size >= 8 else None
    az10 = float(np.percentile(pf, 10)) if pf.size >= 8 else None
    bank = None
    if bank2 is not None and bank2.any():
        bv = B[bank2]
        bv = bv[np.isfinite(bv)]
        bank = float(np.median(bv)) if bv.size >= 10 else None
    extra = {}
    if parts:
        for name, pm in parts.items():
            v = B[pm]
            v = v[np.isfinite(v)]
            extra[name + "_m"] = float(np.median(v)) if v.size >= 10 else None
    return dict(**extra, dip_m=dip, dip_sigma=sigma,
                cnr=float(dip / sigma) if sigma > 0 else None,
                az_frac_dip=az_frac, dip_az25=az25, dip_az10=az10,
                dip_by_section=[None if not np.isfinite(v) else round(v, 3) for v in per],
                bank_m=bank, valid_frac=valid_frac,
                n_band=int(band_vals.size), n_surr=int(surr_vals.size))


def ring_metrics(dem, g, x0, y0, cx, cy, r, halfw, dr=1.0, nth=72):
    """Ditch/bank statistics on the annulus |radius - r| <= halfw."""
    rmax = r + max(45.0, halfw + 40.0)
    rr, P, X, Y = polar(dem, g, x0, y0, cx, cy, rmax, dr=dr, nth=nth)
    band = np.abs(rr - r) <= halfw
    inner = rr < max(2.5, r - halfw - 4.0)
    outer = (rr > r + halfw + 4.0) & (rr < r + max(40.0, halfw + 35.0))
    surr = inner | outer
    bank = (rr > r + halfw + 1.5) & (rr < r + halfw + 10.0)
    B = quad_resid(X, Y, P, surr, np.isfinite(P))
    band2 = np.broadcast_to(band[None, :], P.shape)
    inside = rr < max(2.5, r - halfw - 4.0)
    outside = rr > r + halfw + 4.0
    parts = dict(interior=np.broadcast_to(inside[None, :], P.shape),
                 exterior=np.broadcast_to(outside[None, :], P.shape))
    out = _stats(B, band2, np.broadcast_to(surr[None, :], P.shape),
                 np.broadcast_to(bank[None, :], P.shape), parts)
    # Narrow-trough test on the raw radial profile (no detrending): a ditch is
    # below BOTH its inner and its outer flank; a platform/terrace edge or a
    # knoll bench is below one and level with the other.  This is the test that
    # kills the "raised interior + skirt" false positive.
    prof = np.nanmedian(P, axis=0)
    bin_ = (rr >= r - 1.0) & (rr <= r + 1.0)
    inw = (rr >= r - halfw - 10.0) & (rr <= r - halfw - 2.0)
    outw = (rr >= r + halfw + 2.0) & (rr <= r + halfw + 10.0)
    if bin_.any() and inw.any() and outw.any():
        bp = np.nanmedian(prof[bin_])
        lp = np.nanmedian(prof[inw])
        rp = np.nanmedian(prof[outw])
        out["trough_in_m"] = float(lp - bp) if np.isfinite(lp - bp) else None
        out["trough_out_m"] = float(rp - bp) if np.isfinite(rp - bp) else None
        if out["trough_in_m"] is not None and out["trough_out_m"] is not None:
            out["trough_m"] = float(min(out["trough_in_m"], out["trough_out_m"]))
        out["trough_note"] = "raw profile: band centre vs inner and outer flanks"
    out["rmax_sampled"] = float(rmax)
    return out


def square_metrics(dem, g, x0, y0, cx, cy, r, halfw, rot_deg=0.0, step=1.0):
    """Ditch statistics for a square ditch: band on |max(|dx|,|dy|) - r|."""
    half = r + max(45.0, halfw + 40.0)
    X, Y, P = grid_box(dem, g, x0, y0, cx, cy, half, step=step)
    a = np.radians(rot_deg)
    dx = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    dy = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
    s = np.maximum(np.abs(dx), np.abs(dy))
    t = s - r
    band2 = np.abs(t) <= halfw
    surr2 = (t < -(halfw + 4.0)) | ((t > halfw + 4.0) & (t < max(40.0, halfw + 35.0)))
    bank2 = (t > halfw + 1.5) & (t < halfw + 10.0)
    B = quad_resid(X, Y, P, surr2, np.isfinite(P))
    return _stats(B, band2, surr2, bank2)


def ring_verdict(m, depth_true=None):
    if m.get("dip_m") is None:
        return "masked"
    need = 0.5 * depth_true if depth_true else 0.45
    hard = 0.3 * depth_true if depth_true else 0.30
    if (m.get("valid_frac") or 0) < 0.35:
        return "masked"
    if m["dip_m"] >= need and (m.get("az_frac_dip") or 0) >= 0.70:
        return "clear"
    if m["dip_m"] >= hard and (m.get("az_frac_dip") or 0) >= 0.40:
        return "partial"
    return "absent"


# ------------------------------------------------------------------ roads ---
def line_metrics(dem, g, x0, y0, p0, p1, width, ds=1.5, off_max=None,
                 sign=1.0):
    """Cross-profile dip along a straight feature plus along-line continuity."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    L = float(np.hypot(*(p1 - p0)))
    u = (p1 - p0) / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    off_max = off_max or max(20.0, 3.0 * width)
    offs = np.arange(-off_max, off_max + 0.5, 1.0)
    ss = np.arange(0.0, L, ds)
    S, O = np.meshgrid(ss, offs, indexing="xy")
    xs = p0[0] + S * u[0] + O * n[0]
    ys = p0[1] + S * u[1] + O * n[1]
    P = ndimage.map_coordinates(dem, [(ys - y0) / g - 0.5, (xs - x0) / g - 0.5],
                                order=1, mode="constant", cval=np.nan)
    core = np.abs(offs) <= width / 2.0
    surr = np.abs(offs) >= max(width, width / 2.0 + 2.0)
    cv, sv = P[core][np.isfinite(P[core])], P[surr][np.isfinite(P[surr])]
    if cv.size < 10 or sv.size < 10:
        return dict(dip_m=None, valid_frac=0.0, err="too few valid samples")
    dip = sign * float(np.median(sv) - np.median(cv))
    sig = float(1.4826 * np.median(np.abs(sv - np.median(sv))))
    # bank: residual crest just outside the feature on either side
    prof = np.nanmedian(P, axis=1)
    ib = np.abs(offs)
    lo = (offs > width / 2.0 + 1.5) & (offs < width / 2.0 + 9.0)
    hi = (offs < -(width / 2.0 + 1.5)) & (offs > -(width / 2.0 + 9.0))
    base = np.nanmedian(prof[np.abs(offs) > max(12.0, width)])
    bank = float(max(np.nanmax(prof[lo]) - base if lo.any() else -9,
                     np.nanmax(prof[hi]) - base if hi.any() else -9))
    bank = bank * sign
    ns = 20
    cuts = np.linspace(0.0, L, ns + 1)
    seg = []
    for k in range(ns):
        m = (ss >= cuts[k]) & (ss < cuts[k + 1])
        c = P[np.ix_(core, m)]
        s = P[np.ix_(surr, m)]
        cvm, svm = c[np.isfinite(c)], s[np.isfinite(s)]
        seg.append(sign * float(np.median(svm) - np.median(cvm))
                   if cvm.size > 5 and svm.size > 5 else np.nan)
    seg = np.asarray(seg)
    return dict(dip_m=dip, dip_sigma=sig, bank_m=round(bank, 3),
                cnr=float(dip / sig) if sig > 0 else None,
                valid_frac=float(np.isfinite(P[core]).mean()),
                frac_length_dip=float(np.nanmean(seg > max(0.15, 0.25 * dip)))
                if np.isfinite(seg).any() else None,
                along_dip=[None if not np.isfinite(v) else round(v, 3) for v in seg])


# ------------------------------------------------------------ mound rings ---
def mound_metrics(dem, g, x0, y0, f):
    cx, cy, R, k = f["cx"], f["cy"], float(f["R"]), int(f["k"])
    hs = f.get("heights") or [1.5] * k
    rs = f.get("radii", [10.0] * k)
    ang = float(f.get("phase", 0.0)) + 2.0 * np.pi * np.arange(k) / k
    wob = float(f.get("pos_wobble", 0.0))
    found = []
    for i in range(k):
        rr = R * (1.0 + wob * np.sin(3.0 * ang[i] + 1.3 * i))
        mx, my = cx + rr * np.cos(ang[i]), cy + rr * np.sin(ang[i])
        rad = float(rs[i])
        rmax = 3.0 * rad
        rr_s, P, _X, _Y = polar(dem, g, x0, y0, mx, my, rmax, dr=0.75, nth=48)
        cen = P[:, rr_s <= rad * 0.6]
        ann = P[:, (rr_s >= 1.6 * rad) & (rr_s <= rmax)]
        cv, av = cen[np.isfinite(cen)], ann[np.isfinite(ann)]
        h = float(np.median(cv) - np.median(av)) if cv.size and av.size else np.nan
        found.append(dict(i=i, x=round(float(mx), 1), y=round(float(my), 1),
                          h_m=None if not np.isfinite(h) else round(h, 3),
                          h_true=float(hs[i]),
                          detected=bool(np.isfinite(h) and h >= 0.4)))
    hh = [m["h_m"] for m in found if m["h_m"] is not None]
    return dict(mounds=found, n_detected=sum(m["detected"] for m in found), k=k,
                median_h_m=float(np.median(hh)) if hh else None)


# ------------------------------------------------------------ noise floor ---
def ring_noise_floor(dem, g, x0, y0, radii, stride_m=40.0, nth=36, margin=None,
                     top=6):
    """Ring statistic on a centre/radius grid with no target -> threshold data."""
    ny, nx = dem.shape
    margin = margin or (max(radii) + 55.0)
    xs = np.arange(x0 + margin, x0 + nx * g - margin, stride_m)
    ys = np.arange(y0 + margin, y0 + ny * g - margin, stride_m)
    res = {float(r): [] for r in radii}
    res_az25 = {}
    res_tr = {}
    hits = []
    n_c = 0
    for cy in ys:
        for cx in xs:
            n_c += 1
            for r in radii:
                hw = max(4.0, 0.15 * r)
                m = ring_metrics(dem, g, x0, y0, cx, cy, r, halfw=hw, nth=nth)
                if m.get("dip_m") is not None:
                    res[float(r)].append(m["dip_m"])
                    if m.get("dip_az25") is not None:
                        res_az25.setdefault(float(r), []).append(m["dip_az25"])
                    if m.get("trough_m") is not None:
                        res_tr.setdefault(float(r), []).append(m["trough_m"])
                    hits.append((m.get("dip_az25") if m.get("dip_az25") is not None
                                 else m["dip_m"], float(cx), float(cy), float(r),
                                 m.get("az_frac_dip"), m.get("valid_frac")))
    out = dict(n_centres=int(n_c), stride_m=float(stride_m),
               radii=[float(r) for r in radii], per_radius={},
               note=("statistic = dip_m (median over azimuth, detrended); "
                     "dip_az25 = 25th percentile of the per-azimuth dips, "
                     "which stays low for texture that dips on only some azimuths"))
    for r, vals in res.items():
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size:
            entry = dict(n=int(v.size), p50=float(np.percentile(v, 50)),
                         p99=float(np.percentile(v, 99)),
                         p999=float(np.percentile(v, 99.9)), max=float(v.max()))
            v25 = np.asarray(res_az25.get(r, []), float)
            v25 = v25[np.isfinite(v25)]
            if v25.size:
                entry.update(az25_p99=float(np.percentile(v25, 99)),
                             az25_max=float(v25.max()))
            vt = np.asarray(res_tr.get(r, []), float)
            vt = vt[np.isfinite(vt)]
            if vt.size:
                entry.update(trough_p99=float(np.percentile(vt, 99)),
                             trough_max=float(vt.max()))
            out["per_radius"]["%.0f" % r] = entry
    hits.sort(key=lambda t: -t[0])
    out["top"] = [dict(dip_m=round(h[0], 3), x=h[1], y=h[2], r=h[3],
                       az_frac_dip=None if h[4] is None else round(h[4], 3),
                       valid_frac=None if h[5] is None else round(h[5], 3))
                  for h in hits[:top]]
    return out


# ----------------------------------------------------------------- driver ---
def feature_metrics(f, dem, g, x0, y0, layer):
    if f["type"] in ("ring_ditch", "hole_halo"):
        hw = max(2.5, 0.5 * float(f.get("width", 6.0)) + 1.5)
        m = ring_metrics(dem, g, x0, y0, f["cx"], f["cy"], float(f["r"]), halfw=hw)
        m["verdict"] = ring_verdict(m, f.get("depth"))
    elif f["type"] == "square_ditch":
        hw = max(2.5, 0.5 * float(f.get("width", 6.0)) + 1.5)
        m = square_metrics(dem, g, x0, y0, f["cx"], f["cy"], float(f["r"]),
                           halfw=hw, rot_deg=float(f.get("rot", 0.0)))
        m["verdict"] = ring_verdict(m, f.get("depth"))
        m["note"] = "square-ditch template (|max(|dx|,|dy|) - r| <= halfw)"
    elif f["type"] in ("road", "causeway"):
        m = line_metrics(dem, g, x0, y0, f["p0"], f["p1"], float(f["width"]),
                         sign=1.0 if f["type"] == "road" else -1.0)
        if m.get("dip_m") is None:
            m["verdict"] = "masked"
        else:
            m["verdict"] = ("clear" if m["dip_m"] >= 0.45
                            and (m.get("frac_length_dip") or 0) >= 0.5 else
                            "partial" if m["dip_m"] >= 0.25 else "absent")
    elif f["type"] == "mound_ring":
        m = mound_metrics(dem, g, x0, y0, f)
        m["verdict"] = ("clear" if m["n_detected"] >= max(5, 0.7 * m["k"])
                        else "partial" if m["n_detected"] >= 3 else "absent")
    elif f["type"] in ("oxbow", "pond"):
        m = ring_metrics(dem, g, x0, y0, f["cx"], f["cy"], float(f["r"]), halfw=6.0)
        m["verdict"] = ("false-positive-risk"
                        if (m.get("dip_m") or 0) >= 0.5
                        and (m.get("az_frac_dip") or 0) >= 0.30 else "quiet")
    elif f["type"] == "meander":
        m = ring_metrics(dem, g, x0, y0, f["x0"] + 0.5 * float(f["length"]),
                         f["yc"], float(f.get("A1", 60.0)), halfw=8.0)
        m["verdict"] = ("false-positive-risk"
                        if (m.get("dip_m") or 0) >= 0.5
                        and (m.get("az_frac_dip") or 0) >= 0.30 else "quiet")
        m["note"] = "river bend treated as an arc: ring statistic at (centre of reach, bend amplitude)"
    elif "p0" in f:
        m = line_metrics(dem, g, x0, y0, f["p0"], f["p1"],
                         float(f.get("width", 10.0)))
        m["verdict"] = ("false-positive-risk"
                        if (m.get("dip_m") or 0) >= 0.5
                        and (m.get("frac_length_dip") or 0) >= 0.5 else "quiet")
    elif "cx" in f and "r" in f:
        m = ring_metrics(dem, g, x0, y0, f["cx"], f["cy"], float(f["r"]), halfw=6.0)
        m["verdict"] = ("false-positive-risk"
                        if (m.get("dip_m") or 0) >= 0.5
                        and (m.get("az_frac_dip") or 0) >= 0.30 else "quiet")
    else:
        m = dict(verdict="n/a", note="no simple geometric statistic for this feature type")
    m["layer"] = layer
    return m


def evaluate(dtm_npz, truth_json, layers=("dem", "relief15"), noise_floor=False,
             radii=None):
    d = np.load(dtm_npz, allow_pickle=True)
    tr = json.load(open(truth_json))
    g = float(d["grid_m"])
    x0, y0 = float(d["x0"]), float(d["y0"])
    out = dict(scene=tr["scene"], dtm=os.path.basename(str(dtm_npz)),
               grid_m=g, method=str(d["method"]) if "method" in d else "?",
               layers=list(layers), features=[])
    for f in tr["features"]:
        rec = dict(id=f["id"], type=f["type"], role=f.get("role"),
                   expect=f.get("expect"), label=f.get("label"),
                   depth_true=f.get("depth"), size_m=2 * float(f.get("r", f.get("R", 0.0))))
        for layer in layers:
            if layer not in d:
                continue
            rec[layer] = feature_metrics(f, np.asarray(d[layer], dtype=np.float64),
                                         g, x0, y0, layer)
        out["features"].append(rec)
    if noise_floor:
        layer = "relief15" if "relief15" in d else "dem"
        dem = np.asarray(d[layer], dtype=np.float64)
        radii = radii or [20, 30, 45, 60, 90, 120, 150, 200]
        out["noise_floor"] = ring_noise_floor(dem, g, x0, y0, radii)
        out["noise_floor"]["layer"] = layer
    return out


def plot_profiles(dtm_npz, truth_json, png, layer="relief15", nth=72):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(dtm_npz, allow_pickle=True)
    tr = json.load(open(truth_json))
    g = float(d["grid_m"]); x0 = float(d["x0"]); y0 = float(d["y0"])
    if layer not in d:
        layer = "dem"
    dem = np.asarray(d[layer], dtype=np.float64)
    rings = [f for f in tr["features"]
             if f["type"] in ("ring_ditch", "square_ditch", "oxbow", "pond",
                              "hole_halo")]
    if not rings:
        return False
    n = len(rings)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), dpi=140, squeeze=False)
    for ax, f in zip(axes[0], rings):
        rmax = float(f["r"]) + max(45.0, 2.0 * float(f.get("width", 6.0)))
        rr, P, _X, _Y = polar(dem, g, x0, y0, f["cx"], f["cy"], rmax, dr=1.0, nth=nth)
        with np.errstate(all="ignore"):
            med = np.nanmedian(P, axis=0)
            q25 = np.nanpercentile(P, 25, axis=0)
            q75 = np.nanpercentile(P, 75, axis=0)
        sel = (rr > float(f["r"]) + 12) & (rr < rmax)
        base = np.nanmedian(med[sel]) if np.isfinite(med[sel]).any() else 0.0
        ax.fill_between(rr, q25 - base, q75 - base, color="0.85",
                        label="IQR over azimuth")
        ax.plot(rr, med - base, "k-", lw=1.2, label="median")
        ax.axvline(float(f["r"]), color="r", ls="--", lw=0.8)
        ax.axhline(0, color="b", lw=0.6)
        ax.set_title("%s  R=%.0f m  (%s)" % (f["id"], float(f["r"]), layer), fontsize=8)
        ax.set_xlabel("radius (m)", fontsize=7)
        ax.set_ylabel("height rel. outer (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm", required=True, nargs="+", help="*_dtm.npz (globs ok)")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--layers", default="dem,relief15")
    ap.add_argument("--noise-floor", action="store_true")
    ap.add_argument("--profiles", action="store_true")
    a = ap.parse_args(argv)
    paths = []
    for p in a.dtm:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    for npz in paths:
        base = npz.replace("_dtm.npz", "")
        cands = [base + "_truth.json",
                 os.path.join(os.path.dirname(npz), "..",
                              os.path.basename(base) + "_truth.json"),
                 os.path.join(os.path.dirname(npz), "..", "out",
                              os.path.basename(base) + "_truth.json")]
        tj = a.truth or next((c for c in cands if os.path.exists(c)), cands[0])
        if not os.path.exists(tj):
            print("SKIP %s: no truth json" % npz, flush=True)
            continue
        res = evaluate(npz, tj, layers=tuple(a.layers.split(",")),
                       noise_floor=a.noise_floor)
        outdir = a.outdir or os.path.dirname(os.path.abspath(npz))
        js = os.path.join(outdir, os.path.basename(base) + "_metrics.json")
        with open(js, "w") as fh:
            json.dump(res, fh, indent=1)
        if a.profiles:
            plot_profiles(npz, tj, js.replace("_metrics.json", "_profiles.png"))
        print("\n%s  [method=%s]" % (res["scene"], res["method"]))
        for rec in res["features"]:
            parts = []
            for layer in ("dem", "relief15", "relief30"):
                m = rec.get(layer)
                if not isinstance(m, dict):
                    continue
                if m.get("dip_m") is not None:
                    parts.append("%s dip=%.2f %s v=%.2f %s" % (
                        layer, m["dip_m"],
                        ("az=%.2f" % m["az_frac_dip"]) if "az_frac_dip" in m
                        else ("len=%.2f" % (m.get("frac_length_dip") or 0)),
                        m.get("valid_frac") or 0, m.get("verdict")))
                elif m.get("n_detected") is not None:
                    parts.append("%s mounds %d/%d v=%.2f %s" % (
                        layer, m["n_detected"], m["k"], m.get("valid_frac") or 0,
                        m.get("verdict")))
            print("  %-10s %-13s %-9s %s" % (rec["id"], rec["type"],
                                             rec.get("expect"),
                                             " | ".join(parts) or "n/a"))
        if "noise_floor" in res:
            nf = res["noise_floor"]["per_radius"]
            print("  noise floor (dip_m): " + ", ".join(
                "r=%s p99=%.2f max=%.2f" % (k, v["p99"], v["max"])
                for k, v in nf.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
