"""linedet.py -- linear earthwork detector (sunken/embanked roads) -- EXPERIMENTAL.

STATUS: NOT VALIDATED. Do not use this module's output to produce candidates.
Measured on the synthetic suite (det/run_synth.py scenes):
  * a planted 300 m, 5 m wide, 0.7 m deep sunken road is MISSED at the operating
    point that yields zero lines in featureless/valley scenes;
  * at a looser operating point it is found, but 9-22 spurious 1-1.7 km "lines"
    appear in a null terrain scene -- i.e. the response map is noise-dominated;
  * the river scene (N1) yields 13 "roads", one of which passes the screening
    rules.  A river reach is a legitimate long linear trough, so this family
    needs the drainage-context test that the skeptic role owns.
Root cause, for whoever continues this: the per-cell response is max over
3 widths x 12 orientations of |cross-dip|, which inflates terrain noise ~2.5 sigma;
the mask therefore covers ~16 % of a pasture scene.  The fix is to smooth the
response *along* the strike (>=60 m) for each orientation before taking the max
and thresholding -- sketched in the docstring below, not implemented.

The cross-profile and Hough machinery here is reusable and unit-tested in the
sense that it recovers the correct azimuth/geometry on the river and road rows it
does report; the thresholding/coherence stage is what fails.

Original design note (kept for continuity)
------------------------------------------

Method (decimated cross-profile matched filter; cheap and auditable)
-------------------------------------------------------------------
1. Candidate grid every `stride` cells (default 2 = 6 m).
2. At each grid point and for each of `n_theta` orientations, the cross-section
   response is measured at three points:
       resp(theta) = 0.5*(h(p + o n) + h(p - o n)) - h(p),   n = unit normal
   for o = w + gap cells with w = 1..3 (a 3-15 m wide feature).  A sunken road is a
   trough -> positive resp; an embanked road -> negative resp.  The orientation
   with the largest |resp| is kept, with the signed value.
3. Components of |resp| above a robust threshold are labelled and measured by PCA:
   length along the principal axis, width across it, azimuth, and the agreement
   between the segment axis and the local filter orientation.  Segments shorter
   than `min_length_m` (200 m default) are dropped -- the length requirement is
   measured, not assumed.
4. For surviving segments a full cross profile (1.5 m steps, +-15 m) is taken at
   five positions along the strike and reduced to depth, half-depth width, bank
   height and asymmetry, so a reviewer can see the section that produced the row.

Deliberate non-goals: no road-network linking, no curvature model, and no
causeway-at-drainage-crossing test (that is detector m3's workstream).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

VALIDATED = False

DEFAULTS = dict(
    stride=2, n_theta=12, widths_cells=(1, 2, 3), flank_gap_cells=2,
    thresh_frac=0.70, thresh_min_m=0.30, min_length_m=200.0, max_width_m=20.0,
    max_lines=120, cross_half_cells=15, n_cross=5, min_amplitude_m=0.30,
    close_cells=2, input_mode="lrm15", max_components=400,
)


def _fields(prep, p):
    from ringdet import _fields as rf
    return rf(prep, p)


def response_map(hz, wgt, cell, p):
    """Cross-section response: the orientation of largest |resp| at every cell.

    Whole-array subpixel shifts (`ndimage.shift`, bilinear) rather than point
    sampling: a 5 m road is only 1.7 cells wide, so a 6 m point grid hits it
    sporadically and the segment breaks into unlinkable dots
    (measured: 0 lines recovered from a planted 300 m road).
    """
    w = (wgt > 0.5)
    z0 = np.where(w, hz, 0.0).astype(np.float32)
    wf = w.astype(np.float32)
    best = np.full(z0.shape, -1.0, np.float32)
    bth = np.zeros(z0.shape, np.float32)
    bw = np.zeros(z0.shape, np.float32)
    ths = np.arange(p["n_theta"]) * math.pi / p["n_theta"]
    for wh in p["widths_cells"]:
        o = wh + p["flank_gap_cells"]
        for th in ths:
            nyv, nxv = -math.sin(th), math.cos(th)
            for sgn in (1.0, -1.0):
                pass
            dy, dx = o * nyv, o * nxv
            hp = ndi.shift(z0, (dy, dx), order=1, mode="constant", cval=0.0,
                           prefilter=False)
            wp = ndi.shift(wf, (dy, dx), order=1, mode="constant", cval=0.0,
                           prefilter=False)
            hm = ndi.shift(z0, (-dy, -dx), order=1, mode="constant", cval=0.0,
                           prefilter=False)
            wm = ndi.shift(wf, (-dy, -dx), order=1, mode="constant", cval=0.0,
                           prefilter=False)
            ok = w & (wp > 0.9) & (wm > 0.9)
            resp = np.where(ok, 0.5 * (hp + hm) - z0, 0.0).astype(np.float32)
            upd = ok & (np.abs(resp) > best)
            best = np.where(upd, np.abs(resp), best)
            bth = np.where(upd, th, bth)
            bw = np.where(upd, float(wh), bw)
    resp_out = np.where(best >= 0, best, np.nan).astype(np.float32)
    resp_out[~w] = np.nan
    return resp_out, bth, bw


def cross_profile(hz, wgt, row, col, nvec, half_cells, of_cells):
    t = np.linspace(-half_cells, half_cells, 2 * int(round(half_cells)) + 1)
    yy = row + t * nvec[0]
    xx = col + t * nvec[1]
    pr = ndi.map_coordinates(hz, np.vstack([yy, xx]), order=1, mode="constant",
                             cval=np.nan, prefilter=False)
    wv = ndi.map_coordinates(wgt.astype(np.float32), np.vstack([yy, xx]), order=1,
                             mode="constant", cval=0.0, prefilter=False)
    pr = np.where(wv > 0.5, pr, np.nan)
    c = len(t) // 2
    of = max(1, int(of_cells))
    inner = pr[c - of:c + of + 1]
    if not np.isfinite(inner).any():
        return dict(depth_m=np.nan, crest_m=np.nan, width_m=np.nan, asym_m=np.nan,
                    valid=False)
    imin = int(np.nanargmin(inner))
    imax = int(np.nanargmax(inner))
    f_in = np.nanmean(pr[max(0, c - of - 3):max(1, c - of)])
    f_out = np.nanmean(pr[min(len(pr), c + of):min(len(pr), c + of + 3)])
    base = 0.5 * (f_in + f_out) if np.isfinite(f_in) and np.isfinite(f_out) else np.nan
    depth = float(base - inner[imin]) if np.isfinite(base) else np.nan
    crest = float(inner[imax] - base) if np.isfinite(base) else np.nan
    width = np.nan
    if np.isfinite(depth) and depth > 0:
        half = base - 0.5 * depth
        j = c - of + imin
        lo = j
        while lo - 1 >= 0 and np.isfinite(pr[lo - 1]) and pr[lo - 1] < half:
            lo -= 1
        hi = j
        while hi + 1 < len(pr) and np.isfinite(pr[hi + 1]) and pr[hi + 1] < half:
            hi += 1
        width = float((hi - lo) * (t[1] - t[0]))
    return dict(depth_m=depth, crest_m=crest, width_m=float(width),
                asym_m=(float(f_out - f_in) if np.isfinite(f_in) and np.isfinite(f_out)
                        else np.nan),
                valid=True, profile_m=t.round(2).tolist(),
                profile_z=[None if not np.isfinite(v) else round(float(v), 3) for v in pr])


def _contiguous_span(along, gap=3.0):
    """Longest contiguously occupied stretch of an along-axis coordinate list."""
    a = np.sort(np.asarray(along, dtype=np.float64))
    if a.size == 0:
        return 0.0, 0.0, 0.0
    best = (a[0], a[0])
    cur0 = a[0]
    prev = a[0]
    for v in a[1:]:
        if v - prev > gap:
            if prev - cur0 > best[1] - best[0]:
                best = (cur0, prev)
            cur0 = v
        prev = v
    if prev - cur0 > best[1] - best[0]:
        best = (cur0, prev)
    return float(best[1] - best[0]), float(best[0]), float(best[1])


def hough_lines(a, min_amp, min_votes, perc=99.5, dtheta_deg=2.0, tol_cells=1.5):
    """Accumulate |resp| over line offsets for a dense set of orientations.

    A line with axis (cos t, sin t) in (row, col) space has normal (-sin t, cos t);
    every cell votes for (t, perpendicular offset).  `min_votes` is the length
    requirement expressed in votes: at least `min_length_m / cell` cells of the
    mask must lie on the line.
    """
    fin = np.isfinite(a)
    vals_all = a[fin]
    if vals_all.size == 0:
        return []
    thr = max(min_amp, 0.6 * float(np.percentile(vals_all, perc)))
    mask = fin & (a >= thr)
    ys, xs = np.nonzero(mask)
    if ys.size < min_votes:
        return []
    v = a[ys, xs]
    out = []
    for tdeg in np.arange(0.0, 180.0, dtheta_deg):
        t = math.radians(tdeg)
        ct, st = math.cos(t), math.sin(t)
        perp = -xs * st + ys * ct
        p0 = float(perp.min())
        bins = np.round(perp - p0).astype(np.int64)
        hist = np.bincount(bins, weights=v)
        cnt = np.bincount(bins)
        # local maxima with enough votes
        pad = np.concatenate([[0.0], hist, [0.0]])
        peak = np.nonzero((hist >= pad[:-2]) & (hist >= pad[2:]) & (cnt >= min_votes))[0]
        for b in peak:
            sel = np.abs(bins - b) <= tol_cells
            if sel.sum() < min_votes:
                continue
            along = xs[sel] * ct + ys[sel] * st
            span, a0, a1 = _contiguous_span(along)
            out.append(dict(theta_deg=float(tdeg), perp=float(b + p0),
                            votes=float(hist[b]), n=int(sel.sum()), span_cells=span,
                            along_min=a0, along_max=a1))
    return out


def detect_lines(prep, params=None, verbose=False):
    p = dict(DEFAULTS, **(params or {}))
    cell = float(prep["cell"])
    hz, wgt = _fields(prep, p)
    resp, th, wmap = response_map(hz, wgt, cell, p)
    a = np.abs(resp)
    # a 5 m road is ~1.7 cells wide, so at most ~50-70 % of the cells along its
    # centreline can carry the response; votes are therefore a relaxed support
    # test and the *contiguous span* below is the real >=200 m requirement.
    min_votes = max(20, int(math.ceil(0.35 * p["min_length_m"] / cell)))
    cands = hough_lines(a, p["min_amplitude_m"], min_votes, perc=99.5,
                        dtheta_deg=180.0 / max(24, 2 * p["n_theta"]))
    if not cands:
        return []
    # a line must also span >= min_length_m of *mask* cells, not just vote count
    cands = [c for c in cands if c["span_cells"] * cell >= p["min_length_m"]]
    cands.sort(key=lambda c: (-c["span_cells"], -c["votes"]))
    out = []
    for c in cands:
        t = math.radians(c["theta_deg"])
        ct, st = math.cos(t), math.sin(t)
        tvec = np.array([ct, st])            # (row, col)
        nvec = np.array([-st, ct])
        # midpoint of the supporting cells along the line
        amid = 0.5 * (c["along_min"] + c["along_max"])
        # perpendicular offset -> a point on the line: solve for (y,x) with
        # -x st + y ct = perp and x ct + y st = amid
        y0 = amid * st + c["perp"] * ct
        x0 = amid * ct - c["perp"] * st
        if not (0 <= y0 < hz.shape[0] and 0 <= x0 < hz.shape[1]):
            continue
        length_cells = c["span_cells"]
        ts = np.linspace(-0.35, 0.35, p["n_cross"]) * length_cells
        cps = []
        for tt in ts:
            yy = y0 + tt * tvec[0]
            xx = x0 + tt * tvec[1]
            if not (0 <= yy < hz.shape[0] and 0 <= xx < hz.shape[1]):
                continue
            cps.append(cross_profile(hz, wgt, yy, xx, nvec, p["cross_half_cells"], 3))
        cps = [q for q in cps if q["valid"]]
        if not cps:
            continue
        depth = float(np.nanmedian([q["depth_m"] for q in cps]))
        crest = float(np.nanmedian([q["crest_m"] for q in cps]))
        wmeas = float(np.nanmedian([q["width_m"] for q in cps]))
        asym = float(np.nanmedian([q["asym_m"] for q in cps]))
        amp = max(np.nan_to_num(depth), np.nan_to_num(crest))
        if amp < p["min_amplitude_m"]:
            continue
        # continuity: how many of the sampled cross sections show the feature
        pos = sum(1 for q in cps if max(np.nan_to_num(q["depth_m"]),
                                        np.nan_to_num(q["crest_m"])) >= 0.5 * amp)
        if pos < max(3, int(0.6 * len(cps))):
            continue
        rr = int(max(8, min(25, length_cells / 4)))
        ys_ = slice(max(0, int(y0) - rr), min(hz.shape[0], int(y0) + rr))
        xs_ = slice(max(0, int(x0) - rr), min(hz.shape[1], int(x0) + rr))
        sub = np.where(wgt[ys_, xs_] > 0.5, hz[ys_, xs_], np.nan)
        sig = (float(1.4826 * np.nanmedian(np.abs(sub - np.nanmedian(sub))))
               if sub.size else np.nan)
        az_geo = (90.0 - c["theta_deg"]) % 180.0
        flags = []
        if np.isfinite(wmeas) and wmeas > p["max_width_m"]:
            flags.append("too_wide_gt_%.0fm" % p["max_width_m"])
        if np.isfinite(asym) and abs(asym) > 0.7 * amp:
            flags.append("cross_section_asymmetric")
        edge = min(y0, hz.shape[0] - 1 - y0, x0, hz.shape[1] - 1 - x0) * cell
        if edge < 0.5 * length_cells * cell:
            flags.append("edge_clipped")
        if pos < len(cps):
            flags.append("discontinuous")
        kind = ("sunken_road" if np.nan_to_num(depth) >= np.nan_to_num(crest)
                else "embanked_road")
        out.append(dict(kind=kind, row=float(y0), col=float(x0), azimuth_deg=float(az_geo),
                        length_m=float(length_cells * cell), width_m=float(wmeas),
                        dip_m=float(depth), crest_m=float(crest), asym_m=float(asym),
                        sig_bg_m=sig,
                        snr=(float(amp / sig) if np.isfinite(sig) and sig > 1e-6 else np.nan),
                        votes=float(c["votes"]), n_px=int(c["n"]),
                        score=float(amp * length_cells * cell),
                        straightness=float(length_cells / max(length_cells, 1.0)),
                        orientation_mismatch_deg=0.0, n_valid_sections=int(pos),
                        n_sections=int(len(cps)), flags=flags, cross_profiles=cps))
    # de-duplicate: same azimuth (+-10 deg) and midpoints within 0.5 length
    out.sort(key=lambda d: -d["score"])
    kept = []
    for d in out:
        dup = False
        for k in kept:
            dd = math.hypot(d["row"] - k["row"], d["col"] - k["col"]) * cell
            da = abs(((d["azimuth_deg"] - k["azimuth_deg"]) + 90.0) % 180.0 - 90.0)
            if da < 10.0 and dd < 0.5 * max(d["length_m"], k["length_m"]):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept[: int(p["max_lines"])]


def line_flags_for_screening(L):
    """Which linear detections may be offered for skeptic review."""
    rej = [f for f in L.get("flags", [])
           if f.startswith("too_wide") or f in ("orientation_inconsistent",
                                                "cross_section_asymmetric")]
    if L.get("length_m", 0) < 200.0:
        rej.append("shorter_than_200m")
    if not np.isfinite(L.get("width_m", np.nan)) or L["width_m"] > 15.0:
        rej.append("width_outside_3_15m")
    if L.get("snr", np.nan) < 3.0:
        rej.append("snr_below_3")
    return (len(rej) == 0), ";".join(rej) if rej else "clean"
