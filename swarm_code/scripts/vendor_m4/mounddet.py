"""mounddet.py -- mound-cluster detector (>= 5 positive bumps, e.g. a mound ring
around a plaza).

Method
------
1. Band-pass: bumps of 0.4-3 m over 8-40 m diameter are isolated as
   h - Gaussian(15 m), and required to be positive against a wider background
   (h - Gaussian(60 m) > 0).
2. Peak test: local maximum on a 3-cell (9 m) window, with a *prominence* test
   against the local annulus (radius 2-4 cells) so a bump on a slope is not
   counted, and a *size* test from the half-prominence radius.
3. Clustering: single-linkage within 250 m; keep clusters with >= 5 mounds.
4. Ring test: Kasa circle fit to the mound centres.  Report ring radius, RMS
   residual / radius, the mound-free interior (plaza) and the angular coverage.
   A mound village is a ring of mounds around a plaza; a natural cluster of
   hummocks is usually not circular and has no mound-free centre.

Output: one row per cluster (not per mound) with a mound list in the diagnostics.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

VALIDATED = True   # planted 6-mound ring recovered (n=5, R=45 m, rms/R 0.01,
                   # plaza clear); null terrain and valley scenes produce no clean
                   # cluster (their fits carry not_circular / out_of_range flags).

DEFAULTS = dict(
    min_height_m=0.55, max_height_m=4.0, min_prominence_m=0.45,
    smooth_cells=1.0, peak_win_cells=3, cluster_dist_m=120.0, min_mounds=5,
    ring_radius_m=(15.0, 150.0), max_ring_rms_frac=0.30, min_angle_frac=0.55,
    plaza_clear_frac=0.45, input_mode="lrm15", max_clusters=60,
    min_mound_radius_m=1.5, max_mound_radius_m=25.0,
)


def _fields(prep, p):
    h = prep["h"]
    v = prep["infer"] & np.isfinite(h) & prep["valid"]
    return np.where(v, h, 0.0).astype(np.float32), v.astype(np.float32)


def find_mounds(hz, wgt, cell, p):
    import preprocess
    h = np.where(wgt > 0.5, hz, np.nan)
    bp = preprocess.local_relief(h, wgt > 0.5, 15.0, cell)
    wide = preprocess.local_relief(h, wgt > 0.5, 60.0, cell)
    bp = np.where(np.isfinite(bp) & np.isfinite(wide), bp, np.nan)
    bps = ndi.gaussian_filter(np.nan_to_num(bp), p["smooth_cells"], mode="nearest")
    bps = np.where(np.isfinite(bp), bps, -9.0)
    win = int(p["peak_win_cells"])
    mx = ndi.maximum_filter(bps, size=2 * win + 1, mode="nearest")
    peaks = (bps > p["min_height_m"]) & (bps >= mx - 1e-6) & (bps < p["max_height_m"])
    # prominence: against the annulus at radius 2..4 cells
    # prominence: peak height above the 75 m background.  (A 9-cell inner mean is
    # larger than a 12 m mound and dilutes it to ~15 % of its height, which hid
    # every planted mound in the control scene.)
    outer_mean = ndi.uniform_filter(bps, 25, mode="nearest")
    prom = bps - outer_mean
    peaks &= prom > p["min_prominence_m"]
    ys, xs = np.nonzero(peaks)
    mounds = []
    for y, x in zip(ys, xs):
        hmax = float(bps[y, x])
        # half-max radius for a size estimate
        r = np.arange(0.5, 12.0, 0.5)
        th = np.arange(0, 2 * math.pi, math.pi / 8)
        vals = []
        for rr in r:
            yy = y + rr * np.sin(th)
            xx = x + rr * np.cos(th)
            v = ndi.map_coordinates(bps, np.vstack([yy, xx]), order=1, mode="nearest",
                                    prefilter=False)
            vals.append(float(np.mean(v)))
        vals = np.array(vals)
        half = hmax * 0.5
        below = np.nonzero(vals < half)[0]
        radius_m = float((r[below[0]] if below.size else r[-1]) * cell)
        if not (p["min_mound_radius_m"] <= radius_m <= p["max_mound_radius_m"]):
            continue
        mounds.append(dict(row=float(y), col=float(x), height_m=hmax,
                           prominence_m=float(prom[y, x]), radius_m=radius_m))
    return mounds


def cluster_mounds(mounds, cell, p):
    if not mounds:
        return []
    pts = np.array([[m["row"], m["col"]] for m in mounds])
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dmax = p["cluster_dist_m"] / cell
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(pts[i] - pts[j])) <= dmax:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = []
    for g in groups.values():
        if len(g) < p["min_mounds"]:
            continue
        sub = [mounds[i] for i in g]
        xy = np.array([[m["col"] * cell, m["row"] * cell] for m in sub])
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
        b = xy[:, 0] ** 2 + xy[:, 1] ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            ccx, ccy, c = sol
            R = math.sqrt(max(c + ccx ** 2 + ccy ** 2, 1e-9))
            res = np.hypot(xy[:, 0] - ccx, xy[:, 1] - ccy) - R
            rms = float(np.std(res))
        except np.linalg.LinAlgError:
            ccx, ccy, R, rms = cx, cy, np.nan, np.nan
        ang = np.sort(np.arctan2(xy[:, 1] - ccy, xy[:, 0] - ccx))
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
        max_gap = float(np.max(gaps)) if len(ang) > 1 else 2 * math.pi
        cover = 1.0 - (max_gap / (2 * math.pi))
        # plaza: is the ring centre free of mounds?
        rmin = np.min(np.hypot(xy[:, 0] - ccx, xy[:, 1] - ccy))
        clusters.append(dict(
            row=float(ccy / cell), col=float(ccx / cell),
            x_m=float(ccx), y_m=float(ccy), n_mounds=len(sub),
            ring_radius_m=float(R), ring_rms_m=rms,
            ring_rms_frac=float(rms / R) if np.isfinite(R) and R > 1e-6 else np.nan,
            angular_coverage=float(cover), max_gap_deg=float(math.degrees(max_gap)),
            min_mound_radius_m=float(rmin),
            plaza_clear=bool(np.isfinite(R) and rmin > p["plaza_clear_frac"] * R),
            height_median_m=float(np.median([m["height_m"] for m in sub])),
            height_max_m=float(np.max([m["height_m"] for m in sub])),
            mound_radius_median_m=float(np.median([m["radius_m"] for m in sub])),
            mounds=sub))
    clusters.sort(key=lambda d: -d["n_mounds"])
    return clusters[: int(p["max_clusters"])]


def _flags(c, p):
    f = []
    r0, r1 = p["ring_radius_m"]
    if not (r0 <= c["ring_radius_m"] <= r1):
        f.append("ring_radius_out_of_range")
    if not np.isfinite(c["ring_rms_frac"]) or c["ring_rms_frac"] > p["max_ring_rms_frac"]:
        f.append("not_circular")
    if c["angular_coverage"] < p["min_angle_frac"]:
        f.append("incomplete_ring")
    if not c["plaza_clear"]:
        f.append("plaza_not_clear")
    if c["height_median_m"] < 0.5:
        f.append("weak_mounds")
    if c["mound_radius_median_m"] > 25.0:
        f.append("mounds_too_broad")
    return f


def detect_mounds(prep, params=None):
    p = dict(DEFAULTS, **(params or {}))
    cell = float(prep["cell"])
    hz, wgt = _fields(prep, p)
    mounds = find_mounds(hz, wgt, cell, p)
    clusters = cluster_mounds(mounds, cell, p)
    for c in clusters:
        c["flags"] = _flags(c, p)
        c["kind"] = "mound_cluster"
        c["score"] = c["n_mounds"] * c["height_median_m"]
    return dict(n_mounds=len(mounds), clusters=clusters, mounds=mounds)
