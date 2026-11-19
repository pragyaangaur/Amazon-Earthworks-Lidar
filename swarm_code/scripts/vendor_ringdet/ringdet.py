"""ringdet.py -- ring / arc (geoglyph) detector on a 3 m DTM.

Method (polar radial-profile Hough; every number is auditable)
-------------------------------------------------------------
Stage A, coarse grid (default stride 4 cells = 12 m).  Around each candidate
centre we sample the working DTM on a polar grid (n_rays directions x radii
r = rmin..rmax) and build the ring-mean radial profile M(r) and the per-ray
profiles H(theta, r).  With `of` = fine flank offset (4 cells = 12 m):

    dipf(r)    = 0.5*(M(r-of) + M(r+of)) - M(r)          local trough depth
    two_sided  = min(in_dip,out_dip) / dipf               ditch vs. step/cliff
    cov(r)     = fraction of rays with per-ray dip >= 0.5*dipf(r)
    V(r)       = fraction of rays with data at r
    z(r)       = dipf / standard error of the ring mean

Stage A gate: local minimum of M in radius (a ring has one radius), dip >=
0.45 m, cov >= 0.55 (0.33 in arc mode), V >= 0.80 (0.45 arc), z >= 2.5,
two_sided >= 0.4.

Stage B, candidates only.  This is where natural terrain is separated from
ditches, using physical constraints that terrain curvature does not obey:

  * NARROWNESS.  A ditch is 3-15 m wide, i.e. a trough only 1-5 cells wide in
    radius.  We fit a straight background through two flank windows ~0.35 r
    away and measure the full width at half depth (FWHM).  Broad terrain
    concavities have FWHM of tens of metres and are rejected.  This is the
    published target geometry, not a tuned number.
  * CONSISTENCY.  Relative std of the per-ray dip (rel_std).  A dug ditch has
    about the same depth all round; a terrain hollow does not.
  * CONSTANT RADIUS.  Per-ray trough radius r(theta), interior argmax only
    (boundary-pinned rays are discarded, which kills the classic failure where
    smooth curvature drives every ray to the window edge).  IQR/median gives
    circularity; a 4-fold cosine fit of r(theta) detects square/angular rings;
    the largest angular gap detects U shapes.
  * AN ARC IS A CIRCLE.  For arc mode we fit a circle (Kasa) to the per-ray
    trough locus: a real arc fits with radius ~ r_star, a near-tangent straight
    road or field edge does not.

Ditch width, banks, interior (plaza) level, canopy, roughness and hole distance
are recorded for each detection so a skeptic can reject natural/modern causes.

Limitations (emitted, never hidden)
  * closed oxbow loops are rings; expect detection and rejection on width
    (>15 m), fluvial context and cross-section asymmetry;
  * cattle ponds are rings with a *basin* interior and no inner rim;
  * partial rings at the transect edge are found in arc mode and flagged
    `edge_clipped`.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

DEFAULTS = dict(
    rmin_m=15.0, rmax_m=200.0, stride_cells=4, n_rays=48, dr_cells=1.0,
    flank_of_cells=4, min_valid_frac=0.80, min_minor_valid_frac=0.45,
    min_dip_m=0.45, min_dip_z=2.5, min_cov=0.55, arc_min_cov=0.33,
    arc_min_dip_m=0.65, max_two_sided_fail=0.25,
    max_fwhm_m=27.0, max_rel_std=0.60, max_arc_radius_ratio=0.45,
    chunk_centers=512, max_dets=400, max_refine=120, refine=True, refine_cells=2,
    dr_fine_cells=0.5,
    nms_dist_frac=0.55, nms_merge_frac=1.45, max_width_m=20.0,
    input_mode="lrm15", min_snr=2.5,
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _shift_last(a, k):
    """Value at index i comes from i-k (zero fill at edges)."""
    n = a.shape[-1]
    if k > 0:
        out = np.zeros_like(a)
        out[..., k:] = a[..., :n - k]
    elif k < 0:
        out = np.zeros_like(a)
        out[..., :n + k] = a[..., -k:]
    else:
        out = a.copy()
    return out


def _wmean(a, wgt, axis=1):
    num = np.sum(a * wgt, axis=axis)
    den = np.sum(wgt, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 1e-6, out, np.nan), den


def sample_polar(hz, wgt, centers, radii_cells, n_rays):
    """Bilinear sample of heights and of the validity field on a polar grid."""
    th = (np.arange(n_rays) * (2.0 * math.pi / n_rays)).astype(np.float32)
    rr = np.asarray(radii_cells, dtype=np.float32)
    dy = np.sin(th)[None, :, None] * rr[None, None, :]
    dx = np.cos(th)[None, :, None] * rr[None, None, :]
    cy = np.asarray(centers, dtype=np.float32)[:, 0][:, None, None]
    cx = np.asarray(centers, dtype=np.float32)[:, 1][:, None, None]
    coords = np.vstack([(cy + dy).ravel().astype(np.float64),
                        (cx + dx).ravel().astype(np.float64)])
    shp = (len(cy), n_rays, len(rr))
    cube = ndi.map_coordinates(hz, coords, order=1, mode="constant", cval=0.0,
                               prefilter=False).reshape(shp).astype(np.float32)
    cw = ndi.map_coordinates(wgt, coords, order=1, mode="constant", cval=0.0,
                             prefilter=False).reshape(shp).astype(np.float32)
    return cube, cw


def radial_profiles(cube, cw, of):
    """Stage-A statistics from the polar cube. Arrays shaped (nc, nr)."""
    fin = cw > 0.5
    V = np.mean(fin, axis=1)
    M, Mden = _wmean(np.where(fin, cube, 0.0), fin.astype(np.float32))
    h = np.where(fin, cube, 0.0)
    hm, hp = _shift_last(h, of), _shift_last(h, -of)
    fm, fp = _shift_last(fin, of), _shift_last(fin, -of)
    ok = fin & fm & fp
    dper = np.where(ok, 0.5 * (hm + hp) - h, np.nan)
    with np.errstate(invalid="ignore"):
        dipf = np.nanmean(dper, axis=1)
    thr = 0.5 * np.nan_to_num(dipf)
    cov = np.nanmean(np.where(ok, (dper >= thr[:, None, :]).astype(np.float32), np.nan), axis=1)
    cov = np.nan_to_num(cov)
    inj, outj = _shift_last(M, of), _shift_last(M, -of)
    in_dip = 0.5 * (inj + _shift_last(M, of - 2)) - M
    out_dip = 0.5 * (outj + _shift_last(M, -of + 2)) - M
    dev = np.where(ok, dper - np.nan_to_num(dipf)[:, None, :], np.nan)
    sig = np.sqrt(np.nanmean(dev ** 2, axis=1))
    neff = np.maximum(np.sum(ok, axis=1) / 1.5, 1.0)
    z = np.nan_to_num(dipf) / (sig / np.sqrt(neff) + 1e-6)
    return dict(M=np.where(Mden > 1e-6, M, np.nan), V=V, dip=dipf, cov=cov,
                in_dip=in_dip, out_dip=out_dip, z=z, ok=ok, sig=sig)


def _stage_a_select(res, radii_cells, cell, p):
    """Best radius per centre + Stage-A gates."""
    dip, cov, V, z = res["dip"], res["cov"], res["V"], res["z"]
    nc, nr = dip.shape
    in_range = (radii_cells >= p["rmin_m"] / cell) & (radii_cells <= p["rmax_m"] / cell)
    minor = V >= p["min_minor_valid_frac"]
    dipf = np.nan_to_num(dip, nan=-1e9)
    locmin = np.zeros_like(dipf, bool)
    locmin[:, 1:-1] = (dipf[:, 1:-1] >= dipf[:, :-2]) & (dipf[:, 1:-1] >= dipf[:, 2:])
    score = np.where(locmin & in_range[None, :] & minor, dipf, -1e9)
    idx = np.argmax(score, axis=1)
    ii = np.arange(nc)
    ok = score[ii, idx] > -1e8
    d = np.where(ok, dip[ii, idx], np.nan)
    c = np.where(ok, cov[ii, idx], np.nan)
    v = np.where(ok, V[ii, idx], np.nan)
    zz = np.where(ok, z[ii, idx], np.nan)
    ind = np.where(ok, res["in_dip"][ii, idx], np.nan)
    outd = np.where(ok, res["out_dip"][ii, idx], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        two = np.minimum(ind, outd) / np.where(np.abs(d) > 1e-6, d, np.nan)
    full = v >= p["min_valid_frac"]
    need_cov = np.where(full, p["min_cov"], p["arc_min_cov"])
    need_dip = np.where(full, p["min_dip_m"], p["arc_min_dip_m"])
    gate = (ok & np.isfinite(d) & np.isfinite(two) & (d >= need_dip) & (c >= need_cov)
            & (zz >= p["min_dip_z"]) & (two >= p["max_two_sided_fail"]))
    return dict(gate=gate, idx=idx, dip=d, cov=c, V=v, z=zz, two_sided=two)


# --------------------------------------------------------------------------- #
# candidate centres
# --------------------------------------------------------------------------- #
def candidate_centers(prep, stride_cells):
    infer = prep["infer"]
    nx, ny = infer.shape
    s = int(max(1, stride_cells))
    rows = np.arange(s // 2, nx, s)
    cols = np.arange(s // 2, ny, s)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    keep = infer[rr, cc]
    return np.c_[rr[keep], cc[keep]].astype(np.float32), s


def _fields(prep, p=None):
    """Working raster for the ring statistic.

    Default `lrm15` = DTM minus its 15 m Gaussian.  For a 3-15 m ditch the local
    relief model keeps essentially the full depth (a 15 m Gaussian is blind to a
    narrow trough) while suppressing broad terrain concavities, which are the
    dominant false-positive family.  `dtm` is kept selectable so the choice can
    be re-measured on any transect.
    """
    mode = (p or {}).get("input_mode", "lrm15")
    h = prep["h"]
    if str(mode).startswith("lrm"):
        sig = int(round(float(str(mode)[3:] or 15)))
        if sig not in prep["lrm"]:
            import preprocess as _pp
            prep["lrm"][sig] = _pp.local_relief(h, prep["valid"], sig, prep["cell"])
        h = prep["lrm"][sig]
    v = prep["infer"] & np.isfinite(h) & prep["valid"]
    return np.where(v, h, 0.0).astype(np.float32), v.astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage B metrics
# --------------------------------------------------------------------------- #
def _ob_for(nr, i):
    """Flank window offset in cells: 0.35 r, at least 2, always inside array."""
    ob = int(np.clip(round(0.35 * i), 8, 22))
    ob = min(ob, max(2, i - 3), max(2, (nr - 1 - i) - 3))
    return max(2, ob)


def _bg_windows(M, i, ob):
    nr = len(M)
    ob = _ob_for(nr, i) if ob is None else ob
    a0, a1 = max(0, i - ob - 2), max(1, i - ob + 1)
    b0, b1 = min(nr, i + ob), min(nr, i + ob + 3)
    if a1 - a0 < 2 or b1 - b0 < 2:
        return None
    va = float(np.nanmean(M[a0:a1]))
    vb = float(np.nanmean(M[b0:b1]))
    if not (np.isfinite(va) and np.isfinite(vb)):
        return None
    ca, cb = 0.5 * (a0 + a1 - 1), 0.5 * (b0 + b1 - 1)
    if cb - ca < 2:
        return None
    return va, vb, ca, cb


def _fit_bg(M, i, ob):
    w = _bg_windows(M, i, ob)
    if w is None:
        return np.nan, False
    va, vb, ca, cb = w
    return va + (vb - va) * (i - ca) / (cb - ca), True


def _fwhm(M, i, ob):
    """Full width at half depth of the trough at index i against a linear
    background through the +-ob flanks. Returns (width_cells, depth_m)."""
    line_i, ok = _fit_bg(M, i, ob)
    if not ok or not np.isfinite(M[i]):
        return np.nan, np.nan
    depth = line_i - M[i]
    if depth <= 0:
        return np.nan, float(depth)
    half = line_i - 0.5 * depth
    nr = len(M)
    j = i
    while j - 1 >= 0 and np.isfinite(M[j - 1]) and M[j - 1] < half and i - j < 25:
        j -= 1
    k = i
    while k + 1 < nr and np.isfinite(M[k + 1]) and M[k + 1] < half and k - i < 25:
        k += 1
    return float(k - j + 1), float(depth)


def _kasa_fit(xy):
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    cx, cy, c = sol
    R = math.sqrt(max(c + cx ** 2 + cy ** 2, 1e-9))
    rms = float(np.std(np.hypot(x - cx, y - cy) - R))
    return float(cx), float(cy), float(R), rms


def candidate_metrics(prep, row, col, idx, radii_cells, cell, params):
    """Full diagnostic set for one candidate at (row, col), near radius index idx.

    Stage A runs on a 1-cell (3 m) radius ladder because it must be cheap over the
    whole transect.  Here, for a single candidate, we re-sample on a 0.5-cell
    (1.5 m) ladder: a 3-15 m ditch is only 1-5 cells wide, so a 3 m ladder
    under-measures depth by up to 40 %.  Depth and width reported downstream come
    from this fine ladder.
    """
    p = params
    n_rays = p["n_rays"]
    dr = float(p.get("dr_fine_cells", 0.5)) * cell
    of_m = float(p["flank_of_cells"]) * cell
    ofb = max(1, int(round(of_m / dr)))              # flank offset in fine bins
    hz, wgt = _fields(prep, p)
    r0 = float(radii_cells[idx] * cell)              # coarse radius estimate (m)
    ob_max_m = min(22 * cell, max(8 * cell, 0.38 * r0))
    # the fine ladder must reach below the *inner flank* sample (r0 - of - ob),
    # otherwise the per-ray dip (which needs both flanks) is undefined at the trough
    lo_m = max(1.5 * cell, r0 - ob_max_m - of_m - 8 * dr)
    hi_m = r0 + ob_max_m + of_m + 4 * cell
    rf = np.arange(lo_m, hi_m + dr * 0.5, dr)
    cube, cw = sample_polar(hz, wgt, np.array([[row, col]], np.float32),
                            rf / cell, n_rays)
    nr = len(rf)
    fin = cw[0] > 0.5
    M = np.nanmean(np.where(fin, cube[0], np.nan), axis=0)
    V = fin.mean(axis=0)
    h = np.where(fin, cube[0], 0.0)
    f16 = fin.astype(np.float32)
    hm = _shift_last(h, ofb)[0]
    hp = _shift_last(h, -ofb)[0]
    fm = _shift_last(f16, ofb)[0] > 0.5
    fp = _shift_last(f16, -ofb)[0] > 0.5
    ok = fin & fm & fp
    dper = np.where(ok, 0.5 * (hm + hp) - h, np.nan)          # (rays, nbins)

    # trough bin: best (background-line minus profile) within +-3 cells of r0
    near = np.nonzero(np.abs(rf - r0) <= 3.0 * cell)[0]
    best_i, best_dep, best_ob = int(near[len(near) // 2]), -1e9, 8
    for j in near:
        ob = _ob_for(nr, j)
        line_j, okj = _fit_bg(M, j, ob)
        if not okj or not np.isfinite(M[j]):
            continue
        dep = line_j - M[j]
        if dep > best_dep:
            best_i, best_dep, best_ob = int(j), dep, ob
    i = best_i
    ob = best_ob
    r_star = float(rf[i])
    fwhm_bins, depth_bg = _fwhm(M, i, ob)
    width_m = fwhm_bins * dr if np.isfinite(fwhm_bins) else np.nan

    # per-ray depth near the trough (each ray's own best fine bin within +-1.5 m)
    wj = np.nonzero(np.abs(rf - r_star) <= 1.5 * cell)[0]
    with np.errstate(invalid="ignore"):
        dip_r = np.nanmax(dper[:, wj], axis=1)
    valid_ray = np.isfinite(dip_r)
    dip = float(np.nanmedian(dip_r[valid_ray])) if valid_ray.any() else np.nan
    wj2 = np.nonzero(np.abs(rf - r_star) <= 4.5)[0]
    with np.errstate(invalid="ignore"):
        dip_own = np.nanmax(dper[:, wj2], axis=1)
    own = dip_own[valid_ray & np.isfinite(dip_own)]
    rel_std = float(np.std(own) / max(abs(np.median(own)), 1e-6)) if own.size else np.nan
    cov = float(np.nanmean(dip_r[valid_ray] >= 0.5 * dip)) if valid_ray.any() else 0.0

    # per-ray trough radius (interior argmax only) -> circularity / square / gap
    sel_w = np.nonzero((rf >= 0.62 * r_star) & (rf <= 1.38 * r_star))[0]
    win = np.where(np.isfinite(dper[:, sel_w]), dper[:, sel_w], -1e9)
    ar = np.arange(n_rays)
    am = np.argmax(win, axis=1)
    good = (win[ar, am] > -1e8) & (am > 0) & (am < win.shape[1] - 1)
    m0 = win[ar, np.clip(am, 0, win.shape[1] - 1)]
    mm = win[ar, np.clip(am - 1, 0, win.shape[1] - 1)]
    pp = win[ar, np.clip(am + 1, 0, win.shape[1] - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = 0.5 * (mm - pp) / (mm - 2 * m0 + pp)
    sh = np.clip(np.nan_to_num(sh), -0.5, 0.5)
    rtheta = np.where(good, rf[sel_w][np.clip(am, 0, win.shape[1] - 1)] + sh * dr, np.nan)
    rt = rtheta[good]
    if rt.size >= 8:
        med = float(np.median(rt))
        circ = float(1.0 - (np.percentile(rt, 75) - np.percentile(rt, 25)) / max(med, 1e-6))
    else:
        med, circ = float("nan"), 0.0

    th = np.arange(n_rays) * (2.0 * math.pi / n_rays)
    good_dip = valid_ray & (np.nan_to_num(dip_r, nan=-1e9) >= 0.35 * max(dip, 1e-6))
    absent = valid_ray & ~good_dip
    gap_rays = 0
    if absent.any() and good_dip.any():
        st = int(np.nonzero(good_dip)[0][0])
        order = np.concatenate([np.arange(st, n_rays), np.arange(0, st)])
        run = bestrun = 0
        for k in order:
            if not valid_ray[k]:
                continue                      # unknown ray: neither gap nor break
            if absent[k]:
                run += 1
                bestrun = max(bestrun, run)
            else:
                run = 0
        gap_rays = bestrun
    gap_deg = 360.0 * gap_rays / n_rays
    r2_4, sq_amp = 0.0, 0.0
    if rt.size >= 16 and np.isfinite(med) and med > 1e-6:
        thg = th[good]
        y = rt / med - 1.0
        A = np.c_[np.ones(len(thg)), np.cos(4 * thg), np.sin(4 * thg)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        var = float(np.var(y))
        r2_4 = float(max(0.0, 1.0 - np.var(res) / var)) if var > 1e-9 else 0.0
        sq_amp = float(np.hypot(coef[1], coef[2]))

    arc_ratio, arc_rms = np.nan, np.nan
    if good_dip.sum() >= 8:
        ang = th[good_dip]
        rtg = rtheta[good_dip]
        m = np.isfinite(rtg)
        if m.sum() >= 8:
            xy = np.c_[rtg[m] * np.cos(ang[m]), rtg[m] * np.sin(ang[m])]
            cx, cy, R, rms = _kasa_fit(xy)
            if np.isfinite(R) and R > 1e-6:
                arc_ratio, arc_rms = float(R / max(r_star, 1e-6)), rms

    of_i = ofb
    fl_in = float(np.nanmean(M[max(0, i - of_i - 2):max(1, i - of_i + 1)]))
    fl_out = float(np.nanmean(M[min(nr, i + of_i):min(nr, i + of_i + 3)]))
    seg = M[max(0, int(round(0.15 * r_star / dr))):max(1, int(round(0.55 * r_star / dr)))]
    interior = float(np.nanmedian(seg) - 0.5 * (fl_in + fl_out)) if np.isfinite(seg).any() else np.nan
    interior_rough = float(np.nanstd(seg)) if np.isfinite(seg).any() else np.nan
    bank_lo = min(nr - 1, i + of_i)
    bank_hi = min(nr, i + of_i + 1 + max(2, int(round(0.15 * r_star / dr))))
    bank = float(np.nanmax(M[bank_lo:bank_hi]) - fl_out) if bank_hi > bank_lo + 1 else np.nan
    bank_in = float(np.nanmax(M[max(0, i - of_i - 8):max(1, i - of_i)]) - fl_in) \
        if i - of_i - 8 > 0 else np.nan

    sig_bg, blank_band, band_frac = _band_stats(
        prep, row, col, r_star, cell, hz, wgt,
        width_m=(width_m if np.isfinite(width_m) else 9.0), p=p)
    depth = float(dip) if np.isfinite(dip) else np.nan
    out = dict(
        row=float(row), col=float(col), radius_m=r_star, diameter_m=2 * r_star,
        dip_m=depth, rel_std=rel_std, cov=cov, valid_frac=float(V[i]),
        in_dip_m=float(fl_in - M[i]) if np.isfinite(M[i]) else np.nan,
        out_dip_m=float(fl_out - M[i]) if np.isfinite(M[i]) else np.nan,
        asym_m=float(fl_out - fl_in),
        width_m=width_m, fwhm_m=width_m, depth_bg_m=depth_bg,
        bank_m=bank, bank_in_m=bank_in, interior_m=interior,
        interior_rough_m=interior_rough, circularity=circ, r_theta_median_m=med,
        gap_deg=gap_deg, square_r2_4=r2_4, square_periodicity=sq_amp,
        angle_frac=float(np.mean(valid_ray)), arc_fit_radius_ratio=arc_ratio,
        arc_fit_rms_m=arc_rms, n_rays=n_rays, n_rays_used=int(good.sum()),
        idx=int(idx), fwhm_bins=fwhm_bins, sig_bg_m=sig_bg,
        snr=(float(abs(dip)) / sig_bg if sig_bg and np.isfinite(sig_bg) and sig_bg > 1e-6
             else np.nan),
        blank_band=blank_band, band_frac=band_frac)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["two_sided"] = float(min(out["in_dip_m"], out["out_dip_m"]) / max(abs(depth), 1e-6)) \
            if np.isfinite(out["in_dip_m"]) and np.isfinite(out["out_dip_m"]) else np.nan
    out["shape"] = _shape(out)
    narrow = 1.0 / (1.0 + np.nan_to_num(width_m, nan=36.0) / 24.0)
    out["score"] = float(abs(depth) * min(1.0, cov / 0.85) * narrow
                         / (1.0 + 0.7 * np.nan_to_num(rel_std, nan=1.0)))
    out.update(_context(prep, row, col, r_star, cell))
    out["flags"] = _flags(out, p)
    out["_profile"] = dict(
        radius_m=rf.round(2).tolist(),
        ring_mean_m=[None if not np.isfinite(v) else round(float(v), 3) for v in M],
        per_ray_dip_m=[None if not np.isfinite(v) else round(float(v), 3) for v in dip_r],
        r_theta_m=[None if not np.isfinite(v) else round(float(v), 2) for v in rtheta])
    return out


def _band_stats(prep, row, col, r_m, cell, hz, wgt, width_m, p):
    """Local background roughness at the ring scale (excluding the ring band)
    and the fraction of the ditch band that cannot be inferred (holes/blanking)."""
    nx, ny = hz.shape
    rr = r_m / cell
    ys, xs = np.mgrid[max(0, int(row - 2.5 * rr)):min(nx, int(row + 2.5 * rr) + 1),
                      max(0, int(col - 2.5 * rr)):min(ny, int(col + 2.5 * rr) + 1)]
    if ys.size == 0 or xs.size == 0:
        return np.nan, np.nan, np.nan
    d = np.hypot(ys - row, xs - col) * cell
    w = wgt[ys, xs] > 0.5
    bg = w & (((d >= max(1.35 * r_m, r_m + 2 * width_m)) & (d <= 2.4 * r_m)) | (d <= 0.5 * r_m))
    sig = np.nan
    if bg.sum() > 20:
        v = hz[ys, xs][bg]
        med = np.median(v)
        sig = float(1.4826 * np.median(np.abs(v - med)))
    half = max(0.5 * width_m + 3.0, 0.35 * cell * 2)
    band = np.abs(d - r_m) <= half
    band_frac = float(band.mean()) if band.any() else np.nan
    if band.any():
        hole = np.asarray(prep["hole_dist_m"])[ys, xs]
        blank = band & ((hole < prep["cfg"]["hole_blank_m"]) | ~w)
        blank_band = float(blank.sum() / band.sum())
    else:
        blank_band = np.nan
    return sig, blank_band, band_frac


def _shape(d):
    """Shape class from per-ray trough radius and angular statistics."""
    cov, circ = d["cov"], d["circularity"]
    r24, gapl = d["square_r2_4"], d["gap_deg"]
    if r24 >= 0.55 and cov >= 0.55:
        return "square/angular"
    if cov >= 0.65 and circ >= 0.40 and r24 < 0.55:
        return "circle"
    if cov >= 0.30 and gapl >= 60.0:
        return "U/arc"
    return "irregular"


def _context(prep, row, col, r_m, cell):
    nx, ny = prep["h"].shape
    rr = r_m / cell
    ys, xs = np.mgrid[max(0, int(row - rr) - 1):min(nx, int(row + rr) + 2),
                      max(0, int(col - rr) - 1):min(ny, int(col + rr) + 2)]
    d = np.hypot(ys - row, xs - col) * cell
    out = {}
    for name, arr in (("hole_dist_m", prep["hole_dist_m"]), ("rough", prep["rough"]),
                      ("canopy", prep["canopy"])):
        out["ctx_" + name + "_disc"] = np.nan
        out["ctx_" + name + "_ann"] = np.nan
        if arr is None:
            continue
        v = np.asarray(arr)[ys, xs].astype(np.float64)
        disc = v[d <= r_m]
        ann = v[(d >= 0.85 * r_m) & (d <= 1.15 * r_m)]
        out["ctx_" + name + "_disc"] = float(np.nanmedian(disc)) if np.isfinite(disc).any() else np.nan
        out["ctx_" + name + "_ann"] = float(np.nanmedian(ann)) if np.isfinite(ann).any() else np.nan
    hole = np.asarray(prep["hole_dist_m"])[ys, xs]
    out["hole_frac_in_disc"] = float(np.mean(hole[d <= r_m] <= 0.0)) if d.size else np.nan
    annsel = (d >= 0.75 * r_m) & (d <= 1.25 * r_m)
    out["ann_min_hole_dist_m"] = float(np.min(hole[annsel])) if annsel.any() else np.nan
    out["edge_dist_m"] = float(min(min(row, nx - 1 - row), min(col, ny - 1 - col)) * cell)
    out["edge_clipped"] = bool(out["edge_dist_m"] < 1.1 * r_m)
    return out


def _flags(d, p):
    f = []
    if abs(d["dip_m"]) < p["min_dip_m"]:
        f.append("shallow_dip")
    if np.isfinite(d["width_m"]) and d["width_m"] > p["max_width_m"]:
        f.append("wide_channel_gt_%.0fm" % p["max_width_m"])
    if np.isfinite(d["fwhm_m"]) and d["fwhm_m"] > p["max_fwhm_m"]:
        f.append("broad_concavity")
    if np.isfinite(d["rel_std"]) and d["rel_std"] > p["max_rel_std"]:
        f.append("inconsistent_depth_around_ring")
    if np.isfinite(d["interior_m"]) and d["interior_m"] < -0.5:
        f.append("basin_interior_pond_like")
    if np.isfinite(d["interior_m"]) and abs(d["interior_m"]) > 0.9:
        f.append("interior_not_level")
    if np.isfinite(d["asym_m"]) and abs(d["asym_m"]) > 0.6 * max(abs(d["dip_m"]), 1e-6):
        f.append("asymmetric_cross_section")
    if d["cov"] < 0.6:
        f.append("arc_only")
    if d["gap_deg"] >= 60.0:
        f.append("angular_gap")
    if d["edge_clipped"]:
        f.append("edge_clipped")
    if np.isfinite(d["blank_band"]) and d["blank_band"] > 0.5:
        f.append("hole_blanked_band")
    if np.isfinite(d["snr"]) and d["snr"] < p["min_snr"]:
        f.append("low_snr_vs_background")
    if np.isfinite(d["ctx_rough_ann"]) and d["ctx_rough_ann"] > 0.35:
        f.append("rough_surface_canopy")
    if np.isfinite(d["arc_fit_radius_ratio"]) and not (0.70 <= d["arc_fit_radius_ratio"] <= 1.45):
        f.append("arc_not_circular")
    return f


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def detect_rings(prep, params=None, verbose=False):
    p = dict(DEFAULTS, **(params or {}))
    cell = float(prep["cell"])
    hz, wgt = _fields(prep, p)
    radii_cells = np.arange(max(2.0, round(0.4 * p["rmin_m"] / cell)),
                            p["rmax_m"] / cell + 12, p["dr_cells"])
    centers, stride = candidate_centers(prep, p["stride_cells"])
    of = int(p["flank_of_cells"])
    cand = []
    step = int(p["chunk_centers"])
    for s0 in range(0, len(centers), step):
        ch = centers[s0:s0 + step]
        cube, cw = sample_polar(hz, wgt, ch, radii_cells, p["n_rays"])
        res = radial_profiles(cube, cw, of)
        sel = _stage_a_select(res, radii_cells, cell, p)
        for k in np.nonzero(sel["gate"])[0]:
            cand.append(dict(row=float(ch[k, 0]), col=float(ch[k, 1]), idx=int(sel["idx"][k]),
                             dip=float(sel["dip"][k]), cov=float(sel["cov"][k]),
                             V=float(sel["V"][k]), z=float(sel["z"][k]),
                             two_sided=float(sel["two_sided"][k]),
                             score=float(sel["dip"][k] * min(1.0, sel["cov"][k] / 0.85))))
        if verbose:
            print("  ring stage A %d/%d centres, %d hits"
                  % (min(s0 + step, len(centers)), len(centers), len(cand)), flush=True)
    if not cand:
        return []
    cand.sort(key=lambda d: -d["score"])
    # Exhaustive refinement is intentional: coarse-NMS and a 300-candidate cap
    # were tested but lost spatially distinct detections. See coordinator log.
    pre = [c for c in cand if c["V"] >= p["min_valid_frac"]][: int(p["max_refine"])]
    arc = [c for c in cand if c["V"] < p["min_valid_frac"]][: max(6, int(p["max_refine"]) // 2)]
    dets = []
    for c in pre + arc:
        best = None
        span = int(p["refine_cells"]) if p["refine"] else 0
        for dy in range(-span, span + 1):
            for dx in range(-span, span + 1):
                row, col = int(c["row"]) + dy, int(c["col"]) + dx
                if not (0 <= row < hz.shape[0] and 0 <= col < hz.shape[1]):
                    continue
                if not prep["infer"][row, col]:
                    continue
                m = candidate_metrics(prep, row, col, c["idx"], radii_cells, cell, p)
                if not _stage_b_gate(m, p):
                    continue
                if best is None or m["score"] > best["score"]:
                    best = m
        if best is not None:
            best["coarse_row"], best["coarse_col"] = c["row"], c["col"]
            best["coarse_dip"] = c["dip"]
            dets.append(best)
    dets.sort(key=lambda d: -d["score"])
    dets = _nms(dets, cell, p)
    for n, d in enumerate(dets[: int(p["max_dets"])]):
        d["det_index"] = n
    return dets[: int(p["max_dets"])]


def _stage_b_gate(m, p):
    if not np.isfinite(m["dip_m"]) or m["dip_m"] < p["min_dip_m"]:
        return False
    if np.isfinite(m["rel_std"]) and m["rel_std"] > p["max_rel_std"]:
        return False
    if np.isfinite(m["fwhm_m"]) and m["fwhm_m"] > p["max_fwhm_m"]:
        return False
    if m["cov"] < p["arc_min_cov"]:
        return False
    if np.isfinite(m["interior_m"]) and m["interior_m"] < -0.75:
        return False
    if np.isfinite(m["width_m"]) and m["width_m"] > p["max_width_m"]:
        return False
    if np.isfinite(m["two_sided"]) and m["two_sided"] < p["max_two_sided_fail"]:
        return False
    if m["cov"] < p["min_cov"]:
        if not (np.isfinite(m["arc_fit_radius_ratio"])
                and 1.0 - p["max_arc_radius_ratio"] <= m["arc_fit_radius_ratio"]
                <= 1.0 + p["max_arc_radius_ratio"]):
            return False
    return True


def _nms(dets, cell, p):
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            dist = math.hypot(d["row"] - k["row"], d["col"] - k["col"]) * cell
            r1, r2 = d["radius_m"], k["radius_m"]
            ratio = r1 / max(r2, 1e-6)
            if dist < max(15.0, p["nms_dist_frac"] * max(r1, r2)) and \
                    (1.0 / p["nms_merge_frac"] < ratio < p["nms_merge_frac"]):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept
