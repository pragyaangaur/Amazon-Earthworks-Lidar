import json
import math
import os
from collections import deque

import numpy as np
from scipy import ndimage as ndi

import dtlib as D

DEFAULTS = dict(
    smooth_m=3.0,
    ridge_scales_m=(3.5, 5.0),
    ridge_floor=0.0040,        # 1/m, absolute curvature floor (~0.35 m crest)
    ridge_seed=0.0070,         # 1/m, hysteresis seed
    planarity_min=0.25,
    relief_win_m=15.0,
    relief_min_m=0.22,         # crest must stand this far above the 15 m minimum
    nms_step_m=1.2,
    dir_smooth_cells=1,
    seg_tol_deg=25.0,
    min_seg_len_m=55.0,
    straight_dev_max_m=3.0,    # 90th pct deviation from the fitted axis
    prof_step_m=6.0,
    prof_half_m=25.0,
    prof_step_fine_m=1.0,
    height_range_m=(0.35, 5.0),
    width_range_m=(2.5, 15.0),   # archaeological embankments are 3-15 m wide
    sym_max=0.65,
    sym_w_max=0.62,
    continuity_min=0.6,
    h_iqr_max_frac=0.35,       # IQR of crest height / median crest height
                               # (a causeway is a *level* embankment; natural ridges taper)
    width_cv_max=0.55,
    acc_min_m2=2500.0,
    opening_cells=11,        # 33 m grey opening: ridges narrower than this are removed
    cross_angle_min_deg=35.0,
    approach_min_m=20.0,
    causeway_width_max_m=12.0,  # calibrated: natural ridges here are 13-15 m, causeways 7-11 m
    causeway_asym_max=0.35,     # calibrated: causeways <=0.27, natural ridge 0.40-0.52
    max_gap_m=12.0,
    hole_fill_cells=5,
    mask_hole_buffer_m=6.0,
    dedupe_dist_m=35.0,
    line_win_m=60.0,          # line-integral window (m): the length a real crest keeps
    radon_step_deg=2.0,
    radon_thr=0.0040,         # along-line mean of the ridge measure (1/m)
    radon_ext_factor=0.55,    # extend the peak until the mean falls to this * thr
    radon_min_len_m=70.0,     # generator gate: shorter lines are not causeway candidates
    radon_min_support=0.45,   # fraction of along-line samples above the ridge floor
    max_candidates=400,       # cap on lines sent to profile assessment (runtime guard)
    merge_gap_max_m=40.0,
    side_touch_m=25.0,
    drain_near_m=25.0,
    cross_pair_win_m=30.0,
    one_frac_max=0.7,
    touch_frac_max=0.95,
)


# --------------------------------------------------------------------------
# line-integral (Radon) search over the ridge-measure field
# --------------------------------------------------------------------------
def _radon_pad(a, pad_value=0.0):
    """Pad to a square big enough that no rotation crops the data."""
    n, m = a.shape
    S = int(np.ceil(np.hypot(n, m))) + 2
    out = np.full((S, S), pad_value, np.float32)
    oi = (S - n) // 2
    oj = (S - m) // 2
    out[oi:oi + n, oj:oj + m] = a
    return out, oi, oj, S


def _rot_back(i_r, j_r, deg, S):
    """Inverse of scipy.ndimage.rotate's coordinate map for a square of side S."""
    th = math.radians(deg)
    c = (S - 1) / 2.0
    di = i_r - c
    dj = j_r - c
    i0 = c + np.cos(th) * di + np.sin(th) * dj
    j0 = c - np.sin(th) * di + np.cos(th) * dj
    return i0, j0


def _radon_lines(g, meta, p):
    """Search straight lines in the ridge-measure field.

    For every angle the ridge-measure field is rotated so that lines of that
    orientation become rows, then averaged along each row over `line_win_m`.  A
    straight crest holds its curvature along the whole window; canopy-noise humps
    average away.  Runs above `radon_thr` are extracted with a label pass (no Python
    loop over pixels), their extent is taken from the lower `radon_ext_factor`
    threshold, and each run becomes a candidate line.

    Returns candidates in world coordinates, before profile verification.
    """
    dx = meta["dx"]
    ok = g["wf"] & (g["dist_to_hole_m"] >= p["mask_hole_buffer_m"])
    meas = np.where(ok & np.isfinite(g["meas"]), np.clip(g["meas"], 0, None), 0.0).astype(np.float32)
    val = ok.astype(np.float32)
    mp, oi, oj, S = _radon_pad(meas, 0.0)
    vp, _, _, _ = _radon_pad(val, 0.0)
    kw = max(3, int(round(p["line_win_m"] / dx)))
    horiz = np.zeros((3, 3), bool)
    horiz[1, :] = True
    cands = []
    thr_hi = p["radon_thr"]
    thr_lo = p["radon_ext_factor"] * p["radon_thr"]
    for deg in np.arange(0.0, 180.0, p["radon_step_deg"]):
        Rm = ndi.rotate(mp, angle=deg, reshape=False, order=1, prefilter=False,
                        mode="constant", cval=0.0)
        Rv = ndi.rotate(vp, angle=deg, reshape=False, order=1, prefilter=False,
                        mode="constant", cval=0.0)
        Ssum = ndi.uniform_filter1d(Rm, kw, axis=1) * kw
        Scnt = ndi.uniform_filter1d(Rv, kw, axis=1) * kw
        good = Scnt >= 0.75 * kw
        Smean = np.where(good, Ssum / np.maximum(Scnt, 1e-6), 0.0)
        hi = good & (Smean >= thr_hi)
        if not hi.any():
            continue
        lo = good & (Smean >= thr_lo)
        lab_hi, n_hi = ndi.label(hi, structure=horiz)
        lab_lo, n_lo = ndi.label(lo, structure=horiz)
        objs_lo = ndi.find_objects(lab_lo)
        cols_lo = {}
        for k, sl in enumerate(objs_lo):
            if sl is not None:
                cols_lo[k + 1] = (sl[1].start, sl[1].stop - 1)
        pk = ndi.maximum_position(np.where(hi, Smean, -np.inf), lab_hi,
                                  np.arange(1, n_hi + 1))
        for k, (i_r, j_pk) in enumerate(pk):
            if not np.isfinite(Smean[i_r, j_pk]):
                continue
            lab_l = lab_lo[i_r, j_pk]
            if lab_l == 0:
                j_lo = j_hi = j_pk
            else:
                j_lo, j_hi = cols_lo.get(int(lab_l), (j_pk, j_pk))
            L = (j_hi - j_lo + 1) * dx
            if L < p["min_seg_len_m"]:
                continue
            i0, j0 = _rot_back(float(i_r), 0.5 * (j_lo + j_hi), deg, S)
            xw, yw = D.index_to_world(i0 - oi, j0 - oj, meta)
            ang_w = (-math.radians(deg)) % math.pi
            seg = Rm[i_r, j_lo:j_hi + 1]
            segv = Rv[i_r, j_lo:j_hi + 1] > 0.5
            supp = float(np.mean(seg[segv] >= p["ridge_floor"])) if segv.any() else 0.0
            cands.append(dict(theta_deg=float(deg), x=float(xw), y=float(yw),
                              axis_angle_deg=math.degrees(ang_w) % 180.0,
                              length_m=float(L), strength=float(Smean[i_r, j_pk]),
                              support=supp,
                              amp=float(np.mean(seg[segv])) if segv.any() else 0.0))
    return cands


def _same_feature(a, b, p):
    """True when two detections describe the same physical ridge."""
    d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    da = abs(math.sin(math.radians(a["axis_angle_deg"] - b["axis_angle_deg"])))
    if d < p["dedupe_dist_m"] and da < math.sin(math.radians(12.0)):
        return True
    if da < math.sin(math.radians(10.0)):
        th = math.radians(b["axis_angle_deg"])
        proj = (a["x"] - b["x"]) * math.cos(th) + (a["y"] - b["y"]) * math.sin(th)
        perp = -(a["x"] - b["x"]) * math.sin(th) + (a["y"] - b["y"]) * math.cos(th)
        if abs(perp) < 10.0 and abs(proj) < 0.5 * (a["length_m"] + b["length_m"]) + \
                p["merge_gap_max_m"]:
            return True
    return False


def _dedupe_cands(cands, p):
    """Greedy non-maximum suppression on (position, orientation)."""
    out = []
    for c in sorted(cands, key=lambda q: -(q["strength"] * q["length_m"])):
        if not any(_same_feature(c, o, p) for o in out):
            out.append(c)
    return out


def self_test_rotation(tol=0.6):
    """Verify _rot_back against scipy.ndimage.rotate.  Returns max error in cells."""
    S = 81
    c = (S - 1) / 2.0
    err = 0.0
    for deg in (0.0, 17.0, 45.0, 90.0, 133.0):
        a = np.zeros((S, S))
        i, j = 22, 58
        a[i, j] = 1.0
        r = ndi.rotate(a, angle=deg, reshape=False, order=1, prefilter=False)
        i_r, j_r = np.unravel_index(np.argmax(r), r.shape)
        i0, j0 = _rot_back(float(i_r), float(j_r), deg, S)
        err = max(err, abs(i0 - i), abs(j0 - j))
    if err > tol:
        raise AssertionError(f"rotation mapping wrong: max error {err:.2f} cells")
    return err


# --------------------------------------------------------------------------
def _prep(z, meta, w, p):
    dx = meta["dx"]
    if w is None:
        w = D.valid_mask(z, meta)
    w = np.asarray(w, bool) & D.valid_mask(z, meta)
    z = np.asarray(z, np.float32).copy()
    wf = D.fill_holes_small(w, p["hole_fill_cells"])
    zf = z.copy()
    todo = wf & ~w
    if (~wf).any() and todo.any():
        ind = ndi.distance_transform_edt(~wf, return_distances=False, return_indices=True)
        zf = np.where(todo, z[tuple(ind)], z)
    base = float(np.nanmedian(z[w])) if w.any() else 0.0
    zs = D.mask_gauss(zf, wf, p["smooth_m"] / dx, fill=base)
    return dict(z=z, zf=zf, zs=zs, w=w, wf=wf, dx=dx, dy=meta["dy"], base=base)


def _hydrology(g, meta, p):
    """Drainage network from a *ridge-removed* surface.

    A raised embankment dams a D8 flow net: water computed on the raw (or filled) DTM
    is deflected around the causeway and the stream is never seen to cross it.  A grey
    opening removes positive features narrower than `opening_cells` (banks, levees,
    causeways, spoil) while preserving channels and broad landforms, so the drainage
    line passes *under* the causeway, which is the physical situation we want to test
    against.  This is stated as a design choice, not hidden.
    """
    dx, dy = g["dx"], g["dy"]
    k = int(p["opening_cells"])
    lowest = float(np.nanmin(g["zs"][g["wf"]]))
    zfor_open = np.where(g["wf"], g["zs"], lowest)
    zopen = ndi.grey_opening(zfor_open, size=k)
    zopen = np.where(g["wf"], zopen, np.nan)
    zfillable = zopen
    zf = D.fill_depressions(zfillable, g["wf"])
    g["z_ridge_removed"] = zopen
    acc, down = D.flow_accumulation(zf, g["wf"], dx, dy)
    thr = p["acc_min_m2"] / (dx * dy)
    drain = np.isfinite(acc) & (acc >= thr)
    g.update(filled=zf, acc=acc, down=down, drain=drain,
             drain_frac=float(np.mean(drain[g["wf"]])), acc_thr_cells=thr)
    return g


def _ridge_fields(g, meta, p):
    dx, dy = g["dx"], g["dy"]
    best = np.full(g["zs"].shape, -np.inf, np.float32)
    plan_b = np.zeros(g["zs"].shape, np.float32)
    ang_b = np.full(g["zs"].shape, np.nan, np.float32)
    meas_by_scale = {}
    for sg in p["ridge_scales_m"]:
        meas, plan = D.ridge_measure(g["zs"], g["wf"], sigma=sg / dx, dx=dx, dy=dy,
                                     fill=g["base"])
        ang = D.hessian_dir(g["zs"], g["wf"], sg / dx, dx, dy)
        meas_by_scale[sg] = meas
        upd = np.isfinite(meas) & (meas > best)
        best = np.where(upd, meas, best)
        plan_b = np.where(upd, plan, plan_b)
        ang_b = np.where(upd, ang, ang_b)
    best[~g["wf"]] = -np.inf
    # circular smoothing of the along-ridge direction (mean of doubled angle)
    c = ndi.gaussian_filter(np.where(np.isfinite(ang_b), np.cos(2 * ang_b), 0), 1.0)
    s = ndi.gaussian_filter(np.where(np.isfinite(ang_b), np.sin(2 * ang_b), 0), 1.0)
    ang_s = 0.5 * np.arctan2(s, c)
    g.update(meas=best, planarity=plan_b, ang=ang_s, meas_by_scale=meas_by_scale)
    # local relief above the 15 m minimum
    k = max(3, int(round(p["relief_win_m"] / dx)))
    g["relief15"] = g["zs"] - ndi.minimum_filter(np.where(g["wf"], g["zs"], np.inf), size=k)
    g["dist_to_hole_m"] = ndi.distance_transform_edt(g["wf"]) * min(dx, dy)
    if g.get("drain") is not None and g["drain"].any():
        dd, (di, dj) = ndi.distance_transform_edt(~g["drain"], return_indices=True)
        g["dist_to_drain_m"] = dd * min(dx, dy)
        g["drain_near_i"], g["drain_near_j"] = di, dj
    else:
        g["dist_to_drain_m"] = np.full(g["zs"].shape, np.inf, np.float32)
    return g


def _nms(meas, ang, meta, step_m):
    """Keep cells that are local maxima across the ridge (1-cell crest lines)."""
    X, Y = D.cell_centres(meas.shape, meta)
    nx, ny = -np.sin(ang), np.cos(ang)          # across-ridge unit vector
    m1 = D.bilinear(np.where(np.isfinite(meas), meas, np.nan), X + nx * step_m,
                    Y + ny * step_m, meta)
    m2 = D.bilinear(np.where(np.isfinite(meas), meas, np.nan), X - nx * step_m,
                    Y - ny * step_m, meta)
    m1 = np.where(np.isfinite(m1), m1, -np.inf)
    m2 = np.where(np.isfinite(m2), m2, -np.inf)
    return np.isfinite(meas) & (meas >= m1) & (meas >= m2)


def _segments_by_dir(line, ang, tol_deg):
    """Grow 8-connected segments keeping the direction within tol of the running mean."""
    lab = np.zeros(line.shape, np.int32)
    n, m = line.shape
    ii, jj = np.nonzero(line)
    tol = math.radians(tol_deg)
    cur = 0
    for i0, j0 in zip(ii.tolist(), jj.tolist()):
        if lab[i0, j0]:
            continue
        cur += 1
        lab[i0, j0] = cur
        q = deque([(i0, j0)])
        a_c, a_s, cnt = 0.0, 0.0, 0
        while q:
            i, j = q.popleft()
            if i < 0 or i >= n or j < 0 or j >= m or not line[i, j] or lab[i, j] != cur:
                continue
            a = ang[i, j]
            if np.isfinite(a):
                a_c += math.cos(2 * a); a_s += math.sin(2 * a); cnt += 1
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < m and line[ni, nj] and not lab[ni, nj]:
                        b = ang[ni, nj]
                        if cnt >= 3 and np.isfinite(b):
                            ref = 0.5 * math.atan2(a_s, a_c)
                            d = abs((b - ref + math.pi / 2) % math.pi - math.pi / 2)
                            if d > tol:
                                continue
                        lab[ni, nj] = cur
                        q.append((ni, nj))
    return lab, cur


# --------------------------------------------------------------------------
def _profiles(z, meta, cx, cy, axis_ang, p, depth=0):
    nrm = axis_ang + math.pi / 2
    s = np.arange(-p["prof_half_m"], p["prof_half_m"] + 1e-9, p["prof_step_fine_m"])
    xs = cx + np.cos(nrm) * s
    ys = cy + np.sin(nrm) * s
    prof = D.bilinear(z, xs, ys, meta)
    ok = np.isfinite(prof)
    if ok.mean() < 0.8:
        return dict(valid=False, valid_frac=float(ok.mean()))
    win = np.abs(s) <= 6.0
    if np.isfinite(prof[win]).any():
        off = float(s[win][np.nanargmax(prof[win])])
        if abs(off) > p["prof_step_fine_m"] and depth < 3:
            return _profiles(z, meta, cx + np.cos(nrm) * off, cy + np.sin(nrm) * off,
                             axis_ang, p, depth + 1)
    crest = float(np.nanmedian(prof[np.abs(s) <= 2.0]))

    def shoulder(sign):
        m = (sign * s >= 3.0) & (sign * s <= 12.0)
        v = prof[m]
        return float(np.nanmin(v)) if np.isfinite(v).any() else np.nan

    zl, zr = shoulder(-1), shoulder(+1)
    if not (np.isfinite(zl) and np.isfinite(zr)):
        return dict(valid=False, valid_frac=float(ok.mean()))
    h = crest - 0.5 * (zl + zr)
    hl, hr = crest - zl, crest - zr
    out = dict(valid=True, valid_frac=float(ok.mean()), crest=crest, zl=zl, zr=zr,
               h=float(h), hl=float(hl), hr=float(hr),
               asym=float(abs(hl - hr) / max(abs(h), 1e-6)))
    lvl = crest - 0.5 * h
    if h > 1e-3:
        li = np.nonzero((s < 0) & (prof <= lvl) & np.isfinite(prof))[0]
        ri = np.nonzero((s > 0) & (prof <= lvl) & np.isfinite(prof))[0]
        out["w50"] = float(s[ri[0]] - s[li[-1]]) if li.size and ri.size else np.nan
        # half-widths each side: shape symmetry, independent of the regional slope
        out["wl"] = float(-s[li[-1]]) if li.size else np.nan
        out["wr"] = float(s[ri[0]]) if ri.size else np.nan
        if np.isfinite(out["wl"]) and np.isfinite(out["wr"]) and (out["wl"] + out["wr"]) > 1e-6:
            out["asym_w"] = float(abs(out["wl"] - out["wr"]) / (out["wl"] + out["wr"]))
        else:
            out["asym_w"] = np.nan
    else:
        out["w50"] = np.nan
        out["asym_w"] = np.nan
    cw = np.abs(s) <= 3.5
    out["crown_flat"] = float(np.nanmean(prof[cw] > crest - 0.10 * max(abs(h), 1e-6)))
    far = np.r_[prof[(s >= 16) & (s <= 22)], prof[(s <= -16) & (s >= -22)]]
    out["flank_dip"] = (float(max(0.0, np.nanmedian(far) - 0.5 * (zl + zr)))
                        if np.isfinite(far).any() else 0.0)
    bad = 0
    for sign in (1, -1):
        seg = prof[(sign * s >= 1.0) & (sign * s <= 9.0)]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 4:
            bad += int(np.sum(np.diff(seg[::max(1, seg.size // 5)]) > 0.35 * max(abs(h), 0.05)))
    out["falloff_viol"] = bad
    out["prof"] = prof.tolist()
    out["s"] = s.tolist()
    return out


def _crossing(g, meta, cx, cy, ang, L, crest_mask, p):
    """Drainage geometry in the candidate's own axis frame.

    A causeway crosses a watercourse, so drainage ground lies close to the axis on
    BOTH sides at the SAME longitudinal position, and the crest survives on both banks.
    A levee runs along a watercourse: its drainage is one-sided and/or the axis stays
    close to drainage for a large fraction of its length.

    Returns
      left_min_m / right_min_m : closest drainage on the -perp / +perp side at the crossing
      t_offset_m               : longitudinal offset between those two nearest cells
      both_sides               : True when drainage is close on both sides at one place
      crossing_t_m             : axis coordinate of that place
      touch_frac               : fraction of the crest span with drainage within 20 m
      touch_len_m              : longest such stretch;  crossings: number of stretches
      crest_left_m/right_m     : crest extent each side of the crossing
      angle_deg                : angle between the axis and the local D8 flow
    """
    near_m = p["drain_near_m"]
    toe_w = p["cross_pair_win_m"]
    half = L / 2 + 25.0
    s = np.arange(-half, half + 1e-9, 1.0)
    xs = cx + np.cos(ang) * s
    ys = cy + np.sin(ang) * s
    out = dict(crossings=0, angle_deg=None, touch_len_m=0.0, touch_frac=0.0,
               d_min_m=float("nan"), left_min_m=float("nan"), right_min_m=float("nan"),
               t_offset_m=float("nan"), crossing_t_m=float("nan"), both_sides=False,
               both_frac=0.0, one_frac=0.0, crest_left_m=0.0, crest_right_m=0.0)
    if not g["drain"].any():
        return out
    fi = np.clip(np.round((meta["y0"] - ys) / meta["dy"] - 0.5).astype(int), 0,
                 g["drain"].shape[0] - 1)
    fj = np.clip(np.round((xs - meta["x0"]) / meta["dx"] - 0.5).astype(int), 0,
                 g["drain"].shape[1] - 1)
    di, dj = g["drain_near_i"][fi, fj], g["drain_near_j"][fi, fj]
    xd, yd = D.index_to_world(di, dj, meta)
    d = np.hypot(xd - xs, yd - ys)
    t_d = (xd - cx) * np.cos(ang) + (yd - cy) * np.sin(ang)
    p_d = -(xd - cx) * np.sin(ang) + (yd - cy) * np.cos(ang)
    if np.isfinite(d).any():
        out["d_min_m"] = float(np.nanmin(d))
    tt = np.isfinite(d) & (d <= near_m)
    out["touch_frac"] = float(tt.mean())
    runs, cur, best = 0, 0, 0
    for v in tt:
        if v:
            cur += 1; best = max(best, cur)
        else:
            if cur:
                runs += 1
            cur = 0
    if cur:
        runs += 1
    out["crossings"] = int(runs)
    out["touch_len_m"] = float(best)
    # rasterise drainage presence per side on the axis sample grid
    north = np.zeros(s.size, bool)
    south = np.zeros(s.size, bool)
    near = np.isfinite(d) & (d <= near_m)
    kk = np.clip(np.round(t_d + half).astype(int), 0, s.size - 1)
    north[kk[near & (p_d >= 4.0)]] = True
    south[kk[near & (p_d <= -4.0)]] = True
    w = max(1, int(round(toe_w)))
    nb = ndi.binary_dilation(north, np.ones(2 * w + 1, bool))
    sb = ndi.binary_dilation(south, np.ones(2 * w + 1, bool))
    both = nb & sb
    out["both_frac"] = float(both.mean())
    out["one_frac"] = float((nb ^ sb).mean())
    if both.any():
        out["both_sides"] = True
        runs = np.nonzero(np.diff(np.r_[0, both.astype(int), 0]) == 1)[0]
        out["crossings"] = int(len(runs))
        inside = both & (np.abs(s) <= L / 2)
        cand = np.nonzero(inside)[0] if inside.any() else np.nonzero(both)[0]
        k = int(cand[np.argmin(np.abs(s[cand]))])
        out["crossing_t_m"] = float(s[k])
        win = np.abs(s - s[k]) <= 2 * toe_w + 5.0
        for side, store in ((p_d >= 4.0, "right_min_m"), (p_d <= -4.0, "left_min_m")):
            m = near & side & win
            if m.any():
                out[store] = float(np.nanmin(np.where(m, d, np.inf)))
        if np.isfinite(out["left_min_m"]) and np.isfinite(out["right_min_m"]):
            out["t_offset_m"] = 0.0
        dmin_i = k
    else:
        if not tt.any():
            return out
        dmin_i = int(np.nanargmin(d))
        out["crossing_t_m"] = float(s[dmin_i])
    # NOTE: crest continuation on both banks is measured from the *profiles* in
    # _assess_line, not from the thresholded crest mask: the mask is fragmented by
    # noise and would under-count continuity of a real embankment.
    i, j = int(fi[dmin_i]), int(fj[dmin_i])
    m_ = g["drain"].shape[1]
    dn = int(g["down"][i * m_ + j])
    if dn >= 0:
        i2, j2 = divmod(dn, m_)
        da = math.atan2(-(i2 - i), (j2 - j))
        diff = abs(math.degrees(ang - da)) % 180.0
        out["angle_deg"] = float(min(diff, 180.0 - diff))
    return out


def _modern_flags(agg, L, straight_dev, h, w50):
    """Indicators of a modern (or non-archaeological) origin.  Flags, not proofs."""
    f = []
    if agg["crown_flat_med"] > 0.75 and w50 < 9:
        f.append("flat_planar_crown")
    if agg["flank_dip_med"] > 0.12:
        f.append("flanking_borrow_ditches")
    if L > 70 and straight_dev < 0.9:
        f.append("dead_straight")
    if h < 0.8 and w50 < 8:
        f.append("low_and_narrow_tracklike")
    return f


# --------------------------------------------------------------------------
def _assess_line(g, meta, x, y, ang, L, crest, p, chm=None):
    """Verify one candidate line with perpendicular profiles and a crossing test."""
    dx = meta["dx"]
    ss = np.arange(-L / 2.0 + 2.0, L / 2.0 - 2.0 + 1e-9, p["prof_step_m"])
    rec = dict(x=float(x), y=float(y), length_m=float(L),
               axis_angle_deg=float(math.degrees(ang) % 180.0), n_profiles=0,
               reject=None)
    if ss.size < 4:
        rec["reject"] = "too_short"
        return rec, []
    profs = []
    for t in ss:
        pr = _profiles(g["z"], meta, x + math.cos(ang) * t, y + math.sin(ang) * t, ang, p)
        pr["t"] = float(t)
        profs.append(pr)
    good = [q for q in profs if q.get("valid") and np.isfinite(q.get("h", np.nan))]
    rec["n_profiles"] = int(len(profs))
    rec["continuity"] = float(len(good) / max(len(profs), 1))
    if len(good) < 3:
        rec["reject"] = "no_usable_profiles"
        return rec, profs
    hs = np.array([q["h"] for q in good], float)
    w50s = np.array([q.get("w50", np.nan) for q in good], float)
    rec.update(h_med=float(np.median(hs)),
               h_iqr=float(np.subtract(*np.percentile(hs, [75, 25]))),
               crest_width_m=float(np.nanmedian(w50s)) if np.isfinite(w50s).any() else float("nan"),
               width_cv=float(np.nanstd(w50s) / max(np.nanmedian(w50s), 1e-6)),
               asym_med=float(np.median([q["asym"] for q in good])),
               asym_w_med=float(np.nanmedian([q.get("asym_w", np.nan) for q in good])),
               crown_flat_med=float(np.median([q["crown_flat"] for q in good])),
               flank_dip_med=float(np.median([q["flank_dip"] for q in good])),
               falloff_viol=int(np.sum([q["falloff_viol"] for q in good])))
    if not (p["height_range_m"][0] <= rec["h_med"] <= p["height_range_m"][1]):
        rec["reject"] = "height_out_of_range"
        return rec, profs
    if not np.isfinite(rec["crest_width_m"]) or not (
            p["width_range_m"][0] <= rec["crest_width_m"] <= p["width_range_m"][1]):
        rec["reject"] = "width_out_of_range"
        return rec, profs
    if not np.isfinite(rec["asym_w_med"]) or rec["asym_w_med"] > p["sym_w_max"]:
        rec["reject"] = "asymmetric_flanks"
        return rec, profs
    if rec["continuity"] < p["continuity_min"]:
        rec["reject"] = "discontinuous"
        return rec, profs
    if rec["h_iqr"] > p["h_iqr_max_frac"] * max(rec["h_med"], 1e-6):
        rec["reject"] = "height_not_constant"
        return rec, profs
    if rec["width_cv"] > p["width_cv_max"]:
        rec["reject"] = "width_not_constant"
        return rec, profs
    cross = _crossing(g, meta, x, y, ang, L, crest, p)
    # crest continuation: profiles either side of the crossing that still show at
    # least half the median crest height (metres of profile support on each bank)
    tc = cross["crossing_t_m"]
    strong = [(q["t"], q["h"]) for q in good if q["h"] >= 0.5 * rec["h_med"]]
    left_m = sum(1 for t, h in strong if -L / 2 <= t <= tc - 5.0) * p["prof_step_m"]
    right_m = sum(1 for t, h in strong if tc + 5.0 <= t <= L / 2) * p["prof_step_m"]
    cross["crest_left_m"] = float(left_m)
    cross["crest_right_m"] = float(right_m)
    rec.update(crossings=cross["crossings"], cross_angle_deg=cross["angle_deg"],
               touch_len_m=cross["touch_len_m"], touch_frac=cross["touch_frac"],
               d_min_m=cross["d_min_m"], left_min_m=cross["left_min_m"],
               right_min_m=cross["right_min_m"], t_offset_m=cross["t_offset_m"],
               both_sides=cross["both_sides"], both_frac=cross.get("both_frac", 0.0),
               one_frac=cross.get("one_frac", 0.0),
               crest_left_m=cross["crest_left_m"], crest_right_m=cross["crest_right_m"])
    st = p["side_touch_m"]
    # Decision rule: chosen for precision on the synthetic controls (results/calibration.md).
    # A causeway must (a) have drainage close on both sides at one place, (b) keep its
    # crest on both banks, (c) be a narrow crest, and (d) have symmetric flanks.  The
    # width and flank-symmetry limits are what separate it from a natural ridge and
    # from the paired banks of a straight stream.
    is_causeway = bool(
        cross["both_sides"] and
        np.isfinite(cross["left_min_m"]) and cross["left_min_m"] <= st and
        np.isfinite(cross["right_min_m"]) and cross["right_min_m"] <= st and
        min(cross["crest_left_m"], cross["crest_right_m"]) >= p["approach_min_m"] and
        rec["crest_width_m"] <= p["causeway_width_max_m"] and
        rec["asym_w_med"] <= p["causeway_asym_max"])
    rec["causeway"] = is_causeway
    rec["type"] = "causeway" if is_causeway else "raised_linear_no_crossing"
    rec["flags"] = _modern_flags(
        dict(crown_flat_med=rec["crown_flat_med"], flank_dip_med=rec["flank_dip_med"]),
        rec["length_m"], 0.0, rec["h_med"], rec["crest_width_m"])
    if rec.get("one_frac", 0.0) > 0.6:
        rec["flags"].append("runs_along_drainage")
    if not cross["both_sides"]:
        rec["flags"].append("crossing_not_confirmed")
    score = (0.30 * float(np.clip((rec["continuity"] - 0.5) / 0.5, 0, 1)) +
             0.25 * float(np.clip((rec["h_med"] - 0.3) / 1.2, 0, 1)) +
             0.20 * (1 - float(np.clip(rec["asym_w_med"] / 0.6, 0, 1))) +
             0.15 * (1.0 if is_causeway else 0.0) +
             0.10 * float(np.clip(rec["length_m"] / 200.0, 0, 1)))
    if [f for f in rec["flags"] if f in ("flat_planar_crown", "flanking_borrow_ditches",
                                         "low_and_narrow_tracklike")]:
        score *= 0.8
    rec["score"] = float(score)
    if chm is not None:
        cv = D.bilinear(np.asarray(chm, np.float32),
                        x + np.cos(ang) * ss, y + np.sin(ang) * ss, meta)
        rec["canopy_m"] = float(np.nanmedian(cv)) if np.isfinite(cv).any() else None
    else:
        rec["canopy_m"] = None
    rec.update(height_m=rec["h_med"], height_iqr_m=rec["h_iqr"], asym=rec["asym_med"],
               asym_w=rec["asym_w_med"], crown_flat=rec["crown_flat_med"],
               flank_dip_m=rec["flank_dip_med"])
    rec["passed"] = True
    return rec, profs


# --------------------------------------------------------------------------
def detect(z, meta, w=None, chm=None, p=None, transect="SYN", id_prefix=None,
           return_candidates=False):
    p = dict(DEFAULTS if p is None else {**DEFAULTS, **p})
    g = _prep(z, meta, w, p)
    if not g["wf"].any():
        return dict(rows=[], diag=[], stats=dict(status="no-valid-data"), params=p,
                    meta=meta, funnel={}, mask={}, note="empty grid")
    g = _hydrology(g, meta, p)
    g = _ridge_fields(g, meta, p)
    # crest mask: used as *evidence* (continuity, crossing banks), not as the detector
    nms = _nms(g["meas"], g["ang"], meta, p["nms_step_m"])
    keep_hole = g["dist_to_hole_m"] >= p["mask_hole_buffer_m"]
    common = (g["wf"] & keep_hole & (g["planarity"] >= p["planarity_min"]) &
              (g["relief15"] >= p["relief_min_m"]))
    weak = common & (g["meas"] >= p["ridge_floor"]) & nms
    seeds = weak & (g["meas"] >= p["ridge_seed"])
    if seeds.any():
        crest = ndi.binary_propagation(seeds, mask=weak, structure=np.ones((3, 3), bool))
    else:
        crest = np.zeros_like(weak)
    # Radon search
    self_test_rotation()
    raw = _radon_lines(g, meta, p)
    # broad-recall generator: keep long, well-supported lines, then cap the number
    # assessed so a large transect cannot blow up the runtime
    raw = [c for c in raw if c["length_m"] >= p["radon_min_len_m"] and
           c["support"] >= p["radon_min_support"]]
    raw.sort(key=lambda q: -(q["strength"] * min(q["length_m"], 400.0)))
    n_raw_total = len(raw)
    raw = raw[:p["max_candidates"]]
    cands = _dedupe_cands(raw, p)
    funnel = dict(n_radon=n_raw_total, n_assessed=len(cands), too_short=0, no_profiles=0,
                  height=0, width=0, asym=0, continuity=0, h_iqr=0, width_cv=0,
                  passed=0, n_deduped=0)
    rows, diag, rejects = [], [], []
    for c in cands:
        rec, profs = _assess_line(g, meta, c["x"], c["y"],
                                  math.radians(c["axis_angle_deg"]), c["length_m"],
                                  crest, p, chm=chm)
        rec.update(radon_strength=c["strength"], radon_support=c["support"],
                   radon_theta_deg=c["theta_deg"])
        if rec.get("reject"):
            key = {"too_short": "too_short", "no_usable_profiles": "no_profiles",
                   "height_out_of_range": "height", "width_out_of_range": "width",
                   "asymmetric_flanks": "asym", "discontinuous": "continuity",
                   "height_not_constant": "h_iqr", "width_not_constant": "width_cv"}.get(
                       rec["reject"], "continuity")
            funnel[key] += 1
            rec["gate"] = rec.get("reject")
            rejects.append(rec)
            continue
        rows.append(rec)
        diag.append(dict(candidate=rec, profiles=profs,
                         profile_step_m=p["prof_step_m"], profile_half_m=p["prof_half_m"]))
    # final dedupe of accepted lines
    order = np.argsort([-r["score"] for r in rows]) if rows else []
    rows2, diag2, dropped = [], [], 0
    for k in order:
        r, dg = rows[k], diag[k]
        dup = any(_same_feature(r, o, p) for o in rows2)
        if dup:
            dropped += 1
        else:
            rows2.append(r)
            diag2.append(dg)
    funnel["n_deduped"] = int(dropped)
    funnel["passed"] = int(len(rows2))
    pre = id_prefix or transect
    for k, r in enumerate(rows2, 1):
        r["id"] = f"{pre}-CW{k:03d}"
        r["transect"] = transect
        diag2[k - 1]["candidate"] = r
        diag2[k - 1]["id"] = r["id"]
    stats = dict(status="ok", n_candidates=len(rows2),
                 n_causeway=int(sum(r["causeway"] for r in rows2)),
                 n_raised_nocross=int(sum(not r["causeway"] for r in rows2)),
                 n_crest_cells=int(crest.sum()), n_radon=n_raw_total,
                 drain_frac=g["drain_frac"], acc_thr_cells=g["acc_thr_cells"],
                 hole_frac=float(np.mean(~g["w"])), ridge_floor=p["ridge_floor"],
                 radon_thr=p["radon_thr"])
    out = dict(rows=rows2, diag=diag2, rejects=rejects, stats=stats, funnel=funnel,
               params=p, meta=meta,
               mask=dict(crest=crest, drain=g["drain"], hole=~g["w"], meas=g["meas"],
                         dist_to_drain_m=g["dist_to_drain_m"]))
    if return_candidates:
        out["radon_candidates"] = cands
    return out


def to_candidate_rows(rows, transect, epsg, lonlat=None):
    """Map detections to the shared candidates.csv columns (extra keys ignored on write)."""
    out = []
    for r in rows:
        lon = lat = ""
        if lonlat is not None:
            lon, lat = lonlat(r["x"], r["y"])
        out.append(dict(
            id=r["id"], state="Possible", transect=transect, EPSG=epsg,
            x=f"{r['x']:.2f}", y=f"{r['y']:.2f}",
            lon="" if lon == "" else f"{lon:.6f}",
            lat="" if lat == "" else f"{lat:.6f}",
            type=r["type"], size_m=f"{r['length_m']:.0f}",
            relief_m=f"{r['height_m']:.2f}",
            canopy_m="" if r["canopy_m"] is None else f"{r['canopy_m']:.1f}",
            nearest_known_db="", nearest_known_m="", literature_checked="",
            certainty="Possible",
            reason=(f"level ridge h={r['height_m']:.2f} m, crest width "
                    f"{r['crest_width_m']:.0f} m, length {r['length_m']:.0f} m, "
                    f"continuity {r['continuity']:.2f}, crossings={r['crossings']} "
                    f"at {r['cross_angle_deg'] if r['cross_angle_deg'] is None else round(r['cross_angle_deg'])} deg, "
                    f"flags={'/'.join(r['flags']) or 'none'}; auto-screen only, "
                    f"skeptic review and record check pending")))
    return out


def write_outputs(res, outdir, transect="SYN"):
    os.makedirs(outdir, exist_ok=True)
    import csv
    rows = res["rows"]
    base = os.path.join(outdir, f"detections_{transect}.csv")
    keys = ["id", "transect", "type", "x", "y", "grid_row", "grid_col", "length_m",
            "band_width_m", "crest_width_m", "width_cv", "height_m", "height_iqr_m",
            "asym", "crown_flat", "flank_dip_m", "continuity", "falloff_viol",
            "straight_dev_m", "elong", "axis_angle_deg", "crossings", "cross_angle_deg",
            "approach_left_m", "approach_right_m", "canopy_m", "score", "flags", "n_cells"]
    with open(base, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            r2 = dict(r); r2["flags"] = ";".join(r["flags"]); wr.writerow(r2)
    with open(os.path.join(outdir, f"diagnostics_{transect}.json"), "w") as fh:
        json.dump(dict(transect=transect, stats=res["stats"], params=res["params"],
                       candidates=[dict(id=d["id"], candidate=d["candidate"],
                                        profile_step_m=d["profile_step_m"],
                                        profile_half_m=d["profile_half_m"],
                                        profiles=[{k: v for k, v in pr.items() if k != "s"}
                                                  for pr in d["profiles"]])
                                   for d in res["diag"]]), fh, default=float)
    return base
