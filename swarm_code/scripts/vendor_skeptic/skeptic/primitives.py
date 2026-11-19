"""Numeric primitives used to falsify ring/line/mound candidates.

Design notes
------------
The archetypal target (Peripato et al. 2023 / Saunaluoma & Schaan geoglyphs) is a
*ring ditch*: a low, closed, roughly constant-radius depression, often with a low
bank on one or both sides, regular planform (circle, square, U), ditch 3-15 m wide
and 0.5-4 m deep, 30-400 m across.

Each false-positive family has a different geometric fingerprint, and these
primitives are chosen so that the fingerprint is measurable without imagery:

  meander / oxbow / scroll bar   dip radius not constant; large angular gaps;
                                 dip continues outward (through-going channel)
  straight stream valley         broad V, no banks: no two-sided ditch at all
  cattle pond                    interior far *below* the exterior trend (a
                                 closed depression), flat bottom, often nodata
  building / modern pad          flat low-interior, one-sided step edges, sharp
  modern road / clearing         straight, canopy-gap corridor (CHM ~ 0)
  empty-data halo                ring path touches nodata, annulus partly void
  natural hilltop + gullies      dip radius wanders, no closure, planform irregular
  transect edge artefact         candidate clipped by the grid border

Everything here is deterministic, needs only numpy/scipy, and records the numbers
it used so a reviewer can re-derive every verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage, signal

from .grid import Grid

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def robust_sigma(v, axis=None):
    """MAD-based sigma (1.4826 * median |v - median v|), NaN-aware."""
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(v - med), axis=axis)
    return 1.4826 * mad


def noise_from_second_difference(prof):
    """Local noise estimate of a 1-D profile that is not inflated by smooth dips.

    For white noise of sigma s, the second difference has sigma s*sqrt(6).
    """
    p = np.asarray(prof, dtype=float)
    if p.size < 5:
        return float("nan")
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    s = robust_sigma(d2[np.isfinite(d2)])
    return float(s / math.sqrt(6.0)) if np.isfinite(s) else float("nan")


def _fill_1d(v):
    """Linearly interpolate interior NaNs, nearest-fill the ends."""
    v = np.asarray(v, dtype=float).copy()
    bad = ~np.isfinite(v)
    if bad.all():
        return v
    if bad.any():
        idx = np.arange(v.size)
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


def _odd(n):
    n = int(max(1, round(n)))
    return n if n % 2 else n + 1


def _circular_gaps_deg(az_deg, valid):
    """Largest contiguous angular gap (deg) with no valid sample."""
    az = np.asarray(az_deg, dtype=float)[valid]
    if az.size == 0:
        return 360.0
    if az.size < 2:
        return 360.0
    step = 360.0 / (np.asarray(az_deg).size)
    a = np.sort(az)
    d = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(max(0.0, d.max() - step))


# ---------------------------------------------------------------------------
# polar sampling
# ---------------------------------------------------------------------------

@dataclass
class RadialField:
    """z sampled on a polar grid around (cx, cy)."""

    cx: float
    cy: float
    r: np.ndarray            # (n_r,) radii in metres
    az: np.ndarray           # (n_az,) azimuths in degrees, 0 = +x, 90 = +y
    Z: np.ndarray            # (n_r, n_az) sampled elevations, NaN where void
    valid: np.ndarray        # (n_r,) fraction of valid azimuth samples per radius
    res: float

    @property
    def dr(self):
        return float(self.r[1] - self.r[0]) if self.r.size > 1 else self.res


def radial_field(grid: Grid, cx, cy, r_max, dr=None, n_az=180) -> RadialField:
    """Bilinearly sample `grid` on a polar grid centred at (cx, cy)."""
    res = grid.res
    dr = float(dr) if dr else res / 2.0
    n_r = int(max(8, math.ceil(r_max / dr)))
    r = np.linspace(dr * 0.5, n_r * dr, n_r)
    az = np.arange(n_az) * (360.0 / n_az)
    th = np.deg2rad(az)
    c, s = np.cos(th)[None, :], np.sin(th)[None, :]
    X = cx + r[:, None] * c
    Y = cy + r[:, None] * s
    Z = grid.sample(X, Y)
    valid = np.isfinite(Z).mean(axis=1)
    return RadialField(float(cx), float(cy), r, az, Z, valid, res)


@dataclass
class RadialProfile:
    r: np.ndarray
    prof: np.ndarray          # robust azimuthal median per radius
    lo: np.ndarray            # 25th percentile
    hi: np.ndarray            # 75th percentile
    valid: np.ndarray
    background: np.ndarray    # wide-median local trend
    resid: np.ndarray         # prof - background
    noise: float              # robust sigma of the residual (2nd difference)

    @property
    def dr(self):
        return float(self.r[1] - self.r[0])


def radial_profile(field: RadialField, trend_window_m=41.0) -> RadialProfile:
    """Azimuthal robust median profile plus a wide local-median trend."""
    Z = field.Z
    prof = np.nanmedian(Z, axis=1)
    lo = np.nanpercentile(Z, 25, axis=1)
    hi = np.nanpercentile(Z, 75, axis=1)
    pf = _fill_1d(prof)
    size = _odd(trend_window_m / field.dr)
    bg = ndimage.median_filter(pf, size=size, mode="nearest")
    bg = ndimage.uniform_filter1d(bg, size=_odd(5), mode="nearest")
    resid = pf - bg
    # noise: second difference *far* from the strongest dip is not needed; the
    # estimator is already insensitive to smooth structure.
    noise = noise_from_second_difference(resid)
    return RadialProfile(field.r, prof, lo, hi, field.valid, bg, resid, noise)


# ---------------------------------------------------------------------------
# dip detection on the radial profile
# ---------------------------------------------------------------------------

@dataclass
class Dip:
    r: float                # radius of the profile minimum (ditch centre line)
    amp: float              # prominence below the local trend, metres (positive)
    width_m: float          # FWHM of the dip along the radius
    snr: float              # amp / noise
    r_lo: float             # search band actually used
    r_hi: float
    noise: float

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def find_dip(prof: RadialProfile, r_lo=None, r_hi=None,
             min_amp=0.12) -> Dip | None:
    """Strongest *closed* dip in the radial profile (negative prominence)."""
    r = prof.r
    res = prof.resid.copy()
    m = np.isfinite(res)
    if m.sum() < 10:
        return None
    res = np.where(m, res, 0.0)
    s = ndimage.uniform_filter1d(res, size=_odd(3), mode="nearest")
    mask = np.ones_like(s, dtype=bool)
    if r_lo is not None:
        mask &= r >= r_lo
    if r_hi is not None:
        mask &= r <= r_hi
    if mask.sum() < 5:
        return None
    peaks, props = signal.find_peaks(-s, prominence=0.0)
    if peaks.size == 0:
        return None
    keep = mask[peaks]
    if not keep.any():
        return None
    peaks = peaks[keep]
    prom = np.asarray(props["prominences"])[keep]
    good = np.isfinite(prom)
    if not good.any():
        return None
    peaks, prom = peaks[good], prom[good]
    k = int(np.argmax(prom))
    if prom[k] < min_amp:
        return None
    p = peaks[k]
    try:
        w_samples = float(signal.peak_widths(-s, [p], rel_height=0.5)[0][0])
    except Exception:
        w_samples = float("nan")
    w = w_samples * prof.dr if np.isfinite(w_samples) else 2 * prof.dr
    noise = prof.noise if np.isfinite(prof.noise) and prof.noise > 1e-3 else 0.02
    return Dip(float(r[p]), float(prom[k]), float(w),
               float(prom[k] / noise), float(r[mask][0]), float(r[mask][-1]),
               float(noise))


# ---------------------------------------------------------------------------
# per-azimuth edge extraction, centre refinement, planform fit
# ---------------------------------------------------------------------------

def find_dips(prof: RadialProfile, r_lo=None, r_hi=None, top=3,
              min_amp=0.10):
    """The strongest few *closed* dips in the radial profile, not just the one.

    A detector's size hint is only approximate (on the real 0414 geoglyph the
    reported radius was 51 m against a true ditch radius of 41 m), and a
    single-dip seed can lock onto a neighbouring low and walk the centre search
    away from the real ring.  Evaluating the top few and choosing by ring quality
    removes that dependency.
    """
    r = prof.r
    res = np.where(np.isfinite(prof.resid), prof.resid, 0.0)
    s = ndimage.uniform_filter1d(res, size=_odd(3), mode="nearest")
    mask = np.ones_like(s, dtype=bool)
    if r_lo is not None:
        mask &= r >= r_lo
    if r_hi is not None:
        mask &= r <= r_hi
    if mask.sum() < 5:
        return []
    peaks, props = signal.find_peaks(-s, prominence=0.0)
    if peaks.size == 0:
        return []
    keep = mask[peaks]
    peaks, prom = peaks[keep], np.asarray(props["prominences"])[keep]
    good = np.isfinite(prom)
    peaks, prom = peaks[good], prom[good]
    order = np.argsort(-prom)[:top]
    noise = prof.noise if np.isfinite(prof.noise) and prof.noise > 1e-3 else 0.02
    out = []
    for k in order:
        if prom[k] < min_amp:
            continue
        p = int(peaks[k])
        try:
            w = float(signal.peak_widths(-s, [p], rel_height=0.5)[0][0]) * prof.dr
        except Exception:
            w = 2 * prof.dr
        out.append(Dip(float(r[p]), float(prom[k]), float(w),
                       float(prom[k] / noise), float(r[mask][0]),
                       float(r[mask][-1]), float(noise)))
    return out


@dataclass
class RingEdge:
    cx: float
    cy: float
    az: np.ndarray
    r: np.ndarray            # radius of the strongest dip along each azimuth
    amp: np.ndarray          # its depth below the trend
    ok: np.ndarray           # usable azimuth
    band_lo: float
    band_hi: float
    width_m: np.ndarray      # per-azimuth width at half amplitude
    coverage: float          # fraction of usable azimuths
    gap_deg: float           # largest angular gap without a usable dip
    r_median: float
    r_rmad: float            # robust relative scatter of r


def ring_edge(field: RadialField, prof: RadialProfile, dip: Dip,
              band=(0.6, 1.5), min_frac=0.30, want_widths=True) -> RingEdge:
    """For each azimuth, the radius of the deepest point in a band around `dip.r`."""
    lo = max(field.r[0], band[0] * dip.r)
    hi = min(field.r[-1], band[1] * dip.r)
    sel = (field.r >= lo) & (field.r <= hi)
    rr = field.r[sel]
    if rr.size < 3:
        rr = field.r
        sel = np.ones(field.r.size, dtype=bool)
    R = field.Z[sel, :] - prof.background[sel][:, None]
    R = np.where(np.isfinite(R), R, np.inf)
    idx = np.argmin(R, axis=0)
    rmin = field.r[sel][idx]
    amin = -R[idx, np.arange(R.shape[1])]
    thr = max(min_frac * dip.amp, 0.10)
    ok = np.isfinite(amin) & (amin >= thr)
    # per-azimuth width at half amplitude, measured on the sampled column
    widths = np.full(field.az.size, np.nan)
    if want_widths:
        for j in np.nonzero(ok)[0]:
            col = np.where(np.isfinite(R[:, j]), R[:, j], 0.0)
            half = -0.5 * amin[j]
            below = col <= half
            k = idx[j]
            a = k
            while a - 1 >= 0 and below[a - 1]:
                a -= 1
            b = k
            while b + 1 < col.size and below[b + 1]:
                b += 1
            widths[j] = (b - a + 1) * field.dr
    r_med = float(np.median(rmin[ok])) if ok.any() else float("nan")
    rmad = float(robust_sigma(rmin[ok]) / r_med) if ok.sum() > 3 else float("nan")
    return RingEdge(field.cx, field.cy, field.az, rmin, amin, ok, lo, hi, widths,
                    float(ok.mean()), _circular_gaps_deg(field.az, ok),
                    r_med, rmad)


def ring_score(grid: Grid, cx, cy, R, amp_ref, n_az=72, dr=None):
    """How well does a ring of radius ~R centred at (cx, cy) explain the dip?

    Objective (minimised): robust relative scatter of the dip radius, plus a
    penalty for missing azimuths, plus a penalty for losing dip depth.  Both
    penalties matter: a centre pushed sideways can make the surviving arcs look
    very constant while covering only half the circle, which is exactly the
    meander-bend failure mode.
    """
    f = radial_field(grid, cx, cy, max(2.2 * R, R + 40.0), dr=dr, n_az=n_az)
    pr = radial_profile(f)
    d = find_dip(pr, r_lo=0.55 * R, r_hi=1.7 * R, min_amp=0.08)
    if d is None:
        return None
    e = ring_edge(f, pr, d, band=(0.55, 1.7), want_widths=False)
    rmad = e.r_rmad if np.isfinite(e.r_rmad) else 1.0
    amp_pen = max(0.0, 0.6 * amp_ref - d.amp) / max(amp_ref, 0.1)
    score = rmad + 0.60 * (1.0 - e.coverage) + 0.25 * amp_pen
    return dict(score=float(score), rmad=float(rmad), coverage=float(e.coverage),
                amp=float(d.amp), r=float(e.r_median if np.isfinite(e.r_median) else d.r),
                dip=d, field=f, prof=pr, edge=e)


def refine_centre(grid: Grid, cx, cy, dip: Dip, R=None, max_range_m=None,
                  verbose=False):
    """Search for the centre that makes the dip radius most constant.

    A coarse lattice search followed by one refinement, guarded so that it can
    only be adopted when it actually improves the ring score.  Divergent
    least-squares chasing of the first harmonic is what this replaces: on a
    superellipse (p != 2) the first harmonic is not the right model and the fit
    can walk the centre away from the ring entirely.
    """
    R = float(R if R is not None else dip.r)
    amp_ref = max(float(dip.amp), 0.1)
    base = ring_score(grid, cx, cy, R, amp_ref)
    info = dict(offset_m=0.0, dx=0.0, dy=0.0, n_eval=1,
                score_before=(base["score"] if base else None),
                score_after=(base["score"] if base else None),
                accepted=False)
    if base is None:
        return float(cx), float(cy), info
    rng = max_range_m if max_range_m is not None else float(np.clip(0.25 * R, 8.0, 45.0))
    best = dict(x=0.0, y=0.0, **base)
    # three shrinking lattices: a single coarse lattice quantises the recovered
    # centre to ~0.1 R, which on the real 0414 control was the difference between
    # finding the ring at all and not finding it (two detector snapshots 3 m apart
    # gave different answers).
    st1 = max(2.0 * grid.res, 0.10 * R)
    for step, half in ((st1, rng),
                       (max(grid.res, 0.5 * st1), st1),
                       (max(0.75 * grid.res, 0.25 * st1), 0.5 * st1)):
        offs = np.arange(-half, half + 1e-9, step)
        for dx in offs:
            for dy in offs:
                s = ring_score(grid, cx + best["x"] + dx, cy + best["y"] + dy,
                               R, amp_ref)
                info["n_eval"] += 1
                if s is not None and s["score"] < best["score"]:
                    best = dict(x=best["x"] + dx, y=best["y"] + dy, **s)
    info["score_after"] = best["score"]
    info["dx"], info["dy"] = float(best["x"]), float(best["y"])
    info["offset_m"] = float(math.hypot(best["x"], best["y"]))
    info["accepted"] = bool(best["score"] < base["score"] - 1e-3)
    if verbose:
        print(info)
    if not info["accepted"]:
        return float(cx), float(cy), info
    return float(cx + best["x"]), float(cy + best["y"]), info


@dataclass
class Planform:
    shape: str               # circle | rounded-square | square | ellipse | rectangle | irregular
    a0: float                # mean apparent radius (m)
    diameter_m: float        # 2*a0 -> comparable with a detector's "size"
    side_m: float            # for square/rectangle models, the long side
    aspect: float            # long/short axis ratio of the best model
    p: float                 # superellipse exponent (2 = circle)
    phi_deg: float           # orientation of the square/rectangle
    resid_rel: float         # median |r - model| / a0 for the chosen model
    circle_resid_rel: float  # same for a plain circle
    gain: float              # circle_resid_rel / resid_rel
    n_used: int

    def as_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


def _superellipse_r(theta, p):
    c = np.abs(np.cos(theta))
    s = np.abs(np.sin(theta))
    c = np.maximum(c, 1e-9)
    s = np.maximum(s, 1e-9)
    return (c ** p + s ** p) ** (-1.0 / p)


def _robust_fit_scale(r, model):
    """Scale a such that a*model fits r under a median-ratio loss."""
    q = r / model
    a = float(np.median(q))
    for _ in range(3):
        e = np.abs(q / a - 1.0)
        s = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-9
        keep = e <= max(2.5 * s, 0.02)
        if keep.sum() < 8:
            break
        a = float(np.median(r[keep] / model[keep]))
    return a


def _loss(r, model, a):
    d = np.abs(r - a * model) / max(a, 1e-9)
    return float(np.median(d))


def fit_planform(edge: RingEdge, prefer_simple=True) -> Planform | None:
    """Fit circle / superellipse / ellipse / rectangle to the observed dip radii.

    Models are compared by *median absolute relative deviation* from the dip
    radius; a more complex model must improve that by 25% to be preferred, so a
    plain circle wins whenever it is adequate (conservative for the target type).
    """
    ok = edge.ok & np.isfinite(edge.r) & (edge.r > 0)
    if ok.sum() < 10:
        return None
    th = np.deg2rad(edge.az[ok])
    r = edge.r[ok].astype(float)
    a0 = float(np.median(r))
    if not np.isfinite(a0) or a0 <= 0:
        return None
    n = r.size

    # --- circle
    model = np.ones(n)
    a_c = _robust_fit_scale(r, model)
    loss_c = _loss(r, model, a_c)

    # --- superellipse (axis-aligned + 90 deg periodic orientation)
    best = (loss_c, 2.0, 0.0, a_c)
    for p in (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 30.0, 45.0):
        for phi in np.arange(0.0, 45.0, 2.5):
            m = _superellipse_r(th - math.radians(phi), p)
            a = _robust_fit_scale(r, m)
            L = _loss(r, m, a)
            if L < best[0]:
                best = (L, p, float(phi), a)
    loss_se, p_se, phi_se, a_se = best

    # The ellipse and rectangle fits are only worth their cost when no simple
    # model already explains the dip radius.
    if min(loss_c, loss_se) > 0.05:
        loss_e, pa, pb, phi_e = _fit_ellipse(th, r, a0)
        loss_rc, ra, rb, phi_rc = _fit_rectangle(th, r, a0)
    else:
        loss_e, pa, pb, phi_e = loss_c * 1.5, a_c, a_c, 0.0
        loss_rc, ra, rb, phi_rc = loss_c * 1.5, a_c, a_c, 0.0

    cands = [("circle", loss_c, dict(p=2.0, phi_deg=0.0,
                                     a0=a_c, aspect=1.0, side_m=2 * a_c)),
             ("superellipse", loss_se, dict(p=p_se, phi_deg=phi_se, a0=a_se,
                                            aspect=1.0, side_m=2 * a_se)),
             ("ellipse", loss_e, dict(p=2.0, phi_deg=phi_e, a0=0.5 * (pa + pb),
                                      aspect=max(pa, pb) / max(min(pa, pb), 1e-9),
                                      side_m=2 * max(pa, pb))),
             ("rectangle", loss_rc, dict(p=np.inf, phi_deg=phi_rc,
                                         a0=0.5 * (ra + rb),
                                         aspect=max(ra, rb) / max(min(ra, rb), 1e-9),
                                         side_m=2 * max(ra, rb)))]
    # choose with a complexity penalty: 25% improvement required
    order = sorted(cands, key=lambda c: c[1])
    chosen = order[0]
    if prefer_simple:
        for c in order:
            if c[0] == "circle":
                chosen = c
                break
            if c[1] < 0.75 * loss_c:
                chosen = c
                break
            chosen = ("circle", loss_c, cands[0][2])
            break
    name, loss, par = chosen
    if name == "superellipse":
        name = ("square" if par["p"] >= 8.0 else
                "rounded-square" if par["p"] > 3.0 else "circle")
    loss = max(loss, 1e-6)
    return Planform(shape=name, a0=float(par["a0"]), diameter_m=2 * float(par["a0"]),
                    side_m=float(par["side_m"]), aspect=float(par["aspect"]),
                    p=float(par["p"]), phi_deg=float(par["phi_deg"]),
                    resid_rel=loss, circle_resid_rel=loss_c,
                    gain=loss_c / loss, n_used=int(n))


def _fit_ellipse(th, r, a0):
    """1/r^2 = cos^2(t)/A^2 + sin^2(t)/B^2 in the ellipse frame; grid over angle."""
    best = (np.inf, a0, a0, 0.0)
    inv2 = 1.0 / np.maximum(r, 1e-9) ** 2
    for phi in np.arange(0.0, 180.0, 2.0):
        c = np.cos(th - math.radians(phi)) ** 2
        s = 1.0 - c
        A = np.column_stack([c, s])
        coef, *_ = np.linalg.lstsq(A, inv2, rcond=None)
        if np.any(coef <= 0):
            continue
        ra, rb = 1.0 / math.sqrt(coef[0]), 1.0 / math.sqrt(coef[1])
        m = ra * rb / np.sqrt((rb * np.cos(th - math.radians(phi))) ** 2 +
                              (ra * np.sin(th - math.radians(phi))) ** 2)
        a = _robust_fit_scale(r, m)
        L = _loss(r, m, a)
        if L < best[0]:
            best = (L, ra * a, rb * a, float(phi))
    return best


def _rect_model(th, a, b, phi):
    c = np.abs(np.cos(th - phi)) / max(a, 1e-9)
    s = np.abs(np.sin(th - phi)) / max(b, 1e-9)
    return 1.0 / np.maximum(np.maximum(c, s), 1e-9)


def _fit_rectangle(th, r, a0):
    from scipy.optimize import minimize
    best = (np.inf, a0, a0, 0.0)

    def cost(v):
        a, b, phi = v
        if a < 5 or b < 5:
            return 1e3
        m = _rect_model(th, a, b, phi)
        return _loss(r, m, 1.0)

    for phi0 in np.arange(0.0, 90.0, 30.0):
        for ratio in (1.0, 0.7):
            x0 = np.array([a0 * ratio, a0 / ratio, math.radians(phi0)])
            try:
                res = minimize(cost, x0, method="Powell",
                               options=dict(maxiter=200, xtol=2e-3, ftol=2e-4))
            except Exception:
                continue
            if res.fun < best[0]:
                a, b, phi = res.x
                best = (float(res.fun), float(abs(a)), float(abs(b)),
                        float(math.degrees(phi) % 90.0))
    return best


# ---------------------------------------------------------------------------
# cross-sections: is it a ditch (two-sided) or a one-sided step?
# ---------------------------------------------------------------------------

@dataclass
class CrossSection:
    n: int
    two_sided_frac: float
    step_frac: float
    bank_frac: float
    median_relief_inner: float
    median_relief_outer: float
    median_width_m: float
    median_bank_m: float

    def as_dict(self):
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def cross_sections(grid: Grid, edge: RingEdge, dip: Dip, prof: RadialProfile,
                   half_width_m=None, max_sections=36) -> CrossSection:
    """Radial cross-sections through the dip, looking for a two-sided ditch.

    A ditch has ground at trend level on *both* sides of its bottom.  A terrace
    edge, building pad, scan strip edge or a channel bank has it on one side only,
    and that one-sidedness is what `step_frac` measures.
    """
    w = max(1.5, 0.5 * dip.width_m) if half_width_m is None else half_width_m
    ok = np.nonzero(edge.ok)[0]
    if ok.size == 0:
        return CrossSection(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    if ok.size > max_sections:
        ok = ok[np.linspace(0, ok.size - 1, max_sections).astype(int)]
    amp = max(dip.amp, 0.2)
    thr = 0.35 * amp
    two = step = bank = 0
    rel_in, rel_out, widths, banks = [], [], [], []
    for j in ok:
        r0 = edge.r[j]
        th = math.radians(edge.az[j])
        wl = np.linspace(r0 - 2.8 * w, r0 + 2.8 * w, 41)
        zz = grid.sample(edge.cx + wl * math.cos(th),
                         edge.cy + wl * math.sin(th))
        bg = np.interp(wl, prof.r, prof.background)
        res = zz - bg
        core = np.abs(wl - r0) <= 1.2 * w
        inner = (wl >= r0 - 2.8 * w) & (wl <= r0 - 1.8 * w)
        outer = (wl >= r0 + 1.8 * w) & (wl <= r0 + 2.8 * w)
        if not (np.isfinite(res[core]).any() and inner.any() and outer.any()):
            continue
        bottom = float(np.nanmin(res[core]))
        ri = float(np.nanmedian(res[inner])) - bottom
        ro = float(np.nanmedian(res[outer])) - bottom
        rel_in.append(ri)
        rel_out.append(ro)
        widths.append(edge.width_m[j] if np.isfinite(edge.width_m[j]) else np.nan)
        if min(ri, ro) >= thr:
            two += 1
        else:
            big, small = max(ri, ro), min(ri, ro)
            if big >= thr and small <= 0.15 * amp:
                step += 1
        # outer bank: a local maximum in the residual outside the ditch
        ob = (wl >= r0 + 0.8 * w) & (wl <= r0 + 2.6 * w)
        ib = (wl >= r0 - 2.6 * w) & (wl <= r0 - 0.8 * w)
        bm = np.nan
        if ob.any():
            pk = float(np.nanmax(res[ob])) - ro - bottom
            if np.isfinite(pk):
                bm = max(bm if np.isfinite(bm) else -9, pk)
        if ib.any():
            pk = float(np.nanmax(res[ib])) - ri - bottom
            if np.isfinite(pk):
                bm = max(bm if np.isfinite(bm) else -9, pk)
        if np.isfinite(bm):
            banks.append(bm)
            if bm >= 0.15:
                bank += 1
    n = len(rel_in)
    if n == 0:
        return CrossSection(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    return CrossSection(n, two / n, step / n, bank / n,
                        float(np.median(rel_in)), float(np.median(rel_out)),
                        float(np.nanmedian(widths)), float(np.nanmedian(banks)))


# ---------------------------------------------------------------------------
# topological closure: does the low actually enclose the centre?
# ---------------------------------------------------------------------------

@dataclass
class PolarClosure:
    closed: bool              # no un-bridged gap anywhere around the ring
    max_gap_arc_m: float      # longest gap, measured as arc length at the dip radius
    gap_arc_total_m: float
    n_gaps: int
    n_gaps_bridged: int
    bridge_m: float
    coverage_arc: float       # 1 - total gap arc / circumference
    r_ref: float

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def polar_closure(edge: RingEdge, bridge_m=12.0) -> PolarClosure:
    """Closure of the dip band, measured as *arc length* rather than degrees.

    A ditch entrance is a few metres wide, so an angular gap tolerance is wrong:
    it depends on the ring size.  Gaps whose arc length is at most `bridge_m` are
    treated as entrances (bridged); any longer gap means the low is open, which
    is what a channel bend, an oxbow mouth or a valley shoulder looks like.
    """
    ok = edge.ok
    n = ok.size
    r_ref = float(np.nanmedian(edge.r[ok])) if ok.any() else float("nan")
    if not np.isfinite(r_ref) or r_ref <= 0:
        return PolarClosure(False, float("nan"), float("nan"), 0, 0, bridge_m,
                            0.0, float("nan"))
    dth = 2 * math.pi / n
    # rotate so that index 0 is inside a gap, if there is one, to make runs circular
    if ok.all():
        return PolarClosure(True, 0.0, 0.0, 0, 0, bridge_m, 1.0, r_ref)
    start = int(np.argmin(ok))
    o = np.roll(ok, -start)
    runs = []
    i = 0
    while i < n:
        if not o[i]:
            j = i
            while j < n and not o[j]:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    arcs = [length * dth * r_ref for _, length in runs]
    bridged = [a for a in arcs if a <= bridge_m]
    long_arcs = [a for a in arcs if a > bridge_m]
    total = float(np.sum(arcs))
    return PolarClosure(bool(len(long_arcs) == 0), float(max(arcs)) if arcs else 0.0,
                        total, len(arcs), len(bridged), bridge_m,
                        float(1.0 - total / (2 * math.pi * r_ref)), r_ref)


@dataclass
class BinaryClosure:
    ratio: float              # enclosed area / (pi R^2), 0 when the low leaks
    enclosed_area_m2: float
    centre_inside: bool
    mask_frac: float          # fraction of the window below the dip threshold
    closed: bool
    touches_border: bool
    window_half_m: float

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def binary_closure(grid: Grid, cx, cy, R, amp, bridge_m=12.0, frac=0.35):
    """Do the low cells enclose the centre as a closed band?

    Threshold the local relief at `frac * amp`, bridge gaps up to ~`bridge_m`
    (so a ditch entrance does not read as "open"), and fill holes.  A closed ring
    ditch encloses roughly pi R^2 of interior; an open arc, a valley shoulder or
    a meander bend encloses nothing, no matter where the centre is put.  This is
    the one closure test that a shifted centre cannot fake.
    """
    half = 1.9 * R + 20.0
    sub = grid.subgrid(cx, cy, half)
    if sub is None:
        return BinaryClosure(0.0, 0.0, False, float("nan"), False, True, half)
    z = np.where(np.isfinite(sub.z), sub.z, np.nanmedian(sub.z))
    sigma = max(R / 3.0, 8.0) / sub.res
    rel = z - ndimage.gaussian_filter(z, sigma, mode="nearest")
    mask = rel < -frac * amp
    n = max(1, int(round(bridge_m / sub.res)))
    st = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    mc = ndimage.binary_closing(mask, structure=st)
    filled = ndimage.binary_fill_holes(mc)
    encl = filled & ~mc
    fi, fj = sub.frac_index(cx, cy)
    ii = int(np.clip(round(float(fi)), 0, sub.ny - 1))
    jj = int(np.clip(round(float(fj)), 0, sub.nx - 1))
    # `filled` = the dip band plus everything it encloses.  If the low leaks, the
    # interior is not filled at all, and a centre in the middle of the feature is
    # not inside `filled`.  A centre that sits ON the band (a sunken road crossing
    # the middle of the enclosure) is still inside it.
    centre_in = bool(filled[ii, jj])
    encl = filled & ~mc
    area = float(encl.sum()) * sub.res ** 2
    ratio = area / (math.pi * R * R) if centre_in else 0.0
    touch = bool(mc[0, :].any() or mc[-1, :].any() or mc[:, 0].any() or mc[:, -1].any())
    return BinaryClosure(float(ratio), area, centre_in, float(mc.mean()),
                         bool(centre_in and ratio > 0.30 and not touch),
                         touch, half)


# ---------------------------------------------------------------------------
# interior vs exterior: pond / pad / plateau detection
# ---------------------------------------------------------------------------

@dataclass
class LevelStats:
    interior_off_m: float     # median (interior - exterior plane) in metres
    interior_p75_m: float     # upper quartile: un-excavated ground level
    interior_p10_m: float
    interior_sigma_m: float   # robust sigma inside
    exterior_off_m: float
    plane: tuple              # fitted exterior plane (a, b, c)
    n_interior: int
    n_exterior: int

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else
                    (list(v) if isinstance(v, tuple) else v))
                for k, v in asdict(self).items()}


def _plane_samples(grid, cx, cy, r_lo, r_hi, n=2400, seed=0):
    rs = np.random.default_rng(seed)
    r = np.sqrt(rs.uniform(r_lo ** 2, r_hi ** 2, n))
    th = rs.uniform(0, 2 * np.pi, n)
    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)
    z = grid.sample(x, y)
    ok = np.isfinite(z)
    return x[ok], y[ok], z[ok]


def level_stats(grid: Grid, cx, cy, R, interior_frac=0.55,
                ext_lo=1.15, ext_hi=1.65) -> LevelStats:
    """Is the interior at the same level as the surrounding terrain?

    A ring *ditch* encloses ground at the natural level.  A pond, a cattle tank or
    a modern excavated pad encloses ground well below (or above) it.  The exterior
    trend is a robustly fitted plane so a regional slope does not masquerade as an
    offset.
    """
    xi, yi, zi = _plane_samples(grid, cx, cy, 0.0, interior_frac * R, n=1500, seed=1)
    if zi.size < 20:
        return LevelStats(np.nan, np.nan, np.nan, np.nan, np.nan,
                          (np.nan,) * 3, int(zi.size), 0)
    xe, ye, ze = _plane_samples(grid, cx, cy, ext_lo * R, ext_hi * R, n=3000, seed=2)
    if ze.size < 30:
        return LevelStats(np.nan, np.nan, np.nan, np.nan, np.nan,
                          (np.nan,) * 3, int(zi.size), int(ze.size))
    A = np.column_stack([np.ones(ze.size), xe - cx, ye - cy])
    w = np.ones(ze.size, dtype=bool)
    coef = np.array([np.median(ze), 0.0, 0.0])
    for _ in range(4):
        if w.sum() < 12:
            break
        coef, *_ = np.linalg.lstsq(A[w], ze[w], rcond=None)
        res = ze - A @ coef
        s = robust_sigma(res[w]) or 0.02
        w = np.abs(res) <= max(2.5 * s, 0.05)
    res_e = ze - A @ coef
    Ai = np.column_stack([np.ones(zi.size), xi - cx, yi - cy])
    off_i = zi - Ai @ coef
    return LevelStats(float(np.median(off_i)),
                      float(np.percentile(off_i, 75)),
                      float(np.percentile(off_i, 10)),
                      float(robust_sigma(off_i)),
                      float(np.median(res_e)), tuple(float(v) for v in coef),
                      int(zi.size), int(ze.size))


# ---------------------------------------------------------------------------
# holes, borders, canopy
# ---------------------------------------------------------------------------

@dataclass
class Support:
    nodata_frac_window: float
    nodata_frac_interior: float
    nodata_frac_ditch_band: float   # the ditch itself (|d-R| <= 1.5 w)
    nodata_frac_annulus: float      # wider annulus (|d-R| <= 3 w)
    ring_path_gap_m: float          # no-data distance from the ditch centre line
    annulus_gap_m: float            # no-data distance from the wide annulus
    void_r_median_m: float          # median radius of the no-data cells near the ring
    void_r_offset_m: float          # void_r_median - R: are the voids ON the ditch?
    border_dist_ring_m: float
    border_dist_centre_m: float
    clipped: bool

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def support_stats(grid: Grid, cx, cy, R, w) -> Support:
    """Ground-data support around a ring: voids, holes and the transect border.

    `w` is the ditch half width.  The distance-to-hole figures are measured from
    the *ditch band* and from a wider annulus separately: measuring only from the
    wide annulus makes the test fire whenever any void exists within tens of
    metres of the ring, which on a real transect is almost always -- and it was
    the first thing that wrongly rejected the real 0414 positive control.
    """
    # A band of 1.5 * (half dip width) can be wider than the ring itself for a
    # small feature (a 12 m "width" on a 20 m radius would swallow the interior
    # and make the ditch band equal to "everything nearby").  Cap it to the ring.
    w = max(1.0, min(float(w), 0.30 * R))
    half = 1.7 * R + 30.0
    sl = grid.window(cx, cy, half)
    if sl is None:
            return Support(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, True)
    si, sj = sl
    nd = ~np.isfinite(grid.z[si, sj])
    dt = ndimage.distance_transform_edt(~nd) * grid.res
    ii, jj = np.meshgrid(np.arange(si.start, si.stop), np.arange(sj.start, sj.stop),
                         indexing="ij")
    x, y = grid.cell_center(ii, jj)
    d = np.hypot(x - cx, y - cy)
    ditch_band = np.abs(d - R) <= 1.5 * w
    ann_band = np.abs(d - R) <= 3.0 * w
    inner = d < 0.55 * R
    core = np.abs(d - R) <= 0.5 * w

    def frac(mask):
        return float(nd[mask].mean()) if np.any(mask) else float("nan")

    def gap(mask):
        return float(dt[mask].min()) if np.any(mask) else float("nan")

    bd_ring = (float(grid.border_distance(x[ann_band], y[ann_band]).min())
               if ann_band.any() else float("nan"))
    bd_c = float(grid.border_distance(cx, cy))
    clipped = bool(bd_c - R < 0 or (np.isfinite(bd_ring) and bd_ring < 0.5))
    near = nd & (np.abs(d - R) <= 6.0 * w)
    vr = float(np.median(d[near])) if near.any() else float("nan")
    return Support(float(nd.mean()), frac(inner), frac(ditch_band), frac(ann_band),
                   gap(core), gap(ann_band), vr,
                   (vr - R) if np.isfinite(vr) else float("nan"),
                   bd_ring, bd_c, clipped)


@dataclass
class Canopy:
    canopy_median_m: float
    canopy_p90_m: float
    canopy_open_frac: float
    interior_canopy_median_m: float

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def canopy_stats(chm: Grid | None, cx, cy, R) -> Canopy:
    if chm is None:
        return Canopy(np.nan, np.nan, np.nan, np.nan)
    ann = _plane_samples(chm, cx, cy, 0.8 * R, 1.25 * R, n=2000, seed=3)
    inner = _plane_samples(chm, cx, cy, 0.0, 0.5 * R, n=800, seed=4)
    if ann[2].size == 0:
        return Canopy(np.nan, np.nan, np.nan, np.nan)
    v = ann[2]
    return Canopy(float(np.median(v)), float(np.percentile(v, 90)),
                  float((v < 2.0).mean()),
                  float(np.median(inner[2])) if inner[2].size else np.nan)


# ---------------------------------------------------------------------------
# outward continuations: a channel passing through, or an attached earthwork?
# ---------------------------------------------------------------------------

@dataclass
class Continuation:
    n_rays: int
    pairs: int                # number of near-opposite ray pairs with outward lows
    pair_amp_m: float         # median amplitude of the best pair
    pair_angle_deg: float     # its azimuth
    style: str                # 'v' | 'banked' | 'flat' | 'none'
    outward_frac: float       # fraction of rays with a significant outward low
    note: str = ""

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def continuations(grid: Grid, cx, cy, R, dip: Dip, prof: RadialProfile,
                  plane=None, n_rays=24, r_inner=None, r_outer=None) -> Continuation:
    """Look beyond the ring for the same low continuing outward.

    A natural channel enters and leaves; a sunken road *attached* to a geoglyph
    also leaves, but it is straight, flat-bottomed and usually banked.  The style
    is therefore reported rather than used as a hard rejection.

    `plane` is the robust exterior plane (a, b, c) from `level_stats`; residuals
    are taken against it, not against the azimuthal median, because a regional
    slope of even 1% puts a 0.5 m bias on a ray 50 m long and would fake a
    channel on two opposite rays.
    """
    amps = np.full(n_rays, np.nan)
    wids = np.full(n_rays, np.nan)
    sty = np.empty(n_rays, dtype=object)
    sty[:] = ""
    r_in = float(r_inner) if r_inner else 1.25 * R
    r_out = float(r_outer) if r_outer else 2.5 * R

    def bg_at(x, y, rr):
        if plane is not None and np.all(np.isfinite(plane)):
            a, b, c = plane
            return a + b * (x - cx) + c * (y - cy)
        return np.interp(rr, prof.r, prof.background)

    for k in range(n_rays):
        th = 2 * np.pi * k / n_rays
        rr = np.arange(r_in, r_out, grid.res / 2)
        x = cx + rr * np.cos(th)
        y = cy + rr * np.sin(th)
        ins = grid.contains(x, y, margin_m=grid.res)
        if ins.sum() < 6:
            continue
        z = grid.sample(x[ins], y[ins])
        rr = rr[ins]
        x = x[ins]
        y = y[ins]
        base = bg_at(x, y, rr)
        res = z - base
        if not np.isfinite(res).any():
            continue
        j = int(np.nanargmin(res))
        amp = -res[j]
        if amp < max(0.3 * dip.amp, 0.20):
            continue
        amps[k] = amp
        # width across the ray at the minimum, detrended with the same plane
        perp = np.linspace(-12, 12, 25)
        xa = x[j] - perp * math.sin(th)
        ya = y[j] + perp * math.cos(th)
        za = grid.sample(xa, ya)
        ra = za - bg_at(xa, ya, np.full_like(perp, rr[j]))
        if np.isfinite(ra).sum() < 8:
            sty[k] = "unknown"
            continue
        ra = np.where(np.isfinite(ra), ra, 0.0)
        sm = ndimage.uniform_filter1d(ra, 3, mode="nearest")
        half = -0.5 * (sm.min())
        below = sm <= half
        c = int(np.argmin(sm))
        a = c
        while a - 1 >= 0 and below[a - 1]:
            a -= 1
        b = c
        while b + 1 < below.size and below[b + 1]:
            b += 1
        wids[k] = perp[b] - perp[a]
        pk, _ = signal.find_peaks(sm, prominence=0.08)
        flat = (np.mean(np.abs(np.diff(sm[max(a, 1):max(b, 2)]))) < 0.01
                if b > a else False)
        left = [q for q in pk if perp[q] < perp[a]]
        right = [q for q in pk if perp[q] > perp[b]]
        if len(left) >= 1 and len(right) >= 1:
            sty[k] = "banked"
        elif flat:
            sty[k] = "flat"
        else:
            sty[k] = "v"
    good = np.isfinite(amps)
    outward_frac = float(good.mean())
    pairs = 0
    best = (0.0, np.nan)
    for k in range(n_rays):
        j = (k + n_rays // 2) % n_rays
        if good[k] and good[j]:
            pairs += 1
            a = min(amps[k], amps[j])
            if a > best[0]:
                best = (float(a), 360.0 * k / n_rays)
    used = [sty[k] for k in range(n_rays) if good[k] and sty[k]]
    style = "none"
    if used:
        vals, cnt = np.unique(used, return_counts=True)
        style = str(vals[int(np.argmax(cnt))])
    return Continuation(n_rays, pairs, best[0],
                        best[1] if np.isfinite(best[1]) else np.nan,
                        style, outward_frac,
                        "opposite rays with an outward low" if pairs else
                        "no through-going low")


# ---------------------------------------------------------------------------
# local relief + hillshade (context rasters for the evidence packet)
# ---------------------------------------------------------------------------

def local_relief(grid: Grid, sigma_m=15.0) -> Grid:
    s = sigma_m / grid.res
    z = grid.z.copy()
    bad = ~np.isfinite(z)
    zf = np.where(bad, np.nanmedian(z), z)
    sm = ndimage.gaussian_filter(zf, s, mode="nearest")
    r = z - sm
    r[bad] = np.nan
    return Grid(r, grid.res, grid.x0, grid.y0, y_ascending=grid.y_ascending,
                epsg=grid.epsg, name=f"{grid.name}/relief{int(sigma_m)}")


def hillshade(grid: Grid, az=315.0, alt=45.0, zf=2.0) -> np.ndarray:
    z = grid.z
    zf_arr = np.array(grid.z, dtype=float)
    bad = ~np.isfinite(zf_arr)
    if bad.any():
        zf_arr[bad] = np.nanmedian(zf_arr)
    gy, gx = np.gradient(zf_arr, grid.res)
    slope = np.arctan(zf * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a = math.radians(az)
    e = math.radians(alt)
    hs = (math.sin(e) * np.cos(slope) +
          math.cos(e) * np.sin(slope) * np.cos(a - aspect))
    hs = np.clip(hs, 0, 1)
    hs[bad] = np.nan
    return hs
