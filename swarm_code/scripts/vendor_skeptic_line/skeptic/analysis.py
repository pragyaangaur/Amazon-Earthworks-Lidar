"""Assemble a complete, JSON-safe diagnostic bundle for one candidate feature.

`sceptic_bundle(grid, cx, cy, ...)` is the single entry point used by the rule
engine, the control runner and the review CLI.  It never decides anything: it
measures.  Everything it measures goes into the returned dict, so a verdict can
be re-derived and argued with.

Pipeline inside:
  radial field -> azimuthal median profile -> strongest dip -> robust centre
  search -> final radial field at full resolution -> ring edge per azimuth ->
  planform fit (circle/square/ellipse/rectangle) -> ditch cross-sections ->
  interior-vs-exterior level -> nodata/border support -> canopy -> outward
  continuations.
"""
from __future__ import annotations

import math

import numpy as np

from .grid import Grid
from . import primitives as P

SCHEMA = "m6.skeptic/1"

DEFAULTS = dict(
    # search band around the detector's size hint, as a fraction of size/2
    band=(0.50, 1.50),
    # tighter band when the radius itself came from the profile screen
    band_auto=(0.78, 1.30),
    n_az=180,
    trend_window_m=41.0,
    screen_max_m=230.0,
    screen_top=3,
    auto_fallback_quality=0.60,
    # ring support
    min_coverage=0.55,
    max_gap_closed_deg=130.0,
    max_gap_u_deg=205.0,
    # planform regularity
    max_planform_resid_rel=0.11,
    max_edge_rmad=0.13,
    # ditch cross-section
    min_two_sided_frac=0.55,
    max_step_frac=0.35,
    # size limits for a real earthwork ditch
    min_size_m=30.0,
    max_size_m=400.0,
    min_width_m=2.5,
    ditch_width_ref_m=15.0,   # published earthwork ditch range is 3-15 m
    max_width_m=18.0,         # beyond this it is a channel, not a ditch
    min_depth_m=0.30,
    max_depth_m=5.0,
    # holes / borders
    hole_ring_min_m=10.0,
    hole_ring_warn_m=20.0,
    max_nodata_ditch_band=0.50,   # real EBA strips under canopy run 20-40% void
    max_nodata_annulus=0.50,
    max_nodata_interior=0.35,
    # level / pond / pad
    pond_off_m=-0.40,
    pond_p75_m=-0.15,
    pad_sigma_m=0.09,
    # canopy
    canopy_warn_m=5.0,
    canopy_dense_m=12.0,
)


def void_fraction(grid: Grid, cx, cy, R=25.0):
    """Fraction of no-data cells in a disc; used to explain a missing dip."""
    sl = grid.window(cx, cy, R)
    if sl is None:
        return None
    si, sj = sl
    nd = ~np.isfinite(grid.z[si, sj])
    ii, jj = np.meshgrid(np.arange(si.start, si.stop),
                         np.arange(sj.start, sj.stop), indexing="ij")
    x, y = grid.cell_center(ii, jj)
    m = np.hypot(x - cx, y - cy) <= R
    return float(nd[m].mean()) if m.any() else None


def _num(v, nd=3):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if not np.isfinite(v) else round(v, nd)
    return v


def _clean(d, nd=3):
    if isinstance(d, dict):
        return {k: _clean(v, nd) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v, nd) for v in d]
    if isinstance(d, np.ndarray):
        return _clean(d.tolist(), nd)
    if isinstance(d, np.bool_):
        return bool(d)
    return _num(d, nd)


# ---------------------------------------------------------------------------
# radial screening: where are the ring-scale dips around a point?
# ---------------------------------------------------------------------------

def screen_radii(grid: Grid, cx, cy, r_max=230.0, n_az=90, top=4, min_amp=0.25):
    """Strongest ring-scale dips in the radial profile around a point.

    Used when a detector gives a position but no reliable size, and by the
    control runner to check that the intended feature is the dominant one.
    """
    from scipy import signal as _sig
    f = P.radial_field(grid, cx, cy, r_max, n_az=n_az)
    pr = P.radial_profile(f)
    s = -pr.resid
    peaks, props = _sig.find_peaks(s, prominence=min_amp * 0.5)
    if peaks.size == 0:
        return []
    order = np.argsort(-np.asarray(props["prominences"]))[:top]
    out = []
    for k in order:
        i = int(peaks[k])
        try:
            w = float(_sig.peak_widths(s, [i], rel_height=0.5)[0][0]) * pr.dr
        except Exception:
            w = float("nan")
        out.append(dict(r=float(pr.r[i]), amp=float(props["prominences"][k]),
                        width_m=w,
                        snr=float(props["prominences"][k] / max(pr.noise, 1e-3))))
    return out


# ---------------------------------------------------------------------------
# one measurement at one assumed radius
# ---------------------------------------------------------------------------

def _measure(grid, cx, cy, R_est, band, p, chm, refine=True, seed_dip=None):
    r_lo = max(3.0, band[0] * R_est)
    r_hi = band[1] * R_est
    r_max = max(2.6 * r_hi, r_hi + 60.0)
    field = P.radial_field(grid, cx, cy, r_max, n_az=p["n_az"])
    prof = P.radial_profile(field, trend_window_m=p["trend_window_m"])
    dip = seed_dip if seed_dip is not None else \
        P.find_dip(prof, r_lo=r_lo, r_hi=r_hi, min_amp=0.08)
    if dip is None:
        return None
    if seed_dip is not None:
        # the seed came from a profile measured at the same centre; re-evaluate
        # its amplitude/width against this (identical) profile so the numbers
        # recorded are consistent
        d2 = P.find_dip(prof, r_lo=max(3.0, 0.8 * seed_dip.r),
                        r_hi=1.25 * seed_dip.r, min_amp=0.08)
        if d2 is not None:
            dip = d2
    ref = dict(offset_m=0.0, dx=0.0, dy=0.0, accepted=False, n_eval=0)
    edge0 = P.ring_edge(field, prof, dip, want_widths=False)
    pcl0 = P.polar_closure(edge0)
    ref["polar_closed_at_reported_centre"] = bool(pcl0.closed)
    ref["max_gap_arc_at_reported_centre_m"] = float(pcl0.max_gap_arc_m)
    if refine:
        cx, cy, ref = P.refine_centre(grid, cx, cy, dip, R=dip.r)
        field = P.radial_field(grid, cx, cy, r_max, n_az=p["n_az"])
        prof = P.radial_profile(field, trend_window_m=p["trend_window_m"])
        d2 = P.find_dip(prof, r_lo=r_lo, r_hi=r_hi, min_amp=0.08)
        if d2 is not None:
            dip = d2

    edge = P.ring_edge(field, prof, dip)
    plan = P.fit_planform(edge)
    cross = P.cross_sections(grid, edge, dip, prof)
    R = dip.r
    level = P.level_stats(grid, cx, cy, R)
    sup = P.support_stats(grid, cx, cy, R, max(1.0, 0.5 * dip.width_m))
    can = P.canopy_stats(chm, cx, cy, R)
    # outward search must start beyond the *outermost* part of the ditch, or a
    # square's corner ditch (1.41 R) is mistaken for an outward continuation
    r_max_edge = (float(np.nanmax(edge.r[edge.ok])) if edge.ok.any() else R)
    cont = P.continuations(grid, cx, cy, R, dip, prof, plane=level.plane,
                           r_inner=1.25 * r_max_edge,
                           r_outer=2.4 * r_max_edge)

    w_med = (float(np.nanmedian(edge.width_m[edge.ok]))
             if edge.ok.any() else float("nan"))
    dp_amp = float(dip.amp)
    pcl = P.polar_closure(edge, bridge_m=min(15.0, max(12.0, 1.5 * (w_med if np.isfinite(w_med) else 0.0))))
    bcl = P.binary_closure(grid, cx, cy, R, dp_amp)
    if ref.get("polar_closed_at_reported_centre") is False and pcl.closed \
            and r_max_edge > 0 and ref["offset_m"] > 0.30 * r_max_edge:
        ref["closure_only_after_shift"] = True
    geom = dict(
        status="ok", cx=float(cx), cy=float(cy), centre_refine=ref,
        radius_m=float(R), diameter_m=float(2 * R),
        outer_diameter_m=float(2 * R + (w_med if np.isfinite(w_med) else 0.0)),
        dip=dip.as_dict(), dip_width_m=float(dip.width_m), dip_amp_m=float(dip.amp),
        edge=dict(coverage=float(edge.coverage), gap_deg=float(edge.gap_deg),
                  r_median=float(edge.r_median), r_rmad=float(edge.r_rmad),
                  band_lo=float(edge.band_lo), band_hi=float(edge.band_hi),
                  n_az=int(edge.az.size), n_ok=int(edge.ok.sum()),
                  amp_median=(float(np.nanmedian(edge.amp[edge.ok]))
                              if edge.ok.any() else float("nan")),
                  width_median=w_med),
        planform=(plan.as_dict() if plan else None),
        cross=cross.as_dict(), level=level.as_dict(),
        support=sup.as_dict(), canopy=can.as_dict(),
        continuation=cont.as_dict(),
        polar_closure=pcl.as_dict(), plan_closure=bcl.as_dict(),
        edge_points=dict(az=[float(a) for a in edge.az[edge.ok]],
                         r=[float(v) for v in edge.r[edge.ok]],
                         amp=[float(v) for v in edge.amp[edge.ok]],
                         width=[float(v) for v in edge.width_m[edge.ok]]),
        radial_profile=dict(r=[float(v) for v in prof.r[::2]],
                            prof=[float(v) for v in prof.prof[::2]],
                            background=[float(v) for v in prof.background[::2]],
                            resid=[float(v) for v in prof.resid[::2]],
                            noise=float(prof.noise)),
    )
    geom["quality"] = _quality(geom, p)
    return geom


def _quality(g, p):
    """Heuristic ringness in [0, 1].

    Used for two things only: choosing between candidate radii, and deciding
    whether a detector's size hint produced anything ring-like at all (below
    `auto_fallback_quality` the free screen gets a second look).  It is *not* the
    verdict — the rules decide that — so it is deliberately dominated by the
    closedness of the band, which is the property a false ring lacks.
    """
    q = 0.0
    pc = g.get("polar_closure") or {}
    cov = g["edge"]["coverage"] or 0.0
    if pc.get("closed"):
        q += 0.30
    elif cov > 0.85:
        q += 0.20
    elif cov > 0.65:
        q += 0.08
    pl = g.get("planform")
    if pl:
        q += 0.25 * max(0.0, 1.0 - (pl["resid_rel"] or 1.0) / 0.12)
    cr = g["cross"]
    if cr.get("two_sided_frac") is not None:
        q += 0.20 * cr["two_sided_frac"]
    q += 0.25 * min(1.0, g["dip_amp_m"] / 0.6)
    if g["dip_width_m"] <= p["max_width_m"]:
        q += 0.10
    return float(min(1.0, q))


def _screen_and_measure(grid, cx, cy, p, chm, refine, bundle):
    """Free radial screen (no size hint): try the strongest few dips, keep the
    best ring.  Also used as a fallback when a size hint yields nothing ring-like."""
    cands = screen_radii(grid, cx, cy, r_max=p["screen_max_m"], n_az=90,
                         top=p["screen_top"])
    best = None
    for c in cands:
        g = _measure(grid, cx, cy, c["r"], p["band_auto"], p, chm, refine)
        if g is None:
            continue
        bundle["alternatives"].append(
            dict(r_screen=c["r"], amp_screen=c["amp"], quality=g["quality"],
                 radius_m=g["radius_m"], diameter_m=g["diameter_m"]))
        if best is None or g["quality"] > best["quality"]:
            best = g
    if best is not None:
        bundle["notes"].append(
            f"free screen: {len(cands)} radii tried, best quality "
            f"{best['quality']:.2f} at {best['outer_diameter_m']:.0f} m")
    return best


# ---------------------------------------------------------------------------
# the main entry point
# ---------------------------------------------------------------------------

def sceptic_bundle(grid: Grid, cx, cy, size_hint=None, kind="ring", chm=None,
                   fid="", transect="", state="", detector="",
                   params=None, refine=True):
    """Measure everything the rules need.  Returns a JSON-safe dict."""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    notes = []
    bundle = dict(schema=SCHEMA, id=fid, transect=transect, state=state,
                  type=kind, detector=detector,
                  centre_in=dict(x=float(cx), y=float(cy)),
                  size_hint_m=(None if size_hint is None else float(size_hint)),
                  grid=grid.origin_dict(), geometry={}, alternatives=[],
                  verdict=None, notes=notes)

    if not bool(np.all(grid.contains(cx, cy))):
        notes.append("candidate centre lies outside the DTM")
        bundle["geometry"] = dict(status="outside_grid")
        return bundle
    void_r25 = void_fraction(grid, cx, cy, 25.0)

    if size_hint is None:
        best = _screen_and_measure(grid, cx, cy, p, chm, refine, bundle)
        if best is None:
            notes.append("no ring-scale dip found within the screening radius")
            bundle["geometry"] = dict(status="no_dip", void_frac_centre=void_r25)
            return bundle
        bundle["geometry"] = _clean(best)
        return bundle

    R_est = 0.5 * float(size_hint)
    r_lo = max(3.0, p["band"][0] * R_est)
    r_hi = max(r_lo + 2 * grid.res, p["band"][1] * R_est)
    r_max = max(2.6 * r_hi, r_hi + 60.0)
    f0 = P.radial_field(grid, cx, cy, r_max, n_az=p["n_az"])
    pr0 = P.radial_profile(f0, trend_window_m=p["trend_window_m"])
    seeds = P.find_dips(pr0, r_lo=r_lo, r_hi=r_hi, top=p["screen_top"], min_amp=0.10)
    if seeds:
        best = None
        for sd in seeds:
            gg = _measure(grid, cx, cy, R_est, p["band"], p, chm, refine,
                          seed_dip=sd)
            if gg is None:
                continue
            bundle["alternatives"].append(
                dict(r_seed=sd.r, amp_seed=sd.amp, quality=gg["quality"],
                     radius_m=gg["radius_m"], diameter_m=gg["diameter_m"]))
            if best is None or gg["quality"] > best["quality"]:
                best = gg
        if best is not None:
            if len(seeds) > 1:
                notes.append(f"{len(seeds)} candidate radii evaluated in the size-hint "
                             f"band; best chosen by ring quality")
            hint_diam = 2.0 * float(size_hint)
            got = best["outer_diameter_m"]
            best["size_hint_agreement"] = float(got / hint_diam) if hint_diam else None
            # If the detector's size hint produced nothing ring-like, re-screen the
            # whole radius range before rejecting: the hint may simply not describe
            # the feature that is there (real 0414: hit radius 51 m, true 41 m).
            if best["quality"] < p["auto_fallback_quality"]:
                alt = _screen_and_measure(grid, cx, cy, p, chm, refine, bundle)
                if alt is not None and alt["quality"] > best["quality"] + 0.05:
                    alt["size_hint_agreement"] = float(
                        alt["outer_diameter_m"] / hint_diam) if hint_diam else None
                    alt["size_hint_miss"] = True
                    notes.append(
                        f"the size hint ({hint_diam:.0f} m) did not describe the best "
                        f"ring-like feature in the window; the free screen found "
                        f"{alt['outer_diameter_m']:.0f} m instead")
                    bundle["geometry"] = _clean(alt)
                    return bundle
            bundle["geometry"] = _clean(best)
            return bundle
    # No usable seed in the hinted band: look at the whole radius range before
    # concluding that nothing is there.
    alt = _screen_and_measure(grid, cx, cy, p, chm, refine, bundle)
    if alt is not None:
        notes.append(f"the size hint ({2.0*R_est:.0f} m) produced no usable dip; the "
                     f"free screen found {alt['outer_diameter_m']:.0f} m")
        alt["size_hint_miss"] = True
        bundle["geometry"] = _clean(alt)
        return bundle
    bundle["geometry"] = dict(status="no_dip", void_frac_centre=void_r25,
                              screen=screen_radii(grid, cx, cy,
                                                  r_max=min(p["screen_max_m"],
                                                            1.6 * R_est + 60),
                                                  n_az=90))
    return bundle
