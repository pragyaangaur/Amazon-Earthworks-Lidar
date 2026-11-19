"""Rule engine: measurements -> a written verdict.

Certainty vocabulary is fixed by the project: Known, Strong, Possible, Rejected.
This module can only ever return **Rejected** or **Possible**, because:

  * Known requires a match in the known-site table or the literature, which m6
    does not have (offline);
  * Strong requires, by the project's own definition, that *no record is found
    within 2 km and no natural or modern explanation is left*.  m6 can only
    establish the second half of that, so it reports `strong_eligible` (geometry
    clean, no explanation left, open canopy) and leaves the record check to the
    coordinator and m7.

Every check is emitted with its measured value, its threshold and a sentence, so
a rejection can be argued with rather than taken on trust.
"""
from __future__ import annotations

import math

import numpy as np

from .analysis import DEFAULTS

# labels used in diagnostics and in the control expectations
LABELS = ("outside", "hole", "edge_fail", "weak_dip", "not_closed", "irregular",
          "one_sided_step", "natural_channel", "pond", "pad_flat", "size",
          "too_deep", "data_void")


def _f(v, nd=2, unit=""):
    if v is None:
        return "n/a"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}{unit}"


def _chk(cid, status, value=None, threshold=None, text="", label=None,
         hard=False, context=False):
    """A check result.  `context=True` marks a warning that does not question the
    ditch geometry itself (something nearby is unexplained, or the ditch bottom is
    unobserved but its walls are) and so does not reduce the geometric grade."""
    return dict(id=cid, status=status, value=value, threshold=threshold,
                text=text, label=label, hard=hard, context=context)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def _c_status(b, p):
    g = b.get("geometry") or {}
    st = g.get("status")
    if st == "outside_grid":
        return _chk("GEOM", "fail", None, None,
                    "The candidate centre is outside the DTM.", "outside", True)
    if st != "ok":
        void = g.get("void_frac_centre")
        if void is not None and void > 0.30:
            return _chk("GEOM", "fail", void, 0.30,
                        f"No closed dip exists because {_f(100*void,0,'%')} of the "
                        f"ground within 25 m of this position has no returns: an "
                        f"empty-data halo, not a feature.", "hole", True)
        return _chk("GEOM", "fail", st, "ok",
                    "No closed dip of the required strength exists in the radial "
                    "profile around this position, so there is no ring to explain.",
                    "not_closed", True)
    return None


def _c_dip(b, p):
    g = b["geometry"]
    amp = g["dip_amp_m"]
    snr = g["dip"]["snr"]
    if amp < p["min_depth_m"]:
        return _chk("DIP_DEPTH", "fail", amp, p["min_depth_m"],
                    f"The dip is only {_f(amp,2,' m')} deep, below the "
                    f"{_f(p['min_depth_m'],2,' m')} floor for an earthwork ditch.",
                    "weak_dip", True)
    if snr is not None and snr < 3.0:
        return _chk("DIP_SNR", "fail", snr, 3.0,
                    f"The dip is only {_f(snr,1,'x')} the local noise level.", "weak_dip", True)
    if amp > p["max_depth_m"]:
        return _chk("DIP_DEPTH", "fail", amp, p["max_depth_m"],
                    f"The dip is {_f(amp,2,' m')} deep, deeper than any known "
                    f"geoglyph ditch; a road cut, channel or modern cut is more likely.",
                    "too_deep", True)
    return _chk("DIP_DEPTH", "pass", amp, (p["min_depth_m"], p["max_depth_m"]),
                f"Dip depth {_f(amp,2,' m')} (SNR {_f(snr,1,'x')}).")


def _c_width(b, p):
    g = b["geometry"]
    w = g["edge"]["width_median"]
    if w is None:
        return _chk("DITCH_WIDTH", "na", None, None, "Radial width not measurable.")
    if w > p["max_width_m"] or w < p["min_width_m"]:
        return _chk("DITCH_WIDTH", "fail", w, (p["min_width_m"], p["max_width_m"]),
                    f"The dip is {_f(w,1,' m')} wide across the radial profile, "
                    f"clearly outside the 3-15 m earthwork ditch range: a channel, "
                    f"an oxbow or a broad valley is more likely.", "irregular", True)
    if w > p["ditch_width_ref_m"] or w < 2.0:
        return _chk("DITCH_WIDTH", "warn", w,
                    (p["min_width_m"], p["ditch_width_ref_m"]),
                    f"The radial dip width is {_f(w,1,' m')} against the published "
                    f"3-15 m ditch range: wide enough to be a small channel.")
    return _chk("DITCH_WIDTH", "pass", w, (p["min_width_m"], p["ditch_width_ref_m"]),
                f"Ditch width {_f(w,1,' m')}.")


def _c_size(b, p):
    g = b["geometry"]
    d = g["outer_diameter_m"]
    if d < p["min_size_m"]:
        return _chk("SIZE", "fail", d, (p["min_size_m"], p["max_size_m"]),
                    f"Only {_f(d,0,' m')} across, below the 30 m screen floor.",
                    "size", True)
    if d > p["max_size_m"]:
        return _chk("SIZE", "fail", d, (p["min_size_m"], p["max_size_m"]),
                    f"{_f(d,0,' m')} across, above the 400 m screen ceiling for "
                    f"targets; a natural basin or a full meander loop is more likely.",
                    "size", True)
    return _chk("SIZE", "pass", d, (p["min_size_m"], p["max_size_m"]),
                f"{_f(d,0,' m')} across, inside the target range.")


def _c_coverage(b, p):
    """Closure: is the dip a closed band around the centre, or an open arc?

    Measured in polar space and reported in metres of arc, because a ditch
    entrance is a fixed few metres wide while a channel mouth is tens of metres.
    """
    g = b["geometry"]
    pc = g.get("polar_closure") or {}
    pl = g.get("planform") or {}
    gap = g["edge"]["gap_deg"]
    cov = g["edge"]["coverage"]
    if not pc:
        return _chk("CLOSURE", "na", None, None, "Closure not measurable.")
    if pc.get("closed"):
        return _chk("CLOSURE", "pass", (pc["max_gap_arc_m"], cov),
                    pc["bridge_m"],
                    f"Closed band: {_f(100*cov,0,'%')} azimuthal coverage, longest "
                    f"interruption {_f(pc['max_gap_arc_m'],1,' m')} of arc "
                    f"({pc['n_gaps_bridged']} entrance-sized gap(s) bridged).")
    rect = (pl.get("shape") in ("square", "rounded-square")
            and (pl.get("resid_rel") or 1.0) <= p["max_planform_resid_rel"])
    if rect:
        return _chk("CLOSURE", "warn", (pc["max_gap_arc_m"], gap),
                    pc["bridge_m"],
                    f"Open on one side: {_f(pc['max_gap_arc_m'],0,' m')} of arc has "
                    f"no ditch, but the planform is rectilinear ({pl.get('shape')}), "
                    f"so a U-shaped enclosure remains plausible.")
    return _chk("CLOSURE", "fail", (pc["max_gap_arc_m"], gap), pc["bridge_m"],
                f"The low is open: {_f(pc['max_gap_arc_m'],0,' m')} of arc carries no "
                f"ditch (coverage {_f(100*cov,0,'%')}) and the planform is circular, "
                f"not rectilinear. A meander bend, an oxbow mouth or a valley "
                f"shoulder produces exactly this, so it is not promoted.",
                "not_closed", True)


def _c_regularity(b, p):
    g = b["geometry"]
    pl = g["planform"]
    e = g["edge"]
    if not pl:
        return _chk("REGULARITY", "fail", None, p["max_planform_resid_rel"],
                    "The dip radius could not be fitted by any regular planform.",
                    "irregular", True)
    res = pl["resid_rel"]
    rmad = e["r_rmad"]
    if res > p["max_planform_resid_rel"] and (rmad is None or rmad > p["max_edge_rmad"]):
        return _chk("REGULARITY", "fail", (res, rmad),
                    (p["max_planform_resid_rel"], p["max_edge_rmad"]),
                    f"The dip radius wanders ({_f(100*res,0,'%')} of the radius "
                    f"around the best {pl['shape']} fit, {_f(100*rmad,0,'%')} raw "
                    f"scatter): the low does not follow one circle or square, so a "
                    f"meander, scroll bar or gully is more likely than a ditch.",
                    "irregular", True)
    if res > p["max_planform_resid_rel"]:
        return _chk("REGULARITY", "warn", (res, rmad), p["max_planform_resid_rel"],
                    f"Planform only loosely regular ({_f(100*res,0,'%')} residual).")
    return _chk("REGULARITY", "pass", (res, rmad), p["max_planform_resid_rel"],
                f"Planform {pl['shape']} with {_f(100*res,1,'%')} radius residual "
                f"(circle fit {_f(100*pl['circle_resid_rel'],1,'%')}).")


def _c_cross(b, p):
    g = b["geometry"]
    cr = g["cross"]
    two = cr["two_sided_frac"]
    step = cr["step_frac"]
    if two is None or step is None:
        return _chk("CROSS_SECTION", "na", None, None,
                    "Cross-sections could not be measured (too little ground data).")
    if step > p["max_step_frac"]:
        return _chk("CROSS_SECTION", "fail", (two, step), p["max_step_frac"],
                    f"{_f(100*step,0,'%')} of cross-sections are one-sided steps, "
                    f"with ground at trend level on one side only: a terrace scarp, "
                    f"building pad, road cut or strip edge, not a ditch.",
                    "one_sided_step", True)
    if two < p["min_two_sided_frac"]:
        return _chk("CROSS_SECTION", "fail", (two, step), p["min_two_sided_frac"],
                    f"Only {_f(100*two,0,'%')} of cross-sections rise back to trend "
                    f"level on both sides of the dip.", "one_sided_step", True)
    return _chk("CROSS_SECTION", "pass", (two, step), p["min_two_sided_frac"],
                f"Two-sided ditch in {_f(100*two,0,'%')} of cross-sections, "
                f"bank in {_f(100*cr['bank_frac'],0,'%')}.")


def _c_level(b, p):
    g = b["geometry"]
    lv = g["level"]
    med, p75, sig = lv["interior_off_m"], lv["interior_p75_m"], lv["interior_sigma_m"]
    sup = g["support"]
    if med is None:
        # interior void: caught by the hole check if the hole is small, else here
        if (sup["nodata_frac_interior"] or 0) > p["max_nodata_interior"]:
            return _chk("INTERIOR_LEVEL", "fail", sup["nodata_frac_interior"],
                        p["max_nodata_interior"],
                        "The interior has almost no ground returns, so the level "
                        "cannot be tested and any ring is an interpolation artefact.",
                        "data_void", True)
        return _chk("INTERIOR_LEVEL", "na", None, None,
                    "Interior level not measurable.")
    if med < p["pond_off_m"] and p75 < p["pond_p75_m"]:
        return _chk("INTERIOR_LEVEL", "fail", (med, p75), p["pond_off_m"],
                    f"The interior sits {_f(-med,2,' m')} below the surrounding "
                    f"trend (p75 {_f(p75,2,' m')}): a closed depression such as a "
                    f"cattle pond, tank or excavation, not a ring ditch.",
                    "pond", True)
    if (p75 is not None and p75 > 0.25 and sig is not None
            and sig < p["pad_sigma_m"]):
        return _chk("INTERIOR_LEVEL", "fail", (med, p75, sig), p["pad_sigma_m"],
                    f"The interior is a flat raised pad ({_f(p75,2,' m')} above trend, "
                    f"roughness only {_f(sig,3,' m')}): a building floor, yard or "
                    f"threshing floor rather than an enclosure ditch.",
                    "pad_flat", True)
    return _chk("INTERIOR_LEVEL", "pass", (med, p75), p["pond_off_m"],
                f"Interior is at the surrounding level (p75 {_f(p75,2,' m')}).")


def _c_holes(b, p):
    """Ground-data support: is the ring drawn through a data void, or does the
    ditch merely hold water?

    The distinction that matters on real transects: a no-data patch *inside* the
    ditch band is the ditch bottom (water, dense vegetation) and the dip is still
    measured from the observed walls; a no-data patch *beside* the ditch is what
    produces a fake ring in a smoothed relief model.
    """
    g = b["geometry"]
    sup = g["support"]
    nfb = sup.get("nodata_frac_ditch_band")
    nda = sup.get("nodata_frac_annulus")
    gap = sup.get("ring_path_gap_m")
    ndi = sup.get("nodata_frac_interior")
    off = sup.get("void_r_offset_m")
    w = g.get("dip_width_m") or 6.0
    on_ditch = off is not None and abs(off) <= max(1.5 * w, 8.0)
    if nfb is not None and nfb > p["max_nodata_ditch_band"]:
        return _chk("NODATA", "fail", nfb, p["max_nodata_ditch_band"],
                    f"{_f(100*nfb,0,'%')} of the ditch band has no ground returns; "
                    f"there is too little of the ditch left to measure.",
                    "hole", True)
    if (gap is not None and gap < p["hole_ring_min_m"]
            and not on_ditch and (nfb or 0) > 0.10):
        return _chk("NODATA", "fail", (gap, nfb), p["hole_ring_min_m"],
                    f"The ditch passes {_f(gap,1,' m')} from a data hole that is not "
                    f"the ditch itself (voids centred {_f(off,1,' m')} off the ring): "
                    f"local relief within 15 m of a hole is masked by the project "
                    f"rules, so a raised rim here can be an empty-data halo.",
                    "hole", True)
    if nfb is not None and nfb > 0.15 and on_ditch:
        return _chk("NODATA", "warn", (nfb, off), 0.15,
                    f"{_f(100*nfb,0,'%')} of the ditch band has no ground returns, "
                    f"but those voids sit on the ditch ({_f(off,1,' m')} off the ring "
                    f"centre line): most likely water or dense vegetation in the "
                    f"ditch bottom, and the dip is measured from the observed walls.",
                    context=True)
    if gap is not None and gap < p["hole_ring_warn_m"]:
        return _chk("NODATA", "warn", gap, p["hole_ring_warn_m"],
                    f"The ditch passes {_f(gap,1,' m')} from a data hole: halo risk.")
    if ndi is not None and ndi > p["max_nodata_interior"]:
        return _chk("NODATA", "fail", ndi, p["max_nodata_interior"],
                    f"{_f(100*ndi,0,'%')} of the interior has no ground returns.",
                    "data_void", True)
    if nda is not None and nda > p["max_nodata_annulus"]:
        return _chk("NODATA", "warn", nda, p["max_nodata_annulus"],
                    f"{_f(100*nda,0,'%')} of the wider annulus has no ground "
                    f"returns; the local relief there is interpolated.")
    return _chk("NODATA", "pass", (nfb, gap), p["hole_ring_min_m"],
                f"Ground data present in the ditch band "
                f"({_f(100*(1-(nfb or 0)),0,'%')}); nearest hole "
                f"{_f(gap,1,' m')} from the ditch.")


def _c_edge(b, p):
    g = b["geometry"]
    sup = g["support"]
    if sup["clipped"]:
        return _chk("TRANSECT_EDGE", "warn", sup["border_dist_ring_m"], None,
                    f"The ring reaches the transect border (centre "
                    f"{_f(sup['border_dist_centre_m'],0,' m')} from the edge): it "
                    f"cannot be closed or confirmed inside this DTM.")
    return _chk("TRANSECT_EDGE", "pass", sup["border_dist_centre_m"], None,
                f"Fully inside the transect ({_f(sup['border_dist_centre_m'],0,' m')} "
                f"from the border).")


def _c_continuation(b, p):
    g = b["geometry"]
    co = g["continuation"]
    amp = g["dip_amp_m"]
    cov = g["edge"]["coverage"]
    pl = g["planform"]
    pairs = co["pairs"]
    pa = co["pair_amp_m"]
    scale_ok = (pa is not None and amp is not None
                and pa <= max(3.0 * amp, 1.0))
    if (pairs >= 2 and co["style"] == "v" and pa is not None and amp is not None
            and pa >= 0.6 * amp and scale_ok
            and (cov is not None and cov < 0.90)
            and (pl is None or pl["resid_rel"] > 0.09)):
        return _chk("CONTINUATION", "fail", (pairs, pa, co["style"]),
                    "0.6 * dip depth",
                    f"A V-shaped low continues outward on {pairs} near-opposite "
                    f"ray(s) ({_f(pa,2,' m')} deep) and the ring is not closed: a "
                    f"natural channel passes through this point.", "natural_channel", True)
    if (pairs >= 2 and co["style"] in ("v", "flat") and pa is not None and amp
            and pa >= 0.6 * amp and scale_ok):
        return _chk("CONTINUATION", "warn", (pairs, pa, co["style"]),
                    "0.6 * dip depth",
                    f"An outward low continues on {pairs} near-opposite ray(s) "
                    f"({_f(pa,2,' m')} deep, {co['style']}); if it is an attached "
                    f"sunken road that supports the earthwork reading, if it is a "
                    f"channel the ring is suspect.", context=True)
    return _chk("CONTINUATION", "pass", (pairs, pa, co["style"]), None,
                "No through-going low beyond the ring.")


def _c_canopy(b, p):
    g = b["geometry"]
    ca = g["canopy"]
    m = ca["canopy_median_m"]
    if m is None:
        return _chk("CANOPY", "na", None, None, "No canopy model supplied.")
    if m >= p["canopy_dense_m"]:
        return _chk("CANOPY", "warn", m, p["canopy_dense_m"],
                    f"Under {_f(m,1,' m')} median canopy: the ground surface is "
                    f"interpolated from sparse returns, so the ditch depth is the "
                    f"only evidence and it cannot be plotted cleanly. Grade capped.")
    if m >= p["canopy_warn_m"]:
        return _chk("CANOPY", "warn", m, p["canopy_warn_m"],
                    f"Under {_f(m,1,' m')} canopy: ground returns are partial.")
    return _chk("CANOPY", "pass", m, p["canopy_warn_m"],
                f"Open canopy ({_f(m,1,' m')}).")


# Order matters: checks that describe *the site* (data voids, a closed
# depression, a raised pad, the transect border) run before checks that describe
# the ditch, because a pond with a water hole is a pond first and a weak dip
# second, and reporting it as a weak dip would be the wrong explanation.
CHECKS = (_c_status, _c_holes, _c_level, _c_edge, _c_dip, _c_width,
          _c_size, _c_coverage, _c_regularity, _c_cross, _c_continuation,
          _c_canopy)


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def judge(bundle, params=None):
    """Turn a measurement bundle into a verdict dict (also stored in the bundle)."""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    checks = []
    for fn in CHECKS:
        try:
            c = fn(bundle, p)
        except Exception as exc:  # a measurement gap must not crash a review
            c = _chk(fn.__name__, "na", None, None, f"check failed: {exc!r}")
        if c:
            checks.append(c)
    hard = [c for c in checks if c["status"] == "fail" and c["hard"]]
    warns = [c for c in checks if c["status"] == "warn"]
    passes = [c for c in checks if c["status"] == "pass"]
    nas = [c for c in checks if c["status"] == "na"]

    g = bundle.get("geometry") or {}
    geom_ok = g.get("status") == "ok"
    res = dict(checks=checks, certainty=None, geometric_grade="none",
               strong_eligible=False, blocking=[c["label"] for c in hard],
               open_questions=[c["id"] for c in warns],
               primary_label=(hard[0]["label"] if hard else None),
               needs_record_check=False, reason="")
    if hard:
        res["certainty"] = "Rejected"
        res["reason"] = "Rejected: " + " ".join(c["text"] for c in hard)
        return res
    if not geom_ok:
        res["certainty"] = "Rejected"
        res["reason"] = "Rejected: no ring geometry to assess."
        return res

    g = bundle["geometry"]
    pl = g["planform"] or {}
    shape = pl.get("shape", "ring")
    size = g["outer_diameter_m"]
    w = g["edge"]["width_median"]
    amp = g["dip_amp_m"]
    n_warn = len(warns)
    core = [c for c in passes if c["id"] in
            ("DIP_DEPTH", "DITCH_WIDTH", "SIZE", "CLOSURE", "REGULARITY",
             "CROSS_SECTION", "INTERIOR_LEVEL", "NODATA")]
    # A warning is context, not doubt, when it only says that something else is
    # nearby (an attached road, a nearby data hole) or that the ground returns are
    # sparse.  Any other warning means the ditch reading itself is uncertain.
    doubt = [c for c in warns if not c.get("context")]
    if n_warn == 0 and len(core) >= 7:
        grade = "A"
    elif len(doubt) == 0 and len(core) >= 7:
        grade = "B"
    elif len(doubt) <= 1 and len(core) >= 6:
        grade = "C"
    else:
        grade = "D"
    res["certainty"] = "Possible"
    res["geometric_grade"] = grade
    # Strong needs no doubt left in the geometry; the record/literature half of
    # the Strong definition is the coordinator's and m7's job, never m6's.
    res["strong_eligible"] = bool(len(doubt) == 0 and len(core) >= 7)
    res["doubt_warnings"] = [c["id"] for c in doubt]
    res["needs_record_check"] = True
    caveat = ""
    if warns:
        caveat = " Caveats: " + " ".join(c["text"] for c in warns)
    res["reason"] = (
        f"Geometrically consistent with a {shape} earthwork ditch: {_f(size,0,' m')} "
        f"across, ditch {_f(w,1,' m')} wide and {_f(amp,2,' m')} deep, "
        f"{_f(100*g['edge']['coverage'],0,'%')} azimuthal closure, planform residual "
        f"{_f(100*pl.get('resid_rel',float('nan')),1,'%')}, two-sided in "
        f"{_f(100*g['cross']['two_sided_frac'],0,'%')} of cross-sections, interior at "
        f"natural level, no data void, and no natural, modern or artefact explanation "
        f"found among the {len(checks)} checks."
        + caveat +
        " Certainty is capped at Possible: Known and Strong require the known-site "
        "table and literature check, which m6 cannot do offline.")
    return res


def judge_bundle(bundle, params=None):
    v = judge(bundle, params)
    bundle["verdict"] = {k: v[k] for k in
                         ("certainty", "geometric_grade", "strong_eligible",
                          "reason", "blocking", "open_questions", "primary_label",
                          "needs_record_check")}
    bundle["checks"] = v["checks"]
    return bundle, v
