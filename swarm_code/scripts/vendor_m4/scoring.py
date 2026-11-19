"""scoring.py -- match detections to synthetic ground truth and score them.

Two layers, kept apart on purpose:

  raw detection   what the detector emits (broad recall)
  screened        what a detector may hand to a skeptic: no reject-class flag
                  and geometry inside the published target envelope
                  (ditch 3-15 m wide, dip >= 0.5 m, near-constant radius).

`certainty` for a screened detection is only ever `Possible`; `Rejected` is used
for anything that trips a reject flag, with the flags as the reason.  `Strong`
and `Known` are coordinator outcomes (record + literature check) and this module
never assigns them.
"""
from __future__ import annotations

import math

import numpy as np

REJECT_FLAGS = {
    "broad_concavity", "wide_channel_gt_20m", "inconsistent_depth_around_ring",
    "basin_interior_pond_like", "arc_not_circular", "shape_irregular",
    "hole_blanked_band", "low_snr_vs_background", "interior_not_level",
    "terrace_step",
}
# flags that weaken a candidate but do not by themselves reject it
SOFT_FLAGS = {"rough_surface_canopy", "arc_only", "angular_gap", "edge_clipped",
              "shallow_dip", "asymmetric_cross_section"}


# Canopy-conditional screening thresholds.  Rationale: penetration bias and
# DTM roughness both grow with canopy height (class-2 contamination reaches tens
# of metres under closed forest), so a dip of the same size is much weaker
# evidence under forest than in pasture.  These are policy choices, stated as
# numbers so a reviewer can move them.
# The SNR floor is calibrated, not guessed: a sweep over the synthetic suite
# (det/run_synth.py + the post-hoc sweep in the validation report) shows that at
# snr_min = 4.0 all planted rings are still screened in while featureless terrain,
# a broad stream valley and a gully head are not.  Lower values triple the false
# positive rate; much higher values start to lose 0.8 m ditches on real terrain.
CANOPY_THRESHOLDS = {
    "open":   dict(canopy_max_m=3.0,  min_dip_m=0.50, snr_min=4.0),
    "mosaic": dict(canopy_max_m=10.0, min_dip_m=0.70, snr_min=4.5),
    "forest": dict(canopy_max_m=1e9,  min_dip_m=1.20, snr_min=5.0),
}
ARC_MIN_DIP_M = 1.20      # a partial ring is weaker evidence than a closed one
ARC_MAX_REL_STD = 0.45


def canopy_class(det):
    can = det.get("ctx_canopy_ann", np.nan)
    if not np.isfinite(can):
        can = det.get("ctx_canopy_disc", np.nan)
    if not np.isfinite(can):
        return "open", np.nan
    for k, v in sorted(CANOPY_THRESHOLDS.items(), key=lambda kv: kv[1]["canopy_max_m"]):
        if can < v["canopy_max_m"]:
            return k, float(can)
    return "forest", float(can)


def screened(det, min_dip=None, max_width=15.0, snr_min=None, arc_rules=True):
    """Return (bool, reason_string): may this detection be handed to a skeptic?

    Rejects on
      (a) reject-class flags (broad concavity, pond basin, blanked band, low SNR...),
      (b) absolute dip below the canopy-conditional floor (or `min_dip` if higher),
      (c) dip below snr_min x local background roughness at the ring scale,
      (d) trough width above the published ditch envelope (15 m),
      (e) partial rings (coverage < 0.60) that are shallow or angularly uneven.
    Certainty is never `Strong` from this function; it only says `Possible` or
    `Rejected` with a reason.
    """
    f = set(det.get("flags", []))
    cls, can = canopy_class(det)
    ctx = CANOPY_THRESHOLDS[cls]
    floor = ctx["min_dip_m"] if min_dip is None else max(ctx["min_dip_m"], float(min_dip))
    need_snr = ctx["snr_min"] if snr_min is None else float(snr_min)
    rej = sorted(f & REJECT_FLAGS)
    if det.get("dip_m", 0.0) < floor:
        rej.append("dip_%.2fm_below_%.2fm_%s" % (det.get("dip_m", 0.0), floor, cls))
    sig = det.get("sig_bg_m", np.nan)
    if np.isfinite(sig) and sig > 1e-6 and det.get("dip_m", 0.0) < need_snr * sig:
        rej.append("dip_below_%.1f_sigma_bg" % need_snr)
    if np.isfinite(det.get("width_m", np.nan)) and det["width_m"] > max_width:
        rej.append("width_%.0fm_gt_%.0fm" % (det["width_m"], max_width))
    if arc_rules and det.get("cov", 0.0) < 0.60:
        if det.get("dip_m", 0.0) < ARC_MIN_DIP_M:
            rej.append("arc_dip_below_%.1fm" % ARC_MIN_DIP_M)
        if np.isfinite(det.get("rel_std", np.nan)) and det["rel_std"] > ARC_MAX_REL_STD:
            rej.append("arc_rel_std_gt_%.2f" % ARC_MAX_REL_STD)
    if rej:
        return False, ";".join(rej)
    return True, ("canopy=" + cls + (";" + ";".join(sorted(f)) if f else ""))


def match_rings(dets, truth, cell, radius_tol=0.35, min_dist_m=15.0):
    """One-to-one greedy match of ring-type detections to ring-type truth."""
    pos = [t for t in truth if t["kind"] in ("ring", "square", "U")
           and t.get("expect") == "positive"]
    used = set()
    out = []
    for t in pos:
        best, bd = None, 1e18
        for k, d in enumerate(dets):
            if k in used:
                continue
            dist = math.hypot(d["col"] * cell - t["cx"], (d["row"] * cell) - t["cy"])
            if np.isfinite(t.get("R", np.nan)) and t["R"] > 0:
                tol = max(min_dist_m, radius_tol * t["R"])
                if not (0.55 * t["R"] <= d["radius_m"] <= 1.6 * t["R"]):
                    continue
            else:
                tol = min_dist_m
            if dist <= tol and dist < bd:
                best, bd = k, dist
        if best is not None:
            used.add(best)
            out.append(dict(truth=t, det=dets[best], dist_m=bd, found=True))
        else:
            out.append(dict(truth=t, det=None, dist_m=np.nan, found=False))
    return out, used


def negative_hits(dets, truth, cell, default_radius_m=250.0):
    """Detections that fall in a negative feature's zone (or anywhere, for
    scene-wide negatives)."""
    hits = []
    for t in truth:
        if t.get("expect") != "negative" and t.get("expect") != "ring_like_negative":
            continue
        scene_wide = t["kind"] in ("none", "river", "valley", "fields", "gullies")
        if t["kind"] in ("hole", "pond", "oxbow"):
            zone = max(2.0 * float(t.get("R", 0.0)), 40.0)
            for k, d in enumerate(dets):
                dist = math.hypot(d["col"] * cell - t["cx"], d["row"] * cell - t["cy"])
                if dist <= zone:
                    hits.append(dict(neg=t, det=d, idx=k, dist_m=dist))
        elif scene_wide:
            for k, d in enumerate(dets):
                hits.append(dict(neg=t, det=d, idx=k, dist_m=np.nan))
    return hits


def score_scene(dets, scene, cell, min_dip=0.5, max_width=15.0):
    truth = scene.get("truth", [])
    matches, used = match_rings(dets, truth, cell)
    nhits = negative_hits(dets, truth, cell)
    screened_flags = {id(d): screened(d, min_dip, max_width) for d in dets}
    # positives
    pos_rows = []
    for m in matches:
        d = m["det"]
        s, reason = screened_flags.get(id(d), (False, "")) if d is not None else (False, "missed")
        pos_rows.append(dict(id=m["truth"]["id"], kind=m["truth"]["kind"],
                             found=m["found"],
                             radius_err_m=(d["radius_m"] - m["truth"]["R"]) if d is not None else np.nan,
                             dist_m=m["dist_m"],
                             dip_m=d["dip_m"] if d is not None else np.nan,
                             width_m=d["width_m"] if d is not None else np.nan,
                             cov=d["cov"] if d is not None else np.nan,
                             rel_std=d["rel_std"] if d is not None else np.nan,
                             fwhm_m=d.get("fwhm_m", np.nan) if d is not None else np.nan,
                             sig_bg_m=d.get("sig_bg_m", np.nan) if d is not None else np.nan,
                             snr=d.get("snr", np.nan) if d is not None else np.nan,
                             interior_m=d.get("interior_m", np.nan) if d is not None else np.nan,
                             two_sided=d["two_sided"] if d is not None else np.nan,
                             circularity=d["circularity"] if d is not None else np.nan,
                             gap_deg=d["gap_deg"] if d is not None else np.nan,
                             shape=d["shape"] if d is not None else "",
                             screened=s, reason=reason,
                             flags=d.get("flags", []) if d is not None else []))
    other = sorted(set(id(d) for d in dets) - used)
    # negatives: a negative is a failure only if a *screened* detection lands there
    neg_rows = []
    seen = set()
    for t in truth:
        if t.get("expect") not in ("negative", "ring_like_negative"):
            continue
        rel = [h for h in nhits if h["neg"] is t]
        any_scr = [h for h in rel if screened_flags.get(id(h["det"]), (False, ""))[0]]
        neg_rows.append(dict(
            id=t["id"], kind=t["kind"], n_dets=len(rel), n_screened=len(any_scr),
            worst=(max((h["det"]["dip_m"] for h in rel), default=np.nan)),
            example_flags=(rel[0]["det"].get("flags", []) if rel else []),
            ok=(len(any_scr) == 0)))
        seen.add(id(t))
    # leftover detections that match no truth entry at all: candidate FPs
    unexplained = [dets[k] for k in range(len(dets)) if k not in used]
    unexplained_scr = [d for d in unexplained if screened_flags.get(id(d), (False, ""))[0]]
    return dict(positives=pos_rows, negatives=neg_rows,
                n_dets=len(dets), n_screened=sum(1 for v in screened_flags.values() if v[0]),
                n_unexplained=len(unexplained),
                n_unexplained_screened=len(unexplained_scr),
                unexplained_screened=[dict(row=d["row"], col=d["col"],
                                           radius_m=d["radius_m"], dip_m=d["dip_m"],
                                           width_m=d["width_m"], flags=d.get("flags"))
                                      for d in unexplained_scr])


def summarise(per_scene, dip_cut=0.5):
    """Aggregate a per-scene score dict into the headline numbers."""
    tp = sum(1 for s in per_scene.values() for r in s["positives"]
             if r["found"] and r["screened"])
    found = sum(1 for s in per_scene.values() for r in s["positives"] if r["found"])
    total = sum(len(s["positives"]) for s in per_scene.values())
    neg_fail = sum(1 for s in per_scene.values() for r in s["negatives"] if not r["ok"])
    neg_total = sum(len(s["negatives"]) for s in per_scene.values())
    return dict(positive_total=total, positive_found=found, positive_screened=tp,
                negatives_total=neg_total, negatives_with_screened_hit=neg_fail,
                total_dets=sum(s["n_dets"] for s in per_scene.values()),
                total_screened=sum(s["n_screened"] for s in per_scene.values()),
                unexplained_screened=sum(s["n_unexplained_screened"] for s in per_scene.values()))
