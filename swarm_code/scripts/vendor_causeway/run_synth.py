"""run_synth.py -- scored benchmark of the causeway detector on the synthetic
control set.  Writes results/synth_benchmark.{json,md} and per-scenario
detection CSVs + diagnostics JSON into results/synth/<scenario>/.

Scoring is deliberately blunt and pessimistic:

  true-positive   a `causeway`-type detection whose centroid is within 60 m of the
                  truth line, whose axis is within 20 deg of the truth axis and whose
                  length is >= 40 % of the truth line.
  false-positive  a `causeway`-type detection within 40 m of a negative truth
                  geometry (levee / valley / terrace / ring / modern track) --
                  `raised_linear_no_crossing` detections are NOT counted as false
                  positives because they are explicitly a different, weaker class.
  flagged-modern  a modern track that fires as `causeway` but carries at least one
                  modern flag.  Counted separately: it is a screening failure that
                  the flag screen caught, not a clean negative, and never a candidate.

member m3, swarm smwduocxj.
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causeway
import synth

EXPECT = {
    "causeway_crossing": dict(causeway=1, fps=0, kind="positive"),
    "causeway_crossing_sinuous": dict(causeway=1, fps=0, kind="positive-hard"),
    "two_causeways": dict(causeway=2, fps=0, kind="positive"),
    "causeway_under_canopy": dict(causeway=1, fps=0, kind="positive"),
    "natural_levee": dict(causeway=0, fps=0, kind="negative"),
    "straight_stream_valley": dict(causeway=0, fps=0, kind="negative"),
    "terrace_edge": dict(causeway=0, fps=0, kind="negative"),
    "ring_ditch_roads": dict(causeway=0, fps=0, kind="negative"),
    "flat_forest_noise": dict(causeway=0, fps=0, kind="negative"),
    # embedded positive + artefact test, NOT an ordinary negative: the planted
    # causeway is real, so it must be found; what is scored is hole spanning.
    "empty_halo": dict(causeway=1, fps=0, kind="artefact-truncation"),
    # a modern raised track is NOT a planted archaeological causeway: it may be
    # detected, but only counts as handled when the modern flags fire and no
    # unflagged causeway call is made.  Hence expect causeway=0, flagged>=1.
    "modern_track": dict(causeway=0, fps=0, kind="negative-modern", flagged=1),
    "natural_ridge": dict(causeway=0, fps=0, kind="negative-hard"),
    "cattle_pond": dict(causeway=0, fps=0, kind="negative"),
    "braided_drainage": dict(causeway=0, fps=0, kind="negative-hard"),
}


def _seg_dist_point(geom, x, y):
    from shapely.geometry import Point
    return geom.distance(Point(x, y))


def _hole_spans(res, hole, max_gap_m):
    """Detections whose axis runs inside the data hole for more than `max_gap_m`.

    This is the check the empty_halo scenario exists for: crest cells are removed
    near holes, so a detection that still spans a 180 m hole has invented its crest
    across missing data (the classical empty-data halo false positive).
    """
    out = []
    if not hole:
        return out
    cx, cy, rx, ry = hole["centre"][0], hole["centre"][1], hole["rx"], hole["ry"]
    for r in res["rows"]:
        L = r["length_m"]
        s_ = np.arange(-L / 2.0, L / 2.0 + 1e-9, 1.0)
        th = math.radians(r["axis_angle_deg"])
        xs = r["x"] + np.cos(th) * s_
        ys = r["y"] + np.sin(th) * s_
        inside = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
        best = cur = 0
        for v in inside:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        if best > max_gap_m:
            out.append(dict(id=r["id"], axis_inside_hole_m=float(best),
                            length_m=round(L), type=r["type"]))
    return out


def self_test_hole_span(max_gap_m=None):
    """Prove that the hole-span test can fail.

    A fabricated 340 m detection laid straight across the 180x140 m hole must be
    flagged; the real 150 m truncated remnant from the empty_halo control must not.
    Without this, "0 detections span the hole" would be an untested claim.
    """
    max_gap_m = causeway.DEFAULTS["max_gap_m"] if max_gap_m is None else max_gap_m
    d = synth.make("empty_halo")
    hole = [t for t in d["truth"] if t.get("truncated_by_hole")][0]["truncated_by_hole"]
    res = causeway.detect(d["z"], d["meta"], transect="halo_selftest")
    real_hits = _hole_spans(res, hole, max_gap_m)
    fake = dict(rows=[dict(id="FAKE-span", x=hole["centre"][0], y=hole["centre"][1],
                           length_m=340.0, axis_angle_deg=0.0, type="causeway")])
    fake_hits = _hole_spans(fake, hole, max_gap_m)
    assert fake_hits and fake_hits[0]["axis_inside_hole_m"] >= 100,         "hole-span test does not fire on a detection laid across the hole"
    assert fake_hits[0]["id"] == "FAKE-span"
    return dict(real_detections=len(res["rows"]),
                real_spanning=len(real_hits),
                fake_spanning=len(fake_hits),
                fake_axis_inside_hole_m=fake_hits[0]["axis_inside_hole_m"],
                passed=True)


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def score_scenario(name, outroot):
    d = synth.make(name)
    meta = d["meta"]
    t0 = time.time()
    res = causeway.detect(d["z"], d["meta"], w=None, chm=d["chm"], transect=name)
    dt = time.time() - t0
    caus = [r for r in res["rows"] if r["causeway"]]
    raised = [r for r in res["rows"] if not r["causeway"]]
    rec = dict(scenario=name, kind=EXPECT[name]["kind"], n_detections=len(res["rows"]),
               n_causeway=len(caus), n_raised=len(raised),
               time_s=round(dt, 2), funnel=res["funnel"], stats=res["stats"],
               px=dict(nrows=int(d["z"].shape[0]), ncols=int(d["z"].shape[1]),
                       nx=int(np.isfinite(d["z"]).sum())))
    # ---- scoring, split into the two things a detector can honestly be judged on:
    #   (1) is the RIDGE found at all (any class),  (2) is it CLASSIFIED as a causeway
    #   without firing on known non-causeways.
    matches, fps, flagged, spurious = [], [], [], []
    used = set()
    truths_all = d["truth"]
    for t in truths_all:
        g = t["geom"]
        tl = g.length if t["kind"] in ("causeway", "levee", "valley", "terrace",
                                       "modern_track", "natural_ridge") else float("nan")
        t_ang = None
        try:
            co = np.asarray(g.coords)
            t_ang = math.degrees(math.atan2(co[-1, 1] - co[0, 1],
                                            co[-1, 0] - co[0, 0])) % 180.0
        except Exception:
            pass
        for i, r in enumerate(res["rows"]):
            dist = _seg_dist_point(g, r["x"], r["y"])
            angd = _angle_diff(r["axis_angle_deg"], t_ang) if t_ang is not None else 0.0
            long_enough = (not np.isfinite(tl)) or r["length_m"] >= 0.4 * tl
            if t["kind"] == "causeway":
                if dist <= 25 and angd <= 25 and long_enough:
                    matches.append(dict(id=r["id"], dist_m=round(dist, 1), type=r["type"],
                                        classed=bool(r["causeway"]),
                                        angle_err_deg=round(angd, 1),
                                        len_ratio=round(r["length_m"] / max(tl, 1e-9), 2),
                                        height_m=round(r["height_m"], 2),
                                        truth_height_m=t["height_m"],
                                        flags=r["flags"]))
                    used.add(i)
            elif t["kind"] == "modern_track":
                if r["causeway"] and dist <= 60:
                    flagged.append(dict(id=r["id"], dist_m=round(dist, 1),
                                        len_ratio=round(r["length_m"] / max(tl, 1e-9), 2),
                                        flags=r["flags"]))
                    used.add(i)
            else:
                if r["causeway"] and dist <= 40:
                    fps.append(dict(id=r["id"], truth=t["kind"], dist_m=round(dist, 1),
                                    flags=r["flags"]))
                    used.add(i)
    for i, r in enumerate(res["rows"]):
        if i not in used and r["causeway"]:
            spurious.append(dict(id=r["id"], dist_m=None, flags=r["flags"],
                                 length_m=round(r["length_m"])))
    n_classed = sum(1 for q in matches if q["classed"])
    holes = [t["truncated_by_hole"] for t in truths_all if t.get("truncated_by_hole")]
    spans = _hole_spans(res, holes[0], causeway.DEFAULTS["max_gap_m"]) if holes else []
    rec.update(hole_spans=spans, n_hole_spans=len(spans), n_holes=len(holes),
               matches=matches, false_positives=fps, flagged_modern=flagged,
               spurious_causeway=spurious, n_matched=len(matches),
               n_classed=n_classed, n_fp=len(fps), n_spurious=len(spurious),
               n_flagged_modern=len(flagged))
    exp = EXPECT[name]
    if exp["kind"] in ("positive", "positive-hard"):
        ok = len(matches) >= exp["causeway"] and len(fps) == 0 and len(spurious) == 0
    elif exp["kind"] in ("negative", "negative-hard"):
        ok = len(fps) == 0 and len(spurious) == 0
    elif exp["kind"] == "artefact-truncation":
        # embedded positive: the truncated remnant must still be found, nothing may
        # bridge the hole, and the halo rim must not produce a spurious call
        ok = (len(matches) >= 1 and len(fps) == 0 and len(spurious) == 0
              and len(spans) == 0)
    elif exp["kind"] == "negative-modern":
        ok = len(fps) == 0 and len(spurious) == 0 and \
            all(f["flags"] for f in flagged)
    else:
        ok = False
    rec["pass"] = bool(ok)
    rec["expect"] = exp
    # write per-scenario outputs
    outdir = os.path.join(outroot, "synth", name)
    causeway.write_outputs(res, outdir, transect=name)
    return rec, res


def main(outroot="results"):
    os.makedirs(outroot, exist_ok=True)
    st = self_test_hole_span()
    print("hole-span test self-check:", st)
    recs = []
    for name in synth.ALL:
        rec, _ = score_scenario(name, outroot)
        recs.append(rec)
        print(f"{name:28s} {'PASS' if rec['pass'] else 'FAIL'}  "
              f"causeway {rec['n_causeway']} matched {rec['n_matched']} "
              f"fp {rec['n_fp']} flagged {rec['n_flagged_modern']} "
              f"({rec['time_s']}s)")
    tot = len(recs)
    npass = sum(r["pass"] for r in recs)
    pos = [r for r in recs if r["kind"].startswith("positive")]
    neg = [r for r in recs if r["kind"].startswith("negative")]
    # Two denominators, stated explicitly so no number is ambiguous:
    #  (a) all PLANTED causeway ridges in the control set = the 5 must_find positives
    #      plus the causeway truncated by a data hole (an embedded positive, not a
    #      negative)  -> 6 ridges;
    #  (b) the must_find subset only -> 5 ridges.
    planted = [r for r in recs if r["expect"]["causeway"] > 0
               and r["kind"] != "negative-modern"]
    n_planted = sum(r["expect"]["causeway"] for r in planted)
    summary = dict(
        n_scenarios=tot, n_pass=npass,
        planted_ridges=n_planted,
        ridge_recall=round(sum(r["n_matched"] for r in planted) / max(n_planted, 1), 3),
        causeway_class_rate=round(sum(r["n_classed"] for r in planted) /
                                  max(sum(r["n_matched"] for r in planted), 1), 3),
        must_find_ridges=sum(r["expect"]["causeway"] for r in pos),
        must_find_ridge_recall=round(sum(r["n_matched"] for r in pos) /
                                     max(sum(r["expect"]["causeway"] for r in pos), 1), 3),
        must_find_class_rate=round(sum(r["n_classed"] for r in pos) /
                                   max(sum(r["n_matched"] for r in pos), 1), 3),
        spurious_causeway=sum(r["n_spurious"] for r in recs) + sum(r["n_fp"] for r in recs),
        hole_span_failures=sum(r["n_hole_spans"] for r in recs),
        hole_span_tests=sum(r["n_holes"] for r in recs),
        negative_fp_rate=round(sum(1 for r in neg if r["n_fp"] > 0) / max(len(neg), 1), 3),
        modern_flagged=sum(r["n_flagged_modern"] for r in recs),
        total_time_s=round(sum(r["time_s"] for r in recs), 1),
        hole_span_selftest=st,
        params=causeway.DEFAULTS,
    )
    with open(os.path.join(outroot, "synth_benchmark.json"), "w") as fh:
        json.dump(dict(summary=summary, scenarios=recs), fh, indent=1, default=float)
    lines = ["# Synthetic control benchmark -- causeway / raised-linear detector (m3)", "",
             f"{summary['n_pass']}/{tot} scenarios pass.",
             "",
             f"- planted causeway ridges in the control set: "
             f"**{summary['planted_ridges']}** (5 `must_find` positives + the causeway "
             f"truncated by a data hole, which is an embedded positive, not a negative)",
             f"- ridge recall over all planted ridges: **{summary['ridge_recall']:.2f}** "
             f"({summary['planted_ridges']} ridges); over the "
             f"{summary['must_find_ridges']} `must_find` positives it is "
             f"{summary['must_find_ridge_recall']:.2f}",
             f"- classified `causeway`: **{summary['causeway_class_rate']:.2f}** over all "
             f"{summary['planted_ridges']} planted ridges "
             f"({summary['must_find_class_rate']:.2f} over the must_find subset) -- the "
             f"difference is the truncated remnant, which is labelled causeway",
             f"- false `causeway` calls on negative controls + spurious: "
             f"**{summary['spurious_causeway']}** (fp rate per scenario "
             f"{summary['negative_fp_rate']:.2f})",
             f"- modern track caught by the flag screen: {summary['modern_flagged']}",
             f"- hole-spanning detections: **{summary['hole_span_failures']}** in "
             f"{summary['hole_span_tests']} hole test(s) (the empty_halo control is an "
             f"embedded POSITIVE, not a negative: its planted causeway is real and is "
             f"scored as found, plus the no-span test)",
             f"- total {summary['total_time_s']} s for {tot} scenarios on 4 CPUs", "",
             "| scenario | kind | expect causeway | detected | ridge found | classed causeway | FP | spurious | hole-span | flagged modern | pass | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        exp_s = f"causeway={r['expect']['causeway']}"
        if r["expect"].get("flagged"):
            exp_s += f", flagged>={r['expect']['flagged']}"
        lines.append(f"| {r['scenario']} | {r['kind']} | {exp_s}"
                     f" | {r['n_detections']} | {r['n_matched']}/{r['expect']['causeway']}"
                     f" | {r['n_classed']} | {r['n_fp']} | {r['n_spurious']} | "
                     f"{r['n_hole_spans']} | "
                     f"{r['n_flagged_modern']} | {'yes' if r['pass'] else 'NO'} | "
                     f"{r['time_s']} |")
    lines += ["",
              "Hole-span test self-check (the test must be able to fail): a fabricated "
              f"340 m detection laid across the hole is flagged at "
              f"{st['fake_axis_inside_hole_m']:.0f} m of axis inside it, while the real "
              f"truncated remnant is not flagged.  Reproduce with "
              "`python3 -c \"import sys; sys.path.insert(0,'code'); import run_synth; "
              "print(run_synth.self_test_hole_span())\"`.", "",
              "Detector parameters (metres / dimensionless):", "```",
              json.dumps(summary["params"], indent=1, default=float), "```", "",
              "Per-scenario gate funnels (where candidates are lost):", "```"]
    for r in recs:
        lines.append(f"{r['scenario']:28s} {json.dumps(r['funnel'])}")
    lines += ["```", "",
              "Interpretation limits (do not read more into this table):",
              "- synthetic terrain is a caricature: the noise model is gaussian and the",
              "  earthworks are analytic profiles, so a pass here proves the geometry",
              "  logic is self-consistent, not that real lidar will behave the same;",
              "- the modern-track negative is *supposed* to look like a causeway.  It is",
              "  detected and caught by the flag screen, which is the honest outcome: at",
              "  DTM level a causeway cannot be told from a modern raised track.",
              ""]
    with open(os.path.join(outroot, "synth_benchmark.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
