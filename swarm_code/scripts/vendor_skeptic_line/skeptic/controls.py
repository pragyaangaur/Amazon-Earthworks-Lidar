"""Control runner: score the skeptic engine against every synthetic scene.

Two modes per feature, because a detector's report can be wrong in two ways:

  auto    the box only knows a position (worst case: no usable size)
  hinted  the box knows position and an approximate outer size (the normal case)

A control PASSES when
  * an expected-positive feature comes back Possible (never Rejected), and
  * an expected-negative feature comes back Rejected for a reason in the family
    listed in the scene truth (a rejection for a different reason is reported as
    `wrong_reason` rather than silently counted as success).

A false positive (negative -> Possible) is the failure that matters most, so it
is counted separately in the summary.

    python3 -m skeptic.controls --out out/controls
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from . import analysis, line, rules, synthetic


def _ctl(g, cx, cy, size_hint, chm, fid):
    b = analysis.sceptic_bundle(g, cx, cy, size_hint=size_hint, kind="ring",
                                chm=chm, fid=fid, transect="synthetic")
    b, v = rules.judge_bundle(b)
    return b, v


def _verdict_row(scene, t, mode, b, v):
    g = b["geometry"]
    ok_geom = g.get("status") == "ok"
    row = dict(scene=scene, id=t["id"], expect=t["expect"], mode=mode,
               note=t.get("note", ""), certainty=v["certainty"],
               label=v["primary_label"], grade=v["geometric_grade"],
               reason=v["reason"][:600],
               radius_m=(g.get("radius_m") if ok_geom else None),
               diameter_m=(g.get("diameter_m") if ok_geom else None),
               centre=(dict(x=g.get("cx"), y=g.get("cy")) if ok_geom else None),
               quality=(g.get("quality") if ok_geom else None),
               dip_amp=(g.get("dip_amp_m") if ok_geom else None),
               dip_width=(g.get("dip_width_m") if ok_geom else None),
               coverage=(g.get("edge", {}).get("coverage") if ok_geom else None),
               planform_resid=(g.get("planform") or {}).get("resid_rel") if ok_geom else None,
               shape=(g.get("planform") or {}).get("shape") if ok_geom else None,
               two_sided=(g.get("cross", {}).get("two_sided_frac") if ok_geom else None),
               step_frac=(g.get("cross", {}).get("step_frac") if ok_geom else None),
               interior_p75=(g.get("level", {}).get("interior_p75_m") if ok_geom else None),
               hole_gap=(g.get("support", {}).get("ring_gap_to_hole_m") if ok_geom else None),
               canopy=(g.get("canopy", {}).get("canopy_median_m") if ok_geom else None),
               checks=[(c["id"], c["status"]) for c in v["checks"]])
    exp_rules = t.get("expect_rule")
    exp_rules = (tuple(exp_rules) if isinstance(exp_rules, (list, tuple))
                 else ((exp_rules,) if exp_rules else ()))
    if t["expect"] == "possible":
        row["outcome"] = "ok" if v["certainty"] == "Possible" else "FALSE_NEGATIVE"
    elif t.get("known_limitation") and v["certainty"] == "Possible":
        row["outcome"] = "documented_limitation"
    else:
        if v["certainty"] != "Rejected":
            row["outcome"] = "FALSE_POSITIVE"
        elif exp_rules and v["primary_label"] not in exp_rules:
            row["outcome"] = "wrong_reason"
        else:
            row["outcome"] = "ok"
    return row


def _jitter(scene, t, grid, chm, offsets=(0.0, 2.0, 5.0, 10.0, 15.0),
            hint=None):
    """How far can the reported centre be off before the verdict changes?"""
    out = []
    for off in offsets:
        if off == 0.0:
            dirs = [(0.0, 0.0)]
        else:
            dirs = [(off * math.cos(math.pi * k / 4), off * math.sin(math.pi * k / 4))
                    for k in range(8)]
        n_pos = n_tot = 0
        errs = []
        for dx, dy in dirs:
            b, v = _ctl(grid, t["cx"] + dx, t["cy"] + dy, hint, chm,
                        f"{t['id']}_j{off:.0f}")
            n_tot += 1
            if v["certainty"] == "Possible":
                n_pos += 1
            g = b["geometry"]
            if g.get("status") == "ok" and g.get("cx") is not None:
                errs.append(math.hypot(g["cx"] - t["cx"], g["cy"] - t["cy"]))
        out.append(dict(offset_m=off, n=n_tot, kept_possible=n_pos,
                        hint_m=hint,
                        frac_kept=(n_pos / n_tot if n_tot else None),
                        centre_err_median=(float(np.median(errs)) if errs else None)))
    return out


def run(outdir="out/controls", jitter=True, neg_hint=60.0, quiet=False):
    os.makedirs(os.path.join(outdir, "bundles"), exist_ok=True)
    rows = []
    jit = []
    t0 = time.time()
    for sc in synthetic.all_scenes():
        g, chm = sc["grid"], sc["chm"]
        est_outer = None
        for t in sc["truth"]:
            b, v = _ctl(g, t["cx"], t["cy"], None, chm, t["id"])
            _dump(os.path.join(outdir, "bundles"), sc["name"], t["id"], "auto", b)
            rows.append(_verdict_row(sc["name"], t, "auto", b, v))
            if t["expect"] == "possible":
                geom = b["geometry"]
                hint = (geom.get("outer_diameter_m") if geom.get("status") == "ok"
                        else None) or est_outer or 60.0
                b2, v2 = _ctl(g, t["cx"], t["cy"], float(hint), chm,
                              f"{t['id']}_hint")
            else:
                b2, v2 = _ctl(g, t["cx"], t["cy"], float(neg_hint), chm,
                              f"{t['id']}_hint")
            _dump(os.path.join(outdir, "bundles"), sc["name"], t["id"], "hint", b2)
            rows.append(_verdict_row(sc["name"], t, "hint", b2, v2))
            if jitter and t["expect"] == "possible":
                jit.extend([dict(scene=sc["name"], id=t["id"], **j)
                            for j in _jitter(sc, t, g, chm)])
                hint = b2["size_hint_m"]
                jit.extend([dict(scene=sc["name"], id=t["id"], **j)
                            for j in _jitter(sc, t, g, chm, offsets=(5.0, 10.0, 15.0),
                                             hint=hint)])
            if not quiet:
                r_auto, r_hint = rows[-2], rows[-1]
                print(f"{sc['name']:22s} {t['id']:4s} {t['expect']:8s} "
                      f"auto={r_auto['certainty']:8s}/{r_auto['label']} "
                      f"hint={r_hint['certainty']:8s}/{r_hint['label']} "
                      f"({r_auto['outcome']},{r_hint['outcome']})", flush=True)
    summary = _summarise(rows, jit, time.time() - t0)
    with open(os.path.join(outdir, "control_report.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows, jitter=jit), f, indent=1)
    with open(os.path.join(outdir, "control_report.md"), "w") as f:
        f.write(_markdown(summary, rows, jit))
    if not quiet:
        print(json.dumps(summary, indent=1))
    return dict(summary=summary, rows=rows, jitter=jit)


def run_line(outdir="out/controls_line", quiet=False):
    """Control suite for the linear (causeway / road / levee / scarp) rules."""
    import time as _t
    os.makedirs(os.path.join(outdir, "bundles"), exist_ok=True)
    rows = []
    t0 = _t.time()
    for sc in synthetic.all_line_scenes():
        for t in sc["truth"]:
            b = line.line_bundle(sc["grid"], t["x1"], t["y1"], t["x2"], t["y2"],
                                 width_hint=7.0, chm=sc["chm"], fid=t["id"],
                                 transect="synthetic", kind=t["kind"])
            b, v = rules.judge_line_bundle(b)
            with open(os.path.join(outdir, "bundles", f"{sc['name']}__{t['id']}.json"),
                      "w") as f:
                json.dump(b, f, indent=1)
            g = b["geometry"]
            exp = (tuple(t["expect_rule"]) if isinstance(t["expect_rule"], (list, tuple))
                   else ((t["expect_rule"],) if t["expect_rule"] else ()))
            if t["expect"] == "possible":
                outcome = "ok" if v["certainty"] == "Possible" else "FALSE_NEGATIVE"
            elif v["certainty"] != "Rejected":
                outcome = "FALSE_POSITIVE"
            elif exp and v["primary_label"] not in exp:
                outcome = "wrong_reason"
            else:
                outcome = "ok"
            rows.append(dict(scene=sc["name"], id=t["id"], expect=t["expect"],
                             certainty=v["certainty"], label=v["primary_label"],
                             grade=v["geometric_grade"], outcome=outcome,
                             amp_m=g.get("amp_m"), extreme_m=g.get("relief_extreme_m"),
                             width_m=g.get("width_m"), continuity=g.get("continuity"),
                             asym=g.get("asym"), width_cv=g.get("width_cv"),
                             banks=g.get("banks"), flat=g.get("flat_bottom_frac"),
                             canopy_ratio=(g.get("canopy") or {}).get("ratio"),
                             reason=v["reason"][:500]))
            if not quiet:
                print(f"{sc['name']:18s} {t['id']} {t['expect']:8s} -> "
                      f"{v['certainty']:8s} {str(v['primary_label']):16s} {outcome}",
                      flush=True)
    summary = dict(n=len(rows),
                   possible=sum(1 for r in rows if r["certainty"] == "Possible"),
                   rejected=sum(1 for r in rows if r["certainty"] == "Rejected"),
                   false_positives=sum(1 for r in rows if r["outcome"] == "FALSE_POSITIVE"),
                   false_negatives=sum(1 for r in rows if r["outcome"] == "FALSE_NEGATIVE"),
                   wrong_reason=sum(1 for r in rows if r["outcome"] == "wrong_reason"),
                   elapsed_s=round(_t.time() - t0, 2))
    with open(os.path.join(outdir, "line_control_report.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)
    md = ["# m6 line / causeway falsification: synthetic control report", "",
          f"n={summary['n']}, Possible={summary['possible']}, "
          f"Rejected={summary['rejected']}, false positives="
          f"{summary['false_positives']}, false negatives="
          f"{summary['false_negatives']}, wrong-reason={summary['wrong_reason']}", "",
          "| scene | id | expect | verdict | reason family | amplitude (m) | "
          "extreme (m) | width (m) | continuity | asym | width CV | banks | outcome |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def f(v, nd=2):
            return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        md.append("| " + " | ".join([r["scene"], r["id"], r["expect"], r["certainty"],
                                     str(r["label"]), f(r["amp_m"]), f(r["extreme_m"]),
                                     f(r["width_m"], 1), f(r["continuity"]),
                                     f(r["asym"]), f(r["width_cv"]), str(r["banks"]),
                                     r["outcome"]]) + " |")
    with open(os.path.join(outdir, "line_control_report.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    if not quiet:
        print(json.dumps(summary, indent=1))
    return dict(summary=summary, rows=rows)


def _dump(d, scene, fid, mode, b):
    p = os.path.join(d, f"{scene}__{fid}__{mode}.json")
    with open(p, "w") as f:
        json.dump(b, f, indent=1)


def _summarise(rows, jit, elapsed):
    def count(pred):
        return sum(1 for r in rows if pred(r))
    pos = [r for r in rows if r["expect"] == "possible"]
    neg = [r for r in rows if r["expect"] == "reject"]
    fp = [r for r in neg if r["outcome"] == "FALSE_POSITIVE"]
    fn = [r for r in pos if r["outcome"] == "FALSE_NEGATIVE"]
    wrong = [r for r in neg if r["outcome"] == "wrong_reason"]
    return dict(
        n_rows=len(rows), n_positive=len(pos), n_negative=len(neg),
        documented_limitation=count(lambda r: r["outcome"] == "documented_limitation"),
        positives_kept_possible=count(lambda r: r["expect"] == "possible"
                                      and r["certainty"] == "Possible"),
        false_positives=len(fp),
        false_negative_ids=[f"{r['scene']}/{r['id']}/{r['mode']}" for r in fn],
        wrong_reason_ids=[f"{r['scene']}/{r['id']}/{r['mode']}" for r in wrong],
        false_positive_ids=[f"{r['scene']}/{r['id']}/{r['mode']}" for r in fp],
        grades_A=[f"{r['scene']}/{r['id']}/{r['mode']}" for r in pos
                  if r["grade"] == "A"],
        jitter=[dict(scene=j["scene"], id=j["id"], offset_m=j["offset_m"],
                     hint_m=j.get("hint_m"),
                     frac_kept=j["frac_kept"],
                     centre_err_median=j["centre_err_median"]) for j in jit],
        elapsed_s=round(elapsed, 1))


def _markdown(summary, rows, jit):
    L = []
    L.append("# m6 skeptic engine: synthetic control report\n")
    L.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
             f"in {summary['elapsed_s']} s.\n")
    L.append("## Outcome\n")
    L.append(f"* features tested: {summary['n_rows']} rows "
             f"({summary['n_positive']} positive rows, {summary['n_negative']} negative rows), "
             f"each in auto and hinted mode")
    L.append(f"* expected-positive rows kept at Possible: "
             f"{summary['positives_kept_possible']}/{summary['n_positive']}")
    L.append(f"* **false positives (negative -> Possible): {summary['false_positives']}**")
    L.append(f"* scenes flagged as a documented limitation that the engine does not "
             f"reject: {summary.get('documented_limitation', 0)}")
    L.append(f"* false negatives: {len(summary['false_negative_ids'])}")
    L.append(f"* rejected for a reason outside the expected family: "
             f"{len(summary['wrong_reason_ids'])}")
    L.append("")
    L.append("## Per-feature results\n")
    L.append("| scene | id | expect | mode | verdict | reason label | grade | R (m) | "
             "dip (m) | width (m) | closure | planform resid | two-sided | outcome |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def f(v, nd=2):
            return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
        L.append("| " + " | ".join([
            r["scene"], r["id"], r["expect"], r["mode"], r["certainty"],
            str(r["label"]), r["grade"], f(r["radius_m"], 1), f(r["dip_amp"]),
            f(r["dip_width"], 1), f(r["coverage"], 2), f(r["planform_resid"], 3),
            f(r["two_sided"], 2), r["outcome"]]) + " |")
    L.append("")
    if jit:
        L.append("## Centre-jitter tolerance (expected-positive features)\n")
        L.append("Detector centres are never exact. Each feature is re-run with the "
                 "reported centre displaced by 2, 5, 10 and 15 m in 8 directions; "
                 "`the robot` keeps the verdict at Possible and recovers the true "
                 "centre by its own search.\n")
        L.append("| scene | id | mode | offset (m) | kept Possible | recovered centre error (m) |")
        L.append("|---|---|---|---|---|---|")
        for j in jit:
            L.append(f"| {j['scene']} | {j['id']} | {'hinted' if j.get('hint_m') else 'auto'} | {j['offset_m']:.0f} | "
                     f"{j['kept_possible']}/{j['n']} | "
                     f"{'' if j['centre_err_median'] is None else round(j['centre_err_median'],1)} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/controls")
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--neg-hint", type=float, default=60.0)
    a = ap.parse_args()
    run(a.out, jitter=not a.no_jitter, neg_hint=a.neg_hint)


if __name__ == "__main__":
    main()
