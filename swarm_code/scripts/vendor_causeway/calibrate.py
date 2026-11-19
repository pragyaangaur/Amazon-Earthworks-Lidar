"""calibrate.py -- collect every assessed candidate on every synthetic control,
label it against ground truth, and dump the metric table used to choose the
causeway decision rule.  Writes results/calibration_candidates.csv and
results/calibration.md (rule search results).

A candidate is labelled TRUE only if it lies within 25 m of a truth line that is
declared must_find and its axis agrees within 25 deg.  Everything else -- including
detections of non-causeway truth objects and of nothing at all -- is FALSE.  That
is a deliberately strict labelling: a detection of a real *something* that is not a
causeway counts against the rule.
"""
import csv
import json
import math
import os
import sys

import numpy as np
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causeway
import synth

METRICS = ["length_m", "height_m", "crest_width_m", "width_cv", "asym_w", "continuity",
           "h_iqr", "both_sides", "left_min_m", "right_min_m", "both_frac", "one_frac",
           "crest_left_m", "crest_right_m", "touch_frac", "cross_angle_deg", "score"]


def collect():
    rows = []
    for name in synth.ALL:
        d = synth.make(name)
        res = causeway.detect(d["z"], d["meta"], chm=d["chm"], transect=name)
        # Label against ANY planted causeway (including the meander-apex hard case and
        # the causeway truncated by a data hole): those ridges are real.  The modern
        # track is excluded from calibration because at DTM level it IS a causeway --
        # its rejection is a context argument, not a geometric one.
        if name == "modern_track":
            continue
        truths = [t for t in d["truth"] if t["kind"] == "causeway"]
        for r in res["rows"]:
            lab, best = False, 1e9
            for t in truths:
                g = t["geom"]
                dist = g.distance(Point(r["x"], r["y"]))
                co = np.asarray(g.coords)
                tang = math.degrees(math.atan2(co[-1, 1] - co[0, 1],
                                               co[-1, 0] - co[0, 0])) % 180.0
                da = abs(r["axis_angle_deg"] - tang) % 180.0
                da = min(da, 180.0 - da)
                if dist < best:
                    best = dist
                if dist <= 25.0 and da <= 25.0:
                    lab = True
            rows.append(dict(scenario=name, id=r["id"], label=bool(lab),
                             x=round(r["x"], 1), y=round(r["y"], 1),
                             truth_dist=round(best, 1),
                             type=r["type"], flags=";".join(r["flags"]),
                             axis_angle_deg=round(r["axis_angle_deg"], 1),
                             **{k: (None if r.get(k) is None else r.get(k)) for k in METRICS}))
    return rows


def rule_eval(rows, fn):
    tp = sum(1 for r in rows if r["label"] and fn(r))
    fp = sum(1 for r in rows if not r["label"] and fn(r))
    fn_ = sum(1 for r in rows if r["label"] and not fn(r))
    return dict(tp=tp, fp=fp, missed=fn_,
                precision=round(tp / max(tp + fp, 1), 3),
                recall=round(tp / max(tp + fn_, 1), 3))


def main(outroot="results"):
    os.makedirs(outroot, exist_ok=True)
    rows = collect()
    keys = ["scenario", "id", "label", "truth_dist", "x", "y", "type",
            "axis_angle_deg"] + METRICS + ["flags"]
    with open(os.path.join(outroot, "calibration_candidates.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} assessed candidates, {sum(r['label'] for r in rows)} labelled TRUE")
    # candidate decision rules, from strict to loose
    rules = {}
    rules["A both_sides & L/R<=15 & crest>=40"] = lambda r: (
        r["both_sides"] and (r["left_min_m"] or 99) <= 15 and (r["right_min_m"] or 99) <= 15
        and (r["crest_left_m"] or 0) >= 40 and (r["crest_right_m"] or 0) >= 40)
    rules["B both_sides & L/R<=20 & crest>=30"] = lambda r: (
        r["both_sides"] and (r["left_min_m"] or 99) <= 20 and (r["right_min_m"] or 99) <= 20
        and (r["crest_left_m"] or 0) >= 30 and (r["crest_right_m"] or 0) >= 30)
    rules["C both_sides only"] = lambda r: bool(r["both_sides"])
    rules["D both_sides & L/R<=20"] = lambda r: (
        r["both_sides"] and (r["left_min_m"] or 99) <= 20 and (r["right_min_m"] or 99) <= 20)
    rules["E B & one_frac<=0.6"] = lambda r: (
        rules["B both_sides & L/R<=20 & crest>=30"](r) and (r["one_frac"] or 0) <= 0.6)
    rules["F B & L>=120"] = lambda r: (
        rules["B both_sides & L/R<=20 & crest>=30"](r) and (r["length_m"] or 0) >= 120)
    rules["G B & w50<=12 & h>=0.5"] = lambda r: (
        rules["B both_sides & L/R<=20 & crest>=30"](r) and (r["crest_width_m"] or 99) <= 12
        and (r["height_m"] or 0) >= 0.5)
    rules["I = exact code rule (both_sides & L/R<=25 & crest>=20 & w50<=12 & asymW<=0.35)"] = lambda r: (
        bool(r["both_sides"]) and (r["left_min_m"] or 99) <= 25
        and (r["right_min_m"] or 99) <= 25 and (r["crest_left_m"] or 0) >= 20
        and (r["crest_right_m"] or 0) >= 20
        and (r["crest_width_m"] or 99) <= 12 and (r["asym_w"] or 99) <= 0.35)
    rules["H = SHIPPED RULE (B & w50<=12 & asymW<=0.35)"] = lambda r: (
        rules["B both_sides & L/R<=20 & crest>=30"](r) and (r["crest_width_m"] or 99) <= 12
        and (r["asym_w"] or 99) <= 0.35)
    res = {k: rule_eval(rows, f) for k, f in rules.items()}
    for k, v in res.items():
        print(f"{k:38s} TP {v['tp']:2d} FP {v['fp']:2d} missed {v['missed']:2d} "
              f"P {v['precision']:.2f} R {v['recall']:.2f}")
    with open(os.path.join(outroot, "calibration.json"), "w") as fh:
        json.dump(dict(n=len(rows), rules=rules and res,
                       candidates=rows), fh, indent=1, default=float)
    lines = ["# Causeway decision-rule calibration (m3)", "",
             f"{len(rows)} candidates assessed on {len(synth.ALL)} synthetic controls, "
             f"{sum(r['label'] for r in rows)} of them true causeways.", "",
             "| rule | TP | FP | missed | precision | recall |", "|---|---|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v['tp']} | {v['fp']} | {v['missed']} | "
                     f"{v['precision']:.2f} | {v['recall']:.2f} |")
    lines += ["", "The rule is chosen for precision first: a causeway candidate is a lead "
              "for a human, and a false lead costs more than a missed one at this stage.",
              "", "All labelled candidates:", "",
              "| scenario | id | label | d_truth | L | h | w50 | asymW | both | L/R | crest L/R | one |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda q: (q["scenario"], not q["label"])):
        lines.append(f"| {r['scenario']} | {r['id']} | {'TRUE' if r['label'] else '.'} | "
                     f"{r['truth_dist']} | {r['length_m']:.0f} | {r['height_m']:.2f} | "
                     f"{r['crest_width_m']} | {r['asym_w']:.2f} | {r['both_sides']} | "
                     f"{r['left_min_m']}/{r['right_min_m']} | "
                     f"{r['crest_left_m']:.0f}/{r['crest_right_m']:.0f} | {r['one_frac']:.2f} |")
    with open(os.path.join(outroot, "calibration.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return rows, res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
