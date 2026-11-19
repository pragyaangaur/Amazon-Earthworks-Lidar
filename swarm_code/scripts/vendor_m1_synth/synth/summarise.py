"""Aggregate the m1 benchmark: verdict table per scene x method, plus the
noise-floor calibration, as CSV and Markdown.

Usage: python synth/summarise.py --out out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

LAYERS = ("dem", "relief15")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.out, "*", "*_metrics.json")))
    rows, noise = [], []
    for fn in files:
        r = json.load(open(fn))
        scene, method = r["scene"], r.get("method", "?")
        for rec in r["features"]:
            for layer in LAYERS:
                m = rec.get(layer)
                if not isinstance(m, dict):
                    continue
                rows.append(dict(
                    scene=scene, method=method, feature=rec["id"],
                    type=rec["type"], expect=rec.get("expect"), layer=layer,
                    dip_m=m.get("dip_m"), az_frac=m.get("az_frac_dip"),
                    trough_m=m.get("trough_m"),
                    line_frac=m.get("frac_length_dip"),
                    valid_frac=m.get("valid_frac"),
                    n_mounds=m.get("n_detected"), k=m.get("k"),
                    verdict=m.get("verdict")))
        nf = r.get("noise_floor")
        if nf:
            for rad, v in nf["per_radius"].items():
                noise.append(dict(scene=scene, method=method, layer=nf.get("layer"),
                                  radius_m=float(rad), n=v["n"], p50=v["p50"],
                                  p99=v["p99"], p999=v["p999"], max=v["max"],
                                  az25_p99=v.get("az25_p99"),
                                  az25_max=v.get("az25_max"),
                                  trough_p99=v.get("trough_p99"),
                                  trough_max=v.get("trough_max")))
    import csv
    with open(os.path.join(a.out, "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["scene", "method", "feature", "verdict"])
        w.writeheader()
        w.writerows(rows)
    if noise:
        with open(os.path.join(a.out, "benchmark_noisefloor.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(noise[0].keys()))
            w.writeheader()
            w.writerows(noise)

    scenes = sorted({r["scene"] for r in rows})
    methods = ["ceiling", "vstruct", "median", "literal", "none"]
    L = ["# Synthetic benchmark summary (m1)", "",
         "Verdict words: **clear** / partial / absent / masked -- a benchmark",
         "instrument, not a certainty grade. `dem` = 3 m DTM, `relief15` = the",
         "15 m Gaussian local relief model (what a detector actually sees).", ""]
    for sc in scenes:
        L.append("## %s" % sc)
        feats = []
        for r in rows:
            if r["scene"] == sc and r["feature"] not in [f[0] for f in feats]:
                feats.append((r["feature"], r["type"], r["expect"]))
        L.append("")
        short = {"vstruct": "vs", "median": "med", "literal": "lit", "none": "raw",
                 "ceiling": "ceil"}
        hdr = "| feature | type | expect | " + " | ".join(
            "%s dem" % short.get(m, m) for m in methods) + " | " + " | ".join(
            "%s r15" % short.get(m, m) for m in methods) + " |"
        L.append(hdr)
        L.append("|" + "---|" * (len(hdr.split("|")) - 2))
        for fid, ftype, fexp in feats:
            cells = []
            for m in methods:
                r = next((x for x in rows if x["scene"] == sc and x["method"] == m
                          and x["feature"] == fid and x["layer"] == "dem"), None)
                cells.append(_cell(r))
            for m in methods:
                r = next((x for x in rows if x["scene"] == sc and x["method"] == m
                          and x["feature"] == fid and x["layer"] == "relief15"), None)
                cells.append(_cell(r))
            L.append("| %s | %s | %s | %s |" % (fid, ftype, fexp, " | ".join(cells)))
        L.append("")
    if noise:
        L += ["## Ring-statistic noise floor (natural-negatives / no-target grid)",
              "", "`dip_m` p99 and max over the scan grid; a detector threshold",
              "has to clear the max of the scene it runs in.", "",
              "| scene | method | radius (m) | dip p99 | dip max | az25 p99 | az25 max | trough p99 | trough max |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n in noise:
            if n["radius_m"] in (20.0, 30.0, 45.0, 60.0, 120.0, 200.0):
                L.append("| %s | %s | %.0f | %.2f | %.2f | %s | %s | %s | %s |" % (
                    n["scene"], n["method"], n["radius_m"], n["p99"], n["max"],
                    "%.2f" % n["az25_p99"] if n.get("az25_p99") is not None else "-",
                    "%.2f" % n["az25_max"] if n.get("az25_max") is not None else "-",
                    "%.2f" % n["trough_p99"] if n.get("trough_p99") is not None else "-",
                    "%.2f" % n["trough_max"] if n.get("trough_max") is not None else "-"))
        L.append("")
    md = os.path.join(a.out, "BENCHMARK.md")
    with open(md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote %s (%d rows), %s" % (
        os.path.join(a.out, "benchmark_summary.csv"), len(rows), md))
    return 0


def _cell(r):
    if r is None:
        return "-"
    if r.get("n_mounds") is not None and r.get("n_mounds") != "":
        return "%s %s/%s" % (r["verdict"], r["n_mounds"], r["k"])
    if r.get("dip_m") is None:
        return "masked(%.2f)" % (r.get("valid_frac") or 0)
    if r.get("trough_m") not in (None, ""):
        return "%s %.2f/tr %.2f" % (r["verdict"], r["dip_m"], float(r["trough_m"]))
    return "%s %.2f" % (r["verdict"], r["dip_m"])


if __name__ == "__main__":
    sys.exit(main())
