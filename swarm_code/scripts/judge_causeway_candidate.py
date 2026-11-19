#!/usr/bin/env python3
"""Apply m6's independent line falsifier to one raised-linear detection.

  .venv/bin/python code/judge_causeway_candidate.py data/derived_v2/0516 0516-CW001

Diagnostic only: it never assigns Known or Strong and never publishes coordinates.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/vendor_skeptic_line"))
from skeptic.grid import Grid
from skeptic.line import line_bundle
from skeptic.rules import judge_line_bundle
from run_skeptic import json_safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("candidate_id")
    args = p.parse_args()
    summary = json.loads((args.derived / "summary.json").read_text())
    meta = json.loads((args.derived / "grid.json").read_text())
    products = np.load(args.derived / "products.npz")
    grid = Grid(products["dtm"], meta["cell"], meta["x0"], meta["y0"],
                y_ascending=True, epsg=meta["epsg"], name=args.derived.name)
    chm = Grid(products["chm"], meta["cell"], meta["x0"], meta["y0"],
               y_ascending=True, epsg=meta["epsg"], name=args.derived.name + "_chm")
    path = args.derived / f"causeway/{args.derived.name}/detections_{args.derived.name}.csv"
    with path.open() as handle:
        row = next(r for r in csv.DictReader(handle) if r["id"] == args.candidate_id)
    x, y = float(row["x"]), float(row["y"])
    half = 0.5 * float(row["length_m"])
    angle = math.radians(float(row["axis_angle_deg"]))
    dx, dy = half * math.cos(angle), half * math.sin(angle)
    bundle = line_bundle(grid, x-dx, y-dy, x+dx, y+dy,
                         width_hint=float(row["crest_width_m"]), chm=chm,
                         fid=args.candidate_id, transect=args.derived.name, kind="raised")
    bundle, verdict = judge_line_bundle(bundle)
    target = args.derived / "causeway" / args.derived.name / "evidence" / args.candidate_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "skeptic_line.json").write_text(json.dumps(json_safe(bundle), indent=2, allow_nan=False) + "\n")
    (target / "skeptic_line_verdict.json").write_text(json.dumps(json_safe(verdict), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"id": args.candidate_id, "certainty": verdict.get("certainty"),
                      "label": verdict.get("primary_label"), "reason": verdict.get("reason"),
                      "blocking": verdict.get("blocking")}, indent=2))


if __name__ == "__main__":
    main()
