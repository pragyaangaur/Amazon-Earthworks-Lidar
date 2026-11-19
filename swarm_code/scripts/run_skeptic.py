#!/usr/bin/env python3
"""Apply the frozen offline skeptic bundle to detector outputs.

Example:
  .venv/bin/python code/run_skeptic.py data/derived/0072 rings_snapshot_v1

This does not perform literature checks or assign final certainty. Automatic
results are diagnostic only; coordinator visual/literature review is separate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/vendor_skeptic"))
from skeptic.grid import Grid
from skeptic.analysis import sceptic_bundle
from skeptic.rules import judge_bundle


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("derived", type=Path)
    parser.add_argument("stem")
    parser.add_argument("--state", default="")
    args = parser.parse_args()
    started = time.monotonic()
    source = json.loads((args.derived / "detections" / f"{args.stem}.json").read_text())
    grid_meta = json.loads((args.derived / "grid.json").read_text())
    products = np.load(args.derived / "products.npz")
    grid = Grid(products["dtm"], grid_meta["cell"], grid_meta["x0"], grid_meta["y0"],
                y_ascending=True, epsg=grid_meta["epsg"], name=source["transect"])
    chm = Grid(products["chm"], grid_meta["cell"], grid_meta["x0"], grid_meta["y0"],
               y_ascending=True, epsg=grid_meta["epsg"], name=source["transect"] + "_chm")
    bundles = []
    failures = []
    for det in source["detections"]:
        started_feature = time.monotonic()
        try:
            bundle = sceptic_bundle(
                grid, det["x"], det["y"], size_hint=2 * det.get("radius_m", 50),
                kind="ring", chm=chm, fid=det["id"], transect=source["transect"],
                state=args.state, detector="frozen_ring_snapshot", refine=False)
            bundle, verdict = judge_bundle(bundle)
            bundle["coordinator_status"] = "automatic_diagnostic_requires_visual_and_record_review"
            bundle["seconds"] = time.monotonic() - started_feature
        except Exception as exc:
            failures.append({"id": det["id"], "error": repr(exc)})
            bundle = {"id": det["id"], "transect": source["transect"],
                      "coordinator_status": "skeptic_error_not_reviewed",
                      "seconds": time.monotonic() - started_feature,
                      "error": repr(exc), "verdict": {"certainty": None}}
        bundles.append(bundle)
        print(f"skeptic {det['id']}: {bundle.get('verdict', {}).get('certainty')} "
              f"({bundle['seconds']:.2f}s)", flush=True)
    output = args.derived / "skeptic"
    output.mkdir(exist_ok=True)
    result = {"transect": source["transect"], "source": f"detections/{args.stem}.json",
              "status": "automatic_diagnostics_not_final_certainty",
              "elapsed_seconds": time.monotonic() - started,
              "verdict_counts": dict(collections.Counter(
                  str(bundle.get("verdict", {}).get("certainty")) for bundle in bundles)),
              "failures": failures, "features": bundles}
    result = json_safe(result)
    bundles = json_safe(bundles)
    temporary = output / f"{args.stem}.json.part"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / f"{args.stem}.json")
    rows = []
    for bundle in bundles:
        verdict = bundle.get("verdict", {})
        geometry = bundle.get("geometry", {})
        rows.append({"id": bundle["id"], "transect": source["transect"],
                     "automatic_certainty": verdict.get("certainty"),
                     "geometric_grade": verdict.get("geometric_grade"),
                     "strong_eligible_before_record_literature": verdict.get("strong_eligible"),
                     "primary_label": verdict.get("primary_label"),
                     "geometry_status": geometry.get("status"),
                     "reason": verdict.get("reason", bundle.get("error", "")),
                     "coordinator_status": bundle["coordinator_status"]})
    with (output / f"{args.stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows
                                else ["id", "transect", "automatic_certainty", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    for bundle in bundles:
        target = args.derived / "review" / bundle["id"]
        if target.exists():
            (target / "skeptic_auto.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(f"SKEPTIC COMPLETE {source['transect']}: {result['verdict_counts']}; "
          f"errors={len(failures)}; seconds={result['elapsed_seconds']:.2f}")


if __name__ == "__main__":
    main()
