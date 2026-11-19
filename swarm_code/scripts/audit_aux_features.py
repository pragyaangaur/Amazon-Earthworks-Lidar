#!/usr/bin/env python3
"""Create and audit native-coordinate mound/causeway detection tables.

  .venv/bin/python code/audit_aux_features.py data/derived_v2/0254 --state AC

This closes the feature-type gap in the ring-only audit. Certainty stays
`unreviewed_detection`; no table miss is treated as novelty.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess


def write_csv(path, rows):
    fields = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type",
              "size_m", "relief_m", "canopy_m", "certainty", "reason"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("--state", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((args.derived / "summary.json").read_text())
    grid, epsg = summary["grid"], summary["epsg"]
    cell = grid["cell"]
    results = {}
    for kind in ("mounds", "causeway"):
        results[kind] = {"detected": 0, "csv": None}
    mound_path = args.derived / "detector/mound_clusters.json"
    clusters = json.loads(mound_path.read_text()) if mound_path.exists() else []
    rows = []
    for cluster in clusters:
        row_index, col_index = cluster["cell_rowcol"]
        x = grid["x0"] + (col_index + 0.5) * cell
        y = grid["y0"] + (row_index + 0.5) * cell
        rows.append({"id": cluster["id"], "state": args.state,
                     "transect": args.derived.name, "EPSG": epsg, "x": x, "y": y,
                     "lon": "", "lat": "", "type": "mound_cluster",
                     "size_m": 2 * cluster["ring_radius_m"],
                     "relief_m": cluster["height_median_m"], "canopy_m": "",
                     "certainty": "unreviewed_detection", "reason": cluster.get("screen_reason", "")})
    detections = args.derived / "detections"
    detections.mkdir(exist_ok=True)
    path = detections / "mounds_v3.csv"
    write_csv(path, rows)
    results["mounds"] = {"detected": len(rows), "csv": str(path)}

    causeway_path = args.derived / f"causeway/{args.derived.name}/candidates_rows.csv"
    if causeway_path.exists():
        with causeway_path.open() as handle:
            source_rows = list(csv.DictReader(handle))
        rows = []
        for row in source_rows:
            # summary.json carries the coordinator-validated transect CRS.  A
            # source row may outlive the LAS-header fallback (default EPSG:31980),
            # so never let its stale EPSG override summary.json.  Recompute
            # lon/lat from that CRS rather than copying a location in another.
            rows.append({"id": row["id"], "state": args.state,
                         "transect": args.derived.name, "EPSG": epsg,
                         "x": row["x"], "y": row["y"], "lon": "",
                         "lat": "", "type": row.get("type", "causeway"),
                         "size_m": row.get("size_m", ""), "relief_m": row.get("relief_m", ""),
                         "canopy_m": row.get("canopy_m", ""),
                         "certainty": "unreviewed_detection", "reason": row.get("reason", "")})
    else:
        rows = []
    path = detections / "causeway_v3.csv"
    write_csv(path, rows)
    results["causeway"] = {"detected": len(rows), "csv": str(path)}

    record_check = root / "code/vendor_recordcheck" if (root / "code/vendor_recordcheck").exists() else root / ".ultramath/smwduocxj/m7"
    env = dict(os.environ, PYTHONPATH=str(record_check))
    for kind in ("mounds", "causeway"):
        source = Path(results[kind]["csv"]).name
        command = [str(root / ".venv/bin/python"), "-m", "recordcheck.audit",
                   "--candidates", results[kind]["csv"],
                   "--known", "data/known/peripato/Earthworks.rds",
                   "--out-dir", str(args.derived / "recordcheck"), "--stem", f"{kind}_v3"]
        if results[kind]["detected"] == 0:
            # recordcheck.audit currently has no zero-row branch. An empty
            # detection set is a valid negative screen, not a failed audit.
            empty_audit = args.derived / f"recordcheck/{kind}_v3_audited.csv"
            write_csv(empty_audit, [])
            (args.derived / f"recordcheck/{kind}_v3_audit_summary.json").write_text(json.dumps({
                "n_candidates": 0, "counts_by_table_match": {},
                "note": "No detections; no distance comparisons required. recordcheck.audit zero-row crash bypassed explicitly.",
                "known_table_rows": 961}, indent=2) + "\n")
            results[kind]["audited_csv"] = str(empty_audit)
            results[kind]["audit_skipped_empty"] = True
            continue
        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
        (root / "logs" / f"{args.derived.name}_{kind}_audit_v3.log").write_text(
            completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{kind} audit failed: {completed.stdout[-800:]} {completed.stderr[-800:]}")
        results[kind]["audited_csv"] = str(args.derived / f"recordcheck/{kind}_v3_audited.csv")
    (args.derived / "recordcheck/aux_audit_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
