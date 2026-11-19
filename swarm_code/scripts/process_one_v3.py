#!/usr/bin/env python3
"""Resumable coordinator pipeline for one EBA transect, without certainty claims.

Example:
  .venv/bin/python code/process_one_v3.py 0254 Acre \
    --record-id 7689909 --zip-key 'transectos acre.zip' \
    --member NP_T-0254.laz --state AC

Stages: verified member download -> m2 v3 products -> frozen ring/mound
screens -> Peripato audit -> evidence + constrained skeptic -> causeway -> notes.
Every error is logged. Raw LAZ is deleted only after products re-open correctly.
Long sunken-road screening and coordinator visual/literature upgrades stay pending.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
RECORD_CHECK = ROOT / "code/vendor_recordcheck" if (ROOT / "code/vendor_recordcheck").exists() else ROOT / ".ultramath/smwduocxj/m7"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_failure(transect, stage, exc):
    entry = {"time_utc": utc(), "stage": stage, "transect": transect,
             "failure": repr(exc), "fatal_for_transect": True}
    with (ROOT / "logs/failures.jsonl").open("a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_stage(transect, name, command, logfile, env=None):
    log_path = ROOT / logfile
    print(f"[{transect}] {name}: {' '.join(map(str, command))}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"{name} failed ({result.returncode}); tail: " + " | ".join(tail))
    print(f"[{transect}] {name}: ok", flush=True)


def update_checkpoint(transect, status, counts=None, error=None):
    queue_path = ROOT / "checkpoint/queue.json"
    queue = json.loads(queue_path.read_text())
    for row in queue:
        if row["transect"] == transect:
            row["processing_status"] = status
            row["last_update_utc"] = utc()
            if counts is not None:
                row["feature_counts_v3"] = counts
            if error is not None:
                row["failure_reason"] = str(error)
            break
    temporary = queue_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(queue, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, queue_path)
    # queue.csv is a derived view; regenerate it while preserving source order.
    import csv
    fields = ["priority", "transect", "state", "state_label", "centroid_lon", "centroid_lat",
              "priority_band", "priority_reason", "criteria", "forest_sampling_proxy",
              "floodplain_status", "nearest_known_db", "nearest_known_m",
              "within_50km_known_cluster", "epsg_l3a", "raw_availability", "source",
              "processing_status", "failure_reason", "elev_mean_l3a", "canopy_rel_l3a",
              "last_update_utc", "feature_counts_v3"]
    with (ROOT / "checkpoint/queue.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(queue)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("transect")
    p.add_argument("state_name")
    p.add_argument("--record-id", type=int, required=True)
    p.add_argument("--zip-key", required=True)
    p.add_argument("--member", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--raw-only", action="store_true", help="reuse existing raw file; do not download")
    args = p.parse_args()
    tid = args.transect
    base = ROOT / "data/derived_v2" / tid
    raw = ROOT / ".scratch/amazon/raw" / args.member
    started = time.monotonic()
    update_checkpoint(tid, "v3_in_progress")
    try:
        products_ready = (base / "summary.json").exists() and (base / "masks.npy").exists()
        if not raw.exists() and not products_ready:
            if args.raw_only:
                raise FileNotFoundError(raw)
            run_stage(tid, "download", [PYTHON, "code/fetch_zip_member.py",
                      "--record-id", args.record_id, "--zip-key", args.zip_key,
                      "--member", args.member], f"logs/{tid}_download_v3.log")
        if not (base / "summary.json").exists() or not (base / "masks.npy").exists():
            run_stage(tid, "process_v3", [PYTHON, "code/vendor_m2/process.py",
                      raw, "-o", base, "--dtm-source", "hybrid"],
                      f"logs/{tid}_process_v3.log")
        else:
            print(f"[{tid}] process_v3: existing summary/masks reused", flush=True)
        summary = json.loads((base / "summary.json").read_text())
        with np.load(base / "rasters.npz") as products:
            dtm = products["dtm"]
            assert dtm.shape == (summary["grid"]["ny"], summary["grid"]["nx"])
            assert np.isfinite(dtm).sum() > 1000
            for key in ("chm", "hillshade", "svf", "relief_10", "relief_15", "relief_30"):
                assert products[key].shape == dtm.shape, key
        if raw.exists():
            source = Path(str(raw) + ".source.json")
            if not source.exists():
                raise RuntimeError("raw provenance missing; refusing deletion")
            shutil.copy2(source, base / "raw_source.json")
            raw.unlink()
            print(f"[{tid}] raw deleted after product validation", flush=True)
        run_stage(tid, "ring_mound", [PYTHON, "code/vendor_m4/run_real.py",
                  "--transect", tid, "--npz", base / "rasters.npz",
                  "--masks", base / "masks.npy", "--summary", base / "summary.json",
                  "--out", base / "detector", "--no-evidence"],
                  f"logs/{tid}_detector_v3.log")
        run_stage(tid, "adapt", [PYTHON, "code/adapt_m2_products.py", base],
                  f"logs/{tid}_adapt_v3.log")
        audit_csv = base / f"recordcheck/rings_v3_audited.csv"
        raw_rings = json.loads((base / "detections/rings_v3.json").read_text())["detections"]
        if raw_rings:
            run_stage(tid, "record_audit", [PYTHON, "-m", "recordcheck.audit",
                      "--candidates", base / "detections/rings_v3.csv",
                      "--known", "data/known/peripato/Earthworks.rds",
                      "--out-dir", base / "recordcheck", "--stem", "rings_v3"],
                      f"logs/{tid}_recordcheck_v3.log",
                      env={**os.environ, "PYTHONPATH": str(RECORD_CHECK)})
        else:
            # recordcheck.audit currently crashes on a valid empty detection set.
            # Create the zero-row audit explicitly; this is a completed negative screen.
            import csv
            with audit_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "transect", "EPSG", "x", "y",
                    "lon", "lat", "nearest_known_db", "nearest_known_m", "d_table_match", "certainty"])
                writer.writeheader()
            (base / "recordcheck/rings_v3_audit_summary.json").write_text(json.dumps({
                "n_candidates": 0, "counts_by_table_match": {},
                "note": "Zero ring detections; explicit zero-row audit (recordcheck.audit empty-table crash bypassed)."},
                indent=2) + "\n")
            print(f"[{tid}] record_audit: zero detections, zero-row audit written", flush=True)
        run_stage(tid, "reviews", [PYTHON, "code/make_detection_reviews.py", base,
                  "--stem", "rings_v3", "--audited", audit_csv],
                  f"logs/{tid}_reviews_v3.log")
        run_stage(tid, "skeptic", [PYTHON, "code/run_skeptic.py", base, "rings_v3",
                  "--state", args.state], f"logs/{tid}_skeptic_v3.log")
        run_stage(tid, "mound_review", [PYTHON, "code/review_mounds.py", base],
                  f"logs/{tid}_mound_review_v3.log")
        run_stage(tid, "causeway", [PYTHON, "code/run_causeway.py", base],
                  f"logs/{tid}_causeway_v3.log")
        run_stage(tid, "aux_record_audit", [PYTHON, "code/audit_aux_features.py", base,
                  "--state", args.state], f"logs/{tid}_aux_audit_v3.log")
        detections = json.loads((base / "detections/rings_v3.json").read_text())["detections"]
        audit_summary = json.loads((base / "recordcheck/rings_v3_audit_summary.json").read_text())
        mounds = json.loads((base / "detector/mound_clusters.json").read_text())
        causeway = json.loads((base / f"causeway/{tid}/summary.json").read_text())
        skeptic = json.loads((base / "skeptic/rings_v3.json").read_text())
        counts = {"raw_rings": len(detections), "mound_clusters": len(mounds),
                  "causeway_candidates": causeway["n_candidates"],
                  "skeptic_verdicts": skeptic["verdict_counts"],
                  "recordcheck": audit_summary.get("counts_by_table_match", {}),
                  "mound_recordcheck": json.loads((base / "recordcheck/mounds_v3_audit_summary.json").read_text()).get("counts_by_table_match", {}),
                  "causeway_recordcheck": json.loads((base / "recordcheck/causeway_v3_audit_summary.json").read_text()).get("counts_by_table_match", {})}
        status = {"transect": tid, "state": args.state, "state_name": args.state_name,
                  "status": "v3_screens_complete_pending_coordinator_review",
                  "updated_utc": utc(), "elapsed_seconds": time.monotonic() - started,
                  "counts": counts, "raw_deleted": not raw.exists(),
                  "unvalidated_remaining": ["long sunken-road screen",
                      "coordinator visual decisions and online literature for any surviving features"],
                  "coordinate_policy": "private local files only; never publish or commit"}
        (base / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        update_checkpoint(tid, status["status"], counts)
        if tid != "0414" and (base / "masks.npy").exists():
            # rasters.npz plus summaries/candidate evidence preserve the final
            # method. Large diagnostic masks are removed after all screens pass.
            size = (base / "masks.npy").stat().st_size
            (base / "masks.npy").unlink()
            status["mask_cleanup_bytes"] = size
            (base / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        print(f"[{tid}] COMPLETE {json.dumps(counts)} elapsed={status['elapsed_seconds']:.1f}s", flush=True)
    except Exception as exc:
        log_failure(tid, "process_one_v3", exc)
        update_checkpoint(tid, "failed", error=exc)
        failure = {"transect": tid, "status": "failed", "error": repr(exc),
                   "updated_utc": utc(), "elapsed_seconds": time.monotonic() - started}
        base.mkdir(parents=True, exist_ok=True)
        (base / "status.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(f"[{tid}] FAILED {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
