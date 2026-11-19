#!/usr/bin/env python3
"""Compile the current, private v3 earthwork deliverable without upgrading claims.

Run after coordinator review of all causeway candidates:
  .venv/bin/python code/compile_results.py
Add `--zip` to write amazon_earthworks_results.zip in the workspace root and
results/. No git repository or public service is modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PROD = ROOT / "data/derived_v2"
COLS = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type", "size_m",
        "relief_m", "canopy_m", "nearest_known_db", "nearest_known_m", "literature_checked",
        "certainty", "reason"]


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=None):
    if value in (None, "", "nan", "NaN"):
        return default
    return float(value)


def native(row, epsg, known=None):
    """x/y and lon/lat for a native audited row, with no public lookup."""
    if known:
        x, y = float(row["x"]), float(row["y"])
        lon, lat = known.transform(x, y)
        return x, y, float(lon), float(lat)
    x, y = float(row["x"]), float(row["y"])
    lon = as_float(row.get("d_lon_used"), as_float(row.get("lon")))
    lat = as_float(row.get("d_lat_used"), as_float(row.get("lat")))
    if lon is None or lat is None:
        transformer = Transformer.from_crs(int(float(epsg)), 4326, always_xy=True)
        lon, lat = transformer.transform(x, y)
    return x, y, lon, lat


def decision_row(fid, state, transect, epsg, xy, typ, size, relief, canopy,
                 db, dbm, lit, certainty, reason):
    return {"id": fid, "state": state, "transect": transect, "EPSG": int(float(epsg)),
            "x": xy[0], "y": xy[1], "lon": xy[2], "lat": xy[3], "type": typ,
            "size_m": size, "relief_m": relief, "canopy_m": canopy,
            "nearest_known_db": db, "nearest_known_m": dbm,
            "literature_checked": lit, "certainty": certainty, "reason": reason}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for directory in ("dtm", "control", "progress", "code", "audit"):
        (OUT / directory).mkdir(exist_ok=True)
    queue = json.loads((ROOT / "checkpoint/queue.json").read_text())
    review_decisions = {}
    decision_path = ROOT / "checkpoint/review_decisions.json"
    if decision_path.exists():
        review_decisions = {row["id"]: row for row in json.loads(decision_path.read_text())}
    attempted = sorted([r for r in queue if (PROD / r["transect"] / "status.json").exists()],
                       key=lambda row: row["priority"])
    # A failed controller writes status.json without a usable product summary.
    # Keep that attempt visible as failed rather than crashing or relabelling it
    # as a completed main-screen transect.
    processed = [r for r in attempted
                 if (PROD / r["transect"] / "summary.json").exists()
                 and json.loads((PROD / r["transect"] / "status.json").read_text()).get("status")
                 != "failed"]
    candidates = []
    rejected = []
    transects = []
    candidate_ids = set()
    counts_by_transect = {}
    for source_row in processed:
        tid = source_row["transect"]
        base = PROD / tid
        summary = json.loads((base / "summary.json").read_text())
        epsg = int(summary["epsg"])
        state = source_row["state"]
        transform = Transformer.from_crs(epsg, 4326, always_xy=True)
        audit_rings = {row["id"]: row for row in read_csv(base / "recordcheck/rings_v3_audited.csv")}
        skeptic_rings = {row["id"]: row for row in read_csv(base / "skeptic/rings_v3.csv")}
        for fid, row in audit_rings.items():
            xy = native(row, epsg, transform)
            verdict = skeptic_rings.get(fid, {})
            decision = review_decisions.get(fid, {})
            certainty = "Known" if fid == "R0414-001" else "Rejected"
            if certainty == "Known":
                ring = next(feature for feature in json.loads(
                    (base / "skeptic/rings_v3.json").read_text())["features"]
                    if feature["id"] == fid)
                geometry = ring["geometry"]
                reason = ("Known by independent record position, not a new site: detector is about "
                          f"{float(row['nearest_known_m']):.1f} m from the Peripato record "
                          "'Multiple: (Amazon Arch | CNSA | PAST)' and about 3.5 m from the "
                          "independent Jacobs placemark named acrc33, described as 'Acre Circle 90m'; "
                          "the Peripato point and acrc33 are about 8.5 m apart. "
                          "The original 40m brief was therefore wrong for this control. Independent "
                          "geometric review gives a closed circular ditch with arc-length closure. "
                          "Local dip estimates and raised-interior interpretation still vary across "
                          "products and are retained as limitations.")
                literature = ("Peripato 2023 Earthworks.rds (961 records) plus Jacobs 2017 Amazon "
                              "Geoglyphs Mendeley DOI 10.17632/5ynjwk7gc7.1 / KML: acrc33 "
                              "Acre Circle 90m, matched offline by geodesic position.")
                item = decision_row(fid, state, tid, epsg, xy, "ring_ditch",
                                    geometry["outer_diameter_m"], geometry["dip_amp_m"],
                                    geometry["canopy"]["canopy_median_m"], row["nearest_known_db"],
                                    as_float(row["nearest_known_m"]), literature, certainty, reason)
                candidates.append(item)
                candidate_ids.add(fid)
            else:
                ring_candidate_path = base / "candidates" / fid / "candidate.json"
                if ring_candidate_path.exists():
                    record = json.loads(ring_candidate_path.read_text())
                    if record.get("certainty") != "Possible":
                        raise RuntimeError(f"unreviewed ring candidate certainty: {tid} {fid} {record.get('certainty')}")
                    if decision.get("decision") != "Possible" and verdict.get("decision") != "Possible":
                        raise RuntimeError(f"ring candidate lacks coordinator Possible decision: {tid} {fid}")
                    if fid not in candidate_ids:
                        candidates.append({column: record.get(column, "") for column in COLS})
                        candidate_ids.add(fid)
                    continue
                automatic = verdict.get("automatic_certainty")
                if automatic != "Rejected" and decision.get("decision") != "Rejected":
                    raise RuntimeError(
                        f"ring lacks an explicit coordinator verdict: {tid} {fid} "
                        f"(automatic={automatic!r}); refusing to compile it as Rejected")
                reason = verdict.get("reason") or row.get("screen_reason") or row.get("reason")
                if not reason:
                    raise RuntimeError(f"ring without rejection reason: {tid} {fid}")
                if decision.get("decision") == "Rejected":
                    # Native visual review can overturn an automatic Possible.
                    reason = decision["reason"]
                size_m = as_float(row.get("diameter_m"))
                if size_m is None:
                    radius_m = as_float(row.get("radius_m"))
                    size_m = 2 * radius_m if radius_m is not None else None
                # The original 0414 audit predates the standard ring columns.
                # Its direct geometry fields are valid source values, not blanks.
                if size_m is None:
                    size_m = as_float(row.get("size_m"))
                rejected.append(decision_row(fid, state, tid, epsg, xy,
                    row.get("shape") or row.get("type") or "ring", size_m,
                    as_float(row.get("dip_m", row.get("relief_m"))),
                    as_float(row.get("ctx_canopy_ann", row.get("canopy_m"))),
                    row.get("nearest_known_db", ""), as_float(row.get("nearest_known_m")),
                    verdict.get("literature_checked", ""), certainty, reason))

        mound_audited = {row["id"]: row for row in read_csv(base / "recordcheck/mounds_v3_audited.csv")}
        candidate_dir = base / "candidates"
        mound_candidates = {}
        if candidate_dir.exists():
            for record_path in candidate_dir.glob("*/candidate.json"):
                record = json.loads(record_path.read_text())
                if record["certainty"] != "Possible":
                    raise RuntimeError(f"unexpected auxiliary certainty: {tid} {record['id']} {record['certainty']}")
                if review_decisions.get(record["id"], {}).get("decision") != "Possible":
                    raise RuntimeError(f"auxiliary candidate lacks explicit coordinator Possible decision: {tid} {record['id']}")
                mound_candidates[record["id"]] = record
                # A ring-specific pack above and the generic scan can refer to
                # this exact same JSON. Reuse it rather than append a duplicate.
                if record["id"] not in candidate_ids:
                    candidates.append({column: record.get(column, "") for column in COLS})
                    candidate_ids.add(record["id"])
        for fid, row in mound_audited.items():
            if fid in mound_candidates:
                continue
            review_path = base / f"mound_review/{fid}.json"
            if not review_path.exists():
                raise RuntimeError(f"mound lacks native review: {tid} {fid}")
            review = json.loads(review_path.read_text())
            decision = review_decisions.get(fid, {})
            if review.get("certainty") != "Rejected" and decision.get("decision") != "Rejected":
                raise RuntimeError(
                    f"mound lacks an explicit candidate/rejection decision: {tid} {fid} "
                    f"(review={review.get('certainty')!r}); refusing to compile it as Rejected")
            if not row.get("reason") and not decision.get("reason"):
                raise RuntimeError(f"mound rejection without reason: {tid} {fid}")
            xy = native(row, epsg, transform)
            reason = (decision["reason"] if decision.get("decision") == "Rejected" else
                      "Rejected by mound geometry and visual relief review: " + row["reason"])
            rejected.append(decision_row(fid, state, tid, epsg, xy, "mound_cluster",
                as_float(row.get("size_m")), as_float(row.get("relief_m")),
                as_float(row.get("canopy_m")), row.get("nearest_known_db", ""),
                as_float(row.get("nearest_known_m")), "", "Rejected", reason))

        cw_audited = read_csv(base / "recordcheck/causeway_v3_audited.csv")
        for row in cw_audited:
            fid = row["id"]
            decision = review_decisions.get(fid, {})
            verdict_path = base / f"causeway/{tid}/evidence/{fid}/skeptic_line_verdict.json"
            if not verdict_path.exists():
                raise RuntimeError(f"raised linear feature lacks an independent verdict: {tid} {fid}")
            verdict = json.loads(verdict_path.read_text())
            # An explicit native coordinator rejection outranks a machine Possible,
            # even if the original automatic file has not been rewritten.
            if decision.get("decision") == "Rejected" and verdict.get("certainty") != "Rejected":
                verdict = {**verdict, "certainty": "Rejected", "reason": decision["reason"]}
            if verdict.get("certainty") == "Possible":
                if not verdict.get("native_visual_review") and decision.get("decision") != "Possible":
                    raise RuntimeError(
                        f"Possible causeway lacks coordinator native review: {tid} {fid}")
                candidate_path = base / "candidates" / fid / "candidate.json"
                if not candidate_path.exists():
                    raise RuntimeError(f"Possible causeway lacks complete evidence: {tid} {fid}")
                record = json.loads(candidate_path.read_text())
                # The generic auxiliary-candidate discovery above already added
                # this exact evidence pack.  Validate the line verdict, but do
                # not append the same feature a second time.
                if fid not in candidate_ids:
                    candidates.append({column: record.get(column, "") for column in COLS})
                    candidate_ids.add(fid)
                continue
            if verdict.get("certainty") != "Rejected":
                raise RuntimeError(f"unresolved causeway candidate: {tid} {fid}: {verdict.get('certainty')}")
            if not verdict.get("reason"):
                raise RuntimeError(f"causeway rejection has no reason: {tid} {fid}")
            xy = native(row, epsg, transform)
            rejected.append(decision_row(fid, state, tid, epsg, xy,
                row.get("type", "causeway"), as_float(row.get("size_m")),
                as_float(row.get("relief_m")), as_float(row.get("canopy_m")),
                row.get("nearest_known_db", ""), as_float(row.get("nearest_known_m")), "",
                "Rejected", verdict["reason"]))
        rings = json.loads((base / "detections/rings_v3.json").read_text())["detections"]
        mounds = json.loads((base / "detector/mound_clusters.json").read_text())
        causeway = json.loads((base / f"causeway/{tid}/summary.json").read_text())
        detb_summary_path = base / "detb_review/summary.json"
        detb_count = json.loads(detb_summary_path.read_text()).get("n_screened", 0) if detb_summary_path.exists() else 0
        counts_by_transect[tid] = len(rings) + len(mounds) + int(causeway["n_candidates"]) + int(detb_count)

    for row in queue:
        tid = row["transect"]
        base = PROD / tid
        if tid in counts_by_transect:
            summary = json.loads((base / "summary.json").read_text())
            source_meta = json.loads((base / "raw_source.json").read_text()) if (base / "raw_source.json").exists() else {}
            process_row = json.loads((base / "transect_row.json").read_text()) if (base / "transect_row.json").exists() else {}
            transects.append({"transect": tid, "state": row["state"], "EPSG": summary["epsg"],
                              "status": "skipped",
                              "download_s": round(float(source_meta.get("elapsed_seconds", 0)), 2) or "",
                              "process_s": process_row.get("process_time_s", ""),
                              "empty_dtm_pct": round(float(summary["empty_dtm_pct"]), 3),
                              "n_features": counts_by_transect[tid],
                              "note": "Main v3 ring/mound/causeway screens complete, but not marked done because the long sunken-road method remains unvalidated; raw LAZ deleted after product validation."})
        elif (base / "status.json").exists():
            failure = json.loads((base / "status.json").read_text())
            transects.append({"transect": tid, "state": row["state"], "EPSG": row.get("epsg_l3a", ""),
                              "status": "failed", "download_s": "", "process_s": "",
                              "empty_dtm_pct": "", "n_features": "",
                              "note": failure.get("error") or row.get("failure_reason") or "Processing failed; see status.json and logs/failures.jsonl."})
        else:
            transects.append({"transect": tid, "state": row["state"], "EPSG": row.get("epsg_l3a", ""),
                              "status": "failed" if row["raw_availability"] == "missing_l1a_inventory" else "skipped",
                              "download_s": "", "process_s": "", "empty_dtm_pct": "",
                              "n_features": "",
                              "note": row.get("failure_reason") or "Never attempted; remains in coordinator's priority queue."})

    all_ids = [row["id"] for row in candidates + rejected]
    if len(all_ids) != len(set(all_ids)):
        duplicates = sorted({fid for fid in all_ids if all_ids.count(fid) > 1})
        raise RuntimeError(f"duplicate feature ids: {duplicates}")
    if set(record["id"] for record in candidates) != candidate_ids:
        raise RuntimeError("candidate id set mismatch")
    certainty = {level: sum(row["certainty"] == level for row in candidates)
                 for level in ("Known", "Strong", "Possible")}
    status_counts = {level: sum(row["status"] == level for row in transects)
                     for level in ("done", "failed", "skipped")}
    failed_ids = [row["transect"] for row in transects if row["status"] == "failed"]
    other_failed = sorted(set(failed_ids) - {"0082", "0865"})
    failed_detail = ("transects 0082 and 0865 have no listed L1A member" if not other_failed else
                     "unavailable L1A entries and/or processing failures; see the per-row notes in transects.csv: "
                     + ", ".join(failed_ids))
    possible_records = [record for record in candidates if record["certainty"] == "Possible"]
    possible_ids = ", ".join(record["id"] for record in possible_records) or "none"
    possible_types = {kind: sum(record["type"] == kind for record in possible_records)
                      for kind in sorted({record["type"] for record in possible_records})}
    possible_type_text = ", ".join(f"{kind}={count}" for kind, count in possible_types.items()) or "none"
    state_counts = {state: sum(row["state"] == state for row in processed)
                    for state in sorted({row["state"] for row in processed})}
    state_count_text = ", ".join(f"{state}={count}" for state, count in state_counts.items())
    forest_blind_controls = []
    for row in processed:
        review_path = PROD / row["transect"] / "control_trees_inpe/review.json"
        if review_path.exists():
            review = json.loads(review_path.read_text())
            if review.get("status") == "under_forest_blind_control_not_detection_validation":
                forest_blind_controls.append(row["transect"])
    forest_control_text = ", ".join(forest_blind_controls) or "none"
    outcome = {
        "generated_utc": now_iso(),
        "transects_total": len(transects),
        "processed_to_dtm_and_main_screens": len(processed),
        "done_full_screen": status_counts["done"], "failed": status_counts["failed"],
        "skipped_or_incomplete": status_counts["skipped"],
        "candidates": certainty, "rejected_features": len(rejected),
        "features_examined": len(candidates) + len(rejected),
        "long_sunken_road_screen_validated": False,
        "under_forest_positive_control_found": False,
        "under_forest_blind_control_transects": forest_blind_controls,
        "new_site_claim_made": False,
        "note": "Main-screen results only. Remaining processing, under-forest validation and long sunken-road work are not complete."
    }
    (OUT / "outcome.json").write_text(json.dumps(outcome, indent=2, ensure_ascii=False) + "\n")
    write_csv(OUT / "candidates.csv", COLS, candidates)
    write_csv(OUT / "rejected.csv", COLS, rejected)
    write_csv(OUT / "transects.csv",
              ["transect", "state", "EPSG", "status", "download_s", "process_s",
               "empty_dtm_pct", "n_features", "note"], transects)

    # Remove stale generated evidence from earlier checkpoints so a feature that
    # was subsequently rejected cannot survive only in the ZIP.  Source products
    # under data/derived_v2 are never touched here.
    (OUT / "candidates").mkdir(exist_ok=True)
    for path in (OUT / "candidates").iterdir():
        if path.is_dir() and path.name not in candidate_ids:
            shutil.rmtree(path)
    current_dtm = {f"{row['transect']}_dtm3m.npz" for row in processed}
    for path in (OUT / "dtm").glob("*_dtm3m.npz"):
        if path.name not in current_dtm:
            path.unlink()
    current_audit = {row["transect"] for row in attempted}
    for path in (OUT / "audit").iterdir():
        if path.is_dir() and path.name not in current_audit:
            shutil.rmtree(path)

    # Copy only vetted evidence; legacy pilot products remain outside the final bundle.
    for record in candidates:
        tid = record["transect"]
        if record["id"] == "R0414-001":
            src = PROD / tid / "review" / record["id"]
        else:
            src = PROD / tid / "candidates" / record["id"]
        if src.exists():
            destination = OUT / "candidates" / record["id"]
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.part", "__pycache__"))
        if record["certainty"] in {"Strong", "Possible"}:
            required_evidence = ("hillshade.png", "relief.png", "sky-view.png", "canopy.png",
                                 "radial_profile.csv", "cross_profiles.csv", "skeptic_notes.md")
            missing_evidence = [name for name in required_evidence
                                if not (destination / name).is_file()]
            if missing_evidence:
                raise RuntimeError(f"{record['id']} evidence pack missing required files: {missing_evidence}")
    for row in attempted:
        tid = row["transect"]
        base = PROD / tid
        if tid in counts_by_transect:
            # Link the source DTM rather than making a second on-disk copy.
            # The archive still contains its own bytes; dedupe later verifies
            # all three copies before dropping this directory entry.
            dtm_target = OUT / "dtm" / f"{tid}_dtm3m.npz"
            if dtm_target.exists() or dtm_target.is_symlink():
                dtm_target.unlink()
            os.link(base / "dtm.npz", dtm_target)
        for name in ("rings_v3_audited.csv", "mounds_v3_audited.csv", "causeway_v3_audited.csv"):
            if tid in counts_by_transect and (base / "recordcheck" / name).exists():
                destination = OUT / "audit" / tid
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base / "recordcheck" / name, destination / name)
        detb_audit = base / "detb_review/recordcheck/lines_detb_audited.csv"
        if detb_audit.exists():
            (OUT / "audit" / tid).mkdir(parents=True, exist_ok=True)
            shutil.copy2(detb_audit, OUT / "audit" / tid / "lines_detb_audited.csv")
        (OUT / "audit" / tid).mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "status.json", OUT / "audit" / tid / "status.json")
        for name in ("raw_source.json", "summary.json", "screen_state.json", "storage_compaction.json", "storage_slimming.json"):
            if (base / name).exists():
                shutil.copy2(base / name, OUT / "audit" / tid / name)
    control_source = PROD / "0414/review/R0414-001"
    shutil.copytree(control_source, OUT / "control/0414", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.part", "__pycache__"))
    (OUT / "control").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/literature/control_0414_acrc33.json", OUT / "control/0414_acrc33.json")
    for row in processed:
        tid = row["transect"]
        control = PROD / tid / "control_trees_inpe"
        if control.exists():
            shutil.copytree(control, OUT / "control" / f"{tid}_trees_inpe", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.part", "__pycache__"))

    if (ROOT / "code/references/record_m7").exists():
        shutil.copytree(ROOT / "code/references/record_m7", OUT / "audit/record_m7", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.part"))

    # Outcome-first report and the automatically derived progress note.
    report = f"""# Amazon EBA airborne-lidar earthwork screening

## Outcome

Generated `{outcome['generated_utc']}`. This is a **main-screen checkpoint**, not a completed survey.

- **{len(processed)} transects processed** to 3 m DTMs and main ring/mound/causeway screens ({state_count_text}).
- **{status_counts['done']}** listed as `done`, because the long sunken-road screen remains unvalidated.
- **{status_counts['failed']} failed**: {failed_detail}.
- **{status_counts['skipped']} skipped or incomplete** in the complete 901-row `transects.csv`; {len(processed)} of these have main-screen results but are not promoted to `done`.
- **Candidates: {certainty['Known']} Known, {certainty['Strong']} Strong, {certainty['Possible']} Possible.**
- **{len(rejected)} Rejected** features; {len(candidates)+len(rejected)} features examined.

**No feature is claimed as a new archaeological site.** Table misses and incomplete public registers are not evidence of novelty.

**Rejection versus recorded places:** **Rejected describes the evaluated detection against the target geometry, not the archaeological status of the surrounding location.** A false or indistinct ring, channel, levee or artefact can fall inside a known register's radius. Rejecting that detection does not change the recorded place's status or prove that no measured earthwork exists there. These rows must not be reported as new/unrecorded absence claims.

## Verified results

### Positive control: Known (`R0414-001`)

The 0414 enclosure is about 5 m from the Peripato record, while independent Jacobs/Mendeley KML geometry places `acrc33`, described as **Acre Circle 90m**, about 8.5 m from that record. Blind `v3` recovery is an approximately 98 m closed circular ditch. This is consistent with the register and shows why the brief's unqualified approximately 40 m-scale control description is not an adequate outer-diameter measurement: the recorded outline has fitted radius 42.5 m and the current blind-control outer diameter is 97.94 m. Product and profile reviews cover local depth, closure, raised-interior interpretation and hole masks rather than hiding their disagreement.

**Measurement provenance:** the early audited row in `audit/0414/rings_v3_audited.csv` carries 84.0 m, 1.67 m and 1.3 m. The current candidate deliberately uses the blind control fit's `outer_diameter_m` (97.94 m), `dip_amp_m` (1.319 m) and `canopy_median_m` (1.485 m) from `skeptic/rings_v3.json`. These are different audit stages/definitions, not averaged quantities or a change to the Known certainty. The separate rejected `R0414-002` row retains its own carried source geometry (33.0 m, 0.99 m, 4.7 m).

### Bounded detB reproduction

The finite 0414 detB control is independently byte-identical to the accepted reference. Six current open-gate rows reproduce **11.755044 km², 337 candidates and 0 accepted features**. The coordinator's fresh open-0414 run is byte-identical to the member's single-scene output; the other five are member detector reruns checked at artifact, frozen-mask and DTM-hash level, not five fresh coordinator detector reruns. The open rows for 0905, 0918, 0862 and 0413 remain `INPUT_CHANGED` and unverified because their canopy rasters were removed. The approximately 47.7 h full-899 cost is an extrapolation, not a completed sweep. This bounded reproduction does not validate production long-road screening, under-forest detection or archaeological absence.

### Possible candidates (not Strong, not novelty claims)

| id | transect | type | certainty | principal limitation |
|---|---|---|---|---|
"""
    for record in candidates:
        if record["certainty"] == "Possible":
            report += f"| {record['id']} | {record['transect']} | {record['type']} | Possible | partial geometry and natural/modern/register explanations remain open |\n"
    report += f"""

## Unverified claims and open questions

- **{len(possible_records)} Possible features** remain ({possible_type_text}). They have incomplete geometry or unresolved natural, modern and register explanations; they are not established settlements, confirmed causeways or unrecorded sites.
- Peripato 2023 and the public Jacobs KML were checked locally against native coordinates. No definitive named match was established for the Possible candidates. A table match is existential: some record lies inside its own radius (2 km for TREES/INPE, 500 m otherwise); it is not the nearest record's radius alone.
- The KML distinguishes site-like placemarks from per-year `surveys > YYYY` flight tracks. Flight-track distances are not site identities, and a shared underlying dataset is not independent archaeological confirmation. Centroid/line/outline proximity is not proof of identity.
- Pärssinen et al. (2026) documents hundreds of recently mapped southwest Amazon earthworks, but no complete public per-coordinate register was obtained for all of them. Novelty therefore remains unresolved, not affirmed.
- Sunken-road detection over distances of at least 200 m remains unvalidated as a general screen. Independently reproduced benchmarks establish only a finite-input, open-ground sunken subset; the processor's documented mask, embanked class, modern-road flags and under-forest/real-transect validation remain unresolved. Raised causeways were tested independently, but that does not close the long-line gap.
- The real detB row `D0414-L001` is retained only as **Possible**: it is a subdued straight linear depression aligned with a field/hedgerow boundary. Farm/drainage/land-division explanations remain, and nearby Peripato/Jacobs records are not identity matches. It is not a real positive control or a new-road claim.
- No real under-forest positive control has validated detection sensitivity. Forest cells with inadequate ground returns cannot be treated as surveyed-negative territory.
- Targeted TREES/INPE footprint blind-control transects: **{forest_control_text}**. Their native reviews are not positive detection controls or absence claims: detector masks admit negligible or zero support around the published points, while displaced positions and reconstruction prevent a conclusion about whether the registered earthworks are present. Full measurements and images are retained under `control/<transect>_trees_inpe/`.

## Failed directions and corrections

- The literal first-pass grey-opening processor could erase real ditches and manufacture annular relief. It was superseded by `v3` (hybrid min/q10, low-outlier guard, enclosure-preserving median guard and pre-clamp mound surface); all its outputs are pilot only.
- An early skeptic fallback substituted a different nearby feature for the intended detection. Secondary centre drift and fallback substitution were disabled before constrained review.
- Record-audit zero-row handling, mound/causeway audit omission and audit-before-causeway ordering were corrected during controller development; no result from those incomplete checks was promoted.
- General search providers returned 402; Science presented a human challenge. OpenAlex/Crossref metadata were used only as leads, while direct registers, Mendeley KML and Europe PMC full text supplied the substantive checks.
- The legacy long-line detector fails a noise-free trench at seven of eight bearings because its direction/normal conventions are transposed. The accepted m2 benchmark reproduces a narrow open-ground sunken detB subset (fixed recall 0.88, held-out 1.00), but overall gates remain failed: documented-mask inputs return nothing, embanked roads are absent, modern tracks lack modern flags, and no real-transect/under-forest validation exists. m1's experimental alternative also fails valley/terrace confounders (412 m and 673 m false lines) and is retained only as measured negative/partial evidence, never as a validated subset. m4's independent real-substrate analysis supports only that same open-ground sunken domain, with contiguous-ground and boundary limitations explicit; embanked/forest/modern/curved/general detection remains open.
- Sensitivity varies substantially with radius, ditch width and reference placement. Detector dip figures are not directly transferable between DTM recipes.

## Suggested next steps (maximum five)

1. Finish a validated long sunken-road detector for features >=200 m, then revisit all {len(processed)} main-screen transects before calling any fully done.
2. Check the {len(possible_records)} Possible features with archaeologists and additional registers; do not contact anyone and do not send the draft letter in this archive without a deliberate human decision.
3. Restore or reprocess the four `INPUT_CHANGED` detB scenes and rerun only the frozen open-gate comparison; do not substitute finite-gate results or launch an unbounded all-899 sweep.
4. Find a published under-forest geoglyph inside a transect and validate detection there; otherwise report canopy limits instead of extrapolating.
5. If stronger evidence emerges, review all Possible candidates again in native images and profiles before any upgrade; reject false positives before optimising throughput.

## Package notes

`transects.csv` has all 901 footprint rows and honest `status` values. `candidates.csv` contains Known/Strong/Possible only; every Rejected feature is in `rejected.csv`. Candidate folders contain native hillshade, relief, sky-view and canopy images plus radial/cross profiles and skeptic notes. Raw LAZ files were deleted only after product validation, except the 0414 control LAZ retained for independent tests. Source checksums, `status.json`, compaction manifests and audit records are included under `audit/`.
"""
    (OUT / "REPORT.md").write_text(report)
    progress = f"""# Main-screen checkpoint {outcome['generated_utc']}

- Transects fully processed through main screens: {len(processed)}
- Failed downloads / unavailable L1A entries: {status_counts['failed']}
- Remaining unprocessed: {status_counts['skipped'] - len(processed)}
- Main-screen results not promoted to `done`: {len(processed)} (long sunken-road validation remains open)
- Known: {certainty['Known']}; Strong: {certainty['Strong']}; Possible: {certainty['Possible']}; Rejected: {len(rejected)}
- Features examined: {len(candidates)+len(rejected)}
- New-site claims: none

Processed-state distribution: {state_count_text}. Possible feature IDs: {possible_ids}. They retain natural/modern and register-match uncertainty. Processing follows checkpoint/queue.json; the long-road and under-forest limitations remain open.
"""
    (OUT / "progress/main_screen_checkpoint.md").write_text(progress)
    (ROOT / "progress/main_screen_checkpoint.md").write_text(progress)
    # Retain the historical filename for links to the initial Acre checkpoint,
    # but write current content rather than stale Acre-only counts.
    (OUT / "progress/acre_main_screen.md").write_text(progress)
    (ROOT / "progress/acre_main_screen.md").write_text(progress)
    shutil.copytree(ROOT / "progress", OUT / "progress", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.part", "__pycache__"))

    # Brazillian-Portuguese and English drafts. Positions appear only in the annex.
    draft = f"""# Rascunho de consulta ao IPHAN e especialistas em geoglifos do Acre / Draft inquiry

## Português — não enviar sem revisão humana

Prezados pesquisadores,

Sou estudante de Física de Engenharia, não arqueólogo, e estou aprendendo a trabalhar com LiDAR aéreo público. Este rascunho descreve uma análise geoespacial, não uma descoberta nem um pedido de validação evasiva.

Revisei os dados públicos EBA e processei os produtos em uma grade de 3 m. A análise incluiu checagem local contra os 961 registros de Peripato et al. (2023), a base pública Jacobs/Mendeley (KML) e literatura — inclusive Pärssinen et al. (2026) e seus anexos. As posições dos candidatos não foram publicadas em repositórios, mapas, issues ou sites de compartilhamento.

No momento não há candidatos classificados como **Strong**. Há **{len(possible_records)}** candidatos classificados como **Possible**: {possible_ids}. Tipos: {possible_type_text}. São feições geométricas incompletas e todas mantêm explicações naturais, modernas ou de registro em aberto. Não afirmo que sejam sítios novos, aldeias, calçadas confirmadas ou geoglifos desconhecidos.

A característica de controle `acrc33` foi verificada como **Known**, pela proximidade com o registro Peripato e a descrição “Acre Circle 90m” na base Jacobs. O relatório de processamento não deve ser interpretado como inventário arqueológico completo; a detecção de estradas lineares longas e o controle sob floresta continuam limitados.

Pergunto, se for do conhecimento dos senhores: esses agrupamentos já estão registrados no CNSA, em bases regionais ou em trabalhos publicados? Existe alguma razão geomorfológica ou de uso atual da terra que deva ser avaliada antes de prosseguir com análise? Posso compartilhar as imagens e perfis de forma privada com profissionais responsáveis, respeitando restrições de localização.

Não enviei este documento e agradeço qualquer orientação metodológica. Nenhuma ação é solicitada a pessoas que não sejam os especialistas apropriados, mediante decisão humana posterior.

Com respeito,
[M nome do remetente, estudante de Física de Engenharia]

## English — do not send without human review

Dear researchers,

I am an engineering-physics student, not an archaeologist, learning to analyse public airborne LiDAR. This is a remote-sensing work in progress, not a discovery claim and not a request for automatic validation.

I processed the public EBA transects to 3 m products, checked the 961 Peripato et al. (2023) records locally together with the public Jacobs/Mendeley KML and literature including Pärssinen et al. (2026) and its supplements. Candidate positions were not posted publicly, in maps, repositories, issues or paste sites.

There are no **Strong** candidates. {len(possible_records)} **Possible** candidates remain: {possible_ids}. Types: {possible_type_text}. They are incomplete geometric features with natural, modern and registration explanations still open; they are not claimed as new sites, mound villages, confirmed causeways or geoglyphs.

The `acrc33` control is **Known** by its position relative to the Peripato record and the Jacobs description “Acre Circle 90m”. Long sunken-road and under-forest detection remain limitations.

Could you please advise whether these features are already recorded in CNSA or regional datasets/papers, or whether a modern/natural explanation should be investigated before further work? Evidence and profiles can be shared privately with appropriate specialists in accordance with any location-sensitivity requirements.

This draft has not been sent. Thank you for any methodological guidance.

Sincerely,
[M name, engineering-physics student]

## Private position annex / Anexo privado de posições — never include in a public message

| id | certainty | transect | EPSG | x | y | lon | lat |
|---|---|---|---|---|---|---|---|
"""
    for row in candidates:
        draft += f"| {row['id']} | {row['certainty']} | {row['transect']} | {row['EPSG']} | {row['x']} | {row['y']} | {row['lon']} | {row['lat']} |\n"
    draft += "\nSeparate annex for authorised recipients only; coordinates must stay private and are permitted here only inside the private ZIP.\n"
    (OUT / "report_draft_IPHAN.md").write_text(draft)

    # Per-candidate and general sources are copied, never any web query with positions.
    literature = ROOT / "literature.md"
    lines = [literature.read_text(), """

# Current v3 Possible candidates

The following entries record checks actually performed; no source establishes novelty.

"""]
    for record in candidates:
        if record["certainty"] == "Possible":
            lines.append(f"## {record['id']}\n\n- Peripato nearest: {record['nearest_known_db']}, {record['nearest_known_m']} m, no table match.\n")
            if (PROD / record["transect"] / "candidates" / record["id"] / "literature.md").exists():
                lines.append((PROD / record["transect"] / "candidates" / record["id"] / "literature.md").read_text())
            lines.append("\n- Nature 2026 full XML and all supplementary files were obtained via Europe PMC. The paper documents many new earthworks, but its public material did not provide a complete per-coordinate register that would close this candidate's novelty question.\n")
    (OUT / "literature.md").write_text("\n".join(lines))

    # Code and exact reproduction commands.
    source_note = f"""# Code in this private bundle

The scripts were run from the project root with the project-local `.venv`. No network access is required by these offline phases except downloading public data.

## Main commands

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'laspy[lazrs]' pyreadr pyproj remotezip pyshp
.venv/bin/python code/download_public_inputs.py --inspect-acre
.venv/bin/python code/inventory_remote_archives.py
.venv/bin/python code/build_queue.py
.venv/bin/python code/vendor_m2/process.py .scratch/amazon/raw/NP_T-0414.laz -o data/derived_v2/0414 --dtm-source hybrid
.venv/bin/python code/vendor_m2/synth.py .scratch/m2_synth
.venv/bin/python code/vendor_m2/process.py .scratch/m2_synth/synth.las -o .scratch/m2_synth_products --dtm-source hybrid
.venv/bin/python code/vendor_m2/validate_synth.py .scratch/m2_synth --out=.scratch/m2_synth_products

# Local known-record footprint/control audit (no network or candidate positions)
.venv/bin/python code/find_known_controls.py
.venv/bin/python code/build_trees_control_evidence.py 0470

# One complete main-screen transect (raw network phase: coordinator only)
.venv/bin/python code/process_one_v3.py 0254 Acre --record-id 7689909 \\
  --zip-key 'transectos acre.zip' --member NP_T-0254.laz --state AC

# Bounded sequential queue; line verdicts are automatic, other upgrades are not
.venv/bin/python code/run_queue_v3.py --limit 25 --min-free-gb 3

# Auxiliary Possible candidate evidence (after coordinator review/audit)
.venv/bin/python code/build_aux_candidate.py data/derived_v2/0254 M0254-013
.venv/bin/python code/judge_causeway_candidate.py data/derived_v2/0516 0516-CW001

# Storage lifecycle: dry-run first; only fully Rejected/empty screens are eligible.
# The second command also drops their validated audit caches; candidates/controls are excluded.
.venv/bin/python code/compact_derived_v2.py --all --dry-run
.venv/bin/python code/compact_derived_v2.py --all --drop-screen-state

# Compilation and validation, never certainty promotion
.venv/bin/python code/preflight_compile.py
.venv/bin/python code/compile_results.py --zip
unzip -t amazon_earthworks_results.zip
.venv/bin/python code/references/package_m7/delivcheck.py --zip amazon_earthworks_results.zip --deep --json checkpoint/delivcheck_final.json
.venv/bin/python code/dedupe_results_dtm.py
.venv/bin/python code/link_archive_mirror.py
.venv/bin/python code/verify_archive.py amazon_earthworks_results.zip logs/verify_archive_final.json
.venv/bin/python code/final_deliverable_audit.py amazon_earthworks_results.zip --json logs/final_deliverable_audit.json
```

`process_one_v3.py` performs verified remote-ZIP download, `v3` processing, ring/mound screens, Peripato auditing, native evidence, constrained skeptic diagnostics, causeway screening and mound/causeway audits. Raw LAZ is deleted only after products re-open successfully. Zero detections are valid results. Auxiliary Possible packs use the immobile-mound model and include Jacobs and Nature register checks. `writer validate` and other swam tests are recorded in `code/references/`.

## Validation actually rerun

- m1 synthetic benchmark `SKIP_GEN=1 bash synth/run_all.sh`: six-scene DTM and outlier comparisons.
- m2 production processor: **11/11** checks on 1,174,347 synthetic points.
- m3 causeway self-test and 14-scenario scorer, including explicit embedded-positive hole-span test.
- m4 ring synthetic suite and **6/6** real-DTM digital spike recoveries.
- m5 independent raw-return v3 control audit rerun: exit 0 in 23.6 s, relief null and independent cloth-DTM control reproduced; reconstructed cells are not mislabelled as deleted.
- m6 skeptic **22** unit tests and **36** control rows: 10/10 positives kept, 0/26 false positives.
- Storage compaction verifies exact source/compact array equality and detector-mask counts before deleting intermediate arrays; 0414, 0072, targeted TREES control leads and all candidate transects retain full products.
- m2 long-road benchmark: **30/30** harness checks, byte-identical fixed/held-out evaluator reruns, and decisive legacy P7 coverage 0.98895 with zero screened false metres over 14.09 km². See `code/scripts/references/road_m2/COORDINATOR_REVIEW.md`; overall road-screen gates still fail and no transect is done.
- m3 detB: byte-identical real 0414 adapter output plus an independently rerun real-DTM spike test (three planted classes recall 1.00 at coverage >=0.5, zero accepted null rows). See `code/scripts/references/road_m3/COORDINATOR_REVIEW.md`; mask, embankment, modern-flag and real/forest validation remain open.
- m1 Track A: fixed/held-out/legacy evaluations reproduced byte-for-byte; corrected own-scorer recalls cal 0.7111 and hold 0.6889 independently rerun. Accepted only as measured negative/partial evidence because valley/terrace confounders fail; see `code/scripts/references/road_m1/COORDINATOR_REVIEW.md`, including the H3 579.85 m (not 589 m) erratum.
- m4 real-substrate review: six mechanical checks plus P7 recovery (336.919 m, azimuth error 0.4267°, zero null-twin rows), independent south-up anchors and an exact finite-gate 0414 audit reproduced. Its original combined open-gate rerun remained unfinished at that review; see `code/scripts/references/road_m4/COORDINATOR_REVIEW.md`, including the coordinate-convention qualification.
- m1 bounded detB follow-up: the finite control is independently byte-identical, and six current open rows reproduce 11.755044 km² / 337 candidates / 0 accepted. Independent open-0414 output is byte-identical to the member's single-scene JSON and exactly matches the accepted 0414 subset; the other five are member detector reruns checked at artifact/gate level. Rows 0905, 0918, 0862 and 0413 remain INPUT_CHANGED because their canopy rasters were removed. The approximately 47.7 h full-899 cost is an extrapolation, not a completed run. See `code/scripts/references/detb_open_m1/COORDINATOR_REVIEW.md`; this is not production-road or under-forest validation.
- m8 writer final audit **66** tests passed; its output was not adopted because its `partial` status is outside the attachment's `done`/`failed`/`skipped` vocabulary.
- m7 record checker: all seven offline test scripts passed, including existential-radius regression, three-backend RDS parity, geodesy against PROJ/Karney, footprint proximity, KML source discrimination and a substantive audit of the current candidate/rejected rows. Evidence and coordinator scope: `code/scripts/references/record_m7/COORDINATOR_REVIEW.md`.

Source hashes and provenance are retained in `code/vendor_*/SOURCE.json` where applicable. Candidate locations and raw L1A payloads are never committed to git.

## Policy

No feature is upgraded by compilation. Auxiliary audit completion requires an independent verdict for every raised linear feature before packaging. The long sunken-road method remains unvalidated; therefore `done_full_screen` is zero, not {len(processed)}.
"""
    (OUT / "code/README.md").write_text(source_note)
    shutil.copytree(ROOT / "code", OUT / "code/scripts", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.part", ".scratch"))
    # Provenance and queue summary, useful for auditing the deliberately private positions.
    shutil.copy2(ROOT / "checkpoint/queue_summary.json", OUT / "queue_summary.json")
    shutil.copy2(ROOT / "checkpoint/state.json", OUT / "checkpoint_state.json")
    (OUT / "SOURCE_DATA.md").write_text("""# Sources and privacy

Public inputs: Zenodo 8083287, 4968706, 7689909, 7636454, 7689693, 7689210; L1A files are CC BY 4.0. Known-site checks use Peripato 2023 Earthworks.rds, Jacobs 2017 Mendeley/KML and direct literature links listed in `literature.md`.

The raw EBA transects are not bundled because of disk constraints. Each verified raw LAZ was deleted only after derived products passed shape and point-count checks, except the 0414 control LAZ retained locally for independent tests; provenance source URLs and CRC/SHA-256 records remain in each transect's `audit/raw_source.json`. Rejected/empty-screen storage compaction occurs only after re-opened compact arrays pass equality and detector-count checks. A later cache-removal step deletes those screen_state caches only for the same eligible negative screens, with `audit/storage_slimming.json` recording cache hashes and deliberate raw-refetch restoration instructions; `dtm.npz`, tiles and audit records are retained.

Candidate and rejected-feature positions, images, DTMs and draft annex are in this **private archive only**. Nothing was pushed, posted, emailed or released publicly.
""")

    if args.zip:
        archive_path = ROOT / "amazon_earthworks_results.zip"
        mirror_path = OUT / "amazon_earthworks_results.zip"
        mirror_temporary = mirror_path.with_suffix(".zip.link.part")
        if mirror_temporary.exists() or mirror_temporary.is_symlink():
            mirror_temporary.unlink()
        temporary = archive_path.with_suffix(".zip.part")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
            for path in sorted(OUT.rglob("*")):
                if path.is_file() and path.suffix != ".zip" and "amazon_earthworks_results.zip.part" not in str(path):
                    archive.write(path, path.relative_to(OUT))
        temporary.replace(archive_path)
        print(f"Wrote {archive_path}: {archive_path.stat().st_size} bytes")
        # The mirror is defined as the same bytes, so do not serialize a second
        # ~1 GB temporary ZIP.  Link it into place atomically and leave no mirror
        # .part behind if the compile is interrupted during the root write.
        os.link(archive_path, mirror_temporary)
        os.replace(mirror_temporary, mirror_path)
        print(f"Linked {mirror_path}: {mirror_path.stat().st_size} bytes")
    print(json.dumps(outcome, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
