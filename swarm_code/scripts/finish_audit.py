#!/usr/bin/env python3
"""Read-only completion audit for the EBA main-screen project (m8, swarm smwj89561).

Recomputes, from files only:
  * queue completion state (processed / queued / failed / missing-L1A);
  * measured per-transect wall time from checkpoint/queue_runner*.json;
  * ETA for the remaining queue at mean / p90 and against a time budget;
  * measured net disk growth per transect from the free-GB history;
  * archive size projection and the transient disk peak the current
    compile_results.py --zip needs, so ENOSPC is predicted before it happens;
  * staleness of the shipped archive relative to the live queue;
  * open coordinator verdict items that would block a compile.

Writes nothing. stdlib only. Exit 0 = every gate passes, 1 = a gate fails.

Usage (from the project root):
  .venv/bin/python code/finish_audit.py
  .venv/bin/python code/finish_audit.py --budget-minutes 600 --json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import shutil
import statistics
import sys
import zipfile
from pathlib import Path

MB = 1e6
GB = 1e9
GIB = 1024 ** 3
ARCHIVE_LIMIT_TARGET = 5 * GB          # project rule: eventual archive must fit 5 GB
RUNNER_DEFAULT_MIN_FREE = 3.0          # code/run_queue_v3.py --min-free-gb default (GiB)
COMPILE_RESULTS_SOURCE = "code/compile_results.py"
RUNNER_SOURCE = "code/run_queue_v3.py"


def q(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 120 else f"{seconds / 60:.2f}min"


def audit(root: Path, budget_minutes: float, floor_gib: float | None = None) -> dict:
    queue_path = root / "checkpoint/queue.json"
    prod = root / "data/derived_v2"
    report: dict = {"root": str(root)}

    # ---------------------------------------------------------------- queue
    queue = json.loads(queue_path.read_text())
    status = collections.Counter(r.get("processing_status", "") for r in queue)
    with_dir = [r for r in queue if (prod / r["transect"] / "status.json").exists()]
    processed = [r for r in with_dir
                 if (prod / r["transect"] / "summary.json").exists()
                 and json.loads((prod / r["transect"] / "status.json").read_text()).get("status") != "failed"]
    queued_available = [r for r in queue
                        if r.get("processing_status") == "queued"
                        and r.get("raw_availability") == "available"
                        and not (prod / r["transect"] / "status.json").exists()]
    missing_l1a = [r["transect"] for r in queue if r.get("raw_availability") != "available"]
    band_remaining = collections.Counter(r.get("priority_band") for r in queued_available)
    report["queue"] = {
        "rows_total": len(queue),
        "aux_sub_transect_rows": sum("_" in r["transect"] for r in queue),
        "processing_status_counts": dict(sorted(status.items())),
        "processed": len(processed),
        "queued_available": len(queued_available),
        "failed_missing_l1a": missing_l1a,
        "deliverable_denominator": len(queue) - len(missing_l1a),
        "remaining_by_band": {str(k): v for k, v in sorted(band_remaining.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "first_remaining_priority": min((r["priority"] for r in queued_available), default=None),
        "last_remaining_priority": max((r["priority"] for r in queued_available), default=None),
    }

    # ----------------------------------------------------------- throughput
    times, runs = [], []
    for path in sorted(glob.glob(str(root / "checkpoint/queue_runner*.json"))):
        try:
            state = json.loads(Path(path).read_text())
        except Exception:
            continue
        for item in state.get("attempted", []):
            if item.get("status") == "screens_complete" and (item.get("elapsed_seconds") or 0) > 0:
                times.append(float(item["elapsed_seconds"]))
        if state.get("status") == "batch_complete" and state.get("free_gb") is not None:
            runs.append({"file": os.path.basename(path), "started": state.get("started_utc"),
                         "free_gb": state["free_gb"],
                         "remaining_queued_available": state.get("remaining_queued_available"),
                         "batch_elapsed_s": state.get("elapsed_seconds"),
                         "n_attempted": len(state.get("attempted", []))})
    times.sort()
    mean_s = statistics.mean(times) if times else float("nan")
    p90_s = times[int(0.9 * (len(times) - 1))] if times else float("nan")
    median_s = statistics.median(times) if times else float("nan")
    remaining = len(queued_available)
    eta_mean = remaining * mean_s / 60.0
    eta_p90 = remaining * p90_s / 60.0
    # batch overhead: wall time of a run vs sum of its item times was <= item sum, so
    # item elapsed_seconds already includes download+process+judge+compact.
    report["throughput"] = {
        "timed_successful_transects": len(times),
        "seconds_mean": round(mean_s, 1), "seconds_median": round(median_s, 1),
        "seconds_p90": round(p90_s, 1), "seconds_max": round(times[-1], 1) if times else None,
        "measured_runner_wall_hours_total": round(sum(times) / 3600, 2),
        "eta_minutes_mean": round(eta_mean, 1), "eta_minutes_p90": round(eta_p90, 1),
        "transects_per_hour_mean": round(3600 / mean_s, 2) if times else None,
    }

    # ---------------------------------------------------------------- disk
    free_now = shutil.disk_usage(root).free
    growth = None
    if len(runs) >= 2:
        first, last = runs[0], runs[-1]
        if first["remaining_queued_available"] and last["remaining_queued_available"]:
            consumed = first["remaining_queued_available"] - last["remaining_queued_available"]
            if consumed > 0:
                growth = GIB * (first["free_gb"] - last["free_gb"]) / consumed
    if growth is None:
        growth = 7.0 * MB
    # per-run growth spread: consecutive completed runs, GiB freed / transects consumed
    steps = []
    for a, b in zip(runs, runs[1:]):
        if not (a["remaining_queued_available"] and b["remaining_queued_available"]):
            continue
        consumed = a["remaining_queued_available"] - b["remaining_queued_available"]
        if consumed > 0:
            steps.append(GIB * (a["free_gb"] - b["free_gb"]) / consumed)
    growth_sd = statistics.pstdev(steps) if len(steps) > 1 else 0.0
    growth_max = max(steps) if steps else growth
    projected_free = free_now - remaining * growth
    projected_free_low = free_now - remaining * (growth + growth_sd)
    projected_free_high = free_now - remaining * max(0.0, growth - growth_sd)

    # ------------------------------------------------- derived product sizes
    dtm_sizes, tiles_bytes, rasters_dirs, prod_bytes = [], 0, 0, 0
    for entry in sorted(os.listdir(prod)):
        base = prod / entry
        if not base.is_dir():
            continue
        dtm = base / "dtm.npz"
        if dtm.exists():
            dtm_sizes.append(dtm.stat().st_size)
        if (base / "rasters.npz").exists():
            rasters_dirs += 1
        for dirpath, _, files in os.walk(base):
            for name in files:
                try:
                    size = os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    continue
                prod_bytes += size
                if os.path.basename(dirpath) == "tiles" and name.endswith(".png"):
                    tiles_bytes += size
    dtm_mean = statistics.mean(dtm_sizes) if dtm_sizes else 0.0
    dtm_max = max(dtm_sizes) if dtm_sizes else 0.0

    # ------------------------------------------------------------- archive
    archive = root / "results/amazon_earthworks_results.zip"
    shipped = {}
    zip_bytes = zip_members = 0
    dtm_uncompressed = dtm_compressed = other_uncompressed = other_compressed = 0
    if archive.exists():
        zip_bytes = archive.stat().st_size
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            zip_members = len(infos)
            for info in infos:
                if info.filename.startswith("dtm/"):
                    dtm_uncompressed += info.file_size
                    dtm_compressed += info.compress_size
                else:
                    other_uncompressed += info.file_size
                    other_compressed += info.compress_size
        outcome_path = root / "results/outcome.json"
        if outcome_path.exists():
            shipped = json.loads(outcome_path.read_text())
        mirror = root / "results/amazon_earthworks_results.zip"
        shipped["mirror_hardlinked"] = (mirror.exists() and os.path.samefile(archive, mirror))
        shipped["mirror_link_count"] = archive.stat().st_nlink
    shipped_processed = shipped.get("processed_to_dtm_and_main_screens")
    dtm_ratio = (dtm_compressed / dtm_uncompressed) if dtm_uncompressed else 0.917
    per_dtm_in_zip = dtm_mean * dtm_ratio
    other_per_transect = (other_compressed / shipped_processed) if shipped_processed else 0.0
    projected_zip = (len(queue) - len(missing_l1a)) * per_dtm_in_zip + len(queue) * other_per_transect
    projected_zip_worst = (len(queue) - len(missing_l1a)) * dtm_max * dtm_ratio + len(queue) * other_per_transect

    # transient peak of compile_results.py --zip, as coded:
    #   results/dtm copies (each dtm.npz copied) + old archive inode (hardlinked once)
    #   + new zip written as .zip.part, and the second path is written while the
    #   first new zip already exists.
    final_dtm_total = (len(queue) - len(missing_l1a)) * dtm_mean
    old_zip = zip_bytes
    new_zip = projected_zip
    now_zip = len(processed) * per_dtm_in_zip + len(processed) * other_per_transect
    now_dtm_total = len(processed) * dtm_mean
    # compile_results.py --zip copies every dtm.npz into results/dtm, then writes
    # each archive path as .zip.part and replaces it.  With both paths kept, the
    # second .part coexists with the first finished archive and with the old one.
    peaks = {
        # production today hardlinks each DTM (adopted from m8), so the only
        # remaining duplication is the two archive writes.
        "P1_as_coded_today": old_zip + 2 * new_zip,
        "P2_single_archive_path": old_zip + new_zip,
        "P3_single_path_and_old_archive_removed": new_zip,
        "P0_if_dtm_copies_had_been_kept": final_dtm_total + old_zip + 2 * new_zip,
        "P_now_as_coded_at_current_processed": old_zip + 2 * now_zip,
    }
    # recommended runner floor: enough room for the no-deletion compile plus
    # 0.10 GiB of slack for page cache, .part metadata and the CSV writes.
    COMPILE_SLACK_GIB = 0.05
    peak_keep_old = peaks["P2_single_archive_path"]      # keep the old archive for rollback
    peak_no_old = peaks["P3_single_path_and_old_archive_removed"]
    recommended_floor = math.ceil((peak_keep_old / GIB + COMPILE_SLACK_GIB) * 20) / 20
    floor = float(floor_gib) if floor_gib else recommended_floor
    floor_bytes = floor * GIB
    peak_at_floor = peak_keep_old
    transects_at_floor = int((free_now - floor_bytes) // growth) if growth > 0 else None
    transects_at_floor_1sd = (int((free_now - floor_bytes) // (growth + growth_sd))
                              if growth > 0 else None)
    completeness_gap = remaining - transects_at_floor if transects_at_floor is not None else None
    # measured maximum member size in the remaining queue: the largest single
    # transient the runner needs while a batch is running.
    max_member = max((r["source"]["size"] for r in queued_available if r.get("source")), default=0)
    report["disk"] = {
        "free_bytes_now": free_now,
        "free_gib_now": round(free_now / GIB, 3),
        "derived_v2_total_gib": round(prod_bytes / GIB, 3),
        "derived_v2_tiles_png_gib": round(tiles_bytes / GIB, 3),
        "dirs_with_rasters_npz_retained": rasters_dirs,
        "net_growth_mb_per_transect_measured": round(growth / MB, 2),
        "net_growth_mb_per_transect_sd": round(growth_sd / MB, 2),
        "net_growth_mb_per_transect_max_step": round(growth_max / MB, 2),
        "net_growth_steps_measured": len(steps),
        "net_growth_steps_with_free_increase": sum(1 for x in steps if x < 0),
        "projected_free_after_queue_gib_sd_low": round(projected_free_low / GIB, 3),
        "projected_free_after_queue_gib_sd_high": round(projected_free_high / GIB, 3),
        "growth_source_runs": [r["file"] for r in runs[:1]] + [r["file"] for r in runs[-1:]],
        "projected_free_after_queue_gib": round(projected_free / GIB, 3),
        "remaining_transects": remaining,
        "runner_default_min_free_gib": RUNNER_DEFAULT_MIN_FREE,
        "transects_until_default_runner_floor": (
            int((free_now - RUNNER_DEFAULT_MIN_FREE * GIB) // growth) if growth > 0 else None),
        "runner_floor_evaluated_gib": floor,
        "recommended_runner_floor_gib": recommended_floor,
        "compile_peak_keep_old_gib": round(peak_keep_old / GIB, 3),
        "compile_peak_no_old_gib": round(peak_no_old / GIB, 3),
        "transects_processable_before_floor": transects_at_floor,
        "completeness_gap_at_floor": completeness_gap,
        "max_remaining_member_bytes": max_member,
        "compile_slack_gib_at_recommended_floor": round(recommended_floor - peak_at_floor / GIB, 3),
        "queue_slack_gib_at_recommended_floor": round((projected_free - recommended_floor * GIB) / GIB, 3),
        "transects_processable_at_measured_growth_to_floor": (
            int((free_now - recommended_floor * GIB) // growth) if growth > 0 else None),
        "transects_processable_at_mean_plus_1sd_to_floor": (
            int((free_now - recommended_floor * GIB) // (growth + growth_sd)) if growth > 0 else None),
    }
    report["archive"] = {
        "zip_bytes_now": zip_bytes, "zip_members_now": zip_members,
        "zip_mib_now": round(zip_bytes / 2 ** 20, 1),
        "dtm_members_now": len(dtm_sizes),
        "dtm_compression_ratio": round(dtm_ratio, 4),
        "per_transect_zip_bytes": round(per_dtm_in_zip, 1),
        "projected_zip_bytes_full_queue": int(projected_zip),
        "projected_zip_bytes_worst_case_full_queue": int(projected_zip_worst),
        "target_limit_bytes": ARCHIVE_LIMIT_TARGET,
        "headroom_factor": round(ARCHIVE_LIMIT_TARGET / projected_zip, 2) if projected_zip else None,
        "shipped_processed": shipped_processed,
        "staleness_transects": (len(processed) - shipped_processed) if shipped_processed is not None else None,
        "compile_peak_bytes_needed": {k: int(v) for k, v in peaks.items()},
        "compile_peak_gib_needed": {k: round(v / GIB, 3) for k, v in peaks.items()},
    }

    # ------------------------------------- offline raw-availability recheck
    availability = {"inventory_present": False}
    inventory_path = root / "data/metadata/l1a_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        members = {row["name"] for row in inventory}
        by_tid = collections.defaultdict(list)
        for row in inventory:
            match = re.search(r"NP_T-([0-9]{4}(?:_[0-9]+)?)\.laz", row["name"])
            if match:
                by_tid[match.group(1)].append(row["name"])
        all_tids = [r["transect"] for r in queue]
        unresolved = [t for t in all_tids if t not in by_tid]
        queued_unresolved = [r["transect"] for r in queued_available if r["transect"] not in by_tid]
        member_mismatch = [{"transect": r["transect"], "declared": r["source"]["member"]}
                           for r in queued_available if r["source"]["member"] not in members]
        availability = {
            "inventory_present": True,
            "inventory_members": len(inventory),
            "distinct_transects_in_inventory": len(by_tid),
            "queue_tids_absent_from_inventory": unresolved,
            "queued_unresolved": queued_unresolved,
            "queued_declared_member_missing": member_mismatch,
            "transects_with_multiple_inventory_copies": sorted(t for t, v in by_tid.items() if len(v) > 1),
            "ok": not queued_unresolved and not member_mismatch,
        }
    report["raw_availability_recheck"] = availability

    # ------------------------------------------------ coordinator review state
    # A feature that the automatics did not reject has to be resolved by a
    # recorded coordinator decision and/or a built candidate pack.  Counting raw
    # "automatic certainty != Rejected" as pending review is wrong: it ignores
    # checkpoint/review_decisions.json and the candidate packs.
    decisions = {}
    decision_path = root / "checkpoint/review_decisions.json"
    if decision_path.exists():
        decisions = {row["id"]: row for row in json.loads(decision_path.read_text())}
    resolved = {"ring": collections.Counter(), "mound": collections.Counter(), "line": collections.Counter()}
    pending = []
    bookkeeping = collections.Counter()

    def classify(kind, tid, fid, automatic, native_done=False):
        """Return 'Known'/'Possible'/'Rejected' when resolved, None when pending."""
        decision = decisions.get(fid, {}).get("decision")
        pack = (prod / tid / "candidates" / fid / "candidate.json").exists()
        if fid == "R0414-001" or decision == "Known":
            resolved[kind]["Known"] += 1
            return "Known"
        if pack or decision == "Possible":
            resolved[kind]["Possible"] += 1
            return "Possible"
        if decision == "Rejected":
            resolved[kind]["Rejected"] += 1
            if automatic != "Rejected":
                bookkeeping[f"{kind}_native_rejected_decision_recorded"] += 1
            return "Rejected"
        if automatic == "Rejected" and not native_done:
            return "Rejected"            # automatics suffice; no review was owed
        pending.append({"kind": kind, "transect": tid, "id": fid,
                        "automatic_certainty": automatic,
                        "has_decision": bool(decision), "has_pack": pack})
        return None

    for entry in sorted(os.listdir(prod)):
        base = prod / entry
        if not base.is_dir():
            continue
        rings_path = base / "skeptic/rings_v3.json"
        if rings_path.exists():
            for feature in json.loads(rings_path.read_text()).get("features", []):
                classify("ring", entry, feature.get("id"),
                         feature.get("verdict", {}).get("certainty"))
        for path in sorted((base / "mound_review").glob("M*.json")):
            review = json.loads(path.read_text())
            classify("mound", entry, path.stem, review.get("certainty"))
        for path in sorted((base / f"causeway/{entry}/evidence").glob("*/skeptic_line_verdict.json")):
            verdict = json.loads(path.read_text())
            fid = path.parent.name
            resolved_line = classify("line", entry, fid, verdict.get("certainty"),
                                     native_done=bool(verdict.get("native_visual_review")))
            decision = decisions.get(fid, {})
            if resolved_line == "Rejected" and verdict.get("certainty") != "Rejected" \
                    and decision.get("decision") != "Rejected":
                bookkeeping["line_automatically_rejected_without_decision"] += 1
    report["review_state"] = {
        "resolved": {kind: dict(counter) for kind, counter in resolved.items()},
        "resolved_totals": {kind: sum(counter.values()) for kind, counter in resolved.items()},
        "pending_total": len(pending),
        "pending": pending[:25],
        "bookkeeping": dict(bookkeeping),
        "decisions_file_entries": len(decisions),
        "note": ("Known/Possible/Rejected counts are resolved: each has a recorded final "
                 "decision and/or a built candidate pack (R0414-001 is the Known control). "
                 "Only 'pending' would block native review."),
    }
    # ------------------------------------------------------------- verdicts
    gates = []

    def gate(name, ok, detail):
        gates.append({"gate": name, "pass": bool(ok), "detail": detail})

    gate("queue_fits_time_budget", eta_mean <= budget_minutes,
         f"ETA {eta_mean:.0f} min at measured mean vs budget {budget_minutes:.0f} min "
         f"(p90 {eta_p90:.0f} min)")
    gate("default_runner_floor_does_not_block",
         report["disk"]["transects_until_default_runner_floor"] is None
         or report["disk"]["transects_until_default_runner_floor"] >= remaining,
         f"with --min-free-gb {RUNNER_DEFAULT_MIN_FREE} the runner would stop after "
         f"{report['disk']['transects_until_default_runner_floor']} of {remaining} transects")
    gate("compile_fits_at_floor", peak_at_floor <= floor_bytes,
         f"keeping the old archive needs {peak_at_floor / GIB:.3f} GiB vs floor {floor:.2f} GiB "
         f"(slack {floor - peak_at_floor / GIB:.3f} GiB); removing it after recording its SHA-256 "
         f"would need {peak_no_old / GIB:.3f} GiB")
    gate("queue_completes_at_floor", projected_free >= floor_bytes,
         f"projected free after {remaining} transects {projected_free / GIB:.3f} GiB vs floor "
         f"{floor:.2f} GiB (slack {(projected_free - floor_bytes) / GIB:.3f} GiB); "
         f"{transects_at_floor} of {remaining} transects fit before the floor stops the runner")
    gate("archive_under_target_even_worst_case", projected_zip_worst <= ARCHIVE_LIMIT_TARGET,
         f"worst-case projected {projected_zip_worst / GB:.2f} GB vs target {ARCHIVE_LIMIT_TARGET / GB:.0f} GB")
    gate("archive_current_with_live_queue",
         shipped_processed is not None and shipped_processed >= len(processed),
         f"shipped archive covers {shipped_processed} processed transects, live queue has "
         f"{len(processed)} (stale by {(len(processed) - (shipped_processed or 0))})")
    gate("remaining_transects_resolvable_offline",
         not availability.get("inventory_present") or availability["ok"],
         "every queued transect has an inventory member and the declared member name matches"
         if availability.get("ok") else
         f"queued unresolved {availability.get('queued_unresolved')} / "
         f"member mismatch {len(availability.get('queued_declared_member_missing', []))}")
    gate("no_pending_coordinator_reviews", len(pending) == 0,
         f"{len(pending)} feature(s) with no recorded decision and no candidate pack"
         + ("" if not pending else ": " + ", ".join(f"{p['id']}@{p['transect']}" for p in pending[:8])))
    report["gates"] = gates
    report["all_gates_pass"] = all(g["pass"] for g in gates)
    return report


def render(report: dict) -> str:
    qs, th, dk, ar = (report["queue"], report["throughput"], report["disk"],
                      report["archive"])
    lines = []
    add = lines.append
    add("EBA main-screen completion audit (read-only)")
    add("=" * 60)
    add(f"queue rows {qs['rows_total']} ({qs['aux_sub_transect_rows']} auxiliary sub-transect ids)")
    add(f"  processed (summary+status, non-failed) : {qs['processed']}")
    add(f"  queued + raw available                 : {qs['queued_available']}  {qs['remaining_by_band']}")
    add(f"  unavailable at source (L1A)            : {len(qs['failed_missing_l1a'])} {qs['failed_missing_l1a']}")
    add(f"  honest deliverable denominator          : {qs['deliverable_denominator']}")
    add("")
    add(f"throughput from {th['timed_successful_transects']} timed transects "
        f"({th['measured_runner_wall_hours_total']} h measured): "
        f"mean {q(th['seconds_mean'])} median {q(th['seconds_median'])} p90 {q(th['seconds_p90'])} "
        f"max {q(th['seconds_max'])}")
    add(f"  ETA remaining queue: {th['eta_minutes_mean']:.0f} min (mean) / {th['eta_minutes_p90']:.0f} min (p90); "
        f"{th['transects_per_hour_mean']} transects/hour")
    add("")
    add(f"disk free now {dk['free_gib_now']:.2f} GiB; derived_v2 {dk['derived_v2_total_gib']:.2f} GiB "
        f"(tiles.png {dk['derived_v2_tiles_png_gib']:.2f} GiB); {dk['dirs_with_rasters_npz_retained']} dirs keep rasters")
    add(f"  measured net growth {dk['net_growth_mb_per_transect_measured']} "
        f"+/- {dk['net_growth_mb_per_transect_sd']} MB/transect over "
        f"{dk['net_growth_steps_measured']} run pairs "
        f"-> free after queue {dk['projected_free_after_queue_gib']:.2f} GiB "
        f"(1 sd range {dk['projected_free_after_queue_gib_sd_low']:.2f}"
        f"..{dk['projected_free_after_queue_gib_sd_high']:.2f})")
    add(f"  runner default --min-free-gb {dk['runner_default_min_free_gib']}: would stop after "
        f"{dk['transects_until_default_runner_floor']} transects")
    add("")
    add(f"archive now {ar['zip_mib_now']} MiB, {ar['zip_members_now']} members, "
        f"dtm compression {ar['dtm_compression_ratio']}, {ar['per_transect_zip_bytes']} B/transect")
    add(f"  projected full-queue archive {ar['projected_zip_bytes_full_queue'] / GB:.2f} GB "
        f"(worst case {ar['projected_zip_bytes_worst_case_full_queue'] / GB:.2f} GB) vs 5 GB target, "
        f"headroom x{ar['headroom_factor']}")
    add(f"  shipped archive covers {ar['shipped_processed']} transects; live queue has {qs['processed']} "
        f"(stale by {ar['staleness_transects']})")
    add("  compile --zip transient peak needed:")
    for key, value in ar["compile_peak_gib_needed"].items():
        add(f"    {key:34s} {value:6.2f} GiB")
    add("")
    rs = report["review_state"]
    add(f"coordinator review state: resolved {rs['resolved_totals']} "
        f"{ {k: v for k, v in rs['resolved'].items() if v} }; pending {rs['pending_total']}; "
        f"bookkeeping {rs['bookkeeping']}")
    add(f"  floor evaluated {dk['runner_floor_evaluated_gib']:.2f} GiB "
        f"(compile needs {dk['compile_peak_keep_old_gib']:.3f} GiB keeping the old archive, "
        f"{dk['compile_peak_no_old_gib']:.3f} GiB without it)")
    add(f"  at that floor {dk['transects_processable_before_floor']} of {dk['remaining_transects']} "
        f"transects fit (gap {dk['completeness_gap_at_floor']}); largest remaining member "
        f"{dk['max_remaining_member_bytes'] / 1e6:.0f} MB")
    add(f"  sensitivity: at mean+1 sd growth ({dk['net_growth_mb_per_transect_measured'] + dk['net_growth_mb_per_transect_sd']:.1f} MB/transect)"
        f" only {dk['transects_processable_at_mean_plus_1sd_to_floor']} transects fit before the floor."
        " The sd is dominated by external free-space events, so the mean is the estimator and the"
        " 1 sd case bounds the risk; it is not a prediction.")
    av = report.get("raw_availability_recheck", {})
    if av.get("inventory_present"):
        add(f"offline raw recheck: {av['inventory_members']} archive members, "
            f"{av['distinct_transects_in_inventory']} distinct transects; "
            f"queue tids absent {av['queue_tids_absent_from_inventory']}; "
            f"queued unresolved {len(av['queued_unresolved'])}; "
            f"declared-member mismatches {len(av['queued_declared_member_missing'])}")
    add("")
    for gate in report["gates"]:
        add(f"[{'PASS' if gate['pass'] else 'FAIL'}] {gate['gate']}: {gate['detail']}")
    add("")
    add(f"ALL GATES PASS: {report['all_gates_pass']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--floor-gib", type=float, default=None,
                    help="evaluate this --min-free-gb instead of the computed recommendation")
    ap.add_argument("--budget-minutes", type=float, default=719.0,
                    help="wall-clock budget for the remaining queue (default 719 = 12 h swarm)")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not (root / "checkpoint/queue.json").exists():
        print(f"not a project root: {root}", file=sys.stderr)
        return 2
    report = audit(root, args.budget_minutes, args.floor_gib)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
