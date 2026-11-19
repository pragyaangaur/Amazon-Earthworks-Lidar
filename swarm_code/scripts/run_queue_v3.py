#!/usr/bin/env python3
"""Run a bounded number of queued EBA transects through the current v3 chain.

The runner is deliberately sequential: one remote LAZ and one processor at a
time.  It does not upgrade ring or mound detections.  Raised-linear detections
receive the already validated independent profile verdict; a Possible line gets
its evidence pack, while ring/mound detections remain coordinator attention
items with full native products retained.  Fully Rejected/empty transects are
compacted only by compact_derived_v2.py's separate eligibility checks.

Example (coordinator only; downloads public raw data but never publishes):
  .venv/bin/python code/run_queue_v3.py --limit 25 --min-free-gb 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
QUEUE = ROOT / "checkpoint/queue.json"
PROD = ROOT / "data/derived_v2"
LOG = ROOT / "checkpoint/queue_runner.json"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run([str(x) for x in command], cwd=ROOT,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log_path.read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"command exited {result.returncode}: {command[1]}; tail: " + " | ".join(tail))


def load_queue() -> list[dict]:
    return json.loads(QUEUE.read_text())


def attention(tid: str) -> list[dict]:
    """List unresolved ring/mound/line items without changing their certainty."""
    base = PROD / tid
    items = []
    decision_path = ROOT / "checkpoint/review_decisions.json"
    decisions = {row["id"]: row for row in json.loads(decision_path.read_text())} if decision_path.exists() else {}
    rings_path = base / "skeptic/rings_v3.json"
    if rings_path.exists():
        for feature in json.loads(rings_path.read_text())["features"]:
            verdict = feature.get("verdict", {})
            fid = feature["id"]
            decision = decisions.get(fid, {})
            candidate_path = base / f"candidates/{fid}/candidate.json"
            native = verdict.get("native_visual_review") or decision.get("native_visual_review")
            if candidate_path.exists() and not native:
                native = json.loads(candidate_path.read_text()).get("native_visual_review")
            resolved_possible = (decision.get("decision") == "Possible"
                                 and native and candidate_path.exists())
            # The compiler's sole Known control lives in control/0414, not in
            # the generic Possible-candidate evidence pack.
            if verdict.get("certainty") != "Rejected" and fid != "R0414-001" and not resolved_possible:
                items.append({"id": fid, "kind": "ring",
                              "automatic_certainty": verdict.get("certainty"),
                              "required": "native coordinator review; do not compile as Rejected by default"})
    detb_path = base / "detb_review/recordcheck/lines_detb_audited.csv"
    if detb_path.exists():
        with detb_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                fid = row["id"]
                if row.get("certainty") == "Rejected":
                    continue
                decision = decisions.get(fid, {})
                if decision.get("decision") == "Rejected":
                    continue
                pack_dir = base / f"candidates/{fid}"
                review_path = pack_dir / "review.json"
                native = decision.get("native_visual_review")
                if review_path.exists() and not native:
                    native = json.loads(review_path.read_text()).get("native_visual_review")
                resolved_possible = (decision.get("decision") == "Possible" and native
                                     and (pack_dir / "candidate.json").exists())
                if not resolved_possible:
                    items.append({"id": fid, "kind": "detb_linear",
                                  "automatic_certainty": row.get("certainty"),
                                  "required": "explicit Possible decision, native-reviewed review.json and candidate evidence; do not ship by pack existence alone"})
    for path in sorted((base / "mound_review").glob("M*.json")):
        review = json.loads(path.read_text())
        decision = decisions.get(path.stem, {})
        resolved_possible = (decision.get("decision") == "Possible"
                             and (base / f"candidates/{path.stem}/candidate.json").exists())
        if review.get("certainty") != "Rejected" and not resolved_possible:
            items.append({"id": path.stem, "kind": "mound",
                          "automatic_certainty": review.get("certainty"),
                          "required": "explicit coordinator Possible decision and evidence pack; do not compile from pack existence alone"})
    for path in sorted((base / f"causeway/{tid}/evidence").glob("*/skeptic_line_verdict.json")):
        verdict = json.loads(path.read_text())
        fid = path.parent.name
        decision = decisions.get(fid, {})
        if (verdict.get("certainty") != "Rejected" and not verdict.get("native_visual_review")
                and decision.get("decision") not in {"Possible", "Rejected"}):
            items.append({"id": fid, "kind": "raised_linear",
                          "automatic_certainty": verdict.get("certainty"),
                          "required": "native coordinator review; compiler refuses Possible without it"})
        if decision.get("decision") == "Rejected" and verdict.get("certainty") != "Rejected":
            items.append({"id": fid, "kind": "raised_linear",
                          "automatic_certainty": verdict.get("certainty"),
                          "required": "persist the existing native Rejected decision into the line verdict before compilation"})
    return items


def judge_lines(tid: str) -> None:
    base = PROD / tid
    evidence = base / f"causeway/{tid}/evidence"
    for path in sorted(evidence.glob("*/evidence.json")):
        fid = path.parent.name
        verdict_path = path.parent / "skeptic_line_verdict.json"
        if not verdict_path.exists():
            run([PYTHON, "code/judge_causeway_candidate.py", base, fid],
                ROOT / f"logs/{tid}_{fid}_judge.log")
        verdict = json.loads(verdict_path.read_text())
        candidate_path = base / f"candidates/{fid}/candidate.json"
        if verdict.get("certainty") == "Possible" and not candidate_path.exists():
            run([PYTHON, "code/build_causeway_candidate.py", base, fid],
                ROOT / f"logs/{tid}_{fid}_candidate.log")
        if verdict.get("certainty") not in {"Rejected", "Possible"}:
            raise RuntimeError(f"{tid} {fid}: unresolved line verdict {verdict.get('certainty')}")


def write_state(state: dict) -> None:
    temporary = LOG.with_suffix(".json.part")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, LOG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    started = time.monotonic()
    state = {"started_utc": utc(), "status": "running", "attempted": [],
             "attention": [], "limit": args.limit, "min_free_gb": args.min_free_gb}
    write_state(state)
    # Only cleanly queued rows are selected.  A v3_in_progress row may be another
    # live coordinator process and is never duplicated by this runner.
    rows = sorted((r for r in load_queue()
                   if r.get("processing_status") == "queued" and r.get("raw_availability") == "available"
                   and not (PROD / r["transect"] / "status.json").exists()),
                  key=lambda r: r["priority"])
    for row in rows[:args.limit]:
        tid = row["transect"]
        free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
        if free_gb < args.min_free_gb:
            state.update(status="disk_threshold_reached", free_gb=free_gb, next_transect=tid)
            write_state(state)
            print(json.dumps(state), flush=True)
            return 3
        source = row["source"]
        item_started = time.monotonic()
        item = {"transect": tid, "priority": row["priority"], "started_utc": utc()}
        state["attempted"].append(item)
        state["current_transect"] = tid
        write_state(state)
        try:
            run(["/usr/bin/time", "-v", PYTHON, "code/process_one_v3.py", tid,
                 row["state_label"], "--record-id", source["record_id"],
                 "--zip-key", source["zip_key"], "--member", source["member"],
                 "--state", row["state"]], ROOT / f"logs/{tid}_queue_v3.log")
            judge_lines(tid)
            new_attention = attention(tid)
            state["attention"].extend(new_attention)
            # This is a no-op retaining full products whenever any candidate or
            # unresolved verdict exists.  Its own manifest records freed bytes.
            run([PYTHON, "code/compact_derived_v2.py", tid, "--drop-screen-state"],
                ROOT / f"logs/{tid}_storage.log")
            item.update(status="screens_complete", attention=[x["id"] for x in new_attention])
        except Exception as exc:
            item.update(status="failed", error=repr(exc))
            with (ROOT / "logs/failures.jsonl").open("a") as handle:
                handle.write(json.dumps({"time_utc": utc(), "stage": "run_queue_v3",
                                         "transect": tid, "failure": repr(exc)}, ensure_ascii=False) + "\n")
        item["elapsed_seconds"] = round(time.monotonic() - item_started, 1)
        item["finished_utc"] = utc()
        state.pop("current_transect", None)
        state["free_gb"] = shutil.disk_usage(ROOT).free / (1024 ** 3)
        write_state(state)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    failures = sum(x["status"] == "failed" for x in state["attempted"])
    remaining = sum(r.get("processing_status") == "queued" and r.get("raw_availability") == "available"
                    for r in load_queue())
    state.update(status="batch_complete", finished_utc=utc(), failures=failures,
                 remaining_queued_available=remaining,
                 elapsed_seconds=round(time.monotonic() - started, 1))
    write_state(state)
    print("SUMMARY " + json.dumps(state, ensure_ascii=False), flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
