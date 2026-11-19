"""Record-check every detection list under a directory tree, with an index.

The swarm produces detector snapshots per transect (`data/derived/<t>/detections/
rings*.csv`).  This walks them, runs the same audit as `recordcheck.audit` for
each, writes one table + summary per list into the output directory, and prints a
compact verdict table.  An index JSON records the SHA-256 of every input list and
its verdict counts, so a re-run shows what changed and nothing is silently
re-checked from a different file.

Usage:
    PYTHONPATH=<this dir> python3 -m recordcheck.check_dir --root data/derived \
        --known data/known/peripato/Earthworks.rds --out-dir out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

from . import audit as audit_mod
from .known_table import load_known_table

SNAPSHOT_HINTS = ("rings", "ring_", "circles", "mounds", "lines", "causeway", "linear", "roads")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def looks_like_detections(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    name = path.name.lower()
    if not any(h in name for h in SNAPSHOT_HINTS):
        return False
    if "audit" in name:
        return False
    try:
        head = path.open("r", encoding="utf-8", errors="replace").readline().lower()
    except OSError:
        return False
    return "lon" in head and "lat" in head


def stem_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    transect = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    name = re.sub(r"[^0-9A-Za-z]+", "_", path.stem)
    return f"det_{transect}_{name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="record-check every detection list under a directory")
    ap.add_argument("--root", default="data/derived")
    ap.add_argument("--known", default="data/known/peripato/Earthworks.rds")
    ap.add_argument("--known-csv", default=None)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--index", default=None, help="default: <out-dir>/recordcheck_index.json")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index) if args.index else out_dir / "recordcheck_index.json"

    if args.known_csv:
        known = audit_mod.load_known_table_from_csv(args.known_csv)
    else:
        known = load_known_table(args.known)
    print(f"known table: {len(known.df)} records via {known.backend}")

    lists = sorted(p for p in root.rglob("*.csv") if looks_like_detections(p))
    if not lists:
        print(f"no detection lists found under {root}")
        return 0

    index = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    rows = []
    for path in lists:
        stem = stem_for(path, root)
        digest = sha256(path)
        try:
            cands = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {path}: unreadable ({exc})")
            continue
        if len(cands) == 0:
            print(f"SKIP {path}: no rows")
            continue
        result = audit_mod.run_audit(cands, known, audit_mod.AuditConfig())
        paths = audit_mod.write_outputs(result, out_dir, stem)
        counts = result["summary"]["counts_by_table_match"]
        known_ids = result["table"].loc[
            result["table"]["d_table_match"] == "known_table", "id"].astype(str).tolist()
        nearest = pd.to_numeric(result["table"]["nearest_known_m"], errors="coerce")
        row = {
            "transect": path.parent.parent.name if path.parent.name == "detections" else path.parent.name,
            "list": path.name,
            "input_sha256": digest,
            "n": int(len(cands)),
            "known_table": int(counts.get("known_table", 0)),
            "no_table_match": int(counts.get("no_table_match", 0)),
            "undetermined": int(counts.get("undetermined", 0)),
            "known_ids": known_ids,
            "nearest_known_m_min": None if nearest.dropna().empty else float(nearest.min()),
            "nearest_known_m_max": None if nearest.dropna().empty else float(nearest.max()),
            "flags": result["summary"]["flag_counts"],
            "audited_csv": paths["csv"],
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rows.append(row)
        index[stem] = row

    # ---- merged, de-duplicated view across every list -------------------
    merged_rows = []
    for row in rows:
        audited = pd.read_csv(row["audited_csv"])
        if len(audited) == 0:
            continue
        for col in ("radius_m", "size_m", "diameter_m", "score", "snr"):
            if col not in audited.columns:
                audited[col] = None
        audited = audited.assign(source_list=row["list"], source_transect=row["transect"],
                                 source_sha256=row["input_sha256"][:16])
        keep = ["source_transect", "source_list", "id", "EPSG", "x", "y", "lon", "lat",
                "radius_m", "size_m", "diameter_m", "score", "snr", "nearest_known_db",
                "nearest_known_m", "d_table_match", "d_match_record_id", "d_match_distance_m",
                "d_match_threshold_m", "d_match_margin_m", "d_nearest_ratio",
                "d_certainty_ceiling_from_table", "d_flags"]
        merged_rows.append(audited[[c for c in keep if c in audited.columns]])
    if merged_rows:
        all_det = pd.concat(merged_rows, ignore_index=True)
        # a feature reported in several snapshots has (almost) the same centre;
        # match by position, never by detector id (ids are unstable between runs)
        key = (all_det["source_transect"].astype(str) + "|"
               + all_det["lat"].round(5).astype(str) + "|" + all_det["lon"].round(5).astype(str))
        all_det["position_key"] = key
        all_det["n_lists_with_same_position"] = all_det.groupby("position_key")["source_list"].transform("nunique")
        counts = all_det["position_key"].value_counts()
        all_det["duplicate_position_across_lists"] = all_det["position_key"].map(counts) > 1
        merged_path = out_dir / "all_detections_recordcheck.csv"
        all_det.to_csv(merged_path, index=False)
        print(f"merged view: {merged_path} ({len(all_det)} rows, "
              f"{all_det['position_key'].nunique()} distinct positions, "
              f"{int(all_det['duplicate_position_across_lists'].sum())} rows share a position with "
              "another list)")

    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    table = pd.DataFrame(rows)
    if len(table):
        show = table[["transect", "list", "n", "known_table", "no_table_match", "undetermined",
                      "nearest_known_m_min", "nearest_known_m_max"]]
        print(show.to_string(index=False))
        print(f"\n{len(table)} detection lists checked; index: {index_path}")
        total_known = int(table["known_table"].sum())
        total_unmatched = int(table["no_table_match"].sum())
        print(f"totals: {total_known} known_table rows, {total_unmatched} no_table_match rows "
              "(a table miss is not a novelty claim)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
