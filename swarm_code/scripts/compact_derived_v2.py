#!/usr/bin/env python3
"""Compact fully reviewed v3 transects without losing the audit trail.

This is a storage-lifecycle step, not a detector or a certainty decision.  Only
transects whose ring, mound and raised-linear screens are all Rejected (or
empty) are eligible.  The positive-control and independent forest-audit
transects, and every transect with a candidate evidence directory, retain their
full products.  Eligible transects keep the DTM, review images, all CSV/JSON
records, and a validated ``screen_state.npz`` containing the arrays needed to
reconstruct the detector mask and inspect the 15 m relief.  Bulky intermediate
arrays are removed only after the compact archive is reopened and checked.

With ``--drop-screen-state``, a second lifecycle step then removes the compact
array cache too.  It requires the same eligibility checks plus a valid DTM, and
writes storage_slimming.json with the cache hash and restoration instructions.
The exact DTM, tiling/inspection products, audits and source provenance remain.
Re-expanded v3 products can be reproduced deliberately from the public raw LAZ.
Do not use that flag on transects whose masks/CHM are still needed by a live
method audit.  It never compact-controls or candidates.

Run ``.venv/bin/python code/compact_derived_v2.py --all --dry-run`` first.
Raw L1A provenance remains in raw_source.json, so a full-resolution re-review
can deliberately fetch the public source again; no source is contacted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "data/derived_v2"
# These are methodological reference products used by the accepted independent
# raster audit and detectability work.  They are never compacted automatically.
KEEP_FULL = {"0414", "0072", "0470", "0476", "0126", "0621", "0333"}
MASK_NAMES = ("data_hole", "big_hole", "removed", "filled", "canopy_mask",
              "forest_mask", "detect_mask", "outlier")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_review_decisions() -> dict:
    path = ROOT / "checkpoint/review_decisions.json"
    if not path.exists():
        return {}
    return {row["id"]: row for row in json.loads(path.read_text())}


def ineligible_reason(tid: str, decisions: dict) -> str | None:
    base = PROD / tid
    status_path = base / "status.json"
    if not status_path.exists():
        return "no status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "v3_screens_complete_pending_coordinator_review":
        return "screens not complete: " + str(status.get("status"))
    if tid in KEEP_FULL:
        return "methodological reference product retained in full"
    if any((base / "candidates").glob("*/candidate.json")):
        return "candidate evidence present; full native products retained"
    rings = json.loads((base / "skeptic/rings_v3.json").read_text())["features"]
    for feature in rings:
        fid = feature["id"]
        certainty = feature.get("verdict", {}).get("certainty")
        if certainty != "Rejected" and decisions.get(fid, {}).get("decision") != "Rejected":
            return f"ring {fid} is not explicitly Rejected ({certainty})"
    for path in (base / "mound_review").glob("M*.json"):
        review = json.loads(path.read_text())
        if review.get("certainty") != "Rejected":
            return f"mound {path.stem} is not Rejected"
    for path in (base / f"causeway/{tid}/evidence").glob("*/skeptic_line_verdict.json"):
        verdict = json.loads(path.read_text())
        if verdict.get("certainty") != "Rejected":
            return f"raised-linear {path.parent.name} is not Rejected"
    # A causeway row without an independent verdict is unresolved, even if its
    # verdict file was not created because an earlier manual step was skipped.
    summary = json.loads((base / f"causeway/{tid}/summary.json").read_text())
    verdicts = list((base / f"causeway/{tid}/evidence").glob("*/skeptic_line_verdict.json"))
    if int(summary.get("n_candidates", 0)) != len(verdicts):
        return "one or more raised-linear detections lack an independent verdict"
    return None


def drop_reviewed_cache(tid: str) -> dict:
    """Remove an already validated negative-screen audit cache, with provenance.

    This does not run unless the caller has already established eligibility via
    ineligible_reason(): every detection is rejected/empty and no candidate pack
    or reference-product exclusion exists.  The exact DTM and all public-source
    provenance stay on disk.  A future audit must deliberately refetch the raw
    member and rerun the v3 processor rather than assuming masks/CHM are cached.
    """
    base = PROD / tid
    target = base / "screen_state.npz"
    dtm_path = base / "dtm.npz"
    summary = json.loads((base / "summary.json").read_text())
    with np.load(dtm_path) as dtm_file:
        dtm = dtm_file["dtm"]
        assert dtm.shape == (summary["grid"]["ny"], summary["grid"]["nx"])
        assert np.isfinite(dtm).sum() > 1000
    removed_bytes = target.stat().st_size
    cache_sha = sha256(target)
    result = {"transect": tid, "status": "slimmed_reviewed_cache",
              "screen_state_sha256": cache_sha, "removed_screen_state_bytes": removed_bytes,
              "grid": summary["grid"], "epsg": summary["epsg"],
              "eligibility": "empty or every ring/mound/raised-linear feature Rejected; no candidate evidence pack and not a reference transect",
              "restoration": "Deliberately refetch raw LAZ using raw_source.json and rerun code/vendor_m2/process.py --dtm-source hybrid. DTMs, tiles and audits retained; screen_state masks/CHM are not asserted to remain cached."}
    (base / "storage_slimming.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    status_path = base / "status.json"
    status = json.loads(status_path.read_text())
    status["screen_state_cache_removed_after_validated_compaction"] = True
    status["screen_state_restoration"] = result["restoration"]
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    target.unlink()
    return result


def compact_one(tid: str, dry_run: bool = False, drop_screen_state: bool = False) -> dict:
    base = PROD / tid
    source_path = base / "rasters.npz"
    products_path = base / "products.npz"
    dtm_path = base / "dtm.npz"
    target = base / "screen_state.npz"
    summary = json.loads((base / "summary.json").read_text())
    shape = (int(summary["grid"]["ny"]), int(summary["grid"]["nx"]))
    slim_manifest = base / "storage_slimming.json"
    if slim_manifest.exists() and not target.exists():
        previous = json.loads(slim_manifest.read_text())
        return {"transect": tid, "status": "already_slimmed",
                "removed_screen_state_bytes": previous.get("removed_screen_state_bytes", 0)}
    if target.exists():
        if drop_screen_state:
            return drop_reviewed_cache(tid)
        return {"transect": tid, "status": "already_compact", "bytes": target.stat().st_size}
    for path in (source_path, products_path, dtm_path):
        if not path.exists():
            raise FileNotFoundError(path)
    source_bytes = sum(p.stat().st_size for p in base.iterdir() if p.is_file())
    with np.load(source_path) as source, np.load(products_path) as product, np.load(dtm_path) as dtm_file:
        dtm = source["dtm"]
        if dtm.shape != shape or dtm_file["dtm"].shape != shape:
            raise ValueError(f"{tid}: DTM/grid shape mismatch")
        if not np.array_equal(dtm, dtm_file["dtm"], equal_nan=True):
            raise ValueError(f"{tid}: standalone DTM differs from native product")
        if np.isfinite(dtm).sum() <= 1000:
            raise ValueError(f"{tid}: too few valid DTM cells")
        chm = source["chm"]
        canopy_mask = np.nan_to_num(chm) > float(summary["canopy"]["threshold_m"])
        forest_mask = ndi.binary_dilation(canopy_mask, iterations=1)
        # ground_pick leaves every big hole as NaN, so finite DTM already
        # excludes big_hole.  This exactly reproduces process.py lines 151-158.
        detect_mask = np.isfinite(dtm) & ~forest_mask
        if int(detect_mask.sum()) != int(summary["detect"]["usable_cells"]):
            raise ValueError(f"{tid}: reconstructed detector-mask count differs from summary")
        arrays = {
            "chm": chm,
            "relief_15": source["relief_15"],
            "data_hole": source["hole"].astype(bool),
            "big_hole": ~np.isfinite(dtm),
            "removed": product["removed"].astype(bool),
            "filled": product["interpolated"].astype(bool),
            "canopy_mask": canopy_mask,
            "forest_mask": forest_mask,
            "detect_mask": detect_mask,
            "unfilled": product["unfilled"].astype(bool),
            "relief_hole_mask": product["relief_hole_mask"].astype(bool),
        }
        if not np.array_equal(arrays["data_hole"], product["original_hole"]):
            raise ValueError(f"{tid}: original-hole mask mismatch")
        if not np.array_equal(arrays["big_hole"], arrays["unfilled"]):
            raise ValueError(f"{tid}: big-hole/unfilled mask mismatch")
        masks_path = base / "masks.npy"
        if masks_path.exists():
            masks = np.load(masks_path, allow_pickle=True).item()
            for name in MASK_NAMES:
                if name == "big_hole":
                    expected = arrays["big_hole"]
                elif name == "filled":
                    expected = arrays["filled"]
                else:
                    expected = arrays.get(name)
                if expected is not None and name in masks and not np.array_equal(expected, masks[name]):
                    raise ValueError(f"{tid}: reconstructed {name} differs from original mask")
                if name in masks and name not in arrays:
                    arrays[name] = masks[name].astype(bool)
        metadata = {
            "transect": tid,
            "schema": "eba.v3.screen_state/1",
            "created_utc": utc(),
            "row_order": "south-up",
            "grid": summary["grid"],
            "epsg": summary["epsg"],
            "detect_mask_rule": "isfinite(dtm) & ~binary_dilation(nan_to_num(chm)>canopy_threshold, iterations=1)",
            "canopy_threshold_m": summary["canopy"]["threshold_m"],
            "source_sha256": {p.name: sha256(p) for p in (source_path, products_path, dtm_path)},
            "source_summary_sha256": sha256(base / "summary.json"),
            "note": "Rejected/empty-screen storage compaction. Full-resolution re-review requires deliberate refetch using raw_source.json; no certainty is changed.",
        }
        if dry_run:
            return {"transect": tid, "status": "eligible", "top_level_source_bytes": source_bytes}
        temporary = target.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        # Validate the actual persisted artifact before deleting any source.
        with np.load(temporary) as check:
            for name, expected in arrays.items():
                if check[name].shape != shape or not np.array_equal(check[name], expected, equal_nan=True):
                    raise ValueError(f"{tid}: compact validation failed for {name}")
        os.replace(temporary, target)
    (base / "screen_state.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    removed = {}
    for name in ("rasters.npz", "products.npz", "diagnostics.npz", "masks.npy"):
        path = base / name
        if path.exists():
            removed[name] = path.stat().st_size
            path.unlink()
    result = {"transect": tid, "status": "compact", "screen_state_bytes": target.stat().st_size,
              "removed_intermediate_bytes": removed,
              "retained": ["dtm.npz", "screen_state.npz", "screen_state.json", "CSV/JSON audits", "review images", "tiles"]}
    (base / "storage_compaction.json").write_text(json.dumps(result, indent=2) + "\n")
    if drop_screen_state:
        return drop_reviewed_cache(tid)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transects", nargs="*")
    parser.add_argument("--all", action="store_true", help="consider every completed v3 status")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--drop-screen-state", action="store_true",
                        help="also remove validated negative-screen caches; DTMs/audits/images/provenance remain")
    args = parser.parse_args()
    if not args.all and not args.transects:
        parser.error("name transects or use --all")
    tids = args.transects
    if args.all:
        tids = sorted(p.parent.name for p in PROD.glob("*/status.json"))
    decisions = load_review_decisions()
    totals = {"compact": 0, "already_compact": 0, "retained_full": 0, "eligible": 0,
              "slimmed_reviewed_cache": 0, "already_slimmed": 0, "removed_intermediate_bytes": 0,
              "removed_screen_state_bytes": 0}
    for tid in tids:
        reason = ineligible_reason(tid, decisions)
        if reason:
            totals["retained_full"] += 1
            print(f"{tid}: RETAIN FULL: {reason}", flush=True)
            continue
        result = compact_one(tid, args.dry_run, args.drop_screen_state)
        totals[result["status"]] += 1
        totals["removed_intermediate_bytes"] += sum(result.get("removed_intermediate_bytes", {}).values())
        totals["removed_screen_state_bytes"] += result.get("removed_screen_state_bytes", 0)
        print(json.dumps(result), flush=True)
    print("SUMMARY " + json.dumps(totals), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
