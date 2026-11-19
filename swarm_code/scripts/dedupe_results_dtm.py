#!/usr/bin/env python3
"""Remove redundant direct results/dtm copies after verifying all three copies.

The source DTM under data/derived_v2 and the member in the private results ZIP
are retained.  A direct results/dtm file is unlinked only after its size and
SHA-256 match both preserved copies.  No detector result, certainty or archive
member is changed.  Rerun after each compile_results.py --zip + archive audit.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    out = ROOT / "results/dtm"
    archive = ROOT / "amazon_earthworks_results.zip"
    files = sorted(out.glob("*_dtm3m.npz"))
    if not files:
        print("ALREADY_DEDUPLICATED results/dtm; manifest and preserved copies unchanged")
        return 0
    records = []
    removed_bytes = 0
    with zipfile.ZipFile(archive) as zipped:
        for path in files:
            tid = path.name.removesuffix("_dtm3m.npz")
            source = ROOT / "data/derived_v2" / tid / "dtm.npz"
            member = "dtm/" + path.name
            assert source.exists(), source
            info = zipped.getinfo(member)
            assert path.stat().st_size == source.stat().st_size == info.file_size, (
                tid, path.stat().st_size, source.stat().st_size, info.file_size)
            result_sha = sha_file(path)
            source_sha = sha_file(source)
            zip_digest = hashlib.sha256()
            with zipped.open(member) as handle:
                for block in iter(lambda: handle.read(4 << 20), b""):
                    zip_digest.update(block)
            assert result_sha == source_sha == zip_digest.hexdigest(), tid
            records.append({"transect": tid, "file": path.name,
                            "bytes": path.stat().st_size, "sha256": source_sha,
                            "zip_member": member})
            removed_bytes += path.stat().st_size
    manifest = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "verified_duplicate_results_dtm_copies_removed",
        "archive": str(archive.relative_to(ROOT)),
        "archive_sha256": sha_file(archive),
        "files": len(records), "removed_bytes": removed_bytes,
        "preserved": ["data/derived_v2/<transect>/dtm.npz",
                      "amazon_earthworks_results.zip member dtm/<transect>_dtm3m.npz",
                      "results/amazon_earthworks_results.zip (identical archive copy)"],
        "restoration": "Rerun code/compile_results.py --zip to recreate direct results/dtm copies; "
                       "this deduplication changes no source data, feature certainty, detector output "
                       "or ZIP member.",
        "verified_records": records,
    }
    destination = ROOT / "checkpoint/results_dtm_deduplication.json"
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
    for record in records:
        (out / record["file"]).unlink()
    print(f"REMOVED_VERIFIED_DUPLICATE_RESULTS_DTM {len(records)} BYTES {removed_bytes}")
    print(f"PRESERVED_SOURCE_AND_ZIP_COPIES {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
