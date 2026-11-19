#!/usr/bin/env python3
"""Deduplicate the two identical private archive paths with a verified hard link.

Run after compile_results.py --zip and the full archive audit.  Both filenames
remain available with the same bytes and checksum; no archive member changes.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "amazon_earthworks_results.zip"
MIRROR = ROOT / "results/amazon_earthworks_results.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(checksum: str) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "archive_mirror_hardlinked_after_sha256_verification",
        "root": str(ARCHIVE.relative_to(ROOT)),
        "mirror": str(MIRROR.relative_to(ROOT)),
        "sha256": checksum,
        "root_inode": ARCHIVE.stat().st_ino,
        "mirror_inode": MIRROR.stat().st_ino,
        "bytes_no_longer_duplicated": ARCHIVE.stat().st_size,
        "note": "Both paths reference the same immutable checkpoint bytes. Future compilation "
                "creates a new checkpoint file; rerun this linker after the archive audit. "
                "No content, certainty or archive member changed.",
    }
    path = ROOT / "checkpoint/archive_mirror_deduplication.json"
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(part, path)


def main() -> int:
    checksum = sha256(ARCHIVE)
    if os.path.samefile(ARCHIVE, MIRROR):
        assert sha256(MIRROR) == checksum
        write_manifest(checksum)
        print("ALREADY_LINKED", checksum)
        return 0
    assert sha256(MIRROR) == checksum, "refusing to replace a different archive mirror"
    temporary = MIRROR.with_suffix(".zip.linktmp")
    if temporary.exists():
        temporary.unlink()
    os.link(ARCHIVE, temporary)
    assert sha256(temporary) == checksum
    os.replace(temporary, MIRROR)
    assert os.path.samefile(ARCHIVE, MIRROR) and sha256(MIRROR) == checksum
    write_manifest(checksum)
    print("HARDLINKED_ARCHIVE_MIRROR", checksum, "BYTES_NO_LONGER_DUPLICATED", ARCHIVE.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
