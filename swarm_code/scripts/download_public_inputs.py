#!/usr/bin/env python3
"""Download and verify the small public inputs; inspect remote EBA ZIP members.

Run from the project root with `.venv/bin/python code/download_public_inputs.py`.
No candidate coordinates are written to network services or git by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
SMALL_INPUTS = {
    8083287: "Vperipato/ade2541-v1.0.0.zip",
    4968706: "L3A_transects_v20210616.zip",
}
LARGE_RECORDS = (7689909, 7636454, 7689693, 7689210)


def get_json(session: requests.Session, url: str) -> dict:
    for attempt in range(6):
        try:
            response = session.get(url, timeout=(20, 90))
            if response.status_code == 429:
                time.sleep(min(60 * 2**attempt, int(response.headers.get("Retry-After", "60"))))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == 5:
                raise
            print(f"metadata retry {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(60, 2**attempt))
    raise RuntimeError("unreachable")


def download(session: requests.Session, url: str, destination: Path,
             expected_size: int, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    for attempt in range(12):
        offset = part.stat().st_size if part.exists() else 0
        if offset > expected_size:
            raise RuntimeError(f"oversized partial file: {part}")
        if offset == expected_size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
                if response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", "60")))
                    continue
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("server ignored resume Range request")
                mode = "ab" if offset else "wb"
                with part.open(mode) as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            output.write(chunk)
        except requests.RequestException as exc:
            print(f"download retry {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(60, 2**attempt))
            continue
    if not part.exists() or part.stat().st_size != expected_size:
        actual = part.stat().st_size if part.exists() else -1
        raise RuntimeError(f"incomplete {part}: {actual} of {expected_size} bytes")
    digest = hashlib.md5()
    with part.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {part}: {digest.hexdigest()} != {expected_md5}")
    os.replace(part, destination)
    print(f"verified {destination.relative_to(ROOT)} ({expected_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-acre", action="store_true",
                        help="List Acre remote ZIP members without downloading the archive")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "EBA-earthworks-research/1.0 (public-data archival research)"
    metadata_dir = ROOT / "data/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for record_id in sorted(set(SMALL_INPUTS) | set(LARGE_RECORDS)):
        record = get_json(session, f"https://zenodo.org/api/records/{record_id}")
        records[record_id] = record
        (metadata_dir / f"zenodo_{record_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"record {record_id}: {len(record.get('files', []))} files", flush=True)
        for entry in record.get("files", []):
            print(f"  {entry['key']} | {entry['size']} bytes", flush=True)
    for record_id, key in SMALL_INPUTS.items():
        entry = next(f for f in records[record_id]["files"] if f["key"] == key)
        destination = ROOT / "data/downloads" / Path(key).name
        download(session, entry["links"]["self"], destination, entry["size"],
                 entry["checksum"].removeprefix("md5:"))
        with zipfile.ZipFile(destination) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f"CRC failure in {destination}: {bad}")
            listing = [{"name": info.filename, "size": info.file_size,
                        "compressed_size": info.compress_size}
                       for info in archive.infolist()]
            (metadata_dir / f"{destination.name}.members.json").write_text(
                json.dumps(listing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {len(listing)} ZIP members; CRC checks passed", flush=True)
    if args.inspect_acre:
        entry = next(f for f in records[7689909]["files"] if f["key"] == "transectos acre.zip")
        with RemoteZip(entry["links"]["self"], session=session) as archive:
            listing = [{"name": info.filename, "size": info.file_size,
                        "compressed_size": info.compress_size,
                        "compress_type": info.compress_type,
                        "CRC": info.CRC, "header_offset": info.header_offset}
                       for info in archive.infolist()]
        (metadata_dir / "transectos_acre.members.json").write_text(
            json.dumps(listing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for item in listing:
            print(f"ACRE MEMBER {item['name']} | {item['size']} bytes", flush=True)


if __name__ == "__main__":
    main()
