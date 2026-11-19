#!/usr/bin/env python3
"""Fetch one member of a large remote Zenodo ZIP using resumable byte ranges.

Example (from project root):
  .venv/bin/python code/fetch_zip_member.py --record-id 7689909 \
    --zip-key "transectos acre.zip" --member NP_T-0414.laz

Only the selected member's compressed bytes are fetched. Deflate output is
verified against the uncompressed size and CRC-32 in the ZIP central directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import time
import zlib

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024


def log_failure(stage: str, message: str) -> None:
    entry = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "stage": stage, "failure": message, "fatal": False}
    with (ROOT / "logs/failures.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def request_range(session: requests.Session, url: str, start: int, end: int,
                  attempts: int = 12) -> bytes:
    expected = end - start + 1
    for attempt in range(attempts):
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}"},
                                   timeout=(20, 150))
            if response.status_code == 429:
                retry = min(300, int(response.headers.get("Retry-After", "60")))
                response.close()
                if attempt >= 2:
                    raise RuntimeError("repeated HTTP 429; stopping rather than hammering Zenodo")
                time.sleep(retry)
                continue
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"expected HTTP 206 for range, got {response.status_code}")
            content_range = response.headers.get("Content-Range", "")
            match = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
            if not match or int(match.group(1)) != start or int(match.group(2)) != end:
                raise RuntimeError(f"unexpected Content-Range {content_range!r} for {start}-{end}")
            data = response.content
            response.close()
            if len(data) != expected:
                raise RuntimeError(f"short range: {len(data)} of {expected} bytes")
            return data
        except (requests.RequestException, RuntimeError) as exc:
            log_failure("remote_range", f"range {start}-{end} attempt {attempt + 1}: {exc}")
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60, 2**attempt))
    raise RuntimeError("unreachable")


def central_directory(session: requests.Session, url: str, cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    with RemoteZip(url, session=session) as archive:
        listing = [{"name": info.filename, "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "compress_type": info.compress_type, "CRC": info.CRC,
                    "header_offset": info.header_offset}
                   for info in archive.infolist()]
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".part")
    temporary.write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, cache)
    return listing


def crc_of_file(path: Path) -> tuple[int, int]:
    crc = 0
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return crc & 0xFFFFFFFF, size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-id", type=int, required=True)
    parser.add_argument("--zip-key", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata_path = ROOT / f"data/metadata/zenodo_{args.record_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entry = next(f for f in metadata["files"] if f["key"] == args.zip_key)
    url = entry["links"]["self"]
    safe_zip = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.zip_key).removesuffix(".zip")
    cache = ROOT / f"data/metadata/{safe_zip}.members.json"
    # Preserve the earlier Acre cache's established filename.
    if args.record_id == 7689909 and args.zip_key == "transectos acre.zip":
        cache = ROOT / "data/metadata/transectos_acre.members.json"

    session = requests.Session()
    session.headers["User-Agent"] = "EBA-earthworks-research/1.0 (public-data archival research)"
    listing = central_directory(session, url, cache)
    info = next(item for item in listing if item["name"] == args.member)
    if info["compress_type"] not in (0, 8):
        raise RuntimeError(f"unsupported ZIP compression method {info['compress_type']}")

    output = args.output or ROOT / ".scratch/amazon/raw" / Path(args.member).name
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        crc, size = crc_of_file(output)
        if size == info["size"] and crc == info["CRC"]:
            print(f"already verified {output} ({size} bytes; CRC {crc:08x})")
            return
        raise RuntimeError(f"existing output fails ZIP verification: {output}")

    local_header = request_range(session, url, info["header_offset"],
                                 info["header_offset"] + 29)
    signature, version, flags, method, mtime, mdate, crc32, csize, usize, name_len, extra_len = \
        struct.unpack("<IHHHHHIIIHH", local_header)
    if signature != 0x04034B50:
        raise RuntimeError(f"bad local ZIP signature {signature:08x}")
    if method != info["compress_type"]:
        raise RuntimeError("local and central ZIP compression methods disagree")
    data_start = info["header_offset"] + 30 + name_len + extra_len
    compressed_size = info["compressed_size"]
    part = output.with_name(output.name + ".deflate.part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > compressed_size:
        raise RuntimeError(f"oversized compressed partial file: {part}")
    started = time.monotonic()
    while offset < compressed_size:
        end = min(offset + CHUNK, compressed_size) - 1
        data = request_range(session, url, data_start + offset, data_start + end)
        with part.open("ab") as out:
            out.write(data)
        offset += len(data)
        elapsed = max(time.monotonic() - started, 0.001)
        print(f"{args.member}: compressed {offset}/{compressed_size} bytes "
              f"({offset / compressed_size:.1%}; {offset / elapsed / 1e6:.1f} MB/s)", flush=True)

    raw_part = output.with_name(output.name + ".part")
    crc = 0
    size = 0
    with part.open("rb") as source, raw_part.open("wb") as out:
        if info["compress_type"] == 8:
            inflater = zlib.decompressobj(-15)
            for chunk in iter(lambda: source.read(CHUNK), b""):
                raw = inflater.decompress(chunk)
                if raw:
                    out.write(raw)
                    crc = zlib.crc32(raw, crc)
                    size += len(raw)
            raw = inflater.flush()
            if raw:
                out.write(raw)
                crc = zlib.crc32(raw, crc)
                size += len(raw)
            if not inflater.eof or inflater.unused_data:
                raise RuntimeError("invalid or trailing deflate stream")
        else:
            for raw in iter(lambda: source.read(CHUNK), b""):
                out.write(raw)
                crc = zlib.crc32(raw, crc)
                size += len(raw)
    crc &= 0xFFFFFFFF
    if size != info["size"] or crc != info["CRC"]:
        raise RuntimeError(f"ZIP verification failed: size {size}/{info['size']}, "
                           f"CRC {crc:08x}/{info['CRC']:08x}")
    os.replace(raw_part, output)
    part.unlink()
    provenance = {"record_id": args.record_id, "zip_key": args.zip_key,
                  "member": args.member, "source_url": url, "size": size,
                  "crc32": f"{crc:08x}", "range_chunk_bytes": CHUNK,
                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "elapsed_seconds": time.monotonic() - started,
                  "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    output.with_name(output.name + ".source.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"VERIFIED {output} size={size} CRC={crc:08x}", flush=True)


if __name__ == "__main__":
    main()
