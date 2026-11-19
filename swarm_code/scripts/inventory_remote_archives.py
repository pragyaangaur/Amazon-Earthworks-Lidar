#!/usr/bin/env python3
"""Cache central-directory inventories for all public EBA L1A archives.

Only ZIP metadata are fetched, not LAZ payloads. Run after
`code/download_public_inputs.py` from the project root.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
RECORD_IDS = (7689909, 7636454, 7689693, 7689210)


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "EBA-earthworks-research/1.0 (public-data archival research)"
    combined = []
    for record_id in RECORD_IDS:
        metadata = json.loads((ROOT / f"data/metadata/zenodo_{record_id}.json").read_text())
        for entry in metadata["files"]:
            key = entry["key"]
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).removesuffix(".zip")
            cache = ROOT / f"data/metadata/{safe}.members.json"
            if record_id == 7689909 and key == "transectos acre.zip":
                cache = ROOT / "data/metadata/transectos_acre.members.json"
            if cache.exists():
                listing = json.loads(cache.read_text())
            else:
                for attempt in range(6):
                    try:
                        with RemoteZip(entry["links"]["self"], session=session) as archive:
                            listing = [{"name": info.filename, "size": info.file_size,
                                        "compressed_size": info.compress_size,
                                        "compress_type": info.compress_type,
                                        "CRC": info.CRC, "header_offset": info.header_offset}
                                       for info in archive.infolist()]
                        break
                    except Exception as exc:
                        with (ROOT / "logs/failures.jsonl").open("a") as log:
                            log.write(json.dumps({"stage": "remote_inventory", "record_id": record_id,
                                                  "zip_key": key, "attempt": attempt + 1,
                                                  "failure": repr(exc), "fatal": False}) + "\n")
                        if attempt == 5:
                            raise
                        time.sleep(min(60, 2**attempt))
                temporary = cache.with_suffix(cache.suffix + ".part")
                temporary.write_text(json.dumps(listing, indent=2, ensure_ascii=False) + "\n")
                temporary.replace(cache)
            laz = [item for item in listing if item["name"].lower().endswith((".laz", ".las"))]
            print(f"{record_id} | {key} | {len(laz)} point-cloud members", flush=True)
            for item in laz:
                combined.append({"record_id": record_id, "zip_key": key,
                                 "source_url": entry["links"]["self"], **item})
    output = ROOT / "data/metadata/l1a_inventory.json"
    output.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    print(f"TOTAL {len(combined)} point-cloud members; inventory: {output}", flush=True)


if __name__ == "__main__":
    main()
