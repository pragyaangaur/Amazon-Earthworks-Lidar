"""Download Acre transects one at a time, build the DTM and tiles. Raw files are kept so canopy views do not need a second download.

Usage: python -I src/batch.py 0413 0415 ...

Only one transect downloads at a time, which suits a slow network. The DTM and tiles are kept under outputs/private.
"""
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/api/records/7689909/files/transectos%20acre.zip/content"


CHUNK = 8 * 1024 * 1024


def fetch_range(start, end, tries=20):
    # One HTTP range request with retries. The connection to Zenodo drops often on a slow line.
    for i in range(tries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            r.raise_for_status()
            if len(r.content) == end - start + 1:
                return r.content
        except requests.RequestException as e:
            print("  retry", i + 1, e.__class__.__name__, flush=True)
        time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"range {start}-{end} failed after {tries} tries")


def download(name, dest):
    # Saves the compressed member bytes to a .part file and resumes from its size, so a dropped
    # connection only costs the current chunk. The member is inflated once it is complete.
    with RemoteZip(URL) as z:
        info = z.getinfo(name)
    head = fetch_range(info.header_offset, info.header_offset + 29)
    n, m = struct.unpack("<HH", head[26:30])
    start = info.header_offset + 30 + n + m
    part = dest.with_suffix(".part")
    have = part.stat().st_size if part.exists() else 0
    with open(part, "ab") as f:
        while have < info.compress_size:
            end = min(info.compress_size, have + CHUNK) - 1
            f.write(fetch_range(start + have, start + end))
            f.flush()
            have = end + 1
    data = part.read_bytes()
    if info.compress_type == 8:
        data = zlib.decompress(data, -15)
    if zlib.crc32(data) != info.CRC:
        part.unlink()
        raise RuntimeError("CRC mismatch, partial file removed")
    dest.write_bytes(data)
    part.unlink()


def run(tid):
    out = ROOT / "outputs" / "private" / f"T-{tid}"
    if (out / "dtm.npy").exists():
        print(tid, "already done", flush=True)
        return
    raw = ROOT / "data" / "raw"
    name = f"NP_T-{tid}.laz"
    t = time.time()
    if not (raw / name).exists():
        download(name, raw / name)
    print(tid, f"downloaded in {time.time() - t:.0f} s", flush=True)
    subprocess.run([sys.executable, "-I", str(ROOT / "src" / "dtm.py"), str(raw / name), str(out)], check=True)



if __name__ == "__main__":
    for tid in sys.argv[1:]:
        try:
            run(tid)
        except Exception as e:
            print(tid, "failed:", e, flush=True)
