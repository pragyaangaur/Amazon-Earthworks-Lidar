"""Download Acre transects one at a time, build the DTM and tiles. Raw files are kept so canopy views do not need a second download.

Usage: python -I src/batch.py 0413 0415 ...

Only one transect downloads at a time, which suits a slow network. The DTM and tiles are kept under outputs/private.
"""
import subprocess
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
URL = "https://zenodo.org/api/records/7689909/files/transectos%20acre.zip/content"


def run(tid):
    out = ROOT / "outputs" / "private" / f"T-{tid}"
    if (out / "dtm.npy").exists():
        print(tid, "already done", flush=True)
        return
    raw = ROOT / "data" / "raw"
    name = f"NP_T-{tid}.laz"
    t = time.time()
    with RemoteZip(URL) as z:
        z.extract(name, raw)
    print(tid, f"downloaded in {time.time() - t:.0f} s", flush=True)
    subprocess.run([sys.executable, "-I", str(ROOT / "src" / "dtm.py"), str(raw / name), str(out)], check=True)



if __name__ == "__main__":
    for tid in sys.argv[1:]:
        try:
            run(tid)
        except Exception as e:
            print(tid, "failed:", e, flush=True)
