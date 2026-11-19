"""LAZ/LAS point streaming through laspy+lazrs (available in repo `.venv`).

The VM's system python has no LAZ decoder, but the coordinator provisioned
``.venv`` with laspy 2.7 / lazrs 0.8 (see repo/.venv).  This module exposes the
same iterator contract as ``lasio.iter_xyz`` so ``raster.accumulate`` and
``process.py`` stay reader-agnostic:

    for x, y, z, blk in iter_xyz(path, block=2_000_000): ...

``blk`` carries a synthetic ``flags`` byte with return number in bits 0-2 and
number-of-returns in bits 3-5, exactly as the LAS spec packs them, so the
return statistics in ``raster.accumulate`` work for both readers.

Run with:  .venv/bin/python process.py INPUT.laz -o OUTDIR
"""
from __future__ import annotations

import numpy as np


def have_laspy():
    try:
        import laspy  # noqa: F401
        return True
    except Exception:
        return False


def check_backend():
    """Return the LAZ backend name actually in use, or raise with a reason."""
    import laspy
    try:
        backend = laspy.LazBackend.detect_available()
    except Exception:
        backend = None
    if not backend:
        raise RuntimeError("laspy present but no LAZ backend (lazrs/laszip) found")
    return backend


def iter_xyz(path, block=2_000_000, header=None, extra=("classification",)):
    """Yield (x, y, z, blk) with float64 metres; bounded by ``block`` points."""
    import laspy
    with laspy.open(str(path)) as f:
        for pts in f.chunk_iterator(int(block)):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float64)
            rn = np.asarray(pts.return_number, dtype=np.uint8) & 0b111
            nr = np.asarray(pts.number_of_returns, dtype=np.uint8) & 0b111
            blk = np.zeros(x.size, dtype=[("flags", "u1"), ("classification", "u1")])
            blk["flags"] = (rn | (nr << 3)).astype(np.uint8)
            try:
                blk["classification"] = np.asarray(pts.classification, dtype=np.uint8)
            except Exception:
                pass
            yield x, y, z, blk


def header_dict(path):
    """Small dict of the facts a transect row needs (no laspy objects leak)."""
    import laspy
    with laspy.open(str(path)) as f:
        h = f.header
        crs = None
        try:
            crs = h.parse_crs()
        except Exception:
            crs = None
        return dict(
            n_points=int(h.point_count), point_format=int(h.point_format.id),
            version=str(h.version), scales=[float(v) for v in h.scales],
            offsets=[float(v) for v in h.offsets],
            bounds=dict(mins=[float(v) for v in h.mins], maxs=[float(v) for v in h.maxs]),
            epsg=(int(crs.to_epsg()) if crs is not None and crs.to_epsg() else None),
            crs_wkt=(crs.to_wkt() if crs is not None else None),
        )


if __name__ == "__main__":
    import sys
    import time
    p = sys.argv[1]
    print("backend:", check_backend())
    print("header:", header_dict(p))
    t0 = time.time()
    n = 0
    for x, y, z, blk in iter_xyz(p, block=1_000_000):
        n += x.size
        if n <= 1_000_000:
            print("first block: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f] returns %s"
                  % (x.min(), x.max(), y.min(), y.max(), z.min(), z.max(),
                     np.bincount(blk["flags"] & 7)))
    dt = time.time() - t0
    print("%d points in %.1f s (%.2f M pts/s)" % (n, dt, n / dt / 1e6))
