"""dtmio.py -- dependency-free DTM input/output for the offline EBA screening.

Why this exists: this VM has numpy/scipy/PIL/matplotlib but NO rasterio, GDAL,
laspy or PDAL, and swarm members have no network. So:
  * .npy / .npz is the primary interchange format (cheap, exact, no deps);
  * a minimal GeoTIFF reader/writer (deflate, strips or tiles, predictor 2/3,
    float32/float64, int16/uint16/int32/uint32, single band) is included so a
    processor that *does* end up with rasterio/GDAL output can be read directly,
    and so we can emit georeferenced float32 rasters (DTM/LRM/hillshade) that
    QGIS reads without GDAL here.

Georeferencing convention
  The DTM array is stored north-up, row 0 = north (max y).
  cell size (sx, sy) in metres, origin (x0, y0) = map coordinate of the
  *outer top-left corner* of pixel (row 0, col 0), EPSG code as string.
  centre of pixel (r, c) = (x0 + (c + 0.5) * sx, y0 - (r + 0.5) * sy)

DTM container contract (.npz) -- produced by processors, consumed by detectors:
  dtm        float32 2-D   heights in metres, NaN = no data
  valid      bool 2-D      optional; if absent valid = isfinite(dtm)
  cell       float scalar  cell size metres (or cellx/celly)
  x0, y0     float         top-left outer corner, projected metres
  epsg       str/int       e.g. 5880
  canopy     float32 2-D   optional CHM
  source     str           optional provenance string
"""
from __future__ import annotations

import json
import os
import struct
import zlib

import numpy as np

ENDIAN = "<"   # module-level TIFF byte order, set by read_tiff


# --------------------------------------------------------------------------- #
# .npy / .npz
# --------------------------------------------------------------------------- #
def load_npz(path):
    """Load a DTM container. Missing optional keys are filled with defaults."""
    z = np.load(path, allow_pickle=False)
    if isinstance(z, np.ndarray):  # bare .npy: an array, no georeferencing
        return dict(dtm=np.asarray(z, dtype=np.float32), valid=np.isfinite(z),
                    cell=1.0, x0=0.0, y0=0.0, epsg="", canopy=None,
                    source=os.path.basename(str(path)))
    out = dict(z.files and {k: z[k] for k in z.files} or {})
    if "dtm" not in out and "h" in out:
        out["dtm"] = out["h"]
    dtm = np.asarray(out["dtm"], dtype=np.float32)
    valid = np.asarray(out["valid"]).astype(bool) if "valid" in out else np.isfinite(dtm)
    valid &= np.isfinite(dtm)
    cell = out.get("cell", out.get("cellx", 3.0))
    return dict(dtm=dtm, valid=valid,
                cell=float(np.asarray(cell).ravel()[0]),
                x0=float(np.asarray(out.get("x0", 0.0)).ravel()[0]),
                y0=float(np.asarray(out.get("y0", 0.0)).ravel()[0]),
                epsg=str(np.asarray(out.get("epsg", "")).ravel()[0]) if "epsg" in out else "",
                canopy=(np.asarray(out["canopy"], dtype=np.float32)
                        if "canopy" in out and np.ndim(out["canopy"]) == 2 else None),
                source=str(np.asarray(out.get("source", os.path.basename(str(path)))).ravel()[0]))


def save_npz(path, dtm, valid=None, cell=3.0, x0=0.0, y0=0.0, epsg="", canopy=None,
             source="", **extra):
    dtm = np.asarray(dtm, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(dtm)
    valid = np.asarray(valid).astype(bool) & np.isfinite(dtm)
    arrs = dict(dtm=dtm, valid=valid, cell=np.float64(cell), x0=np.float64(x0),
                y0=np.float64(y0), epsg=np.array(str(epsg)), source=np.array(str(source)))
    if canopy is not None:
        arrs["canopy"] = np.asarray(canopy, dtype=np.float32)
    for k, v in extra.items():
        arrs[k] = np.asarray(v)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# minimal TIFF / GeoTIFF reader
# --------------------------------------------------------------------------- #
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 11: "f", 12: "d"}


def _read_ifd(buf, endian, off):
    (n,) = struct.unpack_from(endian + "H", buf, off)
    tags = {}
    for i in range(n):
        base = off + 2 + 12 * i
        tag, typ, cnt = struct.unpack_from(endian + "HHI", buf, base)
        nbytes = _TYPE_SIZE.get(typ, 1) * cnt
        if nbytes <= 4:
            raw = buf[base + 8: base + 8 + nbytes]
        else:
            (voff,) = struct.unpack_from(endian + "I", buf, base + 8)
            raw = buf[voff: voff + nbytes]
        tags[tag] = (typ, cnt, raw)
    nxt = off + 2 + 12 * n
    next_off = struct.unpack_from(endian + "I", buf, nxt)[0] if nxt + 4 <= len(buf) else 0
    return tags, next_off


def _dec(tags, tag, default=None):
    if tag not in tags:
        return default
    typ, cnt, raw = tags[tag]
    if typ == 2:
        return raw.split(b"\x00")[0].decode("latin-1")
    fmt = _TYPE_FMT.get(typ)
    if fmt is None:
        return default
    vals = struct.unpack(ENDIAN + fmt * cnt, raw[:struct.calcsize(fmt) * cnt])
    return vals[0] if cnt == 1 else list(vals)


def _unpred(raw, pred, dt, cols, rows):
    a = np.frombuffer(raw, dtype=dt, count=cols * rows).reshape(rows, cols)
    if pred in (None, 1):
        return a.astype(np.float64)
    if dt.kind in "iu":
        u = np.cumsum(a.view(dt), axis=1, dtype=dt)
        return u.astype(np.float64)
    if pred == 3:
        # horizontal then vertical byte-plane differencing
        rb = np.frombuffer(raw, dtype=np.uint8, count=cols * rows * dt.itemsize)
        rb = rb.reshape(rows, cols * dt.itemsize)
        h = np.cumsum(rb.astype(np.uint8), axis=1, dtype=np.uint8)
        h = np.cumsum(h, axis=0, dtype=np.uint8)
        out = h.reshape(rows, cols, dt.itemsize).tobytes()
        return np.frombuffer(out, dtype=dt).reshape(rows, cols).astype(np.float64)
    raise NotImplementedError("predictor %r" % pred)


def read_tiff(path):
    """Minimal TIFF reader -> (array float64 2-D, geo dict). Single band only."""
    global ENDIAN
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:2] == b"II":
        ENDIAN = "<"
    elif buf[:2] == b"MM":
        ENDIAN = ">"
    else:
        raise ValueError("not a TIFF")
    magic, = struct.unpack_from(ENDIAN + "H", buf, 2)
    if magic not in (42, 43):
        raise ValueError("bad TIFF magic %r" % magic)
    off = struct.unpack_from(ENDIAN + "Q", buf, 8)[0] if magic == 43 else \
        struct.unpack_from(ENDIAN + "I", buf, 4)[0]
    t, _ = _read_ifd(buf, ENDIAN, off)
    W = _dec(t, 256)
    H = _dec(t, 257)
    bits = _dec(t, 258)
    comp = _dec(t, 259, 1)
    sf = _dec(t, 339, 1)
    spp = _dec(t, 277, 1)
    pred = _dec(t, 317, 1)
    planar = _dec(t, 284, 1)
    bits = bits[0] if isinstance(bits, list) else bits
    sf = sf[0] if isinstance(sf, list) else sf
    spp = spp[0] if isinstance(spp, list) else spp
    planar = planar[0] if isinstance(planar, list) else planar
    if spp not in (1, None) and planar != 2:
        raise NotImplementedError("interleaved multi-band TIFF; export band 1 separately")
    dt = np.dtype({1: "u", 2: "i", 3: "f"}.get(sf, "u") + str(bits // 8)).newbyteorder(ENDIAN)

    def decode(o, c):
        raw = buf[o: o + c]
        if comp in (8, 32946):
            raw = zlib.decompress(raw)
        elif comp not in (1, None):
            raise NotImplementedError("TIFF compression %r" % comp)
        return raw

    if 322 in t:
        tw, th = _dec(t, 322), _dec(t, 323)
        tw = tw[0] if isinstance(tw, list) else tw
        th = th[0] if isinstance(th, list) else th
        offsets, counts = _dec(t, 324), _dec(t, 325)
        offsets = offsets if isinstance(offsets, list) else [offsets]
        counts = counts if isinstance(counts, list) else [counts]
        tiles_x = (W + tw - 1) // tw
        out = np.empty((H, W), dtype=np.float64)
        for k, (o, c) in enumerate(zip(offsets, counts)):
            chip = _unpred(decode(o, c), pred, dt, tw, th)
            ty, tx = divmod(k, tiles_x)
            y0, x0 = ty * th, tx * tw
            hh, ww = min(th, H - y0), min(tw, W - x0)
            out[y0:y0 + hh, x0:x0 + ww] = chip[:hh, :ww]
    else:
        rps = _dec(t, 278, H)
        rps = rps[0] if isinstance(rps, list) else rps
        offsets, counts = _dec(t, 273), _dec(t, 279)
        offsets = offsets if isinstance(offsets, list) else [offsets]
        counts = counts if isinstance(counts, list) else [counts]
        out = np.empty((H, W), dtype=np.float64)
        row = 0
        for o, c in zip(offsets, counts):
            nrow = min(rps, H - row)
            out[row:row + nrow, :] = _unpred(decode(o, c), pred, dt, W, nrow)
            row += nrow
    geo = {}
    sc, tp = _dec(t, 33550), _dec(t, 33922)
    if sc is not None and tp is not None:
        geo["sx"], geo["sy"] = float(sc[0]), float(sc[1])
        geo["x0"], geo["y0"] = float(tp[3]), float(tp[4])
    gk = _dec(t, 34735)
    if gk:
        keys = {}
        for i in range(4, len(gk) - 3, 4):
            keys[gk[i]] = (gk[i + 1], gk[i + 2], gk[i + 3])
        for code in (3072, 2048):
            if code in keys and keys[code][1] == 0:
                geo["epsg"] = str(keys[code][2])
    geo["nodata"] = _dec(t, 42113)
    return out, geo


def load_dtm(path):
    """Dispatch on suffix: .npz/.npy -> load_npz, .tif/.tiff -> read_tiff."""
    p = str(path).lower()
    if p.endswith((".npz", ".npy")):
        return load_npz(path)
    if p.endswith((".tif", ".tiff")):
        arr, geo = read_tiff(path)
        cell = float(geo.get("sx", 1.0))
        dtm = arr.astype(np.float32)
        valid = np.isfinite(dtm)
        nd = geo.get("nodata")
        if nd is not None:
            try:
                valid &= ~np.isclose(dtm, float(nd))
            except Exception:
                pass
        dtm = np.where(valid, dtm, np.nan).astype(np.float32)
        return dict(dtm=dtm, valid=valid, cell=cell, x0=float(geo.get("x0", 0.0)),
                    y0=float(geo.get("y0", 0.0)), epsg=str(geo.get("epsg", "")),
                    canopy=None, source=os.path.basename(str(path)))
    raise ValueError("unsupported DTM suffix: %s" % path)


# --------------------------------------------------------------------------- #
# minimal GeoTIFF writer (uncompressed, stripped, float32, single band)
# --------------------------------------------------------------------------- #
def write_tiff(path, arr, cell=1.0, x0=0.0, y0=0.0, epsg="", nodata=None):
    """Write a north-up single-band float32 GeoTIFF (PixelIsArea, top-left origin)."""
    a = np.asarray(arr, dtype="<f4")
    H, W = a.shape
    data = a.tobytes()
    entries = [(256, 4, 1, struct.pack("<I", W)),
               (257, 4, 1, struct.pack("<I", H)),
               (258, 3, 1, struct.pack("<H", 32)),
               (259, 3, 1, struct.pack("<H", 1)),
               (262, 3, 1, struct.pack("<H", 1)),
               (277, 3, 1, struct.pack("<H", 1)),
               (284, 3, 1, struct.pack("<H", 1)),
               (339, 3, 1, struct.pack("<H", 3)),
               (278, 4, 1, struct.pack("<I", H)),
               (273, 4, 1, struct.pack("<I", 0)),        # patched below
               (279, 4, 1, struct.pack("<I", len(data))),
               (33550, 12, 3, struct.pack("<3d", cell, cell, 0.0)),
               (33922, 12, 6, struct.pack("<6d", 0.0, 0.0, 0.0, x0, y0, 0.0))]
    gk = [1, 1, 0, 0]
    if str(epsg).strip():
        code = int(str(epsg).strip())
        gk += [3072 if code >= 2000 else 2048, 0, 1, code]
    gk += [1024, 0, 1, 2, 1025, 0, 1, 1]
    entries.append((34735, 3, len(gk), struct.pack("<%dH" % len(gk), *gk)))
    nd = b"nan\x00" if nodata is None else (("%.10g" % float(nodata)).encode() + b"\x00")
    entries.append((42113, 2, len(nd), nd))
    entries.sort(key=lambda e: e[0])
    ifd_off = 8
    extra_off = ifd_off + 2 + 12 * len(entries) + 4
    blobs, offs = b"", {}
    for tag, typ, cnt, payload in entries:
        if len(payload) <= 4:
            offs[tag] = None
        else:
            offs[tag] = extra_off + len(blobs)
            blobs += payload
            if len(blobs) % 2:
                blobs += b"\x00"
    data_off = extra_off + len(blobs)
    data_off += (4 - data_off % 4) % 4
    out = bytearray(b"II" + struct.pack("<HI", 42, ifd_off) + struct.pack("<H", len(entries)))
    for tag, typ, cnt, payload in entries:
        o = offs[tag]
        if tag == 273:
            payload, o = struct.pack("<I", data_off), None
        out += struct.pack("<HHI", tag, typ, cnt)
        out += payload.ljust(4, b"\x00") if o is None else struct.pack("<I", o)
    out += struct.pack("<I", 0) + blobs
    out += b"\x00" * (data_off - len(out)) + data
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(bytes(out))
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


if __name__ == "__main__":
    import tempfile
    a = (np.random.rand(37, 53) * 100).astype(np.float32)
    fd = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "."))
    p = os.path.join(fd, "t.tif")
    write_tiff(p, a, cell=3.0, x0=123456.0, y0=8765432.0, epsg=5880)
    b, geo = read_tiff(p)
    npz = os.path.join(fd, "t.npz")
    save_npz(npz, a, cell=3.0, x0=1.0, y0=2.0, epsg="5880")
    c = load_npz(npz)
    print("tiff roundtrip maxabs err", float(np.max(np.abs(b - a))), geo)
    print("npz roundtrip maxabs err", float(np.nanmax(np.abs(c["dtm"] - a))), c["cell"], c["epsg"])
