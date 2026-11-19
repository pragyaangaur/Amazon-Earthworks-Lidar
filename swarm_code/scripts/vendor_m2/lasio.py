"""Minimal, dependency-free LAS 1.0-1.4 reader/writer (uncompressed).

Why this file exists: the VM has no laspy / lazrs / PDAL / rasterio and this
swarm is forbidden from using the network, so every ingest path must be built
locally.  This module is the *reliable* half of ingest: it memory-maps a LAS
file, parses the header, VLRs and CRS, and hands out fixed-size blocks of
points so that peak RAM stays bounded regardless of file size.

Conventions used by the rest of the pipeline
--------------------------------------------
* Coordinates are returned as float64 metres, computed in chunks with
  ``X * scale + offset`` exactly as the LAS spec requires.
* ``iter_points`` yields *structured* numpy arrays; callers pick the columns
  they need.  Nothing here keeps all points in memory.

Public API
----------
read_header(path)                 -> Header (dict-like)
iter_points(path, block=...)      -> yields structured arrays
las_epsg(header)                  -> int | None
write_las(path, x, y, z, ...)     -> write a LAS 1.2/1.4 file (test fixtures)
"""
from __future__ import annotations

import struct
import numpy as np

# ---------------------------------------------------------------- header ----

_HDR_BASE = [
    ("file_signature", "4s"), ("file_source_id", "<u2"), ("global_encoding", "<u2"),
    ("guid1", "<u4"), ("guid2", "<u2"), ("guid3", "<u2"), ("guid4", "8s"),
    ("version_major", "u1"), ("version_minor", "u1"),
    ("system_id", "32s"), ("software", "32s"),
    ("creation_doy", "<u2"), ("creation_year", "<u2"),
    ("header_size", "<u2"), ("offset_to_points", "<u4"),
    ("n_vlr", "<u4"), ("point_format", "u1"), ("point_len", "<u2"),
    ("n_legacy", "<u4"), ("n_by_return_legacy", "<u4", (5,)),
    ("sx", "<f8"), ("sy", "<f8"), ("sz", "<f8"),
    ("ox", "<f8"), ("oy", "<f8"), ("oz", "<f8"),
    ("maxx", "<f8"), ("minx", "<f8"), ("maxy", "<f8"),
    ("miny", "<f8"), ("maxz", "<f8"), ("minz", "<f8"),
]

_POINT_FMT_CORE = {0: 20, 1: 28, 2: 26, 3: 34, 6: 30, 7: 36, 8: 38, 9: 59, 10: 67}


def _dtype_from(fields):
    """Build a numpy dtype, expanding the (name, fmt, shape) triples."""
    out = []
    for f in fields:
        name, fmt = f[0], f[1]
        shape = f[2] if len(f) > 2 else None
        if isinstance(fmt, str):
            if fmt[:-1].isdigit() and fmt.endswith("s"):   # LAS "32s" -> numpy "S32"
                fmt = "S" + fmt[:-1]
            if shape is None:
                out.append((name, fmt))
            else:
                out.append((name, fmt, shape))
        else:  # already a dtype, possibly with shape
            out.append((name, fmt) if shape is None else (name, fmt, shape))
    return np.dtype(out)


_BASE_DT = _dtype_from(_HDR_BASE)


class Header(dict):
    """dict subclass: keys are the LAS header fields, plus derived info."""

    @property
    def n_points(self):
        return int(self.get("n_points") or 0)

    def xyz(self, X, Y, Z):
        return (X * self["sx"] + self["ox"],
                Y * self["sy"] + self["oy"],
                Z * self["sz"] + self["oz"])

    def __getattr__(self, k):          # convenience: h.sx
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def read_header(path):
    with open(path, "rb") as f:
        raw = f.read(375)
    if raw[:4] != b"LASF":
        raise ValueError("%s: not a LAS file (signature %r)" % (path, raw[:4]))
    if len(raw) < 227:
        raise ValueError("%s: truncated LAS header" % path)
    h = Header()
    rec = np.frombuffer(raw[:_BASE_DT.itemsize], dtype=_BASE_DT, count=1)[0]
    h.update({k: (rec[k].item() if np.ndim(rec[k]) == 0 else rec[k].tolist())
              for k in _BASE_DT.names})
    h["n_by_return_legacy"] = list(h["n_by_return_legacy"])
    vmaj, vmin = h["version_major"], h["version_minor"]
    if vmaj == 1 and vmin >= 3:
        off = 227
        h["start_waveform"], = struct.unpack_from("<Q", raw, off)
        off += 8
        if vmin >= 4:
            h["start_evlr"], = struct.unpack_from("<Q", raw, off)
            h["n_evlr"], = struct.unpack_from("<I", raw, off + 8)
            h["n_points"], = struct.unpack_from("<Q", raw, off + 12)
            h["n_by_return"] = list(struct.unpack_from("<15Q", raw, off + 20))
    if "n_points" not in h:
        h["n_points"] = h["n_legacy"]
        h["n_by_return"] = h["n_by_return_legacy"]
    h["n_points"] = int(h["n_points"])
    h["path"] = str(path)
    h["vlrs"] = _read_vlrs(path, h["offset_to_points"], h["n_vlr"])
    h["epsg"] = las_epsg(h)
    return h


# ------------------------------------------------------------------ VLRs ----

def _read_vlrs(path, offset_to_points, n_vlr):
    out = []
    if n_vlr <= 0:
        return out
    with open(path, "rb") as f:
        f.seek(0)
        head = f.read(offset_to_points)
    pos = _BASE_DT.itemsize
    # LAS 1.3+ header is longer than the base dtype; skip to header_size
    hsize = struct.unpack_from("<H", head, 94)[0]
    pos = hsize
    for _ in range(int(n_vlr)):
        if pos + 54 > len(head):
            break
        reserved, user_id, rec_id, rec_len, desc = struct.unpack_from(
            "<H16sHH32s", head, pos)
        pos += 54
        data = head[pos:pos + rec_len]
        pos += rec_len
        out.append(dict(reserved=reserved, user_id=user_id.rstrip(b"\0").decode("ascii", "replace"),
                        record_id=rec_id, length=rec_len,
                        description=desc.rstrip(b"\0").decode("ascii", "replace"),
                        data=data))
    return out


def las_epsg(h):
    """Best-effort EPSG from the WKT VLR (2112) or GeoTIFF keys (34735)."""
    for v in h.get("vlrs", []):
        if v["record_id"] == 2111 and v["data"][:4] in (b"\x00\x00\x00\x00",):
            continue
        if v["record_id"] == 2112:  # OGC WKT
            txt = v["data"].split(b"\0")[0].decode("ascii", "replace")
            i = txt.rfind("EPSG\",\"")
            if i >= 0:
                j = txt.find('"', i + 7)
                try:
                    return int(txt[i + 7:j])
                except ValueError:
                    pass
            for key in ("AUTHORITY", "ID["):
                i = txt.rfind(key)
                if i >= 0:
                    seg = txt[i:i + 40]
                    digits = "".join(c if c.isdigit() else " " for c in seg).split()
                    if digits:
                        return int(digits[-1])
        if v["record_id"] == 34735:  # GeoKeyDirectory
            import array
            n = v["length"] // 2
            vals = array.array("H")
            vals.frombytes(v["data"][:n * 2])
            if len(vals) >= 4:
                nkeys = vals[3]
                for k in range(nkeys):
                    base = 4 + 4 * k
                    if base + 3 >= len(vals):
                        break
                    key_id, tiff_tag, count, value = vals[base:base + 4]
                    if key_id in (3072, 2048) and count == 1 and value < 32767:
                        return int(value)
    return None


def laszip_vlr(h):
    for v in h.get("vlrs", []):
        if v["user_id"] == "laszip encoded" and v["record_id"] == 22204:
            return _parse_laszip_vlr(v["data"])
    return None


def _parse_laszip_vlr(d):
    (compressor, coder, vmaj, vmin, vrev, options, chunk_size,
     n_special, off_special, n_items) = struct.unpack_from("<HHBBHIIqqH", d, 0)
    items = []
    pos = 34
    for _ in range(n_items):
        typ, size, ver = struct.unpack_from("<HHH", d, pos)
        items.append(dict(type=typ, size=size, version=ver))
        pos += 6
    return dict(compressor=compressor, coder=coder, version=(vmaj, vmin, vrev),
                options=options, chunk_size=chunk_size, n_items=n_items,
                items=items, num_special_evlrs=n_special)


def is_laz(path):
    """True when the file is LAZ (LASzip) compressed."""
    try:
        h = read_header(path)
    except Exception:
        with open(path, "rb") as f:
            return f.read(4) == b"LASF" and b"laszip encoded" in f.read(4096)
    return laszip_vlr(h) is not None


# ----------------------------------------------------------- point blocks ----

def point_dtype(h):
    """Structured dtype for one point record, padded to ``point_len`` bytes."""
    pf, plen = int(h["point_format"]), int(h["point_len"])
    if pf > 10:
        pf %= 128   # LAS 1.4 PDRF >= 6 may be stored as 128+base by LAZ tools
    fields = [("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"),
              ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
              ("user_data", "u1"), ("point_source", "<u2")]
    base = 20
    if pf in (1, 3, 4, 5):
        fields.append(("gps_time", "<f8")); base += 8
    if pf in (2, 3, 5):
        fields.append(("red", "<u2")); fields.append(("green", "<u2")); fields.append(("blue", "<u2")); base += 6
    if plen < base:
        base = plen
    dt = np.dtype(fields)
    if plen > dt.itemsize:
        dt = np.dtype(dt.descr + [("_pad", "V%d" % (plen - dt.itemsize))])
    return dt


def iter_points(path, block=2_000_000, header=None, columns=None):
    """Yield structured blocks of points.  Peak RAM ~ block * point_len."""
    h = header if header is not None else read_header(path)
    dt = point_dtype(h)
    n = h["n_points"]
    off = int(h["offset_to_points"])
    plen = int(h["point_len"])
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(n * plen,))
    step = int(block)
    for i in range(0, n, step):
        j = min(i + step, n)
        raw = np.asarray(mm[i * plen:j * plen]).view(dt)
        yield raw if columns is None else raw[list(columns)]


def iter_xyz(path, block=2_000_000, header=None, extra=("classification",)):
    """Yield (x, y, z, block) tuples in float64 metres plus requested extras."""
    h = header if header is not None else read_header(path)
    sx, sy, sz = h["sx"], h["sy"], h["sz"]
    ox, oy, oz = h["ox"], h["oy"], h["oz"]
    cols = ["X", "Y", "Z"] + list(extra)
    for blk in iter_points(path, block=block, header=h, columns=cols):
        x = blk["X"].astype(np.float64) * sx + ox
        y = blk["Y"].astype(np.float64) * sy + oy
        z = blk["Z"].astype(np.float64) * sz + oz
        yield x, y, z, blk


# ---------------------------------------------------------------- writer ----

def write_las(path, x, y, z, scale=0.001, offset=None, epsg=None,
              gps_time=None, classification=None, intensity=None,
              system_id="ultramath m2", software="lasio.py"):
    """Write an uncompressed LAS 1.2 file (PDRF 0 or 1).  Test fixtures only."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64); z = np.asarray(z, np.float64)
    n = x.size
    assert y.size == n and z.size == n
    if offset is None:
        offset = (np.floor(np.array([x.min(), y.min(), z.min()]) / 1000.0) * 1000.0)
    offx, offy, offz = (float(v) for v in offset)
    X = np.rint((x - offx) / scale).astype("<i4")
    Y = np.rint((y - offy) / scale).astype("<i4")
    Z = np.rint((z - offz) / scale).astype("<i4")
    pf = 1 if gps_time is not None else 0
    plen = _POINT_FMT_CORE[pf]
    vlr = b""
    n_vlr = 0
    if epsg is not None:
        wkt = ('PROJCS["WGS 84 / UTM zone %dS",GEOGCS["WGS 84",DATUM["WGS_1984",'
               'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
               'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
               'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",%d],'
               'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
               'PARAMETER["false_northing",10000000],UNIT["metre",1],'
               'AUTHORITY["EPSG","%d"]]') % (abs(epsg - 32600 if epsg < 32700 else epsg - 32700),
                                          -69 if epsg == 32719 else 0, epsg)
        data = wkt.encode("ascii") + b"\0"
        vlr = struct.pack("<H16sHH32s", 0, b"LASF_Projection", 2112, len(data),
                          b"OGC WKT") + data
        n_vlr = 1
    hsize = 227
    off_points = hsize + len(vlr)
    hdr = bytearray(hsize)
    struct.pack_into("<4s", hdr, 0, b"LASF")
    struct.pack_into("<H", hdr, 4, 0)
    struct.pack_into("<H", hdr, 6, 0)
    hdr[8:24] = b"\0" * 16
    struct.pack_into("<BB", hdr, 24, 1, 2)
    hdr[26:58] = system_id.encode()[:31].ljust(32, b"\0")
    hdr[58:90] = software.encode()[:31].ljust(32, b"\0")
    struct.pack_into("<HH", hdr, 90, 1, 2024)
    struct.pack_into("<H", hdr, 94, hsize)
    struct.pack_into("<I", hdr, 96, off_points)
    struct.pack_into("<I", hdr, 100, n_vlr)
    struct.pack_into("<B", hdr, 104, pf)
    struct.pack_into("<H", hdr, 105, plen)
    struct.pack_into("<I", hdr, 107, n)
    nr = np.zeros(5, "<u4")
    struct.pack_into("<5I", hdr, 111, *nr)
    struct.pack_into("<3d", hdr, 131, scale, scale, scale)
    struct.pack_into("<3d", hdr, 155, offx, offy, offz)
    struct.pack_into("<6d", hdr, 179, x.max(), x.min(), y.max(), y.min(), z.max(), z.min())

    dt = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("intensity", "<u2"),
                   ("flags", "u1"), ("classification", "u1"), ("scan_angle", "i1"),
                   ("user_data", "u1"), ("point_source", "<u2")]
                  + ([("gps_time", "<f8")] if pf == 1 else []))
    pts = np.zeros(n, dt)
    pts["X"] = X; pts["Y"] = Y; pts["Z"] = Z
    pts["intensity"] = 0 if intensity is None else np.asarray(intensity, "<u2")
    pts["classification"] = 1 if classification is None else np.asarray(classification, "u1")
    if pf == 1:
        pts["gps_time"] = np.zeros(n) if gps_time is None else gps_time
    with open(path, "wb") as f:
        f.write(bytes(hdr)); f.write(vlr); f.write(pts.tobytes())
    return dict(path=str(path), n=int(n), point_format=pf, epsg=epsg,
                offset=(offx, offy, offz), scale=scale)
