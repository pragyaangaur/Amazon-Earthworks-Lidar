"""Minimal ESRI shapefile + DBF reader (no GDAL, no pyshp, no network).

Only what the EBA footprint inventory needs: 2-D polyline/polygon shapes and a
DBF attribute table.  Everything else is refused with a named error rather than
guessed.

Validated against the real inventory: data/footprints/transectos_total_v20210616
has 901 records, shape type 5 (polygon), EPSG:5880, and the ``transect`` field
reproduces the coordinator's independently produced transect_ids.json exactly
(tests/test_transect_proximity.py).

.shp layout used here (ESRI Technical Description, 1998):
  file header 100 B  >i file code 9994, >i file length in 16-bit words,
                     <i version 1000, <i shape type, <4d bbox, <4d z/m ranges
  record       8 B   >i record number (1-based), >i content length in words
  content            <i shape type, then per type: bbox, nParts, nPoints,
                     parts (index of first point per part), points (x,y)
.shx holds the same 8-byte records and is used here only to count records.
DBF layout: 32 B header, 32 B field descriptors, 0x0D terminator, fixed-width
records, 0x20 deletion flag.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ShpError", "ShapeRecord", "read_shp", "read_dbf", "read_shapefile"]

POLYGON_TYPES = {5, 15, 25}   # Polygon, PolygonZ, PolygonM
POLYLINE_TYPES = {3, 13, 23}  # PolyLine, PolyLineZ, PolyLineM


class ShpError(Exception):
    pass


@dataclass
class ShapeRecord:
    number: int
    shape_type: int
    parts: list  # list of (n, 2) float64 arrays, projected units
    bbox: tuple

    @property
    def is_polygon(self) -> bool:
        return self.shape_type in POLYGON_TYPES

    def n_points(self) -> int:
        return int(sum(p.shape[0] for p in self.parts))

    def all_points(self) -> np.ndarray:
        if not self.parts:
            return np.zeros((0, 2))
        return np.vstack(self.parts)


def _read_shp_header(fh) -> dict:
    head = fh.read(100)
    if len(head) < 100:
        raise ShpError("file shorter than the 100-byte shapefile header")
    code, = struct.unpack(">i", head[0:4])
    if code != 9994:
        raise ShpError(f"not a shapefile (file code {code}, expected 9994)")
    length_words, = struct.unpack(">i", head[24:28])
    version, = struct.unpack("<i", head[28:32])
    shape_type, = struct.unpack("<i", head[32:36])
    bbox = struct.unpack("<4d", head[36:68])
    return {"length_bytes": length_words * 2, "version": version,
            "shape_type": shape_type, "bbox": bbox}


def read_shp(path: str | Path, max_records: int | None = None) -> tuple[list[ShapeRecord], dict]:
    path = Path(path)
    records: list[ShapeRecord] = []
    with open(path, "rb") as fh:
        header = _read_shp_header(fh)
        declared = header["length_bytes"]
        while True:
            rec_head = fh.read(8)
            if len(rec_head) < 8:
                break
            number, content_words = struct.unpack(">ii", rec_head)
            content = fh.read(content_words * 2)
            if len(content) < content_words * 2:
                raise ShpError(f"record {number} truncated: wanted {content_words * 2} bytes")
            st, = struct.unpack("<i", content[0:4])
            if st == 0:  # null shape
                records.append(ShapeRecord(number, st, [], (0.0, 0.0, 0.0, 0.0)))
                continue
            if st not in POLYGON_TYPES | POLYLINE_TYPES:
                raise ShpError(f"record {number}: shape type {st} not supported (2-D/3-D polylines and "
                               "polygons only)")
            if len(content) < 44:
                raise ShpError(f"record {number}: content too short for a {st} shape")
            bbox = struct.unpack("<4d", content[4:36])
            nparts, npoints = struct.unpack("<ii", content[36:44])
            off = 44
            parts_idx = struct.unpack(f"<{nparts}i", content[off:off + 4 * nparts]) if nparts else ()
            off += 4 * nparts
            need = off + 16 * npoints
            if len(content) < need:
                raise ShpError(f"record {number}: {npoints} points but only {len(content) - off} bytes")
            xy = np.frombuffer(content, dtype="<f8", count=2 * npoints, offset=off).reshape(npoints, 2)
            parts = []
            bounds = list(parts_idx) + [npoints]
            for a, b in zip(bounds[:-1], bounds[1:]):
                parts.append(np.array(xy[a:b], dtype=float))
            records.append(ShapeRecord(number, st, parts, bbox))
            if max_records is not None and len(records) >= max_records:
                break
    got = path.stat().st_size
    if abs(got - declared) > 0:
        # not fatal: some writers pad; report it so a truncated download is visible
        header["size_mismatch_bytes"] = got - declared
    return records, header


def read_dbf(path: str | Path) -> list[dict]:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 32:
        raise ShpError("dbf shorter than its header")
    nrec, = struct.unpack("<i", raw[4:8])
    hlen, = struct.unpack("<h", raw[8:10])
    rlen, = struct.unpack("<h", raw[10:12])
    fields = []
    off = 32
    while off < hlen and raw[off] != 0x0D:
        name = raw[off:off + 11].split(b"\x00")[0].decode("latin-1").strip()
        typ = chr(raw[off + 11])
        length = raw[off + 16]
        dec = raw[off + 17]
        fields.append((name, typ, length, dec))
        off += 32
    out = []
    pos = hlen
    for _ in range(nrec):
        rec = raw[pos:pos + rlen]
        pos += rlen
        if len(rec) < rlen:
            break
        if rec[0:1] == b"*":
            continue  # deleted record
        cursor = 1
        row = {}
        for name, typ, length, dec in fields:
            chunk = rec[cursor:cursor + length]
            cursor += length
            text = chunk.decode("latin-1").strip()
            if typ in ("N", "F"):
                # dBase writes '*' (or a run of them) when a value overflows its
                # field width; treat that as missing rather than as a number.
                if text in ("", ".", "-") or (text and set(text) == {"*"}):
                    row[name] = None
                else:
                    try:
                        row[name] = float(text)
                    except ValueError:
                        row[name] = None
            elif typ == "L":
                row[name] = None if text in ("", "?") else text.upper() == "T"
            elif typ == "D":
                row[name] = text or None
            else:
                row[name] = text
        out.append(row)
    return out


def read_shapefile(shp_path: str | Path, dbf_path: str | Path | None = None):
    """Return (records, attributes, header); attributes are zipped by record order."""
    records, header = read_shp(shp_path)
    attrs = read_dbf(dbf_path) if dbf_path else []
    if attrs and len(attrs) != len(records):
        raise ShpError(f"{len(records)} shapes but {len(attrs)} attribute rows")
    return records, attrs, header
