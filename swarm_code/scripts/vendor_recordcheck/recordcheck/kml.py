"""Minimal KML reader + nearest-geometry check (no GDAL, no network).

The workspace holds a second public register, `data/known/amazon_geoglyphs.kml`
(Jacobs, 6298 placemarks: points, linestrings and polygons).  The deliverable
cites distances to named placemarks for its Known and Possible candidates, so
those numbers have to be reproducible.  This module parses the placemarks and
computes geodesic distances from a candidate to the nearest vertex and to the
nearest segment of each placemark.

Caveat that must survive into any report sentence: the KML's own header says its
database reviews the Science article's data, so a KML point near a Peripato
record is not an independent confirmation -- it is often the same source re-served.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geodesy as G

KML_NS = "{http://www.opengis.net/kml/2.2}"


@dataclass
class Placemark:
    name: str
    parts: list = field(default_factory=list)  # list of (n,2) arrays of lon,lat
    folder: str = ""                           # ' > '-joined KML folder path

    def n_points(self) -> int:
        return int(sum(len(p) for p in self.parts))


def _parse_coords(text: str) -> np.ndarray:
    rows = []
    for token in text.replace("\n", " ").split():
        bits = token.split(",")
        if len(bits) >= 2:
            try:
                rows.append((float(bits[0]), float(bits[1])))
            except ValueError:
                continue
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 2))


def read_kml(path: str | Path) -> list[Placemark]:
    """All placemarks with their folder path.

    The Jacobs file mixes the actual register (folders `geoglyphs`,
    `mound sites`, `outlines`, `lines`) with per-year `surveys` folders whose
    placemarks are named by date and are flight tracks/areas, not sites.  A
    distance quoted against a survey placemark is therefore not a distance to a
    recorded site, and the folder is carried here so a caller cannot confuse them.
    """
    tree = ET.parse(str(path))
    out: list[Placemark] = []

    def walk(node, path_parts):
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag in ("Document", "Folder"):
                nm = (child.findtext(f"{KML_NS}name") or "").strip()
                walk(child, path_parts + ([nm] if nm else []))
            elif tag == "Placemark":
                name = (child.findtext(f"{KML_NS}name") or "").strip()
                parts = []
                for geom in child.iter():
                    gtag = geom.tag.split("}")[-1]
                    if gtag in ("Point", "LineString", "LinearRing", "Track"):
                        coords = geom.find(f"{KML_NS}coordinates")
                        if coords is not None and coords.text:
                            arr = _parse_coords(coords.text)
                            if len(arr):
                                parts.append(arr)
                if parts:
                    out.append(Placemark(name=name, parts=parts, folder=" > ".join(path_parts)))
            else:
                walk(child, path_parts)

    walk(tree.getroot(), [])
    return out


def is_survey(pm: Placemark) -> bool:
    return "surveys" in pm.folder.lower()


def nearest_placemark(lon: float, lat: float, placemarks: list[Placemark],
                      refine_segments: bool = True) -> dict:
    """Nearest placemark by geodesic distance to its vertices (and segments)."""
    best = {"name": None, "distance_m": math.inf, "n_points": 0}
    for pm in placemarks:
        pts = np.vstack(pm.parts)
        h = G.haversine(lat, lon, pts[:, 1], pts[:, 0])
        if not np.isfinite(h).any():
            continue
        j = int(np.nanargmin(h))
        d = float(G.vincenty_inverse(lat, lon, float(pts[j, 1]), float(pts[j, 0]))[0])
        if refine_segments:
            # refine against the segments adjacent to the closest vertex, using a
            # local tangent plane (segments in this KML are short)
            for k in range(max(0, j - 1), min(len(pts) - 2, j + 1) + 1):
                if k + 1 >= len(pts):
                    continue
                a, b = pts[k], pts[k + 1]
                cand = _point_segment_m(lon, lat, a, b)
                if cand < d:
                    d = cand
        if d < best["distance_m"]:
            best = {"name": pm.name, "distance_m": d, "n_points": pm.n_points(),
                    "folder": pm.folder, "is_survey": is_survey(pm)}
    return best


def _point_segment_m(lon: float, lat: float, a: np.ndarray, b: np.ndarray) -> float:
    lat0 = math.radians(lat)
    mx = 111_320.0 * math.cos(lat0)
    my = 110_574.0
    ax, ay = (a[0] - lon) * mx, (a[1] - lat) * my
    bx, by = (b[0] - lon) * mx, (b[1] - lat) * my
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(px, py)


def distances_to_named(placemarks: list[Placemark], lon: float, lat: float,
                       names: list[str]) -> list[dict]:
    """Distances to specific placemarks by name (case-insensitive)."""
    wanted = {n.lower() for n in names}
    out = []
    for pm in placemarks:
        if pm.name.lower() in wanted:
            pts = np.vstack(pm.parts)
            h = G.haversine(lat, lon, pts[:, 1], pts[:, 0])
            j = int(np.nanargmin(h))
            d = float(G.vincenty_inverse(lat, lon, float(pts[j, 1]), float(pts[j, 0]))[0])
            out.append({"name": pm.name, "distance_m": round(d, 1), "n_points": pm.n_points()})
    return sorted(out, key=lambda r: r["distance_m"])
