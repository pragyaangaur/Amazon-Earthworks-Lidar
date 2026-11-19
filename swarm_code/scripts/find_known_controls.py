#!/usr/bin/env python3
"""Find published earthwork records inside public EBA transect footprints.

Local-only geometry check for the under-forest positive-control side task.  It
never queries a network service and never reads candidate coordinates.  Footprint
polygons are transformed from their stated EPSG:5880 CRS to WGS84; Peripato uses
point intersection.  The cached Jacobs file has centroids/first vertices, not
complete outlines, so those matches are explicitly *centroid screening*, not
proof that a mapped polygon overlaps the lidar.

Output is a local control-research lead list, not a set of detections.  The
transect-wide canopy_rel_l3a index cannot establish canopy at a particular site.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import shape, Point
from shapely.ops import transform
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/literature/known_control_footprint_intersections.json"


def main() -> None:
    prj = Path(ROOT / "data/footprints/transectos_total_v20210616.prj").read_text()
    source_crs = CRS.from_wkt(prj)
    transformer = Transformer.from_crs(source_crs, 4326, always_xy=True)
    queue = {r["transect"]: r for r in json.loads((ROOT / "checkpoint/queue.json").read_text())}
    sf = shapefile.Reader(str(ROOT / "data/footprints/transectos_total_v20210616.shp"))
    field_names = [field[0] for field in sf.fields[1:]]
    transects, polygons = [], []
    for index, record in enumerate(sf.iterRecords()):
        values = dict(zip(field_names, record))
        tid = values["transect"].split("_")[0].zfill(4)
        geometry = shape(sf.shape(index).__geo_interface__)
        projected = transform(transformer.transform, geometry)
        if not projected.is_valid:
            projected = projected.buffer(0)
        transects.append({"transect": tid, "canopy_rel_l3a": values.get("canopy_rel"),
                          "epsg_l3a": values.get("epsg"), "record": values})
        polygons.append(projected)
    tree = STRtree(polygons)

    peripato = []
    with (ROOT / "data/known/earthworks.csv").open(newline="") as handle:
        for record_index, row in enumerate(csv.DictReader(handle)):
            point = Point(float(row["Longitude"]), float(row["Latitude"]))
            for polygon_index in tree.query(point, predicate="intersects"):
                base = transects[int(polygon_index)]
                tid = base["transect"]
                qr = queue.get(tid, {})
                peripato.append({"record_index": record_index, "database": row["Database"],
                                 "longitude": point.x, "latitude": point.y,
                                 "transect": tid, "canopy_rel_l3a": base["canopy_rel_l3a"],
                                 "processing_status": qr.get("processing_status", "not_in_queue"),
                                 "priority": qr.get("priority"), "raw_availability": qr.get("raw_availability"),
                                 "match_kind": "point_intersects_public_footprint"})

    jacobs = []
    geometry = json.loads((ROOT / "data/known/amazon_geoglyphs_geometry.json").read_text())
    for record_index, row in enumerate(geometry):
        # `centroid` is lon/lat in this cached geometry summary.  A one-coordinate
        # placemark is a real point; all others remain centroid leads only.
        point = Point(float(row["centroid"][0]), float(row["centroid"][1]))
        for polygon_index in tree.query(point, predicate="intersects"):
            base = transects[int(polygon_index)]
            tid = base["transect"]
            qr = queue.get(tid, {})
            jacobs.append({"record_index": record_index, "name": row.get("name", ""),
                           "description": row.get("description", ""), "n_coords": row.get("n_coords"),
                           "longitude": point.x, "latitude": point.y, "transect": tid,
                           "canopy_rel_l3a": base["canopy_rel_l3a"],
                           "processing_status": qr.get("processing_status", "not_in_queue"),
                           "priority": qr.get("priority"), "raw_availability": qr.get("raw_availability"),
                           "match_kind": "placemark_point_intersects" if row.get("n_coords") == 1
                           else "centroid_lead_requires_outline_verification"})

    peripato.sort(key=lambda r: (-float(r["canopy_rel_l3a"] or 0), r["priority"] or 9999,
                                 r["transect"], r["record_index"]))
    jacobs.sort(key=lambda r: (r["match_kind"] != "placemark_point_intersects",
                               -float(r["canopy_rel_l3a"] or 0), r["priority"] or 9999,
                               r["transect"], r["record_index"]))
    result = {
        "schema": "known-control-footprint-intersections/1",
        "method": "Local Shapely point/polygon intersection after EPSG:5880 footprint-to-WGS84 transformation.",
        "source_crs": source_crs.to_string(),
        "footprint_count": len(polygons),
        "peripato_input_rows": sum(1 for _ in csv.DictReader((ROOT / "data/known/earthworks.csv").open())),
        "jacobs_cached_geometry_groups": len(geometry),
        "peripato_match_count": len(peripato),
        "jacobs_match_count": len(jacobs),
        "canopy_limit": "canopy_rel_l3a is transect-wide and does not prove forest at the record position.",
        "association_limit": "Being inside a footprint does not establish which detected landform, if any, is the record.",
        "peripato": peripato,
        "jacobs": jacobs,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("source_crs", "footprint_count", "peripato_match_count", "jacobs_match_count")}, indent=2))
    print("Top Peripato canopy-proxy controls:")
    for row in peripato[:20]:
        print(json.dumps(row, ensure_ascii=False))
    print("First Jacobs point-control leads:")
    for row in [r for r in jacobs if r["match_kind"] == "placemark_point_intersects"][:20]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
