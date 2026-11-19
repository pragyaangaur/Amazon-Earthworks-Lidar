#!/usr/bin/env python3
"""Build a deduplicated 901-row processing queue from L3A and L1A inventories.

Run from project root after inventory_remote_archives.py:
  .venv/bin/python code/build_queue.py

Coordinates are kept in local checkpoint files only. Floodplain status is not
inferred from undocumented elevation/canopy columns: it remains `unassessed`.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import shapefile
from pyproj import Geod, Transformer
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "acre": ("Acre", "AC"), "rondonia": ("Rondônia", "RO"),
    "amazonas": ("Amazonas", "AM"), "mato grosso": ("Mato Grosso", "MT"),
    "para": ("Pará", "PA"), "roraima": ("Roraima", "RR"),
    "amapa": ("Amapá", "AP"), "maranhao": ("Maranhão", "MA"),
    "tocantins": ("Tocantins", "TO"),
}


def transect_id(member_name: str) -> str:
    base = member_name.rsplit("/", 1)[-1]
    match = re.match(r"NP_T-(.+?)(?:_dn|\.la[sz])$", base, re.I)
    if not match:
        raise ValueError(f"unrecognised member name {member_name}")
    return match.group(1)


def state_for_zip(zip_key: str) -> tuple[str, str]:
    lowered = zip_key.lower()
    for token, value in GROUPS.items():
        if token in lowered:
            return value
    raise ValueError(f"no state in {zip_key}")


def band(row: dict) -> int:
    state, lat = row["state"], row["centroid_lat"]
    if state == "AC":
        return 0
    if state == "RO":
        return 1
    if state == "AM" and lat < -4.5:
        return 2
    if state == "MT" and lat > -13.5:
        return 3
    if state == "AM":
        return 5
    if state == "MT":
        return 6
    if state in ("PA", "RR", "AP"):
        return 7
    return 9


def main() -> None:
    inventory = json.loads((ROOT / "data/metadata/l1a_inventory.json").read_text())
    known = pd.read_csv(ROOT / "data/known/earthworks.csv")
    by_id = collections.defaultdict(list)
    for item in inventory:
        by_id[transect_id(item["name"])].append(item)
    reader = shapefile.Reader(str(ROOT / "data/footprints/transectos_total_v20210616.shp"))
    transformer = Transformer.from_crs(5880, 4326, always_xy=True)
    geod = Geod(ellps="WGS84")
    rows = []
    for sr in reader.iterShapeRecords():
        record = sr.record
        tid = str(record["transect"])
        polygon = shape(sr.shape.__geo_interface__)
        centroid = polygon.centroid
        lon, lat = transformer.transform(centroid.x, centroid.y)
        _, _, distances = geod.inv(np.full(len(known), lon), np.full(len(known), lat),
                                   known.Longitude.to_numpy(), known.Latitude.to_numpy())
        nearest = int(np.nanargmin(distances))
        nearest_distance = float(distances[nearest])
        candidates = by_id.get(tid, [])
        # Deduplicate only exact uncompressed size + CRC matches. A conflicting
        # duplicate would need independent processing, not silent collapse.
        unique = {}
        for item in candidates:
            unique.setdefault((item["size"], item["CRC"]), []).append(item)
        chosen_group = None
        if unique:
            size_crc, copies = sorted(
                unique.items(),
                key=lambda pair: min(band({"state": state_for_zip(x["zip_key"])[1],
                                           "centroid_lat": lat}) for x in pair[1]))[0]
            copies = sorted(copies, key=lambda x: band({"state": state_for_zip(x["zip_key"])[1],
                                                       "centroid_lat": lat}))
            chosen = copies[0]
            state, uf = state_for_zip(chosen["zip_key"])
            chosen_group = {"record_id": chosen["record_id"], "zip_key": chosen["zip_key"],
                            "member": chosen["name"], "size": chosen["size"],
                            "CRC": chosen["CRC"], "duplicate_archive_copies": len(candidates),
                            "distinct_size_crc_variants": len(unique),
                            "other_archive_keys": sorted({x["zip_key"] for x in candidates
                                                         if x is not chosen})}
        else:
            state, uf = "Unknown (no L1A member)", ""
        row = {"transect": tid, "state": uf or "XX", "state_label": state,
               "centroid_lon": float(lon), "centroid_lat": float(lat),
               "criteria": str(record["criteria"]),
               "forest_sampling_proxy": "PRODES+TERRACLASS" if record["criteria"] == "PRODES+TERRACLASS" else "other",
               "floodplain_status": "unassessed",
               "nearest_known_db": str(known.iloc[nearest]["Database"]),
               "nearest_known_m": nearest_distance,
               "within_50km_known_cluster": nearest_distance <= 50000,
               "epsg_l3a": str(record["epsg"]),
               "raw_availability": "available" if unique else "missing_l1a_inventory",
               "source": chosen_group,
               "processing_status": "queued" if unique else "failed",
               "failure_reason": "" if unique else "No matching L1A member in the four public Zenodo records",
               "elev_mean_l3a": float(record["elev_mean"]) if record["elev_mean"] is not None else None,
               "canopy_rel_l3a": float(record["canopy_rel"]) if record["canopy_rel"] is not None else None}
        row["priority_band"] = band(row) if unique else 99
        row["priority_reason"] = (
            f"band {row['priority_band']}; " +
            ("within 50 km of known earthworks; " if row["within_50km_known_cluster"] else "outside 50 km known-earthwork buffer; ") +
            f"sampling criterion {row['criteria']}; floodplain unassessed")
        rows.append(row)
    for row in rows:
        row["within_band_sort"] = (
            0 if row["within_50km_known_cluster"] else 1,
            0 if row["forest_sampling_proxy"] == "PRODES+TERRACLASS" else 1,
            row["nearest_known_m"], row["transect"])
    rows.sort(key=lambda r: (r["priority_band"], r["within_band_sort"]))
    for priority, row in enumerate(rows, 1):
        row["priority"] = priority
        if row["transect"] == "0414":
            row["processing_status"] = "control_in_review"
            row["priority"] = 0
    fieldnames = ["priority", "transect", "state", "state_label", "centroid_lon", "centroid_lat",
                  "priority_band", "priority_reason", "criteria", "forest_sampling_proxy",
                  "floodplain_status", "nearest_known_db", "nearest_known_m",
                  "within_50km_known_cluster", "epsg_l3a", "raw_availability",
                  "source", "processing_status", "failure_reason", "elev_mean_l3a", "canopy_rel_l3a"]
    (ROOT / "checkpoint").mkdir(exist_ok=True)
    with (ROOT / "checkpoint/queue.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    queue_json = ROOT / "checkpoint/queue.json"
    queue_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "footprint_rows": len(rows),
        "raw_available_transects": sum(r["raw_availability"] == "available" for r in rows),
        "missing_l1a": [r["transect"] for r in rows if r["raw_availability"] != "available"],
        "raw_archive_members": len(inventory),
        "duplicate_extra_members": len(inventory) - sum(r["raw_availability"] == "available" for r in rows),
        "transects_with_multiple_size_crc_variants": [r["transect"] for r in rows
            if r["source"] and r["source"]["distinct_size_crc_variants"] > 1],
        "priority_bands": {str(k): int(v) for k, v in collections.Counter(r["priority_band"] for r in rows).items()},
        "states": dict(collections.Counter(r["state"] for r in rows)),
        "known_within_50km": sum(r["within_50km_known_cluster"] for r in rows),
        "floodplain_method": "unassessed: L3A has no such field; IPP S4 model requires bayesPO and was not used",
        "ordering": "Acre; Rondônia; southern Amazonas; northern Mato Grosso; other Amazonas; other Mato Grosso; Pará/Roraima/Amapá; Maranhão/Tocantins; missing L1A. Within band: known-earthwork proximity, forest-sampling proxy, distance.",
        "control_priority": {"0414": 0},
    }
    (ROOT / "checkpoint/queue_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
