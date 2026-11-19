#!/usr/bin/env python3
"""Adapt m2 v3 products to the coordinator review/skeptic NPZ contract.

  .venv/bin/python code/adapt_m2_products.py data/derived_v2/0414

Writes grid.json and products.npz beside rasters.npz. Relief arrays in this
compatibility product are blanked within 15 m of original no-return cells before
review images are made; rasters.npz is not modified.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import numpy as np
from scipy import ndimage as ndi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("--stem", default="rings_v3")
    args = p.parse_args()
    summary = json.loads((args.derived / "summary.json").read_text())
    grid = {**summary["grid"], "epsg": summary["epsg"], "row_order": "south-up"}
    (args.derived / "grid.json").write_text(json.dumps(grid, indent=2) + "\n")
    source = np.load(args.derived / "rasters.npz")
    dtm = source["dtm"]
    original_hole = source["hole"].astype(bool)
    unfilled = ~np.isfinite(dtm)
    hole_distance = ndi.distance_transform_edt(~original_hole) * grid["cell"]
    relief_hole_mask = hole_distance <= 15.0
    removed = np.zeros_like(original_hole)
    interpolated = np.zeros_like(original_hole)
    masks_path = args.derived / "masks.npy"
    if masks_path.exists():
        masks = np.load(masks_path, allow_pickle=True).item()
        removed = masks["removed"].astype(bool)
        interpolated = masks["filled"].astype(bool)
    arrays = {"dtm": dtm, "chm": source["chm"], "hillshade": source["hillshade"],
              "svf": source["svf"], "count": np.zeros(dtm.shape, np.uint32),
              "original_hole": original_hole, "unfilled": unfilled,
              "removed": removed, "interpolated": interpolated,
              "relief_hole_mask": relief_hole_mask}
    for sigma in (10, 15, 30):
        array = source[f"relief_{sigma}"].copy()
        array[relief_hole_mask | unfilled] = np.nan
        arrays[f"relief{sigma}"] = array
    temporary = args.derived / "products.npz.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.derived / "products.npz")
    dtm_file = args.derived / "dtm.npz"
    with dtm_file.open("wb") as handle:
        np.savez_compressed(handle, dtm=dtm, x0=grid["x0"], y0=grid["y0"],
                            cell=grid["cell"], epsg=grid["epsg"], row_order="south-up")
    # Detector CSV -> generic review JSON. Detector lon/lat may be blank, so we
    # always calculate them from native x/y with pyproj in the review scripts.
    detections_json = json.loads((args.derived / "detector/detections.json").read_text())
    profiles = {row["id"]: row for row in detections_json}
    features = []
    with (args.derived / "detector/detections.csv").open() as handle:
        for row in csv.DictReader(handle):
            profile = profiles.get(row["id"], {})
            features.append({"id": row["id"], "x": float(row["x"]), "y": float(row["y"]),
                             "lon": float(row["lon"]) if row["lon"] else None,
                             "lat": float(row["lat"]) if row["lat"] else None,
                             "radius_m": float(row["size_m"]) / 2,
                             "diameter_m": float(row["size_m"]),
                             "dip_m": float(row["relief_m"]),
                             "shape": row.get("type") or profile.get("shape"),
                             "flags": (profile.get("flags") or []) + [row.get("screen_reason", "")],
                             "screen_certainty": row["certainty"],
                             "certainty": "unreviewed_detection"})
    detections_dir = args.derived / "detections"
    detections_dir.mkdir(exist_ok=True)
    (detections_dir / f"{args.stem}.json").write_text(json.dumps({
        "transect": summary.get("source", args.derived.name), "status": "unreviewed_detections_from_m4",
        "input_layer": "lrm15", "detections": features}, indent=2) + "\n")
    # CSV with the exact native header expected by the record audit and review.
    csv_target = detections_dir / f"{args.stem}.csv"
    (args.derived / "recordcheck").mkdir(exist_ok=True)
    with csv_target.open("w", newline="") as handle:
        fieldnames = ["id", "transect", "EPSG", "x", "y", "lon", "lat", "radius_m", "dip_m", "shape", "certainty"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            writer.writerow({"id": feature["id"], "transect": args.derived.name,
                             "EPSG": grid["epsg"], "x": feature["x"], "y": feature["y"],
                             "lon": feature["lon"] or "", "lat": feature["lat"] or "",
                             "radius_m": feature["radius_m"], "dip_m": feature["dip_m"],
                             "shape": feature["shape"], "certainty": feature["certainty"]})
    print(f"Adapted {args.derived}: {len(features)} ring detections; relief hole mask 15m")


if __name__ == "__main__":
    main()
