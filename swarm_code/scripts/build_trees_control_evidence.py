#!/usr/bin/env python3
"""Build local native evidence for published TREES/INPE footprint controls.

Example: .venv/bin/python code/build_trees_control_evidence.py 0470

No network and no candidate-coordinate input.  The points come from the local
Peripato footprint-intersection audit.  The output is a control assessment, not
an earthwork detection and not evidence that a displaced register point matches
any particular landform in the crop.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from process_laz import colorize


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_trees_control_evidence.py <transect>")
    tid = sys.argv[1]
    base = ROOT / "data/derived_v2" / tid
    registry = json.loads((ROOT / "data/literature/known_control_footprint_intersections.json").read_text())
    records = [r for r in registry["peripato"] if r["transect"] == tid and r["database"] == "TREES/INPE"]
    if not records:
        raise SystemExit(f"no TREES/INPE point intersection registered for {tid}")
    summary = json.loads((base / "summary.json").read_text())
    grid = summary["grid"]
    epsg = int(summary["epsg"])
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    for record in records:
        record["x"], record["y"] = transformer.transform(record["longitude"], record["latitude"])
    cell = float(grid["cell"])
    margin = 300.0
    c0 = max(0, int(math.floor((min(r["x"] for r in records) - margin - grid["x0"]) / cell)))
    c1 = min(grid["nx"], int(math.ceil((max(r["x"] for r in records) + margin - grid["x0"]) / cell)) + 1)
    r0 = max(0, int(math.floor((min(r["y"] for r in records) - margin - grid["y0"]) / cell)))
    r1 = min(grid["ny"], int(math.ceil((max(r["y"] for r in records) + margin - grid["y0"]) / cell)) + 1)
    out = base / "control_trees_inpe"
    out.mkdir(exist_ok=True)
    source = np.load(base / "rasters.npz")
    product = np.load(base / "products.npz")
    dtm = source["dtm"]
    chm = source["chm"]
    canopy_threshold = float(summary["canopy"]["threshold_m"])
    forest = ndi.binary_dilation(np.nan_to_num(chm) > canopy_threshold, iterations=1)
    detect = np.isfinite(dtm) & ~forest
    if (base / "masks_control.npz").exists():
        with np.load(base / "masks_control.npz") as masks:
            if not np.array_equal(detect, masks["detect_mask"]):
                raise ValueError("reconstructed detect mask differs from preserved source mask")
            detect = masks["detect_mask"]
            forest = masks["forest_mask"]
    layers = {"dtm": dtm, "chm": chm, "hillshade": source["hillshade"],
              "svf": source["svf"], "relief_15": source["relief_15"],
              "relief_pre_15": source["relief_pre_15"],
              "detect_mask": detect.astype(float), "reconstructed": product["removed"].astype(float)}
    pixel_rows = []
    colours = [(255, 30, 30), (255, 0, 255), (0, 255, 255), (255, 130, 0)]
    for record in records:
        row = int((record["y"] - grid["y0"]) / cell)
        col = int((record["x"] - grid["x0"]) / cell)
        record["pixel_row"] = row
        record["pixel_col"] = col
        pixel_rows.append((record, row, col))
    for name, array in layers.items():
        crop = array[r0:r1, c0:c1]
        kind = "chm" if name == "chm" else ("relief15" if "relief" in name else name)
        if name in {"detect_mask", "reconstructed"}:
            rgb = np.zeros((*crop.shape, 3), dtype=np.uint8)
            rgb[..., 0] = np.where(np.isfinite(crop), 255 * crop, 0).astype(np.uint8)
            rgb[..., 1] = rgb[..., 0]
            rgb[..., 2] = rgb[..., 0]
        else:
            rgb = colorize(crop, kind)
        image = Image.fromarray(rgb[::-1])
        draw = ImageDraw.Draw(image)
        for index, (record, row, col) in enumerate(pixel_rows):
            x = col - c0
            y = (r1 - r0 - 1) - (row - r0)
            colour = colours[index % len(colours)]
            draw.line((x - 4, y, x + 4, y), fill=colour, width=1)
            draw.line((x, y - 4, x, y + 4), fill=colour, width=1)
            draw.text((x + 5, y - 5), str(record["record_index"]), fill=colour)
        image.save(out / f"{name}.png", optimize=True)
    measurements = []
    yy, xx = np.mgrid[:dtm.shape[0], :dtm.shape[1]]
    for record, row, col in pixel_rows:
        window = (xx - col) ** 2 + (yy - row) ** 2 <= (100.0 / cell) ** 2
        observed = ~source["hole"].astype(bool)
        valid_values = chm[window & observed & np.isfinite(chm)]
        measurements.append({
            "record_index": record["record_index"],
            "window_radius_m": 100.0,
            "window_cells": int(window.sum()),
            "observed_pct": float(100 * observed[window].mean()),
            "detect_mask_pct": float(100 * detect[window].mean()),
            "reconstructed_pct_of_observed": float(100 * product["removed"][window & observed].mean()) if (window & observed).any() else None,
            "canopy_median_m_observed": float(np.median(valid_values)) if valid_values.size else None,
            "canopy_p90_m_observed": float(np.percentile(valid_values, 90)) if valid_values.size else None,
            "note": "Window measurements show available detector support; they do not identify or reject the registered earthwork."})
    metadata = {
        "transect": tid,
        "status": "native_control_review_pending",
        "public_record_source": "Peripato 2023 Earthworks.rds / TREES-INPE",
        "record_displacement_assumption_m": 2000.0,
        "association_warning": "TREES/INPE coordinates may be displaced. Point proximity is not identity; this crop alone neither confirms nor rejects the registered site.",
        "epsg": epsg,
        "source_grid": grid,
        "crop": {"row0": r0, "row1": r1, "col0": c0, "col1": c1, "pixel_size_m": cell,
                 "x0": grid["x0"] + c0 * cell, "y0": grid["y0"] + r0 * cell,
                 "raster_row_order": "south-up", "png_orientation": "north-up"},
        "records": records,
        "measurements": measurements,
        "raw_ring_detection_count": len(json.loads((base / "detections/rings_v3.json").read_text())["detections"]),
        "raw_mound_cluster_count": len(json.loads((base / "detector/mound_clusters.json").read_text())),
        "layers": list(layers),
    }
    (out / "control.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"transect": tid, "records": len(records), "crop_pixels": [r1-r0, c1-c0],
                      "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
