#!/usr/bin/env python3
"""Create private visual/profile evidence packs for unreviewed detections.

Example:
  .venv/bin/python code/make_detection_reviews.py data/derived/0414 \
    --stem rings_relief15_recall --audited \
    data/derived/0414/recordcheck/rings_relief15_recall_audited.csv

No certainty is assigned here. Native evidence PNGs remain 3 m per pixel;
contact sheets are thumbnails only and are not measurement products.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_laz import write_png

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("derived", type=Path)
    parser.add_argument("--stem", default="rings_relief15_recall")
    parser.add_argument("--audited", type=Path)
    args = parser.parse_args()
    detections = json.loads((args.derived / "detections" / f"{args.stem}.json").read_text())
    grid = json.loads((args.derived / "grid.json").read_text())
    products = np.load(args.derived / "products.npz")
    audited = {}
    if args.audited:
        with args.audited.open() as handle:
            audited = {row["id"]: row for row in csv.DictReader(handle)}
    output = args.derived / "review"
    output.mkdir(exist_ok=True)
    layers = ("dtm", "chm", "hillshade", "svf", "relief10", "relief15", "relief30")
    manifest = []
    contact_items = []
    for det in detections["detections"]:
        fid = det["id"]
        radius = float(det.get("radius_m", 50))
        half = max(75.0, min(400.0, 1.65 * radius))
        cell = float(grid["cell"])
        row = int(round((det["y"] - grid["y0"]) / cell - 0.5))
        col = int(round((det["x"] - grid["x0"]) / cell - 0.5))
        width = int(math.ceil(half / cell))
        r0, r1 = max(0, row - width), min(grid["ny"], row + width + 1)
        c0, c1 = max(0, col - width), min(grid["nx"], col + width + 1)
        if not (r0 <= row < r1 and c0 <= col < c1):
            raise ValueError(f"detection {fid} lies outside raster")
        target = output / fid
        target.mkdir(exist_ok=True)
        crops = {}
        for name in layers:
            crops[name] = products[name][r0:r1, c0:c1]
            write_png(target / f"{name}.png", crops[name], name)
        # Support mask: white = measured relief support; grey = hole-halo mask;
        # black = unfilled/outside data. This is a display, not a new measurement.
        unfilled = products["unfilled"][r0:r1, c0:c1]
        hole_mask = products["relief_hole_mask"][r0:r1, c0:c1]
        support = np.zeros(crops["dtm"].shape + (3,), np.uint8)
        support[~unfilled] = (110, 110, 110)
        support[np.isfinite(crops["relief15"])] = (240, 240, 240)
        support[hole_mask & ~np.isfinite(crops["relief15"])] = (65, 65, 65)
        Image.fromarray(support[::-1]).save(target / "support.png", optimize=True)

        yy = grid["y0"] + (np.arange(r0, r1) + 0.5) * cell
        xx = grid["x0"] + (np.arange(c0, c1) + 0.5) * cell
        distance = np.hypot(xx[None, :] - det["x"], yy[:, None] - det["y"])
        max_radius = min(half, float(np.nanmax(distance)))
        bins = np.arange(0, max_radius + cell, cell)
        radial = []
        relief = crops["relief15"]
        for lo, hi in zip(bins[:-1], bins[1:]):
            selected = (distance >= lo) & (distance < hi) & np.isfinite(relief)
            values = relief[selected]
            radial.append({"radius_inner_m": lo, "radius_outer_m": hi,
                           "radius_mid_m": (lo + hi) / 2, "n": int(values.size),
                           "relief15_median_m": float(np.median(values)) if values.size else "",
                           "relief15_q25_m": float(np.percentile(values, 25)) if values.size else "",
                           "relief15_q75_m": float(np.percentile(values, 75)) if values.size else ""})
        with (target / "radial_profile.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(radial[0]))
            writer.writeheader()
            writer.writerows(radial)
        cross = []
        local_r, local_c = row - r0, col - c0
        for orientation, zvalues, rvalues, offsets in (
            ("east_west", crops["dtm"][local_r, :], relief[local_r, :], xx - det["x"]),
            ("north_south", crops["dtm"][:, local_c], relief[:, local_c], yy - det["y"]),
        ):
            for offset, z, local in zip(offsets, zvalues, rvalues):
                cross.append({"orientation": orientation, "offset_m": float(offset),
                              "dtm_m": float(z) if np.isfinite(z) else "",
                              "relief15_m": float(local) if np.isfinite(local) else ""})
        with (target / "cross_profiles.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cross[0]))
            writer.writeheader()
            writer.writerows(cross)
        audit = audited.get(fid, {})
        notes = {"id": fid, "status": "unreviewed",
                 "natural_modern_artefact_explanations": [],
                 "drainage_context": "pending", "hole_mask_review": "pending",
                 "canopy_review": "pending", "edge_review": "pending",
                 "decision": "pending", "reason": "",
                 "nearest_known_db": audit.get("nearest_known_db", ""),
                 "nearest_known_m": audit.get("nearest_known_m", ""),
                 "table_match": audit.get("d_table_match", ""),
                 "literature_checked": "pending coordinator review if feature survives skeptic"}
        (target / "skeptic_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
        entry = {"id": fid, "crop": {"row": r0, "col": c0, "row_end": r1, "col_end": c1},
                 "native_pixel_size_m": cell,
                 "detector_radius_m": radius,
                 "detector_dip_m_lrm_statistic": det.get("dip_m"),
                 "detector_shape": det.get("shape"), "detector_flags": det.get("flags"),
                 "nearest_known_db": audit.get("nearest_known_db", ""),
                 "nearest_known_m": audit.get("nearest_known_m", ""),
                 "table_match": audit.get("d_table_match", ""),
                 "status": "unreviewed"}
        manifest.append(entry)
        contact_items.append((fid, crops["hillshade"], crops["relief15"], crops["chm"], entry))
    (output / "manifest.json").write_text(json.dumps({"transect": detections["transect"],
        "source_detections": f"detections/{args.stem}.json", "status": "unreviewed",
        "features": manifest}, indent=2) + "\n")

    for page, start in enumerate(range(0, len(contact_items), 5), 1):
        subset = contact_items[start:start + 5]
        cell_w, cell_h, label_h = 220, 220, 32
        sheet = Image.new("RGB", (3 * cell_w, len(subset) * (cell_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, (fid, hill, local, canopy, entry) in enumerate(subset):
            y0 = row_index * (cell_h + label_h)
            draw.text((4, y0 + 4), f"{fid} | {entry['detector_shape']} | r={entry['detector_radius_m']:.0f}m", fill="black")
            for column, (name, array) in enumerate((("hillshade", hill), ("relief15", local), ("canopy", canopy))):
                rgb = __import__("process_laz").colorize(array, "chm" if name == "canopy" else name)
                image = Image.fromarray(rgb[::-1])
                # Thumbnails are explicitly enlarged for review; native PNGs
                # retain the measured 3 m pixels used for all decisions.
                image = image.resize((cell_w, cell_h), Image.Resampling.NEAREST)
                draw.text((column * cell_w + 4, y0 + 18), name, fill="black")
                sheet.paste(image, (column * cell_w, y0 + label_h))
        draw.text((4, sheet.height - 14), "Thumbnails only; use native 3m evidence PNGs for decisions.", fill="black")
        sheet.save(output / f"contact_{page:02d}.jpg", quality=88)
    print(f"Review packs complete: {len(manifest)} detections; {(output / 'manifest.json')}")


if __name__ == "__main__":
    main()
