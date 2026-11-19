#!/usr/bin/env python3
"""Create native-resolution visual checks for rejected mound clusters.

  .venv/bin/python code/review_mounds.py data/derived_v2/0414

These are reviews of m4's rejected clusters, not new candidates or certainty
upgrades. Red dots are detected bumps; the green ellipse is the fitted ring.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_laz import colorize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    args = p.parse_args()
    summary = json.loads((args.derived / "summary.json").read_text())
    grid = summary["grid"]
    source = np.load(args.derived / "rasters.npz")
    clusters = json.loads((args.derived / "detector/mound_clusters.json").read_text())
    out = args.derived / "mound_review"
    out.mkdir(exist_ok=True)
    contact = []
    for cluster in clusters:
        row, col = cluster["cell_rowcol"]
        radius = float(cluster["ring_radius_m"])
        half = max(150.0, 1.5 * radius)
        w = int(math.ceil(half / grid["cell"]))
        r0, r1 = max(0, int(row) - w), min(grid["ny"], int(row) + w + 1)
        c0, c1 = max(0, int(col) - w), min(grid["nx"], int(col) + w + 1)
        names = ["dtm", "chm", "hillshade", "svf", "relief_15", "relief_pre_15"]
        crops = {}
        for name in names:
            if name not in source.files:
                continue
            a = source[name][r0:r1, c0:c1]
            crops[name] = a
            kind = "chm" if name == "chm" else ("relief15" if "relief" in name else name)
            Image.fromarray(colorize(a, kind)[::-1]).save(out / f"{cluster['id']}_{name}.png", optimize=True)
        base = colorize(crops.get("relief_pre_15", crops["relief_15"]), "relief15")[::-1]
        image = Image.fromarray(base)
        draw = ImageDraw.Draw(image)
        cx, cy = col - c0, grid["ny"] - 1 - (row - r0)  # PNG north-up within crop
        # Correct north-up crop coordinates: source row r0 is south; last PNG row
        # corresponds to source r0, so y_png = (r1-r0-1)-(row-r0).
        cy = (r1 - r0 - 1) - (row - r0)
        rp = radius / grid["cell"]
        draw.ellipse((cx-rp, cy-rp, cx+rp, cy+rp), outline=(0, 220, 0), width=1)
        for mound in cluster["mounds"]:
            mx, my = mound["col"] - c0, (r1 - r0 - 1) - (mound["row"] - r0)
            draw.ellipse((mx-2, my-2, mx+2, my+2), fill=(230, 25, 25))
        image.save(out / f"{cluster['id']}_pre_relief_annotated.png", optimize=True)
        notes = {k: cluster.get(k) for k in ("id", "n_mounds", "ring_radius_m",
                 "ring_rms_frac", "angular_coverage", "plaza_clear", "height_median_m",
                 "certainty", "screen_reason")}
        review_status = ("automatic_rejection_visual_reviewed"
                         if cluster.get("certainty") == "Rejected"
                         else "automatic_possible_requires_coordinator_review")
        notes.update({"status": review_status, "pixel_size_m": grid["cell"],
                      "review": "native-resolution visual products created for coordinator review; no automatic certainty change"})
        (out / f"{cluster['id']}.json").write_text(json.dumps(notes, indent=2) + "\n")
        contact.append((cluster["id"], base, notes))
    if contact:
        thumb = 260
        sheet = Image.new("RGB", (2*thumb, ((len(contact)+1)//2)*(thumb+25)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (fid, array, notes) in enumerate(contact):
            x, y = (i % 2)*thumb, (i//2)*(thumb+25)
            im = Image.fromarray(array).resize((thumb, thumb), Image.Resampling.NEAREST)
            sheet.paste(im, (x, y+25))
            draw.text((x+3, y+4), f"{fid} | {notes['screen_reason']}", fill="black")
        sheet.save(out / "contact.jpg", quality=90)
    print(f"Mound review: {len(clusters)} rejected clusters written to {out}")


if __name__ == "__main__":
    main()
