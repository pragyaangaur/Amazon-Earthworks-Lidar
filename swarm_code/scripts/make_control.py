#!/usr/bin/env python3
"""Generate private positive-control crops and profiles for transect 0414.

Run after process_laz.py and creation of data/derived/0414/known_records.json:
  .venv/bin/python code/make_control.py --transect 0414

The crop is centred on the exact Peripato record inside the footprint, not on
an adjusted image correlation. Products are 3 m per pixel. Coordinates stay in
local files for the private deliverable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_laz import write_png

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transect", default="0414")
    parser.add_argument("--radius-m", type=float, default=150.0)
    args = parser.parse_args()
    base = ROOT / "data/derived" / args.transect
    known = json.loads((base / "known_records.json").read_text())
    if len(known["inside"]) != 1:
        raise ValueError(f"expected exactly one known record inside control, got {len(known['inside'])}")
    site = known["inside"][0]
    grid = json.loads((base / "grid.json").read_text())
    products = np.load(base / "products.npz")
    transformer = Transformer.from_crs(4326, grid["epsg"], always_xy=True)
    x, y = transformer.transform(site["Longitude"], site["Latitude"])
    cell = grid["cell"]
    center_col = (x - grid["x0"]) / cell - 0.5
    center_row = (y - grid["y0"]) / cell - 0.5
    radius_cells = int(math.ceil(args.radius_m / cell))
    r = int(round(center_row))
    c = int(round(center_col))
    r0, r1 = max(0, r - radius_cells), min(grid["ny"], r + radius_cells + 1)
    c0, c1 = max(0, c - radius_cells), min(grid["nx"], c + radius_cells + 1)
    if not (r0 <= r < r1 and c0 <= c < c1):
        raise ValueError("known control falls outside LAZ raster bounds")
    out = base / "control"
    out.mkdir(exist_ok=True)
    names = ["dtm", "chm", "hillshade", "svf", "relief10", "relief15", "relief30"]
    crops = {}
    for name in names:
        crops[name] = products[name][r0:r1, c0:c1]
        write_png(out / f"{name}.png", crops[name], name)

    dtm = crops["dtm"]
    relief = crops["relief15"]
    yy = grid["y0"] + (np.arange(r0, r1) + 0.5) * cell
    xx = grid["x0"] + (np.arange(c0, c1) + 0.5) * cell
    dx = xx[None, :] - x
    dy = yy[:, None] - y
    distance = np.hypot(dx, dy)
    bin_width = cell
    bins = np.arange(0, args.radius_m + bin_width, bin_width)
    radial_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        select = (distance >= lo) & (distance < hi) & np.isfinite(relief)
        values = relief[select]
        radial_rows.append({
            "radius_inner_m": lo, "radius_outer_m": hi,
            "radius_mid_m": (lo + hi) / 2, "n": int(values.size),
            "relief15_median_m": float(np.median(values)) if values.size else "",
            "relief15_q25_m": float(np.percentile(values, 25)) if values.size else "",
            "relief15_q75_m": float(np.percentile(values, 75)) if values.size else "",
        })
    with (out / "radial_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(radial_rows[0]))
        writer.writeheader()
        writer.writerows(radial_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    mids = np.array([row["radius_mid_m"] for row in radial_rows])
    med = np.array([row["relief15_median_m"] if row["relief15_median_m"] != "" else np.nan
                    for row in radial_rows], dtype=float)
    q25 = np.array([row["relief15_q25_m"] if row["relief15_q25_m"] != "" else np.nan
                    for row in radial_rows], dtype=float)
    q75 = np.array([row["relief15_q75_m"] if row["relief15_q75_m"] != "" else np.nan
                    for row in radial_rows], dtype=float)
    ax.fill_between(mids, q25, q75, alpha=.2, label="annular IQR")
    ax.plot(mids, med, marker=".", label="annular median")
    ax.axhline(0, color="black", linewidth=.6)
    ax.set(xlabel="Radius from recorded site (m)", ylabel="15 m local relief (m)",
           title="0414 positive control: radial profile at exact record")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "radial_profile.png", dpi=160)
    plt.close(fig)

    cross_rows = []
    for orientation, values, relief_values, offsets in (
        ("east_west", dtm[r - r0, :], relief[r - r0, :], xx - x),
        ("north_south", dtm[:, c - c0], relief[:, c - c0], yy - y),
    ):
        for offset, z, local in zip(offsets, values, relief_values):
            cross_rows.append({"orientation": orientation, "offset_m": float(offset),
                               "dtm_m": float(z) if np.isfinite(z) else "",
                               "relief15_m": float(local) if np.isfinite(local) else ""})
    with (out / "cross_profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_rows[0]))
        writer.writeheader()
        writer.writerows(cross_rows)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, orientation in zip(axes, ("east_west", "north_south")):
        rows = [row for row in cross_rows if row["orientation"] == orientation]
        offsets = [row["offset_m"] for row in rows]
        values = [row["relief15_m"] if row["relief15_m"] != "" else np.nan for row in rows]
        ax.plot(offsets, values, marker=".")
        ax.axhline(0, color="black", linewidth=.6)
        ax.set(ylabel="15 m relief (m)", title=orientation.replace("_", "-"))
    axes[-1].set_xlabel("Offset from recorded site (m)")
    fig.tight_layout()
    fig.savefig(out / "cross_profiles.png", dpi=160)
    plt.close(fig)

    within_50 = distance <= 50
    within_100 = distance <= 100
    metrics = {
        "transect": args.transect, "status": "metrics_generated_visual_review_pending",
        "record_database": site["Database"], "record_lon": site["Longitude"],
        "record_lat": site["Latitude"], "projected_epsg": grid["epsg"],
        "projected_x": float(x), "projected_y": float(y),
        "crop_origin": {"row": r0, "col": c0, "row_end": r1, "col_end": c1},
        "pixel_size_m": cell, "crop_radius_m": args.radius_m,
        "valid_dtm_fraction_within_50m": float(np.isfinite(dtm[within_50]).mean()),
        "valid_relief15_fraction_within_50m": float(np.isfinite(relief[within_50]).mean()),
        "valid_dtm_fraction_within_100m": float(np.isfinite(dtm[within_100]).mean()),
        "valid_relief15_fraction_within_100m": float(np.isfinite(relief[within_100]).mean()),
        "minimum_radial_median_10_to_35m": float(np.nanmin(med[(mids >= 10) & (mids <= 35)])),
        "note": "Metrics do not by themselves prove control success; inspect crop and profiles."
    }
    (out / "control_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (base / "control_npz_manifest.json").write_text(json.dumps({
        "crop_files": {name: f"control/{name}.png" for name in names},
        "radial_profile": "control/radial_profile.csv",
        "cross_profiles": "control/cross_profiles.csv",
        "source_products": "products.npz"}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k not in
                      ("record_lon", "record_lat", "projected_x", "projected_y")}, indent=2))


if __name__ == "__main__":
    main()
