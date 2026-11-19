#!/usr/bin/env python3
"""Create native evidence for screened detB rows without promoting the method.

Usage: .venv/bin/python code/build_detb_review.py <transect> <0414_detB.json>

This prepares candidate AUDIT inputs only. `detB` itself remains a methodological
review track and never sets the transect's long-road screen to done.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from process_laz import colorize


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_detb_review.py <transect> <screen.json>")
    tid, screen_path = sys.argv[1], Path(sys.argv[2])
    base = ROOT / "data/derived_v2" / tid
    summary = json.loads((base / "summary.json").read_text())
    grid = summary["grid"]
    payload = json.loads(screen_path.read_text())
    rows = [r for r in payload["accepted_rows"] if r.get("accepted") and r.get("screened")]
    if not rows:
        (base / "detb_review").mkdir(exist_ok=True)
        (base / "detb_review/summary.json").write_text(json.dumps({"n_screened": 0}, indent=2) + "\n")
        print(json.dumps({"transect": tid, "n_screened": 0}))
        return
    source = np.load(base / "rasters.npz")
    product = np.load(base / "products.npz")
    cell = float(grid["cell"])
    forest = ndi.binary_dilation(np.nan_to_num(source["chm"]) > summary["canopy"]["threshold_m"], iterations=1)
    detect = np.isfinite(source["dtm"]) & ~forest
    review_root = base / "detb_review"
    review_root.mkdir(exist_ok=True)
    audit_rows = []
    for index, row in enumerate(rows, 1):
        fid = f"D{tid}-L{index:03d}"
        out = review_root / fid
        out.mkdir(exist_ok=True)
        x0, y0, x1, y1 = (float(row[k]) for k in ("x0", "y0", "x1", "y1"))
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        pad = 100.0
        c0 = max(0, int((min(x0, x1) - pad - grid["x0"]) / cell))
        c1 = min(grid["nx"], int((max(x0, x1) + pad - grid["x0"]) / cell) + 1)
        r0 = max(0, int((min(y0, y1) - pad - grid["y0"]) / cell))
        r1 = min(grid["ny"], int((max(y0, y1) + pad - grid["y0"]) / cell) + 1)
        xy_to_png = lambda x, y: ((x - (grid["x0"] + c0 * cell)) / cell - 0.5,
                                  (r1 - r0) - (y - (grid["y0"] + r0 * cell)) / cell - 0.5)
        layers = {"dtm": source["dtm"], "chm": source["chm"], "hillshade": source["hillshade"],
                  "svf": source["svf"], "relief_15": source["relief_15"],
                  "detect_mask": detect.astype(float), "reconstructed": product["removed"].astype(float)}
        for name, array in layers.items():
            crop = array[r0:r1, c0:c1]
            if name in {"detect_mask", "reconstructed"}:
                rgb = np.repeat(np.nan_to_num(crop * 255, nan=0).astype(np.uint8)[..., None], 3, axis=2)
            else:
                kind = "chm" if name == "chm" else ("relief15" if "relief" in name else name)
                rgb = colorize(crop, kind)
            image = Image.fromarray(rgb[::-1]); draw = ImageDraw.Draw(image)
            p0, p1 = xy_to_png(x0, y0), xy_to_png(x1, y1)
            draw.line((*p0, *p1), fill=(255, 25, 25), width=1)
            for point in (p0, p1):
                draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), outline=(255, 25, 25), width=1)
            image.save(out / f"{name}.png", optimize=True)
        length = float(np.hypot(x1-x0, y1-y0))
        ux, uy = (x1-x0) / length, (y1-y0) / length
        nx, ny = -uy, ux
        profiles = []
        for station in np.linspace(0.1, 0.9, 9):
            sx, sy = x0 + station * (x1-x0), y0 + station * (y1-y0)
            for offset in sorted(set(np.arange(-40.0, 40.01, cell).tolist()) | {0.0}):
                px, py = sx + offset * nx, sy + offset * ny
                col = (px-grid["x0"]) / cell - 0.5
                row_grid = (py-grid["y0"]) / cell - 0.5
                values = {"station": station, "offset_m": offset, "x": px, "y": py}
                for name in ("dtm", "chm", "relief_15"):
                    values[name] = float(ndi.map_coordinates(source[name], [[row_grid], [col]], order=1,
                                                            mode="constant", cval=np.nan)[0])
                values["detect"] = int(0 <= row_grid < detect.shape[0] and 0 <= col < detect.shape[1]
                                       and detect[int(round(row_grid)), int(round(col))])
                values["reconstructed"] = int(0 <= row_grid < detect.shape[0] and 0 <= col < detect.shape[1]
                                              and product["removed"][int(round(row_grid)), int(round(col))])
                profiles.append(values)
        with (out / "cross_profiles.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(profiles[0]))
            writer.writeheader(); writer.writerows(profiles)
        mid_chm = [p["chm"] for p in profiles if p["offset_m"] == 0 and np.isfinite(p["chm"])]
        metadata = {"id": fid, "method": "m3 detB methodological review", "transect": tid,
                    "certainty": "unreviewed", "long_road_screen_validated": False,
                    "row_order": "south-up", "pixel_size_m": cell,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "x_mid": xm, "y_mid": ym,
                    "length_m": row["length_m"], "length_supported_m": row["length_supported_m"],
                    "depth_m": row["depth_m"], "width_m": row["width_m"], "azimuth_deg": row["azimuth_deg"],
                    "canopy_frac": row["canopy_frac"], "kind": row["kind"],
                    "canopy_median_on_centreline_m": float(np.median(mid_chm)) if mid_chm else None,
                    "detB_row": row,
                    "crop": {"r0": r0, "r1": r1, "c0": c0, "c1": c1},
                    "epsg": summary["epsg"], "screen_json": str(screen_path)}
        (out / "detection.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        audit_rows.append({"id": fid, "transect": tid, "EPSG": summary["epsg"], "x": xm, "y": ym,
                           "lon": "", "lat": "", "radius_m": row["length_m"]/2,
                           "dip_m": row["depth_m"], "shape": row["kind"], "certainty": "unreviewed_detection"})
    audit_path = review_root / "candidates_input.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader(); writer.writerows(audit_rows)
    (review_root / "summary.json").write_text(json.dumps({"n_screened": len(rows), "ids": [r["id"] for r in audit_rows]}, indent=2) + "\n")
    print(json.dumps({"transect": tid, "n_screened": len(rows), "ids": [r["id"] for r in audit_rows], "audit_csv": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
