#!/usr/bin/env python3
"""Run the swarm's independent annular detector on coordinator NPZ products.

Example:
  .venv/bin/python code/run_ring_detector.py data/derived/0414 --transect 0414

Detections are unreviewed measurements, not candidate certainty claims. Output
coordinates are private local data and must only enter the final private ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/vendor_ringdet"))
import ringdet
from process_laz import nan_gaussian


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("derived", type=Path)
    parser.add_argument("--transect", required=True)
    parser.add_argument("--stride-cells", type=int, default=4)
    parser.add_argument("--max-refine", type=int, default=1000)
    parser.add_argument("--layer", choices=("dtm", "relief10", "relief15", "relief30"), default="relief15")
    parser.add_argument("--output-stem", default="rings")
    args = parser.parse_args()
    started = time.monotonic()
    grid = json.loads((args.derived / "grid.json").read_text())
    products = np.load(args.derived / "products.npz")
    dtm = products["dtm"]
    chm = products["chm"]
    # Detect on local relief, as required by the project method. Using absolute
    # DTM heights failed the initial blind control because regional slopes
    # overwhelmed a sub-metre ditch. The relief is already hole-masked.
    h = nan_gaussian(dtm, 0.5).astype(np.float32)
    valid = np.isfinite(dtm)
    selected_relief = products[args.layer] if args.layer != "dtm" else h
    infer = valid & np.isfinite(selected_relief)
    hole_dist = np.minimum(ndi.distance_transform_edt(valid) * grid["cell"], 90).astype(np.float32)
    rough = np.sqrt(np.clip(nan_gaussian(products["relief10"] ** 2, 4), 0, None)).astype(np.float32)
    prep = {"h": h, "valid": valid, "infer": infer, "hole_dist_m": hole_dist,
            "lrm": {10: products["relief10"], 15: products["relief15"], 30: products["relief30"]},
            "rough": rough, "canopy": chm, "cell": float(grid["cell"]),
            "cfg": {"hole_blank_m": 15.0}}
    # Coarse Stage A deliberately favours recall on incomplete annuli. Stage B
    # still requires >=0.45m dip, narrowness, consistency and two-sided relief;
    # every resulting feature remains an unreviewed detection.
    params = dict(ringdet.DEFAULTS, stride_cells=args.stride_cells,
                  input_mode=("dtm" if args.layer == "dtm" else args.layer.replace("relief", "lrm")),
                  max_refine=args.max_refine, min_dip_z=1.0,
                  arc_min_dip_m=0.25, arc_min_cov=0.45,
                  max_two_sided_fail=0.25)
    detections = ringdet.detect_rings(prep, params=params, verbose=True)
    transformer = Transformer.from_crs(grid["epsg"], 4326, always_xy=True)
    rows = []
    serialisable = []
    for index, det in enumerate(detections):
        x = grid["x0"] + (det["col"] + 0.5) * grid["cell"]
        y = grid["y0"] + (det["row"] + 0.5) * grid["cell"]
        lon, lat = transformer.transform(x, y)
        fid = f"{args.transect}_ring_{index:04d}"
        row = {"id": fid, "transect": args.transect, "EPSG": grid["epsg"],
               "x": x, "y": y, "lon": lon, "lat": lat,
               **{k: v for k, v in det.items() if k not in ("_profile", "flags")},
               "flags": ";".join(det["flags"]), "certainty": "unreviewed_detection"}
        rows.append(row)
        serialisable.append({**det, "id": fid, "x": x, "y": y, "lon": lon, "lat": lat,
                             "EPSG": grid["epsg"], "certainty": "unreviewed_detection"})
    output = args.derived / "detections"
    output.mkdir(exist_ok=True)
    temporary = output / f"{args.output_stem}.json.part"
    temporary.write_text(json.dumps({"transect": args.transect,
                                    "status": "unreviewed_detections",
                                    "input_layer": args.layer,
                                    "coarse_stage": "high_recall; not a certainty decision",
                                    "elapsed_seconds": time.monotonic() - started,
                                    "parameters": params,
                                    "products_sha256": hashlib.sha256((args.derived / "products.npz").read_bytes()).hexdigest(),
                                    "detections": serialisable}, indent=2) + "\n")
    temporary.replace(output / f"{args.output_stem}.json")
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{args.output_stem}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with (output / f"{args.output_stem}.csv").open("w", newline="") as handle:
            handle.write("id,transect,EPSG,x,y,lon,lat,certainty\n")
    print(f"RING DETECTION COMPLETE transect={args.transect} detections={len(detections)} "
          f"seconds={time.monotonic() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
