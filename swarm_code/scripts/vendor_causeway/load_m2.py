"""load_m2.py -- read m2's processor products into this module's north-up grid
convention, and write them as the authoritative .npy + .npy.json pair.

m2's rasters are documented south-up: row 0 is the SOUTH edge and
`ys = y0 + (arange(ny) + 0.5) * cell` (their raster.py, their README).  dtlib is
north-up, so every array is flipped vertically and y0 becomes
`y0_south + ny * cell`.  Getting this wrong mirrors every northing about the grid
centre, so the conversion is done once here and asserted.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import dtlib as D


def load(outdir, which=("dtm", "chm", "relief_15"), mask_name="detect_mask"):
    meta_json = os.path.join(outdir, "summary.json")
    with open(meta_json) as fh:
        summ = json.load(fh)
    g = summ["grid"]
    r = np.load(os.path.join(outdir, "rasters.npz"))
    out = {}
    for k in which:
        if k in r:
            out[k] = np.ascontiguousarray(np.flipud(r[k]).astype(np.float32))
    masks = np.load(os.path.join(outdir, "masks.npy"), allow_pickle=True).item()
    for k, v in masks.items():
        if hasattr(v, "shape") and v.shape == r["dtm"].shape:
            out["mask_" + k] = np.ascontiguousarray(np.flipud(v))
    y0_north = float(g["y0"]) + g["ny"] * g["cell"]
    meta = D.make_meta(dx=g["cell"], dy=g["cell"], x0=g["x0"], y0=y0_north,
                       epsg=int(summ.get("epsg") or 0),
                       source=f"m2:{outdir}",
                       note=("north-up conversion of m2 south-up rasters; "
                             "y0 = y0_south + ny*cell"))
    if mask_name and ("mask_" + mask_name) in out:
        meta["note"] += f"; validity mask = m2 {mask_name}"
    return out, meta, summ
