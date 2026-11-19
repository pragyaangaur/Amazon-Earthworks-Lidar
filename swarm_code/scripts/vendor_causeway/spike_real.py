"""spike_real.py -- spike-in validation on the REAL transect DTM.

Synthetic controls prove the geometry logic is self-consistent; they do not prove
the detector works on a real lidar background (real residuals, real drainage, real
holes).  This script injects a known embankment into the real DTM and measures
recovery, and runs the same block unmodified to measure the null false-positive
rate on real terrain.

Usage
  python3 code/spike_real.py --dtm-npz <m2 out dir> --blocks 3 --out results/real/spike
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import load_m2
import causeway
import dtlib as D
import synth


def pick_blocks(meta, z, mask, block_cells=200, n=3, min_frac=0.5):
    """Blocks with substantial usable data, spread over the transect."""
    cand = []
    for i in range(0, z.shape[0] - block_cells, block_cells // 2):
        for j in range(0, z.shape[1] - block_cells, block_cells // 2):
            sub = mask[i:i + block_cells, j:j + block_cells] & np.isfinite(
                z[i:i + block_cells, j:j + block_cells])
            frac = sub.mean()
            if frac > min_frac:
                cand.append((frac, i, j))
    cand.sort(reverse=True)
    out, used = [], set()
    for frac, i, j in cand:
        if all(abs(i - u[0]) > block_cells or abs(j - u[1]) > block_cells for u in used):
            used.add((i, j))
            out.append((i, j, frac))
        if len(out) >= n:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm-npz", required=True, help="m2 out dir with rasters.npz")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--block-cells", type=int, default=200)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--out", default="results/real/spike")
    a = ap.parse_args(argv)
    out, meta, summ = load_m2.load(a.dtm_npz)
    z = out["dtm"]
    w = out["mask_detect_mask"] | ~out["mask_data_hole"]
    w = np.isfinite(z) & (~out["mask_data_hole"])
    chm = out["chm"]
    os.makedirs(a.out, exist_ok=True)
    bc = a.block_cells
    blocks = pick_blocks(meta, z, w, block_cells=bc, n=a.blocks, min_frac=a.min_frac)
    print(f"{len(blocks)} blocks with >70 % usable data")
    rows = []
    for bi, (i0, j0, frac) in enumerate(blocks):
        sub = z[i0:i0 + bc, j0:j0 + bc].copy()
        m = dict(meta)
        m["x0"] = meta["x0"] + j0 * meta["dx"]
        m["y0"] = meta["y0"] - i0 * meta["dy"]
        ws = w[i0:i0 + bc, j0:j0 + bc]
        cs = chm[i0:i0 + bc, j0:j0 + bc]
        # null run on real terrain
        null = causeway.detect(sub, m, w=ws, chm=cs, transect=f"real-null{bi}")
        # spike: a 7 m x 1.2 m embankment across the middle of the block
        cy = m["y0"] - (bc // 2) * meta["dy"]
        line = LineString([(m["x0"] + 0.15 * bc * meta["dx"], cy),
                           (m["x0"] + 0.85 * bc * meta["dx"], cy)])
        spike = sub.copy()
        spike = synth.build_ridge(spike, m, line, 7.0, 1.2, dps_m=2.5, crown_flat=0.45)
        sp = causeway.detect(spike, m, w=ws, chm=cs, transect=f"real-spike{bi}")
        # recovery: a causeway or raised line centred near the injected line
        rec = [r for r in sp["rows"]
               if abs(r["y"] - cy) < 25 and abs(r["axis_angle_deg"]) < 20]
        got = rec[0] if rec else None
        r = dict(block=bi, row0=i0, col0=j0, usable_frac=round(float(frac), 3),
                 null_candidates=len(null["rows"]),
                 null_causeway=null["stats"]["n_causeway"],
                 null_types=[q["type"] for q in null["rows"]],
                 spike_candidates=len(sp["rows"]),
                 spike_causeway=sp["stats"]["n_causeway"],
                 recovered=bool(got),
                 recovered_type=(got["type"] if got else None),
                 recovered_len_m=(round(got["length_m"]) if got else None),
                 recovered_h_m=(round(got["height_m"], 2) if got else None),
                 recovered_w_m=(round(got["crest_width_m"], 1) if got else None))
        rows.append(r)
        print(json.dumps(r))
    ok = sum(1 for r in rows if r["recovered"])
    fp_null = sum(r["null_causeway"] for r in rows)
    summary = dict(blocks=len(rows), recovered=ok, null_causeway=fp_null,
                   param=causeway.DEFAULTS, rows=rows,
                   note=("an embankment identical to the synthetic positive is injected "
                         "into the real 3 m DTM of transect 0414; recovery means a "
                         "detection of the same class within 25 m and 20 deg"))
    with open(os.path.join(a.out, "spike_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\nrecovered {ok}/{len(rows)} spikes; null causeway calls {fp_null}")
    return summary


if __name__ == "__main__":
    main()
