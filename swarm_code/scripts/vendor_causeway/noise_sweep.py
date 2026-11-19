"""noise_sweep.py -- false-positive base rate of the causeway detector on pure
survey-noise tiles that contain no earthwork (fractal terrain + one stream).

This is the honest counterpart of the designed controls: it estimates how many
`causeway` calls the detector invents per square kilometre of noise, which is what
decides whether its output is usable on real transects or just a lead generator.

Writes results/noise_sweep.json / .md.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causeway
import dtlib as D
import synth


def noise_tile(seed, n=400, dx=3.0, canopy_amp=0.45, with_stream=True):
    meta = D.make_meta(dx=dx, x0=500000.0, y0=9600000.0, epsg=0, source=f"noise{seed}")
    z = synth.fractal_noise((n, n), seed, dx=dx)
    z += synth.fractal_noise((n, n), seed + 1, amps=((250.0, 1.6), (800.0, 2.5)), dx=dx)
    if with_stream:
        sx = 0.5 * n * dx
        stream = synth.sinuous((sx, 0), (sx, n * dx), 40, 500)
        z = synth.carve_channel(z, meta, stream, 12.0, 1.5)
    z = synth.add_survey_noise(z, seed, white=0.05, canopy_amp=canopy_amp,
                               canopy_sig=25.0)
    return z.astype(np.float32), meta


def main(outroot="results", n_tiles=24):
    os.makedirs(outroot, exist_ok=True)
    recs = []
    for k in range(n_tiles):
        seed = 100 + k
        canopy = 0.55 if k % 3 else 0.0          # mix open and forested tiles
        z, meta = noise_tile(seed, canopy_amp=canopy)
        t0 = time.time()
        res = causeway.detect(z, meta, transect=f"noise{seed:03d}")
        dt = time.time() - t0
        area_km2 = (z.shape[0] * meta["dx"] / 1000.0) * (z.shape[1] * meta["dy"] / 1000.0)
        recs.append(dict(seed=seed, tile="noise%03d" % seed, canopy_amp=canopy,
                         area_km2=round(area_km2, 3), time_s=round(dt, 2),
                         n_causeway=res["stats"]["n_causeway"],
                         n_raised=res["stats"]["n_raised_nocross"],
                         n_crest_cells=res["stats"]["n_crest_cells"],
                         funnel=res["funnel"],
                         causeway_flags=[r["flags"] for r in res["rows"] if r["causeway"]],
                         causeway_len_m=[round(r["length_m"]) for r in res["rows"] if r["causeway"]]))
        print(f"{recs[-1]['tile']} canopy {canopy:.2f}: causeway {recs[-1]['n_causeway']} "
              f"raised {recs[-1]['n_raised']} ({dt:.1f}s)")
    area = sum(r["area_km2"] for r in recs)
    n_caus = sum(r["n_causeway"] for r in recs)
    n_raised = sum(r["n_raised"] for r in recs)
    summary = dict(n_tiles=len(recs), total_area_km2=round(area, 2),
                   causeway_calls=n_caus, raised_linear_calls=n_raised,
                   causeway_per_km2=round(n_caus / max(area, 1e-9), 3),
                   raised_per_km2=round(n_raised / max(area, 1e-9), 3),
                   tiles_with_causeway=sum(1 for r in recs if r["n_causeway"]),
                   mean_time_s=round(float(np.mean([r["time_s"] for r in recs])), 2),
                   max_time_s=round(float(np.max([r["time_s"] for r in recs])), 2),
                   params=causeway.DEFAULTS)
    with open(os.path.join(outroot, "noise_sweep.json"), "w") as fh:
        json.dump(dict(summary=summary, tiles=recs), fh, indent=1, default=float)
    lines = ["# False-positive base rate on pure noise tiles (m3 causeway detector)", "",
             f"{len(recs)} tiles, {area:.2f} km2 total, fractal terrain + one stream, "
             f"canopy residual 0.45-0.55 m, no earthwork.", "",
             f"- `causeway` calls: **{n_caus}** ({summary['causeway_per_km2']} / km2), "
             f"in {summary['tiles_with_causeway']}/{len(recs)} tiles",
             f"- `raised_linear_no_crossing` calls: {n_raised} "
             f"({summary['raised_per_km2']} / km2)",
             f"- runtime {summary['mean_time_s']} s per 1.2 x 1.2 km tile "
             f"(max {summary['max_time_s']} s)", "",
             "| tile | canopy | causeway | raised | s |", "|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['tile']} | {r['canopy_amp']} | {r['n_causeway']} | "
                     f"{r['n_raised']} | {r['time_s']} |")
    lines += ["", "Read this as: any real-tile causeway call has to be judged against "
              "this base rate, and a call with a long length and a high score is worth "
              "more than a short one.", ""]
    with open(os.path.join(outroot, "noise_sweep.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results",
         int(sys.argv[2]) if len(sys.argv) > 2 else 24)
