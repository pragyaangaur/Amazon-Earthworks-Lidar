"""Dose-response: below-ground ('multipath') returns vs measured ditch depth (m1).

m5 (skeptic, d7 f8) showed that ONE return 1.5 m below ground near a ring centre
can destroy the ring signal in the recorded recipe, because the grey-open reference
surface collapses to the pit and the enclosure interior is then rewritten as "too
high".  This script measures the same dose-response for the recipe m1 recommends
(median reference surface + vertical-structure test) and for the literal recipe, on
the same synthetic control, so the swarm can see which one survives what rate.

Rates are fractions of returns pushed 0.8-2.0 m below the surface (isolated).

Usage:
  python synth/outlier_dose.py --scene s1_ring40_open --rates 0,3e-5,1e-4,3e-4,1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refdtm                       # noqa: E402
from metrics import ring_metrics     # noqa: E402


def inject(src, dst, rate, rng, centre=None, depth_range=(0.8, 2.0)):
    """Copy a LAZ, pushing `rate` of the returns below ground (and optionally one
    single return at `centre`)."""
    import laspy
    with laspy.open(src, laz_backend=laspy.LazBackend.LazrsParallel) as f:
        las = f.read()
    z = np.asarray(las.z, dtype=np.float64)
    n = z.size
    k = int(round(rate * n))
    if k:
        idx = rng.choice(n, k, replace=False)
        z[idx] -= rng.uniform(*depth_range, k)
    if centre is not None:
        cx, cy = centre
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        i = int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))
        z[i] -= float(depth_range[1]) * 0.75
    las.z = z
    las.write(dst)
    return k + (1 if centre is not None else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="s1_ring40_open")
    ap.add_argument("--workdir", default=".scratch/dose")
    ap.add_argument("--out", default="out/outlier_dose.json")
    ap.add_argument("--rates", default="0,3e-5,1e-4,3e-4,1e-3")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    rates = [float(v) for v in a.rates.split(",")]
    os.makedirs(a.workdir, exist_ok=True)
    src = os.path.join("out", a.scene + ".laz")
    tr = json.load(open(os.path.join("out", a.scene + "_truth.json")))
    ring = next(f for f in tr["features"] if f["type"] == "ring_ditch")
    cx, cy, R = ring["cx"], ring["cy"], float(ring["r"])
    hw = max(2.5, 0.5 * float(ring["width"]) + 1.5)
    rng = np.random.default_rng(a.seed)
    rows = []
    cases = [("none injected", 0.0, False)]
    cases += [("rate %.0e" % r, r, False) for r in rates if r > 0]
    cases += [("ONE return at ring centre", 0.0, True)]
    for label, rate, centre_pt in cases:
        dst = os.path.join(a.workdir, "%s_%s.laz" % (a.scene, label.replace(" ", "_")))
        k = inject(src, dst, rate, rng, centre=(cx, cy) if centre_pt else None)
        row = dict(case=label, rate=rate, injected=k)
        for method in ("vstruct", "literal", "none"):
            out, st = refdtm.build_dtm(dst, method=method, svf=False, verbose=False)
            dem = np.asarray(out["dem"], dtype=np.float64)
            m = ring_metrics(dem, float(out["grid_m"]), float(out["x0"]),
                             float(out["y0"]), cx, cy, R, halfw=hw)
            row[method] = dict(dip_m=None if m["dip_m"] is None else round(m["dip_m"], 3),
                               trough_m=None if m.get("trough_m") is None else round(m["trough_m"], 3),
                               valid_frac=round(m.get("valid_frac") or 0, 3),
                               removed_pct=round(st["removed_pct"], 2),
                               removed_low_pct=round(st.get("removed_low_pct", 0.0), 3))
        rows.append(row)
        print("%-28s k=%-5d " % (label, k) + "  ".join(
            "%s dip %5.2f trough %5.2f rm%% %5.2f" % (
                mth, row[mth]["dip_m"] if row[mth]["dip_m"] is not None else float("nan"),
                row[mth]["trough_m"] if row[mth]["trough_m"] is not None else float("nan"),
                row[mth]["removed_pct"]) for mth in ("vstruct", "literal")), flush=True)
    meta = dict(scene=a.scene, ring=dict(id=ring["id"], cx=cx, cy=cy, r=R,
                                         depth=ring.get("depth"), width=ring.get("width")),
                rates=rates, seed=a.seed, rows=rows,
                note=("each rate = fraction of returns pushed 0.8-2.0 m below the "
                      "surface; 'ONE return at ring centre' is m5's d7-f8 case"))
    with open(a.out, "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
