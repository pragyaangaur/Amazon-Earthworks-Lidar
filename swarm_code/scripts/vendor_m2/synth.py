"""Synthetic EBA-like point cloud with ground truth (m2 self-contained control).

Everything downstream of ingest is validated against this rather than against
real data first: if the product chain cannot recover a 40 m ring ditch of known
depth from a cloud with canopy, holes and a tilted, undulating surface, its
output on transect 0414 means nothing.

Scene (~900 x 900 m, 3 m cells):
  * tilted plane + two low-frequency undulations (regional relief)
  * 40 m ring ditch, 6 m wide, 1.5 m deep, with a 0.3 m low outer bank
  * two straight sunken roads (4 m, 0.7 m) far enough not to cut the ring
  * one embanked road (3 m, +0.5 m)
  * canopy: 6 emergent blobs + scattered trees
  * data holes: one straight swath gap, one blob gap, one edge notch
  * one meandering river negative (18 m wide, 4 m deep) and one cattle pond

Usage
-----
    python3 synth.py out_dir [--seed 7] [--size 900] [--density 1.6]

Writes  out_dir/synth.las  and  out_dir/synth_truth.json
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

import lasio


def _ground(x, y, x0, y0):
    """Regional surface: tilt + two long-wavelength undulations."""
    u = (x - x0) / 1000.0
    v = (y - y0) / 1000.0
    return (100.0 + 6.0 * u - 3.0 * v
            + 1.4 * np.sin(2 * np.pi * u / 0.22)
            + 1.1 * np.cos(2 * np.pi * v / 0.31))


def _ring_ditch(x, y, cx, cy, r0=20.0, half_w=3.0, depth=1.5,
                bank_h=0.30, bank_w=2.0):
    """Torus depression with optional low outer bank (metres)."""
    r = np.hypot(x - cx, y - cy)
    dz = np.zeros_like(r)
    inner = np.abs(r - r0) < half_w
    # smooth profile: parabolic across the ditch
    t = np.clip((r - r0) / half_w, -1, 1)
    dz = np.where(inner, -depth * (1.0 - t * t), 0.0)
    banked = (r >= r0 + half_w) & (r < r0 + half_w + bank_w)
    bt = (r - (r0 + half_w)) / bank_w
    dz = np.where(banked, bank_h * (1.0 - bt), dz)
    return dz


def _trench_line(x, y, p0, p1, half_w, depth):
    """Depth of a straight (sunken, depth>0) or embanked (depth<0) road."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = ((x - p0[0]) * d[0] + (y - p0[1]) * d[1]) / L2
    t = np.clip(t, 0, 1)
    px = p0[0] + t * d[0]; py = p0[1] + t * d[1]
    dist = np.hypot(x - px, y - py)
    tt = np.clip(dist / half_w, 0, 1)
    return -depth * (1.0 - tt * tt)


def _mound(x, y, cx, cy, radius, height):
    """Paraboloid mound (positive relief), zero outside the radius."""
    d = np.hypot(x - cx, y - cy)
    return np.where(d < radius, height * (1.0 - (d / radius) ** 2), 0.0)


def _meander(x, y, y0, amp=60.0, wl=260.0, half_w=9.0, depth=4.0):
    """Meandering river negative: trench following a sine axis."""
    yc = y0 + amp * np.sin(2 * np.pi * x / wl)
    d = np.abs(y - yc)
    tt = np.clip(d / half_w, 0, 1)
    return -depth * (1.0 - tt * tt)


def build(size=900.0, density=1.6, seed=7, canopy_density=1.1):
    rng = np.random.default_rng(seed)
    x0, y0 = 300000.0, 9000000.0          # UTM-like origin (EPSG:32719)
    cx, cy = x0 + size * 0.45, y0 + size * 0.55

    # ---- ground returns -------------------------------------------------
    n_ground = int(density * size * size)
    gx = x0 + rng.uniform(0, size, n_ground)
    gy = y0 + rng.uniform(0, size, n_ground)
    gz = _ground(gx, gy, x0, y0)
    gz += _ring_ditch(gx, gy, cx, cy)
    gz += _trench_line(gx, gy, (x0 + 520, y0 + 80), (x0 + 560, y0 + 820), 2.0, 0.7)
    gz += _trench_line(gx, gy, (x0 + 60, y0 + 700), (x0 + 860, y0 + 660), 2.0, 0.7)
    gz += _trench_line(gx, gy, (x0 + 80, y0 + 120), (x0 + 840, y0 + 160), 1.5, -0.5)
    gz += _meander(gx, gy, y0 + 300.0)
    # mound ring around a plaza: 6 mounds 0.5-3 m high, 16 m across, 40 m from
    # the plaza centre; plus one isolated 2 m mound.  These are positive features
    # of exactly the size the opening cascade is designed to remove, so they test
    # whether the DTM keeps them.
    mound_ring_c = (x0 + 150.0, y0 + 400.0)
    mounds = []
    for i, h in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
        th = 2 * np.pi * i / 6.0
        mounds.append(dict(id="S-MOUND-%d" % (i + 1),
                           x=mound_ring_c[0] + 40.0 * np.cos(th),
                           y=mound_ring_c[1] + 40.0 * np.sin(th),
                           radius_m=8.0, height_m=h))
    mounds.append(dict(id="S-MOUND-SOLO", x=x0 + 780.0, y=y0 + 560.0,
                       radius_m=12.0, height_m=2.0))
    for m in mounds:
        gz += _mound(gx, gy, m["x"], m["y"], m["radius_m"], m["height_m"])
    # cattle pond: 12 m radius circular pit, 1.8 m deep
    rr = np.hypot(gx - (x0 + 700), gy - (y0 + 430))
    gz += np.where(rr < 12, -1.8 * (1 - (rr / 12) ** 2), 0.0)
    gz += rng.normal(0, 0.03, gx.size)

    # ---- canopy returns -------------------------------------------------
    blobs = [((x0 + 180, y0 + 180), 14.0, 22.0), ((x0 + 640, y0 + 250), 11.0, 18.0),
             ((x0 + 250, y0 + 560), 16.0, 25.0), ((x0 + 700, y0 + 760), 13.0, 20.0),
             ((x0 + 500, y0 + 430), 9.0, 14.0), ((x0 + 100, y0 + 820), 10.0, 17.0)]
    # one large closed-canopy patch: the honest hard case for any DTM filter
    big_forest = ((x0 + 620.0, y0 + 60.0), 75.0, 26.0)
    cxs, cys, czs = [], [], []
    for (bx, by), R, H in blobs + [big_forest]:
        n = int(canopy_density * np.pi * R * R)
        rr = R * np.sqrt(rng.uniform(0, 1, n))
        th = rng.uniform(0, 2 * np.pi, n)
        px = bx + rr * np.cos(th); py = by + rr * np.sin(th)
        ph = H * np.sqrt(np.maximum(0.0, 1 - (rr / R) ** 2))
        pz = _ground(px, py, x0, y0) + 1.5 + ph * rng.uniform(0.55, 1.0, n)
        cxs.append(px); cys.append(py); czs.append(pz + rng.normal(0, 0.25, n))

    # ---- canopy occlusion: forest hides the ground ----------------------
    cover = np.zeros(gx.size)
    for (bx, by), R, H in blobs + [big_forest]:
        rr = np.hypot(gx - bx, gy - by)
        c = np.clip(1.0 - (rr / R) ** 2, 0.0, 1.0)
        cover = np.maximum(cover, c)
    p_ground = np.exp(-6.0 * cover)          # dense canopy -> almost no ground hits
    kg = rng.random(gx.size) < p_ground
    gx, gy, gz = gx[kg], gy[kg], gz[kg]
    # scattered trees
    n_tree = 220
    px = x0 + rng.uniform(0, size, n_tree); py = y0 + rng.uniform(0, size, n_tree)
    pz = _ground(px, py, x0, y0) + rng.uniform(4, 18, n_tree)
    cxs.append(px); cys.append(py); czs.append(pz + rng.normal(0, 0.3, n_tree))

    X = np.concatenate([gx] + cxs)
    Y = np.concatenate([gy] + cys)
    Z = np.concatenate([gz] + czs)
    n_canopy = sum(c.size for c in cxs)
    cls = np.concatenate([np.full(gx.size, 2, np.uint8),
                          np.full(n_canopy, 5, np.uint8)])
    # class 2 deliberately contaminated (mirrors the EBA problem): a slice of
    # canopy returns is labelled ground.
    n_cont = int(0.25 * n_canopy)
    idx = rng.choice(np.arange(gx.size, Z.size), n_cont, replace=False)
    cls[idx] = 2

    # ---- data holes (swath gaps) ---------------------------------------
    keep = np.ones(Z.size, bool)
    keep &= ~((X > x0 + 300) & (X < x0 + 340))                # straight gap
    keep &= ~(np.hypot(X - (x0 + 430), Y - (y0 + 700)) < 45)  # blob gap
    keep &= ~((X > x0 + 780) & (Y > y0 + 640))                # edge notch
    X, Y, Z, cls = X[keep], Y[keep], Z[keep], cls[keep]

    # LAS wants an unordered point stream; shuffle to mimic flight order noise
    p = rng.permutation(X.size)
    X, Y, Z, cls = X[p], Y[p], Z[p], cls[p]

    truth = dict(
        scene="synthetic EBA-like control (m2)",
        seed=seed, size_m=size, epsg=32719, utm=(x0, y0),
        density_pts_m2=density, n_points=int(X.size), n_ground=int(gx.size),
        surface=dict(base_z=100.0, tilt=(6.0, -3.0),
                     waves=[dict(amp=1.4, wl=220.0, axis="x"),
                            dict(amp=1.1, wl=310.0, axis="y")]),
        features=[
            dict(id="S-RING-001", type="ring_ditch", x=cx, y=cy, radius_m=20.0,
                 diameter_m=40.0, width_m=6.0, depth_m=1.5, bank_m=0.30,
                 note="positive control: 40 m ring, low outer bank"),
            dict(id="S-ROAD-001", type="sunken_road",
                 p0=[x0 + 520, y0 + 80], p1=[x0 + 560, y0 + 820],
                 width_m=4.0, depth_m=0.7),
            dict(id="S-ROAD-002", type="sunken_road",
                 p0=[x0 + 60, y0 + 700], p1=[x0 + 860, y0 + 660],
                 width_m=4.0, depth_m=0.7),
            dict(id="S-ROAD-003", type="embanked_road",
                 p0=[x0 + 80, y0 + 120], p1=[x0 + 840, y0 + 160],
                 width_m=3.0, height_m=0.5),
        ] + [dict(type="mound", **m) for m in mounds],
        mound_ring=dict(x=mound_ring_c[0], y=mound_ring_c[1], plaza_radius_m=30.0,
                        mound_ring_radius_m=40.0, n_mounds=6),
        negatives=[
            dict(id="N-MEANDER", type="meandering_river", y_axis=y0 + 300.0,
                 amp_m=60.0, wl_m=260.0, width_m=18.0, depth_m=4.0),
            dict(id="N-POND", type="cattle_pond", x=x0 + 700, y=y0 + 430,
                 radius_m=12.0, depth_m=1.8),
        ],
        canopy_blobs=[dict(x=b[0][0], y=b[0][1], radius_m=b[1], height_m=b[2])
                      for b in blobs],
        big_forest=dict(x=big_forest[0][0], y=big_forest[0][1],
                        radius_m=big_forest[1], height_m=big_forest[2]),
        holes=[dict(type="strip", x_range=[x0 + 300, x0 + 340]),
               dict(type="blob_gap", x=x0 + 430, y=y0 + 700, radius_m=45),
               dict(type="edge_notch", x_min=x0 + 780, y_min=y0 + 640)],
        contamination=dict(class2_canopy_fraction=0.25),
        ground_occlusion="ground returns kept with p=exp(-6.0*canopy_cover)",
        n_ground_after_occlusion=int(gx.size),
    )
    return X, Y, Z, cls, truth


def main(argv):
    out = argv[1] if len(argv) > 1 else "synth_out"
    kw = {}
    for a in argv[2:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            kw[k] = float(v) if v.replace(".", "").isdigit() else v
    size = float(kw.get("size", 900.0))
    density = float(kw.get("density", 1.6))
    seed = int(kw.get("seed", 7))
    os.makedirs(out, exist_ok=True)
    X, Y, Z, cls, truth = build(size=size, density=density, seed=seed)
    las = os.path.join(out, "synth.las")
    lasio.write_las(las, X, Y, Z, epsg=32719, classification=cls)
    truth["las"] = las
    with open(os.path.join(out, "synth_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    print("wrote %s  points=%d  extent=%.0f m" % (las, X.size, size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
