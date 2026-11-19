"""Synthetic airborne-lidar simulator for EBA-like transects (m1 benchmark).

What it models (deliberately, because each item changes what a processor's
ground filter can recover):

* line-scan sampling: parallel flight lines, ~0.5 m along-track spacing,
  1.0 m across-track at the nominal 2 pulses / m^2;
* a real canopy: Poisson crown positions per 1 m cell, ellipsoidal crowns,
  a canopy height model and a fractional cover model;
* ground returns are *occluded*: P(ground return) = exp(-0.10 * cover * CHM),
  so under closed forest most 3 m cells have no ground return and the lowest
  return is canopy - this is exactly the EBA contamination the previous run
  found, and it is reproduced here on purpose;
* first / intermediate / last return numbering, understorey returns;
* class-2 contamination: ~93% of *all* returns are labelled class 2,
  including canopy, so "ignore the class field" is testable;
* vertical noise, gross high/low outliers (birds, multipath), and
* data holes: swath gaps, rectangles and circles with no pulses at all,
  plus optional "empty halo" geometry around a hole.

Everything is emitted in chunks, so a 12.5 km x 300 m transect never exists
as one in-memory cloud (the same constraint the real ingest faces).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from surface import ground

# ------------------------------------------------------------------ canopy --
def build_canopy(scene, rng):
    """Return (chm, cover, density) 1 m rasters in local scene coordinates.

    `scene['canopy']` is a list of regions applied in order, each
    ``{"bbox": [x0, y0, x1, y1], "density": crowns/m^2, "hmin": m,
       "hmax": m, "rc": [r_min, r_max]}``.  A region with density 0 clears
    canopy (used for pasture and clearings inside a forest block).
    """
    g = float(scene.get("canopy_grid_m", 1.0))
    Lx = scene["x1"] - scene["x0"]
    Ly = scene["y1"] - scene["y0"]
    nx, ny = int(np.floor(Lx / g)), int(np.floor(Ly / g))
    dens = np.zeros((ny, nx), dtype=np.float32)
    for reg in scene.get("canopy", []):
        bx0, by0, bx1, by1 = reg["bbox"]
        i0 = max(0, int((bx0 - scene["x0"]) / g))
        i1 = min(nx, int(np.ceil((bx1 - scene["x0"]) / g)))
        j0 = max(0, int((by0 - scene["y0"]) / g))
        j1 = min(ny, int(np.ceil((by1 - scene["y0"]) / g)))
        dens[j0:j1, i0:i1] = float(reg.get("density", 0.0))

    chm = np.zeros((ny, nx), dtype=np.float32)
    cov = np.zeros((ny, nx), dtype=np.float32)
    if dens.max() <= 0:
        return chm, cov, dens

    # crowns per cell -> crown list
    ncell = rng.poisson(dens)
    cells = np.nonzero(ncell.ravel())[0]
    reps = ncell.ravel()[cells]
    ii = np.repeat(cells % nx, reps)
    jj = np.repeat(cells // nx, reps)
    nc = len(ii)
    if nc == 0:
        return chm, cov, dens
    hmin = np.full(nc, 8.0, dtype=np.float32)
    hmax = np.full(nc, 26.0, dtype=np.float32)
    rcmin = np.full(nc, 2.5, dtype=np.float32)
    rcmax = np.full(nc, 5.0, dtype=np.float32)
    # per-region parameters (later regions win) - resolved per crown
    reg_of = np.zeros(nc, dtype=np.int32)
    for k, reg in enumerate(scene.get("canopy", [])):
        bx0, by0, bx1, by1 = reg["bbox"]
        inside = ((ii * g + scene["x0"] >= bx0) & (ii * g + scene["x0"] < bx1)
                  & (jj * g + scene["y0"] >= by0) & (jj * g + scene["y0"] < by1))
        reg_of[inside] = k
    for k, reg in enumerate(scene.get("canopy", [])):
        m = reg_of == k
        if not m.any():
            continue
        hmin[m] = float(reg.get("hmin", 8.0))
        hmax[m] = float(reg.get("hmax", 26.0))
        rc = reg.get("rc", [2.5, 5.0])
        rcmin[m] = float(rc[0])
        rcmax[m] = float(rc[1])
    cx = scene["x0"] + (ii + rng.random(nc)) * g
    cy = scene["y0"] + (jj + rng.random(nc)) * g
    # trunk-height distribution: more small trees than giants
    u = rng.random(nc)
    h = hmin + (hmax - hmin) * u ** 1.9
    rc = rcmin + (rcmax - rcmin) * rng.random(nc)

    # paint crowns: max-blend for CHM, additive cover weight
    rmax_cells = int(np.ceil(rc.max() / g)) + 1
    step = 20000
    for s in range(0, nc, step):
        e = min(nc, s + step)
        cxi = (cx[s:e] - scene["x0"]) / g
        cyi = (cy[s:e] - scene["y0"]) / g
        for k in range(e - s):
            ci, cj = cxi[k], cyi[k]
            r = rc[s + k] / g
            i0, i1 = max(0, int(ci - r)), min(nx, int(ci + r) + 1)
            j0, j1 = max(0, int(cj - r)), min(ny, int(cj + r) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            xs = np.arange(i0, i1) - ci
            ys = np.arange(j0, j1) - cj
            d2 = (ys[:, None] ** 2 + xs[None, :] ** 2) / (r * r)
            w = np.clip(1.0 - d2, 0.0, 1.0)
            val = h[s + k] * w ** 0.65
            np.maximum(chm[j0:j1, i0:i1], val, out=chm[j0:j1, i0:i1])
            cov[j0:j1, i0:i1] += w * 0.55
    cover = 1.0 - np.exp(-1.9 * cov)
    np.clip(cover, 0.0, 1.0, out=cover)
    return chm, cover, dens


def sample_raster(arr, scene, x, y, g=1.0):
    """Bilinear sample of a (ny, nx) raster given local scene coordinates."""
    col = (x - scene["x0"]) / g - 0.5
    row = (y - scene["y0"]) / g - 0.5
    return ndimage.map_coordinates(arr, [row, col], order=1, mode="nearest")


# ------------------------------------------------------------- pulse grid ---
def pulse_chunk(scene, rng, xlo, xhi):
    """Pulses inside the x-slab [xlo, xhi): line-scan geometry plus jitter."""
    sc = scene["scan"]
    along = float(sc.get("along_m", 0.5))
    across = float(sc.get("across_m", 1.0))
    jx = float(sc.get("jitter_along_m", 0.18))
    jy = float(sc.get("jitter_across_m", 0.10))
    n_along = max(0, int(np.floor((xhi - xlo) / along)))
    ys = np.arange(scene["y0"] + across * 0.5, scene["y1"], across)
    if n_along == 0 or len(ys) == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32))
    xs = xlo + (np.arange(n_along) + 0.5) * along
    X, Y = np.meshgrid(xs, ys)
    x = X.ravel() + rng.normal(0.0, jx, X.size)
    y = Y.ravel() + rng.normal(0.0, jy, Y.size)
    line = np.repeat(np.arange(len(ys), dtype=np.int32), n_along)
    keep = (x >= scene["x0"]) & (x < scene["x1"]) & (y >= scene["y0"]) & (y < scene["y1"])
    x, y, line = x[keep], y[keep], line[keep]
    # data holes: no pulses at all inside them
    for h in scene.get("holes", []):
        keep = np.ones(x.size, dtype=bool)
        typ = h.get("type", "rect")
        if typ == "rect":
            bx0, by0, bx1, by1 = h["bbox"]
            keep &= ~((x >= bx0) & (x < bx1) & (y >= by0) & (y < by1))
        elif typ == "circle":
            keep &= np.hypot(x - h["cx"], y - h["cy"]) >= float(h["r"])
        elif typ == "band":            # e.g. a missing flight strip
            v = x if h.get("axis", "x") == "x" else y
            keep &= np.abs(v - float(h["center"])) >= float(h["halfwidth"])
        else:
            raise KeyError("unknown hole type %r" % typ)
        x, y, line = x[keep], y[keep], line[keep]
    return x, y, line


# --------------------------------------------------------------- returns ----
def simulate_returns(scene, rng, x, y, chm, cover, line):
    """Turn pulses into returns. Returns a dict of flat per-return arrays."""
    if x.size == 0:
        e = np.zeros(0)
        return dict(x=e, y=e, z=e, rn=e.astype(np.int8), nr=e.astype(np.int8),
                    cls=e.astype(np.uint8), src=e.astype(np.uint16), gps=e,
                    p_zg=e, p_H=e, p_q=e, p_nret=e.astype(np.int32),
                    p_x=e, p_y=e)
    sc = scene.get("returns", {})
    sigma = float(sc.get("sigma_z_m", 0.06))
    zg = ground(x, y, scene)
    H = np.clip(sample_raster(chm, scene, x, y), 0.0, None)
    C = np.clip(sample_raster(cover, scene, x, y), 0.0, 1.0)
    alpha = float(sc.get("occlusion_alpha", 0.10))
    q = np.exp(-alpha * C * H)
    q = np.clip(q, 0.015, 1.0)

    has_can = H > 1.0
    N = x.size
    ground_hit = np.where(has_can, rng.random(N) < q, True)

    u1 = 0.78 + 0.22 * rng.random(N)
    z1 = np.where(has_can, zg + H * u1, zg)
    have2 = has_can & (rng.random(N) < float(sc.get("p_second_return", 0.45)))
    under = rng.random(N) < float(sc.get("p_understorey", 0.15))
    frac = np.where(rng.random(N) < 0.58,
                    0.46 + 0.30 * rng.random(N),
                    0.14 + 0.30 * rng.random(N))
    z2 = np.where(under, zg + 0.4 + 2.2 * rng.random(N), zg + H * frac)
    z3 = zg
    m3 = has_can & ground_hit

    n_ret = 1 + have2.astype(np.int32) + m3.astype(np.int32)
    total = int(n_ret.sum())
    off = np.concatenate(([0], np.cumsum(n_ret)))
    start = off[:-1]
    z = np.empty(total, dtype=np.float64)
    z[start] = z1
    if have2.any():
        z[start[have2] + 1] = z2[have2]
    if m3.any():
        z[start[m3] + 1 + have2[m3]] = z3[m3]
    z += rng.normal(0.0, sigma, total)

    # gross outliers: birds/clouds above, multipath below
    n_out_hi = int(sc.get("p_outlier_high", 2e-4) * total)
    n_out_lo = int(sc.get("p_outlier_low", 2e-4) * total)
    cls = np.full(total, 2, dtype=np.uint8)
    if n_out_hi:
        i = rng.choice(total, n_out_hi, replace=False)
        z[i] += rng.uniform(2.0, 60.0, n_out_hi)
        cls[i] = 5
    if n_out_lo:
        i = rng.choice(total, n_out_lo, replace=False)
        z[i] -= rng.uniform(1.0, 25.0, n_out_lo)
        cls[i] = 2
    # class-2 contamination: most *canopy* returns also claim class 2
    u = rng.random(total)
    cls[u < 0.05] = 1
    cls[(u >= 0.05) & (u < 0.07)] = 5

    rn = (np.arange(total) - np.repeat(start, n_ret) + 1).astype(np.int8)
    nr = np.repeat(n_ret, n_ret).astype(np.int8)
    src = np.repeat(line, n_ret).astype(np.uint16)
    xs = np.repeat(x, n_ret)
    ys = np.repeat(y, n_ret)
    gps = np.linspace(0.0, 1.0, total)
    return dict(x=xs, y=ys, z=z, rn=rn, nr=nr, cls=cls, src=src, gps=gps,
                p_zg=zg, p_H=H, p_q=q, p_nret=n_ret.astype(np.int32),
                p_x=x, p_y=y)


# ----------------------------------------------------------------- output ---
def write_laz(path, chunks, scene, epsg=32719):
    """Write an iterable of return-chunks to LAZ (chunked, bounded memory)."""
    import laspy
    import pyproj
    h = laspy.LasHeader(point_format=1, version="1.2")
    h.offsets = [float(np.floor(scene["x0"])), float(np.floor(scene["y0"])), 0.0]
    h.scales = [0.001, 0.001, 0.001]
    h.add_crs(pyproj.CRS.from_epsg(int(epsg)))
    h.system_identifier = "m1-synth v1"[:32]
    h.generating_software = "m1 synth/lidar.py"[:32]
    n_total = 0
    with laspy.open(path, mode="w", header=h,
                    laz_backend=laspy.LazBackend.LazrsParallel) as w:
        for ch in chunks:
            n = int(ch["x"].size)
            if n == 0:
                continue
            rec = laspy.ScaleAwarePointRecord.zeros(n, header=w.header)
            rec.x = ch["x"]
            rec.y = ch["y"]
            rec.z = ch["z"]
            rec.intensity = np.zeros(n, dtype=np.uint16)
            rec.return_number = ch["rn"]
            rec.number_of_returns = ch["nr"]
            rec.classification = ch["cls"]
            rec.scan_direction_flag = np.zeros(n, dtype=np.uint8)
            rec.edge_of_flight_line = np.zeros(n, dtype=np.uint8)
            rec.scan_angle_rank = np.zeros(n, dtype=np.int8)
            rec.user_data = np.zeros(n, dtype=np.uint8)
            rec.point_source_id = ch["src"]
            rec.gps_time = ch["gps"] + n_total * 1e-3
            w.write_points(rec)
            n_total += n
    return n_total
