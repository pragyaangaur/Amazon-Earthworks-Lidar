"""synth.py -- synthetic EBA-like 3 m DTMs with ground truth, for detector tests.

The point of these scenes is to be *realistic enough that thresholds mean
something*, and to contain one honest negative per false-positive family named
in the project brief.  Calibration targets (Acre terra firme, cleared pasture
and canopy, from the brief's run-1 lessons):

  * broad relief: a few metres over 1 km (regional slopes 0.2-2 %);
  * local relief at 15 m scale: 0.15-0.35 m RMS (this is what competes with a
    ditch), plus 0.03-0.08 m micro-roughness;
  * canopy penetration bias: up to +0.5-0.7 m in patches where the DTM is
    contaminated by vegetation returns (the class-2 problem);
  * data holes 1-4 % of cells, plus scene-specific big gaps;
  * ditches 3-15 m wide, 0.5-2.0 m deep, often with a low bank.

Every scene returns h (float32, NaN = no data), canopy (CHM proxy), the ground
truth list and metadata.  `truth.expect` is one of
  positive             must be found and screened in
  ring_like_negative   may be found geometrically but must carry reject flags
  negative             must not be screened in (finding it is a false positive)
  line_positive / mound_positive   for the line and mound detectors
"""
from __future__ import annotations

import math

import numpy as np

CELL = 3.0


# --------------------------------------------------------------------------- #
# random fields
# --------------------------------------------------------------------------- #
def selfaffine(n, H, seed, cell=CELL):
    """Unit-RMS self-affine surface with Hurst exponent H (0=white, 1=smooth)."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(n, d=cell)
    k = np.hypot(ky[:, None], ky[None, :])
    k[0, 0] = 1e-9
    amp = k ** (-(H + 1.0))
    amp[0, 0] = 0.0
    ph = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    z = np.real(np.fft.ifft2(amp * ph))
    z -= z.mean()
    s = z.std()
    return (z / s if s > 0 else z).astype(np.float32)


def local_relief_rms(z, cell=CELL, sigma_m=15.0):
    """RMS of z minus its Gaussian(sigma_m) -- the statistic the scenes target."""
    from scipy import ndimage as ndi
    return float(np.std(z - ndi.gaussian_filter(z, sigma_m / cell, mode="nearest")))


# --------------------------------------------------------------------------- #
# feature builders (all add to the ground surface)
# --------------------------------------------------------------------------- #
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ditch(d, width, depth, edge=2.0):
    """Trough SDF profile: depth at the centreline, zero at |d| >= width/2."""
    half = 0.5 * float(width)
    t = _smoothstep((half - np.abs(d)) / max(edge, 0.5))
    return -float(depth) * t


def bank(d, width, height, offset=1.5, sigma=1.8):
    half = 0.5 * float(width)
    return float(height) * np.exp(-((np.abs(d) - half - offset) / sigma) ** 2)


def add_ring(g, X, Y, cx, cy, R, width, depth, bank_h=0.0, phi0=0.0, phi1=360.0,
             plaza_dm=0.0, edge=2.0):
    """Circular ditch (optionally an arc/U with rounded ends). Metres in, adds to g."""
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy)
    prof = ditch(r - R, width, depth, edge) + bank(r - R, width, bank_h)
    if not (abs(phi1 - phi0) >= 359.9):
        phi = np.degrees(np.arctan2(dy, dx)) % 360.0
        p0, p1 = phi0 % 360.0, phi1 % 360.0
        inside = ((phi >= p0) & (phi <= p1)) if p1 > p0 else ((phi >= p0) | (phi <= p1))
        e0 = (cx + R * math.cos(math.radians(p0)), cy + R * math.sin(math.radians(p0)))
        e1 = (cx + R * math.cos(math.radians(p1)), cy + R * math.sin(math.radians(p1)))
        d0 = np.hypot(X - e0[0], Y - e0[1])
        d1 = np.hypot(X - e1[0], Y - e1[1])
        # rounded caps: use the plane distance to the end points outside the arc
        prof_cap = 0.5 * (ditch(np.maximum(d0 - width * 0.5, 0) - width * 0.5, width, depth, edge)
                          + ditch(np.maximum(d1 - width * 0.5, 0) - width * 0.5, width, depth, edge))
        prof = np.where(inside, prof, prof_cap)
    if plaza_dm:
        inner = _smoothstep((0.75 * R - r) / max(0.2 * R, 1.0))
        prof = prof + float(plaza_dm) * inner
    g += prof
    return g


def rect_sdf(dx, dy, a, b):
    """Signed distance to the boundary of the a x b rectangle (half sizes)."""
    qx = np.abs(dx) - a
    qy = np.abs(dy) - b
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return np.abs(out + inside)


def add_rect(g, X, Y, cx, cy, a, b, width, depth, bank_h=0.0, angle_deg=0.0, edge=2.0):
    dx, dy = X - cx, Y - cy
    th = math.radians(angle_deg)
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    d = rect_sdf(u, v, a, b)
    g += ditch(d, width, depth, edge) + bank(d, width, bank_h)
    return g


def _seg_dist(X, Y, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((X - x0) * vx + (Y - y0) * vy) / max(L2, 1e-9), 0.0, 1.0)
    return np.hypot(X - (x0 + t * vx), Y - (y0 + t * vy))


def add_road(g, X, Y, x0, y0, x1, y1, width, depth, bank_h=0.0, edge=1.2):
    d = _seg_dist(X, Y, x0, y0, x1, y1)
    g += ditch(d, width, depth, edge) + bank(d, width, bank_h)
    return g


def add_polyline_trough(g, X, Y, pts, width, depth, levee=0.0, edge=1.5):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(pts))
    d, _ = tree.query(np.c_[X.ravel(), Y.ravel()], workers=1)
    d = d.reshape(X.shape)
    g += ditch(d, width, depth, edge)
    if levee:
        g += bank(d, width, levee, offset=2.0, sigma=2.5)
    return g


def add_pond(g, X, Y, cx, cy, R, depth, dam_h=0.8, dam_arc=(200.0, 290.0)):
    r = np.hypot(X - cx, Y - cy)
    bowl = -float(depth) * _smoothstep((R - r) / (0.35 * R))
    phi = (np.degrees(np.arctan2(Y - cy, X - cx)) + 360.0) % 360.0
    inarc = (phi >= dam_arc[0]) & (phi <= dam_arc[1])
    dam = float(dam_h) * np.exp(-((r - R - 1.0) / 2.5) ** 2) * inarc
    g += bowl + dam
    return g


def add_valley(g, X, Y, x0, y0, x1, y1, width, depth):
    d = _seg_dist(X, Y, x0, y0, x1, y1)
    g += -float(depth) * np.exp(-(d / float(width)) ** 2)
    return g


def add_hilltop_gullies(g, X, Y, cx, cy, hill_h=1.5, hill_sigma=60.0, n_gullies=6,
                        gully_depth=1.0, seed=0):
    r = np.hypot(X - cx, Y - cy)
    g += float(hill_h) * np.exp(-(r / float(hill_sigma)) ** 2)
    rng = np.random.default_rng(seed)
    for k in range(n_gullies):
        th = rng.uniform(0, 2 * math.pi)
        L = hill_sigma * 3.0
        ex, ey = cx + L * math.cos(th), cy + L * math.sin(th)
        d = _seg_dist(X, Y, cx, cy, ex, ey)
        widen = 4.0 + 0.10 * r
        g += -float(gully_depth) * np.exp(-(d / widen) ** 2) * _smoothstep((r - 8.0) / 25.0)
    return g


def add_mound(g, X, Y, cx, cy, R, h):
    r = np.hypot(X - cx, Y - cy)
    g += float(h) * np.exp(-(r / float(R)) ** 2)
    return g


# --------------------------------------------------------------------------- #
# scene assembly
# --------------------------------------------------------------------------- #
def make_scene(n=420, cell=CELL, seed=0, canopy="open", lr15=0.22, micro=0.05,
               slope=0.004, slope_dir=30.0, veg_bias=0.0, holes=0.02,
               big_holes=(), features=(), name="scene"):
    """Assemble one synthetic scene.

    canopy: 'open' (CHM ~0-1 m), 'mosaic' (patches to 25 m), 'forest' (closed).
    big_holes: sequence of (cx, cy, R_m) discs with no data at all.
    features: list of (kind, kwargs) applied to the *ground* before the canopy bias.
    """
    rng = np.random.default_rng(seed)
    broad = selfaffine(n, 0.85, seed + 1, cell)
    rough = selfaffine(n, 0.45, seed + 2, cell)
    # scale the rough part so the 15 m local relief matches the target
    rr = local_relief_rms(rough, cell, 15.0)
    rough *= (lr15 / max(rr, 1e-6))
    br = local_relief_rms(broad, cell, 15.0)
    broad *= 1.0
    broad_lr = 0.12 * lr15
    broad *= (broad_lr / max(br, 1e-6))
    gy, gx = np.mgrid[0:n, 0:n]
    X = gx * cell
    Y = (n - 1 - gy) * cell            # north-up: row 0 = max y
    g = broad + rough
    th = math.radians(slope_dir)
    g = g + slope * (X * math.cos(th) + (Y - Y.mean()) * math.sin(th)) + 40.0
    g = g + float(micro) * rng.normal(size=(n, n)).astype(np.float32)

    # canopy field
    chm = np.zeros((n, n), np.float32)
    if canopy != "open":
        f = selfaffine(n, 0.9, seed + 7, cell)
        if canopy == "forest":
            chm = 22.0 + 6.0 * f
        else:  # mosaic of pasture and forest patches
            cov = _smoothstep((f - 0.15) / 0.5)
            chm = (2.0 + 24.0 * cov) * (1.0 + 0.15 * selfaffine(n, 0.4, seed + 8, cell))
        chm = np.clip(chm, 0.0, 40.0).astype(np.float32)

    ground = g.copy()
    for kind, kw in features:
        fn = {"ring": add_ring, "rect": add_rect, "road": add_road, "pond": add_pond,
              "valley": add_valley, "gullies": add_hilltop_gullies, "mound": add_mound,
              "poly": add_polyline_trough}[kind]
        fn(ground, X, Y, **kw)

    # DTM = ground + canopy penetration bias + noise, sampled on the 3 m grid
    bias = np.zeros((n, n), np.float32)
    if veg_bias > 0 and canopy != "open":
        f2 = np.abs(selfaffine(n, 0.5, seed + 9, cell))
        bias = (veg_bias * (0.3 + 0.7 * np.clip(f2 / max(f2.max(), 1e-6), 0, 1))
                * _smoothstep((chm - 2.0) / 6.0)).astype(np.float32)
    dtm = ground + bias + (0.02 + 0.04 * np.clip(chm / 25.0, 0, 1)) * \
        rng.normal(size=(n, n)).astype(np.float32)

    valid = np.ones((n, n), bool)
    if holes > 0:
        # clustered gaps (dense canopy patches, water, buildings) + sparse dropouts:
        # scattered single-cell holes are not what EBA transects look like and they
        # make the 15 m hole blanking destroy every scene.
        hf = selfaffine(n, 0.95, seed + 11, cell)
        valid &= hf > np.quantile(hf, holes * 0.8)
        drop = np.random.default_rng(seed + 12).random((n, n)) < holes * 0.2
        valid &= ~drop
    for (hx, hy, hr) in big_holes:
        # cx,cy given in metres in the scene frame
        col = hx / cell - 0.5
        row = (Y.max() - hy) / cell - 0.5
        valid &= np.hypot(gx - col, gy - row) > hr / cell
    # rim artefact around big holes: a raised lip (edge interpolation blunder)
    rim = np.zeros((n, n), np.float32)
    for (hx, hy, hr) in big_holes:
        col = hx / cell - 0.5
        row = (Y.max() - hy) / cell - 0.5
        d = np.hypot(gx - col, gy - row) * cell
        rim += 0.5 * np.exp(-((d - hr - 2 * cell) / (1.5 * cell)) ** 2)
    dtm += rim

    dtm = np.where(valid, dtm, np.nan).astype(np.float32)
    return dict(h=dtm, canopy=(chm if canopy != "open" else None), cell=cell,
                x0=0.0, y0=0.0, n=n, X=X, Y=Y, name=name,
                diag=dict(lr15_meas=local_relief_rms(np.where(valid, dtm, np.nanmean(dtm))),
                          valid_frac=float(valid.mean()), chm_med=float(np.median(chm)),
                          relief_std=float(np.std(np.where(valid, dtm, np.nanmedian(dtm))))),
                features=features)


def _m(X, Y):
    return X, Y


SCENES = {}


def _ring_scene(name, **kw):
    SCENES[name] = kw


def build_all(n=420, seed0=100):
    """The validation suite: 4 ring positives, 1 U/arc, 1 large ring under canopy,
    1 mound ring, 1 road, and 6 negatives (one per documented FP family)."""
    S = {}
    base = dict(n=n, cell=CELL, can=dict())
    S["P1_ring40_open"] = dict(
        seed=seed0 + 1, canopy="open", lr15=0.18, veg_bias=0.0,
        truth=[dict(id="P1", kind="ring", cx=630.0, cy=630.0, R=20.0, width=6.0,
                    depth=1.2, expect="positive")],
        features=[("ring", dict(cx=630.0, cy=630.0, R=20.0, width=6.0, depth=1.2, bank_h=0.30)),
                  ("road", dict(x0=650.0, y0=630.0, x1=760.0, y1=630.0, width=5.0,
                                depth=0.7, bank_h=0.25)),
                  ("road", dict(x0=630.0, y0=650.0, x1=630.0, y1=740.0, width=5.0,
                                depth=0.7, bank_h=0.25))])
    S["P2_ring250_open"] = dict(
        seed=seed0 + 2, canopy="open", lr15=0.20, veg_bias=0.0,
        truth=[dict(id="P2", kind="ring", cx=630.0, cy=630.0, R=125.0, width=9.0,
                    depth=2.0, expect="positive")],
        features=[("ring", dict(cx=630.0, cy=630.0, R=125.0, width=9.0, depth=2.0, bank_h=0.40))])
    S["P3_square90"] = dict(
        seed=seed0 + 3, canopy="open", lr15=0.20, veg_bias=0.0,
        truth=[dict(id="P3", kind="square", cx=630.0, cy=630.0, R=49.0, width=8.0,
                    depth=1.5, expect="positive")],
        features=[("rect", dict(cx=630.0, cy=630.0, a=45.0, b=45.0, width=8.0, depth=1.5,
                                bank_h=0.30))])
    S["P4_U120"] = dict(
        seed=seed0 + 4, canopy="open", lr15=0.18, veg_bias=0.0,
        truth=[dict(id="P4", kind="U", cx=630.0, cy=630.0, R=60.0, width=7.0, depth=1.3,
                    expect="positive")],
        features=[("ring", dict(cx=630.0, cy=630.0, R=60.0, width=7.0, depth=1.3, bank_h=0.30,
                                phi0=20.0, phi1=300.0))])
    S["P5_ring210_mosaic"] = dict(
        holes=0.02, seed=seed0 + 5, canopy="mosaic", lr15=0.30, veg_bias=0.45, 
        big_holes=[(400.0, 300.0, 25.0)],
        truth=[dict(id="P5", kind="ring", cx=630.0, cy=630.0, R=105.0, width=10.0,
                    depth=1.8, expect="positive")],
        features=[("ring", dict(cx=630.0, cy=630.0, R=105.0, width=10.0, depth=1.8,
                                bank_h=0.35))])
    S["P6_moundring"] = dict(
        seed=seed0 + 6, canopy="open", lr15=0.18, veg_bias=0.0,
        truth=[dict(id="P6", kind="moundring", cx=630.0, cy=630.0, R=45.0, width=12.0,
                    depth=1.5, expect="mound_positive")],
        features=[("mound", dict(cx=630.0 + 45 * math.cos(k * math.pi / 3),
                                 cy=630.0 + 45 * math.sin(k * math.pi / 3), R=6.0, h=1.5))
                  for k in range(6)])
    S["P7_road300"] = dict(
        seed=seed0 + 7, canopy="open", lr15=0.20, veg_bias=0.0,
        truth=[dict(id="P7", kind="road", cx=630.0, cy=630.0, R=0.0, width=5.0, depth=0.9,
                    expect="line_positive", x0=300.0, y0=400.0, x1=650.0, y1=560.0)],
        features=[("road", dict(x0=300.0, y0=400.0, x1=650.0, y1=560.0, width=5.0,
                                depth=0.9, bank_h=0.30))])
    # ---- negatives --------------------------------------------------------
    S["N1_meander_oxbow"] = dict(
        holes=0.01, seed=seed0 + 11, canopy="open", lr15=0.25, veg_bias=0.0, 
        truth=[dict(id="N1ox", kind="oxbow", cx=700.0, cy=485.0, R=55.0, width=22.0,
                    depth=2.4, expect="ring_like_negative"),
               dict(id="N1riv", kind="river", cx=0.0, cy=0.0, R=0.0, width=22.0,
                    depth=3.0, expect="negative")],
        features=[(["poly", dict(pts=[(x, 430.0 + 120.0 * math.sin(2 * math.pi * x / 700.0))
                                     for x in np.arange(-100.0, 1400.0, 6.0)],
                                width=22.0, depth=3.0, levee=0.4)]),
                  ("ring", dict(cx=700.0, cy=485.0, R=55.0, width=22.0, depth=2.4))])
    S["N2_pond"] = dict(
        seed=seed0 + 12, canopy="open", lr15=0.20, veg_bias=0.0, holes=0.02,
        big_holes=[],
        truth=[dict(id="N2", kind="pond", cx=620.0, cy=620.0, R=20.0, width=40.0,
                    depth=1.5, expect="ring_like_negative")],
        features=[("pond", dict(cx=620.0, cy=620.0, R=20.0, depth=1.5, dam_h=0.8))])
    S["N3_valley"] = dict(
        seed=seed0 + 13, canopy="open", lr15=0.22, veg_bias=0.0,
        truth=[dict(id="N3", kind="valley", cx=0.0, cy=630.0, R=0.0, width=60.0,
                    depth=2.0, expect="negative")],
        features=[("valley", dict(x0=-100.0, y0=630.0, x1=1400.0, y1=630.0, width=30.0,
                                  depth=2.0))])
    S["N4_holehalo"] = dict(
        seed=seed0 + 14, canopy="open", lr15=0.20, veg_bias=0.0, holes=0.01,
        big_holes=[(620.0, 620.0, 40.0), (350.0, 350.0, 22.0), (900.0, 800.0, 15.0)],
        truth=[dict(id="N4", kind="hole", cx=620.0, cy=620.0, R=40.0, width=0.0,
                    depth=0.0, expect="negative")], features=[])
    S["N5_hilltop_gullies"] = dict(
        seed=seed0 + 15, canopy="open", lr15=0.24, veg_bias=0.0,
        truth=[dict(id="N5", kind="gullies", cx=630.0, cy=630.0, R=0.0, width=0.0,
                    depth=0.0, expect="negative")],
        features=[("gullies", dict(cx=630.0, cy=630.0, hill_h=1.5, hill_sigma=60.0,
                                   n_gullies=6, gully_depth=0.9, seed=3))])
    S["N6_fields_modern"] = dict(
        seed=seed0 + 16, canopy="open", lr15=0.20, veg_bias=0.0,
        truth=[dict(id="N6", kind="fields", cx=630.0, cy=630.0, R=0.0, width=0.0,
                    depth=0.0, expect="negative")],
        features=[("rect", dict(cx=560.0, cy=560.0, a=90.0, b=70.0, width=4.0, depth=0.35,
                                angle_deg=12.0)),
                  ("road", dict(x0=200.0, y0=900.0, x1=900.0, y1=760.0, width=6.0,
                                depth=0.30))])
    S["N7_terrain_null"] = dict(
        seed=seed0 + 17, canopy="open", lr15=0.24, veg_bias=0.0,
        truth=[dict(id="N7", kind="none", cx=0.0, cy=0.0, R=0.0, width=0.0, depth=0.0,
                    expect="negative")], features=[])
    S["N9_pond_wet"] = dict(
        holes=0.01, seed=seed0 + 19, canopy="open", lr15=0.20, veg_bias=0.0, 
        big_holes=[(620.0, 620.0, 17.0)],
        truth=[dict(id="N9", kind="hole", cx=620.0, cy=620.0, R=20.0, width=0.0,
                    depth=0.0, expect="negative")],
        features=[("pond", dict(cx=620.0, cy=620.0, R=20.0, depth=1.5, dam_h=0.8))])
    S["N8_forest_null"] = dict(
        holes=0.05, seed=seed0 + 18, canopy="forest", lr15=0.34, veg_bias=0.55, 
        truth=[dict(id="N8", kind="none", cx=0.0, cy=0.0, R=0.0, width=0.0, depth=0.0,
                    expect="negative")], features=[])
    return S


def build(name, spec):
    holes = spec.get("holes", 0.005)
    sc = make_scene(n=spec.get("n", 420), cell=CELL, seed=spec["seed"],
                    canopy=spec.get("canopy", "open"), lr15=spec.get("lr15", 0.22),
                    micro=spec.get("micro", 0.05), slope=spec.get("slope", 0.004),
                    veg_bias=spec.get("veg_bias", 0.0), holes=holes,
                    big_holes=spec.get("big_holes", ()),
                    features=spec.get("features", []), name=name)
    sc["truth"] = spec.get("truth", [])
    return sc
