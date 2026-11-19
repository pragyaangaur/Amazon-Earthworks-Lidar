"""Analytic ground surface for synthetic EBA-like airborne-lidar scenes (m1).

Every feature is a plain dict so a whole scene is JSON-serialisable
(that is what makes the ground truth auditable).

Conventions
-----------
* Local scene coordinates in metres; `x` east, `y` north, `z` up.
* A feature returns a vertical displacement added to the base surface.
* Depths are negative displacements; banks / mounds / causeways positive.
* All helpers are vectorised over numpy arrays of x, y.

Feature types (see `FEATURE_TYPES`)
-----------------------------------
ring_ditch, square_ditch, road, causeway, mound, mound_ring, meander,
oxbow, scroll, vvalley, pond, hill, gully, modern_road, pad.

The cultural features (ring_ditch, square_ditch, road, causeway, mound_ring)
are the detection targets; the rest exist to generate honest false-positive
material (meandering rivers, oxbows, scroll bars, broad V stream valleys,
cattle ponds, hilltops cut by gullies, graded modern roads, clearings).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------- helpers --
def smoothstep(u):
    """C1 ramp from 0 to 1 over u in [0, 1]."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def flat_top(t, half, edge=0.5):
    """1 for |t| < edge*half, smooth fall to 0 at |t| = half.

    Ditches and channels have a flat floor and steep walls; a Gaussian would
    be too smooth and would flatter the detectors.  `half` may be an array
    (a river channel that widens and narrows along its course).
    """
    half = np.asarray(half, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.all(half <= 0.0):
        return np.zeros(np.broadcast(t, half).shape)
    u = (np.abs(t) - edge * half) / np.maximum((1.0 - edge) * half, EPS)
    return 1.0 - smoothstep(u)


def arc_gate(ang_deg, arc, taper_deg=8.0):
    """Smooth angular window: 1 inside arc=(a0, a1) degrees, 0 outside."""
    a0, a1 = float(arc[0]), float(arc[1])
    span = a1 - a0
    if span >= 359.0:
        return np.ones_like(ang_deg, dtype=float)
    d0 = np.mod(ang_deg - a0, 360.0)
    inside = d0 <= span
    g = smoothstep(d0 / taper_deg) * smoothstep((span - d0) / taper_deg)
    return np.where(inside, g, 0.0)


def seg_dist(x, y, p0, p1):
    """Distance to segment p0->p1 and the unclipped normalised projection t."""
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = vx * vx + vy * vy
    t = ((x - p0[0]) * vx + (y - p0[1]) * vy) / (l2 + EPS)
    tc = np.clip(t, 0.0, 1.0)
    d = np.hypot(x - (p0[0] + tc * vx), y - (p0[1] + tc * vy))
    return d, t


def _end_gate(t, taper):
    """Taper a linear feature to zero over `taper` metres at both ends."""
    if taper <= 0:
        return np.ones_like(t, dtype=float)
    length = 1.0  # t is normalised; caller passes taper as a fraction
    return smoothstep(t / taper) * smoothstep((length - t) / taper)


# ------------------------------------------------------------------- base ---
def base_surface(x, y, scene):
    """Regional relief: plane + smooth blobs. No cultural signal here."""
    b = scene["base"]
    z = b["z0"] + b["sx"] * (x - scene["x0"]) + b["sy"] * (y - scene["y0"])
    for blob in b.get("blobs", []):
        cx, cy, amp, sig = blob
        z = z + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sig * sig))
    for wave in b.get("waves", []):
        amp, lam, ang, ph = wave
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        u = (x - scene["x0"]) * ca + (y - scene["y0"]) * sa
        z = z + amp * np.sin(2.0 * np.pi * u / lam + ph)
    return z


# --------------------------------------------------------------- features ---
def ring_ditch(x, y, f):
    """Circular / elliptical ditch with optional low bank and angular extent."""
    dx, dy = x - f["cx"], y - f["cy"]
    eps = float(f.get("eps", 0.0))          # 0 = circle, 0.1 = 10% ellipse
    ang = np.degrees(np.arctan2(dy, dx))
    rr = np.hypot(dx / (1.0 + eps), dy)
    r = float(f["r"])
    wob = f.get("wob")                      # (amplitude m, cycles, phase rad)
    if wob:
        r_eff = r + wob[0] * np.sin(wob[1] * np.radians(ang) + wob[2])
    else:
        r_eff = r
    t = rr - r_eff
    w = float(f["width"])
    dz = -float(f["depth"]) * flat_top(t, w / 2.0)
    bank = float(f.get("bank", 0.0))
    if bank:
        bw = float(f.get("bank_w", 0.55 * w))
        c = w / 2.0 + bw / 2.0
        dz = dz + bank * np.exp(-((np.abs(t) - c) ** 2) / (2.0 * (bw / 2.5) ** 2))
    return dz * arc_gate(ang, f.get("arc", (0.0, 360.0)), float(f.get("taper", 8.0)))


def square_ditch(x, y, f):
    """Square (or U-shaped) ditch: flat sides at constant |dx|,|dy| = r."""
    dx, dy = x - f["cx"], y - f["cy"]
    rot = np.radians(float(f.get("rot", 0.0)))
    if rot:
        ca, sa = np.cos(rot), np.sin(rot)
        dx, dy = ca * dx + sa * dy, -sa * dx + ca * dy
    s = np.maximum(np.abs(dx), np.abs(dy))
    t = s - float(f["r"])
    w = float(f["width"])
    dz = -float(f["depth"]) * flat_top(t, w / 2.0)
    bank = float(f.get("bank", 0.0))
    if bank:
        bw = float(f.get("bank_w", 0.55 * w))
        c = w / 2.0 + bw / 2.0
        dz = dz + bank * np.exp(-((np.abs(t) - c) ** 2) / (2.0 * (bw / 2.5) ** 2))
    opening = f.get("opening")
    if opening:  # remove one side (or its middle) to make a U / C shape
        side = opening["side"]              # 'east','west','north','south'
        gap = float(opening.get("gap", 2.0 * f["r"]))
        if side in ("east", "west"):
            perp, along = dx, dy
            sign = 1.0 if side == "east" else -1.0
        else:
            perp, along = dy, dx
            sign = 1.0 if side == "north" else -1.0
        s_along = smoothstep((0.5 * gap - np.abs(along)) / (0.15 * gap + EPS))
        s_side = smoothstep(sign * perp / (0.25 * w + EPS))
        dz = dz * (1.0 - s_along * s_side)
    return dz


def road(x, y, f):
    """Straight sunken (depth>0) or embanked (depth<0) road with side banks."""
    p0, p1 = f["p0"], f["p1"]
    d, t = seg_dist(x, y, p0, p1)
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    taper = float(f.get("taper_m", 0.0)) / max(length, EPS)
    g = _end_gate(t, taper) if taper > 0 else ((t >= 0.0) & (t <= 1.0)).astype(float)
    w = float(f["width"])
    dz = -float(f["depth"]) * flat_top(d, w / 2.0) * g
    bank = float(f.get("bank", 0.0))
    if bank:
        bw = float(f.get("bank_w", 0.5 * w))
        c = w / 2.0 + bw / 2.0
        dz = dz + bank * np.exp(-((d - c) ** 2) / (2.0 * (bw / 2.5) ** 2)) * g
    return dz


def causeway(x, y, f):
    """Embanked causeway: a long low ridge with sloping sides."""
    p0, p1 = f["p0"], f["p1"]
    d, t = seg_dist(x, y, p0, p1)
    length = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    taper = float(f.get("taper_m", 25.0)) / max(length, EPS)
    g = _end_gate(t, taper)
    w = float(f["width"])
    return float(f["height"]) * flat_top(d, w / 2.0, edge=0.65) * g


def mound(x, y, f):
    """Single mound: gaussian dome, `r` is the radius where the flank ends."""
    r = np.hypot(x - f["cx"], y - f["cy"])
    sig = float(f["r"]) / 2.2
    return float(f["h"]) * np.exp(-(r ** 2) / (2.0 * sig * sig))


def mound_ring(x, y, f):
    """Ring of k mounds around a plaza; returns displacement of all mounds."""
    cx, cy = f["cx"], f["cy"]
    k = int(f["k"])
    R = float(f["R"])
    a0 = float(f.get("phase", 0.0))
    hs = f["heights"]
    rs = f.get("radii", [8.0] * k)
    wobble = f.get("pos_wobble", 0.05)
    ang = a0 + 2.0 * np.pi * np.arange(k) / k
    dz = np.zeros_like(x, dtype=float)
    for i in range(k):
        rr = R * (1.0 + wobble * np.sin(3.0 * ang[i] + 1.3 * i))
        mx, my = cx + rr * np.cos(ang[i]), cy + rr * np.sin(ang[i])
        sig = float(rs[i]) / 2.2
        dz = dz + float(hs[i]) * np.exp(
            -(((x - mx) ** 2 + (y - my) ** 2)) / (2.0 * sig * sig))
    return dz


def meander(x, y, f):
    """Sinusoidal river: incised channel (flat floor) + optional levees."""
    X = x - f["x0"]
    lam1, lam2 = float(f["L1"]), float(f["L2"])
    a1, a2 = float(f["A1"]), float(f["A2"])
    p1, p2 = float(f.get("ph1", 0.0)), float(f.get("ph2", 0.0))
    fc = f["yc"] + a1 * np.sin(2 * np.pi * X / lam1 + p1) + a2 * np.sin(2 * np.pi * X / lam2 + p2)
    df = (a1 * 2 * np.pi / lam1 * np.cos(2 * np.pi * X / lam1 + p1)
          + a2 * 2 * np.pi / lam2 * np.cos(2 * np.pi * X / lam2 + p2))
    d = (y - fc) / np.sqrt(1.0 + df * df)
    W = float(f["width"]) * (1.0 + float(f.get("wvar", 0.3)) * np.sin(2 * np.pi * X / float(f.get("Lw", 400.0)) + float(f.get("phw", 0.0))))
    inx = smoothstep(X / 30.0) * smoothstep((float(f["length"]) - X) / 30.0)
    dz = -float(f["depth"]) * flat_top(d, W / 2.0, edge=0.55) * inx
    lev = float(f.get("levee", 0.0))
    if lev:
        lw = float(f.get("levee_w", 12.0))
        c = W / 2.0 + lw / 2.0
        dz = dz + lev * np.exp(-((np.abs(d) - c) ** 2) / (2.0 * (lw / 2.5) ** 2)) * inx
    return dz


def oxbow(x, y, f):
    """Abandoned meander loop: an arc-shaped channel, often a near-complete ring."""
    dx, dy = x - f["cx"], y - f["cy"]
    ang = np.degrees(np.arctan2(dy, dx))
    rr = np.hypot(dx, dy)
    w = float(f["width"])
    t = rr - float(f["r"])
    dz = -float(f["depth"]) * flat_top(t, w / 2.0, edge=0.55)
    lev = float(f.get("levee", 0.0))
    if lev:
        lw = float(f.get("levee_w", 10.0))
        c = w / 2.0 + lw / 2.0
        dz = dz + lev * np.exp(-((np.abs(t) - c) ** 2) / (2.0 * (lw / 2.5) ** 2))
    return dz * arc_gate(ang, f["arc"], float(f.get("taper", 20.0)))


def scroll(x, y, f):
    """Scroll-bar ridge-and-swale texture on a floodplain."""
    X = x - f["x0"]
    fc = f["yc"] + float(f.get("tilt", 0.0)) * X
    d = y - fc
    env = np.exp(-(d ** 2) / (2.0 * float(f["sigma"]) ** 2))
    inx = smoothstep(X / 40.0) * smoothstep((float(f["length"]) - X) / 40.0)
    return float(f["amp"]) * np.sin(2 * np.pi * d / float(f["lam"]) + float(f.get("ph", 0.0))) * env * inx


def vvalley(x, y, f):
    """Broad V-shaped stream valley with no banks (classic false positive)."""
    d, t = seg_dist(x, y, f["p0"], f["p1"])
    hw = float(f["halfwidth"])
    inx = ((t >= 0.0) & (t <= 1.0)).astype(float)
    return -float(f["depth"]) * np.clip(1.0 - d / hw, 0.0, 1.0) ** 1.4 * inx


def pond(x, y, f):
    """Cattle pond: flat-bottomed depression with an earth dam on one side.

    The dam is a short positive arc ridge at the pond rim - exactly the kind
    of half-ring that a radial screen can mistake for a banked ditch.
    """
    dx, dy = x - f["cx"], y - f["cy"]
    rr = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))
    r = float(f["r"])
    # flat floor out to 0.85 r, steep rim wall, nothing outside r
    dz = -float(f["depth"]) * (1.0 - smoothstep((rr - 0.85 * r) / (0.15 * r)))
    dam = float(f.get("dam_h", 0.0))
    if dam:
        w = float(f.get("dam_w", 6.0))
        dz = dz + dam * np.exp(-((rr - r) ** 2) / (2.0 * (w / 2.5) ** 2)) \
            * arc_gate(ang, f["dam_arc"], 12.0)
    return dz


def hill(x, y, f):
    """Natural hilltop: broad dome, a false positive for relief-based screens."""
    r = np.hypot(x - f["cx"], y - f["cy"])
    return float(f["h"]) * np.exp(-(r ** 2) / (2.0 * (float(f["r"]) / 1.6) ** 2))


def gully(x, y, f):
    """Gully: narrow incision, negative ridge."""
    d, t = seg_dist(x, y, f["p0"], f["p1"])
    inx = ((t >= 0.0) & (t <= 1.0)).astype(float)
    return -float(f["depth"]) * np.exp(-(d ** 2) / (2.0 * (float(f["width"]) / 2.0) ** 2)) * inx


def modern_road(x, y, f):
    """Graded modern road: flat cut plus small lateral berms."""
    d, t = seg_dist(x, y, f["p0"], f["p1"])
    w = float(f["width"])
    inx = ((t >= 0.0) & (t <= 1.0)).astype(float)
    dz = -float(f.get("cut", 0.35)) * flat_top(d, w / 2.0, edge=0.7)
    berm = float(f.get("berm", 0.25))
    if berm:
        dz = dz + berm * np.exp(-((d - w / 2.0 - 2.0) ** 2) / (2.0 * 1.5 ** 2))
    return dz * inx


def pad(x, y, f):
    """Modern clearing / house pad: a levelled rectangle cut into the slope."""
    cx, cy = f["cx"], f["cy"]
    hw, hh = float(f["w"]) / 2.0, float(f["h"]) / 2.0
    ins = (np.abs(x - cx) < hw) & (np.abs(y - cy) < hh)
    return np.where(ins, float(f.get("dz", 0.0)), 0.0)


def sample_field(arr, scene, x, y, g=1.0):
    """Bilinear sample of a raster given scene-local coordinates."""
    ny, nx = arr.shape
    col = np.clip((x - scene["x0"]) / g - 0.5, 0.0, nx - 1.0)
    row = np.clip((y - scene["y0"]) / g - 0.5, 0.0, ny - 1.0)
    i0 = np.floor(col).astype(np.int64)
    j0 = np.floor(row).astype(np.int64)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    fx = col - i0
    fy = row - j0
    a = arr[j0, i0] * (1 - fx) + arr[j0, i1] * fx
    b = arr[j1, i0] * (1 - fx) + arr[j1, i1] * fx
    return a * (1 - fy) + b * fy


FEATURE_TYPES = {
    "ring_ditch": ring_ditch,
    "square_ditch": square_ditch,
    "road": road,
    "causeway": causeway,
    "mound": mound,
    "mound_ring": mound_ring,
    "meander": meander,
    "oxbow": oxbow,
    "scroll": scroll,
    "vvalley": vvalley,
    "pond": pond,
    "hill": hill,
    "gully": gully,
    "modern_road": modern_road,
    "pad": pad,
}

CULTURAL = {"ring_ditch", "square_ditch", "road", "causeway", "mound", "mound_ring"}
# truth-only annotations: they describe a control (e.g. an empty-data halo)
# but contribute no terrain displacement.
ANNOTATION = {"hole_halo"}
NATURAL = {"meander", "oxbow", "scroll", "vvalley", "pond", "hill", "gully"}
MODERN = {"modern_road", "pad"}


def ground(x, y, scene, skip=()):
    """Full ground surface: base relief + micro-relief + every feature."""
    z = base_surface(x, y, scene)
    rough = scene.get("_rough_field")
    if rough is not None:
        z = z + sample_field(rough, scene, x, y, g=scene.get("_rough_grid_m", 1.0))
    for f in scene["features"]:
        if f["id"] in skip:
            continue
        if f["type"] in ANNOTATION:
            continue
        fn = FEATURE_TYPES.get(f["type"])
        if fn is None:
            raise KeyError("unknown feature type %r" % f["type"])
        z = z + fn(x, y, f)
    return z
