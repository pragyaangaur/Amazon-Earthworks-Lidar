"""synth.py -- seeded synthetic 3 m DTMs with ground truth for validating the
causeway / raised-linear-earthwork detector before it is trusted on real lidar.

Each scenario returns a dict:
    name   : scenario name
    z      : float32 (nrows, ncols) DTM, NaN where the survey has no data
    meta   : dtlib meta (synthetic local grid, EPSG 0)
    chm    : float32 canopy height model (0 = open pasture), or None
    truth  : list of objects {kind, geom, width_m, height_m, must_find}
    note   : what the scenario is testing

The grid bbox is derived from the scenario geometry, so nothing is clipped; every
returned scenario is checked with `validate()` before use.

member m3 (DETECTOR), swarm smwduocxj
"""
import math

import numpy as np
from scipy import ndimage as ndi
import shapely
from shapely.geometry import LineString, Point

import dtlib as D

DX = 3.0


# ---------------------------------------------------------------- grid helper
def _meta_grid(xmin, xmax, ymin, ymax, dx=DX):
    n = int(np.ceil((ymax - ymin) / dx))
    m = int(np.ceil((xmax - xmin) / dx))
    return D.make_meta(dx=dx, dy=dx, x0=xmin, y0=ymax, epsg=0,
                       source="synthetic"), n, m


def _bbox(geoms, pad=90.0):
    b = np.array([g.bounds for g in geoms], float)
    return (b[:, 0].min() - pad, b[:, 2].max() + pad,
            b[:, 1].min() - pad, b[:, 3].max() + pad)


def _dist(z, meta, geom):
    X, Y = D.cell_centres(z.shape, meta)
    return shapely.distance(shapely.points(X.ravel(), Y.ravel()),
                            geom).reshape(z.shape)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- primitives
def fractal_noise(shape, seed, amps=((6.0, 0.04), (14.0, 0.10), (40.0, 0.22),
                                     (120.0, 0.40)), dx=DX):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.float64)
    cap = max(1.0, min(shape) / 8.0)          # a sigma wider than the tile is meaningless
    for sig_m, amp in amps:
        sig = float(np.clip(sig_m / dx, 0.5, cap))
        g = ndi.gaussian_filter(rng.normal(size=shape), sig, mode="wrap")
        g -= g.mean()
        sd = g.std()
        if sd < 1e-9:
            continue
        out += amp * (g / sd)
    return out


def carve_channel(z, meta, line, width_m, depth_m, floor_frac=0.4):
    """Flat-bottomed channel of full depth over floor_frac of the width, then a
    smooth bank rising to 0 at width_m/2."""
    d = _dist(z, meta, line)
    w2 = width_m * 0.5
    u = d / w2
    inside = np.clip((1.0 - u) / (1.0 - floor_frac), 0, 1)
    prof = np.where(u <= floor_frac, 1.0, 0.5 * (1 + np.cos(np.pi * inside)))
    prof = np.where(u <= 1.0, prof, 0.0)
    z = z - depth_m * prof
    return z


def build_ridge(z, meta, line, width_m, height_m, dps_m=2.5, crown_flat=0.4,
                taper_frac=0.0, crown_frac=0.55):
    """Embankment: flat crown of height h, smooth flank down to `crown_frac`*h at
    the base half-width, then 0 over dps_m beyond it.  Optional end taper."""
    d = _dist(z, meta, line)
    w2 = width_m * 0.5
    inside = 1.0 - (1.0 - crown_frac) * smoothstep(crown_flat, 1.0, np.clip(d / w2, 0, 1))
    outside = crown_frac * (1.0 - smoothstep(0.0, 1.0, np.clip((d - w2) / dps_m, 0, 1)))
    prof = np.where(d <= w2, inside, outside)
    if taper_frac > 0:
        X, Y = D.cell_centres(z.shape, meta)
        t = shapely.line_locate_point(line, shapely.points(X.ravel(), Y.ravel()))
        t = t.reshape(z.shape) / max(line.length, 1e-9)
        prof = prof * np.clip(np.minimum(t, 1 - t) / taper_frac, 0.0, 1.0)
    return z + height_m * prof


def step_terrace(z, meta, line, height_m, shoulder_m=6.0, side=1):
    """One-sided step: ground raised on `side` of the line, riser smoothed over
    +-shoulder_m.  Produces a single (not paired) shoulder."""
    X, Y = D.cell_centres(z.shape, meta)
    coords = np.asarray(line.coords)
    nx, ny = -(coords[-1, 1] - coords[0, 1]), (coords[-1, 0] - coords[0, 0])
    nl = np.hypot(nx, ny)
    nx, ny = nx / nl, ny / nl
    s = (X - coords[0, 0]) * nx + (Y - coords[0, 1]) * ny
    z = z + height_m * smoothstep(-shoulder_m, shoulder_m, s * side)
    return z


def ring_ditch(z, meta, centre, radius, width_m, depth_m, bank_h=0.4, bank_w=2.5):
    X, Y = D.cell_centres(z.shape, meta)
    r = np.hypot(X - centre[0], Y - centre[1])
    u = np.abs(r - radius) / (width_m * 0.5)
    ditch = -depth_m * np.where(u <= 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(u, 0, 1))), 0.0)
    b = np.abs(r - radius - width_m * 0.5 - bank_w * 0.5) / (bank_w * 0.5)
    bank = bank_h * np.where(b <= 1.0, 0.5 * (1 + np.cos(np.pi * np.clip(b, 0, 1))), 0.0)
    return z + ditch + bank


def sinuous(a, b, amp, wavelength, npts=120, phase=0.0):
    t = np.linspace(0, 1, npts)
    xs = a[0] + (b[0] - a[0]) * t
    ys = a[1] + (b[1] - a[1]) * t
    L = np.hypot(b[0] - a[0], b[1] - a[1])
    nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
    off = amp * np.sin(2 * np.pi * t * L / wavelength + phase)
    return LineString(np.stack([xs + nx * off, ys + ny * off], 1))


def add_survey_noise(z, seed, white=0.04, canopy_amp=0.30, canopy_sig=22.0,
                     stripe=0.04, slope=(0.0008, -0.009)):
    """Approximate the residual noise structure of a 3 m lidar DTM: white noise,
    broad canopy/vegetation bias, and faint scan-line striping."""
    rng = np.random.default_rng(seed + 991)
    n, m = z.shape
    Xi, Yi = np.meshgrid(np.arange(m) * DX, np.arange(n) * DX)
    z = z + slope[0] * Xi + slope[1] * Yi
    z = z + white * rng.normal(size=z.shape)
    if canopy_amp > 0:
        g = ndi.gaussian_filter(rng.normal(size=z.shape), canopy_sig / DX, mode="wrap")
        g /= max(np.std(g), 1e-12)
        z = z + canopy_amp * g
    if stripe > 0:
        z = z + stripe * np.sin(2 * np.pi * np.arange(n)[:, None] / 9.0)
    return z


def render(scen, pad=90.0, dx=DX):
    """Turn a scenario spec (name, geoms, ops, chm_spec, truth, note) into a dict."""
    name, geoms, ops, chm_spec, truth, note = scen
    xmin, xmax, ymin, ymax = _bbox(geoms, pad)
    meta, n, m = _meta_grid(xmin, xmax, ymin, ymax, dx)
    z = fractal_noise((n, m), scen_seed(name), dx=dx)
    # a regional trend that actually drains: without it the carved channel has no
    # down-valley gradient and a D8 flow net will ignore it (found by testing)
    z += fractal_noise((n, m), scen_seed(name) + 1,
                       amps=((300.0, 0.9), (900.0, 1.4)), dx=dx)
    for op in ops:
        z = op(z, meta)
    chm = None
    if chm_spec is not None:
        rng = np.random.default_rng(scen_seed(name) + 77)
        base = chm_spec["base"] + chm_spec.get("var", 4.0) * rng.normal(size=z.shape)
        chm = np.clip(base, 0, None).astype(np.float32)
        if "zero_frac" in chm_spec:                      # pasture fraction
            X, Y = D.cell_centres(z.shape, meta)
            g = ndi.gaussian_filter(rng.normal(size=z.shape), chm_spec.get("gap_sig_m", 90) / dx,
                                    mode="wrap")
            chm[g > np.quantile(g, 1 - chm_spec["zero_frac"])] = 0.0
    return dict(name=name, z=np.asarray(z, np.float32), meta=meta, chm=chm,
                truth=truth, note=note)


def scen_seed(name):
    return abs(hash(name)) % 9999 if False else sum(bytearray(name.encode())) + 7


# ------------------------------------------------------------------ scenarios
def _s_causeway():
    stream = sinuous((0, 0), (0, 1200), 28, 700)
    # y=700 is where the stream is transverse to the causeway (offset zero, steepest
    # lateral gradient); y=560 would sit at a meander apex, which is the hard case
    # kept separately in causeway_crossing_sinuous
    caus = LineString([(-170, 700), (170, 700)])
    h, w = 1.2, 7.0
    return ("causeway_crossing", [stream, caus],
            [lambda z, m: carve_channel(z, m, stream, 14.0, 1.6),
             lambda z, m: build_ridge(z, m, caus, w, h, dps_m=2.5, crown_flat=0.45),
             # culverted under the causeway: re-carve the stream away from it
             lambda z, m: carve_channel(z, m, stream.difference(caus.buffer(6.0)), 14.0, 1.6),
             lambda z, m: add_survey_noise(z, 11)],
            None,
            [dict(kind="causeway", geom=caus, width_m=w, height_m=h, must_find=True)],
            "straight 7 m x 1.2 m causeway crossing a sinuous stream, open pasture")


def _s_causeway_sinuous():
    stream = sinuous((0, 0), (0, 1200), 55, 420)
    caus = LineString([(-170, 560), (170, 560)])
    h, w = 1.2, 7.0
    return ("causeway_crossing_sinuous", [stream, caus],
            [lambda z, m: carve_channel(z, m, stream, 14.0, 1.6),
             lambda z, m: build_ridge(z, m, caus, w, h, dps_m=2.5, crown_flat=0.45),
             lambda z, m: carve_channel(z, m, stream.difference(caus.buffer(6.0)), 14.0, 1.6),
             lambda z, m: add_survey_noise(z, 23)],
            None,
            [dict(kind="causeway", geom=caus, width_m=w, height_m=h, must_find=False)],
            "same causeway where the stream crosses at a meander apex (nearly parallel "
            "to the causeway): a documented hard case, may be missed")


def _s_levee():
    stream = sinuous((0, 0), (0, 1200), 45, 380)
    levee = stream.parallel_offset(17.0, "right", join_style=2)
    return ("natural_levee", [stream, levee],
            [lambda z, m: carve_channel(z, m, stream, 16.0, 1.8),
             lambda z, m: build_ridge(z, m, levee, 14.0, 0.7, dps_m=4.0,
                                      crown_flat=0.1, taper_frac=0.22),
             # the channel stays open through the levee (younger/contemporaneous)
             lambda z, m: carve_channel(z, m, stream, 16.0, 1.8),
             lambda z, m: add_survey_noise(z, 12)],
            None,
            [dict(kind="levee", geom=levee, width_m=14.0, height_m=0.7, must_find=False)],
            "levee parallel to its channel, tapering: no crossing, not a causeway")


def _s_valley():
    line = LineString([(-600, 1150), (800, 150)])
    return ("straight_stream_valley", [line],
            [lambda z, m: carve_channel(z, m, line, 18.0, 2.2),
             lambda z, m: add_survey_noise(z, 13)],
            None, [dict(kind="valley", geom=line, width_m=18.0, height_m=2.2,
                        must_find=False)],
            "straight incised valley: an inverse ridge, must not fire")


def _s_terrace():
    line = LineString([(-600, 250), (800, 350)])
    return ("terrace_edge", [line],
            [lambda z, m: step_terrace(z, m, line, 1.1, shoulder_m=7.0, side=1),
             lambda z, m: add_survey_noise(z, 14)],
            None, [dict(kind="terrace", geom=line, width_m=np.nan, height_m=1.1,
                        must_find=False)],
            "one-sided 1.1 m riser: no paired shoulders, must not fire")


def _s_ring():
    c = Point(0, 600)
    r1 = LineString([(0, 400), (0, 0)])
    r2 = LineString([(0, 800), (0, 1200)])
    return ("ring_ditch_roads", [c, r1, r2],
            [lambda z, m: ring_ditch(z, m, (c.x, c.y), 20.0, 5.0, 1.2),
             lambda z, m: carve_channel(z, m, r1, 5.0, 0.8),
             lambda z, m: carve_channel(z, m, r2, 5.0, 0.8),
             lambda z, m: add_survey_noise(z, 15)],
            dict(base=0.0, zero_frac=0.85),
            [dict(kind="ring", geom=c, width_m=5.0, height_m=1.2, must_find=False)],
            "40 m ring ditch with straight sunken roads in cleared pasture "
            "(0414-like); no causeway may fire")


def _s_halo():
    stream = sinuous((0, 0), (0, 1200), 55, 420)
    caus = LineString([(-170, 560), (170, 560)])
    return ("empty_halo", [stream, caus, Point(60, 560)],
            [lambda z, m: carve_channel(z, m, stream, 14.0, 1.6),
             lambda z, m: build_ridge(z, m, caus, 7.0, 1.2, dps_m=2.5, crown_flat=0.45),
             lambda z, m: _punch_hole(z, m, (60, 560), 90, 70),
             lambda z, m: add_survey_noise(z, 16)],
            None, [dict(kind="causeway", geom=caus, width_m=7.0, height_m=1.2,
                        must_find=True,
                        # the planted ridge is REAL; what this scenario tests is that the
                        # 180x140 m data hole with its 0.8 m halo rim does not (a) make
                        # the truncated remnant disappear into a false negative, nor
                        # (b) let a detection bridge the hole or chase the halo rim.
                        truncated_by_hole=dict(centre=(60.0, 560.0), rx=90.0, ry=70.0,
                                               rim_m=0.8))],
            "EMBEDDED POSITIVE + no-span test: a real causeway cut by a 180x140 m data "
            "hole with a 0.8 m halo rim.  Scored on: the truncated remnant is still "
            "found, no detection spans the hole, and the halo rim is not chased.")


def _punch_hole(z, meta, centre, rx, ry, rim=0.8):
    X, Y = D.cell_centres(z.shape, meta)
    hole = ((X - centre[0]) / rx) ** 2 + ((Y - centre[1]) / ry) ** 2 <= 1.0
    rim_m = ndi.binary_dilation(hole, np.ones((3, 3), bool)) & ~hole
    z[rim_m] += rim
    z[hole] = np.nan
    return z


def _s_track():
    stream = sinuous((-200, 0), (900, 1200), 60, 500)
    track = LineString([(-600, 800), (800, 800)])
    left = track.parallel_offset(4.6, "left", join_style=2)
    right = track.parallel_offset(4.6, "right", join_style=2)
    return ("modern_track", [stream, track, left, right],
            [lambda z, m: carve_channel(z, m, stream, 10.0, 1.2),
             lambda z, m: build_ridge(z, m, track, 6.0, 0.6, dps_m=1.5, crown_flat=0.85),
             lambda z, m: carve_channel(z, m, left, 2.0, 0.35),
             lambda z, m: carve_channel(z, m, right, 2.0, 0.35),
             lambda z, m: add_survey_noise(z, 17)],
            dict(base=0.0, zero_frac=0.9),
            [dict(kind="modern_track", geom=track, width_m=6.0, height_m=0.6,
                  must_find=False)],
            "modern raised track with flat crown and both borrow ditches crossing a "
            "drainage: geometrically causeway-like, must be flagged modern/suspect")


def _s_forest():
    stream = sinuous((0, 0), (0, 1200), 35, 300)
    return ("flat_forest_noise", [stream],
            [lambda z, m: carve_channel(z, m, stream, 12.0, 1.5),
             lambda z, m: add_survey_noise(z, 18, white=0.05, canopy_amp=0.55,
                                           canopy_sig=25.0)],
            dict(base=19.0, var=3.5, zero_frac=0.10, gap_sig_m=70),
            [], "closed-canopy forest, no earthwork, canopy-noise residual only")


def _s_two():
    s1 = sinuous((-200, 0), (-200, 1500), 40, 500)
    s2 = sinuous((-500, 900), (1000, 700), 50, 600)
    c1 = LineString([(-600, 850), (800, 850)])
    c2 = LineString([(450, 560), (450, 940)])
    return ("two_causeways", [s1, s2, c1, c2],
            [lambda z, m: carve_channel(z, m, s1, 12.0, 1.4),
             lambda z, m: carve_channel(z, m, s2, 12.0, 1.4),
             lambda z, m: build_ridge(z, m, c1, 9.0, 1.6, dps_m=3.0, crown_flat=0.35),
             lambda z, m: build_ridge(z, m, c2, 5.0, 0.7, dps_m=2.0, crown_flat=0.3),
             lambda z, m: carve_channel(z, m, s1.difference(c1.buffer(6)), 12.0, 1.4),
             lambda z, m: add_survey_noise(z, 19)],
            None,
            [dict(kind="causeway", geom=c1, width_m=9.0, height_m=1.6, must_find=True),
             dict(kind="causeway", geom=c2, width_m=5.0, height_m=0.7, must_find=True)],
            "one large (9 m x 1.6 m) and one small (5 m x 0.7 m) causeway")


def _s_canopy_causeway():
    stream = sinuous((0, 0), (0, 1200), 50, 420)
    caus = LineString([(-170, 600), (170, 600)])
    return ("causeway_under_canopy", [stream, caus],
            [lambda z, m: carve_channel(z, m, stream, 14.0, 1.5),
             lambda z, m: build_ridge(z, m, caus, 7.0, 1.0, dps_m=2.5, crown_flat=0.4),
             lambda z, m: carve_channel(z, m, stream.difference(caus.buffer(6.0)), 14.0, 1.5),
             lambda z, m: add_survey_noise(z, 21, white=0.05, canopy_amp=0.55,
                                           canopy_sig=25.0)],
            dict(base=16.0, var=4.0, zero_frac=0.25, gap_sig_m=60),
            [dict(kind="causeway", geom=caus, width_m=7.0, height_m=1.0, must_find=True)],
            "1.0 m causeway under partly closed canopy: does canopy residual hide it?")


def _s_natural_ridge():
    """NEGATIVE (hard): a natural elongated ridge, curved and tapering, that happens
    to cross a small drainage -- the single most dangerous natural false positive."""
    stream = sinuous((150, 0), (150, 1200), 30, 600)
    pts = [(150 + 18 * math.sin(t / 60.0), 200 + t) for t in np.linspace(0, 700, 60)]
    ridge = LineString(pts)
    return ("natural_ridge", [stream, ridge],
            [lambda z, m: carve_channel(z, m, stream, 8.0, 1.0),
             lambda z, m: build_ridge(z, m, ridge, 15.0, 1.1, dps_m=6.0, crown_flat=0.05,
                                      taper_frac=0.30),
             lambda z, m: carve_channel(z, m, stream.difference(ridge.buffer(4.0)), 8.0, 1.0),
             lambda z, m: add_survey_noise(z, 25, white=0.06, canopy_amp=0.45)],
            dict(base=6.0, var=3.0, zero_frac=0.4),
            [dict(kind="natural_ridge", geom=ridge, width_m=15.0, height_m=1.1,
                  must_find=False)],
            "curved, tapering natural ridge crossing a drainage: morphology alone is "
            "ambiguous with a causeway")


def _s_pond():
    """NEGATIVE: cattle pond with a raised spoil rim (a closed curved ridge)."""
    c = Point(0, 600)
    return ("cattle_pond", [c],
            [lambda z, m: _pond(z, m, (c.x, c.y), 22.0, 16.0, 1.2),
             lambda z, m: add_survey_noise(z, 26)],
            dict(base=0.0, zero_frac=0.9),
            [dict(kind="pond", geom=c, width_m=22.0, height_m=1.2, must_find=False)],
            "cattle pond with 1.2 m spoil rim: closed curved ridge, must not fire")


def _pond(z, meta, centre, rx, ry, depth, rim=0.5):
    X, Y = D.cell_centres(z.shape, meta)
    r = np.hypot((X - centre[0]) / rx, (Y - centre[1]) / ry)
    z = z - depth * np.clip(1 - r, 0, 1) ** 0.6
    z = z + rim * np.clip(1 - np.abs(r - 1.15) / 0.25, 0, 1)
    return z


def _s_braided():
    """NEGATIVE (hard): floodplain with four sinuous streams and no earthwork.  The
    paired banks of a straight stream reach are the documented look-alike for a pair
    of causeway-like embankments."""
    s1 = sinuous((-350, 0), (-350, 1300), 45, 420)
    s2 = sinuous((-80, 0), (-80, 1300), 35, 330)
    s3 = sinuous((260, 0), (260, 1300), 55, 500)
    s4 = sinuous((520, 0), (520, 1300), 25, 280)
    return ("braided_drainage", [s1, s2, s3, s4],
            [lambda z, m: carve_channel(z, m, s1, 16.0, 1.8),
             lambda z, m: carve_channel(z, m, s2, 10.0, 1.2),
             lambda z, m: carve_channel(z, m, s3, 14.0, 2.0),
             lambda z, m: carve_channel(z, m, s4, 8.0, 1.0),
             lambda z, m: build_ridge(z, m, s1.parallel_offset(14.0, "left", join_style=2),
                                      12.0, 0.5, dps_m=3.0, crown_flat=0.1, taper_frac=0.15),
             lambda z, m: build_ridge(z, m, s3.parallel_offset(13.0, "right", join_style=2),
                                      12.0, 0.6, dps_m=3.0, crown_flat=0.1, taper_frac=0.15),
             lambda z, m: add_survey_noise(z, 27, white=0.05, canopy_amp=0.45)],
            dict(base=4.0, var=3.0, zero_frac=0.5),
            [], "floodplain with four streams and natural levees, no earthwork: the "
                "paired-bank / leveee look-alike case")


BUILDERS = dict(causeway_crossing=_s_causeway,
                causeway_crossing_sinuous=_s_causeway_sinuous, natural_levee=_s_levee,
                straight_stream_valley=_s_valley, terrace_edge=_s_terrace,
                ring_ditch_roads=_s_ring, empty_halo=_s_halo,
                modern_track=_s_track, flat_forest_noise=_s_forest,
                two_causeways=_s_two, causeway_under_canopy=_s_canopy_causeway,
                natural_ridge=_s_natural_ridge, cattle_pond=_s_pond,
                braided_drainage=_s_braided)


def make(name, dx=DX):
    return render(BUILDERS[name](), dx=dx)


ALL = list(BUILDERS)


def validate(d, verbose=True):
    """Check that the scenario really contains what its truth claims."""
    z, meta, truth = d["z"], d["meta"], d["truth"]
    w = np.isfinite(z)
    msgs = []
    ok = True
    b = dict(xmin=meta["x0"], xmax=meta["x0"] + z.shape[1] * meta["dx"],
             ymin=meta["y0"] - z.shape[0] * meta["dy"], ymax=meta["y0"])
    for t in truth:
        g = t["geom"]
        if not (b["xmin"] <= g.bounds[0] and g.bounds[2] <= b["xmax"] and
                b["ymin"] <= g.bounds[1] and g.bounds[3] <= b["ymax"]):
            ok = False
            msgs.append(f"  !! {t['kind']} extends outside the grid bbox")
    for t in truth:
        if t["kind"] in ("causeway", "modern_track", "levee"):
            # crest height above flanking ground on a cross-section at the middle
            p = t["geom"].interpolate(0.5, normalized=True)
            ln = t["geom"]
            coords = np.asarray(ln.coords)
            mid = coords[len(coords) // 2]
            ang = np.arctan2(coords[-1, 1] - coords[0, 1], coords[-1, 0] - coords[0, 0])
            nx, ny = -np.sin(ang), np.cos(ang)
            s = np.arange(-25, 25.01, 1.0)
            xs = mid[0] + nx * s
            ys = mid[1] + ny * s
            prof = D.bilinear(z, xs, ys, meta)
            crest = np.nanmax(prof[np.abs(s) <= 3])
            flank = np.nanmedian(np.r_[prof[np.abs(s - (-18)) < 2], prof[np.abs(s - 18) < 2]])
            got = crest - flank
            msgs.append(f"  {t['kind']}: crest-flank at mid = {got:.2f} m "
                        f"(declared {t['height_m']})")
            if not (0.5 * t["height_m"] <= got <= 2.0 * t["height_m"] + 0.5):
                ok = False
                msgs.append("  !! crest height not reproduced")
    hinv = 1.0 - w.mean()
    msgs.append(f"  invalid cells: {hinv * 100:.2f} %")
    if verbose:
        print(f"[{d['name']}] grid {z.shape} bbox {b['xmin']:.0f}..{b['xmax']:.0f} / "
              f"{b['ymin']:.0f}..{b['ymax']:.0f}  {'OK' if ok else 'FAIL'}")
        for m_ in msgs:
            print(m_)
    return ok


if __name__ == "__main__":
    bad = []
    for name in ALL:
        d = make(name)
        if not validate(d):
            bad.append(name)
    print("scenarios failing self-check:", bad or "none")
