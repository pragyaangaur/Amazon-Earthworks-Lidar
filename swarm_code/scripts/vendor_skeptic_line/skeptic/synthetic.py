"""Synthetic control scenes for the m6 skeptic harness.

Every scene is a 3 m DTM (optionally with a CHM), painted in map coordinates, and
carries a list of `truth` entries describing what a detector would plausibly
report and what the correct verdict is:

    truth = [dict(id=..., cx=..., cy=..., expect="possible"|"reject",
                  expect_rule="...", note=...), ...]

The scenes are deliberately *not* trivially separable: positives get regional
slope, roughness, micro-noise, an off-centre ditch, missing returns and attached
sunken roads, and negatives get real dip-like structures (channels, ponds, pads).

Usage:
    python3 -m skeptic.synthetic --out out/synthetic      # write npz + truth json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from scipy import ndimage

from .grid import Grid


# ---------------------------------------------------------------------------
# terrain helpers
# ---------------------------------------------------------------------------

def smooth_noise(shape, res, sigma_m, amp_m, seed):
    rs = np.random.default_rng(seed)
    w = rs.standard_normal(shape)
    f = ndimage.gaussian_filter(w, sigma_m / res, mode="nearest")
    f = f / (f.std() + 1e-12)
    return amp_m * f


def base_terrain(nx, ny, res, slope=(0.012, -0.006), rough_m=0.6, micro_m=0.05,
                 seed=0, x0=500000.0, y0=9000000.0):
    X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * res,
                       y0 - (np.arange(ny) + 0.5) * res)
    z = slope[0] * (X - x0) + slope[1] * (Y - y0)
    z = z + smooth_noise((ny, nx), res, 90.0, rough_m, seed + 1)
    z = z + smooth_noise((ny, nx), res, 25.0, 0.35 * rough_m, seed + 2)
    z = z + np.random.default_rng(seed + 3).standard_normal((ny, nx)) * micro_m
    return X, Y, z


def _bell(d, half_width, power=2.0):
    """Smooth 1 at d=0, 0 at d=half_width."""
    t = np.clip(1.0 - (d / half_width) ** 2, 0.0, 1.0)
    return t ** (power * 0.5)


def paint_ditch(z, X, Y, cx, cy, size_m, width_m, depth_m, bank_m=0.0,
                shape="circle", p=2.0, phi_deg=0.0, gap_deg=None,
                flat_bottom=0.35, bank_offset_m=2.0):
    """Subtract a ditch of constant width along a closed superellipse."""
    r = np.hypot(X - cx, Y - cy)
    th = np.arctan2(Y - cy, X - cx)
    a = size_m / 2.0
    phi = math.radians(phi_deg)
    if p <= 2.01:
        Reff = np.full_like(r, a)
    else:
        Reff = a * ((np.abs(np.cos(th - phi)) ** p +
                     np.abs(np.sin(th - phi)) ** p) ** (-1.0 / p))
    d = np.abs(r - Reff)
    hw = width_m / 2.0
    prof = -depth_m * np.clip(1.0 - ((d / hw) ** 2), 0.0, 1.0) ** 0.7
    # flat bottom: no extra depth within the inner flat part
    if flat_bottom > 0:
        inner = d < hw * flat_bottom
        prof = np.where(inner, -depth_m, prof)
    mask = d <= hw
    if gap_deg is not None:
        # gap centred on +y (90 deg); remove the ditch inside it
        g = np.abs(((np.degrees(th) - 90.0 + 180.0) % 360.0) - 180.0)
        mask &= g > gap_deg / 2.0
    z = z + np.where(mask, prof, 0.0)
    if bank_m > 0:
        db = np.abs(d - (hw + bank_offset_m))
        z = z + np.where(db < 2.5, bank_m * _bell(db, 2.5, 2.0), 0.0)
    return z


def paint_line(z, X, Y, x1, y1, x2, y2, width_m, depth_m, bank_m=0.0,
               flat_bottom=0.5, taper_ends=False):
    """Subtract a straight or curved (piecewise) ditch along a segment."""
    px, py = X - x1, Y - y1
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    t = np.clip((px * dx + py * dy) / (L * L), 0.0, 1.0)
    d = np.hypot(px - t * dx, py - t * dy)
    hw = width_m / 2.0
    prof = -depth_m * np.clip(1.0 - ((d / hw) ** 2), 0.0, 1.0) ** 0.7
    if flat_bottom > 0:
        prof = np.where(d < hw * flat_bottom, -depth_m, prof)
    mask = d <= hw
    z = z + np.where(mask, prof, 0.0)
    if bank_m > 0:
        db = np.abs(d - (hw + 2.0))
        z = z + np.where(db < 2.5, bank_m * _bell(db, 2.5, 2.0), 0.0)
    return z


def paint_channel(z, X, Y, cy, amp, wavelength, width_m, depth_m,
                  bank_m=0.0, x_from=None, x_to=None, phase=0.0):
    """Meandering channel along y = cy + amp*sin(2*pi*x/wavelength + phase)."""
    k = 2 * math.pi / wavelength
    yc = cy + amp * np.sin(k * X + phase)
    dy = amp * k * np.cos(k * X + phase)
    d = np.abs(Y - yc) / np.sqrt(1.0 + dy ** 2)
    hw = width_m / 2.0
    prof = -depth_m * np.clip(1.0 - ((d / hw) ** 2), 0.0, 1.0) ** 0.6
    z = z + np.where(d <= hw, prof, 0.0)
    if bank_m > 0:
        db = np.abs(d - (hw + 3.0))
        z = z + np.where(db < 3.0, bank_m * _bell(db, 3.0, 2.0), 0.0)
    return z


def paint_arc_channel(z, X, Y, cx, cy, R, th_from_deg, th_to_deg, width_m,
                      depth_m, flat_bottom=0.5):
    """A C-shaped (open) channel arc: an abandoned meander loop, not a closed ring.

    Distance is |r - R| inside the angular span and the distance to the nearer
    arc end outside it, so the channel has two open mouths.
    """
    r = np.hypot(X - cx, Y - cy)
    th = np.degrees(np.arctan2(Y - cy, X - cx)) % 360.0
    lo, hi = (th_from_deg % 360.0), (th_to_deg % 360.0)
    inside = ((th >= lo) & (th <= hi)) if lo <= hi else ((th >= lo) | (th <= hi))
    d = np.abs(r - R)
    x1, y1 = cx + R * math.cos(math.radians(lo)), cy + R * math.sin(math.radians(lo))
    x2, y2 = cx + R * math.cos(math.radians(hi)), cy + R * math.sin(math.radians(hi))
    d_end = np.minimum(np.hypot(X - x1, Y - y1), np.hypot(X - x2, Y - y2))
    d = np.where(inside, d, d_end)
    hw = width_m / 2.0
    prof = -depth_m * np.clip(1.0 - ((d / hw) ** 2), 0.0, 1.0) ** 0.6
    if flat_bottom > 0:
        prof = np.where(d < hw * flat_bottom, -depth_m, prof)
    return z + np.where(d <= hw, prof, 0.0)


def paint_valley(z, X, Y, cx, width_m, depth_m, ang_deg=70.0):
    """Broad straight V-shaped valley with no banks (natural drainage)."""
    a = math.radians(ang_deg)
    nxv, nyv = -math.sin(a), math.cos(a)
    d = np.abs((X - cx) * nxv + (Y - Y.mean()) * nyv)
    hw = width_m / 2.0
    prof = -depth_m * np.clip(1.0 - d / hw, 0.0, 1.0) ** 1.5
    return z + prof


def paint_pond(z, X, Y, cx, cy, R, depth_m, rim_m=0.4):
    r = np.hypot(X - cx, Y - cy)
    inside = r <= R
    prof = np.where(inside, -depth_m, 0.0)
    rim = np.abs(r - R) < 2.0
    z = z + prof + np.where(rim, rim_m, 0.0)
    return z


def paint_pad(z, X, Y, cx, cy, w, h, dh, ang_deg=0.0):
    a = math.radians(ang_deg)
    xr = (X - cx) * math.cos(a) + (Y - cy) * math.sin(a)
    yr = -(X - cx) * math.sin(a) + (Y - cy) * math.cos(a)
    inside = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    return z + np.where(inside, dh, 0.0)


def add_gullies(z, X, Y, cx, cy, hill_amp=9.0, hill_sigma=70.0, n=9, seed=5):
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    z = z + hill_amp * np.exp(-r2 / (2 * hill_sigma ** 2))
    th0 = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    r = np.sqrt(r2)
    th = np.arctan2(Y - cy, X - cx)
    for t in th0:
        dd = np.abs(((th - t + np.pi) % (2 * np.pi)) - np.pi)
        along = np.clip((r - 15.0) / (hill_sigma * 1.6), 0.0, 1.0)
        z = z - 1.8 * np.exp(-(dd ** 2) / (2 * math.radians(4.0) ** 2)) * along
    return z


def mask_disc(grid: Grid, cx, cy, R):
    X, Y = _mapcoords(grid)
    grid.z[np.hypot(X - cx, Y - cy) <= R] = np.nan


def _mapcoords(grid):
    X, Y = np.meshgrid(grid.x0 + (np.arange(grid.nx) + 0.5) * grid.res,
                       grid.y0 - (np.arange(grid.ny) + 0.5) * grid.res)
    return X, Y


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------

def _scene(name, nx, ny, res, seed, slope, rough, micro, chm=None):
    X, Y, z = base_terrain(nx, ny, res, slope=slope, rough_m=rough, micro_m=micro, seed=seed)
    g = Grid(z, res, 500000.0, 9000000.0, epsg=5880, name=name)
    xm, ym = g.x0 + nx * res / 2, g.y0 - ny * res / 2
    return X, Y, g, xm, ym


def scene_ring_40_pasture():
    """Positive control analogue of transect 0414: ~40 m ring ditch + straight
    sunken roads in cleared pasture, off-centre, on a slope, with noise."""
    X, Y, g, xm, ym = _scene("pos_ring_40_pasture", 300, 300, 3.0, 11,
                             (0.010, -0.008), 0.35, 0.035)
    cx, cy = xm - 23.0, ym + 17.0
    g.z = paint_ditch(g.z, X, Y, cx, cy, 41.0, 6.0, 1.05, bank_m=0.30,
                      flat_bottom=0.35)
    # two straight sunken roads attached to the ring, plus banks
    g.z = paint_line(g.z, X, Y, cx - 160, cy, cx + 160, cy, 7.0, 0.55, bank_m=0.2)
    g.z = paint_line(g.z, X, Y, cx, cy - 160, cx, cy + 160, 6.0, 0.50, bank_m=0.2)
    truth = [dict(id="P1", cx=cx, cy=cy, expect="possible", expect_rule=None,
                  note="40 m ring ditch + straight sunken roads (0414 analogue)")]
    return dict(name="pos_ring_40_pasture", grid=g, chm=None, truth=truth)


def scene_ring_120_open():
    """Larger 120 m ring ditch, square-ish rounded (p=4), thinner ditch."""
    X, Y, g, xm, ym = _scene("pos_ring_120_open", 340, 340, 3.0, 21,
                             (-0.006, 0.009), 0.5, 0.04)
    cx, cy = xm + 8.0, ym - 5.0
    g.z = paint_ditch(g.z, X, Y, cx, cy, 124.0, 9.0, 1.6, bank_m=0.25, p=2.6)
    truth = [dict(id="P2", cx=cx + 6.0, cy=cy - 4.0, expect="possible", expect_rule=None,
                  note="120 m ring ditch, centre deliberately offset 7 m")]
    return dict(name="pos_ring_120_open", grid=g, chm=None, truth=truth)


def scene_square_150():
    """Square geoglyph 150 m across, 10 m ditch, 2 m deep, plus an attached road."""
    X, Y, g, xm, ym = _scene("pos_square_150", 340, 340, 3.0, 31,
                             (0.008, 0.004), 0.45, 0.04)
    cx, cy = xm, ym
    g.z = paint_ditch(g.z, X, Y, cx, cy, 150.0, 10.0, 1.8, bank_m=0.3, p=18.0,
                      phi_deg=15.0, flat_bottom=0.5)
    g.z = paint_line(g.z, X, Y, cx + 75 * math.cos(math.radians(15)),
                     cy + 75 * math.sin(math.radians(15)),
                     cx + 260 * math.cos(math.radians(15)),
                     cy + 260 * math.sin(math.radians(15)), 9.0, 0.8, bank_m=0.25)
    truth = [dict(id="P3", cx=cx, cy=cy, expect="possible", expect_rule=None,
                  note="150 m square geoglyph, rotated 15 deg, with a road")]
    return dict(name="pos_square_150", grid=g, chm=None, truth=truth)


def scene_u_shape():
    """U-shaped enclosure: a *rectilinear* three-sided ditch, open to the north,
    with a sunken road running out of the open side (the real U-shaped type,
    not a circular arc)."""
    X, Y, g, xm, ym = _scene("pos_u_shape", 320, 320, 3.0, 41,
                             (0.004, 0.010), 0.4, 0.04)
    cx, cy = xm, ym
    g.z = paint_ditch(g.z, X, Y, cx, cy, 130.0, 8.0, 1.4, bank_m=0.25,
                      p=14.0, phi_deg=0.0, gap_deg=95.0)
    g.z = paint_line(g.z, X, Y, cx, cy + 60.0, cx, cy + 240.0, 7.0, 0.6, bank_m=0.2)
    truth = [dict(id="P4", cx=cx, cy=cy, expect="possible", expect_rule=None,
                  note="rectilinear U-shaped ditch open to the north + sunken road")]
    return dict(name="pos_u_shape", grid=g, chm=None, truth=truth)


def scene_ring_forest():
    """80 m ring under 20 m canopy: sparse-return noise (0.25 m) + CHM."""
    X, Y, g, xm, ym = _scene("pos_ring_80_forest", 320, 320, 3.0, 51,
                             (0.011, 0.003), 0.7, 0.05)
    cx, cy = xm + 4.0, ym - 9.0
    g.z = paint_ditch(g.z, X, Y, cx, cy, 84.0, 8.0, 1.5, bank_m=0.2)
    g.z = g.z + smooth_noise(g.shape, g.res, 5.0, 0.22, 515)
    chm = Grid(np.full(g.shape, 20.0) + smooth_noise(g.shape, g.res, 30.0, 4.0, 52),
               g.res, g.x0, g.y0, epsg=g.epsg, name="pos_ring_80_forest/chm")
    truth = [dict(id="P5", cx=cx, cy=cy, expect="possible", expect_rule=None,
                  note="80 m ring under 20 m canopy; interpolation noise 0.22 m")]
    return dict(name="pos_ring_80_forest", grid=g, chm=chm, truth=truth)


def scene_meander():
    """Meandering river with oxbow and scroll bars: the classic ring false positive."""
    X, Y, g, xm, ym = _scene("neg_meander", 360, 360, 3.0, 61,
                             (0.002, -0.004), 0.8, 0.05)
    cy = ym + 40.0
    g.z = paint_channel(g.z, X, Y, cy, 95.0, 420.0, 34.0, 3.2, bank_m=0.6)
    # scroll bars: faint parallel ridges flanking the channel
    for i, off in enumerate((40.0, 55.0, 70.0)):
        g.z = g.z + 0.28 * np.exp(-((np.abs(Y - cy - 95.0 * np.sin(2 * math.pi * X / 420.0)) - off) ** 2) / 18.0) * (1 - i * 0.25)
    # an abandoned oxbow: C-shaped loop open to the south-east (210 deg of arc)
    ox, oy, R = xm + 120.0, ym - 140.0, 58.0
    g.z = paint_arc_channel(g.z, X, Y, ox, oy, R, -30.0, 180.0, 20.0, 2.4)
    truth = [
        dict(id="N1a", cx=xm, cy=cy, expect="reject",
             expect_rule=("not_closed", "irregular", "natural_channel", "weak_dip"),
             note="centre of a meander bend: dip radius not constant"),
        dict(id="N1b", cx=ox, cy=oy, expect="reject",
             expect_rule=("not_closed", "irregular", "natural_channel"),
             note="abandoned C-shaped oxbow loop (210 deg of arc), 20 m wide"),
    ]
    return dict(name="neg_meander", grid=g, chm=None, truth=truth)


def scene_pond():
    """Cattle pond: circular closed depression, water surface partly without returns."""
    X, Y, g, xm, ym = _scene("neg_pond", 260, 260, 3.0, 71,
                             (0.006, -0.003), 0.3, 0.04)
    cx, cy = xm, ym
    g.z = paint_pond(g.z, X, Y, cx, cy, 32.0, 1.7, rim_m=0.45)
    mask_disc(g, cx, cy, 22.0)
    # a second, dry pond that keeps its ground returns (dry-season cattle tank)
    cx2, cy2 = xm + 95.0, ym - 80.0
    g.z = paint_pond(g.z, X, Y, cx2, cy2, 26.0, 1.2, rim_m=0.40)
    truth = [dict(id="N2", cx=cx, cy=cy, expect="reject",
                  expect_rule=("pond", "hole", "data_void"),
                  note="cattle pond with a water hole: interior 1.7 m below trend"),
             dict(id="N2b", cx=cx2, cy=cy2, expect="reject",
                  expect_rule=("pond", "not_closed"),
                  note="dry cattle tank: closed depression with ground returns")]
    return dict(name="neg_pond", grid=g, chm=None, truth=truth)


def scene_valley():
    """Straight broad V-shaped stream valley, 60 m wide, no banks."""
    X, Y, g, xm, ym = _scene("neg_valley", 300, 300, 3.0, 81,
                             (0.003, 0.002), 0.5, 0.04)
    g.z = paint_valley(g.z, X, Y, xm - 60.0, 62.0, 2.6, ang_deg=68.0)
    truth = [dict(id="N3", cx=xm - 60.0, cy=ym, expect="reject",
                  expect_rule=("not_closed", "weak_dip", "irregular"), note="straight V valley, no ditch, no closure")]
    return dict(name="neg_valley", grid=g, chm=None, truth=truth)


def scene_building():
    """Modern farm building: flat rectangular pad + roof with no returns."""
    X, Y, g, xm, ym = _scene("neg_building", 260, 260, 3.0, 91,
                             (0.004, 0.005), 0.25, 0.03)
    cx, cy = xm, ym
    g.z = paint_pad(g.z, X, Y, cx, cy, 46.0, 24.0, 0.18, ang_deg=20.0)
    g.z = paint_pad(g.z, X, Y, cx + 30.0, cy - 22.0, 28.0, 18.0, 0.16, ang_deg=20.0)
    mask_disc(g, cx, cy, 20.0)
    truth = [dict(id="N4", cx=cx, cy=cy, expect="reject", expect_rule=("not_closed", "one_sided_step", "pad_flat", "irregular", "hole"),
                  note="building pad + roof hole: one-sided edges, flat interior")]
    return dict(name="neg_building", grid=g, chm=None, truth=truth)


def scene_hole_halo():
    """Empty-data halo in open terrain: pond edge / dense canopy hole with no
    topographic ring at all."""
    X, Y, g, xm, ym = _scene("neg_hole_halo", 260, 260, 3.0, 101,
                             (0.005, 0.004), 0.4, 0.03)
    cx, cy = xm, ym
    mask_disc(g, cx, cy, 26.0)
    truth = [dict(id="N5", cx=cx, cy=cy, expect="reject", expect_rule=("hole", "data_void", "not_closed"),
                  note="26 m data hole in flat ground; any ring here is an artefact")]
    return dict(name="neg_hole_halo", grid=g, chm=None, truth=truth)


def scene_hilltop_gullies():
    """Natural hill cut by gullies."""
    X, Y, g, xm, ym = _scene("neg_hilltop", 300, 300, 3.0, 111,
                             (0.0, 0.0), 0.4, 0.04)
    cx, cy = xm, ym
    g.z = add_gullies(g.z, X, Y, cx, cy, hill_amp=9.0, hill_sigma=65.0, n=11)
    truth = [dict(id="N6", cx=cx, cy=cy + 12.0, expect="reject",
                  expect_rule=("not_closed", "weak_dip", "irregular"), note="hilltop cut by radial gullies")]
    return dict(name="neg_hilltop", grid=g, chm=None, truth=truth)


def scene_modern_road_forest():
    """Modern logging road under forest: straight canopy gap + compacted flat track."""
    X, Y, g, xm, ym = _scene("neg_modern_road", 300, 300, 3.0, 121,
                             (0.007, -0.004), 0.6, 0.05)
    cx, cy = xm, ym
    g.z = paint_line(g.z, X, Y, cx - 250, cy + 90, cx + 250, cy - 110, 7.0, -0.25)
    chm = Grid(np.full(g.shape, 18.0) + smooth_noise(g.shape, g.res, 30.0, 4.0, 122),
               g.res, g.x0, g.y0, epsg=g.epsg, name="neg_modern_road/chm")
    # canopy gap along the road
    for t in np.linspace(0, 1, 200):
        x = cx - 250 + t * 500
        y = cy + 90 - t * 200
        d = np.hypot(X - x, Y - y)
        chm.z = np.where(d < 8.0, 1.5, chm.z)
    truth = [dict(id="N7", cx=cx + 50.0, cy=cy - 20.0, expect="reject",
                  expect_rule=("not_closed", "weak_dip", "one_sided_step", "irregular", "hole"), note="modern road: canopy gap corridor")]
    return dict(name="neg_modern_road", grid=g, chm=chm, truth=truth)


def scene_arc_channel():
    """A lone circular arc channel in a floodplain: the oxbow / meander-bend
    false positive that looks most like a U-shaped enclosure."""
    X, Y, g, xm, ym = _scene("neg_arc_channel", 320, 320, 3.0, 141,
                             (0.003, -0.002), 0.6, 0.05)
    cx, cy = xm, ym
    g.z = paint_arc_channel(g.z, X, Y, cx, cy, 58.0, -30.0, 180.0, 20.0, 2.4)
    truth = [dict(id="N9", cx=cx, cy=cy, expect="reject",
                  expect_rule=("not_closed", "irregular"),
                  note="abandoned oxbow arc, 20 m wide, open over ~150 deg")]
    return dict(name="neg_arc_channel", grid=g, chm=None, truth=truth)


def scene_pad_with_ditch():
    """Modern house pad with a perimeter drainage ditch: a *closed*, two-sided
    ring ditch around a flat raised interior.  Only the interior-level and pad
    checks can reject this one."""
    X, Y, g, xm, ym = _scene("neg_pad_ditch", 280, 280, 3.0, 151,
                             (0.005, 0.002), 0.3, 0.03)
    cx, cy = xm, ym
    # raised flat pad, 54 x 38 m, 0.35 m up
    g.z = paint_pad(g.z, X, Y, cx, cy, 54.0, 38.0, 0.35, ang_deg=10.0)
    # perimeter drainage ditch just outside the pad edge
    g.z = paint_ditch(g.z, X, Y, cx, cy, 72.0, 4.0, 0.55, bank_m=0.0,
                      p=14.0, phi_deg=10.0, flat_bottom=0.6)
    truth = [dict(id="N10", cx=cx, cy=cy, expect="reject",
                  expect_rule=("pad_flat", "not_closed", "irregular", "one_sided_step"),
                  note="house pad + perimeter drainage ditch: closed ring around a "
                       "flat raised interior")]
    return dict(name="neg_pad_ditch", grid=g, chm=None, truth=truth)


def scene_neck_cutoff():
    """A nearly cut-off meander neck: 300 deg of circular channel, so the low is
    almost closed and almost perfectly circular.

    This scene is here to *document a limitation*, not to be passed: with a
    300 deg closed arc the planform, closure and cross-section checks all look
    like a ring ditch, and only the drainage context (which is a map, not a
    profile) separates them.  The control report counts it separately.
    """
    X, Y, g, xm, ym = _scene("neg_neck_cutoff", 340, 340, 3.0, 161,
                             (0.004, -0.003), 0.7, 0.05)
    cx, cy = xm, ym
    g.z = paint_arc_channel(g.z, X, Y, cx, cy, 52.0, 20.0, 320.0, 17.0, 2.2)
    truth = [dict(id="N11", cx=cx, cy=cy, expect="reject", known_limitation=True,
                  expect_rule=("not_closed", "irregular", "natural_channel", "weak_dip"),
                  note="near-closed meander neck (300 deg arc): profile alone cannot "
                       "separate this from a ring ditch")]
    return dict(name="neg_neck_cutoff", grid=g, chm=None, truth=truth)


def scene_terrace_edge():
    """Terrace / floodplain scarp: a long curved one-sided step, no ditch."""
    X, Y, g, xm, ym = _scene("neg_scarp", 300, 300, 3.0, 131,
                             (0.002, 0.003), 0.4, 0.04)
    cx, cy = xm, ym
    # a curved scarp: y_scarp = cy + 0.0004*(X-cx)^2
    ys = cy + 0.0009 * (X - cx) ** 2 - 20.0
    step = 1.4 / (1.0 + np.exp(-(Y - ys) / 3.0))
    g.z = g.z - step
    truth = [dict(id="N8", cx=cx, cy=cy + 0.0009 * 0.0 - 20.0, expect="reject",
                  expect_rule=("not_closed", "one_sided_step", "pad_flat", "irregular"),
                  note="curved one-sided terrace scarp (1.4 m step, no far bank)")]
    return dict(name="neg_scarp", grid=g, chm=None, truth=truth)


# ---------------------------------------------------------------------------
# linear features (causeway / road / levee / scarp / valley / modern track)
# ---------------------------------------------------------------------------

def paint_ridge(z, X, Y, x1, y1, x2, y2, width_m, height_m, ditch_m=0.0,
                one_sided=False, wobble=0.0, wave_m=180.0, seed=0):
    """A raised embankment along a segment, optional flanking ditches.

    `one_sided` lowers the ground on one side of the ridge (levee on a river
    bank); `wobble` makes the ridge wander sinusoidally along its length.
    """
    px, py = X - x1, Y - y1
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    along = px * ux + py * uy
    across = -px * uy + py * ux
    if wobble:
        across = across - wobble * np.sin(2 * math.pi * along / wave_m + seed)
    d = np.abs(across)
    inside = (along >= 0) & (along <= L)
    hw = width_m / 2.0
    prof = height_m * np.clip(1.0 - (d / hw) ** 2, 0.0, 1.0) ** 0.5
    z = z + np.where(inside, prof, 0.0)
    if ditch_m > 0:
        for sgn in (-1, 1):
            dd = np.abs(across - sgn * (hw + 2.5))
            z = z - np.where(inside, ditch_m * np.clip(1.0 - (dd / 2.5) ** 2, 0.0, 1.0), 0.0)
    if one_sided:
        drop = 1.2 * np.clip((across - hw) / 8.0, 0.0, 1.0)
        z = z - np.where(inside, drop, 0.0)
    return z


def scene_causeway():
    """Raised causeway with flanking ditches, crossing a rough floodplain."""
    X, Y, g, xm, ym = _scene("pos_causeway", 360, 360, 3.0, 171,
                             (0.004, -0.003), 0.7, 0.05)
    x1, y1 = xm - 200.0, ym - 120.0
    x2, y2 = xm + 200.0, ym + 130.0
    g.z = paint_ridge(g.z, X, Y, x1, y1, x2, y2, 7.0, 0.95, ditch_m=0.35)
    truth = [dict(id="L1", x1=x1, y1=y1, x2=x2, y2=y2, expect="possible",
                  expect_rule=None, kind="raised",
                  note="0.95 m raised causeway, 7 m wide, with flanking ditches")]
    return dict(name="pos_causeway", grid=g, chm=None, truth=truth)


def scene_levee():
    """Natural levee: one-sided, pinching and swelling, wandering along its length."""
    X, Y, g, xm, ym = _scene("neg_levee", 360, 360, 3.0, 181,
                             (0.003, 0.002), 0.8, 0.05)
    x1, y1 = xm - 210.0, ym - 60.0
    x2, y2 = xm + 210.0, ym + 20.0
    g.z = paint_ridge(g.z, X, Y, x1, y1, x2, y2, 13.0, 0.7, one_sided=True,
                      wobble=9.0, wave_m=150.0, seed=3)
    # the river side: a wide shallow channel next to the levee
    g.z = paint_channel(g.z, X, Y, ym - 90.0, 25.0, 700.0, 40.0, 1.4)
    truth = [dict(id="L2", x1=x1, y1=y1, x2=x2, y2=y2, expect="reject",
                  expect_rule=("one_sided_step", "irregular", "natural_channel"),
                  kind="raised",
                  note="natural levee: one-sided, width varies, crest wanders")]
    return dict(name="neg_levee", grid=g, chm=None, truth=truth)


def scene_scarp_line():
    """Floodplain terrace scarp: a long one-sided step."""
    X, Y, g, xm, ym = _scene("neg_terrace_line", 320, 320, 3.0, 191,
                             (0.002, 0.003), 0.4, 0.04)
    x1, y1 = xm - 180.0, ym + 60.0
    x2, y2 = xm + 180.0, ym + 60.0
    across = (Y - y1)
    g.z = g.z - 1.5 / (1.0 + np.exp(-across / 2.5))
    truth = [dict(id="L3", x1=x1, y1=y1, x2=x2, y2=y2, expect="reject",
                  expect_rule=("one_sided_step", "irregular"), kind="raised",
                  note="1.5 m terrace scarp: one-sided step, no far shoulder")]
    return dict(name="neg_terrace_line", grid=g, chm=None, truth=truth)


def scene_valley_line():
    """Broad V-shaped stream valley, no flat floor, no banks."""
    X, Y, g, xm, ym = _scene("neg_valley_line", 320, 320, 3.0, 201,
                             (0.003, 0.001), 0.6, 0.05)
    x1, y1 = xm - 150.0, ym - 40.0
    x2, y2 = xm + 150.0, ym + 40.0
    g.z = paint_valley(g.z, X, Y, xm, 55.0, 2.8, ang_deg=75.0)
    truth = [dict(id="L4", x1=x1, y1=y1, x2=x2, y2=y2, expect="reject",
                  expect_rule=("natural_channel", "irregular", "weak_dip"),
                  kind="sunken",
                  note="V-shaped valley: no flat floor and no flanking banks")]
    return dict(name="neg_valley_line", grid=g, chm=None, truth=truth)


def scene_track_forest():
    """Modern track under 18 m canopy: flat, straight, and a canopy gap."""
    X, Y, g, xm, ym = _scene("neg_track_forest", 320, 320, 3.0, 211,
                             (0.005, -0.002), 0.6, 0.05)
    x1, y1 = xm - 190.0, ym + 70.0
    x2, y2 = xm + 190.0, ym - 90.0
    g.z = paint_line(g.z, X, Y, x1, y1, x2, y2, 6.0, 0.22, flat_bottom=0.8)
    chm = Grid(np.full(g.shape, 18.0) + smooth_noise(g.shape, g.res, 30.0, 3.5, 212),
               g.res, g.x0, g.y0, epsg=g.epsg, name="neg_track_forest/chm")
    ux, uy = (x2 - x1) / math.hypot(x2 - x1, y2 - y1), (y2 - y1) / math.hypot(x2 - x1, y2 - y1)
    px, py = -uy, ux
    S = np.linspace(0, 1, 200)
    for tt in S:
        xx = x1 + tt * (x2 - x1)
        yy = y1 + tt * (y2 - y1)
        dd = np.abs((X - xx) * px + (Y - yy) * py)
        chm.z = np.where(dd < 4.5, 1.2, chm.z)
    truth = [dict(id="L5", x1=x1, y1=y1, x2=x2, y2=y2, expect="reject",
                  expect_rule=("modern_linear", "weak_dip", "irregular"),
                  kind="sunken",
                  note="modern track in a canopy gap under 18 m forest")]
    return dict(name="neg_track_forest", grid=g, chm=chm, truth=truth)


def all_scenes():
    return [scene_ring_40_pasture(), scene_ring_120_open(), scene_square_150(),
            scene_u_shape(), scene_ring_forest(),
            scene_meander(), scene_pond(), scene_valley(), scene_building(),
            scene_hole_halo(), scene_hilltop_gullies(), scene_modern_road_forest(),
            scene_terrace_edge(), scene_arc_channel(), scene_pad_with_ditch(),
            scene_neck_cutoff()]


def all_line_scenes():
    return [scene_causeway(), scene_levee(), scene_scarp_line(),
            scene_valley_line(), scene_track_forest()]


def save_scenes(outdir):
    os.makedirs(outdir, exist_ok=True)
    index = []
    for sc in all_scenes():
        g = sc["grid"]
        d = dict(z=g.z, res=g.res, x0=g.x0, y0=g.y0,
                 y_ascending=g.y_ascending, epsg=g.epsg, name=sc["name"])
        if sc["chm"] is not None:
            d["chm"] = sc["chm"].z
        p = os.path.join(outdir, sc["name"] + ".npz")
        np.savez_compressed(p, **d)
        meta = dict(name=sc["name"], path=p, res=g.res, x0=g.x0, y0=g.y0,
                    epsg=g.epsg, shape=list(g.shape), truth=sc["truth"])
        with open(os.path.join(outdir, sc["name"] + ".truth.json"), "w") as f:
            json.dump(meta, f, indent=1)
        index.append(meta)
    with open(os.path.join(outdir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/synthetic")
    a = ap.parse_args()
    idx = save_scenes(a.out)
    for m in idx:
        print(f"{m['name']:24s} {m['shape']} features={len(m['truth'])} -> {m['path']}")


if __name__ == "__main__":
    main()
