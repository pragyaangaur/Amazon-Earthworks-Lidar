"""Scene definitions for the m1 synthetic benchmark.

Six scenes, each with an explicit, JSON-serialisable ground truth:

  s1_ring40_open   published-0414 analogue: 40 m ring ditch + sunken roads in
                   cleared pasture, sparse canopy.  MUST be recovered clearly.
  s2_natneg        natural false-positive material only (meander, oxbow,
                   scroll bars, broad V valley, cattle pond with dam, hilltop
                   cut by gullies, graded modern road, clearing).
  s3_mound_village ring of 9 mounds, causeway, roads, one small ring ditch.
  s4_forest_stress ring ditch *under closed canopy* + swath gaps: the honest
                   hard case; expect a weak/possible signature at best.
  s5_holehalo      one empty circular data hole (halo risk) next to a real
                   ring ditch: the halo must be masked, the ring must survive.
  s6_transect_0414sim  12.5 km x 300 m transect at the real 2 pulses/m^2:
                   ingest stress test with targets and one natural stream.

Targets are the features with role "target"; everything else exists to make
detectors earn their keep.  `expect` is what the *benchmark* asserts, not
what the real EBA data will do.
"""
from __future__ import annotations

import math

EPSG = 32719          # WGS 84 / UTM 19S, the Acre transects' zone


def _scene(name, x0, y0, w, h, **kw):
    s = dict(name=name, x0=float(x0), y0=float(y0),
             x1=float(x0 + w), y1=float(y0 + h), epsg=EPSG,
             grid_m=3.0,
             scan=dict(along_m=0.5, across_m=1.0, jitter_along_m=0.18,
                       jitter_across_m=0.10),
             returns=dict(sigma_z_m=0.06, occlusion_alpha=0.18,
                          p_second_return=0.45, p_understorey=0.15,
                          p_outlier_high=2e-4, p_outlier_low=2e-4),
             roughness=dict(field_amp_m=0.18, pit_density=3.5e-4,
                            pit_min_m=0.35, pit_max_m=1.6, pit_sigma=5.0,
                            scales=[[6.0, 0.55], [15.0, 0.32], [35.0, 0.18]],
                            note=("micro-relief calibrated to the measured "
                                  "open-pasture ring-dip null of the real 0414 "
                                  "product; see notes/validation.md")),
             canopy=[], holes=[], features=[],
             base=dict(z0=180.0, sx=0.004, sy=-0.002, blobs=[], waves=[]))
    s.update(kw)
    return s


# --------------------------------------------------------------- scene 1 ----
def s1_ring40_open():
    X, Y = 400000.0, 8900000.0
    s = _scene("s1_ring40_open", X, Y, 700, 700,
               base=dict(z0=182.0, sx=0.0045, sy=-0.0028,
                         blobs=[[X + 180, Y + 320, 1.6, 190.0]], waves=[]),
               canopy=[dict(bbox=[X, Y, X + 700, Y + 700], density=0.0012,
                            hmin=5.0, hmax=11.0, rc=[2.0, 3.5]),
                       dict(bbox=[X + 520, Y + 520, X + 700, Y + 700],
                            density=0.030, hmin=14.0, hmax=26.0, rc=[3.0, 5.0])],
               holes=[dict(type="band", axis="y", center=Y + 300.0, halfwidth=6.0,
                           note="missing flight strip, 12 m, crosses the ring"),
                      dict(type="circle", cx=X + 105.0, cy=Y + 615.0, r=11.0,
                           note="small no-return patch"),
                      dict(type="rect", bbox=[X + 640, Y, X + 700, Y + 55],
                           note="tile corner with no data")],
               features=[])
    s["features"] = [
        dict(id="S1-F1", type="ring_ditch", role="target", shape="circle",
             cx=X + 300.0, cy=Y + 350.0, r=20.0, width=6.0, depth=1.2,
             bank=0.35, bank_w=4.0, arc=[0.0, 360.0], eps=0.0,
             wob=[0.8, 2.0, 0.7],
             label="40 m ring ditch, 6 m wide, 1.2 m deep, low bank",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S1-F2", type="road", role="target", shape="line",
             p0=[X + 170.0, Y + 322.0], p1=[X + 470.0, Y + 372.0],
             width=5.0, depth=0.9, bank=0.25, taper_m=15.0,
             label="straight sunken road, 5 m wide, 0.9 m deep",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S1-F3", type="road", role="target", shape="line",
             p0=[X + 300.0, Y + 380.0], p1=[X + 305.0, Y + 540.0],
             width=6.0, depth=0.7, bank=0.20, taper_m=20.0,
             label="second sunken road leaving the ring to the north",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S1-F4", type="square_ditch", role="target", shape="square",
             cx=X + 560.0, cy=Y + 150.0, r=30.0, width=5.0, depth=0.8,
             bank=0.22, bank_w=3.0, rot=12.0,
             label="60 m square ditch (rotated 12 deg)",
             expect="detect", expect_certainty_in_sim="strong-like"),
    ]
    return s


# --------------------------------------------------------------- scene 2 ----
def s2_natneg():
    X, Y = 410000.0, 8910000.0
    s = _scene("s2_natneg", X, Y, 1000, 1000,
               roughness=dict(field_amp_m=0.32, pit_density=8e-4,
                              pit_min_m=0.4, pit_max_m=2.0, pit_sigma=6.0,
                              scales=[[6.0, 0.55], [15.0, 0.32], [35.0, 0.18]]),
               base=dict(z0=205.0, sx=0.0030, sy=0.0015,
                         blobs=[[X + 780, Y + 800, 4.0, 210.0],
                                [X + 120, Y + 120, -1.5, 260.0]],
                         waves=[[0.35, 620.0, 25.0, 1.1]]),
               canopy=[dict(bbox=[X, Y, X + 1000, Y + 1000], density=0.0010,
                            hmin=5.0, hmax=12.0, rc=[2.0, 3.5]),
                       dict(bbox=[X, Y, X + 260, Y + 320], density=0.030,
                            hmin=15.0, hmax=28.0, rc=[3.0, 5.5]),
                       dict(bbox=[X + 700, Y + 700, X + 1000, Y + 1000],
                            density=0.022, hmin=12.0, hmax=24.0, rc=[3.0, 5.0])],
               holes=[dict(type="band", axis="x", center=X + 640.0, halfwidth=5.0,
                           note="missing strip")],
               features=[])
    s["features"] = [
        dict(id="S2-N1", type="meander", role="negative", shape="meander",
             x0=X, yc=Y + 505.0, A1=95.0, L1=880.0, ph1=0.6, A2=22.0,
             L2=300.0, ph2=2.1, width=46.0, depth=5.5, levee=0.7,
             levee_w=14.0, length=1000.0, Lw=380.0, wvar=0.25, phw=0.4,
             label="meandering river, 46 m channel with levees",
             expect="not_detect"),
        dict(id="S2-N2", type="oxbow", role="negative", shape="arc",
             cx=X + 262.0, cy=Y + 782.0, r=86.0, width=38.0, depth=4.2,
             arc=[15.0, 305.0], taper=18.0, levee=0.5, levee_w=12.0,
             label="oxbow lake remnant: 290 deg arc of channel",
             expect="not_detect"),
        dict(id="S2-N3", type="scroll", role="negative", shape="texture",
             x0=X, yc=Y + 430.0, length=1000.0, amp=0.35, lam=32.0,
             sigma=170.0, tilt=0.02, ph=0.0,
             label="scroll-bar ridge and swale texture",
             expect="not_detect"),
        dict(id="S2-N4", type="vvalley", role="negative", shape="line",
             p0=[X + 120.0, Y], p1=[X + 305.0, Y + 1000.0],
             halfwidth=48.0, depth=2.6,
             label="broad V stream valley, 96 m wide, no banks",
             expect="not_detect"),
        dict(id="S2-N5", type="pond", role="negative", shape="circle",
             cx=X + 700.0, cy=Y + 205.0, r=22.0, depth=1.7,
             dam_h=1.1, dam_w=7.0, dam_arc=[250.0, 340.0],
             label="cattle pond with earth dam (half-ring bank)",
             expect="not_detect"),
        dict(id="S2-N6", type="hill", role="negative", shape="dome",
             cx=X + 780.0, cy=Y + 800.0, r=150.0, h=6.0,
             label="natural hilltop",
             expect="not_detect"),
        dict(id="S2-N7", type="gully", role="negative", shape="line",
             p0=[X + 780.0, Y + 800.0], p1=[X + 690.0, Y + 1000.0],
             width=10.0, depth=0.8, label="gully cutting the hilltop flank",
             expect="not_detect"),
        dict(id="S2-N8", type="gully", role="negative", shape="line",
             p0=[X + 780.0, Y + 800.0], p1=[X + 920.0, Y + 950.0],
             width=13.0, depth=0.6, label="second gully",
             expect="not_detect"),
        dict(id="S2-N9", type="modern_road", role="modern", shape="line",
             p0=[X, Y + 150.0], p1=[X + 1000.0, Y + 285.0],
             width=7.0, cut=0.40, berm=0.3,
             label="graded modern road crossing the scene",
             expect="not_detect"),
        dict(id="S2-N10", type="pad", role="modern", shape="rect",
             cx=X + 600.0, cy=Y + 650.0, w=44.0, h=26.0, dz=-0.6,
             label="modern clearing / house pad", expect="not_detect"),
    ]
    return s


# --------------------------------------------------------------- scene 3 ----
def s3_mound_village():
    X, Y = 420000.0, 8920000.0
    s = _scene("s3_mound_village", X, Y, 800, 800,
               roughness=dict(field_amp_m=0.12, pit_density=2.0e-4,
                              pit_min_m=0.3, pit_max_m=1.2, pit_sigma=5.0,
                              scales=[[6.0, 0.55], [15.0, 0.32], [35.0, 0.18]]),
               base=dict(z0=196.0, sx=0.0020, sy=0.0025,
                         blobs=[[X + 400, Y + 400, 1.2, 260.0]], waves=[]),
               canopy=[dict(bbox=[X, Y, X + 800, Y + 800], density=0.0040,
                            hmin=6.0, hmax=16.0, rc=[2.0, 4.0]),
                       dict(bbox=[X, Y, X + 210, Y + 800], density=0.028,
                            hmin=14.0, hmax=27.0, rc=[3.0, 5.0]),
                       dict(bbox=[X + 300, Y + 300, X + 520, Y + 520],
                            density=0.0008, hmin=5.0, hmax=10.0, rc=[2.0, 3.0])],
               holes=[dict(type="band", axis="y", center=Y + 620.0, halfwidth=7.0,
                           note="missing strip")],
               features=[])
    s["features"] = [
        dict(id="S3-F1", type="mound_ring", role="target", shape="mound_ring",
             cx=X + 400.0, cy=Y + 400.0, R=90.0, k=9, phase=0.30,
             heights=[2.0, 1.4, 1.1, 1.8, 0.9, 1.5, 1.2, 2.2, 1.0],
             radii=[12.0, 10.0, 9.0, 11.0, 8.0, 10.0, 9.0, 12.0, 8.0],
             pos_wobble=0.05,
             label="ring of 9 mounds (0.9-2.2 m) around a 180 m plaza",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S3-F2", type="causeway", role="target", shape="line",
             p0=[X + 400.0, Y + 490.0], p1=[X + 400.0, Y + 690.0],
             width=10.0, height=0.8, taper_m=20.0,
             label="embanked causeway leaving the plaza to the north",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S3-F3", type="road", role="target", shape="line",
             p0=[X + 120.0, Y + 330.0], p1=[X + 690.0, Y + 375.0],
             width=6.0, depth=0.6, bank=0.2, taper_m=25.0,
             label="sunken road, 6 m wide, 0.6 m deep",
             expect="detect", expect_certainty_in_sim="possible"),
        dict(id="S3-F4", type="ring_ditch", role="target", shape="circle",
             cx=X + 650.0, cy=Y + 170.0, r=15.0, width=4.0, depth=0.7,
             bank=0.2, bank_w=2.5, arc=[0.0, 360.0], eps=0.03,
             label="small 30 m ring ditch",
             expect="detect", expect_certainty_in_sim="possible"),
        dict(id="S3-N1", type="vvalley", role="negative", shape="line",
             p0=[X + 40.0, Y], p1=[X + 250.0, Y + 800.0], halfwidth=40.0,
             depth=2.2, label="natural drainage line at the west edge",
             expect="not_detect"),
    ]
    return s


# --------------------------------------------------------------- scene 4 ----
def s4_forest_stress():
    X, Y = 430000.0, 8930000.0
    s = _scene("s4_forest_stress", X, Y, 900, 900,
               base=dict(z0=210.0, sx=-0.0035, sy=0.0020,
                         blobs=[[X + 500, Y + 500, 2.0, 240.0]], waves=[]),
               returns=dict(occlusion_alpha=0.24, p_second_return=0.55),
               canopy=[dict(bbox=[X, Y, X + 900, Y + 900], density=0.030,
                            hmin=12.0, hmax=30.0, rc=[3.0, 5.5]),
                       dict(bbox=[X + 620, Y + 620, X + 900, Y + 900],
                            density=0.0010, hmin=5.0, hmax=11.0,
                            rc=[2.0, 3.5]),
                       dict(bbox=[X + 660, Y + 120, X + 860, Y + 330],
                            density=0.0008, hmin=5.0, hmax=11.0,
                            rc=[2.0, 3.5])],
               holes=[dict(type="band", axis="y", center=Y + 455.0, halfwidth=12.0,
                           note="24 m missing strip through the ring"),
                      dict(type="band", axis="x", center=X + 430.0, halfwidth=9.0,
                           note="18 m missing strip, perpendicular"),
                      dict(type="circle", cx=X + 250.0, cy=Y + 700.0, r=26.0,
                           note="no-return patch: pond or clearing")],
               features=[])
    s["features"] = [
        dict(id="S4-F1", type="ring_ditch", role="target", shape="circle",
             cx=X + 450.0, cy=Y + 450.0, r=60.0, width=8.0, depth=1.5,
             bank=0.4, bank_w=5.0, arc=[0.0, 360.0], eps=0.0,
             wob=[1.0, 3.0, 1.2],
             label="120 m ring ditch under closed canopy, crossed by two swath gaps",
             expect="judge", expect_certainty_in_sim="possible-at-best"),
        dict(id="S4-F2", type="road", role="target", shape="line",
             p0=[X + 60.0, Y + 610.0], p1=[X + 840.0, Y + 640.0],
             width=7.0, depth=0.9, bank=0.25, taper_m=30.0,
             label="sunken road under canopy",
             expect="judge", expect_certainty_in_sim="possible-at-best"),
        dict(id="S4-N1", type="vvalley", role="negative", shape="line",
             p0=[X + 800.0, Y], p1=[X + 120.0, Y + 900.0], halfwidth=45.0,
             depth=2.4, label="natural drainage crossing the forest",
             expect="not_detect"),
    ]
    return s


# --------------------------------------------------------------- scene 5 ----
def s5_holehalo():
    X, Y = 440000.0, 8940000.0
    s = _scene("s5_holehalo", X, Y, 500, 500,
               base=dict(z0=175.0, sx=0.0015, sy=0.0010, blobs=[], waves=[]),
               canopy=[dict(bbox=[X, Y, X + 500, Y + 500], density=0.0010,
                            hmin=5.0, hmax=10.0, rc=[2.0, 3.0])],
               holes=[dict(type="circle", cx=X + 150.0, cy=Y + 250.0, r=20.0,
                           note="EMPTY HOLE r=20 m: interpolation halo must not "
                                "become a ring candidate"),
                      dict(type="rect", bbox=[X + 260, Y + 350, X + 320, Y + 370],
                           note="small rectangular hole")],
               features=[])
    s["features"] = [
        dict(id="S5-F1", type="ring_ditch", role="target", shape="circle",
             cx=X + 370.0, cy=Y + 150.0, r=25.0, width=5.0, depth=1.0,
             bank=0.25, bank_w=3.5, arc=[0.0, 360.0], eps=0.0,
             label="50 m ring ditch, 5 m wide, 1.0 m deep",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S5-N1", type="hole_halo", role="artefact", shape="circle",
             cx=X + 150.0, cy=Y + 250.0, r=20.0,
             label="empty-data halo control: relief must be masked within 15 m",
             expect="not_detect"),
    ]
    return s


# --------------------------------------------------------------- scene 6 ----
def s6_transect_0414sim():
    X, Y = 450000.0, 8950000.0
    L, W = 12500.0, 300.0
    s = _scene("s6_transect_0414sim", X, Y, L, W,
               base=dict(z0=190.0, sx=0.0008, sy=0.0030,
                         blobs=[[X + 3000, Y + 150, 2.5, 900.0],
                                [X + 9000, Y + 150, -2.0, 1100.0]],
                         waves=[[0.6, 2400.0, 5.0, 0.3]]),
               canopy=[dict(bbox=[X, Y, X + L, Y + W], density=0.0006,
                            hmin=5.0, hmax=12.0, rc=[2.0, 3.5]),
                       dict(bbox=[X, Y, X + 2100, Y + W], density=0.030,
                            hmin=14.0, hmax=28.0, rc=[3.0, 5.5]),
                       dict(bbox=[X + 11400, Y, X + L, Y + W], density=0.030,
                            hmin=14.0, hmax=28.0, rc=[3.0, 5.5])],
               holes=[dict(type="band", axis="x", center=X + 5200.0,
                           halfwidth=13.0, note="26 m missing strip"),
                      dict(type="band", axis="x", center=X + 9400.0,
                           halfwidth=11.0, note="22 m missing strip"),
                      dict(type="band", axis="y", center=Y + 210.0,
                           halfwidth=2.0, note="4 m swath seam along the corridor")],
               features=[])
    s["features"] = [
        dict(id="S6-F1", type="ring_ditch", role="target", shape="circle",
             cx=X + 2200.0, cy=Y + 150.0, r=20.0, width=6.0, depth=1.2,
             bank=0.35, bank_w=4.0, arc=[0.0, 360.0], wob=[0.7, 2.0, 2.0],
             label="transect-0414 analogue: 40 m ring in cleared pasture",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S6-F2", type="square_ditch", role="target", shape="square",
             cx=X + 4800.0, cy=Y + 140.0, r=45.0, width=7.0, depth=1.0,
             bank=0.25, bank_w=4.0, rot=18.0,
             label="90 m square ditch",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S6-F3", type="ring_ditch", role="target", shape="circle",
             cx=X + 7500.0, cy=Y + 160.0, r=100.0, width=10.0, depth=1.5,
             bank=0.4, bank_w=6.0, arc=[0.0, 360.0], eps=0.04,
             label="200 m ring ditch (upper size limit)",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S6-F4", type="mound_ring", role="target", shape="mound_ring",
             cx=X + 10300.0, cy=Y + 150.0, R=70.0, k=8, phase=0.15,
             heights=[1.6, 1.1, 2.0, 1.3, 0.9, 1.7, 1.2, 1.4],
             radii=[11.0, 9.0, 12.0, 10.0, 8.0, 11.0, 9.0, 10.0],
             pos_wobble=0.06,
             label="ring of 8 mounds around a 140 m plaza",
             expect="detect", expect_certainty_in_sim="possible"),
        dict(id="S6-F5", type="road", role="target", shape="line",
             p0=[X + 1500.0, Y + 60.0], p1=[X + 4400.0, Y + 70.0],
             width=6.0, depth=0.8, bank=0.2, taper_m=40.0,
             label="sunken road running along the corridor, 2.9 km",
             expect="detect", expect_certainty_in_sim="strong-like"),
        dict(id="S6-N1", type="meander", role="negative", shape="meander",
             x0=X, yc=Y + 255.0, A1=26.0, L1=420.0, ph1=0.9, A2=7.0,
             L2=150.0, ph2=1.7, width=12.0, depth=1.8, levee=0.35,
             levee_w=8.0, length=L, Lw=260.0, wvar=0.3, phw=1.2,
             label="small meandering stream along the corridor",
             expect="not_detect"),
        dict(id="S6-N2", type="vvalley", role="negative", shape="line",
             p0=[X + 8000.0, Y], p1=[X + 8300.0, Y + W], halfwidth=35.0,
             depth=2.0, label="broad V valley crossing the corridor",
             expect="not_detect"),
    ]
    return s


SCENES = {f.__name__.replace("_", "-"): f for f in [
    s1_ring40_open, s2_natneg, s3_mound_village, s4_forest_stress,
    s5_holehalo, s6_transect_0414sim]}


def get(name):
    key = name if name in SCENES else name.replace("_", "-")
    return SCENES[key]()
