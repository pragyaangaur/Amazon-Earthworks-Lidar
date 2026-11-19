"""Line / embankment / causeway / sunken-road falsification.

The ring harness measures a closed band around a centre.  A causeway, embanked
road, sunken road, natural levee, terrace scarp or stream valley is instead a
*strip* whose cross-profile is roughly constant along its length.  This module
measures that strip; `rules.judge_line` turns the measurements into a verdict.

Everything is sampled in a frame (s along the line, t across it), so width,
height, flatness, symmetry, continuity and straightness are direct measurements.

Input is two endpoints.  A detector that reports centre + axis angle + length
converts to endpoints itself.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .grid import Grid
from . import primitives as P

DEFAULTS = dict(
    n_along=180,
    half_width_min_m=18.0,
    min_length_m=40.0,
    min_amp_m=0.30,
    max_amp_m=5.0,
    min_width_m=2.0,
    max_width_m=25.0,
    min_continuity=0.55,
    max_asym=0.55,
    max_width_cv=0.45,
    max_straight_dev_frac=0.06,
    canopy_gap_frac=0.35,
    flank_canopy_min_m=8.0,
    flat_bottom_frac=0.30,
    bank_min_m=0.15,
)


def _run_width(t, prof, k, half_amp, positive):
    """Contiguous run around index k where the anomaly exceeds half amplitude."""
    n = prof.size
    above = (prof > half_amp) if positive else (prof < -half_amp)
    a = k
    while a - 1 >= 0 and above[a - 1]:
        a -= 1
    b = k
    while b + 1 < n and above[b + 1]:
        b += 1
    dt = t[1] - t[0] if n > 1 else 1.0
    return float(t[b] - t[a] + dt), bool(a == 0 or b == n - 1)


def line_bundle(grid: Grid, x1, y1, x2, y2, width_hint=None, chm=None,
                fid="", transect="", kind="raised", params=None):
    """Measure a linear feature.  Returns a JSON-safe bundle (no verdict)."""
    p = dict(DEFAULTS)
    if params:
        p.update(params)
    L = math.hypot(x2 - x1, y2 - y1)
    out = dict(id=fid, transect=transect, type=kind,
               endpoints=dict(x1=float(x1), y1=float(y1),
                              x2=float(x2), y2=float(y2)),
               length_m=float(L), width_hint_m=width_hint,
               grid=grid.origin_dict(), geometry={}, notes=[])
    if L < p["min_length_m"]:
        out["geometry"] = dict(status="too_short", length_m=float(L))
        return out
    if not (bool(np.all(grid.contains(x1, y1, 0.0)) and
                np.all(grid.contains(x2, y2, 0.0)))):
        out["geometry"] = dict(status="outside_grid", length_m=float(L))
        return out

    half_w = max(p["half_width_min_m"], 2.5 * (width_hint or 8.0))
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    px, py = -uy, ux
    ds = max(grid.res / 2.0, L / p["n_along"])
    nt = int(max(11, round(2 * half_w / (grid.res / 2.0))))
    s = np.arange(-0.08 * L, L * 1.08 + 1e-9, ds)
    t = np.linspace(-half_w, half_w, nt)
    S, T = np.meshgrid(s, t, indexing="ij")
    Z = grid.sample(x1 + S * ux + T * px, y1 + S * uy + T * py)
    if np.isfinite(Z).mean() < 0.25:
        out["geometry"] = dict(status="no_data", length_m=float(L))
        return out

    # Background = the across-band median at each station, smoothed along the line.
    # Deliberately NOT a plane: a plane fitted to the outer band absorbs a
    # one-sided step (it tilts to follow it) and a terrace scarp then looks like a
    # two-shouldered ridge.  A per-station across-median cannot do that: half the
    # band is high and half is low, so the median sits between them and the
    # profile stays monotone across.
    bg = np.nanmedian(Z, axis=1)
    if np.isfinite(bg).sum() < 5:
        out["geometry"] = dict(status="no_data", length_m=float(L))
        return out
    idx = np.arange(bg.size)
    good_bg = np.isfinite(bg)
    bg = np.interp(idx, idx[good_bg], bg[good_bg])
    bg = ndimage.uniform_filter1d(bg, size=int(max(3, round(15.0 / ds)) | 1),
                                  mode="nearest")
    R = Z - bg[:, None]

    w_hint = float(width_hint or max(4.0, 0.3 * half_w))
    corridor = np.abs(T) <= 0.5 * w_hint
    edge = (np.abs(T) >= 0.72 * half_w) & (np.abs(T) <= 0.92 * half_w)
    corridor1 = np.abs(t) <= 0.5 * w_hint
    edge1 = (np.abs(t) >= 0.72 * half_w) & (np.abs(t) <= 0.92 * half_w)
    crest = float(np.nanmedian(R[corridor]))
    flank = float(np.nanmedian(R[edge]))
    amp = crest - flank                               # >0 raised, <0 sunken
    positive = amp >= 0
    prof_t = np.nanmedian(R, axis=0)
    if not np.isfinite(prof_t).any():
        out["geometry"] = dict(status="no_data", length_m=float(L))
        return out
    k = int(np.nanargmax(prof_t if positive else -prof_t))
    width, w_clipped = _run_width(t, prof_t, k, 0.5 * abs(amp), positive)
    core = (prof_t > 0.8 * abs(amp)) if positive else (prof_t < -0.8 * abs(amp))
    flat_frac = float(np.sum(core) * (t[1] - t[0]) / max(width, 1e-6))

    left = float(np.nanmedian(R[(T < -0.3 * half_w)]))
    right = float(np.nanmedian(R[(T > 0.3 * half_w)]))
    asym = float(abs(left - right) / max(abs(amp), 1e-6))

    per_s = np.nanmedian(R[:, corridor1], axis=1) - np.nanmedian(R[:, edge1], axis=1)
    ok_s = np.isfinite(per_s)
    cont = float(np.mean(np.abs(per_s[ok_s]) >= 0.5 * abs(amp))) if ok_s.any() else float("nan")
    amp_iqr = float(np.nanpercentile(per_s[ok_s], 75) - np.nanpercentile(per_s[ok_s], 25)) if ok_s.any() else float("nan")

    # width and crest position along the line
    widths, crest_t = [], []
    inner_idx = np.nonzero(np.abs(t) <= 0.5 * half_w)[0]
    Rs = np.empty_like(R)
    for i in range(R.shape[0]):
        lo, hi = max(0, i - 2), min(R.shape[0], i + 3)
        Rs[i] = np.nanmedian(R[lo:hi], axis=0)
    for i in range(Rs.shape[0]):
        row = Rs[i]
        if not np.isfinite(row).any():
            continue
        j = inner_idx[int(np.argmax(row[inner_idx] if positive else -row[inner_idx]))]
        crest_t.append(float(t[j]))
        wj, _ = _run_width(t, row, j, 0.5 * abs(amp), positive)
        widths.append(wj)
    widths = np.asarray(widths, float)
    crest_t = np.asarray(crest_t, float)
    width_cv = float(np.nanstd(widths) / max(np.nanmedian(widths), 1e-6)) if widths.size > 5 else float("nan")
    straight_dev = float("nan")
    if crest_t.size > 8:
        ss = np.linspace(0, L, crest_t.size)
        b1, b0 = np.polyfit(ss, crest_t, 1)
        straight_dev = float(np.nanmax(np.abs(crest_t - (b0 + b1 * ss))))

    # flanking ditches: a bank pair is what separates a raised road from a scarp
    banks = 0
    for side in (-1, 1):
        sel = (t > 0.8 * width / 2 + 1.0) & (t < 0.8 * half_w) if side > 0 else \
              (t < -0.8 * width / 2 - 1.0) & (t > -0.8 * half_w)
        if sel.any():
            dip = -float(np.nanmin(prof_t[sel])) if positive else \
                  float(np.nanmax(prof_t[sel]))
            if dip >= p["bank_min_m"]:
                banks += 1

    canopy = dict(corridor_m=None, flank_m=None, ratio=None, modern_corridor=False)
    if chm is not None:
        C = chm.sample(x1 + S * ux + T * px, y1 + S * uy + T * py)
        c_in = float(np.nanmedian(C[corridor])) if np.isfinite(C[corridor]).any() else float("nan")
        c_fl = float(np.nanmedian(C[edge])) if np.isfinite(C[edge]).any() else float("nan")
        ratio = (c_in / c_fl) if (np.isfinite(c_in) and np.isfinite(c_fl) and c_fl > 0.5) else float("nan")
        canopy = dict(corridor_m=c_in, flank_m=c_fl, ratio=ratio,
                      modern_corridor=bool(np.isfinite(ratio) and c_fl >= p["flank_canopy_min_m"]
                                           and ratio < p["canopy_gap_frac"]))

    relief_extreme = float(prof_t[k] - np.nanmin(prof_t)) if positive else \
        float(prof_t[k] - np.nanmax(prof_t))
    geom = dict(status="ok", length_m=float(L), amp_m=float(amp),
                relief_extreme_m=abs(relief_extreme),
                raised=bool(positive), width_m=float(width), width_clipped=w_clipped,
                flat_bottom_frac=flat_frac, asym=asym, continuity=cont,
                amp_iqr_m=amp_iqr, width_cv=width_cv,
                straight_dev_m=straight_dev,
                straight_dev_frac=(straight_dev / L if np.isfinite(straight_dev) else float("nan")),
                banks=int(banks), canopy=canopy,
                background="per-station across-median, smoothed along the line",
                n_stations=int(R.shape[0]), half_width_m=float(half_w),
                cross_profile=dict(t=[float(v) for v in t],
                                   r=[float(v) if np.isfinite(v) else None for v in prof_t]),
                per_station_amp=[float(v) if np.isfinite(v) else None for v in per_s])
    out["geometry"] = _clean(geom)
    return out


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, np.ndarray):
        return _clean(d.tolist())
    if isinstance(d, (np.bool_, bool)):
        return bool(d)
    if isinstance(d, (np.floating, float)):
        return None if not np.isfinite(float(d)) else round(float(d), 4)
    if isinstance(d, (np.integer, int)):
        return int(d)
    return d
