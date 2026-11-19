"""preprocess.py -- DTM hygiene and context rasters used by all detectors.

Design rules taken from the project brief (run 1 lessons):
  * 3 m cells; NaN/absent = no data.
  * fill only *small* holes (<= 5 cells), never big ones -- big ones are real
    data gaps (ponds, buildings, dense canopy) and must stay masked;
  * blank any derived relief within 15 m of a hole, because hole halos produce
    perfect fake rings;
  * keep a canopy/roughness raster so a detector's evidence can be judged
    (under-canopy DTM noise fakes 0.3 m dips).

Everything is NaN-aware: sums are done as normalised convolutions so a hole
never drags a mean toward zero.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _norm_conv(x, w, sigma_cells):
    """Normalised (NaN-aware) Gaussian convolution. w = 0/1 valid weights."""
    num = ndi.gaussian_filter(np.where(w, x, 0.0).astype(np.float64), sigma_cells, mode="nearest")
    den = ndi.gaussian_filter(w.astype(np.float64), sigma_cells, mode="nearest")
    out = np.full(x.shape, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    return out


def fill_small_holes(dtm, valid, max_cells=5):
    """Fill missing cells belonging to components of <= max_cells cells.

    Returns (filled_dtm, valid_filled, filled_mask). Big gaps are untouched so
    that the empty-data mask used for inference stays honest.
    """
    dtm = np.asarray(dtm, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dtm)
    hole = ~valid
    if not hole.any():
        return dtm.copy(), valid.copy(), np.zeros_like(valid)
    lab, n = ndi.label(hole, structure=np.ones((3, 3), bool))
    if n == 0:
        return dtm.copy(), valid.copy(), np.zeros_like(valid)
    sizes = np.bincount(lab.ravel())
    small = sizes <= max_cells
    small[0] = False
    fill = small[lab]
    if not fill.any():
        return dtm.copy(), valid.copy(), fill
    h = np.where(valid, dtm, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    est = _norm_conv(h, w, 1.5)                      # smooth local estimate
    out = dtm.copy()
    need = fill & np.isfinite(est)
    out[need] = est[need].astype(np.float32)
    vf = valid | need
    # any cell still without an estimate keeps NaN and stays invalid
    vf &= np.isfinite(out)
    return out, vf, need


def hole_distance_m(valid, cell, cap_m=90.0):
    """Distance in metres from every cell to the nearest invalid cell."""
    if valid.all():
        return np.full(valid.shape, cap_m, dtype=np.float32)
    d = ndi.distance_transform_edt(valid) * float(cell)
    return np.minimum(d, cap_m).astype(np.float32)


def local_relief(dtm, valid, sigma_m, cell):
    """h minus Gaussian( sigma_m ) of h, NaN-aware. Blanks where support is thin."""
    sigma_cells = max(0.6, float(sigma_m) / float(cell))
    base = _norm_conv(np.where(valid, dtm, np.nan), valid.astype(np.float64), sigma_cells)
    lrm = np.where(valid, dtm - base, np.nan).astype(np.float32)
    return lrm


def residual_roughness(dtm, valid, cell, sigma_m=6.0, win_m=24.0):
    """Local RMS of the high-pass residual: canopy/vegetation noise proxy (m)."""
    hp = local_relief(dtm, valid, sigma_m, cell)
    hp2 = np.where(valid & np.isfinite(hp), hp ** 2, 0.0)
    w = (valid & np.isfinite(hp)).astype(np.float64)
    s = max(1.0, float(win_m) / float(cell) / 2.0)
    num = ndi.gaussian_filter(hp2, s, mode="nearest")
    den = ndi.gaussian_filter(w, s, mode="nearest")
    out = np.full(dtm.shape, np.nan)
    ok = den > 0.05
    out[ok] = np.sqrt(num[ok] / den[ok])
    return out.astype(np.float32)


def plane_fit(dtm, valid, cell):
    """Robust-ish global plane (removes regional tilt; row/col are y/x metres)."""
    if valid.sum() < 10:
        return np.zeros_like(dtm), np.zeros(3)
    r, c = np.nonzero(valid)
    z = dtm[r, c].astype(np.float64)
    step = max(1, len(r) // 200000)
    rs, cs, zs = r[::step], c[::step], z[::step]
    A = np.c_[rs * cell, cs * cell, np.ones(len(rs))]
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    for _ in range(2):  # re-fit on inliers (dtm noise + vegetation spikes)
        res = zs - (coef[0] * rs * cell + coef[1] * cs * cell + coef[2])
        s = np.median(np.abs(res)) / 0.6745
        keep = np.abs(res) < 3.0 * max(0.2, s)
        if keep.sum() < 10:
            break
        coef, *_ = np.linalg.lstsq(A[keep], zs[keep], rcond=None)
    resid = np.full(dtm.shape, np.nan, dtype=np.float32)
    resid[r, c] = (z - (coef[0] * r * cell + coef[1] * c * cell + coef[2])).astype(np.float32)
    return resid, coef


def hillshade(dtm, cell, valid=None, az_deg=315.0, alt_deg=35.0, zfactor=2.0, blur_m=1.5):
    """Standard hillshade 0..255, holes set to 0 (black) so they are obvious."""
    h = np.array(dtm, dtype=np.float64)
    if valid is not None:
        h[~valid] = np.nan
    hb = _norm_conv(h, np.isfinite(h).astype(np.float64), max(0.5, blur_m / cell)) \
        if np.isfinite(h).any() else h
    hb = np.where(np.isfinite(hb), hb, np.nanmedian(hb))
    gy, gx = np.gradient(hb, cell)
    slope = np.arctan(zfactor * np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.deg2rad(az_deg), np.deg2rad(alt_deg)
    sh = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    sh = np.clip(sh, 0.0, 1.0) * 255.0
    if valid is not None:
        sh[~valid] = 0.0
    return sh.astype(np.uint8)


DEFAULT_CFG = dict(fill_max_cells=5, hole_blank_m=15.0, smooth_m=3.0,
                   lrm_sigmas_m=(10.0, 15.0, 30.0), rough_sigma_m=6.0, rough_win_m=24.0)


def prepare(dtm, valid=None, cell=3.0, canopy=None, cfg=None):
    """Build the working set of rasters. Returns a dict; never mutates inputs."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    dtm = np.asarray(dtm, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(dtm)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dtm)
    dtm = np.where(valid, dtm, np.nan).astype(np.float32)

    hf, vf, filled = fill_small_holes(dtm, valid, cfg["fill_max_cells"])
    hole_d = hole_distance_m(vf, cell, cap_m=4.0 * cfg["hole_blank_m"])
    near_hole = hole_d < cfg["hole_blank_m"]
    resid, plane_coef = plane_fit(hf, vf, cell)

    h_work = _norm_conv(hf, vf.astype(np.float64), max(0.4, cfg["smooth_m"] / cell / 2.0))
    h_work = np.where(vf, h_work, np.nan).astype(np.float32)

    lrm = {}
    for s in cfg["lrm_sigmas_m"]:
        r = local_relief(h_work, vf, s, cell)
        r[near_hole] = np.nan                       # hole halos are not evidence
        r[~vf] = np.nan
        lrm[int(round(s))] = r.astype(np.float32)

    rough = residual_roughness(h_work, vf, cell, cfg["rough_sigma_m"], cfg["rough_win_m"])
    rough[near_hole] = np.nan

    chm = None
    if canopy is not None and np.ndim(canopy) == 2 and np.shape(canopy) == np.shape(dtm):
        chm = np.asarray(canopy, dtype=np.float32)
        chm = np.where(np.isfinite(chm), chm, np.nan)

    # inference mask: valid cells that are not in a hole-halo and are not edge
    infer = vf & ~near_hole
    infer &= ~np.isnan(h_work)

    return dict(dtm=dtm, h=h_work, valid=vf, orig_valid=valid, filled=filled,
                hole_dist_m=hole_d, near_hole=near_hole, detrended=resid.astype(np.float32),
                plane_coef=plane_coef, lrm=lrm, rough=rough, canopy=chm,
                infer=infer, cell=float(cell), cfg=cfg)


def blank_near_holes(arr, prep, radius_m=None):
    """Copy of arr with everything within radius_m of a hole set to NaN."""
    r = prep["cfg"]["hole_blank_m"] if radius_m is None else radius_m
    out = np.array(arr, dtype=np.float32, copy=True)
    out[prep["hole_dist_m"] < r] = np.nan
    return out
