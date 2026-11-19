"""diagnose.py -- per-detection evidence: a 4-panel PNG plus a JSON profile.

Panels (what a skeptic needs to try a natural/modern explanation):
  1. hillshade of the raw DTM around the feature, with the detected circle and
     the hole mask, so a terraced edge or gully shows itself;
  2. the detector input raster (local relief) with the same overlays;
  3. the ring-mean radial profile M(r) with the flank reference line and the
     half-depth crossings used for the width estimate;
  4. per-ray trough depth and trough radius against azimuth - a dug ditch is
     flat in depth and flat in radius; a river bend, gully head or vegetation
     patch is not.

Coordinates are *never* written into the PNG or the JSON: the panel is labelled
with the local detection id and grid cells only.  Map coordinates are exported
separately (run_scan/run_real write them to the CSV that the coordinator keeps
private).
"""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _window(arr, row, col, half):
    nx, ny = arr.shape
    r0, r1 = max(0, int(row - half)), min(nx, int(row + half) + 1)
    c0, c1 = max(0, int(col - half)), min(ny, int(col + half) + 1)
    return arr[r0:r1, c0:c1], (r0, c0)


def panel_png(path, prep, det, det_id, half_m=None, extra_lines=None, dpi=110):
    cell = float(prep["cell"])
    row, col = det["row"], det["col"]
    r_m = det["radius_m"]
    half = int(round((half_m or max(3.0 * r_m, 90.0)) / cell))
    hs, (r0, c0) = _window(prep.get("hillshade_dtm", prep["h"]), row, col, half)
    lrm = (prep["lrm"].get(15) if prep.get("lrm") else None)
    if lrm is None:
        lrm = prep.get("h")
    lrf, _ = _window(np.nan_to_num(lrm, nan=np.nan), row, col, half)
    prof = det.get("_profile") or det.get("profile") or {}
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    ext = (-(col - c0) * cell, (hs.shape[1] - (col - c0)) * cell,
           -(hs.shape[0] - (row - r0)) * cell, (row - r0) * cell)
    for a, im, ttl, cmap in ((ax[0, 0], hs, "DTM hillshade", "gray"),
                             (ax[0, 1], lrf, "detector input (LRM 15 m)", "RdBu_r")):
        vmax = np.nanpercentile(np.abs(im), 98) if np.isfinite(im).any() else 1.0
        kw = dict(extent=ext, origin="upper", cmap=cmap)
        if cmap == "RdBu_r":
            kw.update(vmin=-vmax, vmax=vmax)
        a.imshow(im, **kw)
        a.add_patch(Circle((0, 0), r_m, fill=False, ec="#00ff00", lw=1.6))
        a.add_patch(Circle((0, 0), max(0.5 * det.get("width_m", 6.0), 3.0) + r_m,
                           fill=False, ec="#ffcc00", lw=0.8, ls="--"))
        a.add_patch(Circle((0, 0), max(r_m - max(0.5 * det.get("width_m", 6.0), 3.0), 1.0),
                           fill=False, ec="#ffcc00", lw=0.8, ls="--"))
        a.plot(0, 0, "+", color="#00ffff", ms=12, mew=1.6)
        for x0, y0, x1, y1, lab in (extra_lines or []):
            a.plot([(x0 - col) * cell, (x1 - col) * cell], [(row - y0) * cell, (row - y1) * cell],
                   "-", color="#ff00ff", lw=1.0)
        a.set_title("%s  (%s)" % (ttl, det_id), fontsize=9)
        a.set_xlabel("east of centre (m)", fontsize=8)
        a.set_ylabel("north of centre (m)", fontsize=8)
        a.tick_params(labelsize=7)
    # radial profile
    a = ax[1, 0]
    if prof.get("radius_m"):
        rr = np.array(prof["radius_m"], float)
        mm = np.array([np.nan if v is None else v for v in prof["ring_mean_m"]], float)
        a.plot(rr, mm, "-", color="#1f77b4", lw=1.4, label="ring-mean height")
        a.axvline(r_m, color="#00aa00", lw=1.0, label="detected radius %.0f m" % r_m)
        if np.isfinite(det.get("width_m", np.nan)):
            hw = 0.5 * det["width_m"]
            a.axvspan(r_m - hw, r_m + hw, color="#00aa00", alpha=0.12,
                      label="half-depth width %.0f m" % det["width_m"])
        if np.isfinite(det.get("dip_m", np.nan)):
            a.set_title("radial profile: dip %.2f m, cov %.2f, rel.std %.2f"
                        % (det["dip_m"], det.get("cov", np.nan), det.get("rel_std", np.nan)),
                        fontsize=9)
    a.set_xlabel("radius (m)", fontsize=8)
    a.set_ylabel("height relative to ring mean (m)", fontsize=8)
    if a.get_legend_handles_labels()[0]:
        a.legend(fontsize=7)
    a.tick_params(labelsize=7)
    a.grid(alpha=0.3)
    # per-ray dip and radius
    a = ax[1, 1]
    if prof.get("per_ray_dip_m"):
        n = len(prof["per_ray_dip_m"])
        ang = np.arange(n) * 360.0 / n
        dp = np.array([np.nan if v is None else v for v in prof["per_ray_dip_m"]], float)
        a.plot(ang, dp, "o-", ms=2.5, lw=0.9, color="#d62728", label="per-ray dip (m)")
        if prof.get("r_theta_m"):
            rt = np.array([np.nan if v is None else v for v in prof["r_theta_m"]], float)
            a2 = a.twinx()
            a2.plot(ang, rt, "s-", ms=2.0, lw=0.7, color="#2ca02c", label="per-ray radius (m)")
            a2.set_ylabel("trough radius (m)", fontsize=8, color="#2ca02c")
            a2.tick_params(labelsize=7, colors="#2ca02c")
        a.axhline(0, color="k", lw=0.6)
        a.set_title("azimuth consistency (0=N, clockwise)", fontsize=9)
    a.set_xlabel("azimuth (deg)", fontsize=8)
    a.set_ylabel("per-ray dip (m)", fontsize=8, color="#d62728")
    a.tick_params(labelsize=7)
    a.grid(alpha=0.3)
    txt = ("id=%s  R=%.1f m  D=%.1f m  dip=%.2f m  width=%s m  bank=%s m  SNR=%.1f\n"
           "cov=%.2f  rel.std=%.2f  two_sided=%.2f  interior=%s m  circ=%.2f  gap=%.0f deg  %s\n"
           "canopy=%s m  rough=%s m  hole_blank_band=%s  shape=%s\n"
           "step_frac=%s  step_const=%s m  interior_flat=%s  snr=%s\nflags: %s") % (
        det_id, r_m, 2 * r_m, det.get("dip_m", np.nan),
        _f(det.get("width_m")), _f(det.get("bank_m")), det.get("snr", np.nan),
        det.get("cov", np.nan), det.get("rel_std", np.nan), det.get("two_sided", np.nan),
        _f(det.get("interior_m")), det.get("circularity", np.nan), det.get("gap_deg", np.nan),
        det.get("shape", ""), _f(det.get("ctx_canopy_ann")), _f(det.get("ctx_rough_ann")),
        _f(det.get("blank_band")), det.get("shape", ""),
        _f(det.get("step_frac")), _f(det.get("step_const_m")),
        det.get("shape", ""), _f(det.get("snr")),
        ", ".join(det.get("flags", [])))
    fig.text(0.01, 0.005, txt, fontsize=7.2, family="monospace", va="bottom")
    fig.suptitle("Earthwork screening evidence %s (local id only; positions in the private CSV)"
                 % det_id, fontsize=10)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _f(v, n=2):
    try:
        if v is None or not np.isfinite(float(v)):
            return "-"
        return ("%%.%df" % n) % float(v)
    except Exception:
        return "-"


def profile_json(det):
    """Small, machine-readable record a skeptic can re-plot."""
    keep = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in det.items() if not str(k).startswith("_")}
    keep["profile"] = det.get("_profile") or det.get("profile") or {}
    return keep


def write_evidence(outdir, prep, det, det_id, extra_lines=None):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "%s_panel.png" % det_id)
    panel_png(png, prep, det, det_id, extra_lines=extra_lines)
    return png
