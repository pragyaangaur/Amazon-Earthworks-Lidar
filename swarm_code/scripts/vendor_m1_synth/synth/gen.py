"""Generate synthetic EBA-like airborne-lidar scenes plus auditable ground truth.

Usage (from .ultramath/smwduocxj/m1):
  ../.../repo/.venv/bin/python synth/gen.py --scene all
  ... --scene s1_ring40_open --seed 1234

Writes, per scene, into out/:
  <scene>.laz            compressed LAS 1.2 point format 1, EPSG:32719
  <scene>_truth.json     scene definition + feature table + counts + timing
  <scene>_truth.npz      noiseless 3 m truth DTM and per-cell point statistics
  <scene>_truth.png      hillshade of the noiseless truth DTM with overlays

The cloud is generated and written in x-slabs, so memory stays bounded; a
12.5 km transect is never held as one array.  Nothing here touches the
network; no real position is used - all coordinates are inside a synthetic
UTM 19S block far from any real site and are labelled as synthetic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes import SCENES, get                    # noqa: E402
from lidar import build_canopy, pulse_chunk, simulate_returns, write_laz  # noqa: E402
from surface import ground                        # noqa: E402
from viz import quicklook                         # noqa: E402


def _grid(scene):
    g = float(scene["grid_m"])
    nx = int(np.floor((scene["x1"] - scene["x0"]) / g))
    ny = int(np.floor((scene["y1"] - scene["y0"]) / g))
    return g, nx, ny


def build_roughness(scene, rng):
    """Micro-relief: smooth random field + sparse tree-throw / hollow pits.

    Real open pasture is NOT smooth.  Measured on the real 0414 product (m1,
    control_check.py, open-canopy cells only) the ring-dip null at r = 20 m has
    p95 = +0.12 m, max = +0.94 m, and the azimuth-robust statistic (az25) reaches
    0.25 m over ~2000 centres.  Without micro-relief a synthetic pasture has a
    null of 0.01 m, i.e. it would make every threshold look safe.  Amplitudes
    here are calibrated to reproduce that tail (see notes/validation.md).
    """
    cfg = scene.get("roughness")
    if not cfg:
        return None
    from scipy import ndimage
    g = 1.0
    nx = int(np.floor((scene["x1"] - scene["x0"]) / g))
    ny = int(np.floor((scene["y1"] - scene["y0"]) / g))
    white = rng.standard_normal((ny, nx)).astype(np.float32)
    field = np.zeros((ny, nx), dtype=np.float32)
    for sig, w in cfg.get("scales", [[6.0, 0.55], [15.0, 0.32], [35.0, 0.18]]):
        f = ndimage.gaussian_filter(white, sig / g, mode="nearest")
        sd = float(f.std())
        if sd > 0:
            field += np.float32(w) * f / sd
    field *= float(cfg.get("field_amp_m", 0.10))
    # sparse hollows / tree-throw pits: the heavy negative tail
    dens = float(cfg.get("pit_density", 2e-4))
    n = int(rng.poisson(dens * (scene["x1"] - scene["x0"]) * (scene["y1"] - scene["y0"])))
    if n:
        px = scene["x0"] + rng.random(n) * (scene["x1"] - scene["x0"])
        py = scene["y0"] + rng.random(n) * (scene["y1"] - scene["y0"])
        amp = -(float(cfg.get("pit_min_m", 0.35))
                + (float(cfg.get("pit_max_m", 1.6)) - float(cfg.get("pit_min_m", 0.35)))
                * rng.random(n) ** 2.2)
        sig = float(cfg.get("pit_sigma", 5.0)) * (0.6 + 0.8 * rng.random(n))
        for k in range(n):
            r = 3.0 * sig[k] / g
            i0 = max(0, int((px[k] - scene["x0"]) / g - r))
            i1 = min(nx, int((px[k] - scene["x0"]) / g + r) + 1)
            j0 = max(0, int((py[k] - scene["y0"]) / g - r))
            j1 = min(ny, int((py[k] - scene["y0"]) / g + r) + 1)
            if i0 >= i1 or j0 >= j1:
                continue
            dx = (np.arange(i0, i1) - (px[k] - scene["x0"]) / g) * g
            dy = (np.arange(j0, j1) - (py[k] - scene["y0"]) / g) * g
            field[j0:j1, i0:i1] += (amp[k] * np.exp(
                -(dx[None, :] ** 2 + dy[:, None] ** 2) / (2.0 * sig[k] ** 2))).astype(np.float32)
    return field


def truth_dem(scene, sub=6):
    """Noiseless 3 m truth DTM, comparable with a 'lowest return per cell' DTM.

    A DTM built as the lowest return in a cell is close to the *minimum* of the
    ground surface over that cell, not to a point sample at the cell centre.
    Sampling only the centre underestimates ditch depth on narrow features
    (measured: a 5 m ditch reads 0.27 m at cell centres but 0.55 m in a
    min-based DTM).  So the truth raster is the min of `sub*sub` sub-samples.
    """
    g, nx, ny = _grid(scene)
    sub = int(sub)
    dg = g / sub
    out_min = np.empty((ny, nx), dtype=np.float32)
    out_mean = np.empty((ny, nx), dtype=np.float32)
    ys = scene["y0"] + (np.arange(ny * sub) + 0.5) * dg
    xs = scene["x0"] + (np.arange(nx * sub) + 0.5) * dg
    step = max(1, int(4e6 // max(1, xs.size)))
    for j in range(0, ny, step):
        j1 = min(ny, j + step)
        Y = ys[j * sub:j1 * sub]
        X, YY = np.meshgrid(xs, Y)
        z = ground(X.ravel(), YY.ravel(), scene).reshape(Y.size, xs.size)
        z = z.reshape(Y.size // sub, sub, nx, sub)
        out_min[j:j1] = z.min(axis=(1, 3))
        out_mean[j:j1] = z.mean(axis=(1, 3))
    return out_min, out_mean


def generate(scene, out_dir, seed=1234, chunk_m=2500.0, verbose=True):
    t_all = time.time()
    rng = np.random.default_rng(seed)
    g, nx, ny = _grid(scene)
    x0, y0 = scene["x0"], scene["y0"]
    name = scene["name"]
    os.makedirs(out_dir, exist_ok=True)

    t = time.time()
    rough = build_roughness(scene, rng)
    if rough is not None:
        scene["_rough_field"] = rough
        scene["_rough_grid_m"] = 1.0
    chm, cover, dens = build_canopy(scene, rng)
    t_canopy = time.time() - t

    pulses_g = np.zeros((ny, nx), dtype=np.int32)
    returns_g = np.zeros((ny, nx), dtype=np.int32)
    ground_g = np.zeros((ny, nx), dtype=np.int32)
    chm_sum = np.zeros((ny, nx), dtype=np.float64)
    cov_sum = np.zeros((ny, nx), dtype=np.float64)

    stats = dict(pulses=0, returns=0, ground=0, canopy_pulses=0)
    cls_hist = np.zeros(16, dtype=np.int64)
    rn_hist = np.zeros(6, dtype=np.int64)

    def chunks():
        nonlocal cls_hist, rn_hist
        x = scene["x0"]
        while x < scene["x1"]:
            xhi = min(scene["x1"], x + chunk_m)
            px, py, line = pulse_chunk(scene, rng, x, xhi)
            ch = simulate_returns(scene, rng, px, py, chm, cover, line)
            # --- per-cell statistics -------------------------------------
            ix = np.clip(((ch["p_x"] - x0) / g).astype(np.int64), 0, nx - 1)
            iy = np.clip(((ch["p_y"] - y0) / g).astype(np.int64), 0, ny - 1)
            flat = iy * nx + ix
            np.add.at(pulses_g.reshape(-1), flat, 1)
            np.add.at(chm_sum.reshape(-1), flat, ch["p_H"])
            np.add.at(cov_sum.reshape(-1), flat, ch["p_q"] * 0.0 + 1.0)
            zg_ret = np.repeat(ch["p_zg"], ch["p_nret"])
            ixr = np.clip(((ch["x"] - x0) / g).astype(np.int64), 0, nx - 1)
            iyr = np.clip(((ch["y"] - y0) / g).astype(np.int64), 0, ny - 1)
            flatr = iyr * nx + ixr
            np.add.at(returns_g.reshape(-1), flatr, 1)
            is_ground = np.abs(ch["z"] - zg_ret) < 0.4
            np.add.at(ground_g.reshape(-1), flatr[is_ground], 1)
            cls_hist += np.bincount(ch["cls"], minlength=16)
            rn_hist += np.bincount(np.clip(ch["rn"], 0, 5), minlength=6)
            stats["pulses"] += int(ch["p_nret"].size)
            stats["returns"] += int(ch["x"].size)
            stats["ground"] += int(is_ground.sum())
            stats["canopy_pulses"] += int((ch["p_H"] > 1.0).sum())
            if verbose:
                print("  slab %8.0f-%-8.0f pulses %8d returns %9d" %
                      (x - x0, xhi - x0, ch["p_nret"].size, ch["x"].size),
                      flush=True)
            yield ch
            x = xhi

    laz = os.path.join(out_dir, name + ".laz")
    t = time.time()
    n_written = write_laz(laz, chunks(), scene, epsg=scene["epsg"])
    t_write = time.time() - t

    # ---- authoritative read-back: does the file parse, and does it agree? --
    t = time.time()
    verify = verify_laz(laz, scene)
    t_verify = time.time() - t
    verify["count_matches_stats"] = bool(n_written == stats["returns"] == verify["count"])
    verify["class_hist_matches"] = bool(
        list(verify["class_hist"].values()) == [int(v) for v in cls_hist])
    verify["ok"] = bool(verify["count_matches_stats"] and verify["crs_ok"]
                        and verify["bbox_covers_scene"])

    # ---- truth rasters ----------------------------------------------------
    dem, dem_mean = truth_dem(scene)
    npz = os.path.join(out_dir, name + "_truth.npz")
    with_pulses = pulses_g > 0
    np.savez_compressed(
        npz, dem_true=dem, dem_true_mean=dem_mean,
        pulses=pulses_g, returns=returns_g,
        ground_returns=ground_g,
        chm_mean=np.where(with_pulses, chm_sum / np.maximum(pulses_g, 1), 0.0).astype(np.float32),
        x0=x0, y0=y0, x1=scene["x1"], y1=scene["y1"], grid_m=g, epsg=scene["epsg"])

    area = (scene["x1"] - scene["x0"]) * (scene["y1"] - scene["y0"])
    empty_cells = int(((returns_g == 0)).sum())
    contam_cells = int(((ground_g == 0) & (returns_g > 0)).sum())
    meta = dict(
        scene=name, epsg=scene["epsg"],
        bbox=[scene["x0"], scene["y0"], scene["x1"], scene["y1"]],
        grid_m=g, npix=[int(nx), int(ny)], area_m2=area, seed=seed,
        laz=os.path.basename(laz), laz_bytes=os.path.getsize(laz),
        las_version="1.2", point_format=1,
        offsets=verify.get("offsets"), scales=verify.get("scales"),
        counts=dict(pulses=stats["pulses"], returns=stats["returns"],
                    ground_returns=stats["ground"],
                    canopy_pulses=stats["canopy_pulses"],
                    class_hist={str(k): int(v) for k, v in enumerate(cls_hist) if v},
                    return_number_hist={str(k): int(v) for k, v in enumerate(rn_hist) if v}),
        density=dict(pulses_per_m2=stats["pulses"] / area,
                     returns_per_m2=stats["returns"] / area,
                     ground_returns_per_m2=stats["ground"] / area),
        cells=dict(total=int(nx * ny), no_returns=empty_cells,
                   no_ground_but_returns=contam_cells,
                   empty_pct=100.0 * empty_cells / (nx * ny),
                   contaminated_pct=100.0 * contam_cells / (nx * ny)),
        canopy=dict(crown_density_max=float(np.max(dens)) if dens.size else 0.0,
                    chm_max=float(np.nanmax(chm)) if chm.size else 0.0,
                    chm_mean=float(np.mean(chm)) if chm.size else 0.0,
                    cover_mean=float(np.mean(cover)) if cover.size else 0.0),
        timing_s=dict(canopy=t_canopy, write=t_write, verify=t_verify,
                      total=time.time() - t_all),
        verify=verify,
        features=scene["features"], holes=scene["holes"],
        truth_npz=os.path.basename(npz), truth_png=os.path.basename(name + "_truth.png"),
        synthetic_note=("SYNTHETIC test scene. Coordinates are an invented block "
                        "in UTM 19S and do not correspond to any real place. "
                        "Generated for pipeline validation only."),
    )
    with open(os.path.join(out_dir, name + "_truth.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # ---- quicklook PNG (tiled for long transects) -------------------------
    step_px = 2000
    for k, i0 in enumerate(range(0, nx, step_px)):
        i1 = min(nx, i0 + step_px)
        suf = "" if nx <= step_px else "_tile%d" % k
        quicklook(os.path.join(out_dir, name + "_truth%s.png" % suf),
                  dem[:, i0:i1], x0 + i0 * g, y0, g,
                  title="%s truth DTM (noiseless) %s" % (name, suf),
                  features=scene["features"], holes=scene["holes"],
                  figsize=(8, max(3.0, 8 * (i1 - i0) / nx * (nx / ny) * 0.55)))
    if verbose:
        print("%-22s pulses %8d returns %9d %.1f MB %.1fs verify=%s empty=%.1f%% contam=%.1f%%"
              % (name, stats["pulses"], stats["returns"], os.path.getsize(laz) / 1e6,
                 time.time() - t_all, verify["ok"], meta["cells"]["empty_pct"],
                 meta["cells"]["contaminated_pct"]), flush=True)
    return meta


def verify_laz(path, scene):
    """Reopen the LAZ with laspy and check counts / CRS / bbox / histograms."""
    import laspy
    out = dict(count=0, class_hist={}, crs=None, crs_ok=False,
               offsets=None, scales=None, bbox_covers_scene=False, bounds=None)
    hist = np.zeros(16, dtype=np.int64)
    mnx = mny = np.inf
    mxx = mxy = -np.inf
    mnz, mxz = np.inf, -np.inf
    with laspy.open(path, laz_backend=laspy.LazBackend.LazrsParallel) as f:
        h = f.header
        out["offsets"] = [float(v) for v in h.offsets]
        out["scales"] = [float(v) for v in h.scales]
        crs = h.parse_crs()
        out["crs"] = str(crs) if crs is not None else None
        out["crs_ok"] = bool(crs is not None and crs.to_epsg() == scene["epsg"])
        for pts in f.chunk_iterator(4_000_000):
            out["count"] += int(len(pts.x))
            hist += np.bincount(np.asarray(pts.classification, dtype=np.int64),
                                minlength=16)
            X = np.asarray(pts.x, dtype=np.float64)
            mnx = min(mnx, float(X.min())); mxx = max(mxx, float(X.max()))
            mny = min(mny, float(pts.y.min())); mxy = max(mxy, float(pts.y.max()))
            mnz = min(mnz, float(pts.z.min())); mxz = max(mxz, float(pts.z.max()))
    out["class_hist"] = {str(k): int(v) for k, v in enumerate(hist) if v}
    out["bounds"] = [mnx, mny, mnz, mxx, mxy, mxz]
    pad = 1.0
    out["bbox_covers_scene"] = bool(
        mnx >= scene["x0"] - pad and mxx <= scene["x1"] + pad
        and mny >= scene["y0"] - pad and mxy <= scene["y1"] + pad)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="s1_ring40_open",
                    help="scene name or 'all' (%s)" % ", ".join(sorted(SCENES)))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "out"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-m", type=float, default=2500.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    names = sorted(SCENES) if a.scene in ("all", "ALL") else [a.scene]
    t = time.time()
    metas = []
    for n in names:
        sc = get(n)
        metas.append(generate(sc, a.out, seed=a.seed, chunk_m=a.chunk_m,
                              verbose=not a.quiet))
    print("total %.1fs -> %s" % (time.time() - t, os.path.abspath(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
