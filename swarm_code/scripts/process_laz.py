#!/usr/bin/env python3
"""Stream an EBA LAS/LAZ transect into memory-bounded 3 m terrain products.

Example:
  .venv/bin/python code/process_laz.py .scratch/amazon/raw/NP_T-0414.laz \
    data/derived/0414 --transect 0414

This is the coordinator's independent implementation. LAS classifications are
ignored. Raw files are never deleted by this script. Arrays use south-up rows;
PNG images are north-up. No data are uploaded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import laspy
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CELL = 3.0


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def nearest_fill(a: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(a)
    if not missing.any():
        return a
    if missing.all():
        raise ValueError("cannot interpolate an entirely empty raster")
    indices = ndi.distance_transform_edt(missing, return_distances=False,
                                         return_indices=True)
    return a[tuple(indices)]


def nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(a)
    numerator = ndi.gaussian_filter(np.where(valid, a, 0.0), sigma, mode="nearest")
    denominator = ndi.gaussian_filter(valid.astype(np.float32), sigma, mode="nearest")
    out = np.full(a.shape, np.nan, np.float32)
    np.divide(numerator, denominator, out=out, where=denominator > 1e-5)
    return out


def fill_small_holes(a: np.ndarray, max_cells: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-fill connected missing components no larger than max_cells."""
    missing = ~np.isfinite(a)
    labels, _ = ndi.label(missing, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    small = sizes <= max_cells
    small[0] = False
    fill = small[labels]
    out = a.copy()
    if fill.any():
        filled = nearest_fill(a)
        out[fill] = filled[fill]
    remaining = ~np.isfinite(out)
    return out.astype(np.float32), fill, remaining


def ground_from_low(low: np.ndarray) -> dict[str, np.ndarray]:
    """Cascade 40/20/10 m grey openings and reject >0.7 m positive residuals.

    Openings use a 3x3-median-smoothed, nearest-filled working surface only as
    a reference. Rejected cells become missing; only components <=5 cells are
    interpolated afterward. Large canopy/building/data gaps are not fabricated.
    """
    original_hole = ~np.isfinite(low)
    surface = low.astype(np.float32, copy=True)
    removed_any = np.zeros(low.shape, bool)
    stages = []
    for window_m in (40, 20, 10):
        size = int(round(window_m / CELL))
        if size % 2 == 0:
            size += 1
        working = nearest_fill(surface)
        working = ndi.median_filter(working, size=3, mode="nearest")
        working = ndi.gaussian_filter(working, sigma=0.65, mode="nearest")
        opened = ndi.maximum_filter(
            ndi.minimum_filter(working, size=size, mode="nearest"),
            size=size, mode="nearest")
        remove = np.isfinite(surface) & ((surface - opened) > 0.7)
        surface[remove] = np.nan
        removed_any |= remove
        stages.append({"window_m": window_m, "window_cells": size,
                       "removed_cells": int(remove.sum())})
        del working, opened, remove
    dtm, interpolated, unfilled = fill_small_holes(surface, 5)
    return {"dtm": dtm, "original_hole": original_hole,
            "removed": removed_any, "interpolated": interpolated,
            "unfilled": unfilled, "stages": stages}


def hillshade(dtm: np.ndarray) -> np.ndarray:
    z = nearest_fill(dtm)
    dy, dx = np.gradient(z, CELL, CELL)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = math.radians(315.0)
    altitude = math.radians(45.0)
    shade = (math.sin(altitude) * np.cos(slope) +
             math.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    shade = np.clip(shade, 0, 1).astype(np.float32)
    shade[~np.isfinite(dtm)] = np.nan
    return shade


def sky_view(dtm: np.ndarray) -> np.ndarray:
    """Eight-direction positive-horizon SVF approximation at 10/20/40/80 m."""
    z = nearest_fill(dtm)
    sine_sum = np.zeros(z.shape, np.float32)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for di, dj in directions:
        horizon = np.zeros(z.shape, np.float32)
        norm = math.hypot(di, dj)
        for distance_m in (10, 20, 40, 80):
            # Round to whole cells; document the resulting sampled distances.
            step = max(1, int(round(distance_m / (CELL * norm))))
            shifted = np.full(z.shape, np.nan, np.float32)
            dy, dx = di * step, dj * step
            src_y = slice(max(0, -dy), min(z.shape[0], z.shape[0] - dy))
            src_x = slice(max(0, -dx), min(z.shape[1], z.shape[1] - dx))
            dst_y = slice(max(0, dy), min(z.shape[0], z.shape[0] + dy))
            dst_x = slice(max(0, dx), min(z.shape[1], z.shape[1] + dx))
            shifted[dst_y, dst_x] = z[src_y, src_x]
            actual_distance = CELL * step * norm
            tangent = (shifted - z) / actual_distance
            # Empty borders must not poison later horizons with NaN.
            np.fmax(horizon, tangent, out=horizon)
        sine_sum += np.sin(np.arctan(horizon))
    svf = np.clip(1.0 - sine_sum / len(directions), 0, 1).astype(np.float32)
    svf[~np.isfinite(dtm)] = np.nan
    return svf


def colorize(a: np.ndarray, kind: str) -> np.ndarray:
    """RGB image with transparent/empty cells rendered dark grey."""
    valid = np.isfinite(a)
    rgb = np.zeros(a.shape + (3,), np.uint8)
    rgb[:] = (35, 35, 35)
    if not valid.any():
        return rgb
    values = a[valid]
    if kind in ("hillshade", "svf"):
        lo, hi = 0.0, 1.0
    elif kind.startswith("relief"):
        lo, hi = np.nanpercentile(values, [2, 98])
        bound = max(abs(lo), abs(hi), 0.15)
        lo, hi = -bound, bound
    elif kind == "chm":
        lo, hi = 0.0, max(5.0, float(np.nanpercentile(values, 98)))
    else:
        lo, hi = np.nanpercentile(values, [2, 98])
        if hi - lo < 0.2:
            hi = lo + 0.2
    norm = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    if kind.startswith("relief"):
        # Blue depressions, pale level ground, ochre banks.
        mid = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        mid = np.nan_to_num(mid, nan=0.0, posinf=1.0, neginf=0.0)
        rgb[..., 0] = np.where(mid < .5, 40 + 430 * mid, 255).clip(0, 255).astype(np.uint8)
        rgb[..., 1] = np.where(mid < .5, 80 + 350 * mid, 255 - 80 * (mid - .5)).clip(0, 255).astype(np.uint8)
        rgb[..., 2] = np.where(mid < .5, 180 + 150 * mid, 215 - 430 * (mid - .5)).clip(0, 255).astype(np.uint8)
    elif kind in ("hillshade", "svf"):
        grey = (norm * 255).astype(np.uint8)
        rgb[:] = grey[..., None]
    elif kind == "chm":
        rgb[..., 0] = (20 + 210 * norm).astype(np.uint8)
        rgb[..., 1] = (60 + 190 * np.sqrt(norm)).astype(np.uint8)
        rgb[..., 2] = (35 + 60 * (1 - norm)).astype(np.uint8)
    else:
        # Compact terrain palette without a matplotlib dependency.
        rgb[..., 0] = (40 + 215 * norm).astype(np.uint8)
        rgb[..., 1] = (80 + 150 * np.sin(norm * math.pi)).astype(np.uint8)
        rgb[..., 2] = (100 + 100 * (1 - norm)).astype(np.uint8)
    rgb[~valid] = (35, 35, 35)
    return rgb


def write_png(path: Path, a: np.ndarray, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = colorize(a, kind)
    Image.fromarray(rgb[::-1]).save(path, optimize=True)


def write_products(output: Path, arrays: dict[str, np.ndarray], grid: dict) -> None:
    path = output / "products.npz"
    temporary = output / "products.npz.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    dtm_path = output / "dtm.npz"
    temporary = output / "dtm.npz.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, dtm=arrays["dtm"],
                            x0=np.float64(grid["x0"]), y0=np.float64(grid["y0"]),
                            cell=np.float64(grid["cell"]), epsg=np.int32(grid["epsg"]),
                            row_order=np.array("south-up"))
    os.replace(temporary, dtm_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transect", required=True)
    parser.add_argument("--chunk", type=int, default=2_000_000)
    parser.add_argument("--skip-tiles", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    processing = {"transect": args.transect, "input": str(args.input),
                  "status": "running", "started_unix": time.time()}
    atomic_json(args.output / "processing.json", processing)

    with laspy.open(args.input) as reader:
        header = reader.header
        crs = header.parse_crs()
        if crs is None or crs.to_epsg() is None:
            raise ValueError("a projected EPSG CRS is required")
        epsg = crs.to_epsg()
        if crs.axis_info[0].unit_name not in ("metre", "meter"):
            raise ValueError(f"projected metre CRS required, got {crs}")
        minx, miny, minz = header.mins
        maxx, maxy, maxz = header.maxs
        x0 = math.floor(minx / CELL) * CELL
        y0 = math.floor(miny / CELL) * CELL
        nx = int(math.ceil((maxx - x0) / CELL)) + 1
        ny = int(math.ceil((maxy - y0) / CELL)) + 1
        ncells = nx * ny
        zmin = np.full(ncells, np.inf, np.float32)
        zmax = np.full(ncells, -np.inf, np.float32)
        count = np.zeros(ncells, np.uint32)
        seen = 0
        outside = 0
        for points in reader.chunk_iterator(args.chunk):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z, dtype=np.float32)
            ix = np.floor((x - x0) / CELL).astype(np.int64)
            iy = np.floor((y - y0) / CELL).astype(np.int64)
            good = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) &
                    np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
            outside += int((~good).sum())
            flat = (iy[good] * nx + ix[good]).astype(np.int64, copy=False)
            zz = z[good]
            np.minimum.at(zmin, flat, zz)
            np.maximum.at(zmax, flat, zz)
            count += np.bincount(flat, minlength=ncells).astype(np.uint32)
            seen += int(good.sum())
            print(f"rasterised {seen}/{header.point_count} points", flush=True)
        if seen != header.point_count:
            raise ValueError(f"point count mismatch: {seen} versus {header.point_count}")
        grid = {"x0": x0, "y0": y0, "nx": nx, "ny": ny, "cell": CELL,
                "epsg": int(epsg), "row_order": "south-up",
                "source_bounds": {"minx": float(minx), "maxx": float(maxx),
                                  "miny": float(miny), "maxy": float(maxy),
                                  "minz": float(minz), "maxz": float(maxz)}}
        raster_seconds = time.monotonic() - started

    observed = count > 0
    zmin[~observed] = np.nan
    zmax[~observed] = np.nan
    low = zmin.reshape(ny, nx)
    high = zmax.reshape(ny, nx)
    counts = count.reshape(ny, nx)
    del zmin, zmax, count
    ground = ground_from_low(low)
    dtm = ground["dtm"]
    # Height of the highest recorded return above the reconstructed ground.
    chm = (high - dtm).astype(np.float32)
    chm[~np.isfinite(dtm) | ~np.isfinite(high)] = np.nan
    chm[np.isfinite(chm) & (chm < 0)] = 0
    relief = {}
    for scale in (10, 15, 30):
        smoothed = nan_gaussian(dtm, scale / CELL)
        relief[scale] = (dtm - smoothed).astype(np.float32)
    # Blank all relief within 15 m of an original no-return cell, including
    # small holes that were interpolated for continuity of the DTM.
    distance_to_hole = ndi.distance_transform_edt(~ground["original_hole"]) * CELL
    relief_hole_mask = distance_to_hole <= 15.0
    for scale in relief:
        relief[scale][relief_hole_mask | ground["unfilled"]] = np.nan
    shade = hillshade(dtm)
    svf = sky_view(dtm)
    arrays = {"dtm": dtm, "chm": chm, "hillshade": shade, "svf": svf,
              "relief10": relief[10], "relief15": relief[15],
              "relief30": relief[30], "count": counts.astype(np.uint32),
              "original_hole": ground["original_hole"],
              "removed": ground["removed"],
              "interpolated": ground["interpolated"],
              "unfilled": ground["unfilled"],
              "relief_hole_mask": relief_hole_mask}
    write_products(args.output, arrays, grid)
    atomic_json(args.output / "grid.json", grid)

    layers = {"dtm": dtm, "chm": chm, "hillshade": shade, "svf": svf,
              "relief10": relief[10], "relief15": relief[15], "relief30": relief[30]}
    for name, array in layers.items():
        write_png(args.output / f"{name}.png", array, name)
    tile_records = []
    if not args.skip_tiles:
        cells_per_tile = int(round(1200 / CELL))
        for row in range(0, ny, cells_per_tile):
            for col in range(0, nx, cells_per_tile):
                r1, c1 = min(row + cells_per_tile, ny), min(col + cells_per_tile, nx)
                valid_fraction = np.isfinite(dtm[row:r1, col:c1]).mean()
                if valid_fraction < 0.01:
                    continue
                tile_id = f"r{row // cells_per_tile:02d}_c{col // cells_per_tile:02d}"
                for name, array in layers.items():
                    write_png(args.output / "tiles" / f"{tile_id}_{name}.png",
                              array[row:r1, col:c1], name)
                tile_records.append({"id": tile_id, "row": row, "col": col,
                                     "row_end": r1, "col_end": c1,
                                     "valid_dtm_fraction": float(valid_fraction),
                                     "x0": x0 + col * CELL, "y0": y0 + row * CELL,
                                     "x1": x0 + c1 * CELL, "y1": y0 + r1 * CELL})
    summary = {
        **processing, **grid, "status": "products_written",
        "finished_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "rasterisation_seconds": raster_seconds,
        "points": int(seen), "points_outside_grid": int(outside),
        "grid_cells": int(ncells), "observed_cells": int(observed.sum()),
        "empty_grid_pct": float(100 * (1 - observed.mean())),
        "empty_dtm_pct_of_grid": float(100 * ground["unfilled"].mean()),
        "empty_dtm_pct_of_observed_support": float(
            100 * ground["unfilled"][counts > 0].sum() / max(1, observed.sum())),
        "mean_returns_per_observed_3m_cell": float(counts[counts > 0].mean()),
        "median_returns_per_observed_3m_cell": float(np.median(counts[counts > 0])),
        "removed_cells": int(ground["removed"].sum()),
        "interpolated_cells": int(ground["interpolated"].sum()),
        "relief_hole_masked_cells": int(relief_hole_mask.sum()),
        "ground_filter_stages": ground["stages"],
        "tile_count": len(tile_records),
        "dtm_valid_min_max_m": [float(np.nanmin(dtm)), float(np.nanmax(dtm))],
        "chm_valid_p50_p95_p99_m": [float(v) for v in np.nanpercentile(chm, [50, 95, 99])],
        "method": {"classes_ignored": True, "cell_m": CELL,
                   "ground": "3x3 median + Gaussian 0.65-cell reference; grey opening 40/20/10m; remove >0.7m positive residual; fill connected missing components <=5 cells",
                   "relief": "DTM minus nan-aware Gaussian at sigma 10/15/30m; blank <=15m from any original empty cell",
                   "svf": "8 directions; approximate horizons at nominal 10/20/40/80m, rounded to grid cells",
                   "raw_deleted": False}
    }
    atomic_json(args.output / "tiles.json", {"tiles": tile_records})
    atomic_json(args.output / "processing.json", summary)
    print("PROCESSING COMPLETE", json.dumps({
        "transect": args.transect, "seconds": summary["elapsed_seconds"],
        "tiles": len(tile_records), "empty_dtm_pct_of_grid": summary["empty_dtm_pct_of_grid"],
        "products_bytes": (args.output / "products.npz").stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
