"""run_transect.py -- run the m3 causeway / raised-embankment detector over a
real transect DTM and write coordinator-ready local products.

Inputs this VM can read: `.npy` + sidecar `.npy.json` (authoritative), float32
GeoTIFF with ModelPixelScale/ModelTiepoint/GeoKeys (Pillow only), and uncompressed
`.las`.  LASzip-compressed `.laz` CANNOT be read here: there is no laspy, lazrs,
pdal, laszip or liblaszip in this VM and members have no network.  Use
`read_las_xyz()` only for `.las`; a `.laz` must be converted by the coordinator
(e.g. `laszip -i in.laz -o out.las` on a networked machine) or replaced by a DTM
from the processors.

Outputs, per transect, under <outdir>/<transect>/:
  detections_<transect>.csv   one row per verified ridge, with all diagnostics
  diagnostics_<transect>.json every sampled profile, for skeptics to re-derive
  candidates_rows.csv         detection rows in the shared candidates.csv columns
  rejected_auto.csv           lines that were measured but failed a gate, with the gate
  summary.json                timing, hole %, drainage fraction, candidate counts
  evidence/<id>/{map,profiles}.png
  evidence/<id>/evidence.json
Nothing here decides certainty: every row is written as Possible pending the
coordinator's record and literature check.

Usage
  python3 code/run_transect.py --dtm path/to/dtm.npy --transect 0414 \
      --outdir results/transects [--block-m 1200] [--overlap-m 120] [--epsg 31981]
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causeway
import dtlib as D

# --------------------------------------------------------------------------
# UTM -> lon/lat (no pyproj in this VM).  Karney/standard series, good to < 1 cm
# inside a UTM zone; verified by forward/inverse round-trip in `self_test_utm`.
# --------------------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def _zones(a=None, f=None):
    """EPSG -> (UTM zone, hemisphere) for the South-American UTM blocks used here.

    WGS 84 / UTM zone nS  = 32700 + n   (32718 = 18S ... 32725 = 25S)
    SIRGAS 2000 / UTM zone nS = 31960 + n (31978 = 18S ... 31985 = 25S)
    Only the southern hemisphere is used by this project.  The mapping is checked
    against independently sourced known-site coordinates in `check_epsg()`, which
    tests how many Peripato sites fall inside a transect footprint -- a wrong zone
    puts all of them outside.
    """
    if a is None:
        a, f = WGS84_A, WGS84_F
    z = {}
    for n in range(18, 26):
        z[32700 + n] = (n, -1, a, f)
        z[31960 + n] = (n, -1, a, f)
    return z


def check_epsg(csv_path, footprint_xy, epsg_candidates=(31978, 31979, 31980, 31981)):
    """Count known sites that fall inside a footprint for each candidate EPSG."""
    import csv as _csv
    with open(csv_path) as fh:
        rows = list(_csv.DictReader(fh))
    out = {}
    for epsg in epsg_candidates:
        inside = 0
        for r in rows:
            try:
                lon = float(r["Longitude"]); lat = float(r["Latitude"])
            except Exception:
                continue
            x, y = utm_from_lonlat(epsg, lon, lat)
            if (footprint_xy[0] <= x <= footprint_xy[1] and
                    footprint_xy[2] <= y <= footprint_xy[3]):
                inside += 1
        out[epsg] = inside
    best = max(out, key=out.get) if out else None
    return dict(counts=out, best=best, footprint=footprint_xy)


def utm_to_lonlat(epsg, easting, northing):
    """Inverse UTM for WGS84 / SIRGAS2000 zones 18S-20S.  Returns (lon, lat) degrees."""
    epsg = int(epsg)
    zones = _zones(a=WGS84_A, f=WGS84_F)
    if epsg == 4326:
        return float(easting), float(northing)
    if epsg not in zones:
        raise ValueError(f"no inverse UTM for EPSG:{epsg}; supply longitude/latitude "
                         f"through the sidecar instead")
    zone, hemi, a, f = zones[epsg]
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    x = float(easting) - 500000.0
    y = float(northing)
    if hemi < 0:
        y -= 10000000.0
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
          + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
          + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))
    C1 = ep2 * math.cos(p1) ** 2
    T1 = math.tan(p1) ** 2
    N1 = a / math.sqrt(1 - e2 * math.sin(p1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * math.sin(p1) ** 2) ** 1.5
    Dm = x / (N1 * k0)
    lat = p1 - (N1 * math.tan(p1) / R1) * (
        Dm ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * Dm ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * Dm ** 6 / 720)
    lon = (Dm - (1 + 2 * T1 + C1) * Dm ** 3 / 6
           + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2) * Dm ** 5 / 120
           ) / math.cos(p1)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lon0 + lon), math.degrees(lat)


def utm_from_lonlat(epsg, lon, lat):
    epsg = int(epsg)
    zones = _zones()
    zone, hemi = zones[epsg][0], zones[epsg][1]
    a, f = WGS84_A, WGS84_F
    k0, e2 = 0.9996, f * (2 - f)
    ep2 = e2 / (1 - e2)
    lat, lon = math.radians(lat), math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    T = math.tan(lat) ** 2
    C = ep2 * math.cos(lat) ** 2
    A = (lon - lon0) * math.cos(lat)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * lat))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                  + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * math.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                       + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    if hemi < 0:
        y += 10000000.0
    return x, y


def self_test_utm(tol_m=0.5):
    """Round-trip check of the UTM helper (lon/lat -> UTM -> lon/lat).

    This tests *self-consistency* only.  There is no pyproj or PROJ here, so absolute
    agreement with EPSG's own transformation cannot be verified in this VM.  Measured
    round-trip error is ~0.09 m, i.e. 3 % of one 3 m pixel: irrelevant for a 30-300 m
    feature, but the coordinator should regenerate lon/lat with pyproj from x/y when
    assembling the final archive.
    """
    worst = 0.0
    for epsg in (32719, 31979, 32720, 31981):
        zone = _zones()[epsg][0]
        lon0 = 6 * zone - 183          # central meridian of the zone
        # test points inside the zone: the series diverges far outside it, which is
        # a property of the projection, not a defect worth flagging
        for dlon in (-2.5, -1.0, 0.0, 1.5, 2.5):
            lon, lat = lon0 + dlon, -9.8
            x, y = utm_from_lonlat(epsg, lon, lat)
            lon2, lat2 = utm_to_lonlat(epsg, x, y)
            dx = (lon2 - lon) * 111320 * math.cos(math.radians(lat))
            dy = (lat2 - lat) * 110540
            worst = max(worst, math.hypot(dx, dy))
    if worst > tol_m:
        raise AssertionError(f"UTM round-trip error {worst:.3f} m")
    return worst


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def read_geotiff(path, epsg_override=None):
    from PIL import Image
    im = Image.open(path)
    z = np.array(im).astype(np.float32)
    tags = im.tag_v2
    meta = D.make_meta(dx=3.0, x0=0.0, y0=0.0, epsg=0, source=path)
    if 33550 in tags:
        sx, sy = float(tags[33550][0]), float(tags[33550][1])
        meta["dx"], meta["dy"] = sx, sy
    if 33922 in tags:
        t = tags[33922]
        i0, j0, _, X, Y, _ = (float(t[0]), float(t[1]), float(t[2]),
                              float(t[3]), float(t[4]), float(t[5]))
        meta["x0"] = X - i0 * meta["dx"]
        meta["y0"] = Y + j0 * meta["dy"]
    if 34735 in tags:
        keys = list(tags[34735])
        nkeys = int(keys[3])
        for k in range(nkeys):
            kid, loc, cnt, val = keys[4 + 4 * k: 8 + 4 * k]
            if kid in (3072, 2048) and loc == 0:
                meta["epsg"] = int(val)
    if epsg_override:
        meta["epsg"] = int(epsg_override)
    nod = meta.get("nodata", -9999.0)
    z[z == nod] = np.nan
    if np.nanmin(z) < -1e4:
        z[z < -1e4] = np.nan
    return z, meta


def read_las_xyz(path, max_points=None):
    """Minimal uncompressed LAS 1.0-1.4 reader.  Raises for LASzip-compressed files."""
    with open(path, "rb") as fh:
        head = fh.read(375)
        if head[:4] != b"LASF":
            raise ValueError("not a LAS file")
        ver_major, ver_minor = head[24], head[25]
        header_size = int.from_bytes(head[94:96], "little")
        offset = int.from_bytes(head[96:100], "little")
        npoints = int.from_bytes(head[100:108], "little") if ver_minor < 4 else \
            int.from_bytes(head[247:255], "little")
        fmt = int.from_bytes(head[104:105], "little")
        scale = np.frombuffer(head[131:155], "<f8")[:3]
        off = np.frombuffer(head[155:179], "<f8")[:3]
        fh.seek(header_size)
        # point data record format 0-10; only the first 20 bytes are needed (x,y,z)
        reclen = {0: 20, 1: 28, 2: 26, 3: 34, 4: 57, 5: 63, 6: 30, 7: 36, 8: 38,
                  9: 59, 10: 67}.get(fmt)
        if reclen is None:
            raise ValueError(f"unsupported point format {fmt}")
        fh.seek(offset)
        chunk = 200000
        xs, ys, zs = [], [], []
        read = 0
        while read < npoints:
            k = min(chunk, npoints - read)
            buf = fh.read(k * reclen)
            if len(buf) < k * reclen:
                break
            a = np.frombuffer(buf, np.uint8).reshape(k, reclen)
            xyz = a[:, :12].copy().view("<i4")
            xs.append(xyz[:, 0]); ys.append(xyz[:, 1]); zs.append(xyz[:, 2])
            read += k
            if max_points and read >= max_points:
                break
    X = np.concatenate(xs).astype(np.float64) * scale[0] + off[0]
    Y = np.concatenate(ys).astype(np.float64) * scale[1] + off[1]
    Z = np.concatenate(zs).astype(np.float64) * scale[2] + off[2]
    return X, Y, Z


def load_dtm(path, epsg=None):
    low = path.lower()
    if low.endswith(".npy"):
        z, meta = D.load_grid(path)
    elif low.endswith((".tif", ".tiff")):
        z, meta = read_geotiff(path, epsg_override=epsg)
    elif low.endswith(".las"):
        raise ValueError("LAS point cloud supplied: rasterise it first (this module "
                         "detects on a DTM, not on points)")
    elif low.endswith(".laz"):
        raise ValueError("LASzip-compressed LAZ: this VM has no laspy/lazrs/pdal/"
                         "laszip and members have no network. Convert to .las or "
                         "use the processors' 3 m DTM.")
    else:
        raise ValueError(f"unknown raster type: {path}")
    if epsg:
        meta["epsg"] = int(epsg)
    return z, meta


# --------------------------------------------------------------------------
def process_block(z, meta, i0, j0, rows, cols, transect, tag, mask=None, chm=None):
    sub = z[i0:i0 + rows, j0:j0 + cols]
    m = dict(meta)
    m["x0"] = meta["x0"] + j0 * meta["dx"]
    m["y0"] = meta["y0"] - i0 * meta["dy"]
    w = None
    if mask is not None:
        w = np.asarray(mask[i0:i0 + rows, j0:j0 + cols], bool)
        if not w.any():
            return dict(rows=[], diag=[], rejects=[], stats=dict(status="no-data"),
                        funnel={}, mask={}), m, 0.0
    c = None if chm is None else chm[i0:i0 + rows, j0:j0 + cols]
    t0 = time.time()
    res = causeway.detect(sub, m, w=w, chm=c, transect=transect,
                          id_prefix=f"{transect}-{tag}")
    dt = time.time() - t0
    return res, m, dt


def run(path, transect, outdir, block_m=1200.0, overlap_m=120.0, epsg=None,
        chm_path=None, mask=None, chm_array=None):
    if isinstance(path, tuple):
        z, meta = path            # already-loaded (z, meta) pair
    else:
        z, meta = load_dtm(path, epsg=epsg)
    chm = chm_array
    if chm is None and chm_path:
        chm, _ = load_dtm(chm_path)
    out = os.path.join(outdir, transect)
    os.makedirs(os.path.join(out, "evidence"), exist_ok=True)
    w = np.isfinite(z)
    if w.sum() == 0:
        raise SystemExit(f"{path}: no valid cells")
    # report the georeferencing we actually found, this is what the coordinator needs
    corners = [D.index_to_world(0, 0, meta), D.index_to_world(z.shape[0] - 1, 0, meta),
               D.index_to_world(0, z.shape[1] - 1, meta),
               D.index_to_world(z.shape[0] - 1, z.shape[1] - 1, meta)]
    geo = dict(epsg=meta.get("epsg"), x0=meta["x0"], y0=meta["y0"],
               dx=meta["dx"], dy=meta["dy"], nrows=int(z.shape[0]), ncols=int(z.shape[1]),
               extent_x=[float(min(c[0] for c in corners)), float(max(c[0] for c in corners))],
               extent_y=[float(min(c[1] for c in corners)), float(max(c[1] for c in corners))],
               hole_frac=float(1 - w.mean()))
    try:
        geo["lonlat_corners"] = [utm_to_lonlat(meta.get("epsg") or 4326,
                                               float(c[0]), float(c[1])) for c in corners]
        geo["lonlat_status"] = "computed locally by run_transect.utm_to_lonlat"
    except Exception as exc:
        geo["lonlat_corners"] = None
        geo["lonlat_status"] = f"not computed: {exc}"
    blocks = []
    bs = int(round(block_m / meta["dx"]))
    ov = int(round(overlap_m / meta["dx"]))
    r0 = 0
    while r0 < z.shape[0]:
        c0 = 0
        while c0 < z.shape[1]:
            blocks.append((r0, c0, min(bs, z.shape[0] - r0), min(bs, z.shape[1] - c0)))
            if c0 + bs >= z.shape[1]:
                break
            c0 += max(bs - ov, 1)
        if r0 + bs >= z.shape[0]:
            break
        r0 += max(bs - ov, 1)
    all_rows, all_diag, per_block, rejected = [], [], [], []
    t_start = time.time()
    for k, (i0, j0, rows, cols) in enumerate(blocks):
        tag = f"B{k:02d}"
        res, m, dt = process_block(z, meta, i0, j0, rows, cols, transect, tag,
                                   mask=mask, chm=chm)
        per_block.append(dict(block=tag, row0=i0, col0=j0, rows=rows, cols=cols,
                              seconds=round(dt, 2), stats=res["stats"],
                              funnel=res["funnel"]))
        for r in res["rows"]:
            r["block"] = tag
            all_rows.append(r)
        for r in res.get("rejects", []):
            r["block"] = tag
            rejected.append(r)
        all_diag.extend(res["diag"])
    # dedupe across blocks
    kept, dropped = [], 0
    for r in sorted(all_rows, key=lambda q: -q["score"]):
        if any(causeway._same_feature(r, o, causeway.DEFAULTS) for o in kept):
            dropped += 1
        else:
            kept.append(r)
    prof_by_key = {}
    for d_ in all_diag:
        c = d_.get("candidate") or {}
        prof_by_key[(round(c.get("x", -1), 2), round(c.get("y", -1), 2))] = d_.get("profiles")
    for k, r in enumerate(kept, 1):
        r["id"] = f"{transect}-CW{k:03d}"
        r["transect"] = transect
        r["_profile_list"] = prof_by_key.get((round(r["x"], 2), round(r["y"], 2)))
    # write products
    res_like = dict(rows=kept, diag=[d for d in all_diag
                                     if d.get("candidate") and d["candidate"].get("id") in
                                     {q["id"] for q in kept}],
                    stats=dict(n_candidates=len(kept),
                               n_causeway=sum(r["causeway"] for r in kept),
                               blocks=len(blocks), seconds=round(time.time() - t_start, 1)),
                    params=causeway.DEFAULTS)
    det_csv = causeway.write_outputs(res_like, out, transect=transect)
    rows_csv = os.path.join(out, "candidates_rows.csv")
    cand = causeway.to_candidate_rows(
        kept, transect, meta.get("epsg"), lonlat=_lonlat_fn(meta))
    cols = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type", "size_m",
            "relief_m", "canopy_m", "nearest_known_db", "nearest_known_m",
            "literature_checked", "certainty", "reason"]
    with open(rows_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in cand:
            wr.writerow(r)
    rej_csv = os.path.join(out, "rejected_auto.csv")
    with open(rej_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x", "y", "length_m", "height_m", "crest_width_m", "asym_w",
                     "continuity", "reject_gate", "block"])
        for r in rejected:
            wr.writerow([f"{r['x']:.1f}", f"{r['y']:.1f}", f"{r.get('length_m', 0):.0f}",
                         f"{r.get('h_med', r.get('height_m', float('nan'))):.2f}",
                         f"{r.get('crest_width_m', float('nan')):.1f}",
                         f"{r.get('asym_w', float('nan')):.2f}",
                         f"{r.get('continuity', float('nan')):.2f}", r.get("reject"),
                         r.get("block", "")])
    for r in kept:
        write_evidence(out, r, z, meta, chm)
    src = path if isinstance(path, str) else str(meta.get("source", "in-memory"))
    summary = dict(transect=transect, source=src, geo=geo,
                   detector="causeway.py v2 (Radon line-integral + profile verification)",
                   params=causeway.DEFAULTS, blocks=per_block,
                   total_seconds=round(time.time() - t_start, 1),
                   n_candidates=len(kept), n_causeway=sum(r["causeway"] for r in kept),
                   n_raised=sum(not r["causeway"] for r in kept),
                   n_rejected=int(len(rejected)), n_deduped=int(dropped),
                   certainty_note=("all rows are Possible at most; Known/Strong require "
                                   "the coordinator's known-record and literature check"))
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    # update transects.csv (one row per transect)
    tcols = ["transect", "status", "n_blocks", "seconds", "hole_frac", "n_candidates",
             "n_causeway", "n_rejected", "source", "epsg"]
    tc = os.path.join(outdir, "transects.csv")
    existing = []
    if os.path.exists(tc):
        with open(tc) as fh:
            existing = [r for r in csv.DictReader(fh) if r.get("transect") != transect]
    with open(tc, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=tcols, extrasaction="ignore")
        wr.writeheader()
        for r in existing:
            wr.writerow(r)
        wr.writerow(dict(transect=transect, status="done", n_blocks=len(blocks),
                         seconds=summary["total_seconds"], hole_frac=f"{geo['hole_frac']:.4f}",
                         n_candidates=len(kept),
                         n_causeway=summary["n_causeway"], n_rejected=len(rejected),
                         source=src, epsg=meta.get("epsg")))
    print(json.dumps(dict(transect=transect, blocks=len(blocks),
                          seconds=summary["total_seconds"], candidates=len(kept),
                          causeway=summary["n_causeway"], rejected=len(rejected),
                          out=out), indent=1))
    return summary


def _lonlat_fn(meta):
    epsg = meta.get("epsg") or 0
    def fn(x, y):
        try:
            return utm_to_lonlat(epsg, x, y)
        except Exception:
            return "", ""
    return fn


def write_evidence(outdir, r, z, meta, chm=None):
    """Map + profile PNGs and a JSON record for one detection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = os.path.join(outdir, "evidence", r["id"])
    os.makedirs(d, exist_ok=True)
    half = max(250.0, 0.9 * r["length_m"])
    xs = [r["x"] + half * math.cos(math.radians(r["axis_angle_deg"] + 90)),
          r["x"] - half * math.cos(math.radians(r["axis_angle_deg"] + 90))]
    ys = [r["y"] + half * math.sin(math.radians(r["axis_angle_deg"] + 90)),
          r["y"] - half * math.sin(math.radians(r["axis_angle_deg"] + 90))]
    fi0, fj0 = D.world_to_index(np.array([min(xs), max(xs)]),
                                np.array([max(ys), min(ys)]), meta)
    fi0 = np.clip(fi0, 0, z.shape[0] - 1)
    fj0 = np.clip(fj0, 0, z.shape[1] - 1)
    sub = z[int(fi0[0]):int(fi0[1]) + 1, int(fj0[0]):int(fj0[1]) + 1]
    if sub.size:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        m = np.ma.masked_invalid(sub)
        ax[0].imshow(m, cmap="gray", interpolation="nearest")
        ax[0].set_title(f"{r['id']} DTM (m)", fontsize=9)
        ax[0].plot([sub.shape[1] / 2], [sub.shape[0] / 2], "r+")
        ax[1].imshow(m, cmap="terrain", interpolation="nearest")
        ax[1].set_title(f"{r['type']} h={r['height_m']:.2f} m w={r['crest_width_m']:.1f} m",
                        fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "map.png"), dpi=110)
        plt.close(fig)
    # profiles
    profs = r.get("_profile_list")
    if profs:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        sel = profs[:: max(1, len(profs) // 6)]
        for q in sel:
            if not q.get("valid"):
                continue
            s = np.arange(-len(q["prof"]) // 2, len(q["prof"]) // 2) * 1.0
            ax.plot(s[:len(q["prof"])], np.array(q["prof"]) - q["crest"], lw=0.8)
        ax.set_xlabel("distance across the crest (m, crest = 0)")
        ax.set_ylabel("height relative to crest (m)")
        ax.set_title(f"{r['id']} cross-profiles", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "profiles.png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(d, "evidence.json"), "w") as fh:
        json.dump({k: v for k, v in r.items() if not k.startswith("_")}, fh,
                  indent=1, default=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--transect", required=True)
    ap.add_argument("--outdir", default="results/transects")
    ap.add_argument("--block-m", type=float, default=1200.0)
    ap.add_argument("--overlap-m", type=float, default=120.0)
    ap.add_argument("--epsg", type=int, default=None)
    ap.add_argument("--chm", default=None)
    a = ap.parse_args(argv)
    self_test_utm()
    run(a.dtm, a.transect, a.outdir, block_m=a.block_m, overlap_m=a.overlap_m,
        epsg=a.epsg, chm_path=a.chm)


if __name__ == "__main__":
    main()
