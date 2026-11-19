"""Review detector output: measure, judge, and write an evidence packet.

    python3 -m skeptic.review --products data/derived/0414/products.npz \
        --detections data/derived/0414/detections/rings_relief15_recall.csv \
        --out out/0414_review --transect 0414 --state AC

Inputs
------
products.npz   needs at least `dtm` (+ origin metadata: x0, y0, cell/res,
               epsg, row_order).  Optional: `chm`, `hillshade`, `svf`,
               `relief10/15/30`.
detections     CSV or JSON with, per row/dict:
                   id, x, y                 (map coordinates, mandatory)
                   radius_m | diameter_m | size_m   (any one; optional)
                   type                     (optional: ring/square/U/line)
                   
Outputs (all under --out)
-------------------------
bundles/<id>.json      full measurement bundle + verdict + every check
candidates/<id>/       skeptic_notes.json, notes.md, map_panels.png, profiles.png
skeptic_review.csv     one row per reviewed feature (all numbers + reason)
candidates_schema.csv  rows in the project's candidates.csv column order, with
                       record-check and literature fields left as pending

Nothing here writes outside --out, and nothing here talks to the network.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np

from .grid import Grid, load_grid
from . import analysis, geo, primitives as P, rules


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

def _origin_sibling(path):
    """Find georeferencing for a products file that carries none.

    The 0414/0072 products.npz written by the processor hold *only* 2-D arrays:
    no x0, y0, cell, epsg or row_order.  The georeferencing lives in the sibling
    dtm.npz (or grid.json).  Consumers that assume otherwise silently lose the
    origin, so this is looked up explicitly and the shape is checked.
    """
    d = os.path.dirname(os.path.abspath(path))
    for name in ("dtm.npz", "grid.json", "processing.json"):
        cand = os.path.join(d, name)
        if not os.path.exists(cand):
            continue
        try:
            if name.endswith(".npz"):
                g = load_grid(cand)
                return dict(res=g.res, x0=g.x0, y0=g.y0,
                            y_ascending=g.y_ascending, epsg=g.epsg, source=cand)
            with open(cand) as f:
                j = json.load(f)
            if not {"x0", "y0"} <= set(j):
                continue
            yo = j.get("row_order")
            return dict(res=float(j.get("cell") or j.get("res")),
                        x0=float(j["x0"]), y0=float(j["y0"]),
                        y_ascending=(str(yo).lower().startswith("south")
                                     if yo else False),
                        epsg=(int(j["epsg"]) if j.get("epsg") else None),
                        source=cand)
        except Exception:
            continue
    return None


def load_products(path):
    """Load dtm (+ chm and any precomputed display layers) from a products npz."""
    try:
        dtm = load_grid(path, name=os.path.basename(path))
        origin_note = "from products file"
    except (KeyError, ValueError):
        meta = _origin_sibling(path)
        if meta is None:
            raise
        with np.load(path, allow_pickle=False) as f:
            for k in ("dtm", "z", "elev", "dem"):
                if k in f.files:
                    z = np.asarray(f[k], dtype=np.float64)
                    break
            else:
                raise KeyError(f"{path}: no elevation array")
        dtm = Grid(z, meta["res"], meta["x0"], meta["y0"],
                   y_ascending=meta["y_ascending"], epsg=meta["epsg"],
                   name=os.path.basename(path))
        origin_note = f"from sibling {os.path.basename(meta['source'])}"
    layers = {}
    chm = None
    with np.load(path, allow_pickle=False) as f:
        for k in f.files:
            a = f[k]
            if not hasattr(a, "shape") or a.ndim != 2 or a.shape != dtm.shape:
                continue
            if k == "chm" or (chm is None and k in ("canopy", "chm_m")):
                chm = Grid(np.asarray(a, dtype=np.float64), dtm.res, dtm.x0, dtm.y0,
                           y_ascending=dtm.y_ascending, epsg=dtm.epsg,
                           name=dtm.name + "/chm")
            elif k in ("hillshade", "svf", "relief10", "relief15", "relief30",
                       "count", "original_hole", "unfilled", "relief_hole_mask"):
                layers[k] = np.asarray(a, dtype=np.float64)
    return dtm, chm, layers, origin_note


_DET_KEYS_X = ("x", "cx", "easting", "x_map", "utm_x")
_DET_KEYS_Y = ("y", "cy", "northing", "y_map", "utm_y")
_DET_KEYS_SIZE = ("diameter_m", "size_m", "outer_diameter_m", "diameter")


def _pick(row, keys):
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return None


def load_detections(path):
    rows = []
    if str(path).endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("detections") or data.get("candidates") or data.get("features") or []
        rows = list(data)
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        x = _pick(r, _DET_KEYS_X)
        y = _pick(r, _DET_KEYS_Y)
        if x is None or y is None:
            continue
        size = _pick(r, _DET_KEYS_SIZE)
        if size is None:
            rad = _pick(r, ("radius_m", "r_m", "radius"))
            size = 2.0 * rad if rad is not None else None
        out.append(dict(id=str(r.get("id") or r.get("fid") or f"det{i:04d}"),
                        x=x, y=y, size_m=size,
                        type=str(r.get("type") or r.get("shape") or "ring"),
                        detector=str(r.get("detector") or r.get("source") or ""),
                        row=r))
    return out


# ---------------------------------------------------------------------------
# evidence packet
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_plots(bundle, dtm, chm, layers, outdir, prefix=""):
    """map_panels.png (dtm / relief / hillshade / chm) and profiles.png."""
    plt = _plt()
    g = bundle.get("geometry") or {}
    if g.get("status") != "ok":
        return []
    cx, cy, R = g["cx"], g["cy"], g["radius_m"]
    w = max(1.0, 0.5 * (g.get("dip_width_m") or 6.0))
    half = 1.9 * R + 25.0
    made = []

    # ---- map panels
    panels = [("DTM (m)", dtm.z, "terrain"),
              ("local relief 15 m (m)", P.local_relief(dtm, 15.0).z, "div")]
    if "hillshade" in layers:
        panels.append(("hillshade", layers["hillshade"], "gray"))
    else:
        panels.append(("hillshade", P.hillshade(dtm), "gray"))
    if chm is not None:
        panels.append(("canopy height (m)", chm.z, "viridis"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False)
    for ax, (title, arr, kind) in zip(axes[0], panels):
        sub = dtm.subgrid(cx, cy, half)
        if sub is None:
            continue
        si, sj = dtm.window(cx, cy, half)
        a = arr[si, sj].astype(float)
        if kind == "div":
            v = 1.0 if not np.isfinite(a).any() else max(
                0.2, float(np.nanpercentile(np.abs(a), 97)))
            im = ax.imshow(a, cmap="RdBu_r", vmin=-v, vmax=v)
        elif kind == "gray":
            im = ax.imshow(a, cmap="gray")
        else:
            im = ax.imshow(a, cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        th = np.linspace(0, 2 * np.pi, 361)
        shape = (g.get("planform") or {}).get("shape")
        ax.plot((cx + R * np.cos(th) - sub.x0) / dtm.res - 0.5,
                (cy + R * np.sin(th) - sub.y0) / dtm.res * (1 if dtm.y_ascending else -1) - 0.5,
                "-", color="yellow", lw=0.9, alpha=0.9)
        ax.plot((cx - sub.x0) / dtm.res - 0.5,
                (cy - sub.y0) / dtm.res * (1 if dtm.y_ascending else -1) - 0.5,
                "+", color="yellow", ms=8)
        ax.set_title(f"{title}\nR={R:.0f} m  {shape or ''}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{bundle.get('id','')}  {bundle.get('transect','')}  "
                 f"({cx:.0f}, {cy:.0f})  verdict "
                 f"{(bundle.get('verdict') or {}).get('certainty','?')} "
                 f"grade {(bundle.get('verdict') or {}).get('geometric_grade','?')}",
                 fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(outdir, f"{prefix}map_panels.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    made.append(p1)

    # ---- profiles
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), squeeze=False)
    rp = g.get("radial_profile") or {}
    ax = axes[0][0]
    if rp:
        r = np.asarray(rp["r"], float)
        pf = np.asarray([np.nan if v is None else v for v in rp["prof"]], float)
        bg = np.asarray([np.nan if v is None else v for v in rp["background"]], float)
        ax.plot(r, pf, "k-", lw=1.1, label="azimuthal median DTM")
        ax.plot(r, bg, "b--", lw=1.0, label="local trend")
        band = g.get("edge", {})
        ax.axvspan(band.get("band_lo") or 0, band.get("band_hi") or 0,
                   color="orange", alpha=0.12, label="search band")
        ax.axvline(R, color="r", lw=0.8, ls=":", label=f"dip R={R:.1f} m")
        ax.legend(fontsize=7)
    ax.set_xlabel("radius (m)"); ax.set_ylabel("elevation (m)")
    ax.set_title("radial profile", fontsize=9)

    ax = axes[0][1]
    ep = g.get("edge_points") or {}
    if ep:
        az = np.deg2rad(np.asarray(ep["az"], float))
        rr = np.asarray(ep["r"], float)
        amp = np.asarray(ep["amp"], float)
        sc = ax.scatter(az, rr, c=amp, s=8, cmap="viridis")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="dip depth (m)")
        ax.axhline(R, color="r", lw=0.8, ls=":")
        pc = g.get("polar_closure") or {}
        ax.set_title(f"dip radius by azimuth (closure "
                     f"{'closed' if pc.get('closed') else 'OPEN'}, "
                     f"max gap {pc.get('max_gap_arc_m')} m)", fontsize=8)
    ax.set_xlabel("azimuth (rad, 0=+x)"); ax.set_ylabel("dip radius (m)")

    ax = axes[0][2]
    cs = g.get("cross") or {}
    if ep:
        az = np.asarray(ep["az"], float)
        rr = np.asarray(ep["r"], float)
        for k in np.linspace(0, len(az) - 1, min(8, len(az))).astype(int):
            th = np.deg2rad(az[k])
            wl = np.linspace(rr[k] - 2.4 * w, rr[k] + 2.4 * w, 41)
            zz = dtm.sample(cx + wl * np.cos(th), cy + wl * np.sin(th))
            bgv = np.interp(wl, np.asarray(rp["r"], float),
                            np.asarray([np.nan if v is None else v
                                        for v in rp["background"]], float))
            ax.plot(wl - rr[k], zz - bgv, lw=0.9, alpha=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"cross-sections: two-sided "
                     f"{100*(cs.get('two_sided_frac') or 0):.0f}%, "
                     f"one-sided step {100*(cs.get('step_frac') or 0):.0f}%", fontsize=8)
    ax.set_xlabel("distance from ditch centre (m)"); ax.set_ylabel("relative height (m)")
    fig.tight_layout()
    p2 = os.path.join(outdir, f"{prefix}profiles.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    made.append(p2)
    return made


def write_layer_pngs(bundle, dtm, chm, layers, outdir):
    """One PNG per layer, cropped to the feature, for the deliverable's
    candidates/<id>/ directory: dtm, hillshade, relief15, svf, canopy."""
    plt = _plt()
    g = bundle.get("geometry") or {}
    if g.get("status") != "ok":
        return []
    cx, cy, R = g["cx"], g["cy"], g["radius_m"]
    half = 1.9 * R + 25.0
    sl = dtm.window(cx, cy, half)
    if sl is None:
        return []
    si, sj = sl
    todo = [("dtm", dtm.z, "viridis")]
    todo.append(("hillshade", layers.get("hillshade", P.hillshade(dtm)), "gray"))
    todo.append(("relief15", layers.get("relief15", P.local_relief(dtm, 15.0).z), "div"))
    if "svf" in layers:
        todo.append(("svf", layers["svf"], "magma"))
    if chm is not None:
        todo.append(("canopy", chm.z, "viridis"))
    made = []
    th = np.linspace(0, 2 * np.pi, 361)
    for name, arr, kind in todo:
        a = np.asarray(arr)[si, sj].astype(float)
        fig, ax = plt.subplots(figsize=(5.2, 5.2 * a.shape[0] / max(a.shape[1], 1)))
        if kind == "div":
            v = max(0.2, float(np.nanpercentile(np.abs(a), 97)))
            ax.imshow(a, cmap="RdBu_r", vmin=-v, vmax=v)
        elif kind == "gray":
            ax.imshow(a, cmap="gray")
        elif kind == "magma":
            ax.imshow(a, cmap="magma")
        else:
            ax.imshow(a, cmap="viridis")
        # fitted circle in pixel coordinates of the crop
        xs, ys = dtm.cell_center(si.start, sj.start)
        px = (cx + R * np.cos(th) - (dtm.x0 + si.start * 0)) / dtm.res
        px = (cx + R * np.cos(th) - (dtm.x0 + sj.start * dtm.res)) / dtm.res
        py = (cy + R * np.sin(th) - ys) / dtm.res * (1 if dtm.y_ascending else -1)
        ax.plot(px, py, "-", color="yellow", lw=0.8, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{bundle.get('id','')} {name}  R={R:.0f} m", fontsize=8)
        fig.tight_layout()
        pth = os.path.join(outdir, f"{name}.png")
        fig.savefig(pth, dpi=100)
        plt.close(fig)
        made.append(pth)
    return made


def write_notes(bundle, outdir, prefix=""):
    v = bundle.get("verdict") or {}
    g = bundle.get("geometry") or {}
    md = [f"# Skeptic review: {bundle.get('id','')}", ""]
    md.append(f"* transect: {bundle.get('transect','')}   state: {bundle.get('state','')}")
    md.append(f"* reported centre: ({bundle['centre_in']['x']:.1f}, "
              f"{bundle['centre_in']['y']:.1f})  size hint: {bundle.get('size_hint_m')}")
    if g.get("status") == "ok":
        md.append(f"* measured centre after search: ({g['cx']:.1f}, {g['cy']:.1f}) "
                  f"(shift {g['centre_refine']['offset_m']:.1f} m)")
        md.append(f"* diameter {g['outer_diameter_m']:.0f} m, ditch "
                  f"{g['dip_width_m']:.1f} m wide, {g['dip_amp_m']:.2f} m deep, "
                  f"planform {g['planform']['shape']} "
                  f"({100*g['planform']['resid_rel']:.1f}% radius residual)")
    md += ["", f"## Decision: **{v.get('certainty','?')}**"
              f" (geometric grade {v.get('geometric_grade','?')}, "
              f"strong-eligible {v.get('strong_eligible')})", "", v.get("reason", ""), ""]
    md.append("## Checks")
    md.append("| check | status | value | threshold | note |")
    md.append("|---|---|---|---|---|")
    for c in bundle.get("checks", []):
        val = c.get("value")
        val = f"{val}" if not isinstance(val, (list, tuple)) else \
            "(" + ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in val) + ")"
        th = c.get("threshold")
        th = f"{th}" if not isinstance(th, (list, tuple)) else \
            "(" + ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in th) + ")"
        md.append(f"| {c['id']} | {c['status']} | {val[:60]} | {th[:40]} | "
                  f"{c['text'].replace('|', '/')} |")
    md.append("")
    md.append("## Scope of this verdict")
    md.append("m6 is offline: this verdict covers *natural, modern and data-artefact* "
              "explanations only. It is never Known and never Strong; Known needs a "
              "match in the known-site table or the literature, and Strong needs the "
              "2 km record check as well. `strong_eligible` only says that nothing "
              "geometric is left unexplained.")
    p = os.path.join(outdir, f"{prefix}notes.md")
    with open(p, "w") as f:
        f.write("\n".join(md) + "\n")
    return p


# ---------------------------------------------------------------------------
# driving
# ---------------------------------------------------------------------------

def review_one(dtm, chm, layers, det, transect, state, params=None, png=True):
    b = analysis.sceptic_bundle(dtm, det["x"], det["y"], size_hint=det.get("size_m"),
                                kind=det.get("type") or "ring", chm=chm,
                                fid=det["id"], transect=transect, state=state,
                                detector=det.get("detector", ""), params=params)
    b, v = rules.judge_bundle(b)
    return b, v


def run(products, detections, outdir, transect="", state="", top=None,
        png=True, quiet=False, params=None):
    dtm, chm, layers, origin_note = load_products(products)
    if not quiet:
        print(f"products: {origin_note}; DTM {dtm}")
    dets = load_detections(detections)
    if top:
        dets = dets[:int(top)]
    os.makedirs(os.path.join(outdir, "bundles"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "candidates"), exist_ok=True)
    rows = []
    zone, north = geo.epsg_to_utm(dtm.epsg) if dtm.epsg else (None, None)
    t0 = __import__("time").time()
    for det in dets:
        try:
            b, v = review_one(dtm, chm, layers, det, transect, state, params=params)
        except Exception:
            traceback.print_exc()
            rows.append(dict(id=det["id"], x=det["x"], y=det["y"], error="exception"))
            continue
        g = b.get("geometry") or {}
        with open(os.path.join(outdir, "bundles", f"{det['id']}.json"), "w") as f:
            json.dump(b, f, indent=1)
        lon = lat = None
        if zone:
            lon, lat = geo.utm_inverse(g.get("cx", det["x"]),
                                      g.get("cy", det["y"]), zone, north)
        row = dict(id=det["id"], transect=transect, state=state,
                   x_in=det["x"], y_in=det["y"], size_in=det.get("size_m"),
                   x=(g.get("cx") if g.get("status") == "ok" else None),
                   y=(g.get("cy") if g.get("status") == "ok" else None),
                   epsg=dtm.epsg, lon=lon, lat=lat,
                   status=g.get("status"),
                   certainty=v["certainty"], grade=v["geometric_grade"],
                   strong_eligible=v["strong_eligible"],
                   label=v["primary_label"],
                   radius_m=g.get("radius_m"), outer_diameter_m=g.get("outer_diameter_m"),
                   dip_amp_m=g.get("dip_amp_m"), dip_width_m=g.get("dip_width_m"),
                   closure=((g.get("polar_closure") or {}).get("closed")),
                   coverage=(g.get("edge") or {}).get("coverage"),
                   shape=((g.get("planform") or {}).get("shape")),
                   planform_resid=((g.get("planform") or {}).get("resid_rel")),
                   two_sided=((g.get("cross") or {}).get("two_sided_frac")),
                   step_frac=((g.get("cross") or {}).get("step_frac")),
                   interior_p75=((g.get("level") or {}).get("interior_p75_m")),
                   nodata_ditch=((g.get("support") or {}).get("nodata_frac_ditch_band")),
                   centre_shift_m=((g.get("centre_refine") or {}).get("offset_m")),
                   canopy_m=((g.get("canopy") or {}).get("canopy_median_m")),
                   warnings=",".join(c["id"] for c in b["checks"] if c["status"] == "warn"),
                   reason=v["reason"][:900])
        rows.append(row)
        ev = os.path.join(outdir, "candidates", det["id"])
        os.makedirs(ev, exist_ok=True)
        notes = dict(id=det["id"], status="skeptic_reviewed",
                     natural_modern_artefact_explanations=v["blocking"],
                     decision=v["certainty"], grade=v["geometric_grade"],
                     strong_eligible=v["strong_eligible"],
                     reason=v["reason"],
                     checks={c["id"]: c["status"] for c in b["checks"]},
                     measurements=dict(radius_m=g.get("radius_m"),
                                       dip_amp_m=g.get("dip_amp_m"),
                                       dip_width_m=g.get("dip_width_m"),
                                       coverage=(g.get("edge") or {}).get("coverage"),
                                       planform=g.get("planform"),
                                       cross=g.get("cross"), level=g.get("level"),
                                       support=g.get("support"),
                                       canopy=g.get("canopy"),
                                       polar_closure=g.get("polar_closure")),
                     nearest_known_db="pending:m7",
                     nearest_known_m="pending:m7",
                     table_match="pending:m7",
                     literature_checked="pending:coordinator")
        with open(os.path.join(ev, "skeptic_notes.json"), "w") as f:
            json.dump(notes, f, indent=1)
        write_notes(b, ev)
        if png:
            try:
                write_plots(b, dtm, chm, layers, ev)
                write_layer_pngs(b, dtm, chm, layers, ev)
            except Exception:
                traceback.print_exc()
        if not quiet:
            print(f"{det['id']:22s} {row['certainty']:8s} {str(row['grade']):4s} "
                  f"{str(row['label']):16s} R={row['radius_m']} "
                  f"dip={row['dip_amp_m']} w={row['dip_width_m']}", flush=True)
    _write_csv(rows, os.path.join(outdir, "skeptic_review.csv"))
    with open(os.path.join(outdir, "rejected.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "transect", "EPSG", "x", "y",
                                          "certainty", "reason_family", "reason"])
        w.writeheader()
        for r in rows:
            if r.get("certainty") == "Rejected":
                w.writerow(dict(id=r["id"], transect=r["transect"], EPSG=r["epsg"],
                                x=(f"{r['x']:.1f}" if r["x"] else ""),
                                y=(f"{r['y']:.1f}" if r["y"] else ""),
                                certainty="Rejected",
                                reason_family=r["label"],
                                reason=r["reason"].replace("\n", " ")))
    _write_candidates_csv(rows, os.path.join(outdir, "candidates_schema.csv"))
    summ = _summary(rows, __import__("time").time() - t0)
    with open(os.path.join(outdir, "review_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    if not quiet:
        print(json.dumps(summ, indent=1))
    return dict(rows=rows, summary=summ)


def _write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


CAND_COLS = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type",
             "size_m", "relief_m", "canopy_m", "nearest_known_db", "nearest_known_m",
             "literature_checked", "certainty", "reason"]


def _write_candidates_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(dict(
                id=r["id"], state=r["state"], transect=r["transect"], EPSG=r["epsg"],
                x=(f"{r['x']:.1f}" if r["x"] is not None else ""),
                y=(f"{r['y']:.1f}" if r["y"] is not None else ""),
                lon=(f"{r['lon']:.8f}" if r["lon"] is not None else ""),
                lat=(f"{r['lat']:.8f}" if r["lat"] is not None else ""),
                type="ring", size_m=(f"{r['outer_diameter_m']:.0f}"
                                     if r["outer_diameter_m"] else ""),
                relief_m=(f"{r['dip_amp_m']:.2f}" if r["dip_amp_m"] else ""),
                canopy_m=(f"{r['canopy_m']:.1f}" if r["canopy_m"] else ""),
                nearest_known_db="pending:m7", nearest_known_m="pending:m7",
                literature_checked="pending:coordinator",
                certainty=r["certainty"], reason=r["reason"].replace("\n", " ")))


def _summary(rows, elapsed):
    def cnt(pred):
        return sum(1 for r in rows if pred(r))
    return dict(n=len(rows), elapsed_s=round(elapsed, 1),
                rejected=cnt(lambda r: r.get("certainty") == "Rejected"),
                possible=cnt(lambda r: r.get("certainty") == "Possible"),
                strong_eligible=cnt(lambda r: r.get("strong_eligible") is True),
                grades={g: cnt(lambda r, g=g: r.get("grade") == g)
                        for g in ("A", "B", "C", "D", "none")},
                labels={lab: cnt(lambda r, lab=lab: r.get("label") == lab)
                        for lab in sorted({r.get("label") for r in rows} - {None})},
                possible_ids=[r["id"] for r in rows if r.get("certainty") == "Possible"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--products", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transect", default="")
    ap.add_argument("--state", default="")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    run(a.products, a.detections, a.out, a.transect, a.state, a.top,
        png=not a.no_png, quiet=a.quiet)


if __name__ == "__main__":
    main()
