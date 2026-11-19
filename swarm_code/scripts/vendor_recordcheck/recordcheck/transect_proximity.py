"""Per-transect proximity to known earthworks (offline, no network).

For every EBA transect footprint it reports the geodesic distance from the
footprint to the nearest known record, which register that record belongs to, and
how many known records lie within 0.5, 2, 5, 10 and 50 km.  This is the
record-context the report and the queue ordering need: a candidate can only be
"Strong" if no known record is within its radius, so that number has to be
computed for every transect that produces a candidate, and it is useful before a
transect is even processed.

Method, with its limits stated:
  * footprint vertices are transformed from the footprint CRS to WGS84 lon/lat
    with pyproj, then all distances are geodesic (Vincenty, this package), so no
    projection scale factor enters an answer;
  * distances are measured to footprint *vertices*, and the nearest site is
    refined with a point-to-segment distance over the neighbouring segments.  The
    EBA footprints are buffered transect lines with vertices a few metres apart
    (the run summary reports the median spacing), so the residual vertex error is
    at most about half that spacing; it is reported, not hidden;
  * candidate sites are prefiltered with a sphere distance from the footprint
    centroid: a site further than ``T + max_vertex_extent`` from the centroid
    cannot be within ``T`` of any vertex.  Selection starts at T = 50 km and
    doubles until the nearest exact distance is inside T, so every reported
    number is exact for the selected sites and every unselected site is provably
    farther than the largest radius requested.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import geodesy as G
from .known_table import load_known_table
from .shapefile import read_shapefile

RADII = (500.0, 2000.0, 5000.0, 10000.0, 50000.0)
MAX_T = 2_000_000.0  # stop growing at 2000 km: no known site anywhere near


def transform_to_lonlat(xy: np.ndarray, epsg: str):
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(CRS.from_user_input(f"EPSG:{str(epsg).strip()}"), "EPSG:4326",
                              always_xy=True)
    lon, lat = tr.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)])


def point_to_segment_m(lon: float, lat: float, a_lon: float, a_lat: float,
                       b_lon: float, b_lat: float) -> float:
    """Distance from a point to a short geodesic segment, via a local tangent plane."""
    lat0 = math.radians(lat)
    mx = 111_320.0 * math.cos(lat0)
    my = 110_574.0
    ax, ay = (a_lon - lon) * mx, (a_lat - lat) * my
    bx, by = (b_lon - lon) * mx, (b_lat - lat) * my
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 0.0:
        c_lon, c_lat = a_lon, a_lat
    else:
        t = max(0.0, min(1.0, -(ax * dx + ay * dy) / seg2))
        c_lon = lon + (ax + t * dx) / mx
        c_lat = lat + (ay + t * dy) / my
    s, _, _ = G.vincenty_inverse(lat, lon, c_lat, c_lon)
    return s


def points_inside_parts(parts_lonlat, points) -> np.ndarray:
    """True where a lon/lat point lies inside any polygon part.

    Uses matplotlib's ray-casting Path (no GDAL).  Each shapefile part is tested
    separately and the results are OR-ed, so a multi-part footprint counts as the
    union of its parts.  Rings are not distinguished from holes -- the EBA
    footprints are buffered transect lines with no holes; if a future shapefile
    has holes, this over-reports "inside", which is the safe direction for the
    nearest-known distance (it can only report 0 where a distance was larger).
    """
    from matplotlib.path import Path as MplPath

    points = np.asarray(points, dtype=float)
    inside = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return inside
    for part in parts_lonlat:
        if len(part) < 3:
            continue
        inside |= MplPath(part).contains_points(points)
    return inside


def _refine(known_lon, known_lat, lonlat, j, d_col, j_local) -> float:
    """Exact distance from site j to the footprint polyline around its nearest vertices.

    ``j`` indexes the known-site arrays, ``j_local`` the column of ``d_col``
    (the vertex x selected-site distance matrix column).
    """
    best = float(d_col[j_local])
    order = np.argsort(np.where(np.isfinite(d_col), d_col, np.inf))[:5]
    for ci in order:
        for seg_end in (ci - 1, ci):
            if 0 <= seg_end and seg_end + 1 < len(lonlat):
                a, b = lonlat[seg_end], lonlat[seg_end + 1]
                s = point_to_segment_m(known_lon[j], known_lat[j], a[0], a[1], b[0], b[1])
                if math.isfinite(s) and s < best:
                    best = s
    return best


def transect_proximity(transects, known, radii=RADII, default_epsg: str = "5880"):
    kdf = known.df[known.df["valid_coord"]].reset_index(drop=True)
    k_lon = kdf["longitude"].to_numpy(dtype=float)
    k_lat = kdf["latitude"].to_numpy(dtype=float)
    k_raw = kdf["database_raw"].tolist()
    k_fam = kdf["source_family"].tolist()
    k_id = kdf["record_id"].to_numpy(dtype=int)
    trees = np.array([f == "trees_inpe" for f in k_fam], dtype=bool)
    r_max = max(radii)

    rows, spacings = [], []
    for t in transects:
        xy = np.asarray(t["xy"], dtype=float)
        epsg = t.get("epsg") or default_epsg
        parts_ll, lonlat = [], np.zeros((0, 2))
        for part in (t.get("parts_xy") or [t["xy"]]):
            ll = transform_to_lonlat(np.asarray(part, dtype=float), epsg)
            ll = ll[np.isfinite(ll).all(axis=1)]
            if len(ll) >= 3:
                parts_ll.append(ll)
            if len(ll):
                lonlat = np.vstack([lonlat, ll]) if len(lonlat) else ll
        if len(lonlat) < 2:
            rows.append({"transect": t["transect"], "status": "no_valid_vertices"})
            continue

        d_edge = G.haversine(lonlat[:-1, 1], lonlat[:-1, 0], lonlat[1:, 1], lonlat[1:, 0])
        d_edge = d_edge[np.isfinite(d_edge)]
        spacing = float(np.median(d_edge)) if d_edge.size else float("nan")
        spacings.append(spacing)

        c_lon, c_lat = float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean())
        ext = float(np.max(G.haversine(c_lat, c_lon, lonlat[:, 1], lonlat[:, 0])))
        dc = G.haversine(c_lat, c_lon, k_lat, k_lon)
        nearest_by_centroid = np.argsort(dc)[:6]

        T = r_max
        sel = d = min_site = None
        while True:
            sel = np.where(dc <= T + ext + 1.0)[0]
            sel = np.union1d(sel, nearest_by_centroid).astype(int)
            d = G.pairwise_distance(lonlat[:, 1], lonlat[:, 0], k_lat[sel], k_lon[sel], chunk=512)
            with np.errstate(invalid="ignore"):
                min_site = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=0)
            if not np.isfinite(min_site).any():
                break
            best = float(np.min(min_site))
            if best <= T or T >= MAX_T:
                break
            T *= 2.0

        if sel is None or min_site is None or not np.isfinite(min_site).any():
            rows.append({"transect": t["transect"], "status": "no_distance"})
            continue

        # a record inside the footprint is at distance 0 from the transect; the
        # boundary-vertex distance would otherwise overstate it (159 m for the
        # 0414 control record, which lies inside the swath)
        inside = np.zeros(len(sel), dtype=bool)
        near = np.where(np.isfinite(min_site) & (min_site <= 500.0))[0]
        if near.size:
            inside[near] = points_inside_parts(parts_ll, np.column_stack([k_lon[sel[near]], k_lat[sel[near]]]))
        d_eff = np.where(inside, 0.0, min_site)

        j_local = int(np.argmin(np.where(np.isfinite(d_eff), d_eff, np.inf)))
        best = float(min_site[j_local])
        j = int(sel[j_local])
        exact = 0.0 if inside[j_local] else _refine(k_lon, k_lat, lonlat, j, d[:, j_local], j_local)
        # a site whose vertex distance is within 50 m of the best could still win
        # after segment refinement
        for jl in np.where(np.isfinite(min_site) & (min_site <= best + 50.0) & ~inside)[0]:
            jj = int(sel[jl])
            cand = _refine(k_lon, k_lat, lonlat, jj, d[:, jl], jl)
            if cand < exact:
                exact, j, j_local = cand, jj, int(jl)

        exact_in = float(d_eff[j_local]) if inside[j_local] else exact
        at = t.get("attrs") or {}
        row = {
            "transect": t["transect"],
            "status": "ok",
            "datafile": at.get("datafile"),
            "criteria": at.get("criteria"),
            "canopy_rel": at.get("canopy_rel"),
            "footprint_epsg": str(epsg),
            "data_epsg": t.get("data_epsg"),
            "n_vertices": int(len(lonlat)),
            "vertex_spacing_m": round(spacing, 2),
            "nearest_known_m": round(exact_in, 1),
            "nearest_known_inside": bool(inside[j_local]),
            "nearest_known_db": k_raw[j],
            "nearest_known_family": k_fam[j],
            "nearest_known_record_id": int(k_id[j]),
            "lon_min": round(float(lonlat[:, 0].min()), 7),
            "lon_max": round(float(lonlat[:, 0].max()), 7),
            "lat_min": round(float(lonlat[:, 1].min()), 7),
            "lat_max": round(float(lonlat[:, 1].max()), 7),
            "centroid_lon": round(c_lon, 7),
            "centroid_lat": round(c_lat, 7),
        }
        fam_trees = trees[sel]
        for mask, name in ((fam_trees, "nearest_trees_inpe_m"), (~fam_trees, "nearest_other_m")):
            if not mask.any():
                row[name] = None
                continue
            # refine the family winner the same way as the overall winner, so
            # every reported distance uses one method
            local = int(np.argmin(np.where(mask, d_eff, np.inf)))
            row[name] = (0.0 if inside[local]
                         else round(_refine(k_lon, k_lat, lonlat, int(sel[local]), d[:, local], local), 1))
        for r in radii:
            row[f"n_known_{int(r)}m"] = int((d_eff <= r).sum())
        row["n_known_inside"] = int(inside.sum())
        row["selection_radius_m"] = T
        rows.append(row)

    df = pd.DataFrame(rows)
    ok = df["status"] == "ok"
    summary = {
        "n_transects": int(len(df)),
        "n_ok": int(ok.sum()),
        "median_vertex_spacing_m": round(float(np.median(spacings)), 2) if spacings else None,
        "radii_m": list(radii),
        "known_records": int(len(kdf)),
        "distance_method": "geodesic to footprint vertices, nearest refined by point-to-segment; "
                           "unselected sites are provably farther than the largest radius",
    }
    for col in ("criteria",):
        if col in df:
            summary[f"n_by_{col}"] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).items()}
    for r in radii:
        col = f"n_known_{int(r)}m"
        if col in df:
            summary[f"n_transects_with_known_within_{int(r)}m"] = int((df[col].fillna(0) > 0).sum())
    if "nearest_known_inside" in df:
        summary["n_transects_with_known_inside_footprint"] = int(df["nearest_known_inside"].fillna(False).sum())
    if "n_known_inside" in df:
        summary["n_known_records_inside_footprints"] = int(df["n_known_inside"].fillna(0).sum())
    return df, summary


def read_prj(shp_path: str | Path) -> tuple[str | None, str]:
    """Return (EPSG code or None, WKT) for the shapefile's own CRS from its .prj.

    The DBF ``epsg`` field is NOT the footprint CRS: in the EBA inventory it holds
    the CRS of each transect's LAZ tile (e.g. EPSG:31979, UTM 19S), while the
    footprint geometry itself is EPSG:5880 Brazil Polyconic.  Using the DBF value
    for the geometry put every footprint at the wrong place -- caught by the
    known-site proximity result for transect 0414, which came out 1,200 km from
    any record.  The footprint CRS therefore comes from the .prj only.
    """
    from pyproj import CRS

    shp_path = Path(shp_path)
    candidates = [shp_path.with_suffix(s) for s in (".prj", ".PRJ")]
    prj = next((p for p in candidates if p.exists()), None)
    if prj is None:
        raise FileNotFoundError(f"no .prj next to {shp_path}: refusing to guess the footprint CRS")
    wkt = prj.read_text(encoding="utf-8", errors="replace").strip()
    crs = CRS.from_wkt(wkt)
    epsg = crs.to_epsg()
    return (str(epsg) if epsg else None), wkt


def load_transects(shp_path: str | Path, dbf_path: str | Path) -> tuple[list[dict], dict]:
    records, attrs, header = read_shapefile(shp_path, dbf_path)
    epsg, wkt = read_prj(shp_path)
    if epsg is None:
        raise ValueError(f"{shp_path}.prj has no EPSG code; refusing to proceed without an identified CRS")
    out = []
    for rec, at in zip(records, attrs):
        if not rec.parts:
            continue
        xy = rec.all_points()
        tname = at.get("transect") or at.get("id") or str(rec.number)
        data_epsg = at.get("epsg")
        if isinstance(data_epsg, float) and data_epsg.is_integer():
            data_epsg = str(int(data_epsg))
        out.append({"transect": str(tname).strip(), "epsg": epsg, "data_epsg": data_epsg or None,
                    "xy": xy, "parts_xy": list(rec.parts), "attrs": at})
    info = {"footprint_epsg": epsg, "footprint_wkt": wkt,
            "n_data_epsg_differs": int(sum(1 for t in out if str(t["data_epsg"]) != str(epsg))),
            "data_epsg_values": sorted({str(t["data_epsg"]) for t in out})}
    return out, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="per-transect distance to the nearest known earthwork")
    ap.add_argument("--shp", required=True)
    ap.add_argument("--dbf", required=True)
    ap.add_argument("--known", default=None)
    ap.add_argument("--known-csv", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "pyreadr", "R", "pure"])
    ap.add_argument("--out-csv", default="out/transect_known_proximity.csv")
    ap.add_argument("--out-json", default="out/transect_known_proximity_summary.json")
    ap.add_argument("--transect-ids-json", default=None,
                    help="optional id list (e.g. the coordinator's transect_ids.json) to cross-check")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N shapes")
    args = ap.parse_args(argv)

    if args.known_csv:
        from .audit import load_known_table_from_csv

        known = load_known_table_from_csv(args.known_csv)
    else:
        known = load_known_table(args.known, backend=args.backend)
    transects, crs_info = load_transects(args.shp, args.dbf)
    if args.limit:
        transects = transects[: args.limit]
    print(f"parsed {len(transects)} transect shapes; known records: {len(known.df)}; "
          f"footprint CRS EPSG:{crs_info['footprint_epsg']}; "
          f"{crs_info['n_data_epsg_differs']} rows have a per-tile CRS different from the footprint CRS")
    df, summary = transect_proximity(transects, known)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    if args.transect_ids_json:
        ref = json.loads(Path(args.transect_ids_json).read_text(encoding="utf-8"))
        got = [t["transect"] for t in transects]
        summary["id_cross_check"] = {
            "reference_file": args.transect_ids_json,
            "reference_count": len(ref),
            "parsed_count": len(got),
            "identical_as_multisets": sorted(ref) == sorted(got),
            "only_in_reference": sorted(set(ref) - set(got))[:20],
            "only_in_parsed": sorted(set(got) - set(ref))[:20],
        }
        print("id cross-check:", {k: summary["id_cross_check"][k] for k in
                                  ("reference_count", "parsed_count", "identical_as_multisets")})
    summary["crs"] = {k: v for k, v in crs_info.items() if k != "footprint_wkt"}
    summary["known_table"] = known.meta()
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    keys = [k for k in summary if k.startswith("n_transects_with_known")] + ["median_vertex_spacing_m"]
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    print("wrote", args.out_csv, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
