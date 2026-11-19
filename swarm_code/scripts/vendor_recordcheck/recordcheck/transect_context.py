"""Known-record context for one transect (default: the 0414 positive control).

Lists every known-earthwork record within a radius of a transect footprint, with
geodesic distances, so that:

  * a detection inside the 0414 control area can be labelled Known against the
    exact record that makes it Known, and
  * the detectors/skeptics can see which parts of a transect are already covered
    by records and where a candidate would have to sit to be more than 2 km from
    anything known.

Writes JSON plus a short Markdown table.  Local files only; no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geodesy as G
from .known_table import load_known_table
from .transect_proximity import load_transects, transform_to_lonlat


def transect_context(transect_id: str, transports, known, radius_m: float = 2000.0) -> dict:
    t = next((x for x in transports if str(x["transect"]) == str(transect_id)), None)
    if t is None:
        raise KeyError(f"transect {transect_id!r} not found in the footprint file")
    lonlat = transform_to_lonlat(t["xy"], t["epsg"])
    lonlat = lonlat[np.isfinite(lonlat).all(axis=1)]

    kdf = known.df[known.df["valid_coord"]].reset_index(drop=True)
    d = G.pairwise_distance(lonlat[:, 1], lonlat[:, 0],
                            kdf["latitude"].to_numpy(float), kdf["longitude"].to_numpy(float),
                            chunk=512)
    with np.errstate(invalid="ignore"):
        min_site = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=0)
        near_vertex = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=0)

    c_lon, c_lat = float(lonlat[:, 0].mean()), float(lonlat[:, 1].mean())
    sel = np.where(min_site <= radius_m)[0]
    order = sel[np.argsort(min_site[sel])]
    records = []
    for j in order:
        v = kdf.iloc[int(j)]
        # bearing from the footprint centroid to the record, so a reader can see
        # which side of the transect an already-recorded complex lies on
        _, az, _ = G.vincenty_inverse(c_lat, c_lon, float(v["latitude"]), float(v["longitude"]))
        records.append({
            "bearing_from_centroid_deg": round(float(az), 1),
            "record_id": int(v["record_id"]),
            "longitude": float(v["longitude"]),
            "latitude": float(v["latitude"]),
            "database_raw": v["database_raw"],
            "source_family": v["source_family"],
            "match_radius_m": float(v["radius_m"]),
            "distance_to_footprint_m": round(float(min_site[j]), 1),
            "inside_match_radius": bool(min_site[j] <= float(v["radius_m"])),
        })
    inside = [r for r in records if r["inside_match_radius"]]
    classified = {}
    for r in records:
        classified.setdefault(r["source_family"], []).append(r["distance_to_footprint_m"])

    return {
        "transect": str(transect_id),
        "footprint_epsg": t["epsg"],
        "data_epsg": t.get("data_epsg"),
        "radius_m": radius_m,
        "n_vertices": int(len(lonlat)),
        "footprint_bbox": {"lon_min": float(lonlat[:, 0].min()), "lon_max": float(lonlat[:, 0].max()),
                           "lat_min": float(lonlat[:, 1].min()), "lat_max": float(lonlat[:, 1].max())},
        "centroid_lon": round(c_lon, 7),
        "centroid_lat": round(c_lat, 7),
        "n_records_within_radius": len(records),
        "n_records_inside_their_match_radius": len(inside),
        "nearest_m_by_family": {k: round(min(v), 1) for k, v in sorted(classified.items())},
        "records": records,
        "interpretation": (
            "Any feature within the match radius of one of these records is Known, not new. "
            f"Records listed as inside_match_radius already make part of transect {transect_id} Known. "
            "A feature elsewhere in the transect is only a candidate if it sits outside every listed "
            "radius; and a table miss is still not novelty until the literature check is done."),
    }


def render_md(ctx: dict) -> str:
    L = [f"# Known-record context: transect {ctx['transect']}", ""]
    bb = ctx["footprint_bbox"]
    L.append(f"- footprint CRS: EPSG:{ctx['footprint_epsg']} (per-tile LAZ CRS: EPSG:{ctx['data_epsg']})")
    L.append(f"- footprint bbox: lon {bb['lon_min']:.4f}..{bb['lon_max']:.4f}, "
             f"lat {bb['lat_min']:.4f}..{bb['lat_max']:.4f}")
    L.append(f"- records within {ctx['radius_m']:.0f} m: {ctx['n_records_within_radius']} "
             f"({ctx['n_records_inside_their_match_radius']} of them inside their own match radius)")
    L.append("")
    L.append("| record | register | family | radius | distance to footprint | bearing from centroid | inside radius |")
    L.append("|---:|---|---|---:|---:|---:|---|")
    for r in ctx["records"]:
        L.append(f"| {r['record_id']} | {r['database_raw']} | {r['source_family']} | "
                 f"{r['match_radius_m']:.0f} m | {r['distance_to_footprint_m']:.1f} m | "
                 f"{r['bearing_from_centroid_deg']:.0f} deg | "
                 f"{'yes' if r['inside_match_radius'] else 'no'} |")
    L.append("")
    L.append(ctx["interpretation"])
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="known-record context for one transect")
    ap.add_argument("--shp", required=True)
    ap.add_argument("--dbf", required=True)
    ap.add_argument("--known", required=True)
    ap.add_argument("--transect", default="0414")
    ap.add_argument("--radius", type=float, default=2000.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args(argv)

    known = load_known_table(args.known)
    transports, _info = load_transects(args.shp, args.dbf)
    ctx = transect_context(args.transect, transports, known, args.radius)
    out_json = Path(args.out_json or f"out/transect_{args.transect}_known_context.json")
    out_md = Path(args.out_md or f"out/transect_{args.transect}_known_context.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(ctx, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_md(ctx), encoding="utf-8")
    print(render_md(ctx))
    print(f"wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
