#!/usr/bin/env python3
"""Per-candidate known-record annex: Peripato table + Jacobs KML, one page each.

For every row of results/candidates.csv writes
`out/candidates_known_context/<id>.md` (and a combined `.json`) holding:

  * the position actually used and the stated certainty;
  * the nearest Peripato records with register label, radius, distance and whether
    the candidate is inside that record's match radius;
  * the nearest Jacobs KML site-like placemarks, with the KML folder, so an
    `outlines` entry is not mistaken for a named site;
  * the record-check verdict and the one thing that must not be read into it
    (a table miss is not novelty).

This is the annex the IPHAN/Acre letter needs ("is this already recorded?"), kept
separate from the images so positions travel only with the private package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[2]
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT))
from recordcheck import geodesy as G  # noqa: E402
from recordcheck.kml import is_survey, read_kml  # noqa: E402
from recordcheck.known_table import load_known_table  # noqa: E402


def kml_site_nearest(placemarks, lon, lat, topk=3):
    sites = [p for p in placemarks if not is_survey(p)]
    best = []
    for pm in sites:
        pts = np.vstack(pm.parts)
        d = G.haversine(lat, lon, pts[:, 1], pts[:, 0])
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(d))
        s = float(G.vincenty_inverse(lat, lon, float(pts[j, 1]), float(pts[j, 0]))[0])
        best.append({"name": pm.name, "folder": pm.folder, "distance_m": round(s, 1)})
    best.sort(key=lambda r: r["distance_m"])
    return best[:topk]


def main() -> int:
    known = load_known_table(REPO / "data" / "known" / "peripato" / "Earthworks.rds")
    kdf = known.df[known.df["valid_coord"]].reset_index(drop=True)
    pms = read_kml(REPO / "data" / "known" / "amazon_geoglyphs.kml")
    cand = pd.read_csv(REPO / "results" / "candidates.csv", dtype={"transect": str})
    outdir = OUT / "candidates_known_context"
    outdir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for _, r in cand.iterrows():
        lon, lat = float(r["lon"]), float(r["lat"])
        d = G.pairwise_distance([lat], [lon], kdf["latitude"].to_numpy(float),
                                kdf["longitude"].to_numpy(float))[0]
        order = np.argsort(d)[:5]
        records = []
        for j in order:
            records.append({
                "record_id": int(kdf["record_id"].iloc[j]),
                "database_raw": kdf["database_raw"].iloc[j],
                "source_family": kdf["source_family"].iloc[j],
                "distance_m": round(float(d[j]), 1),
                "match_radius_m": float(kdf["radius_m"].iloc[j]),
                "inside_match_radius": bool(d[j] <= kdf["radius_m"].iloc[j]),
            })
        inside = [x for x in records if x["inside_match_radius"]]
        kml = kml_site_nearest(pms, lon, lat, topk=3)
        payload = {
            "id": str(r["id"]),
            "transect": str(r["transect"]),
            "type": r.get("type"),
            "certainty": r.get("certainty"),
            "lon": lon, "lat": lat,
            "epsg": int(r["EPSG"]) if pd.notna(r.get("EPSG")) else None,
            "x": float(r["x"]) if pd.notna(r.get("x")) else None,
            "y": float(r["y"]) if pd.notna(r.get("y")) else None,
            "table_match": "known_table" if inside else "no_table_match",
            "peripato_nearest": records,
            "jacobs_kml_nearest_sites": kml,
            "literature_checked_present": bool(str(r.get("literature_checked", "")).strip()),
        }
        combined[payload["id"]] = payload

        lines = [f"# {payload['id']} — known-record context", ""]
        lines.append(f"- transect {payload['transect']}, type {payload['type']}, "
                     f"stated certainty **{payload['certainty']}**")
        lines.append(f"- position (private): {lon:.6f}, {lat:.6f}"
                     + (f" (EPSG:{payload['epsg']} {payload['x']:.1f}, {payload['y']:.1f})" if payload['x'] is not None else ""))
        lines.append(f"- record-check verdict: **{'table match' if inside else 'no table match'}**")
        lines.append("")
        lines.append("## Peripato et al. 2023 table (the register the brief names)")
        lines.append("")
        lines.append("| record | register label | distance | match radius | inside |")
        lines.append("|---:|---|---:|---:|---|")
        for x in records:
            lines.append(f"| {x['record_id']} | {x['database_raw']} | {x['distance_m']:.1f} m | "
                         f"{x['match_radius_m']:.0f} m | {'yes' if x['inside_match_radius'] else 'no'} |")
        lines.append("")
        lines.append("## Jacobs KML, site-like placemarks only")
        lines.append("")
        lines.append("| placemark | folder | distance |")
        lines.append("|---|---|---:|")
        for x in kml:
            lines.append(f"| {x['name']} | `{x['folder'].split(' > ')[-1]}` | {x['distance_m']:.1f} m |")
        lines.append("")
        lines.append("`geoglyphs` and `mound sites` are named registers; `outlines` and `lines` are "
                     "geometric outlines and weaker evidence. Survey-track placemarks (named by date) "
                     "are excluded here because they are flight tracks, not sites.")
        lines.append("")
        lines.append("## What this does and does not mean")
        lines.append("")
        lines.append("- A table match means the position is already recorded; the feature is `Known` "
                     "and not a new site.")
        lines.append("- No table match means only that the Peripato register has nothing inside the "
                     "rule. It is **not** evidence of novelty: the register is a list of known sites, "
                     "not a survey, and the literature check remains the coordinator's step.")
        (outdir / f"{payload['id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (outdir / "candidates_known_context.json").write_text(json.dumps(combined, indent=2, sort_keys=True),
                                                        encoding="utf-8")
    print(f"wrote {len(combined)} candidate pages to {outdir}")
    for cid, p in combined.items():
        print(f"  {cid:<14} {p['certainty']:<9} {p['table_match']:<15} "
              f"nearest Peripato {p['peripato_nearest'][0]['distance_m']:>10.1f} m; "
              f"nearest KML site {p['jacobs_kml_nearest_sites'][0]['name']!r} "
              f"{p['jacobs_kml_nearest_sites'][0]['distance_m']:.1f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
