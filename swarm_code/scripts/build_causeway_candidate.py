#!/usr/bin/env python3
"""Build a private evidence pack for a Possible raised-linear candidate.

  .venv/bin/python code/build_causeway_candidate.py data/derived_v2/0875 0875-CW004

Requires the independent line verdict to be Possible. Never labels Strong/new.
Coordinates stay in private deliverable files only.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import shutil
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def jacobs_check(lon, lat):
    path = ROOT / "data/known/amazon_geoglyphs_geometry.json"
    if not path.exists():
        return {}, "Jacobs/KML unavailable"
    shapes = json.loads(path.read_text())
    geod = Geod(ellps="WGS84")
    scored = []
    for p in shapes:
        _, _, distance = geod.inv(lon, lat, p["centroid"][0], p["centroid"][1])
        scored.append((float(distance), p))
    scored.sort(key=lambda value: value[0])
    nearest_geometry = scored[0]
    simple = [(d, p) for d, p in scored if p["n_coords"] == 1][0]
    result = {
        "nearest_geometry_name": nearest_geometry[1]["name"],
        "nearest_geometry_description": re.sub(r"<[^>]+>", " ", nearest_geometry[1]["description"]).strip(),
        "nearest_geometry_distance_m": nearest_geometry[0],
        "nearest_simple_name": simple[1]["name"],
        "nearest_simple_description": re.sub(r"<[^>]+>", " ", simple[1]["description"]).strip(),
        "nearest_simple_distance_m": simple[0],
    }
    text = (f"Jacobs/KML checked offline: geometry '{result['nearest_geometry_name']}' "
            f"{result['nearest_geometry_distance_m']:.1f} m; simple named point "
            f"'{result['nearest_simple_name']}' {result['nearest_simple_distance_m']:.1f} m. "
            "Centroid or outline proximity is not an assumed match.")
    return result, text


def crop_save(source, name, row, col, radius_cells, target, filename):
    ny, nx = source.shape
    r0, r1 = max(0, row-radius_cells), min(ny, row+radius_cells+1)
    c0, c1 = max(0, col-radius_cells), min(nx, col+radius_cells+1)
    array = source[r0:r1, c0:c1]
    import sys
    sys.path.insert(0, str(ROOT / "code"))
    from process_laz import write_png
    kind = "chm" if name == "chm" else ("relief15" if name.startswith("relief") else name)
    write_png(target / filename, array, kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("candidate_id")
    args = p.parse_args()
    verdict_path = args.derived / f"causeway/{args.derived.name}/evidence/{args.candidate_id}/skeptic_line_verdict.json"
    verdict = json.loads(verdict_path.read_text())
    if verdict.get("certainty") != "Possible":
        raise ValueError(f"independent verdict is not Possible: {verdict.get('certainty')}")
    with (args.derived / f"causeway/{args.derived.name}/detections_{args.derived.name}.csv").open() as handle:
        row = next(r for r in csv.DictReader(handle) if r["id"] == args.candidate_id)
    audited = pd.read_csv(args.derived / "recordcheck/causeway_v3_audited.csv")
    audit = audited.loc[audited.id == args.candidate_id].iloc[0].to_dict()
    # The transect summary is the authoritative CRS after coordinator or
    # footprint correction.  Audited rows may still carry the earlier LAS
    # fallback plus its cross-zone distance for provenance.
    summary = json.loads((args.derived / "summary.json").read_text())
    grid = summary["grid"]
    epsg = int(summary["epsg"] or audit["EPSG"])
    x, y = float(row["x"]), float(row["y"])
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    jacobs, lit_note = jacobs_check(lon, lat)
    nearest_m = float(audit["nearest_known_m"])
    reason = (
        "Possible, not a new-site claim. The retained segment has two shoulders and continuous "
        "relief, but it is short and crosses no drainage confidently. A terrace remnant, natural "
        "levee, pad edge or modern track remains possible. The register miss and incomplete public "
        "2026 mapping do not establish novelty. " + lit_note
    )
    literature = (
        f"Peripato 2023 (961 rows), nearest {audit['nearest_known_db']} {nearest_m:.1f} m, no table match. "
        + lit_note + " Pärssinen et al. 2026 Nature (10.1038/s41586-026-10835-7), full article and "
        "supplements inspected: no complete public per-coordinate register was available for a "
        "definitive per-site match, which is not proof of novelty."
    )
    record = {"id": args.candidate_id, "state": audit.get("state", "RO"),
              "transect": args.derived.name, "EPSG": epsg, "x": x, "y": y,
              "lon": lon, "lat": lat, "type": row["type"], "size_m": float(row["length_m"]),
              "relief_m": float(row.get("height_m", audit.get("relief_m"))),
              "canopy_m": float(row.get("canopy_m", audit.get("canopy_m"))),
              "nearest_known_db": audit["nearest_known_db"], "nearest_known_m": nearest_m,
              "literature_checked": literature, "certainty": "Possible", "reason": reason,
              "jacobs_kml_check": jacobs}
    target = args.derived / "candidates" / args.candidate_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "candidate.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    source = np.load(args.derived / "rasters.npz")
    # Derive grid indices in the same south-up convention used by rasters.npz.
    grid_json = json.loads((args.derived / "summary.json").read_text())["grid"]
    col = int(round((x-grid_json["x0"]) / grid_json["cell"] - .5))
    row_idx = int(round((y-grid_json["y0"]) / grid_json["cell"] - .5))
    radius_cells = int(np.ceil(150 / grid_json["cell"]))
    for name, dest in (("hillshade", "hillshade.png"), ("relief_15", "relief.png"),
                       ("svf", "sky-view.png"), ("chm", "canopy.png"), ("dtm", "dtm.png")):
        crop_save(source[name], "chm" if name == "chm" else name, row_idx, col, radius_cells, target, dest)
    for name in ("map.png", "profiles.png", "skeptic_line.json", "evidence.json"):
        source_path = args.derived / f"causeway/{args.derived.name}/evidence/{args.candidate_id}/{name}"
        if source_path.exists():
            shutil.copy2(source_path, target / name)
    # Native radial/cross profiles around the raised segment's centre.
    cell = grid_json["cell"]
    rs = np.arange(0, 150 + cell, cell)
    yy, xx = np.mgrid[0:grid_json["ny"], 0:grid_json["nx"]]
    distance = np.hypot(yy-row_idx, xx-col) * cell
    relief = source["relief_15"]
    dtm = source["dtm"]
    radial = []
    for lo, hi in zip(rs[:-1], rs[1:]):
        selected = (distance >= lo) & (distance < hi)
        radial.append({"radius_inner_m": lo, "radius_outer_m": hi,
                       "radius_mid_m": (lo+hi)/2,
                       "n_dtm": int(np.isfinite(dtm[selected]).sum()),
                       "relief15_median_m": (float(np.nanmedian(relief[selected]))
                                            if np.isfinite(relief[selected]).any() else "")})
    with (target / "radial_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(radial[0])); writer.writeheader(); writer.writerows(radial)
    cross = []
    for orientation, offsets, dtm_values, relief_values in (
        ("east_west", (np.arange(grid_json["nx"])-col)*cell, dtm[row_idx, :], relief[row_idx, :]),
        ("north_south", (np.arange(grid_json["ny"])-row_idx)*cell, dtm[:, col], relief[:, col])):
        for offset, z, local in zip(offsets, dtm_values, relief_values):
            cross.append({"orientation": orientation, "offset_m": float(offset),
                          "dtm_m": float(z) if np.isfinite(z) else "",
                          "relief15_m": float(local) if np.isfinite(local) else ""})
    with (target / "cross_profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross[0])); writer.writeheader(); writer.writerows(cross)
    notes = ("# Skeptic notes: " + args.candidate_id + "\n\n- **Certainty:** Possible.\n- "
             f"- **Detector:** {row['type']}, length {row['length_m']} m, height {row['height_m']} m, "
             f"crest width {row['crest_width_m']} m, crossings {row['crossings']}, flags {row['flags']}.\n- "
             f"- **Nearest record:** {audit['nearest_known_db']}, {nearest_m:.1f} m, no match.\n- "
             f"- **Independent line verdict:** {verdict['reason']}\n- "
             f"- **Coordinator constraint:** short, unconfirmed drainage crossing; natural levee/terrace or modern track remains open. No Strong/new claim.\n- "
             f"- **Literature:** {literature}\n")
    (target / "skeptic_notes.md").write_text(notes)
    (target / "literature.md").write_text("# Literature checked\n\n" + literature + "\n")
    # Simple profile figure.
    fig, ax = plt.subplots(figsize=(7, 4))
    mids = [r["radius_mid_m"] for r in radial]
    values = [r["relief15_median_m"] if r["relief15_median_m"] != "" else np.nan for r in radial]
    ax.plot(mids, values, marker="."); ax.axhline(0, color="black", linewidth=.5)
    ax.set(xlabel="radius (m)", ylabel="relief (m)", title=f"{args.candidate_id}: radial profile")
    fig.tight_layout(); fig.savefig(target / "radial_profile.png", dpi=150); plt.close(fig)
    print(json.dumps({k: record[k] for k in ("id", "transect", "nearest_known_m", "certainty", "jacobs_kml_check")}, indent=2))


if __name__ == "__main__":
    main()
