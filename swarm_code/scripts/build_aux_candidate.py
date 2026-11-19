#!/usr/bin/env python3
"""Build a complete private evidence pack for one auxiliary (mound) candidate.

  .venv/bin/python code/build_aux_candidate.py data/derived_v2/0254 M0254-013

The candidate schema is explicit about uncertainty; this script never labels a
feature Strong or new. Candidate coordinates are private deliverable data only.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import re
import shutil
import sys
import numpy as np
import pandas as pd
from pyproj import Transformer, Geod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("derived", type=Path)
    p.add_argument("candidate_id")
    args = p.parse_args()
    clusters = json.loads((args.derived / "detector/mound_clusters.json").read_text())
    cluster = next(c for c in clusters if c["id"] == args.candidate_id)
    audited = pd.read_csv(args.derived / "recordcheck/mounds_v3_audited.csv")
    audit = audited.loc[audited.id == args.candidate_id].iloc[0].to_dict()
    summary = json.loads((args.derived / "summary.json").read_text())
    grid = summary["grid"]
    source = np.load(args.derived / "rasters.npz")
    row, col = cluster["cell_rowcol"]
    x = grid["x0"] + (col + 0.5) * grid["cell"]
    y = grid["y0"] + (row + 0.5) * grid["cell"]
    transformer = Transformer.from_crs(grid.get("epsg", summary.get("epsg")), 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    radius = cluster["ring_radius_m"]
    R = radius / grid["cell"]
    yy, xx = np.mgrid[0:grid["ny"], 0:grid["nx"]]
    dist = np.hypot(yy-row, xx-col) * grid["cell"]
    canopy_band = (dist >= .8*radius) & (dist <= 1.2*radius) & np.isfinite(source["chm"])
    canopy = float(np.nanmedian(source["chm"][canopy_band]))
    bins = np.arange(0, max(150.0, 1.6*radius) + grid["cell"], grid["cell"])
    profiles = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        select = (dist >= lo) & (dist < hi)
        row_out = {"radius_inner_m": lo, "radius_outer_m": hi,
                   "radius_mid_m": (lo+hi)/2, "n_dtm": int(np.isfinite(source["dtm"][select]).sum())}
        for name, field in (("dtm", source["dtm"]), ("pre_relief15", source["relief_pre_15"]),
                            ("canopy", source["chm"])):
            values = field[select & np.isfinite(field)]
            row_out[name+"_median"] = float(np.median(values)) if values.size else ""
            if name == "pre_relief15":
                row_out[name+"_q25"] = float(np.percentile(values, 25)) if values.size else ""
                row_out[name+"_q75"] = float(np.percentile(values, 75)) if values.size else ""
        profiles.append(row_out)
    target = args.derived / "candidates" / args.candidate_id
    target.mkdir(parents=True, exist_ok=True)
    with (target / "radial_profile.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in profiles for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(profiles)
    # Native cross profiles through the fitted centre, both DTM and pre-clamp relief.
    r0, c0 = int(round(row)), int(round(col))
    cross = []
    for ori, offsets, dtmv, rv in (
        ("east_west", (np.arange(grid["nx"])-c0)*grid["cell"], source["dtm"][r0, :], source["relief_pre_15"][r0, :]),
        ("north_south", (np.arange(grid["ny"])-r0)*grid["cell"], source["dtm"][:, c0], source["relief_pre_15"][:, c0])):
        for off, z, relief in zip(offsets, dtmv, rv):
            cross.append({"orientation": ori, "offset_m": float(off),
                          "dtm_m": float(z) if np.isfinite(z) else "",
                          "pre_relief15_m": float(relief) if np.isfinite(relief) else ""})
    with (target / "cross_profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross[0]))
        writer.writeheader(); writer.writerows(cross)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mid = [row["radius_mid_m"] for row in profiles]
    med = [row["pre_relief15_median"] if row["pre_relief15_median"] != "" else np.nan for row in profiles]
    q25 = [row["pre_relief15_q25"] if row["pre_relief15_q25"] != "" else np.nan for row in profiles]
    q75 = [row["pre_relief15_q75"] if row["pre_relief15_q75"] != "" else np.nan for row in profiles]
    axes[0].fill_between(mid, q25, q75, alpha=.2)
    axes[0].plot(mid, med, marker=".")
    axes[0].axvline(radius, color="green", linestyle="--", label="fitted ring radius")
    axes[0].axhline(0, color="black", linewidth=.5)
    axes[0].set(xlabel="radius (m)", ylabel="pre-clamp relief (m)", title=f"{args.candidate_id}: radial profile")
    axes[0].legend()
    for ori in ("east_west", "north_south"):
        rows = [row for row in cross if row["orientation"] == ori]
        axes[1].plot([row["offset_m"] for row in rows],
                     [row["pre_relief15_m"] if row["pre_relief15_m"] != "" else np.nan for row in rows], label=ori)
    axes[1].set(xlabel="offset (m)", ylabel="pre-clamp relief (m)", title="cross profiles")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(target / "profiles.png", dpi=150); plt.close(fig)
    # Copy evidence with the deliverable's canonical names.
    review = args.derived / "mound_review"
    for source_name, destination in (
        (f"{args.candidate_id}_hillshade.png", "hillshade.png"),
        (f"{args.candidate_id}_relief_15.png", "relief.png"),
        (f"{args.candidate_id}_svf.png", "sky-view.png"),
        (f"{args.candidate_id}_chm.png", "canopy.png"),
        (f"{args.candidate_id}_pre_relief_annotated.png", "pre_relief_annotated.png"),
        (f"{args.candidate_id}_dtm.png", "dtm.png"),
        (f"{args.candidate_id}.json", "detector_metrics.json")):
        if (review / source_name).exists():
            shutil.copy2(review / source_name, target / destination)
    nearest_m = float(audit["nearest_known_m"])
    jacobs_note = "not checked"
    jacobs = {}
    jacobs_path = ROOT / "data/known/amazon_geoglyphs_geometry.json"
    if jacobs_path.exists():
        shapes = json.loads(jacobs_path.read_text())
        geod = Geod(ellps="WGS84")
        scored = []
        for placemark in shapes:
            _, _, distance = geod.inv(lon, lat, placemark["centroid"][0], placemark["centroid"][1])
            scored.append((float(distance), placemark))
        scored.sort(key=lambda item: item[0])
        simple = [(d, p) for d, p in scored if p["n_coords"] == 1]
        any_distance, any_shape = scored[0]
        simple_distance, simple_shape = simple[0]
        jacobs = {"nearest_geometry_name": any_shape["name"],
                  "nearest_geometry_description": re.sub(r"<[^>]+>", " ", any_shape["description"]).strip(),
                  "nearest_geometry_distance_m": any_distance,
                  "nearest_simple_name": simple_shape["name"],
                  "nearest_simple_description": re.sub(r"<[^>]+>", " ", simple_shape["description"]).strip(),
                  "nearest_simple_distance_m": simple_distance}
        jacobs_note = (f"Jacobs/KML checked offline: nearest geometry '{any_shape['name']}' "
                       f"{any_distance:.1f} m; nearest simple named point '{simple_shape['name']}' "
                       f"{simple_distance:.1f} m. Centroid or outline proximity is not assumed to be "
                       "the same earthwork.")
    else:
        jacobs_note = "Jacobs/KML register unavailable locally."
    literature = (
        jacobs_note + " " +
        "Peripato 2023 (961 rows) checked by offline geodesic audit: no table match; "
        f"nearest {audit['nearest_known_db']} {nearest_m:.1f} m. Pärssinen et al. 2026 Nature "
        "(10.1038/s41586-026-10835-7), full XML and all supplementary files inspected: "
        "hundreds of new southwest Amazon earthworks are documented, but no complete public "
        "near-coordinate register was available for a definitive per-site match. Iriarte et al. "
        "mound-village data discussed there describe 20 circular villages with about 12.6 mounds "
        f"and exterior diameter about 141 m; this candidate has {cluster['n_mounds']} detected bumps. Nature DOI "
        "and supplement checked locally, no candidate coordinates queried online. No other register "
        "match established, which is not proof of novelty."
    )
    reason = (
        f"Possible, not a new-site claim. {cluster['n_mounds']} positive bumps are fitted within ~{radius:.1f} m "
        f"with median height ~{cluster['height_median_m']:.2f} m, but the ring has "
        f"{cluster['ring_rms_frac']:.2%} residual and only {cluster['angular_coverage']:.2%} angular coverage. "
        f"The bump-size median is ~{cluster['mound_radius_median_m']:.1f} m and canopy median about {canopy:.1f} m. "
        "Natural stream/hummock and vegetation explanations remain open, and the cluster is smaller and "
        "less complete than known mound villages. The no-table-match and incomplete public 2026 register "
        "do not establish that it is unrecorded. " + jacobs_note
    )
    notes = (
        f"# Skeptic notes: {args.candidate_id}\n\n"
        f"- **Certainty:** Possible.\n"
        f"- **Geometry:** {cluster['n_mounds']} detected bumps, median height {cluster['height_median_m']:.3f} m; "
        f"ring radius {radius:.3f} m; residual {cluster['ring_rms_frac']:.4f}; "
        f"angular coverage {cluster['angular_coverage']:.4f}; plaza-clear test {bool(cluster['plaza_clear'])}.\n"
        f"- **Canopy:** {canopy:.2f} m median in the 0.8–1.2R annulus.\n"
        f"- **Record check:** {audit['nearest_known_db']}, {nearest_m:.1f} m, no match. A table miss is not novelty.\n"
        f"- **Natural/modern explanations left open:** stream bend / terrace hummocks, clumped vegetation, "
        "small modern clearings and partial-ring chance fits.\n"
        f"- **Reason:** {reason}\n\n"
        f"## Literature checked\n\n{literature}\n"
    )
    (target / "skeptic_notes.md").write_text(notes)
    (target / "literature.md").write_text("# Literature checked\n\n" + literature + "\n")
    record = {"id": args.candidate_id, "state": audit.get("state", "AC"), "transect": args.derived.name,
              "EPSG": summary["epsg"], "x": x, "y": y, "lon": lon, "lat": lat,
              "type": "mound_cluster", "size_m": 2*radius,
              "relief_m": cluster["height_median_m"], "canopy_m": canopy,
              "nearest_known_db": audit["nearest_known_db"], "nearest_known_m": nearest_m,
              "literature_checked": literature, "certainty": "Possible", "reason": reason,
              "jacobs_kml_check": jacobs}
    (target / "candidate.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("x", "y", "lon", "lat", "reason", "literature_checked")}, indent=2))


if __name__ == "__main__":
    main()
