"""Audit of the local known-earthwork table itself.

The record check is only as good as the table, so before any candidate is judged
this module answers, from the local files only:

  * are the three RDS readers and the coordinator's CSV conversion in agreement
    (row-by-row, value-by-value)?
  * what register labels exist, and how many records does each carry?
  * how many records are duplicates / how many distinct places does the table
    actually describe (single-linkage clusters at 2 km)?
  * where are the TREES/INPE records, and how close is each to a record from
    another register?  The swarm brief says those positions may be deliberately
    displaced, so the distance to the nearest other-register record is the
    evidence for or against that.
  * any coordinate that is missing, out of range, or outside the Amazon envelope?

Outputs (all local): out/known_table_audit.md, out/known_table_audit.json and
out/trees_inpe_records.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import geodesy as G
from .known_table import KnownTable, load_known_table, sha256_file


def cluster_count(lat, lon, link_m: float) -> int:
    """Number of single-linkage clusters at ``link_m`` (union-find)."""
    n = len(lat)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = G.pairwise_distance(lat, lon, lat, lon)
    for i in range(n):
        for j in np.where(d[i] <= link_m)[0]:
            if j > i:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[rb] = ra
    return len({find(i) for i in range(n)})


def cross_check_csv(table: KnownTable, csv_path: str | Path) -> dict:
    """Compare the canonical table with another CSV conversion of the same RDS."""
    other = pd.read_csv(csv_path)
    out: dict = {"path": str(csv_path), "sha256": sha256_file(csv_path), "rows": int(len(other)),
                 "columns": list(other.columns)}
    if len(other) != len(table.df):
        out["row_count_mismatch"] = True
        return out
    cols = {c.lower(): c for c in other.columns}
    lon_c = cols.get("longitude") or cols.get("lon")
    lat_c = cols.get("latitude") or cols.get("lat")
    db_c = cols.get("database") or cols.get("source")
    if not (lon_c and lat_c and db_c):
        out["error"] = "expected longitude/latitude/database columns"
        return out
    dlon = np.abs(pd.to_numeric(other[lon_c], errors="coerce").to_numpy(float) - table.df["longitude"].to_numpy(float))
    dlat = np.abs(pd.to_numeric(other[lat_c], errors="coerce").to_numpy(float) - table.df["latitude"].to_numpy(float))
    same_label = (other[db_c].astype(str).str.strip().to_numpy() == table.df["database_raw"].astype(str).str.strip().to_numpy())
    out.update({
        "max_abs_dlon": float(np.nanmax(dlon)),
        "max_abs_dlat": float(np.nanmax(dlat)),
        "n_rows_differing": int(((dlon > 1e-9) | (dlat > 1e-9)).sum()),
        "n_labels_differing": int((~same_label).sum()),
        "identical_within_1e-9": bool(np.nanmax(dlon) <= 1e-9 and np.nanmax(dlat) <= 1e-9 and same_label.all()),
        "labels_only_in_other": sorted(set(other[db_c].astype(str).str.strip()) - set(table.df["database_raw"].astype(str).str.strip()))[:10],
    })
    return out


def build_audit(table: KnownTable, coordinator_csv: str | Path | None = None,
                cluster_links=(500.0, 2000.0, 5000.0)) -> dict:
    df = table.df
    valid = df[df["valid_coord"]].reset_index(drop=True)
    lat = valid["latitude"].to_numpy(float)
    lon = valid["longitude"].to_numpy(float)

    d = G.pairwise_distance(lat, lon, lat, lon)
    iu = np.triu_indices(len(valid), k=1)
    pair_d = d[iu]
    fam = valid["source_family"].to_numpy()
    fam_pair = fam[iu[0]] != fam[iu[1]]

    dup_thresholds = (1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0)
    dup = {}
    for t in dup_thresholds:
        m = np.isfinite(pair_d) & (pair_d <= t)
        dup[f"pairs_within_{int(t)}m"] = int(m.sum())
        dup[f"pairs_within_{int(t)}m_cross_family"] = int((m & fam_pair).sum())

    trees = valid[valid["source_family"] == "trees_inpe"].copy()
    other = valid[valid["source_family"] != "trees_inpe"]
    rows = []
    if len(trees) and len(other):
        dt = G.pairwise_distance(trees["latitude"].to_numpy(float), trees["longitude"].to_numpy(float),
                                 other["latitude"].to_numpy(float), other["longitude"].to_numpy(float))
        for i in range(len(trees)):
            j = int(np.nanargmin(dt[i]))
            rows.append({
                "record_id": int(trees["record_id"].iloc[i]),
                "longitude": float(trees["longitude"].iloc[i]),
                "latitude": float(trees["latitude"].iloc[i]),
                "nearest_other_register_db": str(other["database_raw"].iloc[j]),
                "nearest_other_register_id": int(other["record_id"].iloc[j]),
                "nearest_other_register_m": round(float(dt[i, j]), 1),
                "within_500m_of_other_register": bool(dt[i, j] <= 500.0),
            })
    trees_df = pd.DataFrame(rows)

    decimals = {}
    for name in ("longitude", "latitude"):
        s = df[name].dropna().astype(str)
        decimals[name] = {"median_decimal_places": float(np.median([len(x.split(".")[1]) if "." in x else 0 for x in s])),
                          "n_with_5_or_more": int(sum(len(x.split(".")[1]) >= 5 for x in s if "." in x))}

    audit = {
        "table": table.meta(),
        "counts_by_raw_label": table.stats.get("counts_by_raw_label", {}),
        "counts_by_family": table.stats.get("counts_by_family", {}),
        "n_valid_coord": int(len(valid)),
        "n_invalid_coord": int((~df["valid_coord"]).sum()),
        "n_outside_amazon_bbox": int((~df["in_amazon_bbox"]).sum()),
        "bbox": table.stats.get("bbox"),
        "decimals": decimals,
        "duplicates": dup,
        "clusters": {f"link_{int(m)}m": cluster_count(lat, lon, m) for m in cluster_links},
        "trees_inpe": {
            "n": int(len(trees)),
            "n_within_500m_of_other_register": int(trees_df["within_500m_of_other_register"].sum()) if len(trees_df) else 0,
            "min_m": float(trees_df["nearest_other_register_m"].min()) if len(trees_df) else None,
            "median_m": float(trees_df["nearest_other_register_m"].median()) if len(trees_df) else None,
            "max_m": float(trees_df["nearest_other_register_m"].max()) if len(trees_df) else None,
            "detail_file": "out/trees_inpe_records.csv",
        },
        "warnings": table.warnings,
    }
    if coordinator_csv is not None and Path(coordinator_csv).exists():
        audit["cross_check_against_csv"] = cross_check_csv(table, coordinator_csv)
    return audit, trees_df


def render_markdown(audit: dict) -> str:
    a = audit
    L = []
    L.append("# Known-earthwork table audit (offline)")
    L.append("")
    L.append(f"Source file: `{a['table']['path']}`")
    L.append(f"SHA-256: `{a['table']['sha256']}`")
    L.append(f"Reader backend: {a['table']['backend']}; rules fingerprint: `{a['table']['rules_fingerprint']}`")
    L.append("")
    L.append("## Records")
    L.append("")
    L.append(f"- rows: {a['table']['rows']}; valid coordinates: {a['n_valid_coord']}; "
             f"invalid: {a['n_invalid_coord']}; outside the Amazon envelope: {a['n_outside_amazon_bbox']}")
    bb = a["bbox"]
    if bb and bb["lon_min"] is not None:
        L.append(f"- bbox: lon {bb['lon_min']:.4f}..{bb['lon_max']:.4f}, "
                 f"lat {bb['lat_min']:.4f}..{bb['lat_max']:.4f}")
    L.append("")
    L.append("### By raw register label")
    L.append("")
    L.append("| label | records |")
    L.append("|---|---:|")
    for k, v in sorted(a["counts_by_raw_label"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")
    L.append("")
    L.append("### By matched source family (decides the match radius)")
    L.append("")
    L.append("| family | records | radius |")
    L.append("|---|---:|---:|")
    radii = {"trees_inpe": 2000, "multiple_registers": 500, "amazon_arch": 500, "past": 500,
             "cnsa": 500, "inrap_dac": 500, "inrap": 500, "dac": 500, "sigplan": 500, "unresolved": 500}
    for k, v in sorted(a["counts_by_family"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} | {radii.get(k, 500)} m |")
    L.append("")
    L.append("## Duplicates and distinct places")
    L.append("")
    L.append("| threshold | pairs | pairs across registers |")
    L.append("|---|---:|---:|")
    for k, v in a["duplicates"].items():
        if k.endswith("_cross_family"):
            base = k[: -len("_cross_family")]
            L.append(f"| {base.replace('pairs_within_', '').replace('m', ' m')} | "
                     f"{a['duplicates'][base]} | {v} |")
    L.append("")
    L.append("Single-linkage clusters: " + ", ".join(f"{k} -> {v}" for k, v in a["clusters"].items()))
    L.append("")
    L.append("## TREES/INPE records (positions possibly displaced)")
    L.append("")
    t = a["trees_inpe"]
    L.append(f"- {t['n']} records; {t['n_within_500m_of_other_register']} lie within 500 m of a record "
             f"from another register")
    if t["min_m"] is not None:
        L.append(f"- distance to the nearest other-register record: min {t['min_m']:.1f} m, "
                 f"median {t['median_m']:.1f} m, max {t['max_m']:.1f} m")
    L.append(f"- per-record detail: `{t['detail_file']}`")
    L.append("")
    if "cross_check_against_csv" in a:
        c = a["cross_check_against_csv"]
        L.append("## Cross-check against the coordinator's CSV conversion")
        L.append("")
        L.append(f"- `{c['path']}` (sha256 `{c['sha256'][:16]}...`), {c['rows']} rows, columns {c['columns']}")
        if c.get("identical_within_1e-9"):
            L.append("- **identical to this reader's output**: same row count, labels and coordinates "
                     f"(max |dlon| {c['max_abs_dlon']:.1e}, max |dlat| {c['max_abs_dlat']:.1e})")
        else:
            L.append(f"- differences: {c.get('n_rows_differing')} rows, {c.get('n_labels_differing')} labels, "
                     f"max |dlon| {c.get('max_abs_dlon')}, max |dlat| {c.get('max_abs_dlat')}")
        L.append("")
    if a["warnings"]:
        L.append("## Warnings raised while loading")
        L.append("")
        for w in a["warnings"]:
            L.append(f"- {w}")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="audit the local known-earthwork table")
    ap.add_argument("--known", required=True)
    ap.add_argument("--coordinator-csv", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "pyreadr", "R", "pure"])
    ap.add_argument("--out-md", default="out/known_table_audit.md")
    ap.add_argument("--out-json", default="out/known_table_audit.json")
    ap.add_argument("--out-trees", default="out/trees_inpe_records.csv")
    args = ap.parse_args(argv)

    table = load_known_table(args.known, backend=args.backend)
    audit, trees_df = build_audit(table, args.coordinator_csv)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(audit), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    trees_df.to_csv(args.out_trees, index=False)
    print(render_markdown(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
