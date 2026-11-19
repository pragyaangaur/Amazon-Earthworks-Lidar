"""Offline record check for candidate features.

For every candidate it answers four separate questions and keeps them separate:

  ``nearest_known_db`` / ``nearest_known_m``
        the closest record in the local known-earthwork table and its distance,
        regardless of whether that counts as a match.
  ``d_table_match``
        ``known_table`` only if some record lies inside *its own* source radius
        (2000 m for TREES/INPE, 500 m for every other register); otherwise
        ``no_table_match``.  ``undetermined`` when the candidate has no usable
        coordinate.
  ``literature_checked``
        never written by this module.  It is the coordinator's online step; a
        table miss is NOT evidence of novelty and this tool says nothing about
        it.
  ``certainty``
        never written by this module either.  ``d_certainty_ceiling_from_table``
        is ``Known`` only when a table match exists; otherwise it records that
        the ceiling cannot be raised above the skeptic's verdict until the
        literature check is done.

Why the match rule is "exists a record inside its own radius" and not "the
nearest record is inside the nearest record's radius": with a non-TREES record at
800 m and a TREES/INPE record at 1500 m, the nearest record (800 m) is outside
500 m, but a TREES/INPE record *is* inside its 2 km radius, so the honest verdict
is a table match.  ``d_within_2km_any`` is reported as a sensitivity check so an
unmapped register label cannot make a candidate look novel.

Coordinates: ``lon``/``lat`` are used when present; otherwise ``x``/``y`` with
``EPSG`` are transformed (needs pyproj).  When both are present the projected
position is transformed back and compared with the supplied lon/lat -- a
mismatch beyond ``--crs-tolerance-m`` is flagged, which catches the classic
axis-order or wrong-CRS mistake before a distance is believed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geodesy as G
from .known_table import KnownTable, load_known_table

SCHEMA = ["id", "state", "transect", "EPSG", "x", "y", "lon", "lat", "type", "size_m", "relief_m",
          "canopy_m", "nearest_known_db", "nearest_known_m", "literature_checked", "certainty", "reason"]

DIAG = ["d_coord_status", "d_lon_used", "d_lat_used", "d_epsg_check", "d_epsg_delta_m",
        "d_nearest_record_id", "d_nearest_m", "d_nearest_source_raw", "d_nearest_source_family", "d_nearest_radius_m",
        "d_nearest_m_by_family", "d_within_2km_any", "d_n_within_2km", "d_n_within_500m",
        "d_table_match", "d_match_record_id", "d_match_source_raw", "d_match_source_family",
        "d_match_distance_m", "d_match_threshold_m", "d_match_margin_m", "d_nearest_ratio",
        "d_certainty_ceiling_from_table", "d_flags"]

FLAG_NOTES = {
    "coord_missing": "no usable coordinate in the row",
    "coord_invalid": "coordinate outside valid lat/lon range or non-finite",
    "coord_from_xy": "lon/lat derived from x/y + EPSG by pyproj",
    "epsg_missing": "x/y present but no EPSG column value",
    "epsg_no_pyproj": "pyproj unavailable, x/y could not be verified or transformed",
    "crs_mismatch": "supplied lon/lat differ from the projected x/y beyond tolerance",
    "crs_suspect_units": "coordinate magnitudes look like a different CRS family",
    "source_family_unresolved": "known-table labels not mapped to a register family",
    "within_2km_any_source": "a known record lies within 2 km whatever its register",
    "nearest_is_trees_inpe": "nearest record is TREES/INPE, whose positions may be displaced",
    "distance_unreliable": "geodesic iteration failed (bad input or near-antipodal)",
    "id_missing": "no id column value; a synthetic id was assigned",
    "id_duplicate": "id appears more than once in the input",
    "known_table_empty": "known table has no valid coordinate; every verdict is undetermined",
    "match_near_threshold": "the table match sits within 50 m (500 m rule) or 200 m (2 km rule) of the "
                            "threshold; the verdict follows the rule but the coordinator should look at it",
    "just_outside_match_radius": "no table match, but some record lies just outside its own radius "
                                 "(distance/radius <= 1.25); the verdict follows the rule, and this is "
                                 "where a rule change would matter",
}


@dataclass
class AuditConfig:
    trees_radius_m: float = 2000.0
    other_radius_m: float = 500.0
    crs_tolerance_m: float = 25.0
    chunk: int = 128
    top_k: int = 5
    # EPSG:5880 (Brazil Polyconic) magnitudes; used only for a warning, never to
    # change a coordinate.
    polyconic_x_range: tuple = (2_000_000.0, 8_000_000.0)
    polyconic_y_range: tuple = (8_000_000.0, 12_000_000.0)


# --------------------------------------------------------------------------
# coordinate resolution
# --------------------------------------------------------------------------
def _get_col(df: pd.DataFrame, name: str):
    for cand in (name, name.lower(), name.upper(), name.capitalize()):
        if cand in df.columns:
            return df[cand]
    return None


def _to_float(series) -> np.ndarray:
    if series is None:
        return np.full(0, np.nan)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _transform_xy(x: np.ndarray, y: np.ndarray, epsg: str):
    """Transform projected x/y to lon/lat with pyproj; returns (lon, lat, status)."""
    try:
        from pyproj import CRS, Transformer
    except Exception:
        return None, None, "pyproj_missing"
    try:
        crs_from = CRS.from_user_input(f"EPSG:{str(epsg).strip()}")
        tr = Transformer.from_crs(crs_from, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(x, y)
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if np.any(~np.isfinite(lon)) or np.any(~np.isfinite(lat)):
            return lon, lat, "transform_nan"
        return lon, lat, "ok"
    except Exception as exc:  # unknown EPSG code, PROJ error, ...
        return None, None, f"error:{type(exc).__name__}:{exc}"


def resolve_coordinates(df: pd.DataFrame, cfg: AuditConfig) -> tuple[pd.DataFrame, dict]:
    """Add d_lon_used, d_lat_used, d_coord_status, d_epsg_check, d_epsg_delta_m."""
    n = len(df)
    lon_in = _to_float(_get_col(df, "lon")) if _get_col(df, "lon") is not None else np.full(n, np.nan)
    lat_in = _to_float(_get_col(df, "lat")) if _get_col(df, "lat") is not None else np.full(n, np.nan)
    x_in = _to_float(_get_col(df, "x")) if _get_col(df, "x") is not None else np.full(n, np.nan)
    y_in = _to_float(_get_col(df, "y")) if _get_col(df, "y") is not None else np.full(n, np.nan)
    epsg_col = _get_col(df, "EPSG")

    lon = lon_in.copy()
    lat = lat_in.copy()
    status = np.array(["given"] * n, dtype=object)
    epsg_check = np.array(["not_applicable"] * n, dtype=object)
    epsg_delta = np.full(n, np.nan)
    flags: list[set] = [set() for _ in range(n)]
    notes = {}

    has_ll = np.isfinite(lon_in) & np.isfinite(lat_in)
    has_xy = np.isfinite(x_in) & np.isfinite(y_in)
    missing_epsg = 0
    crs_mismatch = 0
    suspect_units = 0
    no_pyproj = False

    for epsg_value, group in _group_by_epsg(epsg_col, n):
        idx = np.array(group, dtype=int)
        idx = idx[has_xy[idx] | has_ll[idx]]
        if idx.size == 0:
            continue
        if epsg_value is None or str(epsg_value).strip() in ("", "nan", "None"):
            rows_xy = idx[has_xy[idx]]
            if rows_xy.size:
                missing_epsg += rows_xy.size
                for i in rows_xy:
                    flags[i].add("epsg_missing")
            continue
        rows_xy = idx[has_xy[idx]]
        if rows_xy.size == 0:
            continue
        lon_p, lat_p, st = _transform_xy(x_in[rows_xy], y_in[rows_xy], epsg_value)
        if st != "ok":
            no_pyproj = st == "pyproj_missing"
            for i in rows_xy:
                flags[i].add("epsg_no_pyproj" if no_pyproj else "crs_mismatch")
                epsg_check[i] = st
            continue
        for k, i in enumerate(rows_xy):
            if not has_ll[i]:
                lon[i], lat[i] = lon_p[k], lat_p[k]
                status[i] = "from_xy"
                flags[i].add("coord_from_xy")
                epsg_check[i] = "ok"
            else:
                d, _, _ = G.vincenty_inverse(lat[i], lon[i], lat_p[k], lon_p[k])
                epsg_delta[i] = d
                if math.isfinite(d) and d > cfg.crs_tolerance_m:
                    epsg_check[i] = "mismatch"
                    crs_mismatch += 1
                    flags[i].add("crs_mismatch")
                else:
                    epsg_check[i] = "ok"
            # unit sanity: polyconic coordinates are large; UTM ones are not
            if str(epsg_value).strip() == "5880":
                if not (cfg.polyconic_x_range[0] <= x_in[i] <= cfg.polyconic_x_range[1]
                        and cfg.polyconic_y_range[0] <= y_in[i] <= cfg.polyconic_y_range[1]):
                    suspect_units += 1
                    flags[i].add("crs_suspect_units")

    bad = ~(np.isfinite(lon) & np.isfinite(lat))
    for i in np.where(bad)[0]:
        status[i] = "missing" if (not has_ll[i] and not has_xy[i]) else "invalid"
        flags[i].add("coord_missing" if status[i] == "missing" else "coord_invalid")
    out_of_range = np.isfinite(lon) & np.isfinite(lat) & ((np.abs(lat) > 90.0) | (np.abs(lon) > 360.0))
    for i in np.where(out_of_range)[0]:
        status[i] = "invalid"
        flags[i].add("coord_invalid")

    out = pd.DataFrame({
        "d_coord_status": status,
        "d_lon_used": lon,
        "d_lat_used": lat,
        "d_epsg_check": epsg_check,
        "d_epsg_delta_m": epsg_delta,
    })
    notes["n_missing_epsg"] = int(missing_epsg)
    notes["n_crs_mismatch"] = int(crs_mismatch)
    notes["n_suspect_units"] = int(suspect_units)
    notes["pyproj_missing"] = bool(no_pyproj)
    notes["n_coord_missing"] = int((status == "missing").sum())
    notes["n_coord_invalid"] = int((status == "invalid").sum())
    notes["n_from_xy"] = int((status == "from_xy").sum())
    return out, {"flags": flags, "notes": notes}


def _group_by_epsg(epsg_col, n: int):
    if epsg_col is None:
        yield None, list(range(n))
        return
    values = epsg_col.astype(object).tolist()
    groups: dict = {}
    for i, v in enumerate(values):
        key = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()
        groups.setdefault(key, []).append(i)
    for key, idx in groups.items():
        yield key, idx


# --------------------------------------------------------------------------
# distance + match logic
# --------------------------------------------------------------------------
def audit_against_known(cands: pd.DataFrame, known: KnownTable, cfg: AuditConfig) -> tuple[pd.DataFrame, list[dict], dict]:
    """Compute distances and match verdicts.  ``cands`` needs d_lon_used/d_lat_used."""
    lon = cands["d_lon_used"].to_numpy(dtype=float)
    lat = cands["d_lat_used"].to_numpy(dtype=float)
    n = len(cands)

    kdf = known.df
    valid = kdf["valid_coord"].to_numpy(dtype=bool)
    rec = kdf.loc[valid].reset_index(drop=False).rename(columns={"index": "orig_index"})
    rec_ids = rec["record_id"].to_numpy(dtype=int)
    rec_raw = rec["database_raw"].tolist()
    rec_fam = rec["source_family"].tolist()
    radii = rec["radius_m"].to_numpy(dtype=float)
    rec_unres = rec["source_unresolved"].to_numpy(dtype=bool) if "source_unresolved" in rec else np.zeros(len(rec), dtype=bool)

    per_candidate: list[dict] = []

    if n == 0:
        # an empty detection list must not crash a pipeline: return a frame with
        # the diagnostic columns so downstream merges still work
        out = pd.DataFrame({c: pd.Series(dtype=object) for c in DIAG})
        for c in ("d_lon_used", "d_lat_used", "d_nearest_m", "d_epsg_delta_m"):
            out[c] = pd.Series(dtype=float)
        out["d_table_match"] = pd.Series(dtype=object)
        for c in ("d_n_within_2km", "d_n_within_500m"):
            out[c] = pd.Series(dtype=object)
        out["d_flags"] = pd.Series(dtype=object)
        return out, [], {"n_records_used": int(len(rec)), "n_candidates": 0,
                         "note": "empty candidate list"}

    if len(rec) == 0:
        out = pd.DataFrame({c: np.array([None] * n, dtype=object) for c in DIAG if c not in ("d_lon_used", "d_lat_used")})
        for c in ("d_lon_used", "d_lat_used"):
            out[c] = cands[c].to_numpy(dtype=float)
        out["d_table_match"] = "undetermined"
        for c in ("d_n_within_2km", "d_n_within_500m"):
            out[c] = 0
        out["d_flags"] = "known_table_empty"
        per_candidate = [{"id": None, "verdict": "undetermined", "reason": "known table empty"} for _ in range(n)]
        return out, per_candidate, {"n_records_used": 0}

    d = G.pairwise_distance(lat, lon, rec["latitude"].to_numpy(), rec["longitude"].to_numpy(),
                            chunk=cfg.chunk)

    families = sorted(set(rec_fam))
    fam_masks = {f: np.array([x == f for x in rec_fam], dtype=bool) for f in families}

    rows = []
    with np.errstate(invalid="ignore"):
        for i in range(n):
            row: dict = {}
            fset = set()
            if not math.isfinite(lon[i]) or not math.isfinite(lat[i]):
                row.update({
                    "d_nearest_record_id": None, "d_nearest_m": None, "d_nearest_source_raw": None,
                    "d_nearest_source_family": None,
                    "d_nearest_radius_m": None, "d_nearest_m_by_family": "{}", "d_within_2km_any": None,
                    "d_n_within_2km": None, "d_n_within_500m": None, "d_table_match": "undetermined",
                    "d_match_record_id": None, "d_match_source_raw": None, "d_match_source_family": None,
                    "d_match_distance_m": None, "d_match_threshold_m": None, "d_match_margin_m": None,
                    "d_nearest_ratio": None,
                    "d_certainty_ceiling_from_table": "unset (no usable coordinate)",
                })
                rows.append(row)
                per_candidate.append({"index": i, "verdict": "undetermined", "reason": "no usable coordinate"})
                continue
            di = d[i]
            finite = np.isfinite(di)
            if not finite.any():
                row.update({
                    "d_nearest_record_id": None, "d_nearest_m": None, "d_nearest_source_raw": None,
                    "d_nearest_source_family": None,
                    "d_nearest_radius_m": None, "d_nearest_m_by_family": "{}", "d_within_2km_any": None,
                    "d_n_within_2km": None, "d_n_within_500m": None, "d_table_match": "undetermined",
                    "d_match_record_id": None, "d_match_source_raw": None, "d_match_source_family": None,
                    "d_match_distance_m": None, "d_match_threshold_m": None, "d_match_margin_m": None,
                    "d_nearest_ratio": None,
                    "d_certainty_ceiling_from_table": "unset (distances unreliable)",
                })
                rows.append(row)
                per_candidate.append({"index": i, "verdict": "undetermined", "reason": "all distances unreliable"})
                continue

            j = int(np.nanargmin(di))
            row["d_nearest_record_id"] = int(rec_ids[j])
            row["d_nearest_m"] = float(di[j])
            row["d_nearest_source_raw"] = rec_raw[j]
            row["d_nearest_source_family"] = rec_fam[j]
            row["d_nearest_radius_m"] = float(radii[j])
            fam_min = {}
            for fam in families:
                m = fam_masks[fam] & finite
                fam_min[fam] = float(np.min(di[m])) if m.any() else None
            row["d_nearest_m_by_family"] = json.dumps(
                {k: (None if v is None else round(v, 1)) for k, v in sorted(fam_min.items())},
                sort_keys=True)
            row["d_within_2km_any"] = bool((di[finite] <= cfg.trees_radius_m).any())
            row["d_n_within_2km"] = int((di[finite] <= cfg.trees_radius_m).sum())
            row["d_n_within_500m"] = int((di[finite] <= cfg.other_radius_m).sum())

            # how close is the best record relative to its OWN radius?  1.0 means on
            # the threshold; the flag below catches near misses on either side
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(finite, di / radii, np.inf)
            rmin = float(np.nanmin(ratio)) if np.isfinite(ratio).any() else float("nan")
            row["d_nearest_ratio"] = None if not math.isfinite(rmin) else round(rmin, 3)

            inside = finite & (di <= radii)
            if inside.any():
                jm = int(np.nanargmin(np.where(inside, di, np.inf)))
                row["d_table_match"] = "known_table"
                row["d_match_record_id"] = int(rec_ids[jm])
                row["d_match_source_raw"] = rec_raw[jm]
                row["d_match_source_family"] = rec_fam[jm]
                row["d_match_distance_m"] = round(float(di[jm]), 1)
                row["d_match_threshold_m"] = float(radii[jm])
                margin = float(radii[jm]) - float(di[jm])
                row["d_match_margin_m"] = round(margin, 1)
                if margin < max(50.0, 0.1 * float(radii[jm])):
                    fset.add("match_near_threshold")
                row["d_certainty_ceiling_from_table"] = "Known"
                if rec_fam[jm] == "trees_inpe":
                    fset.add("nearest_is_trees_inpe")
            else:
                row["d_table_match"] = "no_table_match"
                row["d_match_record_id"] = None
                row["d_match_source_raw"] = None
                row["d_match_source_family"] = None
                row["d_match_distance_m"] = None
                row["d_match_threshold_m"] = None
                row["d_match_margin_m"] = None
                row["d_certainty_ceiling_from_table"] = "unset (table miss is not novelty; needs literature check)"
                if math.isfinite(rmin) and 1.0 < rmin <= 1.25:
                    fset.add("just_outside_match_radius")
            if row["d_within_2km_any"]:
                fset.add("within_2km_any_source")
            if rec_fam[j] == "trees_inpe":
                fset.add("nearest_is_trees_inpe")
            if rec_unres[j]:
                fset.add("source_family_unresolved")
            if row.get("d_match_record_id") is not None:
                jm_rec = np.where(rec_ids == row["d_match_record_id"])[0]
                if jm_rec.size and rec_unres[jm_rec[0]]:
                    fset.add("source_family_unresolved")
            row["_rowflags"] = fset
            rows.append(row)

            top = np.argsort(np.where(finite, di, np.inf))[: cfg.top_k]
            per_candidate.append({
                "index": i,
                "nearest": [
                    {"record_id": int(rec_ids[k]), "source_raw": rec_raw[k], "source_family": rec_fam[k],
                     "distance_m": round(float(di[k]), 2), "radius_m": float(radii[k]),
                     "within_threshold": bool(di[k] <= radii[k])}
                    for k in top if math.isfinite(di[k])
                ],
                "verdict": row["d_table_match"],
                "n_within_2km": row["d_n_within_2km"],
            })

    out = pd.DataFrame(rows)
    summary = {
        "n_records_used": int(len(rec)),
        "n_records_dropped_invalid_coord": int((~valid).sum()),
        "families": families,
        "counts_by_family": {f: int(fam_masks[f].sum()) for f in families},
    }
    return out, per_candidate, summary


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------
def _certainty_summary(candidates: pd.DataFrame):
    if "certainty" not in candidates.columns:
        return "absent (the record checker does not write certainty)"
    out = {str(k): int(v) for k, v in candidates["certainty"].value_counts(dropna=False).items()}
    out["note"] = "passed through unchanged from the input; not written by this tool"
    return out


def run_audit(candidates: pd.DataFrame, known: KnownTable, cfg: AuditConfig | None = None) -> dict:
    cfg = cfg or AuditConfig()
    df = candidates.copy()
    n = len(df)
    if "id" not in df.columns:
        df["id"] = [f"cand_{i:04d}" for i in range(n)]
    id_missing = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
    if id_missing.any():
        df.loc[id_missing, "id"] = [f"cand_{i:04d}" for i in np.where(id_missing)[0]]
    id_dupes = df["id"].astype(str).duplicated(keep=False)

    if n == 0:
        # keep the deliverable schema and the diagnostic columns even with no rows
        empty = pd.DataFrame({c: pd.Series(dtype=object) for c in SCHEMA + DIAG})
        for c in ("EPSG", "x", "y", "lon", "lat"):
            empty[c] = pd.Series(dtype=float)
        summary = {
            "generated_by": "recordcheck.audit",
            "n_candidates": 0,
            "known_table": known.meta(),
            "counts_by_table_match": {},
            "flag_counts": {},
            "note": "empty candidate list: nothing to check",
            "match_rule": "known_table iff any record has geodesic distance <= the radius of its own register",
        }
        return {"table": empty, "details": [], "summary": summary}

    coord_df, coord_info = resolve_coordinates(df, cfg)
    dist_df, per_candidate, dist_summary = audit_against_known(coord_df, known, cfg)

    out = df.copy()
    # keep the deliverable schema columns that exist, and add empty ones for the rest
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = ""
    # fill nearest-known fields from the distance stage
    # the deliverable's lon/lat must be the coordinates actually used: when the
    # detector list carries only EPSG + x/y, they are derived by pyproj and the
    # original empty columns would otherwise leave the candidate without a position
    out["lon"] = coord_df["d_lon_used"].values
    out["lat"] = coord_df["d_lat_used"].values
    out["nearest_known_db"] = dist_df["d_nearest_source_raw"].values
    out["nearest_known_m"] = [
        (None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 1))
        for v in dist_df["d_nearest_m"].values
    ]
    if "literature_checked" in df.columns:
        out["literature_checked"] = df["literature_checked"].fillna("").astype(str)
    else:
        out["literature_checked"] = ""
    out["d_note_literature"] = "never set by record checker; requires the coordinator's online literature step"

    for col in ("d_coord_status", "d_lon_used", "d_lat_used", "d_epsg_check", "d_epsg_delta_m"):
        out[col] = coord_df[col].values
    for col in DIAG:
        if col in dist_df.columns:
            out[col] = dist_df[col].values
    out["d_lon_used"] = out["d_lon_used"].astype(float)
    out["d_lat_used"] = out["d_lat_used"].astype(float)

    # combine per-row flags from coordinate stage and distance stage
    combined = []
    for i in range(len(out)):
        fset = set(coord_info["flags"][i]) if i < len(coord_info["flags"]) else set()
        extra = dist_df.iloc[i].get("_rowflags", set()) if "_rowflags" in dist_df.columns else set()
        fset |= set(extra) if isinstance(extra, (set, list, tuple, frozenset)) else set()
        if id_missing.iloc[i]:
            fset.add("id_missing")
        if id_dupes.iloc[i]:
            fset.add("id_duplicate")
        combined.append(";".join(sorted(fset)))
    out["d_flags"] = combined

    # attach the ids to the per-candidate detail
    for detail, cid in zip(per_candidate, out["id"].astype(str).tolist()):
        detail["id"] = cid

    counts = out["d_table_match"].value_counts().to_dict()
    flag_counts: dict = {}
    for f in out["d_flags"]:
        for token in (f.split(";") if f else []):
            flag_counts[token] = flag_counts.get(token, 0) + 1

    uncertain_sources = sorted(set(known.df.loc[known.df["source_unresolved"], "database_raw"]))
    summary = {
        "generated_by": "recordcheck.audit",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "thresholds": {
            "trees_inpe_radius_m": cfg.trees_radius_m,
            "other_source_radius_m": cfg.other_radius_m,
            "crs_tolerance_m": cfg.crs_tolerance_m,
        },
        "match_rule": "known_table iff any record has geodesic distance <= the radius of its own register; "
                      "nearest_known_* report the nearest record regardless of the threshold",
        "known_table": known.meta(),
        "n_candidates": int(len(out)),
        "counts_by_table_match": {str(k): int(v) for k, v in counts.items()},
        "flag_counts": flag_counts,
        "coordinate_notes": coord_info["notes"],
        "distance_summary": dist_summary,
        "unresolved_source_labels": uncertain_sources,
        "certainty": {
            "Known": int((out["d_certainty_ceiling_from_table"] == "Known").sum()),
            "unset": int((out["d_certainty_ceiling_from_table"] != "Known").sum()),
            "note": "Known is the only certainty this offline tool can assert; no feature is new because "
                    "the table misses it. Strong/Possible/Rejected need the skeptic and the literature check.",
        },
        "flag_notes": FLAG_NOTES,
        "lon_lat_note": "lon/lat in the audited table are the coordinates actually used for the "
                        "distances; rows flagged coord_from_xy were derived from EPSG + x/y with pyproj",
        "input_certainty_column": _certainty_summary(candidates),
    }
    return {"table": out, "details": per_candidate, "summary": summary}


def write_candidate_notes(result: dict, out_dir: str | Path) -> list[str]:
    """One evidence note per candidate under <out_dir>/candidates/<id>/.

    These are the record-check pages that belong in the final zip's
    ``candidates/<id>/`` directories: what the nearest known records are, at what
    distance, whether a table match exists, and the ceiling that implies.  They
    contain coordinates and are local/private by construction.
    """
    out_dir = Path(out_dir)
    written = []
    table = result["table"].reset_index(drop=True)
    details = {str(d.get("id")): d for d in result["details"]}
    for i, row in table.iterrows():
        cid = str(row["id"])
        detail = details.get(cid, {})
        d = out_dir / "candidates" / cid
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": cid,
            "transect": row.get("transect"),
            "lon": float(row["d_lon_used"]) if pd.notna(row["d_lon_used"]) else None,
            "lat": float(row["d_lat_used"]) if pd.notna(row["d_lat_used"]) else None,
            "table_match": row["d_table_match"],
            "certainty_ceiling_from_table": row["d_certainty_ceiling_from_table"],
            "nearest_known_db": None if pd.isna(row["nearest_known_db"]) else row["nearest_known_db"],
            "nearest_known_m": None if pd.isna(row["nearest_known_m"]) else float(row["nearest_known_m"]),
            "match_record_id": None if pd.isna(row["d_match_record_id"]) else int(row["d_match_record_id"]),
            "match_distance_m": None if pd.isna(row["d_match_distance_m"]) else float(row["d_match_distance_m"]),
            "match_threshold_m": None if pd.isna(row["d_match_threshold_m"]) else float(row["d_match_threshold_m"]),
            "flags": row["d_flags"],
            "nearest_records": detail.get("nearest", []),
            "note": "Offline record check only. literature_checked and certainty are NOT set here; "
                    "a table miss is not novelty.",
        }
        (d / "known_records.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        lines = [f"# Record check: {cid}", "",
                 f"- position: {payload['lon']}, {payload['lat']} (used for the distances)",
                 f"- table match: **{payload['table_match']}**",
                 f"- certainty ceiling from the table: {payload['certainty_ceiling_from_table']}",
                 f"- nearest known record: {payload['nearest_known_db']} at {payload['nearest_known_m']} m"]
        if payload["match_record_id"] is not None:
            lines.append(f"- matching record: id {payload['match_record_id']} at "
                         f"{payload['match_distance_m']} m of a {payload['match_threshold_m']:.0f} m radius")
        lines.append(f"- flags: {payload['flags']}")
        lines += ["", "| record | register | family | distance | radius | inside radius |",
                  "|---:|---|---|---:|---:|---|"]
        for r in payload["nearest_records"]:
            lines.append(f"| {r['record_id']} | {r['source_raw']} | {r['source_family']} | "
                         f"{r['distance_m']:.1f} m | {r['radius_m']:.0f} m | "
                         f"{'yes' if r['within_threshold'] else 'no'} |")
        lines += ["", payload["note"], ""]
        (d / "known_records.md").write_text("\n".join(lines), encoding="utf-8")
        written.append(str(d))
    return written


def write_outputs(result: dict, out_dir: str | Path, stem: str = "candidates",
                  candidate_dirs: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = result["table"]
    cols = SCHEMA + [c for c in table.columns if c not in SCHEMA and c != "_rowflags"]
    csv_path = out_dir / f"{stem}_audited.csv"
    table[cols].to_csv(csv_path, index=False)
    detail_path = out_dir / f"{stem}_audit_detail.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(result["details"], fh, indent=2, sort_keys=True)
    summary_path = out_dir / f"{stem}_audit_summary.json"
    summary = dict(result["summary"])
    summary["written_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["outputs"] = {"table": str(csv_path), "detail": str(detail_path), "summary": str(summary_path)}
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    out = {"csv": str(csv_path), "detail": str(detail_path), "summary": str(summary_path)}
    if candidate_dirs:
        out["candidate_dirs"] = write_candidate_notes(result, out_dir)
    return out


# --------------------------------------------------------------------------
# control-point mode
# --------------------------------------------------------------------------
def check_control_point(lon: float, lat: float, known: KnownTable, cfg: AuditConfig | None = None,
                        label: str = "control") -> dict:
    """Record check for a single point, e.g. the 0414 positive control."""
    cfg = cfg or AuditConfig()
    df = pd.DataFrame([{"id": label, "lon": lon, "lat": lat}])
    res = run_audit(df, known, cfg)
    row = res["table"].iloc[0]
    detail = res["details"][0]
    return {
        "label": label,
        "lon": lon,
        "lat": lat,
        "table_match": row["d_table_match"],
        "certainty_ceiling_from_table": row["d_certainty_ceiling_from_table"],
        "nearest_known_db": row["nearest_known_db"],
        "nearest_known_m": row["nearest_known_m"],
        "flags": row["d_flags"],
        "candidates": detail.get("nearest", []),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="offline record check for candidate earthworks")
    ap.add_argument("--candidates", help="candidates CSV (needs id plus lon/lat or EPSG+x+y)")
    ap.add_argument("--known", default=None, help="path to Earthworks.rds")
    ap.add_argument("--known-csv", default=None, help="already-normalised known_sites.csv")
    ap.add_argument("--search-root", default=None, help="root to search for Earthworks.rds")
    ap.add_argument("--backend", default="auto", choices=["auto", "pyreadr", "R", "pure"])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--stem", default="candidates")
    ap.add_argument("--trees-radius", type=float, default=2000.0)
    ap.add_argument("--other-radius", type=float, default=500.0)
    ap.add_argument("--crs-tolerance-m", type=float, default=25.0)
    ap.add_argument("--control-lon", type=float, default=None, help="longitude of a control point")
    ap.add_argument("--control-lat", type=float, default=None, help="latitude of a control point")
    ap.add_argument("--control-lonlat", default=None,
                    help="'lon,lat' of a control point; use the --control-lonlat=-67.28,-9.87 form "
                         "because argparse reads a leading minus as an option")
    ap.add_argument("--control-label", default="control")
    ap.add_argument("--write-known-copy", action="store_true")
    ap.add_argument("--candidate-dirs", action="store_true",
                    help="also write candidates/<id>/known_records.{json,md} evidence notes")
    args = ap.parse_args(argv)

    cfg = AuditConfig(trees_radius_m=args.trees_radius, other_radius_m=args.other_radius,
                      crs_tolerance_m=args.crs_tolerance_m)
    if args.known_csv:
        known = load_known_table_from_csv(args.known_csv)
    else:
        known = load_known_table(args.known, backend=args.backend, search_root=args.search_root)
    print(f"known table: {len(known.df)} records ({known.stats.get('counts_by_family')}) via {known.backend}")

    if (args.control_lon is None) != (args.control_lat is None):
        ap.error("give both --control-lon and --control-lat")
    if args.control_lonlat and args.control_lon is not None:
        ap.error("give either --control-lonlat or --control-lon/--control-lat, not both")
    if args.control_lonlat or args.control_lon is not None:
        if args.control_lonlat:
            lon, lat = (float(v) for v in args.control_lonlat.split(","))
        else:
            lon, lat = float(args.control_lon), float(args.control_lat)
        res = check_control_point(lon, lat, known, cfg, label=args.control_label)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(args.out_dir) / f"{args.control_label}_record_check.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, sort_keys=True)
        print(json.dumps({k: res[k] for k in ("label", "table_match", "certainty_ceiling_from_table",
                                              "nearest_known_db", "nearest_known_m", "flags")}, indent=2))
        for c in res["candidates"]:
            print(f"  {c['distance_m']:10.1f} m  {c['source_raw']:<12} (radius {c['radius_m']:.0f} m, "
                  f"{'inside' if c['within_threshold'] else 'outside'})")
        print(f"wrote {path}")
        if not args.candidates:
            return 0

    if not args.candidates:
        ap.error("--candidates is required unless only a control point is checked")
    cands = pd.read_csv(args.candidates)
    result = run_audit(cands, known, cfg)
    paths = write_outputs(result, args.out_dir, args.stem, candidate_dirs=args.candidate_dirs)
    if args.write_known_copy:
        known.df.to_csv(Path(args.out_dir) / "known_sites.csv", index=False)
    s = result["summary"]
    print(f"candidates: {s['n_candidates']}  verdicts: {s['counts_by_table_match']}")
    print(f"flags: {s['flag_counts']}")
    for p in paths.values():
        print("wrote", p)
    return 0


def load_known_table_from_csv(path: str | Path) -> KnownTable:
    df = pd.read_csv(path)
    from .known_table import KnownTable as _KT

    stats = {
        "n_rows": int(len(df)),
        "counts_by_family": {k: int(v) for k, v in df["source_family"].value_counts().items()}
        if "source_family" in df else {},
        "loaded_from_csv": True,
    }
    return _KT(df=df, path=Path(path), sha256="(csv)", backend="csv",
               rules_fingerprint="(unknown)", stats=stats,
               warnings=["loaded from a normalised CSV: source rules fingerprint unavailable"])


if __name__ == "__main__":
    raise SystemExit(main())
