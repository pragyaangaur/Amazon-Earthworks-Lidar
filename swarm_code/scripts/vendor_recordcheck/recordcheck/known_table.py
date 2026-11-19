"""Loader and validator for the local known-earthwork table.

The table is the Peripato et al. 2023 data release: ``Database/Earthworks.rds``
holding 961 records with columns ``Longitude``, ``Latitude`` and ``Database``
(the register each record comes from).  Nothing here touches the network.

Three interchangeable backends, tried in this order by ``backend='auto'``:

  1. ``pyreadr``   fastest; present in the repo ``.venv`` when it exists.
  2. ``R``         base-R ``readRDS`` -> temporary CSV (R 4.5.3 lives in /opt/math).
  3. ``pure``      ``recordcheck.rds``, a dependency-free RDS reader, so the
                   check still runs in a bare python3 environment.

Every load records the file SHA-256, the backend used, the row/column names and
the hash of the source-classification rules in ``known_sites_meta.json``.  A
distance reported by the audit is therefore always traceable to the exact file
and the exact rule set that produced it.

Source classification decides the match radius (2000 m for TREES/INPE, 500 m for
every other register).  Unrecognised labels are flagged, not guessed silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import geodesy as G

RULES_PATH = Path(__file__).with_name("source_rules.json")

LON_NAMES = ("longitude", "long", "lon", "x", "longitude_deg", "lon_deg")
LAT_NAMES = ("latitude", "lat", "y", "latitude_deg", "lat_deg")
SRC_NAMES = ("database", "source", "db", "origem", "fonte", "register")

# Brazilian Amazon envelope, deliberately generous; used only to flag records,
# never to drop them.
AMAZON_BBOX = (-80.0, -34.0, -25.0, 8.0)  # lon_min, lon_max, lat_min, lat_max


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# source classification
# --------------------------------------------------------------------------
def load_rules(path: str | Path = RULES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        rules = json.load(fh)
    compiled = {}
    for fam, spec in rules["families"].items():
        compiled[fam] = {
            "radius_m": float(spec["radius_m"]),
            "patterns": [re.compile(p, re.IGNORECASE) for p in spec["patterns"]],
            "note": spec.get("note", ""),
        }
    rules = dict(rules)
    rules["_compiled"] = compiled
    return rules


def rules_fingerprint(rules: dict) -> str:
    """SHA-256 of the rule set without the compiled regex objects."""
    payload = {
        "families": {
            fam: {"radius_m": spec["radius_m"], "patterns": spec.get("patterns", [])}
            for fam, spec in rules["families"].items()
        },
        "default_family": rules.get("default_family", "unresolved"),
        "default_radius_m": rules.get("default_radius_m", 500),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def classify_source(raw: object, rules: dict) -> tuple[str, float, bool]:
    """Map a raw ``Database`` label to (family, radius_m, unresolved_flag)."""
    text = "" if raw is None else str(raw).strip()
    if isinstance(raw, float) and np.isnan(raw):
        text = ""
    for fam, spec in rules["_compiled"].items():
        for pat in spec["patterns"]:
            if pat.search(text):
                return fam, float(spec["radius_m"]), False
    return rules.get("default_family", "unresolved"), float(rules.get("default_radius_m", 500)), True


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def _pick(columns, candidates) -> str | None:
    low = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # tolerate punctuation/spacing differences
    squash = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand)
        if key in squash:
            return squash[key]
    return None


def _read_pyreadr(path: Path) -> pd.DataFrame:
    import pyreadr  # noqa: F401

    result = pyreadr.read_r(str(path))
    if len(result) != 1:
        # an .rds holds exactly one object; be explicit if that changes
        keys = list(result.keys())
        if len(keys) != 1:
            raise ValueError(f"pyreadr returned {len(keys)} objects for {path}: {keys}")
    return next(iter(result.values()))


def _find_rscript() -> str | None:
    for cand in (shutil.which("Rscript"), "/opt/math/bin/Rscript", "/usr/bin/Rscript"):
        if cand and os.path.exists(cand):
            return cand
    return None


def _read_r(path: Path) -> pd.DataFrame:
    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found")
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "earthworks.csv"
        code = (
            f"d <- readRDS({json.dumps(str(path))});"
            " stopifnot(is.data.frame(d));"
            f" write.csv(d, {json.dumps(str(out_csv))}, row.names=FALSE, na='')"
        )
        proc = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"Rscript failed: {proc.stderr.strip()[:800]}")
        return pd.read_csv(out_csv)


def _read_pure(path: Path) -> pd.DataFrame:
    from . import rds

    value = rds.read_rds(path)
    return rds.to_dataframe(value)


BACKENDS = {"pyreadr": _read_pyreadr, "R": _read_r, "pure": _read_pure}


def _read_with_backend(path: Path, backend: str) -> tuple[pd.DataFrame, str]:
    order = ["pyreadr", "R", "pure"] if backend == "auto" else [backend]
    errors: list[str] = []
    for name in order:
        try:
            return BACKENDS[name](path), name
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no RDS backend worked for " + str(path) + " :: " + " | ".join(errors))


# --------------------------------------------------------------------------
# table object
# --------------------------------------------------------------------------
@dataclass
class KnownTable:
    df: pd.DataFrame
    path: Path
    sha256: str
    backend: str
    rules_fingerprint: str
    source_columns: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    # ---- convenience -----------------------------------------------------
    @property
    def lon(self) -> np.ndarray:
        return self.df["longitude"].to_numpy(dtype=float)

    @property
    def lat(self) -> np.ndarray:
        return self.df["latitude"].to_numpy(dtype=float)

    @property
    def valid_mask(self) -> np.ndarray:
        return self.df["valid_coord"].to_numpy(dtype=bool)

    def valid(self) -> "KnownTable":
        sub = self.df[self.valid_mask].reset_index(drop=True)
        return KnownTable(sub, self.path, self.sha256, self.backend, self.rules_fingerprint,
                          self.source_columns, self.warnings, self.stats)

    def family_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for fam in sorted(self.df["source_family"].unique()):
            mask = (self.df["source_family"] == fam).to_numpy()
            out[fam] = mask
        return out

    def meta(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "backend": self.backend,
            "rows": int(len(self.df)),
            "columns_found": self.source_columns,
            "rules_fingerprint": self.rules_fingerprint,
            "stats": self.stats,
            "warnings": self.warnings,
            "generated_by": "recordcheck.known_table",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }


def _near_duplicate_stats(df: pd.DataFrame, thresholds=(1.0, 10.0, 50.0)) -> dict:
    """Count record pairs closer than thresholds (metres, ellipsoidal)."""
    if len(df) < 2:
        return {f"pairs_within_{int(t)}m": 0 for t in thresholds}
    d = G.pairwise_distance(df["latitude"], df["longitude"], df["latitude"], df["longitude"])
    iu = np.triu_indices(len(df), k=1)
    vals = d[iu]
    vals = vals[np.isfinite(vals)]
    out = {f"pairs_within_{int(t)}m": int((vals <= t).sum()) for t in thresholds}
    out["min_separation_m"] = float(vals.min()) if vals.size else None
    return out


def build_table(df: pd.DataFrame, path: Path, sha256: str, backend: str, rules: dict,
                extra_columns: dict | None = None) -> KnownTable:
    """Normalise a raw data.frame into the canonical known-site table."""
    warnings: list[str] = []
    cols = list(df.columns)
    lon_col = _pick(cols, LON_NAMES)
    lat_col = _pick(cols, LAT_NAMES)
    src_col = _pick(cols, SRC_NAMES)
    if lon_col is None or lat_col is None:
        raise ValueError(f"could not find longitude/latitude columns among {cols}")
    if src_col is None:
        warnings.append(f"no source column found among {cols}; all records treated as 'unresolved'")
    out = pd.DataFrame({
        "record_id": np.arange(1, len(df) + 1, dtype=int),
        "longitude": pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float),
        "latitude": pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float),
        "database_raw": (df[src_col].astype("object").to_numpy() if src_col is not None
                         else np.full(len(df), "", dtype=object)),
    })
    out["database_raw"] = [("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v).strip())
                           for v in out["database_raw"]]
    fams = [classify_source(v, rules) for v in out["database_raw"]]
    out["source_family"] = [f[0] for f in fams]
    out["radius_m"] = [f[1] for f in fams]
    out["source_unresolved"] = [f[2] for f in fams]

    finite = np.isfinite(out["longitude"]) & np.isfinite(out["latitude"])
    in_range = finite & (out["latitude"].abs() <= 90.0) & (out["longitude"].abs() <= 360.0)
    lon_min, lon_max, lat_min, lat_max = AMAZON_BBOX
    in_amazon = in_range & out["longitude"].between(lon_min, lon_max) & out["latitude"].between(lat_min, lat_max)
    out["valid_coord"] = in_range
    out["in_amazon_bbox"] = in_amazon
    n_bad = int((~in_range).sum())
    n_outside = int((in_range & ~in_amazon).sum())
    if n_bad:
        warnings.append(f"{n_bad} records with missing/non-finite/out-of-range coordinates (kept, flagged)")
    if n_outside:
        warnings.append(f"{n_outside} records inside valid lat/lon ranges but outside the generous Amazon "
                        f"bbox {AMAZON_BBOX} (kept, flagged)")
    n_unres = int(out["source_unresolved"].sum())
    if n_unres:
        warnings.append(f"{n_unres} records with an unrecognised Database label: "
                        f"{sorted(set(out.loc[out['source_unresolved'], 'database_raw']))[:12]} "
                        "(evaluated with the 500 m default, flagged)")

    if extra_columns:
        for name, values in extra_columns.items():
            if name not in out.columns:
                out[name] = values

    stats = {
        "n_rows": int(len(out)),
        "n_valid_coord": int(in_range.sum()),
        "n_in_amazon_bbox": int(in_amazon.sum()),
        "n_source_unresolved": n_unres,
        "n_distinct_database_labels": int(out["database_raw"].nunique()),
        "database_labels": sorted(out["database_raw"].unique().tolist()),
        "counts_by_family": {k: int(v) for k, v in out["source_family"].value_counts().items()},
        "counts_by_raw_label": {k: int(v) for k, v in out["database_raw"].value_counts().items()},
        "bbox": {
            "lon_min": float(np.nanmin(out["longitude"])) if in_range.any() else None,
            "lon_max": float(np.nanmax(out["longitude"])) if in_range.any() else None,
            "lat_min": float(np.nanmin(out["latitude"])) if in_range.any() else None,
            "lat_max": float(np.nanmax(out["latitude"])) if in_range.any() else None,
        },
        "exact_duplicate_rows": int(out.duplicated(subset=["longitude", "latitude"]).sum()),
    }
    stats.update(_near_duplicate_stats(out[in_range].reset_index(drop=True)))
    return KnownTable(out, path, sha256, backend, rules_fingerprint(rules),
                      {"longitude": lon_col, "latitude": lat_col, "database": src_col},
                      warnings, stats)


# --------------------------------------------------------------------------
# discovery + entry point
# --------------------------------------------------------------------------
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}


def find_known_table(root: str | Path, names=("earthworks.rds", "earthworks.rda", "known_sites.rds"),
                     max_depth: int = 8) -> list[Path]:
    """Search a local tree for candidate RDS tables (no network, no symlink chase)."""
    root = Path(root)
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_depth = len(Path(dirpath).relative_to(root).parts)
        if rel_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.lower() in names:
                hits.append(Path(dirpath) / fn)
    return sorted(hits)


def load_known_table(path: str | Path | None = None, backend: str = "auto",
                     rules_path: str | Path = RULES_PATH,
                     search_root: str | Path | None = None) -> KnownTable:
    """Load the known-earthwork table.

    ``path=None`` searches ``search_root`` (default: three levels above this
    file, i.e. the project repo) for ``Earthworks.rds``.
    """
    rules = load_rules(rules_path)
    if path is None:
        root = Path(search_root) if search_root else Path(__file__).resolve().parents[4]
        hits = find_known_table(root)
        if not hits:
            raise FileNotFoundError(
                f"no Earthworks.rds under {root}; pass --known <path> (the coordinator places "
                "Database/Earthworks.rds in the workspace)")
        if len(hits) > 1:
            preferred = [h for h in hits if "earthworks" in h.name.lower() and "database" in str(h).lower()]
            hits = preferred or hits
        path = hits[0]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sha = sha256_file(path)
    df, used = _read_with_backend(path, backend)
    return build_table(df, path, sha, used, rules)


def write_known_csv(table: KnownTable, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.df.to_csv(out, index=False)
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="load/validate the local known-earthwork table")
    ap.add_argument("--known", default=None, help="path to Earthworks.rds")
    ap.add_argument("--search-root", default=None, help="root to search when --known is absent")
    ap.add_argument("--backend", default="auto", choices=["auto", "pyreadr", "R", "pure"])
    ap.add_argument("--out", default="out/known_sites.csv")
    ap.add_argument("--meta", default="out/known_sites_meta.json")
    args = ap.parse_args(argv)

    table = load_known_table(args.known, backend=args.backend, search_root=args.search_root)
    p1 = write_known_csv(table, args.out)
    meta = table.meta()
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    with open(args.meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f"loaded {meta['rows']} records from {meta['path']} via {meta['backend']}")
    print(f"sha256 {meta['sha256']}")
    print(f"families: {meta['stats']['counts_by_family']}")
    print(f"raw labels: {meta['stats']['counts_by_raw_label']}")
    for w in meta["warnings"]:
        print("WARNING:", w)
    print(f"wrote {p1} and {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
