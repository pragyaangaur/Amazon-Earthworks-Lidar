#!/usr/bin/env python3
"""Dry-run code/compile_results.py against the live derived products, writing nothing to production.

The real compiler is imported and executed unchanged except for three
redirections:

  * ROOT  -> a sandbox directory whose data/, checkpoint/ and code/ are
             read-only symlinks to the project, and whose progress/ is a copy
             (the compiler rewrites progress/*.md inside ROOT);
  * OUT   -> sandbox/results;
  * shutil.copy2/copytree -> size-saving shims that skip only the bulk DTM
             copy into OUT/dtm (818 MB now, ~1.06 GB when the queue is done).

Everything else - every validation RuntimeError, every CSV row, every count in
outcome.json - is produced by the production code path. So this answers "will
the next real checkpoint compile, and what will it say" without touching
results/, progress/ or the private archive.

Usage (from the project root; use the project venv, the compiler needs pandas/pyproj):
  .venv/bin/python code/preflight_compile.py
  .venv/bin/python code/preflight_compile.py --keep

Exit codes: 0 dry run succeeded, 1 the compiler refused (message printed),
2 setup problem.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


class RecordingShim:
    """shutil proxy: real behaviour except bulk copies into OUT/dtm."""

    def __init__(self, module, skipped, out_dtm: Path, max_tree_bytes: int = 200 << 20):
        self._real = shutil
        self._module = module
        self.skipped = skipped
        self.out_dtm = out_dtm
        self.max_tree_bytes = max_tree_bytes

    def copy2(self, src, dst, *args, **kwargs):
        src, dst = Path(src), Path(dst)
        try:
            same_parent = dst.parent.resolve() == self.out_dtm.resolve()
        except OSError:
            same_parent = False
        if same_parent:
            self.skipped.append({"op": "copy2", "src": str(src), "bytes": src.stat().st_size})
            return dst
        return self._real.copy2(src, dst, *args, **kwargs)

    def copytree(self, src, dst, *args, **kwargs):
        total = 0
        for dirpath, _, files in os.walk(src):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        if total > self.max_tree_bytes:
            self.skipped.append({"op": "copytree", "src": str(src), "bytes": total})
            Path(dst).mkdir(parents=True, exist_ok=True)
            return Path(dst)
        return self._real.copytree(src, dst, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def build_sandbox(root: Path, sandbox: Path) -> None:
    if sandbox.exists():
        shutil.rmtree(sandbox)
    (sandbox / "results").mkdir(parents=True)
    for name in ("data", "checkpoint", "code"):
        os.symlink(root / name, sandbox / name)
    os.symlink(root / "literature.md", sandbox / "literature.md")
    shutil.copytree(root / "progress", sandbox / "progress")


def load_compiler(root: Path):
    spec = importlib.util.spec_from_file_location("compile_results_under_test",
                                                 root / "code/compile_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counts(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {"rows": sum(1 for _ in csv.DictReader(handle))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--sandbox", default=".scratch/preflight_compile")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox for inspection")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sandbox = Path(args.sandbox).resolve()
    if not (root / "code/compile_results.py").exists():
        print(f"not a project root: {root}", file=sys.stderr)
        return 2

    build_sandbox(root, sandbox)
    module = load_compiler(root)
    module.ROOT = sandbox
    module.OUT = sandbox / "results"
    module.PROD = sandbox / "data/derived_v2"
    skipped: list[dict] = []
    module.shutil = RecordingShim(module, skipped, module.OUT / "dtm")

    print(f"preflight root   : {sandbox}")
    print(f"preflight products: {module.PROD}")
    argv, sys.argv = sys.argv, [str(root / "code/compile_results.py")]
    try:
        module.main()
    except SystemExit as exc:
        print(f"compiler exited: {exc}")
        return 1
    except Exception:
        print("COMPILER REFUSED THE DRY RUN:")
        traceback.print_exc()
        return 1
    finally:
        sys.argv = argv

    dry = sandbox / "results"
    outcome_dry = json.loads((dry / "outcome.json").read_text())
    outcome_live = json.loads((root / "results/outcome.json").read_text())
    print("\nskipped bulk copies: %d (%d MB)"
          % (len(skipped), sum(x["bytes"] for x in skipped) // 10 ** 6))
    print("\noutcome.json: live vs dry run")
    keys = ["transects_total", "processed_to_dtm_and_main_screens", "done_full_screen",
            "failed", "skipped_or_incomplete", "rejected_features", "features_examined"]
    for key in keys:
        print(f"  {key:38s} {str(outcome_live.get(key)):>8s} -> {str(outcome_dry.get(key)):>8s}")
    for key in ("Known", "Strong", "Possible"):
        print(f"  candidates.{key:29s} {str(outcome_live.get('candidates', {}).get(key)):>8s}"
              f" -> {str(outcome_dry.get('candidates', {}).get(key)):>8s}")
    print("\nCSV row counts: live vs dry run")
    for name in ("candidates.csv", "rejected.csv", "transects.csv"):
        print(f"  {name:20s} {counts(root / 'results' / name).get('rows')!s:>8s}"
              f" -> {counts(dry / name).get('rows')!s:>8s}")
    live_ids = {row["id"] for row in csv.DictReader((root / "results/candidates.csv").open(newline="", encoding="utf-8-sig"))}
    dry_ids = {row["id"] for row in csv.DictReader((dry / "candidates.csv").open(newline="", encoding="utf-8-sig"))}
    if dry_ids - live_ids:
        print("  new candidate ids in dry run:", sorted(dry_ids - live_ids))
    if live_ids - dry_ids:
        print("  WARNING candidate ids that would disappear:", sorted(live_ids - dry_ids))
    print("\nPREFLIGHT OK: the real compile would succeed on the current tree.")
    if not args.keep:
        shutil.rmtree(sandbox)
        print("sandbox removed (pass --keep to inspect it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
