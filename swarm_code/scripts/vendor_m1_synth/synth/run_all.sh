#!/usr/bin/env bash
# Regenerate the whole m1 synthetic benchmark from scratch.
#   bash synth/run_all.sh [MEDIAN_METHODS] [EXTRA_METHODS]
# Safe to re-run: every step is deterministic for a given --seed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"          # repo root (holds .venv)
VP="$ROOT/.venv/bin/python"
cd "$HERE/.."
SEED="${SEED:-1234}"
t0=$(date +%s)
SKIP_GEN="${SKIP_GEN:-0}"
if [ "$SKIP_GEN" = "1" ]; then echo "== 1-2b skipped (SKIP_GEN=1) =="; else
echo "== 1. generate scenes (seed $SEED) =="
"$VP" synth/gen.py --scene all --seed "$SEED" --quiet
echo "== 2. build reference products per method =="
for scene in s1_ring40_open s2_natneg s3_mound_village s4_forest_stress \
             s5_holehalo s6_transect_0414sim; do
  for m in vstruct median literal none; do
    echo "-- $scene / $m"
    "$VP" synth/refdtm.py "out/$scene.laz" --method "$m" \
        --out "out/${scene}__$m" --png --truth
  done
done
echo "== 2b. ceiling products from the noiseless truth DTM =="
for scene in s1_ring40_open s2_natneg s3_mound_village s4_forest_stress \
             s5_holehalo s6_transect_0414sim; do
  "$VP" synth/ceiling.py "out/${scene}_truth.npz" --out "out/${scene}__ceiling" --png
done
fi
echo "== 3. acceptance metrics =="
for scene in s1_ring40_open s2_natneg s3_mound_village s4_forest_stress \
             s5_holehalo s6_transect_0414sim; do
  NF=""
  case "$scene" in s2_natneg|s4_forest_stress|s6_transect_0414sim) NF="--noise-floor";; esac
  for m in vstruct median literal none ceiling; do
    "$VP" synth/metrics.py --dtm "out/${scene}__$m/${scene}_dtm.npz" \
        --truth "out/${scene}_truth.json" $NF --profiles
  done
done
echo "== 4. DTM vs truth =="
"$VP" synth/compare.py --dtm 'out/*/*_dtm.npz' --truth 'out/*_truth.npz' \
    --out out/benchmark_compare.csv
echo "== 4b. below-ground outlier dose-response =="
"$VP" synth/outlier_dose.py --scene s1_ring40_open --out out/outlier_dose.json
echo "== 5. summary table =="
"$VP" synth/summarise.py --out out
echo "done in $(( $(date +%s) - t0 )) s"
