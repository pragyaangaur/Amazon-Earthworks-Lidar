# Code in this private bundle

The scripts were run from the project root with the project-local `.venv`. No network access is required by these offline phases except downloading public data.

## Main commands

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'laspy[lazrs]' pyreadr pyproj remotezip pyshp
.venv/bin/python code/download_public_inputs.py --inspect-acre
.venv/bin/python code/inventory_remote_archives.py
.venv/bin/python code/build_queue.py
.venv/bin/python code/vendor_m2/process.py .scratch/amazon/raw/NP_T-0414.laz -o data/derived_v2/0414 --dtm-source hybrid
.venv/bin/python code/vendor_m2/synth.py .scratch/m2_synth
.venv/bin/python code/vendor_m2/process.py .scratch/m2_synth/synth.las -o .scratch/m2_synth_products --dtm-source hybrid
.venv/bin/python code/vendor_m2/validate_synth.py .scratch/m2_synth --out=.scratch/m2_synth_products

# Local known-record footprint/control audit (no network or candidate positions)
.venv/bin/python code/find_known_controls.py
.venv/bin/python code/build_trees_control_evidence.py 0470

# One complete main-screen transect (raw network phase: coordinator only)
.venv/bin/python code/process_one_v3.py 0254 Acre --record-id 7689909 \
  --zip-key 'transectos acre.zip' --member NP_T-0254.laz --state AC

# Bounded sequential queue; line verdicts are automatic, other upgrades are not
.venv/bin/python code/run_queue_v3.py --limit 25 --min-free-gb 3

# Auxiliary Possible candidate evidence (after coordinator review/audit)
.venv/bin/python code/build_aux_candidate.py data/derived_v2/0254 M0254-013
.venv/bin/python code/judge_causeway_candidate.py data/derived_v2/0516 0516-CW001

# Storage lifecycle: dry-run first; only fully Rejected/empty screens are eligible.
# The second command also drops their validated audit caches; candidates/controls are excluded.
.venv/bin/python code/compact_derived_v2.py --all --dry-run
.venv/bin/python code/compact_derived_v2.py --all --drop-screen-state

# Compilation and validation, never certainty promotion
.venv/bin/python code/preflight_compile.py
.venv/bin/python code/compile_results.py --zip
unzip -t amazon_earthworks_results.zip
.venv/bin/python code/references/package_m7/delivcheck.py --zip amazon_earthworks_results.zip --deep --json checkpoint/delivcheck_final.json
.venv/bin/python code/dedupe_results_dtm.py
.venv/bin/python code/link_archive_mirror.py
.venv/bin/python code/verify_archive.py amazon_earthworks_results.zip logs/verify_archive_final.json
.venv/bin/python code/final_deliverable_audit.py amazon_earthworks_results.zip --json logs/final_deliverable_audit.json
```

`process_one_v3.py` performs verified remote-ZIP download, `v3` processing, ring/mound screens, Peripato auditing, native evidence, constrained skeptic diagnostics, causeway screening and mound/causeway audits. Raw LAZ is deleted only after products re-open successfully. Zero detections are valid results. Auxiliary Possible packs use the immobile-mound model and include Jacobs and Nature register checks. `writer validate` and other swam tests are recorded in `code/references/`.

## Validation actually rerun

- m1 synthetic benchmark `SKIP_GEN=1 bash synth/run_all.sh`: six-scene DTM and outlier comparisons.
- m2 production processor: **11/11** checks on 1,174,347 synthetic points.
- m3 causeway self-test and 14-scenario scorer, including explicit embedded-positive hole-span test.
- m4 ring synthetic suite and **6/6** real-DTM digital spike recoveries.
- m5 independent raw-return v3 control audit rerun: exit 0 in 23.6 s, relief null and independent cloth-DTM control reproduced; reconstructed cells are not mislabelled as deleted.
- m6 skeptic **22** unit tests and **36** control rows: 10/10 positives kept, 0/26 false positives.
- Storage compaction verifies exact source/compact array equality and detector-mask counts before deleting intermediate arrays; 0414, 0072, targeted TREES control leads and all candidate transects retain full products.
- m2 long-road benchmark: **30/30** harness checks, byte-identical fixed/held-out evaluator reruns, and decisive legacy P7 coverage 0.98895 with zero screened false metres over 14.09 km². See `code/scripts/references/road_m2/COORDINATOR_REVIEW.md`; overall road-screen gates still fail and no transect is done.
- m3 detB: byte-identical real 0414 adapter output plus an independently rerun real-DTM spike test (three planted classes recall 1.00 at coverage >=0.5, zero accepted null rows). See `code/scripts/references/road_m3/COORDINATOR_REVIEW.md`; mask, embankment, modern-flag and real/forest validation remain open.
- m1 Track A: fixed/held-out/legacy evaluations reproduced byte-for-byte; corrected own-scorer recalls cal 0.7111 and hold 0.6889 independently rerun. Accepted only as measured negative/partial evidence because valley/terrace confounders fail; see `code/scripts/references/road_m1/COORDINATOR_REVIEW.md`, including the H3 579.85 m (not 589 m) erratum.
- m4 real-substrate review: six mechanical checks plus P7 recovery (336.919 m, azimuth error 0.4267°, zero null-twin rows), independent south-up anchors and an exact finite-gate 0414 audit reproduced. Its original combined open-gate rerun remained unfinished at that review; see `code/scripts/references/road_m4/COORDINATOR_REVIEW.md`, including the coordinate-convention qualification.
- m1 bounded detB follow-up: the finite control is independently byte-identical, and six current open rows reproduce 11.755044 km² / 337 candidates / 0 accepted. Independent open-0414 output is byte-identical to the member's single-scene JSON and exactly matches the accepted 0414 subset; the other five are member detector reruns checked at artifact/gate level. Rows 0905, 0918, 0862 and 0413 remain INPUT_CHANGED because their canopy rasters were removed. The approximately 47.7 h full-899 cost is an extrapolation, not a completed run. See `code/scripts/references/detb_open_m1/COORDINATOR_REVIEW.md`; this is not production-road or under-forest validation.
- m8 writer final audit **66** tests passed; its output was not adopted because its `partial` status is outside the attachment's `done`/`failed`/`skipped` vocabulary.
- m7 record checker: all seven offline test scripts passed, including existential-radius regression, three-backend RDS parity, geodesy against PROJ/Karney, footprint proximity, KML source discrimination and a substantive audit of the current candidate/rejected rows. Evidence and coordinator scope: `code/scripts/references/record_m7/COORDINATOR_REVIEW.md`.

Source hashes and provenance are retained in `code/vendor_*/SOURCE.json` where applicable. Candidate locations and raw L1A payloads are never committed to git.

## Policy

No feature is upgraded by compilation. Auxiliary audit completion requires an independent verdict for every raised linear feature before packaging. The long sunken-road method remains unvalidated; therefore `done_full_screen` is zero, not 899.
