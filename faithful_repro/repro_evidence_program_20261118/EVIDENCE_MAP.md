# EVIDENCE_MAP — truy vết bằng chứng → claim (REV4)

| Claim | Artifact | Verifier | Status |
|---|---|---|---|
| Strict tensor = 411×271×5, 11,748 links, 8,735 pairs, density 10.5476% | strict npz sha 85bb9ea4 | `metric_feasibility_check.py` re-measure | VERIFIED |
| Density "4.6%" claim | n/a (no source) | `DENSITY_ERRATUM.md` | RETRACTED |
| Strict tensor khớp paper summary stats (shape/links/per-type) | DATASET_IDENTITY_EVIDENCE.md | measured vs paper table | L1 MATCH |
| Oracle macro-recall bound 0.8263 < paper R 0.9421 | `metric_feasibility_check.py` | deterministic fixture PASS | VERIFIED |
| k4 release metric bỏ hết v3.2 | `test_metric_contracts.py` | 11/11 PASS | VERIFIED |
| pair-fold ≠ triplet-fold | `test_split_contracts.py` | 3/3 PASS | VERIFIED |
| U1 outer-test 0.4558/0.4924 vs prior 0.448 | RESULTS_CROSSWALK.csv + RESCORE_VERIFICATION | 50/50 match | VERIFIED (retrospective) |
| Leakage inflate v2.0 +0.049 / v3.2 ~0 | paired diag_leak results JSON | same config/code/seed/folds | VERIFIED (1 seed) |
| Conventions đảo thứ hạng model-vs-prior | METRIC_BRIDGE.json | `metric_bridge.py` 50 rows | VERIFIED |
| Loss denominators khác author | `test_fidelity_microfixtures.py` | 6/6 PASS | VERIFIED |
| Filter paper <2/<3 | boundary stratum measure | DATASET_IDENTITY_EVIDENCE §L2 | UNRESOLVED |
| Author v3.2 eval input byte-identical | không artifact | AUTHOR_ARTIFACT_REQUEST_READY | EXTERNAL_REQUIRED |
| Upstream mới | git ls-remote 2026-11-19 | main@06f9ba3 unchanged | VERIFIED no-change |
