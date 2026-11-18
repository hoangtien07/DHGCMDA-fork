# DUPLICATION_MATRIX — same question asked by more than one artifact

Purpose: a renamed experiment is not new evidence. Rows group artifacts
that share data + fold membership + config family + target metric.

| Group | Artifacts | Shared | Distinct | Verdict |
|---|---|---|---|---|
| D1 inner-val U1 cap10 | E06 (5fold3seed), E07 (seed_stability), E09 (bank1 Windows partial), fold0 seed1234 in E10 bundles | same bundles, same fold partition, same cfg cap10, same metric | seeds/OS coverage | One evidence family; E06 is the canonical complete set. E09 is a partial re-run of the same cells (stopped by user) — do not count twice. |
| D2 cap1 arm | E10 (bank1_linux) | same bundles/folds as D1 | cap1 negative bank only | New arm, not duplicate; comparison vs cap10 is paired per fold/seed. |
| D3 outer-test U1 | E08 (outer_refit dyn+static) | same strict tensor, pair folds (group candidates) | arm dynamic vs static; fixed-650 | Canonical pair-fold outer-test evidence; priors inside same package. No other outer-test export exists → rescore here, no retrain. |
| D4 CVtriplet | E11 (triplet_u1 U1X + 2 logistic pools) | triplet split family rng(0) | model vs baselines | Canonical triplet evidence; distinct task from D3. |
| D5 v2.0 author lineage | E01 (+controls E02–E04, aborted E05), E12 (j1 bilinear) | shipped v2.0 data+folds | de-leakage controls, predictor variant | E01 baseline; E02–E04 are sequential single-change controls on the same folds — complementary, not duplicates; all share leaked-input lineage caveat. |
| D6 v3.2 wang track | E13 | wang tensor (different entity list from strict) | metric fixed, dense control | Different tensor → cannot merge with D3; label OUR_INTERPRETATION. |
| D7 audits/diagnostics | E14, E15, E17, E18 | — | — | Non-scoring evidence; no duplication issue. |

## Duplicate claims to avoid in WP4/WP7

- Do not average E09 cells into E06 (same cells, partial re-run).
- Do not report E06 inner-val numbers and E08 outer-test numbers as the
  same metric — different evaluation unit (inner-val vs outer-test)
  AND different selection rule (best-epoch vs fixed-650).
- Do not report E11 Task-T AUROC as comparable to CVtype F1.
- fold0 seed1234 inner-val predictions exist in E06 and inside E10's
  run dir — same cell; cite once.
