# EXISTING_RESULTS_REEVALUATION — rescore of stored predictions (no retraining)

WP4 output. Everything recomputed from row-level `pairs/y/logits` by
`analysis_rescore.py` (50/50 checks PASS — rescore reproduces the
originally reported macro_f1 exactly, and priors rebuilt from fold
counts match reported baseline metrics to 1e-9).
See `RESCORE_VERIFICATION.json`, rows in `RESULTS_CROSSWALK.csv`.

## 1. Pair-fold outer-test U1 (E08), rescored under distinct metrics

Mean ± SD over the 15 fold×seed cells (correlated; NOT independent):

| Arm | GENERAL_K_TOP1_HARMONIC (firstpos) | (anyhit) | PER_CLASS_MACRO_F1 | MICRO_F1 | ANY_HIT |
|---|---:|---:|---:|---:|---:|
| U1 dynamic | 0.4558 ± 0.026 | 0.4589 | 0.4045 | 0.4349 | 0.5099 |
| U1 static | 0.4924 ± 0.011 | 0.4936 | 0.4403 | 0.4703 | 0.5514 |
| prior_global | 0.2497 | 0.2783 | 0.1256 | 0.3903 | 0.4576 |
| prior_disease | 0.4241 | 0.4351 | 0.3631 | 0.5539 | 0.6495 |
| prior_mirna | 0.2835 | 0.3180 | 0.2582 | 0.4115 | 0.4824 |
| prior_product_laplace1 | 0.4480 | 0.4680 | 0.4296 | 0.5385 | 0.6314 |

Facts:
- The paper-family number (`GENERAL_K_TOP1_HARMONIC`, first-positive
  convention) is ~0.46–0.49 — NOT the previously quoted 0.40/0.44
  macro-F1. Metric definition moves the headline by ~0.05.
- Static still beats dynamic in 14/15 paired cells under every
  metric variant; product prior beats both arms on micro/any-hit.
- Under the harmonic aggregate the gap U1-vs-product shrinks to
  ~0.01–0.04 — the model's advantage is concentrated in macro-F1.

## 2. Author-release v2.0 (E01)

Raw per-row predictions are UNAVAILABLE_LOCAL (fold_*.npz left on
Windows; SHA in evidence_manifest). Reported figures stand as
documented: author Top-1 0.5568 (vs paper's v2.0 0.5970),
macro-F1 0.5226 — all on a pipeline with full-association features
in inputs (oracle leakage demonstrated). Rescore without artifact:
`NOT_RESCORABLE_WITHOUT_ARTIFACT`. No v2.0↔v3.2 comparison drawn.

## 3. Crosswalk of historical numbers (different questions!)

| Number | Dataset | Split unit | Metric | Model/selection | Exposure |
|---|---|---|---|---|---|
| 0.8600 | paper v3.2 (unreleased) | unknown | unknown (harmonic family) | unknown | — |
| 0.5568 | shipped v2.0 | pair | AUTHOR_RELEASE_TOP1_HARMONIC k4 | author pipeline, leaked inputs | leaked |
| 0.6350 | shipped v2.0 | pair | same | full-bilinear fork | leaked |
| 0.2682–0.3620 | wang v3.2 tensor | pair | corrected general-K | diag/full-bilinear, 300–650ep | leaked inputs |
| 0.4558–0.4924 | strict v3.2 | pair (disjoint) | GENERAL_K_TOP1_HARMONIC | U1 outer refit fixed-650 | post-dev |
| 0.4045–0.4403 | strict v3.2 | pair (disjoint) | PER_CLASS_MACRO_F1 | same | post-dev |
| 0.4370–0.4427 | strict v3.2 | pair, inner-val only | macro-F1 best-epoch | cap1/cap10 | dev |
| 0.8548 AUROC | strict v3.2 | TRIPLET | AUROC binary | U1X CVtriplet | post-dev |

These are NOT a trajectory; they differ in tensor, unit, metric,
exposure and selection rule. No single "current F1" exists.

## 4. What can and cannot be inferred

- Paired static-vs-dynamic on identical folds/seeds: static ≥ dynamic
  consistently → under our config the refresh machinery does not help.
  Cannot infer the paper's dynamic update is harmful (different data,
  G update path, exposure).
- Priors competitive on aggregate metrics → the pair-fold task on
  this tensor carries strong degree priors; model's residual value
  shows mostly in macro-F1 on minority types.
- All cells share development history → no p-values, no blind-test
  claims.

## 5. Missing artifacts (stopped branches)

- v2.0 raw predictions: NOT_RESCORABLE_WITHOUT_ARTIFACT.
- Any author v3.2 fold/predictions: EXTERNAL_ARTIFACT_REQUIRED.
