# REUSE_MATRIX — Track P5 reuse vs rebuild inventory

Rule: reuse existing code/tests where they exist; `NO_DUPLICATE_TRAINING`
for already-run configurations; rebuild only where evidence shows a gap.

| Capability | Existing artifact (path) | Reuse decision |
|---|---|---|
| HGCN operator Eq.4 (cosine KNN + Dv^-1/2 H W De^-1 H^T Dv^-1/2) | `code/faithful.py::knn_G` (L81-93) | REUSE as implementation-under-test; P5 tests carry an independent oracle |
| Standard GCN fallback (ablation) | `code/faithful.py::gcn_G` (L96-106) | REUSE; ablation definition only |
| Dual-view HGCN + GRACE intra-CL | `code/faithful.py::CLHGCN` (L118-140) | REUSE; denominators tested against independent reference |
| Inter-view InfoNCE + margin | `code/faithful.py::inter_loss` (L143-153) | REUSE; sampling/RNG documented as interpretation |
| AVF gate + learnable beta | `code/faithful.py::AVF` (L156-164) | REUSE; gate shape is declared assumption |
| Reconstruction Gram head | `code/faithful.py::Recon` (L167-173) | REUSE; width d→d/2 is a choice, flagged in decision register |
| Predictor diag/full bilinear + softmax type head | `code/faithful.py::Predictor` (L176-202) | REUSE; softmax-multihot CE locked as named convention |
| HGT block (2L/4heads, 4 relations) | `code/faithful.py::HGTBlock`+`META` (L205-215) | REUSE; edge directions tested |
| Dynamic refresh (Sec 2.8) | `code/faithful.py` L406-412 | AUDIT: post-refresh `G[m1]==G[m2]` — see T1 in decision register; test behavior, not silently change |
| GIP kernel | `code/faithful.py::gip` (L76-78) | REUSE for synthetic tests; provenance gate decides admissibility on real data |
| Pair/triplet split builder | `code/faithful.py::make_splits` (L300-322) | REUSE for protocol semantics; locked fold manifests are the actual evaluation partitions |
| top1 harmonic + binary metrics | `code/faithful.py::top1`,`binary` (L260-278) | REUSE only as `RELEASE_CONVENTION` family; P5 primary is `MULTIHOT_MACRO_F1` (track_n contract) |
| Locked multihot macro-F1 + firstpos harmonic | `track_n/metrics_g2c.py::metrics_for` | REUSE formulas as P5 contract reference (thresholds inner-val only) |
| Strong baselines (prior/logistic/labelprop) | `track_n/baselines.py` + `run_g2c.py` + `runs_g2c/` | REUSE for later G6 comparison ONLY if split/feature/metric identical; `NO_DUPLICATE_TRAINING` |
| Cold-start splits | `track_n/splits/*` + `build_coldstart_splits.py` | REUSE if entity-cold questions preregistered later |
| Old smoke tests | `code/test_faithful.py` | NOT portable (absolute `~/dhg_runs`, `~/repos/DHGCMDA` paths); superseded by `track_p5/tests/` — do not run as acceptance |
| Strict tensor + locked folds | `data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`; `independent_v32_outer_refit_20261117/evaluation/fold*_static.npz` | REUSE (hash-pinned in SOURCE_MANIFEST) |
| Four-view provenance classes | `independent_repro_forensics_20261119/FOUR_VIEW_PROVENANCE_CORRECTED.md` | REUSE: 3-layer model (raw inputs / GIP-augmented IM·ID / full-A features) adopted as FEATURE_PROVENANCE_DAG vocabulary |
| Joint P/R + aggregation results | `independent_repro_forensics_20261119/` | REUSE as closed evidence; no re-litigation |
| Run/training driver | `code/faithful.py::run_fold`,`main` | DO NOT EXECUTE now — audit only (G5+ requires owner approval) |

## Explicitly NOT reused

- `runs_g2/` (marked SUPERSEDED_UNVERIFIED) — pre-repair baselines, frozen history only.
- Any test-fitted threshold/scaler — P5 refits on inner-val only.
- `MODEL_LOSS_FIDELITY_TRACE` verdicts as test oracles — tests use
  independent reference implementations, not the code under test.
