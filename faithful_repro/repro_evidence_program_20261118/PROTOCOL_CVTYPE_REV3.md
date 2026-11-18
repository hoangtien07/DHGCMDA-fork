# PROTOCOL_CVTYPE_REV3 — the default reproduction question

Status: locked BEFORE any further training. All names use the
WP1 contract vocabulary (METRIC_SEMANTICS.md, SPLIT_SEMANTICS.md).

## Question (single, default)

> On the pinned strict paper-first v3.2 tensor, under PAIR_CVTYPE
> (group-level pair folds, all types of a pair held out together),
> what does U1 (DHGCMDA-family model, adaptation declared) score
> versus label-unaware count priors — when neither model nor
> fold-bundled feature rows observe any held-out test pair's
> existence or type, under declared labels, seeds, epochs, and
> selection rule, using the pinned general-K top-1 harmonic F1 as
> primary metric?

## Locks (all pre-registered)

- DATASET_LOCK: `faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
  sha256 `85bb9ea4…` — elementwise-identical to reference 135e2050.
- FOLD_LOCK: group-disjoint pair folds; existing family verified
  legal (WP1 `test_real_outer_refit_pair_disjoint`: test∩train=∅,
  test∩val=∅ for all 30 eval files; seeds 1234/42/2025).
- METRIC_LOCK: primary GENERAL_K_TOP1_HARMONIC (first_positive);
  secondary PER_CLASS_MACRO_F1, MICRO_F1, ANY_HIT. The k4 author
  release metric is contractually inadmissible on this data.
- FEATURE_LOCK: bundled similarity rows contain no held-out pair
  signal (features from external sources; train-only for
  association-derived rows where applicable — per
  FEATURE_PROVENANCE_MATRIX).
- EXPOSED_RETRO_LOCK: fold bundles used during development count as
  retrospective, not blind — reported as such.

## Arms

- U1 static & dynamic, 5 folds × seeds {1234,42,2025}, fixed-650ep
  (outer refit, WP4 archive).
- Priors: global / mirna / disease / product_laplace1.

## Change vs earlier drafts (errata E-1)

NEXT_REPRO_PROTOCOL.md §P.3 proposed a new triplet-derived fold —
SUPERSEDED: triplet permutation splits pairs (WP1 counterexample),
so it does not implement PAIR_CVTYPE and cannot answer this
question. Reuses the existing pair-fold family instead.
