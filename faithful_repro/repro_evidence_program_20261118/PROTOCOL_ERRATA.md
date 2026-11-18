# PROTOCOL_ERRATA — versioned corrections to earlier protocol text

Corrections are issued as errata; historical files are NOT edited in
place (non-negotiable rule #2).

## E-1: NEXT_REPRO_PROTOCOL.md §P.3 "fold_lock" is INVALID as written

**Original text** (independent_v32_overnight_20261118/NEXT_REPRO_PROTOCOL.md):
"Same permutation as CVtriplet protocol (rng(0) over positives,
array_split 5) — already pinned and sidecar-hashed."

**Problem:** the CVtriplet split permutes positive TRIPLETS
(mi,dis,type). Reusing it for the type task (a) leaks pair identity
across partitions (~38% of test triplets share their pair with
train), and (b) does not match PAIR_CVTYPE, which the paper's own
CVtype description requires. triplet_u1's split is legal for Task T
only.

**Correction (v2 of the protocol's fold lock):** CVtype folds must be
built over UNIQUE POSITIVE PAIRS: `default_rng(0).permutation(n_pairs)`
→ `array_split(5)`. The existing group-candidate fold family used by
independent_v32_5fold3seed_20261117 and
independent_v32_outer_refit_20261117 already implements pair-disjoint
folds (verified by R/test_split_contracts.py on all 5 folds and by
bundle_audit.json: group_crossings=0). **Reuse that fold family**;
do not mint a third split family.

## E-2: "generalized Top-1 macro-F1" name was ambiguous

P.4 calls the primary metric "generalized Top-1 macro-F1 over K=5".
Two different formulas answer to that name (harmonic-of-aggregates
vs mean-of-per-class-F1; fixture shows 0.2154 vs 0.0971 on one
dataset). Correction: name the metric `GENERAL_K_TOP1_HARMONIC`
(harmonic-of-aggregates, first_positive convention) as the paper-
family comparative, and `PER_CLASS_MACRO_F1` as the separate internal
diagnostic. See METRIC_SEMANTICS.md.

## E-3: "identical to the author's 4-type metric" scope

P.4 states the corrected metric is "proven identical to the author's
4-type metric on v2.0 (Plan K)". Confirmed independently here by
fixture `test_k4_equivalence_vs_pinned_release` on 300 random rows
plus multi-hot cases (exact match to 1e-12). Scope of the claim:
equivalence holds ONLY where the release can score (K=4, scalar or
multi-hot truth, len-4/5 predictions). It proves nothing about the
unreleased v3.2 eval.
