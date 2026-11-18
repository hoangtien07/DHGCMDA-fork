# SPLIT_SEMANTICS — PAIR_CVTYPE vs TRIPLET_CVTRIPLET

WP1.1 tribunal output. Executable contracts verified by
`test_split_contracts.py` (3 fixtures PASS).

## Paper wording (as cited in pinned sources)

- DHGCMDA Evaluation settings: CVtype splits **positive pairs** into
  five parts; CVtriplet splits **positive triplets** into five parts
  then augments with unknowns (per CVTRIPLET_SOURCE_COMPARISON.md,
  Springer link). Fold code for v3.2 was never released.
- SPLDHyperAWNTF (bib 2022): CVtype described as splitting positive
  pairs into K parts; CVtriplet "same way" per their text.
- Consequence measured in-repo (triplet_audit.json): under a triplet
  split on our candidate tensor, ~38.2% of positive test triplets
  share their pair with a different-type train triplet. Under a pair
  split this is structurally impossible.

## Executable contracts (R/test_split_contracts.py)

- `pair_fold_assign(pairs, k, seed)`: unique positive (mi,dis) pairs
  → `default_rng(seed).permutation` → `array_split(k)`. Every type of
  a pair stays in ONE partition. Inner-validation must also be
  pair-disjoint and sampled from outer-train only.
- `triplet_fold_assign(triplets, k, seed)`: (mi,dis,type) units;
  pair straddling is legal and expected.
- Fixture `test_pair_vs_triplet_nonequivalence`: a 2-type pair
  split across two hypothetical triplet partitions is ILLEGAL under
  PAIR_CVTYPE — the contracts are proven non-equivalent, so their
  scores must never be compared as "the same CV".

## Verified property of the REAL evidence (outer-refit)

`test_real_outer_refit_pair_disjoint` loads all 5 selection bundles
+ 30 evaluation npz:
- test ∩ train = ∅ AND test ∩ val = ∅ on every fold;
- identical eval pair set across 3 seeds × 2 arms;
- truth multi-hot K=5, type-5 support 763–801 per fold.
→ The E08 outer-refit folds ARE legal PAIR_CVTYPE folds
(pair-disjoint, train-only association features rebuilt per
bundle_audit.json: heldout_in_mask=0, group_crossings=0,
functional_semantic_rebuilt_equal=true).

## Exposure caveat (unchanged by this audit)

Outer-test fold0 was viewed during earlier development; pairs recur
across fold boundaries in the development history; epoch=650 and
architecture were chosen after observing validation. These runs are
retrospective post-development evaluations — NOT fresh blind tests.
