# TRACK_N_BASELINE_AND_SPLIT_SPEC

## Data lock

`faithful_repro/data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
(sha256 85bb9ea4…). Labels: multi-hot 5-type per pair. All
association-derived features (GIP/functional similarity, any
graph edges) rebuilt from the training fold's labeled pairs ONLY —
established before graph construction (hard rule 3).

## Splits (three evaluation tracks, each a separate claim)

| Track | Split | Unit | Purpose | Status |
|---|---|---|---|---|
| T-pair | group-disjoint pair folds ×5, seeds {1234,42,2025} | pair (all types of a pair leave together) | primary benchmark | existing family reused; retrospective flag kept |
| T-miRNA-cold | miRNA-group folds: all pairs of a held-out miRNA cluster/segment | entity | cold-start generalization | NEW, must be built + verified disjoint at entity level |
| T-disease-cold | disease-group folds | entity | cold-start generalization | NEW, same rules |

Fold generation: `default_rng(seed).permutation(entity_ids)` →
`array_split(k)`; indices frozen to JSON with sha256 before any
training. A fold is legal only if its verify script checks
entity/pair disjointness and prints counts.

## Baselines (minimum, all on identical folds/metrics)

1. `prior_product_laplace1` (log1p(mi)+log1p(dis)−log1p(global),
   train counts; global fallback +1e-12) — strongest known floor.
2. Multi-label logistic regression on pair-level count/degree and
   similarity-row features (train-only rows).
3. Label propagation on the similarity graphs restricted to
   train pairs.
4. U1 static (existing outer-refit predictions already frozen —
   reuse, no retrain for the pair track; cold-start tracks need new
   runs only if protocol approved).
5. Optional: TDRC under same split/metric (independently
   reproducible tensor factorization; adds external-method anchor).

## New arm (exactly ONE, preregistered)

Weighted multi-label sigmoid-BCE classifier on the same embedding
inputs U1 uses, with EN class weights (β=0.99999) and optional
PU-aware pair-existence auxiliary head (auxiliary only; never
merged into the type metric). Rationale: directly targets
minority-type ranking and multi-label capacity that scalar-target
CE cannot express. No kitchen-sink changes; architecture fixed
before any test score is seen.

## Leakage canary (mandatory)

For every fold: an oracle feature read that would reveal a held-out
pair's label (e.g., a feature channel exposing A_test rows) must be
demonstrably absent; `oracle_feature_canary` script asserts every
similarity/edge matrix entry is computable from train labels alone.
Fail ⇒ fold invalid.

## Metrics (preregistered; convention named explicitly)

- **Primary**: `multihot_macroF1` (per-type macro-F1 on full
  5-type multi-hot truth) — reflects minority classes.
- **Secondary**: `K5_firstpos` harmonic (paper-family convention,
  for continuity), `multihot_microF1`, type-macro AUPRC,
  minority-type recall (epigenetics class), `multitype_anyhit`
  (corrected definition REV4.1), per-type support reports.
- **Forbidden**: k4 release metric (drops type-5); any metric
  chosen after seeing test scores.
- Existence evaluation (if any): hard-negative subsets + PR
  curves; PU qualification stated, never "unknown = negative".

## Uncertainty & stopping

- Paired fold×seed descriptive stats (mean±SD over 15 cells);
  no p-values without a real replication design; effect-size target
  for H1: Δmacro-F1 ≥ +0.03 over best baseline (precedent: leak
  effect was +0.049).
- Stop after preregistered cells: no escalating to more seeds
  to chase significance; a second seed family is allowed only as
  a pre-registered confirmation replicate.
