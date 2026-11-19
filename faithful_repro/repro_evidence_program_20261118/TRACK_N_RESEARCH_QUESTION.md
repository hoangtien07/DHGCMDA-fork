# TRACK_N_RESEARCH_QUESTION

## Primary task

Multi-label 5-type classification **conditional on known positive
miRNA–disease pairs** on the strict HMDD v3.2-derived benchmark
(411×271×5, 8,735 positive pairs, 11,748 typed links, 24.2%
multi-type pairs). Binary pair existence is out of scope as a
primary question (PU semantics make "negative" unverifiable; any
existence evaluation is secondary and explicitly qualified).

## Primary hypothesis (preregistered, falsifiable)

> H1: A leakage-controlled model with a **multi-label type-aware
> objective** outperforms strong simple baselines (count priors,
> multi-label logistic, label propagation) AND the U1-static
> reference on **minority-type recall and multi-type-pair
> performance**, under pair-disjoint folds — not merely on
> aggregate accuracy dominated by frequent types.

Secondary hypotheses:
- H2: The same ranking of methods persists under **entity cold-start
  splits** (miRNA-group, disease-group) — generalization under
  shift, not just pair novelty.
- H3 (negative-control): no method approaches degree priors on
  pair-existence once PU negatives are honestly treated —
  quantifying the degree-prior floor.

## Why this is new (not re-litigating the paper)

- Independent of the 0.8600 reproduction question: own benchmark,
  own metric contracts, own splits. Results stand even if the
  paper's number is never verified.
- Novelty claim is **leakage-controlled, imbalance-aware multi-label
  type prediction + honest baselines + generalization under shift**,
  not a higher F1.
- Resolves a question the field's numbers cannot answer: how much
  of published "type prediction" performance is degree prior vs.
  learned signal — our strata data (prior ≥ model on every any-hit
  stratum, REV4.1) suggests the honest answer is uncomfortable for
  many published models.

## What would falsify / end the project

- If no method beats `prior_product_laplace1` on the preregistered
  primary metric on the pair-disjoint track → negative result,
  reported as such; the benchmark + protocol itself is the
  contribution.
- If labels prove unreliable (see label-trust review in
  TRACK_N_INDEPENDENT_REVIEW) → reframe to benchmark/protocol
  paper only.
