# INDEPENDENT_REPRO_VERDICT

## Required outcome (WP5): **UNRESOLVED_WITH_PUBLIC_ARTIFACTS**

### What is already proven (prior work, re-verified on this branch)
- K4 released evaluator cannot score v3.2 (drops all K5 rows) — fixture-verified.
- Strict tensor = 411×271×5, 8735 pairs / 11,748 links — L1 identity with SPLD MDAv3.2-3.
- Optimal multi-hot macro-recall bound 0.8263 (1 pred/pair).
- U1 rescore 0.4558–0.4924 top1 harmonic; priors ≈90% of model; Track N A1 negative (−0.014 vs baseline).
- Density 10.5476% matches paper; density is NOT the cause of the F1 gap.

### Newly proven here (WP1)
The published pair (P=0.7915, R=0.9421) is **mathematically unreachable** under
every evaluation convention derivable from public sources, on this tensor:
- pooled K5 scalar (any truth assignment): R ≤ 0.9089
- fold-wise K5: ≤ 0.9088
- first/last-positive scalar: ≤ 0.8745 / 0.9066
- multi-hot 1-pred: ≤ 0.758 (hard cap 0.849 even at P=1)
- K4-filtered scalar: ≤ 0.9173
Reaching R=0.9421 at P=0.7915 needs a single scalar class of ≥~6287 samples —
impossible (max realizable 3997). Proof: `joint_pr_feasibility.py` +
brute-force fixtures in `JOINT_PR_FIXTURES.json`.

### Exact reason v3.2 is not verified
Two independent missing artifacts: (1) the evaluation convention (scorer +
aggregation + truth-unit) — provably none of the source-supported candidates
fit; (2) author fold/feature inputs — L1 stats match, L2/L3 unverifiable.
Without (1) even a perfect model cannot be compared to 0.8600; without (2)
the input cannot be claimed equivalent. Author contact is forbidden by owner,
so both are **permanent external blockers**, not implementation gaps.

### Author-equivalent scoring protocol feasibility
Not feasible under evidence: candidate conventions C1–C7 all ruled out
(PAPER_METRIC_CONVENTION_MATRIX). C8–C10 remain only as HYPOTHESIS with no
source support. This is recorded as `UNRESOLVED_EVALUATION_CONVENTION` —
explicitly not an accusation.

### Minimal next experiment worth funding
**NO_TRAIN_NECESSITY** (WP4): the remaining unknown is the evaluator, and no
amount of training can answer a metric-definition question. Every archived
run already bounds the space. The only unblockable artifact is the author's
eval code/data — owner forbids contact, so no further experiment is funded.
