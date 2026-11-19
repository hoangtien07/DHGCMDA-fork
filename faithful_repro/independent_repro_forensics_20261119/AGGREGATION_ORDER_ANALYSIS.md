# AGGREGATION_ORDER_ANALYSIS — P/R/F1 aggregation conventions

Published: P=0.7915, R=0.9421, F1=0.8600.

## Finding 1 — pooled harmonic cannot produce 0.8600
- harmonic(0.7915, 0.9421) = 0.8602586 → rounds to **0.8603**.
- Over the full rounding rectangle P∈[0.79145,0.79155), R∈[0.94205,0.94215),
  harmonic ∈ [0.86021, 0.86031] — 0.8600 is **outside** the range.
- ⇒ If the paper's F1 is a pooled harmonic of the same P,R it reports,
  0.8600 is arithmetically inconsistent with 0.7915/0.9421 (small but real).

## Finding 2 — mean-of-fold-F1 CAN produce 0.8600
Existence construction (NOT an author claim, no fold values invented):
- fold A: P=0.5599, R=0.7105 → F1≈0.626
- folds B–E: P=0.8494, R=0.99996 → F1≈0.918
- mean P = 0.7915, mean R = 0.9421, mean F1 ≈ 0.8601

Interpretation: 0.8600 is consistent with mean-of-fold-F1 aggregation only
when fold metrics are highly heterogeneous (one weak fold, four near-ceiling
folds with R≈1.0). R≈0.99996 in four folds borders the implausible — flagged
as existence-only. A more moderate spread cannot reach 0.8600 because
mean-F1 is only second-order below the pooled harmonic (concavity effect
~1e-4 for ±0.05 spreads — measured).

## Finding 3 — other conventions
- Pooled P,R then harmonic: inconsistent (Finding 1).
- Mean-of-fold P,R then harmonic on the means: equals pooled harmonic here
  (equal-size folds) — same inconsistency.
- Per-(pair,type)-cell or micro aggregation: unbounded conventions — no
  source support; documented as HYPOTHESIS only.

## Documented conclusion
The published triplet is internally consistent ONLY under an aggregation
order different from pooled harmonic — either fold-wise F1 averaging with
extreme fold heterogeneity, or F1 computed from underlying values different
from the rounded published P,R. Neither is decidable from public sources.
