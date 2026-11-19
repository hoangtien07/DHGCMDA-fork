# AGGREGATION_ORDER_ANALYSIS — P/R/F1 aggregation conventions

Published: P=0.7915, R=0.9421, F1=0.8600.

## Finding 1 — pooled harmonic cannot produce 0.8600
- harmonic(0.7915, 0.9421) = 0.8602586 → rounds to **0.8603**.
- Over the full rounding rectangle P∈[0.79145,0.79155), R∈[0.94205,0.94215),
  harmonic ∈ [0.86021, 0.86031] — 0.8600 is **outside** the range.
- ⇒ If the paper's F1 is a pooled harmonic of the same P,R it reports,
  0.8600 is arithmetically inconsistent with 0.7915/0.9421 (small but real).

## Finding 2 — mean-of-fold-F1 CAN produce 0.8600 (errata: corrected witness)
Existence construction (NOT an author claim; the following fold values are a
mathematical witness produced by `joint_pr_feasibility_rev11.py`, not inferred
author fold results):
- fold A: P=0.6816, R=0.7504 → F1≈0.7143
- folds B–E: P=0.8190, R=0.9900 → F1≈0.8964
- mean P = 0.7915, mean R = 0.9421, **mean F1 = 0.8600** (residual < 1e-9)

Interpretation: 0.8600 is consistent with mean-of-fold-F1 aggregation under
fold heterogeneity (one weaker fold against four stronger folds). Because the
published mean recall 0.9421 is itself very high, any witness necessarily
assigns high recall to the majority of folds — but we make no claim that this
particular shape, nor near-perfect recall, is necessary; only that a feasible
family exists. This is existence evidence for a convention, not evidence about
the authors' actual folds.

## Finding 3 — other conventions
- Pooled P,R then harmonic: inconsistent (Finding 1).
- Mean-of-fold P,R then harmonic on the means: equals pooled harmonic here
  (equal-size folds) — same inconsistency.
- Per-(pair,type)-cell or micro aggregation: unbounded conventions — no
  source support; documented as HYPOTHESIS only.

## Documented conclusion
The published triplet is internally consistent ONLY under an aggregation
order different from pooled harmonic — either fold-wise F1 averaging with
heterogeneous folds (feasible witness above), or F1 computed from underlying
values different from the rounded published P,R. Neither is decidable from
public sources.
