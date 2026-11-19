# JOINT_PR_FEASIBILITY_THEOREM — published P/R pair vs measurable conventions

Base: strict tensor `data/v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
(N=8,735 positive pairs, L=11,748 links, K=5, support [1155,403,2293,3997,3900]).
Published: P=0.7915, R=0.9421, F1=0.8600 (4-decimal rounding intervals used).
Script: `joint_pr_feasibility.py` → `JOINT_PR_FIXTURES.json`. No training.

## Inequality (proved by exhaustive fixture)
With one prediction per pair and macro recall R = 1 − (1/K)·Σ FN_k/n_k,
n_k ≤ N_max ⇒ Σ FN_k/n_k ≥ E/N_max, hence

    R ≤ 1 − E / (K · N_max),   E = N·(1−P)

The bound is tight: it is attained iff every error lands in the largest class
(tiny-case exhaustive enumeration in JOINT_PR_FIXTURES.json).

## Rigorous bounds (P=0.7915 ±0.00005, E_max=1822.2)

| convention | K | N | N_max | R upper bound | R≥0.94205 possible? |
|---|---:|---:|---:|---:|---|
| pooled K5, scalar truth (any assignment) | 5 | 8735 | ≤3997 | **0.9089** | NO — needs class ≥6287 |
| fold-wise mean, K5 scalar (actual locked folds) | 5 | 1747/fold | 784–832/fold | **0.9088** | NO |
| first-positive scalar truth | 5 | 8735 | 2903 | 0.8745 | NO |
| last-positive scalar truth | 5 | 8735 | 3900 | 0.9066 | NO |
| multi-hot truth, 1 pred/pair, anyhit P | 5 | 8735 | 3997 | 0.7581 at P=0.79155; **0.8492 even at P=1** | NO (impossible at any P) |
| K4 filtered scalar (types 0–3, N4=6342) | 4 | 6342 | 3997 | 0.9173 | NO — needs class ≥5706 |

## Interpretation (conditional, not an accusation)
Under EVERY convention derivable from the public sources (scalar truth
assignment of any flavor, pooled or fold-wise aggregation, first/last-positive,
multi-hot any-hit, K4 filter), the joint pair (P=0.7915, R=0.9421) exceeds the
rigorous upper bound by ≥0.03–0.09. Reaching R=0.9421 with P=0.7915 requires a
single scalar class of ≥ ~6287 samples — no such class exists in any support
count of this tensor.

The gap can therefore only be closed by an evaluation-population or unit
difference not visible in public sources: e.g. a different label universe
(more pairs/different supports), a prediction unit emitting >1 type per pair,
a non-macro recall definition (e.g. micro-R can reach 0.94: micro R = TP/N
bounded only by P-like accuracy), filtered subsets, or per-class averaging
with different absent-class handling. The published K4 evaluator cannot score
v3.2 at all (drops every K5 row) — verified previously.

## 2-point consistency (v2.0)
v2.0 tensor: 495×383, 1498 pairs, 4 types, support [443,199,356,681].
Paper reports only F1=0.5970 for v2.0 (per convention matrix F1=2PR/(P+R)).
Same inequality with K=4, N=1498, N_max=681: for any plausible P the bound is
not restrictive (e.g. P=0.6 → R ≤ 1−1498·0.4/(4·681) = 0.780 vs F1 0.597
which implies moderate R). No contradiction for v2.0; the v3.2 pair is the
outlier. Detailed numbers in JOINT_PR_FIXTURES.json.

## Verdict on WP1 question
An author-equivalent scoring protocol that yields (0.7915, 0.9421) is **NOT
feasible under any source-supported convention** on this tensor → recorded as
`UNRESOLVED_EVALUATION_CONVENTION`, not fraud: the missing artifact could be a
different label population or a per-prediction-unit (multiple predictions per
pair, e.g. per-(pair,type) cells) evaluation.
