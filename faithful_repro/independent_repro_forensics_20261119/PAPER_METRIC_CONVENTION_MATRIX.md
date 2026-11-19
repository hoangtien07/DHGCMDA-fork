# PAPER_METRIC_CONVENTION_MATRIX — candidate Top-1 conventions vs published (P=0.7915, R=0.9421, F1=0.8600)

| # | convention | unit | truth | aggregation | source support | R bound @P=0.7915 | reaches 0.9421? | verdict |
|---|---|---|---|---|---|---:|---|---|
| C1 | K5 scalar, first-positive | pair | first true type | pooled | released evaluator semantics (K4 analog) | 0.8745 | no | RULED_OUT |
| C2 | K5 scalar, last-positive | pair | last true type | pooled | variant of released evaluator | 0.9066 | no | RULED_OUT |
| C3 | K5 scalar, arbitrary assignment | pair | any observed type | pooled | genericization of C1/C2 | 0.9089 | no | RULED_OUT |
| C4 | K5 scalar, fold-wise mean | pair | scalar | mean of folds | paper says 5-fold CV | 0.9088 | no | RULED_OUT |
| C5 | multi-hot truth, 1 pred | pair | all 5 labels | pooled | locked Track N convention | 0.758 / cap 0.849 | no | RULED_OUT |
| C6 | K4 filtered scalar | pair | types 0–3 only | pooled | released evaluator drops type-5 | 0.9173 | no | RULED_OUT |
| C7 | K4 collapsed (any-hit binary) | pair | any label | pooled | release "collapse" mode | degenerate | n/a | NOT_COMPARABLE |
| C8 | micro recall (TP/N) | pair | scalar | pooled | TDRC-style micro | ≤P-ish scale | possible in principle | HYPOTHESIS — micro-R=0.9421 needs P≈0.94 too, inconsistent with P=0.7915 unless denominators differ |
| C9 | per-(pair,type) cell unit | cell | per type flag | pooled | prediction unit >1/pair | unbounded | possible | HYPOTHESIS — no source support |
| C10 | different label population | — | — | — | curated eval subset | unknown | possible | HYPOTHESIS — unverifiable publicly |

Selection rule (owner directive): no convention may be selected because it
produces the highest F1. C1–C7 are the only source-supported conventions and
all are mathematically ruled out ⇒ **UNRESOLVED_EVALUATION_CONVENTION**.

F1 check: 2·P·R/(P+R) with P=0.7915,R=0.9421 → 0.8600 ✓ (F1 formula itself is
consistent; only the P/R pair is infeasible under C1–C7).
