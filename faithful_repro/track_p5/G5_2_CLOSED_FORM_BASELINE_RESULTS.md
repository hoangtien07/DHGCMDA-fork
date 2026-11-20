# G5.2 Closed-Form Baselines — fold0 × seed1234

Label: **POST_HOC_FOLD0_DESCRIPTIVE_COMPARATOR** — not a prospective benchmark.
Protocol: identical locked fold, multi-hot convention, 19-point inner-val grid,
MULTIHOT_MACRO_F1 evaluator, no test-based threshold selection.

## Inputs (recorded per directive)
- Train n=6289; class counts [855, 297, 1613, 2905, 2831]; priors [0.1360, 0.0472, 0.2565, 0.4619, 0.4502]
- B1 prior below min grid (0.05): class1 only (0.0472)
- B1 val-selected τ = [0.05, 0.05, 0.05, 0.05, 0.05] (all at grid floor)

## Results (outer test n=1747, support [201,74,471,788,772])
| arm | class F1s | macro-F1 | TP | FP | FN |
|---|---|---:|---|---|---|
| **P5 model** | [0.2514, 0.1327, 0.4247, 0.6217, 0.6129] | **0.4087** | [116,27,471,788,772] | [606,306,1276,959,975] | [85,47,0,0,0] |
| **B0 always-pos** | [0.2064, 0.0813, 0.4247, 0.6217, 0.6129] | **0.3894** | [201,74,471,788,772] | [1546,1673,1276,959,975] | zeros |
| **B1 prior-const** | [0.2064, 0.0000, 0.4247, 0.6217, 0.6129] | **0.3731** | [201,0,471,788,772] | [1546,0,1276,959,975] | [0,74,0,0,0] |

- P5 − B0 = **+0.0193** (all from classes 0,1); P5 − B1 = +0.0356; B0 > B1 because B1's class-1 prior (0.0472) sits below the grid floor → all-negative → F1=0.
- Classes 2–4: P5 = B0 = B1 exactly — decision collapse confirmed at thresholding level.
- Caveats: single fold, single seed, exploratory profile (UNKNOWN_PROVENANCE views); differences of ±0.02 carry no significance claim.
