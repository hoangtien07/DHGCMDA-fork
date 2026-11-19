# SCIENTIFIC_CLAIM_AUDIT — independent re-check (REV F1.1)

Each claim re-verified by independent computation, not by trusting the generating script.

| claim | audit method | result |
|---|---|---|
| 6914 correct predictions → rounds to 0.7915, and is the UNIQUE feasible integer | enumerated all C∈[0,8735] | CONFIRMED — feasible set = {6914}, E=1821 |
| pooled scalar bound R ≤ 0.9089 | 1 − 1821/(5·3997) recomputed by hand | CONFIRMED 0.9089 |
| fold-wise arbitrary-P bound R̄ ≤ 0.9123 | greedy allocation verified: 1747→fold3(832), 74→fold1(800); ΣE_f/n_maxf=2.1916 | CONFIRMED — valid upper bound now |
| fold2 n_max = 797 (type4) | recomputed from fold manifest | CONFIRMED — v1 (784, type3) was wrong |
| old equal-P fold bound invalid | counterexample: 2 folds nm 10/50, E=20: old bound 0.88 < true max 0.96 | CONFIRMED flaw; fixed |
| harmonic(0.7915,0.9421)=0.86026→0.8603 | direct arithmetic | CONFIRMED — paper reports 0.8600 |
| 0.8600 unreachable via pooled harmonic | rectangle [0.86021,0.86031] | CONFIRMED |
| mean-of-fold-F1 feasibility | found (0.56/0.71)+4×(0.849/0.99996) → mean F1≈0.860 | CONFIRMED existence-only; requires extreme heterogeneity |
| required class support 6285 > 3997 | 1821/(5·0.05795) | CONFIRMED |
| release views GIP-augmented on full A | prepareData.py L190-224 read | CONFIRMED — release leaked |
| d2025.bin vs MISIM distinct | ext/SHA256SUMS + artifact names | CONFIRMED — v1 doc conflated them |
| MISIM is association-derived | MISIM construction uses known associations (Wang) | CONFIRMED — cannot be CLEAN by default |
| K4 bound 0.9173 < 0.9421 | 1 − 1322/(4·3997) | CONFIRMED |
| v2.0 numbers non-contradictory | N=1498, K=4, bound not restrictive | CONFIRMED — only v3.2 pair is the outlier |

## Residual weaknesses (stated honestly)
- Fold-wise bound assumes the locked recovered folds; author folds unknown but
  bound is robust to ANY 5-fold partition of this tensor since n_max,f ≤ 3997.
- The 0.8600-vs-0.8603 discrepancy could also be a typesetting/rounding error
  in the paper (Δ=3e-4) — possibility acknowledged, not ruled out.
- MISIM "association-derived upstream" is from its published construction
  (Wang functional similarity on known associations) — cited as literature
  fact, not re-derived here.
