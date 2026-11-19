# REAL_DATA_BASELINE_RESULTS — G2 (45 cells frozen)

Metrics recomputed independently from row-level `scores.npz` +
`taus.json` (`metrics_g2.py` — no training-time numbers used).
Primary: multihot_macroF1. Cell artifacts: `runs_g2/<track>/
fold<f>_seed<s>/{scores.npz,taus.json,metrics.json}` + MANIFEST.

## T-pair (15 cells: 5 folds × 3 seeds)

| Arm | macroF1 | microF1 | multitype_F1 | min-type R | card |
|---|---:|---:|---:|---:|---:|
| B0_product_laplace1 | 0.4967 | 0.5841 | 0.6482 | 0.3703 | 2.38 |
| B0_disease | 0.4080 | 0.4911 | 0.5564 | 0.3806 | 2.14 |
| B0_mirna | 0.1120 | 0.1384 | 0.1581 | 0.0579 | 0.76 |
| B0_global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| B1_logistic | **0.5184** | 0.6109 | 0.6448 | **0.3960** | 2.18 |
| B2_labelprop | 0.4559 | 0.5640 | 0.6091 | 0.2172 | 2.29 |
| A1_bce (fold0/s1234 sanity only) | 0.5201 | 0.6139 | 0.6342 | 0.3378 | 1.98 |

B1_logistic is the strongest T-pair baseline (macro-F1 0.5184);
B0_product is close (0.4967) and beats it on multitype_F1 —
the bar for A1 H1 is real (Δ≥+0.03 → needs ≥0.548).

## T-miRNA-cold (15 cells)

| Arm | macroF1 | microF1 | multitype_F1 |
|---|---:|---:|---:|
| B2_labelprop | **0.4488** | 0.5459 | 0.6012 |
| B0_mirna | 0.3959 | 0.4234 | 0.6119 |
| B1_logistic | 0.3224 | 0.4894 | 0.3221 |
| B0_disease | 0.3263 | 0.3745 | 0.4583 |
| B0_product | 0.2050 | 0.3352 | 0.3044 |

Surprise: label propagation (inductive similarity) leads on
miRNA-cold — consistent with cold-start favoring similarity-based
over count-based features.

## T-disease-cold (15 cells)

| Arm | macroF1 | microF1 | multitype_F1 |
|---|---:|---:|---:|
| B0_disease | **0.3931** | 0.4209 | 0.6054 |
| B0_mirna | 0.1636 | 0.2009 | 0.2376 |
| B1_logistic | 0.1236 | 0.1200 | 0.1615 |
| B2_labelprop | 0.0639 | 0.1014 | 0.1140 |

Disease-cold is much harder for learned/geometry baselines;
count prior on seen-side disease counts leads.

## Cardinality / cold-start failures

- B0_global degenerates (card=0: τ pushes all scores below
  threshold — constant-score arm; honest, kept).
- No execution failures in 45 cells; per-cell per-type support in
  each metrics.json.
