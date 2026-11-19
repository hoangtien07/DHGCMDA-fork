# CORRECTED_BASELINE_RESULTS — 45 cells, corrected nested-validation (runs_g2c)

Primary = multihot macro-F1 (mean over 15 cells/track). Old runs_g2 values SUPERSEDED_UNVERIFIED.

## pair (T-pair)
| arm | macroF1 | microF1 | multitype | minorityR | card |
|---|---:|---:|---:|---:|---:|
| B1_logistic | **0.5298** | 0.6311 | 0.6303 | 0.3197 | 1.90 |
| B0 product_laplace1 | 0.4967 | 0.5841 | 0.6482 | 0.3703 | 2.38 |
| B2_labelprop | 0.4651 | 0.5664 | 0.6226 | 0.3113 | 2.44 |
| B0 disease | 0.4080 | 0.4911 | 0.5564 | 0.3806 | 2.14 |
| B0 mirna | 0.1120 | 0.1384 | 0.1581 | 0.0579 | 0.76 |
| B0 global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 (degenerate) |

**Corrected H1 target**: best baseline = B1_logistic **0.5298** → A1 must reach ≥ 0.5598 (Δ=+0.03 unchanged).

## mirna_cold
B2_labelprop 0.4569 | B0_mirna 0.3959 | B0_disease 0.3263 | B1 0.2757 | B0_product 0.2050 | global 0.0000

## disease_cold
B0_disease 0.3931 | B2 0.0824 | B1 0.0725 | B0_mirna 0.1636 | B0_product 0.0074 | global 0.0000

Artifacts: `runs_g2c/<track>/fold{f}_seed{s}/{scores.npz,iv_scores.npz,taus.json,metrics.json}` + `runs_g2c/AGGREGATE.json`. Metrics recomputed independently from row-level arrays by metrics_g2c.py.
