# G6 Prediction & Ranking Diagnostics — folds 1–4

## Decision collapse persists on all folds
Positive-prediction rates (P5 outer test):
| fold | c0 | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.37 | 0.19 | 1.00 | 1.00 | 1.00 |
| 2 | 0.38 | 0.15 | 1.00 | 1.00 | 1.00 |
| 3 | 0.44 | 0.25 | 1.00 | 1.00 | 1.00 |
| 4 | 0.43 | 0.19 | 1.00 | 1.00 | 1.00 |

Classes 2–4 hit 100% positive rate on every fold (τ at grid floor 0.05
everywhere) — identical to fold 0's finding.

## Ranking diagnostics (per-class AUROC / AP, folds 1–4)
| class | AUROC f1–f4 | AP f1–f4 |
|---|---|---|
| 0 | 0.619/0.624/0.622/0.640 | see G6_FOLD_TABLE |
| 1 | 0.702/0.732/0.648/0.684 | " |
| 2 | 0.533/0.514/0.564/0.536 | " |
| 3 | 0.608/0.647/0.635/0.562 | " |
| 4 | 0.436/0.466/0.415/0.458 | " |

- Minority classes (0,1) carry the ranking signal (AUROC ~0.62–0.73).
- Class 4 AUROC < 0.5 on all four folds — the model's score ordering is
  WORSE than chance on the second-largest class; its F1 is entirely a
  threshold artifact.
- Classes 2–3 hover near chance (0.51–0.65). Ranking collapse on the
  high-support classes is therefore NOT only a thresholding artifact —
  score ranking itself is weak-to-inverted there (descriptive, fold
  level only).

## Refresh / stability (recorded in fold diagnostics.json)
Refresh fired every 5 epochs on all folds; loss structure mirrors fold 0
(l_rec ~99.5% of objective, clip_active 100%). Graph state G/ei of the
selected epoch is now persisted inside each checkpoint.
