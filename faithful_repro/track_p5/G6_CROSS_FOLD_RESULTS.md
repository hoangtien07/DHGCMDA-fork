# G6 Cross-Fold Results — folds 1–4, locked seed-1234 manifests

Label: exploratory_only. Fold 0 reported separately (previously exposed).
B0/B1 = POST_HOC_DESCRIPTIVE comparators, same thresholds/metric protocol.

## Per-fold table (outer test, each evaluated once after freeze)
| fold | P5 macro-F1 | B0 always-pos | B1 prior-const | P5−B0 | P5−B1 | steps | sel_ep | wall_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0* | 0.4087 | 0.3894 | 0.3731 | +0.0193 | +0.0355 | 160 | 60 | 31.0 |
| 1 | 0.4220 | 0.3953 | 0.3788 | +0.0267 | +0.0432 | 160 | 50 | — |
| 2 | 0.4205 | 0.3959 | 0.3765 | +0.0246 | +0.0440 | 160 | 60 | — |
| 3 | 0.4143 | 0.4007 | 0.3834 | +0.0137 | +0.0310 | 180 | 70 | — |
| 4 | 0.4248 | 0.4012 | 0.3824 | +0.0236 | +0.0424 | 160 | 50 | — |

*fold0 = previously exposed exploratory evidence (kept separate).

## Folds 1–4 summary (fresh runs only)
- P5: mean 0.4204, sd 0.0038
- P5−B0: mean +0.0222, sd 0.0050 (all 4 folds positive)
- P5−B1: mean +0.0401, sd 0.0053 (all 4 folds positive)
- B0 folds1–4 mean 0.3983; B1 mean 0.3803.

## Combined 5-fold descriptive (fold0 exposed + folds1–4 fresh)
- P5 mean 0.4181, sd 0.0062; P5−B0 mean +0.0216.

## Read
The decision-collapse and the small-above-null margin are NOT fold-0
artifacts: every fold shows pos-pred rate = 1.0 on classes 2–4 and
P5−B0 in [+0.0137, +0.0267]. No significance claim, no author-faithful
or leakage-free claim — same caveats as G5.
