# G6 Reporting Errata Rev-1 — corrections to G6_CROSS_FOLD_RESULTS.md
Originals preserved; values taken directly from committed fold `result.json`.

| fold | optimizer_steps | selected_epoch | wall_s (correct) |
|---|---:|---:|---:|
| 1 | **280** | **180** | 72.88 |
| 2 | **190** | **90** | 52.61 |
| 3 | **250** | **150** | 66.65 |
| 4 | **220** | **120** | 49.59 |

The prior table carried wrong steps/epochs (copied from fold 0's pattern) and
omitted wall times. All macro-F1 values were and remain correct.

Standard deviations: population SD, **ddof=0** (numpy default) for every
reported sd (P5 0.0038, P5−B0 0.0050, P5−B1 0.0053; combined-5-fold 0.0062).
