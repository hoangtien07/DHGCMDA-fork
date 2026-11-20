# G5.1 Objective + Graph Diagnostics — existing logs only, no new training

## WP2 — Numerical attribution (160 epochs)
Weighted components (epoch1 → epoch160, fraction of total):
| component | ep1 | ep160 | frac_end |
|---|---:|---:|---:|
| total | 8242.5 | 3132.4 | 1.0 |
| l_type (type CE) | 1.10 | 1.08 | 0.0003 |
| lam1·l_intra | 13.28 | 11.93 | 0.0038 |
| lam2·l_inter | 1.66 | 1.67 | 0.0005 |
| lam3·l_rec | 8226.4 | 3117.7 | **0.9953** |
| exist_w·l_ex | 0.07 | 0.02 | ~0.0000 |

- Reconstruction term carries 99.5% of the objective throughout (sum reduction, locked) — the type head is trained on a loss whose gradient is numerically dominated by recon. **Not inferred from magnitude alone**: per-component gradient norms were not recorded — see gaps.
- Preclip grad norm min/med/max = 1067 / 5161 / 51 679; **clip_active on 160/160 steps**, postclip norm 1.0 exactly — the optimizer sees only clipped directions.
- Actual param updates ||Δθ||_total: 0.152 → 0.094 (monotone-ish decay); top-update param `hgt.0.conv.kqv_lin.lins.miRNA.weight` throughout.
- Selected checkpoint ep60 vs stop ep160: best val macro-F1 0.4219 @ep60; 10 subsequent evals never exceeded it → patience stop. Model returned ~100 epochs of stale checkpoint.
- **Unavailable from logs** (marked, not inferred): per-component gradient norms; per-class val F1 history; prediction-score distributions; which parameters carry recon vs type gradients. Optional checkpoint-level forward/backward attribution on the frozen TRAIN fold (0 optimizer steps, no test) is possible ONLY under separate explicit authorization — not executed.

## WP3 — Graph dynamics (32 refresh events)
- Associate edges constant 6289/6289 every refresh (train positives — expected).
- miRNA-similar edges after-refresh: 884 → range [884, 44 832], all 32 distinct; wild oscillation (e.g. 24 620 → 40 300 → 27 370 in last three).
- disease-similar: 64 150 at first refresh → collapses to ~200 by refresh 4 (dense→sparse transition), then regrows to 4 500–8 500 and oscillates.
- Densities: miRNA-sim max ≈ 44832/168 510 ≈ 27%; disease-sim initial ≈ 64150/73 170 ≈ 88% → ~0.3% at minimum.
- Val history (16 evals, 0.40–0.42 plateau) shows no visible step-change aligned with topology swings — **no causal claim**; correlation at this resolution is uninformative.
- Plausible mechanism (recorded evidence supports, not proven): S_m/S_d reconstructions feed mm01 + θ=0.5 hard thresholding; early reconstructions are poorly fit → similarity masses sit near 0.5 → small θ-crossings flip thousands of edges. Whether mm01 normalization or θ specifically causes the instability is **UNTESTABLE from existing logs** (S statistics not recorded) — flagged as open hypothesis.
