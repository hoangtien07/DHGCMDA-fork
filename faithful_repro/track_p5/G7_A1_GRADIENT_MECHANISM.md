# G7-A1 Gradient Mechanism — conditioned attribution vs matched G7.0 sum
Same protocol as G7.0 (eval mode, dropout off, seed 0, train-fold labels only,
zero optimizer steps) but with the **actual `mean` recon objective** — the
hard-coded `sum` config was NOT reused. Label: CHECKPOINT_CONDITIONED_GRADIENT_DIAGNOSTIC.

## Replay integrity
4/4 checkpoints: `graph_state='selected'`, eval-mode forward reproduces
recorded per-class TP/FP/FN exactly. Fresh checkpoint SHAs recorded in the
gate (no prior registry existed for A1).

## Per-objective total gradient L2 (fold 1→4)
| comp | A1 (mean) | G7.0 P5 (sum) |
|---|---|---|
| type | **0.342 / 0.242 / 2.666 / 0.693** | 0.06–0.07 |
| recon | 0.045 / 0.068 / 0.029 / 0.063 | 8 818–15 964 |
| intra | 1.004 / 0.672 / 0.932 / 1.505 | 1.2–1.4 |
| inter | 0.036 / 0.034 / 0.036 / 0.029 | 0.2–0.3 |
| exist | 0.034 / 0.045 / 0.063 / 0.028 | 0.1–0.3 |

- Under `mean`, the conditioned gradient is now task/contrastive-dominated:
  recon < type on every fold (recon/type ≈ 0.01–0.27 vs ≈10⁵ under `sum`).
- cos(type, recon): −0.041 / −0.009 / −0.013 / +0.048 — still ≈ orthogonal.
- Group split (f1): type grad now reaches predictor (0.17) + encoder (0.29); recon grad lands only on recon_heads/encoder as designed.

## Mechanistic reading
The single locked-key change produces BOTH a predictive gain (+0.0395 mean ΔF1)
AND the expected mechanistic effect (recon gradient reduced ~10⁵×, balance
restored). Per the preregistered interpretation these are consistent evidence
for H1 but jointly still not a causal proof — `mean` also changes the
miRNA:disease block weighting and the checkpoint was retrained, not replayed.
