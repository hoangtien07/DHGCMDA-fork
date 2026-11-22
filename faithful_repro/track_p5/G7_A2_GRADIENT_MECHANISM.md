# G7-A2 Gradient Mechanism — conditioned attribution vs A1
Same CHECKPOINT_CONDITIONED_GRADIENT_DIAGNOSTIC protocol (eval mode,
seed 0, train-fold labels only, zero optimizer steps) with the ACTUAL
sigmoid-BCE objective — restore via `restore_checkpoint` (fail-closed).

## Integrity
4/4 checkpoints: SHA recorded in gate, `graph_state='selected'`,
eval-mode forward reproduces recorded TP/FP/FN exactly, and checkpoint
metadata consistent with the run manifest (`sigmoid_bce`, amendment id,
`recon_reduction=mean`).

## Per-objective total gradient L2 (fold 1→4)
| comp | A2 (sigmoid-BCE) | A1 (softmax+mean) | P5 (softmax+sum) |
|---|---|---|---|
| type | 0.109 / 0.157 / 0.070 / 0.069 | 0.342 / 0.242 / 2.666 / 0.693 | ~0.06–0.07 |
| recon | 0.022 / 0.026 / 0.081 / 0.059 | 0.045 / 0.068 / 0.029 / 0.063 | 8.8k–16k |
| intra | 0.931 / 1.277 / 1.064 / 1.213 | 1.004 / 0.672 / 0.932 / 1.505 | 1.2–1.4 |
| inter | 0.055 / 0.051 / 0.017 / 0.030 | ~0.03 | 0.2–0.3 |
| exist | 0.058 / 0.067 / 0.055 / 0.032 | ~0.03–0.06 | 0.1–0.3 |

- Recon stays small (mean reduction active) — the balance problem A1 fixed
  does not return.
- **Type gradient under sigmoid-BCE is weaker than intra on all 4 folds**
  (type/intra ≈ 0.06–0.16), and weaker than under softmax on folds 1,3,4.
  Under this setup the BCE objective does not deliver a stronger task
  signal at the selected checkpoint — mechanistically consistent with the
  observed F1 deterioration (descriptive, not causal proof).
- cos(type, recon): −0.057 / +0.060 / −0.026 / +0.211 — near-orthogonal
  to mildly positive; no adversarial interaction measured.
