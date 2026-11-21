# G7.0 Checkpoint Gradient Attribution — folds 1–4
Label: **CHECKPOINT_CONDITIONED_GRADIENT_DIAGNOSTIC** — eval-mode (dropout
off), torch.manual_seed(0), zero optimizer steps, train-fold labels only.
NOT exact training-time gradients (dropout masks/RNG/opt state unlogged).

## WP1 integrity
- All 4 checkpoint SHA256 match `G6_OWNER_GATE.json`; `graph_state='selected'`; G, ei, state_dict, selected_epoch present.
- **Replay EXACT on all 4 folds**: reloading model+selected G/ei in eval mode reproduces recorded per-class test TP/FP/FN identically.
- No claim of exact historical gradient replay.

## WP2 measured gradient norms (weighted components, fold 1–4)
| component | total grad L2 (fold1..4) |
|---|---|
| l_type (task) | 0.06–0.07 |
| lam3·l_recon | **8 818 / 15 071 / 11 208 / 15 964** |
| lam1·l_intra | 1.2–1.4 |
| lam2·l_inter | 0.2–0.3 |
| exist_w·l_ex | 0.1–0.3 |

- Recon gradient norm is **~10^4–10^5× the task gradient**; the unclipped step direction is numerically the recon direction. After clipping to norm 1.0, the applied update direction still ≈ recon direction (clip rescales the already-dominated vector; it cannot recover the task component).
- cos(type-grad, recon-grad) on shared params: −0.039 / +0.007 / −0.005 / +0.028 → **near-orthogonal**; recon does not systematically oppose type learning, it simply swamps it.
- Per-group: recon grad lands on encoder (~5.4k) + recon_heads (~6.9k); type grad concentrates on predictor (0.058) + encoder (0.028); graph_layers take ~0.01 type grad.
- **Answer to the directive question**: YES — measured, not inferred: the reconstruction gradient dominates the total direction both before and after the clip rescaling.
