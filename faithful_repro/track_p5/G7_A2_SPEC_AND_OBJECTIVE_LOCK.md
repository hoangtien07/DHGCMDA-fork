# G7-A2 Spec & Objective Lock — sigmoid-BCE type head (H3)
Hypothesis H3: the 5-way softmax multi-hot objective is mismatched with
multilabel association-type prediction (a pair may carry several true types;
softmax forces a single-mode distribution).

## Amendment (one registered entry)
`G7_A2_TYPE_SIGMOID_BCE` → `{recon_reduction: 'mean', type_objective: 'sigmoid_bce'}`
Keeps A1's mean recon; every other LOCKED_HYPERPARAMS value enforced.

## Objective semantics (frozen before any real-data evaluation)
- Predictor head: same diag-bilinear `einsum('ih,kh,jh->ijk')` logits.
- Temperature: the existing learnable `temp` (clamped to [0.5,5]) scales the
  logits EXACTLY as the softmax path does; both paths share it.
- Probabilities for evaluation/thresholds: `p = sigmoid(logits / temp)` —
  5 independent probabilities; no sum-to-1 constraint.
- Training loss on TRAIN-positive pairs only (test/val labels never enter):
  `l_type = mean( BCEWithLogitsLoss(scaled_logits, y_multihot, pos_weight=w, reduction='none') )`
  where `w` = the SAME smoothed per-class weights already used by the
  softmax path (beta=0.99999, normalized to sum 5) — applied as
  `pos_weight` (re-weights the positive term of each class). No new
  hyperparameters, no search.
- Implementation: runner-side `apply_type_objective()` patches
  `model.pred.forward` on the instance (faithful.py untouched), returning
  sigmoid probabilities as `tp` so the locked evaluator/threshold grid is
  reused verbatim, and stashing `last_logits` for the BCE loss.
