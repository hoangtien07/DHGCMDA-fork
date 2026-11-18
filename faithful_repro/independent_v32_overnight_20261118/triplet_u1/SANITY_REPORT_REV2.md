# CVtriplet-U1 REV2 — sanity evidence (fold0, seed1234)

Scope: PROTOCOL_CVTRIPLET_U1_REV2 §7 — design validation only.
These numbers are NOT an experiment result; full 5×3 remains blocked
pending review approval.

## Gates

| Gate | Status | Evidence |
|---|---|---|
| G1 unit fixtures (9) | PASS | preflight.json `fixtures` |
| G1 preflight (pins + canary on real bundles) | PASS | preflight.json (15 entries) |
| G2 bundle build ×5 | PASS | build_report.json; fold0: 6754 train pairs, 599 val pairs, 327 inner-val straddle dropped, 2350 test triplets |
| G4 sanity fold0 seed1234 | PASS | runs/fold0_seed1234: 650/650 epochs, not aborted, all-terms finite |
| G4 independent verify | PASS | verification.json — 26 checks |
| G5 full 5×3 | BLOCKED | requires user approval per REV2 §7 |

## Sanity fold0 signals (design validation, not results)

Trained: U1 + existence head, lam_exist=0.3, 67540 pinned mask-zero
negatives (rng seed+77), exist head init at rng(seed+4007).

- `val_exist_gap` (val-pair mean σ vs mask-zero probe mean) = +0.182 at
  epoch 600 — head separates mask pairs from mask-zero cells.
- Task T joint score s=σ(exist)·softmax_k: AUROC 0.849, AP 0.809
  (mean over 5 pinned eval seeds).
- Task P pair score σ(exist): AUROC 0.815, AP 0.817 on pair-unseen
  positives vs pair-zero negatives.
- Train-only count logistic on the SAME replicates: AUROC 0.933 —
  count features dominate this eval construction at fold0; documents
  the task difficulty is mostly in easily-counted strata.
- F1@0.5 Task T = 0.01 — joint scores live at ≲0.2 scale so the fixed
  threshold is degenerate; AUROC/AP unaffected. Declared, not hidden.
- Strata (seed0): neg_on_train_pair n=115 score_mean 0.077;
  neg_on_heldout_pair n=34 0.078; neg_pair_zero n=2201 0.042 —
  negative mass is dominated by pair-zero cells.
- Positive strata: 854 test triplets on train-seen pairs vs 1496 on
  unseen pairs; joint-score means 0.126 vs 0.139 — head does not
  collapse on unseen pairs.

## Files

bundles/fold{0..4}/ (npz + sidecars), contracts/fold{F}_seed{S}/
(cfg+expected+neg idx, 15), runs/fold0_seed1234/ (out/manifest,
checkpoints, predictions, exist_scores.npz, eval.json,
verification.json, launch.json, progress.log).
