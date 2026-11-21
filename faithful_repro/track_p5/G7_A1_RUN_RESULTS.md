# G7-A1 Run Results — recon_reduction='mean' (amendment G7_A1_RECON_MEAN)
Scope: folds 1–4, seed 1234, all other locks preserved. EXPLORATORY — folds already exposed.

## Launch checks
`--dry-run` now applies the amendment (fixed; regression test added):
resolved `recon_reduction='mean'`, `lam3=1.0`, `amendment='G7_A1_RECON_MEAN'`;
data sha `85bb9ea4…` identical to G6; outputs under `runs/g7a1_fold*_s1234` (new dirs).

## Per-fold outcomes (all `completed`, no interrupts, no hard-stops)
| fold | opt. steps | sel. epoch | wall s | P5 F1 | A1 F1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 650 | 600 | 181.1 | 0.4220 | 0.4385 | +0.0165 |
| 2 | 490 | 390 | 166.0 | 0.4205 | 0.4599 | +0.0395 |
| 3 | 650 | 650 | 217.0 | 0.4143 | 0.4664 | +0.0521 |
| 4 | 650 | 640 | 156.5 | 0.4248 | 0.4748 | +0.0500 |

Mean Δ = **+0.0395 ± 0.0141 (ddof=0)**, positive on 4/4 folds.
(A1 mean 0.4599; vs B0 always-positive mean 0.3983 → +0.0617 mean.)

## Collapse metrics improved but not eliminated
- val-selected τ no longer pinned at floor: τ ∈ [0.05, 0.55] (P5: all 0.05–0.25);
  pos-pred rates on high-support classes now 0.27–1.0 (P5: 1.0 everywhere).
- Class-4 AUROC 0.57–0.59 across folds (P5: 0.42–0.47, below chance); class-2 0.66–0.70 (P5 ~0.51–0.56); class-1 0.78–0.83.
- Class-3 pos-rate still ~0.93–1.0 on folds 1,3,4 — residual collapse on the largest-support class persists.

## Loss/gradient behavior during training
- l_rec_w dropped from ~99.5% of total to a minor term; preclip grad norms collapsed from 10³–10⁵ to **0.4–5.3**; clipping still active on 74–83% of steps (most steps now near/below the cap).
- Refreshes every 5 epochs as locked (f1/f3/f4: 130 events at max epochs).

## Interpretation (per preregistered rules)
- The +0.04 paired delta is **descriptive** evidence the `mean` variant performs better under this setup on already-exposed folds — NOT confirmatory, NOT a causal proof about the original failure.
- Mechanistic gradient findings are in `G7_A1_GRADIENT_MECHANISM.md`; Adam update direction is NOT claimed reconstructed.
- `mean` also reweights the miRNA vs disease recon blocks (411²+271² elements → 2 means); the intervention is not a pure scale change.
