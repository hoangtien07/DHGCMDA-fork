# G7-A2 Run Results — sigmoid-BCE type head (amendment G7_A2_TYPE_SIGMOID_BCE)
Scope: folds 1–4, seed 1234, A1 (mean recon) + independent sigmoid type head.
EXPLORATORY — folds already exposed. Outcome: **deterioration vs A1 on all 4 folds** — reported as-is, no rescue.

## Guards (verified before launch)
Pinned commit `2654583`; data sha `85bb9ea4…` (matches G6/A1); resolved
`recon_reduction=mean`, `type_objective=sigmoid_bce`, `lam3=1.0`,
`amendment=G7_A2_TYPE_SIGMOID_BCE`; new `runs/g7a2_fold*` dirs; per-fold
artifacts verified on disk (not only exit codes): result/manifest/
diagnostics/checkpoint all present, checkpoint metadata consistent with
manifest (`sigmoid_bce`, amendment, `graph_state='selected'`).

## Per-fold outcomes (all `completed`, no interrupts)
| fold | steps | sel. epoch | wall s | A1 F1 | A2 F1 | Δ(A2−A1) | Δ(A2−P5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 200 | 161.6 | 0.4385 | 0.4138 | −0.0247 | −0.0083 |
| 2 | 280 | 180 | 135.8 | 0.4599 | 0.4199 | −0.0401 | −0.0006 |
| 3 | 650 | 650 | 534.5 | 0.4664 | 0.4471 | −0.0194 | +0.0327 |
| 4 | 460 | 360 | 306.7 | 0.4748 | 0.4181 | −0.0567 | −0.0067 |

Mean Δ(A2−A1) = **−0.0352 ± 0.0145 (ddof=0)**, negative on 4/4 folds.
A2 mean 0.4247 ≈ P5 level (0.4204) — the sigmoid-BCE head removes A1's gain.

## Paradox: better ranking, worse decisions
- Class-3 AUROC improved: 0.64–0.68 under A2 vs 0.57–0.59 under A1 (and
  <0.5 under P5). Class-4 0.55–0.68.
- Yet decision collapse returned: τ back at/near grid floor (0.05–0.30 vs
  A1's 0.05–0.55), pos-pred rates ≈1.0 on classes 2–4 in folds 1,2,4
  (A1 had real spread: 0.27–1.0). Cardinality 2.8–3.5.
- Reading: sigmoid-BCE improves per-class *ranking* but produces poorly
  calibrated probabilities for this threshold-fitting setup — the
  post-threshold F1 is decided at the decision layer, not the ranking
  layer. Consistent with the A2 gradient finding (type grad weaker than
  intra on all folds — see `G7_A2_GRADIENT_MECHANISM.md`).

## Interpretation (per preregistration)
Mixed-to-negative for H3 as specified: AUROC up, macro-F1 down 4/4.
Descriptive evidence that under this exact configuration the sigmoid-BCE
variant is *not* the better objective at decision level. No significance
or causal claim; folds are exposed; no author-metric comparison.
