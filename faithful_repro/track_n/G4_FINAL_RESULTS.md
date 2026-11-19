# G4_FINAL_RESULTS — A1 fleet 5 folds × 3 seeds, T-pair (runs_g4)

Frozen config: MLP(→256→5) on concat(Sm[i],Sd[j]), BCE + EN pos_weight β=0.99999, PU OFF, epochs 300, lr 1e-3, wd 1e-4, inner-val 10%, τ grid 0.05–0.95. No config change after results.

## Headline (paired 15-cell means)
| metric | A1 | Δ vs best corrected baseline | Δ vs U1 static |
|---|---:|---:|---:|
| multihot macroF1 (primary) | 0.5166 ± 0.0129 | **−0.0136** | +0.0484 |
| top1_firstpos_harmonic | 0.4421 ± 0.0107 | +0.0022 | −0.0527 |
| minority_recall | 0.3494 ± 0.0862 | −0.0098 | −0.3186 |
| multitype_macroF1 | 0.6316 ± 0.0187 | −0.0078 | +0.0328 |

Best corrected baseline per cell: B1_logistic (macroF1 0.5298).
Per-fold diffs vs best baseline (15 values): −0.0090 −0.0058 −0.0089 −0.0073 −0.0071 −0.0213 −0.0175 −0.0277 −0.0115 +0.0012 −0.0021 −0.0330 −0.0176 −0.0106 −0.0261 — 14/15 negative.

## H1 verdict
**REJECTED.** A1 does not beat the corrected best baseline by Δ≥+0.03 (observed −0.0136). Secondary conditions (minority_recall, multitype_macroF1 ≥ best baseline) also fail. A1 does beat U1 static on macroF1 (+0.048) but that was never the H1 test. U1-static comparison used fixed τ=0.5 (no inner-val exists for frozen logits) — flagged approximation.

## Negative result — reported as negative. No rescue runs, no seed escalation.
Artifacts: `runs_g4/fold{f}_seed{s}/{scores.npz,ckpt.pt,taus.json,train_log.json}` + `runs_g4/AGGREGATE_G4.json` (paired diffs, per-fold rows).
