# TRACK_N_PREREGISTERED_PROTOCOL — REV N.1

Supersedes TRACK_N_PREREGISTERED_PROTOCOL.md (draft). Approved for
code-only + preflight (council 2026-11-19); training gated G0–G5.

## 1. Task

Multi-label 5-type classification conditional on known positive
miRNA–disease pairs. Data lock: strict npz sha 85bb9ea4.

## 2. Metrics — locked

- **Primary**: `multihot_macroF1` = mean over 5 types of per-type
  F1 computed on multi-hot truth vs binarized multi-label
  predictions (threshold rule in §4). Reflects minority classes;
  consistent with H1.
- **Secondary** (named, non-substitutable): `multihot_microF1`,
  per-type precision/recall/F1 + support, `minority_recall`
  (epigenetics channel), `multitype_macroF1` (macro-F1 restricted
  to pairs with ≥2 true types), `top1_firstpos_harmonic`
  (continuity with paper-family convention — projection in §4),
  `pred_cardinality` (mean |ŷ| per pair), type-macro AUPRC.
- **Forbidden**: k4 release metric; any metric selected after
  seeing test scores; unlabeled thresholds tuned on test.

## 3. Hypotheses & criteria — locked

- **H1**: new arm beats best baseline AND U1-static on primary
  `multihot_macroF1` on T-pair by Δ ≥ +0.03 (paired 15-cell mean).
  **Success**: Δ ≥ +0.03 AND minority_recall AND multitype_macroF1
  both ≥ corresponding best-baseline values.
  **Failure**: Δ < +0.03 → H1 falsified on T-pair; report negative.
  Inconclusive zone: none — criterion is binary.
- **H2**: ranking on primary metric persists in sign on at least
  one cold-start track (descriptive; no success threshold — flagged
  exploratory if cells < 15).
- **H3**: existence evaluated only on hard-negative strata with
  PU qualification; never merged into type metrics.

## 4. Prediction rule — locked

- Multi-label probabilities → binary ŷ via per-type sigmoid
  threshold τ_k chosen on **inner-validation only** (10% of train,
  stratified by train-mask; same selection bundle family as outer
  refit): τ_k = argmax_τ F1_k on inner-val over a fixed grid
  {0.05..0.95 step 0.05}; ties → lower τ.
- `top1_firstpos_harmonic` projects ŷ → argmax probability
  (single class) for the continuity metric only.
- Cardinality check (must run on every eval): report
  mean/median |ŷ|; alert (not fail) if outside [1, 4].
- Thresholds frozen per fold into the run manifest.

## 5. Arms — locked

- B0 priors: global/miRNA/disease/product_laplace1 (train counts).
- B1 multi-label logistic (per-type, L2, features: degree counts +
  similarity-row aggregates; train-only).
- B2 label propagation on train-restricted similarity graph.
- B3 U1-static: T-pair reuses frozen outer-refit predictions —
  no retrain; cold-start tracks = new runs (deep-model job,
  stopping rule §7).
- B4 (optional) TDRC port — only after B0–B3 verified.
- **A1 new arm — exactly one**: weighted multi-label sigmoid-BCE
  on U1 embedding inputs, EN weights β=0.99999. **PU auxiliary
  head OFF in arm-1** (council lock). Architecture/hparams copied
  from U1 where shared; no sweep.

## 6. Splits — locked

- T-pair: reuse verified group-pair fold family (retrospective
  flag; seeds 1234/42/2025).
- T-miRNA-cold: `default_rng(seed).permutation(411 miRNA)` →
  array_split(5); all pairs of held-out miRNAs are test.
- T-disease-cold: same over 271 diseases.
- Indices frozen JSON + sha256 before training; verify script
  asserts entity disjointness + prints per-fold per-type support.

## 7. Training budget (computed, not launched)

| Track | Jobs | Est. per job | Total |
|---|---|---|---|
| splits+canary+audit | — | minutes | <10 min |
| B0–B2 × 3 tracks | 15×3 cells | <1 min | <30 min |
| B3 U1-static cold-start | 2×5×3 = 30 | ~11 min | ~5.5 h |
| A1 T-pair 5×3 | 15 | ~11 min | ~2.7 h |
| A1 cold-start 2×5×3 | 30 | ~11 min | ~5.5 h |

Tracks run SEQUENTIALLY: splits/canary → B0–B2 → A1 sanity →
G4 T-pair → G5 cold-start (or SKIPPED_WITH_REASON).

## 8. Stopping rules — locked

- No reconfig after seeing test metrics; no added seeds to rescue
  negatives; second seed family only as preregistered replicate.
- Early stop on numerical failure only.
- G5 cold-start deep jobs run only if G0–G4 pass AND protocol
  coverage complete; else `SKIPPED_WITH_REASON` recorded.

## 9. Execution gates

G0 protocol revised + fixtures PASS → G1 splits + canaries PASS →
G2 baselines verified on shared splits/metrics → G3 1f×1s sanity
with independent metric recompute → G4 T-pair fleet → G5
cold-start deep jobs per §8.
