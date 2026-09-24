# IMPROVEMENT RESULTS — TDHGCMDA track on exact artifact MDAv3.2-3

**Protocol (locked):** SPLD golden pair-fold (8,735 pairs, 5×1747; whole
5-channel mask per test pair). Evaluator `eval_top1.py` verified identical to
SPLD internal (fold membership + metric semantics). Provenance: DSSM = static
external; MISIM = label-derived (legacy/leaky only); GIP/J = per-fold train.

## Leaderboard — legacy SPLD Top-1 metrics (outer 5-fold)

| Model | Track | P | micro-R | macro-R | **legacy-F1** | macro-AUPR |
|---|---|---:|---:|---:|---:|---:|
| simknn_misim | LEAKY | 0.6456 | 0.4800 | 0.5264 | **0.5799** | 0.4379 |
| simknn_j (vote + J) | honest | 0.6275 | 0.4666 | 0.5094 | **0.5623** | 0.4254 |
| **simknn_gip (vote)** | honest | 0.6253 | 0.4650 | 0.5070 | **0.5599** | 0.4536 |
| **SPLDHyperAWNTF (golden)** | honest* | 0.6219 | 0.4624 | 0.5069 | **0.5585** | — |
| DHGCMDA encoder (honest) | honest | 0.5094 | 0.3789 | 0.4000 | **0.4481** | 0.3531 |
| cooccur_prior | floor | 0.4576 | 0.3403 | 0.3461 | **0.3941** | 0.2690 |
| bilinear (BCE d32) | honest | 0.3328 | 0.2476 | 0.2577 | **0.2904** | 0.3217 |
| distmult | honest | 0.2835 | 0.2107 | 0.2122 | **0.2427** | 0.2792 |
| DHGCMDA-honest (R3 triplet-fold) | honest | — | — | — | 0.31 | 0.40 |
| DHGCMDA released (leaky) | leaky | — | — | — | 0.19 | — |
| **DHGCMDA paper claim** | — | 0.7915 | — | 0.9421 | **0.8600** | — |

*SPLD honest caveat: uses static MISIM internally (mild leak per M0).

Golden SPLD per-fold F1: 0.5647 / 0.5646 / 0.5505 / 0.5573 / 0.5556 →
aggregate P=0.6219, miR=0.4624, maR=0.5069 — exactly matching DHGCMDA Table-3
SPLD baseline row. Artifacts: `golden_spld_eval.json`, `golden_spld_eval_pred_fold*.npy`.

## Paired bootstrap (10k resamples on pooled outer-test pairs)

| Comparison | Δ legacy-F1 | 95% CI | verdict |
|---|---:|---|---|
| simknn_gip − SPLD | +0.0014 | [−0.0075, +0.0104] | **statistically tied** |
| simknn_misim − SPLD | +0.0214 | [+0.0131, +0.0297] | significant +2pt (leak channel) |
| DHGCMDA − SPLD | −0.1104 | — | far below |

## Findings

1. **A zero-parameter similarity vote matches the published tensor SOTA.**
   `simknn_gip` (row-normalized GIP_mi × DSSM-weighted votes) = 0.5599 vs
   SPLDHyperAWNTF 0.5585 (Δ=+0.0014, CI ⊇ 0). SPLD's model adds nothing beyond
   similarity-weighted label propagation — and itself uses the leaky MISIM.
2. **The paper's own architecture underperforms the vote baseline by −11pt**
   (0.448 vs 0.559) under honest eval — the dual-view CL machinery does not
   deliver what the paper claims.
3. **MISIM leak channel is real but modest**: +0.02 legacy-F1 (vote 0.5599→0.5799).
   `mi_fun_sim_3.2_3.csv` = MISIM 2.0 exactly (label-derived from HMDD).
4. **Co-occurrence J prior adds ~0** (0.5599→0.5623); embedding-only
   factorization is far below neighborhood voting (bilinear 0.29).
5. **Paper claim remains unreachable and partially impossible**: R=0.9421 >
   single-pred-per-pair macro-R ceiling 0.8642; no honest method > ~0.58.

## Milestone status

- M0 provenance ✅ (F24) · M1 golden eval ✅ (fold/semantics gates PASS) ·
  M2 tournament ✅ · M3 encoder ✅ (worse than vote) · M5 bootstrap ✅
- M4 typed encoder: **deprioritized** — vote ≥ encoders ⇒ encoder story moot;
  remaining value = presentation of honest ceiling + vote-as-SOTA result.
- M6 write-up: this file + REPRO_LEDGER F24-F28.
