# IMPROVEMENT RESULTS — TDHGCMDA track on exact artifact MDAv3.2-3

**Protocol (locked):** SPLD golden pair-fold (8,735 pairs, 5×1747; whole
5-channel mask per test pair). Evaluator `eval_top1.py` verified identical to
SPLD internal (fold membership + metric semantics). Provenance: DSSM = static
external; MISIM = label-derived (legacy/leaky only); GIP/J = per-fold train.

## Leaderboard — legacy SPLD Top-1 metrics (outer 5-fold)

| Model | Track | P | micro-R | macro-R | **legacy-F1** | macro-AUPR |
|---|---|---:|---:|---:|---:|---:|
| **gip_only (mi-side vote)** | PRIMARY HONEST | 0.6558 | 0.4877 | 0.5380 | **0.5910** | 0.4108 |
| simknn_misim | LEAKY CONTROL | 0.6456 | 0.4800 | 0.5264 | **0.5799** | 0.4379 |
| fmisim_dssm (fold-MISIM vote) | honest | 0.6390 | 0.4752 | 0.5199 | **0.5733** | 0.4383 |
| gip_gip | honest | 0.6305 | 0.4688 | 0.5126 | **0.5655** | 0.4274 |
| simknn_j (vote + J) | honest | 0.6275 | 0.4666 | 0.5094 | **0.5623** | 0.4254 |
| **simknn_gip (vote)** | PRIMARY HONEST | 0.6253 | 0.4650 | 0.5070 | **0.5599** | 0.4536 |
| **SPLDHyperAWNTF (golden)** | LEGACY COMPARISON | 0.6219 | 0.4624 | 0.5069 | **0.5585** | — |
| SPLD no-MISIM (MGSM only) | LEGACY corrected | 0.6227 | 0.4631 | 0.5078 | **0.5594** | — |
| SPLD fold-MISIM (train-rebuilt) | LEGACY corrected | 0.6203 | 0.4613 | 0.5054 | **0.5570** | — |
| dssm_only (dis-side vote) | honest | 0.4781 | 0.3555 | 0.3702 | **0.4173** | 0.3977 |
| DHGCMDA encoder (honest) | honest | 0.5094 | 0.3789 | 0.4000 | **0.4481** | 0.3531 |
| cooccur_prior | floor | 0.4576 | 0.3403 | 0.3461 | **0.3941** | 0.2690 |
| bilinear (BCE d32) | honest | 0.3328 | 0.2476 | 0.2577 | **0.2904** | 0.3217 |
| distmult | honest | 0.2835 | 0.2107 | 0.2122 | **0.2427** | 0.2792 |
| DHGCMDA-honest (R3 triplet-fold) | honest | — | — | — | 0.31 | 0.40 |
| DHGCMDA released (leaky) | leaky | — | — | — | 0.19 | — |
| **DHGCMDA paper claim** | — | 0.7915 | — | 0.9421 | **0.8600** | — |

*SPLD MISIM channel measured directly (F32): removing MISIM entirely
(MGSM/GIP only) gives 0.5594 (+0.0009 vs static) and fold-reconstructed MISIM
gives 0.5570 (−0.0015) → SPLD's published 0.5585 contains essentially no
leak benefit; its tensor model does not exploit the static MISIM matrix
(unlike the vote, where static MISIM adds +0.02).

Golden SPLD per-fold F1: 0.5647 / 0.5646 / 0.5505 / 0.5573 / 0.5556 →
aggregate P=0.6219, miR=0.4624, maR=0.5069 — exactly matching DHGCMDA Table-3
SPLD baseline row. Artifacts: `golden_spld_eval.json`, `golden_spld_eval_pred_fold*.npy`.
Corrected legacy runs: `golden_spld_eval_fold.json`, `golden_spld_eval_none.json`.

## Paired bootstrap (10k resamples on pooled outer-test pairs)

| Comparison | Δ legacy-F1 | 95% CI | verdict |
|---|---:|---|---|
| simknn_gip − SPLD | +0.0014 | [−0.0075, +0.0104] | **statistically tied** |
| simknn_misim − SPLD | +0.0214 | [+0.0131, +0.0297] | significant +2pt (leak channel) |
| DHGCMDA − SPLD | −0.1104 | — | far below |

## Findings

1. **A zero-parameter similarity vote matches the published tensor SOTA.**
   `simknn_gip` (row-normalized GIP_mi × DSSM-weighted votes) = 0.5599 vs
   SPLDHyperAWNTF 0.5585 (Δ=+0.0014, CI ⊇ 0): we detect no statistically
   significant advantage over the similarity-voting baseline under this
   protocol (SPLD's own score is unaffected by the MISIM channel — F32). The
   single-view miRNA-side vote (`gip_only`, 0.5910) is the best observed
   performance overall.
2. **The publicly available/reconstructed DHGCMDA implementation
   underperforms the vote baseline by −11pt** (0.448 vs 0.559) under honest
   eval — the released dual-view CL machinery does not deliver what the paper
   claims (private training/eval code was never found).
3. **MISIM leak channel is real but modest**: +0.02 legacy-F1 (vote 0.5599→0.5799).
   `mi_fun_sim_3.2_3.csv` = MISIM 2.0 exactly (label-derived from HMDD).
4. **Co-occurrence J prior adds ~0** (0.5599→0.5623); embedding-only
   factorization is far below neighborhood voting (bilinear 0.29).
5. **Paper claim remains unreachable and partially impossible**: R=0.9421 >
   single-pred-per-pair macro-R ceiling 0.8642; best observed performance
   among tested methods ≈ 0.59 (`gip_only`).
6. **SPLD's result is MISIM-independent** (F32): static 0.5585 ≈ fold-MISIM
   0.5570 ≈ no-MISIM 0.5594 — the leak channel that inflates the vote
   (+0.02) leaves SPLD untouched, consistent with SPLD's model barely using
   the similarity matrix it is fed.
7. **Ablation verdict** (F30): miRNA-side similarity vote alone
   (`gip_only` = 0.5910) is the single best observed method; adding the
   DSSM disease-side vote *reduces* F1 (−3.1pt), J prior adds +0.001.
   Performance lives in miRNA functional neighborhood + local propagation,
   not in multi-view/contrastive machinery.

## Milestone status

- M0 provenance ✅ (F24) · M1 golden eval ✅ (fold/semantics gates PASS) ·
  M2 tournament ✅ · M3 encoder ✅ (worse than vote) · M5 bootstrap ✅
- M4 typed encoder: **deprioritized** — vote ≥ encoders ⇒ encoder story moot;
  remaining value = presentation of the best observed honest performance +
  vote-as-SOTA result.
- M6 write-up: this file + REPRO_LEDGER F24-F32. **EXPERIMENTS FROZEN** at
  final-validation-suite completion — frozen leaderboard in
  `FROZEN_LEADERBOARD.md/json`; next phase = manuscript.
