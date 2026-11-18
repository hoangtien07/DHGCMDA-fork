# CVtriplet-U1X FINAL AUDIT — post-hoc, no retrain

WP1 of COUNCIL decision 2026-11-18 (ACCEPT_NEGATIVE_RESULT).
All numbers below are computed from the LOCKED `eval_arrays.npz`
artifacts (`audit_posthoc.json`, produced by `audit_posthoc.py`).
No metric re-definition, no config change, no training. The original
5×3 results stand; this audit explains WHERE the baseline advantage
comes from.

## 1. Negative-result confirmed with paired statistics

Paired difference (same eval replicates, per fold, mean over eval
seeds) U1X − BL[mask_zero_pairs] Task-T AUROC:

| fold | U1X | BL_mz | BL_or | Δ_mz | Δ_or |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8535 | 0.9221 | 0.9327 | −0.0686 | −0.0792 |
| 1 | 0.8606 | 0.9205 | 0.9327 | −0.0599 | −0.0721 |
| 2 | 0.8605 | 0.9181 | 0.9303 | −0.0576 | −0.0698 |
| 3 | 0.8566 | 0.9150 | 0.9286 | −0.0584 | −0.0720 |
| 4 | 0.8431 | 0.9197 | 0.9307 | −0.0767 | −0.0876 |

Negative on EVERY fold (grand mean Δ_mz = −0.0642). The negative
verdict is stable and not a seed artifact.

## 2. Corrected interpretation of `all_unknown` (council fix)

`all_unknown` negatives are sampled from cells where Y==0 — i.e.
**label-verified negatives**: the pool excludes every held-out
positive by construction because it is generated from the full label
tensor. This is a transductive ORACLE (uses test-time label
information that neither train_mask nor the model could have). It is
NOT "contaminated by held-out positives" — the earlier phrasing was
wrong; corrected here. Its remaining oracle advantage: negatives are
guaranteed true zeros, so the baseline never trains on a held-out
positive labeled negative. `mask_zero_pairs` remains the only
information-matched variant (PU semantics identical to U1X's
existence-head negatives and to upstream zero_index).

## 3. Hard-negative AUROC by stratum (seed0 replicate, fold0; n_pos=2350)

| stratum | n_neg | prevalence | U1X | BL_mz | BL_or |
|---|---:|---:|---:|---:|---:|
| neg_on_train_pair | 115 | 0.953 | **0.732** | 0.320 | 0.679 |
| neg_on_heldout_pair | 34 | 0.986 | 0.715 | **0.832** | 0.810 |
| neg_pair_zero | 2201 | 0.516 | 0.857 | **0.953** | 0.947 |

- **Inversion on the hardest stratum**: on negatives sitting on
  train-mask pairs (cells the model saw as "pair exists"), U1X
  outperforms the info-matched baseline 0.732 vs 0.320. Count
  features make those negatives look strongly positive (miRNA/disease
  have rich train counts) → baseline collapses; the exist-head ranks
  them correctly. Sample size is small (115) — treat as directional.
- The overall 0.92-vs-0.85 gap is produced almost entirely by
  `neg_pair_zero` (94% of negatives), where counts trivially mark the
  negative via pair_count=0. The headline AUROC comparison is
  dominated by the easy stratum — the hard strata tell the opposite
  story. This is a data/eval-construction limit, not a U1X victory
  claim: the joint experiment's eval design cannot separate "good
  model" from "easy negatives".
- On held-out-pair negatives (n=34, smallest) the baseline wins
  0.832 vs 0.715 — again small-n, directional.

Positive-side split (vs all negatives): pos_pair_seen 854 — U1X
0.850, BL_mz **0.992**; pos_pair_unseen 1496 — U1X 0.848, BL_mz
**0.879**. Baseline's edge lives in count-rich seen pairs.

## 4. Sensitivity — train/eval negative overlap removed

Dropping eval negatives that overlap each baseline's train-negative
sample (recomputed deterministically from the pinned rng):

| pool | dropped (of 2350) | U1X | BL | vs. unfiltered |
|---|---:|---:|---:|---|
| mask_zero_pairs | 31 (1.3%) | 0.8494 | 0.9198 | Δ ~ +0.0006/−0.0023 |
| all_unknown | 37 (1.6%) | 0.8490 | 0.9328 | Δ ~ −0.0005/−0.0000 |

Overlap is ~1.5% and immaterial to every conclusion (paired diffs
unchanged to the third decimal). Declared, quantified, closed.

## 5. Conclusions allowed by this audit (no new claims)

1. The negative result is driven by eval construction: ~94% pair-zero
   negatives are separable by pair-count alone.
2. U1X is directionally BETTER on the informative hard stratum
   (neg_on_train_pair) — consistent with a learned existence
   representation rather than count memorization, at n=115.
3. Baseline's transductive-oracle gain (+0.012) quantifies the value
   of label-verified negatives: small but present.
4. Nothing here changes the acceptance verdict; the experiment stays
   a controlled supplementary negative result. AP interpretation is
   deliberately avoided in strata with prevalence 0.95+ (ranking
   metrics only, per REV2).
