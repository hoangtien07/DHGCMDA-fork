# CVtriplet-U1 REV2 — full 5×3 results (final)

15/15 jobs RUN_PASS (TRAIN+EVAL+VERIFY per job). Protocol:
PROTOCOL_CVTRIPLET_U1_REV2 + AMENDMENT_REV2_B1_B4. No config tuning
after sanity; primary metric = AUROC Task T. This is a controlled
supplementary experiment — NOT paper-equivalence evidence; no
comparison to paper F1, no author-faithful claim.

## Aggregate (mean ± std over 15 jobs; per-job means over 5 eval seeds)

| Model | Task T AUROC | Task T AP | Task P AUROC |
|---|---:|---:|---:|
| U1X (exist×softmax joint) | 0.8548 ± 0.0126 | 0.8184 ± 0.0179 | 0.8052 ± 0.0121 |
| BL logistic `mask_zero_pairs` (info-matched) | 0.9191 ± 0.0024 | — | — |
| BL logistic `all_unknown` (oracle-only, separate) | 0.9310 ± 0.0015 | — | — |

**Verdict: negative result.** Under information-matched eval replicates
the train-only count-feature logistic baseline outperforms U1X by
~0.064 AUROC on Task T. The oracle variant adds ~0.012 — quantifying
the information asymmetry the original sampling had.

## Per-job Task T AUROC (fold × seed)

| Fold | seed1234 | seed42 | seed2025 |
|---|---:|---:|---:|
| 0 | 0.8488 | 0.8664 | 0.8452 |
| 1 | 0.8758 | 0.8673 | 0.8387 |
| 2 | 0.8453 | 0.8664 | 0.8698 |
| 3 | 0.8604 | 0.8632 | 0.8461 |
| 4 | 0.8494 | 0.8487 | 0.8312 |

Task P per-job AUROC range 0.773–0.821. Baseline identical across seeds
within a fold (baseline uses rng(3000+fold) only — expected).

## Strata (per fold, seed0 replicate)

Negatives ≈ 94% `neg_pair_zero` (~2200), ~5% `neg_on_train_pair`
(~110), ~1.5% `neg_on_heldout_pair` (~35). Positives: 34–37% on
train-seen pairs, the rest unseen. train/eval neg overlap ≈ 35–40 of
2350 (~1.6%) — measured and declared (PU-style pool overlap is
structural, not leakage).

F1@0.5 Task T ≈ 0.005–0.01 — fixed-threshold degenerate on the joint
score scale (≲0.2); AUROC/AP are the meaningful metrics (declared in
REV2).

## Interpretation limits

- The eval construction is dominated by pair-zero negatives; count
  features separate those trivially via pair-count zero/one — the
  task's difficulty is mostly in the easy stratum, which the baseline
  exploits and the deep model does not need to.
- Existence-head supervision is PU (mask-zero includes held-out
  positives, upstream semantics preserved) — F1 calibrations are
  untrustworthy; AUROC is rank-based and unaffected.
- No sweep was run; U1X is the single pinned configuration. The gap
  may shrink with tuning but nothing suggests crossing 0.92.
- Execution errors: zero aborted jobs, zero verification failures;
  two early session crashes (eval tuple bug, meshgrid OOM) were fixed
  BEFORE the locked fleet ran — recorded in job logs.

## Files

runs/fold{0..4}_seed{1234,42,2025}/{status.json,completion.json,
eval_full.json,eval_arrays.npz,verification_full.json,progress.log,
job_logs}; supervisor_status.json; preflight.json (B1–B4 pins);
preflight_rev2_sanity.json (history preserved).
