# G6 Baseline Preregistration — DRAFT (not executed)

## Purpose
Give fold-level context for the P5 exploratory macro-F1 by scoring a
meaningful constant/prior baseline under IDENTICAL protocol: same locked
fold, same label convention (multi-hot 5-type), same threshold procedure
(inner-val grid 0.05–0.95 step 0.05, per-class argmax-F1), same primary
metric (MULTIHOT_MACRO_F1, exclude_and_report zero support), same
aggregation (mean-of-fold + pooled TP/FP/FN).

## Proposed arms (all constant/prior — no learned parameters)
- **G6-B1 per-class prior constant**: score p_k = train-set label frequency
  of class k for every test pair; thresholds fit on inner-val exactly as P5.
- **G6-B2 always-positive** (τ=0): analytic reference; included for
  completeness, labeled null not baseline.

## Frozen decisions BEFORE any run
- Fold: fold0_seed1234_static (same SHA-verified manifest).
- No model, no training — predictions are closed-form.
- Metrics: primary + aux TOP1_ANYHIT/FIRSTPOS; TP/FP/FN per class recorded.
- Comparison rule: report model−baseline Δ per class and macro; NO
  significance claims from a single fold; NO tuning on exposed fold-0 test.

## Explicit non-goals
No additional folds/seeds, no learned baseline, no rescue arm, no use of
fold-0 test result to select any hyperparameter.
