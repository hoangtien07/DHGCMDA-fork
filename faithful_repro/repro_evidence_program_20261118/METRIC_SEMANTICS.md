# METRIC_SEMANTICS — explicit Top-1 definitions, no aliasing

WP1.2 tribunal output. Verified by `test_metric_contracts.py`
(11 fixtures PASS, including byte-level equivalence against the
pinned author function on all inputs it can score).

## The two non-identical "Top-1 F1" families

| Name | Formula | Truth convention | Source |
|---|---|---|---|
| `AUTHOR_RELEASE_TOP1_HARMONIC` (impl: `release_top1_harmonic_k4`) | P = top-1 accuracy; R = mean over types present of per-type recall; F1 = 2PR/(P+R) | scalar 1-4 map, or multi-hot → FIRST positive index | pinned_release/Calculate_Metrics.py L263-378 (main@06f9ba3) |
| `GENERAL_K_TOP1_HARMONIC` (impl: `generalized_top1_harmonic`) | identical formula family, K arbitrary | declared convention: `first_positive` (author-faithful generalization) or `any_hit` | R/metric_contracts.py |
| `PER_CLASS_MACRO_F1` (impl: `per_class_macro_f1`) | argmax-onehot vs multi-hot; per-class F1 averaged over ALL K channels | multi-hot | paper_guided/u1.../source/metrics.py; sklearn-identical (fixture PASS) |
| `MICRO_F1`, `ANY_HIT` | pooled TP/FP/FN; fraction argmax hits any truth channel | multi-hot | same |

Harmonic-of-aggregates ≠ macro-F1: fixture counterexample
(0.2154 vs 0.0971 on the same predictions). Averaging direction
differs (per-class mean of F1 vs F1 of two means) — these must never
be pooled in one column.

## Verified defects / limits in the released metric

1. Scalar truth map only covers types 1–4; `else: continue` drops
   type-5 rows (Calculate_Metrics.py L309-318).
2. Prediction accepted only at length 5 ([exist|4 types]) or 4;
   v3.2's 6-channel output is dropped entirely (L323-331).
   Net effect on v3.2: `valid_samples == 0` → F1 = 0.0 regardless of
   model quality (also proven in-repo: synthetic perfect predictor
   scores 0.0 — CLAUDE.md §12).
3. Multi-hot truth collapses to first positive index — a documented
   convention, not a bug, but it makes the metric blind to
   181 collapsed labels on v2.0 data and ~24% multi-type pairs on v3.2.

## What the paper's 0.8600 uses — INSUFFICIENT EVIDENCE

Paper p.24 reports CV_type Top-1 P=0.7915/R=0.9421/F1=0.8600; the F1
is the harmonic mean of the reported P and R (arithmetically
consistent). But (a) the released metric cannot score five types;
(b) no v3.2 eval code or fold indices were released. Therefore the
0.8600 was produced by an UNRELEASED metric/split configuration.
Any honest comparison must label ours `GENERAL_K_TOP1_HARMONIC`
(a family member, not proven author-identical) and report
`PER_CLASS_MACRO_F1` separately — never silently substituted.

## Multi-label handling (predeclared)

Predictions score a single type per pair (argmax). Against multi-hot
truth we evaluate BOTH declared conventions and keep them separate:
- `first_positive` — author-faithful; penalizes correct-but-not-first.
- `any_hit` — lenient; counts a hit on ANY positive channel.
- `per_class_macro_f1` — channel-wise view, convention-free.
Confusion matrices and per-type supports are always exported
(`confusion_and_support`) so no type — especially type 5 — is dropped.
