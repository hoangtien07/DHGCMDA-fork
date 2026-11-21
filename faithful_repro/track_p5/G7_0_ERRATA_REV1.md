# G7.0 Errata Rev-1 — graph counts & gradient interpretation
Original `G7_0_GRAPH_MECHANISM_REPORT.md` and
`G7_0_CHECKPOINT_GRADIENT_ATTRIBUTION.md` preserved unchanged.

## WP1 — corrected selected-checkpoint edge entries
Source: `G7_0_ATTRIBUTION_RAW.json` (persisted selected `ei`), cross-checked
against the G6 diagnostics `refreshes` event `edge_counts_after` at each
selected epoch — every value matches exactly.

| fold | sel. epoch | assoc (each direction) | miRNA-sim | disease-sim |
|---|---:|---:|---:|---:|
| 1 | 180 | 6 289 | 63 728 | 7 412 |
| 2 | 90 | 6 289 | 44 954 | 540 |
| 3 | 150 | 6 289 | 53 662 | 5 642 |
| 4 | 120 | 6 289 | 32 420 | 7 042 |

**RETRACTED**: the fold-2 disease-sim figure of 148 082 edges and the
">100% density / duplicate or self edges" conclusion — both were errors in
the Rev-0 table (fold-2 disease-sim is actually 540, the sparsest selected
graph). No duplicates were measured anywhere.

Corrected densities (directed entries / N(N−1); N_miRNA=411 → 168 510,
N_disease=271 → 73 170). These are edge *entries* in `ei`; uniqueness of
the underlying node pairs was not separately measured.

| fold | miRNA-sim density | disease-sim density |
|---|---:|---:|
| 1 | 37.8% | 10.1% |
| 2 | 26.7% | 0.7% |
| 3 | 31.8% | 7.7% |
| 4 | 19.2% | 9.6% |

Revised qualitative claim (unchanged in kind): selected graphs still span a
wide density range across folds (disease-sim 0.7%→10.1%), consistent with
refresh-time instability — still **not** a causal claim about F1.

## WP2 — corrected gradient interpretation (measured values kept)
Keep all measured norms (recon 8.8k–16k vs type ~0.06) and cosines
(−0.039…+0.028). The precise statements are:

- Reconstruction dominates the **global checkpoint-conditioned gradient
  norm** (measured).
- Global-norm clipping rescales the **combined** gradient vector; it cannot
  restore the relative contribution of the type term inside that vector.
- Adam's actual parameter-update **direction** is NOT established by this
  measurement — Adam's first/second moment state was not persisted, so the
  conditioned single-step gradient is not the applied update.
- The recorded replay verifies **TP/FP/FN equality**, not bitwise logit
  equality.
- All measurements are `CHECKPOINT_CONDITIONED_GRADIENT_DIAGNOSTIC` —
  not exact historical training-time attribution (dropout masks, RNG and
  optimizer states were not persisted).
