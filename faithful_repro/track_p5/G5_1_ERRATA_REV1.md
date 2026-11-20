# G5.1 Errata Rev-1 — corrections to G5_1 documents (originals preserved)

Applies to: `G5_1_PREDICTION_COLLAPSE_AUDIT.md`, `G5_1_OBJECTIVE_AND_GRAPH_DIAGNOSTICS.md` (unchanged on disk).

1. **Coverage ratio corrected**: "≈95%" → **95.28%** = always-positive null macro-F1 (0.3894) / model macro-F1 (0.4087), measured exactly.
2. **Retracted**: the assertion that reconstruction *gradients* are numerically dominant. Retained measured facts only: l_rec = 99.53% of the total loss at ep160 and clip_active on 160/160 steps. Per-component gradient norms were never logged; dominance of the gradient by recon is UNVERIFIED.
3. **Terminology corrected**: the finding is **decision collapse after thresholding** (τ at grid floor → 100% positive predictions on classes 2–4). Whether the underlying score ranking is also collapsed is UNVERIFIED — not claimed.
