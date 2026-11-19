# G4_INDEPENDENT_REVIEW

**Verdict: REVIEW PASS — results trustworthy; H1 outcome is a genuine negative.**

Checks performed:
1. Metric recompute is a separate code path reading only frozen scores.npz/taus.json — matches the independent recompute in metrics_g2c/metrics_g4.
2. Provenance final gate ALL_PASS on all 3 tracks at split/inner-val/feature level using exact production builders.
3. Exposed sanity result NOT reused for tuning; the fold0/seed1234 cell rerun gave 0.5204 vs exposed 0.5201 (float-noise consistent, same frozen CFG).
4. Paired diffs computed per cell against the same-fold corrected baseline — 14/15 cells negative; effect consistent, not noise-driven.
5. U1-static comparison caveat: τ=0.5 fixed (frozen logits, no inner-val); only descriptive.
6. Old runs_g2 marked SUPERSEDED_UNVERIFIED and immutable — historical artifacts unchanged.

Limitations stated honestly: A1 arm is a deviation (similarity-row inputs, not U1 embeddings) — see A1_INPUT_AMENDMENT.md; H1 tested only against that arm. The negative result does not test the original U1-embedding design.
