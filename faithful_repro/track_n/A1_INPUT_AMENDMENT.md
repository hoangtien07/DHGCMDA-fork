# A1_INPUT_AMENDMENT — Track N protocol amendment

**Base**: `3f872a1` → implemented on `devin/20261118-repro-evidence-program`.
**Date**: 2026-11-19

## Amendment (owner-selected, G4 decision)
A1 uses **option (a)**: similarity-row MLP — input = `concat(Sm[i,:], Sd[j,:])` with `Sm=(sim_m1+sim_m2)/2`, `Sd=(sim_d1+sim_d2)/2` — with weighted multi-label BCE (EN pos_weight, β=0.99999), PU auxiliary OFF.

## Declared deviation
- **Not** "U1 embeddings" as the protocol text originally implied. No U1 embeddings are generated (that would require its own training fleet).
- This is an architecture/input deviation from the preregistered protocol REV N.1; the arm tests "can a trainable non-linear model on similarity-view rows beat frequency/count baselines and U1 static", a defensible interpretation of H1 but narrower than the original U1-embedding arm.

## Exposure notice
- The fold0/seed1234 sanity result (macroF1 = 0.5201) was **already exposed** before this fleet run. It was NOT used to tune architecture, hyperparameters or thresholds; the frozen CFG is identical to the approved sanity spec. The G4 rerun of the same cell produced 0.5204 (deterministic re-run within float noise of the exposed 0.5201 — consistent).
