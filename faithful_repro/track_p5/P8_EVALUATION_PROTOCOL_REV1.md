# P8 Evaluation Protocol REV1 — DESIGN ONLY
Supersedes P8_EVALUATION_PROTOCOL_DRAFT.md (kept, unchanged).

## Claim tiers (unchanged structure, sharpened)
- **T1 author-faithful reproduction** — requires the exact shipped four-view
  inputs incl. bundled M_MISIM/D_MESH and author splits; ceiling limited by
  unverifiable provenance and unknown folds. May only claim "benchmark
  configuration behavior", never leakage-free skill.
- **T2 scientific ablation** — paired locked comparisons on exposed folds;
  exploratory forever once folds seen.
- **T3 independent biological generalization** — requires T3-inputs
  (label-free or train-masked) AND fresh splits AND frozen config BEFORE
  any confirmatory data access. **No author-faithfulness claim may be made
  for a modified three-view architecture** — it is a declared variant, tier T3.

## Generalization axes — separate evaluations, not one
| axis | held out | what it tests |
|---|---|---|
| pair-level | (m,d) pairs | link/type prediction on seen entities |
| miRNA-cold | entire miRNA rows | generalization to unseen miRNA — all views' m-side rows for held miRNAs must be computable label-free |
| disease-cold | entire disease rows | generalization to unseen diseases — same requirement d-side |

## Permitted external features for held-out entities
- miRNA-cold: sequence similarity (sim_m1-class: miRBase/GFF3, label-free) OK;
  functional (MISIM-class) **forbidden unless recomputed from outer-train
  associations only**; existence/existence-side features derived from the
  entity's own non-MDA data OK.
- disease-cold: semantic (MeSH tree, label-free) OK; gene-based FunSim
  (CTD/HumanNet, no MDA input) OK; any feature consuming the full
  association matrix forbidden.
- Any feature built on the complete A (GIP-on-full-A, MISIM-on-full-A) is
  **forbidden in all T3 evaluations**.

## Provenance constraints
- Every input needs a recorded source snapshot + hash + construction
  date; unverifiable inputs are excluded, not assumed clean.
- Splits must be regenerated after input freeze; fold manifests hashed and
  bound to input hashes.
- Model + full configuration frozen BEFORE first access to any
  confirmatory dataset; freeze recorded in a preregistration lock.

## Statistics
- Repeated folds/seeds on the same data are **dependent measurements** —
  report paired differences and descriptive dispersion (ddof declared);
  no independence assumption, no significance claims on exposed folds.
- Genuinely independent validation = external dataset or temporal split
  (post-cutoff labels) — different claim tier, stated separately.

## Distinction: new CV splits ≠ independent validation
Fresh splits of the same benchmark control fold-exposure only; they do NOT
fix source-level contamination (sim_m2) or external validity. Label them
`fresh_split_same_benchmark`, never `independent`.
