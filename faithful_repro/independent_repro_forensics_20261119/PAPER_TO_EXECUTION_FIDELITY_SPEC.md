# PAPER_TO_EXECUTION_FIDELITY_SPEC — WP3

Consolidates `repro_evidence_program_20261118/MODEL_LOSS_FIDELITY_TRACE.md`
(authoritative table) + 6 micro-fixtures (all PASS on this branch).

## Three named protocols — never mix results
| name | definition | data | metric |
|---|---|---|---|
| RELEASE_K4_V20 | shipped pipeline as-is, K4 evaluator | v2.0 495×383×4 | release Top-1 K4 |
| PAPER_LITERAL_K5 | paper-equation reconstruction | strict 411×271×5 tensor | locked K5 suite |
| CLEAN_REPRO_K5 | mask-safe independent inference | same tensor + PUBLIC_RECONSTRUCTED_4VIEW | same |

## Paper-equation → op trace (condensed; full table in MODEL_LOSS_FIDELITY_TRACE)
- Predictor: paper "bilinear" — release is diagonal bilinear (measured −5.9% on v2.0 vs full). PAPER_LITERAL_K5 should use full bilinear; choice documented.
- Type loss: paper Eq.32 combined; release = scalar-target CE (cannot represent multi-label pairs — 24.2% of pairs). PAPER_LITERAL_K5 needs a defined multi-label rule → **underdetermined by paper** → option list: (a) first-positive scalar, (b) multi-hot weighted CE. Each must be a named option, not silently chosen.
- Existence loss: focal γ=2, α=0.5 on negatives (fixture-verified formula).
- EN weights β=0.99999 normalized to sum=C (fixture-verified).
- Reconstruction: paper Gram+Frobenius λ3=1.0; release MSE coef 0.15 — divergent; underdetermined → options.
- Fusion: release fixed 0.6/0.4, paper "adaptive attention" — learned gate is the paper reading.
- Graph refresh: every 5 epochs; HGCN G matrices are prebuilt (HGT edge_index updates only) — fixture pins G vs edge_index distinction.

## Underdetermined paper choices (must be explicit options)
1. scalar vs multi-label type-loss target (release is scalar-first)
2. fusion weights (fixed vs learned gate)
3. reconstruction formulation + coefficient
4. negative pool semantics for existence head (PU zero_index)
5. KNN/hypergraph K and similarity normalization inside views
