# CONTRADICTIONS_AND_ERRATA — ledger of corrected historical claims

Errata are additive; historic numbers are never altered in place.

| # | Historical claim (location) | Corrected status | Evidence |
|---|---|---|---|
| C1 | "Official release ignores reconstructed similarity Ŝ" (report/U1_FIDELITY_GAP_AUDIT_2026-11-15.md) | PARTIALLY WRONG: `create_hetero_data_optimized` does consume `mi_sim_recon`/`dis_sim_recon` and updates HGT edge_index; but prebuilt HGCN G matrices are NOT rebuilt. | FIDELITY_MATRIX.md §Sửa; upstream_graph_fixture.json; upstream code L-sites |
| C2 | "v3.2 Top-1 F1 = 0.0 = class collapse" (Plan F–J reports) | VOID: artifact of the released metric dropping all type-5 rows and len-6 predictions. Real signal on our tensor ≈0.27–0.36 under corrected metric. | CLAUDE.md §12; R/test_metric_contracts.py `test_k4_release_drops_type5_and_len6` |
| C3 | "loss fully MATCH after declared reductions" (several audits) | OVERSTATED: denominators differ (weighted positive mass vs sample-mean; Gram vs L2-MSE recon). Names of λ match; arithmetic does not. | R/test_fidelity_microfixtures.py |
| C4 | "triplet split reusable for type task" (NEXT_REPRO_PROTOCOL §P.3) | WRONG for CVtype: triplet permutation straddles pairs; superseded by PROTOCOL_ERRATA E-1 — reuse existing pair-fold family. | R/test_split_contracts.py |
| C5 | "all_unknown pool = contaminated by held-out positives" (early triplet report) | WRONG interpretation: it is a label-verified oracle pool from full Y, excluding held-out positives (true-zero guaranteed); reported separately as oracle-only, not info-matched. | triplet_u1/CVTRIPLET_U1X_FINAL_AUDIT.md |
| C6 | "outer-test is a blind evaluation" (implicit in early runs) | CORRECTED framing: fold0 test was viewed during development; pairs recur across folds; 650-epoch horizon chosen post-hoc. All outer numbers are retrospective, not blind confirmatory. | outer_refit PROTOCOL.md exposure note |
| C7 | Density as the gap lever | REFUTED by control: v3.2_wang_dense (385×275 @10.3%) scored 0.3336 ≈ wang 0.3301 — matching paper density does NOT close the gap. | results/v32_dense_fullbilinear.json |
| C8 | "harmonic Top-1 = our macro-F1" (informal) | WRONG: distinct formulas (fixture counterexample 0.2154 vs 0.0971). | R/test_metric_contracts.py |
