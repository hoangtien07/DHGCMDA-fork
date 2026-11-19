# REPRO_MASTER_EVIDENCE_LEDGER — DHGCMDA independent reproduction forensics

Base: `31272c5` (branch `devin/20261119-ind-repro-forensics`).
Scope: reconstruct/test the DHGCMDA HMDD v3.2 CVtype result. NO author contact.
Every claim below cites its source level: [paper] [code] [measured] [vendor] [suy luận] [giả định].

## Errata hierarchy (newest wins)
| claim | status | superseded by |
|---|---|---|
| density "4.6%" | RETRACTED | measured 10.5476% (DENSITY_ERRATUM.md) |
| "v3.2 all predictions collapse → F1=0" | RETRACTED | evaluator K4 bug (metric_feasibility fixtures); real F1 0.27–0.36 |
| "dataset unavailable" | RETRACTED | strict tensor recovered from SPLD MDAv3.2-3 (identity L1) |
| "dataset gap = cause of 0.86 gap" | RETRACTED | density-controlled run showed density not the cause (J/K plans) |
| runs_g2 baselines | SUPERSEDED_UNVERIFIED | runs_g2c (inner-val leakage fixed) |

## Claims ledger
| # | claim | source | status |
|---|---|---|---|
| L1 | paper v3.2 CVtype: P=0.7915 R=0.9421 F1=0.8600 | paper p.24 table [paper] | published value |
| L2 | v2.0 F1=0.5970, evaluator K4-only drops K5 rows | paper p.24; Calculate_Metrics.py L309-331 [code] | verified |
| L3 | strict tensor 411×271×5, N=8735, L=11748, support [1155,403,2293,3997,3900], 2110 multi | sha 85bb9ea4 measured | measured |
| L4 | tensor == SPLDHyperAWNTF MDAv3.2-3 snapshot | DATASET_IDENTITY_EVIDENCE.md | L1 stats identical; L2/L3 not byte-verified vs author |
| L5 | density 10.5476% matches paper | measured | consistent |
| L6 | oracle R bound 0.8263 (multi-hot, 1 pred) | metric_feasibility_check.py | proven |
| L7 | joint (P=.7915,R=.9421) infeasible under ALL source-supported conventions | joint_pr_feasibility.py this WP | NEW — see JOINT_PR_FIXTURES.json |
| L8 | evaluator code releases K4-only, silently drops K5 | Calculate_Metrics.py L309-331 | verified |
| L9 | U1 rescore: harmonic top1 0.4558–0.4924; priors ≈90% of model | RESCORE_VERIFICATION (50/50) | verified, OUR_INTERPRETATION |
| L10 | leak-paired delta: v2.0 +0.049 F1, v3.2 ≈0 | EXISTING_RESULTS_REEVALUATION | paired, verified |
| L11 | Track N A1 −0.014 vs B1 baseline | track_n G4 | negative, this session |
| L12 | ablation inversion (w/o-CL/HGT beat baseline) | Plan B/F, 4 methods | verified, genuine |
| L13 | full bilinear v2.0 0.6350 > paper 0.5970 | J-1 | verified |

## Reproducibility obstacles (ranked by impact)
1. **UNRESOLVED_EVALUATION_CONVENTION** (NEW, WP1): no source-supported convention reaches R=0.9421 at P=0.7915 — bounds 0.758–0.917 < 0.9421.
2. Author eval code / fold split / exact feature inputs unreleased.
3. Feature view provenance partially UNKNOWN (see WP2).
4. K5 evaluator absent in release (K4 only).

## Reconciled contradictions
- "all v3.2 collapsed" → evaluator bug (L8), real F1 0.27–0.36 (Plan K).
- "dataset unavailable" → recovered (L4).
- runs_g2 baselines 0.5184 → corrected 0.5298 after leakage fix.
