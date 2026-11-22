# P8 Evaluation Protocol Draft — DESIGN ONLY, no launches

## Claim tiers — never conflate
| tier | question | evidence standard |
|---|---|---|
| T1 Author-faithful reproduction | does the released pipeline reproduce paper numbers? | exact released code+data+protocol; ceiling limited by provenance gaps |
| T2 Scientific ablation of this implementation | which component causes failure? | locked paired comparisons vs preregistered baselines; exploratory while folds exposed |
| T3 Independent biological generalization | does the model predict unseen MDA types? | requires leakage-controlled splits AND certified label-free features |

## Leakage-control requirements for T3
1. **Split-aware feature construction**: every input (sims, GIP, graphs,
   negative pools, class weights) reconstructed from outer-train pairs
   only. Bundled views blocked until provenance-certified (P8 readiness:
   sim_m2 and bundled arrays currently cannot be certified).
2. **Two leakage axes separated**:
   - pair-level: held-out (m,d) pairs — current G5–G7 protocol.
   - entity-level: unseen miRNA or disease rows/cols removed before ANY
     feature computation (similarity rows for held entities excluded).
   - source-level: upstream label dependence of each view certified or
     excluded (worst offender: sim_m2/MISIM).
3. **Evaluation**: primary metric preregistered (MULTIHOT_MACRO_F1);
   inner-val thresholds; single outer-test score; B0/B1 analytic
   comparators computed per split; per-class AUROC/AP reported.
4. **Statistics**: ≥3 seeds × 5 folds on FRESH (never exposed) splits;
   paired-difference CIs; exploratory labels removed only when folds are
   genuinely unseen.

## Proposed sequence (design only)
P8.1 certify/reconstruct bundled views on train-A → P8.2 entity-level
splits + provenance canary → P8.3 frozen config re-run → P8.4 report.
Nothing in this document authorizes an experiment.
