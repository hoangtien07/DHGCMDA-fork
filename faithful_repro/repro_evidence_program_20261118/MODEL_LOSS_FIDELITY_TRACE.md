# MODEL_LOSS_FIDELITY_TRACE — paper ↔ author v2.0 ↔ U1 v3.2

WP3 output. Each row cites paper location, author code (pinned
main@06f9ba3, files under `independent_v32_author_review_20261117/`
and `independent_v20_release_20261116/pinned_release/`), and internal
U1 code. Categories: `same` / `declared_port` / `adaptation` /
`unknown` / `incompatible`. Micro-fixtures in
`test_fidelity_microfixtures.py` (6 PASS) pin the denominators.

| Computation | Paper | Author code v2.0 | U1 v3.2 | Category |
|---|---|---|---|---|
| Predictor | "bilinear predictor" (presence + type) | `SimplifiedTypePredictor`: diagonal bilinear `σ(mi·diag(r)·dis)` existence, no temperature; softmax(type/T), T learnable init 2.0 clamp [0.5,5] (upstream_hetero_model.py L417–515) | `DiagBilinearType` type logits only (no existence channel in U1); U1X adds exist channel as declared adaptation | declared_port + adaptation (diag ≠ full bilinear — measured on v2.0: full beats diag by +5.9%, J-1) |
| Existence loss | implied by presence term | focal γ=2.0, α=0.5, on one_index vs subsampled zero_index (upstream_main L115–133); zero_index ≤10× one-shot, PU — includes held-out positives | U1: no existence objective; U1X: same focal + pinned rng(seed+77) PU pool | declared_port (U1X), absent in U1 |
| Type loss | Eq.32 combined | CE(EN weights β=0.99999, label_smoothing=0.1, reduction='mean' over positive samples) — scalar targets 1..4 (upstream_main L56–161) | normalized weighted multi-hot softmax CE: −Σw·y·logsoftmax / Σw·y over multi-label rows | adaptation — different denominator AND different label space (multi-hot vs scalar). Fixture: denominators diverge when multi-label rows present. Author code cannot represent a 2-type pair. |
| EN class weights | not specified | (1−βⁿ)/(1−β), β=0.99999, normalized to sum C — saturating curve, minority 403 gets ~9.7× weight of majority 3997 (fixture) | same formula, computed on train labels only | declared_port |
| Contrastive inter/intra | dual-view CL | mi_cl + dis_cl estimated `*0.3/(1+0.3)` (L874–875) | U1 objective: (λ1·intra + λ2·(inter + .1·margin))/M reduction | adaptation — same family, declared non-identical normalization |
| Reconstruction | Eq.19–21 Gram + Frobenius, λ3=1 | L2-normalize + MSE per modality, coef 0.15 | Gram raw, mask coverage, divided by weighted positive mass | adaptation — different formulation AND denominator; cannot compare coefficients alone |
| Graph update | dynamic every 5 epochs | `create_hetero_data_optimized` consumes `mi_sim_recon/dis_sim_recon` when reconstruction change exceeds threshold — HGT edge_index updated (NOT necessarily HGCN G matrices, which are prebuilt) | refresh every 5 epochs, rebuild graph/H from Gram + offdiag-minmax, θ=.5 | adaptation — same trigger family, different construction detail |
| Fusion | adaptive attention | fixed 0.6/0.4 mix in release | learned gate | incompatible |
| Eval unit | CVtype pairs / CVtriplet triplets | v2.0 folds in shipped pipeline; metric hardcodes 4 types | pair-fold group splits (verified disjoint), K=5 | task-level: pair fold same family; metric incompatible (WP1) |
| Checkpoint | early stop on validation | per code | inner-val best-of-65 (selection runs) / fixed-650 (outer refit) | adaptation — declared |
| Optimizer/init | — | Adam, params per param.py | same family, pinned cfg | same |

## Author-code fidelity vs paper-equation fidelity

Verified on the micro-fixtures: matching the released code
(`release_top1_harmonic_k4` byte-equivalence) does NOT establish
paper fidelity — the release cannot score the v3.2 task at all, and
its own predictor is a degenerate diag bilinear that measurably
underperforms the paper's described full bilinear (+5.9% on v2.0 when
replaced). "Faithful to the release" and "faithful to the paper" are
different claims; we hold them separate everywhere.

## Reduction-denominator findings (new, fixture-verified)

1. U1 type_loss denominator = weighted positive mass (multi-hot);
   author CE denominator = #positive samples (scalar labels). Same
   loss family, different effective per-sample weight when pairs are
   multi-label (24.2% of pairs).
2. Author focal existence applies α=0.5 to the negative term only;
   positives unweighted — replicated exactly in U1X.
3. `0.3·exist + 0.7·type` combination is upstream code, not paper
   Eq.32 — documented as paper spec gap (M5 in prior audits).
