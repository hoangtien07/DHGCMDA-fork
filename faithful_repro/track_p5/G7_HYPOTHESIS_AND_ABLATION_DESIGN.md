# G7 Hypothesis & Ablation Design — DESIGN ONLY, no launch

## Evidence-ranked hypotheses
| rank | hypothesis | support (existing evidence) |
|---|---|---|
| **H1** (strongest) | Reconstruction term distorts task-oriented gradients | measured: recon grad ~10^4× task grad, all steps clipped, cosine ≈0 → applied update direction ≈ recon. Loss share 99.5% CONFIRMED by attribution. |
| **H3** | Softmax multi-hot objective mismatched with multilabel task | consistent with decision collapse: per-class softmax mass concentrates; val-optimal τ hits grid floor on classes with ≥30% prevalence; class-4 AUROC<0.5. Indirect (no counterfactual loss tested). |
| **H2** | Refresh instability hurts | mechanistically plausible (S mass at θ=0.5 boundary; fold graphs span 6%→>100% dense) but val-history shows no aligned step changes — weakest causal support. |

## Proposed ablation arms (each = ONE fold × seed1234 × locked config, exploratory)
- **G7-A1 (H1)**: `recon_reduction='mean'` (and/or lam3=0.01) — tests whether reducing recon magnitude frees task gradients. Requires unlock of ONE locked key under explicit owner amendment — flagged as protocol deviation, not silent.
- **G7-A2 (H3)**: replace softmax type head with per-class sigmoid + BCE (code change in p5_runner only) — tests objective-mismatch.
- **G7-A3 (H2)**: `update_every=650` (single refresh at end ≈ static graph) or freeze refresh — tests instability.
- Controls: B0/B1 comparators rerun on every arm; metrics/threshold/grid identical.

## Rules
- Every arm exploratory (all 5 folds already exposed); NO confirmatory claim; no tuning on exposed test results; each arm preregistered before launch with a fixed decision rule comparing macro-F1 vs P5 (Δ>+0.01 = hypothesis supported, else not).
- Do NOT bundle arms; one mechanism variable per run.
