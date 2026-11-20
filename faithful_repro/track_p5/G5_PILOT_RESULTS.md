# G5 Pilot Results — fold0 × seed1234 × P5_BENCHMARK_UNKNOWN_PROVENANCE

**Verdict: `G5_COMPLETED_VALID_EXPLORATORY`** (claim ceiling: exploratory_only — NOT an author-metric comparison)

## Run outcome
- outcome: `completed` — early stop at epoch 160 (best val at ep60 + patience 10 evals × eval_every 10)
- optimizer_steps: 160 · selected_epoch: 60 · wall_s: 31.0 (< cap 10800) · stop: patience exhausted
- hard stops: none triggered; all losses/grads finite throughout

## Primary result (outer test, evaluated exactly once after freeze)
- **MULTIHOT_MACRO_F1 = 0.4087** (support [201,74,471,788,772], none excluded)
- per-class F1: [0.2514, 0.1327, 0.4247, 0.6217, 0.6129]
- best val macro-F1: 0.4219 @ep60 · thresholds: [0.20,0.25,0.05,0.05,0.05] (inner-val only)
- prediction cardinality mean: 3.60
- aux: see result.json aux_top1_*

## Numerical diagnostics (G5_NUMERICAL_DIAGNOSTICS.json)
- Loss composition: l_rec dominates (~99.6% of total, recon_reduction=sum — locked, expected); l_type ~1.08 flat; l_ex decays →0.018; contrastive ~14.5 weighted.
- Gradients: preclip norms 2.3k–51k → always clipped to 1.0 (clip_active on all 160 steps). No NaN/Inf.
- Param updates: ||Δθ||_total decays 0.152→0.094; top-update param consistently `hgt.0.conv.kqv_lin.lins.miRNA.weight`.
- Dynamic refresh: fired at every multiple of 5 (32 refreshes); graph objects changed every time; edge counts shift substantially at each refresh.
- val_history: 16 evals, plateau ~0.40–0.42.

## Baseline comparison (directive 7)
NO genuinely-equivalent comparator exists: Track N baselines ran on a different feature pipeline, different pair splits and a different threshold protocol. No comparison claimed.

## Interpretation caveats
- BENCHMARK profile uses bundled sim_* views = UNKNOWN_PROVENANCE → exploratory only.
- softmax+multi-hot CE remains an interpretation, not independent-Bernoulli.
- Result is ONE fold × ONE seed — no variance estimate; 0.4087 is a point value, not a fleet statistic.
- Author joint (P=0.7915,R=0.9421) remains unreachable per Track R theorem; no claim against it.

## Artifacts
`runs/g5_fold0_s1234/`: result.json, resolved_manifest.json (resolved args + data/fold SHAs + pinned grid + aggregation), diagnostics.json, checkpoint.pt (kept on VM, not committed — bulky).
