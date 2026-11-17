# U1 implementation-fidelity audit — paper equations vs executed source
# (2026-11-15, read-only)

**Scope.** Paper equations (`_pdf_text/p15–p23, p25`) vs the sealed
executed source (`evidence_nested_20261115/pkg_root/src/`: `model.py`,
`stages/{c2_model,c4_model,c1_runner,c4_runner,intra,inter,
reconstruction,reconstruction_adapter,reconstruction_targets,
dynamic_graph}.py`, `losses.py`, `metrics.py`, `contracts.py`) and the
frozen arm-A config (`inputs/config_u1.json`: lam_intra=1.0,
lam_inter=0.3, lam_recon=1.0, t=.5, m=.5, margin_coef=.1, cap=10,
period=5, θ=.5, K=13, seed 1234, 650 epochs).
**Tags:** [F] file/text-verified · [I] inference · [U] unspecified in
paper · [C] paper-internal conflict.
No fitting, no scoring, no artifact changes.

## 1. Verified faithful (equation ↔ executed code)

| Eq/§ | Paper statement | Executed U1 | Verdict |
|---|---|---|---|
| 10–12 | `S_cross = ẐM ẐDᵀ / t`, row-normalized | `inter.py:152-154` `_norm_rows` + `/t`, t=.5 | MATCH |
| 13 | InfoNCE denom `exp(S_ij)+Σ_{k∈N_i}exp(S_ik)`, sampled `N_i` | `inter.py:158-170` logaddexp over row bank, positives excluded, `sample_negative_bank` per-row uniform w/o replacement | MATCH (N_i sampling rule [U]; cap-10 declared) |
| 14–15 | `max(0, m-(S̄P−S̄N))`, `L=InfoNCE+λ·margin` | `inter.py:174-178`, m=.5/λ=.1 **declared fallbacks [U]** | MATCH/unit-declared |
| 16–18 | AVF: GAP→2-layer FC→sigmoid α, `Ẑv=ReLU(αvZv)`, learnable β | `model.py` gate MLP (`W1: dim→dim, W2: dim→1`) + sigmoid + β — follows paper (official used fixed .6/.4) | MATCH |
| 6–9 | intra CL on view embeddings, proj head, α=.5, τ=.5 | `c1_runner._intra_terms` on `out['views']` (z1m/z2m/z1d/z2d) via `proj_head`s; Eq.7-exact denominator, hand-verified `stages/test_intra.py` | MATCH |
| 19–20 | `Z_proj=W2ReLU(W1 Zfused)`, `Ŝ=Z_proj Z_projᵀ` raw | `reconstruction.py:97` `projected @ projected.T` raw Gram — not cosine, unclipped | MATCH (official's cosine was the deviation; U1 corrected) |
| 21 | `‖S−Ŝ‖²_F` Frobenius | masked valid-cell residual², sum+mean both emitted; applied term = `(sum_mi+sum_dis)/(valid_mi+valid_dis)` — declared "global valid-cell weighted mean" | MATCH shape; reduction unit declared (not paper's literal n² sum) |
| §2.6/2.8 | **reconstructed Ŝ drives dynamic update**; edges θ=.5; every 5 epochs; KNN hypergraphs rebuilt; assoc edges unchanged | `dynamic_graph.py:166-240` consumes `gram_mi/gram_dis`, min-max off-diag norm → binary edges θ=.5 + `_knn_incidence` Laplacians; assoc/rev edges cloned unchanged; `should_update(period=5)` | MATCH — better than official (which never rebuilt H and ignored Ŝ) |
| 22–29 | type-proj → 2×HGT 4-head, residual+LN+dropout | `model.py` HGTConv stack, heads=4, layers=2, hgt_dropout=.3 | MATCH ([U] dropout rate) |
| 30–31 vs §2.7 | binary threshold edges vs "edges weighted by Ŝ" | binary `edge_index`, no edge_attr consumed | [C] paper-internal; binary reading realized, declared (M24) |
| 32 | `L=Ltype+λ1Lintra+λ2Linter+λ3Lrecon`, λ=1.0/.3/1.0 | `c4_runner` docstring + config: identical composition, inter λ applied **once** (official double-added) | MATCH |
| 33 | `−Σ_F Σ_k w_k y_ij^k log p̂_ij^k` — **the paper's own formula is multi-hot** | `metrics.type_loss` weighted multi-label log-softmax CE, normalized by weighted positive mass | MATCH formula; normalization declared (optimum-equivalent under fixed w,y) |
| 34 | `w_k=(1−β)/(1−β^{n_k})`, β→1 | `contracts.class_weights_train` on **train** multi-hot, β=.99999, sum-normalized to C | MATCH (sum-norm declared) |
| §2.7 | "bilinear predictor … existence AND type" | `DiagBilinearType` type-only | **SEE R1/R2** |

## 2. Disposition of every remaining deviation — no demonstrable defect

Every non-matching behavior is either (a) a **declared contract**
recorded in manifests (normalized reductions, m/λ fallbacks, coverage
masks, train-only association access — a *stricter* regime than the
paper's transductive protocol, not a defect), (b) a **paper-internal
conflict** the code resolves openly ([C] edge weights; [C] existence
head promised in §2.7 but absent from Eq. 32), or (c) **[U]**
unspecified (dropout rates, hidden dims, N_i sampling rule, margin
values, early-stopping detail). None survives as an *objectively
demonstrable* defect: for each, "paper says X, code does Y, X≠Y" cannot
be established — either the paper is silent or the deviation is a
frozen, disclosed engineering unit. The two items below are the only
**unresolved actionable** fidelity issues, ranked.

## 3. Ranked unresolved actionable issues

### R1 — Diagonal bilinear readout (highest impact, actionable)

- **[F]** Paper §2.7 (p19): embeddings "fed into a **bilinear
  predictor**". Standard term-of-art reading is a full per-type matrix
  `z_miᵀ W_k z_dis` (diagonal forms are named DistMult/diagonal in the
  literature).
- **[F]** U1 executes `DiagBilinearType` (`model.py:160-186`): per-type
  **diagonal** `Σ_d mi_d·W_kd·di_d` — a degenerate rank-d special case,
  silently chosen. 5×128×128=81,920 vs 5×128=640 predictor params.
- **[F]** This exact contrast was measured on the OLD v2.0 pipeline
  (Plan J-1): full bilinear .6350 vs diag .5996 Top-1 F1 (+3.5pp, and
  .6350 vs paper's reported .597) — descriptive numerics only per REV3
  (cross-paper metric/fold alignment unestablished), but direct evidence
  the readout parameterization materially moves results. Untested on
  the U1/v3.2 path.
- **[I]** Consequence: readout rank caps type discrimination; on a
  5-class multi-hot task the diagonal form cannot model cross-feature
  interactions. Highest expected information per unit cost of any
  remaining fidelity question.
- Not a "defect" (underspecification), not disproven, not previously
  tested here → **predeclared experiment FB1 below**.

### R2 — Existence head (blocked, paper-internal)

- [F/C] §2.7 promises existence+type; Eq. 32 defines no existence term.
  U1 is Eq.32-literal (type-only). Official's `0.3·focal_exist` is an
  author-code addition that changes the task unit (C6 in the mechanism
  audit — blocked pending a declared estimand: does CV_type score
  existence?). **Not actionable without a coordinator declaration** —
  listed for completeness, not selected.

### R3 — Recon target view choice (declared, low impact)

- [F] U1 supervises mi:S_func + dis:S_sem (of four views), coverage-
  masked (uncovered disease rows excluded, diagonal excluded, miRNAs
  with zero train associations unsupervised). Paper says "the original
  similarity networks" without naming which of four → [U]; U1's choice
  is declared and hash-pinned. Recon is a regularizer; lower expected
  impact than R1.

### R4 — Inter margin constants (declared fallbacks)

- m=.5, λ=.1 are coordinator values [U]; at λ=.1 the term is near-inert.
  Low impact.

### R5 — Selection convention

- Fixed-650 + best-epoch inner-val selection (65 looks) vs paper's
  unspecified "early stopping … determined by validation" — disclosed
  convention; the boundary-epoch caveat is already on record.

## 4. Predeclared experiment FB1 — full-bilinear readout arm

**Question.** Does replacing `DiagBilinearType` with a full per-type
bilinear `miᵀ W_k di` (same readout projections, same C=5) change the
objective trajectory/metrics on the consumed fold-0 inner splits under
the sealed arm-A recipe?

**Status of evidence.** Descriptive mechanism contrast only: fold-0
inner-val/sub_check are development-exposed (65-look selection, prior
diagnostics); no fold is an unbiased estimate; the paper's "bilinear"
specification is [U] — neither arm can claim paper fidelity over the
other; result is reported as sensitivity, never as improvement or
reproduction.

**Design (frozen before any run).**
- Arm A0: sealed arm-A checkpoint/manifest (already exists — no refit;
  baseline reused byte-identical).
- Arm A1: identical everything except `readout_mode=full_bilinear`;
  same seed 1234, data bundle sha `85bb9ea4…`, bank sha `ba07599d…`,
  graphs, dynamic refresh, 650-epoch horizon, eval_every=10, optimizer.
- Implementation: isolated additive flag on the readout module only —
  default `diag` path byte-identical to current (F1-style anchor test:
  same inputs → same logits/params within 1e-6); `FullBilinearType`
  initialized INSIDE fork_rng so global RNG state is unchanged when the
  flag is off and deterministically shifted when on (recorded).
- Fixtures (must pass before the arm): hand-computed `miᵀW_kdi` vs
  implementation on 2×3 tensors; gradcheck; per-type independence (W_k
  receives grad only from its own logit); anchor test for diag path;
  RNG-state equality test for flag-off.
- Endpoints (all reported, none hidden): full val curves, final AND
  max-selected macro-F1/any-hit/micro-F1/weighted-CE (labeled order
  statistics), per-class F1/TP/FP/FN, paired hit-overlap vs arm-A and
  vs train-only priors, parameter count delta (+81,280).
- Cost: one 650-epoch fold-0 run on the sealed Colab CPU recipe
  (~2–4 h wall, same envelope as arm-A); local preflight dry-runs only.
- Stops: integrity-only (hash drift, contract failure). Underperformance
  is a reported result. No outer-test scoring anywhere.

## 5. Next executable action

Implement `FullBilinearType` + `readout_mode` flag + the five fixtures
(isolated, default-unchanged), then run arm A1 on the sealed Colab
recipe — one run, all endpoints published. Everything else ranked here
is either already declared or needs a coordinator declaration first
(R2's task-unit question).
