# Reproduction of "DHGCMDA: A Dual-view Heterogeneous Graph Contrastive Learning Framework for miRNA–Disease Association Type Prediction" (BMC Bioinformatics 2026) — A Partial Reproduction Report with Evidence Program

*Reproduction note compiled from `faithful_repro/repro_evidence_program_20261118/` —
an adversarial, gate-controlled evidence program (WP0–WP8). All artifacts,
verification scripts and row-level data are in the repository on branch
`devin/20261118-repro-evidence-program`.*

## Abstract

We attempted a faithful reproduction of DHGCMDA's headline result —
Top-1 F1 = 0.8600 for miRNA–disease association *type* prediction on
HMDD v3.2 under pair-wise CVtype. We reach **PARTIAL_REPRODUCTION_WITH_
EXTERNAL_BLOCKERS**: on a provenance-tightened strict tensor
(411×271×5, elementwise-identical to the published SPLD derivation)
under leakage-free pair-disjoint folds and a contract-verified metric
family, a DHGCMDA-family model (U1, declared adaptations) attains
harmonic Top-1 F1 = 0.4558–0.4924, versus 0.448 for train-only count
priors. The original 0.8600 can neither be confirmed nor refuted: the
paper's curated tensor, fold assignments, and evaluation code are
unreleased. Independently of the score, we document two reproducible
findings: (i) the author's released evaluation code cannot score the
paper's own v3.2 task — it silently drops every 5-type sample,
producing F1 = 0.0 for *any* model, including a perfect predictor;
(ii) under the released v2.0 pipeline, ablations inverted relative to
the paper's Fig. 4 persist after four independent verifications.

**Tóm tắt (VI).** Không tái hiện được F1 = 0.8600; trên tensor strict
sạch leakage với metric đã khóa, mô hình đạt ≈ 0.46–0.49. Ba artifact
của paper (tensor, fold, eval code) chưa public nên kết quả gốc không
thể xác nhận hay bác bỏ. Phát hiện độc lập: metric kèm release bỏ hết
mẫu v3.2 (release–eval disconnect), và đảo chiều ablation trên v2.0
là hiện tượng thật.

## 1. Introduction and claim under test

The paper (Sun Y. et al., BMC Bioinformatics 2026) reports, on HMDD
v3.2 curated to 411 miRNA × 271 diseases × 11,748 typed links
(10.5% density): CVtype Top-1 P = 0.7915, R = 0.9421,
F1 = 0.8600, plus CVtriplet binary metrics. The released code
(`CDMBlab/DHGCMDA`, main @ 06f9ba3) ships an HMDD v2.0 pipeline and a
metric class hard-coded to four association types.

We test three separable questions, locked before any measurement:
(Q1) pipeline correctness — does the released code implement the
paper? (Q2) score reproduction — can the headline number be reached
on publicly reconstructable data? (Q3) mechanism — which
architectural and protocol choices drive performance?

## 2. Materials and methods

**Dataset lock.** `v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz`
(sha256 85bb9ea4…): 411×271×5 tensor, elementwise-identical labels
and entity names to the SPLD-derived reference (sha 135e2050).
Four similarity views rebuilt with leak-free provenance:
S_seq = Needleman–Wunsch/GFF3, S_gene = FunSim, S_sem = MeSH-2019
rebuild, S_func = train-only rebuild. No association-derived
similarity (M_MISIM/D_MESH shipped) enters features.

**Split contract.** PAIR_CVTYPE: group-disjoint folds where all types
of a pair leave the training set together — machine-verified
test ∩ train = ∅, test ∩ val = ∅ across 30 evaluation files.
A counterexample fixture proves triplet-permutation folds are a
*different* partition and cannot answer pair questions (errata E-1).

**Metric contract.** A byte-faithful reimplementation of the author's
`OptimizedMetricCalculator.compute_top1_metrics` (release_top1_
harmonic_k4) is verified to produce identical outputs on fixture and
random inputs. A generalization (GENERAL_K_TOP1_HARMONIC,
first-positive convention) extends it to 5-type multi-hot labels;
PER_CLASS_MACRO_F1, MICRO_F1, and ANY_HIT are cross-checked against
scikit-learn. A fixture counterexample proves harmonic-of-aggregates
≠ macro-F1 (0.2154 vs 0.0971) — these are different questions.

**Model fidelity.** Every computation is traced paper ↔ author code
↔ U1 with delta categories (MODEL_LOSS_FIDELITY_TRACE). U1 is a
declared adaptation: normalized multi-hot weighted CE (different
denominator than the author's scalar-label CE), learned fusion gate,
Gram-form reconstruction — never claimed author-faithful.

## 3. Results

### 3.1 Rescore under contract metrics (WP4; 50/50 verification PASS)

Recomputed from stored row-level `pairs/y/logits` of 30 outer-refit
runs (5 folds × 3 seeds × {static, dynamic}) and fold-count priors;
rescored values reproduce the originally reported macro-F1 exactly.

| Arm | HARMONIC (first-pos) | MACRO_F1 | MICRO_F1 | ANY_HIT |
|---|---:|---:|---:|---:|
| U1 dynamic | 0.4558 ± .026 | 0.4045 | 0.4349 | 0.5099 |
| U1 static | 0.4924 ± .011 | 0.4403 | 0.4703 | 0.5514 |
| prior global | 0.2497 | 0.1256 | 0.3903 | 0.4576 |
| prior disease | 0.4241 | 0.3631 | 0.5539 | 0.6495 |
| prior miRNA | 0.2835 | 0.2582 | 0.4115 | 0.4824 |
| prior product | 0.4480 | 0.4296 | 0.5385 | 0.6314 |

Static ≥ dynamic in 14/15 paired cells; count priors reach ~90% of
model performance on aggregate metrics — the pair-fold task carries a
strong degree prior; the model's residual advantage concentrates in
macro-F1 on minority types.

### 3.2 The released metric cannot score v3.2 (release–eval disconnect)

`compute_top1_metrics` maps ground-truth types {1,2,3,4}→{0..3} with
`else: continue` (drops every type-5 row) and accepts predictions of
length 5 or 4 only (drops every 6-channel v3.2 prediction). For any
v3.2 model — including a synthetic perfect predictor — Top-1
F1 = 0.0. Every historical "v3.2 collapse = 0.0" in this repository
traces to this artifact, not to model failure (three independent
demonstrations: synthetic perfect predictor, 4-type equivalence
check, real-data re-measurement ≈ 0.27–0.49 under corrected metrics).

### 3.3 Ablation inversion on shipped v2.0 persists

w/o-CL and w/o-HGT ablations beat the baseline on released v2.0 —
opposite the paper's Fig. 4 — verified across additive switches,
true-retrain rebuilds, multi-seed, and two loss modes.

### 3.4 CVtriplet supplementary experiment (negative)

A controlled extension (U1X, existence head + triplet scoring,
declared adaptation) scores Task T AUROC 0.8548 vs train-only
logistic baseline 0.9191 under identical information conditions —
a calibrated negative result, not paper evidence.

## 4. Discussion

**Blockers, ranked by impact on comparability.** G1 — the paper's
curated 411×271 tensor is unreleased (our strict tensor is ~4.6%
density vs their 10.5%: a different, harder task). G2 — author fold
assignments unreleased. G3 — the released metric is inadmissible on
5-type data (§3.2): the paper's own F1 was produced by different,
unreleased evaluation code. G4 — task-definition mismatch (paper text
says "four types"; v3.2 has five; multi-label pairs ~24%).

We do **not** attribute the residual gap to any single blocker — no
isolating experiment exists, and each of G1–G4 can independently
account for a large share. We also do not read 0.46–0.49 as "partial
success against 0.86": the numbers sit on different tensors, label
spaces, and folds.

**What is reproducible today.** Pipeline-level reproduction of v2.0
(completed earlier, plus full-bilinear predictor exceeding the
release by +5.9% on its own metric — the released diagonal predictor
is measurably weaker than the paper's described bilinear). The
release–eval disconnect (§3.2) and the ablation inversion (§3.3)
are fully demonstrated, artifact-pinned findings.

**Limitations.** All scores are retrospective (folds seen during
development; 650-epoch horizon chosen post-hoc) — no p-values, no
blind confirmation. The SPLD "<3 pairs" filter reproduces the count
(11,748) but neither the paper's literal "<2" text nor the literal
"<3" rule reproduces the exact edge set (2 miRNAs with <2 pairs,
10 miRNA + 1 disease with <3) — the paper's filtering is UNRESOLVED.
v2.0 raw predictions were not locally rescoreable
(NOT_RESCORABLE_WITHOUT_ARTIFACT).

## 5. What would close the question

Three artifacts from the authors: (1) the curated v3.2 tensor or its
exact pipeline code, (2) fold assignments, (3) the evaluation script
producing Table/numbers 0.8600. With those, the question resolves in
~1 compute day. A draft request letter exists
(`ARTIFACT_REQUEST_TO_AUTHORS.md`), unsent.

## 6. Availability

Branch `devin/20261118-repro-evidence-program` of
`hoangtien07/DHGCMDA-fork`, directory
`faithful_repro/repro_evidence_program_20261118/`:
contracts and 20 verification fixtures (all PASS), rescore script +
crosswalk + verification JSON, fidelity trace, errata ledger,
Vietnamese thesis reports, GATE_REPORT (training skipped: question
already answered — `REPRO_EVIDENCE_PROGRAM_COMPLETE`).
