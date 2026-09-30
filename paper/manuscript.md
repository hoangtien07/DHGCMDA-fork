# Reproducing DHGCMDA: artifact recovery, evaluation audit, and a minimal leakage-free baseline for multi-type miRNA–disease association prediction on HMDD v3.2

**Status:** structured draft skeleton — all numbers frozen from `forensics/FROZEN_LEADERBOARD.json`; [ ] marks prose/expansion work remaining.

**Target venue (draft):** ReScience C replication track, or a short reproducibility note at a bioinformatics venue (Briefings in Bioinformatics reproducibility section / GigaScience technical note).

---

## Abstract (draft)

[To write — ~200 words. Spine: DHGCMDA reports Top-1 F1 = 0.860 for five-type miRNA–disease association prediction on HMDD v3.2, but its evaluation dataset (411 × 271 × 11,748 triplets) and evaluation code were never released. We recover the exact dataset byte-for-byte from a deleted companion repository via Software Heritage; replicate the strongest published baseline (SPLDHyperAWNTF) exactly on it (P=0.6219, micro-R=0.4624 — its Table-3 row in DHGCMDA); show the reported recall (0.9421) is mathematically impossible under the documented protocol (single-prediction macro-recall ceiling 0.8642); audit similarity-matrix provenance (MISIM is label-derived); and find that a zero-parameter similarity vote — GIP over the training fold alone — reaches F1 = 0.5910, the best observed performance, statistically indistinguishable from the published tensor SOTA and far above the released DHGCMDA implementation (0.4481).]

**Keywords:** reproducibility, miRNA–disease association, multi-type prediction, evaluation audit, benchmark hygiene

---

## 1. Introduction

- [ ] Task definition: five association types (genetics, epigenetics, circulation, target, tissue) — predicting *which types* link a miRNA–disease pair, not just whether one exists.
- [ ] The DHGCMDA claim: Top-1 hybrid F1 = 0.8600 (P=0.7915, R=0.9421) on HMDD v3.2 — nearly twice the strongest prior published baseline (SPLDHyperAWNTF ≈ 0.53–0.56).
- [ ] Problem: the 411×271 dataset subset, the authors' evaluation code, and the training configuration were never released; public code cannot even score v3.2 (its evaluator hard-codes four association types).
- Contributions (4): **(C1)** byte-exact recovery of the evaluation artifact; **(C2)** anchored reproducibility audit — exact replication of the SPLD baseline row vs. impossibility of the DHGCMDA claim; **(C3)** provenance/evaluation correction — three-track protocol separating honest, legacy, and leaky settings; **(C4)** a minimal leakage-free baseline that matches published SOTA with zero parameters.

## 2. Materials

### 2.1 The MDAv3.2-3 artifact (recovered)
- 411 miRNAs × 271 diseases × 11,748 typed triplets over 5 types
  [genetics 1,155; epigenetics 403; circulation 2,293; target 3,997; tissue 3,900];
  8,735 associated pairs (density 10.5%); 24.2% of associated pairs are multi-type.
- Similarity matrices shipped in the artifact: `mi_fun_sim` (411²), `DSSM` (271²),
  `MGSM` (411² GIP-style), plus entity name lists.
- **Figure 1** (`figures/F1_lineage`): curation lineage raw 1049×758 → TDRC 713×447 → MDAv3.2-3 411×271. The artifact was absent from the paper's repository; we recovered it from the deleted sibling repository `Ouyang-Dong/SPLDHyperAWNTF_Model` via Software Heritage (snapshot 2024-08-21; canonical SHA256 `c8f36c77…`); every table fingerprint matches the paper exactly.

### 2.2 Similarity provenance (audit result)
- `mi_fun_sim` ≡ **MISIM 2.0** to ~1e-10 → computed from HMDD associations → **label-derived → leaky as a static input**.
- `DSSM` correlates 0.90 with independently recomputed MeSH/DO similarity (TDRC Dis_sim) → ontology-derived → **static external (conditionally clean)**.
- GIP-style matrices rebuilt per fold from the masked training matrix → honest.

### 2.3 Evaluation protocol (documented vs. actual)
- Paper's stated protocol: 5-fold pair-level CV; mask all five channels of each test pair; one top-1 type prediction per pair; hybrid P/R/F1.
- **We verify this protocol exactly**: our `eval_top1.py` reproduces SPLD's fold membership (seed-0 shuffle) and metric semantics bit-for-bit.

## 3. Methods

### 3.1 Three-track evaluation protocol
- **PRIMARY HONEST:** similarities computed from the training fold only (GIP) plus external static ontology similarity (DSSM).
- **LEGACY COMPARISON:** SPLDHyperAWNTF as published (includes static MISIM internally).
- **LEAKY CONTROL:** static MISIM fed to baselines — measures the leak channel directly.

### 3.2 Models evaluated
- Similarity-vote family (`vote_*`/`simknn_*`): score(t) = row-norm(S_mi)·Y_train(t) [+ row-norm(S_dis)·Y_train(t)ᵀ]ᵀ [+ λ·tanh(S)⊗J]. Zero trained parameters.
- Factorization baselines: bilinear (d=32), DistMult (d=32), BCE + early stop on inner-val.
- Published models: SPLDHyperAWNTF (PARAFAC-style tensor, author code, golden replay); publicly available/reconstructed DHGCMDA implementation (leakage-free variant, multilabel head, honest eval).
- Floors: type co-occurrence prior, marginal predictor.

### 3.3 Significance testing
- Paired bootstrap (10,000 resamples over pooled outer-test pairs) for ΔF1 CIs.
- Learned models: init seeds 0/1/2 on fixed golden folds; vote family: repeated pair-CV draws (secondary).

## 4. Results

### 4.1 Anchored replication
- Our independent SPLD run on the recovered artifact: P=0.6219, micro-R=0.4624, macro-R=0.5069 → **exactly** the SPLD row in DHGCMDA Table 3. The artifact and our harness are thereby anchored to published reality.

### 4.2 The claimed result is unreachable under the documented protocol
- Single-prediction-per-pair bounds recall: micro ≤ 8,735/11,748 = **0.7435**; macro ≤ mean(1/posᵢ) = **0.8642**. The claimed R=0.9421 exceeds both → the paper's own eval cannot be the documented one (per-triplet or multi-prediction accounting, or a different split/metric — never released).
- **Figure 4** (`figures/F4_recall_ceiling`).

### 4.3 Frozen leaderboard
- **Table 1** (`tables/T1_leaderboard.tex`), **Figure 2** (`figures/F2_leaderboard`).
- Headline: `vote GIP-mi only` **0.5910** = best observed; SPLD static 0.5585 / no-MISIM 0.5594 / fold-MISIM 0.5570; dual-view vote 0.5599–0.5623; DHGCMDA public impl. 0.4481; bilinear 0.283±0.006; DistMult 0.252±0.007; co-occurrence floor 0.394.

### 4.4 The MISIM leak channel
- **Figure 3** (`figures/F3_leak_channel`). Vote: static MISIM +0.020 F1 (95% CI [+0.013, +0.030], significant); fold-reconstructed MISIM recovers only +0.013 of that. SPLD: all three configs within ±0.002 (n.s.) — its tensor model does not exploit the static matrix.
- **Table 2** (`tables/T2_bootstrap.tex`): vote vs SPLD Δ=+0.0014, CI ⊇ 0 → **no statistically significant advantage over the similarity-voting baseline under this protocol**.

### 4.5 Ablation of the vote
- **Table 3** (`tables/T3_ablation.tex`): GIP-mi alone 0.5910 > +DSSM 0.5599 > GIP+GIP 0.5655 > +J 0.5623 > fold-MISIM 0.5733 > static-MISIM (leak) 0.5799 > DSSM-only 0.4173. The miRNA functional neighborhood carries the signal; disease-side similarity voting actively hurts.

### 4.6 Robustness
- **Table 4** (`tables/T4_robustness.tex`): seeds 0/1/2 — bilinear 0.283±0.006, DistMult 0.252±0.007; vote rep-CV 0.561±0.001. Rankings are stable; conclusions do not depend on seed or fold draw.

## 5. Discussion

- [ ] Where performance comes from: miRNA functional similarity + local label propagation; neither tensor decomposition nor dual-view contrastive machinery adds detectable value.
- [ ] The released DHGCMDA code cannot reproduce the paper: four-type evaluator drops every v3.2 sample; three transductive leak channels; scalar-collapse of multi-type pairs; and the reported number exceeds the protocol's recall ceiling.
- [ ] Recommendation: benchmark hygiene for MDA prediction — label-derived similarities must be recomputed per fold; multi-type evaluation needs per-pair budgets stated explicitly; artifacts must be released.
- [ ] Limits: private author eval/training code never obtained; fold-MISIM reconstruction is approximate (corr≈0.80 to static — residual attributed to MISIM having been built on a different HMDD disease set); "best observed" is not a protocol ceiling (0.8642 stands).
- [ ] Ethical/reporting framing: measured tone — we report what is verifiable; we do not claim misconduct.

## 6. Reproducibility statement

- Code, artifact (SHA256-pinned), per-fold predictions, seeds, runtimes: repo `hoangtien07/DHGCMDA-fork` under `forensics/` + `paper/`; findings ledger `REPRO_LEDGER.md` (F1–F32); frozen leaderboard `FROZEN_LEADERBOARD.json`.
- Runtime footprint: vote suite ~2 min; golden SPLD replay ~33 min/fold CPU; learned baselines ~4.5 min/seed.

## 7. Conclusion

[To write — 4 sentences: recovered artifact → anchored audit → impossibility proof → minimal baseline; call for released artifacts and protocol-complete evaluation.]

---

## Figure/table inventory

| ID | File | Status |
|---|---|---|
| F1 lineage | `figures/F1_lineage.{png,pdf}` | done |
| F2 leaderboard | `figures/F2_leaderboard.{png,pdf}` | done |
| F3 leak channel | `figures/F3_leak_channel.{png,pdf}` | done |
| F4 recall ceiling | `figures/F4_recall_ceiling.{png,pdf}` | done |
| T1 leaderboard | `tables/T1_leaderboard.tex` | done |
| T2 bootstrap | `tables/T2_bootstrap.tex` | done |
| T3 ablation | `tables/T3_ablation.tex` | done |
| T4 robustness | `tables/T4_robustness.tex` | done |

## Wording discipline (binding for all prose)

- "we detect no statistically significant advantage over the similarity-voting baseline under this protocol" — never "adds nothing".
- "the publicly available/reconstructed DHGCMDA implementation" — never "the paper's own architecture" (private eval/training code unfound).
- "best observed performance" for ~0.59 — "ceiling" only for the proven 0.8642 bound.
