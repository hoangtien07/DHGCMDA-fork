# Baseline Audit — DHGCMDA (fork)

**Scope:** software & research-methodology review only. Read-only inspection of the
reproduction pipeline for HMDD v2.0 (the repo default and the paper's headline dataset).
No source code was modified. All claims below are traced to specific `file:line`
locations and, where relevant, to the paper text in `_pdf_text/`.

**Reference paper:** Sun Y. et al., *DHGCMDA: a dual-view heterogeneous graph contrastive
learning framework for miRNA–disease association* (BMC Bioinformatics, 2026), PDF at repo root.

**Audited entry point:** `python main_experiments_hetero1.py --device cpu` (default args in
[param.py](../../param.py)).

---

## 0. TL;DR — severity-ranked findings

| # | Finding | Severity | Type | Evidence |
|---|---|---|---|---|
| F1 | **Transductive leakage**: the *full* association matrix (incl. held-out test positives) is fed into hypergraph construction, the heterogeneous graph edges, GIP similarity, and the inter-view contrastive target — by default. | **High** | Leakage | [main_experiments_hetero1.py:970-998](../../main_experiments_hetero1.py#L970), [trainData.py:106](../../trainData.py#L106), [prepareData.py:208-209](../../prepareData.py#L208) |
| F2 | **Multi-label collapse**: 161 pairs (20.4% of rows) carry ≥2 association types; the matrix keeps only the *last* type ("last-wins"), silently discarding the rest. | **High** | Data / metric fidelity | [prepareData.py:74-76](../../prepareData.py#L74); verified on CSV (below) |
| F3 | **~10% of data is silently dropped** from CV: indices are split 10 ways, only 9 chunks are kept, then re-split into 5 folds. Reported CV is over ~90% of pairs, not the full set. | **Medium** | Split | [prepareData.py:285-304](../../prepareData.py#L285) |
| F4 | **Test-set threshold tuning**: binary F1/Accuracy/Recall/Precision pick the F1-optimal threshold *on the test fold itself*. AUC/AUPR are unaffected. | **Medium** | Metric optimism | [Calculate_Metrics.py:104-136](../../Calculate_Metrics.py#L104) |
| F5 | **Two paper protocols conflated into one run**: paper uses separate CVtriplet (AUC/AUPR/F1) and CVtype (Top-1) protocols; the code derives both from a single CV loop. | **Medium** | Protocol mismatch | paper [p24](../../_pdf_text/p24.txt); [main_experiments_hetero1.py:1618-1744](../../main_experiments_hetero1.py#L1618) |
| F6 | **"miRNA sequence view" (View 1) is GIP** (association-derived `M_GSM.txt`), not sequence. It is not an independent view and re-injects association signal. | **Medium** | Leakage / labelling | [prepareData.py:389-408](../../prepareData.py#L389) |
| F7 | **Balanced 1:1 negatives at eval** inflate AUPR relative to the true ~1:125 prevalence. (Paper's CVtriplet also balances, so this matches the paper but should be stated.) | **Low** | Metric interpretation | [main_experiments_hetero1.py:1481-1494](../../main_experiments_hetero1.py#L1481); paper [p24](../../_pdf_text/p24.txt) |
| F8 | **Independent test set defined but unused**; its construction overlaps the CV train pool and would leak if ever enabled. | **Low (latent)** | Dead code / latent leak | [prepareData.py:306-309](../../prepareData.py#L306) |

Positives worth recording: the seed→split plumbing is now correct (multi-seed truly
re-splits, [prepareData.py:277](../../prepareData.py#L277)); train/test *index* sets within
a fold are genuinely disjoint (no direct label leak); K=13 matches the paper; and the repo
already ships an opt-in `--leakage_free` flag that partially addresses F1.

---

## 1. Implemented model architecture

Main class: `HeterogenousGraphCLAMIR` ([hetero_model.py:598](../../hetero_model.py#L598)).
Pipeline for one forward pass ([hetero_model.py:758-949](../../hetero_model.py#L758)):

1. **Dual-view hypergraph encoder** (`CL_HGCN`, [hetero_model.py:218](../../hetero_model.py#L218)).
   For each node type (miRNA, disease) two hypergraph views are encoded by a single-layer
   `HGNN_conv` (`out = G·(x·W)+b`, [hetero_model.py:184](../../hetero_model.py#L184)) and tied
   by an intra-view NT-Xent contrastive loss (τ=0.5). Input to each view is
   `concat(association_matrix, similarity_features)`.
2. **View fusion** (`HGCN_Attention_Mechanism`, [hetero_model.py:282](../../hetero_model.py#L282)):
   despite the name this is a **fixed weighted sum `0.6·view1 + 0.4·view2`**, not learned
   attention (the docstring itself flags this).
3. **Inter-view (cross-modal) contrastive loss** (`InterViewContrastiveLoss`,
   [hetero_model.py:33](../../hetero_model.py#L33)): InfoNCE + margin ranking, pulling associated
   (miRNA, disease) embeddings together. Weight `inter_view_weight=0.3`.
4. **Similarity reconstruction** decoders (`SimpleHypergraphDecoder`,
   [hetero_model.py:315](../../hetero_model.py#L315)), self-supervised MSE against functional /
   semantic similarity. *Note:* only active in `.train()`; at eval the decoder is replaced by an
   identity matrix ([hetero_model.py:882-885](../../hetero_model.py#L882)).
5. **HGT layers** (`EnhancedHGTLayer` wrapping PyG `HGTConv`, [hetero_model.py:351](../../hetero_model.py#L351)):
   `nlayer=2`, `n_head=4`, over 4 edge types (assoc ↔, similar-mi, similar-dis).
6. **Type predictor** (`SimplifiedTypePredictor`, [hetero_model.py:449](../../hetero_model.py#L449)):
   two-head bilinear scorer → existence sigmoid + per-type score, output shape
   `[mi, dis, 1+num_types]`. Default `predictor_mode='diag'` (BilinearDiag, rank-d);
   `full_bilinear` (full `d×d` per type) is available and, per project notes, is the
   recommended v2.0 default.

**Loss** (`SimplifiedMultiTypeAssociationLoss`, [main_experiments_hetero1.py:75](../../main_experiments_hetero1.py#L75)):
`total = exist_weight·focal_existence + type_weight·(weighted CE + label-smoothing)`, added to
`1·(mi_cl+dis_cl) + 1·(mi_recon+dis_recon) + 1e-4·L2` at
[main_experiments_hetero1.py:1104](../../main_experiments_hetero1.py#L1104). Default
`exist_weight=0.3`. Class weights use the Effective-Number formula (β=0.99999).

**Architecture vs paper — matches:** dual-view hypergraph CL, HGT (`n_head=4`),
2 HGT layers, dynamic graph update every 5 epochs
([main_experiments_hetero1.py:1056](../../main_experiments_hetero1.py#L1056)), λ₁=λ₃=1.0, K=13.

**Architecture vs paper — divergences to note (not necessarily wrong):**
- "Attention-guided fusion" is a fixed 0.6/0.4 sum, not attention (F-arch-1).
- The paper's loss (Eq. 32) has no standalone existence term; the code adds a
  `0.3·focal_existence` head (project notes: `exist_weight=0.1` closes most of the gap).
- Reconstruction decoder is disabled at inference — reconstruction contributes to training
  regularization only.

---

## 2. Dataset & preprocessing

Loader: `prepare_data_optimized` ([prepareData.py:328](../../prepareData.py#L328)). For
`v2.0_495m383D`:

| Tensor | Source file | Role |
|---|---|---|
| `md_p` (association) | `multi_all_mirna_disease_pairs_without_negative.csv` | labels + graph edges |
| `dis_sem` | `D_SSM1.txt` | disease view 2 (semantic) |
| `d_gs` | `D_SSM2.txt` | disease view 1 (gene) |
| `mi_fun` | `M_FSM.txt` | miRNA view 2 (functional) |
| `m_ss` | `M_GSM.txt` | miRNA view 1 — **GIP, not sequence** (F6) |
| `ID`,`IM` | computed | integrated sim (Gauss GIP fused with `dd`/`mm`) |

**Association matrix build** ([prepareData.py:49-97](../../prepareData.py#L49)):
`association_matrix[i,j] = atype` in a `for` loop — a repeated `(i,j)` **overwrites** with the
later type. Verified on the CSV:

```
total rows: 1679 | unique (mi,dis) pairs: 1498
pairs appearing >1 time (multi-label): 161  → 342 rows (20.4%) collapsed to single label
type dist (rows):      {1:443, 2:199, 3:356, 4:681}
type dist (collapsed): {1:367, 2:157, 3:293, 4:681}   ← this is what the model trains on
mirna idx range: 1..493   disease idx range: 1..383
```

So **F2** is confirmed: one in five multi-type associations loses ≥1 of its types, and the
retained type is order-dependent (whatever appears last in the CSV). This directly affects both
training targets and the Top-1 metric (a test pair with types {2,3} is scored correct only if
the model predicts the single retained type, not either true type).

**GIP similarity leakage (F6):** `M_GSM.txt` is a Gaussian-interaction-profile kernel derived
from the association profile; the code comment at
[prepareData.py:390-393](../../prepareData.py#L390) says so explicitly and adds a
`--mirna_seq_sim_path` flag to substitute real sequence features. As shipped, miRNA "View 1"
therefore encodes association information rather than an independent modality — and the integrated
`IM`/`ID` are also computed from the full association matrix via `Gauss_M/Gauss_D`
([prepareData.py:208-209](../../prepareData.py#L208)).

---

## 3. Train / validation / test split

Function `preprocess_indices` ([prepareData.py:232-325](../../prepareData.py#L232)):

1. Collect all nonzero (positive) and zero (negative) cell indices; shuffle with `seed`
   (seed now correctly threaded — [prepareData.py:277](../../prepareData.py#L277)).
2. `split(size/10)` → **10 chunks** (plus a small remainder chunk).
3. `cross_* = cat(splits[0..8])` — keeps **only 9 chunks (~90%)**;
   the remaining ~10% is used only for the (unused) "independent" set.
4. `cross_*` is re-split into `validation=5` folds → `cv_data`.

Consequences:
- **F3 (data dropped):** roughly 10% of positives and negatives never enter the CV. Each fold's
  test set is `(1/5)·90% ≈ 18%` of positives and train is `~72%` — not a standard 80/20 5-fold
  over the whole dataset. Reported numbers are on ~90% of the data.
- Train/test index sets **within** a fold are disjoint (`test = split[i]`,
  `train = cat(others)`), so there is **no direct label leakage** in the index assignment. The
  leakage in this pipeline is structural (Section 4), not index-level.
- There is **no separate validation split**: the model trains for a fixed `epoch=650` and the
  final-epoch model is evaluated on the test fold. No early stopping / model selection on a
  held-out validation set (so no selection leakage, but also no checkpoint selection).

**F8 (latent):** `independent_test` ([prepareData.py:306-309](../../prepareData.py#L306)) sets
`train = cross_index` (chunks 0..8) and `test = splits[-2]`. With the remainder chunk present,
`splits[-2]` sits *outside* 0..8, so today there is no overlap — but this is incidental to the
chunk arithmetic. The independent set is never consumed by `main_optimized`, so it is currently
dead code; if re-enabled it should be re-derived to guarantee disjointness.

---

## 4. Data leakage, duplicate edges, negative sampling

### 4.1 Transductive leakage (F1 — High)

The `Dataset` returns the **shared full** association matrix `md_p` as element 4 of every fold's
tuple ([trainData.py:106](../../trainData.py#L106)); it is identical across folds and contains
**all** positives, including the current fold's test positives. In `train_epoch_optimized` this
full matrix is used to:

- build the miRNA/disease hypergraph views:
  `concat(association_matrix, features)` → `constructHW_knn`
  ([main_experiments_hetero1.py:970-998](../../main_experiments_hetero1.py#L970)). The KNN
  hyperedges are formed from Euclidean distance over rows that **include the raw association
  columns** (`Eu_dis(X)`, [hypergraph_construct_KNN.py:135](../../hypergraph_construct_KNN.py#L135)),
  so held-out positives directly determine which nodes share hyperedges;
- build the heterogeneous-graph `associates` edges
  ([main_experiments_hetero1.py:576-587](../../main_experiments_hetero1.py#L576)) — test-positive
  edges are message-passing edges during both train and test;
- form the inter-view contrastive positive mask
  ([hetero_model.py:853-856](../../hetero_model.py#L853)) — the training objective pulls
  test-positive (miRNA, disease) pairs together;
- the GIP-derived `m_ss`/`IM`/`ID` (Section 2) are precomputed from the full matrix.

This is the classic transductive-leakage pattern of the miRNA–disease-association model family:
the graph "sees" the edges being predicted. It inflates held-out performance and is the most
important caveat for any reproduction claim.

**Mitigation already in-repo:** `--leakage_free`
([main_experiments_hetero1.py:932-941](../../main_experiments_hetero1.py#L932)) zeroes test
positives out of `association_matrix` before hypergraph/graph construction. **However, it is
off by default, and even when on it does not mask the GIP-derived `m_ss`/`IM`/`ID`** (those are
built once from the full matrix in `prepareData`). So `--leakage_free` is a partial fix.
Default runs (and, by the numbers in `CLAUDE.md`, the headline results) are leaky.

### 4.2 Duplicate / multi-label edges (F2)

Covered in Section 2: 161 duplicate `(mi,dis)` pairs are silently overwritten. No de-duplication
warning is emitted. Not a crash, but a real information loss and a metric-fidelity problem.

### 4.3 Negative sampling

- **Training:** all zero cells are candidates; the loader samples `10× |positives|` negatives per
  fold ([main_experiments_hetero1.py:912-918](../../main_experiments_hetero1.py#L912)). Negatives
  are unobserved pairs treated as true negatives — a PU-learning assumption the **paper itself
  flags as a limitation** ([p36](../../_pdf_text/p36.txt)). Not a bug, but a known bias.
- **Evaluation:** negatives are subsampled to **1:1** with positives, averaged over 5 reseeds
  ([main_experiments_hetero1.py:1481-1494](../../main_experiments_hetero1.py#L1481)). This is a
  balanced test set → AUPR is optimistic vs the native ~0.8% prevalence (**F7**). The paper's
  CVtriplet also uses an "equal number of randomly sampled unknown" test set
  ([p24](../../_pdf_text/p24.txt)), so this *matches the paper*, but the AUPR should be read as
  "balanced AUPR", not prevalence-adjusted.
- **Sampling validity:** sampled negatives are drawn from cells with value 0 in the full matrix,
  so they are never labelled positives; no invalid (positive-as-negative) sampling was found. The
  only caveat is the PU assumption above.

---

## 5. Evaluation metrics vs paper

Computed in `Calculate_Metrics.py` and aggregated in
`evaluate_optimized_with_comprehensive_metrics`
([main_experiments_hetero1.py:1425](../../main_experiments_hetero1.py#L1425)).

| Metric group | Code | Paper | Match? |
|---|---|---|---|
| AUC | ROC-AUC, threshold-free | CVtriplet AUC | ✅ |
| AUPR | PR-AUC on 1:1 balanced test | CVtriplet AUPR (balanced) | ✅ (balanced, F7) |
| Binary F1 / Acc / P / R | **threshold chosen to maximize F1 on the test fold** ([Calculate_Metrics.py:104-136](../../Calculate_Metrics.py#L104)) | CVtriplet F1 | ⚠ optimistic (F4) |
| Top-1 Precision | micro: `correct/total` ([Calculate_Metrics.py:354](../../Calculate_Metrics.py#L354)) | CVtype Top-1 P | ≈ |
| Top-1 Recall | **macro** over types ([Calculate_Metrics.py:357-363](../../Calculate_Metrics.py#L357)) | CVtype Top-1 R | ≈ (micro/macro asymmetry) |
| Top-1 F1 | harmonic mean of the above | CVtype Top-1 F1 | ✅ (harmonic-mean form matches paper) |

Notes:
- **F5 — protocol conflation.** The paper runs *two* independent 5-fold protocols: CVtriplet
  (novel-association discovery; AUC/AUPR/F1) and CVtype (type discrimination on known pairs;
  Top-1) — [p23-24](../../_pdf_text/p24.txt). The code runs a *single* CV over positives+negatives
  and extracts both metric families from it. Numerically the headline metrics are comparable, but
  they are not produced under the paper's two distinct data partitions, so a strict "same
  protocol" claim is not supported.
- **Top-1 P/R asymmetry.** Precision is micro-averaged while recall is macro-averaged
  ([Calculate_Metrics.py:353-367](../../Calculate_Metrics.py#L353)). This is an unusual pairing;
  the resulting F1 differs from a pure micro- or pure macro-F1. The paper text does not specify the
  averaging, so this cannot be asserted identical.
- **Known metric bug (out of v2.0 scope, documented).** `compute_top1_metrics` only accepts
  prediction vectors of length 4 or 5 ([Calculate_Metrics.py:321-331](../../Calculate_Metrics.py#L321));
  for 5-type v3.2 (length 6) every sample is skipped → Top-1 F1 = 0. This does not affect v2.0
  (length 5) but invalidates any v3.2 Top-1 number computed through this path.

---

## 6. Repository vs paper — consolidated

| Aspect | Paper | Repo (default) | Verdict |
|---|---|---|---|
| Dataset | HMDD v2.0, 495×383 | same | ✅ |
| Hypergraph K | 13 | 13 | ✅ |
| HGT layers / heads | 2 / 4 | 2 / 4 | ✅ |
| λ₁, λ₃ | 1.0, 1.0 | 1.0, 1.0 | ✅ |
| Dynamic update | every 5 epochs | every 5 epochs | ✅ |
| View fusion | "attention-guided" | fixed 0.6/0.4 sum | ⚠ divergent |
| miRNA View 1 | (implied independent modality) | GIP (association-derived) | ⚠ leakage (F6) |
| Existence loss term | none (Eq. 32) | `0.3·focal` added | ⚠ divergent |
| CV protocol | 2 separate (CVtriplet, CVtype) | 1 combined | ⚠ (F5) |
| Data coverage | (implied full) | ~90% (F3) | ⚠ |
| Multi-label pairs | (types per triplet) | collapsed last-wins (F2) | ⚠ |
| Graph edges at eval | — | full matrix incl. test (F1) | ⚠ leakage |

---

## 7. Recommendations (analysis only — no code changes made or proposed to source)

For a defensible reproduction/extension, the following would need to be addressed or at minimum
disclosed as caveats in any write-up. Listed by priority; all are methodology notes, not edits:

1. **Report leakage-controlled numbers.** Re-run with `--leakage_free` *and* additionally
   confirm the GIP-derived `m_ss`/`IM`/`ID` are recomputed from train-only edges; otherwise state
   that headline numbers are transductive.
2. **Disclose multi-label collapse (F2)** and, ideally, evaluate Top-1 as "predicted type ∈ known
   types" to match the paper's intent for multi-type pairs.
3. **Use all data in CV (F3)** or explicitly report that CV covers ~90% of pairs.
4. **Separate the two CV protocols (F5)** to match the paper, and report threshold-free metrics
   (AUC/AUPR) as primary since F1/Acc are threshold-tuned on test (F4).
5. **State the balanced-negative caveat (F7)** wherever AUPR is quoted.

These map cleanly onto flags/behaviours the repo already exposes (`--leakage_free`,
`--mirna_seq_sim_path`, `--predictor_mode`, `--exist_weight`), so they are largely
configuration/reporting decisions rather than new code.

---

*Prepared as a read-only baseline audit. No wet-lab content; in-silico ML review only. No
results were re-run for this document — findings are from static code/data inspection plus the
CSV statistics shown in Section 2, which were computed read-only.*
