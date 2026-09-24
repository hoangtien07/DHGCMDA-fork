# FORENSICS_MODEL_PATH — execution trace of released DHGCMDA code + leakage analysis

Date: 2026-09-23. Track R0 (released-code reproduction).

## 1. Execution path

`main_experiments_hetero1.py` → `prepare_data()` (prepareData.py) → `Dataset` (trainData.py)
→ per-fold `train_model(...)` → `HeterogenousGraphCLAMIR` (hetero_model.py, lineage: HGCLAMIR)
→ `SimplifiedMultiTypeAssociationLoss` → `Calculate_Metrics.compute_top1_metrics`.

## 2. Stage-by-stage findings

### Loader — scalar collapse (VERIFIED)
`read_association_csv` (prepareData.py:74-76): `association_matrix[i,j] = atype` — multi-label
pairs collapse to a single scalar (last row wins). For v3.2 (~23% multi-label pairs) the canonical
Y[m,d,type] tensor is destroyed before it ever reaches the model. `type_indices` in
`preprocess_indices` only tracks type1..4 → the 5th type (Tissue) is invisible to split bookkeeping.

### Split (VERIFIED)
`preprocess_indices`: flatten `nonzero` PAIRS of the collapsed scalar matrix + `zero` pairs,
`np.random.shuffle` (seed=opt.seed), 10 splits → 9 for k-fold, 1 for "independent". CV is over
pairs, not triplets.

### Leakage chain (VERIFIED — three independent channels)
1. **GIP similarity from the full label matrix**: `preprocess_similarity_matrices` computes
   `DGSM`/`MGSM` (Gaussian interaction profile) over the FULL `association_matrix` — including
   held-out test entries — BEFORE `preprocess_indices` splits anything (prepareData.py:208-218,
   called at :405-409). Result fused into `ID`/`IM` ("integrated" sims) and also fed as fallback
   hypergraph features.
2. **Test labels in node features + KNN hypergraphs**: in the training loop,
   `association_matrix = train_data[4]` is the full matrix (trainData.py:106 returns `md_p`
   unmasked for every fold). It is concatenated into all four view feature blocks
   (main_experiments_hetero1.py:902-923) and KNN-hypergraphs `G_mi_view1/2`, `G_dis_view1/2`
   are built on top (K=13). Held-out pair membership is directly visible to the KNN geometry.
3. **Test edges in the message-passing graph**: `create_hetero_data_optimized` builds
   `('miRNA','associates','disease')` edge_index from `torch.nonzero(association_matrix > 0)`
   (main_experiments_hetero1.py:513-528) — i.e., ALL edges including the test fold. The HGT
   aggregates over test associations every epoch; the "dynamic update" only swaps sim edges
   (`mi_sim_recon`/`dis_sim_recon`), the association edges stay full.

Only the loss and the evaluator consult the fold split. The pipeline is **fully transductive**:
every "held-out" pair is simultaneously a graph edge, a node-feature bit, and a GIP contributor.

### Predictor vs paper (VERIFIED, known)
`SimplifiedTypePredictor` = diagonal bilinear (rank-d, shared emb), paper says "bilinear".
`num_association_types` hardcoded 4 in places; `compute_top1_metrics` accepts only len-5
predictions and type ids 1..4 → **all v3.2 predictions discarded → Top-1 F1 = 0.0 regardless
of model quality** (proven via synthetic perfect predictor, Plan K).

## 3. Consequences

- Binary metrics reproduced at ~paper level (AUC 0.92 vs 0.9181) are **inflated by transductive
  leakage**, not evidence of matched generalization.
- Any R2 (leakage-free) reconstruction must: build GIP/features/KNN/edges from train edges only
  (per fold), preserve Y[m,d,type], and fix the evaluator to K types. This is a material rewrite —
  released code cannot produce a leakage-free number.
- The paper's own v3.2 run necessarily used different code (evaluator can't score 5 types) and a
  different dataset (see FORENSICS_V32_DATASET.md) → released repo ≠ paper pipeline on two axes.
