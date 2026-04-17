# Architecture Diagram — DHGCMDA

Sơ đồ ASCII để hình dung kiến trúc overall + map từng khối về code cụ thể.

---

## 🗺️ Overall architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            RAW DATA (v2.0_495m383D/)                          │
│                                                                               │
│   M_GSM.txt      M_FSM.txt      D_SSM2.txt     D_SSM1.txt    pairs.csv       │
│   (495×495)      (495×495)      (383×383)      (383×383)     (1679 rows)     │
│   mi Seq sim     mi Func sim    dis Gene sim   dis Sem sim   associations    │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
                            prepareData.prepare_data_optimized()
                            + Gauss_M/Gauss_D (integrate missing values)
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          TENSORS (trong data_set dict)                        │
│                                                                               │
│  m_ss [495×495]   mi_fun [495×495]   d_gs [383×383]   dis_sem [383×383]     │
│  (miRNA V1)       (miRNA V2)         (Disease V1)     (Disease V2)           │
│                                                                               │
│  md_p [495×383]   IM [495×495]   ID [383×383]   md (5-fold split)           │
│  assoc matrix     integrated mi  integrated dis                              │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
                          trainData.OptimizedDataset
                          → __getitem__(fold_idx) → 12-tuple
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│              TRAIN LOOP — main_experiments_hetero1.train_epoch_optimized      │
│                                                                               │
│   ┌─── Stage A: Build 4 hypergraphs (KNN, K=13) ─────────────────────┐       │
│   │                                                                    │       │
│   │   cat(md_p, m_ss)      →  G_mi_view1   [495×495]                 │       │
│   │   cat(md_p, mi_fun)    →  G_mi_view2   [495×495]                 │       │
│   │   cat(md_p.T, d_gs)    →  G_dis_view1  [383×383]                 │       │
│   │   cat(md_p.T, dis_sem) →  G_dis_view2  [383×383]                 │       │
│   │                                                                    │       │
│   └──────────────────────────────────────────────────────────────────┘       │
│                                        │                                      │
│                                        ▼                                      │
│   ┌─── Stage B: Build heterogeneous graph ────────────────────────────┐      │
│   │                                                                     │      │
│   │   edges:                                                            │      │
│   │     (miRNA)─associates─(disease)  ← from md_p                      │      │
│   │     (disease)─associates─(miRNA)  ← reverse                        │      │
│   │     (miRNA)─similar─(miRNA)       ← from IM > 0.5                  │      │
│   │     (disease)─similar─(disease)   ← from ID > 0.5                  │      │
│   │                                                                     │      │
│   └──────────────────────────────────────────────────────────────────┘        │
│                                        │                                      │
│                                        ▼                                      │
│   ┌─── Stage C: Forward qua HeterogenousGraphCLAMIR ──────────────┐         │
│   │   (chi tiết bên dưới)                                           │         │
│   └────────────────────────────────────────────────────────────────┘         │
│                                        │                                      │
│                                        ▼                                      │
│   ┌─── Stage D: Compute loss ─────────────────────────────────────┐          │
│   │                                                                 │          │
│   │   recover_loss  = 0.3·focal(exist) + 0.7·wCE(type)             │          │
│   │   mi_cl_loss    = intra_mi + 0.3·inter_cl                      │          │
│   │   dis_cl_loss   = intra_dis + 0.3·inter_cl                     │          │
│   │   recon_loss    = MSE(sim_recon, sim_gt)                       │          │
│   │   reg_loss      = L2(model.parameters)                         │          │
│   │                                                                 │          │
│   │   total = recover + mi_cl + dis_cl + 0.15·recon + 1e-4·reg     │          │
│   │                                                                 │          │
│   └──────────────────────────────────────────────────────────────┘            │
│                                        │                                      │
│                                        ▼                                      │
│                          backward + optimizer.step()                          │
│                          (grad clip max_norm=1.0)                             │
│                                                                               │
│   Loop epoch × 5-fold CV                                                      │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          test_optimized → AUC, AUPR, F1, Top-1 metrics
```

---

## 🧠 HeterogenousGraphCLAMIR.forward() — Stage C detail

```
 INPUT:
   concat_mi_tensor   [495, 878]   ← cat(md_p, m_ss)
   concat_dis_tensor  [383, 878]   ← cat(md_p.T, d_gs)
   G_mi_Kn, G_mi_Km   [495, 495]   ← 2 miRNA hypergraphs
   G_dis_Kn, G_dis_Km [383, 383]   ← 2 disease hypergraphs
   hetero_data                       ← HeteroData object
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                                │
        ▼                                                ▼
 ╔══════════════════╗                           ╔══════════════════╗
 ║   CL_HGCN_mi     ║                           ║   CL_HGCN_dis    ║
 ║ (dual-view HGCN) ║                           ║ (dual-view HGCN) ║
 ╚════════╦═════════╝                           ╚════════╦═════════╝
          │                                              │
          │  mi_feat1, mi_feat2, mi_intra_loss           │  dis_feat1, dis_feat2, dis_intra_loss
          │  [495, 256]                                   │  [383, 256]
          ▼                                              ▼
 ╔══════════════════╗                           ╔══════════════════╗
 ║   AM_mi (0.6/0.4)║                           ║  AM_dis (0.6/0.4)║
 ╚════════╦═════════╝                           ╚════════╦═════════╝
          │                                              │
          │  mi_fused [495, 256]                          │  dis_fused [383, 256]
          ├──────────────────────┬───────────────────────┤
          │                      │                       │
          ▼                      ▼                       ▼
 ╔═══════════════╗   ╔══════════════════╗      ╔═══════════════╗
 ║ miRNA_decoder ║   ║ InterViewContCL  ║      ║ disease_decoder║
 ║ → mi_sim_recon║   ║ (mi vs dis)       ║      ║ → dis_sim_recon║
 ║ [495, 495]    ║   ║ → inter_view_loss ║      ║ [383, 383]    ║
 ╚═══════════════╝   ╚══════════════════╝      ╚═══════════════╝
          │                                              │
          │  (for recon loss)                            │
          │                                              │
          └──────────────────┐         ┌────────────────┘
                             ▼         ▼
                  ┌──────────────────────────┐
                  │   node_transformers       │
                  │   Linear(256 → 256)       │
                  └────────────┬──────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │   EnhancedHGTLayer × 2    │
                  │ (heterogeneous graph      │
                  │  transformer, 8 heads)    │
                  └────────────┬──────────────┘
                               │
                               │  mi_emb [495, 256]
                               │  dis_emb [383, 256]
                               ▼
                  ┌──────────────────────────┐
                  │ SimplifiedTypePredictor   │
                  │ (BilinearDiag scoring)    │
                  └────────────┬──────────────┘
                               │
                               ▼
                  score [495, 383, 5]
                  [exist_prob, p_t1, p_t2, p_t3, p_t4]

 OUTPUT:
   return score, mi_cl_loss, dis_cl_loss, mi_sim_recon, dis_sim_recon
   (mi_cl_loss = mi_intra + 0.3 * inter_view)
```

---

## 🎯 CL_HGCN internal detail

```
  INPUT:
    x1 (concat_mi, View 1)  [495, 878]
    adj1 (G_mi_Kn)           [495, 495]
    x2 (concat_mi, View 2)  [495, 878]  ← same content in this project
    adj2 (G_mi_Km)           [495, 495]
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
  ╔═════════════╗          ╔═════════════╗
  ║   hgcn1     ║          ║   hgcn2     ║
  ║ HGCN(878,256)║         ║ HGCN(878,256)║
  ╚══════╦══════╝          ╚══════╦══════╝
         │                         │
         │ z1 [495, 256]           │ z2 [495, 256]
         │                         │
         ├──────────┬──────────────┤
         │          │              │
         ▼          │              ▼
     projection    │         projection
     h1 [495, 256] │         h2 [495, 256]
                   │
                   ▼
              InfoNCE/SimCLR
              contrastive loss
              (tau=0.5)
                   │
                   ▼
              intra_loss (scalar)

  OUTPUT: z1, z2, intra_loss
```

---

## 🎯 InterViewContrastiveLoss detail

```
  INPUT:
    mi_fused  [495, 256]
    dis_fused [383, 256]
    assoc_matrix_binary [495, 383]
                │
                ▼
         L2 normalize both
                │
                ▼
         M_norm @ D_norm.T / τ
         → S [495, 383]
                │
      ┌─────────┴──────────┐
      ▼                    ▼
 positive_mask        negative_mask
 (where assoc > 0)    (where assoc = 0)
                │
                ▼
       For each positive (i, j):
         pos_sim     = exp(S[i,j])
         neg_sum[i]  = Σ exp(S[i,k]) over negatives
         loss = -log(pos_sim / (pos_sim + neg_sum[i]))
                │
                ▼
        infonce_loss = mean(losses)
                │
                + 0.1 · margin_loss
                │
                ▼
         inter_view_loss
```

---

## 🗂️ File → function → purpose map

### Core flow

| File | Function/Class | Line | Purpose |
|------|---------------|------|---------|
| [main_experiments_hetero1.py](../main_experiments_hetero1.py) | `main_optimized()` | 1322 | Entry point, 5-fold CV |
| [main_experiments_hetero1.py](../main_experiments_hetero1.py) | `train_epoch_optimized()` | 647 | Train loop cho 1 fold |
| [main_experiments_hetero1.py](../main_experiments_hetero1.py) | `test_optimized()` | 913 | Evaluate 1 fold |
| [main_experiments_hetero1.py](../main_experiments_hetero1.py) | `SimplifiedMultiTypeAssociationLoss` | 56 | Existence + type loss |
| [hetero_model.py](../hetero_model.py) | `HeterogenousGraphCLAMIR` | 518 | Main model |
| [hetero_model.py](../hetero_model.py) | `HeterogenousGraphCLAMIR.forward` | 622 | Forward pass 5 stages |

### Layers

| File | Class | Line | Role |
|------|-------|------|------|
| [hetero_model.py](../hetero_model.py) | `HGNN_conv` | 147 | 1 hypergraph conv layer |
| [hetero_model.py](../hetero_model.py) | `HGCN` | 181 | HGNN_conv + LeakyReLU |
| [hetero_model.py](../hetero_model.py) | `CL_HGCN` | 201 | 2 HGCN + contrastive head |
| [hetero_model.py](../hetero_model.py) | `HGCN_Attention_Mechanism` | 257 | Weighted sum (0.6/0.4) |
| [hetero_model.py](../hetero_model.py) | `InterViewContrastiveLoss` | 33 | Cross-modal CL |
| [hetero_model.py](../hetero_model.py) | `SimpleHypergraphDecoder` | 283 | Sim reconstruction |
| [hetero_model.py](../hetero_model.py) | `EnhancedHGTLayer` | 319 | HGTConv wrapper |
| [hetero_model.py](../hetero_model.py) | `SimplifiedTypePredictor` | 417 | BilinearDiag classifier |

### Data

| File | Function | Line | Purpose |
|------|----------|------|---------|
| [prepareData.py](../prepareData.py) | `prepare_data_optimized` | 322 | Load toàn bộ data |
| [prepareData.py](../prepareData.py) | `read_txt` / `read_association_csv` | 23/49 | File readers |
| [prepareData.py](../prepareData.py) | `Gauss_M_optimized` / `Gauss_D_optimized` | 127/138 | Gaussian kernel sim |
| [trainData.py](../trainData.py) | `OptimizedDataset` | 9 | Dataset class với 5-fold CV |
| [create_hetero_data.py](../create_hetero_data.py) | `create_hetero_data_optimized` | 5 | Build HeteroData |

### Hypergraph

| File | Function | Line | Purpose |
|------|----------|------|---------|
| [hypergraph_construct_KNN.py](../hypergraph_construct_KNN.py) | `Eu_dis` | 6 | Euclidean distance |
| [hypergraph_construct_KNN.py](../hypergraph_construct_KNN.py) | `construct_H_with_KNN` | 125 | KNN incidence matrix |
| [hypergraph_construct_KNN.py](../hypergraph_construct_KNN.py) | `_generate_G_from_H` | 73 | Laplacian từ H |
| [hypergraph_construct_kmeans.py](../hypergraph_construct_kmeans.py) | `construct_H_with_Kmeans` | 50 | K-means incidence |
| [ConstructHW.py](../ConstructHW.py) | `constructHW_knn` / `constructHW_kmean` | 7/18 | Wrapper |

### Evaluation

| File | Function | Line | Purpose |
|------|----------|------|---------|
| [Calculate_Metrics.py](../Calculate_Metrics.py) | `Metric_fun` | - | AUC, AUPR, F1, Top-1 |

### Config

| File | Function | Line | Purpose |
|------|----------|------|---------|
| [param.py](../param.py) | `parameter_parser` | 4 | CLI args |
| [param.py](../param.py) | `validate_and_adjust_parameters` | 210 | Validation + auto class_weights |

### Helpers

| File | Function | Purpose |
|------|----------|---------|
| [utils.py](../utils.py) | `Myloss`, `get_L2reg` | MSE loss + L2 regularization |
| [wieghts.py](../wieghts.py) | `DataDistributionAnalyzer` | Phân tích class distribution |
| [check_v2.0_495m383D.py](../check_v2.0_495m383D.py) | - | Verify data loading |

---

## 🧬 Biology mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                    miRNA (495 nodes)                             │
│                                                                   │
│   View 1 (Sequence): nucleotide similarity                       │
│   View 2 (Function): pathway/target co-occurrence                │
│                                                                   │
│   → Dual embedding z1, z2 ∈ ℝ^256                                │
│   → Fused: 0.6·z1 + 0.4·z2                                       │
└─────────────────────────────────────────────────────────────────┘

                    Association types:
                    ┌────────────────────────────┐
                    │ 1. Circulation — 367       │ blood biomarkers
                    │ 2. Epigenetics — 157       │ gene regulation
                    │ 3. Target      — 293       │ direct mRNA binding
                    │ 4. Genetics    — 681       │ genetic variants
                    └────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Disease (383 nodes)                           │
│                                                                   │
│   View 1 (Gene): shared disease-gene profile                     │
│   View 2 (Semantic): MeSH / ontology distance                    │
│                                                                   │
│   → Dual embedding z1, z2 ∈ ℝ^256                                │
│   → Fused: 0.6·z1 + 0.4·z2                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key hyperparameters cheat

```
            ┌─────────────┐
 input ─────┤  CL_HGCN    ├─── tau=0.5, alpha=0.5
            └─────────────┘

            ┌─────────────┐
     ──────┤  AM (fusion)├─── w1=0.6, w2=0.4 (hard-coded)
            └─────────────┘

            ┌─────────────┐
     ──────┤ InterViewCL ├─── tau=0.5, margin=0.5
            └─────────────┘          weight in total loss=0.3

            ┌─────────────┐
     ──────┤  HGTConv    ├─── nlayer=2, heads=8
            └─────────────┘          dropout=0.3

            ┌─────────────┐
     ──────┤SimpleDecoder├─── output_dim = N (mi or dis)
            └─────────────┘

            ┌─────────────┐
     ──────┤ TypePred    ├─── node_dim=256, hidden=128
            └─────────────┘          num_types=4, T=2.0 (learnable)

 Loss weights:
   existence  : type      = 0.3 : 0.7
   intra_CL   : inter_CL  = 1.0 : 0.3
   recon_mi   : recon_dis = 0.15 : 0.15 (of total)
   reg (L2)              = 0.0001
```
