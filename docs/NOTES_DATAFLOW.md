# Data Flow — DHGCMDA

Tài liệu này mô tả **data flow từ file .txt → prediction**, kèm tensor shape ở mỗi bước. Dùng làm bản đồ khi đọc code.

---

## 0. Tổng quan 1 câu

```
4 file similarity (.txt) + 1 file association (.csv)
        ↓ prepareData.py
Tensor 4 views + association matrix (495×383)
        ↓ trainData.OptimizedDataset (chia 5 fold)
        ↓ main_experiments_hetero1.train_epoch_optimized()
Hypergraph (KNN) × 2 views × 2 loại node (miRNA, disease)
        ↓ hetero_model.HeterogenousGraphCLAMIR.forward()
Score [495, 383, 5] — (existence, type_1, type_2, type_3, type_4)
        ↓ SimplifiedMultiTypeAssociationLoss
Loss scalar → backward → update weights
```

---

## 1. Raw data — `v2.0_495m383D/`

| File | Format | Shape | Ý nghĩa |
|------|--------|-------|---------|
| `M_GSM.txt` | space-separated floats | 495×495 | miRNA **sequence** similarity (View 1) |
| `M_FSM.txt` | space-separated floats | 495×495 | miRNA **functional** similarity (View 2) |
| `D_SSM2.txt` | space-separated floats | 383×383 | Disease **gene** similarity (View 1) |
| `D_SSM1.txt` | space-separated floats | 383×383 | Disease **semantic** similarity (View 2) |
| `multi_all_mirna_disease_pairs_without_negative.csv` | `[miRNA_id, disease_id, type]` | 1679 rows | Các cặp association + loại (1-4) |

**Đọc file**: `read_txt()` và `read_association_csv()` trong [prepareData.py:23](prepareData.py#L23) và [prepareData.py:49](prepareData.py#L49).

**Lưu ý** — file CSV dùng **1-based index**; code convert sang 0-based bằng `- 1` ở [prepareData.py:65-66](prepareData.py#L65).

---

## 2. `prepareData.prepare_data_optimized()` — build tensor

Output là **dict** `data_set` chứa:

| Key | Shape | Ý nghĩa |
|-----|-------|---------|
| `ID` | `[383, 383]` | Disease similarity *integrated* (SSM + Gaussian kernel) |
| `IM` | `[495, 495]` | miRNA similarity *integrated* (FSM + Gaussian kernel) |
| `md_p` | `[495, 383]` | Association matrix (giá trị 0, 1, 2, 3, 4) |
| `md_true` | `[495, 383]` | Copy của `md_p` (ground truth cho eval) |
| `d_gs` | `[383, 383]` | Disease View 1 (từ D_SSM2.txt) |
| `m_ss` | `[495, 495]` | miRNA View 1 (từ M_GSM.txt) |
| `dis_sem` | `[383, 383]` | Disease View 2 (từ D_SSM1.txt) |
| `mi_fun` | `[495, 495]` | miRNA View 2 (từ M_FSM.txt) |
| `md` | list 5 fold dict | mỗi fold có `train`/`test` indices |
| `independent` | list 1 phần tử | independent test set |

**Quan trọng** — `ID` và `IM` KHÔNG phải là raw D_SSM1/M_FSM. Chúng là **integrated similarity** được tính bằng:
```
IM = M_FSM ⊕ Gauss_M(md_p)     # khi ô nào trong FSM = 0 thì thay bằng Gaussian
```
Xem [prepareData.py:127](prepareData.py#L127) `Gauss_M_optimized()` và `Gauss_D_optimized()`.

---

## 3. `trainData.OptimizedDataset.__getitem__(fold_idx)` — 12-tuple

Khi main script gọi `data_set[fold_idx]`, nhận về tuple 12 phần tử ([trainData.py:101](trainData.py#L101)):

| Index | Tensor | Shape | Dùng ở đâu |
|-------|--------|-------|------------|
| 0 | `dis_sem` | `[383, 383]` | Disease View 2 |
| 1 | `mi_fun` | `[495, 495]` | miRNA View 2 |
| 2 | `train` indices | `[pos, neg]` list | Positive/negative pairs để train |
| 3 | `test` indices | `[pos, neg]` list | Pairs để evaluate fold này |
| 4 | `md_p` | `[495, 383]` | Association matrix |
| 5 | `md_true` | `[495, 383]` | Ground truth |
| 6 | `independent.train` | indices | - |
| 7 | `independent.test` | indices | - |
| 8 | `d_gs` | `[383, 383]` | Disease View 1 |
| 9 | `m_ss` | `[495, 495]` | miRNA View 1 |
| 10 | `ID` | `[383, 383]` | Integrated disease sim |
| 11 | `IM` | `[495, 495]` | Integrated miRNA sim |

**Nhớ mnemonic**: index 8, 9 = "View 1" (gene/sequence); index 0, 1 = "View 2" (semantic/functional).

---

## 4. Hypergraph construction — build 4 matrix G

Trong [main_experiments_hetero1.py:691](main_experiments_hetero1.py#L691) — `train_epoch_optimized()`:

### 4a. Concatenate features

```python
concat_miRNA_view1 = cat([association_matrix, m_ss_data], dim=1)
#  shape: [495, 383 + 495] = [495, 878]

concat_miRNA_view2 = cat([association_matrix, mi_fun_data], dim=1)
#  shape: [495, 383 + 495] = [495, 878]

concat_disease_view1 = cat([association_matrix.t(), d_gs_data], dim=1)
#  shape: [383, 495 + 383] = [383, 878]

concat_disease_view2 = cat([association_matrix.t(), dis_sem_data], dim=1)
#  shape: [383, 495 + 383] = [383, 878]
```

Ý tưởng: mỗi node (miRNA hoặc disease) được mô tả bởi **association pattern + 1 loại similarity**.

### 4b. Build hypergraph Laplacian G bằng KNN

Với K=13 (default `args.K_neigs=[13]`):

```python
G_mi_view1  = constructHW_knn(concat_mi_tensor_view1, K=13)  → [495, 495]
G_mi_view2  = constructHW_knn(concat_mi_tensor_view2, K=13)  → [495, 495]
G_dis_view1 = constructHW_knn(concat_dis_tensor_view1, K=13) → [383, 383]
G_dis_view2 = constructHW_knn(concat_dis_tensor_view2, K=13) → [383, 383]
```

**Bước tính bên trong** (xem [hypergraph_construct_KNN.py:125](hypergraph_construct_KNN.py#L125)):

1. `Eu_dis(X)`: tính khoảng cách Euclidean N×N
2. `construct_H_with_KNN_from_distance()`: với mỗi node, chọn K láng giềng gần nhất → tạo hyperedge
   - Kết quả: incidence matrix **H** shape `[N, N]` (binary: H[i,j] = 1 nếu node i thuộc hyperedge j)
3. `_generate_G_from_H(H)`: tính Laplacian
   ```
   G = D_v^(-½) · H · W · D_e^(-1) · Hᵀ · D_v^(-½)
   ```
   - `D_v`: degree matrix của node
   - `D_e`: degree matrix của hyperedge
   - `W`: weight của hyperedge (mặc định = 1)
4. G là matrix vuông `[N, N]` — dùng như "adjacency smoothing operator" trong hypergraph convolution

---

## 5. Heterogeneous graph — `create_hetero_data_optimized()`

Khác với hypergraph (phẳng, chỉ 1 loại node), **heterogeneous graph** có:
- 2 node types: `miRNA` (495 node), `disease` (383 node)
- 4 edge types:
  - `(miRNA, associates, disease)` — từ association matrix
  - `(disease, associates, miRNA)` — reverse
  - `(miRNA, similar, miRNA)` — từ similarity > threshold (0.5)
  - `(disease, similar, disease)` — từ similarity > threshold

Dùng `torch_geometric.data.HeteroData`. Xem [create_hetero_data.py:5](create_hetero_data.py#L5).

---

## 6. `HeterogenousGraphCLAMIR.forward()` — pipeline chính

[hetero_model.py:622](hetero_model.py#L622). Nhận:

```
concat_mi_tensor    [495, 878]
concat_dis_tensor   [383, 878]
G_mi_Kn, G_mi_Km    [495, 495]    ← 2 hypergraph views của miRNA
G_dis_Kn, G_dis_Km  [383, 383]    ← 2 hypergraph views của disease
hetero_data         (HeteroData object)
```

**5 giai đoạn**:

### Stage 1: Hypergraph convolution (intra-view CL)

```python
mi_feature1, mi_feature2, mi_intra_loss = self.CL_HGCN_mi(
    concat_mi_tensor, G_mi_Kn,    # View 1
    concat_mi_tensor, G_mi_Km     # View 2
)
# mi_feature1, mi_feature2: [495, 256]  (hidden_dim=256)
# mi_intra_loss: scalar (InfoNCE contrastive loss)
```

Tương tự cho disease. Xem `CL_HGCN.forward()` [hetero_model.py:216](hetero_model.py#L216).

### Stage 2: Attention fusion

```python
mi_feature_fused = AM_mi([mi_feature1, mi_feature2])
# Đơn giản: 0.6 * feature1 + 0.4 * feature2  (không phải softmax attention!)
# shape: [495, 256]
```

Xem [hetero_model.py:257](hetero_model.py#L257). **Ghi chú**: tên là "Attention Mechanism" nhưng thực chất chỉ weighted sum tĩnh.

### Stage 3: Inter-view contrastive (cross-modal)

```python
inter_view_loss = self.inter_view_cl(
    mi_feature_fused,       # [495, 256]
    dis_feature_fused,      # [383, 256]
    association_matrix_binary  # [495, 383]
)
```

InfoNCE giữa miRNA ↔ disease: cặp có association → pull, không → push.
Xem [hetero_model.py:33](hetero_model.py#L33).

### Stage 4: Similarity reconstruction (self-supervised regularization)

```python
mi_sim_reconstructed  = self.miRNA_decoder(mi_feature_fused)   # [495, 495]
dis_sim_reconstructed = self.disease_decoder(dis_feature_fused) # [383, 383]
```

Decoder ép feature embedding tái tạo được similarity matrix gốc → regularization.
Xem `SimpleHypergraphDecoder` [hetero_model.py:283](hetero_model.py#L283).

### Stage 5: HGT layers + final prediction

```python
x_dict = {
    'miRNA':   self.node_transformers['miRNA'](mi_feature_fused),     # [495, 256]
    'disease': self.node_transformers['disease'](dis_feature_fused),  # [383, 256]
}
for layer in self.hgt_layers:   # nlayer=2 layers
    x_dict = layer(x_dict, edge_index_dict)

mi_emb  = x_dict['miRNA']   # [495, 256]
dis_emb = x_dict['disease'] # [383, 256]

score = self.association_predictor(mi_emb, dis_emb)
# shape: [495, 383, 5] = [existence_prob, type1_prob, type2_prob, type3_prob, type4_prob]
```

---

## 7. Return của forward

```python
return score, mi_cl_loss, dis_cl_loss, mi_sim_reconstructed, dis_sim_reconstructed
```

| Output | Shape | Dùng ở đâu |
|--------|-------|------------|
| `score` | `[495, 383, 5]` | Feed vào `SimplifiedMultiTypeAssociationLoss` + eval metrics |
| `mi_cl_loss` | scalar | Thêm vào total loss |
| `dis_cl_loss` | scalar | Thêm vào total loss |
| `mi_sim_reconstructed` | `[495, 495]` | MSE với `mi_fun_data` (recon loss) |
| `dis_sim_reconstructed` | `[383, 383]` | MSE với `dis_sem_data` (recon loss) |

**Lưu ý**: `mi_cl_loss = mi_intra_loss + 0.3 * inter_view_loss` (inter_view_weight = 0.3).

---

## 8. Loss computation — [main_experiments_hetero1.py:804](main_experiments_hetero1.py#L804)

```python
# Loss chính: existence + type
recover_loss = regression_crit(one_index, zero_index, score, association_matrix)
#   = 0.3 * focal_loss_existence + 0.7 * weighted_CE_type

# Reconstruction regularization
mi_recon_loss  = F.mse_loss(mi_sim_recon, mi_fun_data)
dis_recon_loss = F.mse_loss(dis_sim_recon, dis_sem_data)

# L2 regularization
reg_loss = get_L2reg(model.parameters())

# Total
tol_loss = recover_loss + mi_cl_loss + dis_cl_loss \
         + 0.15 * (mi_recon_loss + dis_recon_loss) \
         + 0.0001 * reg_loss
```

Focal loss chi tiết trong `_compute_existence_loss` ([main_experiments_hetero1.py:115](main_experiments_hetero1.py#L115)):
```
pos_loss = -(1 - p)^γ · log(p)        với γ=2.0
neg_loss = -p^γ · log(1 - p) · α      với α=0.5
```

Type loss dùng **weighted cross-entropy** với class weights theo **Effective Number** (`beta=0.99999`):
```
weights = [0.780, 1.822, 0.977, 0.421]   # cho 4 class
```

---

## 9. Evaluation

Sau khi train xong các epoch của 1 fold, `test_optimized()` chạy model ở eval mode, lấy `score`, rồi:

- **Binary metrics**: lấy `score[:, :, 0]` (existence) để tính AUC, AUPR, F1
- **Top-1 metrics**: với mỗi positive pair, lấy `argmax(score[i, j, 1:])` so với ground truth type

Xem [Calculate_Metrics.py](Calculate_Metrics.py) `Metric_fun`.

---

## 10. Key shape cheatsheet

```
495 = số miRNA
383 = số disease
256 = hidden_dim (có thể chỉnh qua hidden_list)
128 = node_dim trong SimplifiedTypePredictor
5   = 1 (existence) + 4 (types)
878 = 383 + 495 = dim sau concat view
13  = K của KNN (args.K_neigs)
9   = số cluster K-means (args.clusters, backup)
2   = nlayer HGT
8   = số attention head
```

Nếu bạn thấy tensor shape lạ, 95% là combo của các số trên.
