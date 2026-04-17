# Model Internals — DHGCMDA

Giải thích từng class trong [hetero_model.py](hetero_model.py) + công thức toán học của các phép tính chính.

---

## Table of Contents

1. [`HGNN_conv` — Hypergraph convolution layer](#1-hgnn_conv)
2. [`HGCN` — Wrapper + activation](#2-hgcn)
3. [`CL_HGCN` — Dual-view contrastive HGCN](#3-cl_hgcn)
4. [`HGCN_Attention_Mechanism` — Fusion](#4-hgcn_attention_mechanism)
5. [`InterViewContrastiveLoss` — Cross-modal CL](#5-interviewcontrastiveloss)
6. [`SimpleHypergraphDecoder` — Similarity reconstruction](#6-simplehypergraphdecoder)
7. [`EnhancedHGTLayer` — Heterogeneous graph transformer](#7-enhancedhgtlayer)
8. [`SimplifiedTypePredictor` — Multi-type classifier](#8-simplifiedtypepredictor)
9. [`HeterogenousGraphCLAMIR` — Main model](#9-heterogenousgraphclamir)
10. [`SimplifiedMultiTypeAssociationLoss` — Total loss](#10-simplifiedmultitypeassociationloss)

---

## 1. HGNN_conv

[hetero_model.py:147](hetero_model.py#L147)

### Công thức

Cho input `x` ∈ ℝ^(N×F) và hypergraph Laplacian `G` ∈ ℝ^(N×N):

```
output = G · (x · W) + b
```

Trong đó `W` ∈ ℝ^(F×F'), `b` ∈ ℝ^(F').

### Ý nghĩa

- `x · W`: linear transform feature per-node
- `G · (x·W)`: **propagate** qua hyperedge — mỗi node nhận weighted sum từ các node chia sẻ hyperedge
- G được build offline (xem [NOTES_DATAFLOW.md §4b](NOTES_DATAFLOW.md)), stable trong epoch

### Khác gì GCN?

GCN: `G = D^(-½) · A · D^(-½)` với A là adjacency matrix (pairwise edges).
HGCN: `G = D_v^(-½) · H · W · D_e^(-1) · Hᵀ · D_v^(-½)` với H là incidence matrix (hyperedge có thể chứa nhiều node).

→ HGCN capture được **high-order relations** (3+ node cùng thuộc 1 hyperedge) mà GCN không làm được.

---

## 2. HGCN

[hetero_model.py:181](hetero_model.py#L181)

Đơn giản: 1 layer `HGNN_conv` + `LeakyReLU(0.25)` activation.

```python
def forward(self, x, G):
    return LeakyReLU(HGNN_conv(x, G))
```

Output shape = `[N, hidden_dim]` (mặc định hidden_dim = 256).

**Note**: dù tên là "HGCN" (gợi ý multi-layer), class này chỉ có **1 layer**. Multi-layer được thực hiện ở `HGT` phía sau, không phải ở đây.

---

## 3. CL_HGCN

[hetero_model.py:201](hetero_model.py#L201) — **Dual-view contrastive hypergraph conv**.

### Cấu trúc

```python
self.hgcn1 = HGCN(in_size, hid_list)   # xử lý View 1
self.hgcn2 = HGCN(in_size, hid_list)   # xử lý View 2
self.fc1   = Linear(hidden, num_proj_hidden)
self.fc2   = Linear(num_proj_hidden, hidden)
self.tau   = 0.5
```

### Forward

```python
z1 = hgcn1(x1, adj1)       # embedding từ View 1
z2 = hgcn2(x2, adj2)       # embedding từ View 2
h1 = projection(z1)         # projection head (2-layer MLP with ELU)
h2 = projection(z2)

loss = α · sim(h1, h2) + (1-α) · sim(h2, h1)
return z1, z2, loss
```

### Contrastive loss `sim(h1, h2)` — NT-Xent / SimCLR-style

```
numerator   = exp(cos(h1[i], h2[i]) / τ)                     ← positive pair (same node, 2 views)
denominator = Σ_j exp(cos(h1[i], h1[j]) / τ)                 ← intra-view negatives
            + Σ_j exp(cos(h1[i], h2[j]) / τ)                 ← inter-view negatives
            - exp(cos(h1[i], h1[i]) / τ)                     ← trừ self
loss[i]     = -log(numerator / denominator)
```

Trung bình trên tất cả i. Intuition: **kéo gần h1[i] với h2[i]** (cùng node qua 2 view), **đẩy xa khỏi node khác**.

Reference: SimCLR (Chen et al., 2020).

---

## 4. HGCN_Attention_Mechanism

[hetero_model.py:257](hetero_model.py#L257)

**KHÔNG phải softmax attention!** Đơn giản:

```python
output = 0.6 * feature1 + 0.4 * feature2
```

Tên gọi "attention" ở đây gây hiểu nhầm. Trong các version trước (xem git log nếu có) có thể là attention thực sự, nhưng đã simplified để stable hơn trên dynamic graph.

Muốn thay bằng attention thật: implement multi-head attention sau đó thay class này.

---

## 5. InterViewContrastiveLoss

[hetero_model.py:33](hetero_model.py#L33) — **Cross-modal CL giữa miRNA và disease**.

### Input
- `mi_embeddings`: `[495, dim]`
- `dis_embeddings`: `[383, dim]`
- `association_matrix`: `[495, 383]` (binary: 1 nếu có association)

### Bước 1: Similarity matrix

```
M_norm  = L2_normalize(mi_embeddings)     [495, dim]
D_norm  = L2_normalize(dis_embeddings)    [383, dim]
S       = (M_norm · D_normᵀ) / τ          [495, 383]    τ=0.5
S_exp   = exp(clamp(S, -10, 10))
```

### Bước 2: InfoNCE với masks

```
positive_mask[i, j] = 1  nếu association[i, j] > 0
negative_mask[i, j] = 1  nếu association[i, j] = 0

# Với mỗi positive pair (i, j):
pos_sim           = S_exp[i, j]
neg_sim_sum[i]    = Σ_k S_exp[i, k] · negative_mask[i, k]
denominator[i, j] = pos_sim + neg_sim_sum[i]
loss[i, j]        = -log(pos_sim / denominator[i, j])

infonce_loss      = mean(loss over all positive pairs)
```

### Bước 3: Margin ranking loss (phụ trợ)

```
pos_mean = mean(S[positive pairs])
neg_mean = mean(S[sampled negative pairs])
margin_loss = ReLU(margin - (pos_mean - neg_mean))     margin=0.5
```

### Total

```
inter_view_loss = infonce_loss + 0.1 * margin_loss
```

### Vì sao kết hợp cả 2?

- **InfoNCE** chuẩn: tốt cho học representation tổng quát
- **Margin**: explicit ép cho khoảng cách positive–negative đủ lớn (> margin)

Chỉ active khi `model.training = True`.

---

## 6. SimpleHypergraphDecoder

[hetero_model.py:283](hetero_model.py#L283)

### Cấu trúc

```
Linear(hidden → hidden) → ReLU → Dropout(0.1) → Linear(hidden → hidden/2) → ReLU
```

Output là **reconstructed similarity matrix**:

```python
projected          = MLP(x)                           # [N, hidden/2]
features_normalized = L2_normalize(projected)
similarity         = features_normalized @ features_normalized.T   # [N, N]

output_sim = eye(output_dim)       # identity init
output_sim[:N, :N] = similarity    # fill vào góc trên trái
return output_sim                   # [output_dim, output_dim]
```

### Dùng để làm gì?

Train loss có thêm term:

```python
mi_recon_loss = MSE(mi_sim_reconstructed, mi_fun_data)
```

Ép embedding phải giữ lại **cấu trúc similarity ban đầu** → regularization. Nếu không có, model có thể học embedding "lệch" khỏi domain knowledge.

---

## 7. EnhancedHGTLayer

[hetero_model.py:319](hetero_model.py#L319)

Wrapper quanh `torch_geometric.nn.HGTConv`:

```python
out_dict = HGTConv(x_dict, edge_index_dict)     # Heterogeneous Graph Transformer
out_dict = {type: Dropout(LayerNorm(out)) for type, out in out_dict.items()}
```

### HGTConv là gì?

Reference: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020).

Core idea: **multi-head attention nhưng với parameter riêng cho từng (src_type, relation, dst_type)**.

Công thức đơn giản hoá — với edge `e = (s, r, t)`:

```
Q_t(v_t)         = W_Q^t · v_t                                 (query từ node đích)
K_s^r(v_s)       = W_K^{s,r} · v_s                             (key với relation r)
V_s^r(v_s)       = W_V^{s,r} · v_s                             (value với relation r)

Attn(s→t)        = softmax(Q_t · K_s^r / sqrt(d)) · μ^r
message          = V_s^r · Attn(s→t)

v'_t = sum over all (s, r, t) edges of message
```

Trong DHGCMDA, 4 edge types:
- `(miRNA, associates, disease)`
- `(disease, associates, miRNA)`
- `(miRNA, similar, miRNA)`
- `(disease, similar, disease)`

→ mỗi type có weight riêng → model tự học cách "lắng nghe" các quan hệ khác nhau.

Mặc định `args.nlayer = 2` → stack 2 layer HGT liên tiếp.

---

## 8. SimplifiedTypePredictor

[hetero_model.py:417](hetero_model.py#L417) — **Classifier cuối**.

### Cấu trúc

```python
mi_projector  = Linear(node_dim, hidden) → LayerNorm → Dropout(0.2)
dis_projector = Linear(node_dim, hidden) → LayerNorm → Dropout(0.2)
exist_relation = Parameter(hidden_dim)                     # vector quan hệ "existence"
type_relations = Parameter(num_types, hidden_dim)          # 4 vector cho 4 type
temperature    = Parameter(2.0)                            # learnable temperature
```

### Công thức — BilinearDiag scoring

Inspired bởi **ComplEx / DistMult** (knowledge graph embedding).

#### Existence score

```
mi_feat = mi_projector(mi_embeddings)      # [495, hidden]
dis_feat = dis_projector(dis_embeddings)   # [383, hidden]

existence_score[i, j] = sigmoid( mi_feat[i] · diag(r_exist) · dis_feat[j]ᵀ )
                     = sigmoid( Σ_k mi_feat[i,k] · r_exist[k] · dis_feat[j,k] )
```

Tức là mỗi chiều `k` của embedding đóng góp theo trọng số `r_exist[k]`.

#### Type score

Với mỗi type t ∈ {0, 1, 2, 3}:

```
type_logit[i, j, t] = mi_feat[i] · diag(r_type_t) · dis_feat[j]ᵀ
```

Sau đó softmax qua 4 type với temperature scaling:

```
type_prob[i, j, :] = softmax(type_logit[i, j, :] / T)        T ∈ [0.5, 5.0] learnable
```

**Vì sao learnable temperature?** Không có T, tất cả prob có xu hướng → 0.25 (uniform). Learnable T cho phép model tự chọn mức sharpness.

### Khởi tạo `type_relations` — orthogonal init

```python
Q, _ = QR(randn(num_types, hidden))
type_relations = Q.T[:num_types] * sqrt(2 / hidden)
```

Orthogonal → đảm bảo các type vector ban đầu **trực giao** → tránh bị collapse về cùng 1 direction.

### Output

```
score = cat([existence_score.unsqueeze(-1), type_probs], dim=-1)
# shape: [495, 383, 5] = [exist_prob, p_type1, p_type2, p_type3, p_type4]
```

---

## 9. HeterogenousGraphCLAMIR

[hetero_model.py:518](hetero_model.py#L518) — **Model chính**.

Xem chi tiết forward ở [NOTES_DATAFLOW.md §6](NOTES_DATAFLOW.md). Tóm tắt module list:

```
HeterogenousGraphCLAMIR:
├── CL_HGCN_mi      (dual-view contrastive cho miRNA)
├── CL_HGCN_dis     (dual-view contrastive cho disease)
├── inter_view_cl   (InterViewContrastiveLoss)
├── AM_mi, AM_dis   (HGCN_Attention_Mechanism — weighted sum)
├── miRNA_decoder   (SimpleHypergraphDecoder)
├── disease_decoder (SimpleHypergraphDecoder)
├── node_transformers (2 linear layers)
├── hgt_layers      (ModuleList of EnhancedHGTLayer × nlayer)
└── association_predictor (SimplifiedTypePredictor)
```

### Dynamic hypergraph update

Trong training, sau mỗi `graph_update_frequency` epoch (mặc định 5), nếu `mi_sim_reconstructed` và `dis_sim_reconstructed` lệch đáng kể (MSE > 0.01) so với lần trước, **rebuild edges** `('miRNA','similar','miRNA')` và `('disease','similar','disease')` trong heterogeneous graph.

Xem `_update_hetero_data_optimized` [hetero_model.py:761](hetero_model.py#L761).

---

## 10. SimplifiedMultiTypeAssociationLoss

[main_experiments_hetero1.py:56](main_experiments_hetero1.py#L56) — **Loss tổng**.

### Class weights — Effective Number (Cui et al., CVPR 2019)

Thay vì weight = 1/count (linear inverse), dùng:

```
E_n = (1 - β^n) / (1 - β)        với β = 0.99999

weight_class_c = 1 / E_n_c
weight_normalized = weight * num_classes / sum(weight)
```

Với counts = [367, 157, 293, 681] → weights ≈ `[0.780, 1.822, 0.977, 0.421]`.

Tại sao Effective Number tốt hơn linear inverse? Linear inverse overweight cực đoan khi count nhỏ. Effective Number có diminishing returns — mỗi sample thêm vào "đóng góp" ít dần do overlap → weight không bị bùng nổ.

### Existence loss (Focal)

Với γ = 2.0, α = 0.5:

```
positive:  L = -(1 - p)^γ · log(p)
negative:  L = -p^γ · log(1 - p) · α
```

Focal loss down-weight easy samples (p gần 1 cho positive hoặc p gần 0 cho negative).

### Type loss (Weighted CE + Label Smoothing)

```
L_type = CrossEntropy(
    logits=type_pred[positive pairs],
    target=type_indices,           # 0, 1, 2, 3
    weight=class_weights,
    label_smoothing=0.1
)
```

Label smoothing = 0.1 → target distribution 90% on true class, 10% uniform on others → regularization chống overconfidence.

### Total

```
L = 0.3 · L_existence + 0.7 · L_type
```

---

## 📚 Reading order gợi ý

Để đọc hiểu model từ dưới lên:

1. `HGNN_conv` → `HGCN`: đơn giản, nắm concept hypergraph conv
2. `CL_HGCN`: hiểu contrastive loss SimCLR-style
3. `InterViewContrastiveLoss`: cross-modal CL
4. `SimplifiedTypePredictor`: bilinear scoring
5. `EnhancedHGTLayer`: đọc paper HGT để hiểu sâu
6. `HeterogenousGraphCLAMIR.forward()`: nối tất cả lại

---

## 🔗 References

- **SimCLR** (Chen et al., 2020): contrastive learning framework
- **HGT** (Hu et al., WWW 2020): Heterogeneous Graph Transformer
- **HGNN** (Feng et al., AAAI 2019): Hypergraph Neural Networks
- **ComplEx/DistMult**: knowledge graph embedding với bilinear scoring
- **Focal Loss** (Lin et al., ICCV 2017): class imbalance
- **Effective Number** (Cui et al., CVPR 2019): class re-weighting
