# Chương 6 — Heterogeneous Graph Transformer (HGT) & bộ dự đoán type

> Paper §2.7 (Eq. 22–29). Giai đoạn cuối: trộn thông tin giữa miRNA và disease, rồi chấm điểm từng cặp.

---

## 6.1. Heterogeneous graph G — "sân chơi" cuối của model

Đến đây ta có $Z_M$ (embedding miRNA đã fuse) và $Z_D$ (disease). Paper dựng một đồ thị dị biệt $G = (V, E, R)$:

- **Node**: $V = M \cup D$ — 2 loại node (miRNA, disease), feature ban đầu = $Z_M, Z_D$.
- **4 loại cạnh** ($R$):
  1. miRNA–disease *association* (từ $A$, chiều miRNA→disease)
  2. disease–miRNA *association* (chiều ngược — để message đi cả 2 hướng)
  3. miRNA–miRNA *similarity* (trọng số $\hat S_M$ — cái được update mỗi 5 epoch ở Chương 5)
  4. disease–disease *similarity* (trọng số $\hat S_D$)

Vì node và edge có *loại* khác nhau, cần GNN "hiểu type" → **HGT** (Heterogeneous Graph Transformer, Hu et al. 2020 — paper mượn kiến trúc này).

## 6.2. Type-aware Q/K/V (Eq. 22–26)

Trước hết, mỗi loại node có chiếu riêng vào không gian chung:

$$X_{miRNA} = W_{miRNA} Z_M, \qquad X_{disease} = W_{disease} Z_D \quad (22-23)$$

Với node $i$ loại $\tau(i)$, mỗi HGT layer tính query/key/value bằng **ma trận riêng theo loại node**:

$$Q^{\tau(i)}_i = W^{\tau(i)}_Q x_i, \quad K^{\tau(i)}_i = W^{\tau(i)}_K x_i, \quad V^{\tau(i)}_i = W^{\tau(i)}_V x_i \quad (24-26)$$

→ miRNA "hỏi" khác disease "hỏi"; đây là *type-aware*.

## 6.3. Attention theo loại cạnh (Eq. 27–28)

Với cạnh loại $r$ từ node $j$ → node $i$:

$$\alpha^{r}_{ij} = \mathrm{softmax}_{j\in T(i)}\left( \frac{\big(W^r_{att} Q^{\tau(i)}_i\big)^T \big(W^r_{msg} K^{\tau(j)}_j\big)}{\sqrt{d_k}} \right) \qquad (27)$$

- $W^r_{att}, W^r_{msg}$: transform riêng *theo loại quan hệ* $r$ — attention "miRNA ảnh hưởng disease" khác "miRNA tương tự miRNA".
- $1/\sqrt{d_k}$: chuẩn scaled dot-product (chống gradient vanish).
- Multi-head: 4 heads (paper ghi "four attention heads per layer").

Cập nhật node $i$ = tổng hợp value của láng giềng, weighted bởi attention, gộp trên mọi loại cạnh:

$$x'_i = \sum_{r\in R}\sum_{j\in T^r_i} \alpha^r_{ij}\,\big(W^r_{msg} V^{\tau(j)}_j\big) \qquad (28)$$

Rồi chuẩn residual + dropout + LayerNorm:

$$x^{(l+1)}_i = \mathrm{LayerNorm}\big(x^{(l)}_i + \mathrm{Dropout}(x'_i)\big) \qquad (29)$$

Paper dùng **2 HGT layers** — đủ để mỗi node "nghe" được hàng xóm 2-hop (miRNA → disease liên kết → miRNA khác cùng liên kết disease đó).

> Trong code: `EnhancedHGTLayer` bọc `torch_geometric.nn.HGTConv` + LayerNorm per node-type + Dropout (hetero_model.py:351–384). `hetero_data` được build ở `create_hetero_data_optimized` với đúng 4 edge types trên (`associates` ×2 chiều + `similar` ×2). Trước HGT là `node_transformers` — Linear riêng cho từng node type, tương ứng Eq. 22–23.

## 6.4. Bilinear predictor — hai câu hỏi một lúc

Embedding cuối của miRNA $i$ và disease $j$ đi vào **bilinear predictor** chấm đồng thời:

- **Existence**: cặp $(i,j)$ có liên kết không? (binary)
- **Type**: nếu có, thuộc type nào trong $C$ loại? (multi-class)

Trực giác bilinear: điểm cho type $t$ là $m_i^T R_t\, d_j$ — mỗi type có một "bản đồ tương tác" $R_t$ riêng; điểm cao khi embedding miRNA và disease "khớp" theo cách đặc trưng của type đó.

> Trong code `SimplifiedTypePredictor`: mỗi type là *vector* $r_t$ thay vì ma trận — `type_logit[i,j,t] = (mi_feat[i] ⊙ r_t) · dis_feat[j]` (bilinear **diag**, rank-d). Repo có thêm `--predictor_mode full_bilinear` ($m_i^T W_t d_j$, $W_t$ đầy đủ d×d) — được thêm vì audit thấy diag form là rewrite không trung thực của paper "bilinear predictor"; kết quả v2.0 tốt hơn hẳn (Chương 9).

## 6.5. Đầu ra

Score tensor: `[n_m, n_d, 1 + C]` — channel 0 là xác suất tồn tại liên kết, các channel sau là phân phối type. Từ đây loss (Chương 7) và metric Top-1 (Chương 8) lấy số liệu.

---

**Chương tiếp**: [07 — Loss tổng hợp & training loop](07-loss-tong-hop-va-training.md) — 4 loss ghép lại thế nào, train 1 epoch gồm những gì.
