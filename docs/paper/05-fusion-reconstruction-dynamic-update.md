# Chương 5 — Attention view fusion, similarity reconstruction & dynamic graph update

> Paper §2.5 (Eq. 16–18), §2.6 (Eq. 19–21), §2.8 (Eq. 30–31). Ba cơ chế "nhỏ" nhưng quan trọng của DHGCMDA.

---

## 5.1. Attention-guided adaptive view fusion (Eq. 16–18)

Sau Chương 4, mỗi modality có 2 embedding $Z_1, Z_2$ (một từ mỗi view). Cần gộp thành 1 embedding duy nhất — nhưng hai view không đóng góp ngang nhau, và mức độ quan trọng còn *thay đổi theo ngữ cảnh*.

Cơ chế 2 tầng của paper:

**Tầng 1 — attention theo "tổng thể" view.** Pool toàn bộ node embeddings của view $v$ rồi cho qua MLP 2 lớp + sigmoid:

$$\alpha_v = \sigma\big(W_2\, \mathrm{ReLU}(W_1\, \mathrm{GAP}(Z_v))\big) \qquad (16)$$

- $\mathrm{GAP}$ = global average pooling — nén cả ma trận $Z_v$ [n×d] thành 1 vector tóm tắt "view này đang cho thông tin gì".
- $\alpha_v \in [0,1]$: trọng số theo từng chiều feature (element-wise).

**Tầng 2 — gating + hệ số học toàn cục.** Điều chế từng view bằng attention, rồi trộn với hệ số $\beta$ learnable:

$$\tilde Z_v = \mathrm{ReLU}(\alpha_v Z_v) \qquad (17)$$
$$Z_{fused} = \sum_{v=1}^{2} \beta_v \tilde Z_v \qquad (18)$$

Paper gọi là "two-stage gating": $\alpha$ bắt tầm quan trọng *cụ thể từng instance*, $\beta$ mã hóa *độ tin cậy toàn cục* của view. Áp dụng độc lập cho miRNA → $Z_M$, và disease → $Z_D$.

> **⚠️ Điểm lệch code–paper đáng chú ý** (chi tiết Chương 9): class `HGCN_Attention_Mechanism` trong code **không phải attention** — chỉ là weighted sum cố định `0.6·Z₁ + 0.4·Z₂` (comment trong code ghi rõ). Tức Eq. 16–18 không được implement; ablation "w/o AVF" của repo đo weighted-average `(Z1+Z2)/2` so với `0.6/0.4` — gần như cùng một phép toán. Đây là một trong các faithfulness gap của bản reproduce.

## 5.2. Similarity reconstruction (Eq. 19–21) — "decoder kiểm chứng"

Một lo lắng khi học embedding: nó có giữ được cấu trúc similarity gốc không? Paper thêm **decoder** nhỏ ép embedding tái tạo lại ma trận similarity ban đầu:

$$Z_{proj} = W^{proj}_2\, \mathrm{ReLU}(W^{proj}_1 Z_{fused}), \quad W^{proj}_1\in\mathbb{R}^{d\times d/2},\ W^{proj}_2\in\mathbb{R}^{d/2\times d} \qquad (19)$$

$$\hat S = Z_{proj} Z_{proj}^T \qquad (20)$$

$$L_{recon} = \|S_M - \hat S_M\|_F^2 + \|S_D - \hat S_D\|_F^2 \qquad (21)$$

Tức: nếu 2 miRNA similar trong $S_M$ gốc thì embedding sau chiếu cũng phải similar. Ba vai trò:

1. **Regularizer** chống overfit (embedding không được bay xa khỏi cấu trúc đã biết).
2. **Cầu nối** giữa "không gian embedding học được" và "không gian similarity gốc" — graph regularization.
3. **Nguyên liệu cho dynamic update** — mục 5.3.

> Trong code: `SimpleHypergraphDecoder` (project xuống d/2 qua Sequential Linear–ReLU–Dropout–Linear–ReLU, rồi `normalize + mm(z, z.T)`) — lưu ý code **chuẩn hóa L2 từng hàng** trước khi nhân, nên $\hat S$ là cosine similarity chứ không phải dot-product thuần như Eq. 20. Loss: `F.mse_loss(mi_sim_recon, mi_fun_data)` (tức so với *một* similarity gốc — FSM cho miRNA, SSM cho disease — chứ không phải tổng hai view, xem Chương 9).

## 5.3. Dynamic hypergraph update (§2.8, Eq. 30–31) — đồ thị "tự sửa mình"

Vấn đề: similarity ban đầu lấy từ database → có noise đo lường. Nếu embedding học được đã "sạch" hơn, tại sao không build lại graph từ embedding?

Cơ chế paper:

- **Mỗi 5 epoch**: lấy $\hat S_M, \hat S_D$ vừa reconstruct, threshold ở $\theta = 0.5$ để tạo cạnh mới:

$$E_M = \{(m_i, m_j)\ |\ \hat S_{M,ij} > \theta,\ i\ne j\} \quad (30)$$
$$E_D = \{(d_i, d_j)\ |\ \hat S_{D,ij} > \theta,\ i\ne j\} \quad (31)$$

- Thay cạnh similarity cũ trong heterogeneous graph bằng tập mới; **giữ nguyên association edges** (ground truth, không được đụng vào — đó là supervision).
- Hypergraph $H$ cũng được dựng lại bằng KNN trên similarity đã update.

→ Vòng lặp kín: *embedding tốt hơn → similarity tái tạo tốt hơn → đồ thị tốt hơn → embedding tốt hơn*. Paper gọi là "co-evolution of graph structure and representations". θ=0.5 là điểm giữa của khoảng [0,1] đã normalize — và vì đồ thị tự update mỗi 5 epoch, lựa chọn θ không quá nhạy.

> Trong code: `main_experiments_hetero1.py:990-995` — `epoch % update_graph_frequency == 0` (default 5) gọi `create_hetero_data_optimized(train_data_list, mi_sim_recon, dis_sim_recon)` rebuild heterogeneous graph với similarity edges threshold > 0.5. ⚠️ Nhưng **G_mi/G_dis (4 hypergraph Laplacian) KHÔNG được rebuild** — chỉ heterogeneous graph cho HGT đổi; hypergraph của CL_HGCN giữ nguyên suốt training. Paper nói "hypergraphs H are reconstructed using KNN" — code không làm phần này (Chương 9).

## 5.4. Đặt vào pipeline

```
Z1, Z2 (mỗi modality) ──(5.1 fusion)──► Z_M, Z_D ──► HGT + predictor (Chương 6)
                    │
                    └──(5.2 decoder)──► Ŝ_M, Ŝ_D ──(mỗi 5 epoch)──► rebuild similarity edges
                                                                         (5.3)
```

---

**Chương tiếp**: [06 — HGT & bộ dự đoán](06-hgt-va-predictor.md) — phần cuối của model: transformer trên đồ thị dị biệt và output 2-đầu.
