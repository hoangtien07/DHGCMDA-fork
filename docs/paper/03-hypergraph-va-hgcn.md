# Chương 3 — Hypergraph & tích chập siêu đồ thị (HGCN)

> Paper §2.3–2.4 (Eq. 3–5). Đây là block đầu tiên của model — "encoder" biến raw similarity thành embedding.

---

## 3.1. Graph thường vs hypergraph — trực giác

- **Graph thường**: cạnh nối đúng 2 node $(v_i, v_j)$. Message passing = "tôi lấy thông tin từng láng giềng một".
- **Hypergraph**: một **hyperedge** $e$ chứa *tập con bất kỳ* các node ($|e| \ge 2$). Message passing = "cả nhóm node trong cùng hyperedge trộn thông tin với nhau một lượt".

Tại sao sinh học cần cái thứ hai? Vì miRNA hoạt động *theo cụm*: một nhóm miRNA cùng điều tiết 1 pathway. Quan hệ "cùng cụm" là quan hệ nhiều-chiều (high-order) — cắt nó thành từng cặp pairwise làm mất thông tin. Đây là hạn chế #2 ở Chương 1.

## 3.2. Incidence matrix H — cách viết siêu đồ thị thành ma trận (Eq. 3)

$$H \in \{0,1\}^{n \times |E|}, \qquad H_{ij} = \begin{cases} 1 & v_i \in e_j \\ 0 & \text{otherwise} \end{cases}$$

Mỗi **cột** là một hyperedge (với KNN: cột $j$ chứa node $j$ + K láng giềng của nó → mỗi cột có đúng $K+1$ số 1).

## 3.3. Normalized hypergraph Laplacian G (Eq. 4)

$$G = D_v^{-1/2}\, H\, W\, D_e^{-1}\, H^T\, D_v^{-1/2}$$

với $D_v$ = ma trận bậc của node (đường chéo, số hyperedge chứa node), $D_e$ = bậc của hyperedge (số node trong edge), $W$ = trọng số edge (mặc định $W=I$ — không trọng số).

**Đọc trực giác — đây là phép "truyền tin 2 pha"** (paper giải thích rất rõ):

1. **Pha 1 — gom**: $H^T$ nhân vào = mỗi hyperedge cộng feature của mọi node nó chứa. $D_e^{-1}$ chia cho kích thước edge → hyperedge to không át vế.
2. **Pha 2 — rải**: $H$ nhân lại = phát feature tổng hợp từ hyperedge về từng node thành viên.
3. $D_v^{-1/2}$ hai bên = node nhiều kết nối không bị phóng đại tín hiệu (giống symmetric normalization của GCN Laplacian).

Vậy $G \in \mathbb{R}^{n \times n}$ đóng vai trò như "adjacency đã chuẩn hóa" — nếu $G_{ij}$ lớn nghĩa là node $i,j$ chia sẻ nhiều hyperedge. Công thức này chính là của **HGNN** (Feng et al. 2019); paper mượn nguyên dạng.

> Trong code: `hypergraph_construct_KNN._generate_G_from_H` làm đúng công thức này (`DV2 * H * W * invDE * HT * DV2`); nhánh `variable_weight` trả về thừa số riêng (dùng cho biến thể có trọng số).

## 3.4. Tích chập trên hypergraph (Eq. 5)

$$X^{(l+1)} = \sigma\big(G\, X^{(l)}\, \Theta^{(l)}\big)$$

- $X^{(0)} = X$: augmented feature ở Chương 2.
- $\Theta^{(l)}$: ma trận trọng số học được ở layer $l$.
- $\sigma$: activation (code dùng `LeakyReLU(0.25)`).

Đọc đúng thứ tự: $X\Theta$ biến đổi feature từng node (linear), rồi $G(\cdot)$ trộn thông tin theo hyperedge. Đây là "một lớp HGCN".

> Trong code `HGNN_conv.forward` viết `x = G @ (x @ W) + b` — thứ tự nhân tương đương (đẩy bias ra ngoài). `HGCN` class chỉ stack **1 layer** conv + activation.

## 3.5. Hai nhánh song song cho hai view

Kiến trúc thực tế (§2.4): **hai HGCN chạy song song**, mỗi cái xử lý một view:

```
X_view1 + G_view1 → HGCN₁ → Z₁
X_view2 + G_view2 → HGCN₂ → Z₂
```

Được áp dụng độc lập cho miRNA lẫn disease → tổng 4 nhánh HGCN (2 cho miRNA, 2 cho disease). Output mỗi bên là 2 embedding $Z_1, Z_2$ — *cùng một tập node, hai cách mã hóa*.

Đây chính là điểm neo cho contrastive learning ở chương sau: vì $Z_1[i]$ và $Z_2[i]$ mô tả **cùng node** $i$, ta có thể ép chúng "đồng ý" với nhau.

## 3.6. Đọc code

| Khái niệm paper | Code |
|---|---|
| $H$ incidence (KNN) | `hypergraph_construct_KNN.construct_H_with_KNN_from_distance` — ⚠️ dùng **Euclidean distance** (`Eu_dis`) thay cosine similarity như paper viết |
| $G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$ | `_generate_G_from_H` |
| $X^{(l+1)} = \sigma(GX^{(l)}\Theta^{(l)})$ | `HGNN_conv` + `HGCN` (1 lớp) |
| 2 nhánh view + projection head | `CL_HGCN` (hgcn1/hgcn2 + fc1/fc2) |

---

**Chương tiếp**: [04 — Contrastive learning](04-contrastive-learning.md) — làm sao ép 2 view "nói cùng một thứ" và căn chỉnh embedding miRNA↔disease.
