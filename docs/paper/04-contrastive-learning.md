# Chương 4 — Contrastive learning hai tầng (intra- & cross-modality)

> Paper §2.4.1–2.4.2 (Eq. 6–15). Phần "contrastive" trong tên DHGCMDA.

---

## 4.0. Contrastive learning là gì — phiên bản 30 giây

Ý tưởng SimCLR/InfoNCE: chọn một **anchor** (điểm neo), kéo embedding của **positive pair** (cặp "cùng một thứ") lại gần trong không gian embedding, đồng thời đẩy các **negative pair** (cặp khác nhau) ra xa. Loss điển hình là dạng softmax tính trên similarity, có **temperature** $t$ điều chỉnh độ sắc nét.

DHGCMDA áp dụng ở *hai chiều*:

| Tầng | Positive pair | Negative pairs | Mục đích |
|------|---------------|----------------|----------|
| **Intra-modality** (§2.4.1) | Cùng node $i$ ở 2 view: $(u_i, v_i)$ | Node khác ở cả 2 view $(u_k, v_k)$, $k\ne i$ | 2 view phải mã hóa *cùng một thực thể* → lọc noise riêng của từng view |
| **Cross-modality** (§2.4.2) | miRNA $i$ & disease $j$ **có association** | miRNA $i$ & disease $k$ không association | Căn chỉnh 2 không gian embedding khác nhau về cùng một hệ quy chiếu |

## 4.1. Intra-modality CL (Eq. 6–9) — "hai view phải đồng thuận"

Với node $v_i$: $u_i$ là embedding từ view 1, $v_i$ là embedding từ view 2 (sau projection head $g$). Loss trên mỗi cặp positive:

$$\ell_{CL}(u_i, v_i) = -\log \frac{\exp(\theta(u_i, v_i)/t)}{\Phi_i} \qquad (6)$$

$$\Phi_i = \exp\big(\theta(u_i, v_i)/t\big) + \sum_{k\ne i}\exp\big(\theta(u_i, u_k)/t\big) + \sum_{k\ne i}\exp\big(\theta(u_i, v_k)/t\big) \qquad (7)$$

Đây là **NT-Xent** (normalized temperature-scaled cross entropy): tử số = similarity của positive pair; mẫu số = positive + *tất cả* negative (cùng view lẫn view đối diện — paper gọi là intra-view và inter-view negatives).

- $\theta(u,v) = s(g(u), g(v))$ — cosine similarity sau khi qua projection head $g$ (Eq. 8). Projection head có mặt vì embedding tốt nhất cho contrastive ≠ embedding tốt nhất cho task chính; $g$ cho phép tách hai mục đích.
- Vì loss trên không đối xứng (anchor ở view 1), paper lấy trung bình hai chiều:

$$L_{intra} = \alpha \sum_i \ell_{CL}(u_i, v_i) + (1-\alpha)\sum_i \ell_{CL}(v_i, u_i), \quad \alpha = 0.5 \qquad (9)$$

> **Chỗ code khác paper** (xem thêm Chương 9): `CL_HGCN.sim` tính mẫu số = `refl_sim.sum(1) + between_sim.sum(1) − refl_sim.diag()` — tức negative = *mọi* node cùng view + *mọi* node khác view (kể cả positive một lần). Khớp Eq. 7. Nhưng projection head trong code là `ELU → Linear` không normalize output cuối; và anchor $u_i$/$v_i$ trong code lấy trực tiếp `z1`/`z2` qua `projection()`, không qua HGT.

## 4.2. Cross-modality CL (Eq. 10–15) — "miRNA và disease liên kết phải gần nhau"

Bây giờ không còn so 2 view *cùng* modality nữa, mà so **miRNA với disease**. Input là embedding đã fused: $Z_M \in \mathbb{R}^{n_m\times d}$, $Z_D \in \mathbb{R}^{n_d\times d}$ (cách fuse xem Chương 5).

Similarity chuẩn hóa:

$$S_{cross} = \frac{\hat Z_M \hat Z_D^T}{t}, \quad \hat Z_M = Z_M/\|Z_M\|_2,\ \hat Z_D = Z_D/\|Z_D\|_2 \qquad (10-12)$$

$S_{cross} \in \mathbb{R}^{n_m \times n_d}$ — mỗi ô là độ giống miRNA–disease.

Hai thành phần loss:

**InfoNCE** (Eq. 13) — với miRNA $i$, positive set $P_i = \{j: A_{ij}=1\}$, negative set $N_i$ = các pair được sample:

$$L^{InfoNCE}_{inter} = -\sum_i \sum_{j\in P_i} \log \frac{\exp(S_{cross,ij})}{\exp(S_{cross,ij}) + \sum_{k\in N_i}\exp(S_{cross,ik})}$$

→ muốn disease "đúng" nhận similarity cao hơn hẳn mọi disease negative.

**Margin ranking loss** (Eq. 14) — bảo đảm *khoảng cách* giữa mean positive similarity và mean negative similarity ít nhất $m$:

$$L^{Margin}_{inter} = \max\big(0,\; m - (\bar S_P - \bar S_N)\big)$$

→ InfoNCE kéo/đẩy tương đối; margin ép thêm chênh lệch tuyệt đối tối thiểu, giúp tách biệt rõ hơn khi data imbalance.

Tổng hợp: $L_{inter} = L^{InfoNCE}_{inter} + \lambda\,L^{Margin}_{inter}$ (Eq. 15; trong code hệ số margin hardcode `0.1`).

> Trong code: `InterViewContrastiveLoss` (hetero_model.py:33–138) implement đúng 2 thành phần này, tính InfoNCE trên toàn bộ negative của hàng thay vì sampled subset, rồi cộng `0.1 × margin_loss`. Loss này được cộng vào cả `mi_cl_loss` và `dis_cl_loss` với hệ số `inter_view_weight = 0.3` (≈ λ₂ của paper — paper chọn 0.3 qua grid search, xem Chương 8).

## 4.3. Vì sao hai tầng?

Trực giác thiết kế của paper: trước khi căn chỉnh miRNA↔disease (cross-modality, khó — hai không gian khác nhau), phải chắc chắn mỗi entity đã có embedding "sạch" và ổn định trong chính modality của nó (intra-modality). Intra lọc nhiễu riêng của view; inter căn hai modality về cùng hệ tọa độ để predictor so sánh được.

---

**Chương tiếp**: [05 — View fusion, reconstruction & dynamic update](05-fusion-reconstruction-dynamic-update.md) — 2 view hợp lại thành 1 embedding, và vòng lặp "đồ thị tự cải thiện".
