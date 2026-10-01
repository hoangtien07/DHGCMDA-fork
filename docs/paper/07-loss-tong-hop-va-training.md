# Chương 7 — Loss tổng hợp & vòng lặp training

> Paper §2.9 (Eq. 32–34) + ghép mọi chương trước thành pipeline hoàn chỉnh.

---

## 7.1. Unified training objective (Eq. 32)

$$L_{total} = L_{type} + \lambda_1 L_{intra} + \lambda_2 L_{inter} + \lambda_3 L_{recon}$$

Bốn thành phần, mỗi cái đã gặp:

| Term | Vai trò | Nguồn | λ |
|------|---------|-------|---|
| $L_{type}$ | **Loss chính**: dự đoán đúng association type | Chương 6 (predictor) | — |
| $L_{intra}$ | Consistency 2 view cùng modality | Chương 4.1 | $\lambda_1$ = 1.0 |
| $L_{inter}$ | Alignment miRNA↔disease | Chương 4.2 | $\lambda_2$ = **0.3** (tuned, Fig. 2a) |
| $L_{recon}$ | Giữ cấu trúc similarity gốc | Chương 5.2 | $\lambda_3$ = 1.0 |

Paper cố tình chỉ tune $\lambda_2$ (vì nó ảnh hưởng trực tiếp cross-modality alignment), giữ $\lambda_1=\lambda_3=1$ để hạn chế số hyperparameter trên dataset nhỏ.

## 7.2. Type loss với class weighting (Eq. 33–34)

$$L_{type} = -\sum_{(i,j)\in F} \sum_{k=1}^{C} w_k\, y^k_{ij} \log \hat p^k_{ij} \qquad (33)$$

Weighted cross-entropy trên tập pair $F$, với class weight theo công thức **effective number**:

$$w_k = \frac{1-\beta}{1-\beta^{n_k}} \qquad (34)$$

$n_k$ = số mẫu thuộc type $k$, $\beta \to 1$ (đủ gần để minority class được boost mạnh). Công thức từ paper "Class-Balanced Loss" (Cui et al. 2019): type ít mẫu ($n_k$ nhỏ, vd Epigenetics 3.4% ở v3.2) → mẫu số nhỏ → $w_k$ lớn → lỗi trên minority bị phạt nặng hơn.

## 7.3. Ghép lại: 1 fold training diễn ra thế nào

```
prepareData: load 4 similarity + A; split 5-fold (positive + sampled negative)

Chuẩn bị (một lần mỗi fold):
  X_view = [S_view ‖ A] cho cả 4 view                     (Ch.2, Eq.1-2)
  G_view = KNN hypergraph Laplacian cho cả 4 view         (Ch.2-3, Eq.3-4, K=13)
  hetero_data = heterogeneous graph 4 edge types          (Ch.6.1)

Mỗi epoch (tối đa 650):
  forward → score[n_m, n_d, 1+C], L_intra (mi+dis), L_inter, Ŝ_M, Ŝ_D
  nếu epoch % 5 == 0: rebuild similarity edges từ Ŝ (θ=0.5) (Ch.5.3, Eq.30-31)
  L_total = L_type + λ₁L_intra + λ₂L_inter + λ₃L_recon    (Eq. 32)
  backward + grad clip + Adam step (lr = 1e-4)

Cuối fold: eval test → AUC/AUPR/F1 (CVtriplet) + Top-1 P/R/F1 (CVtype)  (Ch.8)
```

> Trong code thực tế, total loss (`main_experiments_hetero1.py:1039`) = `recover + 1.0·(mi_cl + dis_cl) + 1.0·recon + 1e-4·L2reg`. Điểm khác paper: `recover` mặc định là **2-head** `0.3·exist_focal + 0.7·type_wCE` thay vì single CE của Eq. 33 (xem `--loss_mode` trong param.py: `two_head`/`softmax_5class`/`paper_literal`/`multilabel_bce`), và có thêm focal loss (γ=2), label smoothing 0.1, negative sampling 10×, L2 regularization — các "trick" không nằm trong Eq. 32. Báo cáo reproduce trong repo (CLAUDE.md M5/M6) đánh giá từng chênh lệch này.

## 7.4. Bảng tra cứu hyperparameter

| Param | Giá trị paper | Vai trò | Trong code |
|-------|---------------|---------|------------|
| Epoch | — (early stopping) | — | `--epoch 650` |
| lr | Adam, scheduling | — | `--lr 1e-4` |
| K (KNN hyperedge) | **13** (Fig. 3) | kích thước hyperedge | `--K_neigs 13` |
| $t$ (temperature CL) | **0.5** (Fig. 2a) | Eq. 6–7 | `tau=0.5` hardcoded trong `CL_HGCN` |
| λ₂ (inter CL weight) | **0.3** (Fig. 2a) | Eq. 32 | `--inter_view_weight 0.3` |
| λ₁, λ₃ | 1.0 | Eq. 32 | hệ số 1.0 trong total loss |
| HGT layers / heads | 2 / 4 | §2.7 | `--nlayer 2`, `--n_head 4` |
| θ (dynamic update) | 0.5, mỗi 5 epoch | Eq. 30–31 | `--similarity_threshold 0.5`, `--update_graph_frequency 5` |
| α (Eq. 9) | 0.5 | hai chiều intra-CL | `--alpha 0.5` |

## 7.5. Negative sampling — điểm paper ít nói nhưng code phải làm

Vì $A_{ij}=0$ không phải negative thật (Chương 1.3), train cần chọn ra tập "unknown đóng vai negative". Code dùng tỉ lệ **10 negative : 1 positive** mỗi fold (sample ngẫu nhiên từ các ô 0). Con số này *không được paper ghi rõ* — một trong những chỗ underspec ảnh hưởng reproduce.

---

**Chương tiếp**: [08 — Đánh giá & kết quả](08-danh-gia-va-ket-qua.md) — CVtype vs CVtriplet, vì sao DHGCMDA vượt baseline, và 4 limitation paper tự thừa nhận.
