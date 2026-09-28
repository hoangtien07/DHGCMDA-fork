# Chương 9 — Paper vs code trong repo này (bản đồ + các chỗ khác nhau + kết quả reproduce)

> Chương quan trọng nhất nếu bạn định chạy lại / reproduce. Repo này là **fork đã được audit sâu** (xem `CLAUDE.md`, `EXPERIMENT_STATE.md`, `forensics/`). Phần dưới tóm tắt bản đồ Eq→code rồi liệt kê các chỗ *code làm khác paper viết* và kết quả reproduce thực tế.

---

## 9.1. Bản đồ công thức → code

| Paper | Khối | Code |
|-------|------|------|
| §2.2 Eq. 1–2 | Augmented feature `[S‖A]` | `main_experiments_hetero1.py:905-926` (`torch.cat([association_matrix, m_ss])`...) — **thứ tự cột ngược** `[A‖S]`, tương đương |
| §2.3 Eq. 3 | Incidence $H$ bằng KNN | `hypergraph_construct_KNN.construct_H_with_KNN*` — **dùng Euclidean distance**, paper nói cosine |
| §2.3 Eq. 4 | $G = D_v^{-1/2}HWD_e^{-1}H^TD_v^{-1/2}$ | `hypergraph_construct_KNN._generate_G_from_H` — đúng công thức |
| §2.4 Eq. 5 | $X^{(l+1)}=\sigma(GX^{(l)}\Theta)$ | `HGNN_conv` + `HGCN` (1 layer, LeakyReLU 0.25) |
| §2.4.1 Eq. 6–9 | Intra-modal NT-Xent, α=0.5 | `CL_HGCN.sim` — gần khớp (mẫu số = mọi node cả 2 view) |
| §2.4.2 Eq. 10–15 | Inter-modal InfoNCE + margin | `InterViewContrastiveLoss` — margin weight hardcode 0.1 |
| §2.5 Eq. 16–18 | Attention fusion GAP+MLP+β | **`HGCN_Attention_Mechanism` KHÔNG phải attention** — chỉ `0.6·Z₁+0.4·Z₂` cố định |
| §2.6 Eq. 19–21 | Reconstruct $S$, loss Frobenius | `SimpleHypergraphDecoder` + `F.mse_loss` — decoder có **normalize hàng** (cosine), target chỉ 1 view (FSM/SSM) |
| §2.7 Eq. 22–29 | HGT 2 layer, 4 head | `EnhancedHGTLayer` bọc `HGTConv` PyG; `node_transformers` = chiếu theo node type |
| §2.7 | Bilinear predictor | `SimplifiedTypePredictor` — mặc định **diag** `(m⊙r_t)·d` (rank d); `--predictor_mode full_bilinear` = $m^TW_td$ (rank d², gần paper hơn) |
| §2.8 Eq. 30–31 | Update đồ thị mỗi 5 epoch, θ=0.5 | `main_...:990-995` rebuild **chỉ heterogeneous graph**; 4 hypergraph Laplacian KHÔNG rebuild |
| §2.9 Eq. 32 | $L_{type}+\lambda_1L_{intra}+\lambda_2L_{inter}+\lambda_3L_{recon}$ | `main_...:1039` + thêm `1e-4·L2reg`; `recover_loss` mặc định = 2-head `0.3·focal+0.7·wCE` (xem `--loss_mode`, `paper_literal` = Eq. 32 gần literal) |
| §2.9 Eq. 33–34 | Weighted CE + effective number | `SimplifiedMultiTypeAssociationLoss` — effective number β=0.99999 + **label smoothing 0.1 + focal γ=2** (trick ngoài paper) |

## 9.2. Các khác biệt đáng nhớ (code ≠ paper)

Nhóm theo mức độ ảnh hưởng đến kết quả reproduce (tổng hợp từ CLAUDE.md Plan A→K):

1. **Predictor**: paper nói "bilinear"; code gốc = diagonal (rank-d). Fork thêm `full_bilinear` → v2.0 Top-1 F1 **0.6350, vượt paper 0.5970 (+6.4%)** → diag là rewrite kém trung thực.
2. **Loss**: paper Eq. 32 chỉ CE; code thêm existence head (w=0.3), focal, class weights, label smoothing. Fix `exist_weight≈0.1` đã *đóng gap* v2.0 (F1 0.5996 vs paper 0.5970).
3. **AVF**: "attention" thực chất weighted sum cố định 0.6/0.4 — Eq. 16–18 không implement.
4. **Dynamic update**: chỉ update heterogeneous graph của HGT; hypergraph cho HGCN giữ cố định (paper nói cả hypergraph được rebuild bằng KNN).
5. **KNN distance**: code dùng Euclidean trên feature concat; paper nói cosine.
6. **Metric Top-1 của chính tác giả có bug với v3.2**: `Calculate_Metrics.compute_top1_metrics` hardcode 4 types → bỏ hết mẫu 5-type (type 5 = Tissue) và bỏ pred vector length-6 → v3.2 ra F1 = 0.0 **bất kể model tốt**. Đây là phát hiện "smoking gun" của Plan K: paper Table 3 (v3.2: F1 0.86) đo bằng eval code *không public*; code release không chấm được dataset của chính họ.

## 9.3. Kết quả reproduce thực tế (trên fork này)

| Mục tiêu | Paper | Reproduce | Đánh giá |
|----------|-------|-----------|----------|
| v2.0 binary (CVtriplet AUC/AUPR/F1) | 0.9669 / 0.9738 / 0.9278 | ~0.97-0.98 | ✅ Khớp/vượt |
| v2.0 Top-1 F1 (CVtype) | 0.5970 | 0.5996–0.6350 (sau fix predictor + exist_weight) | ✅ Khớp/vượt |
| v2.0 ablation Fig. 4 | mọi component "critical" | **pattern ngược** — bỏ CL/HGT *tăng* F1, verify multi-seed + rebuild | ❌ Không khớp (finding thật, không phải noise) |
| v3.2 Top-1 F1 | 0.8600 | ~0.27–0.36 (metric đúng) | ❌ Gap lớn — chủ yếu do **data curation 411×271 chưa công bố** (raw 1049×758 → paper filter ra 411×271, density 10.5% vs public-preprocess 3.9%) |
| v3.2 binary | AUC 0.9181 | ~0.92 | ✅ Khớp |
| TDRC baseline (kiểm chứng chéo) | F1 0.4207 | 0.4378 | ✅ Khớp → pipeline của ta đúng |

**Kết luận của fork** (Plan I): trần reproduce không-có-author ≈ 62–68%; v3.2 Top-1 cần data curation của tác giả (không public) → gap còn lại là *non-reproducibility thật*, không phải lỗi implementation. Chi tiết: `CLAUDE.md` §7–12, `forensics/REPRO_LEDGER.md`.

## 9.4. Checklist tự chạy nhanh

```bash
# env (xem requirements.txt — pin torch 2.5.1 +cpu, py3.12)
python check_v2.0_495m383D.py                                # verify data
python main_experiments_hetero1.py --device cpu \
    --epoch 3 --validation 2                                 # smoke test ~phút
python main_experiments_hetero1.py --device cpu              # full 5-fold ~50' CPU
python main_experiments_hetero1.py --ablation no_cl ...      # 1 variant ablation
```

Sau khi chạy: `python parse_metrics.py logs/<file>.log results/<out>.json` để parse metric.

---

**Hết series.** Quay lại [mục lục](README.md) hoặc đọc sâu code ở [../NOTES_MODEL.md](../NOTES_MODEL.md), [../ARCHITECTURE.md](../ARCHITECTURE.md).
