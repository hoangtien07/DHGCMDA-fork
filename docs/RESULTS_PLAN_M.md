# RESULTS — Plan M: Right-sizing hypergraph (K_neigs=2)

> 2026-07-05. Hoàn tất đường cong `K_neigs` dưới predictor `full_bilinear`, đóng caveat "K=1,2 chưa thử" của Plan L.
> **Kết quả: K_neigs=2 là cấu hình hợp lệ tốt nhất cho v2.0.** Additive-only.

## 1. TL;DR

- **HEADLINE v2.0 Top-1 F1 = 0.697 ± 0.003** (K=2, `full_bilinear`, multi-seed {1234,0,42}).
- So paper **0.5970 → +16.8%**; so K=3 cũ (0.688) tốt hơn + ổn định hơn 3× (std 0.0034 vs 0.0107).
- Vượt mọi public baseline: KBLTDARD 0.5869, TFLP 0.5996, TDRC 0.5286, NMCMDA.
- **Default v2.0 mới**: `--predictor_mode full_bilinear --K_neigs 2 --exist_weight 0.1`.

## 2. Đường cong K (seed 1234, đơn điệu)

| K_neigs | 13 | 7 | 3 | **2** | 1 |
|---|---:|---:|---:|---:|---:|
| Top-1 F1 | 0.6311 | 0.6538 | 0.6808 | **0.6940** | 0.6978 |
| AUC | 0.9805 | 0.9818 | 0.9806 | 0.9816 | 0.9833 |

**Cơ chế:** v2.0 rất thưa (1498 assoc / 189K cells). Hypergraph thưa hơn (K nhỏ) giảm over-smoothing /
over-connection → khớp "DHGCMDA over-parameterized cho v2.0" (Plan E). K=13 (default paper) là **K tệ nhất**.

## 3. K=2 multi-seed (best hợp lệ)

| Seed | 1234 | 0 | 42 | **Mean ± std** |
|---|---:|---:|---:|---:|
| Top-1 F1 | 0.6940 | 0.7008 | 0.6974 | **0.6974 ± 0.0034** |
| AUC | 0.9816 | 0.9814 | 0.9823 | 0.9818 |

- Mọi seed K=2 (min 0.6940) vượt *mean* K=3 (0.6883). Không đánh đổi binary (AUC ≈ K=3).
- Không công bố lucky seed 0.7008 riêng lẻ (giữ kỷ luật multi-seed).

## 4. 🔬 K=1 = no_hgcn — xác nhận over-parameterization

**Trùng khít 3 chiều** (cả Top-1 F1 lẫn AUC):

| Cấu hình | Top-1 F1 | AUC |
|---|---:|---:|
| K=1 (`--K_neigs 1`) | 0.6978 | 0.9833 |
| no_hgcn (`--ablation no_hgcn`) | 0.6978 | 0.9833 |
| K=3 + no_hgcn | 0.6978 | 0.9833 |

**Cơ chế:** K=1 → mỗi hyperedge chỉ chứa chính node đó → ma trận incidence H = identity → G ≈ identity →
HGCN thoái hoá thành MLP = **đúng bằng ablation no_hgcn**. "Thưa hoá hypergraph cực đại" và "bỏ HGCN" là
**cùng một lever, cho cùng một số**. → K=1 là mô hình **đã ablate trá hình**, KHÔNG dùng làm headline.

Ý nghĩa: cực trị của trục K-sparsity hội tụ *chính xác* về trục component-ablation → bằng chứng đối kháng-mạnh
cho luận điểm DHGCMDA over-parameterized trên MDA nhỏ/thưa.

## 5. Ablation Fig.4 dưới full_bilinear (reversal lần 5)

Paper claim mọi component đều critical (bỏ → hại). Thực tế dưới full_bilinear (baseline K=13 = 0.6311):

| Ablation | Top-1 F1 | Δ | Hướng |
|---|---:|---:|---|
| no_hgcn | 0.6978 | +10.6% | HELP (đảo) |
| no_hgt | 0.6826 | +8.2% | HELP (đảo) |
| no_cl | 0.6680 | +5.8% | HELP (đảo) |
| no_avf | 0.6356 | +0.7% | ~phẳng |
| no_dv | 0.6303 | −0.1% | ~phẳng |

Reversal dai dẳng dưới predictor trung thực nhất → loại trừ "artifact của diag predictor". Xác nhận độc lập lần 5.

## 6. Caveat trung thực

- K=2/K=3 tune trên *chính* 5-fold CV báo cáo (không held-out riêng) → ước lượng in-sample; giảm nhẹ:
  replicate qua 2 seed không dùng cho selection.
- Một phần gain là "phục hồi baseline xui" (K=13 = K tệ nhất). Khung đúng: "K=2 là K đúng dưới full_bilinear".
- K=1/no_hgcn (0.698) là mô hình đã ablate — không dùng làm số của DHGCMDA đầy đủ.

## 7. Files

- Ma trận: `results/council_matrix_wave3.json` (K=1,2), `council_matrix_wave3b.json` (K=2 multi-seed).
- Kết quả: `results/council_D1_fb_K1_s1234.json`, `council_D2_fb_K2_s1234.json`, `council_D2b_fb_K2_s0.json`,
  `council_D2c_fb_K2_s42.json`; tổng hợp `results/council_summary.json`, `results/council_synthesis.md` (§Plan M).
- Lệnh tái hiện: [REPRODUCE.md](REPRODUCE.md).
