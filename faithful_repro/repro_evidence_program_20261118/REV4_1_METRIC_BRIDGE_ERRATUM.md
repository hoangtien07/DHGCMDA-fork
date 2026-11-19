# REV4.1_METRIC_BRIDGE_ERRATUM — hai lỗi trong metric_bridge.py, đã sửa

Regression fixtures: `test_metric_bridge_regressions.py` — 5/5 PASS
(mỗi lỗi có fixture bắt riêng).

## Lỗi 1 — `multitype_anyhit`

**Bug:** so `pred[multi, None]` (class id 0–4) với ma trận nhị phân
`np.where(y>0,1,0)` — so class ID với nhãn 0/1.
**Fix:** `y[i, pred[i]] > 0` per row.
**Hệ quả:** trước khi sửa, giá trị `multitype_anyhit` trong
METRIC_BRIDGE.json là sai ngẫu nhiên (so 0–4 với 0/1 → hầu như luôn
False ở cột lớp ≥2 và true giả ở cột 0–1). Giá trị đúng (mean 15 ô):

| Arm | multitype_anyhit (đúng) | singletype_acc (đúng) |
|---|---:|---:|
| U1 dynamic | 0.6136 | 0.4769 |
| U1 static | 0.6557 | 0.5182 |
| prior product | **0.8303** | **0.5678** |

**Claim bị đảo:** `NEXT_RESEARCH_DECISION.md` viết "U1 thắng priors
rõ trên `multitype_anyhit`/`singletype_acc` strata" — **RETRACTED**:
đúng ra prior product thắng U1 ở CẢ hai strata. Lợi thế còn lại của
U1 chỉ là macro-F1 aggregate (0.4403 vs 0.4296) và K5_firstpos
harmonic (0.4924 vs 0.4480). Track N không thể giả định mô hình họ
U1 mạnh ở multi-type pairs.

## Lỗi 2 — `K4filtered_F1`

**Bug:** recall tính trung bình trên `range(5)` trong khi chỉ có
4 lớp → lớp thứ 5 đóng góp 0.0, deflate R ~20%.
**Fix:** trung bình trên `range(4)`; fixture perfect-predictor →
F1 = 1.0. Đổi tên ghi chú: **KHÔNG phải metric chính thức tác giả**,
chỉ là approximation 4-class gần nhất chấm được trên v3.2.
**Hệ quả:** K4filtered_F1 tăng sau sửa:

| Arm | Trước (buggy) | Sau (đúng) |
|---|---:|---:|
| U1 dynamic | 0.4607 | 0.5178 |
| U1 static | 0.4936 | 0.5551 |
| prior product | 0.4689 | 0.5336 |

## Kiểm chứng metric lịch sử không đổi

- `K5_firstpos_*`, `K5_lastpos_*`, `anyhit_*`, `multihot_*`,
  `anyhit_rate`, `n_multitype_pairs`, `singletype_acc` giống hệt
  bản cũ (recompute diff = 0).
- WP4 `RESCORE_VERIFICATION.json` 50/50 vẫn PASS — không liên quan
  hai hàm lỗi.
- `metric_contracts.py` không bị chạm — bug chỉ nằm trong
  `metric_bridge.py` (file mới của REV4).

## File ghi erratum trỏ đến

NEXT_RESEARCH_DECISION.md (claim strata), METRIC_BRIDGE_REPORT.md
(bảng + đọc kết quả) — đánh dấu bằng note trỏ file này.
