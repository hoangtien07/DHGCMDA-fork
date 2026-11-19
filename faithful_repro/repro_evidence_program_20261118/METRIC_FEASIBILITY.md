# METRIC_FEASIBILITY — có thể hay không R=0.9421 trên tensor này (REV4 WP-A0)

Fixture chạy được: `metric_feasibility_check.py` (PASS, deterministic).

## Setup

- 8,735 cặp dương, mỗi cặp đúng 1 dự đoán Top-1.
- Truth multi-hot 5-type: 11,748 nhãn dương, support
  Nk = [1155, 403, 2293, 3997, 3900].

## Oracle bounds (chứng minh bằng code)

| Convention | Bound | Paper R=0.9421 khả thi? |
|---|---:|---|
| macro-recall 5-type multi-hot (mean_k hits_k/Nk, oracle gán mỗi cặp cho loại hiếm nhất nó chứa — tối ưu vì hits_k ≤ Nk luôn) | **0.8263** | KHÔNG |
| council rough bound (1 − (11748−8735)/(5·3997)) | 0.8492 | KHÔNG |
| micro-recall (hits/labels ≤ 8735/11748) | 0.7435 | KHÔNG |
| scalar-collapse truth (1 nhãn/cặp, 8,735 nhãn) | ≤ 1.0 | CÓ THỂ |
| any-hit (cặp trúng ≥1 loại) | ≤ 1.0 | CÓ THỂ |

## Kết luận

R = 0.9421 của paper **không thể** là pooled 5-type multi-hot
macro-recall hay micro-recall trên tensor này với 1 dự đoán/cặp —
kể cả oracle hoàn hảo cũng chỉ đạt 0.8263. Convention của paper
phải là một trong:

1. scalar-collapsed truth (mỗi cặp 1 nhãn đại diện, vd. first/last
   positive hoặc nhãn "chính") — với đó recall ≤ 1 khả thi;
2. any-hit/pair-hit convention (R = tỷ lệ cặp trúng ≥1 loại);
3. đơn vị đánh giá khác pair (vd. theo triplet, hay lọc subset);
4. trung bình per-fold với denominator khác (fold-wise truth
   khác pooled).

Đây là đặc trưng hóa metric semantics — KHÔNG bác bỏ số paper;
với scalar-collapse (cách author code đọc truth: scalar 1..4)
0.9421 hoàn toàn khả thi về mặt toán học. Việc phân biệt các khả
năng đòi hỏi eval script gốc (mục 3 trong author request).

## Chú ý tương quan

Nếu paper dùng scalar-collapse/any-hit, thì kết quả "đúng metric"
của ta (multi-hot 5-type) và số của họ không cùng denominator —
so sánh trực tiếp 0.49 vs 0.86 là không hợp lệ ở cả hai chiều.
Đây là cấu trúc bất khả so sánh G3, độc lập với G1.
