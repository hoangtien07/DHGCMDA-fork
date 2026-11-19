# METRIC_BRIDGE_REPORT — cùng frozen outputs, mọi convention có tên (REV4 WP-A2)

Source: `METRIC_BRIDGE.json` (50 rows = 30 model + 20 prior; plus
oracle ceilings per fold). Legacy macro-F1 khớp reported bit-exact
(WP4 RESCORE_VERIFICATION 50/50) — các cột mới chỉ là convention
bổ sung, KHÔNG thay primary metric hậu nghiệm.

## Conventions (mỗi cột một định nghĩa riêng, không bao giờ trộn)

- `K5_firstpos_*`: truth = first positive index, harmonic P/R pooled
- `K5_lastpos_*`: truth = last positive index
- `anyhit_*`: pair đúng nếu argmax(pred) ∈ bất kỳ type thật nào
- `multihot_*`: full 5-type multi-hot
- `K4filtered_*`: bỏ mọi cặp mang nhãn type-5, chấm 4 kênh
  (convention gần release nhất khả thi trên v3.2)
- strata: `n_multitype_pairs`, `multitype_anyhit`,
  `singletype_acc`

## Headline per arm (mean over 15 cells)

| Arm | K5_firstpos F1 | K5_lastpos F1 | anyhit F1 | multi-hot macro | multi-hot micro | K4filtered F1 |
|---|---:|---:|---:|---:|---:|---:|
| U1 dynamic | 0.4558 | 0.4341 | 0.4589 | 0.4045 | 0.4349 | 0.5178 |
| U1 static | 0.4924 | 0.4673 | 0.4936 | 0.4403 | 0.4703 | 0.5551 |
| prior product | 0.4480 | 0.4791 | 0.4680 | 0.4296 | 0.5385 | 0.5336 |

## Oracle ceilings (per fold, METRIC_BRIDGE.json `oracle`)

- macro-recall bound ≈ 0.826 (fold-wise), micro bound ≈ 0.744,
  any-hit bound 1.0 — khớp METRIC_FEASIBILITY.
- Dưới convention scalar-collapse/any-hit, R=0.9421 của paper là khả
  thi; dưới multi-hot pooled, không thể.

## Đọc kết quả

- Thứ hạng U1-static vs prior-product đảo giữa các convention:
  firstpos +0.044 cho U1; lastpos/micro/anyhit prior cao hơn —
  KHÔNG có một "model thắng" duy nhất; phải nêu convention.
- `multitype_anyhit` và `singletype_acc` có sẵn per-row trong JSON
  cho phân tích strata (SAU REV4.1: prior thắng U1 trên cả hai strata —
  xem REV4_1_METRIC_BRIDGE_ERRATUM.md).
- Pair-fold đúng CVtype verified (WP1); triplet-fold chỉ áp cho
  U1X namespace khác (không trộn).


---

> **[ERRATUM REV4.1]** Giá trị `multitype_anyhit`/`K4filtered_F1` trong file này đã được sửa — priors thắng U1 ở cả hai strata, K4filtered tăng ~0.05. Xem `REV4_1_METRIC_BRIDGE_ERRATUM.md`.
