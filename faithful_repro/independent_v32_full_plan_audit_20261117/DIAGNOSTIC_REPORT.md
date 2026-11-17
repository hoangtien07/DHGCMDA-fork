# Đối chứng spike trên ba seed cùng fold0

**Tiến độ plan:** Đã hoàn tất và kiểm tra ba seed chẩn đoán; đang bổ sung 8 lượt validation để đủ 5 fold × 3 seed, còn thiếu bằng chứng nguồn tác giả.

Cùng candidate fold0, cùng source/bundle/650 epoch, ba lượt instrumented Windows CPU; seed model và seed negative bank thay cùng nhau, nên không tách hiệu ứng từng nguồn. Ngưỡng spike >2 lần được quyết định trước các lượt này nhưng bắt nguồn từ quan sát hồi cứu. Cùng state/RNG, không optimizer trong hai nhánh đồ thị:

| Seed | Epoch | Loss G/cạnh cũ → mới | Reconstruction đóng góp cũ → mới | Norm gradient cũ → mới | Số hàng hòa điểm top-k |
|---:|---:|---:|---:|---:|---:|
| 42 | 231 | 5,494 → 143,581 | 0,964 → 138,714 | 14,20 → 4979,06 | 29 |
| 2025 | 221 | 5,488 → 19,547 | 0,930 → 14,931 | 9,88 → 743,83 | 398 |
| 2025 | 481 | 4,925 → 70,990 | 0,811 → 66,726 | 11,62 → 2996,41 | 398 |
| 1234 | 211 | 5,724 → 139,651 | 1,127 → 134,907 | 15,21 → 4309,88 | 13 |

Đồ thị mới trực tiếp làm loss và gradient tăng ở bốn checkpoint; nhánh mới khớp loss thực và phép lặp gradient/RNG đạt. Gram producer, normalized similarity, incidence và G khớp hash dynamic_history. Các lượt cùng fold và có trace: không phải ba fold/test độc lập, không kết luận tần suất phổ quát, tác động lên F1 hoặc hiệu quả của một cách sửa. Chi tiết `three_seed_diagnostics.json`, `verify_seed.py`, các `incidence_*.json`/`paired_*.json` trong ba thư mục seed. Các hiện vật giữ cả control và toàn bộ event vượt ngưỡng.

Kết quả chuỗi seed42 khác đã cho thấy H turnover lớn và mức tập trung cao cũng xảy ra khi không có spike, và hòa điểm không cần thiết trong mọi state. Vì vậy không quy spike riêng cho tie-break hoặc top-k. Trên cơ sở so sánh này, chưa chọn sửa đổi pipeline.
