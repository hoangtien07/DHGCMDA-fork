# U1 v3.2: 5 fold × 3 seed trên inner-validation

**Tiến độ plan:** Đã hoàn tất và kiểm tra 15 lượt validation; còn đánh giá outer-test/ablation theo protocol cố định và xác minh nguồn tác giả.

Các lượt seed 1234 đã có trước; thêm seed 42 và 2025 trên fold 1–4, cùng model/negative-bank seed, horizon 650 và bundle selection. Đây là validation đã dùng chọn checkpoint và phát triển pipeline, không phải 15 test độc lập, không phải kết quả tương đương paper. Một phần seed 1234 chạy Linux/Colab; các lượt mới chạy Windows CPU.

| Fold | Seed 1234 | Seed 42 | Seed 2025 | Product prior | Disease prior |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.4708 | 0.4474 | 0.4711 | 0.4156 | 0.3533 |
| 1 | 0.4333 | 0.4167 | 0.4086 | 0.3960 | 0.3349 |
| 2 | 0.4533 | 0.4394 | 0.4422 | 0.4439 | 0.3800 |
| 3 | 0.4474 | 0.4157 | 0.4444 | 0.4131 | 0.3661 |
| 4 | 0.4586 | 0.4625 | 0.4293 | 0.4574 | 0.3429 |

Bảng ghi macro-F1 của checkpoint tốt nhất trên chính validation; prior được fit từ train và cố định theo fold. Mỗi lượt đánh giá 65 lần, nên best là thống kê có thiên lệch lựa chọn.

| Seed | Best macro-F1: mean ± SD theo 5 fold | Mean macro-F1 tại epoch 650 |
|---:|---:|---:|
| 1234 | 0.4527 ± 0.0138 | 0.4413 |
| 42 | 0.4364 ± 0.0202 | 0.4060 |
| 2025 | 0.4391 ± 0.0228 | 0.4036 |

Trung bình mô tả trên 15 ô: U1 best macro 0.4427, micro 0.4728, any-hit 0.5532; product prior tương ứng 0.4252, 0.5252, 0.6145; disease prior 0.3554, 0.5384, 0.6299. Các ô chia sẻ dữ liệu giữa fold/seed, không dùng làm 15 quan sát độc lập.

Kết quả `results.json` chứa đầy đủ 15 lượt, metric từng lớp và hash; `score.py` đối chiếu manifest, nhãn và pair validation, tính lại metric bằng scikit-learn. Macro được định nghĩa argmax-onehot so với truth multi-hot đủ 5 loại; chưa xác minh tương đương Top-1 F1 paper. Các fold đã tham gia quá trình phát triển và một outer-test fold0 từng được xem; không coi đây là thử nghiệm mù hay chứng minh tái hiện 0,8600.

Còn thiếu: nguồn và fold tác giả, công thức metric gốc, đánh giá outer-test/ablation 5 fold × 3 seed theo protocol cố định và refit được kiểm tra. Kết quả này không thay thế các hạng mục đó.
