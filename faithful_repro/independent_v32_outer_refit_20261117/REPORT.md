# U1 outer-refit: dynamic vs static, 5 fold × 3 seed

**Tiến độ plan:** Hoàn tất refit và chấm 30 checkpoint theo protocol nội bộ; còn xác minh dữ liệu/metric gốc để kết luận tái hiện paper.

Đây là đánh giá sau phát triển, không phải thử nghiệm mù: fold0 từng có outer-test được xem, các cặp và kết quả đã góp mặt trong lịch sử phát triển. Dữ liệu, features và metric là bản thích nghi, chưa đồng nhất paper. Cả hai arm refit toàn bộ outer-train 650 epoch, không validation, và đánh giá checkpoint cuối.

| Fold | Seed | Dynamic macro | Static macro | Product prior | Disease prior |
|---:|---:|---:|---:|---:|---:|
| 0 | 1234 | 0.3536 | 0.4458 | 0.4555 | 0.3701 |
| 0 | 42 | 0.4299 | 0.4490 | 0.4555 | 0.3701 |
| 0 | 2025 | 0.4186 | 0.4367 | 0.4555 | 0.3701 |
| 1 | 1234 | 0.3970 | 0.4291 | 0.4197 | 0.3613 |
| 1 | 42 | 0.3666 | 0.4286 | 0.4197 | 0.3613 |
| 1 | 2025 | 0.4141 | 0.4342 | 0.4197 | 0.3613 |
| 2 | 1234 | 0.4553 | 0.4430 | 0.4388 | 0.3585 |
| 2 | 42 | 0.3685 | 0.4468 | 0.4388 | 0.3585 |
| 2 | 2025 | 0.4292 | 0.4469 | 0.4388 | 0.3585 |
| 3 | 1234 | 0.4198 | 0.4365 | 0.4245 | 0.3574 |
| 3 | 42 | 0.4135 | 0.4314 | 0.4245 | 0.3574 |
| 3 | 2025 | 0.3807 | 0.4220 | 0.4245 | 0.3574 |
| 4 | 1234 | 0.4063 | 0.4436 | 0.4093 | 0.3682 |
| 4 | 42 | 0.4004 | 0.4719 | 0.4093 | 0.3682 |
| 4 | 2025 | 0.4134 | 0.4388 | 0.4093 | 0.3682 |

| Seed | Dynamic mean ± SD qua fold | Static mean ± SD qua fold | Mean dynamic − static |
|---:|---:|---:|---:|
| 1234 | 0.4064 ± 0.0369 | 0.4396 ± 0.0068 | -0.0332 |
| 42 | 0.3958 ± 0.0278 | 0.4455 ± 0.0173 | -0.0498 |
| 2025 | 0.4112 ± 0.0182 | 0.4357 ± 0.0090 | -0.0245 |

Các SD là mô tả theo fold, không phải khoảng tin cậy; các ô chia sẻ dữ liệu và seed thay cả negative bank. Xem `evaluation/results.json` để biết micro-F1, any-hit, per-class và tất cả baseline; `verify_eval.py` đối chiếu truth, hash, công thức và scikit-learn.

Metric primary là argmax-onehot so với truth multi-hot 5 loại; chưa chứng minh tương đương F1 0,8600. Kết quả không xác nhận tái hiện paper. Dynamic/static chỉ là ablation của pipeline nội bộ; chưa xác định cấu hình nguồn tác giả.
