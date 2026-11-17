# Giới hạn negative bank 10 so với 1: validation 5 fold × 3 seed

Trạng thái: **PARTIAL**; đã xác minh 6/15 cặp.

Đây là thí nghiệm hậu nghiệm, inner-validation đã dùng phát triển; không phải 15 quan sát độc lập, test mù hoặc tái hiện chính xác paper. Cap1 thay cả số negatives và objective, không chứng minh riêng tác động false-negative.

| Fold | Seed | Macro best cap10 | Macro best cap1 | Macro epoch650 cap10 | Macro epoch650 cap1 | Spike cap10 / cap1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1234 | 0.4708 | 0.4384 | 0.4708 | 0.4160 | 1 / 1 |
| 0 | 42 | 0.4474 | 0.4705 | 0.4224 | 0.4492 | 1 / 1 |
| 0 | 2025 | 0.4711 | 0.4453 | 0.4335 | 0.4382 | 2 / 1 |
| 1 | 1234 | 0.4333 | 0.4404 | 0.4289 | 0.4404 | 1 / 1 |
| 1 | 42 | 0.4167 | 0.4327 | 0.4097 | 0.3897 | 2 / 1 |
| 1 | 2025 | 0.4086 | 0.4119 | 0.3981 | 0.3370 | 2 / 1 |

Trung bình mô tả trên các ô đã xong (không phải CI):

| Arm | Best macro | Best micro | Best any-hit | Final macro | Final micro | Final any-hit |
|---|---:|---:|---:|---:|---:|---:|
| raw10 | 0.4413 | 0.4769 | 0.5569 | 0.4272 | 0.4610 | 0.5383 |
| raw1 | 0.4399 | 0.4677 | 0.5461 | 0.4118 | 0.4393 | 0.5130 |

Kết quả best được chọn trên cùng validation; final là mốc cố định 650. Per-class, max gradient, hash và những ô lỗi/chưa hoàn tất trong `results.json`. `preflight.json` khóa nguồn, bundle, seed và hash trước fit. Không dùng outer-test trong thí nghiệm này.

Các ô chưa hoàn thành/kiểm tra lỗi: [{"key": "fold2_seed1234", "state": "not_completed"}, {"key": "fold2_seed42", "state": "not_completed"}, {"key": "fold2_seed2025", "state": "not_completed"}, {"key": "fold3_seed1234", "state": "not_completed"}, {"key": "fold3_seed42", "state": "not_completed"}, {"key": "fold3_seed2025", "state": "not_completed"}, {"key": "fold4_seed1234", "state": "not_completed"}, {"key": "fold4_seed42", "state": "not_completed"}, {"key": "fold4_seed2025", "state": "not_completed"}]
