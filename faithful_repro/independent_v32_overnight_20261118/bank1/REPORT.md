# Giới hạn negative bank 10 so với 1: validation 5 fold × 3 seed

Trạng thái: **PARTIAL**; đã xác minh 2/15 cặp.

Đây là thí nghiệm hậu nghiệm, inner-validation đã dùng phát triển; không phải 15 quan sát độc lập, test mù hoặc tái hiện chính xác paper. Cap1 thay cả số negatives và objective, không chứng minh riêng tác động false-negative.

| Fold | Seed | Macro best cap10 | Macro best cap1 | Macro epoch650 cap10 | Macro epoch650 cap1 | Spike cap10 / cap1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1234 | 0.4708 | 0.4384 | 0.4708 | 0.4160 | 1 / 1 |
| 0 | 42 | 0.4474 | 0.4705 | 0.4224 | 0.4492 | 1 / 1 |

Trung bình mô tả trên các ô đã xong (không phải CI):

| Arm | Best macro | Best micro | Best any-hit | Final macro | Final micro | Final any-hit |
|---|---:|---:|---:|---:|---:|---:|
| raw10 | 0.4591 | 0.4929 | 0.5770 | 0.4466 | 0.4751 | 0.5561 |
| raw1 | 0.4544 | 0.4813 | 0.5633 | 0.4326 | 0.4573 | 0.5353 |

Kết quả best được chọn trên cùng validation; final là mốc cố định 650. Per-class, max gradient, hash và những ô lỗi/chưa hoàn tất trong `results.json`. `preflight.json` khóa nguồn, bundle, seed và hash trước fit. Không dùng outer-test trong thí nghiệm này.

Các ô chưa hoàn thành/kiểm tra lỗi: [{"key": "fold0_seed2025", "state": "not_completed"}, {"key": "fold1_seed1234", "state": "not_completed"}, {"key": "fold1_seed42", "state": "not_completed"}, {"key": "fold1_seed2025", "state": "not_completed"}, {"key": "fold2_seed1234", "state": "not_completed"}, {"key": "fold2_seed42", "state": "not_completed"}, {"key": "fold2_seed2025", "state": "not_completed"}, {"key": "fold3_seed1234", "state": "not_completed"}, {"key": "fold3_seed42", "state": "not_completed"}, {"key": "fold3_seed2025", "state": "not_completed"}, {"key": "fold4_seed1234", "state": "not_completed"}, {"key": "fold4_seed42", "state": "not_completed"}, {"key": "fold4_seed2025", "state": "not_completed"}]
