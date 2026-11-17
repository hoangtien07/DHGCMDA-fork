# Giới hạn negative bank 10 so với 1: validation 5 fold × 3 seed

Trạng thái: **PASS**; đã xác minh 15/15 cặp.

Đây là thí nghiệm hậu nghiệm, inner-validation đã dùng phát triển; không phải 15 quan sát độc lập, test mù hoặc tái hiện chính xác paper. Cap1 thay cả số negatives và objective, không chứng minh riêng tác động false-negative.

| Fold | Seed | Macro best cap10 | Macro best cap1 | Macro epoch650 cap10 | Macro epoch650 cap1 | Spike cap10 / cap1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1234 | 0.4708 | 0.4282 | 0.4708 | 0.4231 | 1 / 1 |
| 0 | 42 | 0.4474 | 0.4477 | 0.4224 | 0.4112 | 1 / 2 |
| 0 | 2025 | 0.4711 | 0.4512 | 0.4335 | 0.4213 | 2 / 1 |
| 1 | 1234 | 0.4333 | 0.4343 | 0.4289 | 0.4106 | 1 / 1 |
| 1 | 42 | 0.4167 | 0.4034 | 0.4097 | 0.3457 | 2 / 3 |
| 1 | 2025 | 0.4086 | 0.4140 | 0.3981 | 0.4064 | 2 / 2 |
| 2 | 1234 | 0.4533 | 0.4531 | 0.4413 | 0.4349 | 2 / 1 |
| 2 | 42 | 0.4394 | 0.4323 | 0.4023 | 0.4035 | 2 / 2 |
| 2 | 2025 | 0.4422 | 0.4337 | 0.3483 | 0.4171 | 1 / 2 |
| 3 | 1234 | 0.4474 | 0.4391 | 0.4242 | 0.3764 | 1 / 1 |
| 3 | 42 | 0.4157 | 0.4405 | 0.4131 | 0.4211 | 1 / 1 |
| 3 | 2025 | 0.4444 | 0.4186 | 0.4088 | 0.4186 | 1 / 2 |
| 4 | 1234 | 0.4586 | 0.4567 | 0.4413 | 0.4087 | 1 / 1 |
| 4 | 42 | 0.4625 | 0.4735 | 0.3827 | 0.4729 | 1 / 2 |
| 4 | 2025 | 0.4293 | 0.4288 | 0.4293 | 0.4004 | 1 / 1 |

Trung bình mô tả trên các ô đã xong (không phải CI):

| Arm | Best macro | Best micro | Best any-hit | Final macro | Final micro | Final any-hit |
|---|---:|---:|---:|---:|---:|---:|
| raw10 | 0.4427 | 0.4728 | 0.5532 | 0.4170 | 0.4451 | 0.5207 |
| raw1 | 0.4370 | 0.4656 | 0.5448 | 0.4115 | 0.4381 | 0.5126 |

Kết quả best được chọn trên cùng validation; final là mốc cố định 650. Per-class, max gradient, hash và những ô lỗi/chưa hoàn tất trong `results.json`. `preflight.json` khóa nguồn, bundle, seed và hash trước fit. Không dùng outer-test trong thí nghiệm này.

Các ô chưa hoàn thành/kiểm tra lỗi: không có
