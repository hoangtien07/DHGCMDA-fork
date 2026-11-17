# Đối chứng trọng số tính theo train

Trên cùng đối chứng v2.0 `independent_v20_fold_gip_20261116`, giữ nguyên mã tác giả, fold, seed, association train-only, GIP train-only, feature, hyperparameters và 5 × 650 epoch. Thay duy nhất buffer class_weights của loss: tính từ số lượng scalar-label tại train positives bằng công thức effective-number beta 0,99999 của tác giả, chuẩn hóa tổng 4. Không thay architecture, loss hoặc sampling. Nếu thiếu lớp trong train thì dừng và báo; không bịa trọng số.

Kiểm tra: test-first công thức, tính bất biến khi sửa test; từng fold assert cặp trùng baseline và in counts/weights; smoke 5 × 1, chấm lại; full 5 × 650 và tính paired delta so với đối chứng GIP. Lưu điểm, hash, log và seed. Kết quả là một seed, không chứng minh loại bỏ mọi rò rỉ: M_FSM và các ma trận tĩnh chưa rõ provenance; negative sampling theo release dùng tập đầy đủ. Không xem kết quả là paper-equivalent.
