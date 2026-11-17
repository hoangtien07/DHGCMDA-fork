# Đối chứng che association và GIP theo fold

Dùng các fold, seed, kiến trúc, loss và số epoch của replay v2.0. Thay ma trận association đầu vào/đồ thị bằng ma trận chỉ có train positives và tính lại mọi GIP có thể truy vết từ ma trận train, bao gồm `M_GSM` binary train-only. GIP chuẩn dùng số node chia tổng bình phương ma trận, như công thức tác giả. Truth đầy đủ chỉ dùng khi chấm test. ID/IM dự phòng tính lại từ train.

So với full binary 5.430 cặp, train-only chỉ chứa 1.072 typed train pairs: đây là thay đổi nguồn thông tin đáng kể, gồm cả các binary-only positives, và không thể quy chênh lệch chỉ cho việc xóa test. Các đặc trưng tĩnh khác có provenance chưa xác minh; weight class dựa toàn bộ dữ liệu và negative sampling theo release giữ cố định. Kết quả là đối chứng chứ không chứng nhận CV sạch hay tương đương paper.

Kiểm thử test-first hàm GIP: biết kết quả bằng tay, nhãn type không ảnh hưởng GIP binary, thay test không ảnh hưởng GIP từ train. Smoke 5 × 1; so fold indices và metric; chỉ chạy 5 × 650 sau smoke. Lưu điểm và log; báo paired per-fold, hai metric, nguồn và giới hạn.
