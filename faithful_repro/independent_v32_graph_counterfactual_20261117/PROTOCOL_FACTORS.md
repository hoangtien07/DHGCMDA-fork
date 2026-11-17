# Phân tách thăm dò G và cạnh tại snapshot epoch 226

Ngày 17/11/2026. Sau khi đã xem kết quả đối chứng old/new, so sánh đủ 4 tổ hợp G cũ/mới × cạnh cũ/mới, cùng snapshot, train mode, target và RNG; không optimizer, không outer test. Đây là phân tích hậu nghiệm tại một state không có spike gốc. Dùng run_pair, lặp nhánh mới với dung sai đã định trong PROTOCOL_RETRY2. Kiểm tra nhánh old/old và new/new tái hiện loss đã ghi trong retry2 với sai lệch tương đối <=1e-5. Tổ hợp trộn có thể không ứng với đồ thị từng xuất hiện trong huấn luyện; không suy ra ảnh hưởng lên F1 hoặc spike gốc.
