# Kết quả tái hiện đến K* — 2026-11-24

Mã phát hành 06f9ba3, PyTorch 2.5.1 cu118, GTX 1660 SUPER, seed 1234, 5 fold × 650 epoch.

| Chạy | AUC | AUPR | F1 nhị phân | Top-1 F1 |
| --- | ---: | ---: | ---: | ---: |
| C1, v2.0 | 0.9741 | 0.9675 | 0.9284 | 0.5670 |
| C2, v2.0 | 0.9658 | 0.9593 | 0.9164 | 0.6049 |
| C3, v2.0, seed 42 | 0.9760 | 0.9688 | 0.9343 | 0.5788 |
| K*, v3.2 | 0.8865 | 0.8906 | 0.8079 | 0.3148 |

Top-1 v2.0 là metric của mã phát hành. Top-1 K* là phép mở rộng 5 loại trên nhãn scalar. Chấm lại K* theo TDRC cho Top-1 F1 0.2632. Evaluator phát hành trả Top-1 = 0 với dự đoán 6 chiều.

G2 đạt với sai lệch lớn nhất 0.00062 theo giả thuyết hoán đổi cột AUC/AUPR của Bảng 4. K* chưa đạt tiêu chí tái hiện v3.2. Phán quyết G3 chờ T*; không suy rộng từ K* sang kết quả của T*.

K* sử dụng GIP dựng từ toàn bộ ma trận liên kết và sao chép D_MESH cho view bệnh thứ hai. Đây là giao thức tái hiện, có thông tin test trong đầu vào.

T* đã khởi chạy tự động và đang chạy khi ghi báo cáo. Dự đoán và kết quả fold chỉ được ghi sau khi fold hoàn tất. Không có checkpoint giữa epoch. Nếu tắt máy trước khi fold hoàn tất, phần tiến độ trong fold đó sẽ mất. Chuỗi hiện tại không tự tiếp tục chính xác fold đang dở sau khi khởi động lại.
