# Kế hoạch đối chứng che association theo fold

Mục tiêu: đo chênh lệch so với lượt chạy v2.0 mã phát hành trên cùng fold, seed và cấu hình; không khẳng định sạch hoàn toàn khi chưa biết nguồn gốc đặc trưng có sẵn.

1. Kiểm thử hàm dựng ma trận chỉ từ các chỉ số train, không thay dữ liệu gốc và không thay khi sửa nhãn ngoài tập train; kiểm tra fold test không giao train.
2. Bọc dữ liệu ngay trước hàm train gốc: trường 4 là ma trận chỉ chứa train; trường 5 vẫn là truth dùng trong hàm test gốc. Tính lại ID/IM từ ma trận train; giữ các file đầu vào khác, hash và nguồn mã. Kiểm tra không có nhãn test tại trường 4, graph xây từ trường 4, đầu vào train không thay khi thay nhãn test.
3. Smoke 5 fold 1 epoch; so khớp chỉ số với fold đã lưu, nhãn gốc, gradient và lỗi fallback; sau đó full 5 × 650 nếu smoke thành công. Lưu điểm thô cùng metric tác giả và đa nhãn.
4. Đối chiếu từng fold và trung bình với baseline; giới hạn: M_FSM và các đặc trưng tĩnh có provenance chưa rõ, kết quả không tương đương CV paper và khác biệt không chứng minh quan hệ nhân quả riêng cho type leakage.
