# Kế hoạch qua đêm: chẩn đoán và thí nghiệm DHGCMDA v3.2

Mục tiêu: đánh giá có bằng chứng vì sao loss cosine ổn định hơn nhưng F1 giảm, và xác định liệu sự khác biệt về trọng số loss hoặc horizon là nguyên nhân có thể kiểm chứng. Không tuyên bố đã tái hiện paper khi dữ liệu tác giả, split và công thức F1 chưa xác định.

1. Kiểm tra lại dữ liệu, provenance, trajectory và dự đoán sẵn có; tính đóng góp từng thành phần objective theo epoch và sai lệch F1 theo lớp cho raw và cosine trên ba seed fold0. Kiểm tra công thức tổng objective, hash artifact và nêu rõ các giới hạn; làm rõ đây là quan sát tương quan.
2. Cố định trước thí nghiệm 2×2: raw/cosine × lambda_recon 1/0,1 với ba seed cùng fold0, 650 epoch. Dùng sáu lượt lambda1 đã có và chạy sáu lượt lambda0,1 mới. Cả bốn góc giữ decoder, đồ thị refresh raw, dữ liệu và objective còn lại; đối chiếu raw1/cosine1 đã pin. Lambda 0,1 là phép can thiệp chẩn đoán, không phải suy ra lambda tác giả; giảm lambda thay đổi trade-off nhiều mục tiêu.
3. Nếu khả thi chạy thêm horizon 1300 cho cosine lambda1 với 3 seed; kiểm tra epoch 650 khớp lượt cosine 650 tương ứng trong dung sai, công bố sự không khớp nếu có. Không so best trên horizon khác nhau như ước lượng độc lập; luôn báo fixed-epoch.
4. Nếu các đối chứng khả dụng và thời gian cho phép, kiểm tra thêm fold khác với phương án khả quan, ghi rõ lựa chọn hậu nghiệm và dùng inner validation, không gọi là kiểm chứng mù. Không tái sử dụng outer-test để chọn hay tuyên bố xác nhận.
5. Tự động ghi nhật ký, hash protocol/script/input, trạng thái từng lượt; giữ lỗi, không xóa kết quả, lập bảng từng seed, lớp, loss, gradient, spike; tự kiểm tra hash và tính lại metric từ prediction. Bảo toàn kết quả và công việc có sẵn trong workspace; không thay code gốc, không commit hoặc gửi thông tin ra ngoài.

Tiêu chí: mọi kết luận bám vào kết quả thực, báo đầy đủ những lượt lỗi hoặc chưa chạy; không kết luận về nguồn dữ liệu v3.2 hay công thức F1 0,8600 nếu chưa có bằng chứng tác giả. Điểm validation đã dùng phát triển; seed cùng fold không độc lập. Ghi rõ mọi thay đổi protocol sau khi bắt đầu và không che giấu kết quả không thuận lợi.
