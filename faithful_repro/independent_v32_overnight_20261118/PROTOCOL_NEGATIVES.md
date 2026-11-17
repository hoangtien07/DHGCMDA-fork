# Kiểm tra false negatives trong negative bank trên inner-validation

Không đọc outer-test. Với mỗi fold/seed trong 15 lượt validation U1 đã lưu, dựng lại negative bank từ train_mask và inter_seed bằng mã đã pin, kiểm tra SHA khớp manifest và train overlap = 0. Đếm bao nhiêu cặp validation dương nằm trong bank; báo số cặp, tỷ lệ trong validation, tỷ lệ trong toàn bank và theo kênh type. Một cặp đa nhãn có thể tính vào nhiều kênh; seed cùng fold dùng cùng cặp nên không phải quan sát độc lập.

Bank được tạo mà không đọc nhãn validation; overlap là false-negative theo nhãn benchmark đã giữ lại, không phải rò rỉ nhãn test vào train hay bằng chứng mọi phần tử khác là âm thật. Phân tích hậu nghiệm, không loại các cặp validation khỏi bank trong train hay dùng chúng để chọn model. Nếu kết quả đủ lớn và có ý nghĩa, chỉ thiết kế thử nghiệm train-only không dùng nhãn held-out.
