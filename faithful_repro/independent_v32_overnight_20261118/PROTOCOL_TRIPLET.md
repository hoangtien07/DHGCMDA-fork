# Kiểm tra cấu trúc CVtriplet v3.2 (predeclared)

Không train, không tạo hoặc chọn mô hình, không gọi split mô phỏng là fold tác giả. Từ tensor nhị phân 411×271×5 đã kiểm tra, lấy tất cả triplet dương, ngẫu nhiên chia 5 tập gần đều bằng numpy.default_rng với seed 0–4; với mỗi fold đếm triplet dương test mà cặp miRNA–bệnh còn có type dương khác trong train, số cặp test distinct, và phân phối theo số type/cặp. Mỗi seed là một mô phỏng nhạy cảm, không ước lượng độc lập. Kiểm tra train/test triplet rời nhau và số dương không mất. Không dùng tập test này để fit hay chọn.

Đếm thêm unknown triplets thuộc cặp đã biết dương ở type khác so với cặp chưa biết dương. Unknown không đồng nghĩa negative sinh học; chưa có bằng chứng cách tác giả lấy mẫu hoặc loại false negatives. Đây là kiểm tra cách hiểu task và khả năng overlap theo định nghĩa CVtriplet của paper, không khẳng định paper sai hay có rò rỉ ngoài định nghĩa task.
