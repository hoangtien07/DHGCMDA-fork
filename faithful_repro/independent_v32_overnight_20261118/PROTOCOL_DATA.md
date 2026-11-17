# Kiểm tra lại danh tính dữ liệu và split (predeclared)

Đối chiếu elementwise Y, tên có thứ tự và hai similarity giữa snapshot SPLD local và reference; đối chiếu Y và tên reference với strict bundle được dùng tạo features. Kiểm tra binary, shape, số lượng nhãn/cặp, phân bố số loại/cặp, singleton và trùng tên; kiểm tra mỗi fold chia pair và group không giao nhau qua train/val/test, mỗi pair test đúng một lần. Chỉ kiểm tra schema/danh tính và index, không chấm hoặc chọn bằng outer-test. Không khẳng định snapshot là tập tác giả DHGCMDA chỉ từ thống kê trùng.
