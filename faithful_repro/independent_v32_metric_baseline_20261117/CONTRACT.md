# Quy ước đối chiếu CV_type v3.2 trên validation đã lưu

Đây là quy ước nội bộ để kiểm tra và so sánh, **không khẳng định là công thức paper**. Dữ liệu gồm các cặp dương đa nhãn; kênh cố định genetics, epigenetics, circulation, target, tissue. Dự đoán đúng một loại bằng `argmax` năm score; hòa điểm chọn index nhỏ nhất theo NumPy. Với mỗi lớp tính TP, FP, FN giữa one-hot và multihot; F1 = 2TP/(2TP+FP+FN), mẫu số 0 → 0. Macro là trung bình đủ năm lớp. Micro gộp TP/FP/FN rồi tính F1. Any-hit là tỷ lệ loại dự đoán có trong tập nhãn thật. Hệ quả: vì chỉ dự đoán một loại, khi một cặp có nhiều nhãn, dự đoán trúng một nhãn vẫn bỏ sót các nhãn còn lại.

Điểm first-label hybrid chỉ là diagnostic kế thừa: truth scalar là loại dương đầu tiên theo thứ tự đã pin; P=accuracy, R=mean recall trên các lớp truth hiện diện, F1 harmonic. Nó nhạy thứ tự kênh trên cặp đa nhãn. Không dùng metric này để chọn cách gần 0,8600. Không xử lý triplet/negative bằng giao thức type này.

So sánh mô hình với baseline đã định nghĩa trong kiểm toán độc lập trước: dự đoán loại theo số nhãn của disease trên train; hòa dùng support toàn cục train, sau đó index nhỏ; disease chưa có train dùng support toàn cục. Không dùng nhãn validation để xây prior. Score mô hình lấy từ export checkpoint tốt nhất theo validation macro đã lưu; 65 lần xem validation, nên đây là phân tích phát triển, không phải ước lượng test hoặc điểm CV 5 fold. Không đọc outer-test, không lựa chọn seed, metric hoặc checkpoint.

Báo cáo macro/micro/any-hit, per-class TP/FP/FN/support/predicted và F1 cho mỗi phương pháp, số cặp trúng riêng/chung; chấm bằng phép tính độc lập, đối chiếu scikit-learn và kết quả kiểm toán trước. Inputs, labels và script được hash; fold0 và fold1 là các tập validation khác nhau, không gộp thành một test set độc lập.
