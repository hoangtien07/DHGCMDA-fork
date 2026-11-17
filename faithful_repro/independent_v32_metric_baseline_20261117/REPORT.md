# Metric v3.2 và baseline theo bệnh trên validation đã lưu

Ngày 2026-11-17. Với quy ước nội bộ dự đoán đúng một loại trên truth đa nhãn, U1 có **macro-F1 cao hơn** baseline theo bệnh ở cả hai fold, nhưng **micro-F1 và any-hit thấp hơn**. Đây là kết quả validation đã được dùng để chọn checkpoint, không phải điểm test độc lập hoặc trung bình CV 5 fold.

| Fold / phương pháp | Số cặp | Macro-F1 | Micro-F1 | Any-hit |
|---|---:|---:|---:|---:|
| 0 — U1 | 695 | 0,4708 | 0,5040 | 0,5899 |
| 0 — prior theo bệnh | 695 | 0,3533 | 0,5409 | 0,6331 |
| 1 — U1 | 701 | 0,4333 | 0,4666 | 0,5435 |
| 1 — prior theo bệnh | 701 | 0,3349 | 0,5242 | 0,6106 |

Prior đếm số nhãn từng loại theo bệnh **chỉ từ train**, hòa theo support toàn cục train, sau đó theo thứ tự index; bệnh chưa thấy dùng support toàn cục. Không có cặp validation nào thuộc bệnh hoàn toàn chưa thấy trong train ở hai fold này. U1 dùng export checkpoint tốt nhất theo macro-F1 validation trong tối đa 65 lần đánh giá; prior không có bước chọn checkpoint. Sự khác nhau về quy trình chọn là giới hạn khi so sánh.

## Vì sao macro và micro cho thứ tự khác nhau?

Prior thiên về các lớp phổ biến: nó không chọn epigenetics lần nào trên cả hai tập validation, trong khi mỗi tập có 34 cặp mang nhãn này. F1 epigenetics của prior là **0**, U1 là **0,3725** và **0,3604**. Với genetics, prior F1 lần lượt **0,0606** và **0,0659**, còn U1 **0,4068** và **0,3280**. Ngược lại, với target, prior đạt **0,6354 / 0,6271**, còn U1 **0,4540 / 0,4044**. Mỗi cặp có thể mang nhiều loại nên số support các lớp không cộng thành số cặp.

Đếm any-hit trên đúng cùng cặp: fold0 có **342** cặp cả hai phương pháp trúng ít nhất một loại, **68** chỉ U1 trúng, **98** chỉ prior trúng, **187** cả hai không trúng; fold1 tương ứng **300 / 81 / 128 / 192**. Các nhóm này mô tả dự đoán, không chứng minh nguyên nhân hay hiệu năng outer-test. Chi tiết TP/FP/FN và support của năm lớp nằm trong `per_class.csv`.

## Định nghĩa metric và đối chiếu paper

`CONTRACT.md` cố định quy ước so sánh nội bộ: `argmax` năm score tạo one-hot; so với truth multihot, tính F1 từng lớp và lấy trung bình đủ năm lớp, mẫu số 0 cho 0. Micro gộp TP/FP/FN; any-hit hỏi loại chọn có thuộc tập nhãn đúng không. Một cặp đa nhãn dự đoán trúng một loại vẫn có false negative ở các loại đúng khác.

Đây **không phải công thức paper đã xác minh**. Mô tả paper nói bốn loại trong phần CV_type trong khi bảng v3.2 và tensor có năm; mã metric phát hành hard-code bốn. First-label hybrid mở rộng được giữ làm diagnostic và phụ thuộc thứ tự kênh; không chọn convention vì gần 0,8600. Các cặp validation, nhãn, score và hash được đối chiếu bundle U1 và kiểm toán trước. Các metric macro/micro được tính lại độc lập và kiểm tra với scikit-learn; fixture đa nhãn được kiểm tra trước khi phân tích.

## Tệp và phạm vi

`analyze.py` đọc hai bundle train/validation và export validation đã lưu; không đọc dự đoán outer-test, không huấn luyện, không sửa đầu vào. `results.json` chứa số liệu đầy đủ, `per_class.csv` chứa từng lớp. Các lượt U1 fold0/1 là hai fold selection khác nhau; không gộp chúng thành một test độc lập. Để kết luận hiệu năng tái hiện cần protocol và fold ngoài được cố định, tránh dùng lại kết quả test đã tham gia chọn mô hình/recipe, và nêu provenance đặc trưng cùng giới hạn không xác minh được công thức paper.
