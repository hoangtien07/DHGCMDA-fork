# Baseline validation v3.2 theo 5 fold — cố định trước khi tính

Đọc bundle selection đã kiểm toán: fold0/1 đã pin của U1, fold2–4 mới build. Dùng duy nhất `train_pairs`, `y_train`, `val_pairs`, `y_val`; không đọc test labels hoặc model checkpoints của fold mới. Dự đoán một trong năm loại theo thứ tự genetics/epigenetics/circulation/target/tissue; tie chọn index nhỏ nhất.

Các quy tắc cố định, không chọn theo kết quả: (1) global chọn support toàn cục train; (2) disease chọn số nhãn train theo disease, tie bằng support global, disease chưa thấy dùng global; (3) miRNA tương tự; (4) product: `log(count_m+1) + log(count_d+1) - log(global_count+1)`, chọn argmax, cộng 1 mỗi loại cố định. Đây là heuristic từ số đếm nhãn multilabel, không diễn giải là xác suất đã hiệu chuẩn.

Metrics giữ `independent_v32_metric_baseline_20261117/CONTRACT.md`: macro-F1 năm lớp, micro-F1, any-hit, per-class TP/FP/FN. Điểm là inner-validation thuộc từng fold — là mốc phát triển, không phải outer-test hay kết quả paper. Fold0/1 prior disease phải khớp kiểm toán trước với sai số 1e-12; các baseline khác là phân tích thăm dò, không dùng để chọn test hay điều chỉnh epoch. Không trình bày giá trị lớn nhất như mô hình đã chọn.
