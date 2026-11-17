# Đối chứng trọng số lớp theo tập train và nguồn đặc trưng v2.0

Ngày hoàn tất 2026-11-17. Trên cùng 5 fold, seed và cấu hình 650 epoch của đối chứng association/GIP train-only, thay trọng số lớp từ số lượng toàn tập sang số lượng của từng tập train cho Top-1 F1 **0,5153 → 0,5117** (−0,0037 điểm) và macro-F1 đa nhãn **0,5025 → 0,4943** (−0,0082 điểm). Đây là một lần chạy với một seed; không suy rộng mức chênh lệch sang mọi seed hoặc dữ liệu khác.

| Fold | Top-1: trọng số toàn tập | Top-1: trọng số train | Macro-F1: toàn tập | Macro-F1: train |
|---:|---:|---:|---:|
| 1 | 0,5549 | 0,5536 | 0,5401 | 0,5411 |
| 2 | 0,4837 | 0,4890 | 0,4731 | 0,4787 |
| 3 | 0,4881 | 0,4889 | 0,4629 | 0,4645 |
| 4 | 0,5266 | 0,5047 | 0,5152 | 0,4731 |
| 5 | 0,5233 | 0,5221 | 0,5211 | 0,5141 |
| Trung bình | 0,5153 | 0,5117 | 0,5025 | 0,4943 |

Trung bình micro-F1 đa nhãn là 0,4990 → 0,4941. Như các lượt trước, Top-1 là công thức scalar của mã tác giả; macro/micro dùng cùng dự đoán argmax và toàn bộ nhãn CSV. Chênh lệch macro chủ yếu đến từ fold 4 trong lượt này; điều đó không chứng minh nguyên nhân của riêng fold này.

## Cách làm và xác minh

- Loss gốc hard-code số lượng `[367, 157, 293, 681]` của toàn bộ v2.0, rồi tính effective-number với beta 0,99999. Đối chứng giữ nguyên công thức và chỉ thay buffer trọng số sau khi loss khởi tạo bằng số lượng scalar-label trong train. Số lượng và trọng số từng fold được tính lại độc lập tại `weights_audit.json`; mỗi fold có 1.072 cặp và đủ bốn lớp.
- Giữ cùng commit tác giả, fold train/test, seed, môi trường, số epoch, các đầu vào, GIP và mô hình của `../independent_v20_fold_gip_20261116`. Kiểm tra so sánh xác nhận chỉ số cặp và nhãn giống nhau. Chỉ một thay đổi chủ động là trọng số; quỹ đạo tối ưu hóa có thể thay đổi theo nó.
- Chạy kiểm thử công thức và tính bất biến khi đổi nhãn ngoài train; smoke 5 × 1 trước full. Lần smoke đầu lỗi khởi tạo do cách bọc lớp `super` và được giữ tại `runs/smoke_init_failure`; wrapper đã sửa rồi smoke thành công trước khi chạy full.
- `runs/full/metric_audit.json` và `comparison.json` báo PASS: các điểm hữu hạn, metric tái tính khớp và cùng cặp đánh giá, các chuỗi lỗi/fallback/NaN được rà soát đều không xuất hiện. `runs/full/` lưu hợp đồng, log, hash, dự đoán và nhãn để kiểm tra lại. Không thấy các chuỗi được rà soát không chứng minh không có mọi loại lỗi.

## Nguồn các đặc trưng tĩnh

Tệp `v2.0_495m383D/介绍.txt` trong commit tác giả gọi `D_SSM1` và `D_SSM2` là hai ma trận tương đồng ngữ nghĩa bệnh; đường chạy lại đặt `D_SSM2` vào vị trí được gọi là disease-gene. File này chỉ xác nhận cách tác giả mô tả tệp, không tái dựng thuật toán hay chứng minh giống đặc trưng gene của paper. `M_GSM` là GIP của 5.430 cặp đầy đủ, đã xác minh ở lượt trước và được tính lại từ train trong đối chứng này.

`M_FSM` được mô tả là tương đồng chức năng miRNA nhưng pipeline tạo nó chưa được khôi phục. Kiểm tra độc lập trong `static_audit.json` cho 122.265 cặp miRNA không lặp: Spearman giữa M_FSM và Jaccard tập bệnh là **0,4478** với ma trận nhị phân 5.430 cặp, và **0,3515** với 1.498 cặp có loại. Đây là tương quan mô tả, không chứng minh M_FSM được tạo từ các liên kết test; nguồn gốc và mức trùng lặp nhãn vẫn chưa biết. Các báo cáo trước so khớp v2.0 M_FSM với hai bản MISIM đã kiểm tra nhưng không thấy trùng; không thể suy ra danh tính nguồn chỉ từ kết quả đó.

Các đặc trưng tĩnh chưa xác minh, cách lấy negative dựa trên dữ liệu đầy đủ và split release 90% vẫn giới hạn khả năng gọi đây là CV sạch hoặc tái hiện paper. Kết quả v2.0 này không xác định công thức F1 0,8600 của v3.2. Các bước tiếp theo có giá trị là truy nguồn/đối chứng thay thế M_FSM theo fold và, khi cần đo độ ổn định, chạy nhiều seed đã định trước cho cả hai điều kiện.
