# Đối chứng tương đồng chức năng tính theo train v2.0

Ngày 2026-11-17. Giữ cùng 5 fold, seed, 650 epoch, GIP và trọng số theo train của lượt trước, thay `M_FSM` phát hành bằng đặc trưng tính theo tập bệnh train cho Top-1 F1 **0,5117 → 0,5060** (−0,0057 điểm) và macro-F1 đa nhãn **0,4943 → 0,4887** (−0,0056 điểm). Đây là kết quả một seed của phép thay đặc trưng; không chứng minh `M_FSM` gốc bị rò rỉ và không phải xác minh tái hiện paper.

| Fold | Top-1: M_FSM phát hành | Top-1: đặc trưng train | Macro-F1: M_FSM | Macro-F1: đặc trưng train |
|---:|---:|---:|---:|
| 1 | 0,5536 | 0,5080 | 0,5411 | 0,4955 |
| 2 | 0,4890 | 0,4795 | 0,4787 | 0,4560 |
| 3 | 0,4889 | 0,4869 | 0,4645 | 0,4642 |
| 4 | 0,5047 | 0,5333 | 0,4731 | 0,5104 |
| 5 | 0,5221 | 0,5224 | 0,5141 | 0,5176 |
| Trung bình | 0,5117 | 0,5060 | 0,4943 | 0,4887 |

Micro-F1 đa nhãn trung bình là 0,4941 → 0,4892. Top-1 là công thức scalar của mã công bố; macro/micro dùng dự đoán argmax và toàn bộ nhãn CSV trên cùng các cặp.

## Phương pháp và giới hạn

Với mỗi miRNA, lấy tập bệnh có association trong train; với hai tập bệnh, tính tổng các giá trị tương đồng bệnh lớn nhất theo hai chiều và chia cho tổng kích thước hai tập. Đường chéo đặt 1, kể cả miRNA không có cạnh train; các phần tử ngoài đường chéo của miRNA đó bằng 0. Ma trận bệnh là D_SSM1 phát hành. Đây là phép thay thế theo họ phương pháp tương đồng chức năng dựa trên bệnh liên quan, được mô tả trong [Wang et al. 2010](https://doi.org/10.1093/bioinformatics/btq241) và [Li et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602518/); không có bằng chứng đặc trưng này khớp pipeline `M_FSM` v2.0 của tác giả.

Đặc trưng thay thế được dùng ở mọi vị trí trước đó nhận M_FSM, gồm similarity của đồ thị, hypergraph và mục tiêu tái tạo. Điều này thay nguồn thông tin lẫn giá trị và quá trình tối ưu hóa. Do vậy chênh lệch điểm không định lượng riêng mức rò rỉ, không chứng minh M_FSM có hay không có dữ liệu test và không cho biết cách tạo F1 0,8600 trên v3.2.

Mỗi fold có 205–215 trong 495 miRNA không có cạnh train; **138/1.340 cặp test** thuộc các miRNA này (22, 24, 28, 31, 33 theo fold). Với các cặp đó đặc trưng mới không có tập bệnh train. `feature_audit.json` ghi accuracy scalar riêng từng nhóm, chỉ nhằm mô tả, không phải ước lượng nhân quả của cold start. D_SSM1/D_SSM2 vẫn là ma trận tĩnh chưa xác minh nguồn và negative sampling của release vẫn phụ thuộc dữ liệu đầy đủ; không gọi thí nghiệm là hoàn toàn sạch.

## Kiểm tra và tệp

Kiểm thử công thức gồm ví dụ tính tay, so sánh vòng lặp độc lập, tính nhị phân và tình huống không có cạnh. Smoke 5 × 1 đạt trước full. `feature_audit.json` xác nhận 5 ma trận đã lưu khớp tính lại từ train và hash log; `runs/full/metric_audit.json` tính lại metric và đối chiếu log, `comparison.json` kiểm tra các cặp và nhãn giữa hai lượt. Cả 5 fold hoàn tất, các điểm hữu hạn; không tìm thấy các chuỗi fallback/NaN được liệt kê trong kiểm toán. Không tìm thấy các chuỗi này không chứng minh tuyệt đối mọi khía cạnh đều không lỗi.
