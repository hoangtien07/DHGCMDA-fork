# Incidence và hòa điểm top-k tại spike U1

**Tiến độ plan:** Đã kiểm tra Gram, incidence và hòa điểm tại một spike mới; còn xác định điều kiện khuếch đại, kiểm tra giải pháp bằng huấn luyện độc lập và xác minh dữ liệu/metric tác giả.

Ngày 17/11/2026. Lượt chẩn đoán U1 seed42 trên candidate fold0 hoàn tất 650 epoch; trace bắt 129 thời điểm sau refresh. Theo ngưỡng đã đặt là loss lớn hơn 2 lần epoch trước, chỉ có một spike: epoch 231, từ **5,4808 lên 143,5814**. Ngưỡng có nguồn gốc hồi cứu, đây là một lượt có trace, không phải bằng chứng tần suất tổng quát hay kết quả outer-test.

Giữ cùng state và RNG tại checkpoint 231, đồ thị cũ cho tổng loss 5,4940 và đồ thị mới cho 143,5814; nhánh mới tái tạo loss trong lúc chạy, kiểm tra lặp gradient đạt. Đây là spike của lượt mới, không phải spike 103,76 của lượt trước.

## Gram → incidence → G

Gram producer trước optimizer, normalized similarity, incidence H và G đã được lưu cho control epoch 6 và spike epoch 231. Dựng lại normalized, H và G bằng mã nguồn đã pin cho kết quả trùng tensor được lưu; hash từng bước khớp dynamic_history của chính lượt chạy. Mỗi cột H gồm tâm và 13 node lân cận.

| Đại lượng incidence miRNA | Control epoch 6 | Spike epoch 231 |
|---|---:|---:|
| Bậc trung vị / lớn nhất | 1 / 411 | 1 / 411 |
| Tỷ lệ tổng lượt tham gia thuộc 13 node bậc cao nhất | 92,84% | 89,49% |
| Hàng hòa điểm chính xác tại ranh giới top-k | 0 / 411 | 29 / 411 |
| Trong 13 node có norm projection mới lớn nhất, số node đạt ngưỡng bậc top 13 | 13 | 13 |

Bậc incidence là số hyperedge chứa node, không phải tổng hàng G. Mức tập trung cao đã xuất hiện cả ở control không spike: **tập trung một mình không đủ giải thích spike**. Lựa chọn top 13 projection là phân tích hậu nghiệm; ở trường hợp hòa bậc, tiêu chí là bậc đạt ngưỡng top 13, không khẳng định thứ hạng duy nhất.

## Kiểm tra quy tắc phá hòa điểm

Tại checkpoint spike 231, thay luật ưu tiên index tăng dần bằng giảm dần khi score bằng nhau, giữ similarity, k, self-edge, model và target; H đổi 29 cột, 76 phần tử, G đổi tương đối theo norm khoảng 14,5%. Tổng residual reconstruction miRNA là **592.021** theo luật gốc và **600.012** theo luật đảo. Thay quy tắc này không làm spike reconstruction biến mất tại state đã kiểm tra. Đây là can thiệp hậu nghiệm tại một checkpoint, không loại trừ ảnh hưởng của các quy tắc hoặc state khác; chưa đo F1.

## Bằng chứng và bước tiếp

`PROTOCOL.md`, `capture.py` và `run/completion.json` ghi điều kiện chạy; checkpoint `run/spike_0231.pt` chứa Gram, H, G, model, optimizer và RNG. `incidence.py`, `evaluate.py` và `tie.py` ghi các đối chứng; `verify.py` kiểm tra hash với manifest, toàn bộ 650 epoch, độ phủ spike và độ lặp. Báo cáo trước về phân rã hàng G nằm ở `../independent_v32_hypergraph_path_20261117/REPORT.md`.

Bước tiếp theo là so sánh các lần refresh có/không spike với cùng phép đo producer và thay đổi membership, để tìm điều kiện tương tác giữa thay đổi G, encoder và decoder. Chưa có bằng chứng rằng một thay đổi pipeline cụ thể cải thiện F1; danh tính dữ liệu và F1 0,8600 của tác giả vẫn chưa xác nhận.
