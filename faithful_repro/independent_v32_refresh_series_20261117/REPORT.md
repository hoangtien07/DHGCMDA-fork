# So sánh toàn bộ refresh trong một lượt U1 v3.2

**Tiến độ plan:** Đã đối chiếu 130 refresh và hai spike của một lượt; còn kiểm nghiệm điều kiện khuếch đại trên các lượt độc lập và xác minh dữ liệu/metric tác giả.

Ngày 17/11/2026. Lượt U1 candidate fold0 seed42 hoàn tất 650 epoch với trace lưu Gram, normalized similarity, incidence và G ở cả **130 refresh**. Có 129 refresh có epoch tiếp theo để so sánh loss; refresh epoch 650 không có epoch 651. Hai spike theo ngưỡng định trước >2 lần là epoch 231 và 601. Runner vẫn đánh giá validation và lưu best checkpoint như cấu hình gốc; phân tích dưới đây chọn sự kiện theo loss, không dùng best checkpoint hay outer-test.

| Chỉ số | Spike 231 | Spike 601 | 127 lần không spike: trung vị / lớn nhất |
|---|---:|---:|---:|
| Loss epoch sau / epoch refresh | 2,017 | 7,171 | 1,000 / 1,741 |
| Thay đổi tương đối norm G miRNA | 0,598 | 0,663 | 0,038 / 0,879 |
| Tỷ phần tổng bậc incidence của top 13 node | 89,56% | 88,79% | 92,86% / 92,86% |
| Số hàng hòa điểm chính xác ranh giới top-k | 0 | 2 | 0 / 411 |
| Tỷ lệ membership H thay so với refresh trước | 90,34% | 87,77% | 0,12% / 92,86%* |

*Membership turnover được tính bằng số phần tử incidence đổi chia cho hai lần tổng phần tử 1 trong H; cột H gồm 13 neighbor và tâm nên giá trị tối đa theo phép đo là 13/14 ≈ 92,86%. Không có H của lần refresh trước epoch 5, vì vậy dòng này có 126 lượt không spike, thay vì 127.

Hai spike đều có turnover cao, song turnover cao **không đủ** để gây spike trong lượt này. Ví dụ bước sau refresh ở epoch 566 có turnover 92,86% nhưng loss chỉ tăng 1,066 lần; epoch 461 có turnover 92,14% và loss giảm còn 0,958 lần. Hòa điểm chính xác không phải điều kiện cần ở đây vì spike 231 có 0 hàng hòa điểm; ngược lại có lượt không spike hòa điểm ở cả 411 hàng. Kết quả này là so sánh mô tả, chỉ có hai spike trong một quỹ đạo được instrument; không chứng minh quan hệ nhân quả hay hiệu quả dự đoán ở lượt khác. Ngưỡng >2 lần có nguồn gốc quan sát hồi cứu trước thí nghiệm.

Đối chứng tại checkpoint cho thấy dùng G/cạnh cũ so với mới, cùng model và RNG, loss lần lượt **5,958 → 12,525** ở epoch 231 và **5,427 → 38,905** ở epoch 601. Nhánh mới tái hiện loss trong lượt chạy; kiểm tra lặp gradient và RNG đạt. Chúng là hai thời điểm trong cùng một lượt, không phải hai seed độc lập.

`analyze.py` xác minh hash producer, normalized similarity, H, G cũ và G mới tại từng refresh với manifest; `verify.py` kiểm tra hash tất cả file, độ phủ sự kiện và checkpoint. `series.json` lưu 130 bản ghi và thống kê; `run/refresh_*.pt` chứa bằng chứng tensor, chiếm khoảng **842 MiB**. Các file này ghi lại lượt instrumented, có thể ảnh hưởng quỹ đạo, không phải dữ liệu/tác giả gốc.

Bước tiếp theo là kiểm tra giả thuyết có kiểm soát về tương tác giữa thay đổi membership và trạng thái encoder/decoder, trên nhiều seed hoặc checkpoint độc lập, với so sánh cùng state và tiêu chí định trước. Chưa có cơ sở sửa pipeline chỉ từ các thống kê này; danh tính dữ liệu tác giả và công thức F1 0,8600 vẫn chưa được xác nhận.
