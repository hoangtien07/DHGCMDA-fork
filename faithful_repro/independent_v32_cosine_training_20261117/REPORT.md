# Đối chứng huấn luyện: cosine reconstruction loss trên validation

Ngày 2026-11-18. Kết quả thăm dò trên cùng inner fold 0 và ba seed 1234, 42, 2025; mỗi lượt 650 epoch. Cả ba lượt cosine hoàn thành không lỗi. Chuẩn hóa loss không cải thiện F1 trong ba lượt này: tại epoch tốt nhất trên validation, macro-F1 của cosine thấp hơn raw cho cả ba seed. Không nên chuyển mặc định sang biến thể này dựa trên bằng chứng hiện có.

| Seed | Macro-F1 tốt nhất raw (epoch) | Macro-F1 tốt nhất cosine (epoch) | Macro-F1 epoch 650 raw / cosine | Số spike raw / cosine | Max norm gradient raw / cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1234 | 0,4708 (650) | 0,3885 (650) | 0,4708 / 0,3885 | 1 / 0 | 1547,9 / 153,8 |
| 42 | 0,4474 (630) | 0,3883 (650) | 0,4224 / 0,3883 | 1 / 0 | 3512,3 / 136,2 |
| 2025 | 0,4711 (610) | 0,3467 (650) | 0,4335 / 0,3467 | 2 / 0 | 1128,0 / 155,5 |

Trung bình mô tả của ba seed trên cùng một fold: macro-F1 tốt nhất 0,4631 raw so với 0,3745 cosine; micro-F1 tốt nhất tương ứng 0,4974 và 0,3946; any-hit tương ứng 0,5823 và 0,4619. Không coi ba seed là ba tập dữ liệu độc lập, và không diễn giải đây là khoảng tin cậy hay hiệu quả tổng quát. Chỉ số tại epoch tốt nhất là chỉ số đã được chọn trên validation; so sánh epoch 650 là mốc cố định. Cả ba lượt cosine đạt giá trị tốt nhất ở epoch 650, do vậy chưa chứng minh hội tụ hoặc rằng thay đổi sẽ còn kém ở mọi horizon.

Spike được định nghĩa trước là train loss lớn hơn hai lần epoch ngay trước. Số spike tương ứng là 4 trên 1.950 epoch raw và 0 trên 1.950 epoch cosine; điều này không chứng minh loại bỏ bất ổn trong mọi lượt. Norm gradient tối đa giảm trong mỗi seed nhưng tổng loss cực đại không nhất thiết giảm: ở seed 1234, cosine là 35,83 còn raw là 30,24. F1 theo từng loại, precision, recall, micro-F1 và any-hit được lưu đầy đủ trong `results.json`; ví dụ seed 1234 F1 các loại của raw là [0,407, 0,373, 0,536, 0,454, 0,585], cosine là [0,331, 0,291, 0,536, 0,334, 0,451].

## Thiết kế và giới hạn

Protocol `PROTOCOL.md` được ghi trước khi chạy. Can thiệp chỉ thay loss reconstruction: L2-normalize đầu ra projection trước Gram cho hai modality, epsilon 1e-8; mask, reduction common-mass và λ giữ nguyên. Hàm refresh đồ thị vẫn nhận Gram thô từ adapter gốc, nhưng đồ thị ở các epoch sau có thể thay đổi vì quỹ đạo trọng số đã khác. Cấu trúc decoder, các loss khác và optimizer không được thay đổi. Đây không phải tái chạy decoder tác giả: bias, dropout, kích thước và reduction trong mã công khai có khác biệt.

Baseline raw là các lượt đã chạy trước cùng fold và seed, không phải phép ngẫu nhiên hóa mới. Inner fold này đã được dùng trong phát triển và các tập outer-test cũng từng được quan sát trong công việc trước; kết quả này không blind và không chứng minh khớp paper. Không đọc outer-test trong thí nghiệm này. Macro-F1 dùng dự đoán một nhãn argmax so với nhãn multi-hot, trung bình đều năm loại; chưa xác minh đó là công thức tạo F1 0,8600 trong paper.

## Xác minh

- `unit_pass.json` ghi kiểm thử tính loss và gradient so với công thức độc lập; Gram thô trả về và RNG không đổi trong một forward cố định.
- `launch.json` lưu hash protocol, wrapper, các nguồn và input; `completion.json` từng seed ghi trạng thái và hash manifest. Can thiệp thực hiện bằng wrapper thay thế hàm trong bộ nhớ, nên hash mã nền trong manifest không mô tả đủ biến thể nếu bỏ qua hash wrapper.
- `report.py` tính lại metric từ prediction tốt nhất, xác nhận hash prediction và cặp/nhãn giống nhau. `verify.py` xác nhận cấu hình, hash data/view/source và manifest trùng khớp từng cặp, đủ 650 epoch và không lỗi; kết quả PASS với cả ba seed.
- Bước tiếp theo hợp lý là phân tích nguyên nhân tương đối của loss reconstruction và gradient theo thời gian, kể cả cơ chế suy giảm F1 từng lớp, trước khi đề xuất chỉnh trọng số hoặc horizon. Mọi lựa chọn mới phải được ghi là thăm dò, không dùng lại outer-test đã lộ làm kiểm chứng mù.
