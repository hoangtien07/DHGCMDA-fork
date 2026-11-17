# Bắt và đối chứng spike loss trong U1 v3.2

**Tiến độ plan:** Đã bắt và đối chứng hai spike trên checkpoint thực; còn truy nguyên cơ chế khuếch đại và kiểm chứng dữ liệu/metric tác giả.

Ngày 17/11/2026. Chạy U1 seed42 trên candidate fold0, 650 epoch và 129 lần kiểm tra ngay sau refresh. Đặt trước tiêu chí lưu sự kiện là loss lớn hơn hai lần epoch ngay trước; tiêu chí này kế thừa quan sát hồi cứu, không phải ngưỡng có ý nghĩa thống kê. Kết quả có hai sự kiện, epoch 226 và 591. Kiểm tra toàn bộ trajectory xác nhận chỉ có hai lần tăng >2 lần và cả hai đều ngay sau refresh. Đây là một lượt có trace, không dùng outer-test và không chứng minh tần suất tổng quát.

Checkpoint ghi model, optimizer, input/target, hai đồ thị và RNG; state được lấy ngay trước forward có spike, sau bước optimizer và refresh của epoch trước. Đối chứng giữ model, input, target và RNG, không bước optimizer. Mỗi nhánh đồ thị mới tái tạo đúng loss đã ghi trong lượt chạy trong dung sai 1e-5; gradient lặp đạt dung sai 1e-4, RNG cuối giống nhau. Checkpoint kiểm soát epoch 6 cũng đạt; optimizer được nạp thành công, nhưng chưa kiểm tra tiếp tục huấn luyện dài hạn từ checkpoint.

| Sự kiện | Loss epoch trước | Đồ thị cũ, cùng state | Đồ thị mới, cùng state | Norm gradient cũ → mới | Reconstruction đóng góp cũ → mới |
|---|---:|---:|---:|---:|---:|
| 226 | 5,430750 | 5,410164 | 30,224018 | 14,12 → 1196,71 | 0,890865 → 25,601713 |
| 591 | 5,325638 | 5,270837 | 31,801659 | 8,86 → 1320,30 | 1,130378 → 27,550390 |

Số loss cũ/mới trong bảng là hai can thiệp cùng một state; loss epoch trước chỉ dùng định nghĩa spike và có state khác. Kết quả chứng minh thay đổi đồng thời G và cạnh trực tiếp gây tăng loss ở hai checkpoint này; không suy ra mọi spike do nguyên nhân đó hoặc hệ quả lên F1.

## Phân tách thăm dò

Các phép trộn thành phần thực hiện **sau khi** quan sát spike, với các tổ hợp có thể chưa từng xuất hiện trong huấn luyện. Khi giữ cạnh cũ:

| Sự kiện | G cũ | Chỉ G miRNA mới | Chỉ G bệnh mới | Cả hai G mới |
|---|---:|---:|---:|---:|
| 226 | 5,410164 | 30,218864 | 5,409989 | 30,220537 |
| 591 | 5,270837 | 31,750679 | 5,319417 | 31,797722 |

Chỉ đổi cạnh khi giữ G cũ cho loss 5,414417 và 5,276546 tương ứng. Tại hai state này, thay G miRNA đủ tạo phần lớn mức tăng. Tổng sai số reconstruction có mask của miRNA tăng 3448,73 → 109323,30 tại epoch 226 và 4443,76 → 117620,78 tại epoch 591. Đây là bằng chứng khoanh vùng đường tác động qua G miRNA và reconstruction; chưa xác định vì sao biểu diễn hoặc Gram khuếch đại, và chưa xác nhận một lỗi triển khai.

## Bằng chứng và giới hạn

- `PROTOCOL.md`, `capture.py`, `evaluate.py` và `run/completion.json` ghi điều kiện chạy, sự kiện và kiểm tra; `run/spike_0226.pt`, `run/spike_0591.pt` lưu checkpoint, `run/paired_*.json` lưu đối chứng.
- Các kết quả phân tách và kiểm tra được ghi trong `factors_*.json`, `g_parts_*.json` cùng giao thức tương ứng. `verify.py` kiểm tra SHA256, 650 epoch, độ phủ sự kiện và dung sai lặp.
- Đây là spike của lượt mới, không phải checkpoint của lượt cũ có loss 103,76. Trace và phép lưu state có thể làm quỹ đạo khác lượt cũ. Hai spike trên cùng một lượt không phải hai lần tái hiện độc lập.
- Bước tiếp theo: truy vết định lượng từ G miRNA qua hypergraph encoder, fused embedding và Gram tới sai số reconstruction trên hai checkpoint; sau đó mới thiết kế đối chứng can thiệp nếu có giả thuyết cụ thể. Danh tính dữ liệu và công thức F1 0,8600 của tác giả vẫn chưa được xác nhận.

