# Truy vết spike qua encoder và reconstruction

**Tiến độ plan:** Đã truy vết hai spike đến projection và Gram; còn xác định cơ chế nhạy với thay đổi G và kiểm chứng dữ liệu/metric tác giả.

Ngày 17/11/2026. Phân tích trên hai checkpoint epoch 226 và 591 của cùng lượt seed42 đã được chọn vì xuất hiện spike. Dùng model và target cố định, RNG giống nhau, thay đồng thời G và cạnh cũ/mới. Kết quả loss reconstruction và RNG khớp đối chứng đã lưu; nhánh mới được lặp lại. Tất cả đây là phân tích hậu nghiệm của hai state, không phải hai lần chạy độc lập.

## Đường khuếch đại đã quan sát

Mã hiện tại tính `G @ (XW+b)`, fusion hai view miRNA, hai lớp projection với ReLU, rồi Gram thô `P @ P.T`; reconstruction cộng bình phương chênh lệch Gram với target trên mask. Đối chứng trước đó đã cho thấy thay riêng G miRNA đủ tạo phần lớn spike.

| Đại lượng | Epoch 226 cũ → mới | Epoch 591 cũ → mới |
|---|---:|---:|
| Norm toàn bộ fused embedding | 31,55 → 34,34 | 43,50 → 39,00 |
| Norm toàn bộ projection decoder | 9,53 → 19,32 | 8,92 → 19,65 |
| Norm projection lớn nhất theo node | 1,06 → 5,57 | 1,46 → 4,63 |
| Số node có norm projection > 2 | 0 → 14 | 0 → 13 |
| Tổng Gram bình phương trên mask | 8.032 → 133.409 | 6.183 → 142.895 |
| Tổng residual bình phương trên mask | 3.449 → 109.323 | 4.444 → 117.621 |

Ngưỡng norm >2 là mô tả hậu nghiệm, không phải tiêu chí lâm sàng hay tiêu chuẩn model. Trong hai trường hợp, 5 hàng miRNA có residual lớn nhất chiếm lần lượt **33,3% và 30,4%** tổng residual sau refresh. Các hàng có mask trên đường chéo bằng không, do đó sự tăng không đến từ residual đường chéo. Giá trị Gram thô trên các ô hợp lệ có max tuyệt đối tăng 1,12 → 28,29 và 1,35 → 21,42 tương ứng.

Mức tăng không đơn giản là mọi embedding fused đều lớn lên: ở epoch 591, norm fused toàn bộ giảm nhưng norm output decoder tăng. Điều này phù hợp với thay đổi hướng/phân bố biểu diễn và đáp ứng phi tuyến decoder; số liệu hiện tại chưa xác định riêng ReLU hay ma trận trọng số nào chịu trách nhiệm.

## Cấu trúc G và giới hạn

Tổng hàng G miRNA sau refresh có median/max khoảng 0,120/2,435 ở epoch 226 và 0,117/2,341 ở epoch 591. Tương quan Pearson giữa tổng hàng G và norm projection sau refresh lần lượt 0,906 và 0,973. Tổng hàng là tính chất của ma trận G chuẩn hóa, không được gọi trực tiếp là bậc hyperedge. Đây là tương quan trên các node trong mỗi state; chưa chứng minh tổng hàng lớn là nguyên nhân riêng của spike, cũng không chứng minh đây là dữ liệu lỗi. G cũ và mới cùng có thể rất không đồng đều; trọng số và input cũng quyết định projection.

Các chỉ số của hai nhánh đã khớp residual của đối chứng checkpoint trước đó; phân rã `||Gram-target||²` khớp tổng theo mask, kiểm tra hash và RNG lặp đạt. Xem `trace.json`, `nodes.json`, `PROTOCOL.md`, `PROTOCOL_NODES.md` và `verify.py`. Các index node được lưu phục vụ điều tra, không được gán danh tính sinh học nếu chưa đối chiếu danh sách thực thể.

Bước tiếp theo là kiểm tra giả thuyết cụ thể về ReLU/decoder và cấu trúc G bằng các can thiệp giữ cố định một thành phần tại checkpoint. Các phép can thiệp chỉ là chẩn đoán; cần thực nghiệm huấn luyện tách biệt trước khi đề xuất thay đổi pipeline hoặc kết luận về F1. Spike loss 103,76 của lượt cũ chưa có checkpoint, không được xem là đã giải thích trực tiếp.
