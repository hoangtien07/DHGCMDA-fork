# Truy vết các hàng hypergraph liên quan spike

**Tiến độ plan:** Đã định vị các hàng G liên quan hai spike; còn thu thập Gram/incidence trước refresh và kiểm tra can thiệp hợp lệ khi huấn luyện, song song với dữ liệu/metric tác giả.

Ngày 17/11/2026. Phân tích hai checkpoint spike 226 và 591 từ cùng một lượt seed42. Nhóm 13 node tại mỗi checkpoint được chọn hậu nghiệm theo norm projection mới, không chọn theo tổng hàng G. Giữ model, dữ liệu và target; không huấn luyện lại hay chấm outer-test.

## Cấu trúc và tín hiệu

| Chỉ số | Epoch 226: cũ → mới | Epoch 591: cũ → mới |
|---|---:|---:|
| Số node trong nhóm cũng thuộc top 13 tổng hàng G | 6 → 11 | 0 → 13 |
| Tổng hàng G trung bình của nhóm | 1,153 → 2,208 | 0,127 → 2,341 |
| Tổng hàng G trung bình của các node còn lại | 0,156 → 0,138 | 0,190 → 0,118 |
| Norm projection trung bình của nhóm | 0,961 → 4,448 | 0,841 → 4,628 |

Đây là tổng hàng của G chuẩn hóa, không đồng nhất với bậc hyperedge. Dấu hiệu trùng hợp và thay đổi theo nhóm là mô tả sau khi chọn state, không tự chứng minh cấu trúc top-k là nguyên nhân gốc.

Với từng view, phân rã chính xác `G @ (XW+b)` thành tín hiệu từ các cột thuộc nhóm 13 và từ các cột còn lại. Ở epoch 591, sau refresh norm trung bình của hai phần tại các hàng nhóm là khoảng 11,96 và 25,72 ở view 1, 16,92 và 18,45 ở view 2; các phần có xu hướng cùng hướng. Norm của hai phần không cộng chính xác và giá trị là trung bình theo node. Điều này cho thấy tín hiệu tại nhóm chịu ảnh hưởng cả các node khác, không thể quy spike cho đặc trưng riêng của nhóm 13.

## Can thiệp trực tiếp lên hàng G

Giữ nguyên các hàng khác và thay riêng 13 hàng miRNA của cả hai view; đo tổng residual Gram miRNA:

| Checkpoint | G cũ | G cũ, thay 13 hàng bằng mới | G mới, trả 13 hàng về cũ | G mới |
|---|---:|---:|---:|---:|
| 226 | 3.449 | 99.456 | 5.212 | 109.323 |
| 591 | 4.444 | 132.899 | 3.122 | 117.621 |

Hai G pha trộn **không đối xứng**, có thể không phải hypergraph Laplacian hợp lệ. Fusion còn dùng pooling toàn node, nên đổi hàng G có thể thay gate và ảnh hưởng các hàng khác. Kết quả chứng minh các hàng này đủ tạo phần lớn thay đổi tại phép tính cục bộ, không chứng minh có thể sửa pipeline bằng phép thay hàng hoặc cải thiện F1.

Hai nhánh gốc tái hiện residual đã lưu; phân rã conv và hash checkpoint/giao thức đạt kiểm tra `verify.py`. Chi tiết nằm trong `hypergraph.json`, `row_intervention.json`, `PROTOCOL.md` và `PROTOCOL_ROWS.md`.

## Phần chưa xác định

Checkpoint bắt spike lưu G cũ/mới nhưng **không lưu Gram producer trước optimizer hay incidence H của lần refresh**. Mã dùng k=13, nhưng chỉ từ G và việc nhóm có 13 node không thể chứng minh tie trong top-k hay cơ chế tạo hyperedge. Cần một lượt có ghi Gram producer, normalized similarity và incidence trước/sau refresh; thiết kế kiểm tra xác nhận tính khớp hash với manifest và checkpoint, rồi mới có cơ sở kiểm tra giả thuyết top-k. Spike 103,76 của lượt cũ vẫn chưa có checkpoint trực tiếp.
