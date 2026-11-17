# Đối chứng decoder và nhóm node tại hai spike

**Tiến độ plan:** Đã khoanh vùng ảnh hưởng vào một nhóm nhỏ embedding tại hai checkpoint; còn kiểm tra cơ chế tạo nhóm này, tác động khi huấn luyện và dữ liệu/metric tác giả.

Ngày 17/11/2026. Hai checkpoint epoch 226 và 591 thuộc cùng lượt seed42 và được chọn sau khi quan sát spike; đây là phân tích hậu nghiệm, không phải hai lần huấn luyện độc lập. Đối chứng cố định model, input/target, không optimizer, không dùng outer-test. Gốc cũ và mới khớp sai số reconstruction miRNA đã lưu trong dung sai 1e-5.

## Phân tách trước ReLU

Với `a = fc1(z)`, viết `ReLU(a) = |a| × 1[a>0]`. Các tổ hợp chéo kết hợp độ lớn `|a|` của một nhánh và mask dương của nhánh kia; chúng là phép can thiệp nhân tạo, không nhất thiết ứng với embedding thực nào.

| Checkpoint | Độ lớn cũ, mask cũ | Độ lớn cũ, mask mới | Độ lớn mới, mask cũ | Độ lớn mới, mask mới |
|---|---:|---:|---:|---:|
| 226 | 3.449 | 3.451 | 106.518 | 109.323 |
| 591 | 4.444 | 24.907 | 195.124 | 117.621 |

Các số là tổng bình phương residual Gram miRNA trên mask. Ở epoch 226, chỉ thay độ lớn trong phép trộn đã tạo phần lớn mức tăng. Ở epoch 591, cả hai yếu tố ảnh hưởng và tương tác mạnh; tổ hợp chéo thậm chí lớn hơn nhánh mới thực. Không thể phân bổ duy nhất spike cho một yếu tố hoặc kết luận ReLU vô hại. Tỷ lệ phần tử đổi dấu lần lượt 0,38% và 2,76%, nhưng tỷ lệ phần tử không biểu thị tầm quan trọng của các phần tử đó.

## Can thiệp 13 node

Tại mỗi checkpoint, chọn **sau khi xem kết quả** 13 node có norm projection mới lớn nhất, sau đó chỉ hoán đổi các hàng fused embedding của nhóm này; decoder và target giữ nguyên. Các nhóm được chọn riêng cho từng checkpoint, không khẳng định là cùng 13 node hay có ý nghĩa sinh học cụ thể.

| Checkpoint | Cũ | Cũ, thay 13 hàng bằng mới | Mới, trả 13 hàng về cũ | Mới |
|---|---:|---:|---:|---:|
| 226 | 3.449 | 91.005 | 5.788 | 109.323 |
| 591 | 4.444 | 100.736 | 3.466 | 117.621 |

Ở nhánh mới, khoảng 6,32% ô hợp lệ chạm ít nhất một trong 13 node đã chọn, nhưng chứa khoảng 95,4% và 97,4% tổng residual tương ứng. Những ô này gồm tương tác giữa node được chọn và các node còn lại; không được xem đây là phần sai số do từng node đơn lẻ gây ra. Việc trả 13 hàng cũ giữ các hàng còn lại ở trạng thái mới, vì vậy kết quả không nhất thiết bằng góc cũ hoàn toàn, như ở epoch 591. Phép trộn chứng minh ảnh hưởng cục bộ của các hàng được chọn tại chính state đó, chưa chứng minh có thể kiểm soát spike bằng cách can thiệp lúc huấn luyện.

Bằng chứng: `counterfactual.json`, `rows.json`, hai giao thức tương ứng và `verify.py`. Script kiểm tra hash checkpoint/giao thức, tái hiện hai góc gốc và tổng hai phần mask. Các phép thử trên không đánh giá F1, không nhận diện thực thể sinh học, và không xác định spike 103,76 của lượt cũ.

Bước tiếp theo là truy vết vì sao refresh tạo các hàng embedding lớn ở nhóm này, đặc biệt top-k và phân bố hypergraph; sau đó kiểm nghiệm một giả thuyết bằng đối chứng huấn luyện tách biệt, nếu điều kiện nguồn và metric cho phép.
