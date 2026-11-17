# Kiểm tra peer review, reconstruction và đối chứng cosine (2026-11-17)

## Kết luận và phạm vi

Nguồn công khai cho thấy có khác biệt quan trọng giữa công thức reconstruction trong bài, mã v2.0 được công bố, và nhánh U1 nội bộ. Chuẩn hóa cosine làm norm gradient nhỏ hơn rõ rệt tại cả bốn checkpoint spike đã kiểm tra, nhưng đây chỉ là đối chứng sau khi quan sát trên trạng thái cố định; không chứng minh huấn luyện ổn định hơn, F1 tăng, hay giải thích toàn bộ gap so với paper. Không dùng kết quả outer-test để chọn cấu hình.

## Nguồn và cấu hình

- [Bài báo, các phương trình 19–21 và 32](https://link.springer.com/article/10.1186/s12859-026-06436-w); [phản hồi tác giả trong open peer review](https://static-content.springer.com/openpeerreview/art%3A10.1186%2Fs12859-026-06436-w/12859_2026_6436_AuthorComment_V1.pdf).
- Phản hồi nêu λ1 = λ3 = 1, K = 13, cập nhật sau mỗi 5 epoch, θ = 0,5, xấp xỉ 2,55 triệu tham số; thông tin này chưa cung cấp danh sách thực thể/nhãn v3.2, fold gốc hay công thức tính F1 năm loại. Bảy tài liệu review được lưu và băm trong `peer_review_manifest.json`.
- Inventory GitHub tại thời điểm 2026-11-17 chỉ thấy một nhánh main, không có tag, release hay issue. Điều này không chứng minh chưa từng tồn tại các tài liệu hoặc phiên bản khác. Mã phân tích từ commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614`: [hetero_model.py](https://github.com/CDMBlab/DHGCMDA/blob/06f9ba3eed92e334d9a2b8003fc06d83596e5614/hetero_model.py), [main_experiments_hetero1.py](https://github.com/CDMBlab/DHGCMDA/blob/06f9ba3eed92e334d9a2b8003fc06d83596e5614/main_experiments_hetero1.py). Bản snapshot hai file lưu tại đây, tránh nhầm với bản fork đã sửa.
- Checkpoint U1 v3.2 có 2.445.355 tham số trainable (`parameter_audit.json`). Không suy ra tương đương hoặc sai khác kiến trúc từ so sánh với con số khoảng 2,55 triệu: cấu hình cụ thể đằng sau con số tác giả không được xác định.

## Ba định nghĩa reconstruction cần phân biệt

| Nguồn | Reconstruction và trọng số |
| --- | --- |
| Paper | Eq. 19–21 mô tả Gram từ biểu diễn và tổng bình phương Frobenius; Eq. 32 và phản hồi đặt λ3 = 1. |
| Mã tác giả công bố cho v2.0 | Decoder có bias, ReLU, dropout, chuẩn hóa L2 trước Gram; train dùng MSE cho từng modality và hệ số 0,15 cho tổng reconstruction. |
| U1 nội bộ đang đánh giá | Decoder bias-free, Gram thô; phần reconstruction có mask, tổng bình phương được chia cho weighted positive mass trong chế độ `common_mass_sums`; λ3 danh nghĩa là 1. |

Ba mô tả này không tương đương chỉ vì cùng tên loss hoặc cùng λ3. Ta chưa biết mã công bố có phải chính phiên bản đã chạy v3.2 trong paper. Cách diễn đạt kích thước các ma trận trong paper cũng chưa đủ rõ để khẳng định hoàn toàn kiến trúc decoder thực tế.

Đối chiếu 3.250 epoch ở năm fold seed 1234 xác nhận phép rút gọn của U1 (`objective_trace.json`): ở fold 0 mass là 4.768,03125, tổng reconstruction có mask ở epoch đầu là 14.578,3434 và phần đóng góp sau chia là 3,05752. Tổng có mask này không phải tổng toàn ma trận của paper; không được suy diễn tỷ lệ đó là chênh lệch chính xác với loss tác giả.

## Đối chứng tại checkpoint spike

Cố định checkpoint, model, graph mới, input, mask, target và RNG; chỉ thay Gram thô bằng Gram của embedding đã chuẩn hóa L2 cho hai modality, giữ nguyên reduction hiện tại. Tính loss và gradient, không cập nhật trọng số. So với kết quả paired cũ, raw loss được khôi phục trong sai số tương đối 1e-5; lần chạy cosine lặp có sai số gradient tương đối < 1e-4. Thiết kế và hash trong `PROTOCOL_COSINE.md` và `cosine_counterfactual.json`.

| Seed / epoch | Loss Gram thô | Loss cosine | Norm gradient Gram thô | Norm gradient cosine |
| --- | ---: | ---: | ---: | ---: |
| 42 / 231 | 143,581 | 43,068 | 4.979,064 | 5,273 |
| 2025 / 221 | 19,547 | 42,782 | 743,826 | 3,949 |
| 2025 / 481 | 70,990 | 42,301 | 2.996,408 | 7,853 |
| 1234 / 211 | 139,651 | 42,760 | 4.309,882 | 7,748 |

Cosine tăng loss ở trường hợp 2025/221 dù giảm gradient. Những checkpoint này được chọn sau khi thấy spike; chúng không đại diện cho mọi trạng thái. Cách thử vẫn dùng decoder, mask và reduction của U1, không tái chạy decoder hoặc toàn bộ pipeline mã tác giả. Chưa có kết quả F1 hay quỹ đạo huấn luyện mới từ đối chứng này.

## Xác minh và bước tiếp theo

`verify_evidence.py` xác nhận hash của bảy PDF và bản trích chữ, script audit, protocol và bốn checkpoint; kiểm tra các control đã ghi. Phép thử gốc tự đối chiếu raw loss, tính gradient và kiểm tra RNG. Đây là xác minh tính toàn vẹn và control của artifact, không thay thế chạy lại thí nghiệm độc lập.

Bước tiếp theo là thiết kế thử nghiệm train có kiểm soát, khóa protocol và cấu hình trước, so sánh raw/cosine trên validation với nhiều seed và theo dõi spike, gradient cùng metric theo lớp; giữ outer-test đã lộ tách khỏi quá trình chọn phương án. Chỉ gọi đó là bản tái hiện mã tác giả khi đồng thời xử lý khác biệt decoder, reduction và các chi tiết dữ liệu chưa xác minh. Kết luận về F1 0,8600 và danh sách nhãn v3.2 của tác giả vẫn chưa được giải quyết bằng nguồn mới này.
