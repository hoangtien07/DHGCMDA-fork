# Chạy lại độc lập mã phát hành DHGCMDA với dữ liệu v2.0

Ngày 2026-11-16. Kết quả: mã phát hành chạy đủ 5 fold × 650 epoch và cho **Top-1 F1 trung bình 0,556788** theo chính hàm metric đi kèm. Đây là kết quả tái chạy mã công bố, **không phải ước lượng CV_type sạch hoặc bằng chứng tái hiện điểm paper**: đầu vào chứa cả nhãn loại của các cặp test, và chưa có fold hoặc định nghĩa metric gốc của thí nghiệm paper để đối chiếu.

| Fold | Top-1 F1 mã công bố | Macro-F1 đa nhãn | Micro-F1 đa nhãn |
|---:|---:|---:|---:|
| 1 | 0,5147 | 0,4747 | 0,4712 |
| 2 | 0,5630 | 0,5281 | 0,5248 |
| 3 | 0,5965 | 0,5534 | 0,5509 |
| 4 | 0,5334 | 0,4958 | 0,5018 |
| 5 | 0,5764 | 0,5608 | 0,5559 |
| Trung bình | 0,5568 | 0,5226 | 0,5209 |

Top-1 của mã dùng một nhãn scalar cho mỗi cặp. Metric đa nhãn ở bảng dùng cùng các điểm dự đoán, chọn một loại bằng argmax, rồi so với toàn bộ nhãn của cặp trong CSV; đây là phép chấm bổ sung, không phải metric train công bố. Trung bình any-hit là 0,5537. Hàm Top-1 công bố gọi accuracy là precision và lấy trung bình recall theo lớp, rồi tính trung bình điều hòa hai giá trị đó. Mã còn in F1 **nhị phân** trung bình 0,9287; không được nhầm với F1 phân loại loại liên kết.

## Vì sao kết quả không phải CV sạch

`prepareData.py` dựng ma trận liên kết đủ loại trước khi chia fold; `main_experiments_hetero1.py` dùng ma trận đầy đủ này trong đặc trưng và đồ thị. Kiểm tra các tensor lưu lại cho thấy scalar type của **1.340/1.340 cặp test** nằm ngay tại vị trí tương ứng trong đặc trưng. `feature_oracle.py` chỉ đọc giá trị đó và đạt Top-1 F1 1,0 trên cả 5 fold. Đây là phép kiểm tra rò rỉ thông tin, không phải kết quả huấn luyện mô hình. Mức ảnh hưởng thực tế của rò rỉ tới điểm mô hình chưa được định lượng bằng đối chứng sạch.

CSV chứa 1.679 liên kết có loại trên 1.498 cặp; 161 cặp có nhiều loại. Loader ghi đè và giữ nhãn scalar cuối cùng, làm mất 181 nhãn trong biểu diễn này. CSV sắp theo mã loại tăng, nên nhãn mã lớn hơn thắng khi trùng cặp. Trong 1.498 cặp dương, 5 fold kiểm định tổng cộng 1.340 cặp; 149 cặp thuộc tập independent được tạo nhưng không được chấm trong đường chạy này, và 9 cặp còn lại không thuộc CV hoặc independent.

## Nguồn và phương pháp

- Mã và 5 tệp dữ liệu cần thiết trích từ commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614`, kiểm tra 19 Git blob theo tree nguồn và lưu SHA-256 trong `source_manifest.json`. Mã phát hành không được sửa.
- Mặc định v2.0, 4 loại, 8 attention heads, 650 epoch, 5 fold; seed khởi tạo 1234 và seed chia NumPy 0 theo mã. CPU, torch 2.5.1, PyG 2.6.1, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2, pandas 2.2.3; cache riêng mới. `PROTOCOL.md` ghi cấu hình và cách lưu điểm.
- Wrapper chỉ ghi lại chỉ số cặp, nhãn và điểm do hàm gốc trả về; kiểm tra trạng thái RNG không thay đổi bởi hook. Lưu dự đoán dương và âm, dữ liệu fold và hash tại `runs/full/`.
- `score_folds.py --mode full` kiểm tra hash 5 fold, train/test không giao nhau, đối chiếu công thức thủ công với hàm tác giả và với kết quả log, và tính F1 đa nhãn bằng scikit-learn. Kết quả PASS tại `runs/full/metric_audit.json`; không tìm thấy các chuỗi lỗi/fallback/NaN được liệt kê trong báo cáo kiểm tra. Việc không tìm thấy các chuỗi này không chứng minh tuyệt đối không có mọi loại lỗi.
- Log lịch sử `baseline_v2.0_full.log` có Top-1 F1 trung bình 0,5521 với 4 heads, cache cũ và mã fork; lần chạy hiện tại dùng 8 heads và mã phát hành. Các phiên bản mã và cache khác nhau, nên chênh lệch không thể quy riêng cho số heads.

## Bước tiếp theo

Xây dựng đối chứng theo fold: che toàn bộ cạnh và type của các cặp test khỏi mọi đầu vào suy ra từ ma trận liên kết, tính lại các đặc trưng phụ thuộc association chỉ từ train, giữ cấu hình và seed cố định; kiểm tra rò rỉ trước khi huấn luyện. Chạy cùng tập fold và báo đồng thời metric scalar của tác giả và metric đa nhãn đã định nghĩa trước. Đây là thí nghiệm mới nhằm định lượng tác động của quy trình, không được gọi là tái hiện chính xác pipeline chưa công bố của paper. Đối với v3.2 cần riêng một định nghĩa 5 loại và xác minh nguồn dữ liệu/nhãn; kết quả v2.0 không giải quyết trực tiếp con số F1 0,8600 của paper.
