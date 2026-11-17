# Đối chứng v2.0: che association và tính lại GIP theo fold

Ngày 2026-11-16. Với cùng 5 fold và cấu hình 650 epoch của lượt chạy mã phát hành, Top-1 F1 trung bình giảm từ **0,5568 xuống 0,5153** (−0,0414 điểm); macro-F1 đa nhãn giảm từ **0,5226 xuống 0,5025** (−0,0201 điểm). Đây là so sánh một seed cho một gói thay đổi, không xác định riêng tác động của từng nguồn rò rỉ và không xác lập kết quả tương đương paper.

| Fold | Top-1 F1 release | Top-1 F1 đối chứng | Macro-F1 đa nhãn release | Macro-F1 đa nhãn đối chứng |
|---:|---:|---:|---:|---:|
| 1 | 0,5147 | 0,5549 | 0,4747 | 0,5401 |
| 2 | 0,5630 | 0,4837 | 0,5281 | 0,4731 |
| 3 | 0,5965 | 0,4881 | 0,5534 | 0,4629 |
| 4 | 0,5334 | 0,5266 | 0,4958 | 0,5152 |
| 5 | 0,5764 | 0,5233 | 0,5608 | 0,5211 |
| Trung bình | 0,5568 | 0,5153 | 0,5226 | 0,5025 |

Micro-F1 đa nhãn trung bình là 0,5209 → 0,4990. Macro/micro chấm dự đoán argmax một loại trên toàn bộ nhãn của cặp trong CSV; Top-1 là công thức scalar phát hành. Metric nhị phân là bài toán khác và không đặt vào bảng này.

## Thiết kế và phát hiện GIP

- Lượt đối chứng dùng cùng source release commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614`, cùng train/test indices, seed mặc định, 8 heads, 5 × 650 epoch và môi trường CPU như baseline. Mỗi fold có 1.072 train positives và 268 test positives. `compare_runs.py` kiểm tra bằng nhau các chỉ số train/test dương/âm và nhãn giữa hai lượt.
- Ma trận association đưa vào đặc trưng, hypergraph và đồ thị chỉ gồm các cạnh train. Nhãn đầy đủ giữ riêng cho phép chấm sau train. Đồ thị PyG tái dựng có đúng 1.072 cạnh train và không có cạnh test mỗi fold (`../independent_v20_masked_20261116/graph_audit.json`).
- `M_GSM` phát hành thực tế khớp GIP tính trên ma trận nhị phân **5.430 cặp** của tác giả, sai số tuyệt đối tối đa `4,44e-16`; cả 268 test positives trong mỗi fold đều hiện diện trong ma trận nguồn. CSV kiểm tra đọc trực tiếp từ Git blob `3b1a253895ce800bddaabea96017510218a543e7` (`../independent_v20_masked_20261116/gip_audit.json`). Đối chứng tính GIP nhị phân lại từ 1.072 train positives; ID/IM dự phòng cũng tính từ association train.
- Việc thay GIP này đồng thời loại 3.932 cặp chỉ có trong tập nhị phân đầy đủ và các cặp typed ngoài train. Vì vậy số giảm không thể quy riêng cho việc loại test positives; thay đổi ma trận, cạnh đồ thị, GIP và quá trình tối ưu hóa tác động đồng thời.

## Giới hạn và kiểm tra

Các ma trận tĩnh M_FSM, D_SSM1, D_SSM2 chưa được xác minh nguồn gốc đầy đủ. Hàm loss vẫn dùng trọng số lớp hard-code tính từ phân bố toàn bộ v2.0, và cách chọn negative theo release vẫn dựa trên ma trận đầy đủ. Do đó đây là đối chứng giảm rò rỉ đã biết, **chưa phải pipeline không rò rỉ hoàn toàn**. Các con số không giải thích trực tiếp F1 0,8600 của paper v3.2.

Kiểm thử hàm che và GIP đã chạy, smoke 5 × 1 hoàn thành trước full. `runs/full/metric_audit.json` cho PASS: hash dữ liệu 5 fold, đối chiếu metric thủ công với hàm tác giả, điểm finite, chia train/test và log; các chuỗi fallback/NaN liệt kê đều bằng 0. `comparison.json` chứa chênh lệch từng fold. `runs/full/contract.json`, `completion.json`, log và các tệp `fold_*.npz` giữ cấu hình, điểm và nhãn để tái kiểm tra. Không tìm thấy chuỗi lỗi được kiểm tra không chứng minh tuyệt đối không có mọi dạng lỗi.

Lượt che association nhưng chưa thay GIP đã dừng trước khi hoàn thành fold đầu, giữ tại `../independent_v20_masked_20261116/runs/full/`; không dùng làm kết quả. Tiếp theo cần truy nguồn các similarity tĩnh và xây phiên bản trọng số lớp chỉ từ train; nếu cần đánh giá độ ổn định, lặp cả hai điều kiện với nhiều seed cố định và cùng fold, không chọn seed theo điểm.
