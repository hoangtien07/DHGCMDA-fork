# U1 v3.2: năm fold inner-validation và baseline cố định

Ngày 2026-11-17. Đã bổ sung U1 selection 650 epoch cho fold 2–4, cùng code/config/seed đã pin như U1 fold0/1. Trung bình **macro-F1 validation của checkpoint tốt nhất là 0,4527**, so với **0,4252** của product count baseline và **0,3554** của disease baseline. Đây là so sánh phát triển, không phải 5-fold outer-test, ước lượng độc lập, hay mức tái hiện F1 paper.

| Fold | Số cặp val | Best epoch | U1 best macro-F1 | U1 epoch 650 macro-F1 | Product macro-F1 | Disease macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 695 | 650 | 0,4708 | 0,4708 | 0,4156 | 0,3533 |
| 1 | 701 | 620 | 0,4333 | 0,4289 | 0,3960 | 0,3349 |
| 2 | 702 | 640 | 0,4533 | 0,4413 | 0,4439 | 0,3800 |
| 3 | 704 | 570 | 0,4474 | 0,4242 | 0,4131 | 0,3661 |
| 4 | 700 | 590 | 0,4586 | 0,4413 | 0,4574 | 0,3429 |
| Trung bình theo fold | — | — | **0,4527** | **0,4413** | **0,4252** | **0,3554** |

Với checkpoint U1 tốt nhất, micro-F1 trung bình **0,4799**, any-hit **0,5614**; product là **0,5252 / 0,6145**, disease **0,5384 / 0,6299**. U1 có macro-F1 cao hơn product ở cả 5 validation fold trong lượt này, song mức ở fold4 chỉ khoảng 0,0012 điểm. U1 chọn tốt nhất sau 65 lần xem chính tập validation trong mỗi fold; baseline số đếm không chọn checkpoint. Vì vậy so sánh có thiên lệch lựa chọn và không chứng minh U1 tổng quát hóa tốt hơn. Chỉ một seed, không có khoảng bất định.

## Điều kiện và giới hạn

- Fold0/1 là export U1 650 cũ trên Linux/Colab; fold2–4 là run mới trên Windows CPU, hai threads. Cùng cấu hình lưu, seed 1234, code/source hash, 650 epoch, nhịp đánh giá 10 epoch và metric theo `U1_PROTOCOL.md`; khác môi trường nên không coi các quỹ đạo là tái lập bitwise.
- Fold2–4 dùng bundle selection do builder đã pin tạo ra. `audit.json` xác nhận train/val/group, train mask, nguồn mã và việc tính lại semantic/functional; nhãn test không được đưa vào train mask hoặc dùng chấm trong các run mới. `BASELINE_PROTOCOL.md` cố định quy tắc prior trước khi chạy fold mới, nhưng các kết quả fold0/1 và nhiều phân tích phát triển khác đã quan sát từ trước.
- Outer-test fold0 từng được chấm với **checkpoint refit 200 epoch khác**, trước các phân tích sau đó. Các outer-test còn lại trùng các cặp từng ở train/validation của fold0 như `exposure.json` mô tả. Chồng lấp train là cấu trúc CV bình thường; các kết quả đã quan sát và quyết định phát triển theo sau hạn chế khả năng coi nghiên cứu sau này như một thử nghiệm hoàn toàn mù. Không chấm lại outer-test trong công việc này.
- Macro-F1 là convention nội bộ argmax-onehot vs multihot đủ 5 loại, không phải công thức Top-1 paper đã xác minh. Nguồn sequence/gene/semantic và công thức objective còn khác hoặc chưa đối chiếu được với tác giả. Không dùng chênh lệch so với 0,8600 để tính tỷ lệ tái hiện.

## Kiểm tra và hiện vật

Mỗi lượt mới lưu `u1_selection/foldN/{launch.json,expected.json,progress.log,completion.json,out/}`. `score_u1.py` kiểm tra completion/manifest hash, cặp và truth validation trùng bundle, hash prediction, rồi tính lại macro/micro/any-hit đối chiếu manifest và scikit-learn. `u1_partial_metrics.json` là bảng máy đọc của cả năm fold và bao gồm chỉ số từng lớp. `validation_baselines.json` lưu bốn baseline, `audit.json` xác minh đầu vào và `exposure.json` mô tả giao cắt chỉ số cặp. Không đọc model hoặc score outer-test cho bảng này.
