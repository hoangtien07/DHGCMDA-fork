# Độ nhạy seed và quỹ đạo U1 v3.2

Ngày 2026-11-17. Hai seed 42 và 2025 được chạy thêm trên **cùng candidate fold0** (6.293 train / 695 validation) với U1 650 epoch; so với seed 1234 đã có, macro-F1 validation của checkpoint tốt nhất nằm trong **0,4474–0,4711**, trung bình mô tả **0,4631**. Đây là ba seed trên một split validation đã dùng chọn checkpoint, không phải ba fold, kết quả outer-test, khoảng tin cậy hay bằng chứng tái hiện điểm paper.

| Seed | Môi trường | Macro-F1 epoch 200 | Epoch 400 | Epoch 650 | Macro-F1 tốt nhất | Best epoch |
|---:|---|---:|---:|---:|---:|---:|
| 1234 | Linux/Colab, lượt cũ | 0,3681 | 0,4169 | 0,4708 | 0,4708 | 650 |
| 42 | Windows CPU | 0,3489 | 0,3841 | 0,4224 | 0,4474 | 630 |
| 2025 | Windows CPU | 0,3110 | 0,4132 | 0,4335 | 0,4711 | 610 |

Micro-F1/any-hit ở checkpoint tốt nhất tương ứng: seed 1234 **0,5040/0,5899**, seed 42 **0,4819/0,5640**, seed 2025 **0,5065/0,5928**. Seed khởi tạo và seed negative bank thay cùng nhau, nên không tách được ảnh hưởng của từng nguồn ngẫu nhiên. Seed 1234 chạy môi trường khác; không khẳng định toàn bộ chênh lệch đến từ seed. Hai seed cũ trong `seed_replication_rev1` dùng split 6.289/699 khác và continuation khác, không được gộp vào bảng.

## Số epoch và dao động

Kiểm tra hồi cứu năm fold seed 1234 cho thấy macro-F1 tại epoch 650 cao hơn epoch 200 ở cả năm fold, trung bình **0,0672 điểm**, nhưng best so với epoch 650 cao hơn trung bình **0,0114 điểm**. Mỗi lượt có 65 lần đánh giá validation; mốc 650 và checkpoint tốt nhất được phân tích sau các kết quả phát triển, nên không chứng minh hội tụ hay tổng quát hóa. Ba seed trên cùng fold cũng đều đạt điểm epoch 650 cao hơn epoch 200, song điểm cuối thấp hơn best ở seed 42/2025.

## Đỉnh loss sau refresh: tín hiệu cần đối chứng

Trong cả ba manifest, đỉnh loss lớn nhất xuất hiện **ngay sau** epoch có ghi refresh đồ thị; thành phần reconstruction chiếm phần lớn loss tại đỉnh:

| Seed | Epoch đỉnh | Loss đỉnh | Loss epoch trước | Phần reconstruction | Gradient norm |
|---:|---:|---:|---:|---:|---:|
| 1234 | 276 | 30,24 | 5,32 | 84,9% | 1.548 |
| 42 | 226 | 103,76 | 5,44 | 95,5% | 3.512 |
| 2025 | 501 | 21,99 | 4,86 | 81,3% | 1.128 |

Phần reconstruction được tính theo hệ số reduction cuối cùng, không so trực tiếp một mean trung gian với tổng loss. Trong kiểm tra hồi cứu các bước liền kề, mỗi seed có 129 bước sau refresh và 520 bước còn lại: số bước loss tăng >2 lần lần lượt **1/129, 1/129, 2/129** sau refresh, và **0/520** ở nhóm còn lại của mỗi seed. Ngưỡng được đặt sau khi đã thấy spike; kết quả là tương quan thời điểm, không chứng minh refresh gây spike. Một đối chứng cùng weights/optimizer/RNG với đồ thị cũ và mới tại thời điểm refresh mới có thể kiểm tra cơ chế trực tiếp.

## Kiểm tra và phạm vi

Hai lượt mới dùng cùng source, bundle, view, config ngoài `seed` và `inter_seed`, được kiểm tra hash. Mỗi lượt hoàn tất 650 epoch và 65 validation looks; `score_seeds.py` xác nhận cặp/nhãn, hash export, tính lại metric qua scikit-learn và đối chiếu manifest. `trajectory_audit.json`, `loss_audit.json`, `transition_audit.json` chứa telemetry và giới hạn; `seed_metrics.json` chứa metric từng lớp. Không chấm dự đoán outer-test trong công việc này. Dataset, feature recipe và định nghĩa F1 paper vẫn có bất định như các báo cáo trước.
