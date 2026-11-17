# V2.0 replay của bản mã tác giả đã phát hành

Ngày 2026-11-16. Mục đích: xác lập baseline có thể kiểm tra của mã công khai, đồng thời tính hai định nghĩa metric trên cùng cặp/điểm số, theo yêu cầu người dùng.

## Ranh giới

- Nguồn `06f9ba3eed92e334d9a2b8003fc06d83596e5614` lấy từ git object tại repo, đối chiếu tree GitHub đã pin trước đó. Chỉ extract code .py và 5 file dữ liệu thực sự được loader dùng, mỗi blob xác minh Git SHA. Không thay đổi code tác giả hoặc workspace gốc.
- Bản phát hành mặc định v2.0, 5 fold, 650 epoch, 8 heads, 4 loại, seed khởi tạo 1234; hàm chia fold lại gọi NumPy seed=0. Dataset/feature/optimizer/negative sampling và evaluation giữ nguyên mã tác giả. CPU torch2.5.1, PyG2.6.1, numpy1.26.4, scipy1.13.1, sklearn1.5.2, pandas2.2.3, 2 threads. Cache mới riêng cho từng lượt, không dùng cache sẵn.
- Smoke: cùng cấu hình nhưng chỉ 1 epoch mỗi fold để kiểm tra import, five-fold, hook, lưu pair/score và hai metric. Không diễn giải hiệu năng smoke. Full: 650 epoch × 5 fold theo release. Nếu một fold lỗi, lưu dấu vết/partial và báo trung thực, không tự xóa hay thay đổi mã để ép chạy.
- Hook bọc `train_epoch_optimized` để lưu, sau khi hàm gốc hoàn tất, train/test pair indices, scalar label và score test mà code gốc trả về. Hook chỉ đọc tensors, không lấy thêm forward, không đổi RNG và không thay nhãn dùng train/eval. Hàm gốc đánh giá vẫn chạy đầy đủ. Thu hồi đủ 5 fold, không dừng vì điểm thấp.
- Lưu cả positive/negative test scores. Dựng multihot truth từ CSV gốc theo channel circulation/epigenetics/target/genetics; đối chiếu scalar label với nhãn cuối cùng trong CSV như loader. Trên chính positive pairs và score `[:,1:5]`: (1) metric gốc first-label scalar, (2) macro/micro F1 argmax-onehot vs multihot và any-hit. Dùng raw pair/score giữ kết quả có thể chạy lại. Không chọn convention theo mức gần paper.
- Theo dõi lỗi bị mã gốc bắt và đường dự phòng điểm hằng qua log/score; xuất đủ evidence. Không tự gọi đây là CV_type sạch hay CV_triplet paper: code tạo tập từ toàn bộ ma trận, đưa full association vào feature/graph, và chia 90% vào 5 phần, 10% khác được giữ riêng. Việc pair existence đã biết hay chưa phải được nêu khi diễn giải.
- Original single-label data có nhiều nhãn cho cùng cặp ở CSV, nhưng loader ghi đè về một nhãn. Multihot chấm phụ nhìn lại trên cùng điểm số; không gán nó là cách mã tác giả đã train.

## Kiểm tra trước khi kết luận

- Hash code, dữ liệu, config, hook, fold NPZ, log; xác nhận đúng 5 fold đầy đủ, không chứa NaN, không thiếu mẫu, không fallback âm thầm.
- Chạy lại metric gốc từ lưu trữ và so với dòng log/mã evaluator; kiểm tra macro/micro độc lập với scikit-learn. Kết quả theo fold và mean P/R/F1 (không suy F1 từ mean P/R nếu code gốc average per fold).
- So sánh với paper v2.0 chỉ mô tả, nêu rõ fold tác giả thực nghiệm và metric paper chưa được xác minh. Không dùng con số để chọn seed/config.
