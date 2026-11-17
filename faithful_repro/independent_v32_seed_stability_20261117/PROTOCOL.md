# Độ ổn định U1 trên một split và hai seed bổ sung

Mục tiêu: đo độ nhạy seed của U1 tại cùng candidate fold0 đã phát triển; không tuyên bố kiểm định độc lập hay tái hiện F1 paper. Chốt trước hai seed 42 và 2025; so với seed 1234 đã chạy trước. Mỗi seed chạy mới từ đầu 650 epoch, đánh giá mỗi 10 epoch trên cùng 6.293 train/695 inner-validation. `seed` và `inter_seed` cùng thay bằng seed tương ứng; các cấu hình/nguồn khác giữ nguyên, không đổi split, feature, objective, optimizer, tiêu chí chọn hoặc baseline. Mã U1/fold0 bundle và sidecar được hash trước/sau.

Ghi metrics validation best và epoch 200/400/650, số lần nhìn, train-val, F1 từng lớp; score từ checkpoint best lưu và đối chiếu labels, hash, scikit-learn. Mọi kết quả validation đã dùng phát triển và chọn best; seed 1234 chạy Linux/Colab, hai seed mới chạy Windows CPU, hai threads, cùng version thư viện. Không chấm outer-test, không chọn seed/horizon để tối đa hóa score; không gọi ba seed là ba fold hay ước lượng độc lập.

Có các lượt seed 42/2025 cũ trên split 6.289/699 và continuation khác; chỉ ghi nhận chúng là bằng chứng lịch sử riêng, không trộn với split candidate 6.293/695. Thêm kiểm toán quỹ đạo 5 fold seed 1234 từ các manifest đã lưu để làm rõ độ nhạy horizon; đây là retrospective diagnostic.
