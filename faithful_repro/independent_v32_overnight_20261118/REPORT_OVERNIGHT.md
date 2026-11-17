# Báo cáo qua đêm: tái hiện DHGCMDA

Cập nhật UTC: 2026-11-17 18:14:43

**Tiến độ plan:** Kiểm toán dữ liệu, split, builder, negative bank, CVtriplet và đối chiếu paper/code đã có kết quả; thí nghiệm cap negative bank: 4/15 lượt hoàn tất, trạng thái RUNNING.

## Kết luận quan trọng

Chưa thể khẳng định tái hiện điểm paper: bộ v3.2 của tác giả, folds và định nghĩa F1 gốc chưa được xác thực. Năm loại là đúng mô tả v3.2; mã công bố hỗ trợ bốn loại thuộc v2.0. CVtype và CVtriplet là hai bài toán khác nhau.

1. **Dữ liệu và chia tập:** tensor SPLD local, reference và strict trùng nhãn và thứ tự tên; 5 loại, 11748 triplet dương và 8735 cặp dương. 2110 cặp nhiều loại. Split theo pair/group không giao segment; chưa xác nhận giống authors. Hai miRNA chỉ có một typed link, cần làm rõ quy tắc lọc của authors.

2. **Khác biệt paper và release:** mã phát hành có dùng reconstructed similarity để thay cạnh graph HGT; audit cũ nói bỏ qua reconstruction là không đúng khi nói chung. Ma trận HGCN G đầu vào vẫn là câu hỏi khác. Reconstruction cosine+MSE của release, Gram thô của paper và U1 dùng mask/reduction riêng; không gộp thành một pipeline. Xem `FIDELITY_MATRIX.md` và `upstream_graph_fixture.json`.

3. **Đặc trưng và split:** canary đổi nhãn validation trên năm fold không đổi train/view tại `build_flat` với đầu vào similarity giữ cố định; không chứng minh sequence/gene ngoài là train-only. Validation hầu hết gồm hai entity từng thấy trong train, không phải cold-start. `canary.json`, `split_audit.json`, `coverage.json`.

4. **Unknown negative:** 72.6–76.0% cặp dương inner-validation nằm trong bank cap10, dù sampler chỉ nhận train mask; không gọi đây là rò rỉ nhãn validation. Với cap1 tỷ lệ tương ứng 7.7–14.5%, nhưng objective và tổng số negatives cũng thay đổi. `negative_overlap.json`, `bank_size.json`.

5. **CVtriplet:** trong mô phỏng chia triplet, trung bình 38.2% positive triplet test có cùng pair với một type train khác. Đây là hệ quả có thể xảy ra của đơn vị chia, không tự chứng minh sai. Count/logistic baseline riêng cho CVtriplet giả lập, test cân bằng positive/unknown, có AUROC 0.9324, AP 0.9294, F1 nhị phân 0.8565; nó không phải DHGCMDA hay F1 Top-1 paper. Danh sách dương đầy đủ dùng xác định tập unknown nên benchmark mang tính transductive. `triplet_audit.json`, `triplet_baseline.json`, `triplet_verify.json`.

6. **Phân tích sai số validation:** mô hình và prior được tính trên cùng pair/nhãn. Những nhóm ánh xạ sequence mơ hồ/không phủ cho hiệu năng mô tả khác, nhưng nhóm khác nhau và chọn checkpoint theo validation, không kết luận nguyên nhân. `strata.json`, `coverage.json`.

## Thí nghiệm đang theo dõi

Cap1 so với cap10: 5 fold × 3 seed, nguyên các thành phần khác, 650 epoch, hai process; chỉ chấm inner-validation, không chấm outer-test. So sánh là can thiệp làm thay cả số lượng negatives và contrastive objective; không quy chênh lệch riêng cho false negatives.

- Trạng thái job: 4 hoàn tất / 15; đang chạy: fold1_seed42, fold1_seed2025; lỗi: không.
- Trên 4 cặp đã tổng hợp, macro-F1 best cap10 0.4556 và cap1 0.4487; tại epoch650 0.4389 và 0.4360. Đây là trung bình mô tả trên các ô đã xong.
- Báo cáo chi tiết tự cập nhật khi queue kết thúc: `bank1/REPORT.md`; trạng thái đầy đủ: `bank1/status.json`; xác minh độc lập: `bank1/verification.json` khi có.

## Kiểm tra và lưu ý

12 artifact phân tích đã qua kiểm tra hash/cross-check (`core_verification.json`: PASS). Các thử nghiệm được thiết kế và công bố sau quá trình phát triển, không phải blind confirmation; không dùng outer-test trong các phép phân tích mới. Những gì còn thiếu là tensor/split/feature recipe v3.2 gốc, công thức metric gốc và một lần đánh giá độc lập thực sự.

Các lượt cosine horizon dài của kế hoạch đầu đã hủy sau khi người dùng yêu cầu mở rộng; kết quả dở và trạng thái trước hủy được giữ trong thư mục. Bản kế hoạch áp dụng: `PLAN_REVISED.md`.
