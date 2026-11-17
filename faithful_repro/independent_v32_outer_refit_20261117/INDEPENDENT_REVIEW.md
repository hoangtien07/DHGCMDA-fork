# Kiểm tra độc lập kết quả outer-refit

**Tiến độ plan:** Đã hoàn tất 30 refit, ablation static và chấm lại outer-test; còn xác minh tập và metric gốc của tác giả, cùng đánh giá độc lập chưa bị dùng phát triển.

Kết quả theo metric nội bộ, trung bình mô tả theo 15 ô fold × seed; hai prior tính một lần trên outer-train của mỗi fold rồi lặp tương ứng ba seed. Các ô chia sẻ dữ liệu và lịch sử phát triển nên không phải 15 test độc lập.

| Arm / baseline | Macro-F1 | Micro-F1 | Any-hit |
|---|---:|---:|---:|
| Dynamic U1 | 0,4045 | 0,4349 | 0,5099 |
| Static U1 | 0,4403 | 0,4703 | 0,5514 |
| Product prior | 0,4296 | 0,5385 | 0,6314 |
| Disease prior | 0,3631 | 0,5539 | 0,6495 |

Static có macro-F1 cao hơn dynamic ở **14/15** cặp fold–seed trong thí nghiệm này. Product prior vượt cả hai model về micro-F1 và any-hit; static có macro trung bình cao hơn product prior nhưng mỗi so sánh có liên quan và không phải test mù. Đây là ablation của pipeline nội bộ với cố định 650 epoch, không khẳng định dynamic graph trong paper bất lợi và không dùng kết quả outer-test để chọn lại cấu hình.

`independent_review.py` độc lập đọc lại 30 export đã khóa, đối chiếu 30 hash, cùng pair theo fold và tính lại macro/micro qua scikit-learn; hash seal/results/previous verification khớp. `evaluation/verification.json` trước đó cũng đối chiếu test truth với tensor, per-class và 20 baseline. Tất cả 15 cặp dynamic/static đã qua `pair_audit.json`: cùng input, negative bank, reconstruction, G/edge ban đầu và 5 bước đầu trong dung sai số học.

Các chỉ số dùng argmax-onehot so truth multi-hot 5 loại. Outer-test fold0 đã từng bị xem và các cặp xuất hiện ở train/validation khi phát triển các fold khác; 650 epoch đã chọn sau development. Dữ liệu author, fold gốc và metric tạo 0,8600 vẫn chưa xác nhận. Không dùng các kết quả này để nói đã tái hiện hay thất bại với con số paper.

Bảng từng fold/seed và SD: `REPORT.md`; kết quả và per-class: `evaluation/results.json`; protocol và seal: `PROTOCOL.md`, `seal.json`. Phần kế tiếp hữu ích là tìm bằng chứng nguồn tác giả hoặc thiết kế một benchmark mới độc lập, được khóa trước khi xem test; bất kỳ thay đổi mô hình dựa vào các outer-test hiện tại phải ghi rõ là phát triển trên test.
