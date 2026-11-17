# Phân tách refresh và truy dấu lệch quỹ đạo — 17/11/2026

**Tiến độ plan:** Đã phân tách tác động trên một snapshot và khoanh vùng sai khác từ epoch đầu; còn tìm nguyên nhân sai khác và kiểm tra spike trên checkpoint thực sự có spike.

## Phân tách thăm dò

Dùng lại snapshot epoch 226 của lần đối chứng trước, giữ model, input, target và RNG, không bước optimizer; đánh giá đủ bốn tổ hợp. G và cạnh hỗn hợp là can thiệp phân tích, có thể không từng xuất hiện trong quá trình học. Thí nghiệm được quyết định sau khi đã xem kết quả old/new, và chỉ áp dụng cho một trạng thái không có spike gốc.

| G | Cạnh | Loss | Norm gradient | Reconstruction đóng góp |
|---|---|---:|---:|---:|
| Cũ | Cũ | 5,402071 | 11,477365 | 0,900225 |
| Cũ | Mới | 5,400978 | 11,456112 | 0,900225 |
| Mới | Cũ | 5,804353 | 14,945760 | 1,083373 |
| Mới | Mới | 5,793993 | 14,868119 | 1,083373 |

Ở state này, thay G khi giữ cạnh cũ tăng loss 0,402282; thay cạnh khi giữ G cũ giảm loss 0,001092. Khi cả hai đổi, loss tăng 0,391923. Vì có tương tác, không cộng máy móc tác động riêng thành tác động tổng. Trong lần refresh này hash G miRNA không đổi, hash G bệnh thay đổi; tập cạnh tương tự của cả hai thay đổi. Các phần reconstruction và inter giống nhau khi chỉ đổi cạnh trong bảng, còn type loss đổi. Kết quả khoanh vùng vai trò G bệnh tại state này; chưa chứng minh G bệnh là nguyên nhân spike 103,76, mức tác động điển hình hoặc cải thiện F1.

Mỗi nhánh mới lặp lại trong dung sai đã nêu ở `PROTOCOL_RETRY2.md`, RNG cuối khớp. Hai nhánh old/old, new/new tái hiện loss đã lưu của lần đối chứng trước. Script `factors.py`, giao thức hậu nghiệm `PROTOCOL_FACTORS.md` và `factors.json` lưu kết quả, hash snapshot, giao thức và script.

## Lệch quỹ đạo xuất hiện khi nào?

So sánh manifest của seed42 gốc với lượt replay: hash dữ liệu, đặc trưng và mã nguồn được ghi giống nhau; khác biệt cấu hình duy nhất là max_epochs 650 so với 230. Trong mã runner đã lưu, max_epochs xuất hiện ở giới hạn vòng lặp và điều kiện xử lý epoch cuối; điều này không loại trừ các khác biệt môi trường hoặc tác động instrumentation khác.

- Ngay epoch 1, norm gradient lần lượt là 4,211147374749 và 4,211147362401; norm update cũng đã khác rất nhỏ.
- Loss có sai khác chính xác đầu tiên ở epoch 16: 7,890195846558 so với 7,890195369720.
- Tại refresh epoch 5, hash Gram và ma trận normalized đã khác; các hash G và cạnh được ghi ở refresh này chưa khác.
- Đến epoch 226, loss của hai lượt là 103,756 và 5,794. Tương quan theo thời gian không chứng minh sai khác rất nhỏ ban đầu là nguyên nhân duy nhất của spike.

`drift_audit.py` và `drift_audit.json` ghi phép so sánh và hash hai manifest. Chưa xác định được nguồn gốc sai khác epoch 1: khác biệt số học giữa lần chạy, tác động trace hay điều kiện chưa được ghi đều là giả thuyết cần kiểm tra. Không khẳng định đã tìm ra lỗi hoặc đã tái hiện spike gốc.

Bước tiếp theo là thiết kế lượt chạy có checkpoint đầy đủ model, optimizer, RNG và đồ thị ngay trước/sau refresh, với điều kiện và instrumentation được ghi lại; khi gặp spike mới, so sánh hai nhánh từ chính checkpoint đó. Các câu hỏi về danh tính dữ liệu tác giả và F1 0,8600 vẫn độc lập với phân tích này.
