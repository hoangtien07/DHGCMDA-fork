# Đối chứng cục bộ tác động cập nhật đồ thị — 17/11/2026

**Tiến độ plan:** Đã hoàn tất đối chứng tại một trạng thái; còn xác định nguyên nhân spike gốc và kiểm chứng dữ liệu/metric của tác giả.

## Kết quả

Trong lượt chạy lại seed 42, tại đầu epoch 226 ngay sau refresh epoch 225, giữ nguyên trọng số, đầu vào, target và trạng thái RNG, không thực hiện bước optimizer trong nhánh đối chứng:

| Đồ thị | Tổng loss | Norm gradient | Thành phần reconstruction đóng góp vào loss |
|---|---:|---:|---:|
| Trước refresh | 5,402071 | 11,477365 | 0,900225 |
| Sau refresh | 5,793993 | 14,868119 | 1,083373 |

Tổng loss tăng 0,391923 (7,26%); norm gradient tăng 29,54%. Độ lệch chuẩn hóa giữa hai gradient lặp lại trên cùng đồ thị mới là 4,05e-8; hai loss lặp giống nhau theo giá trị đã ghi và RNG cuối giống nhau. Gradient không trùng bit tuyệt đối, do đó không khẳng định tính tất định hoàn toàn. Nhánh đồ thị mới khớp loss thực tế ở epoch 226; norm gradient là 14,867988 trong runner, sai khác nằm trong dung sai định trước. Hiệu ứng so sánh lớn hơn sai lệch lặp quan sát được.

Như vậy refresh *có thể trực tiếp* làm thay đổi loss và gradient tại trạng thái này. Hai thành phần G và tập cạnh thay đồng thời: chưa thể quy tác động riêng cho thành phần nào. Số cạnh tương tự miRNA đổi 1626→1730, bệnh 1244→1000; hai tập cạnh liên kết đều có 6293 cạnh. Giá trị reconstruction sum của nhánh bệnh đổi 363,43→1148,15. Những số này chưa xác lập ảnh hưởng lên F1 hoặc tác động trung bình trên các epoch.

## Giới hạn và lịch sử kiểm tra

Lượt chạy lại không khớp graph hash và quỹ đạo của lượt 650 epoch seed 42 trước đó. Nó có loss epoch 225 là 5,424 và epoch 226 là 5,794, không tái hiện spike 103,756 của lượt gốc. Vì vậy kết quả này **không chứng minh nguyên nhân spike gốc**. Trường `limits` trong JSON dùng cụm “one selected spike” dễ gây hiểu nhầm: thời điểm được chọn theo spike cũ nhưng lượt đối chứng hiện tại không có spike đó.

Lượt đầu dừng vì hook trace gặp cùng dòng nhiều lần; retry1 dừng vì yêu cầu gradient trùng từng bit. Retry2 đặt dung sai trước khi xem kết quả, lưu state và vượt các kiểm tra nêu trên. Nguyên nhân lệch quỹ đạo chưa được xác định. Không dùng kết quả outer-test để chọn mô hình trong đối chứng này.

## Bằng chứng và bước tiếp

- Giao thức và dung sai: `PROTOCOL_RETRY2.md`; kết quả máy đọc `result_retry2.json`; snapshot `replay_retry2/probe_state.pt`; kiểm tra toy `test_paired.py`.
- Cần truy dấu sự không tất định và lưu checkpoint trước refresh trong lượt có spike, sau đó chạy lại đối chứng trên chính checkpoint đó. Có thể tách riêng tác động G và cạnh trên snapshot hiện tại như phân tích thăm dò, không xem là giải thích spike gốc.
- Metric 5 loại và danh sách dữ liệu thật sự của tác giả vẫn cần bằng chứng độc lập; đối chứng này không giải quyết hai câu hỏi đó.
