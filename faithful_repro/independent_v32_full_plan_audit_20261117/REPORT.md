# Báo cáo thực thi plan độc lập DHGCMDA v3.2

**Tiến độ plan:** Đã hoàn tất kiểm tra nguồn, ba seed chẩn đoán và 5 fold × 3 seed validation; còn outer-test/ablation theo protocol cố định và xác minh bản dữ liệu, metric của tác giả.

Ngày 17/11/2026. Kết luận: **chưa thể xác nhận tái hiện paper**, và hiệu năng theo metric nội bộ vẫn có khoảng trống. Plan chẩn đoán và mở rộng validation đã chạy đủ, nhưng đây không thay thế toàn bộ đánh giá paper hoặc nghiên cứu outer-test/ablation.

## Dữ liệu và metric

[Bài báo](https://link.springer.com/article/10.1186/s12859-026-06436-w) ghi v3.2 có 411 miRNA, 271 bệnh, 11.748 liên kết theo loại, 5 loại; v2.0 có 4 loại. [Mã tác giả công bố](https://github.com/CDMBlab/DHGCMDA) được kiểm tra tại commit `06f9ba3eed92e334d9a2b8003fc06d83596e5614`: hàm Top-1 nhận bốn điểm loại hoặc một điểm tồn tại cộng bốn điểm loại. Test trực tiếp xác nhận vector sáu điểm bị bỏ, và năm điểm loại trực tiếp bị hiểu nhầm thành một điểm tồn tại + bốn loại. Dữ liệu 5 loại và hàm 4 loại phản ánh phiên bản dữ liệu khác nhau; không thể dùng nguyên hàm đó để đánh giá 5 loại.

Tensor tham chiếu nội bộ trùng từng phần tử, tên và hai similarity với snapshot SPLD đã khôi phục: 11.748 nhãn dương thuộc 8.735 cặp, trong đó 2.110 cặp đa loại. Đây là đồng nhất giữa hai hiện vật này, **chưa chứng minh** trùng danh sách tập tác giả DHGCMDA dùng. Paper còn có số F1 khác giữa bảng và hình ablation; chưa xác định được công thức tạo 0,8600. Chi tiết và fixture: `SOURCE_REPORT.md`.

## Kết quả validation đã mở rộng

Chạy thêm 8 lượt U1 cố định trên fold 1–4 với seed 42/2025, mỗi lượt 650 epoch; cộng 7 lượt sẵn có thành đủ **5 fold × 3 seed**. Mọi lượt mới hoàn tất, không abort; kiểm tra 15 export, pair/nhãn, hash, metric từng lớp và mean/SD đạt.

| Seed | Mean macro-F1 best validation qua 5 fold | Mean macro-F1 epoch 650 |
|---:|---:|---:|
| 1234 | 0,4527 | 0,4413 |
| 42 | 0,4364 | 0,4060 |
| 2025 | 0,4391 | 0,4036 |

Trung bình mô tả của 15 ô: macro-F1 best U1 **0,4427** so với product prior **0,4252**; micro-F1 U1 **0,4728** so với product **0,5252** và disease prior **0,5384**. Any-hit U1 **0,5532** so với product **0,6145** và disease **0,6299**. Best được chọn sau 65 lần xem cùng validation, còn prior cố định theo fold; không diễn giải đây là kiểm định thắng baseline. Các ô chia sẻ dữ liệu, fold đã tham gia phát triển, seed1234 có khác môi trường; không phải 15 test độc lập. Metric là argmax-onehot đối chiếu truth multi-hot 5 loại, chưa được xác minh là F1 của paper. Bảng từng ô, độ lệch chuẩn theo fold, per-class và giới hạn: `../independent_v32_5fold3seed_20261117/REPORT.md`.

## Đối chứng spike qua seed

Ba lượt instrumented cùng fold0 với seed 42, 2025, 1234 hoàn tất 650 epoch; model seed và negative-bank seed đổi cùng nhau. Bốn sự kiện vượt ngưỡng loss >2 lần epoch trước được đối chứng trên state/RNG cố định. Đồ thị cũ → mới cho loss **5,494 → 143,581** (42/epoch231), **5,488 → 19,547** và **4,925 → 70,990** (2025/epoch221,481), **5,724 → 139,651** (1234/epoch211). Reconstruction chiếm phần lớn mức tăng; hash Gram/H/G, loss mới và repeat gradient đạt. Cùng fold và instrument nên không suy ra tần suất tổng quát hay F1. Phân tích 130 refresh của một lượt khác cho thấy H turnover lớn hoặc tập trung cao không đủ gây spike, và hòa điểm không phải điều kiện cần. Chi tiết: `DIAGNOSTIC_REPORT.md`.

## Hạng mục còn lại

1. Thu được hoặc xác nhận từ nguồn đáng tin cậy danh sách thực thể/liên kết/nhãn, feature recipes, folds và công thức metric gốc; chỉ khi đó mới so F1 paper có cơ sở. Không tự suy diễn 0,8600 từ phép tính nội bộ.
2. Nếu cần một đánh giá pipeline nội bộ đầy đủ, khóa protocol/refit, baselines và ablations trước khi chấm outer-test, ghi rõ fold0 đã từng lộ kết quả; thực hiện đủ 5 fold × 3 seed với kiểm soát chi phí. Lượt validation vừa xong không phải bước đó.
3. Trước khi đổi dynamic graph/decoder, kiểm định giả thuyết trên nhiều state/seed và chạy ablation có kiểm soát; không coi can thiệp trộn tensor cục bộ là bản sửa hợp lệ hoặc bằng chứng F1 tốt hơn.

Không có outer-test mới hay liên hệ tác giả trong lượt thực thi này. Không có con số tái hiện phần trăm được tuyên bố.
