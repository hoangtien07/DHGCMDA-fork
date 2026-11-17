# Nguồn sơ cấp về dòng MDAv3.2-3 và sáu nhóm gốc

Ngày tra cứu 2026-11-18. [Bài SPLDHyperAWNTF năm 2022](https://doi.org/10.1093/bib/bbac390) mô tả HMDD v3.2 có sáu nhóm evidence tổng quát: circulation, epigenetics, genetics, target, tissue và other. Trong nghiên cứu đó, nhóm `other` không được tải vì ý nghĩa association không rõ; vì vậy benchmark mà họ xây dựng có năm loại, không phải khẳng định mọi bản HMDD v3.2 chỉ có năm category.

Cùng bài gọi một bộ là **MDAv3.2-3**, mô tả 411 miRNA × 271 bệnh × 5 loại và 11.748 association, đồng thời mô tả ngưỡng lọc ít hơn ba association. Đó là bộ thống kê trùng chính xác Table 1 của [DHGCMDA](https://link.springer.com/article/10.1186/s12859-026-06436-w), bài cũng dẫn SPLDHyperAWNTF là một phương pháp so sánh. Đó là bằng chứng về một nguồn benchmark ứng viên đã tồn tại trước DHGCMDA, **không chứng minh** tác giả DHGCMDA dùng cùng byte, cùng danh sách thực thể/cặp hay chính recipe SPLD.

DHGCMDA mô tả loại miRNA/bệnh dưới hai association, trong khi bài SPLD nêu dưới ba cho MDAv3.2-3. Tensor local phục hồi có hai miRNA với một typed link sau xử lý. Các mô tả này cần được đối chiếu theo nguồn gốc, thời điểm và đơn vị đếm; chưa đủ để quyết định bộ local sai hoặc tự loại hai miRNA.

Bằng chứng local: `data_audit.json` xác nhận snapshot SPLD, reference và strict trùng elementwise labels và tên có thứ tự; `../independent_v32_full_plan_audit_20261117/identity.json` xác nhận hai similarity trùng giữa snapshot và reference. Việc thiếu v3.2 trong code DHGCMDA công bố làm giới hạn kết luận author-identical. Không nhận diện thứ sáu `other` bằng cách tự gộp hoặc thêm nhãn vào tensor năm loại.
