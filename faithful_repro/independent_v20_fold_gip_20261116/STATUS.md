# Đối chứng v2.0 theo fold: trạng thái và bằng chứng

Ngày 2026-11-16. Lượt đầu chỉ che ma trận association đã dừng sau epoch 100 fold 1, trước khi đánh giá, do phát hiện M_GSM vẫn dùng GIP từ toàn bộ 5.430 cặp nhị phân. Lượt dừng được giữ trong `../independent_v20_masked_20261116/runs/full/`, không sử dụng như kết quả thí nghiệm.

`../independent_v20_masked_20261116/audit_gip.py` đọc CSV nhị phân từ Git blob được pin `3b1a253895ce800bddaabea96017510218a543e7`, tính GIP và so sánh M_GSM bản release: sai số tuyệt đối tối đa 4,440892098500626e-16. Trong cả 5 fold, toàn bộ 268 cặp test đều thuộc 5.430 cặp đầu vào của GIP. Xác minh độc lập tại `../independent_v20_masked_20261116/gip_audit.json`.

Lượt hiện tại dựng ma trận association từ đúng 1.072 train positives mỗi fold và tính lại GIP nhị phân từ đó; tách truth chỉ cho đánh giá sau train. Đồ thị liên kết được xác minh riêng trong `../independent_v20_masked_20261116/graph_audit.json`: 1.072 cạnh train, 0 cạnh test ở mỗi fold. Chạy thử 5 fold × 1 epoch hoàn thành và kiểm toán metric đạt; đây không phải kết quả hiệu năng. Lượt đầy đủ 5 × 650 được lưu tại `runs/full/`.

Nguồn tĩnh `M_FSM`, `D_SSM1`, `D_SSM2` chưa chứng minh không chứa thông tin test. Trọng số lớp cố định từ toàn bộ dữ liệu và cách lấy negative vẫn như release. M_GSM train-only cũng bỏ thông tin của 3.932 cặp chỉ có trong ma trận nhị phân đầy đủ, bên cạnh các cặp typed ngoài train. Vì thế không gán thay đổi điểm cho duy nhất một nguồn rò rỉ hoặc coi kết quả là CV hoàn toàn sạch.
