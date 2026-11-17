# Đối chứng đồ thị: retry sau lỗi hook và sai khác quỹ đạo

Lượt đầu dừng ngay trước refresh epoch225 vì Python trace báo cùng dòng nhiều lần, hook từ chối lần thứ hai. Chưa thực hiện probe hoặc có kết quả. Quỹ đạo tái chạy đã lệch nhẹ từ khoảng epoch80 so với seed42 650 trước đó. Không thay đổi các artifact/log của lượt đầu.

Sửa hook bắt đúng một lần và tái chạy trong namespace `replay_retry1`; giữ seed42, dữ liệu, source, 230 epoch và phép đối chứng giống `PROTOCOL.md`. Nếu graph hash hoặc quỹ đạo không khớp lượt 650, **không gọi đây là tái hiện spike gốc**. Vẫn có thể diễn giải sự khác biệt old/new của **chính state tái chạy** nếu nhánh new khớp loss và gradient của epoch226 trong lượt tái chạy, RNG/weights bất biến, và nhánh new lặp lại giống hệt. Ghi riêng graph hash và độ lệch so với lượt gốc; không coi sai khác source hash/nhãn dữ liệu là chấp nhận được. Không dùng outer-test hoặc thay model.
