# Đã dừng theo yêu cầu người dùng — 2026-11-18

Người dùng yêu cầu dừng và lưu để máy nghỉ. Supervisor, hai train process, watcher và finalizer đã được dừng; không tự khởi động lại khi chưa có yêu cầu tiếp tục. `bank1/status.json` ghi `PAUSED_BY_USER`, `bank1/pause_record.json` lưu thời điểm, epoch cuối và SHA checkpoint, `bank1/status_before_user_stop.json` giữ snapshot ngay trước khi dừng.

- **6/15 lượt hoàn tất**: fold0 và fold1, mỗi fold đủ seed 1234, 42, 2025. Kết quả đã tổng hợp và push trong snapshot trước.
- **2 lượt dở, không tính hoàn tất**: fold2 seed1234 dừng sau log epoch 275; fold2 seed42 sau epoch 274. Mỗi lượt còn `ckpt_best.pt` và log tại máy gốc; SHA trong `pause_record.json`. Đây không chắc là checkpoint đúng epoch dừng, không có `completion.json` và không được chấm như lượt đủ 650.
- **7 lượt chưa bắt đầu**: fold2 seed2025; fold3 và fold4, mỗi fold 3 seed.

Khi user yêu cầu tiếp tục: kiểm tra trạng thái thực và đường dẫn; giữ nguyên thư mục dở, tạo namespace mới, tái dựng preflight; vì chưa có code tiếp tục từ checkpoint được kiểm thử, chạy lại hai lượt dở từ đầu và bảy lượt chưa chạy, rồi gộp với sáu lượt hoàn tất theo protocol, đối chiếu hash/pair/metric. Không tự sửa status thành PASS hay ghi đè thư mục. Có thể khảo sát resume từ checkpoint riêng nếu xác minh đầy đủ optimizer/RNG/graph/epoch, nhưng hiện chưa đủ căn cứ hứa tiếp tục chính xác từ epoch 275/274.

Các checkpoint lớn được giữ local tại workspace gốc, không được push lên Git. Snapshot Git chứa logs và metadata nhỏ; khôi phục máy mới cần chuyển các file lớn riêng nếu thực sự sử dụng chúng. Không thay đổi cấu hình điện lâu dài; yêu cầu giữ máy thức của supervisor kết thúc khi process bị dừng.
