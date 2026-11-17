# Xác minh checkout mới của nhánh bàn giao

Ngày 2026-11-18. Tạo worktree detached mới trực tiếp từ commit `e96f101` đã push, trên cùng máy Windows có `core.autocrlf=true`, không copy file từ workspace nghiên cứu vào checkout này. Chạy `verify_core.py`: **PASS 12 artifact và cross-check**. Sao chép bốn file protocol/preflight/train/report vào namespace mới trong checkout và chạy preflight bằng các dependency đã pin: **PASS đủ 15 cấu hình**. Thí nghiệm kiểm tra source/bundle/baseline manifests và xác nhận Git checkout giữ byte phù hợp với các hash đó.

Đây là checkout mới trên cùng host, sử dụng cache dependency hiện có; không phải clone độc lập qua mạng, không chạy training, không kiểm tra GPU hay Linux. `PORTABILITY_CHECK.md` mô tả cách tạo namespace mới và các ràng buộc đường dẫn. Khi tái tạo nơi khác, vẫn phải chạy preflight của chính môi trường mới.
