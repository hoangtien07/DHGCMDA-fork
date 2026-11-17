# Namespace bank1_linux — chạy lại hàng đợi cap1 trên Linux/Devin

Ngày tạo: 2026-11-18 (UTC). Ngữ cảnh: hàng đợi `bank1/` gốc được tạo và chạy trên máy Windows (`E:/1.AppCode/1.VSCode/DHGCMDA-fork`, supervisor ghim `uv` + `ctypes.windll`, preflight ghi đường dẫn tuyệt đối Windows). Snapshot Git tại thời điểm bàn giao chỉ có 6/15 lượt hoàn tất. Người dùng xác nhận không kết nối lại máy Windows; toàn bộ công việc chạy trên VM Linux này. Không restart supervisor trên cùng output cũ — đây là namespace mới, thư mục mới, preflight mới, hash mới.

## Quan hệ với bank1 gốc

- `PROTOCOL.md`, `train.py`, `preflight.py`, `report.py`, `verify_final.py` được copy nguyên byte từ `bank1/` (cùng protocol, cùng logic trainer/report/verify). Hash của `PROTOCOL.md` trong preflight mới giống hệt preflight Windows — có thể đối chiếu trực tiếp.
- `supervisor.py` được viết lại cho Linux: bỏ `ctypes.windll.SetThreadExecutionState`, thay launcher `uv run --python C:/...` bằng `.venv-bank1/bin/python` (Python 3.12.16, torch 2.5.1+cpu, PyG 2.6.1, numpy 1.26.4, scipy 1.13.1, sklearn 1.5.2, pandas 2.2.3), giữ nguyên 2 worker và format `status.json`. Không còn watcher riêng — `verify_final.py` chạy thủ công sau khi supervisor kết thúc.
- `preflight.py` sinh `preflight.json` mới với đường dẫn Linux và hash mới; các hash bundle/manifest/expected/bank được kiểm lại độc lập (PASS 15/15, 2026-11-18).

## Khác biệt đã biết và giới hạn

- OS/BLAS khác Windows → các run cap1 Linux không bitwise-identical với 6 run cap1 Windows đã có; đây là hạn chế reproducibility giống lưu ý đã ghi trong PROTOCOL.md cho baseline seed1234 fold0/1.
- `verify_final.py` đối chiếu arm `raw10` (15 baseline cap10 cũ, có sẵn manifest + predictions_val.npz trong repo) với arm `raw1` mới — baseline cap10 vẫn là kết quả lịch sử chạy trên Windows, không chạy lại.
- Không sửa `bank1/` gốc; mọi output mới nằm trong `bank1_linux/runs/`, `contracts/`, `status.json`, `results.json`, `REPORT.md`, `verification.json`.
