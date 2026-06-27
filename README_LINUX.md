# Chạy DHGCMDA trên Linux (Ubuntu)

> Branch `linux-run` — port từ workflow Windows/PowerShell sang Linux/bash.
> Code Python KHÔNG đổi logic; chỉ port orchestrator `.ps1` → `.sh` và dựng lại venv.

## 1. Dựng môi trường (1 lệnh)

```bash
./setup_linux.sh
```

Script này: cài `uv` (user-level, không cần sudo) → tải CPython 3.12 do uv quản lý →
tạo `venv/` → cài deps từ `requirements_linux.txt` (torch 2.5.1+cpu, torch-geometric,
numpy/pandas/scipy/sklearn, python-docx, matplotlib).

**Lưu ý quan trọng:** KHÔNG dùng `/usr/bin/python3.12` của Ubuntu để tạo venv — Debian
patch `site.py` khiến venv không nạp `site-packages`, `import torch` sẽ fail dù đã cài.
`setup_linux.sh` ép uv dùng bản CPython standalone riêng để tránh lỗi này.

Kích hoạt (tùy chọn — mọi script đã trỏ thẳng `venv/bin/python`):
```bash
source venv/bin/activate
```

## 2. Khác biệt so với Windows

| Windows (PowerShell) | Linux (bash) |
|---|---|
| `.\venv\Scripts\python.exe` | `venv/bin/python` |
| `$env:PYTHONUTF8 = 1` | `export PYTHONUTF8=1` (thường không cần — locale Linux đã UTF-8) |
| `.\run_xxx.ps1` | `./run_xxx.sh` |
| `*>&1 \| Tee-Object log` | `2>&1 \| tee log` |
| `Start-Job` (parallel) | hàm bash chạy nền `&` + `wait` |
| Tee tạo UTF-16 LE | `tee` tạo UTF-8 (`parse_metrics.py` auto-detect cả hai) |

Mỗi `run_*.ps1` có một `run_*.sh` tương ứng cùng tên, cùng logic/flags/tên log+json.
File `.ps1` gốc được giữ nguyên (không xóa).

## 3. Lệnh chạy chính (bản Linux)

| Mục đích | Lệnh |
|---|---|
| Smoke test | `venv/bin/python main_experiments_hetero1.py --device cpu --epoch 3 --validation 2` |
| Baseline v2.0 | `venv/bin/python main_experiments_hetero1.py --device cpu` |
| 5 ablation Fig.4 | `./run_ablations.sh` (smoke: `./run_ablations.sh --smoke`) |
| Baseline + 5 ablation song song | `./run_full_rerun.sh` |
| Seed sweep | `./run_seed_sweep.sh` |
| K sweep | `./run_k_sweep.sh --seed 1` |
| Full bilinear benchmark (best) | `./run_j1_benchmark.sh` |
| Parse log → JSON | `venv/bin/python parse_metrics.py logs/X.log results/X.json` |
| Sinh báo cáo .docx | `venv/bin/python generate_report.py` |

## 4. Số thread CPU

`main_experiments_hetero1.py` giờ tự lấy `os.cpu_count()` làm mặc định (máy này 32 cores;
trước hardcode 14 cho Xeon). Override khi cần:
```bash
DHGCMDA_N_THREADS=16 venv/bin/python main_experiments_hetero1.py --device cpu
```
