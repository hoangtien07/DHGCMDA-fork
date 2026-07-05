# REPRODUCE — Plan M (K-sweep) + Conformal

> Mọi lệnh chạy từ gốc repo trên Linux. CPU-only (~50 phút/run 650ep×5fold; v3.2 300ep×5fold ~75 phút).
> Chi tiết môi trường: [../README_LINUX.md](../README_LINUX.md).

## 0. Môi trường

```bash
./setup_linux.sh                       # dựng venv (uv + CPython 3.12 managed), cài requirements_linux.txt
# biến môi trường: PYTHONUTF8=1 (đồng nhất Windows); DHGCMDA_N_THREADS=<n> override số thread (mặc định = os.cpu_count())
```

## 1. Plan M — Đường cong K (K=1,2) + K=2 multi-seed

```bash
# K=1, K=2 (seed 1234) — hoàn tất đường cong
./run_council_matrix.sh --matrix results/council_matrix_wave3.json --lanes 2
# K=2 multi-seed {0,42} — mean±std
./run_council_matrix.sh --matrix results/council_matrix_wave3b.json --lanes 2
# tổng hợp bảng + JSON
venv/bin/python summarize_council.py            # -> results/council_summary.json
```

Chạy trực tiếp 1 cấu hình (không qua orchestrator):

```bash
DHGCMDA_N_THREADS=16 PYTHONUTF8=1 venv/bin/python main_experiments_hetero1.py \
    --device cpu --predictor_mode full_bilinear --K_neigs 2 --exist_weight 0.1 \
    --seed 1234 --epoch 650 --validation 5 2>&1 | tee logs/fb_K2_s1234.log
venv/bin/python parse_metrics.py logs/fb_K2_s1234.log results/fb_K2_s1234.json
```

**Kỳ vọng:** K=2 Top-1 F1 ≈ 0.694 (seed1234), multi-seed mean ≈ 0.697±0.003. Đường cong đơn điệu K13<K7<K3<K2<K1.

## 2. Conformal — v2.0 (4-type)

```bash
# (a) train best config + DUMP per-fold held-out predictions
DHGCMDA_N_THREADS=16 PYTHONUTF8=1 venv/bin/python main_experiments_hetero1.py \
    --device cpu --predictor_mode full_bilinear --K_neigs 2 --exist_weight 0.1 --seed 1234 \
    --dump_scores results/conformal/v2_dump/ 2>&1 | tee logs/conformal_v2_dump.log
# (b) phân tích conformal (APS / RAPS / Mondrian / negative-control)
venv/bin/python conformal_type_prediction.py --dump_dir results/conformal/v2_dump/ \
    --alpha 0.1 0.05 --out results/conformal/v2_conformal_report.json
```

**Kỳ vọng:** APS @90% coverage ≈0.92, set-size ≈2.29/4; shuffle-control set-size ≈3.57.

## 3. Conformal — v3.2 (5-type, bypass metric-bug)

```bash
# (a) train v3.2_wang + dump (bypass Calculate_Metrics 4-type bug — script đọc raw scores)
DHGCMDA_N_THREADS=16 PYTHONUTF8=1 venv/bin/python main_experiments_hetero1.py \
    --device cpu --dataset v3.2_wang --predictor_mode full_bilinear --exist_weight 0.1 \
    --loss_mode two_head --epoch 300 --validation 5 \
    --dump_scores results/conformal/v32_dump/ 2>&1 | tee logs/conformal_v32_dump.log
# (b) conformal 5-type — chú ý Mondrian phục hồi coverage Tissue
venv/bin/python conformal_type_prediction.py --dump_dir results/conformal/v32_dump/ \
    --alpha 0.1 0.05 --out results/conformal/v32_conformal_report.json
```

**Kỳ vọng:** official Top-1 F1=0.0 (metric bug) nhưng conformal in ra real acc ≈0.30; APS T5(Tissue) coverage ≈0.64
→ Mondrian ≈0.96.

## 4. Smoke test nhanh (kiểm schema dump, ~1 phút)

```bash
venv/bin/python main_experiments_hetero1.py --device cpu --predictor_mode full_bilinear \
    --K_neigs 2 --exist_weight 0.1 --epoch 3 --validation 2 --dump_scores results/conformal/_smoke/
venv/bin/python conformal_type_prediction.py --dump_dir results/conformal/_smoke/ --alpha 0.1
rm -rf results/conformal/_smoke
```

## 5. Flags liên quan (param.py)

| Flag | Giá trị | Ý nghĩa |
|---|---|---|
| `--predictor_mode` | `full_bilinear` | predictor đầy đủ (mᵀWₜd), vượt diag |
| `--K_neigs` | `2` | số neighbor KNN hypergraph (best v2.0) |
| `--exist_weight` | `0.1` | trọng số existence loss |
| `--loss_mode` | `two_head` | existence + type (mặc định) |
| `--dataset` | `v2.0_495m383D` \| `v3.2_wang` | dataset |
| `--dump_scores` | `DIR` | **mới** — dump per-fold held-out cho conformal (additive, rỗng=không ảnh hưởng) |

## 6. Lưu ý

- KHÔNG sửa `Calculate_Metrics.py` (giữ nguyên metric-bug để chứng minh, conformal né bằng cách đọc raw scores).
- Kết quả mới dùng prefix riêng (`council_D*`, `results/conformal/`), KHÔNG ghi đè Plan A–L.
- Nền tảng học thuật + kế hoạch bài: `research/` (gốc repo).
