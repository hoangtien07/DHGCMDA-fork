# CLAUDE.md — DHGCMDA-fork project context

> File này được Claude đọc tự động khi mở project. Mục đích: giúp lần làm việc sau bắt đầu ngay không phải dò lại setup.

## 1. Mục tiêu dự án

Fork của repo [CDMBlab/DHGCMDA](https://github.com/CDMBlab/DHGCMDA), kèm bài báo gốc dạng PDF tại root: `[2026] DHGCMDA a dual-view heterogeneous graph constrastive learning framework for miRNA-disaese.pdf` (BMC Bioinformatics 2026, Sun Y. et al.). Hai mục tiêu chính người dùng (Tien) đang theo đuổi:

1. **Reproduce** kết quả paper trên HMDD v2.0 + 5 ablation variant.
2. **Báo cáo phân tích** dạng Word `.docx` tiếng Việt — file output: [BaoCao_DHGCMDA.docx](BaoCao_DHGCMDA.docx).

Nội dung báo cáo gồm 4 phần: Tổng quan + Phân tích phương pháp / Kết quả + Phê bình / Báo cáo Reproduce / Hướng mở rộng. Plan đã chốt nằm tại `C:\Users\hungld\.claude\plans\download-code-v-data-async-lovelace.md`.

## 2. Môi trường — đã setup, dùng lại trực tiếp

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = 1   # bắt buộc — code có emoji + tên file tiếng Trung
```

- **Python**: 3.12.10 (tải từ python.org, KHÔNG dùng Microsoft Store / Python 3.14 vì PyG chưa đầy đủ).
- **PyTorch**: 2.5.1+cpu (đã downgrade từ 2.11 do lỗi DLL init Windows). Pin trong [requirements.txt](requirements.txt).
- **GPU**: GT 625 không hỗ trợ → CPU only. Mỗi fold ~10 phút trên Xeon E5-2680 v4.
- Các deps đầy đủ: `torch-geometric==2.7.0`, `pandas==3.0.2`, `numpy==2.4.3`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `openpyxl`, `xlrd`, `python-docx`, `matplotlib`.
- Reinstall nếu mất venv: `py -3.12 -m venv venv; .\venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple/` (cần `--extra-index-url` vì PyTorch CPU index không có pandas/sklearn).

## 3. Lệnh chạy chính

| Mục đích | Lệnh |
|---|---|
| Verify data | `python "check_v2.0_495m383D.py"` |
| Train baseline v2.0 (~50 phút) | `python main_experiments_hetero1.py --device cpu` |
| Train một ablation variant | `python main_experiments_hetero1.py --device cpu --ablation no_cl` (modes: `no_cl|no_hgcn|no_avf|no_hgt|no_dv`) |
| Smoke test | `python main_experiments_hetero1.py --device cpu --epoch 3 --validation 2` |
| Run 5 ablations tuần tự | `.\run_ablations.ps1` (hoặc `-SmokeTest` cho 3 epochs × 2 folds) |
| Parse log → JSON | `python parse_metrics.py logs\X.log results\X.json` |
| Compile final + regenerate báo cáo | `.\compile_final.ps1` |
| Sinh báo cáo .docx riêng | `python generate_report.py` |
| Sinh hình kiến trúc | `python generate_arch_figure.py` |

## 4. Files quan trọng — KHÔNG xoá

- [param.py](param.py) — đã thêm `--ablation` flag (additive, default `none` giữ behavior cũ).
- [hetero_model.py](hetero_model.py) — đã inject 5 ablation branches trong forward (`HeterogenousGraphCLAMIR.forward`):
  - `no_dv`: dùng cùng 1 hypergraph cho cả 2 view.
  - `no_hgcn`: thay G bằng identity → degenerate HGCN thành MLP.
  - `no_avf`: thay attention fusion bằng `(z1+z2)/2`.
  - `no_cl`: zero-out cả intra + inter contrastive loss.
  - `no_hgt`: bypass HGT layers, fed fused embedding thẳng vào predictor.
- [main_experiments_hetero1.py](main_experiments_hetero1.py) — main training, không sửa logic chính.
- [generate_report.py](generate_report.py) — sinh `.docx` Vietnamese, đọc từ `results/baseline_v2.0_metrics.json` + `results/ablation_results.json`.
- [parse_metrics.py](parse_metrics.py) — auto-detect UTF-8/UTF-16 encoding (PowerShell Tee tạo UTF-16 LE BOM ff fe).
- [run_ablations.ps1](run_ablations.ps1), [compile_final.ps1](compile_final.ps1) — orchestrator scripts.
- [requirements.txt](requirements.txt) — pinned deps.
- [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md) — trạng thái thí nghiệm cuối + cách tiếp tục.

## 5. Code-paper discrepancies — STATUS sau Plan B (2026-05-08)

| Điểm | Paper | Code (sau fix) | Trạng thái |
|---|---|---|---|
| HGT attention heads | 4 | **4** ([param.py:35](param.py#L35)) | ✅ Đã fix |
| Dynamic graph update | mỗi 5 epoch | **mỗi 5 epoch** (epoch-modulo, [main_experiments_hetero1.py:793-810](main_experiments_hetero1.py#L793-L810)) | ✅ Đã fix |
| Reconstruction loss λ₃ | 1.0 | **1.0** ([main_experiments_hetero1.py:871](main_experiments_hetero1.py#L871)) | ✅ Đã fix |
| `num_association_types` | flexible | hardcode 4 ([hetero_model.py:643](hetero_model.py#L643)) | ⏸ Skip (chỉ ảnh hưởng v3.2, không reproduce) |

## 6. Phát hiện thực nghiệm bất thường — VERIFIED LẦN 2 (Plan B)

**Sửa 3/4 discrepancies KHÔNG fix pattern Fig. 4 paper.** Cụ thể:
- Phase A (code gốc): Top-1 F1 baseline = 0.5485, ablation w/o CL = 0.6381 (+16.3%), w/o HGT = 0.6405 (+16.8%).
- Phase B-C (sau fix): Top-1 F1 baseline = 0.5521 (+0.7%), ablation w/o CL = 0.6206 (+12.4%), w/o HGT = 0.6415 (+16.2%).

→ Gain của ablation giảm nhẹ nhưng **vẫn ngược paper**. Finding strengthen → đây là **legitimate observation**, không phải artifact của code-paper discrepancies. Có thể nguyên nhân chính là (i) ablation implementation additive switch không tương đương paper re-train từ đầu, hoặc (ii) loss formulation `0.3*existence + 0.7*type` khác Eq. 32 paper. Chi tiết trong [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md).

## 7. Plan B status — HOÀN THÀNH 5/5 phase

**Tất cả phase Plan B đã xong:**
- ✅ B-A: Sửa 3 discrepancies (n_head=4, λ₃=1.0, dynamic update mỗi 5 epoch)
- ✅ B-B: Smoke test pass
- ✅ B-C: Rerun baseline + 5 ablation parallel (~2.5h CPU)
- ✅ B-D: Case study breast + HCC (~9 phút train + cache score) → `results/case_study_*.csv`
- ✅ B-E: Update [generate_report.py](generate_report.py) Section 3.2/3.4/3.5/3.6 → `BaoCao_DHGCMDA.docx` (308 para, 11 bảng)

## 8. Phát hiện thêm từ case study (legitimate observation)

Case study ranking top-15 miRNA cho 2 disease cho thấy **model collapse về 1 type/disease**:
- Breast neoplasms: 15/15 top miRNAs predict type = "target", chỉ 1/15 trùng paper Table 5.
- HCC: 15/15 top miRNAs predict "epigenetics", 0/15 trùng paper Table 6.

Score raw cao đều >0.99 nhưng all cùng type → confirm class collapse. Đây là dấu hiệu thêm cho thấy implementation có vấn đề về multi-type prediction (ngoài pattern Fig. 4 đã note).

Chi tiết kết quả + analysis: xem [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md).

## 7. Output đã có

- `logs/baseline_v2.0_full.log` — log baseline đầy đủ.
- `logs/ablation_no_*.log` — log 5 ablations (UTF-16 do PowerShell Tee).
- `results/baseline_v2.0_metrics.json`, `results/ablation_*.json` — metrics đã parse.
- `results/architecture_overview.png`, `results/ablation_chart.png` — hình minh hoạ.
- `BaoCao_DHGCMDA.docx` — báo cáo cuối cùng (~25-30 trang).
- `_pdf_text/p01-42.txt` — toàn bộ text paper extracted bằng pypdf.

## 8. Caveat / lessons learned

- **PowerShell Tee mặc định UTF-16 LE với BOM** — `parse_metrics.py` đã auto-detect, không cần lo. Nếu thêm script mới đọc log: dùng `parse_metrics._read_log()`.
- **PowerShell 5.1 KHÔNG có `&&` chain** — dùng `;` hoặc `if ($?) { ... }`.
- **PyTorch 2.11 + Windows + CPU cũ** = lỗi DLL init khó debug. Stick với 2.5.1.
- **Microsoft Store Python** có WindowsApps redirect, hay sinh permission error khi pip install.
- README gốc viết tên file `check_v2_0_495m383D.py` (gạch dưới) nhưng **file thực có dấu chấm**: `check_v2.0_495m383D.py`.
- File tên tiếng Trung `介绍.txt` trong v2.0_495m383D — cần PYTHONUTF8=1 để không crash.

## 9. Dataset HMDD v3.2 — KHÔNG có sẵn

Repo gốc và fork chỉ có v2.0. Nếu muốn reproduce kết quả v3.2 trong paper:
- Download raw từ http://www.cuilab.cn/hmdd
- Tự preprocess MeSH semantic similarity (Wang method) + disease-gene similarity → **8-12 giờ work**
- Hoặc liên hệ tác giả CDMBlab.

User đã chốt scope **không** reproduce v3.2.

## 10. Plan workflow đã chốt với user

Lưu ở `C:\Users\hungld\.claude\plans\download-code-v-data-async-lovelace.md`. Đọc lại nếu cần align scope.
