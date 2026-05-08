# Experiment State — Reproduce DHGCMDA

> File này ghi lại trạng thái thực nghiệm. Cập nhật mỗi khi run xong một experiment.

## Lần cập nhật cuối
**2026-05-08, sau khi hoàn thành Phase B-C (rerun với 3 discrepancies đã fix).**

---

## ⏸ TRẠNG THÁI HIỆN TẠI: Đang dở Plan B (2/5 phase còn lại)

### Đã hoàn thành
- ✅ **Phase A (initial reproduce)**: baseline + 5 ablation chạy với code gốc → kết quả lưu ở [BaoCao_DHGCMDA_v1_before_fix.docx](BaoCao_DHGCMDA_v1_before_fix.docx) (snapshot báo cáo cũ).
- ✅ **Phase B-A**: Đã sửa 3 code-paper discrepancies trong [param.py](param.py) + [main_experiments_hetero1.py](main_experiments_hetero1.py):
  1. `n_head` default: 8 → **4** (param.py:35)
  2. `update_graph_frequency` default: 50 → **5** (param.py:160)
  3. `λ₃_recon` weight: 0.15 → **1.0** (main_experiments_hetero1.py:871)
  4. Block dynamic graph update: thay MSE-threshold → epoch-modulo (main_experiments_hetero1.py:793-810). Đã add `print("[INFO] Hypergraph updated at epoch X")`.
- ✅ **Phase B-B**: Smoke test 12 epochs × 2 fold pass — không NaN, hypergraph update đúng kì.
- ✅ **Phase B-C**: Rerun baseline + 5 ablation parallel 2 jobs (~2.5h wall trên Xeon E5-2680 v4 CPU).

### CÒN LẠI (cần làm khi mở máy lại)
- ⏸ **Phase B-D**: Chạy [case_study.py](case_study.py) (đã viết sẵn, ~150 lines). Train DHGCMDA trên 100% data (1498 associations), predict top-15 miRNA cho **breast neoplasms** + **hepatocellular carcinoma**, cross-check với paper Table 5/6. Output: `results/case_study_breast.csv`, `results/case_study_hcc.csv`, `results/case_study_summary.json`.
  - Lệnh: `cd d:\Tien\DHGCMDA-fork; .\venv\Scripts\Activate.ps1; $env:PYTHONUTF8=1; python case_study.py`
  - Thời gian dự kiến: ~50 phút CPU.
- ⏸ **Phase B-E**: Update [generate_report.py](generate_report.py) — thêm Section 3.5 (Case study) + 3.6 (Updated reproduce summary). Sửa Section 3.2 (đánh dấu 3/4 discrepancies "Đã fix"). Sửa Section 3.4.4 (kết luận: discrepancies KHÔNG fix pattern Fig. 4 → finding strengthen). Sau đó chạy `.\compile_final.ps1` để regenerate `BaoCao_DHGCMDA.docx`.
  - Thời gian dự kiến: ~30 phút edit + 2 phút regenerate.

---

## Kết quả Phase B-C (mới — sau khi fix 3 discrepancies)

| Variant | AUC | AUPR | F1 | Top-1 F1 | Δ Top-1 vs new Full | Time |
|---|---:|---:|---:|---:|---:|---:|
| **Full DHGCMDA (new)** | 0.9752 | 0.9701 | 0.9298 | **0.5521** | — | (parallel) |
| w/o CL | 0.9763 | 0.9716 | 0.9305 | 0.6206 | **+12.4%** | — |
| w/o HGCN | 0.9748 | 0.9696 | 0.9268 | 0.6091 | **+10.3%** | — |
| w/o AVF | 0.9756 | 0.9701 | 0.9296 | 0.5392 | -2.3% | — |
| w/o HGT | 0.9697 | 0.9693 | 0.9159 | **0.6415** | **+16.2%** | — |
| w/o DV | 0.9739 | 0.9681 | 0.9252 | 0.5608 | +1.6% | — |

### Phát hiện CHÍNH (đã được verify lần 2)

**Sửa 3 discrepancies KHÔNG fix pattern bất thường của Fig. 4.** Cụ thể:

1. **Baseline Top-1 F1**: 0.5485 (old) → 0.5521 (new) — chỉ tăng +0.7%, vẫn cách paper 0.5970 (-7.5%).
2. **Pattern ablation**: w/o CL (+12.4%), w/o HGCN (+10.3%), w/o HGT (+16.2%) **vẫn vượt baseline** — không match Fig. 4 paper (paper bảo TẤT CẢ ablation hurt).
3. **Magnitude giảm**: gain của ablation giảm so với run trước (no_cl 16.3% → 12.4%, no_hgt 16.8% → 16.2%) → discrepancies có giải thích MỘT PHẦN nhưng **không phải root cause chính**.

→ **Finding strengthen**: phát hiện ablation bất thường không phải artifact của 3 discrepancies. Có thể nguyên nhân thực sự là (i) ablation implementation không tương đương paper (additive switch thay vì re-train kiến trúc rút gọn), hoặc (ii) discrepancy còn lại không sửa (Q4 num_types — chỉ ảnh hưởng v3.2), hoặc (iii) loss formulation khác paper (0.3*existence + 0.7*type không có trong Eq. 32).

---

## So sánh 3 phase: paper vs phase A vs phase B-C

| Metric | Paper | Phase A (orig code) | Phase B-C (3 fix) | Δ Phase A vs paper | Δ Phase B-C vs paper |
|---|---:|---:|---:|---:|---:|
| AUC (binary) | 0.9669 | 0.9738 | **0.9752** | +0.71% | **+0.86%** |
| AUPR | 0.9738 | 0.9671 | **0.9701** | -0.69% | -0.38% |
| F1 (binary) | 0.9278 | 0.9295 | 0.9298 | +0.18% | +0.22% |
| Top-1 Precision | 0.5842 | 0.5075 | **0.5176** | -13.13% | **-11.39%** |
| Top-1 Recall | 0.6341 | 0.5979 | **0.6010** | -5.71% | -5.22% |
| Top-1 F1 | 0.5970 | 0.5485 | **0.5521** | -8.12% | **-7.52%** |

**Kết luận quantitative**: Sửa 3 discrepancies cải thiện baseline metrics đều, gap vs paper rút từ -8.12% → -7.52% trên Top-1 F1. Cải thiện thực chất nhưng nhỏ → discrepancies KHÔNG là root cause chính của gap.

---

## Cách tiếp tục lần sau

### Option 1 — Hoàn thành Plan B (recommend)

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = 1

# Phase B-D: Case study (~50 phút)
python case_study.py *>&1 | Tee-Object logs\case_study.log

# Phase B-E: Update report
# (Manual edit generate_report.py để thêm Section 3.5, 3.6 — xem chi tiết bên dưới)
.\compile_final.ps1
```

**Phase B-E cụ thể** (sửa [generate_report.py](generate_report.py)):

1. **Section 3.2 (Code-paper discrepancies)**: thêm cột "Trạng thái fix" vào bảng — 3/4 đã fix, Q4 (num_types) skip.
2. **Section 3.4 (Ablation)**: thay số mới (xem bảng Phase B-C ở trên). Subsection 3.4.4 update kết luận: "Sửa 3 discrepancies KHÔNG fix pattern Fig. 4. Finding bất thường được strengthen như legitimate observation."
3. **Section 3.5 (NEW — Case study)**: load `results/case_study_summary.json`, render 2 bảng top-15 (breast, HCC) với cột `in_paper_top15`, summary "X/15 miRNAs khớp paper Table 5/6, Y/15 type cũng khớp".
4. **Section 3.6 (NEW — Updated reproduce summary)**: bảng "% reproduce" tổng kết (xem bảng so sánh ở trên).

### Option 2 — Bỏ B-D, chỉ làm B-E

Nếu không muốn tốn 50 phút thêm cho case study:
- Skip Phase B-D.
- Trong Phase B-E, ghi rõ Section 3.5 = "Skipped — không có case study reproduction".
- Vẫn update 3.2, 3.4, 3.6 với data mới đã có.

### Option 3 — Investigate sâu hơn (sau khi xong B-D, B-E)

Để lý giải tại sao gap Top-1 F1 vẫn -7.5% sau khi fix:
- Multi-seed evaluation (3 seeds × baseline) — đo variance.
- Sửa Q4 (num_types flexible) → train lại baseline.
- Profile training loss components để xem có loss nào collapsed không.

---

## Files đã modify trong Plan B (tracked changes)

| File | Thay đổi | Đã commit? |
|---|---|---|
| [param.py](param.py) | n_head 8→4 (line 35), update_graph_frequency 50→5 (line 160) | Chưa |
| [main_experiments_hetero1.py](main_experiments_hetero1.py) | λ₃ 0.15→1.0 (line 871), refactor block 793-810 (epoch-modulo update) | Chưa |
| [case_study.py](case_study.py) | NEW — full-data train + top-15 ranking | Chưa |
| [run_full_rerun.ps1](run_full_rerun.ps1) | NEW — orchestrator parallel 2 jobs | Chưa |
| [BaoCao_DHGCMDA_v1_before_fix.docx](BaoCao_DHGCMDA_v1_before_fix.docx) | NEW — snapshot báo cáo trước Plan B | Chưa |

**Khuyến nghị commit**: sau khi xong B-D + B-E, chạy `git status` để xem diff full, commit với message kiểu `"Plan B: fix 3 code-paper discrepancies + case study + updated report"`.

⚠️ **Cảnh báo bảo mật vẫn có**: GitHub Personal Access Token (`ghp_Ctx4...`) còn trong git remote URL. Cần revoke + reset remote — chi tiết ở response trước (đã ghi trong ExitPlanMode).

---

## Files cần backup nếu di chuyển workspace

- `requirements.txt`, `CLAUDE.md`, `EXPERIMENT_STATE.md`
- `logs/baseline_v2.0_full.log`, `logs/ablation_*.log` (cả old + new — old đã bị overwrite trong Phase B-C, chỉ còn new)
- `results/*.json`, `results/*.png`, `results/case_study_*.csv` (sau Phase B-D)
- `BaoCao_DHGCMDA.docx` + `BaoCao_DHGCMDA_v1_before_fix.docx`
- Tất cả source `.py` đã modify (param.py, hetero_model.py, generate_report.py, parse_metrics.py, generate_arch_figure.py, case_study.py)
- Plan: `C:\Users\hungld\.claude\plans\download-code-v-data-async-lovelace.md`
