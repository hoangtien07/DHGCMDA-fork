# Experiment State — Reproduce DHGCMDA

> File này ghi lại trạng thái thực nghiệm. Cập nhật mỗi khi run xong một experiment.

## Lần cập nhật cuối
**2026-05-09, hoàn thành toàn bộ Plan B (5/5 phase).**

---

## ✅ TRẠNG THÁI: HOÀN THÀNH PLAN B (mọi phase done)

- ✅ **Phase A (initial reproduce)**: baseline + 5 ablation với code gốc → snapshot báo cáo ở [BaoCao_DHGCMDA_v1_before_fix.docx](BaoCao_DHGCMDA_v1_before_fix.docx).
- ✅ **Phase B-A**: Đã sửa 3 code-paper discrepancies:
  1. `n_head` default: 8 → **4** (param.py:35)
  2. `update_graph_frequency` default: 50 → **5** (param.py:160)
  3. `λ₃_recon` weight: 0.15 → **1.0** (main_experiments_hetero1.py:871)
  4. Block dynamic graph update: MSE-threshold → epoch-modulo (main_experiments_hetero1.py:793-810).
- ✅ **Phase B-B**: Smoke test pass (không NaN, hypergraph update đúng kì).
- ✅ **Phase B-C**: Rerun baseline + 5 ablation parallel (~2.5h wall, Xeon E5-2680 v4 CPU).
- ✅ **Phase B-D**: Case study trained on 100% data (~9 min CPU), predict top-15 cho breast neoplasms + HCC. Score tensor cached tại [results/case_study_score.npy](results/case_study_score.npy).
- ✅ **Phase B-E**: Updated `generate_report.py` Section 3.2/3.4/3.5/3.6, regenerated `BaoCao_DHGCMDA.docx` (308 paragraphs, 11 tables).

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

## Kết quả Phase B-D (Case study)

### Setup
- Train DHGCMDA trên TOÀN BỘ 1498 associations (không CV split), 650 epochs (~8 phút CPU sau Plan B fixes).
- Predict tensor [495, 383, 5] (existence + 4 types).
- Rank top-15 miRNAs theo `max(P(type_k))` cho 2 disease.
- Cross-check với paper Table 5 (breast top-15) + Table 6 (HCC top-15).

### Kết quả

| Disease | Trùng paper | Type khớp | Paper báo confirmed |
|---|---:|---:|---:|
| Breast neoplasms (idx=49) | **1/15** | 0/15 | 13/15 (PMID) |
| Hepatocellular carcinoma (idx=58) | **0/15** | 0/15 | 12/15 (PMID) |

### Phát hiện CHÍNH (case study)

**Model collapse vào 1 type/disease**: 
- Top-15 cho breast: **TẤT CẢ 15 đều predict type = "target"** (score 0.989-0.996)
- Top-15 cho HCC: **TẤT CẢ 15 đều predict type = "epigenetics"** (score 0.994-0.996)

→ Confirm pattern class collapse — model không phân biệt type giữa các miRNAs cho cùng 1 disease, chỉ đổi type giữa các disease. Đây là evidence thêm cho thấy multi-type prediction head có vấn đề (ngoài pattern Fig. 4 đã note).

**Diễn giải khả dĩ**:
- Per-disease, 1 type chiếm đa số associations → model học "default type" cho mỗi disease, ranking tất cả miRNAs cùng kiểu.
- Class weighting (focal_gamma=2.5 + Effective Number) có thể đẩy về majority type per disease.
- Ranking criteria `max prob across types` thiên về type prediction strong nhất, mất đi distinction giữa miRNAs.

---

## Files output cuối Plan B

- [BaoCao_DHGCMDA.docx](BaoCao_DHGCMDA.docx) — báo cáo cuối cùng (~245 KB, 308 paragraphs, 11 bảng)
- [BaoCao_DHGCMDA_v1_before_fix.docx](BaoCao_DHGCMDA_v1_before_fix.docx) — snapshot trước Plan B
- [results/baseline_v2.0_metrics.json](results/baseline_v2.0_metrics.json) + [results/ablation_*.json](results/) (×5) — metrics Plan B
- [results/case_study_breast.csv](results/case_study_breast.csv), [results/case_study_hcc.csv](results/case_study_hcc.csv), [results/case_study_summary.json](results/case_study_summary.json), [results/case_study_score.npy](results/case_study_score.npy) — case study outputs
- [logs/](logs/) — tất cả training + case study logs (UTF-16 từ PowerShell Tee, parse_metrics auto-handle)
- [CLAUDE.md](CLAUDE.md), [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md) — context cho Claude session sau

---

## Cách tiếp tục lần sau (nếu muốn extend)

### Option 1 — Investigate thêm class collapse (recommend nếu muốn extend)

Phát hiện top-15 collapse vào 1 type/disease là evidence mạnh nhất hiện tại. Để verify:

```powershell
# Re-run case study với threshold khác (vd: log_softmax thay max raw prob)
# Hoặc multi-seed để đo variance
python case_study.py *>&1 | Tee-Object logs\case_study_seed42.log  # cần edit seed
```

Cần edit `case_study.py:269` thêm `args.seed = 42` (hoặc command line arg) để chạy seed khác.

### Option 2 — Implement ablation theo CHUẨN paper (re-train kiến trúc rút gọn)

Hiện tại `no_hgcn = identity G` chỉ là approximation. Để tương đương paper:
- `no_hgcn`: replace HGCN module bằng GCNConv thực thay vì identity → cần edit [hetero_model.py](hetero_model.py).
- `no_hgt`: bỏ luôn `node_transformers` (không chỉ skip hgt_layers) → re-init forward sequence.

Với cách này, có thể pattern Fig. 4 sẽ được tái lập.

### Option 3 — Multi-seed cho statistical significance

```powershell
foreach ($seed in 42, 100, 2024) {
  python main_experiments_hetero1.py --device cpu --seed $seed *>&1 | Tee-Object "logs\baseline_seed$seed.log"
}
```

Sau đó tổng kết mean ± std. Effort: ~1.5h cho 3 seeds × baseline.

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
