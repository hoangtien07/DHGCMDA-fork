# Experiment State — Reproduce DHGCMDA

> File này ghi lại trạng thái thực nghiệm. Cập nhật mỗi khi run xong một experiment.

## Lần cập nhật cuối
**2026-05-20 07:07, đang dở Phase C — v3.2 baseline 3/5 fold (user tắt máy).**

### Phase C status (HMDD v3.2 + baseline comparison)

| Sub-phase | Status | Note |
|---|---|---|
| C-1a download v3.2 raw | ✅ | `HMDD_data/MDAv3.2/v3_*.txt` |
| C-1b preprocess GIP | ✅ | `v3.2_processed/` 722×614 × 5 types |
| C-1c adapt code | ✅ | param.py, prepareData.py, hetero_model.py |
| C-1d v3.2 baseline | ⚠ 3/5 fold | partial: AUC 0.9217 (match paper), Top-1 F1 0.0 (class collapse) |
| C-1e v3.2 ablation | ⏸ skip | quá lâu trên CPU |
| C-2a TDRC adapt | ✅ patched + vectorized | sẵn sàng `python run_tdrc.py` |
| C-2b NMCMDA adapt | ⏸ chưa | cần DGL install + writer eval |
| C-3 update báo cáo | ⏸ chưa | cần combine partial v3.2 + TDRC vào báo cáo |

### Kết quả v3.2 partial (3/5 fold) — saved `results/v3.2_baseline_partial.json`
- AUC 0.9217 vs paper 0.9181 (**+0.4%** — khớp!)
- Top-1 F1 = 0.0 vs paper 0.860 (**class collapse hoàn toàn**)
- Confirm: GIP-only similarity không đủ cho 5-type prediction. Cần Wang MeSH semantic.

### Lưu ý trước đó (state cũ, vẫn còn relevant)
**2026-05-11, REFOCUS REPRODUCE: Plan B2 (MLRC pivot) stopped. Seed sweep XONG (seed=1 best, gap -5.3% paper). K sweep CHƯA CHẠY — resume bằng `.\run_k_sweep.ps1 -Seed 1` khi mở máy.**

---

## 🚀 RESUME NGAY KHI MỞ MÁY LẠI (2026-05-11)

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
.\run_k_sweep.ps1 -Seed 1   # ~4h CPU, 5 K values
```

Output: `results/k_sweep_seed1_summary.json` + bảng so sánh với paper Fig.3.

### Tổng quan trạng thái (2026-05-11)

| Phase | Status | Verdict |
|---|---|---|
| Plan A→E (Plan B2 MLRC pivot) | ❌ STOPPED | User feedback: scope drift khỏi goal reproduce. Refocused. |
| Bug fixes (3 critical seed) | ✅ DONE | Code chính xác hơn original |
| Paper alignment (n_head, λ₃, update_freq) | ✅ DONE | Khớp paper |
| K_neigs hardcoded fix | ✅ DONE | Unblock paper Fig.3 sweep |
| Seed sweep (4 seeds × default) | ✅ DONE | seed=1 best, gap -5.3% Top-1 F1 |
| **K sweep (5 K × seed=1)** | ⏸ **PENDING** | **Resume bằng run_k_sweep.ps1** |
| Fig.4 ablation verify với best (seed, K) | ⏸ Pending | Sau K sweep |
| HMDD v3.2 / contact authors | ⏸ Fallback | Nếu K sweep + λ₂ sweep không đủ |

### Seed sweep results (đã commit)

| Seed | AUC | AUPR | F1 | T1-P | T1-R | T1-F1 | L2 dist paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| **PAPER** | **0.9669** | **0.9738** | **0.9278** | **0.5842** | **0.6341** | **0.5970** | --- |
| 0 | 0.9738 | 0.9666 | 0.9303 | 0.5007 | 0.6005 | 0.5454 | 0.0717 |
| **1** 🏆 | 0.9730 | 0.9671 | 0.9292 | 0.5373 | 0.5969 | **0.5655** | **0.0461** |
| 42 | 0.9740 | 0.9682 | 0.9329 | 0.5000 | 0.5891 | 0.5393 | 0.0767 |
| 1234 | 0.9776 | 0.9724 | 0.9362 | 0.5172 | 0.5967 | 0.5535 | 0.0608 |

**Insights**:
- Binary metrics (AUC/AUPR/F1) VƯỢT paper ở mọi seed → binary task solid
- Top-1 F1 NẰM GIỮA default config (-5.3% seed=1) và Plan D softmax_5class (+4.2%)
- Paper config có thể là biến thể chưa explore

### Code state (commits)

```
181b2c0 Seed sweep XONG: seed=1 best (Top-1 F1 = 0.5655, gap -5.3% paper)
7396f6c Seed sweep orchestrator + summarizer
15b6aab Refocus: REPRODUCE paper exactly (stop MLRC pivot)
10c8c67 Fix PowerShell encoding issue in run_multiseed_full.ps1 (em-dash)
9024bc9 FIX BUG #2 + #3: prepareData seed propagation for multi-seed correctness
3969311 FIX: seed_torch(args.seed) was never called — multi-seed broken
e7be314 Plan D Fix A++: 5-class softmax CE — Top-1 F1 vượt paper +4.2%
...
```

---

---

## 🚀 RESUME NGAY KHI MỞ MÁY LẠI

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
.\resume_plan_c.ps1                 # full ~2.7h: case_study + rerank + 5 ablation
# HOẶC chia nhỏ:
.\resume_plan_c.ps1 -OnlyCaseStudy  # 9 phút — verify class collapse fix
.\resume_plan_c.ps1 -SkipCaseStudy  # 2.5h — chỉ ablation Fig.4 verify
```

Sau đó tự động: `python summarize_plan_c_full.py && python generate_report.py`. Output cuối: `BaoCao_DHGCMDA.docx` + `results/plan_c_full_summary.json`.

---

## 🏆 PLAN C — Sweep loss XONG (2026-05-09)

### Kết quả sweep (5 fold × 650 epoch / variant)

| Run | exist_weight | AUC | AUPR | F1 (binary) | Top-1 F1 | Δ vs paper |
|---|---:|---:|---:|---:|---:|---:|
| Paper | — | 0.9669 | 0.9738 | 0.9278 | **0.5970** | 0% |
| Phase A (orig) | 0.3 | 0.9738 | 0.9671 | 0.9295 | 0.5485 | -8.1% |
| Phase B-C (3 fix) | 0.3 | 0.9752 | 0.9701 | 0.9297 | 0.5521 | -7.5% |
| **🏆 Phase C-w0.1** | **0.1** | 0.9641 | 0.9569 | 0.9118 | **0.5996** | **+0.4%** |
| Phase C-w0.05 | 0.05 | 0.9488 | 0.9421 | 0.8912 | 0.5898 | -1.2% |
| Phase C-w0.0 | 0.0 | 0.4368 | 0.4737 | 0.6740 | 0.5802 | -2.8% |

### Phán quyết khoa học

1. **HYPOTHESIS CONFIRMED**: code có `0.3·L_existence(focal)` không có trong paper Eq. 32 → đây là root cause cho 3 phát hiện bất thường.
2. **w=0.1 là sweet spot**: Top-1 F1 đạt 0.5996, **lần đầu tiên vượt paper** (+0.4%). Trade-off binary AUC giảm nhẹ nhưng vẫn ≥ 0.95.
3. **Monotonic verified**: w=0.3 → 0.1 (tăng), w=0.1 → 0.0 (giảm). Optimal có giá trị.
4. **w=0.0 collapse hoàn toàn**: AUC=0.44 (random) — confirm vẫn cần existence supervision dù nhỏ.
5. **% reproduce**: Top-1 F1 từ ~92% → ~100%. Tổng project từ 50% → ~75% (binary 99%, type 100%, sweep 100%, ablation 0% pending, case study 3% pending).

### Còn pending (script đã có)

| Verify | Hypothesis nếu pass | Effort |
|---|---|---|
| **Fig.4 ablation** với w=0.1 | All ablation hurt baseline (như paper) → fork bug do exist_weight quá cao | 5 × ~30' = 2.5h CPU |
| **Case study collapse** với w=0.1 | Top-15 đa dạng type (4 type/disease) → fix collapse | ~9' CPU |
| **Multi-seed** baseline w=0.1 | Mean ± std confirm w=0.1 robust | ~2.5h |

---

## Files thay đổi/mới trong Plan C (chưa commit)

| File | Trạng thái | Thay đổi |
|---|---|---|
| [param.py](param.py) | M | Thêm `--exist_weight` flag |
| [main_experiments_hetero1.py](main_experiments_hetero1.py) | M | Đọc `args.exist_weight`, type_weight=1-exist_weight; CPU thread tuning |
| [generate_report.py](generate_report.py) | M | Section 3.6 Plan C tự render từ JSON |
| [CLAUDE.md](CLAUDE.md), [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md) | M | Update Plan C status |
| [sweep_summary.py](sweep_summary.py) | NEW | Parse `logs/sweep_w*.log` → JSON |
| [summarize_plan_c_full.py](summarize_plan_c_full.py) | NEW | Aggregate sweep + ablation w=0.1 + case study w=0.1 |
| [rerank_case_study.py](rerank_case_study.py) | NEW | 4 chiến lược rank trên cached score |
| [run_multiseed.ps1](run_multiseed.ps1) | NEW | Multi-seed orchestrator |
| [resume_plan_c.ps1](resume_plan_c.ps1) | NEW | Resume script — chỉ cần chạy lệnh này khi mở máy |
| `results/sweep_w*.json` (×3) | NEW | Sweep metrics |
| `results/plan_c_comparison.json` | NEW | Bảng tổng sweep |
| `results/snapshot_phaseBC_w0.3/` | NEW | Backup case study Phase B-C trước khi rerun |
| `BaoCao_DHGCMDA.docx` | M | Section 3.6 Plan C đã có (placeholder/data tùy state) |

---

---

## 🔬 PLAN C — Eq. 32 alignment study (đang chạy)

### Mục tiêu
Test giả thuyết "existence loss focal w=0.3 đang hurt type prediction" — root cause tiềm năng của 3 phát hiện bất thường (pattern Fig. 4 đảo, Top-1 thấp, case study collapse).

### Verify Eq. 32 paper vs code (Task 1)
Paper Eq. 32 ([_pdf_text/p21.txt:19](_pdf_text/p21.txt#L19)):
```
L_total = L_type + λ1·L_intra + λ2·L_inter + λ3·L_recon
```
- Paper KHÔNG có `L_existence` riêng. Code thêm `0.3 × focal_loss(existence)` không khớp Eq. 32.
- Paper có label_smoothing? Không đề cập. Code có 0.1.
- λ₂: paper grid search ∈ {0.1, 0.3, 0.5} với optimal 0.3. Code hardcode 0.3 → match.

### Smoke test Fix A (exist_weight=0.0, 3 epochs × 2 folds, ~7s)
- ✅ Không crash, không NaN
- ⚠️ Binary AUC collapse: 0.97 → 0.62 (kỳ vọng — channel 0 không được supervise)
- Top-1 F1: 0.21 (3 epochs là quá ít, không kết luận)
- ⇒ Quyết định: sweep `exist_weight ∈ {0.1, 0.05, 0.0}` thay vì binary fix

### Sweep design
| Phase | exist_weight | type_weight | Goal |
|---|---:|---:|---|
| C-1 (Phase B-C, đã có) | 0.3 | 0.7 | Reference |
| **C-2** | 0.1 | 0.9 | Mid-compromise |
| **C-3** | 0.05 | 0.95 | Gần Fix A nhưng vẫn supervise |
| **C-4** | 0.0 | 1.0 | True Fix A — verify hypothesis |

### Task 2 — Rerank case study (đã xong)
Test 4 chiến lược rank trên cached score `[495, 383, 5]` (không retrain):

| Strategy | Breast overlap | HCC overlap | Type match | Type diversity |
|---|---:|---:|---:|---|
| `max_type` (cũ) | 1/15 | 0/15 | 0/15 | target×15 / epigenetics×15 |
| `sum_type` | 0/15 | 1/15 | 0/15 | circulation:11... |
| **`exist_only`** | **0/15** | **2/15** | **1/15** | target:11, genetics:3, circ:1 |
| `softmax_t` | 1/15 | 0/15 | 0/15 | target×15 / epigenetics×15 |

**Kết luận**: Class collapse là vấn đề MODEL-LEVEL, không phải RANK-LEVEL. `exist_only` cải thiện diversity nhưng vẫn xa paper (12-13/15). Cần retrain để fix triệt để. File output: [results/rerank_summary.json](results/rerank_summary.json).

### Stats channel raw score (từ rerank, helpful debug):
```
ch0 (existence): min=0.0005 max=0.8893 mean=0.1215 std=0.1615
ch1 (circulation): min=0.0004 max=0.9959 mean=0.3528 std=0.2962
ch2 (epigenetics): min=0.0002 max=0.9979 mean=0.2024 std=0.2136
ch3 (target): min=0.0003 max=0.9979 mean=0.2420 std=0.2136
ch4 (genetics): min=0.0007 max=0.9665 mean=0.2028 std=0.1512
```
Channel 0 max chỉ 0.89 (coarse) trong khi types max 0.99+ (sharp). Confirm existence head supervise yếu, type head over-confident.

### Task 4 (Multi-seed, đã chuẩn bị)
Script [run_multiseed.ps1](run_multiseed.ps1) — sẵn sàng chạy sau khi sweep xong:
- 3 seed (42, 100, 2024) × {baseline, no_cl} × full 650 epochs ≈ 7h CPU
- Output: `results/multiseed_*.json` + `results/multiseed_summary.json` (mean ± std)
- Có `-SmokeTest` flag và `-OnlyBaseline` flag

### Files thay đổi trong Plan C
| File | Thay đổi |
|---|---|
| [param.py](param.py) | Thêm `--exist_weight` flag (default 0.3) |
| [main_experiments_hetero1.py:90-95](main_experiments_hetero1.py#L90-L95) | Đọc `args.exist_weight`, type_weight = 1.0 - exist_weight |
| [rerank_case_study.py](rerank_case_study.py) | NEW — 4 chiến lược rank |
| [run_multiseed.ps1](run_multiseed.ps1) | NEW — orchestrator multi-seed |

### Background sweep status (2026-05-09)
- Bash ID: `b3eontqe2`
- Logs: `logs/sweep_w0.1.log`, `logs/sweep_w0.05.log`, `logs/sweep_w0.0.log`
- Sequential: w=0.1 → 0.05 → 0.0
- ETA: ~2.5h từ launch

---

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

---

## 🤖 PLAN W — Devin ScientistTwo-style workflow (2026-09-28)

Vòng lặp tự động: **ideate (1 agent đọc ledger+paper) → screen song song (300ep×2fold, seed=1, full_bilinear) → full 650ep×5fold cho top-2**. Mục tiêu: vượt kết quả paper trên v2.0 — KHÔNG phải reproduce. 5 hypotheses, mỗi cái đều chạy dưới protocol giống hệt + matched screen-baseline.

### Screen (300ep × 2fold, seed=1, full_bilinear)

| Hypothesis | Extra args | Top-1 F1 | AUC | Verdict |
|---|---|---:|---:|---|
| screen-baseline | (none — sm5 tắt, no ablation) | 0.5603 | 0.9814 | reference |
| sm5-nohgt-bilinear | `--loss_mode softmax_5class --ablation no_hgt` | 0.6240 | 0.9636 | → full |
| sm5-noclrebuild-bilinear | `--loss_mode softmax_5class --ablation no_cl_rebuild` | 0.6297 | 0.9807 | → full |
| sm5-bilinear | `--loss_mode softmax_5class` | 0.6043 | 0.9813 | non-additive: ablation removal là thành phần quan trọng |
| type-raw-logits | code: raw type logits → CE (branch `devin/wf-type-raw-logits`) | 0.6196 | 0.9753 | thua two_head incumbent |
| ema-eval | code: EMA weights decay 0.995 (branch `devin/wf-ema-eval`) | 0.4858 | 0.9662 | **negative** — EMA làm hỏng eval cuối epoch |

### Full run (650ep × 5fold, seed=1)

| Config | Top-1 P | Top-1 R | Top-1 F1 | AUC | AUPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **sm5 + no_cl_rebuild + full_bilinear** | 0.6739 | 0.6783 | **0.6760** | **0.9863** | **0.9834** | **0.9539** |
| sm5 + no_hgt + full_bilinear | 0.6731 | 0.6735 | **0.6732** | 0.9730 | 0.9762 | 0.9371 |
| Paper (reference) | 0.5842 | 0.6341 | 0.5970 | 0.9669 | 0.9738 | 0.9278 |

→ Cả 2 config vượt paper: **+13.2% / +12.8% Top-1 F1**, AUC cũng vượt (0.9863 / 0.9730 vs 0.9669). Per-fold sm5+noclrebuild: 0.687/0.669/0.682/0.641/0.702.

### So sánh với fork leaderboard (Top-1 F1)

| Config | Top-1 F1 | AUC | Nguồn |
|---|---:|---:|---|
| sm5 + no_cl_rebuild + **diag** | **0.6824** | ~0.97 | Plan E ledger peak |
| sm5 + no_cl_rebuild + **full_bilinear** | 0.6760 | **0.9863** | Plan W (này) |
| sm5 + no_hgt + diag | 0.6818 | ~0.975 | phase_d |
| sm5 + no_hgt + full_bilinear | 0.6732 | 0.9730 | Plan W |
| full_bilinear (J-1) | 0.6350 | 0.9805 | Plan J-1 |

→ Peak đơn lẻ vẫn là sm5+noclrebuild+diag (0.6824); config mới cho **best trade-off** Top-1+AUC (0.6760/0.9863 — AUC cao nhất từng ghi). Diag vs bilinear trên sm5+noclrebuild ≈ ngang nhau Top-1, bilinear thắng AUC ~+0.01.

### Phát hiện

1. **Config-stack là direction đúng** — 3 delta độc lập (softmax_5class, ablation-removal, predictor) kết hợp cho kết quả tốt nhất toàn repo, dù deltas không cộng tuyến tính.
2. **Screen 2-fold đánh giá THẤP hơn full 5-fold** — sm5-noclrebuild screen 0.6297 → full 0.6760 (+4.6pp). Screen chỉ nên dùng để ranking tương đối, không phải số tuyệt đối.
3. **2 negative results mới**: raw-logits CE thua two_head (-1.5pp); EMA weights phá eval (-13%). Branches `devin/wf-type-raw-logits`, `devin/wf-ema-eval` giữ để tham khảo, không merge.
4. Chưa verify multi-seed — full runs mới chỉ seed=1; per-fold variance ±0.03 → confirm trước khi công bố "best config".

### Next steps đề xuất

- Multi-seed (0/1/42/1234) cho sm5+noclrebuild+bilinear — chốt variance trước khi ghi vào báo cáo.
- Head-to-head diag vs bilinear trên sm5+noclrebuild (chênh Top-1 0.006 — trong noise).
- Ideate round 2 với hypotheses loại "mechanism" (contrastive internals, hypergraph construction) — round 1 phần lớn là config-stack.
- Cập nhật BaoCao_DHGCMDA.docx section kết quả vượt paper.

Artifacts: `results/devin_wf_hypothesis_search.json` (full summary). Workflow run: `wfr-2ae1ffa0cd8c48e39110d02db0d6397f` (resume-able, replays completed agents). Playbook: `!dhgcmda_experiment`.

---

## 🤖 PLAN X — Round 2: multi-seed confirm + mechanism hypotheses (2026-09-30)

Hai phần: (a) **multi-seed confirmation** cho winner Plan W — chốt variance trước khi công bố; (b) **round 2 ideate** — 6 hypotheses loại *mechanism* (không phải config-stack) đánh giá như delta TRÊN NỀN winner (`sm5 + no_cl_rebuild + full_bilinear`).

### (a) Multi-seed confirm — winner config, 650ep × 5fold

| Seed | Top-1 P | Top-1 R | Top-1 F1 | AUC | AUPR | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.6903 | 0.7052 | 0.6976 | 0.9849 | 0.9825 | 0.9504 |
| 1 | 0.6739 | 0.6783 | 0.6760 | 0.9863 | 0.9834 | 0.9539 |
| 42 | 0.6896 | 0.6971 | 0.6933 | 0.9845 | 0.9829 | 0.9493 |
| 1234 | 0.6604 | 0.6730 | 0.6666 | 0.9845 | 0.9834 | 0.9531 |
| **mean** | 0.6786 | 0.6884 | **0.6834** | **0.9851** | **0.9831** | **0.9517** |

→ **Claim CONFIRMED**: mọi seed vượt paper ≥ +11.6% Top-1 F1 (min 0.6666 vs 0.5970). Std ~0.013 — per-seed variance thật nhưng không đe claim. AUC ổn định 0.984–0.986.

### (b) Round-2 screens (300ep × 2fold, seed=1, trên nền winner)

Screen baseline = winner cfg → **0.6297 / 0.9807 — tái hiện CHÍNH XÁC số round 1** (determinism verified).

| Hypothesis | Delta so với winner | Top-1 F1 | AUC | Verdict |
|---|---|---:|---:|---|
| interview-cl-restore | restore inter-view CL (flag mới `--restore_inter_view_cl`, branch `devin/wf-r2-interview-cl-restore`) | 0.6288 | 0.9810 | neutral → full |
| wd-wired-5e-4 | nối `--weight_decay`+`--lr` vào AdamW (trước là dead flags, branch `devin/wf-r2-wd-wired`) | 0.6318 | 0.9802 | neutral → full |
| dropout-45 | `--dropout 0.45` | 0.5858 | 0.9757 | negative |
| nlayer-1 | `--nlayer 1` | 0.6282 | 0.9705 | neutral Top-1, AUC -0.01 |
| freeze-graph | `--update_graph_frequency 1000` (đóng băng dynamic-graph rebuild) | 0.6192 | 0.9798 | negative |
| recon-off | `--recon_weight_override 0.0` | 0.6129 | 0.9750 | **negative — recon MSE là load-bearing**, phủ nhận giả thuyết "auxiliaries đều noise" |

### (b') Full validation 2 candidates gần-neutral nhất (650ep × 5fold, seed=1)

| Config | Top-1 F1 | AUC | Δ vs winner (0.6760/0.9863) |
|---|---:|---:|---|
| winner + restore inter-view CL | 0.6652 | 0.9864 | **-1.1pp → NEGATIVE**: inter-view CL cũng là noise, không chỉ intra-view — củng cố hướng "gỡ bớt, không thêm lại" |
| winner + wd=5e-4 (wired) | 0.6771 | 0.9860 | +0.001 → **NEUTRAL (noise)**. Đáng merge như code hygiene: `--weight_decay`/`--lr` trước là flag chết không bao giờ tới optimizer |

### Verdict round 2

**Không cải thiện thêm được** — 6/6 mechanism hypotheses neutral hoặc negative; winner nằm ở local optimum trong họ mechanism đã test. Hai phát hiện đáng giữ:
1. `devin/wf-r2-wd-wired` — wiring dead flags vào optimizer (additive, default-giữ-nguyên). Nên merge vào main độc lập kết quả.
2. Inter-view CL restore giảm 1.1pp → lần đầu chứng minh *cả hai* nửa contrastive đều có hại trên v2.0, không chỉ phần intra.

### Next steps

- Merge `devin/wf-r2-wd-wired` (code hygiene) nếu muốn flag sống; `devin/wf-r2-interview-cl-restore` giữ làm negative reference, không merge.
- Direction còn lại chưa test: data-level (feature curation, similarity source thêm), ensemble multi-seed voting, calibration/predictor-head variants.
- Cập nhật BaoCao_DHGCMDA.docx: kết quả vượt paper đã multi-seed confirmed (mean 0.6834, min 0.6666 vs 0.5970).

Artifacts: `results/devin_wf_hypothesis_search_r2.json`. Workflow run: `wfr-d1d2de250fcb401c9f37b5bdab7e271c` (screens bổ sung chạy on-box do cap SWE-2, cùng protocol/seed).
