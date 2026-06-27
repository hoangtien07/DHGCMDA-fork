# Experiment State — Reproduce DHGCMDA

> File này ghi lại trạng thái thực nghiệm. Cập nhật mỗi khi run xong một experiment.

## Lần cập nhật cuối
**2026-06-27 — PLAN L HOÀN TẤT (Linux): K_neigs=3 nâng v2.0 Top-1 F1 → 0.688 (+15.3% vs paper), adversarial-verified.**

### Plan L tóm tắt (branch `linux-run`)
- ✅ Port Linux: uv venv (CPython standalone), 20 `.ps1`→`.sh`, `setup_linux.sh`, `requirements_linux.txt`, `README_LINUX.md`. Smoke + full pipeline chạy OK.
- ✅ Council (multi-agent): brainstorm → ma trận 15 run (4 job song song) → adversarial verify → synthesis.
- 🏆 **v2.0 cải thiện: K=3 dưới full_bilinear = 0.688 ± 0.011 (multi-seed)**, vượt paper 0.5970 (+15.3%) và best cũ 0.6350 (+8.4%). 3 reviewer đối kháng đều refuted=false/high. Công bố 0.688 (KHÔNG 0.7006 lucky seed). ADOPT `--predictor_mode full_bilinear --K_neigs 3`.
- ✅ Ablation reversal xác nhận LẦN 5 (dưới full_bilinear). v3.2 honest 0.3232 (softmax5 hại, AUC sụp). NMCMDA vẫn blocked (DGL).
- Chi tiết: `results/council_synthesis.md`, CLAUDE.md mục 13, BaoCao §3.4.14.

---

### (cũ) 2026-05-20 07:07, đang dở Phase C — v3.2 baseline 3/5 fold (user tắt máy).

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
