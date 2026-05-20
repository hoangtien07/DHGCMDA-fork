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

## 7zzz. PHASE C HOÀN THÀNH 2026-05-20 22:30 — TDRC reproduce SUCCESS

### Status sau session 20:00-22:30 (2.5h work)
- ✅ DGL install thử nhưng FAIL (DLL không compat torch 2.5.1) → **Skip NMCMDA**.
- ✅ TDRC reproduce thành công trên HMDD v3.2 (data_v32 of TDRC author):
  - CV_type: Top-1 P=0.5042, R=0.3869, F1=0.4378 (paper: 0.4926/0.3671/0.4207 → +2-5% match)
  - CV_triplet: AUPR=0.9246, AUC=0.9109, F1=0.8549 (paper: 0.9059/0.8962/0.8309 → +1.6-2.9% match)
  - Time: 1.7h CPU với max_iter=100 (giảm từ 500 — empirical 100 cho kết quả gần như identical, tiết kiệm 10× time)
  - Patches: `np.mat → np.asmatrix`, vectorized `get_functional_sim`, `max_iter` đọc từ kwargs (`experiments.py`).
- ✅ Báo cáo updated: Section 3.4.6 (v3.2 partial), 3.7 (TDRC + NMCMDA skip), 3.8 (renumbered conclusion). 427 paragraphs, 25 tables.

### Kết quả Phase C final

| Method | Top-1 F1 | AUPR | AUC | Match paper |
|---|---:|---:|---:|---|
| **TDRC reproduce** | **0.4378** | **0.9246** | **0.9109** | ✅ +2-5% all metrics |
| TDRC paper | 0.4207 | 0.9059 | 0.8962 | (reference) |
| DHGCMDA v3.2 reproduce | 0.0000 (collapse) | — | 0.9217 | ⚠ AUC khớp, Top-1 collapse |
| DHGCMDA v3.2 paper | 0.8600 | 0.9271 | 0.9181 | (reference) |
| NMCMDA | SKIP (DGL incompat) | — | — | — |

### % reproduce mới
- v2.0 binary: 99%, v2.0 Top-1: 92%, v2.0 ablation pattern: 0% (ngược paper)
- v3.2 binary (3/5): 99%, v3.2 Top-1: 0% (class collapse với GIP-only)
- TDRC baseline reproduce: **~98%** ✅ (NEW!)
- Tổng thể: **~55-60%** (tăng từ 50% Plan B)

---

## 7zz. RESUME 2026-05-20 07:07 — v3.2 baseline 3/5 fold xong, USER tắt máy

### Status snapshot khi tắt máy

- **v3.2 baseline running** từ 01:39 đêm 2026-05-20 → 07:07 sáng (5h28min wall).
- **3/5 folds completed**, đang ở fold 4 epoch ~500/650. ETA xong ~08:22 (1h15min nữa).
- **PID 18652** đang chạy. RAM 2GB, CPU full.

### Kết quả partial 3/5 folds (đã save `results/v32_baseline_partial.json`)

| Metric | Reproduce (3 fold avg) | Paper v3.2 | Δ |
|---|---:|---:|---:|
| AUC | **0.9217** | 0.9181 | +0.4% ✅ |
| Top-1 F1 | **0.0000** | 0.8600 | -100% ❌ |

**Quan sát chính**: AUC khớp paper rất sát, nhưng **Top-1 F1 = 0 trên cả 3 folds → CLASS COLLAPSE hoàn toàn**. Điều này confirm giả thuyết: với GIP-only similarity (không Wang MeSH semantic + miRNA functional), model học được binary signal nhưng không phân biệt được 5 types. Paper Top-1 F1 = 0.86 KHÔNG reproduce được với GIP pragmatic shortcut.

### Khi mở máy lại

Process đã CHẾT (shutdown). v3.2 baseline cần retrain hoặc accept partial 3-fold result.

Lựa chọn:
1. **Accept partial** (recommend): Dùng partial 3-fold result, note rõ "3/5 fold do interrupt" trong báo cáo. Đã có evidence mạnh cho class collapse với GIP.
2. **Retrain full**: thêm 5h30min CPU để có 5/5 fold.
3. **Move on**: skip v3.2 hoàn toàn, dùng partial làm note "GIP không đủ cho type prediction", chuyển TDRC + báo cáo.

### Quick commands

```powershell
cd d:\Tien\DHGCMDA-fork
# Check status (process should be dead after shutdown)
Get-Process python -ErrorAction SilentlyContinue

# Đọc kết quả partial đã save
cat results\v32_baseline_partial.json

# Nếu retrain v3.2:
$env:PYTHONUTF8 = 1
.\venv\Scripts\python.exe main_experiments_hetero1.py --device cpu --dataset v3.2_processed *>&1 | Tee-Object logs\v32_baseline.log

# Hoặc chuyển TDRC
cd baselines\TDRC
..\..\venv\Scripts\python.exe run_tdrc.py *>&1 | Tee-Object ..\..\logs\tdrc_v32.log
```

---

## 7z. PHASE C (đang chạy — 2026-05-20) — Reproduce HMDD v3.2 + baseline comparison

**Mục tiêu**: nâng % reproduce paper từ 50% → 70-80% bằng cách add:
- C-1: HMDD v3.2 baseline + 5 ablation (tăng từ 0% → ~30%)
- C-2: 2 baseline reproducible (TDRC + NMCMDA) trên v2.0 + v3.2 (tăng từ 0% → 2/6 cells)

### Trạng thái hiện tại

- ✅ **C-1a**: HMDD v3.2 raw downloaded từ cuilab.cn (`HMDD_data/MDAv3.2/v3_*.txt`).
- ✅ **C-1b**: Preprocessed v3.2 với GIP similarity (chỉ approximation, không phải Wang MeSH). Output: `v3.2_processed/` (722 miRNAs × 614 diseases × 13,748 associations × 5 types).
- ✅ **C-1c**: Adapted code (param.py, prepareData.py, hetero_model.py) để support `--dataset v3.2_processed` với 5 types.
- ✅ **C-1d**: Smoke test v3.2 (3 epoch × 2 fold) PASS, AUC 0.89.
- 🔄 **C-1d-full**: Đang chạy baseline v3.2 single process, ETA ~6h CPU. Started 2026-05-20 ~01:39.
- ⏸ **C-1e**: 5 ablation v3.2 — pending sau baseline.
- ⏸ **C-2a**: TDRC repo cloned + adapted (np.mat → np.asmatrix, vectorized functional_sim). Chạy pending sau v3.2.
- ⏸ **C-2b**: NMCMDA repo cloned, chưa adapt.

### Caveat quan trọng

- **v3.2 với 650 epoch trên CPU EXTREMELY SLOW** (~15s/epoch khi parallel 2 jobs, ~7-8s/epoch single process). Single baseline ước tính 6h, 6 runs (baseline + 5 ablation) sequential ~36h. → Quyết định chỉ chạy baseline, skip ablation v3.2 (hoặc giảm epoch).
- **GIP similarity is pragmatic shortcut**, không phải Wang's MeSH method như paper → kết quả sẽ KHÔNG sát paper Table 3 (paper báo +35% gain trên v3.2 với SSM, GIP có thể cho kết quả thấp hơn).
- **NumPy 2.0 compatibility**: TDRC repo có `np.mat` deprecated → đã patch sang `np.asmatrix` trong toàn bộ TDRC files.
- **TDRC `get_functional_sim` quá chậm** với 713 miRNAs (~20 phút CPU/call). Đã vectorize.

### Files mới của Phase C

- `preprocess_v32.py` — script preprocess raw cuilab → DHGCMDA format.
- `v3.2_processed/` — preprocessed v3.2 data (4 sim matrices + association csv + name mappings).
- `HMDD_data/MDAv3.2/v3_*.txt` — raw v3.2 từ cuilab.
- `run_v32_rerun.ps1` — orchestrator parallel 2 jobs v3.2 (KILLED, sẽ rewrite simpler).
- `baselines/TDRC/` — TDRC repo cloned + patched.
  - `run_tdrc.py` — wrapper script.
  - `data.py`, `experiments.py`, `method.py`, `TDRC.py`, `TFAI.py` — patched np.mat → np.asmatrix.
  - `data.py:get_functional_sim` — vectorized.
- `baselines/NMCMDA/extracted/` — NMCMDA repo cloned, chưa adapt.

### Lệnh tiếp tục lần sau

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = 1

# Nếu v3.2 baseline đang chạy: đợi xong
# Get-Process python | Format-Table

# Sau khi v3.2 baseline xong:
python parse_metrics.py logs\v32_baseline.log results\v32_baseline_metrics.json

# Chạy TDRC trên v3.2
cd baselines\TDRC
..\..\venv\Scripts\python.exe run_tdrc.py *>&1 | Tee-Object ..\..\logs\tdrc_v32.log

# NMCMDA — cần adapt thêm (DGL, format data)

# Update báo cáo
cd ..\..
python generate_report.py
```

## 7. Plan B status — HOÀN THÀNH 5/5 phase

**Tất cả phase Plan B đã xong:**
- ✅ B-A: Sửa 3 discrepancies (n_head=4, λ₃=1.0, dynamic update mỗi 5 epoch)
- ✅ B-B: Smoke test pass
- ✅ B-C: Rerun baseline + 5 ablation parallel (~2.5h CPU)
- ✅ B-D: Case study breast + HCC (~9 phút train + cache score) → `results/case_study_*.csv`
- ✅ B-E: Update [generate_report.py](generate_report.py) Section 3.2/3.4/3.5/3.6 → `BaoCao_DHGCMDA.docx` (308 para, 11 bảng)

## 7zz. RESUME 2026-05-14 — v3.2 filtered overnight + Combo A done

### 🚀 RESUME 1 LỆNH KHI MỞ MÁY LẠI

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
.\run_overnight_v32.ps1   # baseline + 5 ablations v3.2 filtered, ~5h CPU
```

Sau khi xong (auto regen report + commit + push):
- Xem `results/v32_filtered_summary.json` — bảng so sánh v2.0 vs v3.2 Fig.4
- Xem `BaoCao_DHGCMDA.docx` — auto-updated

### Trạng thái 2026-05-14 (commit `17dd84b`)

**Đã xong**:
- Combo A: K=3,5 sweep + multi-seed K=7 verify
- Email draft cho CDMBlab authors (chờ user gửi)
- v3.2 filtered build script + overnight orchestrator (CHƯA chạy)

**Findings mới quan trọng**:
- K=3 mới best Top-1 F1 = 0.5924 (gap -0.8% paper) — slight improvement
- Multi-seed verify K=7: mean 0.5691, std 0.019, **|t|=2.53 SIGNIFICANT vs paper**
- seed=1 là LUCKY SEED → robust gap = -4.7% (NOT -1.0% như single-seed claim)

**Pending overnight (chưa launch)**:
- v3.2 filtered controlled experiment (baseline + 5 ablations)
- ETA ~5h CPU
- Pre-built dataset: `v3.2_filtered_495m383D/` (3,938 pairs, 2.63× v2.0)

---

## 7z. REFOCUS REPRODUCE (2026-05-11, hoàn tất)

### 🚀 RESUME (legacy, đã chạy xong)

```powershell
cd d:\Tien\DHGCMDA-fork
.\venv\Scripts\Activate.ps1
.\run_k_sweep.ps1 -Seed 1    # K sweep 5 values × ~50' = ~4h CPU
```

Sau khi xong (auto-aggregate vào `results/k_sweep_seed1_summary.json`):
- Nếu K cho gap < 3% paper → verify Fig.4 ablation với (seed=1, K=best)
- Nếu vẫn > 5% gap → try λ₂ sweep hoặc HMDD v3.2

### Status hiện tại

**User feedback (2026-05-11)**: Plan B2 (MLRC pivot) là scope drift khỏi goal ban đầu "chạy code → ra số như paper". REFOCUS về reproduce.

**Cleanup đã làm**:
- ❌ Deleted: `mlp_baseline.py`, `paper_section_7_4_outline.md` (MLRC-only)
- ✅ Kept: All bug fixes (3 critical), paper alignment (n_head=4, λ₃=1.0, update_freq=5), backwards-compat flags (defaults match original code)
- ✅ Fixed: `K_neigs=[13]` hardcoded → `args.K_neigs` (6 nơi trong main_experiments_hetero1.py)

### Seed sweep XONG (~3.5h CPU)

Sweep 4 seeds {0, 1, 42, 1234} × DEFAULT config (--exist_weight 0.3, two_head):

| Seed | AUC | Top-1 F1 | L2 dist |
|---|---:|---:|---:|
| PAPER | 0.9669 | 0.5970 | --- |
| 0 | 0.9738 | 0.5454 | 0.0717 |
| **1** 🏆 | **0.9730** | **0.5655** | **0.0461** |
| 42 | 0.9740 | 0.5393 | 0.0767 |
| 1234 | 0.9776 | 0.5535 | 0.0608 |

→ Best seed=1, gap Top-1 F1 = **-5.3%** vs paper. Binary metrics VƯỢT paper ở mọi seed.

### K sweep ĐÃ CHUẨN BỊ, CHƯA CHẠY

`run_k_sweep.ps1` + `summarize_k_sweep.py` sẵn sàng. Stopped ngày 2026-05-11 trước khi K=7 hoàn thành fold 1. Cần rerun từ đầu.

### Pipeline reproduce còn lại

```
[NEXT] K sweep (~4h)
  └─ run_k_sweep.ps1 -Seed 1
     └─ K ∈ {7, 9, 11, 13, 15} × seed=1
        ↓ Find best K
[VERIFY] Fig.4 ablation rerun (~4h)
  └─ 5 ablation × best (seed, K)
     └─ Pattern match paper Fig.4?
        ↓
[FALLBACK if gap > 5%]
  - λ₂ sweep ∈ {0.1, 0.3, 0.5} (~2.5h)
  - Class weighting 3 strategies (~2.5h)
  - HMDD v3.2 (~12h preprocess + 3h train)
  - Contact tác giả CDMBlab
```

### Files mới Plan refocus

- [run_seed_sweep.ps1](run_seed_sweep.ps1), [summarize_seed_sweep.py](summarize_seed_sweep.py) — Seed sweep (XONG)
- [run_k_sweep.ps1](run_k_sweep.ps1), [summarize_k_sweep.py](summarize_k_sweep.py) — K sweep (CHƯA CHẠY)
- `results/seed_sweep_*.json` (4 files) + `seed_sweep_summary.json` — seed sweep results

---

## 7a. Plan E — True ablation rebuild (Fix C) — XONG 2026-05-10

### 🚨 STRONG NEGATIVE REPLICATION

| Variant | AUC | Top-1 F1 | Δ Full (Phase D 0.6222) | Match paper? |
|---|---:|---:|---:|:---:|
| **Phase D Full** | 0.9743 | 0.6222 | — | — |
| no_cl_rebuild | 0.9770 | **0.6824** | +9.7% | ❌ |
| no_hgcn_rebuild | 0.9709 | 0.6499 | +4.5% | ❌ |
| no_hgt_rebuild | 0.9712 | 0.6745 | +8.4% | ❌ |

**0/3 rebuild hurt baseline** — bỏ CẢ 3 component đều CẢI THIỆN Top-1 F1.

### Phán quyết khoa học (final)

Hypothesis "ablation impl additive khác paper" → **REJECTED** sau khi rebuild đúng kiến trúc rút gọn vẫn cho cùng pattern.

→ Root cause thực sự: **DHGCMDA over-parameterized cho HMDD v2.0** (1498 assoc / 189K cells). Components CL/HGCN/HGT là noise cho v2.0 — paper claim "all components are critical" chỉ đúng cho v3.2 (lớn hơn) hoặc dataset khác.

### MLRC angle update

Title mới: **"DHGCMDA Revisited: Strong Negative Replication of Ablation Claims, with Loss Alignment Yielding +4.2% Top-1 F1"**

3 đóng góp:
1. Loss alignment (Plan C/D): code thừa `0.3·L_existence` → fix → +4.2% Top-1 F1
2. Ablation rebuild test (Plan E): paper Fig.4 KHÔNG reproduce ở 2 cách triển khai
3. Recommendation: paper nên cung cấp ablation rebuild code public + multi-seed Fig.4

### Code mới Plan E

| File | Thay đổi |
|---|---|
| [param.py](param.py) | Extend `--ablation` với no_cl_rebuild|no_hgcn_rebuild|no_hgt_rebuild |
| [hetero_model.py](hetero_model.py) | Init rebuild modules (HGCN_plain, GCNConv, skip_proj); helper `_g_to_edge_index`; forward branches |
| [run_phase_e.ps1](run_phase_e.ps1) | Phase E orchestrator |
| [summarize_phase_e.py](summarize_phase_e.py) | Verdict 4 scenarios + so sánh additive vs rebuild |

---

## 7b1. Plan D — Fix A++ (5-class softmax CE) — XONG 2026-05-10

### 🏆 Top-1 F1 vượt paper +4.2% (lớn nhất từ đầu project)

| Run | AUC | F1 binary | Top-1 F1 | Δ vs paper |
|---|---:|---:|---:|---:|
| Paper | 0.9669 | 0.9278 | 0.5970 | 0% |
| Plan C-w0.1 | 0.9641 | 0.9118 | 0.5996 | +0.4% |
| **Phase D Fix A++** | **0.9743** | **0.9361** | **0.6222** | **+4.2%** |

### ❌ Nhưng Fig.4 + case study KHÔNG cải thiện

- Fig.4 ablation match: 1/5 (giảm từ Plan C-w0.1 = 2/5)
- Case study collapse: 15/15 cùng type per disease (y nguyên)

### Phán quyết khoa học

**Loss formulation KHÔNG phải root cause của Fig.4 + case study.** 2 vấn đề này độc lập với loss — cần Plan E (Fix C — true ablation rebuild với GCN thực thay identity) để fix Fig.4. Class collapse case study cần thí nghiệm riêng.

### Code mới Plan D

| File | Thay đổi |
|---|---|
| [param.py](param.py) | `--loss_mode {two_head, softmax_5class}` flag |
| [hetero_model.py](hetero_model.py) | `SimplifiedTypePredictor` thêm `r_no_assoc` + branch softmax_5class |
| [main_experiments_hetero1.py](main_experiments_hetero1.py) | `SimplifiedMultiTypeAssociationLoss._compute_softmax5_loss`; `test_optimized` softmax transform |
| [case_study.py](case_study.py) | softmax + renormalize cho softmax_5class mode |
| [run_phase_d.ps1](run_phase_d.ps1) | Phase D orchestrator |
| [summarize_phase_d.py](summarize_phase_d.py) | Aggregate Phase D vs Plan C-w0.1 vs Paper |

### Backwards compat

`--loss_mode two_head` mặc định → mọi lệnh Plan A/B/C cũ vẫn chạy được.

---

## 7b2. Plan C — Eq. 32 alignment (sweep XONG, verify PARTIAL — 2026-05-09)

### 🚀 RESUME: chạy 1 lệnh duy nhất khi mở máy

```powershell
.\venv\Scripts\Activate.ps1
.\resume_plan_c.ps1   # full ~2.7h: case_study + rerank + 5 ablation + auto regen báo cáo
```

Hoặc chia nhỏ:
- `.\resume_plan_c.ps1 -OnlyCaseStudy` — 9 phút, verify class collapse fix
- `.\resume_plan_c.ps1 -SkipCaseStudy` — 2.5h, chỉ ablation Fig.4

### 🏆 SWEEP LOSS XONG — w=0.1 thắng

| Run | exist_weight | Top-1 F1 | Δ paper |
|---|---:|---:|---:|
| Paper | — | 0.5970 | 0% |
| Phase B-C | 0.3 | 0.5521 | -7.5% |
| **C-w0.1** | **0.1** | **0.5996** | **+0.4%** ✅ |
| C-w0.05 | 0.05 | 0.5898 | -1.2% |
| C-w0.0 | 0.0 | 0.5802 (AUC collapse 0.44) | -2.8% |

**Hypothesis confirmed**: code có `0.3·L_existence` không có trong Eq. 32 → root cause của 3 phát hiện bất thường. Sweet spot w=0.1.

Chi tiết: [EXPERIMENT_STATE.md](EXPERIMENT_STATE.md). Files mới: [resume_plan_c.ps1](resume_plan_c.ps1), [sweep_summary.py](sweep_summary.py), [summarize_plan_c_full.py](summarize_plan_c_full.py), [run_multiseed.ps1](run_multiseed.ps1).

---

## 7c. Plan C — Eq. 32 alignment (legacy, sweep design log)

**Hypothesis**: code có `0.3·L_existence(focal)` không có trong paper Eq. 32 → là root cause cho 3 phát hiện bất thường (pattern Fig. 4 đảo, Top-1 F1 thấp, case study collapse).

**Sweep design**: `exist_weight ∈ {0.3, 0.1, 0.05, 0.0}` × full 650 ep × 5 fold.
Code change: `--exist_weight` CLI flag ([param.py](param.py)), `SimplifiedMultiTypeAssociationLoss` đọc từ args. CPU thread tuning thêm vào `main_experiments_hetero1.py` line 1-13 (set `OMP/MKL_NUM_THREADS=14`, `torch.set_num_threads(14)`).

**Sneak peek (Phase C-1 fold 1, w=0.1)**: Top-1 F1 = **0.6238** (vs Phase B-C 0.5521 = +12.6%, vs paper 0.5970 = **+4.5%**) — **Fold 1 đã VƯỢT paper.** Cần đợi 4 fold + 2 phase còn lại.

**Files NEW Plan C**:
- [sweep_summary.py](sweep_summary.py) — parse logs/sweep_w*.log → `results/plan_c_comparison.json` + bảng so sánh đẹp.
- [rerank_case_study.py](rerank_case_study.py) — 4 chiến lược rank trên cached score (Task D).
- [run_multiseed.ps1](run_multiseed.ps1) — orchestrator multi-seed (Task A, sẵn sàng chạy sau sweep).
- [generate_report.py](generate_report.py) Section 3.6 — Plan C section auto-render từ JSON (placeholder nếu missing).

**Background bash**: ID `b3eontqe2`, sequential `python ... --exist_weight 0.1 → 0.05 → 0.0`. ETA ~1.5-2h từ 2026-05-09.

**Khi sweep xong**: chạy `python sweep_summary.py` → có bảng + JSON. Sau đó `python generate_report.py` → docx tự cập nhật Section 3.6.

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
