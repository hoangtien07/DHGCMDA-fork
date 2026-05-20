# Wake-up Checklist — sáng 2026-05-20

> File này hướng dẫn bước-by-bước để check status v3.2 baseline (treo máy chạy qua đêm) + tiếp tục Phase C.

## Bước 1: Check status v3.2 baseline (30 giây)

```powershell
cd d:\Tien\DHGCMDA-fork

# Check Python process còn chạy không?
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, CPU, StartTime -AutoSize

# Check log file size + tail
Get-Item logs\v32_baseline.log | Select-Object Length, LastWriteTime
```

**Khả năng**:

### Case A — Python vẫn chạy
- Log có grow → đang ở fold nào đó. Đợi tiếp.
- Để check epoch hiện tại:
  ```powershell
  .\venv\Scripts\python.exe -c "import re; t=open('logs/v32_baseline.log','rb').read(); s=t.decode('utf-16-le' if t[:2]==b'\xff\xfe' else 'utf-8', errors='replace'); folds=re.findall(r'Fold (\d+) completed', s); ep=re.findall(r'Epoch (\d+),', s); print(f'Folds done: {len(folds)}/5, latest epoch: {ep[-1] if ep else \"-\"}'); [print(f'  Fold {f}: AUC={a}, Top-1 F1={f1}') for f,a,f1 in re.findall(r'Fold (\d+) completed - AUC: ([\d.]+), Top-1 F1: ([\d.]+)', s)]"
  ```

### Case B — Python KHÔNG còn chạy (đã xong hoặc crash)
- Check log tail để xem có "FINAL COMPREHENSIVE RESULTS" không:
  ```powershell
  .\venv\Scripts\python.exe -c "t=open('logs/v32_baseline.log','rb').read(); s=t.decode('utf-16-le' if t[:2]==b'\xff\xfe' else 'utf-8', errors='replace'); print(s[-2000:])"
  ```
- Nếu có FINAL COMPREHENSIVE RESULTS → DONE → tiếp Bước 2.
- Nếu lỗi/crash → xem traceback, retry.

## Bước 2: Parse v3.2 metrics (1 phút)

```powershell
$env:PYTHONUTF8 = 1
.\venv\Scripts\python.exe parse_metrics.py logs\v32_baseline.log results\v32_baseline_metrics.json
cat results\v32_baseline_metrics.json
```

So sánh với paper Table 3 v3.2:
- Paper: Top-1 P = 0.7915, Top-1 R = 0.9421, Top-1 F1 = 0.8600, AUPR = 0.9271, AUC = 0.9181, F1 = 0.8674
- Reproduce kỳ vọng: thấp hơn nhiều do dùng GIP thay vì Wang's MeSH semantic.

## Bước 3: Decision — có chạy ablation v3.2 không?

Mỗi ablation v3.2 ~6h. 5 ablation tuần tự = 30h.

**Option 1 (recommend)**: Skip ablation v3.2. Move on TDRC + NMCMDA + báo cáo.

**Option 2**: Chỉ chạy 1-2 ablation quan trọng (vd no_cl, no_hgt) để check pattern. Mỗi cái 6h trên CPU.

## Bước 4: Chạy TDRC trên v3.2 (~3-4h)

```powershell
cd d:\Tien\DHGCMDA-fork\baselines\TDRC
$env:PYTHONUTF8 = 1
..\..\venv\Scripts\python.exe run_tdrc.py *>&1 | Tee-Object ..\..\logs\tdrc_v32.log
```

**Lưu ý**:
- TDRC `get_functional_sim` đã vectorize, ~20 phút CPU/call.
- CV_type: 5 fold × 20 phút = ~2h
- CV_triplet: 5 fold × 10 negative samples = lâu hơn, có thể ~3-5h
- → Tổng TDRC ~4-7h

Khi xong, sẽ tạo `results/baseline_TDRC_v32.json`.

## Bước 5: NMCMDA setup (cần adapt thêm)

NMCMDA có code [baselines/NMCMDA/extracted/NMR-RGCN/main.py](baselines/NMCMDA/extracted/NMR-RGCN/main.py) dùng DGL (Deep Graph Library). Cần:

1. Install DGL:
   ```powershell
   .\venv\Scripts\python.exe -m pip install dgl
   ```
2. Check format data `baselines/NMCMDA/extracted/data/MCD4|MCD6|MCD20/`. MCD4 = 4 types (giống v2.0), MCD6 = 6 types.
3. main.py KHÔNG có evaluation block — chỉ training. Cần viết runner riêng để eval CV_type + CV_triplet.

**ETA**: 2-3h work + 2-4h train.

## Bước 6: Update báo cáo (~30 phút)

Sau khi có v3.2 baseline + TDRC + (optional) NMCMDA:

```powershell
# Update generate_report.py:
#   Section 3.4.6 (NEW): Reproduce v3.2 (1 baseline)
#   Section 3.7 (NEW): Comparison với 2 baselines (TDRC + NMCMDA nếu có)

.\compile_final.ps1
```

## Files quan trọng đã chuẩn bị

| File | Mục đích |
|---|---|
| [preprocess_v32.py](preprocess_v32.py) | Preprocess HMDD v3.2 raw → format DHGCMDA |
| [v3.2_processed/](v3.2_processed/) | v3.2 đã preprocess (722×614 × 5 types) |
| [run_v32_rerun.ps1](run_v32_rerun.ps1) | Orchestrator parallel 2 jobs cho v3.2 (KILLED, không dùng) |
| [baselines/TDRC/run_tdrc.py](baselines/TDRC/run_tdrc.py) | Wrapper chạy TDRC trên v3.2 |
| [baselines/TDRC/data.py](baselines/TDRC/data.py) | Đã patch + vectorize |
| [baselines/NMCMDA/extracted/](baselines/NMCMDA/extracted/) | NMCMDA code+data, chưa adapt |

## Kết quả Plan B (đã có, không thay đổi)

Xem [BaoCao_DHGCMDA.docx](BaoCao_DHGCMDA.docx) (308 paragraphs, 11 bảng).

- v2.0 baseline: AUC 0.9752, AUPR 0.9701, Top-1 F1 0.5521
- v2.0 ablation: 5 variants (w/o CL +12.4%, w/o HGT +16.2%, ...)
- Case study breast/HCC: 1/15 + 0/15 trùng paper (class collapse)

Phase C sẽ ADD vào báo cáo, không thay đổi gì v2.0.
