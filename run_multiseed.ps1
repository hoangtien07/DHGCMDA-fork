# Multi-seed reproducibility — chạy baseline + ablation w/o CL với 3 seed:
# 42, 100, 2024 để đo variance. Phục vụ statistical significance trong report.
#
# Tổng thời gian ước tính: ~7h CPU (3 seed × 2 variant × ~50min). Run sau
# khi sweep loss weight (run_full_rerun.ps1) đã xong.
#
# Cách dùng:
#     .\run_multiseed.ps1                 # full
#     .\run_multiseed.ps1 -SmokeTest      # 3 epochs × 2 folds × 6 jobs ≈ 1 phút
#     .\run_multiseed.ps1 -OnlyBaseline   # skip ablation, chỉ chạy baseline 3 seed (~2.5h)
#
param(
    [switch]$SmokeTest = $false,
    [switch]$OnlyBaseline = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$seeds = @(42, 100, 2024)

if ($SmokeTest) {
    $extraArgs = '--epoch', '3', '--validation', '2'
    Write-Host "[Smoke test] 3 epochs x 2 folds per job" -ForegroundColor Yellow
} else {
    $extraArgs = @()
    Write-Host "[Full] 650 epochs x 5 folds per job" -ForegroundColor Cyan
}

# Variant list: tuple (label, ablation_arg)
if ($OnlyBaseline) {
    $variants = @(@('baseline', ''))
} else {
    $variants = @(@('baseline', ''), @('no_cl', 'no_cl'))
}

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $label = $variant[0]
        $abl = $variant[1]

        Write-Host ""
        Write-Host "================================================================"
        Write-Host " SEED=$seed VARIANT=$label"
        Write-Host "================================================================"

        $logFile = "logs\multiseed_${label}_seed${seed}.log"
        $jsonFile = "results\multiseed_${label}_seed${seed}.json"

        $cmd = @('main_experiments_hetero1.py', '--device', 'cpu', '--seed', $seed)
        if ($abl) { $cmd += @('--ablation', $abl) }
        if ($extraArgs.Count -gt 0) { $cmd += $extraArgs }

        & ".\venv\Scripts\python.exe" @cmd *>&1 | Tee-Object -FilePath $logFile

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] $label seed=$seed exited with $LASTEXITCODE" -ForegroundColor Red
        }

        Write-Host "[Parse] $logFile -> $jsonFile"
        & ".\venv\Scripts\python.exe" parse_metrics.py $logFile $jsonFile
    }
}

# Tổng hợp mean ± std
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATING MULTI-SEED STATS"
Write-Host "================================================================"

$pyScript = @"
import json, glob, os
import numpy as np
from collections import defaultdict

files = glob.glob('results/multiseed_*.json')
groups = defaultdict(list)
for f in files:
    base = os.path.basename(f).replace('.json', '').replace('multiseed_', '')
    parts = base.rsplit('_seed', 1)
    label = parts[0]
    with open(f, 'r', encoding='utf-8') as fh:
        groups[label].append(json.load(fh))

agg = {}
keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
for label, runs in groups.items():
    stats = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and r[k] is not None]
        if vals:
            stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
    agg[label] = stats

with open('results/multiseed_summary.json', 'w', encoding='utf-8') as f:
    json.dump(agg, f, indent=2, ensure_ascii=False)

print('=== Multi-seed mean ± std ===')
for label, stats in agg.items():
    print(f'\n{label}:')
    for k, v in stats.items():
        print(f'  {k}: {v["mean"]:.4f} ± {v["std"]:.4f}  (n={v["n"]})')
"@

$pyScript | & ".\venv\Scripts\python.exe" -

Write-Host "Saved: results\multiseed_summary.json"
