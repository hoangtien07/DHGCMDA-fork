# Fig.4 ablation verify with REPRODUCE config (seed=1, K=7, DEFAULT loss).
#
# Context: K sweep tim ra (seed=1, K=7) cho Top-1 F1 = 0.5909, gap -1.0%
# vs paper 0.5970 — essentially reproduced. Now verify Fig.4 ablation
# pattern voi cau hinh nay.
#
# Paper Fig.4 claim: ALL 5 ablations (no_cl, no_hgcn, no_avf, no_hgt,
# no_dv) HURT baseline Top-1 F1.
#
# Estimated: 5 ablation x ~50 min = ~4h CPU.
#
# Usage:
#     .\run_fig4_verify.ps1
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"
$seed = 1
$K = 7
$ablations = @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')

Write-Host "================================================================"
Write-Host " FIG.4 VERIFY with REPRODUCE config"
Write-Host " seed=$seed, K_neigs=$K, DEFAULT loss (--exist_weight 0.3, two_head)"
Write-Host " Paper baseline T1-F1 = 0.5970"
Write-Host " Our baseline T1-F1 = 0.5909 (gap -1.0%)"
Write-Host "================================================================"

$idx = 0
foreach ($abl in $ablations) {
    $idx++
    Write-Host ""
    Write-Host "[$idx/$($ablations.Count)] ablation=$abl (seed=$seed, K=$K)"
    Write-Host "----------------------------------------------------------------"
    $log = "logs\fig4_verify_${abl}.log"
    & $py main_experiments_hetero1.py `
        --device cpu --seed $seed --K_neigs $K --ablation $abl `
        *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] ablation $abl exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " FIG.4 VERIFY SUMMARY"
Write-Host "================================================================"
& $py summarize_fig4_verify.py
