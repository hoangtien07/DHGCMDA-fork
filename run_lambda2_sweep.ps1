# Lambda2 sweep - paper Fig.2a reproduce check.
#
# Goal: paper Fig.2(a) sweep lambda2 (inter_view_weight) E {0.1, 0.3, 0.5}
# tim cau hinh closest to paper. Co the lambda2 la missing piece cho Fig.4
# reproduce.
#
# Note: --inter_view_weight da co san o param.py:112 (default 0.3). KHONG
# can sua code.
#
# Estimated: 3 lambda2 values x ~50 min = ~2.5h CPU.
#
# Usage:
#     .\run_lambda2_sweep.ps1
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"
$seed = 1
$K = 7
$lambda2_values = @(0.1, 0.5)  # 0.3 da co tu fig4_verify baseline

Write-Host "================================================================"
Write-Host " LAMBDA2 SWEEP - paper Fig.2a reproduce"
Write-Host " Fixed: seed=$seed, K=$K, DEFAULT loss (--exist_weight 0.3, two_head)"
Write-Host " lambda2 values: $($lambda2_values -join ', ')"
Write-Host " Note: lambda2=0.3 baseline da co tu fig4_verify (K=7, seed=1)"
Write-Host "================================================================"

$idx = 0
foreach ($l2 in $lambda2_values) {
    $idx++
    Write-Host ""
    Write-Host "[$idx/$($lambda2_values.Count)] lambda2=$l2 (seed=$seed, K=$K)"
    Write-Host "----------------------------------------------------------------"
    $log = "logs\lambda2_sweep_l2_${l2}.log"
    & $py main_experiments_hetero1.py `
        --device cpu --seed $seed --K_neigs $K --inter_view_weight $l2 `
        *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] lambda2=$l2 exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " LAMBDA2 SWEEP SUMMARY"
Write-Host "================================================================"
& $py summarize_lambda2_sweep.py
