# Combo A: K=3, K=5 sweep + multi-seed verify best config.
#
# Goal: while waiting for CDMBlab authors reply, do 2 things:
#  1. Complete K sweep with K=3, K=5 (paper Fig.3 tests K=1,3,5,7,9,11,13,15)
#  2. Multi-seed verify best config (K=7, seed=1) with seeds 42, 7 to get error bar
#
# Estimated: 4 runs x ~50min = ~3.5h CPU.
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"

Write-Host "================================================================"
Write-Host " COMBO A: K=3,5 extend + Multi-seed verify (best config)"
Write-Host " Total: 4 runs x ~50min = ~3.5h CPU"
Write-Host "================================================================"

# Phase 1: K=3, K=5 with seed=1 (complete Fig.3 coverage)
Write-Host ""
Write-Host "[Phase 1/2] K=3 and K=5 sweep (seed=1, default config)"
Write-Host "----------------------------------------------------------------"

foreach ($K in @(3, 5)) {
    Write-Host ""
    Write-Host "K_neigs=$K (seed=1)"
    $log = "logs\k_sweep_K${K}_seed1.log"
    & $py main_experiments_hetero1.py --device cpu --seed 1 --K_neigs $K *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] K=$K exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Phase 2: Multi-seed verify (K=7 best config, seeds 42 and 7)
Write-Host ""
Write-Host "[Phase 2/2] Multi-seed verify best config (K=7, seeds 42 and 7)"
Write-Host "----------------------------------------------------------------"

foreach ($seed in @(42, 7)) {
    Write-Host ""
    Write-Host "K_neigs=7 (seed=$seed)"
    $log = "logs\multiseed_best_seed${seed}.log"
    & $py main_experiments_hetero1.py --device cpu --seed $seed --K_neigs 7 *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] seed=$seed exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " COMBO A SUMMARY"
Write-Host "================================================================"

# Re-run K sweep summary (now includes K=3, 5)
& $py summarize_k_sweep.py --seed 1
& $py summarize_multiseed_best.py

# Re-run final report
& $py final_reproduce_report.py

Write-Host ""
Write-Host "DONE Combo A." -ForegroundColor Green
