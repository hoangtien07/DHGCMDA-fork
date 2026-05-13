# Overnight chain: v3.2 filtered baseline + 5 Fig.4 ablations.
#
# Goal: controlled experiment - thay v2.0 associations bang v3.2 associations
# (filter to v2.0 entities), KEEP v2.0 similarity. Test Fig.4 pattern voi
# new associations.
#
# Pre-requisite: build_v32_filtered.py da chay - tao folder v3.2_filtered_495m383D/
#
# Estimated: 1 baseline + 5 ablations = 6 runs x ~50min = ~5h CPU.
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"

Write-Host "================================================================"
Write-Host " OVERNIGHT v3.2 FILTERED EXPERIMENT"
Write-Host " Config: seed=1, K=7, default loss, dataset=v3.2_filtered_495m383D"
Write-Host " Total: 6 runs x ~50min = ~5h CPU"
Write-Host "================================================================"

# Step 1: Build v3.2 filtered (idempotent - skip if exists)
if (-not (Test-Path "v3.2_filtered_495m383D\multi_all_mirna_disease_pairs_without_negative.csv")) {
    Write-Host ""
    Write-Host "[Prep] Build v3.2 filtered dataset..."
    & $py build_v32_filtered.py
} else {
    Write-Host ""
    Write-Host "[Prep] v3.2 filtered dataset exists - skip" -ForegroundColor Yellow
}

# Step 2: Baseline v3.2 filtered (seed=1, K=7, no ablation)
Write-Host ""
Write-Host "================================================================"
Write-Host " [1/6] BASELINE v3.2 filtered (seed=1, K=7)"
Write-Host "================================================================"
$log = "logs\v32_baseline.log"
& $py main_experiments_hetero1.py --device cpu --seed 1 --K_neigs 7 --dataset v3.2_filtered_495m383D *>&1 | Tee-Object $log
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] baseline exited with $LASTEXITCODE" -ForegroundColor Red
}

# Step 3: 5 ablations v3.2 filtered
$ablations = @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')
$idx = 1
foreach ($abl in $ablations) {
    $idx++
    Write-Host ""
    Write-Host "================================================================"
    Write-Host " [$idx/6] ABLATION $abl (v3.2 filtered, seed=1, K=7)"
    Write-Host "================================================================"
    $log = "logs\v32_$abl.log"
    & $py main_experiments_hetero1.py --device cpu --seed 1 --K_neigs 7 --dataset v3.2_filtered_495m383D --ablation $abl *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] $abl exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Step 4: Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATE v3.2 FILTERED RESULTS"
Write-Host "================================================================"
& $py summarize_v32_filtered.py

# Step 5: Auto regen report + commit + push
Write-Host ""
Write-Host "[Final] Regen report + commit + push..."
& $py generate_report.py
& $py final_reproduce_report.py

# Git commit + push
git add -A
git commit -m "Overnight v3.2 filtered: baseline + 5 Fig.4 ablations (auto)"
git push origin main

Write-Host ""
Write-Host "DONE overnight v3.2. See results/v32_filtered_summary.json" -ForegroundColor Green
