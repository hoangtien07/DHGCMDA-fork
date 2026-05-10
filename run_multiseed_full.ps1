# Multi-seed full - 3 seeds x 3 baseline variants for MLRC paper.
#
# Bug fixed (2026-05-11): seed_torch + prepareData seed propagation.
# All previous Plan A/B/C/D/E results used seed=0 data split (bugged).
# Multi-seed v3 reruns 3 seeds fresh (1234, 42, 7).
#
# Usage:
#     .\run_multiseed_full.ps1                  # full ~7.5h: 9 runs
#     .\run_multiseed_full.ps1 -OnlyVariant plan_d  # 3 runs ~2.5h
#
param(
    [switch]$AllSeeds = $true,
    [string]$OnlyVariant = ''
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"

# Variants: hashtable - args list
$variants = @{
    'plan_c_w01' = @('--exist_weight', '0.1')
    'plan_d'     = @('--loss_mode', 'softmax_5class')
    'plan_e'     = @('--loss_mode', 'softmax_5class', '--ablation', 'no_cl_rebuild')
}

# 3 seeds fresh after bug fix
$seeds = @(1234, 42, 7)

# Filter variants
if ($OnlyVariant -ne '') {
    if (-not $variants.ContainsKey($OnlyVariant)) {
        Write-Host "[ERROR] Unknown variant: $OnlyVariant" -ForegroundColor Red
        exit 1
    }
    $selected_variants = @{ $OnlyVariant = $variants[$OnlyVariant] }
} else {
    $selected_variants = $variants
}

$total_runs = $seeds.Count * $selected_variants.Count
Write-Host "================================================================"
Write-Host " MULTI-SEED FULL"
Write-Host " Total runs: $total_runs (50 min each)"
Write-Host " Seeds: $($seeds -join ', ')"
Write-Host " Variants: $($selected_variants.Keys -join ', ')"
Write-Host "================================================================"

$run_idx = 0
foreach ($seed in $seeds) {
    foreach ($vname in $selected_variants.Keys) {
        $run_idx++
        $args_list = $selected_variants[$vname]
        Write-Host ""
        Write-Host "[$run_idx/$total_runs] $vname seed=$seed"
        Write-Host "----------------------------------------------------------------"

        $log_path = "logs\multiseed_${vname}_seed${seed}.log"
        & $py main_experiments_hetero1.py --device cpu --seed $seed @args_list *>&1 | Tee-Object $log_path
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] $vname seed=$seed exited with $LASTEXITCODE" -ForegroundColor Red
        }
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATE MULTI-SEED FULL"
Write-Host "================================================================"
& $py summarize_multiseed.py
Write-Host ""
Write-Host "DONE Multi-seed. See results\multiseed_full_summary.json" -ForegroundColor Green
