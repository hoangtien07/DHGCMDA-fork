# Multi-seed full — 3 seed × 3 baseline = 9 runs (~7.5h CPU).
#
# Phuc vu MLRC paper: cap nhat error bar cho 3 chinh claims:
#  - Plan C-w0.1 baseline (loss tweak, two_head)
#  - Plan D baseline (full Eq.32 alignment, softmax_5class)
#  - Plan E baseline (no_cl_rebuild — best Plan E variant)
#
# Seed 1234 da co tu Plan A/B/C/D/E. Script nay re-chay seed 42 va 7
# cho ca 3 variant + parse ket qua → mean ± std.
#
# Cach dung:
#     .\run_multiseed_full.ps1                # full ~5h (skip 1234)
#     .\run_multiseed_full.ps1 -AllSeeds      # rerun ca 1234 ~7.5h
#     .\run_multiseed_full.ps1 -OnlyVariant plan_d   # 1 variant × 3 seed
#
param(
    [switch]$AllSeeds = $false,
    [string]$OnlyVariant = ''
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"

# Variants: (label, extra_args list)
$variants = @{
    'plan_c_w01' = @('--exist_weight', '0.1')                                          # Plan C-w0.1, two_head implicit
    'plan_d'     = @('--loss_mode', 'softmax_5class')                                  # Plan D baseline
    'plan_e'     = @('--loss_mode', 'softmax_5class', '--ablation', 'no_cl_rebuild')   # Plan E best variant
}

# Seeds
if ($AllSeeds) {
    $seeds = @(1234, 42, 7)
} else {
    $seeds = @(42, 7)
    Write-Host "[INFO] Skip seed 1234 (da co tu Plan A/B/C/D/E)" -ForegroundColor Yellow
}

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

$total = $seeds.Count * $selected_variants.Count
Write-Host "================================================================"
Write-Host " MULTI-SEED FULL ($total runs, ~50 phut/run)"
Write-Host " Seeds: $($seeds -join ', ')"
Write-Host " Variants: $($selected_variants.Keys -join ', ')"
Write-Host "================================================================"

$run_idx = 0
foreach ($seed in $seeds) {
    foreach ($vname in $selected_variants.Keys) {
        $run_idx++
        $args_list = $selected_variants[$vname]
        Write-Host ""
        Write-Host "[$run_idx/$total] $vname seed=$seed"
        Write-Host "----------------------------------------------------------------"

        $log_path = "logs\multiseed_${vname}_seed${seed}.log"
        & $py main_experiments_hetero1.py `
            --device cpu --seed $seed @args_list `
            *>&1 | Tee-Object $log_path
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
Write-Host "DONE Multi-seed. Xem results\multiseed_full_summary.json" -ForegroundColor Green
