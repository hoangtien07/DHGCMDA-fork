# K_neigs sweep - paper Fig.3 reproduce.
#
# Goal: Paper Fig.3(b) claim Top-1 max tai K=13. Sweep K {7, 9, 11, 13, 15}
# voi seed=1 (best calibration) + DEFAULT config khac → find K cho paper match.
#
# Estimated time: 5 K values x ~50 min = ~4h CPU.
#
# Usage:
#     .\run_k_sweep.ps1
#     .\run_k_sweep.ps1 -Seed 1234   # override seed
#
param(
    [int]$Seed = 1
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"
$K_values = @(7, 9, 11, 13, 15)

Write-Host "================================================================"
Write-Host " K_neigs SWEEP - paper Fig.3 reproduce"
Write-Host " Seed: $Seed (best calibration from seed sweep)"
Write-Host " K values: $($K_values -join ', ')"
Write-Host " Config: DEFAULT (--exist_weight 0.3, two_head, no ablation)"
Write-Host "================================================================"

$idx = 0
foreach ($K in $K_values) {
    $idx++
    Write-Host ""
    Write-Host "[$idx/$($K_values.Count)] K_neigs=$K (seed=$Seed)"
    Write-Host "----------------------------------------------------------------"
    $log = "logs\k_sweep_K${K}_seed${Seed}.log"
    & $py main_experiments_hetero1.py --device cpu --seed $Seed --K_neigs $K *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] K=$K seed=$Seed exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " K SWEEP SUMMARY"
Write-Host "================================================================"
& $py summarize_k_sweep.py --seed $Seed
