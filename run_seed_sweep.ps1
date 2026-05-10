# Seed sweep - tim seed match paper baseline closest.
#
# Goal (REPRODUCE): paper bao AUC=0.9669, Top-1 F1=0.5970 cho HMDD v2.0.
# Sau bug fix #2/#3, default split = args.seed. Sweep {0, 1, 42, 1234}
# voi DEFAULT CONFIG (--exist_weight 0.3, --loss_mode two_head,
# --ablation none, --K_neigs 13) de tim seed cho metrics gan paper nhat.
#
# Estimated time: 4 seeds x ~50 min = ~3.5h CPU.
#
# Usage:
#     .\run_seed_sweep.ps1
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"
$seeds = @(0, 1, 42, 1234)

Write-Host "================================================================"
Write-Host " SEED SWEEP - find seed closest to paper baseline"
Write-Host " Paper: AUC=0.9669, Top-1 F1=0.5970"
Write-Host " Seeds to test: $($seeds -join ', ')"
Write-Host " Config: DEFAULT (--exist_weight 0.3, two_head, no ablation)"
Write-Host "================================================================"

$idx = 0
foreach ($seed in $seeds) {
    $idx++
    Write-Host ""
    Write-Host "[$idx/$($seeds.Count)] seed=$seed (DEFAULT config)"
    Write-Host "----------------------------------------------------------------"
    $log = "logs\seed_sweep_seed${seed}.log"
    & $py main_experiments_hetero1.py --device cpu --seed $seed *>&1 | Tee-Object $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] seed=$seed exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " SEED SWEEP SUMMARY"
Write-Host "================================================================"
& $py summarize_seed_sweep.py
