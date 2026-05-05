# Chạy 5 ablation experiments tuần tự cho paper Fig. 4.
# Tổng thời gian ước tính: ~5 giờ trên CPU Xeon E5-2680 v4.
#
# Cách dùng:
#     .\run_ablations.ps1
#     # Hoặc smoke test:
#     .\run_ablations.ps1 -SmokeTest
#
param(
    [switch]$SmokeTest = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$modes = @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')

if ($SmokeTest) {
    $extraArgs = '--epoch', '3', '--validation', '2'
    Write-Host "[Smoke test mode] 3 epochs x 2 folds per ablation" -ForegroundColor Yellow
} else {
    $extraArgs = @()
    Write-Host "[Full mode] 650 epochs x 5 folds per ablation" -ForegroundColor Cyan
}

foreach ($mode in $modes) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host " ABLATION: $mode"
    Write-Host "================================================================"

    $logFile = "logs\ablation_$mode.log"
    $jsonFile = "results\ablation_$mode.json"

    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py `
        --device cpu `
        --ablation $mode `
        @extraArgs *>&1 | Tee-Object -FilePath $logFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Ablation $mode exited with code $LASTEXITCODE" -ForegroundColor Red
    }

    Write-Host "[Parse] $logFile -> $jsonFile"
    & ".\venv\Scripts\python.exe" parse_metrics.py $logFile $jsonFile
}

# Compile tổng hợp
Write-Host ""
Write-Host "================================================================"
Write-Host " COMPILING SUMMARY"
Write-Host "================================================================"

$summary = @{}
foreach ($mode in $modes) {
    $jsonFile = "results\ablation_$mode.json"
    if (Test-Path $jsonFile) {
        $summary[$mode] = Get-Content $jsonFile | ConvertFrom-Json
    }
}
$summary | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 "results\ablation_results.json"
Write-Host "Saved: results\ablation_results.json"
