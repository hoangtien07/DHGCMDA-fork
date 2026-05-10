# Plan E - True ablation rebuild full pipeline (3 modes x 5 fold).
#
# Verify hypothesis: Fig.4 ablation pattern bi dao co phai do additive switch
# implementation hay khong. Plan E rebuild 3 ablation dao (CL, HGCN, HGT) voi
# kien truc rut gon thuc su (HGCN plain, GCNConv, skip transformers).
#
# Cach dung:
#     .\run_phase_e.ps1                      # full ~3.5h: 3 ablations sequential
#     .\run_phase_e.ps1 -OnlyMode no_cl_rebuild   # chi 1 mode (~50')
#
param(
    [string]$OnlyMode = '',
    [switch]$SkipSummarize = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

$py = ".\venv\Scripts\python.exe"

# Modes to run
if ($OnlyMode -ne '') {
    $modes = @($OnlyMode)
} else {
    $modes = @('no_cl_rebuild', 'no_hgcn_rebuild', 'no_hgt_rebuild')
}

Write-Host "================================================================"
Write-Host " PLAN E - TRUE ABLATION REBUILD (loss_mode=softmax_5class)"
Write-Host " Modes: $($modes -join ', ')"
Write-Host " ETA: ~50 phut/mode * $($modes.Count) modes"
Write-Host "================================================================"

foreach ($mode in $modes) {
    Write-Host ""
    Write-Host "[Plan E] $mode (softmax_5class) ~50 phut"
    Write-Host "----------------------------------------------------------------"
    & $py main_experiments_hetero1.py `
        --device cpu --loss_mode softmax_5class --ablation $mode `
        *>&1 | Tee-Object "logs\phase_e_$mode.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] $mode exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

if (-not $SkipSummarize) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host " AGGREGATE PHASE E"
    Write-Host "================================================================"
    & $py summarize_phase_e.py
    & $py generate_report.py
    Write-Host ""
    Write-Host "DONE Plan E. Xem BaoCao_DHGCMDA.docx + results\phase_e_summary.json" -ForegroundColor Green
}
