# Resume Plan C — verify w=0.1 fix Fig.4 + case study collapse.
#
# Cách dùng:
#     .\resume_plan_c.ps1                  # full sequential ~2.7h
#     .\resume_plan_c.ps1 -SkipCaseStudy   # chỉ 5 ablation (~2.5h)
#     .\resume_plan_c.ps1 -OnlyCaseStudy   # chỉ case study + rerank (~9')
#
param(
    [switch]$SkipCaseStudy = $false,
    [switch]$OnlyCaseStudy = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

# Backup snapshot Phase B-C neu chua co (idempotent)
if (-not (Test-Path "results\snapshot_phaseBC_w0.3")) {
    New-Item -ItemType Directory "results\snapshot_phaseBC_w0.3" | Out-Null
    foreach ($f in @('case_study_score.npy', 'case_study_breast.csv', 'case_study_hcc.csv', 'case_study_summary.json')) {
        if (Test-Path "results\$f") {
            Copy-Item "results\$f" "results\snapshot_phaseBC_w0.3\$f"
        }
    }
    Write-Host "[backup] Snapshot Phase B-C -> results\snapshot_phaseBC_w0.3\" -ForegroundColor Cyan
}

$py = ".\venv\Scripts\python.exe"

# Step 1: Case study voi w=0.1
if (-not $SkipCaseStudy) {
    if (Test-Path "results\case_study_score.npy") {
        Remove-Item "results\case_study_score.npy"
        Write-Host "[clean] Deleted cached score.npy (se retrain voi w=0.1)" -ForegroundColor Yellow
    }
    Write-Host "================================================================"
    Write-Host " 1/3 CASE STUDY (w=0.1) ~9 phut"
    Write-Host "================================================================"
    & $py case_study.py --exist_weight 0.1 *>&1 | Tee-Object "logs\case_study_w0.1.log"

    Write-Host "================================================================"
    Write-Host " 2/3 RERANK (4 chien luoc, ~5 giay)"
    Write-Host "================================================================"
    & $py rerank_case_study.py *>&1 | Tee-Object "logs\rerank_w0.1.log"
}

# Step 2: 5 ablation voi w=0.1
if (-not $OnlyCaseStudy) {
    Write-Host "================================================================"
    Write-Host " 3/3 ABLATION (w=0.1) - 5 variants x ~30 phut = ~2.5h"
    Write-Host "================================================================"
    foreach ($mode in @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')) {
        Write-Host ""
        Write-Host "[ABL] $mode (w=0.1)"
        & $py main_experiments_hetero1.py --device cpu --exist_weight 0.1 --ablation $mode *>&1 | Tee-Object "logs\abl_w0.1_$mode.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] ablation $mode exited with $LASTEXITCODE" -ForegroundColor Red
        }
    }
}

# Step 3: aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATE + REGEN BAO CAO"
Write-Host "================================================================"
& $py summarize_plan_c_full.py
& $py generate_report.py
Write-Host ""
Write-Host "DONE. Xem BaoCao_DHGCMDA.docx + results\plan_c_full_summary.json" -ForegroundColor Green
