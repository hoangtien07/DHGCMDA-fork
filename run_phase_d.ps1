# Phase D — Fix A++ (5-class softmax CE) full pipeline.
#
# Verify Eq. 32 alignment hoan toan (single L_type, no L_existence).
# Compare voi Plan C-w0.1 (two_head) de xem fix Fig.4 + case study collapse hay khong.
#
# Cach dung:
#     .\run_phase_d.ps1                  # full ~5-6h: baseline + 5 ablation + case study
#     .\run_phase_d.ps1 -SkipCaseStudy
#     .\run_phase_d.ps1 -OnlyBaseline    # chi baseline (~50') de smoke kiem tra Top-1 F1
#
param(
    [switch]$SkipCaseStudy = $false,
    [switch]$OnlyBaseline = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

# Backup Plan C w=0.1 case study results truoc khi rerun (idempotent)
if (-not (Test-Path "results\snapshot_planC_w0.1")) {
    New-Item -ItemType Directory "results\snapshot_planC_w0.1" | Out-Null
    foreach ($f in @('case_study_score.npy', 'case_study_breast.csv',
                     'case_study_hcc.csv', 'case_study_summary.json',
                     'rerank_summary.json')) {
        if (Test-Path "results\$f") {
            Copy-Item "results\$f" "results\snapshot_planC_w0.1\$f"
        }
    }
    Write-Host "[backup] Snapshot Plan C-w0.1 -> results\snapshot_planC_w0.1\" -ForegroundColor Cyan
}

$py = ".\venv\Scripts\python.exe"

# Step 1: Phase D baseline (~50')
Write-Host "================================================================"
Write-Host " 1/3 PHASE D BASELINE (loss_mode=softmax_5class) ~50 phut"
Write-Host "================================================================"
& $py main_experiments_hetero1.py --device cpu --loss_mode softmax_5class *>&1 | Tee-Object "logs\phase_d_baseline.log"

if ($OnlyBaseline) {
    Write-Host ""
    Write-Host "DONE baseline only. Run summarize_phase_d.py to compare." -ForegroundColor Green
    exit 0
}

# Step 2: Case study with softmax_5class (~9')
if (-not $SkipCaseStudy) {
    if (Test-Path "results\case_study_score.npy") {
        Remove-Item "results\case_study_score.npy"
        Write-Host "[clean] Deleted cached score.npy (se retrain voi softmax_5class)" -ForegroundColor Yellow
    }
    Write-Host "================================================================"
    Write-Host " 2/3 CASE STUDY (softmax_5class) ~9 phut"
    Write-Host "================================================================"
    & $py case_study.py --loss_mode softmax_5class *>&1 | Tee-Object "logs\phase_d_casestudy.log"

    & $py rerank_case_study.py *>&1 | Tee-Object "logs\phase_d_rerank.log"
}

# Step 3: 5 ablation voi softmax_5class
Write-Host "================================================================"
Write-Host " 3/3 ABLATION (softmax_5class) - 5 variants x ~30 phut = ~2.5h"
Write-Host "================================================================"
foreach ($mode in @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')) {
    Write-Host ""
    Write-Host "[ABL] $mode (softmax_5class)"
    & $py main_experiments_hetero1.py --device cpu --loss_mode softmax_5class --ablation $mode *>&1 | Tee-Object "logs\phase_d_$mode.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] ablation $mode exited with $LASTEXITCODE" -ForegroundColor Red
    }
}

# Step 4: aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATE + REGEN BAO CAO"
Write-Host "================================================================"
& $py summarize_phase_d.py
& $py generate_report.py
Write-Host ""
Write-Host "DONE Phase D. Xem BaoCao_DHGCMDA.docx + results\phase_d_summary.json" -ForegroundColor Green
