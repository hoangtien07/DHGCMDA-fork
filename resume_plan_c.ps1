# Resume Plan C — verify w=0.1 fix Fig.4 + case study collapse.
#
# Trạng thái khi script được tạo (2026-05-09):
# ✅ Sweep loss xong: w=0.1 thắng (Top-1 F1 = 0.5996, vượt paper 0.5970).
# ⏸ Verify pending: cần rerun case_study.py (~9') + 5 ablation (~2.5h CPU)
#    với exist_weight=0.1 để kiểm tra Fig.4 pattern + class collapse có fix
#    được không.
#
# Cách dùng:
#     .\resume_plan_c.ps1                  # full sequential ~2.7h
#     .\resume_plan_c.ps1 -SkipCaseStudy   # chỉ 5 ablation (~2.5h)
#     .\resume_plan_c.ps1 -OnlyCaseStudy   # chỉ case study + rerank (~9')
#
# Sau khi xong:
#     python summarize_plan_c_full.py      # parse + so sánh paper Fig.4/Bảng 5/6
#     python generate_report.py            # regen BaoCao_DHGCMDA.docx
#
param(
    [switch]$SkipCaseStudy = $false,
    [switch]$OnlyCaseStudy = $false
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

if (-not (Test-Path "logs"))    { New-Item -ItemType Directory logs    | Out-Null }
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }

# Backup snapshot Phase B-C nếu chưa có (idempotent)
if (-not (Test-Path "results\snapshot_phaseBC_w0.3")) {
    New-Item -ItemType Directory "results\snapshot_phaseBC_w0.3" | Out-Null
    foreach ($f in @('case_study_score.npy', 'case_study_breast.csv',
                     'case_study_hcc.csv', 'case_study_summary.json')) {
        if (Test-Path "results\$f") {
            Copy-Item "results\$f" "results\snapshot_phaseBC_w0.3\$f"
        }
    }
    Write-Host "[backup] Snapshot Phase B-C → results\snapshot_phaseBC_w0.3\" -ForegroundColor Cyan
}

# Step 1: Case study với w=0.1
if (-not $SkipCaseStudy) {
    # Xóa cache để case_study.py train lại với w=0.1
    if (Test-Path "results\case_study_score.npy") {
        Remove-Item "results\case_study_score.npy"
        Write-Host "[clean] Deleted cached score.npy (sẽ retrain với w=0.1)" -ForegroundColor Yellow
    }
    Write-Host "================================================================"
    Write-Host " 1/3 CASE STUDY (w=0.1) ~9 phút"
    Write-Host "================================================================"
    & ".\venv\Scripts\python.exe" case_study.py --exist_weight 0.1 *>&1 | Tee-Object "logs\case_study_w0.1.log"

    Write-Host "================================================================"
    Write-Host " 2/3 RERANK (4 chiến lược, ~5 giây)"
    Write-Host "================================================================"
    & ".\venv\Scripts\python.exe" rerank_case_study.py *>&1 | Tee-Object "logs\rerank_w0.1.log"
}

# Step 2: 5 ablation với w=0.1
if (-not $OnlyCaseStudy) {
    Write-Host "================================================================"
    Write-Host " 3/3 ABLATION (w=0.1) — 5 variants × ~30 phút = ~2.5h"
    Write-Host "================================================================"
    foreach ($mode in @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')) {
        Write-Host ""
        Write-Host "[ABL] $mode (w=0.1)"
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py `
            --device cpu --exist_weight 0.1 --ablation $mode `
            *>&1 | Tee-Object "logs\abl_w0.1_$mode.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] ablation $mode exited with $LASTEXITCODE" -ForegroundColor Red
        }
    }
}

# Step 3: aggregate
Write-Host ""
Write-Host "================================================================"
Write-Host " AGGREGATE + REGEN BÁO CÁO"
Write-Host "================================================================"
& ".\venv\Scripts\python.exe" summarize_plan_c_full.py
& ".\venv\Scripts\python.exe" generate_report.py
Write-Host ""
Write-Host "✅ DONE. Xem BaoCao_DHGCMDA.docx + results\plan_c_full_summary.json" -ForegroundColor Green
