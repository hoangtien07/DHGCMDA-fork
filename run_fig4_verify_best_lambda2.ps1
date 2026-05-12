# Auto-chain script: rerun Fig.4 ablation voi best lambda2 from lambda2 sweep.
#
# Logic:
#   1. Read results/lambda2_sweep_summary.json
#   2. Identify best_lambda2 (closest to paper Top-1 F1=0.5970)
#   3. If best_lambda2 == 0.3, skip (da co tu fig4_verify_summary.json)
#   4. Else: rerun 5 ablation voi (seed=1, K=7, inter_view_weight=best_lambda2)
#
# Estimated: 0-5 ablation x ~50min = 0-4h CPU (depends on best lambda2).
#
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

$py = ".\venv\Scripts\python.exe"

# Read best lambda2 from summary JSON
$summary_path = "results\lambda2_sweep_summary.json"
if (-not (Test-Path $summary_path)) {
    Write-Host "[ERROR] $summary_path not found. Lambda2 sweep chua xong?" -ForegroundColor Red
    exit 1
}

$summary = Get-Content $summary_path -Raw | ConvertFrom-Json
$best_l2 = $summary.best_lambda2
$best_t1 = $summary.best_top1_f1

Write-Host "================================================================"
Write-Host " AUTO-CHAIN: Fig.4 verify with BEST lambda2"
Write-Host " Best lambda2 from sweep: $best_l2 (T1-F1 = $best_t1)"
Write-Host " Paper T1-F1: 0.5970"
Write-Host "================================================================"

if ($best_l2 -eq 0.3) {
    Write-Host ""
    Write-Host "[SKIP] Best lambda2 = 0.3 (default) — Fig.4 da co tu fig4_verify_summary.json" -ForegroundColor Yellow
    Write-Host "       Khong can rerun. Verdict: lambda2 KHONG fix Fig.4." -ForegroundColor Yellow

    # Auto-trigger final summary
    & $py final_reproduce_report.py
    exit 0
}

# Rerun Fig.4 ablation with best lambda2
$seed = 1
$K = 7
$ablations = @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')

Write-Host ""
Write-Host "Running 5 ablations with lambda2 = $best_l2..."

$idx = 0
foreach ($abl in $ablations) {
    $idx++
    Write-Host ""
    Write-Host "[$idx/$($ablations.Count)] ablation=$abl (lambda2=$best_l2)"
    Write-Host "----------------------------------------------------------------"
    $log = "logs\fig4_l2_${best_l2}_${abl}.log"
    & $py main_experiments_hetero1.py `
        --device cpu --seed $seed --K_neigs $K --inter_view_weight $best_l2 --ablation $abl `
        *>&1 | Tee-Object $log
}

# Aggregate Fig.4 with new lambda2
Write-Host ""
Write-Host "================================================================"
Write-Host " FIG.4 VERIFY (lambda2=$best_l2) SUMMARY"
Write-Host "================================================================"
& $py summarize_fig4_best_lambda2.py --lambda2 $best_l2

# Final report
Write-Host ""
& $py final_reproduce_report.py

Write-Host ""
Write-Host "DONE chain. Xem results\final_reproduce_report.json" -ForegroundColor Green
