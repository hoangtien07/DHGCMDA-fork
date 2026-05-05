# Compile final results sau khi tất cả ablation xong:
#   1. Re-parse tất cả ablation logs (handle UTF-16 từ Tee-Object)
#   2. Tổng hợp ablation_results.json đúng format
#   3. Re-generate báo cáo .docx với data đầy đủ

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1

$modes = @('no_cl', 'no_hgcn', 'no_avf', 'no_hgt', 'no_dv')

Write-Host "=== Step 1: Re-parse all ablation logs ===" -ForegroundColor Cyan
foreach ($mode in $modes) {
    $log = "logs\ablation_$mode.log"
    $json = "results\ablation_$mode.json"
    if (Test-Path $log) {
        & ".\venv\Scripts\python.exe" parse_metrics.py $log $json | Out-Null
        Write-Host "  Parsed: $mode"
    } else {
        Write-Host "  MISSING: $log" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Step 2: Compile ablation_results.json ===" -ForegroundColor Cyan
$summary = [ordered]@{}
foreach ($mode in $modes) {
    $json = "results\ablation_$mode.json"
    if (Test-Path $json) {
        $summary[$mode] = Get-Content $json -Encoding UTF8 | ConvertFrom-Json
    }
}
$summary | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 "results\ablation_results.json"
Write-Host "  Saved: results\ablation_results.json"

Write-Host ""
Write-Host "=== Step 3: Re-generate report ===" -ForegroundColor Cyan
& ".\venv\Scripts\python.exe" generate_report.py

Write-Host ""
Write-Host "=== Step 4: Print summary table ===" -ForegroundColor Cyan
$baseline = Get-Content "results\baseline_v2.0_metrics.json" -Encoding UTF8 | ConvertFrom-Json
Write-Host ""
Write-Host ("{0,-12} {1,-8} {2,-8} {3,-12} {4,-12} {5,-12}" -f 'Variant', 'AUC', 'AUPR', 'Top1-Prec', 'Top1-Rec', 'Top1-F1')
Write-Host ("-" * 70)
Write-Host ("{0,-12} {1,-8:N4} {2,-8:N4} {3,-12:N4} {4,-12:N4} {5,-12:N4}" -f `
    'Full', $baseline.AUC, $baseline.AUPR, $baseline.top1_precision, $baseline.top1_recall, $baseline.top1_f1)
foreach ($mode in $modes) {
    $m = $summary[$mode]
    if ($m -and $m.AUC) {
        $name = "w/o " + $mode.Substring(3).ToUpper()
        Write-Host ("{0,-12} {1,-8:N4} {2,-8:N4} {3,-12:N4} {4,-12:N4} {5,-12:N4}" -f `
            $name, $m.AUC, $m.AUPR, $m.top1_precision, $m.top1_recall, $m.top1_f1)
    }
}
