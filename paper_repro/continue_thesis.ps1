param(
    [int]$ReproductionProcessId,
    [ValidateSet('cuda','cpu')]
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$trackPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
if ($ReproductionProcessId -gt 0 -and (Get-Process -Id $ReproductionProcessId -ErrorAction SilentlyContinue)) {
    Write-Output "Waiting for v3.2 reproduction process $ReproductionProcessId"
    Wait-Process -Id $ReproductionProcessId -ErrorAction SilentlyContinue
}
& $trackPython -B -u paper_repro/clean_experiment.py all --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$report = if ($env:DHGCMDA_RUN_ROOT) {
    Join-Path $env:DHGCMDA_RUN_ROOT 'THESIS_RESULTS.md'
} else {
    '../outputs/track-p5-thesis-results.md'
}
& $trackPython -B -u paper_repro/report_thesis.py --device $Device --output $report
exit $LASTEXITCODE
