$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:DHGCMDA_RUN_ROOT = Join-Path $PSScriptRoot 'runs/cpu'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONUTF8 = '1'
$trackPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
$c1 = Join-Path $env:DHGCMDA_RUN_ROOT 'c1-cpu'
if (-not (Test-Path -LiteralPath (Join-Path $c1 'completion.json'))) {
    if (Test-Path -LiteralPath $c1) {
        throw "Partial run preserved for diagnosis at $c1"
    }
    & $trackPython -B -u paper_repro/run_v20.py c1 --device cpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
foreach ($stage in @('calibration', 'reproduction', 'thesis')) {
    & (Join-Path $PSScriptRoot "continue_${stage}.ps1") -Device cpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
