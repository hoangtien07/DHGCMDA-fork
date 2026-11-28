param(
    [int]$C1ProcessId,
    [ValidateSet('cuda','cpu')]
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$trackPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
if ($Device -eq 'cpu') { $env:CUDA_VISIBLE_DEVICES = '-1' }
$runRoot = if ($env:DHGCMDA_RUN_ROOT) { $env:DHGCMDA_RUN_ROOT } else { Join-Path $PSScriptRoot 'runs' }
if ($C1ProcessId -gt 0 -and (Get-Process -Id $C1ProcessId -ErrorAction SilentlyContinue)) {
    Write-Output "Waiting for C1 process $C1ProcessId"
    Wait-Process -Id $C1ProcessId -ErrorAction SilentlyContinue
}
& $trackPython -B -u paper_repro/calibration.py g2 --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($mode in @('c2', 'c3')) {
    Get-Date -Format o
    $run = Join-Path $runRoot "$mode-$Device"
    if (Test-Path -LiteralPath (Join-Path $run 'completion.json')) { continue }
    if (Test-Path -LiteralPath $run) {
        throw "Partial run preserved for diagnosis at $run"
    }
    & $trackPython -B -u paper_repro/run_v20.py $mode --device $Device
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $trackPython -B -u paper_repro/calibration.py freeze --device $Device
exit $LASTEXITCODE
