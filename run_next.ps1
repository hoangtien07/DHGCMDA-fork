[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$PythonBin = Join-Path $Root 'venv\Scripts\python.exe'

if (-not $env:PYTHONUTF8) {
    $env:PYTHONUTF8 = '1'
}
if (-not $env:DHGCMDA_N_THREADS) {
    $env:DHGCMDA_N_THREADS = '10'
}

& $PythonBin (Join-Path $Root 'run_next.py') --python $PythonBin @Arguments
exit $LASTEXITCODE
