[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$PythonBin = Join-Path $Root 'venv\Scripts\python.exe'
& $PythonBin (Join-Path $Root 'run_next.py') --python $PythonBin @Arguments
exit $LASTEXITCODE
