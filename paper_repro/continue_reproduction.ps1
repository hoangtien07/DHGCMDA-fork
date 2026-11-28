param(
    [int]$CalibrationProcessId,
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
if ($CalibrationProcessId -gt 0 -and (Get-Process -Id $CalibrationProcessId -ErrorAction SilentlyContinue)) {
    Write-Output "Waiting for calibration process $CalibrationProcessId"
    Wait-Process -Id $CalibrationProcessId -ErrorAction SilentlyContinue
}
if (-not (Test-Path -LiteralPath (Join-Path $runRoot 'calibration_lock.json'))) {
    throw 'Calibration lock is missing. Inspect the calibration log for G2 or run failures.'
}

function Invoke-ReproductionRun {
    param([string]$Profile, [int]$Seed = 1234, [string]$Diagnostic, [switch]$Background)
    $runVariant = if ($Diagnostic) { "-$Diagnostic" } else { '' }
    $run = Join-Path $runRoot "v32-$Profile-$Seed$runVariant-$Device"
    if (Test-Path -LiteralPath (Join-Path $run 'completion.json')) { return }
    if (Test-Path -LiteralPath $run) {
        throw "Partial run preserved for diagnosis at $run"
    }
    Write-Host "v3.2 profile $Profile seed $Seed $Diagnostic"
    $runOptions = @('v32', '--profile', $Profile, '--seed', $Seed, '--device', $Device)
    if ($Diagnostic) { $runOptions += @('--diagnostic', $Diagnostic) }
    if ($Background) {
        return Start-Process -FilePath $trackPython -ArgumentList (@('-B', '-u', 'paper_repro/run_v20.py') + $runOptions) `
            -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$run.log" -RedirectStandardError "$run.err.log"
    }
    & $trackPython -B -u paper_repro/run_v20.py @runOptions
    if ($LASTEXITCODE -ne 0) { throw "v3.2 run exited with code $LASTEXITCODE" }
}

$profiles = @('K_star', 'T_star')
if ($Device -eq 'cpu') {
    $jobs = @{}
    try {
        foreach ($profile in $profiles) {
            $process = Invoke-ReproductionRun -Profile $profile -Background
            if ($process) { $jobs[$profile] = $process }
        }
    } finally {
        foreach ($process in $jobs.Values) { $process.WaitForExit() }
    }
    foreach ($profile in $jobs.Keys) {
        if ($jobs[$profile].ExitCode -ne 0) {
            throw "v3.2 $profile exited with code $($jobs[$profile].ExitCode). See v32-$profile-1234-$Device.err.log."
        }
    }
} else {
    foreach ($profile in $profiles) { Invoke-ReproductionRun -Profile $profile }
}
& $trackPython -B -u paper_repro/v32_runs.py --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$decision = Get-Content -Raw -LiteralPath (Join-Path $runRoot 'g3.json') | ConvertFrom-Json
if ($decision.status -eq 'matched') {
    foreach ($seed in @(42, 2025)) {
        Invoke-ReproductionRun -Profile $decision.closer_profile -Seed $seed
    }
} elseif ($decision.status -eq 'near') {
    foreach ($variant in @('heads', 'disease-gip')) {
        Invoke-ReproductionRun -Profile T_star -Diagnostic $variant
    }
}
& $trackPython -B -u paper_repro/score_predictions.py --device $Device
exit $LASTEXITCODE
