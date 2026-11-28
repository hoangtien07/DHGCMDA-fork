param(
    [string]$TaskName = 'DHGCMDA CPU Plan',
    [string]$OrcaWorkspace,
    [string]$OrcaCli = 'orca'
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$run = Join-Path $PSScriptRoot 'runs/cpu'
$progressLog = Join-Path $run 'watch-progress.log'
$pushed = Join-Path $run 'watch-pushed.txt'
$branch = 'research/track-p5-pretrain-2'
$env:GIT_TERMINAL_PROMPT = '0'

function Write-ProgressRecord([string]$Message) {
    Add-Content -LiteralPath $progressLog -Value "$(Get-Date -Format o) $Message" -Encoding utf8
}

function Invoke-GitChecked {
    & git @args 2>&1 | ForEach-Object { Write-ProgressRecord "$_" }
    if ($LASTEXITCODE -ne 0) { throw "Git command failed with exit code $LASTEXITCODE" }
}

function Publish-OrcaProgress([string]$Message) {
    if (-not $OrcaWorkspace) { return $true }
    $comment = "DHGCMDA $(Get-Date -Format 'dd/MM HH:mm') $Message"
    try {
        $reply = & $OrcaCli worktree set --worktree $OrcaWorkspace --comment $comment --unread --json 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Orca exit code ${LASTEXITCODE}: $reply" }
        $receipt = $reply | Out-String | ConvertFrom-Json
        if (-not $receipt.ok) { throw 'Orca did not confirm the update' }
        Write-ProgressRecord "Orca updated: $comment"
        return $true
    } catch {
        Write-ProgressRecord "Orca delivery failed: $($_.Exception.Message)"
        return $false
    }
}

function Complete-Watch {
    $commit = (Get-Content -LiteralPath $pushed).Trim()
    if (Publish-OrcaProgress "hoàn tất, đã push commit $($commit.Substring(0, 8)).") {
        Disable-ScheduledTask -TaskName $TaskName | Out-Null
    }
}

try {
    if (Test-Path -LiteralPath $pushed) { Complete-Watch; return }
    $pipelineId = [int](Get-Content -LiteralPath (Join-Path $run 'pipeline.pid'))
    $pipeline = Get-CimInstance Win32_Process -Filter "ProcessId = $pipelineId"
    $active = $pipeline.CommandLine -like '*continue_cpu.ps1*'
    $models = @(Get-ChildItem (Join-Path $run 'clean_models') -Recurse -Filter completion.json -File -ErrorAction SilentlyContinue)
    $baselines = @(Get-ChildItem (Join-Path $run 'clean_baselines') -Recurse -Filter completion.json -File -ErrorAction SilentlyContinue)
    $stage = if (Test-Path (Join-Path $run 'improvements-cpu.json')) { '6' }
             elseif (Test-Path (Join-Path $run 'controls-cpu.json')) { '5' }
             elseif (Test-Path (Join-Path $run 'g3.json')) { '4' }
             elseif (Test-Path (Join-Path $run 'calibration_lock.json')) { '3' }
             else { '2' }
    $pipelineLog = Join-Path $run 'pipeline.log'
    $latest = Get-Content -LiteralPath $pipelineLog -Tail 80 |
        Select-String 'Epoch \d+, Total Loss|^.*(?:train_controls|clean_baselines)\.py' |
        Select-Object -Last 1 -ExpandProperty Line
    Write-ProgressRecord "stage=$stage active=$active models=$($models.Count) baselines=$($baselines.Count) $latest"
    $parallelLog = Join-Path $run 'parallel-controls.log'
    if (Test-Path -LiteralPath $parallelLog) {
        $parallelLatest = Get-Content -LiteralPath $parallelLog -Tail 80 |
            Select-String 'Epoch \d+, Total Loss|(?:train_controls|clean_baselines)\.py|^Published |^Stopping ' |
            Select-Object -Last 1 -ExpandProperty Line
        if ($parallelLatest) { Write-ProgressRecord "parallel $parallelLatest" }
    }
    if ($active) {
        $totals = if ($stage -eq '4') { ' (mốc giai đoạn 4: 55 mô hình, 40 baseline)' } else { '' }
        $delivered = Publish-OrcaProgress "giai đoạn $stage đang chạy, đã xong $($models.Count) mô hình và $($baselines.Count) baseline$totals."
        if (-not $delivered) { exit 1 }
        return
    }

    $finalFiles = @('THESIS_RESULTS.md', 'calibration_lock.json', 'g2.json', 'g3.json',
                    'rescored-cpu.json', 'controls-cpu.json', 'improvements-cpu.json')
    foreach ($name in $finalFiles) {
        $path = Join-Path $run $name
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "Pipeline stopped before producing $name. See pipeline.err.log."
        }
    }
    if ((git branch --show-current) -ne $branch) { throw "Expected branch $branch" }
    if ((git remote get-url --push origin) -ne 'https://github.com/hoangtien07/DHGCMDA-fork.git') {
        throw 'Unexpected push remote'
    }

    $evidence = 'paper_repro/evidence/cpu-plan'
    New-Item -ItemType Directory -Path $evidence -Force | Out-Null
    foreach ($name in $finalFiles) { Copy-Item -LiteralPath (Join-Path $run $name) -Destination $evidence -Force }
    $reproRuns = Get-ChildItem -LiteralPath $run -Directory |
        Where-Object Name -match '^(c[123]|v32-.*)-cpu$'
    foreach ($repro in $reproRuns) {
        $destination = Join-Path $evidence $repro.Name
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        $reproFiles = @('config.json', 'completion.json') + @(1..5 | ForEach-Object { "fold_$_.npz" })
        foreach ($name in $reproFiles) { Copy-Item -LiteralPath (Join-Path $repro.FullName $name) -Destination $destination -Force }
    }
    $folds = [ordered]@{}
    foreach ($file in ($models + $baselines | Sort-Object FullName)) {
        $folds[$file.Directory.Name] = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json -AsHashtable
    }
    $folds | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $evidence 'clean-folds.json') -Encoding utf8

    $paths = @($evidence, 'paper_repro/.gitattributes', 'paper_repro/README.md', 'paper_repro/calibration.py',
               'paper_repro/clean_protocol.py', 'paper_repro/continue_calibration.ps1', 'paper_repro/continue_cpu.ps1',
               'paper_repro/continue_reproduction.ps1', 'paper_repro/continue_thesis.ps1', 'paper_repro/run_v20.py',
               'paper_repro/score_predictions.py', 'paper_repro/v32_runs.py', 'paper_repro/watch_cpu_plan.ps1')
    if (Test-Path 'paper_repro/precompute_controls.py') { $paths += 'paper_repro/precompute_controls.py' }
    Invoke-GitChecked add -- @paths
    & git diff --cached --quiet -- @paths
    if ($LASTEXITCODE -eq 1) {
        Invoke-GitChecked commit --only -m 'Record completed DHGCMDA CPU thesis experiments' -- @paths
    } elseif ($LASTEXITCODE -ne 0) {
        throw "Git diff failed with exit code $LASTEXITCODE"
    }
    Invoke-GitChecked -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin "HEAD:refs/heads/$branch"
    git rev-parse HEAD | Set-Content -LiteralPath $pushed -Encoding utf8
    Write-ProgressRecord "Push completed at $(Get-Content -LiteralPath $pushed)"
    Complete-Watch
} catch {
    Write-ProgressRecord "ERROR $($_.Exception.Message)"
    $null = Publish-OrcaProgress "job theo dõi gặp lỗi. Xem watch-progress.log để biết chi tiết."
    throw
}
