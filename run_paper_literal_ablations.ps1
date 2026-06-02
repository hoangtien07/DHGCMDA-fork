# Plan F-D: Run 3 ablations (no_cl, no_hgt, no_dv) với loss_mode paper_literal.
# Parallel 2 jobs: Job1 = no_cl + no_hgt, Job2 = no_dv.

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

$job1 = Start-Job -Name "PL-Job1" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14

    Write-Host "[PL-Job1] no_cl..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu `
        --loss_mode paper_literal --ablation no_cl *>&1 | `
        Tee-Object "logs\ablation_paper_literal_no_cl.log"

    Write-Host "[PL-Job1] no_hgt..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu `
        --loss_mode paper_literal --ablation no_hgt *>&1 | `
        Tee-Object "logs\ablation_paper_literal_no_hgt.log"

    Write-Host "[PL-Job1] DONE"
} -ArgumentList (Get-Location).Path

$job2 = Start-Job -Name "PL-Job2" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14

    Write-Host "[PL-Job2] no_dv..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu `
        --loss_mode paper_literal --ablation no_dv *>&1 | `
        Tee-Object "logs\ablation_paper_literal_no_dv.log"

    Write-Host "[PL-Job2] DONE"
} -ArgumentList (Get-Location).Path

Write-Host "Job1 (PID $($job1.Id)): no_cl + no_hgt"
Write-Host "Job2 (PID $($job2.Id)): no_dv"

while ($job1.State -eq 'Running' -or $job2.State -eq 'Running') {
    Start-Sleep -Seconds 60
    $elapsed = (Get-Date) - $startTime
    Write-Host ("[{0:hh\:mm\:ss}] Job1={1}, Job2={2}" -f $elapsed, $job1.State, $job2.State)
}

Receive-Job $job1 | Select-Object -Last 3
Receive-Job $job2 | Select-Object -Last 3
Remove-Job $job1, $job2

$total = (Get-Date) - $startTime
Write-Host ("DONE in {0:hh\:mm\:ss}" -f $total) -ForegroundColor Green
