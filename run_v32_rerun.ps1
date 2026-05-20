# Phase C-1d/e: Rerun baseline + 5 ablation trên HMDD v3.2 (5 types, GIP similarity).
# Parallel 2 jobs, mỗi job 14 thread.

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

# Job 1: baseline + no_cl + no_hgcn
$job1 = Start-Job -Name "v32-Job1" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    $env:MKL_NUM_THREADS = 14

    Write-Host "[v32-Job1] Starting baseline..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed *>&1 | Tee-Object "logs\v32_baseline.log"

    Write-Host "[v32-Job1] Starting no_cl..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_cl *>&1 | Tee-Object "logs\v32_ablation_no_cl.log"

    Write-Host "[v32-Job1] Starting no_hgcn..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_hgcn *>&1 | Tee-Object "logs\v32_ablation_no_hgcn.log"

    Write-Host "[v32-Job1] DONE"
} -ArgumentList (Get-Location).Path

# Job 2: no_avf + no_hgt + no_dv
$job2 = Start-Job -Name "v32-Job2" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    $env:MKL_NUM_THREADS = 14

    Write-Host "[v32-Job2] Starting no_avf..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_avf *>&1 | Tee-Object "logs\v32_ablation_no_avf.log"

    Write-Host "[v32-Job2] Starting no_hgt..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_hgt *>&1 | Tee-Object "logs\v32_ablation_no_hgt.log"

    Write-Host "[v32-Job2] Starting no_dv..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_dv *>&1 | Tee-Object "logs\v32_ablation_no_dv.log"

    Write-Host "[v32-Job2] DONE"
} -ArgumentList (Get-Location).Path

Write-Host "v3.2 jobs started. Job1 PID=$($job1.Id), Job2 PID=$($job2.Id)"

while ($job1.State -eq 'Running' -or $job2.State -eq 'Running') {
    Start-Sleep -Seconds 60
    $elapsed = (Get-Date) - $startTime
    Write-Host ("[{0:hh\:mm\:ss}] Job1={1}, Job2={2}" -f $elapsed, $job1.State, $job2.State)
}

Receive-Job $job1 | Select-Object -Last 3
Receive-Job $job2 | Select-Object -Last 3
Remove-Job $job1, $job2

$total = (Get-Date) - $startTime
Write-Host ("v3.2 rerun done in {0:hh\:mm\:ss}" -f $total) -ForegroundColor Green
