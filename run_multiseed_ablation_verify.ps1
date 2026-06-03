# Plan H-2: Multi-seed verification of ablation reversal (M14).
# Question: is the "ablation IMPROVES Top-1 F1" reversal REAL or NOISE?
# Run baseline + no_cl + no_hgt at exist_weight=0.1 (matched config) across 3 NEW seeds.
# seed=1 already have: baseline 0.5996, no_cl 0.6286, no_hgt 0.6452.
# Parallel 2 jobs.

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

# Job 1: seeds 0, 42 — baseline + no_cl + no_hgt each
$job1 = Start-Job -Name "MS-Job1" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    foreach ($seed in 0, 42) {
        Write-Host "[MS-Job1] seed=$seed baseline..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed *>&1 | Tee-Object "logs\ms_baseline_seed$seed.log"
        Write-Host "[MS-Job1] seed=$seed no_cl..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed --ablation no_cl *>&1 | Tee-Object "logs\ms_no_cl_seed$seed.log"
        Write-Host "[MS-Job1] seed=$seed no_hgt..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed --ablation no_hgt *>&1 | Tee-Object "logs\ms_no_hgt_seed$seed.log"
    }
    Write-Host "[MS-Job1] DONE"
} -ArgumentList (Get-Location).Path

# Job 2: seed 1234 — baseline + no_cl + no_hgt
$job2 = Start-Job -Name "MS-Job2" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    foreach ($seed in 1234) {
        Write-Host "[MS-Job2] seed=$seed baseline..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed *>&1 | Tee-Object "logs\ms_baseline_seed$seed.log"
        Write-Host "[MS-Job2] seed=$seed no_cl..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed --ablation no_cl *>&1 | Tee-Object "logs\ms_no_cl_seed$seed.log"
        Write-Host "[MS-Job2] seed=$seed no_hgt..."
        & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed $seed --ablation no_hgt *>&1 | Tee-Object "logs\ms_no_hgt_seed$seed.log"
    }
    Write-Host "[MS-Job2] DONE"
} -ArgumentList (Get-Location).Path

Write-Host "Job1 (PID $($job1.Id)): seeds 0,42 (6 runs)"
Write-Host "Job2 (PID $($job2.Id)): seed 1234 (3 runs)"

while ($job1.State -eq 'Running' -or $job2.State -eq 'Running') {
    Start-Sleep -Seconds 120
    $elapsed = (Get-Date) - $startTime
    Write-Host ("[{0:hh\:mm\:ss}] Job1={1}, Job2={2}" -f $elapsed, $job1.State, $job2.State)
}

Receive-Job $job1 | Select-Object -Last 2
Receive-Job $job2 | Select-Object -Last 2
Remove-Job $job1, $job2

$total = (Get-Date) - $startTime
Write-Host ("DONE in {0:hh\:mm\:ss}" -f $total) -ForegroundColor Green
