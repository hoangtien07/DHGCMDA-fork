# Plan B Phase B-C: Rerun baseline + 5 ablation sau khi sửa 3 discrepancies.
# Chạy 2 PowerShell Job song song, mỗi job sequential 3 runs.
# CPU Xeon E5-2680 v4: 14 core / 28 thread → split 14 thread/process.

$ErrorActionPreference = 'Stop'
$startTime = Get-Date

# Job 1: baseline + no_cl + no_hgcn
$job1 = Start-Job -Name "Job1" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    $env:MKL_NUM_THREADS = 14

    Write-Host "[Job1] Starting baseline..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu *>&1 | Tee-Object "logs\baseline_v2.0_full.log"

    Write-Host "[Job1] Starting no_cl..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --ablation no_cl *>&1 | Tee-Object "logs\ablation_no_cl.log"

    Write-Host "[Job1] Starting no_hgcn..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --ablation no_hgcn *>&1 | Tee-Object "logs\ablation_no_hgcn.log"

    Write-Host "[Job1] DONE"
} -ArgumentList (Get-Location).Path

# Job 2: no_avf + no_hgt + no_dv
$job2 = Start-Job -Name "Job2" -ScriptBlock {
    param($workdir)
    Set-Location $workdir
    $env:PYTHONUTF8 = 1
    $env:OMP_NUM_THREADS = 14
    $env:MKL_NUM_THREADS = 14

    Write-Host "[Job2] Starting no_avf..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --ablation no_avf *>&1 | Tee-Object "logs\ablation_no_avf.log"

    Write-Host "[Job2] Starting no_hgt..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --ablation no_hgt *>&1 | Tee-Object "logs\ablation_no_hgt.log"

    Write-Host "[Job2] Starting no_dv..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --ablation no_dv *>&1 | Tee-Object "logs\ablation_no_dv.log"

    Write-Host "[Job2] DONE"
} -ArgumentList (Get-Location).Path

Write-Host "Both jobs started. Polling every 60s..."
Write-Host "Job1 (PID: $($job1.Id)): baseline, no_cl, no_hgcn"
Write-Host "Job2 (PID: $($job2.Id)): no_avf, no_hgt, no_dv"

# Poll until both done
while ($job1.State -eq 'Running' -or $job2.State -eq 'Running') {
    Start-Sleep -Seconds 60
    $elapsed = (Get-Date) - $startTime
    Write-Host ("[{0:hh\:mm\:ss}] Job1={1}, Job2={2}" -f $elapsed, $job1.State, $job2.State)
}

# Receive output
Write-Host ""
Write-Host "=== Job1 final output ==="
Receive-Job $job1 | Select-Object -Last 5
Write-Host ""
Write-Host "=== Job2 final output ==="
Receive-Job $job2 | Select-Object -Last 5

Remove-Job $job1, $job2

$total = (Get-Date) - $startTime
Write-Host ""
Write-Host ("Total time: {0:hh\:mm\:ss}" -f $total) -ForegroundColor Green
Write-Host "Now run: .\compile_final.ps1"
