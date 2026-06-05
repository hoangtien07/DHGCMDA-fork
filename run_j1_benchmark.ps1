$ErrorActionPreference='Continue'
$startTime = Get-Date
# Job1: v2.0 full 5-fold full_bilinear (reference vs diag 0.5996)
$job1 = Start-Job -Name "J1-v2" -ScriptBlock {
  param($wd); Set-Location $wd; $env:PYTHONUTF8=1; $env:OMP_NUM_THREADS=14
  & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --predictor_mode full_bilinear *>&1 | Tee-Object "logs\j1_v2_full_bilinear.log"
} -ArgumentList (Get-Location).Path
# Job2: v3.2 1-fold full_bilinear smoke (collapse recovery check)
$job2 = Start-Job -Name "J1-v32" -ScriptBlock {
  param($wd); Set-Location $wd; $env:PYTHONUTF8=1; $env:OMP_NUM_THREADS=14
  & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu --dataset v3.2_wang --exist_weight 0.1 --predictor_mode full_bilinear --epoch 300 --validation 2 *>&1 | Tee-Object "logs\j1_v32_full_bilinear.log"
} -ArgumentList (Get-Location).Path
while ($job1.State -eq 'Running' -or $job2.State -eq 'Running') {
  Start-Sleep -Seconds 120
  Write-Host ("[{0:hh\:mm\:ss}] v2={1} v32={2}" -f ((Get-Date)-$startTime), $job1.State, $job2.State)
}
Receive-Job $job1 | Select-Object -Last 2; Receive-Job $job2 | Select-Object -Last 2
Remove-Job $job1,$job2
Write-Host ("DONE {0:hh\:mm\:ss}" -f ((Get-Date)-$startTime)) -ForegroundColor Green
