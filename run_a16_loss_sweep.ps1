# Plan I A16: diagnostic — does reducing CL/recon weight unlock v3.2 collapse?
# 2 folds x 3 configs. STOP signal if F1 < 0.05 (collapse not tunable).
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = 1
$env:OMP_NUM_THREADS = 14

# config: (cl_weight, recon_weight, label)
$configs = @(
    @(1.0, 1.0, 'baseline'),
    @(0.05, 0.3, 'aggressive'),
    @(0.1, 0.5, 'mid')
)
foreach ($c in $configs) {
    $cl = $c[0]; $rec = $c[1]; $lbl = $c[2]
    Write-Host "[A16] config=$lbl cl=$cl recon=$rec ..."
    & ".\venv\Scripts\python.exe" main_experiments_hetero1.py --device cpu `
        --dataset v3.2_wang --loss_mode two_head --exist_weight 0.1 `
        --cl_weight_override $cl --recon_weight_override $rec `
        --epoch 200 --validation 2 *>&1 | Tee-Object "logs\a16_$lbl.log"
}
Write-Host "[A16] DONE" -ForegroundColor Green
