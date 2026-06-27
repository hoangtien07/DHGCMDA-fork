#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Plan I A16: diagnostic — does reducing CL/recon weight unlock v3.2 collapse?
# 2 folds x 3 configs. STOP signal if F1 < 0.05 (collapse not tunable).
export PYTHONUTF8=1
export OMP_NUM_THREADS=14
PY="venv/bin/python"

# config: cl_weight recon_weight label
configs=(
    "1.0 1.0 baseline"
    "0.05 0.3 aggressive"
    "0.1 0.5 mid"
)
for c in "${configs[@]}"; do
    read -r cl rec lbl <<< "$c"
    echo "[A16] config=$lbl cl=$cl recon=$rec ..."
    "$PY" main_experiments_hetero1.py --device cpu \
        --dataset v3.2_wang --loss_mode two_head --exist_weight 0.1 \
        --cl_weight_override "$cl" --recon_weight_override "$rec" \
        --epoch 200 --validation 2 2>&1 | tee "logs/a16_$lbl.log"
done
echo "[A16] DONE"
