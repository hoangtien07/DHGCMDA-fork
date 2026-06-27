#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

PY="venv/bin/python"
start=$SECONDS

# Job1: v2.0 full 5-fold full_bilinear (reference vs diag 0.5996)
job1() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --predictor_mode full_bilinear 2>&1 | tee "logs/j1_v2_full_bilinear.log"
}

# Job2: v3.2 1-fold full_bilinear smoke (collapse recovery check)
job2() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_wang --exist_weight 0.1 --predictor_mode full_bilinear --epoch 300 --validation 2 2>&1 | tee "logs/j1_v32_full_bilinear.log"
}

job1 & P1=$!
job2 & P2=$!

echo "v2 (PID $P1), v32 (PID $P2)"
wait $P1
wait $P2

echo "DONE $((SECONDS-start))s"
