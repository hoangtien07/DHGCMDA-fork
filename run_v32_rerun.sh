#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Phase C-1d/e: Rerun baseline + 5 ablation tren HMDD v3.2 (5 types, GIP similarity).
# Parallel 2 jobs, moi job 14 thread.

PY="venv/bin/python"
start=$SECONDS

# Job 1: baseline + no_cl + no_hgcn
job1() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    export MKL_NUM_THREADS=14

    echo "[v32-Job1] Starting baseline..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed 2>&1 | tee "logs/v32_baseline.log"

    echo "[v32-Job1] Starting no_cl..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_cl 2>&1 | tee "logs/v32_ablation_no_cl.log"

    echo "[v32-Job1] Starting no_hgcn..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_hgcn 2>&1 | tee "logs/v32_ablation_no_hgcn.log"

    echo "[v32-Job1] DONE"
}

# Job 2: no_avf + no_hgt + no_dv
job2() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    export MKL_NUM_THREADS=14

    echo "[v32-Job2] Starting no_avf..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_avf 2>&1 | tee "logs/v32_ablation_no_avf.log"

    echo "[v32-Job2] Starting no_hgt..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_hgt 2>&1 | tee "logs/v32_ablation_no_hgt.log"

    echo "[v32-Job2] Starting no_dv..."
    "$PY" main_experiments_hetero1.py --device cpu --dataset v3.2_processed --ablation no_dv 2>&1 | tee "logs/v32_ablation_no_dv.log"

    echo "[v32-Job2] DONE"
}

job1 & P1=$!
job2 & P2=$!

echo "v3.2 jobs started. Job1 PID=$P1, Job2 PID=$P2"

wait $P1
wait $P2

echo "v3.2 rerun done in $((SECONDS-start))s"
