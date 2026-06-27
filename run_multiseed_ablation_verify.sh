#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Plan H-2: Multi-seed verification of ablation reversal (M14).
# Question: is the "ablation IMPROVES Top-1 F1" reversal REAL or NOISE?
# Run baseline + no_cl + no_hgt at exist_weight=0.1 (matched config) across 3 NEW seeds.
# seed=1 already have: baseline 0.5996, no_cl 0.6286, no_hgt 0.6452.
# Parallel 2 jobs.

PY="venv/bin/python"
start=$SECONDS

# Job 1: seeds 0, 42 — baseline + no_cl + no_hgt each
job1() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    for seed in 0 42; do
        echo "[MS-Job1] seed=$seed baseline..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" 2>&1 | tee "logs/ms_baseline_seed$seed.log"
        echo "[MS-Job1] seed=$seed no_cl..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" --ablation no_cl 2>&1 | tee "logs/ms_no_cl_seed$seed.log"
        echo "[MS-Job1] seed=$seed no_hgt..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" --ablation no_hgt 2>&1 | tee "logs/ms_no_hgt_seed$seed.log"
    done
    echo "[MS-Job1] DONE"
}

# Job 2: seed 1234 — baseline + no_cl + no_hgt
job2() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    for seed in 1234; do
        echo "[MS-Job2] seed=$seed baseline..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" 2>&1 | tee "logs/ms_baseline_seed$seed.log"
        echo "[MS-Job2] seed=$seed no_cl..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" --ablation no_cl 2>&1 | tee "logs/ms_no_cl_seed$seed.log"
        echo "[MS-Job2] seed=$seed no_hgt..."
        "$PY" main_experiments_hetero1.py --device cpu --exist_weight 0.1 --seed "$seed" --ablation no_hgt 2>&1 | tee "logs/ms_no_hgt_seed$seed.log"
    done
    echo "[MS-Job2] DONE"
}

job1 & P1=$!
job2 & P2=$!

echo "Job1 (PID $P1): seeds 0,42 (6 runs)"
echo "Job2 (PID $P2): seed 1234 (3 runs)"

wait $P1
wait $P2

echo "DONE in $((SECONDS-start))s"
