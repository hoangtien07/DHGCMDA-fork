#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Plan B Phase B-C: Rerun baseline + 5 ablation sau khi sua 3 discrepancies.
# Chay 2 job song song, moi job sequential 3 runs.
# CPU Xeon E5-2680 v4: 14 core / 28 thread -> split 14 thread/process.

export PYTHONUTF8=1
PY="venv/bin/python"
start=$SECONDS

# Job 1: baseline + no_cl + no_hgcn
job1() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    export MKL_NUM_THREADS=14

    echo "[Job1] Starting baseline..."
    "$PY" main_experiments_hetero1.py --device cpu 2>&1 | tee "logs/baseline_v2.0_full.log"

    echo "[Job1] Starting no_cl..."
    "$PY" main_experiments_hetero1.py --device cpu --ablation no_cl 2>&1 | tee "logs/ablation_no_cl.log"

    echo "[Job1] Starting no_hgcn..."
    "$PY" main_experiments_hetero1.py --device cpu --ablation no_hgcn 2>&1 | tee "logs/ablation_no_hgcn.log"

    echo "[Job1] DONE"
}

# Job 2: no_avf + no_hgt + no_dv
job2() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14
    export MKL_NUM_THREADS=14

    echo "[Job2] Starting no_avf..."
    "$PY" main_experiments_hetero1.py --device cpu --ablation no_avf 2>&1 | tee "logs/ablation_no_avf.log"

    echo "[Job2] Starting no_hgt..."
    "$PY" main_experiments_hetero1.py --device cpu --ablation no_hgt 2>&1 | tee "logs/ablation_no_hgt.log"

    echo "[Job2] Starting no_dv..."
    "$PY" main_experiments_hetero1.py --device cpu --ablation no_dv 2>&1 | tee "logs/ablation_no_dv.log"

    echo "[Job2] DONE"
}

job1 & P1=$!
job2 & P2=$!

echo "Both jobs started."
echo "Job1 (PID $P1): baseline, no_cl, no_hgcn"
echo "Job2 (PID $P2): no_avf, no_hgt, no_dv"

wait $P1
wait $P2

echo ""
echo "Total time: $((SECONDS-start))s"
echo "Now run: ./compile_final.sh"
