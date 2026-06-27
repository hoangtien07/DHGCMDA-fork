#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Plan F-D: Run 3 ablations (no_cl, no_hgt, no_dv) voi loss_mode paper_literal.
# Parallel 2 jobs: Job1 = no_cl + no_hgt, Job2 = no_dv.

PY="venv/bin/python"
start=$SECONDS

job1() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14

    echo "[PL-Job1] no_cl..."
    "$PY" main_experiments_hetero1.py --device cpu \
        --loss_mode paper_literal --ablation no_cl 2>&1 | \
        tee "logs/ablation_paper_literal_no_cl.log"

    echo "[PL-Job1] no_hgt..."
    "$PY" main_experiments_hetero1.py --device cpu \
        --loss_mode paper_literal --ablation no_hgt 2>&1 | \
        tee "logs/ablation_paper_literal_no_hgt.log"

    echo "[PL-Job1] DONE"
}

job2() {
    export PYTHONUTF8=1
    export OMP_NUM_THREADS=14

    echo "[PL-Job2] no_dv..."
    "$PY" main_experiments_hetero1.py --device cpu \
        --loss_mode paper_literal --ablation no_dv 2>&1 | \
        tee "logs/ablation_paper_literal_no_dv.log"

    echo "[PL-Job2] DONE"
}

job1 & P1=$!
job2 & P2=$!

echo "Job1 (PID $P1): no_cl + no_hgt"
echo "Job2 (PID $P2): no_dv"

wait $P1
wait $P2

echo "DONE in $((SECONDS-start))s"
