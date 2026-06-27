#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Combo A: K=3, K=5 sweep + multi-seed verify best config.
#
# Goal: while waiting for CDMBlab authors reply, do 2 things:
#  1. Complete K sweep with K=3, K=5 (paper Fig.3 tests K=1,3,5,7,9,11,13,15)
#  2. Multi-seed verify best config (K=7, seed=1) with seeds 42, 7 to get error bar
#
# Estimated: 4 runs x ~50min = ~3.5h CPU.

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

echo "================================================================"
echo " COMBO A: K=3,5 extend + Multi-seed verify (best config)"
echo " Total: 4 runs x ~50min = ~3.5h CPU"
echo "================================================================"

# Phase 1: K=3, K=5 with seed=1 (complete Fig.3 coverage)
echo ""
echo "[Phase 1/2] K=3 and K=5 sweep (seed=1, default config)"
echo "----------------------------------------------------------------"

for K in 3 5; do
    echo ""
    echo "K_neigs=$K (seed=1)"
    log="logs/k_sweep_K${K}_seed1.log"
    "$PY" main_experiments_hetero1.py --device cpu --seed 1 --K_neigs "$K" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] K=$K exited nonzero"
    fi
done

# Phase 2: Multi-seed verify (K=7 best config, seeds 42 and 7)
echo ""
echo "[Phase 2/2] Multi-seed verify best config (K=7, seeds 42 and 7)"
echo "----------------------------------------------------------------"

for seed in 42 7; do
    echo ""
    echo "K_neigs=7 (seed=$seed)"
    log="logs/multiseed_best_seed${seed}.log"
    "$PY" main_experiments_hetero1.py --device cpu --seed "$seed" --K_neigs 7 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] seed=$seed exited nonzero"
    fi
done

# Aggregate
echo ""
echo "================================================================"
echo " COMBO A SUMMARY"
echo "================================================================"

# Re-run K sweep summary (now includes K=3, 5)
"$PY" summarize_k_sweep.py --seed 1
"$PY" summarize_multiseed_best.py

# Re-run final report
"$PY" final_reproduce_report.py

echo ""
echo "DONE Combo A."
