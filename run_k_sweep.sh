#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# K_neigs sweep - paper Fig.3 reproduce.
#
# Goal: Paper Fig.3(b) claim Top-1 max tai K=13. Sweep K {7, 9, 11, 13, 15}
# voi seed=1 (best calibration) + DEFAULT config khac -> find K cho paper match.
#
# Estimated time: 5 K values x ~50 min = ~4h CPU.
#
# Usage:
#     ./run_k_sweep.sh
#     ./run_k_sweep.sh --seed 1234   # override seed

Seed=1
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) Seed="$2"; shift 2 ;;
        *) shift ;;
    esac
done

export PYTHONUTF8=1
PY="venv/bin/python"
K_values=(7 9 11 13 15)

mkdir -p logs results

echo "================================================================"
echo " K_neigs SWEEP - paper Fig.3 reproduce"
echo " Seed: $Seed (best calibration from seed sweep)"
echo " K values: ${K_values[*]}"
echo " Config: DEFAULT (--exist_weight 0.3, two_head, no ablation)"
echo "================================================================"

idx=0
for K in "${K_values[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "[$idx/${#K_values[@]}] K_neigs=$K (seed=$Seed)"
    echo "----------------------------------------------------------------"
    log="logs/k_sweep_K${K}_seed${Seed}.log"
    "$PY" main_experiments_hetero1.py --device cpu --seed "$Seed" --K_neigs "$K" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] K=$K seed=$Seed exited nonzero"
    fi
done

# Aggregate
echo ""
echo "================================================================"
echo " K SWEEP SUMMARY"
echo "================================================================"
"$PY" summarize_k_sweep.py --seed "$Seed"
