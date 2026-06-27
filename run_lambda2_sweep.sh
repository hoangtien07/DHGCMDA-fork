#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Lambda2 sweep - paper Fig.2a reproduce check.
#
# Goal: paper Fig.2(a) sweep lambda2 (inter_view_weight) E {0.1, 0.3, 0.5}
# tim cau hinh closest to paper. Co the lambda2 la missing piece cho Fig.4
# reproduce.
#
# Note: --inter_view_weight da co san o param.py:112 (default 0.3). KHONG
# can sua code.
#
# Estimated: 3 lambda2 values x ~50 min = ~2.5h CPU.
#
# Usage:
#     ./run_lambda2_sweep.sh

export PYTHONUTF8=1
PY="venv/bin/python"
seed=1
K=7
lambda2_values=(0.1 0.5)  # 0.3 da co tu fig4_verify baseline

mkdir -p logs results

echo "================================================================"
echo " LAMBDA2 SWEEP - paper Fig.2a reproduce"
echo " Fixed: seed=$seed, K=$K, DEFAULT loss (--exist_weight 0.3, two_head)"
echo " lambda2 values: ${lambda2_values[*]}"
echo " Note: lambda2=0.3 baseline da co tu fig4_verify (K=7, seed=1)"
echo "================================================================"

idx=0
for l2 in "${lambda2_values[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "[$idx/${#lambda2_values[@]}] lambda2=$l2 (seed=$seed, K=$K)"
    echo "----------------------------------------------------------------"
    log="logs/lambda2_sweep_l2_${l2}.log"
    "$PY" main_experiments_hetero1.py \
        --device cpu --seed $seed --K_neigs $K --inter_view_weight "$l2" \
        2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] lambda2=$l2 exited nonzero"
    fi
done

# Aggregate
echo ""
echo "================================================================"
echo " LAMBDA2 SWEEP SUMMARY"
echo "================================================================"
"$PY" summarize_lambda2_sweep.py
