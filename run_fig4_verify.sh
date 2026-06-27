#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Fig.4 ablation verify with REPRODUCE config (seed=1, K=7, DEFAULT loss).
#
# Context: K sweep tim ra (seed=1, K=7) cho Top-1 F1 = 0.5909, gap -1.0%
# vs paper 0.5970 — essentially reproduced. Now verify Fig.4 ablation
# pattern voi cau hinh nay.
#
# Paper Fig.4 claim: ALL 5 ablations (no_cl, no_hgcn, no_avf, no_hgt,
# no_dv) HURT baseline Top-1 F1.
#
# Estimated: 5 ablation x ~50 min = ~4h CPU.
#
# Usage:
#     ./run_fig4_verify.sh

export PYTHONUTF8=1
PY="venv/bin/python"
seed=1
K=7
ablations=(no_cl no_hgcn no_avf no_hgt no_dv)

mkdir -p logs results

echo "================================================================"
echo " FIG.4 VERIFY with REPRODUCE config"
echo " seed=$seed, K_neigs=$K, DEFAULT loss (--exist_weight 0.3, two_head)"
echo " Paper baseline T1-F1 = 0.5970"
echo " Our baseline T1-F1 = 0.5909 (gap -1.0%)"
echo "================================================================"

idx=0
for abl in "${ablations[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "[$idx/${#ablations[@]}] ablation=$abl (seed=$seed, K=$K)"
    echo "----------------------------------------------------------------"
    log="logs/fig4_verify_${abl}.log"
    "$PY" main_experiments_hetero1.py \
        --device cpu --seed $seed --K_neigs $K --ablation "$abl" \
        2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] ablation $abl exited nonzero"
    fi
done

# Aggregate
echo ""
echo "================================================================"
echo " FIG.4 VERIFY SUMMARY"
echo "================================================================"
"$PY" summarize_fig4_verify.py
