#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Seed sweep - tim seed match paper baseline closest.
#
# Goal (REPRODUCE): paper bao AUC=0.9669, Top-1 F1=0.5970 cho HMDD v2.0.
# Sau bug fix #2/#3, default split = args.seed. Sweep {0, 1, 42, 1234}
# voi DEFAULT CONFIG (--exist_weight 0.3, --loss_mode two_head,
# --ablation none, --K_neigs 13) de tim seed cho metrics gan paper nhat.
#
# Estimated time: 4 seeds x ~50 min = ~3.5h CPU.
#
# Usage:
#     ./run_seed_sweep.sh

export PYTHONUTF8=1
PY="venv/bin/python"
seeds=(0 1 42 1234)

mkdir -p logs results

echo "================================================================"
echo " SEED SWEEP - find seed closest to paper baseline"
echo " Paper: AUC=0.9669, Top-1 F1=0.5970"
echo " Seeds to test: ${seeds[*]}"
echo " Config: DEFAULT (--exist_weight 0.3, two_head, no ablation)"
echo "================================================================"

idx=0
for seed in "${seeds[@]}"; do
    idx=$((idx+1))
    echo ""
    echo "[$idx/${#seeds[@]}] seed=$seed (DEFAULT config)"
    echo "----------------------------------------------------------------"
    log="logs/seed_sweep_seed${seed}.log"
    "$PY" main_experiments_hetero1.py --device cpu --seed "$seed" 2>&1 | tee "$log"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] seed=$seed exited nonzero"
    fi
done

# Aggregate
echo ""
echo "================================================================"
echo " SEED SWEEP SUMMARY"
echo "================================================================"
"$PY" summarize_seed_sweep.py
