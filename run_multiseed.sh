#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Multi-seed reproducibility — chay baseline + ablation w/o CL voi 3 seed:
# 42, 100, 2024 de do variance. Phuc vu statistical significance trong report.
#
# Tong thoi gian uoc tinh: ~7h CPU (3 seed x 2 variant x ~50min). Run sau
# khi sweep loss weight (run_full_rerun.sh) da xong.
#
# Cach dung:
#     ./run_multiseed.sh                 # full
#     ./run_multiseed.sh --smoke         # 3 epochs x 2 folds x 6 jobs ~ 1 phut
#     ./run_multiseed.sh --only-baseline # skip ablation, chi chay baseline 3 seed (~2.5h)

SMOKE=0
ONLY_BASELINE=0
for a in "$@"; do
    [ "$a" = "--smoke" ] && SMOKE=1
    [ "$a" = "--only-baseline" ] && ONLY_BASELINE=1
done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

seeds=(42 100 2024)

EXTRA=()
if [ "$SMOKE" -eq 1 ]; then
    EXTRA=(--epoch 3 --validation 2)
    echo "[Smoke test] 3 epochs x 2 folds per job"
else
    echo "[Full] 650 epochs x 5 folds per job"
fi

# Variant list: "label ablation_arg" (empty ablation = baseline)
if [ "$ONLY_BASELINE" -eq 1 ]; then
    variants=("baseline ")
else
    variants=("baseline " "no_cl no_cl")
fi

for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
        label="${variant%% *}"
        abl="${variant#* }"

        echo ""
        echo "================================================================"
        echo " SEED=$seed VARIANT=$label"
        echo "================================================================"

        logFile="logs/multiseed_${label}_seed${seed}.log"
        jsonFile="results/multiseed_${label}_seed${seed}.json"

        cmd=(main_experiments_hetero1.py --device cpu --seed "$seed")
        if [ -n "$abl" ]; then cmd+=(--ablation "$abl"); fi
        if [ "${#EXTRA[@]}" -gt 0 ]; then cmd+=("${EXTRA[@]}"); fi

        "$PY" "${cmd[@]}" 2>&1 | tee "$logFile"

        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo "[WARN] $label seed=$seed exited nonzero"
        fi

        echo "[Parse] $logFile -> $jsonFile"
        "$PY" parse_metrics.py "$logFile" "$jsonFile"
    done
done

# Tong hop mean +- std
echo ""
echo "================================================================"
echo " AGGREGATING MULTI-SEED STATS"
echo "================================================================"

"$PY" - <<'PYEOF'
import json, glob, os
import numpy as np
from collections import defaultdict

files = glob.glob('results/multiseed_*.json')
groups = defaultdict(list)
for f in files:
    base = os.path.basename(f).replace('.json', '').replace('multiseed_', '')
    parts = base.rsplit('_seed', 1)
    label = parts[0]
    with open(f, 'r', encoding='utf-8') as fh:
        groups[label].append(json.load(fh))

agg = {}
keys = ['AUC', 'AUPR', 'F1', 'top1_precision', 'top1_recall', 'top1_f1']
for label, runs in groups.items():
    stats = {}
    for k in keys:
        vals = [r[k] for r in runs if k in r and r[k] is not None]
        if vals:
            stats[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}
    agg[label] = stats

with open('results/multiseed_summary.json', 'w', encoding='utf-8') as f:
    json.dump(agg, f, indent=2, ensure_ascii=False)

print('=== Multi-seed mean +- std ===')
for label, stats in agg.items():
    print(f'\n{label}:')
    for k, v in stats.items():
        print(f'  {k}: {v["mean"]:.4f} +- {v["std"]:.4f}  (n={v["n"]})')
PYEOF

echo "Saved: results/multiseed_summary.json"
