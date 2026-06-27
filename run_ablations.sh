#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

# Chay 5 ablation experiments tuan tu cho paper Fig. 4.
# Tong thoi gian uoc tinh: ~5 gio tren CPU Xeon E5-2680 v4.
#
# Cach dung:
#     ./run_ablations.sh
#     # Hoac smoke test:
#     ./run_ablations.sh --smoke

SMOKE=0
for a in "$@"; do [ "$a" = "--smoke" ] && SMOKE=1; done

export PYTHONUTF8=1
PY="venv/bin/python"

mkdir -p logs results

modes=(no_cl no_hgcn no_avf no_hgt no_dv)

EXTRA=()
if [ "$SMOKE" -eq 1 ]; then
    EXTRA=(--epoch 3 --validation 2)
    echo "[Smoke test mode] 3 epochs x 2 folds per ablation"
else
    echo "[Full mode] 650 epochs x 5 folds per ablation"
fi

for mode in "${modes[@]}"; do
    echo ""
    echo "================================================================"
    echo " ABLATION: $mode"
    echo "================================================================"

    logFile="logs/ablation_$mode.log"
    jsonFile="results/ablation_$mode.json"

    "$PY" main_experiments_hetero1.py \
        --device cpu \
        --ablation "$mode" \
        "${EXTRA[@]}" 2>&1 | tee "$logFile"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[WARN] Ablation $mode exited nonzero"
    fi

    echo "[Parse] $logFile -> $jsonFile"
    "$PY" parse_metrics.py "$logFile" "$jsonFile"
done

# Compile tong hop
echo ""
echo "================================================================"
echo " COMPILING SUMMARY"
echo "================================================================"

"$PY" - "${modes[@]}" <<'PYEOF'
import sys, json, os
modes = sys.argv[1:]
summary = {}
for m in modes:
    p = f"results/ablation_{m}.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as fh:
            summary[m] = json.load(fh)
with open("results/ablation_results.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
print("Saved: results/ablation_results.json")
PYEOF
